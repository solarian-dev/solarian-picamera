from __future__ import annotations

import os
import signal
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, TYPE_CHECKING

import numpy as np

from solarian.core.camera_bus import CameraBus, ColorMode
from solarian.core.data import Data, ImageData
from solarian_picamera.config import PiCameraSensorConfig

if TYPE_CHECKING:
    from solarian_picamera.camera import PiCamera


import logging
logger = logging.getLogger("picamera_bus")
# Root is often DEBUG; keep picamera2's INFO+ but drop its per-capture
# "Execute job" debug lines.
logging.getLogger("picamera2").setLevel(logging.INFO)


class Rotation(int, Enum):
    NO_ROTATION = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = -90


#: The clock libcamera stamps SensorTimestamp on. Only Linux has BOOTTIME;
#: elsewhere this module is imported but never run against real hardware, and
#: on a board that does not suspend the two clocks agree anyway.
_SENSOR_CLOCK = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)

# CSI cameras (IMX477 HQ especially) drop frames or stall on ribbon/power
# glitches. libcamera's dequeue watchdog is ~1s; after that capture_array
# hangs or raises and the pipeline is dead until Picamera2 is recreated.
_CAPTURE_TIMEOUT_S = 1.5
_AE_SETTLE_S = 0.5
_RECONNECT_SETTLE_S = 0.0
_RECONNECT_MIN_DELAY_S = 0.05
_RECONNECT_MAX_DELAY_S = 1.0
_CLOSE_TIMEOUT_S = 2.0
_BUFFER_COUNT = 6

# How long a leftover holder gets to exit after SIGTERM before SIGKILL.
# picamera2's own close is the same order of magnitude; this is just the
# gap between the two signals.
_LEFTOVER_TERM_WAIT_S = 0.5

# libcamera's Pi pipeline, as it shows up in sysfs video-node names.
# A USB webcam's name does not contain these, so it is left alone.
_CSI_VIDEO_MARKERS = ("pisp", "rp1-cfe", "unicam", "imx")


def _process_group(pid: int, proc_root: Path) -> int | None:
    """``pgrp`` from ``/proc/<pid>/stat``, or ``None`` if it cannot be read."""
    try:
        stat = (proc_root / str(pid) / "stat").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    # comm is parenthesised and may contain spaces; fields after it are fixed.
    fields = stat.rsplit(")", 1)[-1].split()
    try:
        return int(fields[2])
    except (IndexError, ValueError):
        return None


def _csi_device_paths(dev_root: Path, sys_root: Path) -> set[str]:
    """Device nodes a Pi CSI pipeline holds open.

    ``/dev/media*`` is the media controller. The matching ``/dev/video*``
    nodes are the ones whose sysfs name belongs to the Pi ISP or sensor,
    not every video device on the machine.
    """
    paths: set[str] = set()
    if dev_root.is_dir():
        for node in dev_root.iterdir():
            if node.name.startswith("media"):
                paths.add(str(node))
    v4l = sys_root / "class" / "video4linux"
    if not v4l.is_dir():
        return paths
    for node in v4l.iterdir():
        try:
            name = (node / "name").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lowered = name.lower()
        if any(marker in lowered for marker in _CSI_VIDEO_MARKERS):
            paths.add(str(dev_root / node.name))
    return paths


def _pids_holding(proc_root: Path, devices: set[str]) -> dict[int, str]:
    """Pid to command line, for processes with an fd open on ``devices``."""
    holders: dict[int, str] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target in devices:
                try:
                    raw = (entry / "cmdline").read_bytes()
                except OSError:
                    raw = b""
                cmdline = (
                    raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
                )
                holders[int(entry.name)] = cmdline
                break
    return holders


def release_leftover_cameras(
    *,
    proc_root: str | Path = "/proc",
    sys_root: str | Path = "/sys",
    dev_root: str | Path = "/dev",
    kill: Callable[[int, int], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """SIGTERM, then SIGKILL, any process from another run holding the Pi camera.

    A previous ``run-robot`` whose camera child outlived it leaves libcamera's
    pipeline locked (``Device or resource busy``). No ``/proc`` — a laptop, or
    a machine with no CSI nodes — is a no-op.

    Processes in this process group are never signaled: forked module
    processes share it, so on a robot with two CSI cameras the second one
    to connect must not kill the first.
    """
    proc = Path(proc_root)
    if not proc.is_dir():
        return
    devices = _csi_device_paths(Path(dev_root), Path(sys_root))
    if not devices:
        return
    me = os.getpid()
    own_group = os.getpgid(0)
    holders = {
        pid: cmd
        for pid, cmd in _pids_holding(proc, devices).items()
        if pid != me and _process_group(pid, proc) != own_group
    }
    if not holders:
        return
    send = os.kill if kill is None else kill
    pause = time.sleep if sleep is None else sleep
    for pid, cmd in sorted(holders.items()):
        logger.warning(
            "Releasing CSI camera held by pid %s (%s)",
            pid,
            cmd or "unknown",
        )
        try:
            send(pid, signal.SIGTERM)
        except OSError:
            logger.debug("SIGTERM pid %s failed", pid, exc_info=True)
    pause(_LEFTOVER_TERM_WAIT_S)
    still = {pid for pid in _pids_holding(proc, devices) if pid in holders}
    for pid in sorted(still):
        logger.warning("pid %s still holding the CSI camera; sending SIGKILL", pid)
        try:
            send(pid, signal.SIGKILL)
        except OSError:
            logger.debug("SIGKILL pid %s failed", pid, exc_info=True)


class PiCameraBus(CameraBus):
    """Sensor bus for Raspberry Pi CSI cameras via picamera2.

    ``port`` is the libcamera CSI index (``"0"`` = cam0, ``"1"`` = cam1),
    not a ``/dev/video*`` path. ``picamera2`` is a system package
    (``apt install python3-picamera2``); use a venv created with
    ``--system-site-packages`` so it is importable.

    Capture uses a timeout. On stall, empty frame, or libcamera error the
    bus closes the handle and recreates ``Picamera2`` until the camera
    comes back, so a CSI glitch does not kill the module process.
    """

    def __init__(self, port: str, components: Dict[str, PiCamera], calibration=None):
        super().__init__(port, components, calibration)
        self.picam2 = None
        self.components = components
        self.capture_timeout_s = _CAPTURE_TIMEOUT_S
        self.ae_settle_s = _AE_SETTLE_S
        self.reconnect_settle_s = _RECONNECT_SETTLE_S
        self.reconnect_min_delay_s = _RECONNECT_MIN_DELAY_S
        self.reconnect_max_delay_s = _RECONNECT_MAX_DELAY_S
        self.close_timeout_s = _CLOSE_TIMEOUT_S
        self.buffer_count = _BUFFER_COUNT

    @property
    def is_connected(self) -> bool:
        return self.picam2 is not None

    def _cfg(self) -> PiCameraSensorConfig:
        return next(iter(self.components.values())).cfg  # type: ignore[return-value]

    def _camera_num(self) -> int:
        if isinstance(self.port, str) and self.port.startswith("/dev/"):
            raise ValueError(
                f"PiCameraBus port must be a CSI camera index (e.g. '0' or '1'), "
                f"not a V4L2 device path: {self.port!r}. "
                f"CSI cameras on Raspberry Pi use libcamera via picamera2."
            )
        try:
            return int(self.port)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"PiCameraBus port must be an integer CSI index, got {self.port!r}"
            ) from e

    @staticmethod
    def _picamera2_class():
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise ImportError(
                "picamera2 is required for PiCameraBus. On Raspberry Pi OS install "
                "with `sudo apt install python3-picamera2` and create your venv with "
                "`--system-site-packages` so the system package is importable."
            ) from e
        return Picamera2

    def _open(self, *, settle: float) -> None:
        from solarian.core.optical_remap import use_one_opencv_thread

        use_one_opencv_thread()
        camera_num = self._camera_num()
        Picamera2 = self._picamera2_class()
        cfg = self._cfg()
        # Instantiate only here (after fork/reconnect in the module child process).
        self.picam2 = Picamera2(camera_num=camera_num)
        main = {"format": "RGB888", "size": (int(cfg.width), int(cfg.height))}
        controls = {"FrameRate": float(cfg.fps)}
        if cfg.exposure_value is not None:
            controls["ExposureValue"] = float(cfg.exposure_value)
        if cfg.brightness is not None:
            controls["Brightness"] = float(cfg.brightness)
        if cfg.contrast is not None:
            controls["Contrast"] = float(cfg.contrast)
        if cfg.shutter is not None:
            controls["ExposureTime"] = int(cfg.shutter)
        if cfg.gain is not None:
            controls["AnalogueGain"] = float(cfg.gain)
        if cfg.shutter is not None and cfg.gain is not None:
            controls["AeEnable"] = False
        try:
            config = self.picam2.create_video_configuration(
                main=main,
                controls=controls,
                buffer_count=int(self.buffer_count),
            )
        except TypeError:
            config = self.picam2.create_video_configuration(
                main=main,
                controls=controls,
            )
        self.picam2.configure(config)
        self.picam2.start()
        if settle > 0:
            time.sleep(settle)

    def connect(self):
        if self.picam2 is not None:
            return
        release_leftover_cameras()
        self._open(settle=self.ae_settle_s)

    @staticmethod
    def _close_picam2(picam2) -> None:
        try:
            cancel = getattr(picam2, "cancel_all_and_flush", None)
            if cancel is not None:
                cancel()
        except Exception:  # noqa: BLE001
            logger.debug("picamera2 cancel_all_and_flush failed", exc_info=True)
        try:
            picam2.stop()
            picam2.close()
        except Exception:  # noqa: BLE001
            logger.exception("Error closing picamera2")

    def disconnect(self):
        picam2 = self.picam2
        self.picam2 = None
        if picam2 is None:
            return
        closer = threading.Thread(
            target=self._close_picam2,
            args=(picam2,),
            name="picamera-close",
            daemon=True,
        )
        closer.start()
        closer.join(timeout=self.close_timeout_s)
        if closer.is_alive():
            logger.warning(
                "picamera2 close timed out on CSI %s; abandoning handle",
                self.port,
            )

    def _capture_frame(self):
        """Return one frame, and record how old it is from the sensor's clock.

        Taken as a request rather than an array so the metadata comes with it.
        libcamera stamps SensorTimestamp when the frame's first row was
        exposed, on CLOCK_BOOTTIME, which is the one number here that owes
        nothing to when this process happened to be scheduled.

        The request holds a DMA buffer out of a small pool, so it is released
        on every path; leaking one starves the pipeline within a few frames.
        """
        if self.picam2 is None:
            raise RuntimeError("Camera not connected")
        timeout = self.capture_timeout_s
        try:
            request = self.picam2.capture_request(wait=timeout)
        except TypeError:
            # Older picamera2: wait is bool/None only.
            request = self.picam2.capture_request()
        if request is None:
            raise RuntimeError("Could not read frame from camera")
        try:
            frame = request.make_array("main")
            sensor_ts = request.get_metadata().get("SensorTimestamp")
        finally:
            request.release()
        if frame is None:
            raise RuntimeError("Could not read frame from camera")
        self.measured_capture_age = (
            None
            if sensor_ts is None
            else time.clock_gettime(_SENSOR_CLOCK) - sensor_ts / 1e9
        )
        return frame

    def _reconnect(self, reason: BaseException | str) -> None:
        logger.warning(
            "PiCamera CSI %s dropped (%s); reconnecting",
            self.port,
            reason,
        )
        self.disconnect()
        delay = self.reconnect_min_delay_s
        attempt = 0
        while True:
            attempt += 1
            if delay > 0:
                time.sleep(delay)
            try:
                self._open(settle=self.reconnect_settle_s)
                logger.warning(
                    "PiCamera CSI %s reconnected after %d attempt(s)",
                    self.port,
                    attempt,
                )
                return
            except Exception as exc:  # noqa: BLE001
                delay = (
                    min(delay * 2.0, self.reconnect_max_delay_s)
                    if delay > 0
                    else self.reconnect_min_delay_s
                )
                logger.warning(
                    "PiCamera CSI %s reconnect attempt %d failed: %s; retry in %.2fs",
                    self.port,
                    attempt,
                    exc,
                    delay,
                )

    def _capture_or_reconnect(self):
        while True:
            try:
                return self._capture_frame()
            except Exception as exc:  # noqa: BLE001
                self._reconnect(exc)

    def read(self, data_type: type[Data], component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        import cv2

        if not issubclass(data_type, ImageData):
            raise ValueError(f"PiCameraBus only supports ImageData, got {data_type}")

        frame = self._capture_or_reconnect()

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[..., :3]

        # picamera2 RGB888 returns a buffer that is already B,G,R for OpenCV
        # (UMI observed quirk). Treat as BGR so color_mode handling matches
        # OpenCVCameraBus.
        processed = np.ascontiguousarray(frame)

        cfg = self._cfg()

        if cfg.rotation == Rotation.ROTATE_90.value:
            processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
        elif cfg.rotation == Rotation.ROTATE_180.value:
            processed = cv2.rotate(processed, cv2.ROTATE_180)
        elif cfg.rotation == Rotation.ROTATE_270.value:
            processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)

        camera_name = next(iter(self.components.keys()))
        processed = self.archive_and_finish(camera_name, processed, cfg)

        if not isinstance(processed, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(processed)}")

        return {camera_name: processed}

    def is_calibrated(self) -> bool:
        return True
