"""PiCameraBus reconnects after CSI stalls without crashing the module process."""

from __future__ import annotations

import os
import signal
import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from solarian.core.data import ImageData, PosData
from solarian_picamera.bus import _SENSOR_CLOCK, PiCameraBus, release_leftover_cameras
from solarian_picamera.module import PiCameraConfig


def _cfg(**overrides):
    values = dict(
        port="0",
        fps=30,
        width=8,
        height=6,
        rotation=0,
        channels=3,
        color_mode="bgr",
        video_format="MJPG",
        resize_width=None,
        resize_height=None,
        archive_width=None,
        archive_height=None,
        optical_remap=None,
        exposure_value=None,
        brightness=None,
        contrast=None,
        shutter=None,
        gain=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class FakePicamera2:
    instances: list["FakePicamera2"] = []
    fail_opens = 0
    fail_captures = 0
    empty_frames = 0
    reject_wait_timeout = False
    capture_exc_type: type[BaseException] = RuntimeError
    sensor_timestamp: int | None = None

    def __init__(self, camera_num=0):
        type(self).instances.append(self)
        if type(self).fail_opens > 0:
            type(self).fail_opens -= 1
            raise RuntimeError("camera busy")
        self.camera_num = camera_num
        self.started = False
        self.closed = False
        self.cancelled = False
        self.captures = 0
        self.config = None
        self.last_wait = None
        #: Requests handed out and not yet given back. The buffer pool is
        #: small, so a leak here is a stalled camera a few frames later.
        self.outstanding = 0

    def create_video_configuration(self, main=None, controls=None, buffer_count=None):
        return {"main": main, "controls": controls, "buffer_count": buffer_count}

    def configure(self, config):
        self.config = config

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def cancel_all_and_flush(self):
        self.cancelled = True

    def capture_request(self, wait=None):
        self.last_wait = wait
        if type(self).reject_wait_timeout and wait not in (None, True, False):
            raise TypeError("wait must be bool or None")
        if type(self).fail_captures > 0:
            type(self).fail_captures -= 1
            raise type(self).capture_exc_type("Camera disconnected")
        if type(self).empty_frames > 0:
            type(self).empty_frames -= 1
            return None
        self.captures += 1
        self.outstanding += 1
        return FakeRequest(self, type(self).sensor_timestamp)


class FakeRequest:
    def __init__(self, camera, sensor_timestamp):
        self.camera = camera
        self.sensor_timestamp = sensor_timestamp
        self.released = False

    def make_array(self, name="main"):
        return np.zeros((6, 8, 3), dtype=np.uint8)

    def get_metadata(self):
        if self.sensor_timestamp is None:
            return {}
        return {"SensorTimestamp": self.sensor_timestamp}

    def release(self):
        self.released = True
        self.camera.outstanding -= 1


@pytest.fixture
def fake_picamera2(monkeypatch):
    FakePicamera2.instances = []
    FakePicamera2.fail_opens = 0
    FakePicamera2.fail_captures = 0
    FakePicamera2.empty_frames = 0
    FakePicamera2.reject_wait_timeout = False
    FakePicamera2.capture_exc_type = RuntimeError
    FakePicamera2.sensor_timestamp = None
    mod = types.ModuleType("picamera2")
    mod.Picamera2 = FakePicamera2
    monkeypatch.setitem(sys.modules, "picamera2", mod)
    monkeypatch.setattr("solarian_picamera.bus.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "solarian_picamera.bus.release_leftover_cameras", lambda **_k: None
    )
    return FakePicamera2


def _bus(**cfg_overrides) -> PiCameraBus:
    bus = PiCameraBus(
        port="0", components={"camera": SimpleNamespace(cfg=_cfg(**cfg_overrides))}
    )
    bus.ae_settle_s = 0.0
    bus.reconnect_settle_s = 0.0
    bus.reconnect_min_delay_s = 0.0
    bus.reconnect_max_delay_s = 0.0
    bus.close_timeout_s = 1.0
    return bus


def test_read_returns_frame(fake_picamera2):
    bus = _bus()
    bus.connect()
    out = bus.read(ImageData, ["camera"])
    assert "camera" in out
    assert out["camera"].shape == (6, 8, 3)
    assert fake_picamera2.instances[0].captures == 1
    assert fake_picamera2.instances[0].last_wait == bus.capture_timeout_s
    assert fake_picamera2.instances[0].config["buffer_count"] == 6


def test_the_sensors_own_timestamp_dates_the_capture(fake_picamera2):
    """A stamp taken on arrival jitters with the scheduler; this one does not."""
    bus = _bus()
    bus.connect()
    fake_picamera2.sensor_timestamp = int(
        (time.clock_gettime(_SENSOR_CLOCK) - 0.040) * 1e9
    )

    bus.read(ImageData, ["camera"])

    assert bus.measured_capture_age == pytest.approx(0.040, abs=5e-3)


def test_a_camera_that_does_not_date_its_frames_falls_back(fake_picamera2):
    """Without a sensor stamp the configured latency is all there is."""
    bus = _bus()
    bus.connect()
    fake_picamera2.sensor_timestamp = None

    bus.read(ImageData, ["camera"])

    assert bus.measured_capture_age is None


def test_every_capture_hands_its_buffer_back(fake_picamera2):
    """The pool is a handful of frames deep; one leak stalls the pipeline."""
    bus = _bus()
    bus.connect()

    for _ in range(8):
        bus.read(ImageData, ["camera"])

    assert fake_picamera2.instances[0].outstanding == 0


def test_a_buffer_is_handed_back_even_when_the_frame_is_unusable(
    fake_picamera2, monkeypatch
):
    """The failure path is the one that leaks, being the one nobody runs."""
    bus = _bus()
    bus.connect()
    camera = fake_picamera2.instances[0]

    def explode(self, name="main"):
        raise RuntimeError("mapping failed")

    monkeypatch.setattr(FakeRequest, "make_array", explode)
    with pytest.raises(RuntimeError, match="mapping failed"):
        bus._capture_frame()

    assert camera.outstanding == 0


def test_rejects_non_image_data(fake_picamera2):
    bus = _bus()
    bus.connect()
    with pytest.raises(ValueError, match="ImageData"):
        bus.read(PosData, ["camera"])


def test_reconnects_after_capture_error(fake_picamera2):
    bus = _bus()
    bus.connect()
    fake_picamera2.fail_captures = 1
    out = bus.read(ImageData, ["camera"])
    assert out["camera"].shape == (6, 8, 3)
    assert len(fake_picamera2.instances) == 2
    assert fake_picamera2.instances[0].closed is True
    assert fake_picamera2.instances[0].cancelled is True
    assert fake_picamera2.instances[1].captures == 1


def test_reconnects_after_empty_frame(fake_picamera2):
    bus = _bus()
    bus.connect()
    fake_picamera2.empty_frames = 1
    out = bus.read(ImageData, ["camera"])
    assert out["camera"].shape == (6, 8, 3)
    assert len(fake_picamera2.instances) == 2


def test_retries_open_until_camera_returns(fake_picamera2):
    bus = _bus()
    bus.connect()
    fake_picamera2.fail_captures = 1
    fake_picamera2.fail_opens = 2
    out = bus.read(ImageData, ["camera"])
    assert out["camera"].shape == (6, 8, 3)
    # original + 2 failed opens + 1 successful reopen
    assert len(fake_picamera2.instances) == 4
    assert fake_picamera2.instances[-1].captures == 1


def test_reconnects_after_capture_timeout(fake_picamera2):
    bus = _bus()
    bus.connect()
    fake_picamera2.capture_exc_type = TimeoutError
    fake_picamera2.fail_captures = 1
    out = bus.read(ImageData, ["camera"])
    assert out["camera"].shape == (6, 8, 3)
    assert len(fake_picamera2.instances) == 2


def test_a_configured_remap_does_not_warp_the_capture(fake_picamera2):
    """The camera runs for everyone; only a policy wants GoPro optics.

    A path that does not exist cannot be loaded, so a bus that still warped
    would fail here rather than quietly cost a warp per frame.
    """
    bus = _bus(optical_remap="conf/calibrations/nothing_here.json")
    bus.connect()

    out = bus.read(ImageData, ["camera"])

    assert out["camera"].shape == (6, 8, 3)


def test_capture_falls_back_when_wait_timeout_unsupported(fake_picamera2):
    fake_picamera2.reject_wait_timeout = True
    bus = _bus()
    bus.connect()
    out = bus.read(ImageData, ["camera"])
    assert out["camera"].shape == (6, 8, 3)


def test_connect_is_idempotent(fake_picamera2):
    bus = _bus()
    bus.connect()
    bus.connect()
    assert len(fake_picamera2.instances) == 1


def test_port_must_be_csi_index(fake_picamera2):
    bus = PiCameraBus(
        port="/dev/video0",
        components={"camera": SimpleNamespace(cfg=_cfg(port="/dev/video0"))},
    )
    with pytest.raises(ValueError, match="CSI camera index"):
        bus.connect()


def test_default_controls_are_only_framerate(fake_picamera2):
    bus = _bus()
    bus.connect()
    assert fake_picamera2.instances[0].config["controls"] == {"FrameRate": 30.0}


def test_isp_controls_map_to_libcamera(fake_picamera2):
    bus = PiCameraBus(
        port="0",
        components={
            "camera": SimpleNamespace(
                cfg=_cfg(
                    exposure_value=-1.0,
                    brightness=-0.1,
                    contrast=1.3,
                    shutter=8000,
                    gain=1.5,
                )
            )
        },
    )
    bus.ae_settle_s = 0.0
    bus.connect()
    assert fake_picamera2.instances[0].config["controls"] == {
        "FrameRate": 30.0,
        "ExposureValue": -1.0,
        "Brightness": -0.1,
        "Contrast": 1.3,
        "ExposureTime": 8000,
        "AnalogueGain": 1.5,
        "AeEnable": False,
    }


def test_pi_camera_config_keeps_isp_fields():
    cfg = PiCameraConfig(
        frequency=100,
        sensors={
            "camera": {
                "port": "0",
                "fps": 30,
                "width": 8,
                "height": 6,
                "rotation": 0,
                "channels": 3,
                "color_mode": "rgb",
                "video_format": "MJPG",
                "resize_width": None,
                "resize_height": None,
                "exposure_value": -1.0,
                "brightness": -0.1,
            }
        },
    )
    cam = cfg.sensors["camera"]
    assert cam.exposure_value == -1.0
    assert cam.brightness == -0.1
    assert cam.contrast is None
    assert cam.shutter is None
    assert cam.gain is None


def _hold(proc, pid: int, device, cmdline: bytes, pgrp: int) -> None:
    entry = proc / str(pid)
    fd_dir = entry / "fd"
    fd_dir.mkdir(parents=True)
    (fd_dir / "3").symlink_to(device)
    (entry / "cmdline").write_bytes(cmdline)
    (entry / "stat").write_text(f"{pid} (python3 x) S 1 {pgrp} {pgrp} 0 -1\n")


def test_connect_releases_leftover_holders_before_opening(fake_picamera2, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "solarian_picamera.bus.release_leftover_cameras",
        lambda **_k: calls.append(len(fake_picamera2.instances)),
    )
    _bus().connect()
    assert calls == [0]


def test_release_leftover_cameras_signals_only_holders_from_another_run(tmp_path):
    proc = tmp_path / "proc"
    dev = tmp_path / "dev"
    sysfs = tmp_path / "sys"
    dev.mkdir()
    media = dev / "media0"
    usb = dev / "video0"
    csi_video = dev / "video1"
    for node in (media, usb, csi_video):
        node.write_bytes(b"")
    (sysfs / "class" / "video4linux" / "video0").mkdir(parents=True)
    (sysfs / "class" / "video4linux" / "video0" / "name").write_text("USB Camera\n")
    (sysfs / "class" / "video4linux" / "video1").mkdir(parents=True)
    (sysfs / "class" / "video4linux" / "video1" / "name").write_text("rp1-cfe\n")

    me = os.getpid()
    own_group = os.getpgid(0)
    old_group = own_group + 1
    sibling = 1_000_001
    holder = 4242
    usb_holder = 4243
    _hold(proc, me, media, b"solarian\x00run-robot\x00", own_group)
    _hold(proc, sibling, media, b"other-camera-module\x00", own_group)
    _hold(proc, holder, media, b"solarian\x00run-robot\x00-d\x00pi\x00", old_group)
    _hold(proc, usb_holder, usb, b"some-webcam\x00", old_group)
    _hold(proc, 4244, csi_video, b"old-camera-child\x00", old_group)

    signaled: list[tuple[int, int]] = []

    def kill(pid, sig):
        signaled.append((pid, sig))

    release_leftover_cameras(
        proc_root=proc,
        sys_root=sysfs,
        dev_root=dev,
        kill=kill,
        sleep=lambda _seconds: None,
    )

    assert signaled == [
        (holder, signal.SIGTERM),
        (4244, signal.SIGTERM),
        (holder, signal.SIGKILL),
        (4244, signal.SIGKILL),
    ]
    assert all(pid not in (me, sibling, usb_holder) for pid, _sig in signaled)
