"""
``solarian list-cameras`` backend for CSI cameras.

Registered under the ``solarian.camera_backends`` entry point as ``picamera2``,
which is the ``backend`` value every entry from :func:`find` carries.
"""

import os


def _quiet_libcamera() -> None:
    # Read once, when libcamera first loads inside the picamera2 import.
    os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:ERROR")


def find() -> list[dict]:
    """CSI / libcamera cameras via picamera2. USB cameras are listed via V4L2 instead."""
    _quiet_libcamera()
    try:
        from picamera2 import Picamera2
    except ImportError:
        return []

    try:
        infos = Picamera2.global_camera_info()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: picamera2 available but global_camera_info() failed: {e}")
        return []

    cameras = []
    for i, info in enumerate(infos):
        cam_id = str(info.get("Id", ""))
        if "usb@" in cam_id:
            continue
        model = info.get("Model", "unknown")
        camera_num = info.get("Num", i)
        cameras.append({
            "port": str(camera_num),
            "backend": "picamera2",
            "model": model,
        })
    return cameras


def iter_frames(port: str, width: int | None = None, height: int | None = None):
    """Yield BGR frames from CSI camera ``port`` until the caller closes the generator."""
    _quiet_libcamera()
    from picamera2 import Picamera2

    camera_num = int(port)
    picam2 = Picamera2(camera_num=camera_num)
    try:
        if width and height:
            size = (int(width), int(height))
        else:
            modes = getattr(picam2, "sensor_modes", None) or []
            if modes:
                best = max(modes, key=lambda m: m["size"][0] * m["size"][1])
                size = tuple(best["size"])
            else:
                size = tuple(picam2.sensor_resolution)

        try:
            config = picam2.create_video_configuration(main={"size": size, "format": "RGB888"})
            picam2.configure(config)
        except Exception:
            config = picam2.create_still_configuration(main={"size": size, "format": "RGB888"})
            picam2.configure(config)

        picam2.start()
        cfg_size = picam2.camera_configuration()["main"]["size"]
        print(f"Opening picamera2:{port} at {cfg_size[0]}x{cfg_size[1]}.")

        while True:
            frame = picam2.capture_array("main")
            if frame is None:
                raise RuntimeError(f"Could not read a frame from picamera2 camera {port}")
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[..., :3]
            # picamera2 RGB888 is already B,G,R in memory (OpenCV BGR). Do not swap.
            yield frame
    finally:
        try:
            picam2.stop()
            picam2.close()
        except Exception:  # noqa: BLE001
            pass
