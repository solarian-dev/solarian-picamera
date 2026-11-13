"""
PiCamera Module

Module for Raspberry Pi CSI cameras via picamera2.

On the Pi: ``sudo apt install python3-picamera2`` and create the project
venv with ``--system-site-packages`` so the system package is importable.
``port`` is the CSI index (``0`` = cam0, ``1`` = cam1), not ``/dev/video*``.
"""

from solarian.core.module import Module
from solarian.core.configs import ModuleConfig
from solarian_picamera.camera import PiCamera
from solarian_picamera.config import PiCameraSensorConfig


class PiCameraConfig(ModuleConfig):
    calibration_file: str | None = None
    # libcamera / picamera2 first start (configure + AE settle + first
    # 1080p capture) routinely exceeds the generic 3s module timeout.
    launch_timeout: float = 15.0
    sensors: dict[str, PiCameraSensorConfig] = {}


class PiCameraModule(Module):
    """Raspberry Pi CSI camera module (picamera2)."""

    kind: str = "camera"

    config_class = PiCameraConfig

    camera = PiCamera()

    def control_step(self):
        """Sensor-only: frames are published by ``update_observations``."""
