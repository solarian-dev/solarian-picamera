"""
Solarian-PiCamera Plugin

Raspberry Pi CSI camera module, sensor and bus for the Solarian robotics
framework, via picamera2.
"""

from solarian_picamera.camera import PiCamera
from solarian_picamera.config import PiCameraSensorConfig
from solarian_picamera.module import PiCameraConfig, PiCameraModule

__version__ = "0.1.0"

__all__ = ["PiCamera", "PiCameraConfig", "PiCameraModule", "PiCameraSensorConfig"]
