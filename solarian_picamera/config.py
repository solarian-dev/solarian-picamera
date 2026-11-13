from pydantic import Field

from solarian.core.configs import CameraConfig


class PiCameraSensorConfig(CameraConfig):
    """libcamera / Picamera2 controls for a Raspberry Pi CSI camera.

    Unset fields leave the driver default (auto-exposure and default ISP).

    Docs: https://solarian.dev/docs/sensors
    """

    #: EV stops for AEC/AGC (``rpicam-hello --ev``).
    exposure_value: float | None = None
    #: Pixel offset after capture, -1.0 (black) to 1.0 (white).
    brightness: float | None = Field(default=None, ge=-1.0, le=1.0)
    #: 1.0 is the default amount of contrast.
    contrast: float | None = Field(default=None, gt=0)
    #: Exposure time in microseconds. Gain can still vary unless ``gain`` is also set.
    shutter: int | None = Field(default=None, gt=0)
    #: Analogue gain. Exposure can still vary unless ``shutter`` is also set.
    gain: float | None = Field(default=None, gt=0)
