# Solarian-PiCamera

Raspberry Pi CSI camera (IMX477 HQ, Camera Module 3, ...) module, sensor and bus
plugin for the Solarian robotics framework, via
[picamera2](https://github.com/raspberrypi/picamera2).

## Installation

picamera2 is a system package on Raspberry Pi OS. Install it with apt, and
create the project venv with `--system-site-packages` so it is importable:

```bash
sudo apt install python3-picamera2
python3 -m venv --system-site-packages .venv
pip install solarian-picamera
```

Or install from source:

```bash
cd solarian-picamera
pip install -e .
```

The plugin imports on any machine (a laptop running the controller or the
simulator only needs the classes); picamera2 is imported when the camera opens.

## Usage

```python
from solarian.core.robot import Robot
from solarian_picamera import PiCameraModule


class MyRobot(Robot):
    wrist_camera = PiCameraModule()
```

The module's yaml names the CSI index as `port` (`0` = cam0, `1` = cam1), not a
`/dev/video*` path. Besides the usual camera fields, `PiCameraSensorConfig`
accepts libcamera ISP controls: `exposure_value`, `brightness`, `contrast`,
`shutter` and `gain`.

`solarian list-cameras` lists and previews CSI cameras once this plugin is
installed.

## Repository

https://github.com/solarian-dev/solarian-picamera
