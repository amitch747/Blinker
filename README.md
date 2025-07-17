# Blinker - Your face is the mouse

![Tests](https://github.com/amitch747/Blinker/actions/workflows/tests.yaml/badge.svg)

## Controls

- Face center translation applies movement commands to the cursor
- Left blink - Left click
- Right blink - Right click
- 'm' - Toggle cursor
- 't' - Toggle left click
- 'c' - Calibrate cursor and EAR thresholds
- 'q' - Quit

## Troubleshooting

- False positive blinks - Close both eyes and press 'c' to recalibrate blink detection
- Bad mouse speed - Play with the variables in settings.py
- Camera problems - If you have multiple devices, change CAMERA_INDEX in settings.py
- CMake problems - Go to chatgpt.com

## Quick Start Tutorial

### Requirements

- Python 3.11 - https://www.python.org/downloads/release/python-3119/
- CMake 3.10 or newer - https://cmake.org/download/
- WebCam
- Windows 10/11 - https://en.wikipedia.org/wiki/Bill_Gates

### Setup

1. **Create python environment**
   ```bash
   py 3.11 -m venv BlinkerEnv
   BlinkerEnv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Build C++ server**

   1. Install Visual Studio Community Edition
   2. Under Workloads check 'Desktop development with C++'
   3. Under Individual components check 'MSVC V143', 'Windows 10/11 SDK', and 'C++ CMake tools'

   ```bash
   mkdir build
   cmake -S . -B build
   cmake --build build --config Release
   ```

3. **Run Blinker**
   ```bash
   python blinker.py
   ```

## Demos

### CS2

Trig based cursor inspired by Jason Orlosky: https://www.youtube.com/watch?v=hImmJDTgXjw
![alt text](misc/v2.gif)

![alt text](misc/M4.gif)
