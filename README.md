# Blinker - Your face is the mouse

## Quick Start Tutorial

### Requirements

- WebCam
- Python 3.11
- CMake (at least 3.10)

### Setup

1. **Create a python env**
   ```bash
   python -m venv BlinkerEnv
   BlinkerEnv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Build C++ server**
   ```bash
   mkdir build
   cmake -S . -B build
   cmake --build build --config Release
   ```
3. **Run Blinker**
   ```bash
   python blinker.ev
   ```

## Demos

### CS2

Trig based cursor inspired by Jason Orlosky: https://www.youtube.com/watch?v=hImmJDTgXjw
![alt text](misc/v2.gif)

![alt text](misc/M4.gif)
