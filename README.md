python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install opencv-python mediapipe numpy

Plug the Waveshare stereo camera.

Run the script:
python stereo_mediapipe_hands.py --mode sbs --cam 0 --size 2560x720 --flip

or
If your device enumerates as two cameras, run:
python stereo_mediapipe_hands.py --mode dual --left 0 --right 1 --size 1280x720 --flip


Tips
• Common SxS sizes: 1280x480 (2× 640×480) or 2560x720 (2× 1280×720).
• If window is mirrored, toggle with f.
• Keys: q/ESC quit, f flip, d draw on/off, p pause, h help overlay.
