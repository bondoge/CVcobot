#!/usr/bin/env python3
"""
stereo_mediapipe_hands.py
---------------------------------
Detect hands from a Waveshare stereo camera with MediaPipe (Python + OpenCV).

Two input modes:
  1) --mode sbs      : single UVC device that outputs a side-by-side frame (left|right)
  2) --mode dual     : two separate UVC devices (use --left, --right)

Quick start (Windows / VS Code):
  python -m venv .venv
  .venv\Scripts\activate
  pip install --upgrade pip
  pip install opencv-python mediapipe numpy
  python stereo_mediapipe_hands.py --mode sbs --cam 0 --size 2560x720 --flip

Typical Waveshare stereo resolutions (side-by-side):
  1280x480  -> two 640x480 streams
  2560x720  -> two 1280x720 streams
Pick the one your camera supports and pass it via --size.

Keys:
  q/ESC  : quit
  f      : toggle flip (mirror) view
  d      : toggle drawing landmarks
  p      : pause/resume
  h      : toggle help overlay

Author: ChatGPT
"""

import argparse
import time
from dataclasses import dataclass
import sys

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] mediapipe is not installed. Run: pip install mediapipe", file=sys.stderr)
    sys.exit(1)


@dataclass
class StereoConfig:
    mode: str                # 'sbs' or 'dual'
    cam: int = 0             # camera index for sbs mode
    left: int = 0            # left camera index for dual mode
    right: int = 1           # right camera index for dual mode
    size: tuple = (2560, 720)  # capture size (W,H) for sbs or per-device desired size for dual
    fps: int = 30
    flip: bool = True        # mirror preview (useful if cams are facing user)
    draw: bool = True        # draw landmarks
    pause: bool = False
    backend: int = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0  # Win: prefer DirectShow


class StereoSource:
    """Abstraction over stereo inputs (side-by-side or two devices)."""

    def __init__(self, cfg: StereoConfig):
        self.cfg = cfg
        self.capL = None
        self.capR = None
        self.sbs_cap = None

        if cfg.mode == "sbs":
            self.sbs_cap = cv2.VideoCapture(cfg.cam, cfg.backend)
            if not self.sbs_cap.isOpened():
                raise RuntimeError(f"Failed to open camera index {cfg.cam}")
            # Try to set resolution & fps (best-effort; may be ignored by driver)
            self.sbs_cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.size[0])
            self.sbs_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.size[1])
            self.sbs_cap.set(cv2.CAP_PROP_FPS, cfg.fps)

        elif cfg.mode == "dual":
            self.capL = cv2.VideoCapture(cfg.left, cfg.backend)
            self.capR = cv2.VideoCapture(cfg.right, cfg.backend)
            if not self.capL.isOpened():
                raise RuntimeError(f"Failed to open LEFT camera index {cfg.left}")
            if not self.capR.isOpened():
                raise RuntimeError(f"Failed to open RIGHT camera index {cfg.right}")
            for cap in (self.capL, self.capR):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.size[1])
                cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        else:
            raise ValueError("cfg.mode must be 'sbs' or 'dual'")

    def read(self):
        """Return (left_frame, right_frame) BGR images or (None, None) on failure."""
        if self.cfg.mode == "sbs":
            ok, frame = self.sbs_cap.read()
            if not ok or frame is None:
                return None, None
            h, w = frame.shape[:2]
            # Expecting side-by-side: left|right horizontally
            w2 = w // 2
            left = frame[:, :w2].copy()
            right = frame[:, w2:].copy()
            return left, right
        else:
            okL, left = self.capL.read()
            okR, right = self.capR.read()
            if not okL or not okR:
                return None, None
            return left, right

    def release(self):
        for cap in (self.capL, self.capR, self.sbs_cap):
            if cap is not None:
                cap.release()


def put_text(img, text, org=(10, 30)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_help(img):
    lines = [
        "Controls: q/ESC=quit  f=flip  d=draw landmarks  p=pause/resume  h=help",
    ]
    y = 30
    for ln in lines:
        put_text(img, ln, (10, y))
        y += 24


def main():
    parser = argparse.ArgumentParser(description="Stereo MediaPipe Hands on Waveshare stereo camera")
    parser.add_argument("--mode", choices=["sbs", "dual"], default="sbs",
                        help="Input mode: sbs=single side-by-side stream, dual=two devices")
    parser.add_argument("--cam", type=int, default=0, help="Camera index for sbs mode")
    parser.add_argument("--left", type=int, default=0, help="Left camera index for dual mode")
    parser.add_argument("--right", type=int, default=1, help="Right camera index for dual mode")
    parser.add_argument("--size", type=str, default="2560x720", help="Capture size, e.g., 2560x720 or 1280x480")
    parser.add_argument("--fps", type=int, default=30, help="Desired FPS")
    parser.add_argument("--flip", action="store_true", help="Mirror the preview")
    parser.add_argument("--no-draw", action="store_true", help="Disable drawing landmarks")
    args = parser.parse_args()

    try:
        w, h = map(int, args.size.lower().split("x"))
    except Exception:
        print("[ERROR] --size must look like 2560x720", file=sys.stderr)
        sys.exit(2)

    cfg = StereoConfig(
        mode=args.mode,
        cam=args.cam,
        left=args.left,
        right=args.right,
        size=(w, h),
        fps=args.fps,
        flip=args.flip,
        draw=not args.no_draw,
    )

    # Init video
    try:
        src = StereoSource(cfg)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(3)

    # Init MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,             # 0=lite, 1=full, 2=heavy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    last_time = time.time()
    fps_smooth = None
    show_help = True
    paused = False
    draw_landmarks = cfg.draw
    flip = cfg.flip

    try:
        while True:
            if not paused:
                left, right = src.read()
                if left is None or right is None:
                    print("[WARN] Failed to read frames. Exiting...")
                    break

                if flip:
                    left = cv2.flip(left, 1)
                    right = cv2.flip(right, 1)

                # Process with MediaPipe (expects RGB)
                left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
                right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

                left_res = hands.process(left_rgb)
                right_res = hands.process(right_rgb)

                # Draw results
                if draw_landmarks:
                    if left_res.multi_hand_landmarks:
                        for hand_landmarks in left_res.multi_hand_landmarks:
                            mp_draw.draw_landmarks(
                                left,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_styles.get_default_hand_landmarks_style(),
                                mp_styles.get_default_hand_connections_style(),
                            )
                    if right_res.multi_hand_landmarks:
                        for hand_landmarks in right_res.multi_hand_landmarks:
                            mp_draw.draw_landmarks(
                                right,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_styles.get_default_hand_landmarks_style(),
                                mp_styles.get_default_hand_connections_style(),
                            )

                # FPS
                now = time.time()
                fps = 1.0 / max(1e-6, (now - last_time))
                last_time = now
                if fps_smooth is None:
                    fps_smooth = fps
                else:
                    fps_smooth = 0.9 * fps_smooth + 0.1 * fps

                put_text(left,  f"L | {int(left.shape[1])}x{int(left.shape[0])}  FPS: {fps_smooth:5.1f}")
                put_text(right, f"R | {int(right.shape[1])}x{int(right.shape[0])}  FPS: {fps_smooth:5.1f}")

                if show_help:
                    draw_help(left)
                    draw_help(right)

                cv2.imshow("Left - MediaPipe Hands", left)
                cv2.imshow("Right - MediaPipe Hands", right)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break
            elif key == ord('f'):
                flip = not flip
            elif key == ord('d'):
                draw_landmarks = not draw_landmarks
            elif key == ord('p'):
                paused = not paused
            elif key == ord('h'):
                show_help = not show_help

    finally:
        hands.close()
        src.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
