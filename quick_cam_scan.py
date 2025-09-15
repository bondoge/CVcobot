# quick_cam_scan.py
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # on mac use CAP_AVFOUNDATION, on Linux leave backend default
    ok = cap.isOpened()
    print(f"{i}: {'OPEN' if ok else '---'}")
    cap.release()
