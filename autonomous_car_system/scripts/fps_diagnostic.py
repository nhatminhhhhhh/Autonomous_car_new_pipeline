import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
import sys as _sys

src = 0
N = 60


def fourcc_str(val):
    val = int(val) & 0xFFFFFFFF
    return val.to_bytes(4, 'little').decode('ascii', errors='replace')


def measure_fps(cap, n=N):
    for _ in range(5):
        cap.read()
    t0 = time.perf_counter()
    ok = 0
    for _ in range(n):
        ret, _ = cap.read()
        if ret:
            ok += 1
    return ok / (time.perf_counter() - t0)


def measure_grab_fps(cap, n=N):
    for _ in range(5):
        cap.grab()
    t0 = time.perf_counter()
    ok = 0
    for _ in range(n):
        if cap.grab():
            ok += 1
    return ok / (time.perf_counter() - t0)


def test(label, backend, w, h, fps_target, fourcc=None):
    cap = cv2.VideoCapture(src, backend)
    if not cap.isOpened():
        print(f"  [{label}] FAIL")
        return

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps_target)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    af = cap.get(cv2.CAP_PROP_FPS)
    acc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))

    grab_fps = measure_grab_fps(cap)
    read_fps = measure_fps(cap)
    cap.release()

    print(f"  [{label:<28}] {aw}x{ah} codec={acc} reported={af:.0f}  "
          f"grab={grab_fps:.1f}fps  read={read_fps:.1f}fps")


def main():
    if _sys.platform.startswith('linux'):
        print("\n=== LINUX (V4L2) ===")
        test("V4L2_MJPG", cv2.CAP_V4L2, 640, 480, 60, fourcc='MJPG')
        test("V4L2_YUYV", cv2.CAP_V4L2, 640, 480, 60, fourcc='YUYV')
        test("V4L2_MJPG_320", cv2.CAP_V4L2, 320, 240, 60, fourcc='MJPG')
    else:
        print("\n=== WINDOWS ===")
        test("DSHOW_MJPG", cv2.CAP_DSHOW, 640, 480, 60, fourcc='MJPG')
        test("MSMF_default", cv2.CAP_MSMF, 640, 480, 60)
        test("DSHOW_MJPG_320", cv2.CAP_DSHOW, 320, 240, 60, fourcc='MJPG')

    print("\nNote: grab_fps = camera raw output, read_fps = after decode")
    print("      Higher read_fps is better for inference pipeline")


if __name__ == '__main__':
    main()
