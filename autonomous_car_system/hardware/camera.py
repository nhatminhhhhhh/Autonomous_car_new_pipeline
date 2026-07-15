import cv2
import threading
import queue
import sys


def detect_platform():
    if sys.platform.startswith('linux'):
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                return 'jetson'
        except (FileNotFoundError, PermissionError, IOError):
            pass
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'tegra' in model.lower() or 'jetson' in model.lower():
                    return 'jetson'
        except (FileNotFoundError, PermissionError, IOError):
            pass
        return 'linux'
    return 'windows'


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=480,
    display_height=360,
    framerate=120,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink max-buffers=2 drop=true sync=false"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def open_camera(src=0, width=640, height=480, fps=60):
    plat = detect_platform()

    if plat == 'jetson':
        pipeline = gstreamer_pipeline(
            sensor_id=src,
            capture_width=1280,
            capture_height=720,
            display_width=width,
            display_height=height,
            framerate=120,
            flip_method=2,
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[Camera] IMX219 GStreamer | display {width}x{height} (capture 1280x720 @ 120fps)")
            return cap

    if plat == 'windows':
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        for backend in backends:
            cap = cv2.VideoCapture(src, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"[Camera] Windows backend {backend} | {width}x{height}")
                return cap

    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        print(f"[Camera] Default fallback")
        return cap

    raise RuntimeError(f"Cannot open camera {src}")


class CameraThread:
    def __init__(self, src=0, width=640, height=480, fps=60, max_queue_size=2):
        self.cap = open_camera(src, width, height, fps)
        self.q = queue.Queue(maxsize=max_queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def read_nowait(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()
