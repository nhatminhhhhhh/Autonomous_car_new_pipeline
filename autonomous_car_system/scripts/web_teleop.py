import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import threading
import time
import cv2
from flask import Flask, Response, render_template_string, request, jsonify

from hardware.camera import CameraThread
from hardware.esp32_controller import ESP32Controller
from configs.config import CONFIG

app = Flask(__name__)

cam = None
esp32 = None
lock = threading.Lock()
current_steering = 0.0
current_throttle = 0.0


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Car Teleoperation</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', Tahoma, sans-serif;
    background: #1a1a2e;
    color: #eee;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    user-select: none;
  }
  h1 { margin: 16px 0 8px; font-size: 1.4rem; letter-spacing: 1px; color: #e94560; }

  .main-container {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 40px;
    padding: 10px;
    flex: 1;
  }

  .stream-section {
    flex-shrink: 0;
  }
  .stream-section img {
    width: 640px;
    height: 480px;
    border: 2px solid #333;
    border-radius: 8px;
    background: #000;
    display: block;
  }
  .status-bar {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 0.85rem;
    color: #aaa;
  }
  .status-bar .connected { color: #4ade80; }
  .status-bar .disconnected { color: #f87171; }

  .control-section {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 40px;
  }

  .slider-group {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }
  .slider-group label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
  }
  .slider-group .value {
    font-size: 1.6rem;
    font-weight: bold;
    font-family: 'Courier New', monospace;
    min-width: 70px;
    text-align: center;
  }

  /* Steering — horizontal */
  .steering-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }
  input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    background: transparent;
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-runnable-track {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(to right, #f87171, #fbbf24, #4ade80);
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    height: 28px;
    width: 28px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid #333;
    margin-top: -10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
  }
  input[type=range]::-moz-range-track {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(to right, #f87171, #fbbf24, #4ade80);
  }
  input[type=range]::-moz-range-thumb {
    height: 24px;
    width: 24px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid #333;
  }

  #steering-slider {
    width: 300px;
  }

  /* Throttle — vertical */
  .throttle-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }
  #throttle-slider {
    writing-mode: vertical-lr;
    direction: rtl;
    height: 300px;
    width: 40px;
  }
  #throttle-slider::-webkit-slider-runnable-track {
    height: 100%;
    width: 8px;
    background: linear-gradient(to top, #f87171, #fbbf24, #4ade80);
    border-radius: 4px;
  }
  #throttle-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    height: 28px;
    width: 28px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid #333;
    margin-left: -10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
  }
  #throttle-slider::-moz-range-track {
    width: 8px;
    background: linear-gradient(to top, #f87171, #fbbf24, #4ade80);
    border-radius: 4px;
  }
  #throttle-slider::-moz-range-thumb {
    height: 24px;
    width: 24px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid #333;
  }

  .steering-value { color: #fbbf24; }
  .throttle-value { color: #60a5fa; }

  .estop-btn {
    margin-top: 12px;
    padding: 10px 28px;
    font-size: 1rem;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: #dc2626;
    color: #fff;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.15s;
  }
  .estop-btn:hover { background: #b91c1c; }
  .estop-btn:active { background: #991b1b; }

  footer {
    margin-top: auto;
    padding: 12px;
    font-size: 0.75rem;
    color: #555;
  }

  @media (max-width: 1100px) {
    .main-container { flex-direction: column; gap: 20px; }
    .stream-section img { width: 100%; max-width: 480px; height: auto; }
  }
</style>
</head>
<body>

<h1>&#9881; Car Teleoperation</h1>

<div class="main-container">
  <div class="stream-section">
    <img id="stream" src="/video_feed" alt="Camera Feed">
    <div class="status-bar">
      <span id="esp-status" class="disconnected">ESP32: DISCONNECTED</span>
      <span id="fps-display">FPS: --</span>
    </div>
  </div>

  <div class="control-section">
    <div class="slider-group">
      <label>Steering</label>
      <div class="steering-wrap">
        <span id="steering-val" class="value steering-value">0.00</span>
        <input type="range" id="steering-slider" min="-100" max="100" value="0" step="1">
        <div style="display:flex;justify-content:space-between;width:300px;font-size:0.65rem;color:#666;">
          <span>Left</span><span>Center</span><span>Right</span>
        </div>
      </div>
    </div>

    <div class="slider-group">
      <label>Throttle</label>
      <div class="throttle-wrap">
        <input type="range" id="throttle-slider" min="-100" max="100" value="0" step="1">
        <span id="throttle-val" class="value throttle-value">0.00</span>
        <div style="display:flex;flex-direction:column;align-items:center;font-size:0.65rem;color:#666;">
          <span>Forward</span><span style="margin:4px 0">Stop</span><span>Reverse</span>
        </div>
      </div>
    </div>
  </div>
</div>

<button class="estop-btn" id="estop-btn">&#9632; Emergency Stop</button>

<footer>Autonomous Car System &mdash; WiFi Teleoperation</footer>

<script>
const steeringSlider = document.getElementById('steering-slider');
const throttleSlider = document.getElementById('throttle-slider');
const steeringVal = document.getElementById('steering-val');
const throttleVal = document.getElementById('throttle-val');
const estopBtn = document.getElementById('estop-btn');
const espStatus = document.getElementById('esp-status');

let commandTimer = null;
const COMMAND_INTERVAL = 50;

function sendCommand() {
  const s = parseInt(steeringSlider.value);
  const t = parseInt(throttleSlider.value);
  fetch('/command', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ steering: s / 100.0, throttle: t / 100.0 })
  }).catch(() => {});
}

function updateDisplay() {
  const s = parseInt(steeringSlider.value) / 100;
  const t = parseInt(throttleSlider.value) / 100;
  steeringVal.textContent = s.toFixed(2);
  throttleVal.textContent = t.toFixed(2);
}

function scheduleCommand() {
  updateDisplay();
  if (commandTimer) clearTimeout(commandTimer);
  commandTimer = setTimeout(() => {
    sendCommand();
    commandTimer = null;
  }, COMMAND_INTERVAL);
}

steeringSlider.addEventListener('input', scheduleCommand);
throttleSlider.addEventListener('input', scheduleCommand);

document.addEventListener('keydown', (e) => {
  const s = parseInt(steeringSlider.value);
  const t = parseInt(throttleSlider.value);
  let changed = false;
  switch (e.key) {
    case 'ArrowLeft':  steeringSlider.value = Math.max(-100, s - 5); changed = true; break;
    case 'ArrowRight': steeringSlider.value = Math.min(100, s + 5); changed = true; break;
    case 'ArrowUp':    throttleSlider.value = Math.min(100, t + 5); changed = true; break;
    case 'ArrowDown':  throttleSlider.value = Math.max(-100, t - 5); changed = true; break;
    case ' ': e.preventDefault(); steeringSlider.value = 0; throttleSlider.value = 0; changed = true; break;
  }
  if (changed) scheduleCommand();
});

estopBtn.addEventListener('click', () => {
  steeringSlider.value = 0;
  throttleSlider.value = 0;
  updateDisplay();
  fetch('/command', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ steering: 0, throttle: 0 })
  });
  fetch('/estop', { method: 'POST' });
});

setInterval(() => {
  fetch('/status')
    .then(r => r.json())
    .then(d => {
      const el = espStatus;
      if (d.esp32_connected) {
        el.textContent = 'ESP32: CONNECTED';
        el.className = 'connected';
      } else {
        el.textContent = 'ESP32: DISCONNECTED';
        el.className = 'disconnected';
      }
      const fpsEl = document.getElementById('fps-display');
      if (d.fps !== undefined) fpsEl.textContent = `FPS: ${d.fps}`;
    })
    .catch(() => {});
}, 2000);

window.addEventListener('beforeunload', () => {
  navigator.sendBeacon('/command', JSON.stringify({ steering: 0, throttle: 0 }));
});
</script>
</body>
</html>
"""


def gen_frames():
    global cam
    fps_counter = 0
    fps_time = time.perf_counter()
    fps_val = 0.0

    while True:
        frame = cam.read_nowait()
        if frame is None:
            time.sleep(0.005)
            continue

        fps_counter += 1
        now = time.perf_counter()
        if now - fps_time >= 1.0:
            fps_val = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/command', methods=['POST'])
def command():
    global current_steering, current_throttle
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'invalid JSON'}), 400

    s = float(data.get('steering', 0.0))
    t = float(data.get('throttle', 0.0))

    with lock:
        current_steering = s
        current_throttle = t
        if esp32 is not None:
            esp32.send_command(s, t)

    return jsonify({'steering': s, 'throttle': t})


@app.route('/estop', methods=['POST'])
def estop():
    global current_steering, current_throttle
    with lock:
        current_steering = 0.0
        current_throttle = 0.0
        if esp32 is not None:
            esp32.send_command(0.0, 0.0)
    return jsonify({'status': 'estop'}), 200


@app.route('/status')
def status():
    with lock:
        connected = esp32 is not None and esp32.ser is not None and esp32.ser.is_open
    return jsonify({
        'esp32_connected': connected,
        'steering': current_steering,
        'throttle': current_throttle,
    })


def parse_args():
    parser = argparse.ArgumentParser(description='Web Teleoperation Server')
    parser.add_argument('--port', type=str, default=None,
                        help='ESP32 serial port (auto-detect if not set)')
    parser.add_argument('--baud', type=int, default=115200,
                        help='ESP32 serial baud rate')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Flask host')
    parser.add_argument('--http-port', type=int, default=5000,
                        help='Flask port')
    parser.add_argument('--camera', type=int, default=CONFIG['camera_index'],
                        help='Camera index')
    parser.add_argument('--camera-width', type=int, default=CONFIG['camera_width'],
                        help='Camera width')
    parser.add_argument('--camera-height', type=int, default=CONFIG['camera_height'],
                        help='Camera height')
    return parser.parse_args()


def main():
    global cam, esp32

    args = parse_args()

    print("[WebTeleop] Initializing camera...")
    cam = CameraThread(args.camera, args.camera_width, args.camera_height)
    print(f"[WebTeleop] Camera started ({args.camera_width}x{args.camera_height})")

    print("[WebTeleop] Connecting to ESP32...")
    esp32 = ESP32Controller(port=args.port, baud=args.baud)

    print(f"[WebTeleop] Starting Flask server on {args.host}:{args.http_port}")
    print(f"[WebTeleop] Open http://{args.host}:{args.http_port} in your browser")
    try:
        app.run(host=args.host, port=args.http_port, threaded=True, debug=False)
    finally:
        if esp32 is not None:
            esp32.close()
        if cam is not None:
            cam.stop()


if __name__ == '__main__':
    main()
