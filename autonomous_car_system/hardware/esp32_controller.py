import serial
import serial.tools.list_ports
import threading
import time


KNOWN_VID_PID = {
    (0x10C4, 0xEA60),  # CP2102/CP2104
    (0x1A86, 0x7523),  # CH340
    (0x0403, 0x6001),  # FT232/FTDI
    (0x2E8A, 0x000A),  # Raspberry Pi Pico (CDC)
    (0x303A, 0x1001),  # ESP32-S3
    (0x303A, 0x0002),  # ESP32-S2
    (0x1A86, 0x55D4),  # CH340 variant
    (0x10C4, 0xEA70),  # CP2105
}


class ESP32Controller:
    def __init__(self, port=None, baud=115200, timeout=0.1, reset_esp=True, connect_retries=3):
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self.lock = threading.Lock()
        self._running = False
        self._read_thread = None
        self.reset_esp = reset_esp

        if port is None:
            port = self._auto_detect_port()

        self.port = port

        if self.port:
            for attempt in range(1, connect_retries + 1):
                if self._connect():
                    break
                print(f"[ESP32] Connection attempt {attempt}/{connect_retries} failed")
                if attempt < connect_retries:
                    time.sleep(1.5)
        else:
            print("[ESP32] No port specified and auto-detect failed — simulation mode")

    def connected(self):
        return self.ser is not None and self.ser.is_open

    def _auto_detect_port(self):
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("[ESP32] No serial ports found")
            return None

        print(f"[ESP32] Scanning {len(ports)} serial port(s)...")
        for p in ports:
            if (p.vid, p.pid) in KNOWN_VID_PID:
                print(f"[ESP32] Auto-detected: {p.device} ({p.description}) [VID:0x{p.vid:04X} PID:0x{p.pid:04X}]")
                return p.device

        for p in ports:
            desc_lower = p.description.lower()
            if any(k in desc_lower for k in ('usb', 'uart', 'cp210', 'ch340', 'ftdi', 'serial')):
                print(f"[ESP32] Auto-detected: {p.device} ({p.description})")
                return p.device

        for p in ports:
            if p.vid is not None or p.pid is not None:
                print(f"[ESP32] Found serial device: {p.device} ({p.description})")
                return p.device

        return None

    def _connect(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                write_timeout=0.1,
            )

            if self.reset_esp:
                self._hard_reset()

            time.sleep(2)
            print(f"[ESP32] Connected to {self.port} @ {self.baud} baud")
            self._running = True
            self._wait_for_ready(timeout=3)
            return True

        except serial.SerialException as e:
            print(f"[ESP32] Failed to connect: {e}")
            self.ser = None
            return False

    def _hard_reset(self):
        try:
            self.ser.dtr = False
            self.ser.rts = False
            time.sleep(0.1)
            self.ser.dtr = True
            self.ser.rts = False
            time.sleep(0.1)
            self.ser.dtr = False
            self.ser.rts = False
            print("[ESP32] Hardware reset via DTR/RTS")
        except Exception:
            print("[ESP32] DTR/RTS reset not available on this platform")

    def _wait_for_ready(self, timeout=3):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ser and self.ser.is_open and self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if 'Ready' in line or 'ready' in line:
                        print(f"[ESP32] Ready signal received: {line}")
                        return True
                except Exception:
                    pass
            time.sleep(0.1)
        print(f"[ESP32] No 'Ready' signal within {timeout}s (ESP32 may still booting)")
        return False

    def send_raw(self, data):
        with self.lock:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(data.encode('utf-8'))
                except serial.SerialException as e:
                    print(f"[ESP32] Write error: {e}")

    def send_command(self, steering, throttle):
        steering = max(-1.0, min(1.0, steering))
        throttle = max(-1.0, min(1.0, throttle))
        s_int = int(round(steering * 100))
        t_int = int(round(throttle * 100))
        cmd = f"S{s_int:+.0f},T{t_int:+.0f}\n"
        with self.lock:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(cmd.encode('utf-8'))
                except serial.SerialException as e:
                    print(f"[ESP32] Write error: {e}")
            else:
                print(f"[ESP32 Sim] {cmd.strip()}")

    def stop(self):
        self.send_command(0.0, 0.0)

    def close(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.stop()
            time.sleep(0.1)
            self.ser.close()
            print(f"[ESP32] Disconnected from {self.port}")
