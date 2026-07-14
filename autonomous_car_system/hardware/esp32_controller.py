import serial
import serial.tools.list_ports
import threading
import time


class ESP32Controller:
    def __init__(self, port=None, baud=115200, timeout=0.1):
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self.lock = threading.Lock()
        self._running = False
        self._read_thread = None

        if port is None:
            port = self._auto_detect_port()
        self.port = port

        if self.port:
            self._connect()
        else:
            print("[ESP32] No port specified and auto-detect failed — simulation mode")

    def _auto_detect_port(self):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if 'USB' in p.description or 'UART' in p.description or 'CP210' in p.description or 'CH340' in p.description or 'FTDI' in p.description:
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
                write_timeout=0.1
            )
            time.sleep(2)
            print(f"[ESP32] Connected to {self.port} @ {self.baud} baud")
            self._running = True
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()
        except serial.SerialException as e:
            print(f"[ESP32] Failed to connect: {e}")
            self.ser = None

    def _read_loop(self):
        while self._running and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"[ESP32 RX] {line}")
            except serial.SerialException:
                break
            except Exception:
                pass

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

    def send_steering(self, value):
        self.send_command(value, 0.0)

    def send_throttle(self, value):
        self.send_command(0.0, value)

    def stop(self):
        self.send_command(0.0, 0.0)

    def close(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.stop()
            time.sleep(0.1)
            self.ser.close()
            print(f"[ESP32] Disconnected from {self.port}")
