import time


class PWMController:
    def __init__(self, chip=0, pwm_frequency=50):
        self.frequency = pwm_frequency
        self.steering_channel = 0
        self.throttle_channel = 1
        self.steering_center = 90
        self.steering_range = 45
        self.throttle_min = 60
        self.throttle_max = 180

        self._initialized = False
        self._simulate = True

        try:
            import Jetson.GPIO as GPIO
            import wiringpi
            wiringpi.wiringPiSetupGpio()
            self._simulate = False
            self._initialized = True
            print("[PWM] Jetson.GPIO + wiringpi initialized")
        except ImportError:
            print("[PWM] Jetson GPIO not available — running in simulation mode")

    def _map_steering(self, value):
        value = max(-1.0, min(1.0, value))
        return self.steering_center + value * self.steering_range

    def _map_throttle(self, value):
        value = max(-1.0, min(1.0, value))
        if value >= 0:
            return self.throttle_min + value * (self.throttle_max - self.throttle_min)
        return self.throttle_min

    def set_steering(self, value):
        pulse = self._map_steering(value)
        if self._simulate:
            print(f"[PWM Sim] Steering: {value:+.2f} -> {pulse:.0f}us")
            return
        import wiringpi
        wiringpi.pwmWrite(self.steering_channel, int(pulse))

    def set_throttle(self, value):
        pulse = self._map_throttle(value)
        if self._simulate:
            print(f"[PWM Sim] Throttle: {value:+.2f} -> {pulse:.0f}us")
            return
        import wiringpi
        wiringpi.pwmWrite(self.throttle_channel, int(pulse))

    def stop(self):
        if not self._simulate:
            import wiringpi
            wiringpi.pwmWrite(self.steering_channel, 0)
            wiringpi.pwmWrite(self.throttle_channel, 0)
        print("[PWM] Stopped")


class SafetyMonitor:
    def __init__(self, pwm_controller, max_throttle=0.5, timeout=2.0):
        self.pwm = pwm_controller
        self.max_throttle = max_throttle
        self.timeout = timeout
        self.last_command_time = time.time()
        self.emergency_stop = False

    def update(self, steering, throttle):
        if self.emergency_stop:
            self.pwm.set_throttle(0.0)
            self.pwm.set_steering(0.0)
            return

        throttle = max(-self.max_throttle, min(self.max_throttle, throttle))
        self.pwm.set_steering(steering)
        self.pwm.set_throttle(throttle)
        self.last_command_time = time.time()

    def check_timeout(self):
        if time.time() - self.last_command_time > self.timeout:
            self.pwm.set_throttle(0.0)
            self.pwm.set_steering(0.0)
            return True
        return False

    def trigger_estop(self):
        self.emergency_stop = True
        self.pwm.stop()
        print("[SAFETY] EMERGENCY STOP ACTIVATED")
