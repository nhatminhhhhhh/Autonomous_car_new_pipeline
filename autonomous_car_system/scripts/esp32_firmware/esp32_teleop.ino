/*
 * ESP32 Teleoperation Firmware
 *
 * Receives commands from Jetson Nano via USB UART (Serial)
 * and controls RC car steering (servo) + DC motor (L298N).
 *
 * Pinout (from user's esp32_code.ino):
 *   GPIO 5  — steering servo (PWM, 540-2500 µs, 50 Hz)
 *   GPIO 18 — motor PWM speed
 *   GPIO 26 — motor direction A (enA)
 *   GPIO 27 — motor direction B (enB)
 *
 * L298N Truth Table:
 *   27=HIGH, 26=LOW  → forward
 *   27=LOW,  26=HIGH → reverse
 *   27=LOW,  26=LOW  → brake (stop)
 *
 * Command format: S<steering>,T<throttle>\n
 *   steering: integer -100..+100  (-100 = full left, 0 = center, +100 = full right)
 *   throttle: integer -100..+100  (-100 = full reverse, 0 = stop, +100 = full forward)
 *
 * Example: S+045,T-030\n  -> steering 45% right, throttle 30% reverse
 *
 * Safety: If no valid command received for SAFETY_TIMEOUT ms, stop the car.
 */

#include <ESP32Servo.h>

// --- Pin Configuration (from esp32_code.ino) ---
#define STEERING_PIN    5
#define MOTOR_PWM_PIN   18
#define MOTOR_DIR_A     26
#define MOTOR_DIR_B     27
#define LED_PIN          2

// --- Steering Servo ---
Servo steeringServo;
#define STEERING_PULSE_MIN    540
#define STEERING_PULSE_CENTER 1520
#define STEERING_PULSE_MAX    2500

// --- Safety ---
#define SAFETY_TIMEOUT    500     // ms without valid command -> stop
#define BAUD_RATE         115200

// --- Globals ---
unsigned long lastCmdTime = 0;
int currentSteering = 0;
int currentThrottle = 0;
bool safetiesEngaged = false;

// --- Steering ---
int mapSteering(int s) {
  return map(constrain(s, -100, 100), -100, 100,
             STEERING_PULSE_MIN, STEERING_PULSE_MAX);
}

// --- Motor ---
void setMotor(int t) {
  t = constrain(t, -100, 100);

  if (t > 0) {
    digitalWrite(MOTOR_DIR_A, LOW);
    digitalWrite(MOTOR_DIR_B, HIGH);
    analogWrite(MOTOR_PWM_CHANNEL, map(t, 0, 100, 0, 255));
  } else if (t < 0) {
    digitalWrite(MOTOR_DIR_A, HIGH);
    digitalWrite(MOTOR_DIR_B, LOW);
    analogWrite(18, map(-t, 0, 100, 0, 255));
  } else {
    digitalWrite(MOTOR_DIR_A, LOW);
    digitalWrite(MOTOR_DIR_B, LOW);
    analogWrite(18, 0);
  }
}

void applyControls() {
  steeringServo.writeMicroseconds(mapSteering(currentSteering));
  setMotor(currentThrottle);
}

void emergencyStop() {
  currentSteering = 0;
  currentThrottle = 0;
  steeringServo.writeMicroseconds(STEERING_PULSE_CENTER);
  setMotor(0);
  safetiesEngaged = true;
  digitalWrite(LED_PIN, HIGH);
  Serial.println("ESTOP");
}

// --- Parse command: S<steering>,T<throttle>\n ---
void parseCommand(const String &line) {
  int s = 0, t = 0;

  int sIdx = line.indexOf('S');
  int tIdx = line.indexOf('T');

  if (sIdx >= 0 && tIdx >= 0) {
    String sStr = line.substring(sIdx + 1, tIdx);
    String tStr = line.substring(tIdx + 1);

    sStr.trim();
    tStr.trim();

    while (sStr.length() > 0 &&
           !isDigit(sStr[sStr.length()-1]) && sStr[sStr.length()-1] != '-') {
      sStr.remove(sStr.length() - 1);
    }
    while (tStr.length() > 0 &&
           !isDigit(tStr[tStr.length()-1]) && tStr[tStr.length()-1] != '-') {
      tStr.remove(tStr.length() - 1);
    }

    if (sStr.length() > 0 && tStr.length() > 0) {
      s = constrain(sStr.toInt(), -100, 100);
      t = constrain(tStr.toInt(), -100, 100);

      currentSteering = s;
      currentThrottle = t;
      lastCmdTime = millis();
      safetiesEngaged = false;
      digitalWrite(LED_PIN, LOW);

      applyControls();

      Serial.print("OK S");
      Serial.print(s);
      Serial.print(" T");
      Serial.println(t);
    }
  }
}

// --- Setup ---
void setup() {
  Serial.begin(BAUD_RATE);
  Serial.println("ESP32 Teleop Ready");

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Motor control pins
  pinMode(MOTOR_DIR_A, OUTPUT);
  pinMode(MOTOR_DIR_B, OUTPUT);
  digitalWrite(MOTOR_DIR_A, LOW);
  digitalWrite(MOTOR_DIR_B, LOW);

  pinMode(18, OUTPUT);
  analogWrite(18,0);
  // Steering servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  steeringServo.setPeriodHertz(50);
  steeringServo.attach(STEERING_PIN, STEERING_PULSE_MIN, STEERING_PULSE_MAX);
  steeringServo.writeMicroseconds(STEERING_PULSE_CENTER);

  Serial.println("ESP32 Teleop Ready — steering=GPIO5, motor(GPIO18,26,27)");
}

// --- Loop ---
void loop() {
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      parseCommand(line);
    }
  }

  if (!safetiesEngaged && (millis() - lastCmdTime > SAFETY_TIMEOUT)) {
    emergencyStop();
    Serial.println("SAFETY: timeout — stopped");
  }

  if (!safetiesEngaged && (millis() - lastCmdTime > 200)) {
    static unsigned long lastBlink = 0;
    if (millis() - lastBlink > 1000) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      lastBlink = millis();
    }
  }
}
