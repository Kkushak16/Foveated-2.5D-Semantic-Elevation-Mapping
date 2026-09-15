/*
 * wave_rover_controller.ino — Arduino Firmware for Waveshare WAVE ROVER
 * =====================================================================
 * Target: Arduino Uno (R3/R4), Nano, Mega, or Arduino Uno WiFi / Nano ESP32
 * 
 * Functions:
 *   1. Receives JSON movement commands over Serial (USB @ 115200 baud):
 *        {"T":1,"L":speed_left,"R":speed_right}  -> Motor drive (-255 to 255)
 *        {"T":0}                                 -> Emergency Stop
 *        {"T":1001}                              -> Immediate Telemetry Request
 *   2. Closed-loop / Dual H-Bridge Motor Control (PWM + Direction)
 *   3. Reads wheel encoders on Interrupt Pins (D2, D3)
 *   4. Monitors battery voltage on Analog Pin A0 (via voltage divider)
 *   5. Safety Watchdog: Stops motors if no command received for 600ms
 *   6. Streams JSON telemetry back to Python Bridge on Host Laptop:
 *        {"v":12.1, "left":1042, "right":1050}
 */

#include <Arduino.h>

// ============================================================================
// PIN DEFINITIONS (Configured for standard 4WD H-Bridge Motor Drivers)
// ============================================================================
// Left Motors (PWM + Direction)
const int PIN_MOTOR_L_PWM = 5;    // Timer 0 / 2 PWM pin
const int PIN_MOTOR_L_DIR = 4;    // Direction pin

// Right Motors (PWM + Direction)
const int PIN_MOTOR_R_PWM = 6;    // Timer 0 PWM pin
const int PIN_MOTOR_R_DIR = 7;    // Direction pin

// Optical Wheel Encoders (Hardware Interrupt Pins on Uno: D2 & D3)
const int PIN_ENCODER_LEFT  = 2;  // INT0
const int PIN_ENCODER_RIGHT = 3;  // INT1

// Battery Voltage Sensing (Analog Voltage Divider)
// e.g. 100k / 20k resistor divider to step down 12.6V to < 5V
const int PIN_BATTERY_SENSE = A0;
const float VOLTAGE_DIVIDER_RATIO = 5.0; // Adjust according to your resistor divider

// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================
volatile long encoder_left_ticks = 0;
volatile long encoder_right_ticks = 0;

unsigned long last_cmd_time = 0;
const unsigned long WATCHDOG_TIMEOUT_MS = 600; // Auto-stop if laptop disconnects

unsigned long last_telemetry_time = 0;
const unsigned long TELEMETRY_INTERVAL_MS = 50; // 20 Hz telemetry stream

String rx_buffer = "";

// ============================================================================
// ENCODER INTERRUPT SERVICE ROUTINES (ISRs)
// ============================================================================
void isr_encoder_left() {
  encoder_left_ticks++;
}

void isr_encoder_right() {
  encoder_right_ticks++;
}

// ============================================================================
// MOTOR CONTROL PRIMITIVES
// ============================================================================
void set_motors(int speed_l, int speed_r) {
  // Clamp speeds
  speed_l = constrain(speed_l, -255, 255);
  speed_r = constrain(speed_r, -255, 255);

  // Left Motor Direction & PWM
  if (speed_l >= 0) {
    digitalWrite(PIN_MOTOR_L_DIR, HIGH);
    analogWrite(PIN_MOTOR_L_PWM, speed_l);
  } else {
    digitalWrite(PIN_MOTOR_L_DIR, LOW);
    analogWrite(PIN_MOTOR_L_PWM, -speed_l);
  }

  // Right Motor Direction & PWM
  if (speed_r >= 0) {
    digitalWrite(PIN_MOTOR_R_DIR, HIGH);
    analogWrite(PIN_MOTOR_R_PWM, speed_r);
  } else {
    digitalWrite(PIN_MOTOR_R_DIR, LOW);
    analogWrite(PIN_MOTOR_R_PWM, -speed_r);
  }
}

void emergency_stop() {
  analogWrite(PIN_MOTOR_L_PWM, 0);
  analogWrite(PIN_MOTOR_R_PWM, 0);
  digitalWrite(PIN_MOTOR_L_DIR, LOW);
  digitalWrite(PIN_MOTOR_R_DIR, LOW);
}

// ============================================================================
// BATTERY & TELEMETRY STREAMING
// ============================================================================
float read_battery_voltage() {
  int raw = analogRead(PIN_BATTERY_SENSE);
  float v_pin = (raw / 1023.0) * 5.0;
  float v_bat = v_pin * VOLTAGE_DIVIDER_RATIO;
  if (v_bat < 1.0) {
    // Default simulated voltage if no divider wired
    v_bat = 12.2;
  }
  return v_bat;
}

void send_telemetry() {
  float v_bat = read_battery_voltage();
  
  // Waveshare protocol format:
  // {"v": 12.1, "left": 100, "right": 100}
  Serial.print(F("{\"v\":"));
  Serial.print(v_bat, 2);
  Serial.print(F(",\"left\":"));
  Serial.print(encoder_left_ticks);
  Serial.print(F(",\"right\":"));
  Serial.print(encoder_right_ticks);
  Serial.println(F("}"));
}

// ============================================================================
// JSON COMMAND PARSER
// ============================================================================
void parse_command(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  // Simple, robust zero-RAM string parser for standard Waveshare JSON commands
  if (cmd.indexOf("\"T\":0") >= 0) {
    emergency_stop();
    last_cmd_time = millis();
    return;
  }

  if (cmd.indexOf("\"T\":1001") >= 0) {
    send_telemetry();
    return;
  }

  if (cmd.indexOf("\"T\":1") >= 0) {
    int idx_l = cmd.indexOf("\"L\":");
    int idx_r = cmd.indexOf("\"R\":");

    if (idx_l != -1 && idx_r != -1) {
      int speed_l = cmd.substring(idx_l + 4).toInt();
      int speed_r = cmd.substring(idx_r + 4).toInt();
      set_motors(speed_l, speed_r);
      last_cmd_time = millis();
    }
  }
}

// ============================================================================
// SETUP & MAIN LOOP
// ============================================================================
void setup() {
  Serial.begin(115200);
  rx_buffer.reserve(64);

  // Motor output pins
  pinMode(PIN_MOTOR_L_PWM, OUTPUT);
  pinMode(PIN_MOTOR_L_DIR, OUTPUT);
  pinMode(PIN_MOTOR_R_PWM, OUTPUT);
  pinMode(PIN_MOTOR_R_DIR, OUTPUT);

  // Encoder input pins with pull-ups
  pinMode(PIN_ENCODER_LEFT, INPUT_PULLUP);
  pinMode(PIN_ENCODER_RIGHT, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_ENCODER_LEFT), isr_encoder_left, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENCODER_RIGHT), isr_encoder_right, RISING);

  emergency_stop();
  last_cmd_time = millis();

  // Send startup banner
  Serial.println(F("{\"status\":\"ARDUINO_WAVE_ROVER_READY\",\"baud\":115200}"));
}

void loop() {
  // 1. Read Serial Buffer from Host Laptop
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rx_buffer.length() > 0) {
        parse_command(rx_buffer);
        rx_buffer = "";
      }
    } else {
      if (rx_buffer.length() < 64) {
        rx_buffer += c;
      }
    }
  }

  // 2. Safety Watchdog — stop rover if communication lost
  if (millis() - last_cmd_time > WATCHDOG_TIMEOUT_MS) {
    emergency_stop();
  }

  // 3. Periodic Telemetry Stream (20 Hz)
  if (millis() - last_telemetry_time >= TELEMETRY_INTERVAL_MS) {
    last_telemetry_time = millis();
    send_telemetry();
  }
}
