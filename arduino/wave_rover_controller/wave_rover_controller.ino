/*
 * wave_rover_controller.ino — Arduino Firmware for Waveshare WAVE ROVER
 * =====================================================================
 * Target: Arduino Uno (R3/R4/Q), Nano, Mega, or Arduino Uno WiFi / Nano ESP32
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

#if defined(ARDUINO_UNOR4_WIFI) || defined(ARDUINO_UNO_Q) || defined(__ZEPHYR__) || __has_include("Arduino_LED_Matrix.h")
#include "Arduino_LED_Matrix.h"
#define HAS_LED_MATRIX 1
Arduino_LED_Matrix matrix;

// 8 rows x 13 columns bitmap patterns for Arduino UNO Q LED Matrix (Section circled in red)
uint8_t MATRIX_OK[8][13] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0},
  {0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0},
  {0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0},
  {0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0},
  {0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

uint8_t MATRIX_SOLID[8][13] = {
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
};

uint8_t MATRIX_HEART[8][13] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0},
  {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0},
  {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
  {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
  {0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0}
};

uint8_t MATRIX_HEART_SM[8][13] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
  {0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};
#else
#define HAS_LED_MATRIX 0
#endif

// LED Indicator Pins (Status & Connection Confirmation)
const int PIN_LED_OK = 13;       // Built-in LED pin (Controls QRB 1 on Uno Q; kept LOW to prevent QRB glow)
const int PIN_LED_STRIP = 12;    // Waveshare rover LED strip signal pin
bool led_ok_active = false;

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
// LED MATRIX & HEARTBEAT STATUS INDICATOR
// ============================================================================
void set_led_state(bool ok, int pattern_type = 0) {
  led_ok_active = ok;
  // Strictly keep PIN 13 LOW so the "QRB 1" LED does NOT glow
  digitalWrite(PIN_LED_OK, LOW);
  digitalWrite(PIN_LED_STRIP, ok ? HIGH : LOW);

#if HAS_LED_MATRIX
  if (ok) {
    if (pattern_type == 1) {
      matrix.renderBitmap(MATRIX_SOLID, 8, 13);
      Serial.println(F("{\"status\":\"LED_OK_ACTIVE\",\"led\":\"MATRIX_SOLID\"}"));
    } else {
      matrix.renderBitmap(MATRIX_OK, 8, 13);
      Serial.println(F("{\"status\":\"LED_OK_ACTIVE\",\"led\":\"MATRIX_OK\"}"));
    }
  } else {
    matrix.clear();
    Serial.println(F("{\"status\":\"LED_OFF\",\"led\":\"OFF\"}"));
  }
#else
  digitalWrite(PIN_LED_OK, ok ? HIGH : LOW);
  if (ok) {
    Serial.println(F("{\"status\":\"LED_OK_ACTIVE\",\"led\":\"OK\"}"));
  } else {
    Serial.println(F("{\"status\":\"LED_OFF\",\"led\":\"OFF\"}"));
  }
#endif
}

void flash_heartbeat_led() {
  digitalWrite(PIN_LED_OK, LOW); // Explicitly ensure QRB 1 stays OFF
  digitalWrite(PIN_LED_STRIP, HIGH);

#if HAS_LED_MATRIX
  // Pulse heart animation directly on the Arduino UNO Q 8x13 LED Matrix (circled section)
  matrix.renderBitmap(MATRIX_HEART_SM, 8, 13);
  delay(120);
  matrix.renderBitmap(MATRIX_HEART, 8, 13);
  delay(250);
  matrix.renderBitmap(MATRIX_HEART_SM, 8, 13);
  delay(120);
  matrix.renderBitmap(MATRIX_HEART, 8, 13);
  delay(300);
#else
  delay(90);
  digitalWrite(PIN_LED_STRIP, LOW);
  delay(110);
  digitalWrite(PIN_LED_STRIP, HIGH);
  delay(180);
#endif

  digitalWrite(PIN_LED_STRIP, LOW);
  // Settle back on steady matrix OK state
  set_led_state(true);
  Serial.println(F("{\"status\":\"LED_HEARTBEAT_ACTIVE\",\"led\":\"HEART_OK\"}"));
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
  // Strictly return genuine physical sensor reading (no fake 12.2V fallback)
  if (v_bat < 0.2) {
    return 0.0;
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

  // 1. Heartbeat / Heart Ping: {"cmd":"heart"}, {"pattern":"heart"}, or {"T":135}
  if (cmd.indexOf("\"cmd\":\"heart\"") >= 0 || cmd.indexOf("\"pattern\":\"heart\"") >= 0 || cmd.indexOf("\"T\":135") >= 0) {
    flash_heartbeat_led();
    return;
  }

  // 2. LED Matrix / OK Command from Python Bridge or Host:
  //    {"cmd":"led","state":"ok"}, {"pattern":"solid"}, {"pattern":"ok"}, {"state":"off"}
  if (cmd.indexOf("\"cmd\":\"led\"") >= 0 || cmd.indexOf("\"T\":133") >= 0 || cmd.indexOf("\"state\":\"ok\"") >= 0 || cmd.indexOf("\"pattern\"") >= 0) {
    if (cmd.indexOf("\"state\":\"off\"") >= 0 || cmd.indexOf("\"state\":\"0\"") >= 0 || cmd.indexOf("\"pattern\":\"off\"") >= 0) {
      set_led_state(false);
    } else if (cmd.indexOf("\"pattern\":\"solid\"") >= 0 || cmd.indexOf("\"state\":\"solid\"") >= 0) {
      set_led_state(true, 1);
    } else {
      set_led_state(true, 0);
    }
    return;
  }

  // 3. Emergency Stop
  if (cmd.indexOf("\"T\":0") >= 0) {
    emergency_stop();
    last_cmd_time = millis();
    return;
  }

  // 4. Telemetry Request
  if (cmd.indexOf("\"T\":1001") >= 0) {
    send_telemetry();
    return;
  }

  // 5. Drive Command
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

  // Status & LED Strip pins
  pinMode(PIN_LED_OK, OUTPUT);
  pinMode(PIN_LED_STRIP, OUTPUT);
  digitalWrite(PIN_LED_OK, LOW); // Explicitly ensure QRB 1 stays OFF
  digitalWrite(PIN_LED_STRIP, LOW);

#if HAS_LED_MATRIX
  matrix.begin();
  // Keep the 8x13 LED Matrix completely OFF on startup
  matrix.clear();
  set_led_state(false);
#endif

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
