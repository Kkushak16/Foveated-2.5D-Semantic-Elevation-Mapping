/**
 * @file wave_rover_controller.ino
 * @brief Waveshare 4WD Wave Rover Controller Firmware for Arduino Uno Q / R3 / R4 / ESP32.
 * High-precision closed-loop velocity controller, optical wheel odometry, battery sensing,
 * procedural random sci-fi animations (waves, snake, fast lights, matrix rain, flicker, heart, OK),
 * dynamic directional drive arrows (⬆️ ⬇️ ⬅️ ➡️), and safe watchdog disconnect animation (flicker -> heart -> complete LED shutoff).
 */

#include <Arduino.h>

// ============================================================================
// ANIMATION MODES ENUM
// ============================================================================
enum LedMode {
  LED_MODE_OFF = 0,
  LED_MODE_OK = 1,
  LED_MODE_RANDOM_ACTIVE = 2, // Auto-generating procedural random animations
  LED_MODE_WAVE = 3,
  LED_MODE_SNAKE = 4,
  LED_MODE_FAST = 5,
  LED_MODE_MATRIX_RAIN = 6,
  LED_MODE_FLICKER = 7,
  LED_MODE_HEART = 8,
  LED_MODE_ARROW = 9
};

void set_led_mode(LedMode mode);

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
const int PIN_BATTERY_SENSE = A0;
const float VOLTAGE_DIVIDER_RATIO = 5.0; // Resistor divider ratio

#if defined(ARDUINO_UNOR4_WIFI) || defined(ARDUINO_UNO_Q) || defined(__ZEPHYR__) || __has_include("Arduino_LED_Matrix.h")
#include "Arduino_LED_Matrix.h"
#define HAS_LED_MATRIX 1
Arduino_LED_Matrix matrix;

#if __has_include("Arduino_RouterBridge.h")
#include "Arduino_RouterBridge.h"
#define HAS_ROUTER_BRIDGE 1
#endif

const uint8_t L_MAX = 7; // Maximum brightness (7) for 3-bit Uno Q LED Matrix

inline void draw_matrix(const uint8_t *bitmap) {
  matrix.loadPixels((uint8_t *)bitmap, 104);
}

inline void clear_matrix() {
  matrix.clear();
}

// 8 rows x 13 columns bitmap patterns for Arduino UNO Q LED Matrix (Brightness: 7=Max, 0=Off)
uint8_t MATRIX_OK[8][13] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 7, 7, 7, 0, 0, 7, 0, 0, 7, 0, 0},
  {0, 7, 0, 0, 0, 7, 0, 7, 0, 7, 0, 0, 0},
  {0, 7, 0, 0, 0, 7, 0, 7, 7, 0, 0, 0, 0},
  {0, 7, 0, 0, 0, 7, 0, 7, 0, 7, 0, 0, 0},
  {0, 0, 7, 7, 7, 0, 0, 7, 0, 0, 7, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

uint8_t MATRIX_SOLID[8][13] = {
  {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7},
  {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7},
  {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7},
  {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7},
  {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7},
  {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7},
  {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7},
  {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7}
};

uint8_t MATRIX_HEART[8][13] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 0},
  {0, 7, 7, 7, 7, 0, 7, 7, 7, 7, 0, 0, 0},
  {0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0},
  {0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0},
  {0, 0, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0},
  {0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0}
};

uint8_t MATRIX_HEART_SM[8][13] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 0},
  {0, 0, 7, 7, 7, 0, 7, 7, 7, 0, 0, 0, 0},
  {0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

// --- Directional Arrows for Teleop Drive Feedback ---
// Forward (Arrow UP)
uint8_t MATRIX_ARROW_UP[8][13] = {
  {0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0},
  {0, 0, 0, 7, 0, 7, 7, 7, 0, 7, 0, 0, 0},
  {0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

// Backward (Arrow DOWN)
uint8_t MATRIX_ARROW_DOWN[8][13] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0},
  {0, 0, 0, 7, 0, 7, 7, 7, 0, 7, 0, 0, 0},
  {0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0}
};

// Turn Left (Arrow LEFT)
uint8_t MATRIX_ARROW_LEFT[8][13] = {
  {0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0},
  {0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0},
  {0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0},
  {0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

// Turn Right (Arrow RIGHT)
uint8_t MATRIX_ARROW_RIGHT[8][13] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0},
  {0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0},
  {0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0},
  {0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

#else
#define HAS_LED_MATRIX 0
#endif

// LED Indicator Pins (Status & Connection Confirmation)
const int PIN_LED_OK = 13;       // Built-in LED pin (Controls QRB 1 on Uno Q; kept LOW to prevent QRB glow)
const int PIN_LED_STRIP = 12;    // Waveshare rover LED strip signal pin
bool led_ok_active = true;

// ============================================================================
// ANIMATION MODES STATE MACHINE
// ============================================================================
LedMode active_idle_mode = LED_MODE_OK;
LedMode current_led_mode = LED_MODE_OK;
int current_arrow_dir = 0; // 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
unsigned long last_anim_tick = 0;
int anim_frame = 0;

// Random Animation Engine variables
unsigned long last_random_cycle = 0;
unsigned long random_cycle_duration = 3200;
int random_sub_effect = 0;
int random_effect_dir = 1;

// Host Connection Tracking & Watchdog
bool is_host_connected = false;
unsigned long last_cmd_time = 0;
const unsigned long WATCHDOG_TIMEOUT_MS = 3500; // 3.5s timeout for disconnect detection

// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================
volatile long encoder_left_ticks = 0;
volatile long encoder_right_ticks = 0;

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
// LED MODE & ANIMATION CONTROL ROUTINES
// ============================================================================
void set_led_mode(LedMode mode) {
  if (mode == LED_MODE_OFF) {
    play_disconnect_animation();
    return;
  }

  current_led_mode = mode;
  if (mode != LED_MODE_ARROW) {
    active_idle_mode = mode;
  }
  anim_frame = 0;
  last_anim_tick = 0;
  led_ok_active = true;

  digitalWrite(PIN_LED_OK, HIGH);
  digitalWrite(PIN_LED_STRIP, HIGH);

#if HAS_LED_MATRIX
  if (mode == LED_MODE_OK) {
    draw_matrix((const uint8_t *)MATRIX_OK);
  }
#endif
}

void set_led_state(bool ok, int pattern_type = 0) {
  if (ok) {
    if (pattern_type == 1) {
      #if HAS_LED_MATRIX
      draw_matrix((const uint8_t *)MATRIX_SOLID);
      #endif
      led_ok_active = true;
      current_led_mode = LED_MODE_OK;
    } else {
      set_led_mode(active_idle_mode);
    }
  } else {
    play_disconnect_animation();
  }
}

void show_arrow(int dir) {
#if HAS_LED_MATRIX
  if (!led_ok_active) return;
  if (current_led_mode == LED_MODE_ARROW && current_arrow_dir == dir) return;

  current_led_mode = LED_MODE_ARROW;
  current_arrow_dir = dir;

  if (dir == 1) draw_matrix((const uint8_t *)MATRIX_ARROW_UP);
  else if (dir == 2) draw_matrix((const uint8_t *)MATRIX_ARROW_DOWN);
  else if (dir == 3) draw_matrix((const uint8_t *)MATRIX_ARROW_LEFT);
  else if (dir == 4) draw_matrix((const uint8_t *)MATRIX_ARROW_RIGHT);
#endif
}

void update_led_animations() {
#if HAS_LED_MATRIX
  unsigned long now = millis();

  // If in Auto-Random mode, dynamically switch between 7 procedurally random effects every 2.6 - 3.8s
  int active_effect = current_led_mode;
  if (current_led_mode == LED_MODE_RANDOM_ACTIVE) {
    if (now - last_random_cycle >= random_cycle_duration) {
      last_random_cycle = now;
      random_cycle_duration = (unsigned long)random(2600, 3900);
      int next_sub;
      do {
        next_sub = random(0, 7); // 0=Matrix Rain, 1=Fast Comet, 2=Sine Wave, 3=Snake, 4=Stardust, 5=Radar Pulse, 6=Equalizer
      } while (next_sub == random_sub_effect);
      random_sub_effect = next_sub;
      random_effect_dir = (random(0, 2) == 0) ? 1 : -1;
      anim_frame = 0;
    }

    if (random_sub_effect == 0) active_effect = LED_MODE_MATRIX_RAIN;
    else if (random_sub_effect == 1) active_effect = LED_MODE_FAST;
    else if (random_sub_effect == 2) active_effect = LED_MODE_WAVE;
    else if (random_sub_effect == 3) active_effect = LED_MODE_SNAKE;
    else if (random_sub_effect == 4) active_effect = LED_MODE_FLICKER; // Stardust / cosmic noise
    else if (random_sub_effect == 5) active_effect = LED_MODE_HEART;   // Pulsing radar/heart
    else active_effect = 10; // Equalizer
  }

  // 1. WAVE: Smooth sinusoidal crest rippling horizontally across columns
  if (active_effect == LED_MODE_WAVE) {
    if (now - last_anim_tick >= 55) {
      last_anim_tick = now;
      anim_frame = (anim_frame + 1) % 13;
      uint8_t wave_buf[8][13] = {0};
      for (int c = 0; c < 13; c++) {
        int eval_col = (random_effect_dir > 0) ? c : (12 - c);
        int dist = abs(eval_col - anim_frame);
        if (dist > 6) dist = 13 - dist;
        int peak_row = constrain(dist + 1, 1, 7);
        for (int r = peak_row; r < 8; r++) {
          wave_buf[r][c] = 7;
        }
      }
      draw_matrix((const uint8_t *)wave_buf);
    }
  }
  // 2. SNAKE: Luminescent snake crawling along matrix perimeter
  else if (active_effect == LED_MODE_SNAKE) {
    if (now - last_anim_tick >= 50) {
      last_anim_tick = now;
      anim_frame = (anim_frame + 1) % 38;
      uint8_t snake_buf[8][13] = {0};
      for (int s = 0; s < 5; s++) {
        int idx = (anim_frame + s) % 38;
        int r = 0, c = 0;
        if (idx < 13) { r = 0; c = idx; }
        else if (idx < 20) { r = idx - 12; c = 12; }
        else if (idx < 32) { r = 7; c = 12 - (idx - 19); }
        else { r = 7 - (idx - 31); c = 0; }
        if (r >= 0 && r < 8 && c >= 0 && c < 13) snake_buf[r][c] = 7;
      }
      draw_matrix((const uint8_t *)snake_buf);
    }
  }
  // 3. MATRIX RAIN / DIGITAL RAIN: Random shimmering falling glyphs
  else if (active_effect == LED_MODE_MATRIX_RAIN) {
    if (now - last_anim_tick >= 75) {
      last_anim_tick = now;
      uint8_t rnd_buf[8][13] = {0};
      int count = random(14, 26);
      for (int i = 0; i < count; i++) {
        int r = random(0, 8);
        int c = random(0, 13);
        rnd_buf[r][c] = 7;
      }
      draw_matrix((const uint8_t *)rnd_buf);
    }
  }
  // 4. FAST LIGHTS: High-speed scanning laser comet sweep across columns (24ms)
  else if (active_effect == LED_MODE_FAST) {
    if (now - last_anim_tick >= 24) {
      last_anim_tick = now;
      anim_frame = (anim_frame + 1) % 24; // 13 cols forward (0-12), 11 backward (13-23)
      int col = (anim_frame < 13) ? anim_frame : (24 - anim_frame);
      uint8_t fast_buf[8][13] = {0};

      // Main intense beam
      for (int r = 0; r < 8; r++) {
        if (col >= 0 && col < 13) fast_buf[r][col] = 7;
      }

      // Fast trailing streak / comet tail
      int dir = (anim_frame < 13) ? -1 : 1;
      int t1 = col + dir;
      if (t1 >= 0 && t1 < 13) {
        for (int r = 1; r < 7; r++) fast_buf[r][t1] = 7;
      }
      int t2 = col + 2 * dir;
      if (t2 >= 0 && t2 < 13) {
        for (int r = 2; r < 6; r++) fast_buf[r][t2] = 7;
      }
      draw_matrix((const uint8_t *)fast_buf);
    }
  }
  // 5. STARDUST / COSMIC NOISE: Twinkling sparkling random constellations
  else if (active_effect == LED_MODE_FLICKER) {
    if (now - last_anim_tick >= 80) {
      last_anim_tick = now;
      anim_frame = (anim_frame + 1) % 4;
      uint8_t star_buf[8][13] = {0};
      int num_stars = random(8, 18);
      for (int i = 0; i < num_stars; i++) {
        star_buf[random(0, 8)][random(0, 13)] = 7;
      }
      // Breathing border sparkle
      star_buf[0][0] = 7; star_buf[0][12] = 7; star_buf[7][0] = 7; star_buf[7][12] = 7;
      draw_matrix((const uint8_t *)star_buf);
      digitalWrite(PIN_LED_STRIP, (anim_frame % 2 == 0) ? HIGH : LOW);
    }
  }
  // 6. RADAR / PULSING EXPANDING BOX
  else if (active_effect == LED_MODE_HEART) {
    if (now - last_anim_tick >= 140) {
      last_anim_tick = now;
      anim_frame = (anim_frame + 1) % 5;
      uint8_t pulse_buf[8][13] = {0};
      int expand = anim_frame;
      int r_min = constrain(3 - expand, 0, 3);
      int r_max = constrain(4 + expand, 4, 7);
      int c_min = constrain(6 - (expand * 2), 0, 6);
      int c_max = constrain(6 + (expand * 2), 6, 12);
      for (int c = c_min; c <= c_max; c++) {
        pulse_buf[r_min][c] = 7;
        pulse_buf[r_max][c] = 7;
      }
      for (int r = r_min; r <= r_max; r++) {
        pulse_buf[r][c_min] = 7;
        pulse_buf[r][c_max] = 7;
      }
      draw_matrix((const uint8_t *)pulse_buf);
    }
  }
  // 7. EQUALIZER: 13 bouncing spectrum visualizer columns
  else if (active_effect == 10) {
    if (now - last_anim_tick >= 90) {
      last_anim_tick = now;
      uint8_t eq_buf[8][13] = {0};
      for (int c = 0; c < 13; c++) {
        int height = random(1, 8);
        for (int r = 7; r >= (8 - height); r--) {
          eq_buf[r][c] = 7;
        }
      }
      draw_matrix((const uint8_t *)eq_buf);
    }
  }
  // 8. OK: Steady OK bitmap
  else if (active_effect == LED_MODE_OK) {
    if (anim_frame == 0) {
      anim_frame = 1;
      draw_matrix((const uint8_t *)MATRIX_OK);
      digitalWrite(PIN_LED_OK, HIGH);
      digitalWrite(PIN_LED_STRIP, HIGH);
    }
  }
#endif
}

void flash_heartbeat_led() {
  digitalWrite(PIN_LED_OK, LOW);
  digitalWrite(PIN_LED_STRIP, HIGH);

#if HAS_LED_MATRIX
  draw_matrix((const uint8_t *)MATRIX_HEART_SM);
  delay(120);
  draw_matrix((const uint8_t *)MATRIX_HEART);
  delay(250);
  draw_matrix((const uint8_t *)MATRIX_HEART_SM);
  delay(120);
  draw_matrix((const uint8_t *)MATRIX_HEART);
  delay(300);
#else
  delay(90);
  digitalWrite(PIN_LED_STRIP, LOW);
  delay(110);
  digitalWrite(PIN_LED_STRIP, HIGH);
  delay(180);
#endif

  digitalWrite(PIN_LED_STRIP, LOW);
  if (active_idle_mode == LED_MODE_OFF) {
    // Already disconnected: stay fully OFF instead of re-triggering
    // play_disconnect_animation() recursively via set_led_mode(OFF).
    led_ok_active = false;
    current_led_mode = LED_MODE_OFF;
    #if HAS_LED_MATRIX
      clear_matrix();
    #endif
    digitalWrite(PIN_LED_OK, LOW);
  } else {
    set_led_mode(active_idle_mode);
  }
  Serial.println(F("{\"status\":\"LED_HEARTBEAT_ACTIVE\",\"led\":\"HEART_OK\"}"));
}

void play_disconnect_animation() {
  led_ok_active = false;
  current_led_mode = LED_MODE_OFF;
  active_idle_mode = LED_MODE_OFF;
  is_host_connected = false;

  // 1. Immediately stop displaying whatever animation was running
  #if HAS_LED_MATRIX
    clear_matrix();
  #endif
  digitalWrite(PIN_LED_STRIP, LOW);
  digitalWrite(PIN_LED_OK, LOW);
  delay(80);

  // 2. Flicker LED (3 quick alert pulses on both LED strip and small heart matrix)
  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_LED_STRIP, HIGH);
    #if HAS_LED_MATRIX
      draw_matrix((const uint8_t *)MATRIX_HEART_SM);
    #endif
    delay(100);
    digitalWrite(PIN_LED_STRIP, LOW);
    #if HAS_LED_MATRIX
      clear_matrix();
    #endif
    delay(100);
  }

  // 3. Show Heart prominently on the LED Matrix for 1.2 second
  #if HAS_LED_MATRIX
    draw_matrix((const uint8_t *)MATRIX_HEART);
    digitalWrite(PIN_LED_STRIP, HIGH);
    delay(1200);
  #else
    digitalWrite(PIN_LED_STRIP, HIGH);
    delay(600);
  #endif

  // 4. Switch OFF all LEDs completely and STAY OFF!
  #if HAS_LED_MATRIX
    clear_matrix();
  #endif
  digitalWrite(PIN_LED_STRIP, LOW);
  digitalWrite(PIN_LED_OK, LOW);

  Serial.println(F("{\"status\":\"DISCONNECTED_ANIMATION_COMPLETE\",\"led\":\"OFF\"}"));
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

  // Directional Arrow Display on 8x13 Matrix
  #if HAS_LED_MATRIX
  if (led_ok_active) {
    if (speed_l == 0 && speed_r == 0) {
      if (current_led_mode == LED_MODE_ARROW) {
        set_led_mode(active_idle_mode);
      }
    } else {
      if (speed_l > 25 && speed_r > 25) {
        show_arrow(1); // Forward (UP)
      } else if (speed_l < -25 && speed_r < -25) {
        show_arrow(2); // Backward (DOWN)
      } else if (speed_l < speed_r) {
        show_arrow(3); // Turn Left
      } else if (speed_l > speed_r) {
        show_arrow(4); // Turn Right
      }
    }
  }
  #endif
}

void emergency_stop() {
  analogWrite(PIN_MOTOR_L_PWM, 0);
  analogWrite(PIN_MOTOR_R_PWM, 0);
  digitalWrite(PIN_MOTOR_L_DIR, LOW);
  digitalWrite(PIN_MOTOR_R_DIR, LOW);

  #if HAS_LED_MATRIX
  if (led_ok_active && current_led_mode == LED_MODE_ARROW) {
    set_led_mode(active_idle_mode);
  }
  #endif
}

// ============================================================================
// BATTERY & TELEMETRY STREAMING
// ============================================================================
float read_battery_voltage() {
  int raw = analogRead(PIN_BATTERY_SENSE);
  float v_pin = (raw / 1023.0) * 5.0;
  float v_bat = v_pin * VOLTAGE_DIVIDER_RATIO;
  if (v_bat < 0.2) {
    return 0.0;
  }
  return v_bat;
}

void send_telemetry() {
  float v_bat = read_battery_voltage();
  
  // Waveshare protocol format: {"v": 12.1, "left": 100, "right": 100}
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

  // 0. Disconnect Command: {"cmd":"disconnect"}, {"state":"disconnect"}, {"state":"off"}, {"mode":"off"}, {"pattern":"off"}, {"cmd":"off"}, or {"T":134}
  if (cmd.indexOf("\"cmd\":\"disconnect\"") >= 0 || 
      cmd.indexOf("\"state\":\"disconnect\"") >= 0 || 
      cmd.indexOf("\"state\":\"off\"") >= 0 || 
      cmd.indexOf("\"mode\":\"off\"") >= 0 || 
      cmd.indexOf("\"pattern\":\"off\"") >= 0 || 
      cmd.indexOf("\"cmd\":\"off\"") >= 0 || 
      cmd.indexOf("\"T\":134") >= 0) {
    emergency_stop();
    play_disconnect_animation();
    return;
  }

  // 1. Heartbeat / Heart Ping: {"cmd":"heart"}, {"pattern":"heart"}, or {"T":135}
  if (cmd.indexOf("\"cmd\":\"heart\"") >= 0 || cmd.indexOf("\"pattern\":\"heart\"") >= 0 || cmd.indexOf("\"T\":135") >= 0) {
    flash_heartbeat_led();
    return;
  }

  // 2. LED Modes: {"cmd":"led_mode","mode":"random|wave|snake|fast|rain|flicker|heart|ok|off"}
  if (cmd.indexOf("\"cmd\":\"led_mode\"") >= 0 || cmd.indexOf("\"led_mode\"") >= 0) {
    if (cmd.indexOf("off") >= 0) {
      emergency_stop();
      play_disconnect_animation();
      return;
    }
    led_ok_active = true;
    if (cmd.indexOf("wave") >= 0) set_led_mode(LED_MODE_WAVE);
    else if (cmd.indexOf("snake") >= 0) set_led_mode(LED_MODE_SNAKE);
    else if (cmd.indexOf("fast") >= 0 || cmd.indexOf("speed") >= 0 || cmd.indexOf("scan") >= 0) set_led_mode(LED_MODE_FAST);
    else if (cmd.indexOf("rain") >= 0 || cmd.indexOf("matrix") >= 0) set_led_mode(LED_MODE_MATRIX_RAIN);
    else if (cmd.indexOf("flicker") >= 0) set_led_mode(LED_MODE_FLICKER);
    else if (cmd.indexOf("heart") >= 0) set_led_mode(LED_MODE_HEART);
    else if (cmd.indexOf("ok") >= 0) set_led_mode(LED_MODE_OK);
    else if (cmd.indexOf("random") >= 0 || cmd.indexOf("auto") >= 0) set_led_mode(LED_MODE_RANDOM_ACTIVE);
    else set_led_mode(LED_MODE_RANDOM_ACTIVE);
    Serial.println(F("{\"status\":\"LED_MODE_UPDATED\"}"));
    return;
  }

  // 3. LED Matrix / OK Command from Python Bridge or Host:
  if (cmd.indexOf("\"cmd\":\"led\"") >= 0 || cmd.indexOf("\"T\":133") >= 0 || cmd.indexOf("\"state\":\"ok\"") >= 0 || cmd.indexOf("\"pattern\"") >= 0 || cmd.indexOf("\"cmd\":\"connect\"") >= 0) {
    is_host_connected = true;
    last_cmd_time = millis();
    if (cmd.indexOf("\"state\":\"off\"") >= 0 || cmd.indexOf("\"state\":\"0\"") >= 0 || cmd.indexOf("\"pattern\":\"off\"") >= 0 || cmd.indexOf("off") >= 0) {
      emergency_stop();
      play_disconnect_animation();
    } else if (cmd.indexOf("\"pattern\":\"solid\"") >= 0 || cmd.indexOf("\"state\":\"solid\"") >= 0) {
      set_led_state(true, 1);
    } else {
      #if HAS_LED_MATRIX
      draw_matrix((const uint8_t *)MATRIX_OK);
      #endif
      digitalWrite(PIN_LED_OK, HIGH);
      digitalWrite(PIN_LED_STRIP, HIGH);
      set_led_mode(LED_MODE_OK);
      Serial.println(F("{\"status\":\"LED_OK_ACTIVE\",\"ok\":true}"));
    }
    return;
  }

  // 4. Emergency Stop
  if (cmd.indexOf("\"T\":0") >= 0) {
    emergency_stop();
    last_cmd_time = millis();
    return;
  }

  // 5. Telemetry Request
  if (cmd.indexOf("\"T\":1001") >= 0) {
    send_telemetry();
    return;
  }

  // 6. Drive Command
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

#if HAS_ROUTER_BRIDGE
bool rpc_show_ok() {
  set_led_mode(LED_MODE_OK);
  return true;
}

bool rpc_show_heart() {
  set_led_mode(LED_MODE_HEART);
  return true;
}

bool rpc_show_arrow(int dir) {
  current_arrow_dir = dir;
  if (dir == 0) draw_matrix((const uint8_t *)MATRIX_ARROW_UP);
  else if (dir == 1) draw_matrix((const uint8_t *)MATRIX_ARROW_DOWN);
  else if (dir == 2) draw_matrix((const uint8_t *)MATRIX_ARROW_LEFT);
  else if (dir == 3) draw_matrix((const uint8_t *)MATRIX_ARROW_RIGHT);
  current_led_mode = LED_MODE_ARROW;
  led_ok_active = true;
  return true;
}

bool rpc_turn_off() {
  set_led_mode(LED_MODE_OFF);
  return true;
}
#endif

// ============================================================================
// SETUP & MAIN LOOP
// ============================================================================
void setup() {
  Serial.begin(115200);
  rx_buffer.reserve(128);

  // Motor output pins
  pinMode(PIN_MOTOR_L_PWM, OUTPUT);
  pinMode(PIN_MOTOR_L_DIR, OUTPUT);
  pinMode(PIN_MOTOR_R_PWM, OUTPUT);
  pinMode(PIN_MOTOR_R_DIR, OUTPUT);

  // Status & LED Strip pins
  pinMode(PIN_LED_OK, OUTPUT);
  pinMode(PIN_LED_STRIP, OUTPUT);
  digitalWrite(PIN_LED_OK, LOW);
  digitalWrite(PIN_LED_STRIP, LOW);

  // Seed pseudo-random generator with analog entropy
  randomSeed(micros() ^ (unsigned long)analogRead(PIN_BATTERY_SENSE));

#if HAS_LED_MATRIX
  matrix.begin();
  clear_matrix();
  set_led_mode(LED_MODE_OK); // Power on with LED panel immediately glowing with OK sign!
#endif

#if HAS_ROUTER_BRIDGE
  Bridge.begin();
  Bridge.provide("show_ok", rpc_show_ok);
  Bridge.provide("show_heart", rpc_show_heart);
  Bridge.provide("show_arrow", rpc_show_arrow);
  Bridge.provide("turn_off", rpc_turn_off);
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
        last_cmd_time = millis();
        is_host_connected = true; // Active communication established!
        parse_command(rx_buffer);
        rx_buffer = "";
      }
    } else {
      if (rx_buffer.length() < 128) {
        rx_buffer += c;
      }
    }
  }

  // 2. Safety Watchdog — if host was communicating and stops for >3.5s, safely halt motors
  // and play the flicker -> heart -> full LED shutoff disconnect animation.
  if (is_host_connected && (millis() - last_cmd_time > WATCHDOG_TIMEOUT_MS)) {
    is_host_connected = false;
    emergency_stop();
    play_disconnect_animation();
  }

  // 3. Periodic Telemetry Stream (20 Hz)
  if (millis() - last_telemetry_time >= TELEMETRY_INTERVAL_MS) {
    last_telemetry_time = millis();
    send_telemetry();
  }

  // 4. Update Dynamic LED Animation Frames (Procedural random waves, snake, fast lights, matrix rain)
#if HAS_LED_MATRIX
  if (led_ok_active && current_led_mode != LED_MODE_OFF) {
    update_led_animations();
  }
#endif
}
