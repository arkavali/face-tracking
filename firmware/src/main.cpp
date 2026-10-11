// Mouth camera streamer: ESP32-CAM sends hardware-JPEG frames to the PC over Serial for
// live view and recordings (tools/mouth_cam.py). No tracking on the chip yet.
//
// Serial commands (921600 baud, one per line):
//   help                  this list
//   stream on|off         send frames (on after power-up)
//   size qvga|cif|vga     frame size (320x240 default, 400x296, 640x480)
//   q N                   JPEG quality 4..63 (lower = better, bigger; default 14)
//   gray on|off           grey picture (default on: smaller JPEG, more fps)
//   exp auto|N            exposure: automatic, or fixed N lines with auto gain
//   led N                 flash LED (GPIO 4) PWM 0..255, 0 = off
//   reg ADDR [VALUE]      read / write a camera register (hex, debugging)
//   status                frame size, quality, fps, bytes per frame
// Frame: "JPG seq w h len ms\n" + len bytes of JPEG. Text replies start with other words.
#include <Arduino.h>

#include "camera_pins.h"
#include "esp_camera.h"

namespace {

constexpr int kLedPin = 4;
bool streaming = true;
int quality = 14;
bool gray = true;
framesize_t frameSize = FRAMESIZE_QVGA;
uint32_t seq = 0, fpsFrames = 0, fpsT0 = 0, bytesSum = 0, camRestarts = 0, lastFrameMs = 0;
float fps = 0, avgBytes = 0;
String line;

bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;
  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = frameSize;
  c.jpeg_quality = quality;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) return false;
  esp_log_level_set("cam_hal", ESP_LOG_NONE);  // driver log lines would land inside JPEG data
  // Grey picture: the IR image has no colour anyway, and without chroma the JPEG is ~30 %
  // smaller, which is what limits the frame rate on the 921600-baud link.
  sensor_t* sen = esp_camera_sensor_get();
  if (sen) sen->set_special_effect(sen, gray ? 2 : 0);
  return true;
}

void restartCamera() {
  esp_camera_deinit();
  delay(50);
  ++camRestarts;
  if (!initCamera()) Serial.println("ERR camera init failed");
}

void printStatus() {
  int w = 320, h = 240;
  if (frameSize == FRAMESIZE_CIF) w = 400, h = 296;
  if (frameSize == FRAMESIZE_VGA) w = 640, h = 480;
  sensor_t* sen = esp_camera_sensor_get();
  Serial.printf("STATUS size %dx%d q %d fps %.1f avg %.0f B/frame stream %d restarts %u PID 0x%02x psram %u\n", w, h,
                quality, millis() - lastFrameMs > 1000 ? 0.0f : fps, avgBytes, streaming, camRestarts,
                sen ? sen->id.PID : 0, ESP.getFreePsram());
}

void handleCommand(String cmd) {
  cmd.trim();
  if (!cmd.length()) return;
  char word[16] = {0}, arg[16] = {0};
  int n = sscanf(cmd.c_str(), "%15s %15s", word, arg);
  String w(word), a(arg);
  sensor_t* sen = esp_camera_sensor_get();
  if (w == "help") {
    Serial.println("INFO stream on|off, size qvga|cif|vga, q N, gray on|off, exp auto|N, led N, reg ADDR [VAL], status");
  } else if (w == "stream" && n == 2) {
    streaming = a == "on";
  } else if (w == "size" && n == 2) {
    frameSize = a == "vga" ? FRAMESIZE_VGA : a == "cif" ? FRAMESIZE_CIF : FRAMESIZE_QVGA;
    if (sen) sen->set_framesize(sen, frameSize);
  } else if (w == "q" && n == 2) {
    quality = constrain(a.toInt(), 4, 63);
    if (sen) sen->set_quality(sen, quality);
  } else if (w == "exp" && n == 2 && sen) {
    if (a == "auto") {
      sen->set_exposure_ctrl(sen, 1);
    } else {
      sen->set_exposure_ctrl(sen, 0);
      sen->set_aec_value(sen, a.toInt());
    }
  } else if (w == "gray" && n == 2) {
    gray = a == "on";
    if (sen) sen->set_special_effect(sen, gray ? 2 : 0);
  } else if (w == "led" && n == 2) {
    analogWrite(kLedPin, constrain(a.toInt(), 0, 255));
  } else if (w == "reg" && n >= 2 && sen) {
    unsigned addr = 0, value = 0;
    int k = sscanf(cmd.c_str(), "%*s %x %x", &addr, &value);
    if (k == 2) sen->set_reg(sen, addr, 0xff, value);
    Serial.printf("REG 0x%03x = 0x%02x\n", addr, sen->get_reg(sen, addr, 0xff));
  } else if (w == "status") {
    printStatus();
  } else {
    Serial.printf("ERR unknown command: %s\n", cmd.c_str());
  }
}

}  // namespace

void setup() {
  Serial.setTxBufferSize(32768);  // must come before begin() on arduino-esp32
  Serial.begin(921600);
  pinMode(kLedPin, OUTPUT);
  analogWrite(kLedPin, 0);
  if (!initCamera()) Serial.println("ERR camera init failed");
  fpsT0 = millis();
  Serial.println("INFO mouth streamer ready");
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') {
      handleCommand(line);
      line = "";
    } else if (line.length() < 64) {
      line += ch;
    }
  }
  if (!streaming) {
    delay(5);
    return;
  }
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    if (millis() - lastFrameMs > 2000) {  // the camera drops out when the board is touched
      restartCamera();
      lastFrameMs = millis();
    }
    return;
  }
  lastFrameMs = millis();
  Serial.printf("JPG %u %u %u %u %u\n", seq++, fb->width, fb->height, (unsigned)fb->len, (unsigned)lastFrameMs);
  Serial.write(fb->buf, fb->len);
  bytesSum += fb->len;
  esp_camera_fb_return(fb);
  ++fpsFrames;
  if (millis() - fpsT0 >= 1000) {
    fps = fpsFrames * 1000.0f / (millis() - fpsT0);
    avgBytes = fpsFrames ? bytesSum / (float)fpsFrames : 0;
    fpsFrames = bytesSum = 0;
    fpsT0 = millis();
  }
}
