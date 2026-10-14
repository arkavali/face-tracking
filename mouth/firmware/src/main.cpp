// Mouth camera: ESP32-CAM either streams hardware-JPEG frames to the PC (live view, recordings:
// tools/mouth_cam.py, calib_rec.py) or runs the mouth CNN on the chip (lib/mouth_net, weights
// from tools/export_c.py) and sends the values (tools/mouth_live.py --board).
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
//   mode stream|track     JPEG streaming (default) or on-chip tracking
//   preview N             track mode: send the 80x60 net input every N-th frame (0 = off, default 2)
//   bench                 track mode: time downscale + net on the current frame
//   fit zero [N]          per-wearing adjustment: neutral face, average N frames (default 20)
//   fit max [S]           per-wearing adjustment: for S seconds (default 8) make every expression as
//                         strongly as possible; the per-channel maximum becomes 1.0
//   fit on|off|clear|show use / ignore / forget / print the adjustment (kept in NVS)
// Frame: "JPG seq w h len ms\n" + len bytes of JPEG.
// Track: "MOUTH seq ms net_ms fit v0 .. v21\n" (values x1000, channel order: "CH" line after mode
//        track; fit = 1 when the values are adjusted: (raw - zero) / (max - zero), clipped 0..1)
//        and "PRV seq 80 60 4800\n" + 4800 grey bytes when preview is on.
// Text replies start with other words.
#include <Arduino.h>
#include <Preferences.h>

#include "camera_pins.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "mouth_net.h"

namespace {

constexpr int kLedPin = 4;
bool streaming = true;
int quality = 14;
bool gray = true;
framesize_t frameSize = FRAMESIZE_QVGA;
uint32_t seq = 0, fpsFrames = 0, fpsT0 = 0, bytesSum = 0, camRestarts = 0, lastFrameMs = 0;
float fps = 0, avgBytes = 0;
String line;
bool tracking = false;
int previewEvery = 2;
float* netA = nullptr;
float* netB = nullptr;
uint8_t small[80 * 60];
float netOut[32];
float netMs = 0;

// Per-wearing adjustment: the mask sits a little differently every time, so the net's neutral
// output is not exactly 0 and the strongest expression not exactly 1.
float fitZero[32], fitMax[32], smooth[32];
bool fitOn = false, fitHave = false;
int fitZeroLeft = 0, fitZeroN = 0;
uint32_t fitMaxUntil = 0;
float zeroAcc[32];
Preferences prefs;

void fitSave() {
  prefs.putInt("n", mouth::outputCount());
  prefs.putBytes("zero", fitZero, sizeof(float) * mouth::outputCount());
  prefs.putBytes("max", fitMax, sizeof(float) * mouth::outputCount());
  prefs.putBool("on", fitOn);
}

void fitLoad() {
  if (prefs.getInt("n", 0) != mouth::outputCount()) return;
  prefs.getBytes("zero", fitZero, sizeof(float) * mouth::outputCount());
  prefs.getBytes("max", fitMax, sizeof(float) * mouth::outputCount());
  fitOn = prefs.getBool("on", false);
  fitHave = true;
}

void fitUpdate(const float* raw) {
  const int n = mouth::outputCount();
  for (int i = 0; i < n; ++i) smooth[i] = 0.5f * smooth[i] + 0.5f * raw[i];
  if (fitZeroLeft > 0) {
    for (int i = 0; i < n; ++i) zeroAcc[i] += raw[i];
    if (--fitZeroLeft == 0) {
      for (int i = 0; i < n; ++i) {
        fitZero[i] = zeroAcc[i] / fitZeroN;
        if (!fitHave || fitMax[i] < fitZero[i] + 0.05f) fitMax[i] = 1.0f;
      }
      fitHave = fitOn = true;
      fitSave();
      Serial.println("INFO fit zero done");
    }
  }
  if (fitMaxUntil) {
    for (int i = 0; i < n; ++i)
      if (smooth[i] > fitMax[i]) fitMax[i] = smooth[i];
    if ((int32_t)(millis() - fitMaxUntil) >= 0) {
      fitMaxUntil = 0;
      fitHave = fitOn = true;
      fitSave();
      Serial.println("INFO fit max done");
    }
  }
}

float fitted(int i, float v) {
  if (!fitOn || !fitHave) return v;
  const float span = fitMax[i] - fitZero[i];
  if (span < 0.05f) return max(0.0f, v - fitZero[i]);
  return constrain((v - fitZero[i]) / span, 0.0f, 1.0f);
}

float* allocFloats(size_t n) {
  // internal RAM is several times faster than PSRAM on the classic ESP32; PSRAM as a fallback
  void* p = heap_caps_malloc(n * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!p) p = heap_caps_malloc(n * sizeof(float), MALLOC_CAP_SPIRAM);
  return (float*)p;
}

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
  c.pixel_format = tracking ? PIXFORMAT_GRAYSCALE : PIXFORMAT_JPEG;
  c.frame_size = frameSize;
  c.jpeg_quality = quality;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) return false;
  esp_log_level_set("cam_hal", ESP_LOG_NONE);  // driver log lines would land inside JPEG data
  sensor_t* s0 = esp_camera_sensor_get();
  // non-JPEG modes run the OV2640 at a quarter clock (12.5 fps); divider 1 = 25 fps (brow_tracker)
  if (tracking && s0 && s0->id.PID == OV2640_PID) s0->set_reg(s0, 0x111, 0xff, 0x01);
  // Grey picture: the IR image has no colour anyway, and without chroma the JPEG is ~30 %
  // smaller, which is what limits the frame rate on the 921600-baud link.
  sensor_t* sen = esp_camera_sensor_get();
  if (sen && !tracking) sen->set_special_effect(sen, gray ? 2 : 0);
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
  Serial.printf("STATUS mode %s size %dx%d q %d fps %.1f avg %.0f B/frame net %.1f ms stream %d restarts %u "
                "PID 0x%02x heap %u psram %u\n", tracking ? "track" : "stream", w, h, quality,
                millis() - lastFrameMs > 1000 ? 0.0f : fps, avgBytes, netMs, streaming, camRestarts,
                sen ? sen->id.PID : 0, ESP.getFreeHeap(), ESP.getFreePsram());
}

void handleCommand(String cmd) {
  cmd.trim();
  if (!cmd.length()) return;
  char word[16] = {0}, arg[16] = {0};
  int n = sscanf(cmd.c_str(), "%15s %15s", word, arg);
  String w(word), a(arg);
  sensor_t* sen = esp_camera_sensor_get();
  if (w == "help") {
    Serial.println("INFO stream on|off, size qvga|cif|vga, q N, gray on|off, exp auto|N, led N, reg ADDR [VAL], "
                   "status, mode stream|track, preview N, bench, fit zero|max|on|off|clear|show");
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
  } else if (w == "mode" && n == 2) {
    bool want = a == "track";
    if (want && !netA) {
      size_t na, nb;
      mouth::scratchSizes(&na, &nb);
      netA = allocFloats(na);
      netB = allocFloats(nb);
      bool inRam = mouth::init([](size_t n) { return heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT); });
      Serial.printf("INFO net weights in %s, heap left %u\n", inRam ? "RAM" : "flash (partly)", ESP.getFreeHeap());
      if (!netA || !netB) {
        Serial.println("ERR no memory for the net");
        return;
      }
    }
    if (want != tracking) {
      tracking = want;
      esp_camera_deinit();
      delay(50);
      if (!initCamera()) Serial.println("ERR camera init failed");
    }
    if (tracking) {
      Serial.print("CH");
      for (int i = 0; i < mouth::outputCount(); ++i) Serial.printf(" %s", mouth::channelName(i));
      Serial.println();
    }
  } else if (w == "fit" && n >= 2) {
    const int nOut = mouth::outputCount();
    if (a == "zero") {
      int frames = 20;
      sscanf(cmd.c_str(), "%*s %*s %d", &frames);
      fitZeroN = fitZeroLeft = constrain(frames, 1, 200);
      for (int i = 0; i < nOut; ++i) zeroAcc[i] = 0;
      Serial.printf("INFO fit zero: keep a neutral face for %d frames\n", fitZeroN);
    } else if (a == "max") {
      int sec = 8;
      sscanf(cmd.c_str(), "%*s %*s %d", &sec);
      if (!fitHave)
        for (int i = 0; i < nOut; ++i) fitZero[i] = 0;
      for (int i = 0; i < nOut; ++i) fitMax[i] = fitZero[i];
      fitMaxUntil = millis() + 1000u * constrain(sec, 1, 60);
      Serial.printf("INFO fit max: make every expression strongly for %d s\n", sec);
    } else if (a == "on" || a == "off") {
      fitOn = a == "on" && fitHave;
      prefs.putBool("on", fitOn);
    } else if (a == "clear") {
      fitHave = fitOn = false;
      prefs.clear();
    }
    Serial.printf("FIT %s", fitOn ? "on" : fitHave ? "off" : "none");
    for (int i = 0; i < nOut && fitHave; ++i)
      Serial.printf(" %s:%d-%d", mouth::channelName(i), (int)(fitZero[i] * 1000), (int)(fitMax[i] * 1000));
    Serial.println();
  } else if (w == "preview" && n == 2) {
    previewEvery = max(0, (int)a.toInt());
  } else if (w == "bench") {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb || !tracking) {
      Serial.println("ERR bench needs mode track");
      if (fb) esp_camera_fb_return(fb);
      return;
    }
    uint32_t t0 = micros();
    mouth::downscale4(fb->buf, fb->width, fb->height, small);
    uint32_t t1 = micros();
    mouth::run(small, netA, netB, netOut);
    uint32_t t2 = micros();
    esp_camera_fb_return(fb);
    Serial.printf("INFO bench downscale %.2f ms net %.1f ms\n", (t1 - t0) / 1000.0f, (t2 - t1) / 1000.0f);
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
  prefs.begin("mouthfit", false);
  fitLoad();
  fpsT0 = millis();
  Serial.println("INFO mouth camera ready");
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
  if (tracking) {
    if (fb->width != 320 || fb->height != 240 || fb->len != 320 * 240) {  // short frame
      esp_camera_fb_return(fb);
      return;
    }
    mouth::downscale4(fb->buf, fb->width, fb->height, small);
    esp_camera_fb_return(fb);  // the net works on the small copy: the driver can refill meanwhile
    uint32_t t0 = micros();
    mouth::run(small, netA, netB, netOut);
    netMs = (micros() - t0) / 1000.0f;
    fitUpdate(netOut);
    char buf[256];
    int len = snprintf(buf, sizeof(buf), "MOUTH %u %u %.1f %d", seq, (unsigned)lastFrameMs, netMs,
                       (fitOn && fitHave) ? 1 : 0);
    for (int i = 0; i < mouth::outputCount() && len < (int)sizeof(buf) - 8; ++i)
      len += snprintf(buf + len, sizeof(buf) - len, " %d", (int)(fitted(i, netOut[i]) * 1000 + 0.5f));
    Serial.println(buf);
    if (previewEvery && seq % previewEvery == 0) {
      Serial.printf("PRV %u 80 60 4800\n", seq);
      Serial.write(small, sizeof(small));
    }
    ++seq;
  } else {
    Serial.printf("JPG %u %u %u %u %u\n", seq++, fb->width, fb->height, (unsigned)fb->len, (unsigned)lastFrameMs);
    Serial.write(fb->buf, fb->len);
    bytesSum += fb->len;
    esp_camera_fb_return(fb);
  }
  ++fpsFrames;
  if (millis() - fpsT0 >= 1000) {
    fps = fpsFrames * 1000.0f / (millis() - fpsT0);
    avgBytes = fpsFrames ? bytesSum / (float)fpsFrames : 0;
    fpsFrames = bytesSum = 0;
    fpsT0 = millis();
  }
}
