// Brow camera: ESP32-CAM grabs 320x240 grey frames and tracks two brows on-chip
// (lib/brow_core, a port of tools/tracker.py). Setup and output over Serial for now;
// sending the values to main_hub over UDP is the next step.
//
// Serial commands (921600 baud, one per line):
//   help                               this list
//   roi L|R x0 y0 x1 y1 left|right     brow ROI in image pixels; L/R = ROI on the image
//                                      left/right; last word = which ROI side faces the nose
//   calib [n]                          neutral face: average n frames (default 15) and calibrate
//   range up|down                      store the current raise as the +1 / -1 end of the scale
//   stream on|off                      print one BROW line per frame
//   snap                               dump the current frame: "SNAP 320 240\n" + raw bytes
//   bench                              time one tracker update on the current frame
//   status                             ROIs, calibration, frame rate
//   reset                              forget ROIs and calibration
// Output line: BROW seq=.. ms=.. rL=.. iL=.. oL=.. vL=.. nL=.. rR=.. iR=.. oR=.. vR=.. nR=..
//   r/i/o = raise / inner end / outer end in px (positive = up), v = valid,
//   n = raise scaled by the "range" calibration (-1 .. +1), empty if not set.
#include <Arduino.h>
#include <Preferences.h>

#include "brow_core.h"
#include "camera_pins.h"
#include "esp_camera.h"

namespace {

constexpr int kW = 320;
constexpr int kH = 240;
constexpr int kMaxCalib = 30;
constexpr uint32_t kStateVersion = 1;

brow::Tracker trackers[2];  // 0 = ROI "L" (image left), 1 = ROI "R" (image right)
bool roiSet[2] = {false, false};
float rangeUp[2] = {0, 0};
float rangeDown[2] = {0, 0};
float lastRaise[2] = {0, 0};
Preferences prefs;
bool streaming = false;
int calibWanted = 0;
int calibHave = 0;
uint8_t* calibBuf = nullptr;  // PSRAM, kMaxCalib frames
uint32_t seq = 0;
uint32_t fpsFrames = 0;
uint32_t fpsT0 = 0;
float fps = 0;
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
  c.pixel_format = PIXFORMAT_GRAYSCALE;  // raw luma: no JPEG decode on the tracking path
  c.frame_size = FRAMESIZE_QVGA;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  return esp_camera_init(&c) == ESP_OK;
}

void saveState() {
  prefs.putUInt("ver", kStateVersion);
  prefs.putBytes("roi", roiSet, sizeof(roiSet));
  prefs.putBytes("t0", &trackers[0], sizeof(brow::Tracker));
  prefs.putBytes("t1", &trackers[1], sizeof(brow::Tracker));
  prefs.putBytes("up", rangeUp, sizeof(rangeUp));
  prefs.putBytes("down", rangeDown, sizeof(rangeDown));
}

void loadState() {
  if (prefs.getUInt("ver", 0) != kStateVersion) return;
  prefs.getBytes("roi", roiSet, sizeof(roiSet));
  if (prefs.getBytesLength("t0") == sizeof(brow::Tracker)) prefs.getBytes("t0", &trackers[0], sizeof(brow::Tracker));
  if (prefs.getBytesLength("t1") == sizeof(brow::Tracker)) prefs.getBytes("t1", &trackers[1], sizeof(brow::Tracker));
  prefs.getBytes("up", rangeUp, sizeof(rangeUp));
  prefs.getBytes("down", rangeDown, sizeof(rangeDown));
}

void printStatus() {
  for (int i = 0; i < 2; ++i) {
    const brow::Tracker& t = trackers[i];
    Serial.printf("ROI %c: %s  [%d %d %d %d]  calibrated=%d  K=%d S=%d h=%d  range +%.1f/%.1f px\n",
                  i ? 'R' : 'L', roiSet[i] ? "set" : "unset", t.x0, t.y0, t.x1, t.y1, t.calibrated(),
                  t.config().half_range, t.config().search, t.config().ncc_half, rangeUp[i], rangeDown[i]);
  }
  Serial.printf("fps %.1f  free heap %u  free psram %u\n", fps, ESP.getFreeHeap(), ESP.getFreePsram());
}

float normalised(int i) {
  float r = lastRaise[i];
  if (r >= 0 && rangeUp[i] > 0.5f) return min(r / rangeUp[i], 1.5f);
  if (r < 0 && rangeDown[i] < -0.5f) return max(-r / rangeDown[i], -1.5f);
  return NAN;
}

void handleCommand(String cmd) {
  cmd.trim();
  if (!cmd.length()) return;
  char word[16] = {0};
  sscanf(cmd.c_str(), "%15s", word);
  String w(word);
  if (w == "help") {
    Serial.println("roi L|R x0 y0 x1 y1 left|right | calib [n] | range up|down | stream on|off | snap | bench | status | reset");
  } else if (w == "roi") {
    char side = 0;
    int x0, y0, x1, y1;
    char inner[8] = {0};
    if (sscanf(cmd.c_str(), "roi %c %d %d %d %d %7s", &side, &x0, &y0, &x1, &y1, inner) != 6 ||
        (side != 'L' && side != 'R') || x0 < 0 || y0 < 0 || x1 > kW || y1 > kH || x1 - x0 < 16 || y1 - y0 < 16) {
      Serial.println("ERR usage: roi L|R x0 y0 x1 y1 left|right (inside 320x240)");
      return;
    }
    int i = side == 'R';
    trackers[i].setup(x0, y0, x1, y1, String(inner) == "right");
    roiSet[i] = true;
    rangeUp[i] = rangeDown[i] = 0;
    saveState();
    Serial.printf("OK roi %c set, run calib with a neutral face\n", side);
  } else if (w == "calib") {
    int n = 15;
    sscanf(cmd.c_str(), "calib %d", &n);
    n = constrain(n, 1, kMaxCalib);
    if (!calibBuf) {
      Serial.println("ERR no PSRAM buffer");
      return;
    }
    calibWanted = n;
    calibHave = 0;
    Serial.printf("OK collecting %d frames - hold a neutral face\n", n);
  } else if (w == "range") {
    bool up = cmd.indexOf("up") > 0;
    for (int i = 0; i < 2; ++i) (up ? rangeUp : rangeDown)[i] = lastRaise[i];
    saveState();
    Serial.printf("OK range %s = %.1f / %.1f px\n", up ? "up" : "down", lastRaise[0], lastRaise[1]);
  } else if (w == "stream") {
    streaming = cmd.indexOf("on") > 0;
    Serial.printf("OK stream %s\n", streaming ? "on" : "off");
  } else if (w == "snap") {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("ERR no frame");
      return;
    }
    Serial.printf("SNAP %d %d\n", fb->width, fb->height);
    Serial.write(fb->buf, fb->len);
    Serial.println();
    esp_camera_fb_return(fb);
  } else if (w == "bench") {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) return;
    for (int i = 0; i < 2; ++i) {
      if (!trackers[i].calibrated()) continue;
      brow::Tracker copy = trackers[i];  // do not disturb the live state
      uint32_t t0 = micros();
      for (int k = 0; k < 20; ++k) copy.update(fb->buf, fb->width, fb->height);
      Serial.printf("bench %c: %.2f ms per update\n", i ? 'R' : 'L', (micros() - t0) / 20000.0f);
    }
    esp_camera_fb_return(fb);
  } else if (w == "status") {
    printStatus();
  } else if (w == "reset") {
    prefs.clear();
    roiSet[0] = roiSet[1] = false;
    trackers[0] = brow::Tracker();
    trackers[1] = brow::Tracker();
    Serial.println("OK reset");
  } else {
    Serial.println("ERR unknown command, try help");
  }
}

void pollSerial() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (line.length()) handleCommand(line);
      line = "";
    } else if (line.length() < 96) {
      line += ch;
    }
  }
}

void finishCalibration() {
  const uint8_t* ptrs[kMaxCalib];
  for (int k = 0; k < calibHave; ++k) ptrs[k] = calibBuf + static_cast<size_t>(k) * kW * kH;
  for (int i = 0; i < 2; ++i) {
    if (!roiSet[i]) continue;
    bool ok = trackers[i].calibrate(ptrs, calibHave, kW, kH);
    Serial.printf("calib %c: %s\n", i ? 'R' : 'L', ok ? "OK" : "FAILED (no brow edge in the ROI)");
  }
  saveState();
  calibWanted = 0;
}

}  // namespace

void setup() {
  Serial.begin(921600);
  Serial.setTxBufferSize(4096);
  delay(300);
  bool cam = initCamera();
  for (int attempt = 0; !cam && attempt < 5; ++attempt) {
    Serial.println("camera init failed - retrying");
    delay(1000);
    cam = initCamera();
  }
  if (!cam) {
    Serial.println("camera init failed - restarting");
    delay(200);
    esp_restart();
  }
  calibBuf = static_cast<uint8_t*>(ps_malloc(static_cast<size_t>(kMaxCalib) * kW * kH));
  prefs.begin("brow", false);
  loadState();
  fpsT0 = millis();
  Serial.println("brow_cam ready, type help");
  printStatus();
}

void loop() {
  pollSerial();
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return;
  if (fb->width != kW || fb->height != kH || fb->format != PIXFORMAT_GRAYSCALE) {
    esp_camera_fb_return(fb);
    return;
  }
  ++fpsFrames;
  uint32_t now = millis();
  if (now - fpsT0 >= 1000) {
    fps = fpsFrames * 1000.0f / (now - fpsT0);
    fpsFrames = 0;
    fpsT0 = now;
  }
  if (calibWanted) {
    memcpy(calibBuf + static_cast<size_t>(calibHave) * kW * kH, fb->buf, kW * kH);
    esp_camera_fb_return(fb);
    if (++calibHave >= calibWanted) finishCalibration();
    return;
  }
  brow::Result res[2] = {};
  bool have[2] = {false, false};
  uint32_t t0 = micros();
  for (int i = 0; i < 2; ++i) {
    if (!trackers[i].calibrated()) continue;
    res[i] = trackers[i].update(fb->buf, kW, kH);
    lastRaise[i] = res[i].raise;
    have[i] = true;
  }
  float ms = (micros() - t0) / 1000.0f;
  esp_camera_fb_return(fb);
  if (streaming && (have[0] || have[1])) {
    char buf[200];
    int n = snprintf(buf, sizeof(buf), "BROW seq=%u ms=%.2f", ++seq, ms);
    for (int i = 0; i < 2; ++i) {
      char s = i ? 'R' : 'L';
      if (!have[i]) continue;
      n += snprintf(buf + n, sizeof(buf) - n, " r%c=%.2f i%c=%.2f o%c=%.2f v%c=%d", s, res[i].raise, s,
                    res[i].inner, s, res[i].outer, s, res[i].valid);
      float nv = normalised(i);
      if (!isnan(nv)) n += snprintf(buf + n, sizeof(buf) - n, " n%c=%.2f", s, nv);
    }
    Serial.println(buf);
  }
}
