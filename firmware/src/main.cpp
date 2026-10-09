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
//   stream on|off                      print one BROW line per frame (on after power-up)
//   snap                               dump the current frame: "SNAP 320 240\n" + raw bytes
//                                      (blocks tracking for ~0.9 s)
//   preview off|1|2|4                  keep sending frames downscaled 1/2/4x for the PC UI, in
//                                      chunks between tracked frames: "IMG seq w h off len\n" +
//                                      len raw bytes. Tracking keeps running at full rate.
//   curve                              calibrated curves: "CURVE L|R a b c sx0 sx1 sx2 .. sxN"
//                                      (y = a x^2 + b x + c in image px, strip edges sx)
//   bench                              time one tracker update on the current frame
//   exp auto|N                         exposure: automatic, or fixed N lines with auto gain
//                                      (experimental: dim light makes auto exposure drop fps)
//   reg ADDR [VALUE]                   read / write a camera register (hex, debugging)
//   status                             ROIs, calibration, frame rate
//   reset                              forget ROIs and calibration
// Output line: BROW seq=.. ms=.. rL=.. iL=.. oL=.. vL=.. nL=.. rR=.. iR=.. oR=.. vR=.. nR=..
//   r/i/o = raise / inner end / outer end in px (positive = up), v = valid,
//   n = raise scaled by the "range" calibration (-1 .. +1), empty if not set.
//   While preview is on, sL=/sR= add the per-strip shifts (px, negative = up, comma list).
#include <Arduino.h>
#include <Preferences.h>

#include "brow_core.h"
#include "camera_pins.h"
#include "esp_camera.h"

// Tracker::update keeps ~4 KB of profiles on the stack (more on the re-acquire path); the
// default 8 KB loopTask stack overflowed there on hardware (stack canary panic, 2026-10-09).
SET_LOOP_TASK_STACK_SIZE(16 * 1024);

namespace {

constexpr int kW = 320;
constexpr int kH = 240;
constexpr int kMaxCalib = 30;
constexpr uint32_t kStateVersion = 2;  // bump when brow::Tracker changes layout

brow::Tracker trackers[2];  // 0 = ROI "L" (image left), 1 = ROI "R" (image right)
bool roiSet[2] = {false, false};
float rangeUp[2] = {0, 0};
float rangeDown[2] = {0, 0};
float lastRaise[2] = {0, 0};
Preferences prefs;
bool streaming = true;  // the values are the device's output: on from power-up
int calibWanted = 0;
int calibHave = 0;
uint8_t* calibBuf = nullptr;  // PSRAM, kMaxCalib frames
uint32_t seq = 0;
uint32_t fpsFrames = 0;
uint32_t fpsT0 = 0;
uint32_t shortFrames = 0;
uint32_t camRestarts = 0;
uint32_t lastFrameMs = 0;
float fps = 0;
String line;

// Preview for the PC UI: a downscaled copy of the latest frame, sent in chunks whenever the
// Serial TX buffer has room, so a frame never blocks the tracking loop.
constexpr size_t kTxBuffer = 16384;
constexpr size_t kChunk = 2048;
int pvScale = 0;  // 0 = off
uint8_t* pvBuf = nullptr;  // PSRAM, one full frame
int pvW = 0, pvH = 0;
size_t pvLen = 0, pvSent = 0;
uint32_t pvSeq = 0;

void previewCapture(const uint8_t* img) {
  const int s = pvScale;
  pvW = kW / s;
  pvH = kH / s;
  const int area = s * s;
  for (int y = 0; y < pvH; ++y) {
    for (int x = 0; x < pvW; ++x) {
      int acc = 0;
      for (int dy = 0; dy < s; ++dy) {
        const uint8_t* row = img + (y * s + dy) * kW + x * s;
        for (int dx = 0; dx < s; ++dx) acc += row[dx];
      }
      pvBuf[y * pvW + x] = static_cast<uint8_t>(acc / area);
    }
  }
  pvLen = static_cast<size_t>(pvW) * pvH;
  pvSent = 0;
  ++pvSeq;
}

void previewPump() {
  while (pvSent < pvLen) {
    const size_t room = Serial.availableForWrite();
    // keep ~1 KB free so the BROW lines never wait behind image data
    if (room < 1024 + 64 + 256) return;
    const size_t n = min(min(kChunk, pvLen - pvSent), room - 1024 - 64);
    Serial.printf("IMG %u %d %d %u %u\n", pvSeq, pvW, pvH, (unsigned)pvSent, (unsigned)n);
    Serial.write(pvBuf + pvSent, n);
    pvSent += n;
  }
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
  c.pixel_format = PIXFORMAT_GRAYSCALE;  // raw luma: no JPEG decode on the tracking path
  c.frame_size = FRAMESIZE_QVGA;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) return false;
  // esp32-camera runs the OV2640 at a quarter of its clock in non-JPEG modes (CLKRC
  // divider 3 -> 12.5 fps at QVGA). Divider 1 gives clean 25 fps on this board; divider 0
  // is too fast for the I2S DMA (every frame arrives a few lines short). Measured 2026-10-09.
  sensor_t* sen = esp_camera_sensor_get();
  if (sen && sen->id.PID == OV2640_PID) sen->set_reg(sen, 0x111, 0xff, 0x01);
  // Short frames are counted in status instead; the driver's log line would land in the
  // middle of the BROW stream.
  esp_log_level_set("cam_hal", ESP_LOG_NONE);
  return true;
}

// NVS keeps the old copy of a blob until the new one is written, so rewriting two ~2 KB
// trackers needs twice the space; removing the old blob first keeps the default 20 KB
// partition from running out.
void putBlob(const char* key, const void* data, size_t len) {
  prefs.remove(key);
  if (prefs.putBytes(key, data, len) != len) Serial.printf("ERR could not save %s to NVS\n", key);
}

void saveState() {
  prefs.putUInt("ver", kStateVersion);
  putBlob("roi", roiSet, sizeof(roiSet));
  putBlob("t0", &trackers[0], sizeof(brow::Tracker));
  putBlob("t1", &trackers[1], sizeof(brow::Tracker));
  putBlob("up", rangeUp, sizeof(rangeUp));
  putBlob("down", rangeDown, sizeof(rangeDown));
}

void loadState() {
  if (prefs.getUInt("ver", 0) != kStateVersion) return;
  prefs.getBytes("roi", roiSet, sizeof(roiSet));
  if (prefs.getBytesLength("t0") == sizeof(brow::Tracker)) prefs.getBytes("t0", &trackers[0], sizeof(brow::Tracker));
  if (prefs.getBytesLength("t1") == sizeof(brow::Tracker)) prefs.getBytes("t1", &trackers[1], sizeof(brow::Tracker));
  prefs.getBytes("up", rangeUp, sizeof(rangeUp));
  prefs.getBytes("down", rangeDown, sizeof(rangeDown));
}

// A full grey frame, or nullptr (the driver occasionally delivers a short one).
camera_fb_t* grabFull() {
  for (int attempt = 0; attempt < 4; ++attempt) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) continue;
    if (fb->width == kW && fb->height == kH && fb->len == kW * kH) return fb;
    esp_camera_fb_return(fb);
  }
  return nullptr;
}

void printStatus() {
  for (int i = 0; i < 2; ++i) {
    const brow::Tracker& t = trackers[i];
    Serial.printf("ROI %c: %s  [%d %d %d %d]  calibrated=%d  K=%d S=%d h=%d  range +%.1f/%.1f px\n",
                  i ? 'R' : 'L', roiSet[i] ? "set" : "unset", t.x0, t.y0, t.x1, t.y1, t.calibrated(),
                  t.config().half_range, t.config().search, t.config().ncc_half, rangeUp[i], rangeDown[i]);
  }
  sensor_t* sen = esp_camera_sensor_get();
  Serial.printf("fps %.1f  short frames %u  camera restarts %u  sensor PID 0x%02x  free heap %u  free psram %u  "
                "min free stack %u\n",
                millis() - lastFrameMs > 1000 ? 0.0f : fps, shortFrames, camRestarts, sen ? sen->id.PID : 0,
                ESP.getFreeHeap(), ESP.getFreePsram(), (unsigned)uxTaskGetStackHighWaterMark(nullptr));
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
    Serial.println(
        "roi L|R x0 y0 x1 y1 left|right | calib [n] | range up|down | stream on|off | snap | preview off|1|2|4 | "
        "curve | bench | exp auto|N | reg ADDR [VALUE] | status | reset");
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
    camera_fb_t* fb = grabFull();
    if (!fb) {
      Serial.println("ERR no frame");
      return;
    }
    Serial.printf("SNAP %d %d\n", fb->width, fb->height);
    Serial.write(fb->buf, fb->len);
    Serial.println();
    esp_camera_fb_return(fb);
  } else if (w == "bench") {
    camera_fb_t* fb = grabFull();
    if (!fb) return;
    for (int i = 0; i < 2; ++i) {
      if (!trackers[i].calibrated()) continue;
      static brow::Tracker copy;  // not on the stack: a Tracker is ~2 KB
      copy = trackers[i];         // do not disturb the live state
      uint32_t t0 = micros();
      for (int k = 0; k < 20; ++k) copy.update(fb->buf, fb->width, fb->height);
      Serial.printf("bench %c: %.2f ms per update\n", i ? 'R' : 'L', (micros() - t0) / 20000.0f);
    }
    esp_camera_fb_return(fb);
  } else if (w == "preview") {
    int s = 0;
    sscanf(cmd.c_str(), "preview %d", &s);
    if (s != 1 && s != 2 && s != 4) s = 0;
    if (s && !pvBuf) {
      Serial.println("ERR no PSRAM buffer");
      return;
    }
    pvScale = s;
    pvLen = pvSent = 0;
    Serial.printf("OK preview %d\n", s);
  } else if (w == "curve") {
    for (int i = 0; i < 2; ++i) {
      const brow::Tracker& t = trackers[i];
      if (!t.calibrated()) continue;
      Serial.printf("CURVE %c %.9g %.9g %.9g", i ? 'R' : 'L', t.curve[0], t.curve[1], t.curve[2]);
      const int S = t.config().strips;
      for (int s = 0; s < S; ++s) Serial.printf(" %d", t.strip_x0[s]);
      Serial.printf(" %d\n", t.strip_x1[S - 1]);
    }
    Serial.println("CURVE end");
  } else if (w == "exp") {
    // exp auto | exp N: automatic exposure, or a fixed exposure of N lines (0..1200) with
    // the gain still automatic. In dim light auto exposure stretches frames (25 -> 3-8 fps).
    sensor_t* sen = esp_camera_sensor_get();
    int v = -1;
    if (!sen || (cmd.indexOf("auto") < 0 && sscanf(cmd.c_str(), "exp %d", &v) != 1)) {
      Serial.println("ERR usage: exp auto | exp N (0..1200)");
      return;
    }
    if (v < 0) {
      sen->set_exposure_ctrl(sen, 1);
    } else {
      sen->set_exposure_ctrl(sen, 0);
      sen->set_aec_value(sen, constrain(v, 0, 1200));
    }
    sen->set_gain_ctrl(sen, 1);
    Serial.printf("OK exp %s\n", v < 0 ? "auto" : String(v).c_str());
  } else if (w == "reg") {
    // reg ADDR [VALUE]: read / write a camera register (hex). OV2640: 0x1xx = sensor bank,
    // 0x0xx = DSP bank (esp32-camera convention).
    unsigned addr = 0, value = 0;
    int n = sscanf(cmd.c_str(), "reg %x %x", &addr, &value);
    sensor_t* sen = esp_camera_sensor_get();
    if (n < 1 || !sen) {
      Serial.println("ERR usage: reg ADDR [VALUE] (hex)");
      return;
    }
    if (n == 2) sen->set_reg(sen, addr, 0xff, value);
    Serial.printf("reg 0x%03x = 0x%02x\n", addr, sen->get_reg(sen, addr, 0xff));
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
    } else if (ch >= ' ' && ch <= '~' && line.length() < 96) {
      // printable only: line noise around a reset must not stick to the next command
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
  Serial.setTxBufferSize(kTxBuffer);  // must come before begin() on arduino-esp32
  Serial.begin(921600);
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
  pvBuf = static_cast<uint8_t*>(ps_malloc(static_cast<size_t>(kW) * kH));
  prefs.begin("brow", false);
  loadState();
  fpsT0 = millis();
  Serial.println("brow_cam ready, type help");
  printStatus();
}

void loop() {
  pollSerial();
  previewPump();
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    // Seen once (2026-10-09, right after a flash + reset): the board kept answering but the
    // camera delivered no frames at all. Restart the camera instead of tracking nothing.
    if (millis() - lastFrameMs > 2000) {
      Serial.println("camera stalled - restarting it");
      esp_camera_deinit();
      if (initCamera()) ++camRestarts;
      lastFrameMs = millis();
    }
    return;
  }
  // The driver occasionally hands over a short frame (DMA lost a few lines): skip it.
  if (fb->width != kW || fb->height != kH || fb->format != PIXFORMAT_GRAYSCALE || fb->len != kW * kH) {
    ++shortFrames;
    esp_camera_fb_return(fb);
    return;
  }
  ++fpsFrames;
  uint32_t now = millis();
  lastFrameMs = now;
  if (now - fpsT0 >= 1000) {
    fps = fpsFrames * 1000.0f / (now - fpsT0);
    fpsFrames = 0;
    fpsT0 = now;
  }
  if (pvScale && pvSent >= pvLen) previewCapture(fb->buf);
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
    char buf[384];
    int n = snprintf(buf, sizeof(buf), "BROW seq=%u ms=%.2f", ++seq, ms);
    for (int i = 0; i < 2; ++i) {
      char s = i ? 'R' : 'L';
      if (!have[i]) continue;
      n += snprintf(buf + n, sizeof(buf) - n, " r%c=%.2f i%c=%.2f o%c=%.2f v%c=%d", s, res[i].raise, s,
                    res[i].inner, s, res[i].outer, s, res[i].valid);
      float nv = normalised(i);
      if (!isnan(nv)) n += snprintf(buf + n, sizeof(buf) - n, " n%c=%.2f", s, nv);
      if (pvScale) {
        const int S = trackers[i].config().strips;
        for (int k = 0; k < S; ++k)
          n += snprintf(buf + n, sizeof(buf) - n, "%s%.1f", k ? "," : (i ? " sR=" : " sL="), res[i].off[k]);
      }
    }
    Serial.println(buf);
  }
}
