// Eye camera: ESP32-S3 CAM grabs 320x240 grey frames and tracks one eye on-chip
// (lib/eye_core, a port of tools/pupil.py + tools/eyemodel.py). Output and setup over the
// native USB serial port for now; sending the values to main_hub is the next step.
//
// Serial commands (one per line):
//   help                    this list
//   stream on|off           print one EYE line per frame (on after power-up)
//   snap                    dump the current frame: "SNAP 320 240\n" + raw bytes
//   preview off|1|2|4       keep sending frames downscaled 1/2/4x for the PC tools, in chunks
//                           between tracked frames: "IMG seq w h off len\n" + len raw bytes
//   bench                   time one detector update on the current frame
//   exp auto|N              exposure: automatic, or fixed N lines with auto gain
//   reg ADDR [VALUE]        read / write a camera register (hex, debugging)
//   status                  frame rate, memory, model state
//   reset                   forget the eye model (re-learns while you look around)
// Output line:
//   EYE seq=.. ms=.. p=0|1 [cx= cy= a= b= ang= con= top=] [ex= ey= R=] [gx= gy= gz=] [g1x= g1y= g2x= g2y=]
//   p = pupil accepted (0 = no pupil: blink or lost), cx/cy/a/b/ang = pupil ellipse (px, rad),
//   con = ring contrast, top = visible share of the pupil under the upper lid,
//   ex/ey/R = eyeball centre and radius in the image (once the model is ready),
//   gx/gy/gz = gaze unit vector in camera axes, g1/g2 = the two nearest corneal glints.
// The wearer-specific calibration (AR target points, glint mapping, blink/squint ranges)
// runs on the hub / PC from these raw values.
#include <Arduino.h>

#include "camera_pins.h"
#include "esp_camera.h"
#include "eye_core.h"

SET_LOOP_TASK_STACK_SIZE(16 * 1024);

namespace {

constexpr int kW = 320;
constexpr int kH = 240;
constexpr float kFocalPx = 308.0f;  // OV2640 66 deg diagonal at 320x240 (pinhole estimate)

eye::real* work = nullptr;  // PSRAM scratch for the detector
eye::PupilDetector* detector = nullptr;
eye::PupilGate gate;
eye::EyeModel* model = nullptr;
bool streaming = true;
uint32_t seq = 0;
uint32_t fpsFrames = 0, fpsT0 = 0, shortFrames = 0, camRestarts = 0, lastFrameMs = 0;
float fps = 0;
String line;

// Preview for the PC tools (same scheme as brow_tracker): a downscaled copy of the latest
// frame, sent in chunks whenever the TX buffer has room, so a frame never blocks tracking.
constexpr size_t kTxBuffer = 16384;
constexpr size_t kChunk = 4096;
int pvScale = 0;
uint8_t* pvBuf = nullptr;
int pvW = 0, pvH = 0;
size_t pvLen = 0, pvSent = 0;
uint32_t pvSeq = 0;

void previewCapture(const uint8_t* img) {
  const int s = pvScale;
  pvW = kW / s;
  pvH = kH / s;
  for (int y = 0; y < pvH; ++y)
    for (int x = 0; x < pvW; ++x) {
      int acc = 0;
      for (int dy = 0; dy < s; ++dy) {
        const uint8_t* row = img + (y * s + dy) * kW + x * s;
        for (int dx = 0; dx < s; ++dx) acc += row[dx];
      }
      pvBuf[y * pvW + x] = static_cast<uint8_t>(acc / (s * s));
    }
  pvLen = static_cast<size_t>(pvW) * pvH;
  pvSent = 0;
  ++pvSeq;
}

void previewPump() {
  while (pvSent < pvLen) {
    const size_t room = Serial.availableForWrite();
    if (room < 1024 + 64 + 256) return;  // keep room so EYE lines never wait behind image data
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
  c.pixel_format = PIXFORMAT_GRAYSCALE;
  c.frame_size = FRAMESIZE_QVGA;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) return false;
  // As on the brow ESP32-CAM: esp32-camera runs the OV2640 at a quarter clock in non-JPEG
  // modes (12.5 fps); CLKRC divider 1 gave 25 fps there. Re-check on the S3 (its camera
  // DMA is different; divider 0 may work too) - see "status" fps and short frames.
  sensor_t* sen = esp_camera_sensor_get();
  if (sen && sen->id.PID == OV2640_PID) sen->set_reg(sen, 0x111, 0xff, 0x01);
  esp_log_level_set("cam_hal", ESP_LOG_NONE);
  return true;
}

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
  sensor_t* sen = esp_camera_sensor_get();
  Serial.printf("fps %.1f  short frames %u  camera restarts %u  sensor PID 0x%02x  free heap %u  free psram %u  "
                "min free stack %u\n",
                millis() - lastFrameMs > 1000 ? 0.0f : fps, shortFrames, camRestarts, sen ? sen->id.PID : 0,
                ESP.getFreeHeap(), ESP.getFreePsram(), (unsigned)uxTaskGetStackHighWaterMark(nullptr));
  if (model->ready())
    Serial.printf("eye model: centre %.1f %.1f  R %.1f px\n", model->cx(), model->cy(), model->radius());
  else
    Serial.println("eye model: learning (look around)");
}

void handleCommand(String cmd) {
  cmd.trim();
  if (!cmd.length()) return;
  char word[16] = {0};
  sscanf(cmd.c_str(), "%15s", word);
  String w(word);
  if (w == "help") {
    Serial.println("stream on|off | snap | preview off|1|2|4 | bench | exp auto|N | reg ADDR [VALUE] | status | reset");
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
    static eye::real* benchWork = nullptr;  // separate detector: keeps the live RANSAC state
    if (!benchWork) benchWork = static_cast<eye::real*>(ps_malloc(sizeof(eye::real) * kW * kH));
    static eye::PupilDetector* det = new eye::PupilDetector(benchWork, kW, kH);
    eye::Ellipse e;
    int found = 0;
    uint32_t t0 = micros();
    for (int k = 0; k < 20; ++k) found += det->detect(fb->buf, &e);
    Serial.printf("bench: %.2f ms per detect (pupil found %d/20)\n", (micros() - t0) / 20000.0f, found);
    esp_camera_fb_return(fb);
  } else if (w == "preview") {
    int s = 0;
    sscanf(cmd.c_str(), "preview %d", &s);
    if (s != 1 && s != 2 && s != 4) s = 0;
    pvScale = s;
    pvLen = pvSent = 0;
    Serial.printf("OK preview %d\n", s);
  } else if (w == "exp") {
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
    delete model;
    model = new eye::EyeModel(kFocalPx, kW, kH);
    gate = eye::PupilGate();
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
      line += ch;
    }
  }
}

}  // namespace

void setup() {
  Serial.setTxBufferSize(kTxBuffer);
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
  work = static_cast<eye::real*>(ps_malloc(sizeof(eye::real) * kW * kH));
  pvBuf = static_cast<uint8_t*>(ps_malloc(static_cast<size_t>(kW) * kH));
  detector = new eye::PupilDetector(work, kW, kH);
  model = new eye::EyeModel(kFocalPx, kW, kH);
  fpsT0 = millis();
  Serial.println("eye_cam ready, type help");
  printStatus();
}

void loop() {
  pollSerial();
  previewPump();
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    if (millis() - lastFrameMs > 2000) {
      Serial.println("camera stalled - restarting it");
      esp_camera_deinit();
      if (initCamera()) ++camRestarts;
      lastFrameMs = millis();
    }
    return;
  }
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

  uint32_t t0 = micros();
  eye::Ellipse e;
  bool found = detector->detect(fb->buf, &e);
  bool ok = gate(found ? &e : nullptr);
  eye::real g[3];
  bool hasGaze = false;
  eye::Glint gl[eye::kMaxGlints];
  int ng = 0;
  if (ok) {
    model->update(e);
    hasGaze = model->gazeCam(e, g);
    ng = eye::findGlints(fb->buf, kW, kH, e, gl);
  }
  float ms = (micros() - t0) / 1000.0f;
  esp_camera_fb_return(fb);
  if (!streaming) return;
  char buf[320];
  int n = snprintf(buf, sizeof(buf), "EYE seq=%u ms=%.2f p=%d", ++seq, ms, ok ? 1 : 0);
  if (ok)
    n += snprintf(buf + n, sizeof(buf) - n, " cx=%.2f cy=%.2f a=%.2f b=%.2f ang=%.3f con=%.0f top=%.2f", e.cx, e.cy,
                  e.a, e.b, e.angle, e.contrast, e.top_vis);
  if (model->ready())
    n += snprintf(buf + n, sizeof(buf) - n, " ex=%.1f ey=%.1f R=%.1f", model->cx(), model->cy(), model->radius());
  if (hasGaze) n += snprintf(buf + n, sizeof(buf) - n, " gx=%.3f gy=%.3f gz=%.3f", g[0], g[1], g[2]);
  if (ng >= 2)
    n += snprintf(buf + n, sizeof(buf) - n, " g1x=%.1f g1y=%.1f g2x=%.1f g2y=%.1f", gl[0].x, gl[0].y, gl[1].x, gl[1].y);
  Serial.println(buf);
}
