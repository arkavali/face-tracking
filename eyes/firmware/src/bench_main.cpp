// Camera-less bench build (env s3bench): the PC sends grey frames over the UART, the chip
// runs the same detector + gate + eye model + glints as the camera firmware and returns
// the result and the time per frame. Used by tools/hw_bench.py to check timing and
// float-precision parity on a real ESP32-S3 before the camera boards arrive.
//
// Protocol (lines, 2 Mbaud):
//   PC:  "F w h\n" + w*h raw bytes      ->  "R ms found cx cy a b ang support contrast top gated
//                                            ready mcx mcy R gx gy gz ng g1x g1y g2x g2y"
//   PC:  "bench N\n"                     ->  "B ms_per_detect" (last frame, N repeats, own detector)
//   PC:  "reset\n"                       ->  fresh detector / gate / model ("OK reset")
//   PC:  "info\n"                        ->  chip, clock, memory
#ifdef EYE_BENCH
#include <Arduino.h>

#include "esp_heap_caps.h"
#include "eye_core.h"

namespace {

constexpr int kMaxW = 320, kMaxH = 240;
uint8_t* frame = nullptr;
uint16_t* work = nullptr;
eye::PupilDetector* det = nullptr;
eye::PupilGate* gate = nullptr;
eye::EyeModel* model = nullptr;
eye::GlintPairer* pairer = nullptr;
int fw = 0, fh = 0;

void fresh(int w, int h) {
  delete det;
  delete gate;
  delete model;
  delete pairer;
  pairer = new eye::GlintPairer();
  det = new eye::PupilDetector(work, w, h);
  gate = new eye::PupilGate();
  model = new eye::EyeModel(308.0f, w, h);
}

bool readExact(uint8_t* dst, size_t n, uint32_t timeout_ms) {
  size_t got = 0;
  uint32_t t0 = millis();
  while (got < n) {
    int a = Serial.available();
    if (a > 0) {
      got += Serial.readBytes(dst + got, min((size_t)a, n - got));
      t0 = millis();
    } else if (millis() - t0 > timeout_ms) {
      return false;
    }
  }
  return true;
}

void runFrame() {
  eye::Ellipse e;
  uint32_t t0 = micros();
  bool found = det->detect(frame, &e);
  bool ok = (*gate)(found ? &e : nullptr);
  eye::real g[3] = {0, 0, 0};
  eye::Glint gl[eye::kMaxGlints];
  int ng = 0;
  if (ok) {
    model->update(e);
    model->gazeCam(e, g);
    ng = eye::findGlints(frame, fw, fh, e, gl);
  }
  float ms = (micros() - t0) / 1000.0f;
  if (!found) e = eye::Ellipse{0, 0, 0, 0, 0, 0, 0, 0, 0};
  Serial.printf("R %.3f %d %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %d %d %.6g %.6g %.6g %.6g %.6g %.6g %d", ms,
                found, e.cx, e.cy, e.a, e.b, e.angle, e.support, e.contrast, e.top_vis, ok, model->ready(),
                model->cx(), model->cy(), model->radius(), g[0], g[1], g[2], ng);
  eye::Glint l{0, 0, 0}, r{0, 0, 0};
  if (ok && !(*pairer)(gl, ng, &l, &r)) l = r = eye::Glint{0, 0, 0};
  Serial.printf(" %.6g %.6g %.6g %.6g", l.x, l.y, r.x, r.y);
  Serial.println();
}

}  // namespace

void setup() {
  Serial.setRxBufferSize(16384);
  Serial.begin(2000000);
  frame = static_cast<uint8_t*>(ps_malloc(kMaxW * kMaxH));
  // detector scratch in internal RAM (falls back to PSRAM)
  work = static_cast<uint16_t*>(heap_caps_malloc(sizeof(uint16_t) * kMaxW * kMaxH, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (!work) work = static_cast<uint16_t*>(ps_malloc(sizeof(uint16_t) * kMaxW * kMaxH));
  fw = kMaxW;
  fh = kMaxH;
  fresh(fw, fh);
  Serial.println("eye_bench ready");
}

void loop() {
  if (!Serial.available()) return;
  String ln = Serial.readStringUntil('\n');
  ln.trim();
  int w, h, n;
  if (sscanf(ln.c_str(), "F %d %d", &w, &h) == 2 && w > 0 && h > 0 && w <= kMaxW && h <= kMaxH) {
    if (w != fw || h != fh) {
      fw = w;
      fh = h;
      fresh(fw, fh);
    }
    if (!readExact(frame, (size_t)w * h, 1000)) {
      Serial.println("ERR short frame");
      return;
    }
    runFrame();
  } else if (sscanf(ln.c_str(), "bench %d", &n) == 1) {
    eye::PupilDetector d(work, fw, fh);  // shares the scratch: frames are processed one at a time
    eye::Ellipse e;
    uint32_t t0 = micros();
    for (int k = 0; k < n; ++k) d.detect(frame, &e);
    Serial.printf("B %.3f  last: blur %u seed %u starburst %u ransac %u checks %u us\n",
                  (micros() - t0) / 1000.0f / max(n, 1), (unsigned)d.prof_us[0], (unsigned)d.prof_us[1],
                  (unsigned)d.prof_us[2], (unsigned)d.prof_us[3], (unsigned)d.prof_us[4]);
  } else if (ln == "reset") {
    fresh(fw, fh);
    Serial.println("OK reset");
  } else if (ln == "info") {
    Serial.printf("chip %s rev %d, %d MHz, free heap %u, free psram %u, psram %u\n", ESP.getChipModel(),
                  ESP.getChipRevision(), ESP.getCpuFreqMHz(), ESP.getFreeHeap(), ESP.getFreePsram(),
                  ESP.getPsramSize());
  } else if (ln.length()) {
    Serial.println("ERR unknown");
  }
}
#endif  // EYE_BENCH
