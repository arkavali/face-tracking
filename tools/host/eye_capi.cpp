// C API around eye_core for the host parity test (ctypes). Not used on the ESP32.
#include <stdint.h>
#include <stdlib.h>

#include "eye_core.h"

extern "C" {

struct Ctx {
  eye::real* work;
  eye::PupilDetector* det;
  eye::PupilGate gate;
  eye::EyeModel* model;
};

__declspec(dllexport) void* eye_new(int w, int h, double f) {
  auto* c = new Ctx();
  c->work = static_cast<eye::real*>(malloc(sizeof(eye::real) * w * h));
  c->det = new eye::PupilDetector(c->work, w, h);
  c->model = new eye::EyeModel(static_cast<eye::real>(f), w, h);
  return c;
}

__declspec(dllexport) void eye_free(void* p) {
  auto* c = static_cast<Ctx*>(p);
  delete c->det;
  delete c->model;
  free(c->work);
  delete c;
}

// out: found, cx, cy, a, b, angle, support, contrast, top_vis, gated, model_ready, mcx, mcy,
//      R, gx, gy, gz, n_glints, then 4 x (glint x, y)
__declspec(dllexport) void eye_update(void* p, const uint8_t* img, int w, int h, double* out) {
  auto* c = static_cast<Ctx*>(p);
  for (int i = 0; i < 26; ++i) out[i] = 0;
  eye::Ellipse e;
  bool found = c->det->detect(img, &e);
  out[0] = found;
  if (found) {
    out[1] = e.cx;
    out[2] = e.cy;
    out[3] = e.a;
    out[4] = e.b;
    out[5] = e.angle;
    out[6] = e.support;
    out[7] = e.contrast;
    out[8] = e.top_vis;
  }
  bool ok = c->gate(found ? &e : nullptr);
  out[9] = ok;
  if (ok) {
    c->model->update(e);
    eye::real g[3];
    if (c->model->gazeCam(e, g)) {
      out[14] = g[0];
      out[15] = g[1];
      out[16] = g[2];
    }
    eye::Glint gl[eye::kMaxGlints];
    int n = eye::findGlints(img, w, h, e, gl);
    out[17] = n;
    for (int i = 0; i < n; ++i) {
      out[18 + 2 * i] = gl[i].x;
      out[19 + 2 * i] = gl[i].y;
    }
  }
  out[10] = c->model->ready();
  out[11] = c->model->cx();
  out[12] = c->model->cy();
  out[13] = c->model->radius();
}

}  // extern "C"
