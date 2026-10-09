// C API around brow::Tracker for the host parity test (ctypes). Not used on the ESP32.
#include <stdint.h>

#include <vector>

#include "brow_core.h"

extern "C" {

__declspec(dllexport) void* brow_new(int x0, int y0, int x1, int y1, int inner_is_right, int strips,
                                     double range_mm, double cand_frac, int lost_frames) {
  brow::Config cfg;
  cfg.strips = strips;
  cfg.range_mm = static_cast<brow::real>(range_mm);
  cfg.cand_frac = static_cast<brow::real>(cand_frac);
  cfg.lost_frames = lost_frames;
  auto* t = new brow::Tracker();
  t->setup(x0, y0, x1, y1, inner_is_right != 0, cfg);
  return t;
}

__declspec(dllexport) void brow_free(void* t) { delete static_cast<brow::Tracker*>(t); }

// frames: n consecutive w*h grey images
__declspec(dllexport) int brow_calibrate(void* t, const uint8_t* frames, int n, int w, int h) {
  std::vector<const uint8_t*> ptrs(n);
  for (int i = 0; i < n; ++i) ptrs[i] = frames + static_cast<size_t>(i) * w * h;
  return static_cast<brow::Tracker*>(t)->calibrate(ptrs.data(), n, w, h) ? 1 : 0;
}

// out: raise, inner, outer, valid, then strips x (off, ok, score)
__declspec(dllexport) void brow_update(void* t, const uint8_t* img, int w, int h, double* out) {
  auto* tr = static_cast<brow::Tracker*>(t);
  brow::Result r = tr->update(img, w, h);
  out[0] = r.raise;
  out[1] = r.inner;
  out[2] = r.outer;
  out[3] = r.valid;
  const int S = tr->config().strips;
  for (int s = 0; s < S; ++s) {
    out[4 + 3 * s] = r.off[s];
    out[5 + 3 * s] = r.ok[s];
    out[6 + 3 * s] = r.score[s];
  }
}

// out: c0 c1 c2, K, edge_half, search, ncc_half, then strips x (x0, x1)
__declspec(dllexport) void brow_info(void* t, double* out) {
  auto* tr = static_cast<brow::Tracker*>(t);
  const brow::Config& c = tr->config();
  out[0] = tr->curve[0];
  out[1] = tr->curve[1];
  out[2] = tr->curve[2];
  out[3] = c.half_range;
  out[4] = c.edge_half;
  out[5] = c.search;
  out[6] = c.ncc_half;
  for (int s = 0; s < c.strips; ++s) {
    out[7 + 2 * s] = tr->strip_x0[s];
    out[8 + 2 * s] = tr->strip_x1[s];
  }
}
}
