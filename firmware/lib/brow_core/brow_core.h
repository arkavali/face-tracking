// Brow tracker core: C++ port of tools/tracker.py (method "ncc"), no Arduino dependencies.
//
// One Tracker per brow. Input frames are 8-bit grey, row-major, stride = width.
// Calibration (neutral face): per ROI column, the topmost bright->dark step with a
// brow-thick dark band below it; a quadratic curve through those points "unrolls"
// the brow. Tracking: per strip along the curve, a 1-D profile is matched against
// the neutral profile with normalised cross-correlation around the previous shift.
//
// Kept numerically identical to the Python reference (same rounding, same order of
// operations) so tools/parity_test.py can compare the two frame by frame.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef BROW_REAL
#define BROW_REAL float  // float on the ESP32 (no double FPU); the host test uses double
#endif

namespace brow {

using real = BROW_REAL;

constexpr int kMaxStrips = 8;
constexpr int kMaxProfile = 255;   // 2 * half_range + 1
constexpr int kMaxTemplate = 127;  // 2 * ncc_half + 1

struct Config {
  int strips = 5;
  // Window sizes in rows; 0 = derive from the ROI height (ROI ~ 30 mm tall).
  int half_range = 0;  // K, profile spans curve-K .. curve+K
  int edge_half = 0;   // edge filter half width (~1.2 mm)
  int search = 0;      // max shift change per frame (~2.5 mm)
  int ncc_half = 0;    // neutral template rows each side of the edge (~4 mm)
  int smooth = 1;      // box radius for profile smoothing
  real min_edge = 4;
  real min_ncc = real(0.5);
  real range_mm = 15;
  real search_mm = real(2.5);
  real dist_penalty = 0;  // NCC penalty per search radius of distance from the previous shift
  real cand_frac = real(0.25);
  int lost_frames = 15;  // invalid this many frames in a row -> search the whole range
};

struct Result {
  real off[kMaxStrips];    // per-strip shift, px, negative = up
  bool ok[kMaxStrips];
  real score[kMaxStrips];  // NCC of the match
  real raise;              // px, positive = up (weighted mean of strips)
  real inner;              // mean of the two strips towards the nose
  real outer;
  bool valid;              // at least half of the strips matched
};

class Tracker {
 public:
  // ROI in pixels [x0, x1) x [y0, y1); inner_is_right: which ROI side is towards the nose.
  void setup(int x0, int y0, int x1, int y1, bool inner_is_right, const Config& cfg = Config());

  // Fit the brow curve on the per-pixel median of n neutral frames (w x h).
  // Allocates ~ (x1 - x0) * h reals temporarily. Returns false if no brow was found.
  bool calibrate(const uint8_t* const* frames, int n, int w, int h);

  // Track one frame. Call only after a successful calibrate().
  Result update(const uint8_t* img, int w, int h);

  bool calibrated() const { return calibrated_; }
  const Config& config() const { return cfg_; }

  // y = curve[0] * x^2 + curve[1] * x + curve[2] (numpy polyval order)
  double curve[3] = {0, 0, 0};
  int strip_x0[kMaxStrips] = {0};
  int strip_x1[kMaxStrips] = {0};
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;

 private:
  // Mean grey along the curve for strip s, rows curve(x)-K .. curve(x)+K.
  // img column index = x - col_offset (0 for full frames, x0 for the ROI median image).
  template <typename Pix>
  void profile(const Pix* img, int stride, int col_offset, int h, int s, real* out) const;
  int topmost_brow_edge(const real* p, const real* g, int n) const;

  Config cfg_;
  bool inner_is_right_ = true;
  bool calibrated_ = false;
  int lost_ = 0;
  real prev_[kMaxStrips] = {0};
  real tz_[kMaxStrips][kMaxTemplate];  // standardised neutral templates (2 * ncc_half + 1)
};

// Plain data only: a calibrated Tracker can be saved/restored as raw bytes (e.g. NVS).
static_assert(__is_trivially_copyable(Tracker), "Tracker must stay trivially copyable");

}  // namespace brow
