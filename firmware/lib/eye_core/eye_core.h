// Eye tracker core: C++ port of tools/pupil.py and tools/eyemodel.py, no Arduino
// dependencies.
//
// PupilDetector: darkest box seed on the raw frame, 3x3 box blur x2 (integer, separable,
// only in a window around the seed) on a 2x-downsampled frame, starburst
// rays (glint skip, sub-pixel step), RANSAC over 5-point subsets with the Fitzgibbon
// direct ellipse fit, refit on inliers, dark-inside / ring-contrast / upper-lid checks.
// PupilGate: rejects weak pupils against the running median contrast.
// findGlints: small saturated blobs near the pupil.
// EyeModel: eyeball centre as the decaying least-squares intersection of the pupil
// ellipses' minor axes, radius from |p - c|^2 = R^2 sin^2(theta), gaze vector.
//
// Kept numerically identical to the Python reference (same order of operations, same
// xorshift RANSAC subsets) so tools/parity_test.py can compare frame by frame.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef EYE_REAL
#define EYE_REAL float  // float on the ESP32-S3 (single-precision FPU); the host test uses double
#endif

namespace eye {

using real = EYE_REAL;

constexpr int kMaxRays = 64;
constexpr int kMaxRaySamples = 128;
constexpr int kGateHist = 100;
constexpr int kMaxGlints = 4;

struct Params {
  // Seed box (px at full resolution), per camera placement: 16 behind the AR mirror,
  // 8 from below (see PupilParams in tools/pupil.py).
  int box = 16;
  int box2 = 0;             // optional second seed box, both followed (0 = off)
  int rays = 36;
  real r_min = 2;
  real r_max = 45;          // px
  real step = 1;
  real grad = 14;           // dark->bright step over 2 px (grey levels)
  real above_seed = 18;     // bright side must be this much above the pupil level
  real glint_level = 235;
  int rounds = 3;
  int ransac_iter = 40;
  real inlier_px = real(1.2);
  int min_points = 10;
  real max_axis_ratio = 4;
  real min_support = real(0.4);
  real max_inside = 25;
};

struct Ellipse {
  real cx, cy;
  real a, b;          // semi-major, semi-minor
  real angle;         // rad, major axis direction
  real support;       // inlier rays / all rays
  int n;
  real contrast;      // ring around the ellipse minus the pupil level
  real top_vis;       // visible share of the pupil's upward extent (1 = lid clear)
};

class XorShift32 {
 public:
  explicit XorShift32(uint32_t s = 2463534242u) : s_(s ? s : 1) {}
  uint32_t next() {
    uint32_t x = s_;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s_ = x;
    return x;
  }

 private:
  uint32_t s_;
};

class PupilDetector {
 public:
  // work: scratch for the blurred window, (2 * (r_max + 12))^2 uint16 (~22 KB with the
  // defaults; w*h always suffices), kept by the caller, internal RAM preferred.
  PupilDetector(uint16_t* work, int w, int h, const Params& p = Params(), uint32_t seed = 0);
  ~PupilDetector();
  // true and *out filled when a pupil is found.
  bool detect(const uint8_t* img, Ellipse* out);
  const Params& params() const { return p_; }
  Params& params() { return p_; }
  // Microseconds spent in blur, seed, starburst, ransac, checks during the last detect()
  // (filled only when built with EYE_PROFILE).
  uint32_t prof_us[5] = {0, 0, 0, 0, 0};

 private:
  void blurWindow(int cx, int cy, int half);
  void seed(int box, real* sx, real* sy, real* level);
  bool detectFrom(real sx, real sy, real level, Ellipse* out);
  uint16_t blurPx(int x, int y) const;
  uint16_t px81(int x, int y) const {  // blurred x 81: from the window if inside, else computed
    return (x >= wx0_ && x < wx1_ && y >= wy0_ && y < wy1_) ? buf_[(y - wy0_) * ww_ + (x - wx0_)] : blurPx(x, y);
  }
  int starburst(real sx, real sy, real level, real* px, real* py);
  real at(real x, real y) const;  // bilinear on the blurred frame, clamped

  const uint8_t* img_ = nullptr;
  uint16_t* buf_;             // blurred window x 81
  uint16_t* rows_ = nullptr;  // 5 horizontally filtered rows of the window (ring)
  int wx0_ = 0, wx1_ = 0, wy0_ = 0, wy1_ = 0, ww_ = 0;
  int w_, h_;
  Params p_;
  XorShift32 rng_;
};

class PupilGate {
 public:
  explicit PupilGate(real share = real(0.45), real min_support = real(0.55)) : share_(share), min_support_(min_support) {}
  bool operator()(const Ellipse* e);  // e may be null

 private:
  real share_, min_support_;
  real hist_[kGateHist];
  int n_ = 0;
};

struct Glint {
  real x, y;
  int size;
};

// Up to kMaxGlints blobs (>= level, <= 40 px) within reach * e.a of the pupil centre,
// nearest first. A blob counts only as an isolated peak: the mean of the 1-px frame 2 px
// outside its bounding box must be <= level - isolate. Returns the count.
int findGlints(const uint8_t* img, int w, int h, const Ellipse& e, Glint* out, int level = 230,
               real reach = 3, real isolate = 50);

// Picks the two LED glints (left, right) among the candidates by their learned offset;
// see GlintPairer in tools/pupil.py.
class GlintPairer {
 public:
  explicit GlintPairer(int learn = 50, real tol = real(0.35)) : learn_(learn), tol_(tol) {}
  bool operator()(const Glint* gl, int n, Glint* left, Glint* right);
  bool learned() const { return n_ >= learn_; }

 private:
  static constexpr int kMaxLearn = 64;
  int learn_;
  real tol_;
  real sx_[kMaxLearn], sy_[kMaxLearn];
  int n_ = 0;
  real ref_[2] = {0, 0};
};

class EyeModel {
 public:
  EyeModel(real f, int w, int h, real half_life_s = 20, real fps = 25, real min_ecc = real(0.15));
  void update(const Ellipse& e);
  bool ready() const { return has_c_ && has_r_; }
  real cx() const { return c_[0]; }
  real cy() const { return c_[1]; }
  real radius() const { return R_; }
  // Unit gaze in camera axes (x right, y down, z forward); false until ready.
  bool gazeCam(const Ellipse& e, real g[3]) const;

 private:
  real f_;
  int w_, h_;
  real decay_, min_ecc_;
  real A_[3] = {0, 0, 0};  // symmetric 2x2: a00, a01, a11
  real bv_[2] = {0, 0};
  real wsum_ = 0;
  real c_[2] = {0, 0};
  bool has_c_ = false;
  real r_num_ = 0, r_den_ = 0;
  int r_count_ = 0;
  real R_ = 0;
  bool has_r_ = false;
};

// Exposed for the host test.
bool fitEllipse(const real* x, const real* y, int n, real conic[6]);
bool conicToParams(const real c[6], real* cx, real* cy, real* a, real* b, real* ang);

}  // namespace eye
