#include "brow_core.h"

#include <math.h>

#include <algorithm>
#include <vector>

namespace brow {
namespace {

const real kNegInf = -INFINITY;

// Python round(): ties to even (nearbyint with the default rounding mode).
inline int rint_even(double v) { return static_cast<int>(nearbyint(v)); }

// Box filter with edge padding, radius r (numpy: pad(mode="edge") + convolve "valid").
void box(const real* p, int n, int r, real* out) {
  if (r <= 0) {
    for (int i = 0; i < n; ++i) out[i] = p[i];
    return;
  }
  const real k = real(1) / real(2 * r + 1);
  for (int i = 0; i < n; ++i) {
    real acc = 0;
    for (int j = -r; j <= r; ++j) acc += p[std::min(std::max(i + j, 0), n - 1)] * k;
    out[i] = acc;
  }
}

// g[k] = mean(p[k-m:k]) - mean(p[k:k+m]) for k in [m, n-m], -inf elsewhere.
void edge_strength(const real* p, int n, int m, real* g) {
  // cumulative sums, c[0] = 0
  real c[kMaxProfile + 1 > 512 ? kMaxProfile + 1 : 512];
  std::vector<real> big;
  real* cs = c;
  if (n + 1 > static_cast<int>(sizeof(c) / sizeof(c[0]))) {
    big.resize(n + 1);
    cs = big.data();
  }
  cs[0] = 0;
  for (int i = 0; i < n; ++i) cs[i + 1] = cs[i] + p[i];
  for (int k = 0; k < n; ++k) g[k] = kNegInf;
  for (int k = m; k <= n - m; ++k) {
    real above = (cs[k] - cs[k - m]) / m;
    real below = (cs[k + m] - cs[k]) / m;
    g[k] = above - below;
  }
}

real parabola(const real* g, int n, int i) {
  if (i > 0 && i < n - 1 && isfinite(g[i - 1]) && isfinite(g[i + 1])) {
    real den = g[i - 1] - 2 * g[i] + g[i + 1];
    if (den < 0) return i + real(0.5) * (g[i - 1] - g[i + 1]) / den;
  }
  return static_cast<real>(i);
}

real mean_of(const real* p, int a, int b) {
  real s = 0;
  for (int i = a; i < b; ++i) s += p[i];
  return s / (b - a);
}

double median_of(std::vector<double> v) {
  size_t n = v.size();
  std::sort(v.begin(), v.end());
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Weighted least-squares quadratic, numpy.polyfit(x, y, 2, w=w) semantics
// (w multiplies the residuals, i.e. weights w^2 in the normal equations).
bool polyfit2(const std::vector<double>& x, const std::vector<double>& y,
              const std::vector<double>& w, double out[3]) {
  if (x.size() < 3) return false;
  double A[3][4] = {{0}};
  for (size_t i = 0; i < x.size(); ++i) {
    double ww = w[i] * w[i];
    double v[3] = {x[i] * x[i], x[i], 1.0};
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) A[r][c] += ww * v[r] * v[c];
      A[r][3] += ww * v[r] * y[i];
    }
  }
  for (int col = 0; col < 3; ++col) {  // Gaussian elimination with partial pivoting
    int piv = col;
    for (int r = col + 1; r < 3; ++r)
      if (fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
    if (fabs(A[piv][col]) < 1e-300) return false;
    for (int c = 0; c < 4; ++c) std::swap(A[col][c], A[piv][c]);
    for (int r = 0; r < 3; ++r) {
      if (r == col) continue;
      double f = A[r][col] / A[col][col];
      for (int c = col; c < 4; ++c) A[r][c] -= f * A[col][c];
    }
  }
  for (int r = 0; r < 3; ++r) out[r] = A[r][3] / A[r][r];
  return true;
}

inline double polyval2(const double c[3], double x) { return (c[0] * x + c[1]) * x + c[2]; }

}  // namespace

void Tracker::setup(int ax0, int ay0, int ax1, int ay1, bool inner_is_right, const Config& cfg) {
  x0 = ax0;
  y0 = ay0;
  x1 = ax1;
  y1 = ay1;
  inner_is_right_ = inner_is_right;
  cfg_ = cfg;
  double ppm = std::max(y1 - y0, 10) / 30.0;
  if (!cfg_.half_range) cfg_.half_range = std::max(12, rint_even(cfg.range_mm * ppm));
  if (!cfg_.edge_half) cfg_.edge_half = std::max(2, rint_even(1.2 * ppm));
  if (!cfg_.search) cfg_.search = std::max(4, rint_even(cfg.search_mm * ppm));
  if (!cfg_.ncc_half) cfg_.ncc_half = std::max(5, rint_even(4 * ppm));
  cfg_.half_range = std::min(cfg_.half_range, (kMaxProfile - 1) / 2);
  cfg_.ncc_half = std::min(cfg_.ncc_half, (kMaxTemplate - 1) / 2);
  cfg_.strips = std::min(std::max(cfg_.strips, 1), kMaxStrips);
  calibrated_ = false;
  lost_ = 0;
  for (int s = 0; s < kMaxStrips; ++s) prev_[s] = 0;
}

int Tracker::topmost_brow_edge(const real* p, const real* g, int n) const {
  const int band = 3 * cfg_.edge_half;
  real gmax = kNegInf;
  for (int i = 0; i < n; ++i)
    if (isfinite(g[i]) && g[i] > gmax) gmax = g[i];
  if (!isfinite(gmax) || gmax < cfg_.min_edge) return -1;
  const real thr = std::max(cfg_.min_edge, cfg_.cand_frac * gmax);
  for (int i = 1; i < n - 1; ++i) {
    if (!(isfinite(g[i]) && g[i] >= thr && g[i] >= g[i - 1] && g[i] >= g[i + 1])) continue;
    if (i - band < 0 || i + band > n) continue;
    if (mean_of(p, i - band, i) - mean_of(p, i, i + band) >= real(0.6) * g[i]) return i;
  }
  return -1;
}

template <typename Pix>
void Tracker::profile(const Pix* img, int stride, int col_offset, int h, int s, real* out) const {
  const int K = cfg_.half_range;
  const int a = strip_x0[s], b = strip_x1[s];
  for (int k = 0; k < 2 * K + 1; ++k) out[k] = 0;
  for (int x = a; x < b; ++x) {
    // floor(yc + k) = floor(yc) + k, so the bilinear weights are the same for every row
    // of this column: one double evaluation per column, float/int work in the loop
    // (the ESP32 has no double FPU).
    const double yc = polyval2(curve, x);
    const double base = floor(yc);
    const real f = static_cast<real>(yc - base);
    const real f0 = 1 - f;
    const int ib = static_cast<int>(base);
    const int cx = x - col_offset;
    for (int k = -K; k <= K; ++k) {
      int r0c = std::min(std::max(ib + k, 0), h - 1);
      int r1c = std::min(std::max(ib + k + 1, 0), h - 1);
      out[k + K] += img[r0c * stride + cx] * f0 + img[r1c * stride + cx] * f;
    }
  }
  const real inv = real(1) / (b - a);
  for (int k = 0; k < 2 * K + 1; ++k) out[k] *= inv;
}

bool Tracker::calibrate(const uint8_t* const* frames, int n, int w, int h) {
  calibrated_ = false;
  if (n < 1 || x1 <= x0 || y1 <= y0) return false;
  // Median image over the ROI columns, full height (profiles may reach outside the ROI rows).
  const int rw = x1 - x0;
  std::vector<real> med(static_cast<size_t>(rw) * h);
  std::vector<uint8_t> v(n);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < rw; ++x) {
      for (int i = 0; i < n; ++i) v[i] = frames[i][y * w + x0 + x];
      std::sort(v.begin(), v.end());
      med[y * rw + x] = (n % 2) ? real(v[n / 2]) : real(0.5) * (real(v[n / 2 - 1]) + real(v[n / 2]));
    }
  }
  // Column edges inside the ROI rows.
  const int rh = y1 - y0;
  std::vector<real> col(rh), p(rh), g(rh);
  std::vector<double> xs, ys, ws;
  for (int x = 0; x < rw; ++x) {
    for (int y = 0; y < rh; ++y) col[y] = med[(y0 + y) * rw + x];
    box(col.data(), rh, cfg_.smooth, p.data());
    edge_strength(p.data(), rh, cfg_.edge_half, g.data());
    int pick = topmost_brow_edge(p.data(), g.data(), rh);
    if (pick >= 0) {
      xs.push_back(x0 + x);
      ys.push_back(y0 + parabola(g.data(), rh, pick));
      ws.push_back(g[pick]);
    }
  }
  if (xs.size() < 8) return false;
  // Robust quadratic fit: 2 passes, drop points far from the curve.
  const size_t m = xs.size();
  std::vector<char> keep(m, 1);
  double c[3] = {0, 0, 0};
  for (int it = 0; it < 2; ++it) {
    std::vector<double> kx, ky, kw;
    for (size_t i = 0; i < m; ++i)
      if (keep[i]) {
        kx.push_back(xs[i]);
        ky.push_back(ys[i]);
        kw.push_back(ws[i]);
      }
    if (!polyfit2(kx, ky, kw, c)) return false;
    std::vector<double> res(m), kres;
    for (size_t i = 0; i < m; ++i) {
      res[i] = fabs(polyval2(c, xs[i]) - ys[i]);
      if (keep[i]) kres.push_back(res[i]);
    }
    double thr = std::max(2.0, 2.5 * median_of(kres));
    for (size_t i = 0; i < m; ++i) keep[i] = res[i] < thr;
  }
  for (int i = 0; i < 3; ++i) curve[i] = c[i];
  double cx0 = 1e18, cx1 = -1e18;
  for (size_t i = 0; i < m; ++i)
    if (keep[i]) {
      cx0 = std::min(cx0, xs[i]);
      cx1 = std::max(cx1, xs[i]);
    }
  if (cx0 > cx1) return false;
  cx1 += 1;
  const int S = cfg_.strips;
  for (int s = 0; s <= S; ++s) {
    int e = rint_even(cx0 + (cx1 - cx0) * s / S);  // numpy linspace(...).round()
    if (s < S) strip_x0[s] = e;
    if (s > 0) strip_x1[s - 1] = e;
  }
  // Neutral templates from the median image (column index relative to the ROI).
  const int K = cfg_.half_range, hn = cfg_.ncc_half;
  real prof[kMaxProfile];
  for (int s = 0; s < S; ++s) {
    profile(med.data(), rw, x0, h, s, prof);
    // standardise the template window around the neutral edge
    const int len = 2 * hn + 1;
    real mean = 0;
    for (int i = 0; i < len; ++i) mean += prof[K - hn + i];
    mean /= len;
    real var = 0;
    for (int i = 0; i < len; ++i) {
      real d = prof[K - hn + i] - mean;
      var += d * d;
    }
    real sd = sqrt(var / len) + real(1e-6);
    for (int i = 0; i < len; ++i) tz_[s][i] = (prof[K - hn + i] - mean) / sd;
  }
  for (int s = 0; s < kMaxStrips; ++s) prev_[s] = 0;
  lost_ = 0;
  calibrated_ = true;
  return true;
}

Result Tracker::update(const uint8_t* img, int w, int h) {
  Result r = {};
  const int S = cfg_.strips, K = cfg_.half_range, hn = cfg_.ncc_half, srch = cfg_.search;
  const int n = 2 * K + 1;
  const int len = 2 * hn + 1;
  real raw[kMaxProfile], p[kMaxProfile], g[kMaxProfile];
  for (int s = 0; s < S; ++s) {
    profile(img, w, 0, h, s, raw);
    box(raw, n, cfg_.smooth, p);
    const int margin = std::max(cfg_.edge_half, hn);
    int c = std::min(std::max(rint_even(prev_[s]) + K, margin), 2 * K - margin);
    int lo = std::max(margin, c - srch);
    int hi = std::min(2 * K + 1 - margin, c + srch + 1);
    for (int k = 0; k < n; ++k) g[k] = kNegInf;
    real best = -2;
    real best_pen = kNegInf;
    int bi = lo;
    for (int cc = lo; cc < hi; ++cc) {
      const real* wv = p + cc - hn;
      real mean = 0;
      for (int i = 0; i < len; ++i) mean += wv[i];
      mean /= len;
      real var = 0;
      for (int i = 0; i < len; ++i) {
        real d = wv[i] - mean;
        var += d * d;
      }
      real sd = sqrt(var / len) + real(1e-6);
      real acc = 0;
      for (int i = 0; i < len; ++i) acc += tz_[s][i] * ((wv[i] - mean) / sd);
      real val = acc / len;
      g[cc] = val;
      // optional prior: prefer matches near the previous position
      real vp = cfg_.dist_penalty != 0 ? val - cfg_.dist_penalty * real(std::abs(cc - c)) / srch : val;
      if (vp > best_pen) {
        best_pen = vp;
        best = val;
        bi = cc;
      }
    }
    real pos = parabola(g, n, bi) - K;
    r.score[s] = best;
    r.ok[s] = best > cfg_.min_ncc;
    r.off[s] = r.ok[s] ? pos : prev_[s];
  }
  // Strips belong to one brow: a strip far from the others jumped to another edge.
  int nok = 0;
  for (int s = 0; s < S; ++s) nok += r.ok[s];
  if (nok >= 3) {
    std::vector<double> v;
    for (int s = 0; s < S; ++s)
      if (r.ok[s]) v.push_back(r.off[s]);
    real med = static_cast<real>(median_of(v));
    real lim = std::max(real(3), real(0.5) * K);
    for (int s = 0; s < S; ++s)
      if (fabs(r.off[s] - med) > lim) {
        r.ok[s] = false;
        r.off[s] = med;
      }
  }
  nok = 0;
  for (int s = 0; s < S; ++s) nok += r.ok[s];
  lost_ = (nok * 2 < S) ? lost_ + 1 : 0;  // ok.mean() < 0.5
  if (cfg_.lost_frames && lost_ >= cfg_.lost_frames) {
    for (int s = 0; s < S; ++s) {
      profile(img, w, 0, h, s, raw);
      box(raw, n, cfg_.smooth, p);
      edge_strength(p, n, cfg_.edge_half, g);
      int i = topmost_brow_edge(p, g, n);
      if (i >= 0) {
        r.off[s] = parabola(g, n, i) - K;
        r.ok[s] = true;
      }
    }
    lost_ = 0;
  }
  for (int s = 0; s < S; ++s) prev_[s] = r.off[s];
  // Outputs (positive = up).
  real wsum = 0, acc = 0;
  nok = 0;
  for (int s = 0; s < S; ++s) {
    real wgt = (r.ok[s] ? real(1) : real(0)) + real(1e-3);
    wsum += wgt;
    acc += r.off[s] * wgt;
    nok += r.ok[s];
  }
  r.raise = -acc / wsum;
  const int ia = inner_is_right_ ? std::max(S - 2, 0) : 0, ib = inner_is_right_ ? S : std::min(2, S);
  const int oa = inner_is_right_ ? 0 : std::max(S - 2, 0), ob = inner_is_right_ ? std::min(2, S) : S;
  real si = 0, so = 0;
  for (int s = ia; s < ib; ++s) si += r.off[s];
  for (int s = oa; s < ob; ++s) so += r.off[s];
  r.inner = -si / (ib - ia);
  r.outer = -so / (ob - oa);
  r.valid = nok * 2 >= S;
  return r;
}

template void Tracker::profile<uint8_t>(const uint8_t*, int, int, int, int, real*) const;
template void Tracker::profile<real>(const real*, int, int, int, int, real*) const;

}  // namespace brow
