#include "eye_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace eye {

namespace {

constexpr double kPi = 3.14159265358979323846;

template <typename T>
inline T clampv(T v, T lo, T hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// In-place insertion sort (small arrays only).
void sortSmall(real* v, int n) {
  for (int i = 1; i < n; ++i) {
    real x = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > x) {
      v[j + 1] = v[j];
      --j;
    }
    v[j + 1] = x;
  }
}

// numpy.median semantics.
real median(const real* v, int n, real* scratch) {
  memcpy(scratch, v, sizeof(real) * n);
  sortSmall(scratch, n);
  return (n & 1) ? scratch[n / 2] : (scratch[n / 2 - 1] + scratch[n / 2]) / 2;
}

// 3x3 solve by Gaussian elimination with partial pivoting; false if singular.
bool solve3(const double A[3][3], const double* b, int nrhs, double* x) {
  double M[3][3 + 3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) M[i][j] = A[i][j];
    for (int k = 0; k < nrhs; ++k) M[i][3 + k] = b[i * nrhs + k];
  }
  for (int c = 0; c < 3; ++c) {
    int piv = c;
    for (int r = c + 1; r < 3; ++r)
      if (fabs(M[r][c]) > fabs(M[piv][c])) piv = r;
    if (fabs(M[piv][c]) < 1e-300) return false;
    if (piv != c)
      for (int j = 0; j < 3 + nrhs; ++j) {
        double t = M[c][j];
        M[c][j] = M[piv][j];
        M[piv][j] = t;
      }
    for (int r = 0; r < 3; ++r) {
      if (r == c) continue;
      double f = M[r][c] / M[c][c];
      for (int j = c; j < 3 + nrhs; ++j) M[r][j] -= f * M[c][j];
    }
  }
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < nrhs; ++k) x[i * nrhs + k] = M[i][3 + k] / M[i][i];
  return true;
}

// Real eigenvalues of a 3x3 matrix (characteristic cubic); returns how many.
int eig3Real(const double M[3][3], double* lam) {
  double a = -(M[0][0] + M[1][1] + M[2][2]);
  double b = M[0][0] * M[1][1] - M[0][1] * M[1][0] + M[0][0] * M[2][2] - M[0][2] * M[2][0] +
             M[1][1] * M[2][2] - M[1][2] * M[2][1];
  double c = -(M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
               M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
               M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]));
  // depressed cubic t^3 + p t + q, x = t - a/3
  double p = b - a * a / 3, q = 2 * a * a * a / 27 - a * b / 3 + c;
  double disc = q * q / 4 + p * p * p / 27;
  if (disc > 0) {
    double s = sqrt(disc);
    lam[0] = cbrt(-q / 2 + s) + cbrt(-q / 2 - s) - a / 3;
    return 1;
  }
  double r = sqrt(-p / 3);
  double arg = r > 0 ? clampv(-q / (2 * r * r * r), -1.0, 1.0) : 0.0;
  double phi = acos(arg);
  for (int k = 0; k < 3; ++k) lam[k] = 2 * r * cos((phi - 2 * kPi * k) / 3) - a / 3;
  return 3;
}

// Null vector of (M - lam I) from the largest cross product of its rows.
void nullVec3(const double M[3][3], double lam, double v[3]) {
  double R[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R[i][j] = M[i][j] - (i == j ? lam : 0);
  double best = -1;
  for (int i = 0; i < 3; ++i) {
    const double* u = R[i];
    const double* w = R[(i + 1) % 3];
    double c[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
    double n = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
    if (n > best) {
      best = n;
      v[0] = c[0];
      v[1] = c[1];
      v[2] = c[2];
    }
  }
}

}  // namespace

// ----------------------------------------------------------------------------- ellipse

bool fitEllipse(const real* xs, const real* ys, int n, real conic[6]) {
  if (n < 5) return false;
  double mx = 0, my = 0;
  for (int i = 0; i < n; ++i) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  double sx = 0, sy = 0;
  for (int i = 0; i < n; ++i) {
    sx = fmax(sx, fabs(xs[i] - mx));
    sy = fmax(sy, fabs(ys[i] - my));
  }
  sx = fmax(sx, 1e-6);
  sy = fmax(sy, 1e-6);
  double S1[3][3] = {}, S2[3][3] = {}, S3[3][3] = {};
  for (int i = 0; i < n; ++i) {
    double x = (xs[i] - mx) / sx, y = (ys[i] - my) / sy;
    double d1[3] = {x * x, x * y, y * y}, d2[3] = {x, y, 1};
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) {
        S1[r][c] += d1[r] * d1[c];
        S2[r][c] += d1[r] * d2[c];
        S3[r][c] += d2[r] * d2[c];
      }
  }
  // T = -S3^-1 S2^T
  double S2T[9];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) S2T[r * 3 + c] = S2[c][r];
  double T[9];
  if (!solve3(S3, S2T, 3, T)) return false;
  for (double& t : T) t = -t;
  double M0[3][3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += S2[r][k] * T[k * 3 + c];
      M0[r][c] = S1[r][c] + s;
    }
  double M[3][3];
  for (int c = 0; c < 3; ++c) {
    M[0][c] = M0[2][c] / 2;
    M[1][c] = -M0[1][c];
    M[2][c] = M0[0][c] / 2;
  }
  double lam[3];
  int nl = eig3Real(M, lam);
  double a1[3];
  bool found = false;
  for (int k = 0; k < nl && !found; ++k) {
    double v[3];
    nullVec3(M, lam[k], v);
    if (4 * v[0] * v[2] - v[1] * v[1] > 0) {
      a1[0] = v[0];
      a1[1] = v[1];
      a1[2] = v[2];
      found = true;
    }
  }
  if (!found) return false;
  double a2[3];
  for (int r = 0; r < 3; ++r) a2[r] = T[r * 3] * a1[0] + T[r * 3 + 1] * a1[1] + T[r * 3 + 2] * a1[2];
  double A = a1[0], B = a1[1], C = a1[2], D = a2[0], E = a2[1], F = a2[2];
  double A2 = A / (sx * sx), B2 = B / (sx * sy), C2 = C / (sy * sy);
  double D2 = D / sx, E2 = E / sy;
  conic[0] = real(A2);
  conic[1] = real(B2);
  conic[2] = real(C2);
  conic[3] = real(-2 * A2 * mx - B2 * my + D2);
  conic[4] = real(-2 * C2 * my - B2 * mx + E2);
  conic[5] = real(A2 * mx * mx + B2 * mx * my + C2 * my * my - D2 * mx - E2 * my + F);
  return true;
}

bool conicToParams(const real c[6], real* cx, real* cy, real* a, real* b, real* ang) {
  double A = c[0], B = c[1], C = c[2], D = c[3], E = c[4], F = c[5];
  double m00 = A, m01 = B / 2, m11 = C;
  double det = 4 * (m00 * m11 - m01 * m01);  // det(2M)
  if (fabs(det) < 1e-300) return false;
  double x = (-D * 2 * m11 + E * 2 * m01) / det;
  double y = (-E * 2 * m00 + D * 2 * m01) / det;
  double k = A * x * x + B * x * y + C * y * y + D * x + E * y + F;
  // eigen of the symmetric 2x2 [[m00, m01], [m01, m11]], ascending like numpy.eigh
  double tr = m00 + m11, dd = sqrt((m00 - m11) * (m00 - m11) / 4 + m01 * m01);
  double l0 = tr / 2 - dd, l1 = tr / 2 + dd;
  double q0 = -k / l0, q1 = -k / l1;
  if (!(q0 > 0 && q1 > 0)) return false;
  double ax0 = sqrt(q0), ax1 = sqrt(q1);
  double lmaj = ax0 >= ax1 ? l0 : l1;
  // eigenvector of lmaj
  double vx, vy;
  if (fabs(m01) > 1e-300) {
    vx = lmaj - m11;
    vy = m01;
  } else {
    bool first = fabs(lmaj - m00) <= fabs(lmaj - m11);
    vx = first ? 1 : 0;
    vy = first ? 0 : 1;
  }
  *cx = real(x);
  *cy = real(y);
  *a = real(fmax(ax0, ax1));
  *b = real(fmin(ax0, ax1));
  *ang = real(atan2(vy, vx));
  return true;
}

namespace {

real ellipseDist(real px, real py, real cx, real cy, real a, real b, real ang) {
  real ca = cos(ang), sa = sin(ang);
  real dx = px - cx, dy = py - cy;
  real u = dx * ca + dy * sa, v = -dx * sa + dy * ca;
  real rr = sqrt((u / a) * (u / a) + (v / b) * (v / b));
  return fabs(rr - 1) * sqrt(u * u + v * v) / fmax(rr, real(1e-9));
}

}  // namespace

// ----------------------------------------------------------------------------- detector

PupilDetector::PupilDetector(real* work, int w, int h, const Params& p, uint32_t seed)
    : buf_(work), w_(w), h_(h), p_(p), rng_(2463534242u + seed) {
  tmp_ = static_cast<real*>(malloc(sizeof(real) * w * h));
}

void PupilDetector::blur(const uint8_t* img) {
  const int w = w_, h = h_;
  for (int i = 0; i < w * h; ++i) buf_[i] = img[i];
  for (int pass = 0; pass < 2; ++pass) {
    // vertical, edge padded (numpy: (up + centre + down) / 3)
    for (int y = 0; y < h; ++y) {
      const real* up = buf_ + (y > 0 ? y - 1 : 0) * w;
      const real* ce = buf_ + y * w;
      const real* dn = buf_ + (y < h - 1 ? y + 1 : h - 1) * w;
      real* o = tmp_ + y * w;
      for (int x = 0; x < w; ++x) o[x] = (up[x] + ce[x] + dn[x]) / 3;
    }
    for (int y = 0; y < h; ++y) {
      const real* r = tmp_ + y * w;
      real* o = buf_ + y * w;
      for (int x = 0; x < w; ++x) {
        real l = r[x > 0 ? x - 1 : 0], c = r[x], rr = r[x < w - 1 ? x + 1 : w - 1];
        o[x] = (l + c + rr) / 3;
      }
    }
  }
}

void PupilDetector::seed(real* sx, real* sy, real* level) {
  const int hw = w_ / 2, hh = h_ / 2;
  const int k = p_.box / 2 > 2 ? p_.box / 2 : 2;
  // integral image of the 2x-downsampled, clipped frame (in tmp_)
  real* ii = tmp_;  // (hh + 1) x (hw + 1)
  const int W1 = hw + 1;
  for (int x = 0; x <= hw; ++x) ii[x] = 0;
  for (int y = 0; y < hh; ++y) {
    ii[(y + 1) * W1] = 0;
    real rowsum = 0;
    for (int x = 0; x < hw; ++x) {
      const real* r0 = buf_ + (2 * y) * w_ + 2 * x;
      const real* r1 = r0 + w_;
      real s = (r0[0] + r1[0] + r0[1] + r1[1]) / 4;
      if (s > 120) s = 120;
      rowsum += s;
      ii[(y + 1) * W1 + x + 1] = ii[y * W1 + x + 1] + rowsum;
    }
  }
  const int bd = 3;
  const int mh = hh + 1 - k, mw = hw + 1 - k;
  real best = real(1e30);
  int bx = 0, by = 0;
  for (int y = bd; y < mh - bd; ++y)
    for (int x = bd; x < mw - bd; ++x) {
      real m = (ii[(y + k) * W1 + x + k] - ii[y * W1 + x + k] - ii[(y + k) * W1 + x] + ii[y * W1 + x]) / (k * k);
      if (m < best) {
        best = m;
        bx = x;
        by = y;
      }
    }
  *sx = real(2 * bx + k);
  *sy = real(2 * by + k);
  *level = best;
}

real PupilDetector::at(real x, real y) const {
  x = clampv(x, real(0), real(w_ - 1.001));
  y = clampv(y, real(0), real(h_ - 1.001));
  int x0 = int(x), y0 = int(y);
  real fx = x - x0, fy = y - y0;
  const real* r0 = buf_ + y0 * w_ + x0;
  const real* r1 = r0 + w_;
  real a = r0[0] * (1 - fx) + r0[1] * fx;
  real b = r1[0] * (1 - fx) + r1[1] * fx;
  return a * (1 - fy) + b * fy;
}

int PupilDetector::starburst(real sx, real sy, real level, real* px, real* py) {
  const int R = clampv(int(ceil((p_.r_max - p_.r_min) / p_.step)), 0, kMaxRaySamples);
  real v[kMaxRaySamples];
  bool ok[kMaxRaySamples];
  int n = 0;
  const real thr = level + p_.above_seed;
  for (int i = 0; i < p_.rays; ++i) {
    real th = real(2 * kPi * i / p_.rays);
    real ct = cos(th), st = sin(th);
    for (int j = 0; j < R; ++j) {
      real r = p_.r_min + j * p_.step;
      real x = sx + ct * r, y = sy + st * r;
      ok[j] = x >= 0 && x < w_ - 1 && y >= 0 && y < h_ - 1;
      v[j] = at(x, y);
    }
    int j = 3;
    while (j < R && ok[j]) {
      if (v[j] >= p_.glint_level) {
        int k = j;
        while (k < R && v[k] >= thr && k - j < 6) ++k;
        if (k < R && v[k] < thr) {
          j = k + 1;
          continue;
        }
      }
      if (v[j] - v[j - 2] > p_.grad && v[j] > thr) {
        real rr;
        if (j + 2 > R) {
          rr = p_.r_min + j * p_.step;
        } else {
          int best = 0;
          real bg = v[j - 1] - v[j - 3];
          for (int m = 1; m < 3; ++m) {
            real g = v[j - 1 + m] - v[j - 3 + m];
            if (g > bg) {
              bg = g;
              best = m;
            }
          }
          int idx = j - 1 + best - 1;
          rr = (idx >= 0 && idx < R) ? p_.r_min + idx * p_.step : p_.r_min + j * p_.step;
        }
        px[n] = sx + ct * rr;
        py[n] = sy + st * rr;
        ++n;
        break;
      }
      ++j;
    }
  }
  return n;
}

bool PupilDetector::detect(const uint8_t* img, Ellipse* out) {
  blur(img);
  real sx, sy, level;
  seed(&sx, &sy, &level);
  real px[kMaxRays], py[kMaxRays], scratch[kMaxRays];
  int n = 0;
  for (int round = 0; round < p_.rounds; ++round) {
    n = starburst(sx, sy, level, px, py);
    if (n < p_.min_points) return false;
    real nx = median(px, n, scratch), ny = median(py, n, scratch);
    bool done = sqrt((nx - sx) * (nx - sx) + (ny - sy) * (ny - sy)) < real(0.5);
    sx = nx;
    sy = ny;
    if (done) break;
  }
  // RANSAC
  bool have = false;
  real best[5] = {0, 0, 0, 0, 0};
  bool bestIn[kMaxRays];
  int bestCount = -1;
  for (int it = 0; it < p_.ransac_iter; ++it) {
    int idx[5], m = 0;
    while (m < 5) {
      int c = int(rng_.next() % uint32_t(n));
      bool dup = false;
      for (int q = 0; q < m; ++q) dup |= idx[q] == c;
      if (!dup) idx[m++] = c;
    }
    real xs[5], ys[5], conic[6], prm[5];
    for (int q = 0; q < 5; ++q) {
      xs[q] = px[idx[q]];
      ys[q] = py[idx[q]];
    }
    if (!fitEllipse(xs, ys, 5, conic)) continue;
    if (!conicToParams(conic, &prm[0], &prm[1], &prm[2], &prm[3], &prm[4])) continue;
    if (prm[3] < 1 || prm[2] / prm[3] > p_.max_axis_ratio) continue;
    int cnt = 0;
    bool in[kMaxRays];
    for (int q = 0; q < n; ++q) {
      in[q] = ellipseDist(px[q], py[q], prm[0], prm[1], prm[2], prm[3], prm[4]) < p_.inlier_px;
      cnt += in[q];
    }
    if (cnt > bestCount) {
      bestCount = cnt;
      memcpy(best, prm, sizeof(best));
      memcpy(bestIn, in, sizeof(bool) * n);
      have = true;
    }
  }
  if (!have || bestCount < p_.min_points * real(0.7)) return false;
  real ix[kMaxRays], iy[kMaxRays];
  int ni = 0;
  for (int q = 0; q < n; ++q)
    if (bestIn[q]) {
      ix[ni] = px[q];
      iy[ni] = py[q];
      ++ni;
    }
  real conic[6], prm[5];
  if (!(fitEllipse(ix, iy, ni, conic) && conicToParams(conic, &prm[0], &prm[1], &prm[2], &prm[3], &prm[4])))
    memcpy(prm, best, sizeof(prm));
  real cx = prm[0], cy = prm[1], a = prm[2], b = prm[3], ang = prm[4];
  if (a > p_.r_max * real(1.2) || ni < p_.min_support * p_.rays) return false;
  // dark inside
  {
    int y0 = int(cy - b), y1 = int(cy + b), x0 = int(cx - b), x1 = int(cx + b);
    if (y0 < 0) y0 = 0;
    if (x0 < 0) x0 = 0;
    double s = 0;
    int c = 0;
    for (int y = y0; y <= y1 && y < h_; ++y)
      for (int x = x0; x <= x1 && x < w_; ++x) {
        s += buf_[y * w_ + x];
        ++c;
      }
    if (c == 0 || real(s / c) - level > p_.max_inside) return false;
  }
  // ring contrast
  real ca = cos(ang), sa = sin(ang);
  real ring[16], rs[16];
  for (int t = 0; t < 16; ++t) {
    real th = real(2 * kPi * t / 16);
    real u = real(1.6) * a * cos(th), v = real(1.6) * b * sin(th);
    ring[t] = at(cx + u * ca - v * sa, cy + u * sa + v * ca);
  }
  real contrast = median(ring, 16, rs) - level;
  // upper lid over the pupil
  real tv[kMaxRays];
  int nt = 0;
  for (int q = 0; q < n; ++q) {
    real dx = px[q] - cx, dy = py[q] - cy;
    real up = real(atan2(-dy, dx) * 180 / kPi);
    if (fabs(up - 90) < 35) {
      real u = dx * ca + dy * sa, v = -dx * sa + dy * ca;
      tv[nt++] = sqrt((u / a) * (u / a) + (v / b) * (v / b));
    }
  }
  real top = 1;
  if (nt >= 2) top = clampv(median(tv, nt, scratch), real(0), real(1));
  out->cx = cx;
  out->cy = cy;
  out->a = a;
  out->b = b;
  out->angle = ang;
  out->support = real(ni) / p_.rays;
  out->n = ni;
  out->contrast = contrast;
  out->top_vis = top;
  return true;
}

// ----------------------------------------------------------------------------- gate

bool PupilGate::operator()(const Ellipse* e) {
  if (!e || e->support < min_support_) return false;
  real ref = 0;
  if (n_ >= 10) {
    real s[kGateHist];
    ref = median(hist_, n_, s);
  }
  if (e->contrast < share_ * ref) return false;
  if (n_ == kGateHist) {
    memmove(hist_, hist_ + 1, sizeof(real) * (kGateHist - 1));
    --n_;
  }
  hist_[n_++] = e->contrast;
  return true;
}

// ----------------------------------------------------------------------------- glints

int findGlints(const uint8_t* img, int w, int h, const Ellipse& e, Glint* out, int level, real reach) {
  const int r = int(reach * e.a) + 2;
  int x0 = int(e.cx) - r, x1 = int(e.cx) + r + 1, y0 = int(e.cy) - r, y1 = int(e.cy) + r + 1;
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > w) x1 = w;
  if (y1 > h) y1 = h;
  const int bw = x1 - x0, bh = y1 - y0;
  if (bw <= 0 || bh <= 0) return 0;
  static uint8_t seen[256 * 256];  // box up to 256 x 256 px (reach * a <= ~126 px)
  if (bw * bh > int(sizeof(seen))) return 0;
  memset(seen, 0, bw * bh);
  constexpr int kMaxFound = 32;
  Glint found[kMaxFound];
  real dist[kMaxFound];
  int nf = 0;
  int stack[256];
  for (int yy = 0; yy < bh; ++yy)
    for (int xx = 0; xx < bw; ++xx) {
      if (seen[yy * bw + xx] || img[(yy + y0) * w + xx + x0] < level) continue;
      int sp = 0, cnt = 0;
      double sxs = 0, sys = 0;
      stack[sp++] = yy * bw + xx;
      seen[yy * bw + xx] = 1;
      while (sp) {
        int id = stack[--sp];
        int a = id / bw, b = id % bw;
        ++cnt;
        sxs += b;
        sys += a;
        const int nb[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (auto& d : nb) {
          int c = a + d[0], dd = b + d[1];
          if (c >= 0 && c < bh && dd >= 0 && dd < bw && !seen[c * bw + dd] && img[(c + y0) * w + dd + x0] >= level) {
            seen[c * bw + dd] = 1;
            if (sp < 256) stack[sp++] = c * bw + dd;
          }
        }
      }
      if (cnt <= 40 && nf < kMaxFound) {
        Glint g{real(sxs / cnt + x0), real(sys / cnt + y0), cnt};
        found[nf] = g;
        dist[nf] = sqrt((g.x - e.cx) * (g.x - e.cx) + (g.y - e.cy) * (g.y - e.cy));
        ++nf;
      }
    }
  // nearest first (stable, like Python's sort)
  for (int i = 1; i < nf; ++i) {
    Glint g = found[i];
    real d = dist[i];
    int j = i - 1;
    while (j >= 0 && dist[j] > d) {
      found[j + 1] = found[j];
      dist[j + 1] = dist[j];
      --j;
    }
    found[j + 1] = g;
    dist[j + 1] = d;
  }
  int m = nf < kMaxGlints ? nf : kMaxGlints;
  for (int i = 0; i < m; ++i) out[i] = found[i];
  return m;
}

// ----------------------------------------------------------------------------- eye model

EyeModel::EyeModel(real f, int w, int h, real half_life_s, real fps, real min_ecc)
    : f_(f), w_(w), h_(h), decay_(real(pow(0.5, 1.0 / (half_life_s * fps)))), min_ecc_(min_ecc) {}

void EyeModel::update(const Ellipse& e) {
  real ecc = 1 - e.b / e.a;
  for (real& v : A_) v *= decay_;
  bv_[0] *= decay_;
  bv_[1] *= decay_;
  wsum_ *= decay_;
  if (ecc > min_ecc_) {
    real mx = cos(e.angle), my = sin(e.angle);
    real n00 = mx * mx * ecc, n01 = mx * my * ecc, n11 = my * my * ecc;
    A_[0] += n00;
    A_[1] += n01;
    A_[2] += n11;
    bv_[0] += n00 * e.cx + n01 * e.cy;
    bv_[1] += n01 * e.cx + n11 * e.cy;
    wsum_ += ecc;
  }
  if (wsum_ > 3) {
    // 2-norm condition number of the symmetric 2x2
    real tr = A_[0] + A_[2], dd = sqrt((A_[0] - A_[2]) * (A_[0] - A_[2]) / 4 + A_[1] * A_[1]);
    real l0 = fabs(tr / 2 - dd), l1 = fabs(tr / 2 + dd);
    real lo = l0 < l1 ? l0 : l1, hi = l0 < l1 ? l1 : l0;
    if (lo > 0 && hi / lo < 1000) {
      real det = A_[0] * A_[2] - A_[1] * A_[1];
      c_[0] = (A_[2] * bv_[0] - A_[1] * bv_[1]) / det;
      c_[1] = (A_[0] * bv_[1] - A_[1] * bv_[0]) / det;
      has_c_ = true;
    }
  }
  if (has_c_) {
    real s2 = 1 - (e.b / e.a) * (e.b / e.a);
    if (s2 < 0) s2 = 0;
    r_num_ *= decay_;
    r_den_ *= decay_;
    if (s2 > real(0.25)) {
      real d2 = (e.cx - c_[0]) * (e.cx - c_[0]) + (e.cy - c_[1]) * (e.cy - c_[1]);
      r_num_ += d2 * s2;
      r_den_ += s2 * s2;
      ++r_count_;
    }
    if (r_count_ >= 20 && r_den_ > 0) {
      R_ = sqrt(r_num_ / r_den_);
      has_r_ = true;
    }
  }
}

bool EyeModel::gazeCam(const Ellipse& e, real g[3]) const {
  if (!ready()) return false;
  real vx = (e.cx - c_[0]) / R_, vy = (e.cy - c_[1]) / R_;
  real n = sqrt(vx * vx + vy * vy);
  if (n > real(0.98)) {
    vx *= real(0.98) / n;
    vy *= real(0.98) / n;
  }
  real l[3] = {vx, vy, -sqrt(1 - vx * vx - vy * vy)};
  real d[3] = {(c_[0] - w_ / real(2) + real(0.5)) / f_, (c_[1] - h_ / real(2) + real(0.5)) / f_, 1};
  real dn = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  for (real& x : d) x /= dn;
  // rotation taking +z to d (Rodrigues with v = z x d)
  real v[3] = {-d[1], d[0], 0};
  real s = sqrt(v[0] * v[0] + v[1] * v[1]), c = d[2];
  if (s < real(1e-12)) {
    g[0] = l[0];
    g[1] = l[1];
    g[2] = l[2];
    return true;
  }
  real K[3][3] = {{0, -v[2], v[1]}, {v[2], 0, -v[0]}, {-v[1], v[0], 0}};
  real KK[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) KK[i][j] = K[i][0] * K[0][j] + K[i][1] * K[1][j] + K[i][2] * K[2][j];
  real f = (1 - c) / (s * s);
  for (int i = 0; i < 3; ++i) {
    real r = 0;
    for (int j = 0; j < 3; ++j) r += ((i == j ? 1 : 0) + K[i][j] + KK[i][j] * f) * l[j];
    g[i] = r;
  }
  return true;
}

}  // namespace eye
