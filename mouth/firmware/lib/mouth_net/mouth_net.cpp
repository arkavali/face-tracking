#include "mouth_net.h"

#include <math.h>
#include <string.h>

#include "mouth_weights.h"

namespace mouth {

namespace {

// Weights in RAM (init()): the flash cache thrashes on them otherwise. nullptr = use flash.
const float* gW[MOUTH_N_LAYERS] = {nullptr};

// One conv3x3 output pixel (all output channels) into acc[cout], ReLU applied by the caller.
inline void convPixel(const float* in, int H, int W, int cin, int cout, const float* w, const float* b, int y,
                      int x, float* acc) {
  for (int o = 0; o < cout; ++o) acc[o] = b[o];
  for (int ky = 0; ky < 3; ++ky) {
    const int yy = y + ky - 1;
    if (yy < 0 || yy >= H) continue;
    for (int kx = 0; kx < 3; ++kx) {
      const int xx = x + kx - 1;
      if (xx < 0 || xx >= W) continue;
      const float* px = in + (yy * W + xx) * cin;
      const float* wk = w + (ky * 3 + kx) * cin;
      const int stride = 9 * cin;
      if (cin == 1) {
        const float v = px[0];
        for (int o = 0; o < cout; ++o) acc[o] += v * wk[o * stride];
        continue;
      }
      for (int o = 0; o < cout; ++o) {
        const float* wo = wk + o * stride;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 4 <= cin; i += 4) {
          s0 += px[i] * wo[i];
          s1 += px[i + 1] * wo[i + 1];
          s2 += px[i + 2] * wo[i + 2];
          s3 += px[i + 3] * wo[i + 3];
        }
        for (; i < cin; ++i) s0 += px[i] * wo[i];
        acc[o] += (s0 + s1) + (s2 + s3);
      }
    }
  }
}

// Conv + ReLU, optionally fused with a 2x2 max pool (floor, as nn.MaxPool2d(2)).
void convLayer(const float* in, int H, int W, const MouthLayerDef& L0, const float* w, float* out, int* outH,
               int* outW) {
  MouthLayerDef L = L0;
  L.w = w;
  float acc[128];
  if (!L.pool) {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        convPixel(in, H, W, L.cin, L.cout, L.w, L.b, y, x, acc);
        float* o = out + (y * W + x) * L.cout;
        for (int c = 0; c < L.cout; ++c) o[c] = acc[c] > 0 ? acc[c] : 0;
      }
    *outH = H;
    *outW = W;
    return;
  }
  const int PH = H / 2, PW = W / 2;
  for (int py = 0; py < PH; ++py)
    for (int px = 0; px < PW; ++px) {
      float* o = out + (py * PW + px) * L.cout;
      for (int c = 0; c < L.cout; ++c) o[c] = 0;  // ReLU output >= 0, so 0 is the max identity
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
          convPixel(in, H, W, L.cin, L.cout, L.w, L.b, py * 2 + dy, px * 2 + dx, acc);
          for (int c = 0; c < L.cout; ++c)
            if (acc[c] > o[c]) o[c] = acc[c];
        }
    }
  *outH = PH;
  *outW = PW;
}

}  // namespace

bool init(void* (*alloc)(size_t)) {
  bool all = true;
  for (int k = 0; k < MOUTH_N_LAYERS; ++k) {
    const MouthLayerDef& L = kMouthLayers[k];
    const size_t n = (size_t)L.cout * 9 * L.cin;
    float* p = alloc ? (float*)alloc(n * sizeof(float)) : nullptr;
    if (!p) {
      all = false;
      continue;
    }
    memcpy(p, L.w, n * sizeof(float));
    gW[k] = p;
  }
  return all;
}

int inputWidth() { return MOUTH_IN_W; }
int inputHeight() { return MOUTH_IN_H; }
int outputCount() { return MOUTH_N_OUT; }
const char* channelName(int i) { return kMouthChannels[i]; }

void scratchSizes(size_t* a, size_t* b) {
  // walk the layers: even layers write A, odd layers write B; the input sits in B
  size_t sz[2] = {0, (size_t)MOUTH_IN_W * MOUTH_IN_H};
  int H = MOUTH_IN_H, W = MOUTH_IN_W;
  for (int k = 0; k < MOUTH_N_LAYERS; ++k) {
    const MouthLayerDef& L = kMouthLayers[k];
    if (L.pool) H /= 2, W /= 2;
    size_t n = (size_t)H * W * L.cout;
    if (n > sz[k % 2]) sz[k % 2] = n;
  }
  *a = sz[0];
  *b = sz[1];
}

void run(const uint8_t* img, float* bufA, float* bufB, float* out) {
  const int n = MOUTH_IN_W * MOUTH_IN_H;
  for (int i = 0; i < n; ++i) bufB[i] = img[i] * (1.0f / 255.0f);
  float* buf[2] = {bufA, bufB};
  const float* cur = bufB;
  int H = MOUTH_IN_H, W = MOUTH_IN_W, C = 1;
  for (int k = 0; k < MOUTH_N_LAYERS; ++k) {
    float* dst = buf[k % 2];
    convLayer(cur, H, W, kMouthLayers[k], gW[k] ? gW[k] : kMouthLayers[k].w, dst, &H, &W);
    C = kMouthLayers[k].cout;
    cur = dst;
  }
  float feat[128];
  for (int c = 0; c < C; ++c) feat[c] = 0;
  for (int p = 0; p < H * W; ++p)
    for (int c = 0; c < C; ++c) feat[c] += cur[p * C + c];
  for (int c = 0; c < C; ++c) feat[c] /= (float)(H * W);
  for (int o = 0; o < MOUTH_N_OUT; ++o) {
    float s = kHeadB[o];
    for (int c = 0; c < C; ++c) s += kHeadW[o * C + c] * feat[c];
    out[o] = 1.0f / (1.0f + expf(-s));
  }
}

void downscale4(const uint8_t* src, int srcW, int srcH, uint8_t* dst) {
  const int W = srcW / 4, H = srcH / 4;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      unsigned s = 0;
      for (int dy = 0; dy < 4; ++dy) {
        const uint8_t* r = src + (y * 4 + dy) * srcW + x * 4;
        s += r[0] + r[1] + r[2] + r[3];
      }
      // round half to even, as OpenCV's saturate_cast in INTER_AREA
      unsigned q = s >> 4, rem = s & 15;
      if (rem > 8 || (rem == 8 && (q & 1))) ++q;
      dst[y * W + x] = (uint8_t)q;
    }
}

}  // namespace mouth

extern "C" {
// Host build (tools/parity_test.py)
#if defined(MOUTH_HOST)
#ifdef _WIN32
#define MOUTH_API __declspec(dllexport)
#else
#define MOUTH_API
#endif
MOUTH_API int mouth_scratch(void) {
  size_t a, b;
  mouth::scratchSizes(&a, &b);
  return (int)(a > b ? a : b);
}
MOUTH_API void mouth_run(const uint8_t* img, float* a, float* b, float* out) { mouth::run(img, a, b, out); }
MOUTH_API void mouth_down4(const uint8_t* src, int w, int h, uint8_t* dst) { mouth::downscale4(src, w, h, dst); }
#endif
}
