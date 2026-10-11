"""CUDA ray-triangle kernel (CuPy RawKernel): one thread per ray, loop over all triangles.

  t, tri, u, v = cast(v0, e1, e2, origins, dirs, tmax=None)
v0/e1/e2: (T, 3) float32 cupy arrays; origins (R, 3) or (1, 3); dirs (R, 3). Nearest hit with
t > 1e-3 (and < tmax per ray when given, for shadow rays); tri = -1 on a miss. Barycentric u, v
give the smooth-normal interpolation. ~100x faster than the broadcast NumPy-style version.
"""
from __future__ import annotations

from gpu import cp

_SRC = r"""
extern "C" __global__
void cast(const float* v0, const float* e1, const float* e2, int nt,
          const float* org, int org_stride, const float* dir, const float* tmax, int nr,
          float* out_t, int* out_i, float* out_u, float* out_v) {
  int r = blockDim.x * blockIdx.x + threadIdx.x;
  if (r >= nr) return;
  const float ox = org[r * org_stride], oy = org[r * org_stride + 1], oz = org[r * org_stride + 2];
  const float dx = dir[r * 3], dy = dir[r * 3 + 1], dz = dir[r * 3 + 2];
  float best = tmax ? tmax[r] : 3.0e38f;
  int bi = -1; float bu = 0, bv = 0;
  for (int k = 0; k < nt; ++k) {
    const float ax = e1[k*3], ay = e1[k*3+1], az = e1[k*3+2];
    const float bx = e2[k*3], by = e2[k*3+1], bz = e2[k*3+2];
    const float px = dy * bz - dz * by, py = dz * bx - dx * bz, pz = dx * by - dy * bx;
    const float det = ax * px + ay * py + az * pz;
    if (fabsf(det) < 1e-9f) continue;
    const float inv = 1.0f / det;
    const float sx = ox - v0[k*3], sy = oy - v0[k*3+1], sz = oz - v0[k*3+2];
    const float u = (sx * px + sy * py + sz * pz) * inv;
    if (u < 0.f || u > 1.f) continue;
    const float qx = sy * az - sz * ay, qy = sz * ax - sx * az, qz = sx * ay - sy * ax;
    const float v = (dx * qx + dy * qy + dz * qz) * inv;
    if (v < 0.f || u + v > 1.f) continue;
    const float t = (bx * qx + by * qy + bz * qz) * inv;
    if (t > 1e-3f && t < best) { best = t; bi = k; bu = u; bv = v; }
  }
  out_t[r] = bi >= 0 ? best : 3.0e38f;
  out_i[r] = bi; out_u[r] = bu; out_v[r] = bv;
}
"""
_kernel = cp.RawKernel(_SRC, "cast") if cp is not None else None


def cast(v0, e1, e2, o, d, tmax=None):
    v0, e1, e2 = (cp.ascontiguousarray(a, dtype=cp.float32) for a in (v0, e1, e2))   # views break the kernel
    o = cp.ascontiguousarray(cp.asarray(o, dtype=cp.float32).reshape(-1, 3))
    d = cp.ascontiguousarray(cp.asarray(d, dtype=cp.float32).reshape(-1, 3))
    nr = d.shape[0]
    stride = 0 if o.shape[0] == 1 else 3
    t = cp.empty(nr, cp.float32)
    i = cp.empty(nr, cp.int32)
    u = cp.empty(nr, cp.float32)
    v = cp.empty(nr, cp.float32)
    tm = None if tmax is None else cp.ascontiguousarray(cp.asarray(tmax, dtype=cp.float32))
    block = 128
    _kernel(((nr + block - 1) // block,), (block,),
            (v0, e1, e2, cp.int32(v0.shape[0]), o, cp.int32(stride), d,
             tm if tm is not None else cp.uint64(0), cp.int32(nr), t, i, u, v))
    t = cp.where(i >= 0, t, cp.inf)
    return t, i, u, v
