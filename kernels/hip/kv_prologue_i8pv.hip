// K/V quantisation for the attn_i8pv_* kernels. Two launches per call:
//
//   kv_prologue_i8pv_stats1_{f16,bf16} grid (B*H), block 256: the K mean (from every `step`-th row; the caller picks
//                                      ~1024 rows), the V mean and the per-channel V scale s_d = max|v - vmean| / 127
//                                      (from every row); sets the call flag (head 0's SV[128 + 2 nt]) to 1; and
//                                      the head's rotation flag SV[129 + 2 nt]: 1 when K's largest channel range (over
//                                      the same sampled rows) is under 16x the median one. Heads with one dominant K
//                                      channel lose accuracy to the rotation (Q's matching channel is large too).
//                                      K is quantised after a 128-point Hadamard rotation on flagged heads.
//   kv_prologue_i8pv_quant_{f16,bf16}  grid (N_pad/64, B*H), block 256: one 64-key tile per block.
//       K: k - kmean, per-token amax. A 16-key group's scale is the group max; the tile takes group scales if every
//          live key's amax is >= TAU * its group max (a group scale then costs any key at most log2(1/TAU) bits),
//          else per-token scales. SK holds each key's scale either way; the tile flag goes to SV[bh, 128 + tile]
//          and a per-token tile clears the call flag. K8 = rne(k / scale), clamped to int8.
//       V: (v - vmean) / s_d, the tile's amax f_j = max / 127 (floored at 1e-4) to SV[bh, 128 + nt + tile],
//          V8T = rint(... / f_j) clamped to [-127, 127], stored transposed (BH, 128, N_pad).
//
// K and V may be any view with unit stride along the head dim (strided HND, NHD, fused QKV); padding rows get
// K8 = 0, SK = 1, V8T = 0. SV must hold N_pad >= 130 + 2 nt floats per head (true for N >= 129).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef unsigned short u16;
typedef unsigned char  u8;
typedef unsigned long long u64;
typedef _Float16       f16;

#define BN   64        // keys per quant block
#define NT   256       // quant threads per block: 64 rows x 4 lanes
#define HD   128
#define VROW 132       // LDS row stride of the byte tile that is transposed for V8T

// round half to even, then clamp to [-128, 127]
__device__ __forceinline__ u8 i8_rne(float x) {
  float fl = floorf(x);
  float d  = x - fl;
  int   odd = ((int)fl) & 1;
  float q = ((d > 0.5f) || (d == 0.5f && odd != 0)) ? (fl + 1.0f) : fl;
  q = fminf(fmaxf(q, -128.0f), 127.0f);
  return (u8)(signed char)q;
}

template <typename T> __device__ __forceinline__ T frombits(u16 b) {
  union { u16 u; T f; } x; x.u = b; return x.f;
}

// 8 elements = one 16-byte load
template <typename T>
__device__ __forceinline__ void load8(const T* p, float* out) {
  uint4 u = *(const uint4*)p;
  u16 hh[8];
  *(uint4*)hh = u;
  #pragma unroll
  for (int t = 0; t < 8; t++) out[t] = (float)frombits<T>(hh[t]);
}

__device__ __forceinline__ void nsum(float& s, float& c, float x) {
  const float t = s + x;
  c += (fabsf(s) >= fabsf(x)) ? ((s - t) + x) : ((x - t) + s);
  s = t;
}

// One block per (b, h): per-channel K mean, V mean and V max / min, the fp32 partial sums of the 16 row lanes
// reduced in fp64 in a fixed order (deterministic). V is read in full; K only every `step`-th row.
template <typename T>
__device__ __forceinline__ void stats1_body(const T* __restrict__ K, const T* __restrict__ V, float* __restrict__ KMEAN,
                                            float* __restrict__ VMEAN, float* __restrict__ SV, long long N, long long N_pad,
                                            long long H, long long step, long long k_sb, long long k_sh, long long k_sr,
                                            long long v_sb, long long v_sh, long long v_sr) {
  __shared__ double rk[16][HD], rv[16][HD];
  __shared__ float rmx[16][HD], rmn[16][HD];
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int bh = blockIdx.x;
  const int b = (int)(bh / H);
  const int h = (int)(bh - (long long)b * H);
  const long long kb = (long long)b * k_sb + (long long)h * k_sh + 8 * tx;
  const long long vb_ = (long long)b * v_sb + (long long)h * v_sh + 8 * tx;
  float ks[8], kc[8], vs[8], vc[8], mx[8], mn[8], kmx[8], kmn[8];
  #pragma unroll
  for (int t = 0; t < 8; t++) {
    ks[t] = kc[t] = vs[t] = vc[t] = 0.f;
    mx[t] = kmx[t] = -INFINITY;
    mn[t] = kmn[t] = INFINITY;
  }
  // V: every row (its channel max sets the V scale, and a missed outlier would cost its tile 5 bits).
  // K: every `step`-th row; any K mean is exact for attention, so a sample only changes the K ranges.
  for (long long n = ty; n < N; n += 16) {
    float vx[8];
    load8<T>(V + vb_ + n * v_sr, vx);
    #pragma unroll
    for (int t = 0; t < 8; t++) {
      nsum(vs[t], vc[t], vx[t]);
      mx[t] = fmaxf(mx[t], vx[t]);
      mn[t] = fminf(mn[t], vx[t]);
    }
  }
  for (long long n = ty * step; n < N; n += 16 * step) {
    float kx[8];
    load8<T>(K + kb + n * k_sr, kx);
    #pragma unroll
    for (int t = 0; t < 8; t++) {
      nsum(ks[t], kc[t], kx[t]);
      kmx[t] = fmaxf(kmx[t], kx[t]);
      kmn[t] = fminf(kmn[t], kx[t]);
    }
  }
  #pragma unroll
  for (int t = 0; t < 8; t++) {
    rk[ty][8 * tx + t] = (double)ks[t] + (double)kc[t];
    rv[ty][8 * tx + t] = (double)vs[t] + (double)vc[t];
    rmx[ty][8 * tx + t] = mx[t];
    rmn[ty][8 * tx + t] = mn[t];
  }
  __syncthreads();
  if (threadIdx.x < HD) {
    const int d = threadIdx.x;
    double sk_ = 0.0, sv_ = 0.0;
    float m1 = -INFINITY, m0 = INFINITY;
    for (int i = 0; i < 16; i++) { sk_ += rk[i][d]; sv_ += rv[i][d]; m1 = fmaxf(m1, rmx[i][d]); m0 = fminf(m0, rmn[i][d]); }
    KMEAN[(long long)bh * HD + d] = (float)(sk_ / (double)((N - 1) / step + 1));
    const float mu = (float)(sv_ / (double)N);
    VMEAN[(long long)bh * HD + d] = mu;
    if (d == 0 && bh == 0) SV[HD + 2 * (N_pad / BN)] = 1.0f;
    SV[(long long)bh * N_pad + d] = __fdiv_rn(fmaxf(fmaxf(m1 - mu, mu - m0), 1e-8f), 127.0f);
  }
  // the rotation flag: K's largest channel range against the median one (the lower of the middle two)
  __shared__ float krg[HD], kmed;
  __syncthreads();
  #pragma unroll
  for (int t = 0; t < 8; t++) {
    rmx[ty][8 * tx + t] = kmx[t];
    rmn[ty][8 * tx + t] = kmn[t];
  }
  __syncthreads();
  if (threadIdx.x < HD) {
    const int d = threadIdx.x;
    float m1 = -INFINITY, m0 = INFINITY;
    for (int i = 0; i < 16; i++) { m1 = fmaxf(m1, rmx[i][d]); m0 = fminf(m0, rmn[i][d]); }
    krg[d] = m1 - m0;
    if (d == 0) kmed = 0.f;   // stays 0 (no rotation) if non-finite K leaves no element at the middle rank
  }
  __syncthreads();
  if (threadIdx.x < HD) {
    const int d = threadIdx.x;
    int rank = 0;
    for (int i = 0; i < HD; i++) rank += (krg[i] < krg[d]) || (krg[i] == krg[d] && i < d);
    if (rank == HD / 2 - 1) kmed = krg[d];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float kpk = 0.f;
    for (int i = 0; i < HD; i++) kpk = fmaxf(kpk, krg[i]);
    SV[(long long)bh * N_pad + HD + 2 * (N_pad / BN) + 1] = kpk < 16.0f * kmed ? 1.0f : 0.0f;
  }
}

#define STATS1_ARGS(T)                                                                                      const T* __restrict__ K, const T* __restrict__ V, float* __restrict__ KMEAN, float* __restrict__ VMEAN,           float* __restrict__ SV, long long N, long long N_pad, long long H, long long step, long long k_sb,            long long k_sh, long long k_sr, long long v_sb, long long v_sh, long long v_sr
#define STATS1_PASS K, V, KMEAN, VMEAN, SV, N, N_pad, H, step, k_sb, k_sh, k_sr, v_sb, v_sh, v_sr

extern "C" __global__ void __launch_bounds__(256) kv_prologue_i8pv_stats1_f16(STATS1_ARGS(f16)) {
  stats1_body<f16>(STATS1_PASS);
}

extern "C" __global__ void __launch_bounds__(256) kv_prologue_i8pv_stats1_bf16(STATS1_ARGS(__bf16)) {
  stats1_body<__bf16>(STATS1_PASS);
}

template <typename T>
__device__ __forceinline__ void quant_body(const T* __restrict__ K, const T* __restrict__ V, u8* __restrict__ K8,
                                           float* __restrict__ SK, u8* __restrict__ V8T, float* __restrict__ SV,
                                           const float* __restrict__ MEAN, const float* __restrict__ VMEAN,
                                           long long N, long long N_pad, long long H,
                                           long long k_sb, long long k_sh, long long k_sr,
                                           long long v_sb, long long v_sh, long long v_sr,
                                           float FLOOR, int SUB_MEAN, float TAU) {
  __shared__ u8 vb[BN * VROW];
  __shared__ float wred[NT / 32];
  __shared__ float gred[NT / 32];
  __shared__ int tok[NT / 32];

  const int tid = threadIdx.x;
  const int rl  = tid >> 2;
  const int q   = tid & 3;
  const long long n = (long long)blockIdx.x * BN + rl;
  const int bh = blockIdx.y;
  const int b = (int)(bh / H);
  const int h = (int)(bh - (long long)b * H);
  const long long kbase = (long long)b * k_sb + (long long)h * k_sh;
  const long long vbase = (long long)b * v_sb + (long long)h * v_sh;
  const bool live = (n < N);
  const u8* kdst_row = K8 + ((long long)bh * N_pad + n) * HD;
  const long long nt = N_pad / BN;

  // ---- K ----
  float kv[32];
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    float tmp[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) tmp[t] = 0.0f;
    if (live) load8<T>(K + kbase + n * k_sr + (long long)j * 32 + (long long)q * 8, tmp);
    #pragma unroll
    for (int t = 0; t < 8; t++) kv[j * 8 + t] = tmp[t];
  }
  if (SUB_MEAN) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      #pragma unroll
      for (int t = 0; t < 8; t++) {
        const long long col = (long long)j * 32 + q * 8 + t;
        kv[j * 8 + t] = live ? (kv[j * 8 + t] - MEAN[(long long)bh * HD + col]) : 0.0f;
      }
    }
  }
  if (SV[(long long)bh * N_pad + HD + 2 * nt + 1] != 0.0f) {
    // the orthonormal 128-point Hadamard rotation of k - kmean (the attention kernel rotates this head's Q the same
    // way). A thread holds dims 32 j + 8 q + t: the t and j bits rotate in registers, the q bits across the row's lanes.
    #pragma unroll
    for (int s = 1; s < 32; s <<= 1)
      #pragma unroll
      for (int i = 0; i < 32; i++)
        if (!(i & s)) {
          const float a = kv[i], c = kv[i + s];
          kv[i] = a + c;
          kv[i + s] = a - c;
        }
    #pragma unroll
    for (int m = 1; m < 4; m <<= 1)
      #pragma unroll
      for (int i = 0; i < 32; i++) {
        const float o = __shfl_xor(kv[i], m, 32);
        kv[i] = (q & m) ? o - kv[i] : kv[i] + o;
      }
    #pragma unroll
    for (int i = 0; i < 32; i++) kv[i] *= 0.08838834764831845f;
  }
  float kmax = 0.0f;
  #pragma unroll
  for (int t = 0; t < 32; t++) kmax = fmaxf(kmax, fabsf(kv[t]));
  kmax = fmaxf(kmax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, kmax, 1));
  kmax = fmaxf(kmax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, kmax, 2));
  const float krow = kmax;
  // the 16-key group: a wave32 holds 8 rows (4 lanes each), so a group is two waves
  kmax = fmaxf(kmax, __shfl_xor(kmax, 4, 32));
  kmax = fmaxf(kmax, __shfl_xor(kmax, 8, 32));
  kmax = fmaxf(kmax, __shfl_xor(kmax, 16, 32));
  if ((tid & 31) == 0) gred[tid >> 5] = kmax;
  __syncthreads();
  const float gmax = fmaxf(gred[(tid >> 5) & ~1], gred[(tid >> 5) | 1]);
  const int wall = __all((!live) || krow >= TAU * gmax);
  if ((tid & 31) == 0) tok[tid >> 5] = wall;
  __syncthreads();
  int tile_ok = 1;
  #pragma unroll
  for (int i = 0; i < NT / 32; i++) tile_ok &= tok[i];
  kmax = tile_ok ? gmax : krow;
  if (tid == 0) SV[(long long)bh * N_pad + HD + blockIdx.x] = tile_ok ? 1.0f : 0.0f;
  // the bits of 1.0f and 0.0f order like the values, so an integer atomicMin is a logical AND
  if (tid == 0 && !tile_ok) atomicMin((int*)&SV[HD + 2 * nt], 0);
  float sk = fmaxf(__fdiv_rn(kmax, 127.0f), FLOOR);
  if (q == 0) SK[(long long)bh * N_pad + n] = live ? sk : 1.0f;
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    u8* dst = (u8*)kdst_row + (long long)j * 32 + q * 8;
    u64 w = 0;
    #pragma unroll
    for (int t = 0; t < 8; t++) w |= ((u64)i8_rne(__fdiv_rn(kv[j * 8 + t], sk))) << (8 * t);
    *(u64*)dst = w;
  }

  // ---- V ----
  float vv[32];
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    float tmp[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) tmp[t] = 0.0f;
    if (live) load8<T>(V + vbase + n * v_sr + (long long)j * 32 + (long long)q * 8, tmp);
    #pragma unroll
    for (int t = 0; t < 8; t++) {
      const long long col = (long long)j * 32 + q * 8 + t;
      vv[j * 8 + t] = live ? __fdiv_rn(tmp[t] - VMEAN[(long long)bh * HD + col],
                                       SV[(long long)bh * N_pad + col]) : 0.0f;
    }
  }
  float vmax = 0.0f;
  #pragma unroll
  for (int t = 0; t < 32; t++) vmax = fmaxf(vmax, fabsf(vv[t]));
  #pragma unroll
  for (int m = 1; m < 32; m <<= 1) vmax = fmaxf(vmax, __shfl_xor(vmax, m, 32));
  if ((tid & 31) == 0) wred[tid >> 5] = vmax;
  __syncthreads();
  float bmax = wred[0];
  #pragma unroll
  for (int i = 1; i < NT / 32; i++) bmax = fmaxf(bmax, wred[i]);
  const float fb = fmaxf(__fdiv_rn(bmax, 127.0f), 1e-4f);
  if (tid == 0) SV[(long long)bh * N_pad + HD + nt + blockIdx.x] = fb;

  #pragma unroll
  for (int j = 0; j < 4; j++) {
    #pragma unroll
    for (int t = 0; t < 8; t++) {
      float x = rintf(__fdiv_rn(vv[j * 8 + t], fb));
      x = fminf(fmaxf(x, -127.0f), 127.0f);
      vb[rl * VROW + j * 32 + q * 8 + t] = (u8)(signed char)x;
    }
  }
  __syncthreads();
  {
    const int d    = tid >> 1;
    const int half = tid & 1;
    const int n0   = half * 32;
    u8 tmp[32];
    #pragma unroll
    for (int t = 0; t < 32; t++) tmp[t] = vb[(n0 + t) * VROW + d];
    u8* dst = V8T + (long long)bh * HD * N_pad + (long long)d * N_pad + (long long)blockIdx.x * BN + n0;
    *(uint4*)(dst)      = *(uint4*)(tmp);
    *(uint4*)(dst + 16) = *(uint4*)(tmp + 16);
  }
}

#define QUANT_ARGS(T)                                                                                       \
  const T* __restrict__ K, const T* __restrict__ V, u8* __restrict__ K8, float* __restrict__ SK,            \
      u8* __restrict__ V8T, float* __restrict__ SV, const float* __restrict__ MEAN,                         \
      const float* __restrict__ VMEAN, long long N, long long N_pad, long long H, long long k_sb,           \
      long long k_sh, long long k_sr, long long v_sb, long long v_sh, long long v_sr, float FLOOR,           \
      int SUB_MEAN, float TAU
#define QUANT_PASS K, V, K8, SK, V8T, SV, MEAN, VMEAN, N, N_pad, H, k_sb, k_sh, k_sr, v_sb, v_sh, v_sr, FLOOR, \
                   SUB_MEAN, TAU

extern "C" __global__ void __launch_bounds__(NT) kv_prologue_i8pv_quant_f16(QUANT_ARGS(f16)) {
  quant_body<f16>(QUANT_PASS);
}

extern "C" __global__ void __launch_bounds__(NT) kv_prologue_i8pv_quant_bf16(QUANT_ARGS(__bf16)) {
  quant_body<__bf16>(QUANT_PASS);
}
