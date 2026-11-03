// fp8 e4m3 linear with one activation scale per row (token), for torch._scaled_mm on RDNA4.
//
//   fp8_rowquant_{f16,bf16}  grid (M), block 256. X (M, K) row-major with row stride x_rs -> X8 e4m3 and SX (M) fp32,
//                            SX[m] = max(|X[m, :]|, 1e-12), X8 = rne(448 X / SX[m]) clamped to +-448. X8 is written as
//                            `parts` contiguous (M, K / parts) blocks, so each K slice is a contiguous GEMM operand
//                            (hipBLASLt runs ~1.6x slower when an operand's leading dimension exceeds its width).
//   fp8_rowscale_{f16,bf16}  grid (M, ceil(N / 2048)), block 256 (M on x: grid y stops at 65535). Y0 = (Y0 [+ Y1]) * SX[m] [+ bias], in place, Y0 / Y1
//                            (M, N) row-major in the GEMM's output type; Y1 and bias may be null. The GEMM runs with
//                            scale_a = 1 / 448, so Y0 holds y / SX[m] and cannot overflow fp16 where y does not.
//
// K must be a multiple of 8 * parts and at most 32768; N a multiple of 8.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef unsigned short u16;
typedef unsigned int   u32;
typedef _Float16       f16;

#define NT 256
#define MAXC 16   // 16-byte chunks per thread: K <= 256 * 8 * 16 = 32768

template <typename T> __device__ __forceinline__ T frombits(u16 b) {
  union { u16 u; T f; } x; x.u = b; return x.f;
}
template <typename T> __device__ __forceinline__ u16 tobits(T v) {
  union { u16 u; T f; } x; x.f = v; return x.u;
}

template <typename T>
__device__ __forceinline__ void rowquant_body(const T* __restrict__ X, uint8_t* __restrict__ X8, float* __restrict__ SX,
                                              int M, int K, int parts, long long x_rs) {
  __shared__ float wmax[NT / 32];
  const int m = blockIdx.x, tid = threadIdx.x;
  const T* xr = X + (long long)m * x_rs;
  const int nch = K / 8;
  float v[MAXC][8];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < MAXC; i++) {
    const int c = tid + i * NT;
    if (c < nch) {
      const uint4 u = *(const uint4*)(xr + 8 * c);
      const u16* h = (const u16*)&u;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        v[i][e] = (float)frombits<T>(h[e]);
        amax = fmaxf(amax, fabsf(v[i][e]));
      }
    }
  }
#pragma unroll
  for (int s = 16; s; s >>= 1) amax = fmaxf(amax, __shfl_xor(amax, s, 32));
  if ((tid & 31) == 0) wmax[tid >> 5] = amax;
  __syncthreads();
  amax = wmax[0];
#pragma unroll
  for (int i = 1; i < NT / 32; i++) amax = fmaxf(amax, wmax[i]);
  amax = fmaxf(amax, 1e-12f);
  const float inv = 448.0f / amax;
  if (tid == 0) SX[m] = amax;
  const int kp = K / parts;
#pragma unroll
  for (int i = 0; i < MAXC; i++) {
    const int c = tid + i * NT;
    if (c < nch) {
      float y[8];
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const float t = v[i][e] * inv;   // a NaN stays NaN (fminf / fmaxf would make it -448)
        y[e] = t > 448.0f ? 448.0f : (t < -448.0f ? -448.0f : t);
      }
      // v_cvt_pk_fp8_f32 packs two values into the low (false) or high (true) 16 bits
      int lo = __builtin_amdgcn_cvt_pk_fp8_f32(y[0], y[1], 0, false);
      lo = __builtin_amdgcn_cvt_pk_fp8_f32(y[2], y[3], lo, true);
      int hi = __builtin_amdgcn_cvt_pk_fp8_f32(y[4], y[5], 0, false);
      hi = __builtin_amdgcn_cvt_pk_fp8_f32(y[6], y[7], hi, true);
      const int k = 8 * c, p = k / kp;
      *(uint2*)(X8 + (long long)p * M * kp + (long long)m * kp + (k - p * kp)) = make_uint2((u32)lo, (u32)hi);
    }
  }
}

template <typename T>
__device__ __forceinline__ void rowscale_body(T* __restrict__ Y0, const T* __restrict__ Y1, const float* __restrict__ SX,
                                              const T* __restrict__ bias, int M, int N) {
  const int m = blockIdx.x;
  const int n = (blockIdx.y * NT + threadIdx.x) * 8;
  if (n >= N) return;
  const float s = SX[m];
  T* yr = Y0 + (long long)m * N + n;
  uint4 a = *(const uint4*)yr;
  u16* ha = (u16*)&a;
  float r[8];
#pragma unroll
  for (int e = 0; e < 8; e++) r[e] = (float)frombits<T>(ha[e]);
  if (Y1) {
    const uint4 b = *(const uint4*)(Y1 + (long long)m * N + n);
    const u16* hb = (const u16*)&b;
#pragma unroll
    for (int e = 0; e < 8; e++) r[e] += (float)frombits<T>(hb[e]);
  }
  if (bias) {
    const uint4 c = *(const uint4*)(bias + n);
    const u16* hc = (const u16*)&c;
#pragma unroll
    for (int e = 0; e < 8; e++) r[e] = fmaf(r[e], s, (float)frombits<T>(hc[e]));
  } else {
#pragma unroll
    for (int e = 0; e < 8; e++) r[e] *= s;
  }
#pragma unroll
  for (int e = 0; e < 8; e++) ha[e] = tobits<T>((T)r[e]);
  *(uint4*)yr = a;
}

extern "C" __global__ void __launch_bounds__(NT)
fp8_rowquant_f16(const f16* X, uint8_t* X8, float* SX, int M, int K, int parts, long long x_rs) {
  rowquant_body<f16>(X, X8, SX, M, K, parts, x_rs);
}

extern "C" __global__ void __launch_bounds__(NT)
fp8_rowquant_bf16(const __bf16* X, uint8_t* X8, float* SX, int M, int K, int parts, long long x_rs) {
  rowquant_body<__bf16>(X, X8, SX, M, K, parts, x_rs);
}

extern "C" __global__ void __launch_bounds__(NT)
fp8_rowscale_f16(f16* Y0, const f16* Y1, const float* SX, const f16* bias, int M, int N) {
  rowscale_body<f16>(Y0, Y1, SX, bias, M, N);
}

extern "C" __global__ void __launch_bounds__(NT)
fp8_rowscale_bf16(__bf16* Y0, const __bf16* Y1, const float* SX, const __bf16* bias, int M, int N) {
  rowscale_body<__bf16>(Y0, Y1, SX, bias, M, N);
}
