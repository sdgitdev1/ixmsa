// int8 Q.K and int8 P.V attention, head_dim 128, for RDNA4. Shared body of the eight attn_i8pv_* objects; each
// .hip file sets three macros and includes this:
//
//   I8PV_MIX      0: serves a call whose every head kv_prologue_i8pv flagged "group" (every 16-key group of the head
//                    kept each key's amax within a factor TAU of the group max), with one K scale per 16 keys on
//                    interior tiles. 1: serves any other call, each head on its own path (per-token K scales on the
//                    heads not flagged). Launch both on the same prologue output; the one the call flag does not
//                    pick returns at once. One launch per call, because two launches that split the heads leave two
//                    partly idle tails: on a real Krea2 call with 24 group heads of 48 that cost 6 %.
//   I8PV_STRIDED  1: Q and O addressed by (row, head, batch) element strides (strided HND views and NHD).
//   I8PV_BF16     1: bf16 Q and O; 0: fp16.
//   I8PV_NAME     symbol prefix: I8PV_NAME_fwd and I8PV_NAME_fwd_causal.
//
// Inputs (from kv_prologue_i8pv): K8 (BH, N_pad, 128) int8 after the K mean, SK (BH, N_pad) its per-key scale;
// V8T (BH, 128, N_pad) int8 of (v - vmean) / s_d / f_j, VMEAN (BH, 128); SV (BH, N_pad) holds s_d at [0, 128),
// the tile flags at [128, 128 + nt), f_j at [128 + nt, 128 + 2 nt), the head flag at 128 + 2 nt and, in head 0's
// row only, the call flag (every head flagged group) at 129 + 2 nt.
// Q is quantised to int8 per token here. sm_scale must be positive (the max of scaled scores is taken as the
// scaled max).
//
// Per 64-key tile: S = Q.K in int32 (WMMA iu8), a row max, P = u8(exp2(s - (tile max - log2 255))) so every
// tile uses the full byte range, the row sum from the same bytes (v_sad_u8), then O += (V^T P^T) in int32
// (WMMA iu8, A = V signed, B = P unsigned). The output and the row sum live in a frame R that moves only when a
// tile's reference exceeds it, and then 8 (a factor 256) past it; each tile is folded in with
// fma(pv, exp2(ref - R) * f_j, o), so the 64-wide rescale of O runs only on those moves. The P.V accumulator
// starts at the bits of 1.5*2^23, so its int32 result read as fp32 is exactly 1.5*2^23 + pv; the accumulated
// offset is tracked and cancelled every 8 tiles. On the group path the Q.K accumulator is seeded the same
// way, which removes the int -> fp32 convert of every score.
//
// The lazy rescale and the P.V seed follow comfy-kitchen's int8 attention (Apache-2.0), as do the single
// LDS buffer and the sched_barrier fences that keep its register count down.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int      i32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef float    f32x4 __attribute__((ext_vector_type(4)));
typedef float    f32x8 __attribute__((ext_vector_type(8)));
typedef int      i32x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef __bf16   bf16x8 __attribute__((ext_vector_type(8)));

#define BM 128
#define BN 64
#define HD 128
#define KSTRIDE 152
#define VSTRIDE 72
#define I8_MAX 127.0f
#define LOG2E 1.4426950408889634f
#define P_LOG2 7.9943534f   // log2(255)
#define HEADROOM 8.0f       // a new frame sits this far (log2) above the tile reference that moved it
#define MAGIC_I 0x4B400000  // bits of 1.5*2^23
#define MAGIC_F 12582912.0f

#if I8PV_BF16
typedef __bf16 qo_t;
typedef bf16x8 qo8_t;
#else
typedef _Float16 qo_t;
typedef f16x8 qo8_t;
#endif

__device__ __forceinline__ float clamp127(float x) { return fminf(fmaxf(x, -128.0f), 127.0f); }

// four already-integral floats -> four int8 in one VGPR (rint first: the hardware pack truncates)
__device__ __forceinline__ int pk4i(float a, float b, float c, float d) {
  const int ia = (int)__builtin_rintf(a);
  const int ib = (int)__builtin_rintf(b);
  const int ic = (int)__builtin_rintf(c);
  const int id = (int)__builtin_rintf(d);
  return (ia & 0xff) | ((ib & 0xff) << 8) | ((ic & 0xff) << 16) | ((id & 0xff) << 24);
}

// four probabilities -> four u8 (v_cvt_pk_u8_f32 rounds to nearest and saturates)
__device__ __forceinline__ unsigned pk4u(float a, float b, float c, float d) {
  unsigned r = __builtin_amdgcn_cvt_pk_u8_f32(a, 0, 0u);
  r = __builtin_amdgcn_cvt_pk_u8_f32(b, 1, r);
  r = __builtin_amdgcn_cvt_pk_u8_f32(c, 2, r);
  return __builtin_amdgcn_cvt_pk_u8_f32(d, 3, r);
}

__device__ __forceinline__ i32x8 wmma_i8(i32x2 a, i32x2 b, i32x8 c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, true);
}

// O^T = V^T P^T: A = V signed, B = P unsigned
__device__ __forceinline__ i32x8 wmma_i8u8(i32x2 a, i32x2 b, i32x8 c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, false, b, c, false);
}

template <bool B> struct btag { static constexpr bool v = B; };

template <bool CAUSAL>
__device__ __forceinline__ void i8pv_body(const qo_t* __restrict__ Q, const uint8_t* __restrict__ K8,
                                          const float* __restrict__ SK, const uint8_t* __restrict__ V8T,
                                          const float* __restrict__ SV, const float* __restrict__ VMEAN,
                                          qo_t* __restrict__ O, int H, int N, int N_pad, float sm_scale
#if I8PV_STRIDED
                                          , int q_rstride, int q_hstride, int q_bstride,
                                          int o_rstride, int o_hstride, int o_bstride
#endif
                                          ) {
  __shared__ __attribute__((aligned(16))) uint8_t sK[BN * KSTRIDE];
  __shared__ __attribute__((aligned(16))) uint8_t sV[HD * VSTRIDE];
  __shared__ __attribute__((aligned(16))) float sS[BN];

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, lr = lane & 15, lh = lane >> 4;
  const int start_m = blockIdx.x;
  const size_t bh = (size_t)blockIdx.z * H + blockIdx.y;
#if I8PV_STRIDED
  const size_t qbase = (size_t)blockIdx.z * (size_t)q_bstride + (size_t)blockIdx.y * (size_t)q_hstride;
  const size_t obase = (size_t)blockIdx.z * (size_t)o_bstride + (size_t)blockIdx.y * (size_t)o_hstride;
#endif
  const int q0 = start_m * BM + w * 16;
  const int qrow = q0 + lr;
  const int n_tiles = N_pad / BN;
  const int j_hi = CAUSAL ? min(n_tiles, 2 * start_m + 2) : n_tiles;
  const int full = CAUSAL ? 2 * start_m : n_tiles;
  // the call flag picks the object; the head flag is per (b, h), so a whole block takes one path
#if I8PV_MIX
  if (SV[HD + 2 * n_tiles + 1] != 0.f) return;
  const bool head_grp = SV[bh * N_pad + HD + 2 * n_tiles] != 0.f;
#else
  if (SV[HD + 2 * n_tiles + 1] == 0.f) return;
#endif

  // ---- Q: raw 16-bit -> int8, one scale per token ----
  qo8_t qh[8];
  float amax = 0.f;
#if I8PV_STRIDED
  const qo_t* qp = Q + qbase + (size_t)qrow * (size_t)q_rstride + 8 * lh;
#else
  const qo_t* qp = Q + (bh * N + qrow) * HD + 8 * lh;
#endif
#pragma unroll
  for (int dt = 0; dt < 8; ++dt) {
    if (qrow < N) {
      qh[dt] = *(const qo8_t*)(qp + 16 * dt);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) qh[dt][e] = (qo_t)0.f;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf((float)qh[dt][e]));
  }
  amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
  amax = fmaxf(amax, 1e-12f);
  const float qinv = I8_MAX / amax;
  const float c_q = sm_scale * LOG2E * (amax / I8_MAX);
  i32x2 qf[8];
#pragma unroll
  for (int dt = 0; dt < 8; ++dt) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = clamp127(rintf((float)qh[dt][e] * qinv));
    qf[dt] = (i32x2){pk4i(v[0], v[1], v[2], v[3]), pk4i(v[4], v[5], v[6], v[7])};
  }

  // ---- K / V tile -> LDS (one buffer: barrier, load, barrier, compute) ----
  // The staging and the epilogue below each come in two forms that compute the same bytes. clang schedules the KV
  // loop differently around them, and each object is faster with one: the contiguous objects by ~1 % with these
  // two, the strided ones by 1-3 % with the other two (kernel time, A/A within 0.3 %).
#if I8PV_STRIDED
  auto stage = [&](int j) {
    const uint8_t* kp = K8 + (bh * N_pad + (size_t)j * BN + (tid >> 2)) * HD + (tid & 3) * 32;
    const u32x4 kr0 = *(const u32x4*)kp, kr1 = *(const u32x4*)(kp + 16);
    const uint8_t* vp = V8T + (bh * HD + (tid >> 1)) * N_pad + (size_t)j * BN + (tid & 1) * 32;
    const u32x4 vr0 = *(const u32x4*)vp, vr1 = *(const u32x4*)(vp + 16);
    uint8_t* kd = &sK[(tid >> 2) * KSTRIDE + (tid & 3) * 32];
    *(u32x2*)(kd + 0) = kr0.xy;
    *(u32x2*)(kd + 8) = kr0.zw;
    *(u32x2*)(kd + 16) = kr1.xy;
    *(u32x2*)(kd + 24) = kr1.zw;
    uint8_t* vd = &sV[(tid >> 1) * VSTRIDE + (tid & 1) * 32];
    *(u32x2*)(vd + 0) = vr0.xy;
    *(u32x2*)(vd + 8) = vr0.zw;
    *(u32x2*)(vd + 16) = vr1.xy;
    *(u32x2*)(vd + 24) = vr1.zw;
    if (tid < 16)
      *(f32x4*)&sS[4 * tid] = *(const f32x4*)(SK + bh * N_pad + (size_t)j * BN + 4 * tid);
  };

#else
  u32x4 kr0, kr1, vr0, vr1;
  f32x4 sr = {0.f, 0.f, 0.f, 0.f};
  auto gload = [&](int j) {
    const uint8_t* kp = K8 + (bh * N_pad + (size_t)j * BN + (tid >> 2)) * HD + (tid & 3) * 32;
    kr0 = *(const u32x4*)kp;
    kr1 = *(const u32x4*)(kp + 16);
    const uint8_t* vp = V8T + (bh * HD + (tid >> 1)) * N_pad + (size_t)j * BN + (tid & 1) * 32;
    vr0 = *(const u32x4*)vp;
    vr1 = *(const u32x4*)(vp + 16);
    if (tid < 16)
      sr = *(const f32x4*)(SK + bh * N_pad + (size_t)j * BN + 4 * tid);
  };
  auto lstore = [&]() {
    uint8_t* kd = &sK[(tid >> 2) * KSTRIDE + (tid & 3) * 32];
    *(u32x2*)(kd + 0) = kr0.xy;
    *(u32x2*)(kd + 8) = kr0.zw;
    *(u32x2*)(kd + 16) = kr1.xy;
    *(u32x2*)(kd + 24) = kr1.zw;
    uint8_t* vd = &sV[(tid >> 1) * VSTRIDE + (tid & 1) * 32];
    *(u32x2*)(vd + 0) = vr0.xy;
    *(u32x2*)(vd + 8) = vr0.zw;
    *(u32x2*)(vd + 16) = vr1.xy;
    *(u32x2*)(vd + 24) = vr1.zw;
    if (tid < 16)
      *(f32x4*)&sS[4 * tid] = sr;
  };
  auto stage = [&](int j) { gload(j); lstore(); };

#endif
  f32x8 acc[8];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[nt][e] = 0.f;
  float m_i = -INFINITY, l_i = 0.f, bias_i = 0.f, R = -INFINITY;
  const float* const fb = SV + bh * N_pad + HD + n_tiles;

  // Interior tiles (every key in range, no causal mask) run a copy of the body with no skip or mask code;
  // only those take the group path.
  auto iter = [&](auto bnd, auto g, int j) {
    constexpr bool BND = decltype(bnd)::v;
    constexpr bool grp = decltype(g)::v && !BND;
    if (j) __syncthreads();
    stage(j);
    __syncthreads();
    const uint8_t* const kb = &sK[lr * KSTRIDE + 8 * lh];
    const uint8_t* const vb = &sV[lr * VSTRIDE + 8 * lh];
    const float* const skb = &sS[8 * lh];

    // ---- phase 1: Q.K, scale, mask, one row max over 64 keys ----
    f32x8 sc[2][2];
    float gq[2][2];
    i32x8 sci[2][2];
    bool skp[2];
    float lmax = -INFINITY;
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const int base = j * BN + st * 32;
      bool skip = false, need_mask = false;
      if (BND) {
        if (CAUSAL) {
          skip = base > q0 + 15;
          need_mask = (j >= full);
        } else {
          skip = base >= N;
          need_mask = (base + 32 > N);
        }
      }
      skp[st] = skip;
      if (skip) continue;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int e = 0; e < 8; ++e) sci[st][kk][e] = grp ? MAGIC_I : 0;
#pragma unroll
      for (int dt = 0; dt < 8; ++dt)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const i32x2 a = *(const i32x2*)(kb + (st * 32 + kk * 16) * KSTRIDE + dt * 16);
          sci[st][kk] = wmma_i8(a, qf[dt], sci[st][kk]);
        }
      // two accumulators in flight, without every K fragment hoisted ahead of them
      __builtin_amdgcn_sched_barrier(0);
      if (grp) {
        // One scale per 16 keys (the group's first key carries it; the prologue wrote the same value for all
        // 16), read as a wave-uniform scalar load. Each seeded score read as fp32 is 1.5*2^23 + s exactly
        // (|s| <= 128*127*127 < 2^22).
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const float g = SK[bh * N_pad + (size_t)j * BN + st * 32 + kk * 16];
          gq[st][kk] = c_q * g;
          float fm = __builtin_bit_cast(float, (int)sci[st][kk][0]);
#pragma unroll
          for (int e = 1; e < 8; ++e) fm = fmaxf(fm, __builtin_bit_cast(float, (int)sci[st][kk][e]));
          lmax = fmaxf(lmax, (fm - MAGIC_F) * g);
        }
        continue;
      }
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const f32x4 k0 = *(const f32x4*)(skb + st * 32 + kk * 16);
        const f32x4 k1 = *(const f32x4*)(skb + st * 32 + kk * 16 + 4);
        const float skv[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) sc[st][kk][e] = (float)sci[st][kk][e] * skv[e];
      }
      if (need_mask) {
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int key = base + kk * 16 + 8 * lh + e;
            if (CAUSAL ? (key > qrow) : (key >= N)) sc[st][kk][e] = -INFINITY;
          }
      }
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int e = 0; e < 8; ++e) lmax = fmaxf(lmax, sc[st][kk][e]);
    }
    lmax = fmaxf(lmax, __shfl_xor(lmax, 16, 32));
    lmax *= c_q;  // c_q > 0, so max(x) * c_q == max(x * c_q); -inf stays -inf
    const float m_new = fmaxf(m_i, lmax);
    const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
    // this tile's reference: its own row max, clamped 60 below the running max
    const float r_t = fmaxf(lmax, m_use - 60.f);
    m_i = m_new;
    const bool upd = r_t > R;
    const float R_new = upd ? r_t + HEADROOM : R;
    if (__any(upd)) {
      const float g = __builtin_amdgcn_exp2f(R - R_new);
      l_i *= g;
      bias_i *= g;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[nt][e] *= g;
    }
    R = R_new;
    const float ts = __builtin_amdgcn_exp2f(r_t - R);

    // ---- phase 2: P bytes and their row sum ----
    const float m_off = r_t - P_LOG2;
    unsigned isum = 0u;
    i32x2 pa[2][2] = {{{0, 0}, {0, 0}}, {{0, 0}, {0, 0}}};
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      if (skp[st]) continue;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        float p[8];
        if (grp) {
          const float cg = fmaf(-MAGIC_F, gq[st][kk], -m_off);
#pragma unroll
          for (int e = 0; e < 8; ++e)
            p[e] = __builtin_amdgcn_exp2f(fmaf(__builtin_bit_cast(float, (int)sci[st][kk][e]), gq[st][kk], cg));
        } else {
#pragma unroll
          for (int e = 0; e < 8; ++e) p[e] = __builtin_amdgcn_exp2f(fmaf(sc[st][kk][e], c_q, -m_off));
        }
        const unsigned lo = pk4u(p[0], p[1], p[2], p[3]);
        const unsigned hi = pk4u(p[4], p[5], p[6], p[7]);
        isum = __builtin_amdgcn_sad_u8(lo, 0u, isum);
        isum = __builtin_amdgcn_sad_u8(hi, 0u, isum);
        pa[st][kk] = (i32x2){(int)lo, (int)hi};
      }
    }
    float psum = (float)isum;
    psum += __shfl_xor(psum, 16, 32);
    l_i = fmaf(psum, ts, l_i);
    const float tv = ts * fb[j];
    bias_i = fmaf(MAGIC_F, tv, bias_i);

    // ---- phase 3: P.V in int32 per 16 channels (seeded), folded into the fp32 output ----
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      i32x8 t = {MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I};
#pragma unroll
      for (int st = 0; st < 2; ++st)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const i32x2 b = *(const i32x2*)(vb + nt * 16 * VSTRIDE + st * 32 + kk * 16);
          t = wmma_i8u8(b, pa[st][kk], t);
        }
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[nt][e] = fmaf(__builtin_bit_cast(float, (int)t[e]), tv, acc[nt][e]);
      if (nt & 1) __builtin_amdgcn_sched_barrier(0);
    }
    // cancel the accumulated seed every 8 tiles, so it cannot grow with N
    if ((j & 7) == 7) {
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[nt][e] -= bias_i;
      bias_i = 0.f;
    }
  };
  const int j_int = min(j_hi, CAUSAL ? full : N / BN);
  int j = 0;
#if I8PV_MIX
  // only the interior loop differs between the two paths; Q, the boundary tiles and the epilogue are shared
  if (head_grp)
    for (; j < j_int; ++j) iter(btag<false>{}, btag<true>{}, j);
  else
    for (; j < j_int; ++j) iter(btag<false>{}, btag<false>{}, j);
#else
  for (; j < j_int; ++j) iter(btag<false>{}, btag<true>{}, j);
#endif
  for (; j < j_hi; ++j) iter(btag<true>{}, btag<false>{}, j);

  // ---- epilogue ----
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[nt][e] -= bias_i;
  const float inv = l_i > 0.f ? 1.0f / l_i : 0.f;
  if (qrow < N) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int d0 = nt * 16 + 8 * lh;
      const f32x4 s0 = *(const f32x4*)(SV + bh * N_pad + d0), s1 = *(const f32x4*)(SV + bh * N_pad + d0 + 4);
#if I8PV_STRIDED
      const f32x4 m0 = *(const f32x4*)(VMEAN + bh * HD + d0), m1 = *(const f32x4*)(VMEAN + bh * HD + d0 + 4);
      const float vsv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
      const float vmv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
      qo8_t o;
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = (qo_t)(acc[nt][e] * inv * vsv[e] + vmv[e]);
#else
      const float vsv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
      qo8_t o;
      // VMEAN is never null here; the test keeps clang from hoisting these loop-invariant loads above the KV
      // loop, where they would hold 7 more VGPRs for the whole kernel (1 % slower).
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = (qo_t)(acc[nt][e] * inv * vsv[e] + (VMEAN ? VMEAN[bh * HD + d0 + e] : 0.f));
#endif
#if I8PV_STRIDED
      *(qo8_t*)(O + obase + (size_t)qrow * (size_t)o_rstride + d0) = o;
#else
      *(qo8_t*)(O + (bh * N + qrow) * HD + d0) = o;
#endif
    }
  }
}

#define I8PV_CAT2(a, b) a##b
#define I8PV_CAT(a, b) I8PV_CAT2(a, b)

#if I8PV_STRIDED
#define I8PV_SARGS , int q_rstride, int q_hstride, int q_bstride, int o_rstride, int o_hstride, int o_bstride
#define I8PV_SPASS , q_rstride, q_hstride, q_bstride, o_rstride, o_hstride, o_bstride
#else
#define I8PV_SARGS
#define I8PV_SPASS
#endif

#define I8PV_ARGS                                                                                   \
  const qo_t *__restrict__ Q, const uint8_t *__restrict__ K8, const float *__restrict__ SK,         \
      const uint8_t *__restrict__ V8T, const float *__restrict__ SV, const float *__restrict__ VMEAN, \
      qo_t *__restrict__ O, int H, int N, int N_pad, float sm_scale I8PV_SARGS

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8)))
I8PV_CAT(I8PV_NAME, _fwd)(I8PV_ARGS) {
  i8pv_body<false>(Q, K8, SK, V8T, SV, VMEAN, O, H, N, N_pad, sm_scale I8PV_SPASS);
}

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8)))
I8PV_CAT(I8PV_NAME, _fwd_causal)(I8PV_ARGS) {
  i8pv_body<true>(Q, K8, SK, V8T, SV, VMEAN, O, H, N, N_pad, sm_scale I8PV_SPASS);
}
