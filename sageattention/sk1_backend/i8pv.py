"""int8 Q.K + int8 P.V attention: `kv_prologue_i8pv` and the `attn_i8pv_{grp,mix}_{f16,bf16}[_strided]` objects
(sources `kernels/hip/attn_i8pv.h`, `kernels/hip/kv_prologue_i8pv.hip`).

A call is two prologue launches (the K mean from ~1024 sampled rows, the V statistics from every row; then the
quantisation) and two attention launches on the same prologue output. Per 64-key tile the prologue takes one K
scale per 16 keys where that costs no key more than log2(1/TAU) = 2 bits, per-token scales elsewhere, and
flags each head whose tiles all took group scales, and the call when every head did. The `grp` object serves a
call whose every head is flagged (group scale and a seeded Q.K on interior tiles); the `mix` object serves any
other call, each head on its own path. The one the call flag does not pick returns at once, so the call runs as
one launch: two launches that split the heads leave two partly idle tails.

Contract: fp16 or bf16, head_dim 128, sm_scale > 0, N_pad >= 130 + 2 * N_pad / 64 (N >= 129), q/k/v logical
(B, H, N, D) views that `strided_view_reason` accepts.
"""
from __future__ import annotations

import ctypes
import os

import torch

from .sk1_loader import HipRuntime, _strided_triple, torch_stream

HERE = os.path.dirname(os.path.abspath(__file__))
HD = 128
BN = 64
TAU = 0.25
K_ROWS = 1024       # rows the K mean is taken from (all of them up to this length)
TAGS = {torch.float16: "f16", torch.bfloat16: "bf16"}


def object_path(name, arch):
    return os.path.join(HERE, "%s.%s.hsaco" % (name, arch or "gfx1201"))


def refusal(q, sm_scale):
    """`None` if the call is inside the contract above (layout and strides are checked by the caller's envelope),
    else the reason."""
    if q.dtype not in TAGS:
        return "i8pv_dtype"
    if q.size(-1) != HD:
        return "i8pv_head_dim"
    if not sm_scale > 0:
        return "i8pv_scale"
    n_pad = (int(q.size(2)) + BN - 1) // BN * BN
    if n_pad < HD + 2 * (n_pad // BN) + 1:
        return "i8pv_short"
    return None


class I8pv:
    """The objects for one arch; each (dtype, strided) pair loads on its first call."""

    def __init__(self, arch):
        self.arch = arch
        self.rt = HipRuntime()
        mod = self.rt.load_module(object_path("kv_prologue_i8pv", arch))
        self.f_stats = {dt: self.rt.get_function(mod, "kv_prologue_i8pv_stats1_" + t) for dt, t in TAGS.items()}
        self.f_quant = {dt: self.rt.get_function(mod, "kv_prologue_i8pv_quant_" + t) for dt, t in TAGS.items()}
        self._fns = {}
        self._keep = None

    def functions(self, dtype, strided):
        """((grp_fwd, grp_fwd_causal), (mix_fwd, mix_fwd_causal)) for one dtype and layout."""
        key = (dtype, strided)
        if key not in self._fns:
            pair = []
            for kind in ("grp", "mix"):
                name = "attn_i8pv_%s_%s%s" % (kind, TAGS[dtype], "_strided" if strided else "")
                mod = self.rt.load_module(object_path(name, self.arch))
                pair.append((self.rt.get_function(mod, name + "_fwd"), self.rt.get_function(mod, name + "_fwd_causal")))
            self._fns[key] = tuple(pair)
        return self._fns[key]

    def prologue(self, k, v, smooth_k, st):
        """k, v: logical (B, H, N, 128) views; st: the stream handle. Returns k8, sk, v8t, sv, vmean, n_pad."""
        B, H, N, D = (int(x) for x in k.shape)
        BH = B * H
        n_pad = (N + BN - 1) // BN * BN
        dev = k.device
        k8 = torch.empty(BH, n_pad, D, dtype=torch.int8, device=dev)
        sk = torch.empty(BH, n_pad, dtype=torch.float32, device=dev)
        v8t = torch.empty(BH, D, n_pad, dtype=torch.int8, device=dev)
        sv = torch.empty(BH, n_pad, dtype=torch.float32, device=dev)
        kmean = torch.empty(BH, D, dtype=torch.float32, device=dev)
        vmean = torch.empty(BH, D, dtype=torch.float32, device=dev)
        P, I64 = ctypes.c_void_p, ctypes.c_int64
        strides = [I64(k.stride(0)), I64(k.stride(1)), I64(k.stride(2)), I64(v.stride(0)), I64(v.stride(1)),
                   I64(v.stride(2))]
        a1 = [P(k.data_ptr()), P(v.data_ptr()), P(kmean.data_ptr()), P(vmean.data_ptr()), P(sv.data_ptr()), I64(N),
              I64(n_pad), I64(H), I64(max(1, N // K_ROWS))] + strides
        a3 = [P(k.data_ptr()), P(v.data_ptr()), P(k8.data_ptr()), P(sk.data_ptr()), P(v8t.data_ptr()),
              P(sv.data_ptr()), P(kmean.data_ptr()), P(vmean.data_ptr()), I64(N), I64(n_pad), I64(H)] + strides + \
            [ctypes.c_float(1e-12), ctypes.c_int(1 if smooth_k else 0), ctypes.c_float(TAU)]
        launches = [self.rt.launch(self.f_stats[k.dtype], (BH, 1, 1), (256, 1, 1), 0, st, a1),
                    self.rt.launch(self.f_quant[k.dtype], (n_pad // BN, BH, 1), (256, 1, 1), 0, st, a3)]
        self._keep_pro = (a1, a3, launches)
        return k8, sk, v8t, sv, vmean, n_pad

    def run(self, qq, kk, vv, out, causal, sm_scale, smooth_k, contiguous_hnd):
        """qq/kk/vv: logical (B, H, N, D) views the caller's envelope accepted (`strided_view_reason`); `out` the
        (B*H, N, D) buffer for contiguous HND, else the logical (B, H, N, D) view to write, allocated by the caller.
        `out` is written in place. Both objects take the same arguments, so they are built once."""
        B, H, N, D = (int(x) for x in qq.shape)
        fns = self.functions(qq.dtype, not contiguous_hnd)   # load (or fail) before any prologue work
        st = torch_stream()
        k8, sk, v8t, sv, vmean, n_pad = self.prologue(kk, vv, smooth_k, st)
        P, I = ctypes.c_void_p, ctypes.c_int
        args = [P(qq.data_ptr()), P(k8.data_ptr()), P(sk.data_ptr()), P(v8t.data_ptr()), P(sv.data_ptr()),
                P(vmean.data_ptr()), P(out.data_ptr()), I(H), I(N), I(n_pad), ctypes.c_float(float(sm_scale))]
        if not contiguous_hnd:
            args += [I(x) for x in _strided_triple(qq, "q")[:3] + _strided_triple(out, "out")[:3]]
        grid = ((N + 127) // 128, H, B)
        launches = [self.rt.launch(f[1] if causal else f[0], grid, (256, 1, 1), 0, st, args)
                    for f in fns]
        # the argument storage must outlive the asynchronous launches
        self._keep = (args, launches, self._keep_pro)
