"""The HIP KV prologue: a ctypes-launched code object that replaces the Triton prologue on the
default SK1 paths. Source: `kernels/hip/kv_prologue.hip`.

It produces the four tensors the SK1 attention objects consume:

    k8  (BH, n_pad, D)   int8 (int8 arm) or e4m3 (fp8 arm), contiguous
    sk  (BH, n_pad)      fp32
    v8t (BH, D, n_pad)   e4m3, contiguous, transposed
    sv  (BH, n_pad)      fp32

Q is not read and `vmean` is not produced. At most two launches (`kseq_mean_*` only when
`smooth_k`, then `quant_kv_*`). Addressing is 64-bit; `strided_view_reason` stays the acceptance
rule, so the prologue serves exactly the views the rest of SK1 accepts.

Loading and launching go through `sk1_loader.HipRuntime`.
"""
from __future__ import annotations

import collections
import ctypes
import os

import torch

from .sk1_loader import HipRuntime, strided_view_reason, torch_stream

HERE = os.path.dirname(os.path.abspath(__file__))
HSACO_NAME = "kv_prologue.gfx1201.hsaco"
HSACO_PATH = os.path.join(HERE, HSACO_NAME)


def hsaco_path_for(arch=None, name=None):
    """The D=128 prologue object for `arch` (`""` / `None` / `gfx1201` -> the gfx1201 object).

    Per-arch (`kv_prologue.gfx1200.hsaco`); the symbols are identical, only the path changes.
    """
    base = name or HSACO_NAME
    if arch in (None, "", "gfx1201"):
        return os.path.join(HERE, base)
    return os.path.join(HERE, base.replace(".gfx1201.hsaco", ".%s.hsaco" % arch))

HD = 128          # the D=128 prologue
BN = 64           # rows per block, matches the Triton BLOCK_N=64
NT = 256          # threads per block
MEAN_BLOCK = (32, 8, 1)

#: (input dtype tag, K quantiser) -> symbol
SYMBOLS = {
    ("f16", "i8"): "quant_kv_f16_i8",
    ("f16", "f8"): "quant_kv_f16_f8",
    ("bf16", "f8"): "quant_kv_bf16_f8",
}
MEAN_SYMBOLS = {"f16": "kseq_mean_f16", "bf16": "kseq_mean_bf16"}

#: every reason `check` can return, so a test can enumerate them
REFUSALS = ("not_cuda", "rank", "shape", "dtype_mismatch", "bf16_int8", "dtype", "head_dim",
            "device", "n_pad", "n_lt_1", "no_symbol", "launch",
            "mean_launch", "mean_dtype", "mean_head_dim") + tuple(
    "k_" + r for r in ("rank", "stride_last", "stride_nonpos", "stride_align", "ptr_align",
                       "stride_overlap", "stride_int32")) + tuple(
    "v_" + r for r in ("rank", "stride_last", "stride_nonpos", "stride_align", "ptr_align",
                       "stride_overlap", "stride_int32"))


def n_pad_for(N: int) -> int:
    """SK1's N_pad contract: a multiple of 64."""
    return ((int(N) + BN - 1) // BN) * BN


class HipPrologue:
    """The loaded HIP prologue. `quant_kv` returns `None` (never raises) when the call is outside
    the contract or an allocation or launch fails, so the caller can run the Triton prologue."""

    def __init__(self, hsaco_path=None, rt=None):
        self.path = hsaco_path or HSACO_PATH
        if not os.path.isfile(self.path):
            raise FileNotFoundError("the HIP prologue object is not built: %s "
                                    "(run tools/build_hsaco.py)" % self.path)
        self.rt = rt if rt is not None else HipRuntime()
        self.mod = self.rt.load_module(self.path)
        self.fns = {k: self.rt.get_function(self.mod, s) for k, s in SYMBOLS.items()}
        self.means = {k: self.rt.get_function(self.mod, s) for k, s in MEAN_SYMBOLS.items()}
        self.n_calls = 0
        self.n_refusals = 0
        self.n_mean_launches = 0
        self.n_kv_launches = 0
        self.refusal_reasons = collections.Counter()
        #: HIP dereferences `kernelParams` asynchronously: keep argument blocks alive past the launch.
        self._keep = collections.deque(maxlen=128)

    def describe(self):
        return "%s sha256=%s symbols=%d" % (
            os.path.basename(self.path), _sha256(self.path)[:16], len(self.fns) + len(self.means))

    # ------------------------------------------------------------------ the contract
    def check(self, k, v, n_pad, int8):
        """`None` if this call is inside the prologue's contract, else a reason string."""
        if not (k.is_cuda and v.is_cuda):
            return "not_cuda"
        if k.dim() != 4 or v.dim() != 4:
            return "rank"
        if tuple(k.shape) != tuple(v.shape):
            return "shape"
        if k.dtype != v.dtype:
            return "dtype_mismatch"
        if k.dtype == torch.bfloat16:
            if int8:
                return "bf16_int8"
        elif k.dtype != torch.float16:
            return "dtype"
        D = int(k.shape[3])
        if D != HD:
            return "head_dim"
        if k.device != v.device:
            return "device"
        r = strided_view_reason(k, "k")
        if r is not None:
            return "k_" + r
        r = strided_view_reason(v, "v")
        if r is not None:
            return "v_" + r
        N = int(k.shape[2])
        if N < 1:
            # N=0 would make the mean kernel compute 0/0 and the KV launch get gridDim.x = 0
            return "n_lt_1"
        if int(n_pad) < N or int(n_pad) % BN:
            return "n_pad"
        return None

    # ------------------------------------------------------------------ the launch
    def quant_kv(self, k, v, n_pad=None, smooth_k=False, int8=True, floor=1e-12):
        """`(k8, sk, v8t, sv)` for a logical `(B, H, N, D)` fp16/bf16 `k`/`v`, or `None` if refused.

        `k`/`v` may be strided views (any pattern `strided_view_reason` accepts); the outputs are
        freshly allocated contiguous tensors. `floor` is the int8-K scale floor only; the fp8-K and
        V scales use a hard-coded `1e-12`.

        Never raises: a failed launch or allocation returns the refusal `"launch"` so the caller
        runs the Triton prologue (a propagated error would route to the gfx12 native path instead).
        """
        if k.dim() != 4 or v.dim() != 4:
            return self._refuse("rank")
        B, H, N, D = (int(x) for x in k.shape)
        BH = B * H
        n_pad = n_pad_for(N) if n_pad is None else int(n_pad)
        reason = self.check(k, v, n_pad, int8)
        if reason is not None:
            return self._refuse(reason)
        dt = "bf16" if k.dtype == torch.bfloat16 else "f16"
        fn = self.fns.get((dt, "i8" if int8 else "f8"))
        if fn is None:
            return self._refuse("no_symbol")

        dev = k.device
        try:
            k8 = torch.empty((BH, n_pad, D),
                             dtype=(torch.int8 if int8 else torch.float8_e4m3fn), device=dev)
            sk = torch.empty((BH, n_pad), dtype=torch.float32, device=dev)
            v8t = torch.empty((BH, D, n_pad), dtype=torch.float8_e4m3fn, device=dev)
            sv = torch.empty((BH, n_pad), dtype=torch.float32, device=dev)

            mean = None
            if smooth_k:
                mean = torch.empty((BH, D), dtype=torch.float32, device=dev)
                self._launch_mean(k, mean, N, H)

            kargs = [ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()),
                     ctypes.c_void_p(k8.data_ptr()), ctypes.c_void_p(sk.data_ptr()),
                     ctypes.c_void_p(v8t.data_ptr()), ctypes.c_void_p(sv.data_ptr()),
                     ctypes.c_void_p(0 if mean is None else mean.data_ptr()),
                     ctypes.c_int64(N), ctypes.c_int64(n_pad), ctypes.c_int64(H),
                     ctypes.c_int64(k.stride(0)), ctypes.c_int64(k.stride(1)),
                     ctypes.c_int64(k.stride(2)),
                     ctypes.c_int64(v.stride(0)), ctypes.c_int64(v.stride(1)),
                     ctypes.c_int64(v.stride(2)),
                     ctypes.c_float(float(floor)), ctypes.c_int(1 if mean is not None else 0)]
            vals, arr = self.rt.launch(fn, (n_pad // BN, BH, 1), (NT, 1, 1), 0, torch_stream(), kargs)
        except Exception:  # noqa: BLE001 - the fallback contract is "never raises"
            return self._refuse("launch")
        self._keep.append((kargs, vals, arr))
        self.n_kv_launches += 1
        self.n_calls += 1
        return k8, sk, v8t, sv

    def _refuse(self, reason):
        self.n_refusals += 1
        self.refusal_reasons[reason] += 1
        return None

    # ------------------------------------------------------------------ the mean, on its own
    def _launch_mean(self, k, mean, N, H):
        dt = "bf16" if k.dtype == torch.bfloat16 else "f16"
        fn = self.means.get(dt)
        if fn is None:
            raise RuntimeError("mean_no_symbol")
        margs = [ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(mean.data_ptr()),
                 ctypes.c_int64(N), ctypes.c_int64(H),
                 ctypes.c_int64(k.stride(0)), ctypes.c_int64(k.stride(1)),
                 ctypes.c_int64(k.stride(2))]
        vals, arr = self.rt.launch(fn, (int(mean.shape[0]), int(k.shape[3]) // 32, 1), MEAN_BLOCK,
                                   0, torch_stream(), margs)
        self._keep.append((margs, vals, arr))
        self.n_mean_launches += 1

    def kseq_mean(self, k):
        """The `smooth_k` sequence-axis mean `(B, H, D)` fp32, or `None` if refused.

        Deterministic: fp64 accumulation of widened fp16/bf16 inputs, one fp32 rounding at the end.
        Never raises: an allocation or launch failure returns the refusal `"mean_launch"`.
        """
        if k.dim() != 4 or not k.is_cuda or k.dtype not in (torch.float16, torch.bfloat16):
            return self._refuse("mean_dtype")
        B, H, N, D = (int(x) for x in k.shape)
        if D != HD:
            return self._refuse("mean_head_dim")
        r = strided_view_reason(k, "k")
        if r is not None:
            return self._refuse("mean_" + r)
        try:
            mean = torch.empty((B * H, D), dtype=torch.float32, device=k.device)
            self._launch_mean(k, mean, N, H)
        except Exception:  # noqa: BLE001 - never raises
            return self._refuse("mean_launch")
        return mean.view(B, H, D)


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
