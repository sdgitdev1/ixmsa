"""fp8 e4m3 linear with one activation scale per token, for RDNA4 (`kernels/hip/fp8_rowquant.hip`).

`linear(x, w8, w_scale, bias=None)` computes x @ (w8 * w_scale)^T in x's dtype through `torch._scaled_mm`:

1. one HIP pass quantizes x to e4m3 with one scale per row (token), its absolute maximum,
2. hipBLASLt runs the fp8 GEMM with scale_a = 1 / 448, so it writes y / row max, which stays in fp16 range; with
   K >= 12288 it runs as two contiguous K halves, because hipBLASLt picks a ~2x slower kernel for the long-K shape
   (Krea2's mlp.down, 16384 -> 6144: 110 TF unsplit, 214 TF per half),
3. one HIP pass sums the halves, applies the row scales and adds the bias.

Returns None when the call is outside that (no GPU object for this arch, activations other than non-empty fp16/bf16,
a weight that is not a contiguous e4m3 (N, K) tensor with one scale, K or N not a multiple of 16, K over 32768, or
autograd needing a gradient through x or bias: the HIP passes are not differentiable), so the caller keeps its own
path.
"""
from __future__ import annotations

import ctypes
import os

import torch

_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sk1_backend")
_FNS = {}
_KEEP = []   # argument storage of the last call's launches, which HIP reads asynchronously
SPLIT_K = 12288
_TAGS = {torch.float16: "f16", torch.bfloat16: "bf16"}


def _functions(device):
    # per device: hipModuleLoadData loads the code object for the current device only
    if device.index not in _FNS:
        from .sk1_backend.sk1_loader import HipRuntime
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        path = os.path.join(_HERE, "fp8_rowquant.%s.hsaco" % arch)
        if not os.path.isfile(path):
            _FNS[device.index] = None
        else:
            rt = HipRuntime()
            with torch.cuda.device(device):
                mod = rt.load_module(path)
            _FNS[device.index] = (rt, {t: (rt.get_function(mod, "fp8_rowquant_" + t), rt.get_function(mod, "fp8_rowscale_" + t))
                                       for t in _TAGS.values()})
    return _FNS[device.index]


def linear(x, w8, w_scale, bias=None):
    if x.dtype not in _TAGS or w8.dtype != torch.float8_e4m3fn or not x.is_cuda or x.numel() == 0:
        return None
    N, K = w8.shape
    if x.shape[-1] != K or K % 16 or N % 16 or K > 32768 or w_scale.numel() != 1 or not w8.is_contiguous():
        return None
    if torch.is_grad_enabled() and (x.requires_grad or (bias is not None and bias.requires_grad)):
        return None
    fns = _functions(x.device)
    if fns is None:
        return None
    rt, f = fns
    f_quant, f_scale = f[_TAGS[x.dtype]]
    shape = x.shape
    x2 = x.reshape(-1, K)
    if x2.stride(1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
        x2 = x2.contiguous()
    M = x2.shape[0]
    parts = 2 if K >= SPLIT_K and K % 32 == 0 else 1   # each half must stay a multiple of 16 for _scaled_mm
    kp = K // parts
    x8 = torch.empty(parts, M, kp, dtype=torch.float8_e4m3fn, device=x.device)
    sx = torch.empty(M, dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    P, I = ctypes.c_void_p, ctypes.c_int
    a1 = [P(x2.data_ptr()), P(x8.data_ptr()), P(sx.data_ptr()), I(M), I(K), I(parts), ctypes.c_int64(x2.stride(0))]
    k1 = rt.launch(f_quant, (M, 1, 1), (256, 1, 1), 0, stream, a1)
    sa = torch.full((), 1 / 448, dtype=torch.float32, device=x.device)
    ws = w_scale.to(device=x.device, dtype=torch.float32)
    if parts == 1:
        y0 = torch._scaled_mm(x8[0], w8.t(), scale_a=sa, scale_b=ws, out_dtype=x.dtype)
        y1 = None
    else:
        y0 = torch._scaled_mm(x8[0], w8[:, :kp].contiguous().t(), scale_a=sa, scale_b=ws, out_dtype=x.dtype)
        y1 = torch._scaled_mm(x8[1], w8[:, kp:].contiguous().t(), scale_a=sa, scale_b=ws, out_dtype=x.dtype)
    b = None if bias is None else bias.to(dtype=x.dtype).contiguous()
    a2 = [P(y0.data_ptr()), P(0 if y1 is None else y1.data_ptr()), P(sx.data_ptr()), P(0 if b is None else b.data_ptr()),
          I(M), I(N)]
    k2 = rt.launch(f_scale, (M, (N + 2047) // 2048, 1), (256, 1, 1), 0, stream, a2)
    _KEEP[:] = (a1, a2, k1, k2)
    return y0.reshape(*shape[:-1], N)
