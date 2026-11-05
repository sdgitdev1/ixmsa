"""Tests for the ComfyUI node in `comfyui/ComfyUI-ROCm-FP8-Fastpath`, loaded by path against stand-ins for `comfy.ops`
and `comfy_kitchen` that copy the signatures the node patches (ComfyUI at the time of v0.6, comfy-kitchen 0.2.41).
`torch._scaled_mm` and comfy-kitchen's HIP kernel are replaced by recorders, so the tests check routing, not math.
Skipped without a HIP build and a GPU: the node does nothing there.
"""
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import torch

NODE = Path(__file__).resolve().parents[1] / "comfyui" / "ComfyUI-ROCm-FP8-Fastpath" / "__init__.py"

pytestmark = pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(),
                                reason="the node only patches ROCm builds")


def _fake_comfy_ops(new_signature):
    ops = types.ModuleType("comfy.ops")

    class CastBiasWeightContext:
        def __init__(self, *args, **kwargs):
            self.slf = args[0] if len(args) else None

        def __enter__(self):
            return (None, None) if self.slf is None else (self.slf.weight, self.slf.bias)

        def __exit__(self, *_args):
            return None

    class MixedPrecisionOp:
        pass

    def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[]):
        class MixedPrecisionOps:
            class Linear(torch.nn.Module, MixedPrecisionOp):
                def _forward(self, input, weight, bias):
                    return torch.nn.functional.linear(input, weight, bias)

                if new_signature:
                    def forward_comfy_cast_weights(self, input, compute_dtype=None, want_requant=False, weight_only_quant=False):
                        return "original"
                else:
                    def forward_comfy_cast_weights(self, input):
                        return "original"
        return MixedPrecisionOps

    def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False, fp8_optimizations=False, model_config=None):
        return mixed_precision_ops()

    ops.CastBiasWeightContext = CastBiasWeightContext
    ops.MixedPrecisionOp = MixedPrecisionOp
    ops.mixed_precision_ops = mixed_precision_ops
    ops.pick_operations = pick_operations
    return ops


def _install(monkeypatch, new_signature=True, ck_hip_kernel=True):
    calls = []
    ck = types.ModuleType("comfy_kitchen")
    ck.dequantize_per_tensor_fp8 = lambda qdata, scale, orig_dtype: qdata.to(orig_dtype) * scale
    tensor = types.ModuleType("comfy_kitchen.tensor")
    base = types.ModuleType("comfy_kitchen.tensor.base")
    fp8 = types.ModuleType("comfy_kitchen.tensor.fp8")
    mm = types.ModuleType("comfy_kitchen.scaled_mm_v2")

    class QuantizedTensor:
        pass

    table = {}

    def register_layout_op(torch_op, layout_cls):
        def decorator(handler_func):
            table.setdefault(torch_op, {})[layout_cls] = handler_func
            return handler_func
        return decorator

    class TensorCoreFP8Layout:
        @classmethod
        def supports_fast_matmul(cls):
            return False

    def ck_kernel(input, weight, scale_a, scale_b, bias=None, out_dtype=None, **kw):
        calls.append(("comfy_kitchen", input.shape[1]))
        return "comfy_kitchen"

    def scaled_mm(mat_a, mat_b, scale_a=None, scale_b=None, bias=None, out_dtype=None, **kw):
        calls.append(("hipblaslt", mat_a.shape[1]))
        return torch.zeros(mat_a.shape[0], mat_b.shape[1], dtype=out_dtype or torch.float16, device=mat_a.device)

    base.QuantizedTensor, base.register_layout_op = QuantizedTensor, register_layout_op
    fp8.TensorCoreFP8Layout, fp8.scaled_mm_v2 = TensorCoreFP8Layout, ck_kernel
    fp8._handle_fp8_mm = fp8._handle_fp8_addmm = fp8._handle_fp8_linear = lambda qt, args, kwargs: None
    if ck_hip_kernel:
        mm._hip_fp8_gemm = lambda *a, **k: None
    ck.tensor, tensor.base, tensor.fp8, ck.scaled_mm_v2 = tensor, base, fp8, mm
    comfy = types.ModuleType("comfy")
    comfy.ops = _fake_comfy_ops(new_signature)
    for name, mod in (("comfy_kitchen", ck), ("comfy_kitchen.tensor", tensor), ("comfy_kitchen.tensor.base", base),
                      ("comfy_kitchen.tensor.fp8", fp8), ("comfy_kitchen.scaled_mm_v2", mm), ("comfy", comfy),
                      ("comfy.ops", comfy.ops)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)
    monkeypatch.delenv("COMFYUI_FP8_UNIFORM_QUANT", raising=False)
    monkeypatch.delenv("COMFYUI_FP8_PERTOKEN", raising=False)
    return comfy.ops, fp8, calls


def _load(caplog):
    spec = importlib.util.spec_from_file_location("rocm_fp8_fastpath_under_test", NODE)
    with caplog.at_level(logging.INFO, logger="rocm_fp8_fastpath"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert "failed to apply" not in caplog.text


def test_active_line_lists_parts(monkeypatch, caplog):
    ops, _, _ = _install(monkeypatch)
    original = ops.pick_operations
    _load(caplog)
    active = [r.getMessage() for r in caplog.records if "ACTIVE:" in r.getMessage()]
    assert len(active) == 1
    for part in ("hipblaslt-gemm", "stride-aware-dequant", "per-token-fp8"):
        assert part in active[0]
    assert ops.pick_operations is not original


def test_per_token_skipped_when_signature_changed(monkeypatch, caplog):
    ops, _, _ = _install(monkeypatch, new_signature=False)
    original = ops.pick_operations
    _load(caplog)
    assert ops.pick_operations is original
    assert "per-token-fp8 skipped" in caplog.text
    active = [r.getMessage() for r in caplog.records if "ACTIVE:" in r.getMessage()]
    assert len(active) == 1 and "per-token-fp8" not in active[0]


def test_k16384_stays_on_comfy_kitchen(monkeypatch, caplog):
    _, fp8, calls = _install(monkeypatch)
    _load(caplog)
    s = torch.ones((), device="cuda")
    calls.clear()
    for k in (6144, 14336, 16384, 24576):
        a = torch.zeros(4, k, device="cuda").to(torch.float8_e4m3fn)
        fp8.scaled_mm_v2(a, torch.zeros(8, k, device="cuda").to(torch.float8_e4m3fn).t(), scale_a=s, scale_b=s,
                         out_dtype=torch.bfloat16)
    assert calls == [("hipblaslt", 6144), ("hipblaslt", 14336), ("comfy_kitchen", 16384), ("comfy_kitchen", 24576)]


def test_without_comfy_kitchen_hip_kernel_every_call_uses_hipblaslt(monkeypatch, caplog):
    _, fp8, calls = _install(monkeypatch, ck_hip_kernel=False)
    _load(caplog)
    s = torch.ones((), device="cuda")
    calls.clear()
    a = torch.zeros(4, 16384, device="cuda").to(torch.float8_e4m3fn)
    fp8.scaled_mm_v2(a, torch.zeros(8, 16384, device="cuda").to(torch.float8_e4m3fn).t(), scale_a=s, scale_b=s,
                     out_dtype=torch.bfloat16)
    assert calls == [("hipblaslt", 16384)]
