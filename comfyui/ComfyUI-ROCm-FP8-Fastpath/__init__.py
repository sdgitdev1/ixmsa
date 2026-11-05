"""ROCm FP8 fast-path patch for comfy-kitchen quantized models.

Problem this solves (AMD RDNA4 / gfx1201, ROCm torch, Windows):

1. comfy_kitchen before 0.2.25 gates its fp8 tensor-core fast path off on
   HIP, so every fp8 linear falls back to "dequantize whole weight + plain
   matmul"; 0.2.25 and later run their own HIP kernel. The LEGACY
   torch._scaled_mm (hipBLASLt) is faster than that kernel: 0.76-0.83x of
   its time at Krea 2's shapes, ~230 TFLOPS on an RX 9070 XT. So the fast
   path goes through it and the gate is re-enabled for per-tensor FP8
   layouts. hipBLASLt drops to ~110 TFLOPS at K = 16384, so calls with
   K >= 16384 keep comfy_kitchen's own kernel where there is one. (The new
   torch.nn.functional.scaled_mm API access-violates on ROCm Windows.)

2. F.linear on QuantizedTensor decomposes to aten.t + aten.mm. The t()
   handler produces a transposed (non-contiguous) fp8 view and
   QuantizedTensor.dequantize() then calls .contiguous() on it — an
   uncoalesced fp8 transpose-copy that costs ~250 ms per large layer.
   The checkpoint-configured "full_precision_matrix_mult" layers (e.g. 96
   layers in Krea 2 fp8_scaled) hit this on every step. We register
   stride-aware mm/addmm handlers that dequantize the contiguous base and
   let the GEMM consume the transposed view natively (hipBLAS handles
   column-major B). Same math, no copy: 283 ms -> 36 ms per layer.

3. Those full-precision fp8 layers still run dequantize + fp16 GEMM. With
   SageAttention-RDNA4 0.6 or later installed, they run as fp8 with one
   activation scale per token instead (sageattention.fp8_linear), whose
   error matches the checkpoint's other fp8 layers. On by default;
   COMFYUI_FP8_PERTOKEN=0 turns it off.

Per-token fp8 (3.) on Krea 2 Turbo fp8_scaled, 1672x1256, RX 9070 XT:
2.43-2.47 -> 2.12-2.14 s/it.

Only active when torch.version.hip is set; no-op on NVIDIA/CPU.
"""

import logging

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

log = logging.getLogger("rocm_fp8_fastpath")


def _apply():
    import torch

    if not getattr(torch.version, "hip", None):
        log.info("[ROCm-FP8-Fastpath] not a ROCm build, skipping")
        return

    try:
        import comfy_kitchen as ck
        import comfy_kitchen.tensor.fp8 as ckfp8
        from comfy_kitchen.tensor.base import QuantizedTensor, register_layout_op
    except ImportError as e:
        log.warning("[ROCm-FP8-Fastpath] comfy_kitchen not importable (%s), skipping", e)
        return

    # sanity: legacy _scaled_mm must actually work on this GPU
    try:
        a = (torch.randn(32, 32, device="cuda") / 8).to(torch.float8_e4m3fn)
        s = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        torch._scaled_mm(a, a.t(), scale_a=s, scale_b=s, out_dtype=torch.float16)
        torch.cuda.synchronize()
    except Exception as e:
        log.warning("[ROCm-FP8-Fastpath] legacy _scaled_mm probe failed (%s), skipping", e)
        return

    # 1) route comfy_kitchen's fp8 matmul through the WORKING legacy API
    #    (the new F.scaled_mm / _scaled_mm_v2 access-violates on ROCm Windows).
    #    hipBLASLt runs K = 16384 at ~110 TFLOPS against ~180 for comfy_kitchen's
    #    HIP kernel (0.2.25+), so calls with K >= 16384 stay on comfy_kitchen.
    import comfy_kitchen.scaled_mm_v2 as ckmm
    ck_hip_mm = ckfp8.scaled_mm_v2 if hasattr(ckmm, "_hip_fp8_gemm") else None

    def scaled_mm_legacy(mat_a, mat_b, scale_a=None, scale_b=None, bias=None, out_dtype=None, **kw):
        if ck_hip_mm is not None and mat_a.shape[1] >= 16384:
            return ck_hip_mm(mat_a, mat_b, scale_a=scale_a, scale_b=scale_b, bias=bias, out_dtype=out_dtype, **kw)
        return torch._scaled_mm(mat_a, mat_b, scale_a=scale_a, scale_b=scale_b, bias=bias, out_dtype=out_dtype)

    ckfp8.scaled_mm_v2 = scaled_mm_legacy

    # 2) re-enable the fast-matmul gate for per-tensor FP8 layouts only
    #    (mxfp8/nvfp4 stay gated — their kernels are genuinely unavailable here)
    ckfp8.TensorCoreFP8Layout.supports_fast_matmul = classmethod(lambda cls: True)

    # 3) stride-aware dequant fallback: avoid the fp8 transpose-copy
    def smart_dequant(qt):
        qd = qt._qdata
        scale = qt._params.scale
        od = qt._params.orig_dtype
        if qd.is_contiguous():
            return ck.dequantize_per_tensor_fp8(qd, scale, od)
        if qd.t().is_contiguous():
            return ck.dequantize_per_tensor_fp8(qd.t(), scale, od).t()
        return ck.dequantize_per_tensor_fp8(qd.contiguous(), scale, od)

    _orig_mm = ckfp8._handle_fp8_mm
    _orig_addmm = ckfp8._handle_fp8_addmm

    def better_mm(qt, args, kwargs):
        a, b = args[0], args[1]
        a_q = isinstance(a, QuantizedTensor)
        b_q = isinstance(b, QuantizedTensor)
        if a_q and b_q:
            return _orig_mm(qt, args, kwargs)  # fp8 x fp8 -> legacy _scaled_mm fast path
        if b_q:
            return torch.mm(a, smart_dequant(b))
        if a_q:
            return torch.mm(smart_dequant(a), b)
        return _orig_mm(qt, args, kwargs)

    def better_addmm(qt, args, kwargs):
        bias, a, b = args[0], args[1], args[2]
        a_q = isinstance(a, QuantizedTensor)
        b_q = isinstance(b, QuantizedTensor)
        if a_q and b_q:
            return _orig_addmm(qt, args, kwargs)
        if b_q:
            return torch.addmm(bias, a, smart_dequant(b))
        if a_q:
            return torch.addmm(bias, smart_dequant(a), b)
        return _orig_addmm(qt, args, kwargs)

    def better_linear(qt, args, kwargs):
        inp, weight = args[0], args[1]
        bias = args[2] if len(args) > 2 else None
        if isinstance(weight, QuantizedTensor) and not isinstance(inp, QuantizedTensor):
            return torch.nn.functional.linear(inp, smart_dequant(weight), bias)
        return ckfp8._handle_fp8_linear(qt, args, kwargs)

    for layout in (ckfp8.TensorCoreFP8Layout,):
        register_layout_op(torch.ops.aten.mm.default, layout)(better_mm)
        register_layout_op(torch.ops.aten.addmm.default, layout)(better_addmm)
        register_layout_op(torch.ops.aten.linear.default, layout)(better_linear)

    # 4) OPT-IN uniform-quant mode (set COMFYUI_FP8_UNIFORM_QUANT=1): fp8-scaled
    #    checkpoints mark some layers "full_precision_matrix_mult" (quality
    #    choice by the publisher; ~96 layers in Krea2-class models). Those run
    #    dequant+bf16 GEMM (~2x the cost of the fp8 fast path per layer). This
    #    override ignores that flag for fp8 formats so EVERY layer uses fp8
    #    hardware compute — int8-file-style uniform speed for fp8 files, at a
    #    small (usually invisible) quality tradeoff. Off by default.
    import os
    if os.environ.get("COMFYUI_FP8_UNIFORM_QUANT") == "1":
        import comfy.ops as _cops
        _orig_lqm = _cops._load_quantized_module

        def _uniform_lqm(module, super_load, state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs, load_extra_params=False):
            r = _orig_lqm(module, super_load, state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs, load_extra_params)
            fmt = getattr(module, "quant_format", None)
            if fmt in ("float8_e4m3fn", "float8_e5m2") \
                    and fmt not in getattr(module, "_disabled_formats", ()) \
                    and getattr(module, "_full_precision_mm", False):
                module._full_precision_mm = False
            return r

        _cops._load_quantized_module = _uniform_lqm
        log.info("[ROCm-FP8-Fastpath] UNIFORM-QUANT mode ON: full-precision-marked fp8 layers forced onto the fp8 fast path")

    # 5) per-token fp8 for the checkpoint's full-precision e4m3 layers in diffusion models (pick_operations; text
    #    encoders build their ops elsewhere and keep dequantize + GEMM). On by default, COMFYUI_FP8_PERTOKEN=0 turns
    #    it off; fp8_linear.linear returns None for calls it does not take, which keep ComfyUI's path.
    elif os.environ.get("COMFYUI_FP8_PERTOKEN", "1") != "0":
        try:
            from sageattention import fp8_linear
        except ImportError:
            fp8_linear = None
            log.info("[ROCm-FP8-Fastpath] per-token fp8 for full-precision layers needs SageAttention-RDNA4 0.6 or later")
        if fp8_linear is not None:
            import comfy.ops as _cops
            _orig_pick = _cops.pick_operations

            def _pertoken_pick(*args, **kwargs):
                ops = _orig_pick(*args, **kwargs)
                if issubclass(ops.Linear, _cops.MixedPrecisionOp):
                    _orig_fccw = ops.Linear.forward_comfy_cast_weights

                    def forward_comfy_cast_weights(self, input, compute_dtype=None, want_requant=False, weight_only_quant=False):
                        if weight_only_quant or not (self._full_precision_mm and self._full_precision_mm_config
                                                     and self.quant_format == "float8_e4m3fn"):
                            return _orig_fccw(self, input, compute_dtype, want_requant, weight_only_quant)
                        with _cops.CastBiasWeightContext(self, input, offloadable=True, compute_dtype=compute_dtype,
                                                         want_requant=want_requant) as (weight, bias):
                            if isinstance(weight, QuantizedTensor):
                                y = fp8_linear.linear(input, weight._qdata, weight._params.scale, bias)
                                if y is not None:
                                    return y
                                weight = weight.dequantize()
                            return self._forward(input, weight, bias)

                    ops.Linear.forward_comfy_cast_weights = forward_comfy_cast_weights
                return ops

            _cops.pick_operations = _pertoken_pick
            log.info("[ROCm-FP8-Fastpath] per-token fp8 ON for full-precision fp8 layers (COMFYUI_FP8_PERTOKEN=0 turns it off)")

    log.info("[ROCm-FP8-Fastpath] ACTIVE: fp8 fast matmul via legacy _scaled_mm + stride-aware dequant fallback")


try:
    _apply()
except Exception:
    log.exception("[ROCm-FP8-Fastpath] failed to apply, ComfyUI will run unpatched")
