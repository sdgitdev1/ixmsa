"""Tests for `sageattention.fp8_linear`, the per-token fp8 e4m3 linear (HIP quantize, `torch._scaled_mm`, HIP
rescale), against the same quantization computed in fp32, and its refusals. Skipped when there is no gfx1201 GPU or no
HIP runtime.
"""
import pytest
import torch

from sageattention import fp8_linear


def _have_gfx1201():
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    return arch.split(":", 1)[0] == "gfx1201"


pytestmark = pytest.mark.skipif(not _have_gfx1201(), reason="needs a gfx1201 GPU and a HIP runtime")

# Against the same per-row e4m3 quantization accumulated in fp32, the differences are summation order, the output
# rounding, and the hardware e4m3 conversion, which rounds about 1 in 15000 values lying a hair from a rounding tie the
# other way from torch's conversion; each such value moves its row's output by about 1e-3 at these K. Measured up to
# 1.9e-3 (fp16) and 3.8e-3 (bf16); a per-tensor instead of per-row scale would be about 2.6e-2.
MAX_REL = {torch.float16: 4e-3, torch.bfloat16: 6e-3}


def _rel(out, ref):
    out = out.float()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def _reference(x, w8, ws, bias):
    sx = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12)
    xq = (x.float() * (448.0 / sx)).clamp(-448, 448).to(torch.float8_e4m3fn).float() * (sx / 448.0)
    y = xq @ (w8.float() * ws).t()
    return y if bias is None else y + bias.float()


def _inputs(shape, N, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    K = shape[-1]
    x = torch.randn(shape, device="cuda", dtype=dtype, generator=g)
    w8 = (torch.randn(N, K, device="cuda", generator=g) * 0.05).to(torch.float8_e4m3fn)
    ws = torch.tensor(0.02, device="cuda")
    bias = torch.randn(N, device="cuda", dtype=dtype, generator=g)
    return x, w8, ws, bias


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,N", [((37, 256), 128), ((2, 19, 6144), 1536), ((300, 12288), 256), ((64, 12304), 64)])
@pytest.mark.parametrize("with_bias", [True, False])
def test_matches_reference(shape, N, with_bias, dtype):
    # (300, 12288) takes the two-GEMM K split; K = 12304 is long but not a multiple of 32, so it stays one GEMM
    x, w8, ws, bias = _inputs(shape, N, dtype)
    bias = bias if with_bias else None
    y = fp8_linear.linear(x, w8, ws, bias)
    assert y.shape == (*shape[:-1], N) and y.dtype == dtype
    assert _rel(y, _reference(x, w8, ws, bias)) < MAX_REL[dtype]


def test_strided_rows():
    x, w8, ws, bias = _inputs((50, 1040), 128, torch.float16)
    xs = x[:, :1024]
    y = fp8_linear.linear(xs, w8[:, :1024].contiguous(), ws, bias)
    assert _rel(y, _reference(xs, w8[:, :1024], ws, bias)) < MAX_REL[torch.float16]


def test_fp16_output_far_above_the_input_max_stays_finite():
    # y / max|x| is about 4000 here, so a GEMM output of y * 448 / max|x| would overflow fp16
    x = (torch.rand(64, 16384, device="cuda") * 0.01).half()
    w8 = torch.full((256, 16384), 0.5, device="cuda").to(torch.float8_e4m3fn)
    ws = torch.tensor(1.0, device="cuda")
    y = fp8_linear.linear(x, w8, ws)
    assert torch.isfinite(y).all()
    assert _rel(y, _reference(x, w8, ws, None)) < MAX_REL[torch.float16]


def test_refusals_return_none():
    x, w8, ws, _ = _inputs((16, 256), 128, torch.float16)
    assert fp8_linear.linear(x.float(), w8, ws) is None
    assert fp8_linear.linear(x[:, :248], w8[:, :248].contiguous(), ws) is None
    assert fp8_linear.linear(x, w8.to(torch.float16).to(torch.float8_e5m2), ws) is None
    assert fp8_linear.linear(x, w8, torch.full((128,), 0.02, device="cuda")) is None
    assert fp8_linear.linear(x, w8.t().contiguous().t(), ws) is None
    assert fp8_linear.linear(x[:0], w8, ws) is None
