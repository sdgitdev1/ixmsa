"""Tests for the hand-written gfx1201 attention kernels, through the public `sageattn` API.

On gfx1201, head_dim 128 calls are served by the prebuilt kernels by default: first by the int8 Q·K +
int8 P·V kernels (fp16 and bf16, with their own HIP prologue), then, for a call those do not take, by the
int8 Q·K kernel (fp16) or the bf16 fp8 kernel, with K and V quantized by the prebuilt HIP prologue. These
tests check accuracy against fp32 SDPA, edge lengths, large V values, that NHD, HND and fused-QKV
inputs give the same bits, the prologue, the environment switches and the fallback. They are
skipped when there is no gfx1201 GPU or no HIP runtime.
"""
import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F


def _have_gfx1201():
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    return arch.split(":", 1)[0] == "gfx1201"


pytestmark = pytest.mark.skipif(not _have_gfx1201(), reason="needs a gfx1201 GPU and a HIP runtime")

# Relative RMS error of the quantized paths against fp32 SDPA on N(0,1) inputs is about 0.03 to 0.055.
MAX_REL_RMS = 0.08

HEAD_DIM = 128
DTYPES = [torch.float16, torch.bfloat16]


def _sageattn():
    from sageattention import sageattn
    return sageattn


def _rand_qkv(b, h, n, seed=0, layout="HND", dtype=torch.float16, head_dim=HEAD_DIM):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (b, h, n, head_dim) if layout == "HND" else (b, n, h, head_dim)
    return [torch.randn(shape, device="cuda", dtype=dtype, generator=g) for _ in range(3)]


def _rel_rms(out, ref):
    out = out.float()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def _reference(q, k, v, causal):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)


def _run_fresh(code, **env):
    # The SAGEATTN_* switches are read at import time, so these checks need a fresh interpreter.
    res = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=dict(os.environ, **env),
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return [l for l in res.stdout.splitlines() if l.startswith("RESULT ")][-1].split()[1:]


@pytest.mark.parametrize("dtype", DTYPES)
def test_hnd_matches_sdpa(dtype):
    q, k, v = _rand_qkv(1, 8, 512, dtype=dtype)
    out = _sageattn()(q, k, v, tensor_layout="HND")
    assert out.dtype == dtype and out.shape == q.shape
    assert torch.isfinite(out).all()
    assert _rel_rms(out, _reference(q, k, v, False)) < MAX_REL_RMS


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 15, 63, 64, 65, 127, 129, 300, 1000])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("smooth_k", [True, False])
def test_edge_lengths(n, causal, smooth_k, dtype):
    q, k, v = _rand_qkv(1, 2, n, seed=n, dtype=dtype)
    out = _sageattn()(q, k, v, tensor_layout="HND", is_causal=causal, smooth_k=smooth_k)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    assert _rel_rms(out, _reference(q, k, v, causal)) < MAX_REL_RMS


@pytest.mark.parametrize("dtype", DTYPES)
def test_large_v_outlier_stays_finite(dtype):
    # All queries are identical and one key is aligned with them, so that key gets almost all of
    # the softmax weight. Its V row has |V| = 20000, far above the e4m3 maximum of 448. A kernel
    # that folds p * amax_v into the fp8 conversion without a clamp overflows here.
    n, outlier = 1024, 300
    q, k, v = _rand_qkv(1, 2, n, seed=1, dtype=dtype)
    q[:] = q[:, :, :1]
    k[:, :, outlier] = 4 * q[:, :, 0]
    v[:, :, outlier] = 20000.0
    out = _sageattn()(q, k, v, tensor_layout="HND")
    assert (~torch.isfinite(out)).sum().item() == 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_nhd_equals_hnd_bitwise(causal, dtype):
    n = 300
    q, k, v = _rand_qkv(2, 4, n, seed=3, dtype=dtype)
    out_hnd = _sageattn()(q, k, v, tensor_layout="HND", is_causal=causal)
    qn, kn, vn = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    out_nhd = _sageattn()(qn, kn, vn, tensor_layout="NHD", is_causal=causal)
    assert out_nhd.shape == (2, n, 4, HEAD_DIM)
    assert torch.equal(out_nhd.transpose(1, 2), out_hnd)


def test_int8_is_the_fp16_default():
    # The default fp16 path computes Q·K in int8, so it differs from the fp8 kernel and should be at
    # least as close to the fp32 reference.
    q, k, v = _rand_qkv(1, 8, 1024, seed=5)
    ref = _reference(q, k, v, False)
    out_int8 = _sageattn()(q, k, v, tensor_layout="HND")
    out_fp8 = _sageattn()(q, k, v, tensor_layout="HND", sk1_int8=False)
    assert not torch.equal(out_int8, out_fp8)
    assert _rel_rms(out_int8, ref) <= 1.05 * _rel_rms(out_fp8, ref)


def test_int8_switch_off_gives_the_fp8_kernel():
    code = """
        import torch
        from sageattention import sageattn
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.float16, generator=g)
                   for _ in range(3)]
        print("RESULT", torch.equal(sageattn(q, k, v), sageattn(q, k, v, sk1_int8=False)))
    """
    assert _run_fresh(code, SAGEATTN_SK1_INT8="0") == ["True"]


@pytest.mark.parametrize("env", [{"SAGEATTN_SK1_BACKEND": "0"}, {"SAGEATTN_SK1_BF16": "0"}])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_switched_off_still_works(env, dtype):
    code = """
        import torch
        import torch.nn.functional as F
        from sageattention import sageattn
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.%s, generator=g)
                   for _ in range(3)]
        out = sageattn(q, k, v, tensor_layout="HND")
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        rel = ((out.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print("RESULT", bool(torch.isfinite(out).all()), rel)
    """ % dtype
    finite, rel = _run_fresh(code, **env)
    assert finite == "True"
    assert float(rel) < MAX_REL_RMS


@pytest.mark.parametrize("dtype", DTYPES)
def test_head_dim_64_falls_back(dtype):
    q, k, v = _rand_qkv(1, 4, 512, seed=7, dtype=dtype, head_dim=64)
    out = _sageattn()(q, k, v, tensor_layout="HND")
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    assert _rel_rms(out, _reference(q, k, v, False)) < MAX_REL_RMS


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("smooth_k", [True, False])
def test_fused_qkv_views_served_bitwise(dtype, smooth_k):
    # MiniMax-H3 style: one (N, 3*H*D) projection split into q, k and v, each viewed as (1, H, N, D),
    # so a row's stride is three heads wide. Served, and the same bits as the data made contiguous.
    from sageattention.sk1_backend import try_sk1_t1
    n, heads = 1000, 4
    g = torch.Generator(device="cuda").manual_seed(11)
    qkv = torch.randn(n, 3 * heads * HEAD_DIM, device="cuda", dtype=dtype, generator=g)
    q, k, v = (t.view(n, heads, HEAD_DIM).transpose(0, 1).unsqueeze(0)
               for t in qkv.split(heads * HEAD_DIM, dim=-1))
    out, reason = try_sk1_t1(q, k, v, tensor_layout="HND", smooth_k=smooth_k)
    assert out is not None, reason
    ref, _ = try_sk1_t1(q.contiguous(), k.contiguous(), v.contiguous(), tensor_layout="HND",
                        smooth_k=smooth_k)
    assert torch.equal(out, ref)


@pytest.mark.skipif(os.environ.get("SAGEATTN_SK1_PROLOGUE", "") not in ("", "hip"),
                    reason="SAGEATTN_SK1_PROLOGUE selects another prologue")
def test_hip_prologue_is_the_default():
    from sageattention import sk1_backend as sk1
    assert sk1.SK1_PROLOGUE == "hip"
    hp = sk1._hip_prologue()
    assert hp is not None
    before = hp.n_calls
    q, k, v = _rand_qkv(1, 8, 512, seed=21)
    _sageattn()(q, k, v, tensor_layout="HND", sk1_i8pv=False)
    assert hp.n_calls > before


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [128, 64])
def test_smooth_k_is_always_on(dtype, head_dim):
    # Stock ComfyUI passes smooth_k=False. Smoothing is exact, so it is applied anyway: the same bits
    # as smooth_k=True, on the kernels here and on PR #368's path (head_dim 64), and keys that share
    # one large direction stay accurate.
    q, k, v = _rand_qkv(1, 8, 1000, seed=5, dtype=dtype, head_dim=head_dim)
    k = k + 8 * torch.randn(1, 1, 1, head_dim, device="cuda", dtype=dtype)
    k_before = k.clone()
    off = _sageattn()(q, k, v, tensor_layout="HND", smooth_k=False)
    assert torch.equal(off, _sageattn()(q, k, v, tensor_layout="HND", smooth_k=True))
    assert torch.equal(k, k_before)
    assert _rel_rms(off, _reference(q, k, v, False)) < MAX_REL_RMS


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_triton_prologue_gives_the_same_bits_without_smooth_k(dtype):
    code = """
        import hashlib, torch
        from sageattention.sk1_backend import try_sk1_t1
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [torch.randn(1, 8, 777, 128, device="cuda", dtype=torch.%s, generator=g)
                   for _ in range(3)]
        out, reason = try_sk1_t1(q, k, v, tensor_layout="HND", smooth_k=False, i8pv=False)
        assert out is not None, reason
        print("RESULT", hashlib.sha256(out.float().cpu().numpy().tobytes()).hexdigest())
    """ % dtype
    assert _run_fresh(code, SAGEATTN_SK1_PROLOGUE="hip") == _run_fresh(code, SAGEATTN_SK1_PROLOGUE="triton")


def test_default_path_compiles_no_triton_kernel(tmp_path):
    # The HIP prologue needs no compiler at run time: a fresh process with an empty Triton cache
    # serves fp16 and bf16, with and without smooth_k, and leaves the cache empty.
    code = """
        import torch
        from sageattention import sageattn
        for dt in (torch.float16, torch.bfloat16):
            for sk in (True, False):
                g = torch.Generator(device="cuda").manual_seed(0)
                q, k, v = [torch.randn(1, 8, 512, 128, device="cuda", dtype=dt, generator=g)
                           for _ in range(3)]
                sageattn(q, k, v, tensor_layout="HND", smooth_k=sk)
        torch.cuda.synchronize()
        print("RESULT ok")
    """
    cache = tmp_path / "triton_cache"
    cache.mkdir()
    assert _run_fresh(code, TRITON_CACHE_DIR=str(cache), SAGEATTN_SK1_PROLOGUE="hip") == ["ok"]
    assert not any(cache.iterdir())


def test_gfx1200_switch_does_nothing_on_this_card():
    # The device picks its own builds; the switch only permits gfx1200, so on gfx1201 it is a no-op.
    code = """
        import hashlib, torch
        from sageattention import sageattn
        from sageattention.sk1_backend import try_sk1_t1
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.float16, generator=g)
                   for _ in range(3)]
        served = try_sk1_t1(q, k, v, tensor_layout="HND")[0] is not None
        out = sageattn(q, k, v, tensor_layout="HND")
        print("RESULT", served, hashlib.sha256(out.float().cpu().numpy().tobytes()).hexdigest())
    """
    on = _run_fresh(code, SAGEATTN_SK1_GFX1200="1")
    assert on[0] == "True"
    assert on == _run_fresh(code, SAGEATTN_SK1_GFX1200="0")


def test_default_path_does_not_import_triton():
    # Importing Triton costs ~0.3 s where its AMD backend finds `rocm-sdk` on PATH, so the default
    # (HIP prologue) path must not import it at all.
    code = """
        import sys, torch
        from sageattention import sageattn
        for dt in (torch.float16, torch.bfloat16):
            q, k, v = [torch.randn(1, 8, 512, 128, device="cuda", dtype=dt) for _ in range(3)]
            sageattn(q, k, v, tensor_layout="HND")
            sageattn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), tensor_layout="NHD")
        torch.cuda.synchronize()
        print("RESULT", "triton" in sys.modules)
    """
    assert _run_fresh(code, SAGEATTN_SK1_PROLOGUE="hip") == ["False"]


# ---- the int8 Q.K + int8 P.V objects (attn_i8pv_*), served first for head_dim 128 ----

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_i8pv_serves_the_default_call(causal, dtype):
    q, k, v = _rand_qkv(1, 8, 1000, seed=3, dtype=dtype)
    sa = _sageattn()
    out = sa(q, k, v, tensor_layout="HND", is_causal=causal)
    assert torch.equal(out, sa(q, k, v, tensor_layout="HND", is_causal=causal, sk1_i8pv=True))
    assert not torch.equal(out, sa(q, k, v, tensor_layout="HND", is_causal=causal, sk1_i8pv=False))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_i8pv_is_more_accurate_than_fp8_pv(causal, dtype):
    # half the heads get a few extreme keys, so both the group-K and the per-token-K object serve the call
    q, k, v = _rand_qkv(1, 8, 2048, seed=5, dtype=dtype)
    k[:, :4, ::97] *= 25
    ref = _reference(q, k, v, causal)
    sa = _sageattn()
    new = sa(q, k, v, tensor_layout="HND", is_causal=causal)
    old = sa(q, k, v, tensor_layout="HND", is_causal=causal, sk1_i8pv=False)
    assert torch.isfinite(new).all()
    assert _rel_rms(new, ref) < 0.8 * _rel_rms(old, ref)


@pytest.mark.parametrize("n, scale, head_dim", [(100, None, 128), (1000, -0.05, 128), (1000, None, 64)])
def test_i8pv_refusals_fall_back(n, scale, head_dim):
    # fewer than 135 tokens, a negative scale and head_dim 64 go to the previous objects
    q, k, v = _rand_qkv(1, 4, n, seed=7, head_dim=head_dim)
    sa = _sageattn()
    out = sa(q, k, v, tensor_layout="HND", sm_scale=scale)
    assert torch.equal(out, sa(q, k, v, tensor_layout="HND", sm_scale=scale, sk1_i8pv=False))


def test_i8pv_strict_raises_on_a_refusal():
    q, k, v = _rand_qkv(1, 4, 100, seed=7)
    with pytest.raises(RuntimeError):
        _sageattn()(q, k, v, tensor_layout="HND", sk1_i8pv=True)


def test_i8pv_switch_off():
    code = """
        import torch
        from sageattention import sageattn
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.float16, generator=g)
                   for _ in range(3)]
        print("RESULT", torch.equal(sageattn(q, k, v), sageattn(q, k, v, sk1_i8pv=False)))
    """
    assert _run_fresh(code, SAGEATTN_SK1_I8PV="0") == ["True"]


@pytest.mark.parametrize("head_dim, int8", [(64, None), (128, False)])
def test_i8pv_strict_raises_where_it_cannot_serve(head_dim, int8):
    # head_dim 64, and an fp16 call that asked for fp8 Q.K, are not served by the int8 P.V objects
    q, k, v = _rand_qkv(1, 4, 512, seed=7, head_dim=head_dim)
    with pytest.raises(RuntimeError):
        _sageattn()(q, k, v, tensor_layout="HND", sk1_i8pv=True, sk1_int8=int8)


def test_i8pv_failed_load_keeps_no_inputs_alive(monkeypatch):
    import gc
    import weakref
    from sageattention import sk1_backend as sk1
    from sageattention.sk1_backend import i8pv

    def refuse(arch):
        raise RuntimeError("no object")
    monkeypatch.setattr(i8pv, "I8pv", refuse)
    monkeypatch.setattr(sk1, "_I8PV", {})
    refs = []
    for _ in range(3):
        q = torch.zeros(1, 2, 256, 128, dtype=torch.float16)
        refs.append(weakref.ref(q))
        assert sk1._try_i8pv(q, q, q, "HND", True, False, 0.1, True, "gfx1201", False)[0] is None
        del q
    gc.collect()
    assert all(r() is None for r in refs)
    assert isinstance(sk1._I8PV["gfx1201"], str)
