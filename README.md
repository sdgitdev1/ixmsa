# SageAttention-RDNA4

Quantized attention for AMD Radeon RX 9070 / RX 9070 XT (RDNA4, `gfx1201`) on Windows, and on Linux
when built from source.

This is a drop-in build of the `sageattention` package. Underneath it is SageAttention 2.2 and the
gfx12 port from [SageAttention PR #368](https://github.com/thu-ml/SageAttention/pull/368). On top of
that sit HIP attention kernels that take over the common case: head_dim 128, fp16 or bf16, every
layout, causal or not. Since v0.6 both products run in 8-bit integers (Q·K in int8, the attention
weights times V in int8), which halves PR #368's error on real activations. Against PR #368's own
kernels they are **1.2–1.3× faster on non-causal attention and 1.4–2.2× faster on causal
attention**, in fp16 and bf16, measured over the full call. The quantization in front of the kernel
is a prebuilt HIP code object too, so nothing is compiled on first use.

ComfyUI users get it through the normal `--use-sage-attention` switch. No node and no code change
is needed.

ComfyUI also has its own int8 attention for this card, `--use-ck-attention` (comfy-kitchen). v0.6
is 1–2 % faster than it on contiguous inputs, up to 6 % faster on the strided NHD and fused-QKV
inputs most models pass, and more accurate on layers with extreme keys. It has a causal mode, which
comfy-kitchen does not (see "Compared with comfy-kitchen" below).

---

## Results

### Against PR #368, full call (quantization included)

Time of this package divided by time of PR #368 on the same tensors. Below 1.0 means faster.
B=1, H=48, D=128, contiguous HND. Each cell is the median of 15 interleaved repetitions in one
process (`bench/bench_attention.py`).

| N | fp16 non-causal | fp16 causal | bf16 non-causal | bf16 causal |
|---|---|---|---|---|
| 4096 | 0.78 | 0.71 | 0.79 | 0.68 |
| 8771 | 0.75 | 0.45 | 0.75 | 0.45 |
| 16384 | 0.84 | 0.53 | 0.84 | 0.53 |

MiniMax-H3's fused-QKV views (bf16, H=56): 0.78 / 0.79 / 0.82 at N = 27 280 / 31 248 / 61 488.
Against PyTorch SDPA the full call is 2.1–2.5× faster non-causal and 2.5–3.2× causal.

### Short calls

Each call timed on its own (synchronized, median of 40), fp16, B=1, H=48, D=128, non-causal, our
time divided by PR #368's. Single calls this short vary by about ±5 % from run to run.

| N | 130 | 288 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| v0.6 | 0.79 | 0.70 | 0.99 | 0.82 | 0.83 | 0.78 |
| v0.5 | 0.84 | 0.78 | 1.11 | 0.91 | 0.90 | 0.87 |

bf16 is within 5 % of these. Calls with fewer than 129 tokens keep v0.5's kernels.

Nothing is compiled at run time and the default path does not import Triton. In a fresh process in
ComfyUI's environment the first call takes 0.02 s (v0.5: 0.08 s; v0.4.0: 0.48 s; v0.3.0, which
compiled Triton kernels on first use: 1.9 s; PR #368's path: 0.24 s). There is no 32-bit limit on
sequence length (F162, F169).

All of these are ratios. On this card a software clock check can confirm that the clock did not
move during a block, but it cannot pin the absolute clock. So the absolute milliseconds are not
quoted here and the ratios are. [docs/METHOD.md](docs/METHOD.md) explains why that matters and
how each cell was measured.

### In a real render (ComfyUI's Krea2 model, 8 steps)

Measured with v0.4's kernels; v0.6 has not been timed in a render yet. Seconds per sampling step:
median of 9 renders (3 prompts × 3 seeds). Each attention backend was called exactly as stock
ComfyUI calls it, so every package used its own defaults.

fp16, a Krea2 int8 fine-tune with no LoRAs:

| | 1 MP | 2 MP (1448²) |
|---|---|---|
| PyTorch SDPA | 1.029 s | 2.387 s |
| SageAttention 1.x (community RDNA4 build) | 0.997 s | 2.167 s |
| **This package** | **0.973 s** | **2.059 s** |
| attention time per call (this / SDPA) | 5.0 / 9.9 ms | 14.1 / 29.6 ms |

These were taken with the fp8 kernel. Switching to the int8 default changed the step time by
−0.3 % at 1 MP and −0.05 % at 2 MP.

bf16, Krea2 Turbo (fp8-scaled weights, computing in bf16):

| | 1 MP | 2 MP |
|---|---|---|
| PyTorch SDPA | 1.156 s | 2.567 s |
| SageAttention 1.x (community RDNA4 build) | 1.113 s | 2.330 s |
| **This package** | **1.076 s** | **2.202 s** |

The hand-written kernels served every self-attention call in these renders. Most of a step is
spent outside attention, in the model's matmuls, which is why a 2× faster attention call becomes
a 5–14 % faster step.

### Accuracy

Q is quantized to int8 per token. K is quantized to int8 after its per-head mean is subtracted, with
one scale per token, or one per 16 keys on heads where that costs no key more than 2 bits. V is int8
with one scale per channel and one per 64-key tile, and the attention weights are unsigned int8
against each tile's own maximum. Both products run on the int8 matrix instructions with exact int32
accumulation. fp16 and bf16 use the same scheme.

Relative RMS error of the full call against fp64 attention, on 18 real captures from Krea2 and
Flux2-Klein (F174):

| | v0.6 | v0.5 | PR #368's path | comfy-kitchen |
|---|---|---|---|---|
| all 18, median, fp16 | 0.0072 | 0.0139 | 0.0126 | 0.0049 |
| all 18, median, bf16 | 0.0100 | 0.0226 | 0.0144 | 0.0081 |
| Krea2, 8 771 tokens, median of 8 blocks, fp16 | 0.0041 | 0.0113 | 0.0074 | 0.0040 |
| the same, bf16 | 0.0046 | 0.0157 | 0.0077 | 0.0045 |
| Krea2 block 0 (extreme keys), 3 captures, fp16 | 0.008–0.015 | 0.022–0.025 | 0.025–0.175 | 0.010–0.051 |

- **Against PR #368:** lower error on all 18 captures in fp16 (at most 0.70× of it) and on 17 of
  18 in bf16. The exception is a capture where exact attention, computed in bf16, is itself worse
  than PR #368's path: rounding the inputs to bf16 dominates there, and v0.6 matches exact bf16
  attention.
- **Against comfy-kitchen:** level on Krea2's 8 771-token layers, more accurate on its first block
  (1.1–3.5×), less accurate on the short captures of 288–1 056 tokens.
- **Causal:** 2.0–2.8× lower error than v0.5's causal kernel.
- **`smooth_k` is on by default** and should stay on. Subtracting the per-head key mean before
  quantizing is exact in real arithmetic, and it is what makes int8 K work on layers where most keys
  share one direction (F063).

In the Krea2 render tests of v0.4 the images from this package were closer to full-precision
attention than SageAttention 1.x's: median PSNR 18.5 vs 13.5 dB at 1 MP and 21.7 vs 15.6 dB at
2 MP in fp16. Take per-image numbers from an 8-step sampler with care, though. A change of 3·10⁻⁶
in the text conditioning alone moves the final latent by 4–58 %, so single-image differences are
mostly trajectory divergence, not precision.

### Compared with comfy-kitchen (`--use-ck-attention`)

ComfyUI ships comfy-kitchen, whose `int8_attention` is behind `--use-ck-attention`. It quantizes Q,
K and V to int8 and the attention weights to unsigned int8, with a Hadamard rotation of Q and K and
one K scale per 16 keys. v0.6 takes several of its ideas: the lazy output rescale, the 1.5·2²³
accumulator seed and its kernel shape (F174). It differs in how K is scaled, keeping one scale per
token on heads with outlier keys, which is where its accuracy advantage on Krea2's first block comes
from.

comfy-kitchen's time divided by this package's (above 1: this package is faster), on the same
tensors, interleaved in one process (comfy-kitchen 0.2.40, torch 2.13, B=1, H=48, D=128,
non-causal):

| | N = 4 096 | 8 771 | 16 384 |
|---|---|---|---|
| fp16, contiguous HND | 1.01 | 1.02 | 1.02 |
| bf16, contiguous HND | 1.01 | 1.01 | 1.02 |
| fp16, NHD as ComfyUI passes it | 1.00 | 1.06 | 1.04 |

MiniMax-H3's fused-QKV views, bf16, H=56: 1.04 / 1.04 / 1.03 at N = 27 280 / 31 248 / 61 488. For NHD input
ComfyUI hands comfy-kitchen transposed views and copies its HND output back into NHD, and that copy
is counted. comfy-kitchen has no causal mode. Error: see Accuracy above.

---

### In a video model (MiniMax-H3, w4a8 int4, 864x480, 124 frames)

Measured with v0.3's kernels.

H3 fuses q, k and v into one projection, so they arrive as views into a single tensor and `k`'s row
stride is `3*H*D`, not its own width. Earlier releases refused that layout, which meant the call
quietly ran PR #368's path even with `SAGEATTN_SK1_BF16=1` set. v0.3 serves it.

The margins are ratios, not milliseconds: this GPU's clock cannot be pinned (see F073), so every
timing block is ratio-only, and every ratio in this section is our time divided by PR #368's, so
below 1 means we are faster. End to end in the same workflow, the sampling phase takes **0.950** of
PR #368's time at 4 steps and **0.847** at 8 steps, which is 5 % and 15 % faster, same checkpoint and
seed. Loading dominates the wall clock, so the "Prompt executed" totals do not resolve this and are
not quoted as a speedup.
The sweep stops at N = 99 792 because that is where the launch's 32-bit addressing runs out, not
because the margin does.

## Install

The prebuilt wheels target exactly this stack:

- Windows 10/11, Python 3.12
- AMD Radeon RX 9070 or RX 9070 XT (`gfx1201`); the RX 9060 XT (`gfx1200`) is opt in, see below
- PyTorch `2.13.0+rocm10.0.0` or `2.14.0+rocm10.1.0` (the ROCm Python packages AMD publishes for
  gfx120X). There is one wheel for each; the one with `-214-` in its name is for 2.14

Download the wheel for your PyTorch from [Releases](../../releases) and install it into the Python
environment that runs your models. For ComfyUI that means ComfyUI's own venv. Close ComfyUI first.

```
python -m pip install --no-deps sageattention-2.2.0+amd.gfx12.5-cp312-cp312-win_amd64.whl       # PyTorch 2.13
python -m pip install --no-deps sageattention-2.2.0+amd.gfx12.5-214-cp312-cp312-win_amd64.whl   # PyTorch 2.14
```

`python -c "import torch; print(torch.__version__)"` tells you which PyTorch you have. Both wheels
compute bit-identical output; installed next to the wrong PyTorch, the import stops and names the
version it needs.

`--no-deps` keeps pip from touching your PyTorch install. The compiled extensions are built
against that exact PyTorch version. For any other version, build from source (see below).

### Linux

There is no prebuilt Linux wheel, but the kernels load on Linux too, thanks to
[boxwrench](https://github.com/boxwrench) ([#1](https://github.com/IxMxAMAR/SageAttention-RDNA4/pull/1)). It was tested on a Radeon AI PRO R9700, which is
also `gfx1201`, under Ubuntu 24.04 with PyTorch `2.9.1+rocm7.2.1`. Build from source with
`PYTORCH_ROCM_ARCH=gfx1201`; [docs/LINUX.md](docs/LINUX.md) has the steps. Since
v0.4.0 the build records the PyTorch version it was built against and the runtime check compares
with that, so a source build against your own PyTorch passes without any override.

## Use

```python
from sageattention import sageattn

out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)   # q, k, v: (B, H, N, D)
```

ComfyUI: start it with `--use-sage-attention`.

The hand-written kernels are used automatically when **all** of these hold. Anything else goes to
the gfx12 kernel from PR #368, with the same results contract:

- the GPU is `gfx1201` (or `gfx1200` with `SAGEATTN_SK1_GFX1200=1`), the HIP runtime resolves,
  and the code object loads;
- fp16 or bf16 inputs and head_dim 128;
- HND or NHD layout, contiguous or any view whose strides the kernel can address: the last axis
  contiguous, strides that are multiples of 8 elements, a 16-byte-aligned start. That covers the
  usual ComfyUI paths and the fused-QKV views MiniMax H3 passes;
- query length equal to key length, either fully causal or non-causal;
- no attention mask, `return_lse=False`, `smooth_v=False`.

A call tries the int8 P·V kernels first (129 tokens or more, a positive `sm_scale`). Then an fp16
call tries v0.5's int8 Q·K kernel, then the fp8 kernel, then PR #368's; a bf16 call tries v0.5's
bf16 kernel, then PR #368's.

| environment variable | effect |
|---|---|
| `SAGEATTN_SK1_BACKEND` unset | default: use the kernels where they apply, fall back silently elsewhere |
| `SAGEATTN_SK1_BACKEND=0` | never use them |
| `SAGEATTN_SK1_BACKEND=1` | strict: raise instead of falling back when the kernels cannot load or the GPU is not `gfx1201` |
| `SAGEATTN_SK1_I8PV` unset | default: the int8 P·V kernels first, v0.5's kernels for a call they do not take |
| `SAGEATTN_SK1_I8PV=0` | v0.5's kernels only, bit-identical to v0.5.0 |
| `SAGEATTN_SK1_I8PV=1` | strict: raise for a call the int8 P·V kernels cannot serve (head_dim 64, under 129 tokens, `sm_scale <= 0`) or when they fail to load |
| `SAGEATTN_SK1_INT8` unset | default: int8 Q·K for fp16 calls |
| `SAGEATTN_SK1_INT8=0` | fp8 Q·K for fp16 calls, bit-identical to v0.1.0 (this also skips the int8 P·V kernels for fp16) |
| `SAGEATTN_SK1_INT8=1` | strict: raise instead of falling back when the int8 kernel is missing or fails to load, or for a head_dim 64 call |
| `SAGEATTN_SK1_BF16` unset | default: bf16 calls use the hand-written kernel |
| `SAGEATTN_SK1_BF16=0` | bf16 calls go to PR #368's path, as in v0.1.0 |
| `SAGEATTN_SK1_BF16=1` | strict: raise instead of falling back when the bf16 kernel is missing or fails to load, or for a head_dim 64 call |
| `SAGEATTN_SK1_PROLOGUE` unset | default: quantize K and V with the prebuilt HIP prologue |
| `SAGEATTN_SK1_PROLOGUE=triton` | use v0.3.0's Triton prologue instead |
| `SAGEATTN_SK1_GFX1200=1` | allow the hand-written kernels on an RX 9060 XT (`gfx1200`). Off by default because nobody has run them on that card yet; the device always picks its own builds, so the switch does nothing on a `gfx1201` card |
| `SAGEATTN_CK=1` | serve non-causal head_dim 128 calls with comfy-kitchen's int8 attention, if comfy-kitchen is installed, for comparison. head_dim 64 and causal calls stay on this package's paths. Off by default |

The variables are read once, when the package is imported. `SAGEATTN_SK1_BACKEND=0` turns all of
them off, except `SAGEATTN_CK`. The same switches exist per call: `sageattn(..., sk1_backend=, sk1_i8pv=, sk1_int8=, sk1_bf16=)`; passing `sk1_backend=` also skips `SAGEATTN_CK`.

Strict mode is about loading the kernels, not about the list above. A call outside it (a mask,
`return_lse`, an unsupported stride pattern) still falls back silently with a switch set to `1`.
To see whether a call is served, call `sageattention.sk1_backend.try_sk1_t1(q, k, v,
tensor_layout=...)` directly: it returns `(None, reason)` for a call it refuses.

If the kernels cannot be used at all, for example on a different GPU, the package logs one line
per process and carries on with the fallback.

### fp8 linear layers

v0.6 also ships `sageattention.fp8_linear.linear(x, w8, w_scale, bias=None)`: `x @ (w8 * w_scale)^T`
for fp16 or bf16 `x` and an e4m3 weight with one scale, computed in fp8 with one activation scale per
token. It is meant for the layers fp8_scaled checkpoints mark `full_precision_matrix_mult` (96 in
Krea2 Turbo), which ComfyUI otherwise runs as dequantize + fp16 GEMM. At Krea2's shapes it takes
0.61–0.70× the time of that path, with the error of the checkpoint's other fp8 layers (F175); a Krea2
step at 1672×1256 goes from 2.38 s to 2.12 s. Neither `sageattn` nor ComfyUI calls it on its own: a
node or patch has to route those layers to it. A call it does not take (other dtypes, a per-channel
scale, K or N not a multiple of 16, K over 32 768, no code object for the GPU, or one autograd needs a
gradient through) returns `None`.

## Build from source

[docs/BUILD.md](docs/BUILD.md) covers the native extensions (`setup.py`, ROCm SDK,
`PYTORCH_ROCM_ARCH=gfx1201`) and rebuilding the kernels' code objects from `kernels/hip/` with
`tools/build_hsaco.py`.

## How this was built

- [docs/JOURNEY.md](docs/JOURNEY.md): the whole path, from a Triton kernel that lost to PR #368
  to hand-written kernels that beat it, including the dead ends.
- [docs/FINDINGS.md](docs/FINDINGS.md): every finding, confirmed or refuted, with its numbers.
- [docs/METHOD.md](docs/METHOD.md): how measurements were taken, and the mistakes that shaped
  those rules.

## Limitations

- The prebuilt kernels target `gfx1201`, plus `gfx1200` behind a switch. The `gfx1200` builds have
  the same machine code as the `gfx1201` ones but have not been run on that card yet
  ([docs/GFX1200-TESTING.md](docs/GFX1200-TESTING.md)). Other GPUs use the fallback path.
- Every speed and accuracy number here was measured on Windows. Linux results from two testers,
  with torch 2.9.1 and a torch 2.14 nightly, are in [docs/LINUX.md](docs/LINUX.md).
- head_dim 128 only. Two head_dim 64 kernels were built and tested, the second on the new prologue,
  and both were slower than PR #368's, so head_dim 64 still uses the fallback (F156, F165).
- Query and key lengths must match. Models that attend a query over a longer cached prefix (Qwen
  Image 2.1, for one) fall back to PR #368's path.
- comfy-kitchen is still more accurate on short sequences (a few hundred to a thousand tokens), and
  about 3 % faster on contiguous input at 4 096 tokens.
- The int8 P·V objects are built with ROCm 10.1's compiler, the older objects with ROCm 10.0's; both
  load under either PyTorch wheel.
- v0.6 has not been timed in a full render yet. The kernels are checked bit for bit across shapes
  and layouts, and against an fp64 reference; v0.5 and earlier ran in Krea2 (fp16 and bf16) and
  MiniMax-H3 (bf16).

## A note from me

I've been busy with my day job and haven't had much time for this. I think the kernel itself could
still improve, but the returns are marginal now, and most of what is left is Triton and the HIP
toolchain rather than the attention kernel. If I could, I'd ship this with my own libraries for both
of them -- but testing and modifying them takes far too long on one GPU.

I've built this with AI coding tools. They wrote most of the kernel code, ran the tests across
the whole matrix and summarized the findings. I chose what to try, read the results and decided
what shipped, and nothing went in without passing the bit-for-bit and timing checks in
docs/METHOD.md. The failures are written up next to the wins in docs/FINDINGS.md.

v0.5 measured this package against comfy-kitchen and found it behind on both speed and accuracy.
v0.6 is the answer: int8 for both products, comfy-kitchen's lazy rescale and kernel shape, and one
K scale per 16 keys only where it costs nothing. That made it level with comfy-kitchen on speed,
with half PR #368's error. Next up is head_dim 64, which needs a new design after two slower
attempts, and query lengths shorter than the key length.

## Credits and license

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).

- [SageAttention](https://github.com/thu-ml/SageAttention) by the THU-ML group: the
  quantized-attention method and the package this builds on.
- The gfx12 port in [PR #368](https://github.com/thu-ml/SageAttention/pull/368) by [jammm](https://github.com/jammm) and [0xDELUXA](https://github.com/0xDELUXA), which
  provides the native fallback kernels and the baseline this work is measured against.
- [boxwrench](https://github.com/boxwrench), for Linux support in the kernel loader ([#1](https://github.com/IxMxAMAR/SageAttention-RDNA4/pull/1)).
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) by Comfy Org (Apache-2.0), whose int8
  attention kernel showed the lazy output rescale, the 1.5·2²³ accumulator seed and the kernel shape
  that v0.6's kernels use.
