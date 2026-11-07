# SageAttention-RDNA4

Quantized attention for AMD Radeon RX 9070 / RX 9070 XT (RDNA4, `gfx1201`) on Windows, and on Linux
when built from source. It is a drop-in build of the `sageattention` package: ComfyUI users get it
through the usual `--use-sage-attention` switch, with no node and no code change.

> **v0.6 is as fast as ComfyUI's own int8 attention, comfy-kitchen (`--use-ck-attention`), and more
> accurate.** On contiguous input the two are level; on the strided layout ComfyUI passes from 8.8k
> tokens this package is 4–5 % faster, on MiniMax-H3's fused QKV 2–3 %, and through ComfyUI's own
> attention functions on Krea2's calls 2 %. Its median error on 18 real captures is 0.0043 against
> comfy-kitchen's 0.0049, and 1.3–3.7× lower on Krea2's first block. It also has a causal mode, which
> comfy-kitchen does not.

Against the gfx12 port in [SageAttention PR #368](https://github.com/thu-ml/SageAttention/pull/368),
which this package builds on and falls back to, it is 1.2–1.4× faster on non-causal attention and
1.5–2.3× faster on causal attention, in fp16 and bf16, measured over the full call, with a third of
its error in fp16 and about half in bf16. Against PyTorch SDPA it is 2.2–2.4× faster non-causal and
2.6–3.2× faster causal. Calls too short to gain from int8 are handed to SDPA.

## What's new in v0.6

- **int8 for both products.** Q·K and the attention weights times V both run on the int8 matrix
  instructions with exact int32 accumulation, in kernels shaped after comfy-kitchen's.
- **K scaled per tile.** One K scale per 16 keys on 64-key tiles where that costs no key more than
  2 bits, one per token on the tiles with outlier keys.
- **A Hadamard rotation of Q and K** on every head without one dominant K channel. It spreads K's
  outliers over all 128 dimensions before int8 and cut the median error from 0.0072 to 0.0043.
- **Queries and keys may differ in length** (non-causal), as when Qwen Image 2.1 attends over its
  cached text prefix: its sampler runs 12 % faster.
- **Short calls go to PyTorch's SDPA**, which is faster than any int8 path below 2³¹ multiply-adds
  per call, and exact. SDXL's cross-attention, for one, takes 0.33–0.64 of the time it did.
- **Causal attention** has 2.7–3.6× lower error than v0.5's kernel and takes 0.89–0.91 of its time.
- **bf16 uses int8 Q·K too**, so fp16 and bf16 get the same scheme.
- **A ComfyUI node for fp8_scaled checkpoints**, [below](#comfyui-node-for-fp8-checkpoints).
- **`sageattention.fp8_linear`**, an fp8 linear layer with one activation scale per token, for the
  layers fp8_scaled checkpoints keep at full precision ([below](#fp8-linear-layers)).
- `SAGEATTN_CK` is gone: ComfyUI's own `--use-ck-attention` does the same thing.

---

## Results

All timings are ratios measured in one process with the variants interleaved. On this card a
software clock check can confirm that the clock did not move during a block, but it cannot pin the
absolute clock, so absolute milliseconds are not quoted. [docs/METHOD.md](docs/METHOD.md) explains
why and how each cell was measured.

### Against comfy-kitchen (`--use-ck-attention`)

comfy-kitchen's time divided by this package's: **above 1 means this package is faster.** Same
tensors, comfy-kitchen 0.2.41, torch 2.13, B=1, H=48, D=128, non-causal; A/A twins within 0.5 %.

| | N = 4 096 | 8 771 | 16 384 |
|---|---|---|---|
| fp16, contiguous HND | 1.002 | 1.009 | 1.013 |
| bf16, contiguous HND | 1.005 | 1.007 | 1.006 |
| fp16, NHD as ComfyUI passes it | 0.998 | 1.046 | 1.038 |
| MiniMax-H3 fused QKV, bf16, H=56, at N = 27 280 / 31 248 / 61 488 | 1.032 | 1.030 | 1.023 |

For NHD input ComfyUI hands comfy-kitchen transposed views and copies its HND output back into NHD,
and that copy is counted, because it is part of what `--use-ck-attention` costs. Through ComfyUI's
own attention functions on Krea2's calls (8 776 tokens, NHD) the ratio is 1.024 in fp16 and bf16.
On Krea2's real 8 771-token activations the two are level (0.997–1.008); on 1 056-token calls this
package is 1–7 % faster, and on 288-token calls, which it hands to SDPA, 2.2–2.4× faster.
comfy-kitchen has no causal mode.

In a full Krea2 render the two take the same time per step ([below](#in-a-real-render)).

### Against PR #368 and PyTorch SDPA

Our time divided by PR #368's on the same tensors, full call including quantization. Below 1.0
means faster. B=1, H=48, D=128, contiguous HND, median of 15 interleaved repetitions
(`bench/bench_attention.py`).

| N | fp16 non-causal | fp16 causal | bf16 non-causal | bf16 causal |
|---|---|---|---|---|
| 4096 | 0.78 | 0.68 | 0.77 | 0.65 |
| 8771 | 0.73 | 0.44 | 0.74 | 0.43 |
| 16384 | 0.85 | 0.52 | 0.85 | 0.52 |

MiniMax-H3's fused-QKV views (bf16, H=56): 0.79 / 0.80 / 0.82 at N = 27 280 / 31 248 / 61 488.
Against PyTorch SDPA the full call is 2.2–2.4× faster non-causal, 2.6–3.2× causal and 2.4–2.5× on
H3's views.

### In a real render

Krea2 Turbo (fp8_scaled), 1672×1256, 8 steps, with the [node below](#comfyui-node-for-fp8-checkpoints);
seconds per sampling step from ComfyUI's progress bar, two renders of each, interleaved:

| | s per step |
|---|---|
| this package (`--use-sage-attention`) | 2.03 / 2.03 |
| comfy-kitchen (`--use-ck-attention`) | 2.03 / 2.02 |
| this package, the node's per-token fp8 layers off | 2.29 / 2.27 |

Attention is well under half of a step here, so per-call differences of 1 % or less do not show:
the two attentions take the same time. The node's per-token fp8 layers save 11 % of each step.

### Short calls and the first call

Each call timed on its own (synchronized, median of 40), fp16, B=1, H=48, D=128, non-causal, time
divided by PR #368's. Calls below 2³¹ multiply-adds (B·H·Nq·Nk·D, here up to 512 tokens) run
PyTorch's SDPA, which is faster there than any int8 path, and exact. Single calls this short vary by
about ±5 % from run to run (an A/A twin of v0.6 read 0.30 / 0.35 / 0.64 / 0.79 / 0.78 / 0.76).

| N | 130 | 288 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| **v0.6** | **0.29** | **0.34** | **0.63** | **0.78** | **0.77** | **0.76** |
| v0.6's int8 kernels (`SAGEATTN_SDPA_SHORT=0`) | 0.63 | 0.65 | 0.83 | 0.77 | 0.77 | 0.76 |
| comfy-kitchen | 0.66 | 0.70 | 0.91 | 0.85 | 0.83 | 0.77 |
| PyTorch SDPA | 0.31 | 0.35 | 0.64 | 0.87 | 1.27 | 1.63 |

head_dim 64 calls have no int8 kernel here, so the short ones gain the most: on NHD calls as
ComfyUI passes them, SDXL's cross-attention to 77 text tokens takes 0.33–0.64 of PR #368's time
and a 1 024-token self-attention 0.78–0.81.

Nothing is compiled at run time and the default path does not import Triton. In a fresh process in
ComfyUI's environment the first call takes 0.02 s (v0.5: 0.08 s; v0.3.0, which compiled Triton
kernels on first use: 1.9 s; PR #368's path: 0.24 s). There is no 32-bit limit on sequence length.

### Accuracy

Q is quantized to int8 per token. K is quantized to int8 after its per-head mean is subtracted, with
one scale per token, or one per 16 keys on 64-key tiles where that costs no key more than 2 bits. On
heads without one dominant K channel, Q and K are first rotated by the same 128-point Hadamard
transform, which leaves Q·K unchanged and spreads K's outliers. V is int8 with one scale per channel
and one per 64-key tile, and the attention weights are unsigned int8 against each tile's own maximum.

Relative RMS error of the full call against fp64 attention on 18 real captures from Krea2 and
Flux2-Klein (lower is better). These are the int8 kernels' errors (`SAGEATTN_SDPA_SHORT=0`); by
default the six 288-token captures run SDPA and come out exact.

| | v0.6 | v0.5 | PR #368's path | comfy-kitchen |
|---|---|---|---|---|
| all 18, median, fp16 | **0.0043** | 0.0139 | 0.0126 | 0.0049 |
| all 18, median, bf16 | **0.0078** | 0.0226 | 0.0144 | 0.0081 |
| Krea2, 8 771 tokens, median of 8 blocks, fp16 | **0.0034** | 0.0113 | 0.0074 | 0.0040 |
| the same, bf16 | **0.0039** | 0.0157 | 0.0077 | 0.0045 |
| Krea2 block 0 (extreme keys), 3 captures, fp16 | **0.007–0.014** | 0.022–0.025 | 0.025–0.175 | 0.010–0.051 |

- **Against PR #368:** lower error on all 18 captures in fp16 (at most 0.58× of it) and on 17 of
  18 in bf16. The exception is a capture where exact attention computed in bf16 is itself worse
  than PR #368's path: rounding the inputs to bf16 dominates there, and v0.6 matches exact bf16
  attention.
- **Against comfy-kitchen:** lower error on 17 of the 18 captures in fp16 and 16 in bf16, and 1.3–3.7×
  lower on Krea2's first block.
- **Causal:** 2.7–2.8× lower error than v0.5's causal kernel in fp16, 3.5–3.6× in bf16.
- **K is always smoothed.** Stock ComfyUI passes `smooth_k=False`, which is ignored on gfx12.
  Subtracting the per-head key mean before quantizing is exact in real arithmetic, and it is what
  makes int8 K work on layers where most keys share one direction.

A note on comparing images: in an 8-step sampler a change of 3·10⁻⁶ in the text conditioning alone
moves the final latent by 4–58 %, and the same run repeated can land on a different trajectory. So
per-image differences are mostly trajectory divergence, not precision; the per-call errors above
are the measure of precision.

### In a video model (MiniMax-H3, w4a8 int4, 864x480, 124 frames)

Measured with v0.3's kernels; v0.6's are 1.22–1.26× PR #368's speed on these views, against
v0.3's 1.09–1.14×. H3 fuses q, k and v into one projection, so they arrive as views into a single
tensor whose rows are `3*H*D` apart; the kernels read that layout in place. End to end in the same
workflow, the sampling phase took **0.950** of PR #368's time at 4 steps and **0.847** at 8 steps,
same checkpoint and seed.

---

## Install

The prebuilt wheels target exactly this stack:

- Windows 10/11, Python 3.12
- AMD Radeon RX 9070 or RX 9070 XT (`gfx1201`); the RX 9060 XT (`gfx1200`) is opt in, see below
- PyTorch `2.13.0+rocm10.0.0` or `2.14.0+rocm10.1.0` (the ROCm Python packages AMD publishes for
  gfx120X). There is one wheel for each; the one with `-214-` in its name is for 2.14

Download the wheel for your PyTorch from [Releases](../../releases) and install it into the Python
environment that runs your models. For ComfyUI that means ComfyUI's own venv. Close ComfyUI first.

```
python -m pip install --no-deps sageattention-2.2.0+amd.gfx12.6-cp312-cp312-win_amd64.whl       # PyTorch 2.13
python -m pip install --no-deps sageattention-2.2.0+amd.gfx12.6-214-cp312-cp312-win_amd64.whl   # PyTorch 2.14
```

`python -c "import torch; print(torch.__version__)"` tells you which PyTorch you have. The kernels
in both wheels compute bit-identical output; calls short enough to run SDPA use PyTorch's own
kernel, which can differ between the two PyTorch versions. Installed next to the wrong PyTorch, the
import stops and names the version it needs. `--no-deps` keeps pip from touching your PyTorch
install.

### Linux

There is no prebuilt Linux wheel, but the kernels load on Linux too, thanks to
[boxwrench](https://github.com/boxwrench) ([#1](https://github.com/IxMxAMAR/SageAttention-RDNA4/pull/1)).
It was tested on a Radeon AI PRO R9700, which is also `gfx1201`, under Ubuntu 24.04 with PyTorch
`2.9.1+rocm7.2.1`. Build from source with `PYTORCH_ROCM_ARCH=gfx1201`;
[docs/LINUX.md](docs/LINUX.md) has the steps. The build records the PyTorch version it was built
against, so a source build against your own PyTorch passes the runtime check without any override.

## Use

```python
from sageattention import sageattn

out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)   # q, k, v: (B, H, N, D)
```

ComfyUI: start it with `--use-sage-attention`.

A call below 2³¹ multiply-adds (B·H·Nq·Nk·D) without `return_lse` runs PyTorch's SDPA. The
hand-written kernels take a larger call when **all** of these hold; anything else goes to the gfx12
kernel from PR #368, with the same results contract:

- the GPU is `gfx1201` (or `gfx1200` with `SAGEATTN_SK1_GFX1200=1`), the HIP runtime resolves,
  and the code object loads;
- fp16 or bf16 inputs and head_dim 128;
- HND or NHD layout, contiguous or any view whose strides the kernel can address: the last axis
  contiguous, strides that are multiples of 8 elements, a 16-byte-aligned start. That covers the
  usual ComfyUI paths and the fused-QKV views MiniMax H3 passes;
- query length equal to key length, causal or not; or, for the int8 P·V kernels, a non-causal call
  with any query length against at least 129 keys;
- no attention mask, `return_lse=False`, `smooth_v=False`.

A call tries the int8 P·V kernels first (129 keys or more, a positive `sm_scale`). Then an fp16
call tries v0.5's int8 Q·K kernel, then the fp8 kernel, then PR #368's; a bf16 call tries v0.5's
bf16 kernel, then PR #368's.

| environment variable | effect |
|---|---|
| `SAGEATTN_SDPA_SHORT` unset | default: calls below 2³¹ multiply-adds run PyTorch's SDPA |
| `SAGEATTN_SDPA_SHORT=0` | keep them on the int8 paths |
| `SAGEATTN_SK1_BACKEND` unset | default: use the kernels where they apply, fall back silently elsewhere |
| `SAGEATTN_SK1_BACKEND=0` | never use them |
| `SAGEATTN_SK1_BACKEND=1` | strict: raise instead of falling back when the kernels cannot load or the GPU is not `gfx1201` |
| `SAGEATTN_SK1_I8PV` unset | default: the int8 P·V kernels first, v0.5's kernels for a call they do not take |
| `SAGEATTN_SK1_I8PV=0` | v0.5's kernels only, bit-identical to v0.5.0 for calls with `smooth_k` on |
| `SAGEATTN_SK1_I8PV=1` | strict: raise for a call the int8 P·V kernels cannot serve (head_dim 64, under 129 keys, `sm_scale <= 0`) or when they fail to load |
| `SAGEATTN_SK1_INT8` unset | default: int8 Q·K for fp16 calls |
| `SAGEATTN_SK1_INT8=0` | fp8 Q·K for fp16 calls, bit-identical to v0.1.0 for calls with `smooth_k` on (this also skips the int8 P·V kernels for fp16) |
| `SAGEATTN_SK1_INT8=1` | strict: raise instead of falling back when the int8 kernel is missing or fails to load, or for a head_dim 64 call |
| `SAGEATTN_SK1_BF16` unset | default: bf16 calls use the hand-written kernel |
| `SAGEATTN_SK1_BF16=0` | bf16 calls go to PR #368's path, as in v0.1.0 |
| `SAGEATTN_SK1_BF16=1` | strict: raise instead of falling back when the bf16 kernel is missing or fails to load, or for a head_dim 64 call |
| `SAGEATTN_SK1_PROLOGUE` unset | default: quantize K and V with the prebuilt HIP prologue |
| `SAGEATTN_SK1_PROLOGUE=triton` | use v0.3.0's Triton prologue instead |
| `SAGEATTN_SK1_GFX1200=1` | allow the hand-written kernels on an RX 9060 XT (`gfx1200`). Off by default because nobody has run them on that card yet; the device always picks its own builds, so the switch does nothing on a `gfx1201` card |

The variables are read once, when the package is imported. `SAGEATTN_SK1_BACKEND=0` turns all the
`SAGEATTN_SK1_*` ones off; short calls still run SDPA unless `SAGEATTN_SDPA_SHORT=0`, and the
bit-identical rows above hold for short calls only with it. The kernel switches also exist per call:
`sageattn(..., sk1_backend=, sk1_i8pv=, sk1_int8=, sk1_bf16=)`.

Strict mode is about loading the kernels, not about the list above. A call outside it (a mask,
`return_lse`, an unsupported stride pattern) still falls back silently with a switch set to `1`.
To see whether a call is served, call `sageattention.sk1_backend.try_sk1_t1(q, k, v,
tensor_layout=...)` directly: it returns `(None, reason)` for a call it refuses.

If the kernels cannot be used at all, for example on a different GPU, the package logs one line
per process and carries on with the fallback.

### ComfyUI node for fp8 checkpoints

[`comfyui/ComfyUI-ROCm-FP8-Fastpath`](comfyui/ComfyUI-ROCm-FP8-Fastpath) is a ComfyUI custom node for
fp8_scaled checkpoints (Krea2 Turbo, for one) on AMD GPUs. It adds no node to place in a workflow:
copy the folder into `ComfyUI/custom_nodes` and restart ComfyUI. On a ROCm build of PyTorch it
patches three things when ComfyUI starts, and on any other build it does nothing.

- **fp8 matmuls through hipBLASLt.** comfy-kitchen's per-tensor fp8 linear layers run on
  `torch._scaled_mm` (hipBLASLt), which takes 0.76–0.97 of the time of comfy-kitchen 0.2.41's own
  HIP kernel at the shapes measured; calls with K of 16384 or more stay on comfy-kitchen's kernel,
  where hipBLASLt is slower. comfy-kitchen before 0.2.25 had no fp8 matmul on HIP at all.
- **No fp8 transpose copy.** Layers that dequantize their weight do it from the contiguous storage
  instead of a transposed copy.
- **Per-token fp8 for the full-precision layers.** The layers an fp8_scaled checkpoint keeps at full
  precision (`full_precision_matrix_mult`, 96 in Krea2 Turbo) run through `sageattention.fp8_linear`
  instead of dequantize + 16-bit matmul. Needs this package, v0.6 or later. A Krea2 Turbo step at
  1672×1256 takes 2.03 s with them against 2.27–2.29 s without, 11 % less.

At startup the log shows what was applied:

```
[ROCm-FP8-Fastpath] ACTIVE: hipblaslt-gemm, stride-aware-dequant, per-token-fp8
```

If a ComfyUI update changes an interface the per-token part relies on, that part is skipped with a
warning naming what changed, and the rest still applies.

| environment variable | effect |
|---|---|
| `COMFYUI_FP8_PERTOKEN=0` | turn the per-token fp8 layers off |
| `COMFYUI_FP8_UNIFORM_QUANT=1` | run the full-precision layers as ordinary per-tensor fp8 instead (faster, less accurate; replaces the per-token part) |

### fp8 linear layers

v0.6 also ships `sageattention.fp8_linear.linear(x, w8, w_scale, bias=None)`: `x @ (w8 * w_scale)^T`
for fp16 or bf16 `x` and an e4m3 weight with one scale, computed in fp8 with one activation scale per
token. It is meant for the layers fp8_scaled checkpoints mark `full_precision_matrix_mult` (96 in
Krea2 Turbo), which ComfyUI otherwise runs as dequantize + fp16 GEMM. At Krea2's shapes it takes
0.61–0.70× the time of that path, with the error of the checkpoint's other fp8 layers. Neither
`sageattn` nor ComfyUI calls it on its own: the node above, or another patch, has to route those
layers to it. A call it does not take (other dtypes, a per-channel scale, K or N not a multiple of
16, K over 32 768, no code object for the GPU, or one autograd needs a gradient through) returns
`None`.

## Build from source

[docs/BUILD.md](docs/BUILD.md) covers the native extensions (`setup.py`, ROCm SDK,
`PYTORCH_ROCM_ARCH=gfx1201`) and rebuilding the kernels' code objects from `kernels/hip/` with
`tools/build_hsaco.py`.

## How this was built

- [docs/JOURNEY.md](docs/JOURNEY.md): the whole path, from a Triton kernel that lost to PR #368
  to hand-written kernels that beat it, including the dead ends.
- [docs/FINDINGS.md](docs/FINDINGS.md): every finding, confirmed or refuted, with its numbers.
  F174 is how v0.6 caught up with comfy-kitchen, F182 the rotation that made it more accurate, F175
  the fp8 linear layer, F176 why K is always smoothed, F185 why short calls go to SDPA.
- [docs/METHOD.md](docs/METHOD.md): how measurements were taken, and the mistakes that shaped
  those rules.

## Limitations

- The prebuilt kernels target `gfx1201`, plus `gfx1200` behind a switch. The `gfx1200` builds have
  the same machine code as the `gfx1201` ones but have not been run on that card yet
  ([docs/GFX1200-TESTING.md](docs/GFX1200-TESTING.md)). Other GPUs use the fallback path.
- Every speed and accuracy number here was measured on Windows. Linux results from two testers,
  with torch 2.9.1 and a torch 2.14 nightly, are in [docs/LINUX.md](docs/LINUX.md).
- The int8 kernels are head_dim 128 only. Longer head_dim 64 calls run PR #368's path: two head_dim
  64 kernels built here were slower than it, and comfy-kitchen's int8 design is level with it on
  long head_dim 64 calls, so a third was not attempted.
- A causal call with unequal query and key lengths runs PR #368's path.
- comfy-kitchen is still a little more accurate on one of the 18 captures in fp16 and two in bf16.
- The int8 P·V objects are built with ROCm 10.1's compiler, the older objects with ROCm 10.0's; both
  load under either PyTorch wheel.

## A note from me

I've been busy with my day job and haven't had much time for this. I think the kernel itself could
still improve, but the returns are marginal now, and most of what is left is the HIP toolchain
rather than the attention kernel. If I could, I'd ship this with my own build of it -- but testing
and modifying it takes far too long on one GPU.

I've built this with AI coding tools. They wrote most of the kernel code, ran the tests across
the whole matrix and summarized the findings. I chose what to try, read the results and decided
what shipped, and nothing went in without passing the bit-for-bit and timing checks in
docs/METHOD.md. The failures are written up next to the wins in docs/FINDINGS.md.

v0.5 measured this package against comfy-kitchen and found it behind on both speed and accuracy.
v0.6 is the answer: int8 for both products, comfy-kitchen's lazy rescale and kernel shape, one K
scale per 16 keys only where it costs nothing, and a Hadamard rotation on the heads where it helps.
That made it as fast as comfy-kitchen and more accurate, with a third of PR #368's error. head_dim
64 still needs a new design.

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
