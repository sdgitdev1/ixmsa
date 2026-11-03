# Changelog

The heading gives the release and the wheel's version string. The `+amd.gfx12.N` part is a local
version identifier, so every wheel can be traced to the release it came from. The numbers below are
summaries; [docs/FINDINGS.md](docs/FINDINGS.md) has the measurements behind them and
[docs/METHOD.md](docs/METHOD.md) how they were taken.

## Unreleased (v0.6)

- **New default kernels: int8 Q·K and int8 P·V** (`attn_i8pv_*`, `kv_prologue_i8pv`) for fp16 and bf16, head_dim 128,
  every layout, causal or not. Against comfy-kitchen's `int8_attention`, through `sageattn`: 1.02 / 1.00 / 1.00 of its
  time on contiguous HND and 1.02 / 0.96 / 0.96 on NHD at 4k / 8.8k / 16k tokens (fp16; bf16 the same within 1 %).
  Against PR #368: 1.2–1.3× faster non-causal, 1.4–2.2× causal. Error on the 18 real captures: median 0.0072 in fp16
  against PR #368's 0.0126 and v0.5's 0.0139, lower than PR #368's on all 18. It uses comfy-kitchen's lazy rescale and
  kernel shape, one K scale per 16 keys on heads where that loses at most 2 bits per key (per-token scales on the
  others, which keeps Krea2's outlier layers accurate), and a seeded Q·K accumulator (F174). bf16 now uses int8 Q·K too.
- `SAGEATTN_SK1_I8PV=0` (or `sageattn(..., sk1_i8pv=False)`) turns the new kernels off; `SAGEATTN_SK1_INT8=0` does too
  for fp16. head_dim 64, fewer than 129 tokens and `sm_scale <= 0` keep the previous objects.
- **`SAGEATTN_CK=1`** hands non-causal head_dim 128 calls to comfy-kitchen's int8 attention when it
  is installed, for comparison (F167, F168). head_dim 64 stays on PR #368's path, where comfy-kitchen is more
  accurate but slower (F171). Off by default.
- The unshipped head_dim 64 prologue path and two unused helpers are gone; the loader closes the
  code object file after reading it.
- `tools/gfx1200_probe.py` now shows the object the device would actually load (it always printed
  the gfx1201 name).
- A test guards v0.5's first-call fix: the default path must not import Triton.
- With `SAGEATTN_SK1_I8PV=0`, output is bit-identical to v0.5.0.

## v0.5.0 — 2.2.0+amd.gfx12.5

**A clean-up release, measured against ComfyUI's own int8 attention. The kernels compute the same
bytes as v0.4.0.**

- **The first call no longer imports Triton.** The default path imported the Triton prologue module
  for one helper, and importing Triton runs a `rocm-sdk` subprocess. In ComfyUI's environment the
  first call drops from 0.48 s to 0.08 s, now below PR #368's 0.24 s.
- **Kernels named after what they do.** `sk1_t6i` is now `attn_int8qk_f16`, `h34_prologue` is
  `kv_prologue`, and so on, with one `.gfx1201.hsaco` / `.gfx1200.hsaco` naming scheme. The machine
  code is byte-identical; docs/BUILD.md has the old-to-new table. Environment variables and the
  Python API are unchanged.
- **Dead code removed.** Three objects nothing could route to (`sk1_t4a1`, `sk1_d64`, the unshipped
  `sk1_d64b`), their loaders, research-only kernel sources, and most of the backend's research-diary
  comments.
- **Compared with comfy-kitchen** (`--use-ck-attention`): it is 6–11 % faster per call and its
  full-call error on real activations is 0.35–0.41× this package's (median). The README says so.
- **Accuracy claims corrected.** The README's "more precise than PR #368 on all ten captures" came
  from comparing Q·K schemes in isolation. End to end, the fp8 P·V makes this package's error about
  1.5–2× PR #368's on ordinary layers; PR #368 is still 4–7× worse on layers with extreme keys.
- Same two Windows wheels as v0.4.0: PyTorch `2.13.0+rocm10.0.0`, and PyTorch `2.14.0+rocm10.1.0`
  with `-214-` in the name.

## v0.4.0 — 2.2.0+amd.gfx12.4

**The quantization step before the kernel is now hand-written HIP as well, so short calls are about
even with PR #368's instead of up to 1.6× slower, and nothing is compiled on first use.**

- **A HIP prologue replaces the Triton one.** Every call quantizes K and V before the attention
  kernel runs. That used to be 3–4 Triton launches with a JIT compiler behind them. It is now 1–2
  launches of a prebuilt code object, on every default path (fp16 with int8 Q·K, fp16 with fp8
  Q·K, bf16; contiguous, strided and fused-QKV layouts). What that changes:
  - **Short calls are faster.** Timed one at a time, a call of 288 to 1 024 tokens takes 0.85–1.13
    of PR #368's time, where v0.3.0 took 1.12–1.59. Production sizes are unchanged within noise.
  - **No compile on first use.** In ComfyUI's environment with an empty Triton cache, the first
    call takes 0.48 s instead of 1.9 s, and nothing is compiled.
  - **No 2^31-element limit.** v0.3.0 stopped serving at about 100 000 tokens at H3's layout,
    because the Triton prologue computed offsets in 32 bits. The HIP prologue uses 64-bit offsets.
  - **Same output.** With `smooth_k` off, the quantized bytes and the attention output are
    bit-identical to v0.3.0, including all-zero rows, values past fp8's range, and NaN/Inf. With
    `smooth_k` on, the key mean is now summed in fp64, which is deterministic and more accurate
    than the old fp32 sum. A small share of outputs therefore differ from v0.3.0 by rounding, with
    the same error against exact attention.
  - `SAGEATTN_SK1_PROLOGUE=triton` brings back v0.3.0's Triton prologue.
- **RX 9060 XT (gfx1200), opt in.** The wheel carries gfx1200 builds of every kernel the default
  paths use. Their machine code is byte-identical to the gfx1201 builds, but nobody has run them on
  a 9060 XT yet, so they stay off unless `SAGEATTN_SK1_GFX1200=1` is set. The device always picks
  its own builds. If you have the card, [docs/GFX1200-TESTING.md](docs/GFX1200-TESTING.md) says
  what to run and what to send back.
- **The build records the torch it was built against.** The runtime check compares against that
  stamp instead of a hard-coded `2.13.0+rocm10.0.0`, which is what wheels for other torch versions
  and for Linux need.

Not in this release:
- **head_dim 64** still uses PR #368's kernel. A head_dim 64 kernel on the new prologue was built
  and measured, and it is still 1.03–5.3× slower than PR #368's. That needs its own design.
- **ROCm 10.1 / PyTorch 2.14** came later, on 2026-10-29 (#3): a second Windows wheel,
  `sageattention-2.2.0+amd.gfx12.4-214-cp312-cp312-win_amd64.whl`, built against
  `2.14.0+rocm10.1.0`. Same code; it passes 97 of 97 tests on two machines and its output is
  bit-identical to the 2.13 wheel. PyTorch 2.14 itself needs a recent driver (it failed on
  32.0.31041.1004 and runs on Adrenalin 26.8.1 and 26.10.1).
- **A Linux wheel.** Linux runs from source; Linux testing of this release happens in a pull
  request.

## v0.3.0 — 2.2.0+amd.gfx12.3

- **Fused q/k/v views are served.** MiniMax-H3, and any model that fuses q, k and v into one
  projection, passes them as views of one tensor, so a row's stride is three heads wide. Those
  calls used to fall back to PR #368's path without saying so. The acceptance rule is now the raw
  launch's own requirement, and a refused call carries its reason.
- **Against PR #368 at production sizes** (N = 27 280 to 99 792): 0.87–0.94 of its time. In a real
  MiniMax-H3 w4a8 render at 864x480 and 124 frames, the sampling phase took 0.950 (4 steps) and
  0.847 (8 steps) of PR #368's time.
- **Linux runs from source**, thanks to [boxwrench](https://github.com/boxwrench)
  ([#1](https://github.com/IxMxAMAR/SageAttention-RDNA4/pull/1)). The wheel is Windows-only.
- The version strings in `setup.py`, `sageattention/__init__.py` and `sageattention/_version.py`
  agree again.

Known issue outside this package: with the ROCm-int8 ComfyUI pack's `OTUNetLoaderW8A8`, the first
forward pass of a w4a8 checkpoint hung the driver on the test machine (6 of 6 attempts). ComfyUI's
own `UNETLoader` handled the same checkpoint 5 times out of 5.

## v0.2.0 — 2.2.0+amd.gfx12.2

- **int8 Q·K is the default for fp16**, head_dim 128. Calls it cannot serve fall back to the fp8
  kernel, then to PR #368's path.
- **bf16 runs on the hand-written kernels**, also by default (fp8 Q·K; there is no int8 bf16
  kernel).
- `SAGEATTN_SK1_INT8=0` and `SAGEATTN_SK1_BF16=0` restore v0.1.0's behaviour exactly. `=1` makes
  the switch strict: it raises when its kernel is missing or fails to load.

## v0.1.0 — 2.2.0+amd.gfx12.1

- First release: a hand-written HIP fp8 attention kernel for fp16, head_dim 128, used by default on
  gfx1201, with a silent fallback to PR #368's path everywhere else and on any load failure.
