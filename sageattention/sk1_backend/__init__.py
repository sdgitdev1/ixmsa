"""Prebuilt SK1 backend for the gfx12 native path (on by default on gfx1201).

The package ships prebuilt gfx1201 code objects (and, opt-in, the same set built for gfx1200) and
the loader that launches them through the HIP runtime DLL torch has already loaded. A `.hsaco` is a
self-contained `elf64-amdgpu` file with no C++ imports, so the `c10` ABI mismatch that breaks a
`.pyd` extension against a different torch build cannot arise.

* `attn_fp8qk_f16` (contiguous HND) and `attn_fp8qk_f16_strided` (strided HND and NHD, six extra
  stride arguments for Q and O). Both take a per-(b,h) scale `S` as a 12th argument:
  `S = clamp_min(max_j sv(j), 1)` is computed on the device and `SV' = SV/S` is staged, so a key
  token with `|V| > 448` cannot overflow the fp8 conversion in the P-fold (the output used to go
  non-finite). `S == 1.0` exactly when every `amax_v <= 448`.
* `attn_int8qk_f16` / `attn_int8qk_f16_strided`: INT8 Q.K variants (K quantised per token to int8,
  V / P.V as fp8). fp16, head_dim 128, used by default.
* `attn_fp8qk_bf16` / `attn_fp8qk_bf16_strided`: the fp8 kernels with a bf16 Q/O element type.
  bf16, head_dim 128, used by default for bf16 input.
* `kv_prologue`: the HIP KV prologue (`kernels/hip/kv_prologue.hip`), used by default.
* `attn_i8pv_*` + `kv_prologue_i8pv` (`i8pv.py`): int8 Q.K and int8 P.V, fp16 and bf16, head_dim 128, every
  layout above, causal or not. Used first by default; a call it does not take goes to the objects above.

No object serves head_dim 64: those calls fall back to the shipped path.

The backend is used only when the device arch is `gfx1201` (or `gfx1200` with
`SAGEATTN_SK1_GFX1200=1`), a HIP runtime resolves, the code object loads, and the call is inside
`ENVELOPE`. Any other case falls back to the shipped path (the gfx12 native extension on gfx12,
Triton on other arches) with at most one short log line per process, and never raises at call time.

Environment variables (read once, at import):

* `SAGEATTN_SK1_BACKEND`: `1` forces the backend on (a load failure then raises), `0` forces it off
  and wins over every other switch, unset means on.
* `SAGEATTN_SK1_INT8`: unset serves fp16 head_dim-128 calls with the int8 objects and falls back to
  the fp8 objects, then to the shipped path. `0` uses the fp8 objects only. `1` is strict: a call
  int8 cannot serve raises.
* `SAGEATTN_SK1_BF16`: unset serves bf16 head_dim-128 calls with the bf16 objects and falls back
  normally. `0` sends bf16 calls to the shipped path (reason `"dtype"`). `1` is strict.
* `SAGEATTN_SK1_PROLOGUE`: unset or `hip` quantises K and V with the HIP prologue, falling back to
  the Triton prologue for a call it cannot serve. `triton` uses the Triton prologue only.
* `SAGEATTN_SK1_I8PV`: unset serves fp16/bf16 head_dim-128 calls with a positive scale and N >= 129 with the
  int8 P.V objects and falls back to the objects above for anything else or on a failure. `0` turns them off.
  `1` is strict: a call they cannot serve raises.
* `SAGEATTN_SK1_GFX1200=1`: lets a gfx1200 device use the `*.gfx1200.hsaco` objects (the same
  kernels built with `--offload-arch=gfx1200`). Unset, a gfx1200 device is refused (reason
  `"arch"`). No effect on gfx1201.

The `.hsaco` files are rebuilt from `kernels/hip/` by `tools/build_hsaco.py` in the source
repository; there is no in-wheel rebuild path.

`rocm_bin_dir()` resolves the HIP runtime at call time, trying in order: `$SK1_HIP_DLL`, the
`amdhip64*.dll` already loaded in this process (torch's own copy), `$SK1_ROCM_BIN`, then the
`_rocm_sdk_core` / `_rocm_sdk_devel` layouts under `site-packages` (see `sk1_loader.py`).
"""
from __future__ import annotations

import os
import sys

# `quant_kv_v8t` does `from quant_triton import ...` as a flat top-level import
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

__all__ = ["available", "try_sk1_t1", "ENVELOPE", "SUPPORTED_ARCHS",
           "device_arch", "arch_supported", "SK1_INT8", "SK1_INT8_STRICT", "int8_status",
           "SK1_BF16", "SK1_BF16_STRICT", "bf16_status",
           "SK1_GFX1200", "SK1_GFX1200_ARCHS", "gfx1200_status", "arch_suffix", "selected_arch"]

# ---- gfx1200 objects, opt-in ---------------------------------------------------------------------
#: Every gfx1201 code object is also shipped built for gfx1200 (RX 9060 XT, the same RDNA4 ISA) as
#: `*.gfx1200.hsaco`. A gfx1200 device is routed to them only when `SAGEATTN_SK1_GFX1200=1`;
#: otherwise it is refused with `reason="arch"` and one log line naming the switch. No effect on
#: gfx1201.
SK1_GFX1200_DEFAULT_ON = False
#: the architectures that have their own object set
SK1_GFX1200_ARCHS = ("gfx1200",)

_sk1_g12_env = os.environ.get("SAGEATTN_SK1_GFX1200", "").strip().lower()
if _sk1_g12_env in ("1", "true", "yes", "on"):
    SK1_GFX1200 = True
elif _sk1_g12_env in ("0", "false", "no", "off"):
    SK1_GFX1200 = False
else:
    SK1_GFX1200 = bool(SK1_GFX1200_DEFAULT_ON)

#: the arch the shipped object set is built for; the default and the fallback
_DEFAULT_ARCH = "gfx1201"


def arch_suffix(arch):
    """The file-name suffix for `arch`'s object set: `".gfx1201"` or `".gfx1200"`."""
    return "." + arch


def selected_arch(device=None):
    """Which object set serves `device`, or `None` if none may.

    The one place the arch/switch decision is made. `gfx1201` -> `"gfx1201"` always; `gfx1200` ->
    `"gfx1200"` iff `SK1_GFX1200`, else `None`; anything else or an unreadable arch -> `None`.
    """
    arch = device_arch(device)
    if not arch:
        return None
    if any(arch.startswith(a) for a in SK1_GFX1200_ARCHS):
        return arch if SK1_GFX1200 else None
    return _DEFAULT_ARCH if arch.startswith(_DEFAULT_ARCH) else None


#: The arch whose file names the base constants below carry. Always `gfx1201`, on purpose:
#: `object_path()` rewrites the literal gfx1201 names, so if this followed the switch a gfx1201
#: device with `SAGEATTN_SK1_GFX1200=1` would be handed a gfx1200 object (`hipModuleLoadData`
#: rc=209). The device chooses the object set, per call, through `_arch_now()`.
_SEL_ARCH = _DEFAULT_ARCH


def _arch_now(device=None):
    """The object set for this device right now: `selected_arch()`, or the gfx1201 default.

    Loaders default to this, never to `_SEL_ARCH`, so the object set always follows the device.
    """
    return selected_arch(device) or _DEFAULT_ARCH


def object_path(base_gfx1201_path, arch):
    """The object path for `arch`, given the gfx1201 path (gfx1200: `attn_fp8qk_f16.gfx1200.hsaco`).

    The suffix comes from the same `arch` value the arch check used, so a gfx1200 name can never
    be produced while the active arch is gfx1201.
    """
    return base_gfx1201_path.replace(".gfx1201.hsaco", "%s.hsaco" % arch_suffix(arch))


def _with_suffix(name):
    """The module's own default-arch spelling of a gfx1201 object name."""
    return object_path(name, _SEL_ARCH)


SK1_HSACO_NAME_S = _with_suffix("attn_fp8qk_f16.gfx1201.hsaco")
SK1_HSACO_PATH_S = os.path.join(_HERE, SK1_HSACO_NAME_S)
SK1_SYMBOLS_S = ("attn_fp8qk_f16_fwd", "attn_fp8qk_f16_fwd_causal")

#: `attn_fp8qk_f16_strided`: `attn_fp8qk_f16`'s 12 arguments plus six Q/O strides. Contiguous HND
#: calls are still routed to `attn_fp8qk_f16`; this one serves only strided HND and NHD.
SK1_HSACO_NAME_N = _with_suffix("attn_fp8qk_f16_strided.gfx1201.hsaco")
SK1_HSACO_PATH_N = os.path.join(_HERE, SK1_HSACO_NAME_N)
SK1_SYMBOLS_N = ("attn_fp8qk_f16_strided_fwd", "attn_fp8qk_f16_strided_fwd_causal")

#: The INT8 Q.K objects (`attn_int8qk_f16`, `attn_int8qk_f16_strided`), the default for fp16, D=128.
#: Q.K runs int8 x int8 on `v_wmma_i32_16x16x16_iu8`; the V / P.V path is the fp8 one.
#:
#: `SAGEATTN_SK1_INT8`: unset -> `SK1_INT8_DEFAULT_ON`, non-strict (int8 serves every fp16 D=128 call
#: it can and falls back normally: fp8 SK1 first, then the gfx12 native path); `0/false/no/off` ->
#: OFF, the fp8 behaviour bit for bit; `1/true/yes/on` -> ON and strict, the only mode in which a
#: refusal is an error. `SAGEATTN_SK1_BACKEND=0` wins over it.
#:
#: fp16 and D=128 only: a `head_dim == 64` call is refused (`reason="int8_d128_only"`), and a bf16
#: call never reaches this path (bf16 takes precedence; there is no int8 bf16 object).
SK1_HSACO_NAME_I8 = _with_suffix("attn_int8qk_f16.gfx1201.hsaco")
SK1_HSACO_PATH_I8 = os.path.join(_HERE, SK1_HSACO_NAME_I8)
SK1_SYMBOLS_I8 = ("attn_int8qk_f16_fwd", "attn_int8qk_f16_fwd_causal")
SK1_HSACO_NAME_I8N = _with_suffix("attn_int8qk_f16_strided.gfx1201.hsaco")
SK1_HSACO_PATH_I8N = os.path.join(_HERE, SK1_HSACO_NAME_I8N)
SK1_SYMBOLS_I8N = ("attn_int8qk_f16_strided_fwd", "attn_int8qk_f16_strided_fwd_causal")
SK1_INT8_DEFAULT_ON = True

_sk1_i8_env = os.environ.get("SAGEATTN_SK1_INT8", "").strip().lower()
if _sk1_i8_env in ("1", "true", "yes", "on"):
    SK1_INT8 = True
    SK1_INT8_STRICT = True
elif _sk1_i8_env in ("0", "false", "no", "off"):
    SK1_INT8 = False
    SK1_INT8_STRICT = False
else:
    SK1_INT8 = bool(SK1_INT8_DEFAULT_ON)
    SK1_INT8_STRICT = False

#: The bf16 objects: the fp16 kernel source with the Q/O element type templated
#: (`kernels/hip/attn_fp8qk_bf16.hip`, `attn_fp8qk_bf16_strided.hip`). The dtype contract is per
#: object (`sk1_loader.Sk1Attn.qo_dtype`): fp16 objects refuse a bf16 Q and these refuse an fp16 one.
#:
#: `SK1_BF16_DEFAULT_ON` is True: unset `SAGEATTN_SK1_BF16` serves bf16 calls with these objects
#: and falls back normally. `=1` forces ON and strict, `=0` forces OFF (reason `"dtype"`).
SK1_HSACO_NAME_B = _with_suffix("attn_fp8qk_bf16.gfx1201.hsaco")
SK1_HSACO_PATH_B = os.path.join(_HERE, SK1_HSACO_NAME_B)
SK1_SYMBOLS_B = ("attn_fp8qk_bf16_fwd", "attn_fp8qk_bf16_fwd_causal")
SK1_HSACO_NAME_BN = _with_suffix("attn_fp8qk_bf16_strided.gfx1201.hsaco")
SK1_HSACO_PATH_BN = os.path.join(_HERE, SK1_HSACO_NAME_BN)
SK1_SYMBOLS_BN = ("attn_fp8qk_bf16_strided_fwd", "attn_fp8qk_bf16_strided_fwd_causal")
SK1_BF16_DEFAULT_ON = True

_sk1_b_env = os.environ.get("SAGEATTN_SK1_BF16", "").strip().lower()
if _sk1_b_env in ("1", "true", "yes", "on"):
    SK1_BF16 = True
    SK1_BF16_STRICT = True
elif _sk1_b_env in ("0", "false", "no", "off"):
    SK1_BF16 = False
    SK1_BF16_STRICT = False
else:
    SK1_BF16 = bool(SK1_BF16_DEFAULT_ON)
    SK1_BF16_STRICT = False

_sk1_v_env = os.environ.get("SAGEATTN_SK1_I8PV", "").strip().lower()
SK1_I8PV = _sk1_v_env not in ("0", "false", "no", "off")
SK1_I8PV_STRICT = _sk1_v_env in ("1", "true", "yes", "on")

#: The arches a packaged code object exists for (prefix test, so `gfx1201:sramecc-:xnack-` matches).
#: Membership means "an object set exists", not "this device is served"; that is `selected_arch()`.
SUPPORTED_ARCHS = ("gfx1201", "gfx1200")

#: The exact envelope `try_sk1_t1` serves; anything else returns `None` (caller falls back).
#: `contiguous` means "contiguous in its own layout, or a physically-NHD view"
#: (`sk1_loader._strided_triple`). A bf16 call needs `SK1_BF16`, else it falls back with reason
#: `"dtype"`. `head_dim` 64 is inside the envelope but no D=64 object is shipped, so it falls back.
ENVELOPE = dict(
    layout=("HND", "NHD"),
    dtype=("fp16", "bf16"),
    head_dim=(64, 128),
    contiguous=("HND-contiguous", "physically-NHD view (HND or NHD logical order)"),
    causal=(0, 1),
    smooth_k=(False, True),
    batch_heads="any",
    seq_len="N >= 1",
    return_lse=False,
    attn_mask=False,
    smooth_v=False,
)

_BACKEND = None
_BACKEND_FAILED = None
_BACKEND_N = None
_BACKEND_N_FAILED = None
_BACKEND_I8 = None
_BACKEND_I8_FAILED = None
_BACKEND_I8N = None
_BACKEND_I8N_FAILED = None
_BACKEND_B = None
_BACKEND_B_FAILED = None
_BACKEND_BN = None
_BACKEND_BN_FAILED = None
_LOGGED = False
#: arch -> loaded `i8pv.I8pv`, or the message of the error its load raised
_I8PV = {}
#: `sk1_loader.strided_view_reason`, imported once on first use: `_envelope_ok` runs the guard three
#: times per attention call, so the import stays off the hot path.
_SVR = None


def _strided_reason(t, name):
    """The widened-acceptance guard. Returns `None` if `t` is servable, else a reason."""
    global _SVR
    if _SVR is None:
        from .sk1_loader import strided_view_reason
        _SVR = strided_view_reason
    return _SVR(t, name)


def device_arch(device=None) -> str:
    """The device's `gcnArchName` (prefix before `:`), or `""` if it cannot be read. Never raises.

    A named function so a test can monkeypatch the arch query without touching the routing logic.
    """
    try:
        import torch
        if device is None:
            idx = 0
        elif isinstance(device, int):
            idx = device
        else:
            idx = device.index if getattr(device, "index", None) is not None else 0
        props = torch.cuda.get_device_properties(idx)
        arch = str(getattr(props, "gcnArchName", "") or "")
        return arch.split(":", 1)[0]
    except Exception:
        return ""


def arch_supported(device=None) -> bool:
    """True iff `device` may be served by a packaged code object right now.

    This is `selected_arch(device) is not None`. The module attribute is consulted through
    `globals()` so a test that replaces this function (even with a `MagicMock`) is honoured.
    """
    import types
    cur = globals().get("arch_supported")
    if not isinstance(cur, types.FunctionType):
        return bool(cur(device))
    return selected_arch(device) is not None


def _arch_refusal_reason(device=None) -> str:
    """The one-line reason for an arch refusal, naming the switch when that is what blocked it."""
    arch = device_arch(device) or "<unknown>"
    if arch and any(arch.startswith(a) for a in SK1_GFX1200_ARCHS) and not SK1_GFX1200:
        return ("device arch %r needs its own objects; set SAGEATTN_SK1_GFX1200=1 to enable them "
                "(they are opt-in until a gfx1200 tester confirms them)" % arch)
    return "device arch %r is not %s" % (arch, _DEFAULT_ARCH)


def _log_once(reason: str) -> None:
    """At most one short line per process, and never an exception out of the logging itself."""
    global _LOGGED
    if _LOGGED:
        return
    _LOGGED = True
    try:
        sys.stderr.write("[sageattention] SK1 backend not used (%s); using the shipped path.\n"
                         % reason)
        sys.stderr.flush()
    except Exception:                                          # pragma: no cover
        pass


# ---- the HIP KV prologue router ------------------------------------------------------------------
#: `SAGEATTN_SK1_PROLOGUE` picks the prologue that runs before the attention object: `triton` (the
#: shipped `quant_kv_v8t*` kernels) or `hip` (`kernels/hip/kv_prologue.hip`, loaded through
#: `sk1_loader.HipRuntime`). The default is `"hip"`; any other value means the default. A call the
#: HIP prologue cannot serve falls back to the Triton prologue, never to the gfx12 native path.
SK1_PROLOGUE_DEFAULT = "hip"
_sk1_prol_env = os.environ.get("SAGEATTN_SK1_PROLOGUE", "").strip().lower()
if _sk1_prol_env in ("hip", "triton"):
    SK1_PROLOGUE = _sk1_prol_env
else:
    SK1_PROLOGUE = SK1_PROLOGUE_DEFAULT

_HIP_P = None
_HIP_P_TRIED = False
#: the arch the loaded prologue was built for
_HIP_P_ARCH = None
_PROLOGUE_LOGGED = 0


def _log_prologue(msg: str) -> None:
    """The prologue's own at-most-8-lines logger; not `_log_once`, which owns the "SK1 backend not
    used" line."""
    global _PROLOGUE_LOGGED
    if _PROLOGUE_LOGGED >= 8:
        return
    _PROLOGUE_LOGGED += 1
    try:
        sys.stderr.write("[sageattention] HIP prologue: %s\n" % msg)
        sys.stderr.flush()
    except Exception:                                          # pragma: no cover
        pass


def _hip_prologue(arch=None):
    """The loaded HIP prologue object, or `None`. Loaded once per process; never raises.

    Per-arch like every `_load_*` (default: `_arch_now()`), so a gfx1201 prologue can never serve a
    gfx1200 call.
    """
    global _HIP_P, _HIP_P_TRIED, _HIP_P_ARCH
    arch = arch or _arch_now()
    if _HIP_P_TRIED and _HIP_P_ARCH == arch:
        return _HIP_P
    _HIP_P_TRIED = True
    _HIP_P_ARCH = arch
    _HIP_P = None
    if SK1_PROLOGUE != "hip":
        return None
    try:
        from .hip_prologue import HipPrologue, hsaco_path_for
        _HIP_P = HipPrologue(hsaco_path=hsaco_path_for(arch))
    except Exception as e:                                      # noqa: BLE001
        _log_prologue("the HIP prologue did not load (%s); the Triton prologue serves the calls" % e)
        _HIP_P = None
    return _HIP_P


def packaged_hsaco_path(arch=None) -> str:
    """The object the packaged path loads: `attn_fp8qk_f16`, for `arch` (default `_arch_now()`)."""
    return object_path(SK1_HSACO_PATH_S, arch or _arch_now())


def gfx1200_status() -> dict:
    """The state of the gfx1200 opt-in without loading anything or needing a device. Never raises.

    Keys: `switch`, `default_on`, `archs`, `supported_archs`, `objects` and `gfx1201_objects`
    (`{name: {"file", "present", "sha256", "size"}}`).
    """
    def one(path):
        out = dict(file=os.path.basename(path), path=path.replace("\\", "/"),
                   present=os.path.isfile(path), sha256=None, size=None)
        try:
            with open(path, "rb") as fh:
                import hashlib
                h = hashlib.sha256()
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
                out["sha256"] = h.hexdigest()
            out["size"] = os.path.getsize(path)
        except Exception:
            pass
        return out
    names = ("attn_fp8qk_f16", "attn_fp8qk_f16_strided", "attn_fp8qk_bf16", "attn_fp8qk_bf16_strided",
             "attn_int8qk_f16", "attn_int8qk_f16_strided", "kv_prologue")
    return dict(switch=bool(SK1_GFX1200), default_on=bool(SK1_GFX1200_DEFAULT_ON),
                archs=list(SK1_GFX1200_ARCHS), supported_archs=list(SUPPORTED_ARCHS),
                base_arch=_SEL_ARCH, selected_now=_arch_now(),
                objects={n: one(object_path(os.path.join(_HERE, n + ".gfx1201.hsaco"), "gfx1200"))
                         for n in names},
                gfx1201_objects={n: one(os.path.join(_HERE, n + ".gfx1201.hsaco"))
                                 for n in names})


def fold_v_scale(sv):
    """Stage `SV' = SV / S` on the device and return `S` (B*H,) fp32.

    `S = clamp_min(max_j sv(j), 1)`, so `S > 1` exactly when some key row has `amax_v > 448`. The
    kernel's P fold is `e4m3(p * amax_v)` with no clamp, so that would overflow e4m3 and make the
    output non-finite; dividing `sv` by `S` keeps the fold's argument `<= 448` and the kernel's
    epilogue multiplies by `S` again. Three device ops, no host sync. When every `amax_v <= 448`,
    `S == 1.0` and `sv` is bit-for-bit unchanged.
    """
    import torch
    s = sv.amax(dim=1).clamp_min(1.0)
    sv.div_(s.unsqueeze(1))
    return s


def available() -> bool:
    """True iff the prebuilt object exists and a HIP runtime can be resolved. Never raises."""
    global _BACKEND, _BACKEND_FAILED
    if _BACKEND is not None:
        return True
    if _BACKEND_FAILED is not None:
        return False
    if not os.path.isfile(packaged_hsaco_path()):
        _BACKEND_FAILED = "missing %s" % packaged_hsaco_path()
        return False
    try:
        from .sk1_loader import HipRuntime, Sk1Attn  # noqa: F401
    except Exception as e:                                   # pragma: no cover
        _BACKEND_FAILED = "%s: %s" % (type(e).__name__, e)
        return False
    return True


#: The arch the cached objects were loaded for. A cache built for one arch must never serve a call
#: routed for the other, so every `_load_*` is keyed on it and a mismatch drops the caches.
_ARCH_LOADED = None

#: Every loader cache and its "already failed" flag. An arch change must drop all of them: a call
#: can load the int8 object and then fall through to the fp8 one, so dropping only `_BACKEND` would
#: leave a module built for the previous arch (only possible when the arch probe is monkeypatched).
_ARCH_CACHES = (
    "_BACKEND", "_BACKEND_FAILED",
    "_BACKEND_N", "_BACKEND_N_FAILED",
    "_BACKEND_I8", "_BACKEND_I8_FAILED",
    "_BACKEND_I8N", "_BACKEND_I8N_FAILED",
    "_BACKEND_B", "_BACKEND_B_FAILED",
    "_BACKEND_BN", "_BACKEND_BN_FAILED",
)


def _cache_arch(arch, cache_name, failed_name):
    """Drop every loader cache if it was built for a different arch, and record the arch.

    The HIP prologue cache also drops itself via `_HIP_P_ARCH`; it is reset here too so the two
    cannot disagree.
    """
    global _ARCH_LOADED, _HIP_P, _HIP_P_TRIED, _HIP_P_ARCH
    if _ARCH_LOADED != arch:
        for name in _ARCH_CACHES:
            globals()[name] = None
        _HIP_P = None
        _HIP_P_TRIED = False
        _HIP_P_ARCH = None
        _ARCH_LOADED = arch
    return arch


def _load(arch=None):
    global _BACKEND, _BACKEND_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND", "_BACKEND_FAILED")
    if _BACKEND is not None:
        return _BACKEND
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        p = object_path(SK1_HSACO_PATH_S, arch)
        _BACKEND = Sk1Attn(p, HipRuntime(), symbols=SK1_SYMBOLS_S, with_s=True)
        return _BACKEND
    except Exception as e:
        _BACKEND_FAILED = "%s: %s" % (type(e).__name__, e)
        raise


def _load_n(arch=None):
    """The widened object `attn_fp8qk_f16_strided`, or `None` if it is not installed or will not load.

    A missing object is a clean refusal (`reason="no_strided_object"`), never a raise on the default
    path. `arch` defaults to this device's set (`_arch_now()`).
    """
    global _BACKEND_N, _BACKEND_N_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND_N", "_BACKEND_N_FAILED")
    if _BACKEND_N is not None:
        return _BACKEND_N
    if _BACKEND_N_FAILED is not None:
        return None
    p = object_path(SK1_HSACO_PATH_N, arch)
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(p):
            _BACKEND_N_FAILED = "missing %s" % p
            return None
        _BACKEND_N = Sk1Attn(p, HipRuntime(), symbols=SK1_SYMBOLS_N,
                             with_s=True, strided=True)
        return _BACKEND_N
    except Exception as e:                                   # pragma: no cover
        _BACKEND_N_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def _load_i8(arch=None):
    """The contiguous-HND INT8 object `attn_int8qk_f16`, or `None` if it is not installed or will not load.

    A missing object is a clean refusal (`reason="no_int8_object"`), never a raise on the default path.
    """
    global _BACKEND_I8, _BACKEND_I8_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND_I8", "_BACKEND_I8_FAILED")
    if _BACKEND_I8 is not None:
        return _BACKEND_I8
    if _BACKEND_I8_FAILED is not None:
        return None
    p = object_path(SK1_HSACO_PATH_I8, arch)
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(p):
            _BACKEND_I8_FAILED = "missing %s" % p
            return None
        _BACKEND_I8 = Sk1Attn(p, HipRuntime(), symbols=SK1_SYMBOLS_I8, with_s=True)
        return _BACKEND_I8
    except Exception as e:                                   # pragma: no cover
        _BACKEND_I8_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def _load_i8n(arch=None):
    """The strided-HND / NHD INT8 object `attn_int8qk_f16_strided`, or `None`.  See `_load_i8()`."""
    global _BACKEND_I8N, _BACKEND_I8N_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND_I8N", "_BACKEND_I8N_FAILED")
    if _BACKEND_I8N is not None:
        return _BACKEND_I8N
    if _BACKEND_I8N_FAILED is not None:
        return None
    p = object_path(SK1_HSACO_PATH_I8N, arch)
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(p):
            _BACKEND_I8N_FAILED = "missing %s" % p
            return None
        _BACKEND_I8N = Sk1Attn(p, HipRuntime(), symbols=SK1_SYMBOLS_I8N,
                               with_s=True, strided=True)
        return _BACKEND_I8N
    except Exception as e:                                   # pragma: no cover
        _BACKEND_I8N_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


_I8PV_LOGGED = 0


def _log_i8pv(msg: str) -> None:
    """The int8 P.V path's own at-most-4-lines logger; the call still runs on the other SK1 objects, so this is
    not `_log_once`'s "SK1 backend not used" line."""
    global _I8PV_LOGGED
    if _I8PV_LOGGED >= 4:
        return
    _I8PV_LOGGED += 1
    try:
        sys.stderr.write("[sageattention] int8 P.V kernels not used (%s); using the other SK1 objects.\n" % msg)
        sys.stderr.flush()
    except Exception:                                          # pragma: no cover
        pass


def _try_i8pv(qq, kk, vv, tensor_layout, contiguous_hnd, is_causal, sm, smooth_k, arch, strict):
    """`(out, None)` from the int8 P.V objects, or `(None, reason)` for the caller to fall through. Raises
    only when `strict`."""
    from . import i8pv
    why = i8pv.refusal(qq, sm)
    if why is not None:
        if strict:
            raise RuntimeError("the int8 P.V objects cannot serve this call (%s)" % why)
        return None, why
    # A failed load is cached as its message, never as the exception: re-raising one exception object on every
    # call would chain each call's frame (and its q/k/v) onto its traceback for the life of the process.
    v = _I8PV.get(arch)
    if isinstance(v, str):
        if strict:
            raise RuntimeError(v)
        return None, "i8pv_load_failed"
    try:
        if v is None:
            try:
                v = i8pv.I8pv(arch)
            except Exception as e:                                  # noqa: BLE001
                _I8PV[arch] = "the int8 P.V objects did not load: %s" % e
                raise
            _I8PV[arch] = v
        import torch
        B, H, N, D = qq.shape
        if contiguous_hnd:
            out = torch.empty(B * H, N, D, device=qq.device, dtype=qq.dtype)
            v.run(qq, kk, vv, out, bool(is_causal), sm, smooth_k, True)
            return out.view(B, H, N, D), None
        if tensor_layout == "NHD":
            out = torch.empty(B, N, H, D, device=qq.device, dtype=qq.dtype)
            v.run(qq, kk, vv, out.transpose(1, 2), bool(is_causal), sm, smooth_k, False)
        else:
            out = torch.empty(B, H, N, D, device=qq.device, dtype=qq.dtype)
            v.run(qq, kk, vv, out, bool(is_causal), sm, smooth_k, False)
        return out, None
    except Exception as e:                                          # noqa: BLE001
        if strict:
            raise
        _log_i8pv(str(e))
        return None, "i8pv_failed"


def int8_status() -> dict:
    """The state of the two int8 objects without loading anything. Never raises.

    Keys: `enabled`, `strict`, `default_on`, and `contig` / `strided` dicts with `file`, `present`,
    `sha256`, `size`, `loaded`, `failed`.
    """
    def one(path, name, loaded, failed):
        out = dict(file=name, present=os.path.isfile(path), sha256=None, size=None,
                   loaded=(loaded is not None), failed=failed)
        try:
            with open(path, "rb") as fh:
                import hashlib
                h = hashlib.sha256()
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
                out["sha256"] = h.hexdigest()
            out["size"] = os.path.getsize(path)
        except Exception:
            pass
        return out
    return dict(enabled=bool(SK1_INT8), strict=bool(SK1_INT8_STRICT),
                default_on=bool(SK1_INT8_DEFAULT_ON),
                contig=one(SK1_HSACO_PATH_I8, SK1_HSACO_NAME_I8, _BACKEND_I8, _BACKEND_I8_FAILED),
                strided=one(SK1_HSACO_PATH_I8N, SK1_HSACO_NAME_I8N, _BACKEND_I8N,
                            _BACKEND_I8N_FAILED))


def _load_b(arch=None):
    """The bf16 object `attn_fp8qk_bf16` (contiguous HND, 12-argument), or `None` if unavailable.

    A missing object is a clean refusal (`reason="no_bf16_object"`), never a raise on the default path.
    """
    global _BACKEND_B, _BACKEND_B_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND_B", "_BACKEND_B_FAILED")
    if _BACKEND_B is not None:
        return _BACKEND_B
    if _BACKEND_B_FAILED is not None:
        return None
    p = object_path(SK1_HSACO_PATH_B, arch)
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(p):
            _BACKEND_B_FAILED = "missing %s" % p
            return None
        import torch
        _BACKEND_B = Sk1Attn(p, HipRuntime(), symbols=SK1_SYMBOLS_B,
                             with_s=True, strided=False, head_dim=128, qo_dtype=torch.bfloat16)
        return _BACKEND_B
    except Exception as e:                                   # pragma: no cover
        _BACKEND_B_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def _load_bn(arch=None):
    """The bf16 object `attn_fp8qk_bf16_strided` (strided-HND / NHD, 18-argument), or `None`."""
    global _BACKEND_BN, _BACKEND_BN_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND_BN", "_BACKEND_BN_FAILED")
    if _BACKEND_BN is not None:
        return _BACKEND_BN
    if _BACKEND_BN_FAILED is not None:
        return None
    p = object_path(SK1_HSACO_PATH_BN, arch)
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(p):
            _BACKEND_BN_FAILED = "missing %s" % p
            return None
        import torch
        _BACKEND_BN = Sk1Attn(p, HipRuntime(), symbols=SK1_SYMBOLS_BN,
                              with_s=True, strided=True, head_dim=128, qo_dtype=torch.bfloat16)
        return _BACKEND_BN
    except Exception as e:                                   # pragma: no cover
        _BACKEND_BN_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def bf16_status() -> dict:
    """The state of the two bf16 objects without loading anything. Never raises.

    Same keys as `int8_status()`.
    """
    def one(path, name, loaded, failed):
        out = dict(file=name, present=os.path.isfile(path), sha256=None, size=None,
                   loaded=(loaded is not None), failed=failed)
        try:
            with open(path, "rb") as fh:
                import hashlib
                h = hashlib.sha256()
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
                out["sha256"] = h.hexdigest()
            out["size"] = os.path.getsize(path)
        except Exception:
            pass
        return out
    return dict(enabled=bool(SK1_BF16), strict=bool(SK1_BF16_STRICT),
                default_on=bool(SK1_BF16_DEFAULT_ON),
                contig=one(SK1_HSACO_PATH_B, SK1_HSACO_NAME_B, _BACKEND_B, _BACKEND_B_FAILED),
                strided=one(SK1_HSACO_PATH_BN, SK1_HSACO_NAME_BN, _BACKEND_BN,
                            _BACKEND_BN_FAILED))


def _envelope_ok(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, smooth_k, smooth_v,
                 attn_mask):
    import torch
    if tensor_layout not in ("HND", "NHD"):
        return "layout"
    if return_lse:
        return "return_lse"
    if attn_mask is not None:
        return "attn_mask"
    if smooth_v:
        return "smooth_v"
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return "device"
    if q.device != k.device or q.device != v.device:
        return "device"
    # fp16 and bf16 are both 16 bits wide, so a mixed q/k/v triple would be silently misread
    if not (q.dtype == k.dtype == v.dtype):
        return "dtype"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return "dtype"
    if not (q.dim() == 4 and k.dim() == 4 and v.dim() == 4):
        return "rank"
    if tensor_layout == "NHD":
        qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    else:
        qq, kk, vv = q, k, v
    # The stride rule lives in `sk1_loader.strided_view_reason`, so this envelope can never accept a
    # pattern the loader's own contract check would refuse.
    for nm, t in (("q", qq), ("k", kk), ("v", vv)):
        r = _strided_reason(t, nm)
        if r is not None:
            return r
    q, k, v = qq, kk, vv
    # the three tensors must agree on head_dim, or a D=64 q against a D=128 k would reach the raw launch
    if q.size(3) not in (64, 128) or k.size(3) != q.size(3) or v.size(3) != q.size(3):
        return "head_dim"
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        return "batch"
    if q.size(1) != k.size(1) or q.size(1) != v.size(1):
        return "heads"
    # the kernel takes a single `N` from `q`: a longer K would be silently truncated and a shorter
    # K would read out of bounds
    if q.size(2) != k.size(2):
        return "seq_len"
    if k.size(2) != v.size(2):
        return "seq_len"
    if q.size(2) < 1 or k.size(2) < 1:
        return "seq_len"
    return None


def try_sk1_t1(q, k, v, *, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False,
               smooth_k=True, smooth_v=False, attn_mask=None, strict=False, int8=None, bf16=None,
               i8pv=None):
    """Run the prebuilt SK1 kernel if the call is inside `ENVELOPE` and the platform supports it.

    Returns `(out, reason)` where `out` is the attention output in the caller's (B,H,N,D) shape, or
    `(None, reason)` when the call is not served and the caller must fall back to the shipped path.

    `strict=False` (the default): an unsupported arch, an unresolvable HIP runtime, a code object
    that will not load, a prologue failure or a launch failure falls back with at most one short log
    line per process and never raises. `strict=True` (`SAGEATTN_SK1_BACKEND=1`, or
    `sk1_backend=True`): the same failures raise, since silently falling back on an explicitly
    demanded kernel would hide a defect.

    `int8=None` reads `SK1_INT8` (`SAGEATTN_SK1_INT8`); `True/False` forces it for this call. The int8
    objects are fp16, head_dim 128 only, so when not strict a call they cannot serve falls back to
    the fp8 objects, and if those cannot serve it either this returns `(None, <the fp8 route's own
    reason>)`, never an int8-specific one. `int8=True` is strict for the int8 path: the same call
    raises. A bf16 call never reaches the int8 path.

    `bf16=None` reads `SK1_BF16` (`SAGEATTN_SK1_BF16`); `True/False` forces it for this call. A bf16
    call with the bf16 objects off is refused with `reason="dtype"`; `head_dim == 64` is refused
    with `bf16_d128_only`.
    """
    reason = _envelope_ok(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, smooth_k,
                          smooth_v, attn_mask)
    if reason is not None:
        return None, reason
    # `arch_supported` is called as a module attribute so a test that replaces it to simulate another
    # device is honoured; `selected_arch()` alone would ignore it.
    if not arch_supported(q.device):
        if strict:
            raise RuntimeError(
                "SK1 backend forced on (SAGEATTN_SK1_BACKEND=1 / sk1_backend=True) but the device "
                "arch is %r; the packaged code objects support %s (gfx1200 needs "
                "SAGEATTN_SK1_GFX1200=1)"
                % (device_arch(q.device) or "<unknown>", ", ".join(SUPPORTED_ARCHS)))
        _log_once(_arch_refusal_reason(q.device))
        return None, "arch"
    # the object set this call uses, derived from the decision the check above approved
    _arch = selected_arch(q.device) or _DEFAULT_ARCH
    # Routing: contiguous HND (all three tensors) goes to `attn_fp8qk_f16`; everything else to the
    # strided object.
    if tensor_layout == "NHD":
        qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    else:
        qq, kk, vv = q, k, v
    contiguous_hnd = bool(tensor_layout == "HND" and q.is_contiguous() and k.is_contiguous()
                          and v.is_contiguous())
    d64 = int(qq.size(3)) == 64
    # bf16 has its own object pair (the int8 and fp16 pairs are fp16 only), so it takes precedence
    # over int8; with `SK1_BF16` off a bf16 call returns `"dtype"`
    import torch
    is_bf16 = bool(q.dtype == torch.bfloat16)
    if bf16 is None:
        want_b, b_strict = bool(SK1_BF16), bool(SK1_BF16_STRICT)
    else:
        want_b, b_strict = bool(bf16), bool(bf16)
    if is_bf16 and not want_b:
        if b_strict or strict:
            raise RuntimeError(
                "bf16 SK1 was demanded (SAGEATTN_SK1_BF16=1 / sk1_bf16=True) but the bf16 objects "
                "are switched off; set SAGEATTN_SK1_BF16=1")
        _log_once("bf16 input but the bf16 objects are switched off")
        return None, "dtype"
    if is_bf16 and d64:
        # a D=64 call on a D=128 object would read the wrong KSTRIDE
        if b_strict or strict:
            raise RuntimeError(
                "the bf16 SK1 objects (%s / %s) are head_dim 128 only; head_dim is 64"
                % (SK1_HSACO_NAME_B, SK1_HSACO_NAME_BN))
        _log_once("bf16 object is head_dim 128 only (head_dim=64)")
        return None, "bf16_d128_only"
    # The int8 P.V objects first; a call they do not take continues to the objects below. An fp16 call with int8
    # off (`int8=False` / `SAGEATTN_SK1_INT8=0`) asked for the fp8 Q.K objects, so it skips them.
    want_v = bool(SK1_I8PV) if i8pv is None else bool(i8pv)
    v_strict = SK1_I8PV_STRICT if i8pv is None else bool(i8pv)
    int8_on = bool(SK1_INT8) if int8 is None else bool(int8)
    if want_v and v_strict and (d64 or not (is_bf16 or int8_on)):
        raise RuntimeError("the int8 P.V objects were demanded (SAGEATTN_SK1_I8PV=1 / sk1_i8pv=True) but %s"
                           % ("they are head_dim 128 only" if d64 else "int8 is off for this fp16 call"))
    if want_v and not d64 and (is_bf16 or int8_on):
        sm_v = float(sm_scale if sm_scale is not None else qq.size(3) ** -0.5)
        out, _ = _try_i8pv(qq, kk, vv, tensor_layout, contiguous_hnd, is_causal, sm_v, bool(smooth_k), _arch,
                           v_strict)
        if out is not None:
            return out, None
    if int8 is None:
        want_i8, i8_strict = bool(SK1_INT8), bool(SK1_INT8_STRICT)
    else:
        want_i8, i8_strict = bool(int8), bool(int8)

    # Re-enters this function with `int8=False` and returns that call's `(out, reason)` as is, so
    # the recursion is one level deep. `strict` never reaches it: every refusal raises first.
    def _i8_fallback():
        return try_sk1_t1(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal,
                          sm_scale=sm_scale, return_lse=return_lse, smooth_k=smooth_k,
                          smooth_v=smooth_v, attn_mask=attn_mask, strict=strict, int8=False,
                          bf16=bf16)

    if want_i8 and d64 and not is_bf16:
        # a D=64 call on a D=128 object would read the wrong KSTRIDE
        if i8_strict or strict:
            raise RuntimeError(
                "the int8 SK1 objects (%s / %s) are head_dim 128 only; head_dim is 64"
                % (SK1_HSACO_NAME_I8, SK1_HSACO_NAME_I8N))
        _log_once("int8 object is head_dim 128 only (head_dim=64)")
        return _i8_fallback()
    if d64:
        if strict:
            raise RuntimeError("no head_dim-64 SK1 object is shipped")
        _log_once("no head_dim-64 object is shipped")
        return None, "no_d64_object"
    try:
        if is_bf16:
            attn = _load_b(_arch) if contiguous_hnd else _load_bn(_arch)
        elif want_i8:
            attn = _load_i8(_arch) if contiguous_hnd else _load_i8n(_arch)
        else:
            attn = _load(_arch) if contiguous_hnd else _load_n(_arch)
    except Exception as e:
        if strict or (want_i8 and not is_bf16 and i8_strict):
            # an explicitly demanded int8 object must not silently serve fp8
            raise
        if want_i8 and not is_bf16:
            _log_once("int8 load failed, falling back to the fp8 SK1 objects: %s" % e)
            return _i8_fallback()
        _log_once("load failed: %s" % e)
        return None, "load_failed"
    if attn is None:
        if is_bf16:
            name = SK1_HSACO_NAME_B if contiguous_hnd else SK1_HSACO_NAME_BN
            fail = _BACKEND_B_FAILED if contiguous_hnd else _BACKEND_BN_FAILED
            if b_strict or strict:
                raise RuntimeError("the bf16 SK1 object %s is not installed (%s)" % (name, fail))
            _log_once("bf16 object unavailable: %s" % fail)
            return None, "no_bf16_object"
        if want_i8:
            name = SK1_HSACO_NAME_I8 if contiguous_hnd else SK1_HSACO_NAME_I8N
            fail = _BACKEND_I8_FAILED if contiguous_hnd else _BACKEND_I8N_FAILED
            if i8_strict or strict:
                raise RuntimeError("the int8 SK1 object %s is not installed (%s)" % (name, fail))
            _log_once("int8 object unavailable (%s), falling back to the fp8 SK1 objects" % fail)
            return _i8_fallback()
        if strict:
            raise RuntimeError("the widened SK1 object %s is not installed (%s)"
                               % (SK1_HSACO_NAME_N, _BACKEND_N_FAILED))
        _log_once("widened object unavailable: %s" % _BACKEND_N_FAILED)
        return None, "no_strided_object"
    # Everything below can fail for reasons unrelated to the code object (a partial install, an
    # allocation failure), and the default path must stay raise-free up to the launch. The Triton
    # prologue modules are imported only where they are the fallback: importing Triton costs ~0.3 s
    # on the first call.
    try:
        import torch
        B, H, N, D = qq.shape
        sm = float(sm_scale if sm_scale is not None else D ** -0.5)
        n_pad = (N + 63) // 64 * 64
        BH = B * H
        hp = _hip_prologue(_arch)   # `None` unless the HIP prologue is selected and loaded
        if is_bf16:
            # The Triton K/V kernels widen every load to fp32 (exact for bf16), so they serve bf16
            # as is. `out` is bf16, the kernel's own O type.
            if contiguous_hnd:
                qr = q.reshape(BH, N, D).contiguous()
                res = (hp.quant_kv(k, v, n_pad=n_pad, smooth_k=bool(smooth_k), int8=False)
                       if hp is not None else None)
                if res is None:
                    from . import quant_kv_v8t_b as QB
                    res = QB.prologue_bf16_fused(k, v, n_pad=n_pad, smooth_k=bool(smooth_k))
                k8, sk, v8t, sv = res
                out = torch.empty(BH, N, D, device=q.device, dtype=q.dtype)
            else:
                res = (hp.quant_kv(kk, vv, n_pad=n_pad, smooth_k=bool(smooth_k), int8=False)
                       if hp is not None else None)
                if res is None:
                    from . import quant_kv_v8t_n as QN
                    res = QN.prologue_fp8_fused_strided(kk, vv, n_pad=n_pad,
                                                        smooth_k=bool(smooth_k))
                k8, sk, v8t, sv = res
                if tensor_layout == "NHD":
                    out = torch.empty(B, N, H, D, device=q.device, dtype=q.dtype)
                else:
                    out = torch.empty(BH, N, D, device=q.device, dtype=q.dtype)
        elif want_i8:
            # K -> int8 (scale amax/127, round-to-nearest-even, clamp), V -> fp8 e4m3
            if contiguous_hnd:
                qr = q.reshape(BH, N, D).contiguous()
                res = (hp.quant_kv(k, v, n_pad=n_pad, smooth_k=bool(smooth_k), int8=True)
                       if hp is not None else None)
                if res is None:
                    from . import quant_kv_v8t_i8_n as QI8N
                    res = QI8N.prologue_i8_fused_contig(k, v, n_pad=n_pad,
                                                        smooth_k=bool(smooth_k))
                k8, sk, v8t, sv = res
                out = torch.empty(BH, N, D, device=q.device, dtype=torch.float16)
            else:
                res = (hp.quant_kv(kk, vv, n_pad=n_pad, smooth_k=bool(smooth_k), int8=True)
                       if hp is not None else None)
                if res is None:
                    from . import quant_kv_v8t_i8_n as QI8N
                    res = QI8N.prologue_i8_fused_strided(kk, vv, n_pad=n_pad,
                                                         smooth_k=bool(smooth_k))
                k8, sk, v8t, sv = res
                if tensor_layout == "NHD":
                    out = torch.empty(B, N, H, D, device=q.device, dtype=torch.float16)
                else:
                    out = torch.empty(BH, N, D, device=q.device, dtype=torch.float16)
        elif contiguous_hnd:
            qr = q.reshape(BH, N, D).contiguous()
            res = (hp.quant_kv(k, v, n_pad=n_pad, smooth_k=bool(smooth_k), int8=False)
                   if hp is not None else None)
            if res is None:
                from . import quant_kv_v8t as QV
                _, _, k8, sk, v8t, sv = QV.prologue_fp8_fused(q, k, v, n_pad=n_pad,
                                                              smooth_k=bool(smooth_k))
            else:
                k8, sk, v8t, sv = res
            out = torch.empty(BH, N, D, device=q.device, dtype=torch.float16)
        else:
            # reads K/V straight out of the strided view (no copy); Q is not quantised
            res = (hp.quant_kv(kk, vv, n_pad=n_pad, smooth_k=bool(smooth_k), int8=False)
                   if hp is not None else None)
            if res is None:
                from . import quant_kv_v8t_n as QN
                res = QN.prologue_fp8_fused_strided(kk, vv, n_pad=n_pad,
                                                    smooth_k=bool(smooth_k))
            k8, sk, v8t, sv = res
            if tensor_layout == "NHD":
                out = torch.empty(B, N, H, D, device=q.device, dtype=torch.float16)
            else:
                out = torch.empty(BH, N, D, device=q.device, dtype=torch.float16)
        sv_s = fold_v_scale(sv) if attn.with_s else None
    except Exception as e:                                          # noqa: BLE001
        if strict or (want_i8 and not is_bf16 and i8_strict):
            raise
        if want_i8 and not is_bf16:
            # A broken int8 prologue must not cost the call its fp8 SK1 service.
            _log_once("int8 prologue failed, falling back to the fp8 SK1 objects: %s" % e)
            return _i8_fallback()
        _log_once("prologue failed: %s" % e)
        return None, "prologue_failed"
    try:
        if contiguous_hnd:
            attn.launch(qr, k8, sk, v8t, sv, None, out, H, N, n_pad, sm, bool(is_causal), s=sv_s)
        else:
            # addressed in the logical (B, H, N, D) reading; for NHD that is the transpose view
            attn.launch_strided(qq, k8, sk, v8t, sv, None,
                                out.view(B, H, N, D) if tensor_layout == "HND"
                                else out.transpose(1, 2),
                                H, N, n_pad, sm, bool(is_causal), s=sv_s)
    except Exception as e:
        if strict or (want_i8 and not is_bf16 and i8_strict):
            raise
        if want_i8 and not is_bf16:
            _log_once("int8 launch failed, falling back to the fp8 SK1 objects: %s" % e)
            return _i8_fallback()
        _log_once("launch failed: %s" % e)
        return None, "launch_failed"
    if tensor_layout == "NHD":
        return out, None
    return out.view(B, H, N, D), None
