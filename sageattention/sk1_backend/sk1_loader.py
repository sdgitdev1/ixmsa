#!/usr/bin/env python3
"""Load and launch a raw `.hsaco` from Python with ctypes, on torch's own HIP runtime.

A code object has no C++ imports, so the `c10` symbol ABI mismatch that breaks a `.pyd` extension
built against a different torch cannot arise; everything here is a plain C-ABI call.

1. Same HIP runtime DLL as torch. `hipModuleLoadData` must be called on the already-loaded
   `amdhip64_7.dll`, or device pointers and streams are not interchangeable. The ROCm SDK's
   `_rocm_sdk_core` and `_rocm_sdk_devel` copies are different builds, so `rocm_bin_dir()` resolves
   to the image that is already loaded and `HipRuntime.same_as_torch` records the evidence.
2. Torch's current stream, passed as a plain integer handle, so the kernel is ordered against the
   torch ops that produced its inputs.
3. Argument lifetime. HIP reads `kernelParams` asynchronously, so the argument values must outlive
   the launch; `launch()` returns them and callers keep them until the next call.

Environment: `SK1_ROCM_BIN` (extra directory to search for the HIP runtime) and `SK1_HIP_DLL`
(explicit path to the runtime library).
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys

_IS_WINDOWS = (os.name == "nt")
try:
    from ctypes import wintypes
except ImportError:  # non-Windows: the Linux paths below do not need it
    wintypes = None

HIP_SUCCESS = 0

# hipModuleLaunchKernel(hipFunction_t, gx, gy, gz, bx, by, bz, sharedMemBytes, stream,
#                       kernelParams, extra): exactly 11 entries in declaration order
_LAUNCH_ARGTYPES = ([ctypes.c_void_p] + [ctypes.c_uint] * 6 +
                    [ctypes.c_uint, ctypes.c_void_p,
                     ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)])


def _enum_amdhip_modules() -> list:
    """Every loaded module whose base name starts with `amdhip`: full path + HMODULE."""
    if not _IS_WINDOWS:
        return _enum_amdhip_modules_linux()
    assert wintypes is not None  # Windows-only branch
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    proc = k32.GetCurrentProcess()
    psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                         ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE,
                                           wintypes.LPWSTR, ctypes.c_uint]
    arr = (wintypes.HMODULE * 2048)()
    need = ctypes.c_uint()
    if not psapi.EnumProcessModules(proc, arr, ctypes.sizeof(arr), ctypes.byref(need)):
        return []
    out = []
    buf = ctypes.create_unicode_buffer(4096)
    for i in range(need.value // ctypes.sizeof(wintypes.HMODULE)):
        psapi.GetModuleFileNameExW(proc, arr[i], buf, 4096)
        p = buf.value
        if os.path.basename(p).lower().startswith("amdhip"):
            out.append(dict(path=p, hmodule=ctypes.cast(arr[i], ctypes.c_void_p).value))
    return out


def _enum_amdhip_modules_linux() -> list:
    """Linux equivalent: mappings in /proc/self/maps whose file is libamdhip*. `hmodule` is None."""
    out = []
    seen = set()
    try:
        with open("/proc/self/maps", "r") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) < 6:
                    continue
                p = parts[5].strip()
                if p.endswith(" (deleted)"):
                    continue
                if p in seen or not os.path.basename(p).lower().startswith("libamdhip"):
                    continue
                seen.add(p)
                out.append(dict(path=p, hmodule=None))
    except OSError:
        pass
    return out


def _hip_lib_names() -> tuple:
    """HIP runtime file names, most-preferred first, for this platform."""
    if _IS_WINDOWS:
        return ("amdhip64_7.dll",)
    return ("libamdhip64.so.7", "libamdhip64.so")


def _candidate_bins() -> list:
    cands = []
    d = os.environ.get("SK1_ROCM_BIN")
    if d:
        cands.append(d)
    import site
    for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
        # `_rocm_sdk_core` first: that is the copy torch loads
        if _IS_WINDOWS:
            cands.append(os.path.join(sp, "_rocm_sdk_core", "bin"))
            cands.append(os.path.join(sp, "_rocm_sdk_devel", "bin"))
        else:
            cands.append(os.path.join(sp, "_rocm_sdk_core", "lib"))
            cands.append(os.path.join(sp, "_rocm_sdk_devel", "lib"))
    if not _IS_WINDOWS:
        cands.extend(glob.glob("/opt/rocm-*/lib") + ["/opt/rocm/lib", "/usr/lib/x86_64-linux-gnu"])
    return cands


def rocm_bin_dir() -> str:
    """Directory holding the `amdhip64_7.dll` that torch already loaded, else the best fallback.

    Windows' loader keys on the resolved path, so loading the DLL from another directory yields a
    second, independent HIP runtime in the process; hence the already-loaded image is preferred.
    """
    override = os.environ.get("SK1_HIP_DLL")
    if override and os.path.isfile(override):
        return os.path.dirname(override)
    # let torch load its own copy before we enumerate
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.current_stream()
    except Exception:
        pass
    for m in _enum_amdhip_modules():
        return os.path.dirname(m["path"])
    for c in _candidate_bins():
        if not c:
            continue
        for name in _hip_lib_names():
            if os.path.isfile(os.path.join(c, name)):
                return c
    raise RuntimeError("%s not found; set SK1_ROCM_BIN or SK1_HIP_DLL"
                       % ("/".join(_hip_lib_names()),))


class HipRuntime:
    """Thin ctypes binding to the HIP runtime that is already loaded in this process."""

    DLL_NAME = "amdhip64_7.dll" if _IS_WINDOWS else "libamdhip64.so.7"

    def __init__(self, bin_dir: str | None = None):
        self.bin_dir = bin_dir or rocm_bin_dir()
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(self.bin_dir)   # the runtime's sibling DLLs
        if _IS_WINDOWS:
            assert wintypes is not None  # Windows-only branch
            self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
            self._k32.GetModuleHandleW.restype = wintypes.HMODULE
        else:
            self._k32 = None
        self.dll_path = os.path.join(self.bin_dir, self.DLL_NAME)
        if _IS_WINDOWS:
            self.hip = ctypes.WinDLL(self.dll_path)
        else:
            # RTLD_GLOBAL mirrors WinDLL's process-global visibility
            self.hip = ctypes.CDLL(self.dll_path, mode=ctypes.RTLD_GLOBAL)
        self._bind()
        self.same_as_torch = self._verify_same_module()
        self.modules: list = []          # keep modules (and their images) alive
        self._keepalive = None           # the last launch's argument storage

    # ---------------------------------------------------------------- binding
    def _bind(self):
        h = self.hip
        h.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        h.hipModuleLoadData.restype = ctypes.c_int
        h.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                           ctypes.c_char_p]
        h.hipModuleGetFunction.restype = ctypes.c_int
        h.hipModuleLaunchKernel.argtypes = _LAUNCH_ARGTYPES
        h.hipModuleLaunchKernel.restype = ctypes.c_int
        h.hipModuleUnload.argtypes = [ctypes.c_void_p]
        h.hipModuleUnload.restype = ctypes.c_int
        for fn in ("hipGetErrorString", "hipGetErrorName"):
            f = getattr(h, fn)
            f.argtypes = [ctypes.c_int]
            f.restype = ctypes.c_char_p
        h.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        h.hipStreamSynchronize.restype = ctypes.c_int

    def err(self, rc: int) -> str:
        try:
            name = self.hip.hipGetErrorName(rc).decode()
            msg = self.hip.hipGetErrorString(rc).decode()
            return "%d (%s: %s)" % (rc, name, msg)
        except Exception:
            return str(rc)

    def loaded_amdhip_modules(self) -> list:
        return _enum_amdhip_modules()

    def _verify_same_module(self) -> dict:
        """Evidence that ctypes and torch share one HIP runtime image.

        Equal `WinDLL` and `GetModuleHandleW` handles (both keyed on the module name) mean Windows
        returned the already-loaded image rather than mapping a second one.
        """
        got = ctypes.cast(self.hip._handle, ctypes.c_void_p).value
        if _IS_WINDOWS:
            assert wintypes is not None  # Windows-only branch
            want = self._k32.GetModuleHandleW(self.DLL_NAME)
        else:
            # dlopen of an already-loaded soname returns the same handle
            try:
                probe = ctypes.CDLL(self.DLL_NAME)
                want = ctypes.cast(probe._handle, ctypes.c_void_p).value
            except OSError:
                want = None
        mods = _enum_amdhip_modules()
        rec = dict(dll=self.DLL_NAME, bin_dir=self.bin_dir, dll_path=self.dll_path,
                   ctypes_hmodule=got, getmodulehandle_hmodule=want,
                   handle_equal=bool(got and want and got == want),
                   loaded_amdhip_modules=mods, n_amdhip_images=len(mods))
        try:
            import torch
            rec["torch_version"] = torch.__version__
            rec["torch_hip_version"] = getattr(torch.version, "hip", None)
            rec["torch_cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                rec["stream_handle"] = int(torch.cuda.current_stream().cuda_stream)
        except Exception as e:                                    # pragma: no cover
            rec["torch_error"] = repr(e)
        return rec

    # ---------------------------------------------------------------- module / function
    def load_module(self, hsaco):
        """`hsaco`: path or raw bytes of an `elf64-amdgpu` code object (not a `__CL` container)."""
        if isinstance(hsaco, (str, os.PathLike)):
            with open(hsaco, "rb") as f:
                image = f.read()
        else:
            image = bytes(hsaco)
        if image[:4] != b"\x7fELF":
            raise RuntimeError("hipModuleLoadData wants a raw ELF; got magic %r "
                               "(unbundle the clang-offload-bundler container first)" % image[:4])
        buf = ctypes.create_string_buffer(image, len(image))       # HIP does not copy
        mod = ctypes.c_void_p()
        rc = self.hip.hipModuleLoadData(ctypes.byref(mod), buf)
        if rc:
            raise RuntimeError("hipModuleLoadData rc=%s" % self.err(rc))
        self.modules.append((mod, buf, image))                     # keep alive
        return mod

    def get_function(self, mod, symbol: str):
        fn = ctypes.c_void_p()
        rc = self.hip.hipModuleGetFunction(ctypes.byref(fn), mod, symbol.encode())
        if rc:
            raise RuntimeError("hipModuleGetFunction(%r) rc=%s" % (symbol, self.err(rc)))
        return fn

    # ---------------------------------------------------------------- launch
    def launch(self, fn, grid, block, shared_bytes, stream, args):
        """`args`: list of ctypes values in the KERNEL's declaration order.

        HIP dereferences `kernelParams` asynchronously, so both the pointer array and the values
        must outlive the launch. They are kept in `self._keepalive` as well as returned.
        """
        vals = [ctypes.addressof(v) for v in args]
        arr = (ctypes.c_void_p * len(vals))(*vals)
        rc = self.hip.hipModuleLaunchKernel(fn, *[ctypes.c_uint(x) for x in grid],
                                            *[ctypes.c_uint(x) for x in block],
                                            ctypes.c_uint(shared_bytes),
                                            ctypes.c_void_p(int(stream)), arr, None)
        if rc:
            raise RuntimeError("hipModuleLaunchKernel rc=%s" % self.err(rc))
        self._keepalive = (args, vals, arr)
        return vals, arr

    def sync(self, stream=0):
        rc = self.hip.hipStreamSynchronize(ctypes.c_void_p(int(stream)))
        if rc:
            raise RuntimeError("hipStreamSynchronize rc=%s" % self.err(rc))


def torch_stream():
    import torch
    return int(torch.cuda.current_stream().cuda_stream)


#: Every reason `strided_view_reason` can return, in evaluation order. `sk1_backend._envelope_ok`
#: returns them verbatim.
STRIDED_REFUSALS = ("rank", "stride_last", "stride_nonpos", "stride_align", "ptr_align",
                    "stride_overlap", "stride_int32")
_INT32_MAX = 2 ** 31 - 1


def strided_view_reason(t, name="t"):
    """`None` if `t` is a servable logical `(B, H, N, D)` view, else a reason string.

    The one place the acceptance rule lives; `_strided_triple` (raises) and
    `sk1_backend._envelope_ok` (returns the string) both call it, so the envelope can never accept
    a pattern the raw launch would refuse. Real models hand the kernel views that are neither
    contiguous HND nor physical NHD (MiniMax-H3's q/k/v are views into one fused `qkv_proj` output,
    row stride `3*H*D`), so the rule is the launch's own requirements:

    1. `rank`            -- not 4-D.
    2. `stride_last`     -- `stride(3) != 1`; the kernel's `Q` access is a 16-byte `f16x8` load.
    3. `stride_nonpos`   -- any stride <= 0; a zero stride aliases (`expand`) and the kernel adds
       the head/batch strides unchecked.
    4. `stride_align`    -- an outer stride not a multiple of 8 elements, so that every
       `(b, h, row)` start is 16-byte aligned. Needed for `Q`; conservative for `K`/`V`/`O`.
    5. `ptr_align`       -- `data_ptr() % 16`; same 16-byte load.
    6. `stride_overlap`  -- the index map is not injective. Drop size-1 dims, sort by ascending
       stride, require `stride[k+1] >= stride[k] * size[k]`. Sufficient, not necessary: it may
       refuse a non-overlapping pattern but never accepts an overlapping one. Size-1 dims must be
       dropped: `unsqueeze(0)` gives `B = 1` a batch stride smaller than the row stride.
    7. `stride_int32`    -- any stride above `2**31 - 1`; the launch passes strides as 32-bit
       arguments. Not covered: Triton's int32 element offsets could still wrap if a stride times
       `N` overflows.
    """
    if t.dim() != 4:
        return "rank"
    if int(t.stride(3)) != 1:
        return "stride_last"
    st = [int(s) for s in t.stride()]
    if any(s <= 0 for s in st):
        return "stride_nonpos"
    # stride(3) == 1 is already required, so only the outer three are checked
    if any(st[i] % 8 for i in (0, 1, 2)):
        return "stride_align"
    if int(t.data_ptr()) % 16:
        return "ptr_align"
    sh = [int(x) for x in t.shape]
    dims = sorted((st[i], sh[i]) for i in range(4) if sh[i] > 1)
    for (s0, n0), (s1, _n1) in zip(dims, dims[1:]):
        if s1 < s0 * n0:
            return "stride_overlap"
    if any(s > _INT32_MAX for s in st):
        return "stride_int32"
    return None


def _strided_triple(t, name):
    """`(row, head, batch)` element strides of a logical `(B, H, N, D)` tensor, plus a layout tag.

    Legal views are exactly those `strided_view_reason` accepts; the launch is raw and nothing
    downstream bounds-checks, so anything else is refused with the guard's own reason. Tags:
    `"hnd_contig"` (contiguous HND), `"nhd_phys"` (physical NHD read as HND, i.e.
    `transpose(1, 2).is_contiguous()`) and `"general"` (e.g. MiniMax-H3's fused-QKV view).
    """
    r = strided_view_reason(t, name)
    if r is not None:
        if r == "rank":
            raise ValueError("%s must be 4-D (B,H,N,D); got rank %d" % (name, t.dim()))
        raise ValueError("%s is not a servable (B,H,N,D) view: %s (shape=%s strides=%s)"
                         % (name, r, tuple(t.shape), tuple(t.stride())))
    if t.is_contiguous():
        tag = "hnd_contig"
    elif t.transpose(1, 2).is_contiguous():
        tag = "nhd_phys"
    else:
        tag = "general"
    return int(t.stride(2)), int(t.stride(1)), int(t.stride(0)), tag


class Sk1Attn:
    """SK1's two entry points behind a torch-tensor interface.

    Tensor contract (exactly the kernel's): Q (B*H, N, 128) fp16 or bf16 RAW; K8 (B*H, N_pad, 128)
    e4m3 zero-padded; SK (B*H, N_pad) fp32; V8T (B*H, 128, N_pad) e4m3 zero-padded (V TRANSPOSED);
    SV (B*H, N_pad) fp32; VMEAN (B*H, 128) fp32 or None; O (B*H, N, 128) same dtype as Q.
    `H` is the head count used to fold (b, h) out of `bh`; N_pad must be a multiple of 64.

    `qo_dtype` is the Q/O dtype the object was built for, defaulting from the symbol pair
    (`attn_fp8qk_bf16*` -> bf16, else fp16): fp16 objects refuse a bf16 call and vice versa.

    Every packaged object takes a 12th argument `S` (B*H,) fp32, the per-`(b,h)` rescale
    `clamp_min(max_j sv(j), 1)` the prologue applied to `SV`, between `VMEAN` and `O`. `with_s`
    selects it and defaults to `True`.
    """

    DEFAULT_SYMBOLS = ("attn_fp8qk_f16_fwd", "attn_fp8qk_f16_fwd_causal")

    #: Kernel argument order of the `attn_fp8qk_f16` object
    ARG_ORDER_WITH_S = ("Q", "K8", "SK", "V8T", "SV", "VMEAN", "S", "O", "H", "N", "N_pad", "sm")

    #: `attn_fp8qk_f16_strided` takes six extra integers after `sm_scale`: the `(row, head, batch)`
    #: element strides of `Q` and `O`. Contiguous HND is still routed to `attn_fp8qk_f16`
    #: (routing is in `__init__.py`).
    ARG_ORDER_STRIDED = ("Q", "K8", "SK", "V8T", "SV", "VMEAN", "S", "O", "H", "N", "N_pad", "sm",
                         "q_rs", "q_hs", "q_bs", "o_rs", "o_hs", "o_bs")

    def __init__(self, hsaco_path: str, rt: HipRuntime | None = None, symbols=None, with_s=None,
                 strided=None, head_dim=None, bm=None, qo_dtype=None):
        """`symbols` = (non-causal, causal) entry points; the default is the `attn_fp8qk_f16` pair.

        `strided`, `head_dim` (64 or 128), `bm` (128 or 256) and `qo_dtype` default from the
        symbol pair, so a caller that only swaps `symbols` cannot launch a D=64 kernel against
        D=128 buffers, a BM=256 kernel with a BM=128 grid, or a bf16 object on fp16 bytes.
        """
        self.rt = rt or HipRuntime()
        self.hsaco_path = hsaco_path
        self.symbols = tuple(symbols) if symbols else self.DEFAULT_SYMBOLS
        if len(self.symbols) != 2:
            raise ValueError("symbols must be (non-causal, causal); got %r" % (self.symbols,))
        self.with_s = True if with_s is None else bool(with_s)
        self.strided = ("_strided_" in self.symbols[0]) if strided is None else bool(strided)
        self.head_dim = int(128 if head_dim is None else head_dim)
        if self.head_dim not in (64, 128):
            raise ValueError("head_dim must be 64 or 128; got %d" % self.head_dim)
        self.bm = int(128 if bm is None else bm)
        if self.bm not in (128, 256):
            raise ValueError("bm must be 128 or 256; got %d" % self.bm)
        if qo_dtype is None:
            import torch
            qo_dtype = (torch.bfloat16 if any("bf16" in s for s in self.symbols) else torch.float16)
        self.qo_dtype = qo_dtype
        self.module = self.rt.load_module(hsaco_path)
        self.funcs = {sym: self.rt.get_function(self.module, sym) for sym in self.symbols}
        self._keep = None

    @staticmethod
    def _check(q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, s=None, with_s=False, head_dim=128,
               qo_dtype=None):
        # The launch is raw: an undersized buffer is a GPU page fault, so refuse any tensor that
        # does not match the contract exactly, before it reaches the device.
        import torch
        qo_dtype = torch.float16 if qo_dtype is None else qo_dtype
        # First: a null `S` is dereferenced by the kernel's `Sb = S[bh]` load
        if with_s and s is None:
            raise ValueError("this SK1 object takes the per-head V scale `S` argument; s=None was passed")
        N, N_pad, H = int(N), int(N_pad), int(H)
        head_dim = int(head_dim)
        if head_dim not in (64, 128):
            raise ValueError("SK1 objects exist only for head_dim in {64,128}; got %d" % head_dim)
        if N < 1 or N_pad % 64 or N_pad < N:
            raise ValueError("need N >= 1, N_pad %% 64 == 0 and N_pad >= N (N=%d, N_pad=%d)"
                             % (N, N_pad))
        bh = q.shape[0]
        want = {
            "q": (q, (bh, N, head_dim), qo_dtype),
            "k8": (k8, (bh, N_pad, head_dim), None),
            "sk": (sk, (bh, N_pad), torch.float32),
            "v8t": (v8t, (bh, head_dim, N_pad), None),
            "sv": (sv, (bh, N_pad), torch.float32),
            "out": (out, (bh, N, head_dim), qo_dtype),
        }
        if vmean is not None:
            want["vmean"] = (vmean, (bh, head_dim), torch.float32)
        if with_s:
            want["s"] = (s, (bh,), torch.float32)
        for name, (t, shape, dtype) in want.items():
            if tuple(t.shape) != shape:
                raise ValueError("%s shape %s != %s" % (name, tuple(t.shape), shape))
            if dtype is not None and t.dtype != dtype:
                raise ValueError("%s dtype %s != %s" % (name, t.dtype, dtype))
            if dtype is None and t.element_size() != 1:
                raise ValueError("%s must be a 1-byte e4m3 tensor, got %s" % (name, t.dtype))
            if not t.is_cuda or not t.is_contiguous():
                raise ValueError("%s must be a contiguous device tensor" % name)
            if t.device != q.device:
                raise ValueError("%s is on %s, q is on %s" % (name, t.device, q.device))

    @staticmethod
    def _check_strided(q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, s=None, head_dim=128,
                       qo_dtype=None, with_s=True):
        """`_check` for the 18-argument object.

        Same contract, except `q` and `out` are logical `(B, H, N, D)` tensors that may be strided
        (`_strided_triple`); the other tensors must still be contiguous. Returns
        `(q_triple, o_triple)` for `launch_strided`. `head_dim` is the object's, not the caller's.
        """
        import torch
        qo_dtype = torch.float16 if qo_dtype is None else qo_dtype
        if with_s and s is None:
            raise ValueError("this SK1 object takes the per-head V scale `S` argument; s=None was passed")
        N, N_pad, H = int(N), int(N_pad), int(H)
        head_dim = int(head_dim)
        if head_dim not in (64, 128):
            raise ValueError("SK1 objects exist only for head_dim in {64,128}; got %d" % head_dim)
        if N < 1 or N_pad % 64 or N_pad < N:
            raise ValueError("need N >= 1, N_pad %% 64 == 0 and N_pad >= N (N=%d, N_pad=%d)"
                             % (N, N_pad))
        if q.dim() != 4 or out.dim() != 4:
            raise ValueError("q/out must be 4-D (B,H,N,D); got ranks %d/%d" % (q.dim(), out.dim()))
        B, Hq, Nq, Dq = (int(x) for x in q.shape)
        if Hq != H:
            raise ValueError("q.shape[1]=%d != H=%d" % (Hq, H))
        if (Nq, Dq) != (N, head_dim):
            raise ValueError("q is %s, expected (B,%d,%d,%d)" % (tuple(q.shape), H, N, head_dim))
        if tuple(out.shape) != (B, H, N, head_dim):
            raise ValueError("out is %s, expected (B,%d,%d,%d)"
                             % (tuple(out.shape), H, N, head_dim))
        bh = B * H
        qtr = _strided_triple(q, "q")
        otr = _strided_triple(out, "out")
        # the Q load is a 16-byte `f16x8`: every (b,h,row) start must be 16-byte aligned
        for nm, tr in (("q", qtr), ("out", otr)):
            rs, hs, bs = tr[0], tr[1], tr[2]
            if rs <= 0 or hs <= 0 or bs <= 0:
                raise ValueError("%s strides must be positive; got %s" % (nm, (rs, hs, bs)))
            if rs % 8 or hs % 8 or bs % 8:
                raise ValueError("%s strides must be multiples of 8 elements (16-byte Q loads); "
                                 "got %s" % (nm, (rs, hs, bs)))
        if q.data_ptr() % 16:
            raise ValueError("q.data_ptr()=0x%x is not 16-byte aligned" % q.data_ptr())
        want = (
            ("q", q, (B, H, N, head_dim), qo_dtype, False),
            ("out", out, (B, H, N, head_dim), qo_dtype, False),
            ("k8", k8, (bh, N_pad, head_dim), None, True),
            ("sk", sk, (bh, N_pad), torch.float32, True),
            ("v8t", v8t, (bh, head_dim, N_pad), None, True),
            ("sv", sv, (bh, N_pad), torch.float32, True),
        )
        for name, t, shape, dtype, need_contig in want:
            if tuple(t.shape) != shape:
                raise ValueError("%s shape %s != %s" % (name, tuple(t.shape), shape))
            if dtype is not None and t.dtype != dtype:
                raise ValueError("%s dtype %s != %s" % (name, t.dtype, dtype))
            if dtype is None and t.element_size() != 1:
                raise ValueError("%s must be a 1-byte e4m3 tensor, got %s" % (name, t.dtype))
            if not t.is_cuda:
                raise ValueError("%s must be a device tensor" % name)
            if t.device != q.device:
                raise ValueError("%s is on %s, q is on %s" % (name, t.device, q.device))
            if need_contig and not t.is_contiguous():
                raise ValueError("%s must be a contiguous device tensor" % name)
        if vmean is not None:
            if tuple(vmean.shape) != (bh, head_dim) or vmean.dtype != torch.float32:
                raise ValueError("vmean must be (%d,%d) fp32; got %s %s"
                                 % (bh, head_dim, tuple(vmean.shape), vmean.dtype))
            if not vmean.is_cuda or not vmean.is_contiguous() or vmean.device != q.device:
                raise ValueError("vmean must be a contiguous device tensor on q's device")
        if with_s:
            if tuple(s.shape) != (bh,) or s.dtype != torch.float32:
                raise ValueError("s must be (%d,) fp32; got %s %s" % (bh, tuple(s.shape), s.dtype))
            if not s.is_cuda or not s.is_contiguous() or s.device != q.device:
                raise ValueError("s must be a contiguous device tensor on q's device")
        return qtr, otr

    def launch(self, q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, sm_scale, causal,
               stream=None, s=None):
        self._check(q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, s=s, with_s=self.with_s,
                    head_dim=self.head_dim, qo_dtype=self.qo_dtype)
        c = ctypes
        args = [c.c_void_p(q.data_ptr()), c.c_void_p(k8.data_ptr()), c.c_void_p(sk.data_ptr()),
                c.c_void_p(v8t.data_ptr()), c.c_void_p(sv.data_ptr()),
                c.c_void_p(vmean.data_ptr() if vmean is not None else 0)]
        if self.with_s:
            args.append(c.c_void_p(s.data_ptr()))
        args += [c.c_void_p(out.data_ptr()),
                 c.c_int(int(H)), c.c_int(int(N)), c.c_int(int(N_pad)), c.c_float(float(sm_scale))]
        bh = q.shape[0]
        B = int(bh) // int(H)
        if B * int(H) != int(bh):
            raise ValueError("q.shape[0]=%d is not H=%d times an integer batch size" % (bh, H))
        # the kernel folds bh = blockIdx.z*H + blockIdx.y; the block is 4 warps x 32 lanes x (bm/128)
        bm = int(self.bm)
        grid = ((N + bm - 1) // bm, int(H), B)
        block = (bm * 2, 1, 1)
        st = torch_stream() if stream is None else int(stream)
        self._keep = self.rt.launch(self.funcs[self.symbols[1] if causal else self.symbols[0]],
                                    grid, block, 0, st, args)
        return out

    def launch_strided(self, q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, sm_scale, causal,
                       stream=None, s=None):
        """Launch the 18-argument kernel on logical `(B, H, N, D)` `q`/`out` (strided views allowed).

        Every other tensor is the same as in `launch`. `out` is written in place and returned.
        """
        if not self.strided:
            raise ValueError("this SK1 object is not the strided form (symbols=%r)"
                             % (self.symbols,))
        qtr, otr = self._check_strided(q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, s=s,
                                       head_dim=self.head_dim, qo_dtype=self.qo_dtype,
                                       with_s=self.with_s)
        c = ctypes
        args = [c.c_void_p(q.data_ptr()), c.c_void_p(k8.data_ptr()), c.c_void_p(sk.data_ptr()),
                c.c_void_p(v8t.data_ptr()), c.c_void_p(sv.data_ptr()),
                c.c_void_p(vmean.data_ptr() if vmean is not None else 0)]
        if self.with_s:
            args.append(c.c_void_p(s.data_ptr()))
        args += [c.c_void_p(out.data_ptr()),
                 c.c_int(int(H)), c.c_int(int(N)), c.c_int(int(N_pad)), c.c_float(float(sm_scale)),
                 c.c_int(int(qtr[0])), c.c_int(int(qtr[1])), c.c_int(int(qtr[2])),
                 c.c_int(int(otr[0])), c.c_int(int(otr[1])), c.c_int(int(otr[2]))]
        B = int(q.shape[0])
        bm = int(self.bm)
        grid = ((N + bm - 1) // bm, int(H), B)
        block = (bm * 2, 1, 1)
        st = torch_stream() if stream is None else int(stream)
        self._keep = self.rt.launch(self.funcs[self.symbols[1] if causal else self.symbols[0]],
                                    grid, block, 0, st, args)
        return out


if __name__ == "__main__":
    import json
    rt = HipRuntime()
    print(json.dumps(rt.same_as_torch, indent=1, sort_keys=True))

