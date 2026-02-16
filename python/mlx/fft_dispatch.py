# Copyright © 2025 Apple Inc.
"""FFT dispatch: use nvmath-python (cuFFT) for CUDA, else mlx.core.fft."""

from __future__ import annotations

import types
from typing import Any, Optional, Sequence, Union

import mlx.core as mx

# Original C++ fft submodule (set when we patch).
_orig_fft: Optional[types.ModuleType] = None

# Lazy nvmath/cupy (only when CUDA FFT is used).
_nvmath = None
_cupy = None
# One-time DLPack compatibility: None = not checked, True/False = MLX GPU <-> CuPy works or not.
_cuda_fft_dlpack_ok: Optional[bool] = None


def _cuda_fft_available() -> bool:
    """True if nvmath and CuPy are available and DLPack MLX GPU <-> CuPy works."""
    global _nvmath, _cupy, _cuda_fft_dlpack_ok
    try:
        if _nvmath is None or _cupy is None:
            import nvmath  # noqa: F401
            import cupy as cp  # noqa: F401
            _nvmath = nvmath
            _cupy = cp
    except ImportError:
        return False
    if not _device_is_cuda(None):
        return False
    if _cuda_fft_dlpack_ok is not None:
        return _cuda_fft_dlpack_ok
    try:
        a = mx.array([1.0 + 0.0j], device=mx.gpu)
        _cupy.from_dlpack(a.__dlpack__())
        _cuda_fft_dlpack_ok = True
    except RuntimeError as e:
        if "different" in str(e).lower() and "backend" in str(e).lower():
            _cuda_fft_dlpack_ok = False
        else:
            raise
    return _cuda_fft_dlpack_ok


def cuda_fft_available() -> bool:
    """Public check: True if CUDA FFT (nvmath + CuPy, DLPack compatible) is usable."""
    return _cuda_fft_available()


def _ensure_cuda_fft():
    """Raise if nvmath/CuPy are not available (for CUDA FFT)."""
    if not _cuda_fft_available():
        raise RuntimeError(
            "CUDA FFT requires nvmath-python and CuPy. "
            "Install with e.g.: pip install nvmath-python[cu12] cupy-cuda12x"
        )


def _device_is_cuda(stream: Any) -> bool:
    if stream is None:
        return mx.default_device() == mx.gpu
    if hasattr(stream, "device"):
        return stream.device == mx.gpu
    return mx.default_device() == mx.gpu


def _to_cupy(a: mx.array):
    _ensure_cuda_fft()
    cp = _cupy
    dl = a.__dlpack__()
    return cp.from_dlpack(dl)


def _from_cupy_to_mlx(cp_arr, stream: Any) -> mx.array:
    from mlx._dlpack import from_dlpack
    out = from_dlpack(cp_arr.__dlpack__())
    if stream is not None and hasattr(stream, "device") and stream.device == mx.gpu:
        out = mx.copy(out, stream)
    elif stream is None and mx.default_device() == mx.gpu:
        out = mx.copy(out, mx.default_device())
    return out


def _norm_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        return axis + ndim
    return axis


def _norm_axes(axes: Optional[Sequence[int]], ndim: int) -> list[int]:
    if axes is None:
        return list(range(ndim))
    return [_norm_axis(a, ndim) for a in axes]


def _prepare_fft_input(
    a: mx.array,
    n: Optional[Union[int, Sequence[int]]],
    axes: list[int],
    real: bool,
    inverse: bool,
    stream: Any,
) -> mx.array:
    """Slice/pad input to match n/s like mlx fft_impl."""
    if n is None:
        return a
    ndim = a.ndim
    if isinstance(n, int):
        n_seq = [n]
    else:
        n_seq = list(n)
    in_shape = list(a.shape)
    for i, ax in enumerate(axes):
        if i < len(n_seq):
            in_shape[ax] = n_seq[i]
    if real and inverse and axes:
        in_shape[axes[-1]] = n_seq[-1] // 2 + 1
    any_less = any(in_shape[i] < a.shape[i] for i in range(ndim))
    any_greater = any(in_shape[i] > a.shape[i] for i in range(ndim))
    in_arr = a
    axes_tup = tuple(range(ndim))
    if any_less:
        in_arr = mx.slice(
            in_arr,
            mx.array([0] * ndim),
            axes_tup,
            tuple(in_shape),
            stream=stream,
        )
    if any_greater:
        tmp = mx.zeros(tuple(in_shape), a.dtype, stream)
        in_arr = mx.slice_update(
            tmp,
            in_arr,
            mx.array([0] * ndim),
            axes_tup,
            stream=stream,
        )
    return in_arr


def _fft_cuda(
    a: mx.array,
    axes: list[int],
    inverse: bool,
    real: bool,
    n: Optional[Sequence[int]],
    s: Any,
    stream: Any,
) -> mx.array:
    _ensure_cuda_fft()
    nvmath = _nvmath
    a = mx.contiguous(a)
    a = _prepare_fft_input(a, n, axes, real, inverse, stream)
    cp_a = _to_cupy(a)
    exec_cuda = nvmath.fft.ExecutionCUDA()
    if real and not inverse:
        r = nvmath.fft.rfft(cp_a, axes=axes, execution=exec_cuda)
    elif real and inverse:
        r = nvmath.fft.irfft(cp_a, axes=axes, execution=exec_cuda)
    elif inverse:
        r = nvmath.fft.ifft(cp_a, axes=axes, execution=exec_cuda)
    else:
        r = nvmath.fft.fft(cp_a, axes=axes, execution=exec_cuda)
    return _from_cupy_to_mlx(r, stream)


def _dispatch_fft(
    name: str,
    real: bool,
    inverse: bool,
    a: mx.array,
    n: Optional[Union[int, Sequence[int]]] = None,
    axis: Optional[int] = None,
    axes: Optional[Sequence[int]] = None,
    s: Optional[Sequence[int]] = None,
    stream: Any = None,
) -> mx.array:
    orig = _orig_fft
    if orig is None:
        raise RuntimeError("fft_dispatch: original mlx.core.fft not set")
    stream = stream if stream is not None else mx.default_device()
    # Use nvmath (CUDA) path only when default/stream is CUDA and nvmath+CuPy are installed.
    if not _device_is_cuda(stream) or not _cuda_fft_available():
        fn = getattr(orig, name)
        if name in ("fft", "ifft", "rfft", "irfft"):
            if n is not None:
                return fn(a, n=n, axis=axis if axis is not None else -1, stream=stream)
            return fn(a, axis=axis if axis is not None else -1, stream=stream)
        if name in ("fft2", "ifft2", "rfft2", "irfft2"):
            return fn(a, s=s, axes=axes, stream=stream)
        if name in ("fftn", "ifftn", "rfftn", "irfftn"):
            return fn(a, s=s, axes=axes, stream=stream)
        return fn(a, stream=stream)

    ndim = a.ndim
    if name in ("fft", "ifft", "rfft", "irfft"):
        ax = _norm_axis(axis if axis is not None else -1, ndim)
        axes_list = [ax]
        n_seq = [n] if n is not None else None
    elif name in ("fft2", "ifft2", "rfft2", "irfft2"):
        axes_list = list(axes) if axes is not None else [ndim - 2, ndim - 1]
        axes_list = [_norm_axis(ax, ndim) for ax in axes_list]
        n_seq = list(s) if s is not None else None
    else:
        axes_list = _norm_axes(axes, ndim)
        n_seq = list(s) if s is not None else None

    return _fft_cuda(a, axes_list, inverse, real, n_seq, s, stream)


def _create_fft_wrapper(original_fft: types.ModuleType) -> types.ModuleType:
    global _orig_fft
    _orig_fft = original_fft

    wrapper = types.ModuleType("mlx.core.fft")
    wrapper.__doc__ = original_fft.__doc__

    def fft_1d(a, n=None, axis=-1, stream=None):
        return _dispatch_fft("fft", False, False, a, n=n, axis=axis, stream=stream)

    def ifft_1d(a, n=None, axis=-1, stream=None):
        return _dispatch_fft("ifft", False, True, a, n=n, axis=axis, stream=stream)

    def rfft_1d(a, n=None, axis=-1, stream=None):
        return _dispatch_fft("rfft", True, False, a, n=n, axis=axis, stream=stream)

    def irfft_1d(a, n=None, axis=-1, stream=None):
        return _dispatch_fft("irfft", True, True, a, n=n, axis=axis, stream=stream)

    def fft_nd(name, real, inverse, a, s=None, axes=None, stream=None):
        return _dispatch_fft(name, real, inverse, a, n=s, axes=axes, s=s, stream=stream)

    wrapper.fft = fft_1d
    wrapper.ifft = ifft_1d
    wrapper.rfft = rfft_1d
    wrapper.irfft = irfft_1d
    wrapper.fft2 = lambda a, s=None, axes=(-2, -1), stream=None: fft_nd("fft2", False, False, a, s=s, axes=axes, stream=stream)
    wrapper.ifft2 = lambda a, s=None, axes=(-2, -1), stream=None: fft_nd("ifft2", False, True, a, s=s, axes=axes, stream=stream)
    wrapper.rfft2 = lambda a, s=None, axes=(-2, -1), stream=None: fft_nd("rfft2", True, False, a, s=s, axes=axes, stream=stream)
    wrapper.irfft2 = lambda a, s=None, axes=(-2, -1), stream=None: fft_nd("irfft2", True, True, a, s=s, axes=axes, stream=stream)
    wrapper.fftn = lambda a, s=None, axes=None, stream=None: fft_nd("fftn", False, False, a, s=s, axes=axes, stream=stream)
    wrapper.ifftn = lambda a, s=None, axes=None, stream=None: fft_nd("ifftn", False, True, a, s=s, axes=axes, stream=stream)
    wrapper.rfftn = lambda a, s=None, axes=None, stream=None: fft_nd("rfftn", True, False, a, s=s, axes=axes, stream=stream)
    wrapper.irfftn = lambda a, s=None, axes=None, stream=None: fft_nd("irfftn", True, True, a, s=s, axes=axes, stream=stream)

    def fftshift(a: mx.array, axes: Optional[Sequence[int]] = None, stream: Any = None):
        return original_fft.fftshift(a, axes=axes, stream=stream)

    def ifftshift(a: mx.array, axes: Optional[Sequence[int]] = None, stream: Any = None):
        return original_fft.ifftshift(a, axes=axes, stream=stream)

    wrapper.fftshift = fftshift
    wrapper.ifftshift = ifftshift

    return wrapper
