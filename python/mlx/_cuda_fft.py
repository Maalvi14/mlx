# Copyright © 2025 Apple Inc.
"""
CUDA FFT implementation using nvmath-python.

This module provides FFT operations for CUDA devices using NVIDIA's nvmath-python
library with zero-copy dlpack integration.
"""

import mlx.core as mx

try:
    import nvmath
    NVMATH_AVAILABLE = True
except ImportError:
    NVMATH_AVAILABLE = False


def _ensure_nvmath():
    """Check if nvmath-python is available, raise helpful error if not."""
    if not NVMATH_AVAILABLE:
        raise ImportError(
            "nvmath-python is required for CUDA FFT operations. "
            "Install it with: pip install nvmath-python"
        )


def _to_cupy_array(arr):
    """Convert MLX array to CuPy array via dlpack (zero-copy)."""
    import cupy as cp
    # Ensure array is evaluated
    arr.eval()
    # Convert via dlpack
    dlpack_capsule = arr.__dlpack__()
    return cp.from_dlpack(dlpack_capsule)


def _from_cupy_array(cp_arr):
    """Convert CuPy array to MLX array via dlpack (zero-copy)."""
    # Convert via dlpack
    dlpack_capsule = cp_arr.toDlpack()
    return mx.from_dlpack(dlpack_capsule)


def _normalize_axes(axes, ndim):
    """Normalize axes to positive integers."""
    if axes is None:
        return None
    normalized = []
    for ax in axes:
        if ax < 0:
            ax += ndim
        if ax < 0 or ax >= ndim:
            raise ValueError(f"Invalid axis {ax} for array with {ndim} dimensions")
        normalized.append(ax)
    return normalized


def _handle_padding_truncation(arr, n, axes):
    """Handle FFT size that differs from input size via padding or truncation."""
    if n is None:
        return arr
    
    shape = list(arr.shape)
    slices = [slice(None)] * arr.ndim
    pad_widths = [(0, 0)] * arr.ndim
    
    for i, (ax, size) in enumerate(zip(axes, n)):
        if size < shape[ax]:
            # Truncate
            slices[ax] = slice(0, size)
        elif size > shape[ax]:
            # Pad
            pad_widths[ax] = (0, size - shape[ax])
    
    # Apply truncation
    arr = arr[tuple(slices)]
    
    # Apply padding if needed
    if any(pw != (0, 0) for pw in pad_widths):
        arr = mx.pad(arr, pad_widths)
    
    return arr


def fft(a, n=None, axis=-1, norm=None):
    """
    One dimensional discrete Fourier Transform (CUDA implementation).
    
    Args:
        a: Input array
        n: Length of transformed axis. If None, uses a.shape[axis]
        axis: Axis along which to perform FFT (default: -1)
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Complex array containing the FFT
    """
    _ensure_nvmath()
    
    # Normalize axis
    axis = axis if axis >= 0 else a.ndim + axis
    
    # Handle padding/truncation
    if n is not None and n != a.shape[axis]:
        a = _handle_padding_truncation(a, [n], [axis])
    
    # Convert to complex if real input
    if a.dtype == mx.float32:
        a = mx.astype(a, mx.complex64)
    
    # Convert to CuPy
    cp_arr = _to_cupy_array(a)
    
    # Execute FFT using nvmath
    result = nvmath.fft.fft(cp_arr, axes=[axis])
    
    # Convert back to MLX
    return _from_cupy_array(result)


def ifft(a, n=None, axis=-1, norm=None):
    """
    One dimensional inverse discrete Fourier Transform (CUDA implementation).
    
    Args:
        a: Input array
        n: Length of transformed axis. If None, uses a.shape[axis]
        axis: Axis along which to perform IFFT (default: -1)
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Complex array containing the inverse FFT
    """
    _ensure_nvmath()
    
    # Normalize axis
    axis = axis if axis >= 0 else a.ndim + axis
    
    # Handle padding/truncation
    if n is not None and n != a.shape[axis]:
        a = _handle_padding_truncation(a, [n], [axis])
    
    # Ensure complex input
    if a.dtype == mx.float32:
        a = mx.astype(a, mx.complex64)
    
    # Convert to CuPy
    cp_arr = _to_cupy_array(a)
    
    # Execute IFFT using nvmath
    result = nvmath.fft.ifft(cp_arr, axes=[axis])
    
    # Convert back to MLX
    return _from_cupy_array(result)


def rfft(a, n=None, axis=-1, norm=None):
    """
    One dimensional FFT of a real input (CUDA implementation).
    
    Args:
        a: Real input array
        n: Length of transformed axis. If None, uses a.shape[axis]
        axis: Axis along which to perform FFT (default: -1)
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Complex array with shape[axis] = n // 2 + 1
    """
    _ensure_nvmath()
    
    # Normalize axis
    axis = axis if axis >= 0 else a.ndim + axis
    
    # Handle padding/truncation
    if n is not None and n != a.shape[axis]:
        a = _handle_padding_truncation(a, [n], [axis])
    
    # Ensure real input
    if a.dtype == mx.complex64:
        a = mx.real(a)
    
    # Convert to CuPy
    cp_arr = _to_cupy_array(a)
    
    # Execute RFFT using nvmath
    result = nvmath.fft.rfft(cp_arr, axes=[axis])
    
    # Convert back to MLX
    return _from_cupy_array(result)


def irfft(a, n=None, axis=-1, norm=None):
    """
    Inverse of rfft (CUDA implementation).
    
    Args:
        a: Complex input array
        n: Length of output axis. If None, uses (a.shape[axis] - 1) * 2
        axis: Axis along which to perform IFFT (default: -1)
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Real array
    """
    _ensure_nvmath()
    
    # Normalize axis
    axis = axis if axis >= 0 else a.ndim + axis
    
    # Determine output size
    if n is None:
        n = (a.shape[axis] - 1) * 2
    
    # Convert to CuPy
    cp_arr = _to_cupy_array(a)
    
    # Execute IRFFT using nvmath
    result = nvmath.fft.irfft(cp_arr, n=n, axes=[axis])
    
    # Convert back to MLX
    return _from_cupy_array(result)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Two dimensional discrete Fourier Transform (CUDA implementation).
    
    Args:
        a: Input array
        s: Shape of output. If None, uses a.shape along axes
        axes: Axes along which to perform FFT (default: (-2, -1))
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Complex array containing the 2D FFT
    """
    return fftn(a, s=s, axes=axes, norm=norm)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Two dimensional inverse discrete Fourier Transform (CUDA implementation).
    
    Args:
        a: Input array
        s: Shape of output. If None, uses a.shape along axes
        axes: Axes along which to perform IFFT (default: (-2, -1))
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Complex array containing the 2D inverse FFT
    """
    return ifftn(a, s=s, axes=axes, norm=norm)


def fftn(a, s=None, axes=None, norm=None):
    """
    N-dimensional discrete Fourier Transform (CUDA implementation).
    
    Args:
        a: Input array
        s: Shape of output. If None, uses a.shape along axes
        axes: Axes along which to perform FFT. If None, uses all axes
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Complex array containing the N-D FFT
    """
    _ensure_nvmath()
    
    # Default axes: all axes
    if axes is None:
        axes = list(range(a.ndim))
    
    # Normalize axes
    axes = _normalize_axes(axes, a.ndim)
    
    # Handle padding/truncation
    if s is not None:
        if len(s) != len(axes):
            raise ValueError(f"Shape {s} and axes {axes} have different lengths")
        a = _handle_padding_truncation(a, s, axes)
    
    # Convert to complex if real input
    if a.dtype == mx.float32:
        a = mx.astype(a, mx.complex64)
    
    # Convert to CuPy
    cp_arr = _to_cupy_array(a)
    
    # Execute FFT using nvmath
    result = nvmath.fft.fft(cp_arr, axes=axes)
    
    # Convert back to MLX
    return _from_cupy_array(result)


def ifftn(a, s=None, axes=None, norm=None):
    """
    N-dimensional inverse discrete Fourier Transform (CUDA implementation).
    
    Args:
        a: Input array
        s: Shape of output. If None, uses a.shape along axes
        axes: Axes along which to perform IFFT. If None, uses all axes
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Complex array containing the N-D inverse FFT
    """
    _ensure_nvmath()
    
    # Default axes: all axes
    if axes is None:
        axes = list(range(a.ndim))
    
    # Normalize axes
    axes = _normalize_axes(axes, a.ndim)
    
    # Handle padding/truncation
    if s is not None:
        if len(s) != len(axes):
            raise ValueError(f"Shape {s} and axes {axes} have different lengths")
        a = _handle_padding_truncation(a, s, axes)
    
    # Ensure complex input
    if a.dtype == mx.float32:
        a = mx.astype(a, mx.complex64)
    
    # Convert to CuPy
    cp_arr = _to_cupy_array(a)
    
    # Execute IFFT using nvmath
    result = nvmath.fft.ifft(cp_arr, axes=axes)
    
    # Convert back to MLX
    return _from_cupy_array(result)


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Two dimensional FFT of a real input (CUDA implementation).
    
    Args:
        a: Real input array
        s: Shape of output. If None, uses a.shape along axes
        axes: Axes along which to perform FFT (default: (-2, -1))
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Complex array with shape[axes[-1]] = s[-1] // 2 + 1
    """
    return rfftn(a, s=s, axes=axes, norm=norm)


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Inverse of rfft2 (CUDA implementation).
    
    Args:
        a: Complex input array
        s: Shape of output. If None, computed from input
        axes: Axes along which to perform IFFT (default: (-2, -1))
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Real array
    """
    return irfftn(a, s=s, axes=axes, norm=norm)


def rfftn(a, s=None, axes=None, norm=None):
    """
    N-dimensional FFT of a real input (CUDA implementation).
    
    Args:
        a: Real input array
        s: Shape of output. If None, uses a.shape along axes
        axes: Axes along which to perform FFT. If None, uses all axes
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Complex array with shape[axes[-1]] = s[-1] // 2 + 1
    """
    _ensure_nvmath()
    
    # Default axes: all axes
    if axes is None:
        axes = list(range(a.ndim))
    
    # Normalize axes
    axes = _normalize_axes(axes, a.ndim)
    
    # Handle padding/truncation
    if s is not None:
        if len(s) != len(axes):
            raise ValueError(f"Shape {s} and axes {axes} have different lengths")
        a = _handle_padding_truncation(a, s, axes)
    
    # Ensure real input
    if a.dtype == mx.complex64:
        a = mx.real(a)
    
    # Convert to CuPy
    cp_arr = _to_cupy_array(a)
    
    # Execute RFFT using nvmath
    result = nvmath.fft.rfft(cp_arr, axes=axes)
    
    # Convert back to MLX
    return _from_cupy_array(result)


def irfftn(a, s=None, axes=None, norm=None):
    """
    Inverse of rfftn (CUDA implementation).
    
    Args:
        a: Complex input array
        s: Shape of output. If None, computed from input
        axes: Axes along which to perform IFFT. If None, uses all axes
        norm: Normalization mode (not yet implemented, kept for compatibility)
    
    Returns:
        Real array
    """
    _ensure_nvmath()
    
    # Default axes: all axes
    if axes is None:
        axes = list(range(a.ndim))
    
    # Normalize axes
    axes = _normalize_axes(axes, a.ndim)
    
    # Determine output size for last axis if not specified
    if s is None:
        s = [a.shape[ax] for ax in axes[:-1]]
        s.append((a.shape[axes[-1]] - 1) * 2)
    
    if len(s) != len(axes):
        raise ValueError(f"Shape {s} and axes {axes} have different lengths")
    
    # Convert to CuPy
    cp_arr = _to_cupy_array(a)
    
    # Execute IRFFT using nvmath
    # Note: nvmath.fft.irfft needs explicit output size for last axis
    result = nvmath.fft.irfft(cp_arr, n=s[-1], axes=axes)
    
    # Convert back to MLX
    return _from_cupy_array(result)

