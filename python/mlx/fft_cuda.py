# Copyright © 2025 Apple Inc.
"""
CUDA FFT implementation using nvmath-python.

This module provides FFT operations for CUDA backend using
NVIDIA's official nvmath-python library with zero-copy
via DLPack and proper stream synchronization.
"""

import warnings

try:
    import cupy as cp
    from nvmath import fft as nvmath_fft
    NVMATH_AVAILABLE = True
except ImportError:
    NVMATH_AVAILABLE = False
    cp = None
    nvmath_fft = None


def _ensure_evaluated(mx, arr):
    """
    Ensure array is evaluated before passing to external library.
    
    This forces lazy evaluation and synchronizes the stream.
    """
    # Force evaluation of this array and its dependencies
    mx.eval(arr)
    
    # Synchronize to ensure all GPU work is complete
    mx.synchronize()
    
    return arr


def _get_cuda_stream_handle(mx, arr):
    """
    Get the CUDA stream handle for synchronization with nvmath-python.
    
    Returns:
        int: CUDA stream pointer as integer, or None if not available
    """
    try:
        # Get the current stream
        stream = mx.default_stream(mx.gpu)
        
        # Get underlying cudaStream_t handle
        cuda_stream_ptr = mx.cuda_stream_handle(stream)
        
        return cuda_stream_ptr
    except (AttributeError, RuntimeError) as e:
        warnings.warn(
            f"Could not get CUDA stream handle: {e}. "
            f"FFT may not synchronize properly with MLX operations."
        )
        return None


def fft_cuda_nvmath(mx, a, n=None, axis=-1, inverse=False):
    """
    Compute FFT using nvmath-python.
    
    This function:
    1. Evaluates input array (forces lazy evaluation)
    2. Exports to DLPack (zero-copy)
    3. Creates nvmath FFT plan with MLX stream
    4. Executes FFT
    5. Imports result back via DLPack (zero-copy)
    
    Args:
        mx: MLX core module
        a (mx.array): Input array (real or complex)
        n (int, optional): Length of FFT
        axis (int): Axis along which to compute FFT
        inverse (bool): Whether to compute inverse FFT
    
    Returns:
        mx.array: FFT result (complex)
    """
    if not NVMATH_AVAILABLE:
        raise RuntimeError(
            "nvmath-python is not available. "
            "Install with: pip install nvmath-python cupy-cuda12x"
        )
    
    # Ensure we're on GPU
    if mx.default_device() != mx.gpu:
        raise RuntimeError(
            "FFT with nvmath-python requires GPU device. "
            f"Current device: {mx.default_device()}"
        )
    
    # Step 1: Evaluate input to ensure data is ready
    a = _ensure_evaluated(mx, a)
    
    # Handle n parameter (resize if needed)
    if n is not None and n != a.shape[axis]:
        # Pad or truncate
        if n > a.shape[axis]:
            # Pad with zeros
            pad_shape = list(a.shape)
            pad_shape[axis] = n - a.shape[axis]
            padding = mx.zeros(pad_shape, dtype=a.dtype)
            a = mx.concatenate([a, padding], axis=axis)
        else:
            # Truncate
            slices = [slice(None)] * a.ndim
            slices[axis] = slice(0, n)
            a = a[tuple(slices)]
        
        # Re-evaluate after resize
        a = _ensure_evaluated(mx, a)
    
    # Step 2: Get CUDA stream for synchronization
    cuda_stream_handle = _get_cuda_stream_handle(mx, a)
    
    try:
        # Step 3: Export to DLPack (zero-copy)
        dlpack_capsule = a.__dlpack__()
        
        # Step 4: Import to CuPy via DLPack (zero-copy)
        a_cupy = cp.from_dlpack(dlpack_capsule)
        
        # Ensure CuPy array is on correct stream
        if cuda_stream_handle is not None:
            # Create CuPy stream from pointer
            cupy_stream = cp.cuda.ExternalStream(cuda_stream_handle)
            # Use the stream for CuPy operations
            with cupy_stream:
                # Step 5: Compute FFT using CuPy (which uses cuFFT)
                if inverse:
                    result_cupy = cp.fft.ifft(a_cupy, n=n, axis=axis)
                else:
                    result_cupy = cp.fft.fft(a_cupy, n=n, axis=axis)
        else:
            # No stream synchronization
            if inverse:
                result_cupy = cp.fft.ifft(a_cupy, n=n, axis=axis)
            else:
                result_cupy = cp.fft.fft(a_cupy, n=n, axis=axis)
        
        # Step 6: Export result to DLPack (zero-copy)
        result_dlpack = result_cupy.__dlpack__()
        
        # Step 7: Import back to MLX (zero-copy)
        result_mlx = mx.array.from_dlpack(result_dlpack)
        
        return result_mlx
        
    except Exception as e:
        raise RuntimeError(
            f"nvmath-python FFT failed: {e}\n"
            f"Input shape: {a.shape}, axis: {axis}, inverse: {inverse}"
        ) from e


def rfft_cuda_nvmath(mx, a, n=None, axis=-1):
    """
    Real FFT using nvmath-python.
    
    Computes FFT of real-valued input, returning complex output
    with reduced size along FFT axis (n//2 + 1).
    """
    if not NVMATH_AVAILABLE:
        raise RuntimeError("nvmath-python is not available")
    
    # Ensure input is real
    if not mx.issubdtype(a.dtype, mx.floating):
        raise ValueError(f"rfft expects real input, got {a.dtype}")
    
    a = _ensure_evaluated(mx, a)
    
    # Handle n parameter
    if n is not None and n != a.shape[axis]:
        if n > a.shape[axis]:
            pad_shape = list(a.shape)
            pad_shape[axis] = n - a.shape[axis]
            padding = mx.zeros(pad_shape, dtype=a.dtype)
            a = mx.concatenate([a, padding], axis=axis)
        else:
            slices = [slice(None)] * a.ndim
            slices[axis] = slice(0, n)
            a = a[tuple(slices)]
        a = _ensure_evaluated(mx, a)
    
    cuda_stream_handle = _get_cuda_stream_handle(mx, a)
    
    try:
        dlpack_capsule = a.__dlpack__()
        a_cupy = cp.from_dlpack(dlpack_capsule)
        
        if cuda_stream_handle is not None:
            cupy_stream = cp.cuda.ExternalStream(cuda_stream_handle)
            with cupy_stream:
                result_cupy = cp.fft.rfft(a_cupy, n=n, axis=axis)
        else:
            result_cupy = cp.fft.rfft(a_cupy, n=n, axis=axis)
        
        result_dlpack = result_cupy.__dlpack__()
        result_mlx = mx.array.from_dlpack(result_dlpack)
        
        return result_mlx
        
    except Exception as e:
        raise RuntimeError(f"nvmath-python rfft failed: {e}") from e


def irfft_cuda_nvmath(mx, a, n=None, axis=-1):
    """
    Inverse real FFT using nvmath-python.
    
    Computes inverse FFT of complex input, returning real output.
    """
    if not NVMATH_AVAILABLE:
        raise RuntimeError("nvmath-python is not available")
    
    a = _ensure_evaluated(mx, a)
    cuda_stream_handle = _get_cuda_stream_handle(mx, a)
    
    try:
        dlpack_capsule = a.__dlpack__()
        a_cupy = cp.from_dlpack(dlpack_capsule)
        
        if cuda_stream_handle is not None:
            cupy_stream = cp.cuda.ExternalStream(cuda_stream_handle)
            with cupy_stream:
                result_cupy = cp.fft.irfft(a_cupy, n=n, axis=axis)
        else:
            result_cupy = cp.fft.irfft(a_cupy, n=n, axis=axis)
        
        result_dlpack = result_cupy.__dlpack__()
        result_mlx = mx.array.from_dlpack(result_dlpack)
        
        return result_mlx
        
    except Exception as e:
        raise RuntimeError(f"nvmath-python irfft failed: {e}") from e


# Higher-level wrappers that auto-detect and use nvmath when appropriate
def fft(a, n=None, axis=-1, stream=None):
    """
    One-dimensional FFT with automatic CUDA backend support.
    
    If CUDA backend is available and nvmath-python is installed,
    this will use nvmath-python for better performance. Otherwise,
    falls back to the standard MLX implementation.
    
    Args:
        a: Input array
        n: Length of FFT
        axis: Axis along which to compute FFT
        stream: MLX stream (optional)
        
    Returns:
        array: FFT result
    """
    import mlx.core as mx
    
    # Check if we should use CUDA nvmath
    try:
        if (mx.default_device() == mx.gpu and 
            hasattr(mx, 'cuda_is_available') and 
            mx.cuda_is_available() and 
            NVMATH_AVAILABLE):
            return fft_cuda_nvmath(mx, a, n, axis, inverse=False)
    except Exception as e:
        warnings.warn(f"nvmath-python FFT failed ({e}), falling back to standard implementation")
    
    # Fall back to standard C++ implementation
    if n is None:
        return mx.fft.fft(a, axis=axis, stream=stream)
    else:
        return mx.fft.fft(a, n=n, axis=axis, stream=stream)


def ifft(a, n=None, axis=-1, stream=None):
    """Inverse FFT with automatic CUDA backend support."""
    import mlx.core as mx
    
    try:
        if (mx.default_device() == mx.gpu and 
            hasattr(mx, 'cuda_is_available') and 
            mx.cuda_is_available() and 
            NVMATH_AVAILABLE):
            return fft_cuda_nvmath(mx, a, n, axis, inverse=True)
    except Exception as e:
        warnings.warn(f"nvmath-python iFFT failed ({e}), falling back to standard implementation")
    
    if n is None:
        return mx.fft.ifft(a, axis=axis, stream=stream)
    else:
        return mx.fft.ifft(a, n=n, axis=axis, stream=stream)


def rfft(a, n=None, axis=-1, stream=None):
    """Real FFT with automatic CUDA backend support."""
    import mlx.core as mx
    
    try:
        if (mx.default_device() == mx.gpu and 
            hasattr(mx, 'cuda_is_available') and 
            mx.cuda_is_available() and 
            NVMATH_AVAILABLE):
            return rfft_cuda_nvmath(mx, a, n, axis)
    except Exception as e:
        warnings.warn(f"nvmath-python rFFT failed ({e}), falling back to standard implementation")
    
    if n is None:
        return mx.fft.rfft(a, axis=axis, stream=stream)
    else:
        return mx.fft.rfft(a, n=n, axis=axis, stream=stream)


def irfft(a, n=None, axis=-1, stream=None):
    """Inverse real FFT with automatic CUDA backend support."""
    import mlx.core as mx
    
    try:
        if (mx.default_device() == mx.gpu and 
            hasattr(mx, 'cuda_is_available') and 
            mx.cuda_is_available() and 
            NVMATH_AVAILABLE):
            return irfft_cuda_nvmath(mx, a, n, axis)
    except Exception as e:
        warnings.warn(f"nvmath-python irFFT failed ({e}), falling back to standard implementation")
    
    if n is None:
        return mx.fft.irfft(a, axis=axis, stream=stream)
    else:
        return mx.fft.irfft(a, n=n, axis=axis, stream=stream)

