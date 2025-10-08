# Copyright © 2025 Apple Inc.
"""
FFT dispatcher that routes to appropriate backend (CUDA or C++).

This module checks the device of input arrays and routes FFT operations
to either the CUDA nvmath-python implementation or the existing C++ implementation.
"""

import mlx.core as mx
from mlx.core import fft as core_fft


def _is_cuda_device(device):
    """Check if a device is a CUDA device."""
    # Check if device is CUDA GPU
    return device.type == mx.gpu and mx.cuda.is_available()


def _dispatch_fft(func_name, a, *args, **kwargs):
    """
    Dispatch FFT function to appropriate backend based on device.
    
    Args:
        func_name: Name of the FFT function to call
        a: Input array
        *args, **kwargs: Additional arguments for the FFT function
        
    Returns:
        Result from either CUDA or C++ FFT implementation
    """
    # Get the device from the input array
    device = a.device if hasattr(a, 'device') else mx.default_device()
    
    # Check if we should use CUDA implementation
    if _is_cuda_device(device):
        try:
            # Import CUDA FFT module
            from mlx import _cuda_fft
            
            # Get the requested function
            cuda_func = getattr(_cuda_fft, func_name)
            
            # Call CUDA implementation
            return cuda_func(a, *args, **kwargs)
            
        except ImportError:
            # nvmath-python not available, fall back to C++
            # This will raise an error in the C++ backend
            pass
        except Exception as e:
            # If CUDA implementation fails, try C++ fallback
            # (though it will likely fail too for CUDA arrays)
            import warnings
            warnings.warn(
                f"CUDA FFT failed: {e}. Attempting C++ fallback (may also fail).",
                RuntimeWarning
            )
    
    # Use C++ implementation (via core.fft)
    core_func = getattr(core_fft, func_name)
    
    # Extract stream if provided
    stream = kwargs.pop('stream', None)
    if stream is not None:
        return core_func(a, *args, stream=stream, **kwargs)
    else:
        return core_func(a, *args, **kwargs)


def fft(a, n=None, axis=-1, stream=None):
    """
    One dimensional discrete Fourier Transform.
    
    Args:
        a (array): The input array.
        n (int, optional): Size of the transformed axis. The
           corresponding axis in the input is truncated or padded with
           zeros to match ``n``. The default value is ``a.shape[axis]``.
        axis (int, optional): Axis along which to perform the FFT. The
           default is ``-1``.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The DFT of the input along the given axis.
    """
    # C++ signature: fft(a, n, axis, stream)
    # CUDA signature: fft(a, n, axis, norm)
    return _dispatch_fft('fft', a, n, axis, stream=stream)


def ifft(a, n=None, axis=-1, stream=None):
    """
    One dimensional inverse discrete Fourier Transform.
    
    Args:
        a (array): The input array.
        n (int, optional): Size of the transformed axis.
        axis (int, optional): Axis along which to perform the IFFT. 
           The default is ``-1``.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The inverse DFT of the input along the given axis.
    """
    return _dispatch_fft('ifft', a, n, axis, stream=stream)


def fft2(a, s=None, axes=(-2, -1), stream=None):
    """
    Two dimensional discrete Fourier Transform.
    
    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``[-2, -1]``.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The 2D DFT of the input along the given axes.
    """
    return _dispatch_fft('fft2', a, s, axes, stream=stream)


def ifft2(a, s=None, axes=(-2, -1), stream=None):
    """
    Two dimensional inverse discrete Fourier Transform.
    
    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes.
        axes (list(int), optional): Axes along which to perform the IFFT.
           The default is ``[-2, -1]``.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The 2D inverse DFT of the input along the given axes.
    """
    return _dispatch_fft('ifft2', a, s, axes, stream=stream)


def fftn(a, s=None, axes=None, stream=None):
    """
    n-dimensional discrete Fourier Transform.
    
    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``None`` in which case the FFT is over all axes.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The DFT of the input along the given axes.
    """
    return _dispatch_fft('fftn', a, s, axes, stream=stream)


def ifftn(a, s=None, axes=None, stream=None):
    """
    n-dimensional inverse discrete Fourier Transform.
    
    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes.
        axes (list(int), optional): Axes along which to perform the IFFT.
           The default is ``None`` in which case the FFT is over all axes.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The inverse DFT of the input along the given axes.
    """
    return _dispatch_fft('ifftn', a, s, axes, stream=stream)


def rfft(a, n=None, axis=-1, stream=None):
    """
    One dimensional discrete Fourier Transform on a real input.
    
    The output has the same shape as the input except along ``axis`` in
    which case it has size ``n // 2 + 1``.
    
    Args:
        a (array): The input array. If the array is complex it will be silently
           cast to a real type.
        n (int, optional): Size of the transformed axis.
        axis (int, optional): Axis along which to perform the FFT.
           The default is ``-1``.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The DFT of the input along the given axis. The output
        data type will be complex.
    """
    return _dispatch_fft('rfft', a, n, axis, stream=stream)


def irfft(a, n=None, axis=-1, stream=None):
    """
    One dimensional inverse of :func:`rfft`.
    
    Args:
        a (array): The input array.
        n (int, optional): Size of the transformed axis in the output.
           The default is ``(a.shape[axis] - 1) * 2``.
        axis (int, optional): Axis along which to perform the IFFT.
           The default is ``-1``.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The inverse real FFT along the given axis. The output
        data type will be real.
    """
    return _dispatch_fft('irfft', a, n, axis, stream=stream)


def rfft2(a, s=None, axes=(-2, -1), stream=None):
    """
    Two dimensional discrete Fourier Transform on a real input.
    
    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``[-2, -1]``.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The 2D real FFT of the input along the given axes.
    """
    return _dispatch_fft('rfft2', a, s, axes, stream=stream)


def irfft2(a, s=None, axes=(-2, -1), stream=None):
    """
    Two dimensional inverse of :func:`rfft2`.
    
    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes.
        axes (list(int), optional): Axes along which to perform the IFFT.
           The default is ``[-2, -1]``.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The 2D inverse real FFT along the given axes.
    """
    return _dispatch_fft('irfft2', a, s, axes, stream=stream)


def rfftn(a, s=None, axes=None, stream=None):
    """
    n-dimensional discrete Fourier Transform on a real input.
    
    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``None`` in which case the FFT is over all axes.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The real FFT of the input along the given axes.
    """
    return _dispatch_fft('rfftn', a, s, axes, stream=stream)


def irfftn(a, s=None, axes=None, stream=None):
    """
    n-dimensional inverse of :func:`rfftn`.
    
    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes.
        axes (list(int), optional): Axes along which to perform the IFFT.
           The default is ``None`` in which case the FFT is over all axes.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The inverse real FFT of the input along the given axes.
    """
    return _dispatch_fft('irfftn', a, s, axes, stream=stream)


def fftshift(a, axes=None, stream=None):
    """
    Shift the zero-frequency component to the center of the spectrum.
    
    Args:
        a (array): The input array.
        axes (list(int), optional): Axes along which to shift.
           The default is ``None`` which shifts all axes.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The shifted array.
    """
    # fftshift doesn't depend on device, always use C++ version
    if axes is None:
        return core_fft.fftshift(a, stream) if stream else core_fft.fftshift(a)
    else:
        return core_fft.fftshift(a, axes, stream) if stream else core_fft.fftshift(a, axes)


def ifftshift(a, axes=None, stream=None):
    """
    The inverse of :func:`fftshift`.
    
    Args:
        a (array): The input array.
        axes (list(int), optional): Axes along which to shift.
           The default is ``None`` which shifts all axes.
        stream (Stream, optional): Stream for the operation.

    Returns:
        array: The shifted array.
    """
    # ifftshift doesn't depend on device, always use C++ version
    if axes is None:
        return core_fft.ifftshift(a, stream) if stream else core_fft.ifftshift(a)
    else:
        return core_fft.ifftshift(a, axes, stream) if stream else core_fft.ifftshift(a, axes)

