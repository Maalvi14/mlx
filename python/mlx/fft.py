# Copyright © 2025 Apple Inc.
"""
FFT operations for MLX arrays.

This module provides Fast Fourier Transform operations that automatically
dispatch to the appropriate backend (CUDA or CPU/Metal) based on the device
of the input array.

For CUDA devices, the implementation uses NVIDIA's nvmath-python library
when available, providing optimal performance on NVIDIA GPUs.
"""

from mlx._fft_dispatch import (
    fft,
    ifft,
    fft2,
    ifft2,
    fftn,
    ifftn,
    rfft,
    irfft,
    rfft2,
    irfft2,
    rfftn,
    irfftn,
    fftshift,
    ifftshift,
)

__all__ = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "fftshift",
    "ifftshift",
]

