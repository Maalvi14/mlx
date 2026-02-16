# Copyright © 2025 Apple Inc.
"""MLX: A framework for machine learning."""

# Ensure mlx.core is loaded and patch FFT for CUDA (nvmath-python) and add from_dlpack.
import mlx.core  # noqa: F401

from mlx._dlpack import from_dlpack
from mlx.fft_dispatch import _create_fft_wrapper

# Patch mlx.core.fft so CUDA uses nvmath-python (cuFFT) via DLPack.
mlx.core.fft = _create_fft_wrapper(mlx.core.fft)

# Expose from_dlpack on mlx.core for DLPack interop (e.g. FFT result from CuPy).
mlx.core.from_dlpack = from_dlpack
