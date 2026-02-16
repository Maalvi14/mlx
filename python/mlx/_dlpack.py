# Copyright © 2025 Apple Inc.
"""DLPack interop: create MLX arrays from DLPack capsules (copy-based)."""

from __future__ import annotations

import numpy as np


def from_dlpack(dlpack_capsule):
    """Create an MLX array from a DLPack capsule.

    Data is copied; the returned array does not share memory with the
    original tensor. Supports CPU (NumPy) and CUDA (CuPy) capsules.

    For CUDA capsules, CuPy must be installed (`pip install cupy-cuda12x` or
    similar). The data is copied from device to MLX (same or host).

    Args:
        dlpack_capsule: A DLPack capsule (e.g. from ``tensor.__dlpack__()``).

    Returns:
        An MLX array with the same shape and dtype.
    """
    import mlx.core as mx

    try:
        arr = np.from_dlpack(dlpack_capsule)
        return mx.array(arr)
    except (TypeError, ValueError):
        pass

    try:
        import cupy as cp
        arr = cp.from_dlpack(dlpack_capsule)
        return mx.array(arr.get())
    except ImportError:
        raise RuntimeError(
            "from_dlpack: CUDA capsule received but CuPy is not installed. "
            "Install with e.g. pip install cupy-cuda12x"
        ) from None
    except (TypeError, ValueError):
        raise RuntimeError(
            "from_dlpack: unsupported DLPack capsule (device or dtype)."
        ) from None
