#!/usr/bin/env python3
"""
Manual test for CUDA FFT (nvmath-python path).

Usage (from repo root):

  # 1. CPU FFT (no nvmath/CuPy needed) – checks dispatch and C++ path:
  uv run python scripts/test_cuda_fft.py --cpu

  # 2. CUDA FFT (requires nvmath + CuPy; install then run):
  pip install nvmath-python[cu12] cupy-cuda12x   # or [cu13] / cupy-cuda13x
  uv run python scripts/test_cuda_fft.py

  # 3. Full FFT test suite (after installing mlx, e.g. pip install -e python):
  cd python && uv run pytest tests/test_fft.py -v

  # 4. Only the CUDA FFT test (skips if CUDA or nvmath/CuPy missing):
  cd python && uv run pytest tests/test_fft.py -v -k TestFFTCUDA
"""

from __future__ import annotations

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Test MLX FFT (optionally on CUDA via nvmath).")
    ap.add_argument("--cpu", action="store_true", help="Force CPU; do not use CUDA even if available.")
    args = ap.parse_args()

    import mlx.core as mx
    import numpy as np

    if args.cpu:
        mx.set_default_device(mx.cpu)
        print("Using CPU")
    else:
        print(f"Default device: {mx.default_device()}")
        if mx.default_device() == mx.gpu:
            try:
                from mlx.fft_dispatch import cuda_fft_available
                if not cuda_fft_available():
                    print("CUDA FFT not available (nvmath/CuPy missing or DLPack incompatible). Using CPU.")
                    mx.set_default_device(mx.cpu)
            except Exception:
                print("CUDA FFT not available. Using CPU.")
                mx.set_default_device(mx.cpu)

    # Basic FFT (works on CPU or GPU when nvmath+CuPy installed)
    a = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.complex64)
    r = mx.fft.fft(a)
    expected = np.fft.fft(np.array(a))
    np.testing.assert_allclose(np.array(r), expected, atol=1e-5, rtol=1e-5)
    print("  fft(1..4) OK")

    # Compare to NumPy on random data
    np.random.seed(42)
    a_np = np.random.rand(32).astype(np.float32) + 1j * np.random.rand(32).astype(np.float32)
    a_mx = mx.array(a_np)
    r_np = np.fft.fft(a_np)
    r_mx = mx.fft.fft(a_mx)
    np.testing.assert_allclose(r_np, np.array(r_mx), atol=1e-4, rtol=1e-3)
    print("  fft(random 32) vs NumPy OK")

    # rfft
    r_np = np.random.rand(32).astype(np.float32)
    r_mx = mx.array(r_np)
    out_np = np.fft.rfft(r_np)
    out_mx = mx.fft.rfft(r_mx)
    np.testing.assert_allclose(out_np, np.array(out_mx), atol=1e-4, rtol=1e-3)
    print("  rfft vs NumPy OK")

    print("All FFT checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
