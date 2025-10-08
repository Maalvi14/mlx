#!/usr/bin/env python3
# Copyright © 2025 Apple Inc.
"""
Example demonstrating CUDA FFT functionality in MLX.

This script shows how to use FFT operations on CUDA devices using
NVIDIA's nvmath-python library integration.
"""

import mlx.core as mx
from mlx import fft
import numpy as np


def check_cuda_available():
    """Check if CUDA is available."""
    if not mx.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-capable GPU.")
        return False
    
    try:
        import nvmath
        print(f"nvmath-python version: {nvmath.__version__}")
        return True
    except ImportError:
        print("nvmath-python is not installed.")
        print("Install it with: pip install nvmath-python")
        return False


def example_1d_fft():
    """Example: 1D FFT of a complex signal."""
    print("\n=== 1D FFT Example ===")
    
    # Create a simple complex signal
    n = 128
    t = mx.arange(n, dtype=mx.float32) / n
    signal = mx.exp(2j * mx.pi * 5 * t)  # 5 Hz complex exponential
    
    print(f"Input signal shape: {signal.shape}")
    print(f"Input device: {signal.device}")
    
    # Compute FFT
    spectrum = fft.fft(signal)
    print(f"FFT result shape: {spectrum.shape}")
    
    # Compute inverse to verify
    recovered = fft.ifft(spectrum)
    error = mx.max(mx.abs(signal - recovered))
    print(f"Roundtrip error: {float(error):.2e}")


def example_real_fft():
    """Example: Real FFT for efficient processing of real signals."""
    print("\n=== Real FFT Example ===")
    
    # Create a real signal (sum of sinusoids)
    n = 256
    t = mx.arange(n, dtype=mx.float32) / n
    signal = (mx.sin(2 * mx.pi * 10 * t) + 
              0.5 * mx.sin(2 * mx.pi * 25 * t) +
              0.3 * mx.sin(2 * mx.pi * 50 * t))
    
    print(f"Real signal shape: {signal.shape}")
    
    # Real FFT (more efficient than complex FFT for real input)
    spectrum = fft.rfft(signal)
    print(f"RFFT result shape: {spectrum.shape} (note: n//2 + 1)")
    
    # Find dominant frequencies
    magnitude = mx.abs(spectrum)
    peak_idx = int(mx.argmax(magnitude))
    peak_freq = peak_idx / n
    print(f"Dominant frequency bin: {peak_idx} (normalized freq: {peak_freq:.3f})")
    
    # Inverse to recover signal
    recovered = fft.irfft(spectrum)
    error = mx.max(mx.abs(signal - recovered))
    print(f"Roundtrip error: {float(error):.2e}")


def example_2d_fft():
    """Example: 2D FFT for image processing."""
    print("\n=== 2D FFT Example (Image Processing) ===")
    
    # Create a 2D "image" with some structure
    size = 64
    x = mx.arange(size, dtype=mx.float32).reshape(-1, 1)
    y = mx.arange(size, dtype=mx.float32).reshape(1, -1)
    
    # Create a pattern (checkerboard-like)
    image = mx.sin(2 * mx.pi * x / 8) * mx.sin(2 * mx.pi * y / 8)
    
    print(f"Image shape: {image.shape}")
    
    # Compute 2D FFT
    freq_domain = fft.fft2(image)
    print(f"2D FFT shape: {freq_domain.shape}")
    
    # Shift zero frequency to center
    freq_shifted = fft.fftshift(freq_domain)
    magnitude = mx.abs(freq_shifted)
    
    # Inverse transform
    recovered = fft.ifft2(freq_domain)
    error = mx.max(mx.abs(image - recovered.real))
    print(f"Roundtrip error: {float(error):.2e}")


def example_nd_fft():
    """Example: N-dimensional FFT."""
    print("\n=== N-D FFT Example ===")
    
    # Create a 3D array
    shape = (16, 16, 16)
    data = mx.random.normal(shape=shape)
    
    print(f"3D data shape: {shape}")
    
    # Full N-D FFT
    spectrum_full = fft.fftn(data)
    print(f"Full N-D FFT shape: {spectrum_full.shape}")
    
    # Partial FFT (only on some axes)
    spectrum_partial = fft.fftn(data, axes=(0, 2))
    print(f"Partial FFT (axes 0,2) shape: {spectrum_partial.shape}")
    
    # Verify roundtrip
    recovered = fft.ifftn(spectrum_full)
    error = mx.max(mx.abs(data - recovered.real))
    print(f"Roundtrip error: {float(error):.2e}")


def example_batch_processing():
    """Example: Batch FFT processing."""
    print("\n=== Batch FFT Processing ===")
    
    # Create multiple signals to process together
    batch_size = 32
    signal_length = 256
    
    # Each row is an independent signal
    signals = mx.random.normal(shape=(batch_size, signal_length))
    
    print(f"Batch shape: {signals.shape}")
    
    # FFT along the last axis (each signal independently)
    spectra = fft.fft(signals, axis=-1)
    print(f"Batch FFT shape: {spectra.shape}")
    
    # Process each spectrum (e.g., filter, modify, etc.)
    # Here we just demonstrate the roundtrip
    recovered = fft.ifft(spectra, axis=-1)
    error = mx.max(mx.abs(signals - recovered.real))
    print(f"Batch roundtrip error: {float(error):.2e}")


def performance_comparison():
    """Compare FFT on different array sizes."""
    print("\n=== Performance Comparison ===")
    
    import time
    
    sizes = [128, 512, 1024, 2048]
    
    for size in sizes:
        # Create complex signal
        signal = mx.random.normal(shape=(size,)) + 1j * mx.random.normal(shape=(size,))
        
        # Warm up
        _ = fft.fft(signal)
        mx.eval(_)
        
        # Time it
        start = time.time()
        for _ in range(100):
            result = fft.fft(signal)
            mx.eval(result)
        elapsed = time.time() - start
        
        print(f"Size {size:4d}: {elapsed*10:.2f} ms per FFT (averaged over 100 runs)")


def main():
    """Run all examples."""
    print("CUDA FFT Examples for MLX")
    print("=" * 50)
    
    # Check CUDA and nvmath availability
    if not check_cuda_available():
        return
    
    # Set CUDA as default device
    print(f"\nDefault device: {mx.default_device()}")
    if mx.default_device().type != mx.gpu:
        print("Setting default device to GPU...")
        mx.set_default_device(mx.gpu)
    print(f"Using device: {mx.default_device()}")
    
    # Run examples
    example_1d_fft()
    example_real_fft()
    example_2d_fft()
    example_nd_fft()
    example_batch_processing()
    performance_comparison()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()

