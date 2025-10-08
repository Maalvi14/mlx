# Copyright © 2025 Apple Inc.

import unittest

import mlx.core as mx
import numpy as np

# Check if CUDA is available
CUDA_AVAILABLE = mx.cuda.is_available()

# Try to import nvmath-python
try:
    import nvmath
    NVMATH_AVAILABLE = True
except ImportError:
    NVMATH_AVAILABLE = False

# Only run tests if both CUDA and nvmath are available
SKIP_CUDA_FFT_TESTS = not (CUDA_AVAILABLE and NVMATH_AVAILABLE)
SKIP_REASON = "CUDA FFT tests require CUDA and nvmath-python"


@unittest.skipIf(SKIP_CUDA_FFT_TESTS, SKIP_REASON)
class TestCUDAFFT(unittest.TestCase):
    """Test FFT operations on CUDA devices using nvmath-python."""

    def setUp(self):
        """Set default device to CUDA for these tests."""
        self.original_device = mx.default_device()
        mx.set_default_device(mx.gpu)

    def tearDown(self):
        """Restore original default device."""
        mx.set_default_device(self.original_device)

    def check_fft_correctness(self, fft_func, np_fft_func, a_np, **kwargs):
        """
        Helper to check FFT results against NumPy.
        
        Args:
            fft_func: MLX FFT function
            np_fft_func: NumPy FFT function
            a_np: NumPy input array
            **kwargs: Additional arguments for FFT functions
        """
        # Import the dispatcher functions
        from mlx import fft as mlx_fft
        
        # Get the MLX FFT function from the dispatcher
        fft_fn = getattr(mlx_fft, fft_func)
        
        # Compute NumPy reference
        out_np = np_fft_func(a_np, **kwargs)
        
        # Compute MLX result on CUDA
        a_mx = mx.array(a_np)
        mx.eval(a_mx)  # Ensure it's on CUDA
        self.assertEqual(a_mx.device.type, mx.gpu)
        
        out_mx = fft_fn(a_mx, **kwargs)
        mx.eval(out_mx)
        
        # Compare results
        np.testing.assert_allclose(
            np.array(out_mx),
            out_np,
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"FFT mismatch for {fft_func}"
        )

    def test_fft_basic(self):
        """Test basic 1D FFT."""
        # Complex input
        r = np.random.rand(100).astype(np.float32)
        i = np.random.rand(100).astype(np.float32)
        a_np = r + 1j * i
        self.check_fft_correctness('fft', np.fft.fft, a_np)

    def test_fft_with_padding(self):
        """Test FFT with padding."""
        r = np.random.rand(100).astype(np.float32)
        i = np.random.rand(100).astype(np.float32)
        a_np = r + 1j * i
        self.check_fft_correctness('fft', np.fft.fft, a_np, n=120)

    def test_fft_with_truncation(self):
        """Test FFT with truncation."""
        r = np.random.rand(100).astype(np.float32)
        i = np.random.rand(100).astype(np.float32)
        a_np = r + 1j * i
        self.check_fft_correctness('fft', np.fft.fft, a_np, n=80)

    def test_fft_different_axes(self):
        """Test FFT along different axes."""
        r = np.random.rand(10, 20, 30).astype(np.float32)
        i = np.random.rand(10, 20, 30).astype(np.float32)
        a_np = r + 1j * i
        
        for axis in [0, 1, 2, -1, -2]:
            with self.subTest(axis=axis):
                self.check_fft_correctness('fft', np.fft.fft, a_np, axis=axis)

    def test_ifft_basic(self):
        """Test basic 1D inverse FFT."""
        r = np.random.rand(100).astype(np.float32)
        i = np.random.rand(100).astype(np.float32)
        a_np = r + 1j * i
        self.check_fft_correctness('ifft', np.fft.ifft, a_np)

    def test_ifft_with_sizes(self):
        """Test inverse FFT with different sizes."""
        r = np.random.rand(100).astype(np.float32)
        i = np.random.rand(100).astype(np.float32)
        a_np = r + 1j * i
        
        self.check_fft_correctness('ifft', np.fft.ifft, a_np, n=80)
        self.check_fft_correctness('ifft', np.fft.ifft, a_np, n=120)

    def test_rfft_basic(self):
        """Test basic 1D real FFT."""
        a_np = np.random.rand(100).astype(np.float32)
        self.check_fft_correctness('rfft', np.fft.rfft, a_np)

    def test_rfft_with_sizes(self):
        """Test real FFT with different sizes."""
        a_np = np.random.rand(100).astype(np.float32)
        
        self.check_fft_correctness('rfft', np.fft.rfft, a_np, n=80)
        self.check_fft_correctness('rfft', np.fft.rfft, a_np, n=120)

    def test_irfft_basic(self):
        """Test basic 1D inverse real FFT."""
        # Create complex input from rfft
        a_real = np.random.rand(100).astype(np.float32)
        a_np = np.fft.rfft(a_real)
        self.check_fft_correctness('irfft', np.fft.irfft, a_np)

    def test_irfft_with_size(self):
        """Test inverse real FFT with explicit size."""
        a_real = np.random.rand(100).astype(np.float32)
        a_np = np.fft.rfft(a_real)
        self.check_fft_correctness('irfft', np.fft.irfft, a_np, n=100)

    def test_fft2_basic(self):
        """Test basic 2D FFT."""
        r = np.random.rand(32, 32).astype(np.float32)
        i = np.random.rand(32, 32).astype(np.float32)
        a_np = r + 1j * i
        self.check_fft_correctness('fft2', np.fft.fft2, a_np)

    def test_fft2_with_shape(self):
        """Test 2D FFT with specific shape."""
        r = np.random.rand(32, 32).astype(np.float32)
        i = np.random.rand(32, 32).astype(np.float32)
        a_np = r + 1j * i
        self.check_fft_correctness('fft2', np.fft.fft2, a_np, s=(40, 40))

    def test_fft2_different_axes(self):
        """Test 2D FFT along different axes."""
        r = np.random.rand(10, 20, 30).astype(np.float32)
        i = np.random.rand(10, 20, 30).astype(np.float32)
        a_np = r + 1j * i
        
        self.check_fft_correctness('fft2', np.fft.fft2, a_np, axes=(0, 1))
        self.check_fft_correctness('fft2', np.fft.fft2, a_np, axes=(1, 2))

    def test_ifft2_basic(self):
        """Test basic 2D inverse FFT."""
        r = np.random.rand(32, 32).astype(np.float32)
        i = np.random.rand(32, 32).astype(np.float32)
        a_np = r + 1j * i
        self.check_fft_correctness('ifft2', np.fft.ifft2, a_np)

    def test_fftn_basic(self):
        """Test basic N-D FFT."""
        r = np.random.rand(8, 8, 8).astype(np.float32)
        i = np.random.rand(8, 8, 8).astype(np.float32)
        a_np = r + 1j * i
        self.check_fft_correctness('fftn', np.fft.fftn, a_np)

    def test_fftn_partial_axes(self):
        """Test N-D FFT on subset of axes."""
        r = np.random.rand(8, 16, 12).astype(np.float32)
        i = np.random.rand(8, 16, 12).astype(np.float32)
        a_np = r + 1j * i
        
        self.check_fft_correctness('fftn', np.fft.fftn, a_np, axes=(0, 2))
        self.check_fft_correctness('fftn', np.fft.fftn, a_np, axes=(1,))

    def test_fftn_with_shape(self):
        """Test N-D FFT with specific shape."""
        r = np.random.rand(8, 8, 8).astype(np.float32)
        i = np.random.rand(8, 8, 8).astype(np.float32)
        a_np = r + 1j * i
        self.check_fft_correctness('fftn', np.fft.fftn, a_np, s=(10, 10, 10))

    def test_ifftn_basic(self):
        """Test basic N-D inverse FFT."""
        r = np.random.rand(8, 8, 8).astype(np.float32)
        i = np.random.rand(8, 8, 8).astype(np.float32)
        a_np = r + 1j * i
        self.check_fft_correctness('ifftn', np.fft.ifftn, a_np)

    def test_rfft2_basic(self):
        """Test basic 2D real FFT."""
        a_np = np.random.rand(32, 32).astype(np.float32)
        self.check_fft_correctness('rfft2', np.fft.rfft2, a_np)

    def test_irfft2_basic(self):
        """Test basic 2D inverse real FFT."""
        a_real = np.random.rand(32, 32).astype(np.float32)
        a_np = np.fft.rfft2(a_real)
        self.check_fft_correctness('irfft2', np.fft.irfft2, a_np)

    def test_rfftn_basic(self):
        """Test basic N-D real FFT."""
        a_np = np.random.rand(8, 8, 8).astype(np.float32)
        self.check_fft_correctness('rfftn', np.fft.rfftn, a_np)

    def test_rfftn_partial_axes(self):
        """Test N-D real FFT on subset of axes."""
        a_np = np.random.rand(8, 16, 12).astype(np.float32)
        self.check_fft_correctness('rfftn', np.fft.rfftn, a_np, axes=(0, 2))

    def test_irfftn_basic(self):
        """Test basic N-D inverse real FFT."""
        a_real = np.random.rand(8, 8, 8).astype(np.float32)
        a_np = np.fft.rfftn(a_real)
        self.check_fft_correctness('irfftn', np.fft.irfftn, a_np)

    def test_fft_roundtrip(self):
        """Test FFT followed by IFFT returns original."""
        from mlx import fft as mlx_fft
        
        r = np.random.rand(100).astype(np.float32)
        i = np.random.rand(100).astype(np.float32)
        a_np = r + 1j * i
        
        a_mx = mx.array(a_np)
        fft_result = mlx_fft.fft(a_mx)
        roundtrip = mlx_fft.ifft(fft_result)
        mx.eval(roundtrip)
        
        np.testing.assert_allclose(
            np.array(roundtrip),
            a_np,
            atol=1e-5,
            rtol=1e-5
        )

    def test_rfft_roundtrip(self):
        """Test RFFT followed by IRFFT returns original."""
        from mlx import fft as mlx_fft
        
        a_np = np.random.rand(100).astype(np.float32)
        
        a_mx = mx.array(a_np)
        fft_result = mlx_fft.rfft(a_mx)
        roundtrip = mlx_fft.irfft(fft_result)
        mx.eval(roundtrip)
        
        np.testing.assert_allclose(
            np.array(roundtrip),
            a_np,
            atol=1e-5,
            rtol=1e-5
        )

    def test_fftshift(self):
        """Test fftshift (should use C++ implementation)."""
        from mlx import fft as mlx_fft
        
        a_np = np.random.rand(100).astype(np.float32)
        expected = np.fft.fftshift(a_np)
        
        a_mx = mx.array(a_np)
        result = mlx_fft.fftshift(a_mx)
        mx.eval(result)
        
        np.testing.assert_allclose(
            np.array(result),
            expected,
            atol=1e-6,
            rtol=1e-6
        )

    def test_ifftshift(self):
        """Test ifftshift (should use C++ implementation)."""
        from mlx import fft as mlx_fft
        
        a_np = np.random.rand(100).astype(np.float32)
        expected = np.fft.ifftshift(a_np)
        
        a_mx = mx.array(a_np)
        result = mlx_fft.ifftshift(a_mx)
        mx.eval(result)
        
        np.testing.assert_allclose(
            np.array(result),
            expected,
            atol=1e-6,
            rtol=1e-6
        )

    def test_large_fft(self):
        """Test FFT with larger arrays."""
        r = np.random.rand(1024).astype(np.float32)
        i = np.random.rand(1024).astype(np.float32)
        a_np = r + 1j * i
        self.check_fft_correctness('fft', np.fft.fft, a_np)

    def test_batch_fft(self):
        """Test batched FFT operations."""
        # 2D array where each row is an independent FFT
        r = np.random.rand(10, 128).astype(np.float32)
        i = np.random.rand(10, 128).astype(np.float32)
        a_np = r + 1j * i
        
        # FFT along last axis (each row independently)
        self.check_fft_correctness('fft', np.fft.fft, a_np, axis=-1)


@unittest.skipIf(SKIP_CUDA_FFT_TESTS, SKIP_REASON)
class TestCUDAFFTFallback(unittest.TestCase):
    """Test that CUDA FFT gracefully falls back when nvmath is not available."""

    def test_import_without_nvmath(self):
        """Test that importing mlx.fft works even without nvmath."""
        # This should not raise, even if nvmath is missing
        import mlx.fft
        
        # All functions should be importable
        self.assertTrue(hasattr(mlx.fft, 'fft'))
        self.assertTrue(hasattr(mlx.fft, 'ifft'))
        self.assertTrue(hasattr(mlx.fft, 'rfft'))


if __name__ == "__main__":
    unittest.main()

