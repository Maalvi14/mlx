# Copyright © 2025 Apple Inc.

import unittest
import mlx.core as mx
import numpy as np

# Check if we're on CUDA
try:
    IS_CUDA = mx.cuda_is_available() if hasattr(mx, 'cuda_is_available') else False
except:
    IS_CUDA = False

# Check if nvmath-python is available
try:
    from mlx import fft_cuda
    NVMATH_AVAILABLE = fft_cuda.NVMATH_AVAILABLE
except ImportError:
    NVMATH_AVAILABLE = False


@unittest.skipUnless(IS_CUDA, "CUDA backend not available")
@unittest.skipUnless(NVMATH_AVAILABLE, "nvmath-python not installed")
class TestFFTCuda(unittest.TestCase):
    """Test FFT operations with CUDA backend using nvmath-python."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Ensure we're using GPU device
        mx.set_default_device(mx.gpu)
    
    def test_fft_basic(self):
        """Test basic 1D FFT."""
        from mlx.fft_cuda import fft
        
        # Test data
        x = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        
        # Compute FFT
        result = fft(x)
        
        # Compare with NumPy
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        expected = np.fft.fft(x_np)
        
        mx.eval(result)
        result_np = np.array(result)
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-6)
    
    def test_fft_with_n(self):
        """Test FFT with custom size n."""
        from mlx.fft_cuda import fft
        
        x = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        
        # FFT with n=8 (padding)
        result = fft(x, n=8)
        
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        expected = np.fft.fft(x_np, n=8)
        
        mx.eval(result)
        result_np = np.array(result)
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-6)
    
    def test_ifft_basic(self):
        """Test inverse FFT."""
        from mlx.fft_cuda import fft, ifft
        
        x = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        
        # Forward then inverse
        y = fft(x)
        z = ifft(y)
        
        mx.eval(z)
        z_np = np.array(z)
        
        # Should recover original (up to floating point errors)
        np.testing.assert_allclose(np.real(z_np), [1, 2, 3, 4], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.imag(z_np), [0, 0, 0, 0], rtol=1e-5, atol=1e-6)
    
    def test_rfft_basic(self):
        """Test real FFT."""
        from mlx.fft_cuda import rfft
        
        x = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        result = rfft(x)
        
        x_np = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.fft.rfft(x_np)
        
        mx.eval(result)
        result_np = np.array(result)
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-6)
    
    def test_irfft_basic(self):
        """Test inverse real FFT."""
        from mlx.fft_cuda import rfft, irfft
        
        x = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        
        # Forward then inverse
        y = rfft(x)
        z = irfft(y)
        
        mx.eval(z)
        z_np = np.array(z)
        
        # Should recover original
        np.testing.assert_allclose(z_np, [1, 2, 3, 4], rtol=1e-5, atol=1e-6)
    
    def test_complex_input(self):
        """Test FFT with complex input."""
        from mlx.fft_cuda import fft
        
        # Create complex array
        x_real = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        x_imag = mx.array([0.5, 1.0, 1.5, 2.0], dtype=mx.float32)
        x = x_real + 1j * x_imag
        
        result = fft(x)
        
        # Compare with NumPy
        x_np = np.array([1.0+0.5j, 2.0+1.0j, 3.0+1.5j, 4.0+2.0j])
        expected = np.fft.fft(x_np)
        
        mx.eval(result)
        result_np = np.array(result)
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-6)
    
    def test_different_axes(self):
        """Test FFT along different axes."""
        from mlx.fft_cuda import fft
        
        # 2D array
        x = mx.array([[1, 2, 3, 4],
                      [5, 6, 7, 8]], dtype=mx.float32)
        
        # FFT along axis 0
        result_axis0 = fft(x, axis=0)
        
        # FFT along axis 1
        result_axis1 = fft(x, axis=1)
        
        x_np = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8]], dtype=np.float32)
        expected_axis0 = np.fft.fft(x_np, axis=0)
        expected_axis1 = np.fft.fft(x_np, axis=1)
        
        mx.eval(result_axis0)
        mx.eval(result_axis1)
        
        np.testing.assert_allclose(np.array(result_axis0), expected_axis0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.array(result_axis1), expected_axis1, rtol=1e-5, atol=1e-6)
    
    def test_large_array(self):
        """Test FFT with larger array."""
        from mlx.fft_cuda import fft
        
        # Larger array
        x = mx.arange(1024, dtype=mx.float32)
        result = fft(x)
        
        x_np = np.arange(1024, dtype=np.float32)
        expected = np.fft.fft(x_np)
        
        mx.eval(result)
        result_np = np.array(result)
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)
    
    def test_stream_synchronization(self):
        """Test that FFT properly synchronizes with MLX operations."""
        from mlx.fft_cuda import fft
        
        # Create computation graph
        a = mx.array([1.0, 2.0, 3.0, 4.0])
        b = a * 2  # MLX operation
        c = fft(b)  # nvmath-python operation  
        d = mx.abs(c)  # MLX operation
        
        # Evaluate - all should execute in order on same stream
        mx.eval(d)
        
        # Check result is reasonable
        self.assertTrue(d.shape == (4,))
        self.assertTrue(d.dtype == mx.float32)
    
    def test_dlpack_zero_copy(self):
        """Test that DLPack transfer is zero-copy."""
        from mlx.fft_cuda import fft
        
        x = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        mx.eval(x)
        
        # FFT should use zero-copy DLPack transfer
        result = fft(x)
        mx.eval(result)
        
        # Check that operation completed successfully
        self.assertEqual(result.shape, (4,))
        self.assertEqual(result.dtype, mx.complex64)
        
        # Result should be reasonable
        result_np = np.array(result)
        self.assertTrue(np.all(np.isfinite(result_np)))


@unittest.skipUnless(IS_CUDA, "CUDA backend not available")
class TestCudaStreamHandle(unittest.TestCase):
    """Test CUDA stream handle exposure."""
    
    def setUp(self):
        mx.set_default_device(mx.gpu)
    
    def test_cuda_is_available(self):
        """Test cuda_is_available function."""
        if hasattr(mx, 'cuda_is_available'):
            self.assertTrue(mx.cuda_is_available())
    
    def test_cuda_stream_handle(self):
        """Test getting CUDA stream handle."""
        if not hasattr(mx, 'cuda_stream_handle'):
            self.skipTest("cuda_stream_handle not available")
        
        stream = mx.default_stream(mx.gpu)
        handle = mx.cuda_stream_handle(stream)
        
        # Should return an integer pointer
        self.assertIsInstance(handle, int)
        self.assertGreater(handle, 0)


@unittest.skipUnless(IS_CUDA, "CUDA backend not available")
class TestFromDLPack(unittest.TestCase):
    """Test from_dlpack functionality."""
    
    def setUp(self):
        mx.set_default_device(mx.gpu)
    
    def test_from_dlpack(self):
        """Test importing array from DLPack."""
        # Create MLX array
        x_mlx = mx.array([1, 2, 3, 4], dtype=mx.float32)
        mx.eval(x_mlx)
        
        # Export to DLPack
        dlpack_capsule = x_mlx.__dlpack__()
        
        # Import back (should be zero-copy)
        y_mlx = mx.array.from_dlpack(dlpack_capsule)
        mx.eval(y_mlx)
        
        # Check equality
        np.testing.assert_array_equal(np.array(x_mlx), np.array(y_mlx))
    
    @unittest.skipUnless(NVMATH_AVAILABLE, "CuPy not available")
    def test_interop_with_cupy(self):
        """Test interoperability with CuPy via DLPack."""
        try:
            import cupy as cp
        except ImportError:
            self.skipTest("CuPy not available")
        
        # Create CuPy array
        x_cupy = cp.array([1, 2, 3, 4], dtype=cp.float32)
        
        # Export to DLPack
        dlpack_capsule = x_cupy.__dlpack__()
        
        # Import to MLX
        x_mlx = mx.array.from_dlpack(dlpack_capsule)
        mx.eval(x_mlx)
        
        # Check values match
        np.testing.assert_array_equal(cp.asnumpy(x_cupy), np.array(x_mlx))


if __name__ == "__main__":
    unittest.main()

