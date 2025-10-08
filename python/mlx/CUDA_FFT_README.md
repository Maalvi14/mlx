# CUDA FFT Support in MLX

MLX now supports Fast Fourier Transform (FFT) operations on NVIDIA CUDA devices using NVIDIA's nvmath-python library.

## Installation

To use FFT operations on CUDA devices, you need to install nvmath-python:

```bash
pip install mlx[cuda]
```

Or install nvmath-python separately:

```bash
pip install nvmath-python
```

**Requirements:**
- CUDA 12.0 or later
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- nvmath-python >= 0.2.0

## Usage

### Basic FFT Operations

The FFT API automatically dispatches to the appropriate backend (CUDA or CPU/Metal) based on the device of your input array:

```python
import mlx.core as mx
from mlx import fft

# Set CUDA as default device
mx.set_default_device(mx.gpu)

# Create input array (will be on CUDA)
x = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)

# Perform FFT (automatically uses CUDA implementation)
result = fft.fft(x)
print(result)
```

### Supported Operations

All standard FFT operations are supported:

#### Complex-to-Complex FFT
```python
from mlx import fft

# 1D FFT
y = fft.fft(x)           # Forward FFT
x_recovered = fft.ifft(y) # Inverse FFT

# 2D FFT
y = fft.fft2(x)          # 2D Forward FFT
x_recovered = fft.ifft2(y) # 2D Inverse FFT

# N-D FFT
y = fft.fftn(x)          # N-D Forward FFT
x_recovered = fft.ifftn(y) # N-D Inverse FFT
```

#### Real-to-Complex FFT
```python
from mlx import fft

# Real input FFT (more efficient for real data)
x_real = mx.array([1.0, 2.0, 3.0, 4.0])
y = fft.rfft(x_real)     # Output is complex, size n//2 + 1
x_recovered = fft.irfft(y) # Recover real signal

# 2D Real FFT
y = fft.rfft2(x_real_2d)
x_recovered = fft.irfft2(y)

# N-D Real FFT
y = fft.rfftn(x_real_nd)
x_recovered = fft.irfftn(y)
```

### Advanced Usage

#### Custom FFT Size
```python
from mlx import fft

# Pad or truncate to specific size
x = mx.array([1.0, 2.0, 3.0, 4.0])
y = fft.fft(x, n=8)  # Pad to size 8
```

#### Specific Axes
```python
from mlx import fft

# FFT along specific axis
x = mx.random.normal(shape=(10, 20, 30))
y = fft.fft(x, axis=1)  # FFT along axis 1

# Multi-dimensional FFT on specific axes
y = fft.fftn(x, axes=(0, 2))  # FFT on axes 0 and 2
```

#### FFT Shift Operations
```python
from mlx import fft

# Shift zero-frequency to center
y_shifted = fft.fftshift(y)

# Inverse shift
y_original = fft.ifftshift(y_shifted)
```

## Performance

The CUDA FFT implementation provides significant performance improvements on NVIDIA GPUs:

- **Zero-copy data transfer**: Uses DLPack for efficient memory sharing
- **NVIDIA cuFFT**: Leverages NVIDIA's highly optimized cuFFT library via nvmath-python
- **Automatic dispatch**: Seamlessly switches between CUDA and CPU/Metal based on array device

## Fallback Behavior

If nvmath-python is not installed or CUDA is not available:
- Arrays on CPU/Metal devices use the existing optimized implementations
- Arrays on CUDA devices will raise an informative error suggesting to install nvmath-python
- The `mlx.fft` module can still be imported and used for CPU/Metal operations

## Migration Guide

### From `mlx.core.fft` to `mlx.fft`

If you're currently using the C++ FFT API:

**Old code:**
```python
import mlx.core as mx

x = mx.array([1, 2, 3, 4])
y = mx.fft.fft(x)  # Uses C++ implementation only
```

**New code for CUDA support:**
```python
import mlx.core as mx
from mlx import fft

mx.set_default_device(mx.gpu)  # Use CUDA
x = mx.array([1, 2, 3, 4])
y = fft.fft(x)  # Automatically uses CUDA if available
```

**Note:** `mlx.core.fft` continues to work and uses the C++ implementation. For CUDA support, use `mlx.fft`.

## API Reference

### 1D FFT Functions
- `fft.fft(a, n=None, axis=-1)` - Forward FFT
- `fft.ifft(a, n=None, axis=-1)` - Inverse FFT
- `fft.rfft(a, n=None, axis=-1)` - Real input FFT
- `fft.irfft(a, n=None, axis=-1)` - Inverse real FFT

### 2D FFT Functions
- `fft.fft2(a, s=None, axes=(-2, -1))` - 2D Forward FFT
- `fft.ifft2(a, s=None, axes=(-2, -1))` - 2D Inverse FFT
- `fft.rfft2(a, s=None, axes=(-2, -1))` - 2D Real input FFT
- `fft.irfft2(a, s=None, axes=(-2, -1))` - 2D Inverse real FFT

### N-D FFT Functions
- `fft.fftn(a, s=None, axes=None)` - N-D Forward FFT
- `fft.ifftn(a, s=None, axes=None)` - N-D Inverse FFT
- `fft.rfftn(a, s=None, axes=None)` - N-D Real input FFT
- `fft.irfftn(a, s=None, axes=None)` - N-D Inverse real FFT

### Helper Functions
- `fft.fftshift(a, axes=None)` - Shift zero-frequency to center
- `fft.ifftshift(a, axes=None)` - Inverse of fftshift

## Troubleshooting

### ImportError: No module named 'nvmath'

Install nvmath-python:
```bash
pip install nvmath-python
```

### CUDA FFT not being used

Ensure your array is on a CUDA device:
```python
import mlx.core as mx

# Check device
print(mx.default_device())  # Should show gpu

# Force CUDA
mx.set_default_device(mx.gpu)
x = mx.array([1, 2, 3, 4])
print(x.device)  # Should be Device(gpu, 0)
```

### Version compatibility

Ensure you have compatible versions:
```bash
python -c "import nvmath; print(nvmath.__version__)"  # Should be >= 0.2.0
python -c "import mlx.core as mx; print(mx.__version__)"
```

## Examples

### Signal Processing
```python
import mlx.core as mx
from mlx import fft
import numpy as np

# Generate a signal
t = mx.linspace(0, 1, 1000, dtype=mx.float32)
signal = mx.sin(2 * mx.pi * 50 * t) + 0.5 * mx.sin(2 * mx.pi * 120 * t)

# Compute FFT
spectrum = fft.fft(signal)
magnitude = mx.abs(spectrum)

# Find dominant frequencies
frequencies = mx.arange(len(signal)) / len(signal)
```

### Image Processing (2D FFT)
```python
import mlx.core as mx
from mlx import fft

# Load or create image
image = mx.random.normal(shape=(512, 512))

# Compute 2D FFT
freq_domain = fft.fft2(image)

# Apply frequency filter
freq_shifted = fft.fftshift(freq_domain)
# ... apply filter ...
filtered = fft.ifftshift(freq_shifted)

# Inverse FFT to get filtered image
filtered_image = fft.ifft2(filtered).real
```

## Performance Tips

1. **Use real FFT for real inputs**: `rfft()` is faster than `fft()` for real-valued data
2. **Batch operations**: Process multiple signals in a single array for better performance
3. **Power-of-2 sizes**: FFT is most efficient when array size is a power of 2
4. **Reuse arrays**: Minimize memory allocations by reusing output arrays when possible

## Citation

If you use CUDA FFT support in MLX, please cite:
- MLX: https://github.com/ml-explore/mlx
- nvmath-python: https://github.com/NVIDIA/nvmath-python

## See Also

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [nvmath-python Documentation](https://docs.nvidia.com/cuda/nvmath-python/)
- [NVIDIA cuFFT](https://developer.nvidia.com/cufft)

