import mlx.core as mx

def main():
    print("Testing MLX FFT with CUDA backend...")
    print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"Default device: {mx.default_device()}")

    # Create a simple 1D signal
    x = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
    print(f"Input array: {x}")

    try:
        # Attempt FFT operation that triggers the CUDA error
        result = mx.fft.fft(x)
        print(f"FFT result: {result}")
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        raise

if __name__ == "__main__":
    main()