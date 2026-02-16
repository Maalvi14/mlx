"""Simple check that MLX is running with CUDA."""
import mlx.core as mx

def main():
    print("MLX CUDA check")
    print("-" * 40)

    # CUDA backend
    cuda_available = mx.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    # GPU (Metal or CUDA)
    gpu_available = mx.is_available(mx.gpu)
    print(f"GPU available:  {gpu_available}")

    # Default device
    default = mx.default_device()
    print(f"Default device: {default}")

    if gpu_available:
        gpu_count = mx.device_count(mx.gpu)
        print(f"GPU count:      {gpu_count}")
        for i in range(gpu_count):
            info = mx.device_info(mx.Device(mx.gpu, i))
            print(f"  GPU {i}: {info.get('device_name', '?')}")

        # Run a tiny computation on GPU
        with mx.stream(mx.gpu):
            a = mx.ones((2, 2))
            b = mx.ones((2, 2))
            c = mx.add(a, b)
            mx.eval(c)
        print(f"\nGPU compute test: 1+1 on GPU -> {c[0, 0].item()} (expected 2.0)")
        print("MLX with CUDA is working.")
    else:
        print("No GPU available (MLX may be using CPU only).")

if __name__ == "__main__":
    main()
