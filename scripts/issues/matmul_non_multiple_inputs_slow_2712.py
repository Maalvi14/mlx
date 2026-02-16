"""Benchmark: slicing k and v to 2047 before matmuls hits a bad kernel (~3x slower).
   Issue: https://github.com/ml-explore/mlx/issues/2712
"""
import time
import mlx.core as mx

H = 16
D = 128
T = 2048


def run_benchmark(Ts):
    q = mx.random.normal(shape=(H, Ts, D))
    k = mx.random.normal(shape=(H, T, D))[:, :Ts, :]
    v = mx.random.normal(shape=(H, T, D))[:, :Ts, :]

    def fun(q):
        for _ in range(10):
            scores = q @ k.swapaxes(-1, -2)
            q = scores @ v
        return q

    # Warmup
    for _ in range(10):
        mx.eval(fun(q))

    tic = time.time()
    for _ in range(10):
        mx.eval(fun(q))
    toc = time.time()
    return 1e3 * (toc - tic)


# Ts=2047: hits bad kernel (non-multiple dims)
ms_2047 = run_benchmark(2047)
print(f"Ts=2047 (sliced, bad kernel): ms={ms_2047:.3f}")

# Ts=2048: good kernel
ms_2048 = run_benchmark(2048)
print(f"Ts=2048 (good kernel):        ms={ms_2048:.3f}")

ratio = ms_2047 / ms_2048
print(f"Ratio (2047/2048):            {ratio:.2f}x")
