.. _fft:

FFT
===

On CUDA, FFT is implemented via `nvmath-python`_ (cuFFT). Install optional
dependencies for CUDA FFT: ``pip install nvmath-python[cu12] cupy-cuda12x``
(or ``[cu13]`` / ``cupy-cuda13x`` to match your CUDA toolkit). Data is passed
via DLPack (zero-copy where possible). See also :func:`mlx.core.from_dlpack`.

.. _nvmath-python: https://github.com/NVIDIA/nvmath-python

.. currentmodule:: mlx.core.fft

.. autosummary:: 
  :toctree: _autosummary

  fft
  ifft
  fft2
  ifft2
  fftn
  ifftn
  rfft
  irfft
  rfft2
  irfft2
  rfftn
  irfftn
  fftshift
  ifftshift
