// Copyright © 2023-2024 Apple Inc.

#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "mlx/stream.h"
#include "mlx/utils.h"

#ifdef MLX_CUDA_AVAILABLE
#include "mlx/backend/cuda/device.h"
#include <cuda_runtime.h>
#endif

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

// Create the StreamContext on enter and delete on exit.
class PyStreamContext {
 public:
  PyStreamContext(mx::StreamOrDevice s) : _inner(nullptr) {
    if (std::holds_alternative<std::monostate>(s)) {
      throw std::runtime_error(
          "[StreamContext] Invalid argument, please specify a stream or device.");
    }
    _s = s;
  }

  void enter() {
    _inner = new mx::StreamContext(_s);
  }

  void exit() {
    if (_inner != nullptr) {
      delete _inner;
      _inner = nullptr;
    }
  }

 private:
  mx::StreamOrDevice _s;
  mx::StreamContext* _inner;
};

void init_stream(nb::module_& m) {
  nb::class_<mx::Stream>(
      m,
      "Stream",
      R"pbdoc(
      A stream for running operations on a given device.
      )pbdoc")
      .def_ro("device", &mx::Stream::device)
      .def(
          "__repr__",
          [](const mx::Stream& s) {
            std::ostringstream os;
            os << s;
            return os.str();
          })
      .def("__eq__", [](const mx::Stream& s, const nb::object& other) {
        return nb::isinstance<mx::Stream>(other) &&
            s == nb::cast<mx::Stream>(other);
      });

  nb::implicitly_convertible<mx::Device::DeviceType, mx::Device>();

  m.def(
      "default_stream",
      &mx::default_stream,
      "device"_a,
      R"pbdoc(Get the device's default stream.)pbdoc");
  m.def(
      "set_default_stream",
      &mx::set_default_stream,
      "stream"_a,
      R"pbdoc(
        Set the default stream.

        This will make the given stream the default for the
        streams device. It will not change the default device.

        Args:
          stream (stream): Stream to make the default.
      )pbdoc");
  m.def(
      "new_stream",
      &mx::new_stream,
      "device"_a,
      R"pbdoc(Make a new stream on the given device.)pbdoc");

  nb::class_<PyStreamContext>(m, "StreamContext", R"pbdoc(
        A context manager for setting the current device and stream.

        See :func:`stream` for usage.

        Args:
            s: The stream or device to set as the default.
  )pbdoc")
      .def(nb::init<mx::StreamOrDevice>(), "s"_a)
      .def("__enter__", [](PyStreamContext& scm) { scm.enter(); })
      .def(
          "__exit__",
          [](PyStreamContext& scm,
             const std::optional<nb::type_object>& exc_type,
             const std::optional<nb::object>& exc_value,
             const std::optional<nb::object>& traceback) { scm.exit(); },
          "exc_type"_a = nb::none(),
          "exc_value"_a = nb::none(),
          "traceback"_a = nb::none());
  m.def(
      "stream",
      [](mx::StreamOrDevice s) { return PyStreamContext(s); },
      "s"_a,
      R"pbdoc(
        Create a context manager to set the default device and stream.

        Args:
            s: The :obj:`Stream` or :obj:`Device` to set as the default.

        Returns:
            A context manager that sets the default device and stream.

        Example:

        .. code-block::python

          import mlx.core as mx

          # Create a context manager for the default device and stream.
          with mx.stream(mx.cpu):
              # Operations here will use mx.cpu by default.
              pass
      )pbdoc");
  m.def(
      "synchronize",
      [](const std::optional<mx::Stream>& s) {
        s ? mx::synchronize(s.value()) : mx::synchronize();
      },
      "stream"_a = nb::none(),
      R"pbdoc(
      Synchronize with the given stream.

      Args:
        stream (Stream, optional): The stream to synchronize with. If ``None``
           then the default stream of the default device is used.
           Default: ``None``.
      )pbdoc");

#ifdef MLX_CUDA_AVAILABLE
  m.def(
      "cuda_stream_handle",
      [](const mx::Stream& s) -> uintptr_t {
        // Validate stream is GPU/CUDA
        if (s.device != mx::Device::gpu) {
          throw std::runtime_error(
              "cuda_stream_handle: Stream is not a GPU stream");
        }

        // Check if CUDA backend is available
        if (!mx::cu::is_available()) {
          throw std::runtime_error(
              "cuda_stream_handle: CUDA backend not available");
        }

        // Get command encoder for this stream
        auto& encoder = mx::cu::get_command_encoder(s);

        // Get underlying cudaStream_t handle
        cudaStream_t cuda_stream = encoder.stream();

        // Return as integer pointer for Python interop
        return reinterpret_cast<uintptr_t>(cuda_stream);
      },
      "stream"_a,
      R"pbdoc(
        Get the CUDA stream handle as an integer pointer.

        This function exposes the underlying cudaStream_t handle for
        interoperability with external CUDA libraries like nvmath-python.
        The returned integer can be cast back to cudaStream_t in C/C++
        code or passed to libraries that accept stream pointers.

        Args:
            stream (Stream): The MLX stream (must be a GPU stream)

        Returns:
            int: The CUDA stream handle as an integer pointer

        Raises:
            RuntimeError: If stream is not a GPU stream or CUDA unavailable

        Example:
            >>> import mlx.core as mx
            >>> stream = mx.default_stream(mx.gpu)
            >>> handle = mx.cuda_stream_handle(stream)
            >>> print(f"Stream handle: 0x{handle:x}")
      )pbdoc");

  m.def(
      "cuda_is_available",
      []() -> bool { return mx::cu::is_available(); },
      R"pbdoc(
        Check if CUDA backend is available.

        Returns:
            bool: True if CUDA backend is compiled and available
      )pbdoc");
#endif
}
