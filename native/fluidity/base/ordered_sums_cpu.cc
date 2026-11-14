#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <utility>

#include "fluidity/base/buffer.h"
#include "fluidity/base/ordered_sums.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/tree_sum.h"
#include "fluidity/base/tree_sum_layers.h"

#if defined(FLUIDITY_HAS_GPU)
#include "fluidity/base/ordered_sums_gpu.h"
#include "fluidity/base/run_on_call.h"
#endif

namespace fluidity {
namespace {

// The GPU sum buffers: max_count products, then the two layer arrays, which start at a multiple of kGpuAreaAlignment
// bytes. BuildSumBuffers and ComputeSumBuffersSize both read these two constants, so the size reported is the size
// allocated. The layer arrays hold ComputeLayerValueCount(max_count) <= 3/4 max_count + 2 doubles, at most 6 bytes
// per value plus 16; with the products' 8 bytes per value and at most 63 bytes of alignment, 14 and 128 cover them,
// and the static_asserts below check chosen counts.
constexpr std::uint32_t kGpuBytesPerItem = 14;
constexpr std::uint64_t kGpuFixedBytes = 128;
// Bytes. The bytes of the sum buffers start at a multiple of kBufferAlignmentBytes, so the second area is aligned
// too, more than the 8 bytes a double needs.
constexpr std::size_t kGpuAreaAlignment = kBufferAlignmentBytes;

// `byte_count` rounded up to a multiple of kGpuAreaAlignment.
constexpr std::size_t AlignUp(std::size_t byte_count) {
  return (byte_count + kGpuAreaAlignment - 1) / kGpuAreaAlignment * kGpuAreaAlignment;
}

// Where the layer arrays start in the bytes of the sum buffers, after max_count products.
constexpr std::size_t ComputeLayerOffset(std::size_t max_count) {
  return AlignUp(sizeof(double) * max_count);
}

// Whether the GPU areas for `max_count` values fit in the bytes BuildSumBuffers allocates for them.
constexpr bool GpuAreasFit(std::size_t max_count) {
  const std::size_t end = ComputeLayerOffset(max_count) + sizeof(double) * ComputeLayerValueCount(max_count);
  return end <= kGpuBytesPerItem * max_count + kGpuFixedBytes;
}

static_assert(
    GpuAreasFit(0) && GpuAreasFit(1) && GpuAreasFit(2) && GpuAreasFit(3) && GpuAreasFit(5) && GpuAreasFit(7)
    && GpuAreasFit(1000) && GpuAreasFit(1000003) && GpuAreasFit(std::size_t{1} << 40U)
);

#if defined(FLUIDITY_HAS_GPU)

GpuSumBufferAreas ToGpuAreas(std::uint8_t* bytes, std::size_t max_count) {
  // The bytes come from Buffer: GPU memory starting at a multiple of kBufferAlignmentBytes (64), and the layer offset
  // is a multiple of it, so each area is aligned for doubles.
  return {
      .products = reinterpret_cast<double*>(bytes),
      .layer_values = reinterpret_cast<double*>(bytes + ComputeLayerOffset(max_count)),
  };
}

// The answer of a GPU reduction, or the BufferError for how it ended.
std::expected<double, BufferError> ToBufferResult(RunOnCallStatus status, double result) {
  if (status != RunOnCallStatus::kDone) {
    return std::unexpected(ToBufferError(status));
  }
  return result;
}

#endif

// The size checks of the sums. `second_values` is null for a sum of one buffer.
std::expected<void, BufferError> CheckSumSizes(
    const Buffer<double>& values,
    const Buffer<double>* second_values,
    std::size_t count,
    const SumBuffers& buffers
) {
  const bool second_is_too_small = second_values != nullptr && count > second_values->size();
  if (count > values.size() || second_is_too_small) {
    // The caller asked for more values than a buffer holds: a bug in the caller.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  if (count > buffers.max_count()) {
    // The sum buffers were built for fewer values; the caller must build larger ones.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  return {};
}

}  // namespace

SumBuffersSize ComputeSumBuffersSize(RunOn run_on) {
  if (run_on == RunOn::kGpu) {
    return {.bytes_per_item = kGpuBytesPerItem, .fixed_bytes = kGpuFixedBytes};
  }
  // math::TreeSum needs no memory beyond its 64 waiting sums on the stack.
  return {.bytes_per_item = 0, .fixed_bytes = 0};
}

std::expected<SumBuffers, BufferError> BuildSumBuffers(RunOn run_on, std::size_t max_count) {
  const SumBuffersSize size = ComputeSumBuffersSize(run_on);
  // The byte count must not wrap around, or the sum buffers would be smaller than asked for.
  const bool byte_count_fits = size.bytes_per_item == 0
      || max_count <= (std::numeric_limits<std::size_t>::max() - size.fixed_bytes) / size.bytes_per_item;
  if (!byte_count_fits) {
    return std::unexpected(BufferError::kOutOfMemory);
  }
  const std::size_t byte_count = (size.bytes_per_item * max_count) + size.fixed_bytes;
  std::expected<Buffer<std::uint8_t>, BufferError> bytes = BuildBuffer<std::uint8_t>(run_on, byte_count);
  if (!bytes.has_value()) {
    return std::unexpected(bytes.error());
  }
  return SumBuffers(std::move(*bytes), max_count);
}

std::expected<double, BufferError>
ComputeTreeSum(const Buffer<double>& values, std::size_t count, SumBuffers& buffers) {
  // Requires: one processor (base/ordered_sums.h). A bug in the caller, so checked in debug builds only.
  assert(buffers.run_on() == values.run_on());
  if (const std::expected<void, BufferError> checked = CheckSumSizes(values, nullptr, count, buffers);
      !checked.has_value()) {
    return std::unexpected(checked.error());
  }
  if (count == 0) {
    return 0.0;
  }
  if (values.run_on() == RunOn::kGpu) {
#if defined(FLUIDITY_HAS_GPU)
    double sum = 0.0;
    const RunOnCallStatus status =
        ComputeGpuTreeSum(values.data(), count, ToGpuAreas(buffers.bytes_.data(), buffers.max_count_), &sum);
    return ToBufferResult(status, sum);
#else
    // Without GPU code BuildBuffer never makes a GPU buffer, so no caller reaches this.
    return std::unexpected(BufferError::kRunOnCallFailed);
#endif
  }
  return math::TreeSum(values.data(), count);
}

std::expected<double, BufferError>
ComputeTreeDot(const Buffer<double>& first, const Buffer<double>& second, std::size_t count, SumBuffers& buffers) {
  // Requires: one processor (base/ordered_sums.h). A bug in the caller, so checked in debug builds only.
  assert(second.run_on() == first.run_on());
  assert(buffers.run_on() == first.run_on());
  if (const std::expected<void, BufferError> checked = CheckSumSizes(first, &second, count, buffers);
      !checked.has_value()) {
    return std::unexpected(checked.error());
  }
  if (count == 0) {
    return 0.0;
  }
  if (first.run_on() == RunOn::kGpu) {
#if defined(FLUIDITY_HAS_GPU)
    double dot = 0.0;
    const RunOnCallStatus status = ComputeGpuTreeDot(
        first.data(),
        second.data(),
        count,
        ToGpuAreas(buffers.bytes_.data(), buffers.max_count_),
        &dot
    );
    return ToBufferResult(status, dot);
#else
    // Without GPU code BuildBuffer never makes a GPU buffer, so no caller reaches this.
    return std::unexpected(BufferError::kRunOnCallFailed);
#endif
  }
  const double* const first_values = first.data();
  const double* const second_values = second.data();
  // The library is built with -ffp-contract=off (/fp:precise on MSVC), so each product is rounded before the tree adds
  // it, as on the GPU.
  return math::TreeSumOf(count, [first_values, second_values](std::size_t index) {
    return first_values[index] * second_values[index];
  });
}

std::expected<double, BufferError>
ComputeLargestMagnitude(const Buffer<double>& values, std::size_t count, SumBuffers& buffers) {
  // Requires: one processor (base/ordered_sums.h). A bug in the caller, so checked in debug builds only.
  assert(buffers.run_on() == values.run_on());
  if (const std::expected<void, BufferError> checked = CheckSumSizes(values, nullptr, count, buffers);
      !checked.has_value()) {
    return std::unexpected(checked.error());
  }
  if (count == 0) {
    return 0.0;
  }
  if (values.run_on() == RunOn::kGpu) {
#if defined(FLUIDITY_HAS_GPU)
    double largest = 0.0;
    const RunOnCallStatus status = ComputeGpuLargestMagnitude(
        values.data(),
        count,
        ToGpuAreas(buffers.bytes_.data(), buffers.max_count_),
        &largest
    );
    return ToBufferResult(status, largest);
#else
    // Without GPU code BuildBuffer never makes a GPU buffer, so no caller reaches this.
    return std::unexpected(BufferError::kRunOnCallFailed);
#endif
  }
  // Starting from +0.0 changes nothing: every magnitude is +0.0 or more, and a NaN wins over it.
  const double* const cpu_values = values.data();
  double largest = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    largest = math::LargerMagnitude(largest, cpu_values[index]);
  }
  return largest;
}

}  // namespace fluidity
