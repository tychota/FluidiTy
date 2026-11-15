// Sums, dot products and largest magnitudes over buffers, in one fixed order on either processor.
//
// Purpose: give every solver's convergence test and every conservation total one way to reduce a buffer of doubles
//   to one double, with the same bits on every run and on both processors, unless the result is NaN, whose bits
//   IEEE 754 leaves open.
//
// Idea: ComputeTreeSum and ComputeTreeDot add in the fixed tree of math::TreeSum (base/tree_sum.h): values padded with
//   zeros to the next power of two, each pair sum its left side plus its right side. The CPU calls math::TreeSum. The
//   GPU adds one layer of the tree at a time, all pairs of a layer at once, in the working buffers of a SumBuffers.
//   Both add the same pairs, and IEEE 754 rounds each addition the same way on both, so the bits are equal unless the
//   result is NaN, whose bits IEEE 754 leaves open. No atomic additions, no library reduction whose order is
//   unspecified. ComputeLargestMagnitude needs no fixed order, because picking the larger of two magnitudes rounds
//   nothing; it returns NaN if any value is NaN. Each sum takes a SumBuffers, built once for up to max_count values
//   on one processor, and allocates nothing. The CPU needs no working array, so CPU sum buffers hold no bytes, but
//   the count is still checked against them, so code written for the GPU runs unchanged on the CPU.
//   Example: a buffer {1e16, 1, -1e16, 1} sums to 0.0 on both processors; a left-to-right loop would give 1.0.
//
// Owns:    SumBuffers, BuildSumBuffers, SumBuffersSize, ComputeSumBuffersSize, ComputeTreeSum, ComputeTreeDot,
//          ComputeLargestMagnitude
// Uses:    Buffer and BufferError (base/buffer.h), RunMemory (base/run_memory.h), RunOn (base/run_on.h), CPU-only
//          guard (base/cpu_only.h)
// Implementation uses: math::TreeSum and math::LargerMagnitude (base/tree_sum.h), the layer sizes
//          (base/tree_sum_layers.h) and, in a build with GPU code, the private GPU half (base/ordered_sums_gpu.h)
// Used by: every solver's convergence test and every conservation total
// Read:    docs/method-paper/fluidity-method.typ <sec-fixed-order-sums>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <utility>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"

namespace fluidity {

class SumBuffers;

// Builds the working buffers of ComputeTreeSum, ComputeTreeDot and ComputeLargestMagnitude on one processor, for up
// to max_count values, so that no sum allocates.
//
// Method: allocates one buffer of ComputeSumBuffersSize(run_on).bytes_per_item * max_count +
//   ComputeSumBuffersSize(run_on).fixed_bytes bytes on `run_on` with BuildBuffer. On the GPU it holds the products
//   of a dot product and the two arrays of the layers (base/tree_sum_layers.h); on the CPU it holds nothing.
// Example: BuildSumBuffers(RunOn::kGpu, 1000) holds 14 * 1000 + 128 = 14,128 bytes; on the CPU, 0 bytes.
// Fails: kOutOfMemory     max_count is too large to count in bytes, or the processor could not provide the memory
//        kRunOnCallFailed RunOn::kGpu was asked for and the GPU cannot run (also in a build without GPU code)
//        On failure nothing stays allocated and no buffers are returned.
// Read: docs/method-paper/fluidity-method.typ <sec-fixed-order-sums>
std::expected<SumBuffers, BufferError> BuildSumBuffers(RunOn run_on, std::size_t max_count);

// Builds the same working buffers as one piece of `run_memory` instead of memory of their own, so that the run's one
// reserved block holds them (base/run_memory.h). Tests and tools use the form above.
//
// Method: one BuildBuffer from `run_memory` of exactly the bytes the form above allocates, so the block hands out
//   ComputeRunMemoryPieceBytes(bytes_per_item * max_count + fixed_bytes) for it. The sums behave the same with either.
// Example: on the GPU, max_count 1000 holds 14,128 bytes and takes a piece of 14,336; on the CPU it takes nothing.
// Requires: run_on == run_memory.run_on(), checked by assert in debug builds.
// Fails: kRunMemoryTooSmall  the piece does not fit in run_memory.unused_bytes(), or max_count is too large to count
//                            in bytes; nothing is taken and no buffers are returned
// Read: docs/method-paper/fluidity-method.typ <sec-run-memory>
std::expected<SumBuffers, BufferError> BuildSumBuffers(RunOn run_on, std::size_t max_count, RunMemory& run_memory);

// Exact bytes BuildSumBuffers allocates: bytes_per_item * max_count + fixed_bytes.
struct SumBuffersSize {
  std::uint32_t bytes_per_item = 0;  // Bytes for each value of max_count.
  std::uint64_t fixed_bytes = 0;  // Bytes that do not depend on max_count.
};

// The bytes BuildSumBuffers allocates on `run_on`, for the memory plan of a module.
//
// Method: the same constants BuildSumBuffers allocates from. The GPU holds max_count products (8 bytes each) and
//   the two layer arrays (at most 3/4 of a double per value, plus a few), the second array starting at a multiple
//   of 64 bytes: 14 bytes per value and 128 bytes cover them. The CPU holds nothing.
//   In a memory estimate the buffers are one entry, so the estimate equals the one piece they take: counted_per
//   kGivenCount, bytes_each 1, given_count bytes_per_item * max_count + fixed_bytes at the plan's max_count. A
//   per-item entry plus a fixed entry would round twice and reserve too much; a per-item entry alone too little.
// Example: ComputeSumBuffersSize(RunOn::kGpu) gives {14, 128}; ComputeSumBuffersSize(RunOn::kCpu) gives {0, 0}.
// Read: docs/method-paper/fluidity-method.typ <sec-fixed-order-sums>
SumBuffersSize ComputeSumBuffersSize(RunOn run_on);

// Computes the sum of the first `count` values of a buffer, in the order of math::TreeSum. Allocates nothing.
//
// Method: on the CPU, math::TreeSum. On the GPU, one layer of the tree per GPU call, in the layer arrays of
//   `buffers`. The buffer of values is never changed.
// Order: same bits on every run and on both processors, unless the result is NaN, whose bits IEEE 754 leaves
//   open: both add the same pairs, and IEEE 754 rounds each addition exactly. The single value -0.0 gives -0.0;
//   padding adds +0.0, so {-0.0, -0.0, -0.0} gives +0.0.
// Example: a buffer {1e16, 1, -1e16, 1, 5}, count 4, sum buffers for 1,000 values: 0.0, since the fifth value is
//   not read. count 0 gives 0.0.
// Requires: values and buffers live on one processor, checked by assert in debug builds.
// Fails: kSizeMismatch    count is above values.size() or buffers.max_count()
//        kRunOnCallFailed a GPU call failed
//        On failure no sum is returned.
// Read: docs/method-paper/fluidity-method.typ <sec-fixed-order-sums>
std::expected<double, BufferError> ComputeTreeSum(const Buffer<double>& values, std::size_t count, SumBuffers& buffers);

// Computes the sum of first[i] * second[i] for i below `count`, in the order of math::TreeSum. Allocates nothing.
//
// Method: each product is rounded once, then the products are added as ComputeTreeSum adds values. Never fused
//   with the addition: the CPU library is built with -ffp-contract=off (/fp:precise on MSVC) and the GPU code with
//   --fmad=false. The GPU writes the products to `buffers` first, then sums them there. Neither buffer of values
//   is changed.
// Order: same bits on every run and on both processors, unless the result is NaN, whose bits IEEE 754 leaves
//   open, as for ComputeTreeSum.
// Example: first {1 + 2^-30, -1}, second {1 - 2^-30, 1}. The first product, 1 - 2^-60, rounds to 1, and 1 + -1
//   gives 0.0. A fused multiply-add would give -2^-60.
// Requires: first, second and buffers live on one processor, checked by assert in debug builds.
// Fails: kSizeMismatch    count is above the size of either buffer of values or buffers.max_count()
//        kRunOnCallFailed a GPU call failed
//        On failure no sum is returned.
// Read: docs/method-paper/fluidity-method.typ <sec-fixed-order-sums>
std::expected<double, BufferError>
ComputeTreeDot(const Buffer<double>& first, const Buffer<double>& second, std::size_t count, SumBuffers& buffers);

// Computes the largest magnitude (absolute value) among the first `count` values of a buffer. Allocates nothing.
//
// Method: math::LargerMagnitude over the values. Any order gives the same bits, since no comparison rounds; the GPU
//   uses the layers of the tree. A NaN anywhere gives NaN, on both processors; its bits are not promised. The
//   buffer of values is never changed.
// Example: {-7, 3, 5} gives 7. {1, NaN, 2} gives NaN. count 0 gives 0.0.
// Requires: values and buffers live on one processor, checked by assert in debug builds.
// Fails: kSizeMismatch    count is above values.size() or buffers.max_count()
//        kRunOnCallFailed a GPU call failed
//        On failure no value is returned.
// Read: docs/method-paper/fluidity-method.typ <sec-fixed-order-sums>
std::expected<double, BufferError>
ComputeLargestMagnitude(const Buffer<double>& values, std::size_t count, SumBuffers& buffers);

// The working buffers of ComputeTreeSum, ComputeTreeDot and ComputeLargestMagnitude on one processor, for up to
// max_count values, built once and passed to every call, so that no sum allocates. Their contents mean nothing
// between calls: each call overwrites what it uses. Move-only.
//
// Moved-from buffers have max_count() 0 and no memory, and keep their run_on(). The destructor frees the memory.
class SumBuffers {
 public:
  SumBuffers(const SumBuffers&) = delete;
  SumBuffers& operator=(const SumBuffers&) = delete;
  SumBuffers(SumBuffers&& other) noexcept
      : bytes_(std::move(other.bytes_)), max_count_(std::exchange(other.max_count_, 0)) {
  }
  SumBuffers& operator=(SumBuffers&& other) noexcept {
    if (this != &other) {
      bytes_ = std::move(other.bytes_);
      max_count_ = std::exchange(other.max_count_, 0);
    }
    return *this;
  }
  ~SumBuffers() = default;

  // The most values one call may use these buffers for.
  [[nodiscard]] std::size_t max_count() const {
    return max_count_;
  }

  // The processor the buffers live on.
  [[nodiscard]] RunOn run_on() const {
    return bytes_.run_on();
  }

  // Bytes held: ComputeSumBuffersSize(run_on()) applied to max_count(), or 0 after a move.
  [[nodiscard]] std::size_t byte_count() const {
    return bytes_.size();
  }

 private:
  friend std::expected<SumBuffers, BufferError> BuildSumBuffers(RunOn run_on, std::size_t max_count);
  friend std::expected<SumBuffers, BufferError>
  BuildSumBuffers(RunOn run_on, std::size_t max_count, RunMemory& run_memory);
  friend std::expected<double, BufferError>
  ComputeTreeSum(const Buffer<double>& values, std::size_t count, SumBuffers& buffers);
  friend std::expected<double, BufferError>
  ComputeTreeDot(const Buffer<double>& first, const Buffer<double>& second, std::size_t count, SumBuffers& buffers);
  friend std::expected<double, BufferError>
  ComputeLargestMagnitude(const Buffer<double>& values, std::size_t count, SumBuffers& buffers);

  // Takes ownership of `bytes`, split by base/ordered_sums_cpu.cc for `max_count` values.
  SumBuffers(Buffer<std::uint8_t> bytes, std::size_t max_count) : bytes_(std::move(bytes)), max_count_(max_count) {
  }

  Buffer<std::uint8_t> bytes_;
  std::size_t max_count_ = 0;
};

}  // namespace fluidity
