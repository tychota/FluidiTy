// Ordering and grouping of items by an integer key, with ties kept in a fixed order, on either processor.
//
// Purpose: give the grid builder, the particle-to-grid transfer and the batches one way to put items in order and
//   to group them, with the same result on every run and on both processors.
//
// Idea: two particles in the same cell have the same key, and the order in which they are added decides the bits
//   of a face sum. So the order is stable: items with equal keys keep the order they came in, never the order in
//   which threads happened to finish. BuildStableOrder places items by one byte of the key at a time, lowest byte
//   first, each byte in one counting pass (papers call this a least-significant-digit radix sort). A counting pass
//   counts the items of each digit value, turns the counts into first places with an exclusive prefix sum, and
//   puts each item at the first place of its digit plus the number of earlier items with that digit. Only
//   integers take part, and each place is a formula of the keys, so the CPU and the GPU give the same bytes. No
//   library sort is used: none promises a stable result that is the same on every toolkit version.
//   Example: keys {5, 3, 5, 1} give the order {3, 1, 0, 2}; items 0 and 2 share key 5 and keep their index order.
//
// Words:
//   stable                items with equal keys keep the order they had before
//   counting pass         one placement of every item by one digit of its key
//   exclusive prefix sum  sums[i] is the sum of the values before index i
//   destination           the group an item belongs to, in BuildGroupsByDestination
// Owns:    OrderBuffers, BuildOrderBuffers, OrderBuffersSize, ComputeOrderBuffersSize, BuildStableOrder,
//          ComputeExclusivePrefixSum, BuildGroupsByDestination
// Uses:    Buffer and BufferError (base/buffer.h), RunMemory (base/run_memory.h), RunOn (base/run_on.h), CPU-only
//          guard (base/cpu_only.h)
// Implementation uses: the digit rule (base/stable_order_digit.h) and, in a build with GPU code, the private GPU
//          half (base/stable_order_gpu.h)
// Used by: the grid builder, the particle-to-grid transfer, the batches
// Read:    docs/method-paper/fluidity-method.typ <sec-stable-order>

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

class OrderBuffers;

// Builds the working buffers of BuildStableOrder, ComputeExclusivePrefixSum and BuildGroupsByDestination on one
// processor, for up to max_count items, so that none allocates.
//
// Method: allocates one buffer of ComputeOrderBuffersSize(run_on).bytes_per_item * max_count +
//   ComputeOrderBuffersSize(run_on).fixed_bytes bytes on `run_on` with BuildBuffer, and splits it into the arrays
//   each processor needs. max_count 0 still allocates the fixed bytes.
// Example: BuildOrderBuffers(RunOn::kCpu, 1000) holds 4 * 1000 + 8192 = 12,192 bytes.
// Fails: kSizeMismatch    max_count is above 2^32 - 1: orders hold 32-bit indexes
//        kOutOfMemory     the processor could not provide the memory
//        kRunOnCallFailed RunOn::kGpu was asked for and the GPU cannot run (also in a build without GPU code)
//        On failure nothing stays allocated and no buffers are returned.
// Read: docs/method-paper/fluidity-method.typ <sec-stable-order>
std::expected<OrderBuffers, BufferError> BuildOrderBuffers(RunOn run_on, std::size_t max_count);

// Builds the same working buffers as one piece of `run_memory` instead of memory of their own, so that the run's one
// reserved block holds them (base/run_memory.h). Tests and tools use the form above.
//
// Method: checks max_count, then one BuildBuffer from `run_memory` of exactly the bytes the form above allocates, so
//   the block hands out ComputeRunMemoryPieceBytes(bytes_per_item * max_count + fixed_bytes) for it. Orders, prefix
//   sums and groups are the same bytes with either form.
// Example: BuildOrderBuffers(RunOn::kCpu, 1000, run_memory) holds 12,192 bytes and takes a piece of 12,288.
// Requires: run_on == run_memory.run_on(), checked by assert in debug builds.
// Fails: kSizeMismatch       max_count is above 2^32 - 1: orders hold 32-bit indexes
//        kRunMemoryTooSmall  the piece does not fit in run_memory.unused_bytes()
//        On failure nothing is taken and no buffers are returned.
// Read: docs/method-paper/fluidity-method.typ <sec-run-memory>
std::expected<OrderBuffers, BufferError> BuildOrderBuffers(RunOn run_on, std::size_t max_count, RunMemory& run_memory);

// order[i] becomes the index of the item at place i when items are sorted by ascending key. Allocates nothing.
//
// Method: Procedure <proc-stable-order> in `buffers`. The CPU counts the digits of all eight bytes in one read of
//   the keys and skips a byte that is the same for every item, since its pass would move nothing; the GPU runs all
//   eight passes. Skipping does not change the result, so both give the same bytes.
// Order: items with equal keys keep the order they have in *previous_order (index order when it is null). Two passes
//   sort by two keys: first by the minor key, then by the major key with the first order as previous_order.
// Example: keys {5, 3, 5, 1}, count 4, no previous_order: order becomes {3, 1, 0, 2}. With previous_order
//   {2, 1, 0, 3}: order becomes {3, 1, 2, 0}, since item 2 comes before item 0 in previous_order.
// Requires: keys, order, *previous_order and buffers live on one processor, checked by assert in debug builds.
//   *previous_order is a permutation of [0, count), not checked.
// Fails: kSizeMismatch    count is above 2^32 - 1, above keys.size(), previous_order->size(), order.size() or
//                         buffers.max_count(); or previous_order points to `order` itself, which this function
//                         would overwrite while it still reads it
//        kRunOnCallFailed a GPU call failed
//        A failed size check writes nothing. After kRunOnCallFailed the values of `order` are unspecified. count 0
//        succeeds and writes nothing.
// Read: docs/method-paper/fluidity-method.typ <proc-stable-order>
std::expected<void, BufferError> BuildStableOrder(
    const Buffer<std::uint64_t>& keys,
    std::size_t count,
    const Buffer<std::uint32_t>* previous_order,
    Buffer<std::uint32_t>& order,
    OrderBuffers& buffers
);

// sums[i] = values[0] + ... + values[i - 1] for i in [0, count]; sums[count] is the total. Exact unsigned
// arithmetic; a total above 2^32 - 1 is kSizeMismatch. sums.size() >= count + 1. `sums` may be the same buffer
// as `values` (in place). Allocates nothing. Same bytes on CPU and GPU.
//
// Method: the total is computed first in 64-bit arithmetic and checked, so an overflow writes nothing. The CPU then
//   writes the running sum from left to right. The GPU adds chunk totals layer by layer in `buffers`, then writes
//   each chunk's sums from its offset; a chunk reads all its values before it writes, so in place is safe.
// Example: values {3, 0, 2, 5, 1}, count 5: sums becomes {0, 3, 3, 5, 10, 11}. count 0 writes sums[0] = 0.
// Requires: values, sums and buffers live on one processor, checked by assert in debug builds.
// Fails: kSizeMismatch    count is above 2^32 - 1, above values.size() or buffers.max_count(), or sums.size() is
//                         below count + 1; or the total is above 2^32 - 1
//        kRunOnCallFailed a GPU call failed
//        Checks run in that order: sizes, then values (the total). A failed check writes nothing. After
//        kRunOnCallFailed the values of `sums` (and of `values`, when in place) are unspecified.
// Read: docs/method-paper/fluidity-method.typ <sec-stable-order>
std::expected<void, BufferError> ComputeExclusivePrefixSum(
    const Buffer<std::uint32_t>& values,
    std::size_t count,
    Buffer<std::uint32_t>& sums,
    OrderBuffers& buffers
);

// Groups items by destination: item_order lists the items of destination 0, then 1, ..., each group in
// ascending item index; first_of_group has destination_count + 1 entries. Allocates nothing.
//
// Method: Procedure <proc-groups-by-destination> in `buffers`: one stable order with the destination as the key,
//   whose passes only cover the bytes of destination_count - 1 on the GPU. first_of_group[d] is the number of items
//   whose destination is below d, so first_of_group[destination_count] is item_count and an empty group starts
//   where the next one does.
// Example: destination {2, 0, 2, 1, 0}, destination_count 4, order buffers for 5 or more items: item_order becomes
//   {1, 4, 3, 0, 2} and first_of_group {0, 2, 3, 5, 5}; destination 3 is empty.
// Requires: destination, first_of_group, item_order and buffers live on one processor, checked by assert in debug
//   builds.
// Fails: kSizeMismatch    item_count is above 2^32 - 1, above destination.size(), item_order.size() or
//                         buffers.max_count(), or first_of_group.size() is below destination_count + 1; or two of
//                         the three buffers of values are the same buffer, which the call would overwrite while it
//                         still reads it; or a destination among the first item_count is destination_count or more
//        kRunOnCallFailed a GPU call failed
//        Checks run in that order: sizes, then values (the destinations). A failed check writes nothing. After
//        kRunOnCallFailed the values of both outputs are unspecified.
// Read: docs/method-paper/fluidity-method.typ <proc-groups-by-destination>
std::expected<void, BufferError> BuildGroupsByDestination(
    const Buffer<std::uint32_t>& destination,
    std::size_t item_count,
    std::uint32_t destination_count,
    Buffer<std::uint32_t>& first_of_group,
    Buffer<std::uint32_t>& item_order,
    OrderBuffers& buffers
);

// Exact bytes BuildOrderBuffers allocates: bytes_per_item * max_count + fixed_bytes.
struct OrderBuffersSize {
  std::uint32_t bytes_per_item = 0;  // Bytes for each item of max_count.
  std::uint64_t fixed_bytes = 0;  // Bytes that do not depend on max_count.
};

// The bytes BuildOrderBuffers allocates on `run_on`, for the memory plan of a module.
//
// Method: the same constants BuildOrderBuffers allocates from. The CPU holds a second sequence (4 bytes per item)
//   and the digit counts of eight passes (8,192 bytes). The GPU holds a second sequence, a table of chunk counts and
//   the chunk totals of its prefix sums: 6 bytes per item and 4,096 bytes, which cover the rounding of chunk counts.
//   In a memory estimate the buffers are one entry, so the estimate equals the one piece they take: counted_per
//   kGivenCount, bytes_each 1, given_count bytes_per_item * max_count + fixed_bytes at the plan's max_count. A
//   per-item entry plus a fixed entry would round twice and reserve too much; a per-item entry alone too little.
// Example: ComputeOrderBuffersSize(RunOn::kGpu) gives {6, 4096}; for 1,000,000 items that is 6,004,096 bytes.
// Read: docs/method-paper/fluidity-method.typ <sec-stable-order>
OrderBuffersSize ComputeOrderBuffersSize(RunOn run_on);

// The working buffers of BuildStableOrder, ComputeExclusivePrefixSum and BuildGroupsByDestination on one processor,
// for up to max_count items, built once and passed to every call, so that none allocates. Their contents mean nothing
// between calls: each call overwrites what it uses. Move-only.
//
// Moved-from buffers have max_count() 0 and no memory, and keep their run_on(). The destructor frees the memory.
class OrderBuffers {
 public:
  OrderBuffers(const OrderBuffers&) = delete;
  OrderBuffers& operator=(const OrderBuffers&) = delete;
  OrderBuffers(OrderBuffers&& other) noexcept
      : bytes_(std::move(other.bytes_)), max_count_(std::exchange(other.max_count_, 0)) {
  }
  OrderBuffers& operator=(OrderBuffers&& other) noexcept {
    if (this != &other) {
      bytes_ = std::move(other.bytes_);
      max_count_ = std::exchange(other.max_count_, 0);
    }
    return *this;
  }
  ~OrderBuffers() = default;

  // The most items one call may use these buffers for.
  [[nodiscard]] std::size_t max_count() const {
    return max_count_;
  }

  // The processor the buffers live on.
  [[nodiscard]] RunOn run_on() const {
    return bytes_.run_on();
  }

  // Bytes held: ComputeOrderBuffersSize(run_on()) applied to max_count(), or 0 after a move.
  [[nodiscard]] std::size_t byte_count() const {
    return bytes_.size();
  }

 private:
  friend std::expected<OrderBuffers, BufferError> BuildOrderBuffers(RunOn run_on, std::size_t max_count);
  friend std::expected<OrderBuffers, BufferError>
  BuildOrderBuffers(RunOn run_on, std::size_t max_count, RunMemory& run_memory);
  friend std::expected<void, BufferError> BuildStableOrder(
      const Buffer<std::uint64_t>& keys,
      std::size_t count,
      const Buffer<std::uint32_t>* previous_order,
      Buffer<std::uint32_t>& order,
      OrderBuffers& buffers
  );
  friend std::expected<void, BufferError> ComputeExclusivePrefixSum(
      const Buffer<std::uint32_t>& values,
      std::size_t count,
      Buffer<std::uint32_t>& sums,
      OrderBuffers& buffers
  );
  friend std::expected<void, BufferError> BuildGroupsByDestination(
      const Buffer<std::uint32_t>& destination,
      std::size_t item_count,
      std::uint32_t destination_count,
      Buffer<std::uint32_t>& first_of_group,
      Buffer<std::uint32_t>& item_order,
      OrderBuffers& buffers
  );

  // Takes ownership of `bytes`, split by base/stable_order_cpu.cc for `max_count` items.
  OrderBuffers(Buffer<std::uint8_t> bytes, std::size_t max_count) : bytes_(std::move(bytes)), max_count_(max_count) {
  }

  Buffer<std::uint8_t> bytes_;
  std::size_t max_count_ = 0;
};

}  // namespace fluidity
