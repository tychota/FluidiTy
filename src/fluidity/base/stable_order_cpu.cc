#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <utility>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/base/stable_order_digit.h"

#if defined(FLUIDITY_HAS_GPU)
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/stable_order_gpu.h"
#endif

namespace fluidity {
namespace {

// The largest item count: an order holds 32-bit indexes, so an item index must fit in 32 bits.
constexpr std::size_t kMaxItemCount = std::numeric_limits<std::uint32_t>::max();

// The passes of a 64-bit key, one per byte. A 32-bit destination takes the first four.
constexpr std::uint32_t kKeyPassCount = sizeof(std::uint64_t);

// The CPU order buffers: the digit counts of every pass first, then the second sequence. BuildOrderBuffers and
// ComputeOrderBuffersSize both read these two constants, so the size reported is the size allocated.
constexpr std::uint32_t kCpuBytesPerItem = sizeof(std::uint32_t);
constexpr std::uint64_t kCpuFixedBytes = std::uint64_t{kKeyPassCount} * kDigitValueCount * sizeof(std::uint32_t);

// The GPU order buffers: second sequence (4 bytes per item), chunk counts (kDigitValueCount entries of 4 bytes for each
// chunk of kItemsPerChunk items: 1 byte per item, plus up to one chunk of rounding), the chunk totals of a prefix
// sum (8 bytes for each chunk of 1,024 values in every layer: below 1/127 byte per item, plus a few entries), and
// the destination flag, each area starting at a multiple of kGpuAreaAlignment bytes. That is at most about
// 5.01 bytes per item plus 1,900 bytes; 6 and 4,096 cover it, and the static_asserts below check chosen counts.
constexpr std::uint32_t kGpuBytesPerItem = 6;
constexpr std::uint64_t kGpuFixedBytes = 4096;
// Bytes. Every area starts at a multiple of this from the start of the bytes of the order buffers, which Buffer
// starts at a multiple of kBufferAlignmentBytes, so every area is aligned to kBufferAlignmentBytes, more than the 8
// bytes a chunk total needs.
constexpr std::size_t kGpuAreaAlignment = kBufferAlignmentBytes;
static_assert(kGpuAreaAlignment % alignof(std::uint64_t) == 0 && kGpuAreaAlignment % alignof(std::uint32_t) == 0);

// `byte_count` rounded up to a multiple of `alignment`.
constexpr std::size_t AlignUp(std::size_t byte_count, std::size_t alignment) {
  return (byte_count + alignment - 1) / alignment * alignment;
}

// Where each GPU area starts in the bytes of the order buffers, and where the last one ends.
struct GpuAreaOffsets {
  std::size_t second_sequence = 0;
  std::size_t chunk_counts = 0;
  std::size_t layer_totals = 0;
  std::size_t destination_flag = 0;
  std::size_t end = 0;
};

// The GPU areas for `max_count` items, as GpuOrderBufferAreas (base/stable_order_gpu.h) describes them.
constexpr GpuAreaOffsets ComputeGpuAreaOffsets(std::size_t max_count) {
  const std::size_t chunk_count_entries = std::size_t{kDigitValueCount} * ComputeChunkCount(max_count);
  // The sort sums the chunk counts; ComputeExclusivePrefixSum sums at most max_count values.
  const std::size_t largest_prefix_sum = std::max(max_count, chunk_count_entries);
  GpuAreaOffsets offsets;
  offsets.second_sequence = 0;
  offsets.chunk_counts = AlignUp(sizeof(std::uint32_t) * max_count, kGpuAreaAlignment);
  offsets.layer_totals = AlignUp(offsets.chunk_counts + sizeof(std::uint32_t) * chunk_count_entries, kGpuAreaAlignment);
  offsets.destination_flag = AlignUp(
      offsets.layer_totals + sizeof(std::uint64_t) * ComputePrefixSumLayerEntryCount(largest_prefix_sum),
      kGpuAreaAlignment
  );
  offsets.end = offsets.destination_flag + sizeof(std::uint32_t);
  return offsets;
}

// Whether the GPU areas for `max_count` items fit in the bytes BuildOrderBuffers allocates for them.
constexpr bool GpuAreasFit(std::size_t max_count) {
  return ComputeGpuAreaOffsets(max_count).end <= kGpuBytesPerItem * max_count + kGpuFixedBytes;
}

static_assert(
    GpuAreasFit(0) && GpuAreasFit(1) && GpuAreasFit(255) && GpuAreasFit(1023) && GpuAreasFit(1024) && GpuAreasFit(1025)
    && GpuAreasFit(std::size_t{1} << 20U) && GpuAreasFit(1000003) && GpuAreasFit(kMaxItemCount)
);

// The CPU areas of an OrderBuffers: the digit counts of all passes, then the second sequence.
struct CpuOrderBufferAreas {
  std::uint32_t* digit_counts = nullptr;  // kKeyPassCount * kDigitValueCount entries, pass by pass.
  std::uint32_t* second_sequence = nullptr;  // max_count entries.
};

CpuOrderBufferAreas ToCpuAreas(std::uint8_t* bytes) {
  // The bytes come from Buffer: CPU memory from ::operator new, which creates the 32-bit values implicitly
  // (implicit object creation), starting at a multiple of kBufferAlignmentBytes (64). kCpuFixedBytes is a multiple
  // of 4, so both areas are aligned for 32-bit values. std::start_lifetime_as_array would say this in code, but
  // libc++ 19 does not provide it.
  static_assert(kCpuFixedBytes % alignof(std::uint32_t) == 0);
  return {
      .digit_counts = reinterpret_cast<std::uint32_t*>(bytes),
      .second_sequence = reinterpret_cast<std::uint32_t*>(bytes + kCpuFixedBytes),
  };
}

// Writes to order[0..count) the indexes of keys[0..count) by ascending key, ties in the order of previous_order
// (index order when it is null). count is 1 or more; order is neither previous_order nor an area of the order buffers.
// Follows Procedure <proc-stable-order>; Key is std::uint64_t for keys and std::uint32_t for destinations.
template <typename Key>
void BuildCpuStableOrder(
    const Key* keys,
    std::size_t count,
    const std::uint32_t* previous_order,
    std::uint32_t* order,
    CpuOrderBufferAreas areas
) {
  constexpr std::uint32_t kPassCount = sizeof(Key);

  // Step 1: the starting sequence is previous_order; null stands for the indexes in ascending order.
  const std::uint32_t* sequence = previous_order;

  // Step 2, for every pass at once: a count does not depend on the order of the sequence, so one read of the keys
  // counts the digits of all passes.
  std::uint32_t* const digit_counts = areas.digit_counts;
  std::fill(digit_counts, digit_counts + std::size_t{kPassCount} * kDigitValueCount, 0U);
  for (std::size_t item = 0; item < count; ++item) {
    for (std::uint32_t pass = 0; pass < kPassCount; ++pass) {
      ++digit_counts[(std::size_t{pass} * kDigitValueCount) + ComputeKeyDigit(keys[item], pass)];
    }
  }

  // Step 5 (decided here, before the passes, because the counts of every byte are already known): a pass in which
  // one digit value holds every item would leave every item where it is, so it is skipped. Skipping does not change
  // the result.
  std::array<std::uint32_t, kKeyPassCount> passes_that_move{};
  std::uint32_t moving_pass_count = 0;
  for (std::uint32_t pass = 0; pass < kPassCount; ++pass) {
    const std::uint32_t* const pass_counts = digit_counts + (std::size_t{pass} * kDigitValueCount);
    const bool one_digit_holds_every_item =
        std::any_of(pass_counts, pass_counts + kDigitValueCount, [count](std::uint32_t digit_count) {
          return digit_count == count;
        });
    if (!one_digit_holds_every_item) {
      passes_that_move[moving_pass_count] = pass;
      ++moving_pass_count;
    }
  }

  for (std::uint32_t moving_pass = 0; moving_pass < moving_pass_count; ++moving_pass) {
    const std::uint32_t pass = passes_that_move[moving_pass];
    // The passes alternate between `order` and the second sequence, chosen so that the last pass writes `order`.
    const bool writes_order = (moving_pass_count - 1 - moving_pass) % 2 == 0;
    std::uint32_t* const next_sequence = writes_order ? order : areas.second_sequence;

    // Step 3: the exclusive prefix sum of the counts gives the first place of each digit value.
    std::uint32_t* const next_place = digit_counts + (std::size_t{pass} * kDigitValueCount);
    std::uint32_t running_count = 0;
    for (std::uint32_t digit = 0; digit < kDigitValueCount; ++digit) {
      const std::uint32_t digit_count = next_place[digit];
      next_place[digit] = running_count;
      running_count += digit_count;
    }

    // Step 4: front to back, each item goes to the next free place of its digit, so an earlier item with the same
    // digit always lands first.
    for (std::size_t position = 0; position < count; ++position) {
      const std::uint32_t item = sequence != nullptr ? sequence[position] : static_cast<std::uint32_t>(position);
      const std::uint32_t digit = ComputeKeyDigit(keys[item], pass);
      next_sequence[next_place[digit]] = item;
      ++next_place[digit];
    }
    sequence = next_sequence;
  }

  // Step 6: the last moving pass wrote `order`. With no moving pass every key is equal, and the order is the
  // starting sequence.
  if (moving_pass_count == 0) {
    for (std::size_t position = 0; position < count; ++position) {
      order[position] = previous_order != nullptr ? previous_order[position] : static_cast<std::uint32_t>(position);
    }
  }
}

#if defined(FLUIDITY_HAS_GPU)

GpuOrderBufferAreas ToGpuAreas(std::uint8_t* bytes, std::size_t max_count) {
  const GpuAreaOffsets offsets = ComputeGpuAreaOffsets(max_count);
  // The bytes come from Buffer: GPU memory starting at a multiple of kBufferAlignmentBytes (64), and every
  // offset is a multiple of kGpuAreaAlignment, which equals it, so each area is aligned for its values.
  return {
      .second_sequence = reinterpret_cast<std::uint32_t*>(bytes + offsets.second_sequence),
      .chunk_counts = reinterpret_cast<std::uint32_t*>(bytes + offsets.chunk_counts),
      .layer_totals = reinterpret_cast<std::uint64_t*>(bytes + offsets.layer_totals),
      .destination_flag = reinterpret_cast<std::uint32_t*>(bytes + offsets.destination_flag),
  };
}

// The BufferError for how a GPU call ended.
std::expected<void, BufferError> ToBufferResult(RunOnCallStatus status) {
  if (status != RunOnCallStatus::kDone) {
    return std::unexpected(ToBufferError(status));
  }
  return {};
}

#endif

// The size checks of BuildStableOrder.
std::expected<void, BufferError> CheckOrderSizes(
    const Buffer<std::uint64_t>& keys,
    std::size_t count,
    const Buffer<std::uint32_t>* previous_order,
    const Buffer<std::uint32_t>& order,
    const OrderBuffers& buffers
) {
  if (count > kMaxItemCount) {
    // An index of the order would not fit in 32 bits.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  if (count > keys.size() || count > order.size()) {
    // The caller asked for more items than a buffer holds: a bug in the caller.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  if (previous_order != nullptr && count > previous_order->size()) {
    return std::unexpected(BufferError::kSizeMismatch);
  }
  if (count > buffers.max_count()) {
    // The order buffers were built for fewer items; the caller must build larger ones.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  if (previous_order == &order) {
    // The passes would overwrite the starting sequence while they still read it; the caller needs two buffers.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  return {};
}

// The entries of first_of_group: one start per destination and the end of the last group.
constexpr std::uint64_t ToGroupStartCount(std::uint32_t destination_count) {
  return std::uint64_t{destination_count} + 1;
}

// The size checks of BuildGroupsByDestination.
std::expected<void, BufferError> CheckGroupSizes(
    const Buffer<std::uint32_t>& destination,
    std::size_t item_count,
    const Buffer<std::uint32_t>& first_of_group,
    const Buffer<std::uint32_t>& item_order,
    const OrderBuffers& buffers,
    std::uint64_t group_start_count
) {
  if (item_count > kMaxItemCount) {
    // An index of item_order would not fit in 32 bits.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  if (item_count > destination.size() || item_count > item_order.size()) {
    // The caller asked for more items than a buffer holds: a bug in the caller.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  if (item_count > buffers.max_count()) {
    // The order buffers were built for fewer items; the caller must build larger ones.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  if (first_of_group.size() < group_start_count) {
    // first_of_group needs one entry per destination plus the end of the last group.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  const bool outputs_share_a_buffer = &first_of_group == &item_order;
  const bool destination_is_an_output = &destination == &first_of_group || &destination == &item_order;
  if (outputs_share_a_buffer || destination_is_an_output) {
    // The function would overwrite one array while it still reads or writes it; the caller needs three buffers.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  return {};
}

// The bytes both forms of BuildOrderBuffers take for `max_count` items. max_count is at most kMaxItemCount, below
// 2^32, so the product stays far below 2^64.
std::uint64_t ComputeOrderBuffersByteCount(RunOn run_on, std::size_t max_count) {
  const OrderBuffersSize size = ComputeOrderBuffersSize(run_on);
  return (std::uint64_t{size.bytes_per_item} * max_count) + size.fixed_bytes;
}

}  // namespace

OrderBuffersSize ComputeOrderBuffersSize(RunOn run_on) {
  if (run_on == RunOn::kGpu) {
    return {.bytes_per_item = kGpuBytesPerItem, .fixed_bytes = kGpuFixedBytes};
  }
  return {.bytes_per_item = kCpuBytesPerItem, .fixed_bytes = kCpuFixedBytes};
}

std::expected<OrderBuffers, BufferError> BuildOrderBuffers(RunOn run_on, std::size_t max_count) {
  if (max_count > kMaxItemCount) {
    // An index of the order would not fit in 32 bits.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  std::expected<Buffer<std::uint8_t>, BufferError> bytes =
      BuildBuffer<std::uint8_t>(run_on, ComputeOrderBuffersByteCount(run_on, max_count));
  if (!bytes.has_value()) {
    return std::unexpected(bytes.error());
  }
  return OrderBuffers(std::move(*bytes), max_count);
}

std::expected<OrderBuffers, BufferError> BuildOrderBuffers(RunOn run_on, std::size_t max_count, RunMemory& run_memory) {
  // Requires: one processor (base/stable_order.h).
  assert(run_on == run_memory.run_on());
  if (max_count > kMaxItemCount) {
    // An index of the order would not fit in 32 bits; checked first, so nothing is taken.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  std::expected<Buffer<std::uint8_t>, BufferError> bytes =
      BuildBuffer<std::uint8_t>(run_on, ComputeOrderBuffersByteCount(run_on, max_count), run_memory);
  if (!bytes.has_value()) {
    return std::unexpected(bytes.error());
  }
  return OrderBuffers(std::move(*bytes), max_count);
}

std::expected<void, BufferError> BuildStableOrder(
    const Buffer<std::uint64_t>& keys,
    std::size_t count,
    const Buffer<std::uint32_t>* previous_order,
    Buffer<std::uint32_t>& order,
    OrderBuffers& buffers
) {
  // Requires: one processor (base/stable_order.h). A bug in the caller, so checked in debug builds only.
  assert(order.run_on() == keys.run_on());
  assert(previous_order == nullptr || previous_order->run_on() == keys.run_on());
  assert(buffers.run_on() == keys.run_on());
  if (const std::expected<void, BufferError> checked = CheckOrderSizes(keys, count, previous_order, order, buffers);
      !checked.has_value()) {
    return checked;
  }
  if (count == 0) {
    return {};
  }
  const std::uint32_t* const previous_order_data = previous_order != nullptr ? previous_order->data() : nullptr;
  if (keys.run_on() == RunOn::kGpu) {
#if defined(FLUIDITY_HAS_GPU)
    return ToBufferResult(BuildGpuStableOrder(
        keys.data(),
        static_cast<std::uint32_t>(count),
        previous_order_data,
        order.data(),
        ToGpuAreas(buffers.bytes_.data(), buffers.max_count_)
    ));
#else
    // Without GPU code BuildBuffer never makes a GPU buffer, so no caller reaches this.
    return std::unexpected(BufferError::kRunOnCallFailed);
#endif
  }
  BuildCpuStableOrder(keys.data(), count, previous_order_data, order.data(), ToCpuAreas(buffers.bytes_.data()));
  return {};
}

std::expected<void, BufferError> ComputeExclusivePrefixSum(
    const Buffer<std::uint32_t>& values,
    std::size_t count,
    Buffer<std::uint32_t>& sums,
    OrderBuffers& buffers
) {
  // Requires: one processor (base/stable_order.h). A bug in the caller, so checked in debug builds only.
  assert(sums.run_on() == values.run_on());
  assert(buffers.run_on() == values.run_on());
  if (count > kMaxItemCount || count > buffers.max_count()) {
    // Too many values for 32-bit indexes, or for the order buffers the caller built.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  if (count > values.size() || count >= sums.size()) {
    // `sums` needs count + 1 entries, the last for the total: a bug in the caller.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  if (values.run_on() == RunOn::kGpu) {
#if defined(FLUIDITY_HAS_GPU)
    return ToBufferResult(ComputeGpuExclusivePrefixSum(
        values.data(),
        static_cast<std::uint32_t>(count),
        sums.data(),
        ToGpuAreas(buffers.bytes_.data(), buffers.max_count_)
    ));
#else
    // Without GPU code BuildBuffer never makes a GPU buffer, so no caller reaches this.
    return std::unexpected(BufferError::kRunOnCallFailed);
#endif
  }

  const std::uint32_t* const value_data = values.data();
  // The total first, in 64 bits, so an overflow is found before anything is written. count values below 2^32
  // each, with count below 2^32, cannot overflow 64 bits.
  std::uint64_t total = 0;
  for (std::size_t index = 0; index < count; ++index) {
    total += value_data[index];
  }
  if (total > std::numeric_limits<std::uint32_t>::max()) {
    // The sums would not fit in 32 bits; the caller's counts are larger than the arrays they describe can be.
    return std::unexpected(BufferError::kSizeMismatch);
  }
  // Each value is read before its sum is written, so `sums` may be `values`.
  std::uint32_t* const sum_data = sums.data();
  std::uint32_t running_sum = 0;
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint32_t value = value_data[index];
    sum_data[index] = running_sum;
    running_sum += value;
  }
  sum_data[count] = running_sum;
  return {};
}

std::expected<void, BufferError> BuildGroupsByDestination(
    const Buffer<std::uint32_t>& destination,
    std::size_t item_count,
    std::uint32_t destination_count,
    Buffer<std::uint32_t>& first_of_group,
    Buffer<std::uint32_t>& item_order,
    OrderBuffers& buffers
) {
  // Groups steps (Procedure <proc-groups-by-destination>): 1 check the sizes  2 the order buffers, given here
  //   3 check the destinations  4 order the items  5 find the group starts

  // Requires: one processor (base/stable_order.h). A bug in the caller, so checked in debug builds only.
  assert(first_of_group.run_on() == destination.run_on());
  assert(item_order.run_on() == destination.run_on());
  assert(buffers.run_on() == destination.run_on());

  // Groups step 1: the sizes.
  if (const std::expected<void, BufferError> checked = CheckGroupSizes(
          destination,
          item_count,
          first_of_group,
          item_order,
          buffers,
          ToGroupStartCount(destination_count)
      );
      !checked.has_value()) {
    return checked;
  }
  if (destination.run_on() == RunOn::kGpu) {
    // Groups steps 3 to 5 run in base/stable_order_gpu.cu.
#if defined(FLUIDITY_HAS_GPU)
    return ToBufferResult(BuildGpuGroupsByDestination(
        destination.data(),
        static_cast<std::uint32_t>(item_count),
        destination_count,
        first_of_group.data(),
        item_order.data(),
        ToGpuAreas(buffers.bytes_.data(), buffers.max_count_)
    ));
#else
    // Without GPU code BuildBuffer never makes a GPU buffer, so no caller reaches this.
    return std::unexpected(BufferError::kRunOnCallFailed);
#endif
  }

  // Groups step 3: every destination must name a group, before anything is written.
  const std::uint32_t* const destination_data = destination.data();
  const bool every_destination_is_in_range =
      std::all_of(destination_data, destination_data + item_count, [destination_count](std::uint32_t group) {
        return group < destination_count;
      });
  if (!every_destination_is_in_range) {
    // The caller computed a destination outside the groups it asked for.
    return std::unexpected(BufferError::kSizeMismatch);
  }

  // Groups step 4: one stable order with the destination as the key keeps each group in ascending item index.
  if (item_count > 0) {
    BuildCpuStableOrder(destination_data, item_count, nullptr, item_order.data(), ToCpuAreas(buffers.bytes_.data()));
  }

  // Groups step 5: first_of_group[d] is the number of items whose destination is below d: count each destination at
  // index destination + 1, then add up from the left.
  const std::uint64_t group_start_count = ToGroupStartCount(destination_count);
  std::uint32_t* const group_starts = first_of_group.data();
  std::fill(group_starts, group_starts + group_start_count, 0U);
  for (std::size_t item = 0; item < item_count; ++item) {
    ++group_starts[std::size_t{destination_data[item]} + 1];
  }
  for (std::uint64_t group = 1; group < group_start_count; ++group) {
    group_starts[group] += group_starts[group - 1];
  }
  return {};
}

}  // namespace fluidity
