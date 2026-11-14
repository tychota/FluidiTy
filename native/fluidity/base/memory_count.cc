#include "fluidity/base/memory_count.h"

#include <cstdint>
#include <limits>
#include <span>

namespace fluidity {
namespace {

constexpr std::uint64_t kLargestBytes = std::numeric_limits<std::uint64_t>::max();

// The product of two counts, or the largest 64-bit number when it does not fit. Plain division keeps this portable
// to compilers without an overflow builtin.
std::uint64_t MultiplySaturating(std::uint64_t left, std::uint64_t right) {
  if (left != 0 && right > kLargestBytes / left) {
    return kLargestBytes;
  }
  return left * right;
}

// The sum of two counts, or the largest 64-bit number when it does not fit.
std::uint64_t AddSaturating(std::uint64_t left, std::uint64_t right) {
  if (right > kLargestBytes - left) {
    return kLargestBytes;
  }
  return left + right;
}

// The number of items of `entry` in this scene. A counted_per outside the enum gives zero.
std::uint64_t CountItems(const MemoryEstimateEntry& entry, const MemoryCounts& counts) {
  switch (entry.counted_per) {
    case CountedPer::kCellBlock:
      return counts.cell_blocks;
    case CountedPer::kCellPlace:
      return counts.cell_places;
    case CountedPer::kFacePlace:
      return counts.face_places;
    case CountedPer::kCornerPlace:
      return counts.corner_places;
    case CountedPer::kParticle:
      return counts.particles;
    case CountedPer::kGivenCount:
      return entry.given_count;
  }
  return 0;
}

// The bytes of one entry: items x bytes_each x copies, saturating.
std::uint64_t ComputeEntryBytes(const MemoryEstimateEntry& entry, const MemoryCounts& counts) {
  const std::uint64_t item_count = CountItems(entry, counts);
  const std::uint64_t bytes_of_one_copy = MultiplySaturating(item_count, entry.bytes_each);
  return MultiplySaturating(bytes_of_one_copy, entry.copies);
}

}  // namespace

std::uint64_t ComputeMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts) {
  std::uint64_t bytes_sum = 0;
  for (const MemoryEstimateEntry& entry : estimate) {
    bytes_sum = AddSaturating(bytes_sum, ComputeEntryBytes(entry, counts));
  }
  return bytes_sum;
}

std::uint64_t
ComputeMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts, MemoryWhere where) {
  std::uint64_t bytes_sum = 0;
  for (const MemoryEstimateEntry& entry : estimate) {
    if (entry.where != where) {
      continue;
    }
    bytes_sum = AddSaturating(bytes_sum, ComputeEntryBytes(entry, counts));
  }
  return bytes_sum;
}

}  // namespace fluidity
