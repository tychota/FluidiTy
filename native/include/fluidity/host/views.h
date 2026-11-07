#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace fluidity {

// Returns the element count for two extents, including zero. Returns kOutOfRange on overflow.
inline absl::StatusOr<std::size_t> CheckedExtentProduct(std::size_t first_extent, std::size_t second_extent) {
  if (second_extent != 0 && first_extent > std::numeric_limits<std::size_t>::max() / second_extent) {
    return absl::OutOfRangeError("view extent product overflows size_t");
  }
  return first_extent * second_extent;
}

// Requires an exact element count. Returns kInvalidArgument for a mismatch.
inline absl::Status ValidateExtent(std::size_t actual, std::size_t expected) {
  if (actual != expected) {
    return absl::InvalidArgumentError("view extent does not match the required size");
  }
  return absl::OkStatus();
}

// Checks two valid byte views without writing to storage. Empty views and adjacent ranges
// are allowed. Returns kInvalidArgument if the ranges overlap. Use std::as_bytes to
// convert typed views, including a single-element span for an entire workspace.
inline absl::Status ValidateDisjoint(std::span<const std::byte> first_view, std::span<const std::byte> second_view) {
  if (first_view.empty() || second_view.empty()) {
    return absl::OkStatus();
  }
  // Compare byte addresses on the supported flat-address hosts. Subtract the earlier start
  // from the later one to avoid constructing a potentially overflowing end address.
  const auto first_start = reinterpret_cast<std::uintptr_t>(first_view.data());
  const auto second_start = reinterpret_cast<std::uintptr_t>(second_view.data());
  bool overlaps;
  if (first_start <= second_start) {
    overlaps = second_start - first_start < first_view.size();
  } else {
    overlaps = first_start - second_start < second_view.size();
  }
  if (overlaps) {
    return absl::InvalidArgumentError("views overlap");
  }
  return absl::OkStatus();
}

}  // namespace fluidity
