#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
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

// Returns whether two valid byte views share storage. Empty views and adjacent ranges
// are disjoint. This check allocates nothing and leaves both views unchanged.
inline bool ByteViewsOverlap(std::span<const std::byte> first_view, std::span<const std::byte> second_view) {
  if (first_view.empty() || second_view.empty()) {
    return false;
  }
  // Subtract the earlier byte address from the later one so no end address can overflow.
  const auto first_start = reinterpret_cast<std::uintptr_t>(first_view.data());
  const auto second_start = reinterpret_cast<std::uintptr_t>(second_view.data());
  if (first_start <= second_start) {
    return second_start - first_start < first_view.size();
  }
  return first_start - second_start < second_view.size();
}

// Returns kInvalidArgument if the valid byte views share storage. Use std::as_bytes
// to convert typed views, including a single-element span for an entire workspace.
inline absl::Status ValidateDisjoint(std::span<const std::byte> first_view, std::span<const std::byte> second_view) {
  if (ByteViewsOverlap(first_view, second_view)) {
    return absl::InvalidArgumentError("views overlap");
  }
  return absl::OkStatus();
}

// Requires each extent to match the expected size. Returns kInvalidArgument for a mismatch.
inline absl::Status CheckExtents(std::size_t expected, std::initializer_list<std::size_t> extents) {
  for (const std::size_t extent : extents) {
    const auto status = ValidateExtent(extent, expected);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

using ByteView = std::span<const std::byte>;

// Read-only views can share storage. Returns kInvalidArgument for overlap between a
// writable view and any other view.
inline absl::Status
CheckWritableViews(std::initializer_list<ByteView> writable, std::initializer_list<ByteView> readable) {
  for (auto view = writable.begin(); view != writable.end(); ++view) {
    for (const ByteView input : readable) {
      const auto status = ValidateDisjoint(*view, input);
      if (!status.ok()) {
        return status;
      }
    }
    for (auto other = writable.begin(); other != view; ++other) {
      const auto status = ValidateDisjoint(*view, *other);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace fluidity
