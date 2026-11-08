#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>

#include "absl/status/statusor.h"
#include "mac_pressure_3d_multigrid.h"

namespace fluidity::pressure_internal {
// Typed stack arena backed by one nothrow reservation. Marks reclaim scratch
// without allocating or freeing; live spans above a reclaimed mark expire.
class MacPressureScratch3D {
 public:
  explicit MacPressureScratch3D(std::span<std::max_align_t> storage) : storage_(std::as_writable_bytes(storage)) {
  }
  template <typename T>
  absl::StatusOr<std::span<T>> Take(std::size_t count) {
    static_assert(std::is_trivially_copyable_v<T> && std::is_nothrow_default_constructible_v<T>);
    static_assert(alignof(T) <= alignof(std::max_align_t));
    const std::size_t padding = (alignof(T) - used_ % alignof(T)) % alignof(T);
    if (padding > storage_.size() - used_) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    const std::size_t offset = used_ + padding;
    if (count > (storage_.size() - offset) / sizeof(T)) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    T* pointer = reinterpret_cast<T*>(storage_.data() + offset);
    for (std::size_t index = 0; index < count; ++index) {
      std::construct_at(pointer + index);
    }
    used_ = offset + count * sizeof(T);
    high_water_ = std::max(high_water_, used_);
    return std::span<T>(pointer, count);
  }
  std::size_t used() const {
    return used_;
  }
  std::size_t high_water() const {
    return high_water_;
  }

 private:
  friend class MacPressureScratchMark3D;
  void Rewind(std::size_t mark) {
    used_ = mark;
  }
  std::span<std::byte> storage_;
  std::size_t used_ = 0;
  std::size_t high_water_ = 0;
};
class MacPressureScratchMark3D {
 public:
  explicit MacPressureScratchMark3D(MacPressureScratch3D& scratch) : scratch_(scratch), mark_(scratch.used()) {
  }
  MacPressureScratchMark3D(const MacPressureScratchMark3D&) = delete;
  MacPressureScratchMark3D& operator=(const MacPressureScratchMark3D&) = delete;
  ~MacPressureScratchMark3D() {
    scratch_.Rewind(mark_);
  }

 private:
  MacPressureScratch3D& scratch_;
  std::size_t mark_;
};
// Prepared coefficients and hierarchy borrow the scratch arena until its next
// reuse. These flat spans can be uploaded without reconstructing owning vectors.
struct PreparedMacPressureView3D {
  std::span<const PreparedMacPressureCell3D> cell_records;
  std::span<const FrozenMacPressureLink3D> link_records;
  std::span<double> rhs;
  std::span<double> rhs_absolute_terms;
  std::span<std::size_t> components;
  std::span<std::size_t> gauges;
  std::span<const PreparedMacPressureCell3D> cells() const {
    return cell_records;
  }
  std::span<const FrozenMacPressureLink3D> links() const {
    return link_records;
  }
  std::span<const double> integrated_rhs() const {
    return rhs;
  }
  std::span<const double> rhs_absolute_term_sums() const {
    return rhs_absolute_terms;
  }
  std::span<const std::size_t> component_ordinals() const {
    return components;
  }
  std::span<const std::size_t> gauge_cell_ordinals() const {
    return gauges;
  }
};
// Same units/order as MacPressureLevel3D; terminal blocks use flat ranges.
struct MacPressureLevelView3D {
  std::span<CellIndex3D> coordinates;
  std::span<std::size_t> component_ordinals;
  std::span<double> anchor_diagonal;
  std::span<double> diagonal;
  std::span<MacPressureEdge3D> edges;
  std::span<std::size_t> row_offsets;
  std::span<MacPressureNeighbor3D> neighbors;
  std::span<std::size_t> aggregate_ordinals;
  std::span<std::size_t> terminal_offsets;  // Offsets into terminal_unknowns, including exclusive end.
  std::span<std::size_t> terminal_unknowns;
  std::span<double> terminal_factors;  // Concatenated row-major block squares, zero above diagonal.
};
inline constexpr std::size_t kMaximumPressureLevels3D = 33;
struct MacPressureHierarchyView3D {
  std::span<std::size_t> unknown_to_cell;
  std::span<std::size_t> cell_to_unknown;
  std::span<MacPressureLevelView3D> levels;
};
// Builds the same ordered Galerkin graph as the owning setup builder. All
// allocations are checked arena spans. Failure leaves scratch unpublished.
// Precondition: system is a successfully prepared original pressure system with
// valid endpoint ordinals, sorted gauges and admitted cell/face bounds. Returned
// hierarchy spans expire on arena rewind/reuse or backing-owner destruction.
absl::StatusOr<MacPressureHierarchyView3D>
BuildBoundedMacPressureHierarchy3D(const PreparedMacPressureView3D& system, MacPressureScratch3D& scratch);
}  // namespace fluidity::pressure_internal
