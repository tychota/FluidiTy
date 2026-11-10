#pragma once

#include <cstddef>
#include <cstdint>

#include "fluidity/base/host_device.h"
#include "liquid_boundary_transfer_3d_bridge.h"

namespace fluidity::cuda_internal {

struct DonorFaceRow3D {
  std::size_t first_face_slot;
  std::size_t past_last_face_slot;
};

// Enumerates rows of candidate slots within two cell widths on each axis.
// Staged anchors must use canonical sparse MAC face order: owners (z,y,x),
// components (X,Y,Z), then local coordinates (z,y,x) in each 4x4x4 block.
// The target is an owned staged anchor and component_index is in [0,3).
// Rows preserve increasing face slots. Callers still apply the spherical
// radius and stage-specific history predicates.
class DonorFaceRows3D {
 public:
  FLUIDITY_HD DonorFaceRows3D(
      boundary_internal::numeric::span<const BoundaryTransferFace3D> anchors,
      Vec3 target_anchor_in_cell_widths,
      std::size_t component_index
  )
      : anchors_(anchors),
        component_face_slot_offset_(component_index * kFacesPerComponent),
        x_(LatticeBounds(target_anchor_in_cell_widths.x, component_index == 0)),
        y_(LatticeBounds(target_anchor_in_cell_widths.y, component_index == 1)),
        z_(LatticeBounds(target_anchor_in_cell_widths.z, component_index == 2)),
        owner_component_first_face_slot_(anchors.size()) {
  }

  // On exhaustion, returns false and leaves row unchanged.
  FLUIDITY_HD bool Next(DonorFaceRow3D& row) {
    if (owner_component_first_face_slot_ == anchors_.size() && !AdvanceOwner()) {
      return false;
    }
    const std::size_t row_first_face_slot =
        owner_component_first_face_slot_ + (local_z_index_ * kBlockWidth + local_y_index_) * kBlockWidth;
    row = {
        .first_face_slot = row_first_face_slot + local_x_.first,
        .past_last_face_slot = row_first_face_slot + local_x_.last + 1,
    };
    if (++local_y_index_ > local_y_.last) {
      local_y_index_ = local_y_.first;
      if (++local_z_index_ > local_z_.last) {
        owner_component_first_face_slot_ = anchors_.size();
      }
    }
    return true;
  }

 private:
  static constexpr int kBlockWidthShift = 2;
  static constexpr std::int32_t kBlockWidth = 1 << kBlockWidthShift;
  static constexpr std::size_t kFacesPerComponent = kBlockWidth * kBlockWidth * kBlockWidth;
  static constexpr std::size_t kFacesPerBlock = 3 * kFacesPerComponent;
  static constexpr std::int64_t kRadiusInHalfCells = 4;

  struct InclusiveRange {
    std::int32_t first;
    std::int32_t last;
  };

  FLUIDITY_HD static InclusiveRange LatticeBounds(double target_coordinate_in_cell_widths, bool donor_normal_axis) {
    using Limits = boundary_internal::numeric::numeric_limits<std::int32_t>;
    const auto target_in_half_cells = static_cast<std::int64_t>(2.0 * target_coordinate_in_cell_widths);
    const std::int64_t donor_shift_in_half_cells = donor_normal_axis ? 0 : 1;
    const std::int64_t center_in_half_cells = target_in_half_cells - donor_shift_in_half_cells;
    // Anchor coordinates are exact half integers. C++20 signed right shift
    // rounds down, so adding one to the lower numerator rounds that bound up.
    const std::int64_t first = (center_in_half_cells - kRadiusInHalfCells + 1) >> 1;
    const std::int64_t last = (center_in_half_cells + kRadiusInHalfCells) >> 1;
    return {
        .first = first < Limits::min() ? Limits::min() : static_cast<std::int32_t>(first),
        .last = last > Limits::max() ? Limits::max() : static_cast<std::int32_t>(last),
    };
  }

  FLUIDITY_HD static InclusiveRange LocalBounds(InclusiveRange lattice, std::int32_t block_index) {
    const std::int32_t block_first = block_index * kBlockWidth;
    const std::int32_t first = lattice.first - block_first;
    const std::int32_t last = lattice.last - block_first;
    return {
        .first = first < 0 ? 0 : first,
        .last = last >= kBlockWidth ? kBlockWidth - 1 : last,
    };
  }

  FLUIDITY_HD static bool IsBefore(Vec3 left, Vec3 right) {
    if (left.z != right.z) {
      return left.z < right.z;
    }
    if (left.y != right.y) {
      return left.y < right.y;
    }
    return left.x < right.x;
  }

  FLUIDITY_HD std::size_t FindOwnerFirstFaceSlot(BlockIndex3D block) const {
    // Each owner's first face is its low X face at local coordinate (0,0,0).
    const Vec3 owner_anchor{
        .x = static_cast<double>(block.x * kBlockWidth),
        .y = block.y * kBlockWidth + 0.5,
        .z = block.z * kBlockWidth + 0.5,
    };
    std::size_t first = 0;
    std::size_t last = anchors_.size() / kFacesPerBlock;
    while (first < last) {
      const std::size_t middle = first + (last - first) / 2;
      if (IsBefore(anchors_[middle * kFacesPerBlock].anchor_in_cell_widths, owner_anchor)) {
        first = middle + 1;
      } else {
        last = middle;
      }
    }
    const std::size_t owner_first_face_slot = first * kFacesPerBlock;
    if (owner_first_face_slot == anchors_.size()) {
      return anchors_.size();
    }
    const Vec3 found_anchor = anchors_[owner_first_face_slot].anchor_in_cell_widths;
    if (found_anchor.x != owner_anchor.x || found_anchor.y != owner_anchor.y || found_anchor.z != owner_anchor.z) {
      return anchors_.size();
    }
    return owner_first_face_slot;
  }

  FLUIDITY_HD bool AdvanceOwner() {
    // At most five lattice indices on an axis intersect at most two owners.
    // Iterate the eight possible high/low combinations in (z,y,x) order.
    while (next_owner_offset_ < 8) {
      const int owner_offset = next_owner_offset_++;
      const BlockIndex3D block{
          .x = (x_.first >> kBlockWidthShift) + (owner_offset & 1),
          .y = (y_.first >> kBlockWidthShift) + ((owner_offset >> 1) & 1),
          .z = (z_.first >> kBlockWidthShift) + (owner_offset >> 2),
      };
      if (block.x > (x_.last >> kBlockWidthShift) || block.y > (y_.last >> kBlockWidthShift)
          || block.z > (z_.last >> kBlockWidthShift)) {
        continue;
      }
      const std::size_t owner_first_face_slot = FindOwnerFirstFaceSlot(block);
      if (owner_first_face_slot == anchors_.size()) {
        continue;
      }
      owner_component_first_face_slot_ = owner_first_face_slot + component_face_slot_offset_;
      local_x_ = LocalBounds(x_, block.x);
      local_y_ = LocalBounds(y_, block.y);
      local_z_ = LocalBounds(z_, block.z);
      local_y_index_ = local_y_.first;
      local_z_index_ = local_z_.first;
      return true;
    }
    return false;
  }

  boundary_internal::numeric::span<const BoundaryTransferFace3D> anchors_;
  std::size_t component_face_slot_offset_;
  InclusiveRange x_;
  InclusiveRange y_;
  InclusiveRange z_;
  std::size_t owner_component_first_face_slot_;
  int next_owner_offset_ = 0;
  InclusiveRange local_x_{};
  InclusiveRange local_y_{};
  InclusiveRange local_z_{};
  std::int32_t local_y_index_ = 0;
  std::int32_t local_z_index_ = 0;
};

}  // namespace fluidity::cuda_internal
