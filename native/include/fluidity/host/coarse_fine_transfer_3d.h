#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {

struct CoarseFinePatch3D {
  Vec3 world_origin;  // Common lattice origin at the junction in meters.
  double coarse_cell_width;  // Positive coarse spacing in meters. Fine spacing is half this width.
  // One entry per world axis. A nonzero sign selects the fine half-space measured from world_origin.
  // Zero leaves that axis unbounded. The intersection of the selected half-spaces is the fine region.
  // With all entries zero, the patch is uniform coarse.
  std::array<int, 3> fine_side;
};

struct CoarseFineStencilRow3D {
  // The owner is the cell with this lattice index at this level, via its low face on owner.axis.
  // Indices are in world-axis order, even on reflected patches.
  FaceIndex3D owner;
  std::uint8_t level;  // 0 is coarse and 1 is fine. Both levels use the patch origin.
  float weight;
  std::array<float, 3> offset_coarse_units;  // Original Greville anchor minus particle, divided by coarse width.
  std::array<float, 3> gradient_coarse_units;  // Derivative with respect to world position, times coarse width.
};

class CoarseFineStencil3D {
 public:
  static constexpr std::size_t kMaxRows = 2 * 4 * 3 * 3;

  // Returns the positive-weight rows in ascending (level, world x, y, z) lattice order.
  std::span<const CoarseFineStencilRow3D> rows() const {
    return std::span{rows_}.first(row_count_);
  }

 private:
  CoarseFineStencil3D() = default;

  friend absl::StatusOr<CoarseFineStencil3D>
  EvaluateCoarseFineStencil3D(const CoarseFinePatch3D&, FaceNormalAxis, Vec3);

  std::array<CoarseFineStencilRow3D, kMaxRows> rows_{};
  std::size_t row_count_{};
};

enum class CoarseFineMaskDisposition3D : std::uint8_t {
  kComplete,
  kPicRequiredIncomplete,
  kPicRequiredRankDeficient,
};

// Evaluates the bounded 2:1 cubic-normal, quadratic-transverse stencil for one MAC face component.
// Fine splines enter where their closed support is contained. Coarse rows retain dyadic child terms
// extending outside the fine region. Weights and normalized derivatives use FP32 arithmetic.
// Returns kInvalidArgument for invalid signs, axis or nonfinite geometry or position, and kOutOfRange
// if the position is outside [-64, 64] coarse units on any axis relative to the patch origin.
// On error no stencil is returned.
absl::StatusOr<CoarseFineStencil3D>
EvaluateCoarseFineStencil3D(const CoarseFinePatch3D& patch, FaceNormalAxis face_normal, Vec3 particle_world_position);

// Classifies the subset of rows whose mask entries are true, in stencil order. Affine rank uses the
// exact lattice anchors. The result specifies whether the full stencil can be used or PIC is required.
// Returns kInvalidArgument for a mask of the wrong size or kFailedPrecondition when no carrier remains.
absl::StatusOr<CoarseFineMaskDisposition3D>
ClassifyCoarseFineMask3D(const CoarseFineStencil3D& stencil, std::span<const bool> retained_rows);

}  // namespace fluidity
