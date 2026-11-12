#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/base/liquid_boundary_transfer_types_3d.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {

class AdaptiveMacTopology3D;
struct AdaptiveMacFaceOwnership3D;

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

class AdaptiveMacStencilBinding3D {
 public:
  // One optional global slot per input stencil row. A missing slot has no
  // geometric owner and must be excluded from deposition and gathering.
  std::span<const std::optional<std::size_t>> slots() const {
    return std::span(slots_).first(row_count_);
  }

 private:
  friend absl::StatusOr<AdaptiveMacStencilBinding3D> BindAdaptiveMacStencil3D(
      const AdaptiveMacTopology3D&,
      const AdaptiveMacFaceOwnership3D&,
      const CoarseFineStencil3D&,
      CellIndex3D,
      std::uint8_t
  );
  std::array<std::optional<std::size_t>, CoarseFineStencil3D::kMaxRows> slots_{};
  std::size_t row_count_ = 0;
};

// Map a patch stencil to the matching frozen topology and its ownership.
// The patch origin is a signed finest-lattice coordinate aligned to the
// supplied coarse exponent (1..30). Level 1 is exactly one exponent finer.
// The caller pairs the stencil with the patch's world origin and width.
// Exact geometry and width are required; a subdivided coarse face has no
// slot of its former width. Returns InvalidArgument for malformed alignment
// or ownership extents and OutOfRange for unrepresentable face coordinates.
// No allocation and no output on failure.
absl::StatusOr<AdaptiveMacStencilBinding3D> BindAdaptiveMacStencil3D(
    const AdaptiveMacTopology3D& topology,
    const AdaptiveMacFaceOwnership3D& ownership,
    const CoarseFineStencil3D& stencil,
    CellIndex3D patch_origin_finest,
    std::uint8_t coarse_width_exponent
);
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

struct CoarseFinePicCarrierSample3D {
  FaceIndex3D owner;
  std::uint8_t level;  // Same level and lattice convention as CoarseFineStencilRow3D.
  double new_face_velocity;  // Caller-certified usable new-stage Eulerian normal velocity, m/s.
  bool low_mass = false;  // True when a certified Eulerian velocity is available despite low deposited mass.
};

// Applies PIC for an incomplete or rank-deficient mask. Every true mask row must have exactly one
// finite, caller-certified sample. Sample order is arbitrary. Low mass is eligible when the caller
// supplies a usable Eulerian velocity. Retained positive weights are normalized in canonical row order.
// Sets the affected component of new_pic_velocity and clears its grid_velocity_increment,
// particle_minus_old_pic_velocity, fitted_velocity_slopes row and velocity_gradient row.
// Disables particle affine use, sets new_pic_fallback and sets low_mass_fallback if a used donor is low mass.
// Other component values and old_pic_fallback are preserved.
// Returns kInvalidArgument for a malformed mask, unexpected/duplicate owner or nonfinite velocity,
// kFailedPrecondition for complete support, missing retained samples or an empty mask, or kOutOfRange
// for nonfinite arithmetic. Errors leave gather unchanged.
absl::Status ApplyCoarseFinePicFallback3D(
    const CoarseFineStencil3D& stencil,
    std::span<const bool> retained_rows,
    std::span<const CoarseFinePicCarrierSample3D> carriers,
    LiquidBoundaryParticleGather3D& gather
);

// Apply the existing PIC fallback to a bound component using certified
// positive-normal velocities in m/s. All carrier spans use global slot order.
// Missing, structurally ineligible or different-side faces are excluded.
// Low-mass carriers must supply a certified finite velocity. Complete support
// and an empty mask return FailedPrecondition. Bad span extents or slots
// return InvalidArgument. Failure preserves gather.
absl::Status ApplyAdaptiveMacPicFallback3D(
    const CoarseFineStencil3D& stencil,
    const AdaptiveMacStencilBinding3D& binding,
    std::span<const double> face_velocities,
    std::span<const LiquidFaceValidity3D> face_validity,
    std::span<const LiquidBoundaryFaceTag3D> face_tags,
    std::int64_t particle_side_tag,
    LiquidBoundaryParticleGather3D& gather
);
}  // namespace fluidity
