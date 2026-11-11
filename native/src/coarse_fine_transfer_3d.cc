#include "fluidity/host/coarse_fine_transfer_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>

#include "absl/status/status.h"
#include "fluidity/base/coarse_fine_transfer_math_3d.h"

namespace fluidity {
namespace {

auto MakeOwnerKey(std::uint8_t level, FaceIndex3D owner) {
  const auto [x, y, z] = owner.lattice_index;
  return std::tuple{level, owner.axis, x, y, z};
}

auto OwnerKey(const CoarseFineStencilRow3D& row) {
  return MakeOwnerKey(row.level, row.owner);
}

using LatticePoint = std::array<int, 3>;

// The finest transverse anchors land at quarter coarse-cell coordinates, so these integers
// preserve exact coplanarity even when the FP32 particle-relative offsets have rounded.
LatticePoint AnchorInQuarterCoarseUnits(const CoarseFineStencilRow3D& row) {
  const auto [x, y, z] = row.owner.lattice_index;
  const std::array indices{x, y, z};
  const int quarter_units_per_level_width = row.level == 0 ? 4 : 2;
  LatticePoint anchor{};
  for (std::size_t axis = 0; axis < anchor.size(); ++axis) {
    const int stagger = axis == static_cast<std::size_t>(row.owner.axis) ? 0 : 1;
    anchor[axis] = indices[axis] * quarter_units_per_level_width + stagger * (quarter_units_per_level_width / 2);
  }
  return anchor;
}

LatticePoint Cross(const LatticePoint& a, const LatticePoint& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

}  // namespace

absl::StatusOr<CoarseFineStencil3D>
EvaluateCoarseFineStencil3D(const CoarseFinePatch3D& patch, FaceNormalAxis face_normal, Vec3 particle_world_position) {
  const internal::CoarseFineCoreQuery3D query{
      .world_origin = patch.world_origin,
      .coarse_cell_width = patch.coarse_cell_width,
      .fine_side = {patch.fine_side[0], patch.fine_side[1], patch.fine_side[2]},
      .face_normal = face_normal,
      .particle_world_position = particle_world_position,
  };
  internal::CoarseFineCoreResult3D result{};
  internal::EvaluateCoarseFineCore3D(query, result);
  switch (result.status) {
    case internal::CoarseFineCoreStatus3D::kInvalidArgument: {
      return absl::InvalidArgumentError("invalid coarse-fine query geometry or position");
    }
    case internal::CoarseFineCoreStatus3D::kOutOfRange: {
      return absl::OutOfRangeError("particle position is outside the bounded coarse-fine patch");
    }
    case internal::CoarseFineCoreStatus3D::kOk: {
      break;
    }
  }
  static_assert(CoarseFineStencil3D::kMaxRows == internal::kMaxCoarseFineCoreRows3D);
  CoarseFineStencil3D stencil;
  stencil.row_count_ = result.row_count;
  for (std::size_t index = 0; index < stencil.row_count_; ++index) {
    const auto& row = result.rows[index];
    stencil.rows_[index] = CoarseFineStencilRow3D{
        .owner = row.owner,
        .level = row.level,
        .weight = row.weight,
        .offset_coarse_units = {row.offset_coarse_units[0], row.offset_coarse_units[1], row.offset_coarse_units[2]},
        .gradient_coarse_units =
            {row.gradient_coarse_units[0], row.gradient_coarse_units[1], row.gradient_coarse_units[2]},
    };
  }
  return stencil;
}
absl::StatusOr<CoarseFineMaskDisposition3D>
ClassifyCoarseFineMask3D(const CoarseFineStencil3D& stencil, std::span<const bool> retained_rows) {
  const auto rows = stencil.rows();
  if (retained_rows.size() != rows.size()) {
    return absl::InvalidArgumentError("coarse-fine mask size differs from stencil size");
  }
  const auto first_retained = std::ranges::find(retained_rows, true);
  if (first_retained == retained_rows.end()) {
    return absl::FailedPreconditionError("coarse-fine mask retains no carrier");
  }
  if (std::ranges::all_of(retained_rows, [](bool retained) {
        return retained;
      })) {
    return CoarseFineMaskDisposition3D::kComplete;
  }

  // A nonzero scalar triple product establishes affine rank three. All retained anchors lie
  // within the combined support of a single bounded query, so differences and products fit in int.
  const auto first_index = static_cast<std::size_t>(first_retained - retained_rows.begin());
  const LatticePoint origin = AnchorInQuarterCoarseUnits(rows[first_index]);
  LatticePoint direction{};
  LatticePoint normal{};
  for (std::size_t index = first_index + 1; index < rows.size(); ++index) {
    if (!retained_rows[index]) {
      continue;
    }
    const auto anchor = AnchorInQuarterCoarseUnits(rows[index]);
    const LatticePoint displacement{anchor[0] - origin[0], anchor[1] - origin[1], anchor[2] - origin[2]};
    if (direction == LatticePoint{}) {
      direction = displacement;
    } else if (normal == LatticePoint{}) {
      normal = Cross(direction, displacement);
    } else {
      const int signed_volume = normal[0] * displacement[0] + normal[1] * displacement[1] + normal[2] * displacement[2];
      if (signed_volume != 0) {
        return CoarseFineMaskDisposition3D::kPicRequiredIncomplete;
      }
    }
  }
  return CoarseFineMaskDisposition3D::kPicRequiredRankDeficient;
}

absl::Status ApplyCoarseFinePicFallback3D(
    const CoarseFineStencil3D& stencil,
    std::span<const bool> retained_rows,
    std::span<const CoarseFinePicCarrierSample3D> carriers,
    LiquidBoundaryParticleGather3D& gather
) {
  const auto disposition = ClassifyCoarseFineMask3D(stencil, retained_rows);
  if (!disposition.ok()) {
    return disposition.status();
  }
  if (*disposition == CoarseFineMaskDisposition3D::kComplete) {
    return absl::FailedPreconditionError("coarse-fine PIC fallback requires incomplete support");
  }

  const auto rows = stencil.rows();
  std::array<const CoarseFinePicCarrierSample3D*, CoarseFineStencil3D::kMaxRows> carrier_by_row{};
  double minimum_velocity = std::numeric_limits<double>::infinity();
  double maximum_velocity = -std::numeric_limits<double>::infinity();
  bool low_mass = false;
  for (const auto& carrier : carriers) {
    const auto key = MakeOwnerKey(carrier.level, carrier.owner);
    const auto row = std::ranges::lower_bound(rows, key, {}, OwnerKey);
    if (row == rows.end() || OwnerKey(*row) != key) {
      return absl::InvalidArgumentError("coarse-fine PIC carrier owner is outside the stencil");
    }
    const auto index = static_cast<std::size_t>(row - rows.begin());
    if (!retained_rows[index] || carrier_by_row[index] != nullptr) {
      return absl::InvalidArgumentError("coarse-fine PIC carrier owner is masked or duplicated");
    }
    if (!std::isfinite(carrier.new_face_velocity)) {
      return absl::InvalidArgumentError("coarse-fine PIC carrier velocity must be finite");
    }
    carrier_by_row[index] = &carrier;
    low_mass = low_mass || carrier.low_mass;
  }

  double total_weight = 0;
  for (std::size_t index = 0; index < rows.size(); ++index) {
    if (!retained_rows[index]) {
      continue;
    }
    const auto* carrier = carrier_by_row[index];
    if (carrier == nullptr) {
      return absl::FailedPreconditionError("coarse-fine PIC requires a valid carrier for each retained row");
    }
    total_weight += rows[index].weight;
    minimum_velocity = std::min(minimum_velocity, carrier->new_face_velocity);
    maximum_velocity = std::max(maximum_velocity, carrier->new_face_velocity);
  }
  if (!std::isfinite(total_weight) || total_weight <= 0) {
    return absl::OutOfRangeError("coarse-fine PIC requires positive finite weight");
  }

  // Scale before accumulating so finite donors near the floating-point limits
  // cannot overflow the sum. Equal donors reproduce their constant exactly.
  double new_pic_velocity = minimum_velocity;
  if (minimum_velocity != maximum_velocity) {
    const double velocity_scale = std::max(-minimum_velocity, maximum_velocity);
    double scaled_weighted_velocity = 0;
    for (std::size_t index = 0; index < rows.size(); ++index) {
      if (const auto* carrier = carrier_by_row[index]; carrier != nullptr) {
        scaled_weighted_velocity += rows[index].weight * (carrier->new_face_velocity / velocity_scale);
      }
    }
    new_pic_velocity = (scaled_weighted_velocity / total_weight) * velocity_scale;
  }
  if (!std::isfinite(new_pic_velocity)) {
    return absl::OutOfRangeError("coarse-fine PIC velocity is nonfinite");
  }

  constexpr std::array velocity_components{&Vec3::x, &Vec3::y, &Vec3::z};
  constexpr std::array matrix_rows{&Mat3::x_row, &Mat3::y_row, &Mat3::z_row};
  const auto normal_axis = static_cast<std::size_t>(rows.front().owner.axis);
  const auto component = velocity_components[normal_axis];
  const auto matrix_row = matrix_rows[normal_axis];
  gather.new_pic_velocity.*component = new_pic_velocity;
  gather.grid_velocity_increment.*component = 0.0;
  gather.particle_minus_old_pic_velocity.*component = 0.0;
  gather.fitted_velocity_slopes.*matrix_row = {};
  gather.velocity_gradient.*matrix_row = {};
  gather.affine_enabled = false;
  gather.new_pic_fallback = true;
  gather.low_mass_fallback = gather.low_mass_fallback || low_mass;
  return absl::OkStatus();
}

}  // namespace fluidity
