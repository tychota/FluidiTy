#include "fluidity/host/coarse_fine_transfer_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>

#include "absl/status/status.h"

namespace fluidity {
namespace {

struct BasisFactor {
  float weight{};
  float derivative{};
};

struct AxisRow {
  int index{};
  BasisFactor uniform;
  BasisFactor outside;
  BasisFactor inside;
};

constexpr float SupportRadius(bool normal) {
  return normal ? 2.0f : 1.5f;
}

// Evaluates a centered cardinal factor in the units of its own lattice.
BasisFactor CardinalFactor(float displacement, bool normal) {
  const float distance = std::abs(displacement);
  const float edge_distance = std::max(SupportRadius(normal) - distance, 0.0f);
  const float sign = displacement < 0 ? -1.0f : 1.0f;
  if (normal) {
    if (distance < 1.0f) {
      return {(4.0f - distance * distance * (6.0f - 3.0f * distance)) / 6.0f, displacement * (1.5f * distance - 2.0f)};
    }
    return {edge_distance * edge_distance * edge_distance / 6.0f, -0.5f * edge_distance * edge_distance * sign};
  }
  if (distance < 0.5f) {
    return {0.75f - displacement * displacement, -2.0f * displacement};
  }
  return {0.5f * edge_distance * edge_distance, -edge_distance * sign};
}

// Split the refinement sum into children whose closed support fits in the fine half-space and
// children that extend outside. Keeping both positive sums avoids cancellation near a support edge.
void SplitChildren(float local_coordinate, float anchor, bool normal, AxisRow& row) {
  constexpr std::array cubic_mask{0.125f, 0.5f, 0.75f, 0.5f, 0.125f};
  constexpr std::array quadratic_mask{0.25f, 0.75f, 0.75f, 0.25f};
  const std::span<const float> mask =
      normal ? std::span<const float>{cubic_mask} : std::span<const float>{quadratic_mask};
  const float radius = SupportRadius(normal);
  for (std::size_t child = 0; child < mask.size(); ++child) {
    const float child_anchor = 2.0f * anchor + (static_cast<float>(child) - radius);
    const BasisFactor factor = CardinalFactor(2.0f * local_coordinate - child_anchor, normal);
    BasisFactor& destination = child_anchor < radius ? row.outside : row.inside;
    destination.weight += mask[child] * factor.weight;
    destination.derivative += mask[child] * factor.derivative;
  }
  row.outside.derivative *= 2.0f;
  row.inside.derivative *= 2.0f;
}

struct AxisRows {
  std::array<AxisRow, 4> coarse;
  std::array<AxisRow, 4> fine;
};

AxisRows EvaluateAxis(float local_coordinate, bool normal, bool bounded) {
  AxisRows rows{};
  const int first_coarse = static_cast<int>(std::floor(local_coordinate)) - 1;
  const int first_fine = static_cast<int>(std::floor(2.0f * local_coordinate)) - 1;
  const std::size_t count = normal ? 4 : 3;
  const float anchor_offset = normal ? 0.0f : 0.5f;
  for (std::size_t index = 0; index < count; ++index) {
    auto& coarse = rows.coarse[index];
    coarse.index = first_coarse + static_cast<int>(index);
    const float coarse_anchor = static_cast<float>(coarse.index) + anchor_offset;
    coarse.uniform = CardinalFactor(local_coordinate - coarse_anchor, normal);
    if (bounded) {
      SplitChildren(local_coordinate, coarse_anchor, normal, coarse);
    }
    auto& fine = rows.fine[index];
    fine.index = first_fine + static_cast<int>(index);
    const float fine_anchor = static_cast<float>(fine.index) + anchor_offset;
    if (!bounded || fine_anchor >= SupportRadius(normal)) {
      fine.uniform = CardinalFactor(2.0f * local_coordinate - fine_anchor, normal);
      fine.uniform.derivative *= 2.0f;
    }
  }
  return rows;
}

struct TensorFactor {
  float weight{};
  std::array<float, 3> gradient{};
};

void AddTensor(const std::array<BasisFactor, 3>& factors, TensorFactor& result) {
  const auto [x, y, z] = factors;
  result.weight += x.weight * y.weight * z.weight;
  result.gradient[0] += x.derivative * y.weight * z.weight;
  result.gradient[1] += x.weight * y.derivative * z.weight;
  result.gradient[2] += x.weight * y.weight * z.derivative;
}

TensorFactor EvaluateCoarseTensor(const std::array<AxisRow, 3>& axes, const std::array<int, 3>& fine_side) {
  std::array factors{axes[0].uniform, axes[1].uniform, axes[2].uniform};
  TensorFactor result{};
  if (fine_side == std::array{0, 0, 0}) {
    AddTensor(factors, result);
    return result;
  }
  // Every retained child contributes under the first axis whose containment test fails.
  // Later axes sum to the unrefined spline by the dyadic identity.
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    if (fine_side[axis] != 0) {
      factors[axis] = axes[axis].outside;
      AddTensor(factors, result);
      factors[axis] = axes[axis].inside;
    }
  }
  return result;
}

auto OwnerKey(const CoarseFineStencilRow3D& row) {
  const auto [x, y, z] = row.owner.lattice_index;
  return std::tuple{row.level, x, y, z};
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
  const auto normal_axis = static_cast<std::size_t>(face_normal);
  if (normal_axis >= 3 || !std::isfinite(patch.coarse_cell_width) || patch.coarse_cell_width <= 0) {
    return absl::InvalidArgumentError("invalid coarse width or MAC face component");
  }
  const auto [origin_x, origin_y, origin_z] = patch.world_origin;
  const auto [position_x, position_y, position_z] = particle_world_position;
  const std::array origin{origin_x, origin_y, origin_z};
  const std::array position{position_x, position_y, position_z};
  std::array<float, 3> coarse_coordinates{};
  std::array<int, 3> signs{};
  for (std::size_t axis = 0; axis < origin.size(); ++axis) {
    const int fine_side = patch.fine_side[axis];
    if (fine_side < -1 || fine_side > 1 || !std::isfinite(origin[axis]) || !std::isfinite(position[axis])) {
      return absl::InvalidArgumentError("invalid fine-side sign, patch origin or particle position");
    }
    const double coordinate = (position[axis] - origin[axis]) / patch.coarse_cell_width;
    if (std::abs(coordinate) > 64.0) {
      return absl::OutOfRangeError("particle position is outside the bounded coarse-fine patch");
    }
    coarse_coordinates[axis] = static_cast<float>(coordinate);
    signs[axis] = fine_side < 0 ? -1 : 1;
  }

  std::array<AxisRows, 3> axes{};
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    const float local_coordinate = static_cast<float>(signs[axis]) * coarse_coordinates[axis];
    axes[axis] = EvaluateAxis(local_coordinate, axis == normal_axis, patch.fine_side[axis] != 0);
  }

  CoarseFineStencil3D stencil;
  const bool adaptive = patch.fine_side != std::array{0, 0, 0};
  const std::uint8_t level_count = adaptive ? 2 : 1;
  for (std::uint8_t level = 0; level < level_count; ++level) {
    const auto axis_rows = [&](std::size_t axis) {
      const auto& rows = level == 0 ? axes[axis].coarse : axes[axis].fine;
      return std::span{rows}.first(axis == normal_axis ? 4 : 3);
    };
    const float level_width = level == 0 ? 1.0f : 0.5f;
    for (const auto& x : axis_rows(0)) {
      for (const auto& y : axis_rows(1)) {
        for (const auto& z : axis_rows(2)) {
          TensorFactor factor{};
          if (level == 0) {
            factor = EvaluateCoarseTensor({x, y, z}, patch.fine_side);
          } else {
            AddTensor({x.uniform, y.uniform, z.uniform}, factor);
          }
          if (factor.weight <= 0) {
            continue;
          }
          const std::array local_indices{x.index, y.index, z.index};
          std::array<int, 3> world_indices{};
          std::array<float, 3> offset{};
          for (std::size_t axis = 0; axis < world_indices.size(); ++axis) {
            const bool transverse = axis != normal_axis;
            // Reflecting i + 1/2 maps to lattice index -i - 1.
            const int index_shift = transverse && signs[axis] < 0 ? -1 : 0;
            world_indices[axis] = signs[axis] * local_indices[axis] + index_shift;
            const float anchor = static_cast<float>(world_indices[axis]) + (transverse ? 0.5f : 0.0f);
            offset[axis] = level_width * anchor - coarse_coordinates[axis];
            factor.gradient[axis] *= static_cast<float>(signs[axis]);
          }
          stencil.rows_[stencil.row_count_++] = CoarseFineStencilRow3D{
              .owner = {face_normal, {world_indices[0], world_indices[1], world_indices[2]}},
              .level = level,
              .weight = factor.weight,
              .offset_coarse_units = offset,
              .gradient_coarse_units = factor.gradient,
          };
        }
      }
    }
  }
  std::ranges::sort(std::span{stencil.rows_}.first(stencil.row_count_), {}, OwnerKey);
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

}  // namespace fluidity
