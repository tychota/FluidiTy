#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#include <cuda/std/cmath>
#else
#include <cmath>
#endif

#include "fluidity/base/host_device.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity::internal {

#if defined(__CUDACC__)
namespace coarse_fine_numeric = cuda::std;
#else
namespace coarse_fine_numeric = std;
#endif

inline constexpr std::size_t kMaxCoarseFineCoreRows3D = 2 * 4 * 3 * 3;

struct CoarseFineCoreQuery3D {
  Vec3 world_origin;  // Common lattice origin in meters.
  double coarse_cell_width;  // Coarse width in meters.
  int fine_side[3];  // -1 or +1 selects an oriented half-space, 0 leaves an axis unbounded.
  FaceNormalAxis face_normal;
  Vec3 particle_world_position;  // Meters.
};

enum class CoarseFineCoreStatus3D : std::uint8_t {
  kOk,
  kInvalidArgument,
  kOutOfRange,
};

struct CoarseFineCoreRow3D {
  FaceIndex3D owner;  // Low-face owner in signed world-axis coordinates on its own level.
  std::uint8_t level;  // 0 for coarse and 1 for fine.
  float weight;
  float offset_coarse_units[3];  // Greville anchor minus particle, divided by coarse cell width.
  float gradient_coarse_units[3];  // World-position derivative times coarse cell width.
};

struct CoarseFineCoreResult3D {
  CoarseFineCoreStatus3D status;
  std::uint32_t row_count;
  CoarseFineCoreRow3D rows[kMaxCoarseFineCoreRows3D];
};

namespace coarse_fine_detail {

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

FLUIDITY_HD constexpr float SupportRadius(bool normal) {
  return normal ? 2.0f : 1.5f;
}

// Evaluates a centered cardinal factor in the units of its own lattice.
FLUIDITY_HD inline BasisFactor CardinalFactor(float displacement, bool normal) {
  const float distance = coarse_fine_numeric::abs(displacement);
  const float support_distance = SupportRadius(normal) - distance;
  const float edge_distance = support_distance > 0.0f ? support_distance : 0.0f;
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
FLUIDITY_HD inline void SplitChildren(float local_coordinate, float anchor, bool normal, AxisRow& row) {
  constexpr float cubic_mask[]{0.125f, 0.5f, 0.75f, 0.5f, 0.125f};
  constexpr float quadratic_mask[]{0.25f, 0.75f, 0.75f, 0.25f};
  const float* mask = normal ? cubic_mask : quadratic_mask;
  const int child_count = normal ? 5 : 4;
  const float radius = SupportRadius(normal);
  for (int child = 0; child < child_count; ++child) {
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
  AxisRow coarse[4];
  AxisRow fine[4];
};

// Stores each axis in ascending world order, reversing the local order on reflected axes.
FLUIDITY_HD inline AxisRows EvaluateAxis(float local_coordinate, bool normal, int fine_side) {
  AxisRows rows{};
  const int first_coarse = static_cast<int>(coarse_fine_numeric::floor(local_coordinate)) - 1;
  const int first_fine = static_cast<int>(coarse_fine_numeric::floor(2.0f * local_coordinate)) - 1;
  const int count = normal ? 4 : 3;
  const float anchor_offset = normal ? 0.0f : 0.5f;
  for (int index = 0; index < count; ++index) {
    const int slot = fine_side < 0 ? count - index - 1 : index;
    auto& coarse = rows.coarse[slot];
    coarse.index = first_coarse + index;
    const float coarse_anchor = static_cast<float>(coarse.index) + anchor_offset;
    coarse.uniform = CardinalFactor(local_coordinate - coarse_anchor, normal);
    if (fine_side != 0) {
      SplitChildren(local_coordinate, coarse_anchor, normal, coarse);
    }
    auto& fine = rows.fine[slot];
    fine.index = first_fine + index;
    const float fine_anchor = static_cast<float>(fine.index) + anchor_offset;
    if (fine_side == 0 || fine_anchor >= SupportRadius(normal)) {
      fine.uniform = CardinalFactor(2.0f * local_coordinate - fine_anchor, normal);
      fine.uniform.derivative *= 2.0f;
    }
  }
  return rows;
}

struct TensorFactor {
  float weight{};
  float gradient[3]{};
};

FLUIDITY_HD inline void AddTensor(BasisFactor x, BasisFactor y, BasisFactor z, TensorFactor& result) {
  result.weight += x.weight * y.weight * z.weight;
  result.gradient[0] += x.derivative * y.weight * z.weight;
  result.gradient[1] += x.weight * y.derivative * z.weight;
  result.gradient[2] += x.weight * y.weight * z.derivative;
}

FLUIDITY_HD inline TensorFactor
EvaluateCoarseTensor(const AxisRow& x, const AxisRow& y, const AxisRow& z, const int (&fine_side)[3], bool adaptive) {
  const AxisRow* axes[]{&x, &y, &z};
  BasisFactor factors[]{x.uniform, y.uniform, z.uniform};
  TensorFactor result{};
  if (!adaptive) {
    AddTensor(factors[0], factors[1], factors[2], result);
    return result;
  }
  // Every retained child contributes under the first axis whose containment test fails.
  // Later axes sum to the unrefined spline by the dyadic identity.
  for (int axis = 0; axis < 3; ++axis) {
    if (fine_side[axis] != 0) {
      factors[axis] = axes[axis]->outside;
      AddTensor(factors[0], factors[1], factors[2], result);
      factors[axis] = axes[axis]->inside;
    }
  }
  return result;
}

}  // namespace coarse_fine_detail

// Evaluates one bounded query with FP32 stencil arithmetic after double normalization.
// Positive rows are written in canonical order. Invalid geometry or position leaves row_count zero.
FLUIDITY_HD inline void EvaluateCoarseFineCore3D(const CoarseFineCoreQuery3D& query, CoarseFineCoreResult3D& result) {
  result.row_count = 0;
  const auto normal_axis = static_cast<int>(query.face_normal);
  if (normal_axis >= 3 || !coarse_fine_numeric::isfinite(query.coarse_cell_width) || query.coarse_cell_width <= 0) {
    result.status = CoarseFineCoreStatus3D::kInvalidArgument;
    return;
  }
  const double origin[]{query.world_origin.x, query.world_origin.y, query.world_origin.z};
  const double position[]{
      query.particle_world_position.x,
      query.particle_world_position.y,
      query.particle_world_position.z,
  };
  float coarse_coordinates[3]{};
  int signs[3]{};
  int axis_row_counts[3]{};
  for (int axis = 0; axis < 3; ++axis) {
    const int fine_side = query.fine_side[axis];
    if (fine_side < -1 || fine_side > 1 || !coarse_fine_numeric::isfinite(origin[axis])
        || !coarse_fine_numeric::isfinite(position[axis])) {
      result.status = CoarseFineCoreStatus3D::kInvalidArgument;
      return;
    }
    const double coordinate = (position[axis] - origin[axis]) / query.coarse_cell_width;
    if (coarse_fine_numeric::abs(coordinate) > 64.0) {
      result.status = CoarseFineCoreStatus3D::kOutOfRange;
      return;
    }
    coarse_coordinates[axis] = static_cast<float>(coordinate);
    signs[axis] = fine_side < 0 ? -1 : 1;
    axis_row_counts[axis] = axis == normal_axis ? 4 : 3;
  }

  coarse_fine_detail::AxisRows axes[3]{};
  for (int axis = 0; axis < 3; ++axis) {
    const float local_coordinate = static_cast<float>(signs[axis]) * coarse_coordinates[axis];
    axes[axis] = coarse_fine_detail::EvaluateAxis(local_coordinate, axis == normal_axis, query.fine_side[axis]);
  }
  const bool adaptive = query.fine_side[0] != 0 || query.fine_side[1] != 0 || query.fine_side[2] != 0;
  const std::uint8_t level_count = adaptive ? 2 : 1;
  for (std::uint8_t level = 0; level < level_count; ++level) {
    const auto* x_rows = level == 0 ? axes[0].coarse : axes[0].fine;
    const auto* y_rows = level == 0 ? axes[1].coarse : axes[1].fine;
    const auto* z_rows = level == 0 ? axes[2].coarse : axes[2].fine;
    const float level_width = level == 0 ? 1.0f : 0.5f;
    for (int x_index = 0; x_index < axis_row_counts[0]; ++x_index) {
      for (int y_index = 0; y_index < axis_row_counts[1]; ++y_index) {
        for (int z_index = 0; z_index < axis_row_counts[2]; ++z_index) {
          const auto& x = x_rows[x_index];
          const auto& y = y_rows[y_index];
          const auto& z = z_rows[z_index];
          coarse_fine_detail::TensorFactor factor{};
          if (level == 0) {
            factor = coarse_fine_detail::EvaluateCoarseTensor(x, y, z, query.fine_side, adaptive);
          } else {
            coarse_fine_detail::AddTensor(x.uniform, y.uniform, z.uniform, factor);
          }
          if (factor.weight <= 0) {
            continue;
          }
          const int local_indices[]{x.index, y.index, z.index};
          int world_indices[3]{};
          auto& row = result.rows[result.row_count++];
          for (int axis = 0; axis < 3; ++axis) {
            const bool transverse = axis != normal_axis;
            // Reflecting i + 1/2 maps to lattice index -i - 1.
            const int index_shift = transverse && signs[axis] < 0 ? -1 : 0;
            world_indices[axis] = signs[axis] * local_indices[axis] + index_shift;
            const float anchor = static_cast<float>(world_indices[axis]) + (transverse ? 0.5f : 0.0f);
            row.offset_coarse_units[axis] = level_width * anchor - coarse_coordinates[axis];
            row.gradient_coarse_units[axis] = factor.gradient[axis] * static_cast<float>(signs[axis]);
          }
          row.owner = {query.face_normal, {world_indices[0], world_indices[1], world_indices[2]}};
          row.level = level;
          row.weight = factor.weight;
        }
      }
    }
  }
  result.status = CoarseFineCoreStatus3D::kOk;
}

}  // namespace fluidity::internal
