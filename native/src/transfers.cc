#include "fluidity/host/transfers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <tuple>

#include "fluidity/host/basis.h"
#include "fluidity/host/views.h"

namespace fluidity {
namespace {

constexpr std::size_t kVelocityComponentCount = 2;
constexpr std::size_t kTransferStencilSlotCount = std::tuple_size_v<decltype(Stencil2D::nodes)>;
constexpr std::size_t kDepositDoublesPerFace = 2;
constexpr std::size_t kMatrixEntryCount = kVelocityComponentCount * kVelocityComponentCount;
constexpr std::size_t kGatherDoublesPerParticle = 2 * kVelocityComponentCount + 2 * kMatrixEntryCount;

bool IsFinite(Vec2 vector) {
  return std::isfinite(vector.x) && std::isfinite(vector.y);
}

bool IsFinite(Mat2 matrix) {
  return std::isfinite(matrix.xx) && std::isfinite(matrix.xy) && std::isfinite(matrix.yx) && std::isfinite(matrix.yy);
}

double Dot(Vec2 lhs, Vec2 rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y;
}

struct FaceStencil {
  Stencil2D basis;
  std::array<std::size_t, kTransferStencilSlotCount> face_indices;
  std::array<Vec2, kTransferStencilSlotCount> face_positions;
};

// FaceCount has validated the lattice sizes. Check the signed nodes before flattening them,
// including zero-weight nodes at exact half-cell ties.
absl::StatusOr<FaceStencil>
BuildFaceStencil(const Grid2D& mac_grid, Vec2 particle_world_position, std::size_t velocity_component_index) {
  const Vec2 particle_position_from_origin = particle_world_position - mac_grid.origin;
  if (!IsFinite(particle_position_from_origin)) {
    return absl::OutOfRangeError("position relative to grid origin is nonfinite");
  }
  const bool is_x_normal = velocity_component_index == 0;
  const Vec2 lattice_offset{
      .x = is_x_normal ? 0.0 : 0.5,
      .y = is_x_normal ? 0.5 : 0.0,
  };
  const auto basis = QuadraticStencil2D(particle_position_from_origin, mac_grid.h, lattice_offset);
  if (!basis.ok()) {
    return basis.status();
  }

  const std::int64_t face_column_count = static_cast<std::int64_t>(mac_grid.nx) + (is_x_normal ? 1 : 0);
  const std::int64_t face_row_count = static_cast<std::int64_t>(mac_grid.ny) + (is_x_normal ? 0 : 1);
  const std::size_t orientation_face_offset =
      is_x_normal ? 0 : (static_cast<std::size_t>(mac_grid.nx) + 1) * mac_grid.ny;
  FaceStencil stencil;
  stencil.basis = *basis;
  for (std::size_t stencil_slot_index = 0; stencil_slot_index < stencil.face_indices.size(); ++stencil_slot_index) {
    const Node2D lattice_node = basis->nodes[stencil_slot_index];
    const bool node_on_face_lattice = lattice_node.x >= 0 && lattice_node.y >= 0 && lattice_node.x < face_column_count
        && lattice_node.y < face_row_count;
    if (!node_on_face_lattice) {
      return absl::InvalidArgumentError("particle does not have complete MAC support");
    }
    const std::size_t face_index = orientation_face_offset
        + static_cast<std::size_t>(lattice_node.y) * static_cast<std::size_t>(face_column_count)
        + static_cast<std::size_t>(lattice_node.x);
    const auto face_geometry = FaceGeometry(mac_grid, face_index);
    if (!face_geometry.ok()) {
      return face_geometry.status();
    }
    stencil.face_indices[stencil_slot_index] = face_index;
    stencil.face_positions[stencil_slot_index] = face_geometry->position;
  }
  return stencil;
}

// With a partition of unity, subtracting this mean offset makes the weighted first moment
// vanish. The same centering is used in the deposition velocity and the affine fit.
absl::StatusOr<Vec2> WeightedOffsetMean(
    Vec2 reference_position,
    std::span<const Vec2> sample_positions,
    std::span<const double> sample_weights
) {
  Vec2 weighted_offset_mean{};
  for (std::size_t sample_index = 0; sample_index < sample_positions.size(); ++sample_index) {
    const Vec2 sample_offset = sample_positions[sample_index] - reference_position;
    weighted_offset_mean = weighted_offset_mean + sample_offset * sample_weights[sample_index];
  }
  if (!IsFinite(weighted_offset_mean)) {
    return absl::OutOfRangeError("weighted offset mean is nonfinite");
  }
  return weighted_offset_mean;
}

// Deposit one complete component stencil into batch scratch. Centering the affine term
// makes its weighted linear momentum zero while preserving its contribution to torque.
absl::Status DepositComponent(
    const FaceStencil& stencil,
    Vec2 particle_world_position,
    double particle_mass,
    double particle_velocity_component,
    Vec2 velocity_fit_slope_row,
    std::span<double> staged_face_masses,
    std::span<double> staged_face_pure_momenta
) {
  const auto weighted_face_offset_mean =
      WeightedOffsetMean(particle_world_position, stencil.face_positions, stencil.basis.weights);
  if (!weighted_face_offset_mean.ok()) {
    return weighted_face_offset_mean.status();
  }
  for (std::size_t stencil_slot_index = 0; stencil_slot_index < stencil.face_indices.size(); ++stencil_slot_index) {
    const std::size_t face_index = stencil.face_indices[stencil_slot_index];
    const Vec2 centered_face_offset =
        (stencil.face_positions[stencil_slot_index] - particle_world_position) - *weighted_face_offset_mean;
    const double deposited_velocity = particle_velocity_component + Dot(velocity_fit_slope_row, centered_face_offset);
    const double mass_contribution = stencil.basis.weights[stencil_slot_index] * particle_mass;
    staged_face_masses[face_index] += mass_contribution;
    staged_face_pure_momenta[face_index] += mass_contribution * deposited_velocity;
    if (!std::isfinite(staged_face_masses[face_index]) || !std::isfinite(staged_face_pure_momenta[face_index])) {
      return absl::OutOfRangeError("deposited mass or momentum is nonfinite");
    }
  }
  return absl::OkStatus();
}

struct BinaryParts {
  double fraction;
  int exponent;
};

// Split both factors before multiplying. Each nonzero fraction product is in [1/4, 1)
// in magnitude even when the full product is outside the range of double.
BinaryParts ProductParts(double first, double second) {
  int first_exponent;
  int second_exponent;
  const double first_fraction = std::frexp(first, &first_exponent);
  const double second_fraction = std::frexp(second, &second_exponent);
  return {
      .fraction = first_fraction * second_fraction,
      .exponent = first_exponent + second_exponent,
  };
}

// Align only the terms in this numerator. A term too small to survive alignment is
// negligible compared to its nonzero partner. Leave the quotient's range to the final scaling.
double DivideProductDifference(BinaryParts first, BinaryParts second, BinaryParts denominator) {
  int numerator_exponent = std::max(first.exponent, second.exponent);
  if (first.fraction == 0.0) {
    numerator_exponent = second.exponent;
  }
  if (second.fraction == 0.0) {
    numerator_exponent = first.exponent;
  }
  const double first_fraction = std::scalbn(first.fraction, first.exponent - numerator_exponent);
  const double second_fraction = std::scalbn(second.fraction, second.exponent - numerator_exponent);
  const double quotient_fraction = (first_fraction - second_fraction) / denominator.fraction;
  return std::scalbn(quotient_fraction, numerator_exponent - denominator.exponent);
}

// Scale the full symmetric moment before forming its determinant. Comparing the determinant
// to 2*epsilon*lambda_max^2 applies the two-dimensional relative rank test without subtracting
// two nearly equal eigenvalues. The caller has checked all four accumulated inputs.
absl::StatusOr<Vec2> SolveAffineMoment(double moment_xx, double moment_xy, double moment_yy, Vec2 offset_value_moment) {
  const double moment_scale = std::max(moment_xx, moment_yy);
  if (moment_scale == 0.0) {
    return absl::FailedPreconditionError("affine moment is singular");
  }
  const double normalized_xx = moment_xx / moment_scale;
  const double normalized_xy = moment_xy / moment_scale;
  const double normalized_yy = moment_yy / moment_scale;
  const double normalized_determinant = std::fma(normalized_xx, normalized_yy, -normalized_xy * normalized_xy);
  const double largest_eigenvalue =
      0.5 * (normalized_xx + normalized_yy + std::hypot(normalized_xx - normalized_yy, 2.0 * normalized_xy));
  constexpr double rank_tolerance = 2.0 * std::numeric_limits<double>::epsilon();
  if (normalized_determinant <= rank_tolerance * largest_eigenvalue * largest_eigenvalue) {
    return absl::FailedPreconditionError("affine moment is numerically singular");
  }

  // Cramer's rule uses the original moment entries in each numerator. Keep
  // the squared scale in the determinant exponent so neither it nor RHS
  // normalization constrains the range of a representable affine coefficient.
  const BinaryParts squared_scale = ProductParts(moment_scale, moment_scale);
  const BinaryParts moment_determinant{
      .fraction = normalized_determinant * squared_scale.fraction,
      .exponent = squared_scale.exponent,
  };
  const Vec2 affine_row{
      .x = DivideProductDifference(
          ProductParts(moment_yy, offset_value_moment.x),
          ProductParts(moment_xy, offset_value_moment.y),
          moment_determinant
      ),
      .y = DivideProductDifference(
          ProductParts(moment_xx, offset_value_moment.y),
          ProductParts(moment_xy, offset_value_moment.x),
          moment_determinant
      ),
  };
  if (!IsFinite(affine_row)) {
    return absl::OutOfRangeError("affine row is nonfinite");
  }
  return affine_row;
}

struct GatherComponentResult {
  double pic_velocity;
  double grid_velocity_increment;
  Vec2 velocity_fit_slope_row;
  Vec2 velocity_gradient_row;
};

// Both face fields are validated before the batch loop. Old and new values use the same
// basis and accumulation order so a shared old/new field gives exactly zero increment.
absl::StatusOr<GatherComponentResult> GatherComponent(
    const FaceStencil& stencil,
    Vec2 particle_world_position,
    std::span<const double> old_face_velocities,
    std::span<const double> new_face_velocities
) {
  std::array<double, kTransferStencilSlotCount> new_face_velocity_samples;
  double old_pic_velocity = 0.0;
  for (std::size_t stencil_slot_index = 0; stencil_slot_index < new_face_velocity_samples.size();
       ++stencil_slot_index) {
    const std::size_t face_index = stencil.face_indices[stencil_slot_index];
    new_face_velocity_samples[stencil_slot_index] = new_face_velocities[face_index];
    old_pic_velocity += stencil.basis.weights[stencil_slot_index] * old_face_velocities[face_index];
  }
  const auto affine_fit =
      FitAffine2D(particle_world_position, stencil.face_positions, stencil.basis.weights, new_face_velocity_samples);
  if (!affine_fit.ok()) {
    return affine_fit.status();
  }
  const auto velocity_gradient_row = VelocityGradient2D(new_face_velocity_samples, stencil.basis.gradients);
  if (!velocity_gradient_row.ok()) {
    return velocity_gradient_row.status();
  }
  const double grid_velocity_increment = affine_fit->interpolated_value - old_pic_velocity;
  if (!std::isfinite(grid_velocity_increment)) {
    return absl::OutOfRangeError("grid velocity increment is nonfinite");
  }
  return GatherComponentResult{
      .pic_velocity = affine_fit->interpolated_value,
      .grid_velocity_increment = grid_velocity_increment,
      .velocity_fit_slope_row = affine_fit->spatial_slopes,
      .velocity_gradient_row = *velocity_gradient_row,
  };
}

}  // namespace

absl::StatusOr<std::size_t> P2GScratchDoubles(const Grid2D& grid) {
  const auto face_count = FaceCount(grid);
  if (!face_count.ok()) {
    return face_count.status();
  }
  return CheckedExtentProduct(*face_count, kDepositDoublesPerFace);
}

absl::StatusOr<std::size_t> G2PScratchDoubles(std::size_t particle_count) {
  return CheckedExtentProduct(particle_count, kGatherDoublesPerParticle);
}

absl::Status P2G(
    const Grid2D& grid,
    ParticleView2D particles,
    std::span<double> face_mass,
    std::span<double> face_momentum,
    std::span<double> scratch
) {
  const auto scratch_count = P2GScratchDoubles(grid);
  if (!scratch_count.ok()) {
    return scratch_count.status();
  }
  const std::size_t face_count = *scratch_count / kDepositDoublesPerFace;
  const std::size_t particle_count = particles.positions.size();
  const auto particle_extents = CheckExtents(
      particle_count,
      {particles.masses.size(), particles.velocities.size(), particles.velocity_fit_slopes.size()}
  );
  if (!particle_extents.ok()) {
    return particle_extents;
  }
  const auto face_extents = CheckExtents(face_count, {face_mass.size(), face_momentum.size()});
  if (!face_extents.ok()) {
    return face_extents;
  }
  const auto scratch_extent = ValidateExtent(scratch.size(), *scratch_count);
  if (!scratch_extent.ok()) {
    return scratch_extent;
  }
  const auto disjoint = CheckWritableViews(
      {std::as_bytes(face_mass), std::as_bytes(face_momentum), std::as_bytes(scratch)},
      {
          std::as_bytes(std::span(&grid, 1)),
          std::as_bytes(particles.positions),
          std::as_bytes(particles.masses),
          std::as_bytes(particles.velocities),
          std::as_bytes(particles.velocity_fit_slopes),
      }
  );
  if (!disjoint.ok()) {
    return disjoint;
  }

  std::ranges::fill(scratch, 0.0);
  auto staged_face_masses = scratch.first(face_count);
  auto staged_face_pure_momenta = scratch.last(face_count);
  for (std::size_t particle_index = 0; particle_index < particle_count; ++particle_index) {
    const Vec2 particle_world_position = particles.positions[particle_index];
    const double particle_mass = particles.masses[particle_index];
    const Vec2 particle_velocity = particles.velocities[particle_index];
    const Mat2 particle_velocity_fit_slopes = particles.velocity_fit_slopes[particle_index];
    if (!IsFinite(particle_world_position) || !IsFinite(particle_velocity) || !IsFinite(particle_velocity_fit_slopes)) {
      return absl::InvalidArgumentError("particle position, velocity and affine fit must be finite");
    }
    const bool valid_mass = std::isfinite(particle_mass) && particle_mass >= 0.0;
    if (!valid_mass) {
      return absl::InvalidArgumentError("particle mass must be finite and nonnegative");
    }
    const std::array component_velocities{particle_velocity.x, particle_velocity.y};
    const std::array velocity_fit_slope_rows{
        Vec2{particle_velocity_fit_slopes.xx, particle_velocity_fit_slopes.xy},
        Vec2{particle_velocity_fit_slopes.yx, particle_velocity_fit_slopes.yy}
    };

    for (std::size_t velocity_component_index = 0; velocity_component_index < kVelocityComponentCount;
         ++velocity_component_index) {
      const auto stencil = BuildFaceStencil(grid, particle_world_position, velocity_component_index);
      if (!stencil.ok()) {
        return stencil.status();
      }
      const auto deposit_status = DepositComponent(
          *stencil,
          particle_world_position,
          particle_mass,
          component_velocities[velocity_component_index],
          velocity_fit_slope_rows[velocity_component_index],
          staged_face_masses,
          staged_face_pure_momenta
      );
      if (!deposit_status.ok()) {
        return deposit_status;
      }
    }
  }

  // All particles have complete support and every accumulated face is finite.
  std::ranges::copy(staged_face_masses, face_mass.begin());
  std::ranges::copy(staged_face_pure_momenta, face_momentum.begin());
  return absl::OkStatus();
}

absl::Status G2P(
    const Grid2D& mac_grid,
    std::span<const Vec2> particle_world_positions,
    std::span<const double> old_face_velocities,
    std::span<const double> new_face_velocities,
    std::span<Vec2> pic_velocities,
    std::span<Vec2> grid_velocity_increments,
    std::span<Mat2> velocity_fit_slopes,
    std::span<Mat2> velocity_gradients,
    std::span<double> gather_scratch
) {
  const auto face_count = FaceCount(mac_grid);
  if (!face_count.ok()) {
    return face_count.status();
  }
  const std::size_t particle_count = particle_world_positions.size();
  const auto scratch_count = G2PScratchDoubles(particle_count);
  if (!scratch_count.ok()) {
    return scratch_count.status();
  }
  const auto face_extents = CheckExtents(*face_count, {old_face_velocities.size(), new_face_velocities.size()});
  if (!face_extents.ok()) {
    return face_extents;
  }
  const auto particle_extents = CheckExtents(
      particle_count,
      {pic_velocities.size(), grid_velocity_increments.size(), velocity_fit_slopes.size(), velocity_gradients.size()}
  );
  if (!particle_extents.ok()) {
    return particle_extents;
  }
  const auto scratch_extent = ValidateExtent(gather_scratch.size(), *scratch_count);
  if (!scratch_extent.ok()) {
    return scratch_extent;
  }
  const auto disjoint = CheckWritableViews(
      {
          std::as_bytes(pic_velocities),
          std::as_bytes(grid_velocity_increments),
          std::as_bytes(velocity_fit_slopes),
          std::as_bytes(velocity_gradients),
          std::as_bytes(gather_scratch),
      },
      {
          std::as_bytes(std::span(&mac_grid, 1)),
          std::as_bytes(particle_world_positions),
          std::as_bytes(old_face_velocities),
          std::as_bytes(new_face_velocities),
      }
  );
  if (!disjoint.ok()) {
    return disjoint;
  }

  const auto finite_face = [](double value) {
    return std::isfinite(value);
  };
  if (!std::ranges::all_of(old_face_velocities, finite_face)
      || !std::ranges::all_of(new_face_velocities, finite_face)) {
    return absl::InvalidArgumentError("old and new face velocities must be finite");
  }

  // Stage the four particle outputs in double slices, independent of Vec2 and Mat2 layout.
  const std::size_t vector_value_count = kVelocityComponentCount * particle_count;
  const std::size_t matrix_value_count = kMatrixEntryCount * particle_count;
  auto staged_pic_velocities = gather_scratch.first(vector_value_count);
  auto staged_grid_velocity_increments = gather_scratch.subspan(vector_value_count, vector_value_count);
  auto staged_velocity_fit_slopes = gather_scratch.subspan(2 * vector_value_count, matrix_value_count);
  auto staged_velocity_gradients = gather_scratch.last(matrix_value_count);
  for (std::size_t particle_index = 0; particle_index < particle_count; ++particle_index) {
    const Vec2 particle_world_position = particle_world_positions[particle_index];
    if (!IsFinite(particle_world_position)) {
      return absl::InvalidArgumentError("particle position must be finite");
    }
    for (std::size_t velocity_component_index = 0; velocity_component_index < kVelocityComponentCount;
         ++velocity_component_index) {
      const auto stencil = BuildFaceStencil(mac_grid, particle_world_position, velocity_component_index);
      if (!stencil.ok()) {
        return stencil.status();
      }
      const auto component_result =
          GatherComponent(*stencil, particle_world_position, old_face_velocities, new_face_velocities);
      if (!component_result.ok()) {
        return component_result.status();
      }
      const std::size_t vector_component_offset = kVelocityComponentCount * particle_index + velocity_component_index;
      const std::size_t matrix_row_offset = kVelocityComponentCount * vector_component_offset;
      staged_pic_velocities[vector_component_offset] = component_result->pic_velocity;
      staged_grid_velocity_increments[vector_component_offset] = component_result->grid_velocity_increment;
      staged_velocity_fit_slopes[matrix_row_offset] = component_result->velocity_fit_slope_row.x;
      staged_velocity_fit_slopes[matrix_row_offset + 1] = component_result->velocity_fit_slope_row.y;
      staged_velocity_gradients[matrix_row_offset] = component_result->velocity_gradient_row.x;
      staged_velocity_gradients[matrix_row_offset + 1] = component_result->velocity_gradient_row.y;
    }
  }

  for (std::size_t particle_index = 0; particle_index < particle_count; ++particle_index) {
    const std::size_t vector_offset = kVelocityComponentCount * particle_index;
    const std::size_t matrix_offset = kMatrixEntryCount * particle_index;
    pic_velocities[particle_index] = {staged_pic_velocities[vector_offset], staged_pic_velocities[vector_offset + 1]};
    grid_velocity_increments[particle_index] = {
        staged_grid_velocity_increments[vector_offset],
        staged_grid_velocity_increments[vector_offset + 1],
    };
    velocity_fit_slopes[particle_index] = {
        .xx = staged_velocity_fit_slopes[matrix_offset],
        .xy = staged_velocity_fit_slopes[matrix_offset + 1],
        .yx = staged_velocity_fit_slopes[matrix_offset + 2],
        .yy = staged_velocity_fit_slopes[matrix_offset + 3],
    };
    velocity_gradients[particle_index] = {
        .xx = staged_velocity_gradients[matrix_offset],
        .xy = staged_velocity_gradients[matrix_offset + 1],
        .yx = staged_velocity_gradients[matrix_offset + 2],
        .yy = staged_velocity_gradients[matrix_offset + 3],
    };
  }
  return absl::OkStatus();
}

absl::StatusOr<AffineFit2D> FitAffine2D(
    Vec2 position,
    std::span<const Vec2> sample_positions,
    std::span<const double> sample_weights,
    std::span<const double> sample_values
) {
  if (!IsFinite(position)) {
    return absl::InvalidArgumentError("fit position must be finite");
  }
  if (sample_positions.empty()) {
    return absl::InvalidArgumentError("affine fit needs samples");
  }
  const auto extents = CheckExtents(sample_positions.size(), {sample_weights.size(), sample_values.size()});
  if (!extents.ok()) {
    return extents;
  }
  // The reference compares the weight sum to one with absolute and relative tolerances of 1e-12.
  constexpr double partition_tolerance = 2e-12;
  double total_weight = 0.0;
  for (std::size_t sample_index = 0; sample_index < sample_positions.size(); ++sample_index) {
    if (!IsFinite(sample_positions[sample_index]) || !std::isfinite(sample_values[sample_index])) {
      return absl::InvalidArgumentError("sample positions and values must be finite");
    }
    const double weight = sample_weights[sample_index];
    const bool valid_weight = std::isfinite(weight) && weight >= 0.0;
    if (!valid_weight) {
      return absl::InvalidArgumentError("weights must be finite and nonnegative");
    }
    total_weight += weight;
    // Nonnegative weights cannot restore a partition once the sum exceeds its upper bound.
    if (total_weight - 1.0 > partition_tolerance) {
      return absl::InvalidArgumentError("weights must form a partition of unity");
    }
  }
  if (1.0 - total_weight > partition_tolerance) {
    return absl::InvalidArgumentError("weights must form a partition of unity");
  }
  const auto mean_offset = WeightedOffsetMean(position, sample_positions, sample_weights);
  if (!mean_offset.ok()) {
    return mean_offset.status();
  }

  double interpolated_value = 0.0;
  double moment_xx = 0.0;
  double moment_xy = 0.0;
  double moment_yy = 0.0;
  Vec2 offset_value_moment{};
  for (std::size_t sample_index = 0; sample_index < sample_positions.size(); ++sample_index) {
    const Vec2 centered_offset = (sample_positions[sample_index] - position) - *mean_offset;
    const Vec2 weighted_centered_offset = centered_offset * sample_weights[sample_index];
    const double weighted_value = sample_weights[sample_index] * sample_values[sample_index];
    interpolated_value += weighted_value;
    moment_xx += weighted_centered_offset.x * centered_offset.x;
    moment_xy += weighted_centered_offset.x * centered_offset.y;
    moment_yy += weighted_centered_offset.y * centered_offset.y;
    offset_value_moment = offset_value_moment + centered_offset * weighted_value;
  }
  const bool finite_moment = std::isfinite(moment_xx) && std::isfinite(moment_xy) && std::isfinite(moment_yy);
  if (!finite_moment || !IsFinite(offset_value_moment) || !std::isfinite(interpolated_value)) {
    return absl::OutOfRangeError("affine moment, right-hand side or gathered value is nonfinite");
  }
  const auto affine_row = SolveAffineMoment(moment_xx, moment_xy, moment_yy, offset_value_moment);
  if (!affine_row.ok()) {
    return affine_row.status();
  }
  return AffineFit2D{
      .interpolated_value = interpolated_value,
      .spatial_slopes = *affine_row,
  };
}

absl::StatusOr<Vec2>
VelocityGradient2D(std::span<const double> face_velocity_samples, std::span<const Vec2> basis_weight_gradients) {
  if (face_velocity_samples.empty()) {
    return absl::InvalidArgumentError("velocity gradient needs samples");
  }
  const auto extent = ValidateExtent(basis_weight_gradients.size(), face_velocity_samples.size());
  if (!extent.ok()) {
    return extent;
  }
  Vec2 velocity_gradient{};
  for (std::size_t sample_index = 0; sample_index < face_velocity_samples.size(); ++sample_index) {
    if (!std::isfinite(face_velocity_samples[sample_index]) || !IsFinite(basis_weight_gradients[sample_index])) {
      return absl::InvalidArgumentError("velocity samples and weight gradients must be finite");
    }
    velocity_gradient = velocity_gradient + basis_weight_gradients[sample_index] * face_velocity_samples[sample_index];
  }
  if (!IsFinite(velocity_gradient)) {
    return absl::OutOfRangeError("velocity gradient is nonfinite");
  }
  return velocity_gradient;
}

absl::StatusOr<VelocityPair> BlendHistory2D(
    Vec2 pic_velocity,
    Vec2 history_velocity_residual,
    double energy_control,
    Phase phase,
    double separation_factor
) {
  if (!IsFinite(pic_velocity) || !IsFinite(history_velocity_residual)) {
    return absl::InvalidArgumentError("PIC and history residual must be finite");
  }
  const bool valid_energy = energy_control >= 0.0 && energy_control <= 1.0;
  if (!valid_energy) {
    return absl::InvalidArgumentError("energy must be in [0, 1]");
  }
  const bool valid_separation = separation_factor >= 0.0 && separation_factor <= 1.0;
  if (!valid_separation) {
    return absl::InvalidArgumentError("separation must be in [0, 1]");
  }

  double history_retention_factor = 0.99 * energy_control;
  double advection_separation_factor = 0.0;
  switch (phase) {
    case Phase::kGas:
      history_retention_factor = 0.0;
      break;
    case Phase::kElastic:
      break;
    case Phase::kLiquid:
    case Phase::kGranular:
    case Phase::kSnow:
      advection_separation_factor = energy_control * separation_factor;
      break;
    default:
      return absl::InvalidArgumentError("unknown history blend phase");
  }

  const VelocityPair velocities{
      .stored = pic_velocity + history_retention_factor * history_velocity_residual,
      .advecting = pic_velocity + (advection_separation_factor * history_retention_factor) * history_velocity_residual,
  };
  if (!IsFinite(velocities.stored) || !IsFinite(velocities.advecting)) {
    return absl::OutOfRangeError("history-blended velocity is nonfinite");
  }
  return velocities;
}

}  // namespace fluidity
