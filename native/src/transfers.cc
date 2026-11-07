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

constexpr std::size_t kComponentCount = 2;
constexpr std::size_t kStencilSize = std::tuple_size_v<decltype(Stencil2D::nodes)>;
constexpr std::size_t kDepositDoublesPerFace = 2;
constexpr std::size_t kMatrixEntryCount = kComponentCount * kComponentCount;
constexpr std::size_t kGatherDoublesPerParticle = 2 * kComponentCount + 2 * kMatrixEntryCount;

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
  std::array<std::size_t, kStencilSize> face_indices;
  std::array<Vec2, kStencilSize> samples;
};

// FaceCount has validated the lattice sizes. Check the signed nodes before flattening them,
// including zero-weight nodes at exact half-cell ties.
absl::StatusOr<FaceStencil> BuildFaceStencil(const Grid2D& grid, Vec2 position, std::size_t velocity_component) {
  const Vec2 relative_position = position - grid.origin;
  if (!IsFinite(relative_position)) {
    return absl::OutOfRangeError("position relative to grid origin is nonfinite");
  }
  const bool is_x_normal = velocity_component == 0;
  const Vec2 lattice_offset{
      .x = is_x_normal ? 0.0 : 0.5,
      .y = is_x_normal ? 0.5 : 0.0,
  };
  const auto basis = QuadraticStencil2D(relative_position, grid.h, lattice_offset);
  if (!basis.ok()) {
    return basis.status();
  }

  const std::int64_t face_column_count = static_cast<std::int64_t>(grid.nx) + (is_x_normal ? 1 : 0);
  const std::int64_t face_row_count = static_cast<std::int64_t>(grid.ny) + (is_x_normal ? 0 : 1);
  const std::size_t orientation_face_offset = is_x_normal ? 0 : (static_cast<std::size_t>(grid.nx) + 1) * grid.ny;
  FaceStencil stencil;
  stencil.basis = *basis;
  for (std::size_t stencil_slot = 0; stencil_slot < stencil.face_indices.size(); ++stencil_slot) {
    const Node2D node = basis->nodes[stencil_slot];
    const bool node_on_face_lattice =
        node.x >= 0 && node.y >= 0 && node.x < face_column_count && node.y < face_row_count;
    if (!node_on_face_lattice) {
      return absl::InvalidArgumentError("particle does not have complete MAC support");
    }
    const std::size_t face_index = orientation_face_offset
        + static_cast<std::size_t>(node.y) * static_cast<std::size_t>(face_column_count)
        + static_cast<std::size_t>(node.x);
    const auto geometry = FaceGeometry(grid, face_index);
    if (!geometry.ok()) {
      return geometry.status();
    }
    stencil.face_indices[stencil_slot] = face_index;
    stencil.samples[stencil_slot] = geometry->position;
  }
  return stencil;
}

// With a partition of unity, subtracting this mean offset makes the weighted first moment
// vanish. The same centering is used in the deposition velocity and the affine fit.
absl::StatusOr<Vec2> WeightedOffsetMean(Vec2 position, std::span<const Vec2> samples, std::span<const double> weights) {
  Vec2 mean{};
  for (std::size_t sample = 0; sample < samples.size(); ++sample) {
    const Vec2 offset = samples[sample] - position;
    mean = mean + offset * weights[sample];
  }
  if (!IsFinite(mean)) {
    return absl::OutOfRangeError("weighted offset mean is nonfinite");
  }
  return mean;
}

// Deposit one complete component stencil into batch scratch. Centering the affine term
// makes its weighted linear momentum zero while preserving its contribution to torque.
absl::Status DepositComponent(
    const FaceStencil& stencil,
    Vec2 position,
    double mass,
    double velocity,
    Vec2 affine_row,
    std::span<double> staged_mass,
    std::span<double> staged_momentum
) {
  const auto mean_offset = WeightedOffsetMean(position, stencil.samples, stencil.basis.weights);
  if (!mean_offset.ok()) {
    return mean_offset.status();
  }
  for (std::size_t stencil_slot = 0; stencil_slot < stencil.face_indices.size(); ++stencil_slot) {
    const std::size_t face_index = stencil.face_indices[stencil_slot];
    const Vec2 centered_offset = (stencil.samples[stencil_slot] - position) - *mean_offset;
    const double deposited_velocity = velocity + Dot(affine_row, centered_offset);
    const double mass_contribution = stencil.basis.weights[stencil_slot] * mass;
    staged_mass[face_index] += mass_contribution;
    staged_momentum[face_index] += mass_contribution * deposited_velocity;
    if (!std::isfinite(staged_mass[face_index]) || !std::isfinite(staged_momentum[face_index])) {
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

struct GatheredComponent {
  double pic;
  double increment;
  Vec2 affine_row;
  Vec2 gradient_row;
};

// Both face fields are validated before the batch loop. Old and new values use the same
// basis and accumulation order so a shared old/new field gives exactly zero increment.
absl::StatusOr<GatheredComponent> GatherComponent(
    const FaceStencil& stencil,
    Vec2 position,
    std::span<const double> old_faces,
    std::span<const double> new_faces
) {
  std::array<double, kStencilSize> new_samples;
  double old_value = 0.0;
  for (std::size_t sample = 0; sample < new_samples.size(); ++sample) {
    const std::size_t face = stencil.face_indices[sample];
    new_samples[sample] = new_faces[face];
    old_value += stencil.basis.weights[sample] * old_faces[face];
  }
  const auto fit = FitAffine2D(position, stencil.samples, stencil.basis.weights, new_samples);
  if (!fit.ok()) {
    return fit.status();
  }
  const auto gradient = VelocityGradient2D(new_samples, stencil.basis.gradients);
  if (!gradient.ok()) {
    return gradient.status();
  }
  const double increment = fit->value - old_value;
  if (!std::isfinite(increment)) {
    return absl::OutOfRangeError("grid velocity increment is nonfinite");
  }
  return GatheredComponent{
      .pic = fit->value,
      .increment = increment,
      .affine_row = fit->row,
      .gradient_row = *gradient,
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
  const auto particle_extents =
      CheckExtents(particle_count, {particles.masses.size(), particles.velocities.size(), particles.affine.size()});
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
          std::as_bytes(particles.affine),
      }
  );
  if (!disjoint.ok()) {
    return disjoint;
  }

  std::ranges::fill(scratch, 0.0);
  auto staged_mass = scratch.first(face_count);
  auto staged_momentum = scratch.last(face_count);
  for (std::size_t particle = 0; particle < particle_count; ++particle) {
    const Vec2 position = particles.positions[particle];
    const double mass = particles.masses[particle];
    const Vec2 velocity = particles.velocities[particle];
    const Mat2 affine = particles.affine[particle];
    if (!IsFinite(position) || !IsFinite(velocity) || !IsFinite(affine)) {
      return absl::InvalidArgumentError("particle position, velocity and affine fit must be finite");
    }
    const bool valid_mass = std::isfinite(mass) && mass >= 0.0;
    if (!valid_mass) {
      return absl::InvalidArgumentError("particle mass must be finite and nonnegative");
    }
    const std::array component_velocities{velocity.x, velocity.y};
    const std::array affine_rows{Vec2{affine.xx, affine.xy}, Vec2{affine.yx, affine.yy}};

    for (std::size_t component = 0; component < kComponentCount; ++component) {
      const auto stencil = BuildFaceStencil(grid, position, component);
      if (!stencil.ok()) {
        return stencil.status();
      }
      const auto deposited = DepositComponent(
          *stencil,
          position,
          mass,
          component_velocities[component],
          affine_rows[component],
          staged_mass,
          staged_momentum
      );
      if (!deposited.ok()) {
        return deposited;
      }
    }
  }

  // All particles have complete support and every accumulated face is finite.
  std::ranges::copy(staged_mass, face_mass.begin());
  std::ranges::copy(staged_momentum, face_momentum.begin());
  return absl::OkStatus();
}

absl::Status G2P(
    const Grid2D& grid,
    std::span<const Vec2> positions,
    std::span<const double> old_faces,
    std::span<const double> new_faces,
    std::span<Vec2> pic,
    std::span<Vec2> increments,
    std::span<Mat2> affine,
    std::span<Mat2> gradient,
    std::span<double> scratch
) {
  const auto face_count = FaceCount(grid);
  if (!face_count.ok()) {
    return face_count.status();
  }
  const std::size_t particle_count = positions.size();
  const auto scratch_count = G2PScratchDoubles(particle_count);
  if (!scratch_count.ok()) {
    return scratch_count.status();
  }
  const auto face_extents = CheckExtents(*face_count, {old_faces.size(), new_faces.size()});
  if (!face_extents.ok()) {
    return face_extents;
  }
  const auto particle_extents =
      CheckExtents(particle_count, {pic.size(), increments.size(), affine.size(), gradient.size()});
  if (!particle_extents.ok()) {
    return particle_extents;
  }
  const auto scratch_extent = ValidateExtent(scratch.size(), *scratch_count);
  if (!scratch_extent.ok()) {
    return scratch_extent;
  }
  const auto disjoint = CheckWritableViews(
      {
          std::as_bytes(pic),
          std::as_bytes(increments),
          std::as_bytes(affine),
          std::as_bytes(gradient),
          std::as_bytes(scratch),
      },
      {
          std::as_bytes(std::span(&grid, 1)),
          std::as_bytes(positions),
          std::as_bytes(old_faces),
          std::as_bytes(new_faces),
      }
  );
  if (!disjoint.ok()) {
    return disjoint;
  }

  const auto finite_face = [](double value) {
    return std::isfinite(value);
  };
  if (!std::ranges::all_of(old_faces, finite_face) || !std::ranges::all_of(new_faces, finite_face)) {
    return absl::InvalidArgumentError("old and new face velocities must be finite");
  }

  // Stage the four particle outputs in double slices, independent of Vec2 and Mat2 layout.
  const std::size_t vector_doubles = kComponentCount * particle_count;
  const std::size_t matrix_doubles = kMatrixEntryCount * particle_count;
  auto staged_pic = scratch.first(vector_doubles);
  auto staged_increments = scratch.subspan(vector_doubles, vector_doubles);
  auto staged_affine = scratch.subspan(2 * vector_doubles, matrix_doubles);
  auto staged_gradient = scratch.last(matrix_doubles);
  for (std::size_t particle = 0; particle < particle_count; ++particle) {
    const Vec2 position = positions[particle];
    if (!IsFinite(position)) {
      return absl::InvalidArgumentError("particle position must be finite");
    }
    for (std::size_t component = 0; component < kComponentCount; ++component) {
      const auto stencil = BuildFaceStencil(grid, position, component);
      if (!stencil.ok()) {
        return stencil.status();
      }
      const auto gathered = GatherComponent(*stencil, position, old_faces, new_faces);
      if (!gathered.ok()) {
        return gathered.status();
      }
      const std::size_t vector_entry = kComponentCount * particle + component;
      const std::size_t matrix_row_offset = kComponentCount * vector_entry;
      staged_pic[vector_entry] = gathered->pic;
      staged_increments[vector_entry] = gathered->increment;
      staged_affine[matrix_row_offset] = gathered->affine_row.x;
      staged_affine[matrix_row_offset + 1] = gathered->affine_row.y;
      staged_gradient[matrix_row_offset] = gathered->gradient_row.x;
      staged_gradient[matrix_row_offset + 1] = gathered->gradient_row.y;
    }
  }

  for (std::size_t particle = 0; particle < particle_count; ++particle) {
    const std::size_t vector_offset = kComponentCount * particle;
    const std::size_t matrix_offset = kMatrixEntryCount * particle;
    pic[particle] = {staged_pic[vector_offset], staged_pic[vector_offset + 1]};
    increments[particle] = {staged_increments[vector_offset], staged_increments[vector_offset + 1]};
    affine[particle] = {
        .xx = staged_affine[matrix_offset],
        .xy = staged_affine[matrix_offset + 1],
        .yx = staged_affine[matrix_offset + 2],
        .yy = staged_affine[matrix_offset + 3],
    };
    gradient[particle] = {
        .xx = staged_gradient[matrix_offset],
        .xy = staged_gradient[matrix_offset + 1],
        .yx = staged_gradient[matrix_offset + 2],
        .yy = staged_gradient[matrix_offset + 3],
    };
  }
  return absl::OkStatus();
}

absl::StatusOr<AffineFit2D> FitAffine2D(
    Vec2 position,
    std::span<const Vec2> samples,
    std::span<const double> weights,
    std::span<const double> values
) {
  if (!IsFinite(position)) {
    return absl::InvalidArgumentError("fit position must be finite");
  }
  if (samples.empty()) {
    return absl::InvalidArgumentError("affine fit needs samples");
  }
  const auto extents = CheckExtents(samples.size(), {weights.size(), values.size()});
  if (!extents.ok()) {
    return extents;
  }
  // The reference compares the weight sum to one with absolute and relative tolerances of 1e-12.
  constexpr double partition_tolerance = 2e-12;
  double total_weight = 0.0;
  for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
    if (!IsFinite(samples[sample_index]) || !std::isfinite(values[sample_index])) {
      return absl::InvalidArgumentError("sample positions and values must be finite");
    }
    const double weight = weights[sample_index];
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
  const auto mean_offset = WeightedOffsetMean(position, samples, weights);
  if (!mean_offset.ok()) {
    return mean_offset.status();
  }

  double value = 0.0;
  double moment_xx = 0.0;
  double moment_xy = 0.0;
  double moment_yy = 0.0;
  Vec2 offset_value_moment{};
  for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
    const Vec2 centered_offset = (samples[sample_index] - position) - *mean_offset;
    const Vec2 weighted_centered_offset = centered_offset * weights[sample_index];
    const double weighted_value = weights[sample_index] * values[sample_index];
    value += weighted_value;
    moment_xx += weighted_centered_offset.x * centered_offset.x;
    moment_xy += weighted_centered_offset.x * centered_offset.y;
    moment_yy += weighted_centered_offset.y * centered_offset.y;
    offset_value_moment = offset_value_moment + centered_offset * weighted_value;
  }
  const bool finite_moment = std::isfinite(moment_xx) && std::isfinite(moment_xy) && std::isfinite(moment_yy);
  if (!finite_moment || !IsFinite(offset_value_moment) || !std::isfinite(value)) {
    return absl::OutOfRangeError("affine moment, right-hand side or gathered value is nonfinite");
  }
  const auto affine_row = SolveAffineMoment(moment_xx, moment_xy, moment_yy, offset_value_moment);
  if (!affine_row.ok()) {
    return affine_row.status();
  }
  return AffineFit2D{
      .value = value,
      .row = *affine_row,
  };
}

absl::StatusOr<Vec2> VelocityGradient2D(std::span<const double> values, std::span<const Vec2> gradients) {
  if (values.empty()) {
    return absl::InvalidArgumentError("velocity gradient needs samples");
  }
  const auto extent = ValidateExtent(gradients.size(), values.size());
  if (!extent.ok()) {
    return extent;
  }
  Vec2 gradient{};
  for (std::size_t sample = 0; sample < values.size(); ++sample) {
    if (!std::isfinite(values[sample]) || !IsFinite(gradients[sample])) {
      return absl::InvalidArgumentError("velocity samples and weight gradients must be finite");
    }
    gradient = gradient + gradients[sample] * values[sample];
  }
  if (!IsFinite(gradient)) {
    return absl::OutOfRangeError("velocity gradient is nonfinite");
  }
  return gradient;
}

absl::StatusOr<VelocityPair> BlendHistory2D(Vec2 pic, Vec2 residual, double energy, Phase phase, double separation) {
  if (!IsFinite(pic) || !IsFinite(residual)) {
    return absl::InvalidArgumentError("PIC and history residual must be finite");
  }
  const bool valid_energy = energy >= 0.0 && energy <= 1.0;
  if (!valid_energy) {
    return absl::InvalidArgumentError("energy must be in [0, 1]");
  }
  const bool valid_separation = separation >= 0.0 && separation <= 1.0;
  if (!valid_separation) {
    return absl::InvalidArgumentError("separation must be in [0, 1]");
  }

  double alpha = 0.99 * energy;
  double beta = 0.0;
  switch (phase) {
    case Phase::kGas:
      alpha = 0.0;
      break;
    case Phase::kElastic:
      break;
    case Phase::kLiquid:
    case Phase::kGranular:
    case Phase::kSnow:
      beta = energy * separation;
      break;
    default:
      return absl::InvalidArgumentError("unknown history blend phase");
  }

  const VelocityPair velocities{
      .stored = pic + alpha * residual,
      .advecting = pic + (beta * alpha) * residual,
  };
  if (!IsFinite(velocities.stored) || !IsFinite(velocities.advecting)) {
    return absl::OutOfRangeError("history-blended velocity is nonfinite");
  }
  return velocities;
}

}  // namespace fluidity
