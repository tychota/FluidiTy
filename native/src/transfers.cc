#include "fluidity/host/transfers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>

#include "fluidity/host/basis.h"
#include "fluidity/host/views.h"

namespace fluidity {
namespace {

constexpr std::size_t kComponentCount = 2;
constexpr std::size_t kStencilSize = 9;
constexpr std::size_t kDepositDoublesPerFace = 2;
constexpr std::size_t kGatherDoublesPerParticle = 12;

bool IsFinite(Vec2 vector) {
  return std::isfinite(vector.x) && std::isfinite(vector.y);
}

bool IsFinite(Mat2 matrix) {
  return std::isfinite(matrix.xx) && std::isfinite(matrix.xy) && std::isfinite(matrix.yx) && std::isfinite(matrix.yy);
}

double Dot(Vec2 lhs, Vec2 rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y;
}

absl::Status CheckExtents(std::size_t expected, std::initializer_list<std::size_t> extents) {
  for (const std::size_t extent : extents) {
    const auto status = ValidateExtent(extent, expected);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

using ByteView = std::span<const std::byte>;

// Read-only views can share storage. Compare every writable view to all other storage it can affect.
absl::Status CheckWritableViews(std::initializer_list<ByteView> writable, std::initializer_list<ByteView> readable) {
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

struct FaceStencil {
  Stencil2D basis;
  std::array<std::size_t, kStencilSize> face_indices;
  std::array<Vec2, kStencilSize> samples;
};

// FaceCount has validated the lattice sizes. Check the signed nodes before flattening them,
// including zero-weight nodes at exact half-cell ties.
absl::StatusOr<FaceStencil> BuildFaceStencil(const Grid2D& grid, Vec2 position, std::size_t component) {
  const Vec2 relative_position = position - grid.origin;
  if (!IsFinite(relative_position)) {
    return absl::OutOfRangeError("position relative to grid origin is nonfinite");
  }
  const bool is_x_normal = component == 0;
  const Vec2 offset = is_x_normal ? Vec2{0.0, 0.5} : Vec2{0.5, 0.0};
  const auto basis = QuadraticStencil2D(relative_position, grid.h, offset);
  if (!basis.ok()) {
    return basis.status();
  }

  const std::int64_t columns = static_cast<std::int64_t>(grid.nx) + (is_x_normal ? 1 : 0);
  const std::int64_t rows = static_cast<std::int64_t>(grid.ny) + (is_x_normal ? 0 : 1);
  const std::size_t first_face = is_x_normal ? 0 : (static_cast<std::size_t>(grid.nx) + 1) * grid.ny;
  FaceStencil stencil;
  stencil.basis = *basis;
  for (std::size_t sample = 0; sample < stencil.face_indices.size(); ++sample) {
    const Node2D node = basis->nodes[sample];
    const bool has_face = node.x >= 0 && node.y >= 0 && node.x < columns && node.y < rows;
    if (!has_face) {
      return absl::InvalidArgumentError("particle does not have complete MAC support");
    }
    const std::size_t face = first_face + static_cast<std::size_t>(node.y) * static_cast<std::size_t>(columns)
        + static_cast<std::size_t>(node.x);
    const auto geometry = FaceGeometry(grid, face);
    if (!geometry.ok()) {
      return geometry.status();
    }
    stencil.face_indices[sample] = face;
    stencil.samples[sample] = geometry->position;
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

// Scale the full symmetric moment before forming its determinant. Comparing the determinant
// to 2*epsilon*lambda_max^2 applies the two-dimensional relative rank test without subtracting
// two nearly equal eigenvalues. The caller has checked all four accumulated inputs.
absl::StatusOr<Vec2> SolveAffineMoment(double xx, double xy, double yy, Vec2 rhs) {
  const double scale = std::max(xx, yy);
  if (scale == 0.0) {
    return absl::FailedPreconditionError("affine moment is singular");
  }
  xx /= scale;
  xy /= scale;
  yy /= scale;

  const double determinant = std::fma(xx, yy, -xy * xy);
  const double largest_eigenvalue = 0.5 * (xx + yy + std::hypot(xx - yy, 2.0 * xy));
  constexpr double rank_tolerance = 2.0 * std::numeric_limits<double>::epsilon();
  if (determinant <= rank_tolerance * largest_eigenvalue * largest_eigenvalue) {
    return absl::FailedPreconditionError("affine moment is numerically singular");
  }
  const Vec2 scaled_rhs{rhs.x / scale, rhs.y / scale};
  const Vec2 row{
      .x = (yy * scaled_rhs.x - xy * scaled_rhs.y) / determinant,
      .y = (xx * scaled_rhs.y - xy * scaled_rhs.x) / determinant,
  };
  if (!IsFinite(row)) {
    return absl::OutOfRangeError("affine row is nonfinite");
  }
  return row;
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
      const auto mean = WeightedOffsetMean(position, stencil->samples, stencil->basis.weights);
      if (!mean.ok()) {
        return mean.status();
      }
      for (std::size_t sample = 0; sample < stencil->face_indices.size(); ++sample) {
        const std::size_t face = stencil->face_indices[sample];
        const Vec2 centered = (stencil->samples[sample] - position) - *mean;
        const double sample_velocity = component_velocities[component] + Dot(affine_rows[component], centered);
        const double mass_contribution = stencil->basis.weights[sample] * mass;
        staged_mass[face] += mass_contribution;
        staged_momentum[face] += mass_contribution * sample_velocity;
        if (!std::isfinite(staged_mass[face]) || !std::isfinite(staged_momentum[face])) {
          return absl::OutOfRangeError("deposited mass or momentum is nonfinite");
        }
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
  const std::size_t vector_doubles = 2 * particle_count;
  const std::size_t matrix_doubles = 4 * particle_count;
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
      const std::size_t vector_entry = 2 * particle + component;
      const std::size_t matrix_row = 2 * vector_entry;
      staged_pic[vector_entry] = gathered->pic;
      staged_increments[vector_entry] = gathered->increment;
      staged_affine[matrix_row] = gathered->affine_row.x;
      staged_affine[matrix_row + 1] = gathered->affine_row.y;
      staged_gradient[matrix_row] = gathered->gradient_row.x;
      staged_gradient[matrix_row + 1] = gathered->gradient_row.y;
    }
  }

  for (std::size_t particle = 0; particle < particle_count; ++particle) {
    const std::size_t vector_offset = 2 * particle;
    const std::size_t matrix_offset = 4 * particle;
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
  for (std::size_t sample = 0; sample < samples.size(); ++sample) {
    if (!IsFinite(samples[sample]) || !std::isfinite(values[sample])) {
      return absl::InvalidArgumentError("sample positions and values must be finite");
    }
    const double weight = weights[sample];
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
  const auto mean = WeightedOffsetMean(position, samples, weights);
  if (!mean.ok()) {
    return mean.status();
  }

  double value = 0.0;
  double moment_xx = 0.0;
  double moment_xy = 0.0;
  double moment_yy = 0.0;
  Vec2 rhs{};
  for (std::size_t sample = 0; sample < samples.size(); ++sample) {
    const Vec2 centered = (samples[sample] - position) - *mean;
    const Vec2 weighted_centered = centered * weights[sample];
    const double weighted_value = weights[sample] * values[sample];
    value += weighted_value;
    moment_xx += weighted_centered.x * centered.x;
    moment_xy += weighted_centered.x * centered.y;
    moment_yy += weighted_centered.y * centered.y;
    rhs = rhs + centered * weighted_value;
  }
  const bool finite_moment = std::isfinite(moment_xx) && std::isfinite(moment_xy) && std::isfinite(moment_yy);
  if (!finite_moment || !IsFinite(rhs) || !std::isfinite(value)) {
    return absl::OutOfRangeError("affine moment, right-hand side or gathered value is nonfinite");
  }
  const auto row = SolveAffineMoment(moment_xx, moment_xy, moment_yy, rhs);
  if (!row.ok()) {
    return row.status();
  }
  return AffineFit2D{
      .value = value,
      .row = *row,
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
