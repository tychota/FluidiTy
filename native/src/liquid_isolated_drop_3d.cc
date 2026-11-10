#include "liquid_isolated_drop_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "absl/status/status.h"
#include "liquid_substep_3d_internal.h"

namespace fluidity::liquid_step_internal {
namespace {

using Matrix = std::array<std::array<double, 3>, 3>;

struct PrincipalAxes {
  Mat3 frame;
  std::array<double, 3> eigenvalues;
};

Vec3 Difference(Vec3 end, Vec3 start) {
  return {end.x - start.x, end.y - start.y, end.z - start.z};
}

double QuadraticForm(Mat3 matrix, Vec3 offset) {
  return matrix.x_row.x * offset.x * offset.x + matrix.y_row.y * offset.y * offset.y
      + matrix.z_row.z * offset.z * offset.z
      + 2
      * (matrix.x_row.y * offset.x * offset.y + matrix.x_row.z * offset.x * offset.z
         + matrix.y_row.z * offset.y * offset.z);
}

absl::Status CheckConnected(std::span<const LiquidParticleState3D> particles, double neighbor_radius) {
  std::array<bool, kMaxIsolatedDropParticles3D> visited{};
  std::array<std::size_t, kMaxIsolatedDropParticles3D> queue{};
  visited[0] = true;
  std::size_t reachable = 1;
  for (std::size_t ordinal = 0; ordinal < reachable; ++ordinal) {
    const std::size_t current = queue[ordinal];
    for (std::size_t candidate = 0; candidate < particles.size(); ++candidate) {
      if (candidate == current) {
        continue;
      }
      const Vec3 offset = Difference(particles[candidate].world_position, particles[current].world_position);
      const double distance = std::hypot(offset.x, offset.y, offset.z);
      if (!std::isfinite(distance)) {
        return absl::OutOfRangeError("nonfinite isolated drop particle separation");
      }
      if (distance == 0) {
        return absl::InvalidArgumentError("duplicate isolated drop particle positions");
      }
      if (!visited[candidate] && distance <= neighbor_radius) {
        visited[candidate] = true;
        queue[reachable++] = candidate;
      }
    }
  }
  if (reachable != particles.size()) {
    return absl::FailedPreconditionError("isolated drop particle graph is disconnected");
  }
  return absl::OkStatus();
}

// Cyclic symmetric Jacobi rotations retain the principal axes in frame columns.
// Verify the eigenpairs against the original inverse shape before using them.
absl::StatusOr<PrincipalAxes> DiagonalizeInverseShape(Mat3 inverse_shape) {
  const auto& [x, y, z] = inverse_shape;
  const Matrix shape{{{x.x, x.y, x.z}, {y.x, y.y, y.z}, {z.x, z.y, z.z}}};
  Matrix diagonal = shape;
  Matrix frame{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  for (int sweep = 0; sweep < 12; ++sweep) {
    for (std::size_t p = 0; p < 2; ++p) {
      for (std::size_t q = p + 1; q < 3; ++q) {
        const double cross = diagonal[p][q];
        if (cross == 0) {
          continue;
        }
        const double tau = (diagonal[q][q] - diagonal[p][p]) / (2 * cross);
        const double tangent = std::copysign(1.0, tau) / (std::abs(tau) + std::hypot(1.0, tau));
        const double cosine = 1 / std::hypot(1.0, tangent);
        const double sine = tangent * cosine;
        diagonal[p][p] -= tangent * cross;
        diagonal[q][q] += tangent * cross;
        diagonal[p][q] = diagonal[q][p] = 0;
        for (std::size_t row = 0; row < 3; ++row) {
          if (row != p && row != q) {
            const double old_p = diagonal[row][p];
            const double old_q = diagonal[row][q];
            diagonal[row][p] = diagonal[p][row] = cosine * old_p - sine * old_q;
            diagonal[row][q] = diagonal[q][row] = sine * old_p + cosine * old_q;
          }
          const double old_p = frame[row][p];
          const double old_q = frame[row][q];
          frame[row][p] = cosine * old_p - sine * old_q;
          frame[row][q] = sine * old_p + cosine * old_q;
        }
      }
    }
  }

  const std::array eigenvalues{diagonal[0][0], diagonal[1][1], diagonal[2][2]};
  if (!std::ranges::all_of(eigenvalues, [](double value) {
        return std::isfinite(value);
      })) {
    return absl::OutOfRangeError("nonfinite isolated drop eigenvalues");
  }
  const double tolerance = 32 * std::numeric_limits<double>::epsilon() * std::ranges::max(eigenvalues);
  if (std::ranges::min(eigenvalues) <= std::max(0.0, tolerance)) {
    return absl::FailedPreconditionError("degenerate isolated drop principal axes");
  }
  for (std::size_t axis = 0; axis < 3; ++axis) {
    for (std::size_t row = 0; row < 3; ++row) {
      double product = 0;
      for (std::size_t column = 0; column < 3; ++column) {
        product += shape[row][column] * frame[column][axis];
      }
      const double residual = product - eigenvalues[axis] * frame[row][axis];
      if (!std::isfinite(residual)) {
        return absl::OutOfRangeError("nonfinite isolated drop eigenpair residual");
      }
      if (std::abs(residual) > tolerance) {
        return absl::FailedPreconditionError("isolated drop eigenpair residual exceeds tolerance");
      }
    }
  }
  return PrincipalAxes{
      .frame =
          {
              {frame[0][0], frame[0][1], frame[0][2]},
              {frame[1][0], frame[1][1], frame[1][2]},
              {frame[2][0], frame[2][1], frame[2][2]},
          },
      .eigenvalues = eigenvalues,
  };
}

}  // namespace

absl::StatusOr<PreparedIsolatedDrop3D> PrepareIsolatedDrop3D(
    std::span<const LiquidParticleState3D> particles,
    double cell_width,
    double physical_radius,
    double seed_covariance_scale
) {
  if (particles.size() < 4 || particles.size() > kMaxIsolatedDropParticles3D) {
    return absl::InvalidArgumentError("isolated drop particle count is outside the supported range");
  }
  const double neighbor_radius = 1.04 * cell_width;
  if (!std::isfinite(neighbor_radius) || cell_width <= 0 || !std::isfinite(physical_radius) || physical_radius <= 0
      || !std::isfinite(seed_covariance_scale) || seed_covariance_scale <= 0) {
    return absl::InvalidArgumentError("invalid isolated drop fit parameters");
  }
  double total_mass = 0;
  Vec3 center{};
  for (const auto& particle : particles) {
    if (!Finite(particle.world_position) || !std::isfinite(particle.mass) || particle.mass <= 0) {
      return absl::InvalidArgumentError("invalid isolated drop particle position or mass");
    }
    center.x += particle.mass * particle.world_position.x;
    center.y += particle.mass * particle.world_position.y;
    center.z += particle.mass * particle.world_position.z;
    total_mass += particle.mass;
  }
  if (!std::isfinite(total_mass)) {
    return absl::OutOfRangeError("nonfinite isolated drop total mass");
  }
  center.x /= total_mass;
  center.y /= total_mass;
  center.z /= total_mass;
  if (!Finite(center)) {
    return absl::OutOfRangeError("nonfinite isolated drop centroid");
  }
  if (const auto status = CheckConnected(particles, neighbor_radius); !status.ok()) {
    return status;
  }

  // Accumulate in trial order, retaining the calibrated seed scale as the drop evolves.
  double moment_xx = 0;
  double moment_yy = 0;
  double moment_zz = 0;
  double moment_xy = 0;
  double moment_xz = 0;
  double moment_yz = 0;
  for (const auto& particle : particles) {
    const Vec3 offset = Difference(particle.world_position, center);
    moment_xx += particle.mass * offset.x * offset.x;
    moment_yy += particle.mass * offset.y * offset.y;
    moment_zz += particle.mass * offset.z * offset.z;
    moment_xy += particle.mass * offset.x * offset.y;
    moment_xz += particle.mass * offset.x * offset.z;
    moment_yz += particle.mass * offset.y * offset.z;
  }
  const double covariance_scale = seed_covariance_scale / total_mass;
  moment_xx *= covariance_scale;
  moment_yy *= covariance_scale;
  moment_zz *= covariance_scale;
  moment_xy *= covariance_scale;
  moment_xz *= covariance_scale;
  moment_yz *= covariance_scale;
  const double cofactor_xx = moment_yy * moment_zz - moment_yz * moment_yz;
  const double cofactor_yy = moment_xx * moment_zz - moment_xz * moment_xz;
  const double cofactor_zz = moment_xx * moment_yy - moment_xy * moment_xy;
  const double cofactor_xy = moment_xz * moment_yz - moment_xy * moment_zz;
  const double cofactor_xz = moment_xy * moment_yz - moment_xz * moment_yy;
  const double cofactor_yz = moment_xy * moment_xz - moment_xx * moment_yz;
  const double determinant = moment_xx * cofactor_xx + moment_xy * cofactor_xy + moment_xz * cofactor_xz;
  if (!std::isfinite(determinant)) {
    return absl::OutOfRangeError("nonfinite isolated drop covariance determinant");
  }
  if (moment_xx <= 0 || cofactor_zz <= 0 || determinant <= 0) {
    return absl::FailedPreconditionError("isolated drop covariance is not positive definite");
  }
  const Mat3 inverse_shape{
      {cofactor_xx / determinant, cofactor_xy / determinant, cofactor_xz / determinant},
      {cofactor_xy / determinant, cofactor_yy / determinant, cofactor_yz / determinant},
      {cofactor_xz / determinant, cofactor_yz / determinant, cofactor_zz / determinant},
  };
  if (!Finite(inverse_shape.x_row) || !Finite(inverse_shape.y_row) || !Finite(inverse_shape.z_row)) {
    return absl::OutOfRangeError("nonfinite isolated drop inverse shape");
  }
  const auto axes = DiagonalizeInverseShape(inverse_shape);
  if (!axes.ok()) {
    return axes.status();
  }

  PreparedIsolatedDrop3D fit;
  fit.center_ = center;
  fit.inverse_shape_ = inverse_shape;
  fit.principal_frame_ = axes->frame;
  fit.physical_radius_ = physical_radius;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    fit.semiaxes_[axis] = 1 / std::sqrt(axes->eigenvalues[axis]);
  }
  return fit;
}

absl::StatusOr<double> PreparedIsolatedDrop3D::QuadraticGridLevelSet(Vec3 world_position) const {
  const double level_set =
      0.5 * physical_radius_ * (QuadraticForm(inverse_shape_, Difference(world_position, center_)) - 1);
  if (!std::isfinite(level_set)) {
    return absl::OutOfRangeError("nonfinite isolated drop grid level set");
  }
  return level_set;
}

absl::StatusOr<double> PreparedIsolatedDrop3D::ParticleSignedDistance(Vec3 world_position) const {
  const Vec3 offset = Difference(world_position, center_);
  const auto& [x, y, z] = principal_frame_;
  const double scale = std::ranges::max(semiaxes_);
  const std::array point{
      std::abs(x.x * offset.x + y.x * offset.y + z.x * offset.z) / scale,
      std::abs(x.y * offset.x + y.y * offset.y + z.y * offset.z) / scale,
      std::abs(x.z * offset.x + y.z * offset.y + z.z * offset.z) / scale,
  };
  if (!std::ranges::all_of(point, [](double value) {
        return std::isfinite(value);
      })) {
    return absl::OutOfRangeError("nonfinite isolated drop particle position");
  }
  const std::array axes{semiaxes_[0] / scale, semiaxes_[1] / scale, semiaxes_[2] / scale};
  const std::array squares{axes[0] * axes[0], axes[1] * axes[1], axes[2] * axes[2]};
  const std::size_t shortest = static_cast<std::size_t>(std::ranges::min_element(squares) - squares.begin());
  const double smallest_square = squares[shortest];
  const std::array shifts{squares[0] - smallest_square, squares[1] - smallest_square, squares[2] - smallest_square};
  const std::array numerators{axes[0] * point[0], axes[1] * point[1], axes[2] * point[2]};
  const double normalized_radius = std::hypot(point[0] / axes[0], point[1] / axes[1], point[2] / axes[2]);
  if (!std::isfinite(normalized_radius)) {
    return absl::OutOfRangeError("nonfinite isolated drop normalized particle position");
  }
  const bool inside = normalized_radius < 1;

  // Shift the closest-point multiplier by the shortest squared semiaxis.
  const auto surface_coordinates = [&](double multiplier) {
    std::array<double, 3> surface{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double denominator = shifts[axis] + multiplier;
      if (denominator > 0) {
        surface[axis] = numerators[axis] / denominator;
      }
    }
    return surface;
  };
  double lower = smallest_square;
  double upper = smallest_square;
  if (inside) {
    lower = std::max({0.0, numerators[0] - shifts[0], numerators[1] - shifts[1], numerators[2] - shifts[2]});
  } else {
    upper += 2 * std::hypot(numerators[0], numerators[1], numerators[2]);
  }
  if (!std::isfinite(upper)) {
    return absl::OutOfRangeError("nonfinite isolated drop closest-point bracket");
  }
  double multiplier = lower;
  const auto lower_surface = surface_coordinates(lower);
  if (std::hypot(lower_surface[0], lower_surface[1], lower_surface[2]) > 1) {
    // Near the medial axis the shifted multiplier can be tiny. Terminate at
    // adjacent representable values instead of using an absolute tolerance.
    for (double middle = lower + 0.5 * (upper - lower); middle > lower && middle < upper;
         middle = lower + 0.5 * (upper - lower)) {
      const auto surface = surface_coordinates(middle);
      if (std::hypot(surface[0], surface[1], surface[2]) > 1) {
        lower = middle;
      } else {
        upper = middle;
      }
    }
    multiplier = lower + 0.5 * (upper - lower);
  }
  auto surface = surface_coordinates(multiplier);
  if (multiplier == 0) {
    // A medial-axis point can reach the surface along a shortest principal axis.
    const double radius = std::hypot(surface[0], surface[1], surface[2]);
    surface[shortest] = std::sqrt(std::max(0.0, 1 - radius * radius));
  }
  const double distance = scale
      * std::hypot(axes[0] * surface[0] - point[0], axes[1] * surface[1] - point[1], axes[2] * surface[2] - point[2]);
  if (!std::isfinite(distance)) {
    return absl::OutOfRangeError("nonfinite isolated drop particle distance");
  }
  return inside ? -distance : distance;
}

absl::StatusOr<double> PreparedIsolatedDrop3D::AirLinkCurvature(Vec3 wet_world_center, Vec3 air_world_center) const {
  const auto wet = QuadraticGridLevelSet(wet_world_center);
  const auto air = QuadraticGridLevelSet(air_world_center);
  if (!wet.ok()) {
    return wet.status();
  }
  if (!air.ok()) {
    return air.status();
  }
  if (*wet >= 0 || *air < 0) {
    return absl::FailedPreconditionError("isolated drop air link does not bracket the surface");
  }

  const Vec3 direction = Difference(air_world_center, wet_world_center);
  double lower = 0;
  double upper = 1;
  for (int iteration = 0; iteration < 36; ++iteration) {
    const double middle = lower + 0.5 * (upper - lower);
    const auto level = QuadraticGridLevelSet(Add(wet_world_center, Scale(direction, middle)));
    if (!level.ok()) {
      return level.status();
    }
    if (*level < 0) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  const Vec3 crossing = Add(wet_world_center, Scale(direction, lower + 0.5 * (upper - lower)));
  const Vec3 offset = Difference(crossing, center_);
  const auto& [x, y, z] = inverse_shape_;
  const Vec3 gradient{Dot(x, offset), Dot(y, offset), Dot(z, offset)};
  const double length = std::hypot(gradient.x, gradient.y, gradient.z);
  if (!std::isfinite(length) || length == 0) {
    return absl::OutOfRangeError("invalid isolated drop interface normal");
  }
  const Vec3 normal{gradient.x / length, gradient.y / length, gradient.z / length};
  const double curvature = (x.x + y.y + z.z - QuadraticForm(inverse_shape_, normal)) / length;
  if (!std::isfinite(curvature)) {
    return absl::OutOfRangeError("nonfinite isolated drop interface curvature");
  }
  return curvature;
}

}  // namespace fluidity::liquid_step_internal
