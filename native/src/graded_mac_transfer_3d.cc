#include "fluidity/host/graded_mac_transfer_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/status.h"

namespace fluidity {
namespace {

using Components3D = std::array<double, 3>;
using Matrix3D = std::array<Components3D, 3>;

constexpr double kSupportRadiusInLeafWidths = 2;
constexpr double kMinimumRelativePivot = 1e-12;

Components3D Components(Vec3 value) {
  return {value.x, value.y, value.z};
}

Vec3 Cartesian(const Components3D& values) {
  return {values[0], values[1], values[2]};
}

bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double Dot(const Components3D& left, const Components3D& right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

struct ComponentSupport {
  double weight_sum = 0;
  Components3D mean_offset{};
  double velocity_scale = 0;
  double scaled_velocity_mean = 0;
  bool uses_low_mass = false;
  Matrix3D covariance{};
  Components3D velocity_covariance{};
};
using ParticleSupport = std::array<ComponentSupport, 3>;

// Evaluate the compact, area-weighted transfer on canonical global slots.
// All offsets use the particle's owning leaf width to condition the fit.
template <typename Visit>
absl::Status VisitSupport(
    std::span<const GradedMacFace3D> faces,
    std::span<const LiquidAdaptiveFaceState3D> carriers,
    const LiquidParticleState3D& particle,
    double leaf_width,
    bool gather,
    Visit visit
) {
  for (std::size_t slot = 0; slot < faces.size(); ++slot) {
    const auto& carrier = carriers[slot];
    if (!carrier.structurally_eligible || carrier.side_tag != particle.side_tag) {
      continue;
    }
    if (gather && carrier.validity == LiquidFaceValidity3D::kMissing) {
      continue;
    }
    const auto& face = faces[slot];
    const Components3D offset{
        (face.world_anchor.x - particle.world_position.x) / leaf_width,
        (face.world_anchor.y - particle.world_position.y) / leaf_width,
        (face.world_anchor.z - particle.world_position.z) / leaf_width,
    };
    if (std::ranges::any_of(offset, [](double value) {
          return std::abs(value) >= kSupportRadiusInLeafWidths;
        })) {
      continue;
    }
    const double radial_weight = 1 - Dot(offset, offset) / (kSupportRadiusInLeafWidths * kSupportRadiusInLeafWidths);
    if (radial_weight <= 0) {
      continue;
    }
    const double width_ratio = face.width_m / leaf_width;
    const double weight = (radial_weight * width_ratio) * (radial_weight * width_ratio);
    if (gather && (carrier.validity > LiquidFaceValidity3D::kLowMass || !std::isfinite(carrier.carrier_velocity))) {
      return absl::InvalidArgumentError("invalid adaptive transfer carrier");
    }
    const auto status = visit(slot, static_cast<std::size_t>(face.axis), offset, weight);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<ParticleSupport> BuildSupport(
    std::span<const GradedMacFace3D> faces,
    std::span<const LiquidAdaptiveFaceState3D> carriers,
    const LiquidParticleState3D& particle,
    double leaf_width,
    bool gather
) {
  ParticleSupport support{};
  auto status = VisitSupport(
      faces,
      carriers,
      particle,
      leaf_width,
      gather,
      [&](std::size_t slot, std::size_t axis, const Components3D& offset, double weight) {
        auto& component = support[axis];
        component.weight_sum += weight;
        for (std::size_t spatial = 0; spatial < 3; ++spatial) {
          component.mean_offset[spatial] += weight * offset[spatial];
        }
        if (gather) {
          component.velocity_scale = std::max(component.velocity_scale, std::abs(carriers[slot].carrier_velocity));
          component.uses_low_mass |= carriers[slot].validity == LiquidFaceValidity3D::kLowMass;
        }
        return absl::OkStatus();
      }
  );
  if (!status.ok()) {
    return status;
  }
  for (auto& component : support) {
    if (!std::isfinite(component.weight_sum)) {
      return absl::OutOfRangeError("adaptive transfer weight sum is nonfinite");
    }
    if (component.weight_sum <= 0) {
      return absl::FailedPreconditionError("adaptive transfer has no compatible face on a Cartesian component");
    }
    for (auto& offset : component.mean_offset) {
      offset /= component.weight_sum;
    }
  }
  status = VisitSupport(
      faces,
      carriers,
      particle,
      leaf_width,
      gather,
      [&](std::size_t slot, std::size_t axis, const Components3D& offset, double weight) {
        auto& component = support[axis];
        const double normalized_weight = weight / component.weight_sum;
        const Components3D centered{
            offset[0] - component.mean_offset[0],
            offset[1] - component.mean_offset[1],
            offset[2] - component.mean_offset[2],
        };
        const double scaled_velocity =
            component.velocity_scale == 0 ? 0 : carriers[slot].carrier_velocity / component.velocity_scale;
        component.scaled_velocity_mean += normalized_weight * scaled_velocity;
        for (std::size_t row = 0; row < 3; ++row) {
          component.velocity_covariance[row] += normalized_weight * centered[row] * scaled_velocity;
          for (std::size_t column = 0; column <= row; ++column) {
            component.covariance[row][column] += normalized_weight * centered[row] * centered[column];
          }
        }
        return absl::OkStatus();
      }
  );
  if (!status.ok()) {
    return status;
  }
  return support;
}

// Factor the centered weighted covariance in dimensionless units. Pivots
// smaller than the relative rank threshold cannot certify an affine fit.
bool SolveAffineSlope(const ComponentSupport& component, Components3D& slope) {
  Matrix3D factor{};
  const auto& covariance = component.covariance;
  const double trace = covariance[0][0] + covariance[1][1] + covariance[2][2];
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      double residual = covariance[row][column];
      for (std::size_t previous = 0; previous < column; ++previous) {
        residual -= factor[row][previous] * factor[column][previous];
      }
      if (column == row) {
        if (residual <= kMinimumRelativePivot * trace || !std::isfinite(residual)) {
          return false;
        }
        factor[row][row] = std::sqrt(residual);
      } else {
        factor[row][column] = residual / factor[column][column];
      }
    }
  }

  Components3D forward{};
  for (std::size_t row = 0; row < 3; ++row) {
    double residual = component.velocity_covariance[row];
    for (std::size_t previous = 0; previous < row; ++previous) {
      residual -= factor[row][previous] * forward[previous];
    }
    forward[row] = residual / factor[row][row];
  }
  for (std::size_t reverse = 3; reverse > 0; --reverse) {
    const std::size_t row = reverse - 1;
    double residual = forward[row];
    for (std::size_t column = row + 1; column < 3; ++column) {
      residual -= factor[column][row] * slope[column];
    }
    slope[row] = residual / factor[row][row];
  }
  return true;
}

bool FitAllComponents(const ParticleSupport& support, Matrix3D& slopes) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (support[axis].uses_low_mass || !SolveAffineSlope(support[axis], slopes[axis])) {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::StatusOr<GradedMacTransfer3D>
GradedMacTransfer3D::Create(const AdaptiveMacTopology3D& topology, Vec3 world_origin) {
  if (!Finite(world_origin)) {
    return absl::InvalidArgumentError("adaptive transfer world origin must be finite");
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  const std::size_t face_count = topology.interior_faces().size() + ownership->exterior_faces.size();
  if (face_count > kMaximumAdaptiveLiquidFaces3D) {
    return absl::ResourceExhaustedError("adaptive transfer face count exceeds the state limit");
  }
  GradedMacTransfer3D transfer;
  transfer.leaves_.reserve(topology.cells().size());
  transfer.faces_.resize(face_count);
  const double finest_width = topology.finest_cell_width();
  const auto origin = Components(world_origin);
  for (const auto& leaf : topology.cells()) {
    const auto corner = leaf.key.lower_corner;
    const std::array lower_indices{corner.x, corner.y, corner.z};
    Components3D lower{};
    Components3D upper{};
    const double lattice_width = std::ldexp(1.0, leaf.key.width_exponent);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      lower[axis] = std::fma(static_cast<double>(lower_indices[axis]), finest_width, origin[axis]);
      upper[axis] = std::fma(static_cast<double>(lower_indices[axis]) + lattice_width, finest_width, origin[axis]);
      if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) || upper[axis] <= lower[axis]) {
        return absl::OutOfRangeError("adaptive transfer world leaf is not representable");
      }
    }
    transfer.leaves_.push_back({.lower = Cartesian(lower), .upper = Cartesian(upper), .width_m = leaf.cell_width});
  }

  const auto stage_face = [&](FaceIndex3D index, std::uint8_t exponent, std::size_t slot) {
    const auto lattice_index = index.lattice_index;
    const std::array lower{lattice_index.x, lattice_index.y, lattice_index.z};
    const auto normal = static_cast<std::size_t>(index.axis);
    const double lattice_width = std::ldexp(1.0, exponent);
    Components3D anchor{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double half_width = axis == normal ? 0 : lattice_width / 2;
      anchor[axis] = std::fma(static_cast<double>(lower[axis]) + half_width, finest_width, origin[axis]);
      if (axis != normal) {
        const double low = std::fma(static_cast<double>(lower[axis]), finest_width, origin[axis]);
        const double high = std::fma(static_cast<double>(lower[axis]) + lattice_width, finest_width, origin[axis]);
        if (!(anchor[axis] > low && anchor[axis] < high)) {
          return false;
        }
      }
    }
    transfer.faces_[slot] = {
        .axis = index.axis,
        .world_anchor = Cartesian(anchor),
        .width_m = std::ldexp(finest_width, exponent),
    };
    return Finite(transfer.faces_[slot].world_anchor);
  };
  for (std::size_t face = 0; face < topology.interior_faces().size(); ++face) {
    const auto& geometry = topology.interior_faces()[face];
    if (!stage_face(geometry.face_index, geometry.width_exponent, ownership->interior_face_slots[face])) {
      return absl::OutOfRangeError("adaptive transfer interior anchor is not representable");
    }
  }
  for (const auto& geometry : ownership->exterior_faces) {
    if (!stage_face(geometry.face_index, geometry.width_exponent, geometry.face_slot)) {
      return absl::OutOfRangeError("adaptive transfer exterior anchor is not representable");
    }
  }
  return transfer;
}

absl::StatusOr<double> GradedMacTransfer3D::ParticleSupportWidth(Vec3 world_position) const {
  if (!Finite(world_position)) {
    return absl::InvalidArgumentError("adaptive transfer particle position must be finite");
  }
  for (const auto& leaf : leaves_) {
    if (world_position.x >= leaf.lower.x && world_position.x < leaf.upper.x && world_position.y >= leaf.lower.y
        && world_position.y < leaf.upper.y && world_position.z >= leaf.lower.z && world_position.z < leaf.upper.z) {
      return leaf.width_m;
    }
  }
  return absl::FailedPreconditionError("adaptive transfer particle has no containing leaf");
}

absl::StatusOr<std::vector<MacFaceDeposit3D>> GradedMacTransfer3D::Deposit(
    std::span<const LiquidParticleState3D> particles,
    std::span<const LiquidAdaptiveFaceState3D> carrier_faces
) const {
  if (carrier_faces.size() != faces_.size()) {
    return absl::InvalidArgumentError("adaptive transfer carrier extent differs from the canonical face count");
  }
  std::vector<MacFaceDeposit3D> deposits(faces_.size());
  for (const auto& particle : particles) {
    if (!std::isfinite(particle.mass) || particle.mass <= 0 || !Finite(particle.world_velocity)
        || (particle.affine_enabled
            && (!Finite(particle.velocity_slopes.x_row) || !Finite(particle.velocity_slopes.y_row)
                || !Finite(particle.velocity_slopes.z_row)))) {
      return absl::InvalidArgumentError("invalid adaptive transfer deposition particle");
    }
    const auto width = ParticleSupportWidth(particle.world_position);
    if (!width.ok()) {
      return width.status();
    }
    const auto support = BuildSupport(faces_, carrier_faces, particle, *width, false);
    if (!support.ok()) {
      return support.status();
    }
    Matrix3D unused_slopes{};
    const bool use_affine = particle.affine_enabled && FitAllComponents(*support, unused_slopes);
    const auto velocity = Components(particle.world_velocity);
    const Matrix3D slopes{
        Components(particle.velocity_slopes.x_row),
        Components(particle.velocity_slopes.y_row),
        Components(particle.velocity_slopes.z_row),
    };
    const auto status = VisitSupport(
        faces_,
        carrier_faces,
        particle,
        *width,
        false,
        [&](std::size_t slot, std::size_t axis, const Components3D& offset, double weight) {
          const auto& component = (*support)[axis];
          const double mass_contribution = particle.mass * (weight / component.weight_sum);
          double face_velocity = velocity[axis];
          if (use_affine) {
            const Components3D centered_offset_m{
                (offset[0] - component.mean_offset[0]) * *width,
                (offset[1] - component.mean_offset[1]) * *width,
                (offset[2] - component.mean_offset[2]) * *width,
            };
            face_velocity += Dot(slopes[axis], centered_offset_m);
          }
          const double momentum_contribution = mass_contribution * face_velocity;
          if (mass_contribution == 0 || (face_velocity != 0 && momentum_contribution == 0)) {
            return absl::OutOfRangeError("adaptive transfer contribution underflows");
          }
          auto& deposit = deposits[slot];
          deposit.deposited_mass += mass_contribution;
          deposit.deposited_pure_momentum += momentum_contribution;
          if (!std::isfinite(deposit.deposited_mass) || !std::isfinite(deposit.deposited_pure_momentum)) {
            return absl::OutOfRangeError("adaptive transfer deposition is nonfinite");
          }
          return absl::OkStatus();
        }
    );
    if (!status.ok()) {
      return status;
    }
  }
  return deposits;
}

absl::StatusOr<GradedMacParticleGather3D> GradedMacTransfer3D::Gather(
    const LiquidParticleState3D& particle,
    std::span<const LiquidAdaptiveFaceState3D> carrier_faces
) const {
  if (carrier_faces.size() != faces_.size()) {
    return absl::InvalidArgumentError("adaptive transfer carrier extent differs from the canonical face count");
  }
  const auto width = ParticleSupportWidth(particle.world_position);
  if (!width.ok()) {
    return width.status();
  }
  const auto support = BuildSupport(faces_, carrier_faces, particle, *width, true);
  if (!support.ok()) {
    return support.status();
  }
  Matrix3D slopes{};
  const bool use_affine = FitAllComponents(*support, slopes);
  if (!use_affine) {
    slopes = {};
  }
  Components3D velocity{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const auto& component = (*support)[axis];
    const double scaled_mean = std::clamp(component.scaled_velocity_mean, -1.0, 1.0);
    velocity[axis] = (scaled_mean - Dot(slopes[axis], component.mean_offset)) * component.velocity_scale;
    for (double& slope : slopes[axis]) {
      slope = slope * component.velocity_scale / *width;
    }
    if (!std::isfinite(velocity[axis]) || !Finite(Cartesian(slopes[axis]))) {
      return absl::OutOfRangeError("adaptive transfer gather is nonfinite");
    }
  }
  return GradedMacParticleGather3D{
      .velocity = Cartesian(velocity),
      .velocity_slopes = {Cartesian(slopes[0]), Cartesian(slopes[1]), Cartesian(slopes[2])},
      .affine_enabled = use_affine,
  };
}

}  // namespace fluidity
