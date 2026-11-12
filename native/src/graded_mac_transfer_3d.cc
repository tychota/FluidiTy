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

constexpr std::size_t kFaceSupportBlockSize = 16;

// A subtree with N canonical blocks has 2N - 1 preorder bounds. Its left
// child gets floor(N/2) blocks and occupies the first nodes after the root.
std::size_t LeftSupportBlockCount(std::size_t node_count) {
  return (node_count + 1) / 4;
}

template <typename Bounds>
void BuildFaceSupportBounds(std::span<const GradedMacFace3D> faces, std::span<Bounds> nodes) {
  auto& root = nodes.front();
  if (nodes.size() == 1) {
    root.lower = Components(faces.front().world_anchor);
    root.upper = root.lower;
    for (const auto& face : faces.subspan(1)) {
      const auto anchor = Components(face.world_anchor);
      for (std::size_t axis = 0; axis < anchor.size(); ++axis) {
        root.lower[axis] = std::min(root.lower[axis], anchor[axis]);
        root.upper[axis] = std::max(root.upper[axis], anchor[axis]);
      }
    }
    return;
  }

  const std::size_t left_block_count = LeftSupportBlockCount(nodes.size());
  const std::size_t right_node = 2 * left_block_count;
  const std::size_t right_slot = left_block_count * kFaceSupportBlockSize;
  BuildFaceSupportBounds(faces.first(right_slot), nodes.subspan(1, right_node - 1));
  BuildFaceSupportBounds(faces.subspan(right_slot), nodes.subspan(right_node));
  for (std::size_t axis = 0; axis < root.lower.size(); ++axis) {
    root.lower[axis] = std::min(nodes[1].lower[axis], nodes[right_node].lower[axis]);
    root.upper[axis] = std::max(nodes[1].upper[axis], nodes[right_node].upper[axis]);
  }
}

// Use the same subtraction and division as the slot filter. Both operations
// are monotone for a fixed position and positive width. If the nearest bound
// in any coordinate fails the slot test, every anchor in the node fails it.
template <typename Bounds>
bool OutsideSupport(const Bounds& bounds, const Components3D& position, double leaf_width) {
  for (std::size_t axis = 0; axis < position.size(); ++axis) {
    if (position[axis] < bounds.lower[axis]) {
      if ((bounds.lower[axis] - position[axis]) / leaf_width >= kSupportRadiusInLeafWidths) {
        return true;
      }
    } else if (position[axis] > bounds.upper[axis]) {
      if ((bounds.upper[axis] - position[axis]) / leaf_width <= -kSupportRadiusInLeafWidths) {
        return true;
      }
    }
  }
  return false;
}

template <typename Bounds, typename Visit>
absl::Status VisitFaceSupportSlots(
    std::span<const Bounds> nodes,
    const Components3D& position,
    double leaf_width,
    std::size_t first_slot,
    std::size_t end_slot,
    const Visit& visit
) {
  if (OutsideSupport(nodes.front(), position, leaf_width)) {
    return absl::OkStatus();
  }
  if (nodes.size() == 1) {
    for (std::size_t slot = first_slot; slot < end_slot; ++slot) {
      const auto status = visit(slot);
      if (!status.ok()) {
        return status;
      }
    }
    return absl::OkStatus();
  }

  // The left subtree precedes the right in global slot order.
  const std::size_t left_block_count = LeftSupportBlockCount(nodes.size());
  const std::size_t right_node = 2 * left_block_count;
  const std::size_t right_slot = first_slot + left_block_count * kFaceSupportBlockSize;
  const auto left_status =
      VisitFaceSupportSlots(nodes.subspan(1, right_node - 1), position, leaf_width, first_slot, right_slot, visit);
  if (!left_status.ok()) {
    return left_status;
  }
  return VisitFaceSupportSlots(nodes.subspan(right_node), position, leaf_width, right_slot, end_slot, visit);
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

bool IsEligibleCarrier(const LiquidAdaptiveFaceState3D& carrier, std::int64_t side_tag, bool gather) {
  return carrier.structurally_eligible && carrier.side_tag == side_tag
      && (!gather || carrier.validity != LiquidFaceValidity3D::kMissing);
}

// Evaluate the compact, area-weighted transfer on canonical global slots.
// All offsets use the particle's owning leaf width to condition the fit.
template <bool IncludeMasked = false, typename Bounds, typename Visit>
absl::Status VisitSupport(
    std::span<const Bounds> bounds,
    std::span<const GradedMacFace3D> faces,
    std::span<const LiquidAdaptiveFaceState3D> carriers,
    const LiquidParticleState3D& particle,
    double leaf_width,
    bool gather,
    Visit visit
) {
  const auto visit_slot = [&](std::size_t slot) {
    const auto& carrier = carriers[slot];
    if constexpr (!IncludeMasked) {
      if (!IsEligibleCarrier(carrier, particle.side_tag, gather)) {
        return absl::OkStatus();
      }
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
      return absl::OkStatus();
    }
    const double radial_weight = 1 - Dot(offset, offset) / (kSupportRadiusInLeafWidths * kSupportRadiusInLeafWidths);
    if (radial_weight <= 0) {
      return absl::OkStatus();
    }
    const double width_ratio = face.width_m / leaf_width;
    const double weight = (radial_weight * width_ratio) * (radial_weight * width_ratio);
    if (gather && (carrier.validity > LiquidFaceValidity3D::kLowMass || !std::isfinite(carrier.carrier_velocity))) {
      return absl::InvalidArgumentError("invalid adaptive transfer carrier");
    }
    return visit(slot, static_cast<std::size_t>(face.axis), offset, weight);
  };
  return VisitFaceSupportSlots(bounds, Components(particle.world_position), leaf_width, 0, faces.size(), visit_slot);
}

template <bool GeometryOnly = false, typename Bounds>
absl::StatusOr<ParticleSupport> BuildSupport(
    std::span<const Bounds> bounds,
    std::span<const GradedMacFace3D> faces,
    std::span<const LiquidAdaptiveFaceState3D> carriers,
    const LiquidParticleState3D& particle,
    double leaf_width,
    bool gather
) {
  ParticleSupport support{};
  auto status = VisitSupport(
      bounds,
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
          if constexpr (!GeometryOnly) {
            component.velocity_scale = std::max(component.velocity_scale, std::abs(carriers[slot].carrier_velocity));
          }
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
      bounds,
      faces,
      carriers,
      particle,
      leaf_width,
      gather,
      [&]([[maybe_unused]] std::size_t slot, std::size_t axis, const Components3D& offset, double weight) {
        auto& component = support[axis];
        const double normalized_weight = weight / component.weight_sum;
        const Components3D centered{
            offset[0] - component.mean_offset[0],
            offset[1] - component.mean_offset[1],
            offset[2] - component.mean_offset[2],
        };
        double scaled_velocity = 0;
        if constexpr (!GeometryOnly) {
          scaled_velocity =
              component.velocity_scale == 0 ? 0 : carriers[slot].carrier_velocity / component.velocity_scale;
          component.scaled_velocity_mean += normalized_weight * scaled_velocity;
        }
        for (std::size_t row = 0; row < 3; ++row) {
          if constexpr (!GeometryOnly) {
            component.velocity_covariance[row] += normalized_weight * centered[row] * scaled_velocity;
          }
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

// Compare the normalized moments in leaf-width units, as used by the
// uniform resampling formulas. A floating-point match is diagnostic only.
bool MatchesUniformMoments(const GradedMacSupportMomentWitness3D& witness, LiquidUniformResamplingBasis3D basis) {
  for (std::size_t axis = 0; axis < witness.rows.size(); ++axis) {
    const auto& row = witness.rows[axis];
    if (row.normalized_weight_sum != 1 || Components(row.mean_offset) != Components3D{}) {
      return false;
    }
    const Matrix3D covariance{
        Components(row.centered_covariance.x_row),
        Components(row.centered_covariance.y_row),
        Components(row.centered_covariance.z_row),
    };
    for (std::size_t spatial = 0; spatial < covariance.size(); ++spatial) {
      Components3D expected{};
      const bool cubic_normal = basis == LiquidUniformResamplingBasis3D::kCubicNormal && spatial == axis;
      expected[spatial] = cubic_normal ? 1.0 / 3 : 0.25;
      if (covariance[spatial] != expected) {
        return false;
      }
    }
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
  return Create(topology, world_origin, std::nullopt);
}

absl::StatusOr<GradedMacTransfer3D>
GradedMacTransfer3D::Create(const AdaptiveMacTopology3D& topology, Vec3 world_origin, std::optional<Vec3> solid_upper) {
  if (!Finite(world_origin) || (solid_upper && !Finite(*solid_upper))) {
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
  transfer.solid_upper_ = solid_upper;
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
  const std::size_t block_count = (face_count + kFaceSupportBlockSize - 1) / kFaceSupportBlockSize;
  transfer.face_support_bounds_.resize(2 * block_count - 1);
  BuildFaceSupportBounds(transfer.faces_, std::span(transfer.face_support_bounds_));
  return transfer;
}

absl::StatusOr<std::size_t> GradedMacTransfer3D::ParticleLeafOrdinal(Vec3 world_position) const {
  if (!Finite(world_position)) {
    return absl::InvalidArgumentError("adaptive transfer particle position must be finite");
  }
  for (std::size_t ordinal = 0; ordinal < leaves_.size(); ++ordinal) {
    const auto& leaf = leaves_[ordinal];
    const bool inside_upper_x = world_position.x < leaf.upper.x
        || (solid_upper_ && world_position.x == leaf.upper.x && leaf.upper.x == solid_upper_->x);
    const bool inside_upper_z = world_position.z < leaf.upper.z
        || (solid_upper_ && world_position.z == leaf.upper.z && leaf.upper.z == solid_upper_->z);
    if (world_position.x >= leaf.lower.x && inside_upper_x && world_position.y >= leaf.lower.y
        && world_position.y < leaf.upper.y && world_position.z >= leaf.lower.z && inside_upper_z) {
      return ordinal;
    }
  }
  return absl::FailedPreconditionError("adaptive transfer particle has no containing leaf");
}

absl::StatusOr<double> GradedMacTransfer3D::ParticleSupportWidth(Vec3 world_position) const {
  const auto ordinal = ParticleLeafOrdinal(world_position);
  if (!ordinal.ok()) {
    return ordinal.status();
  }
  return leaves_[*ordinal].width_m;
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
    const auto status = AccumulateParticleDeposit(particle, carrier_faces, deposits);
    if (!status.ok()) {
      return status;
    }
  }
  return deposits;
}

absl::Status GradedMacTransfer3D::AccumulateParticleDeposit(
    const LiquidParticleState3D& particle,
    std::span<const LiquidAdaptiveFaceState3D> carrier_faces,
    std::span<MacFaceDeposit3D> deposits
) const {
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
  const auto support = BuildSupport(std::span(face_support_bounds_), faces_, carrier_faces, particle, *width, false);
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
  return VisitSupport(
      std::span(face_support_bounds_),
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
  const auto support = BuildSupport(std::span(face_support_bounds_), faces_, carrier_faces, particle, *width, true);
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

absl::StatusOr<GradedMacSupportMomentWitness3D> GradedMacTransfer3D::MeasureSupportMoments(
    const LiquidParticleState3D& particle,
    std::span<const LiquidAdaptiveFaceState3D> carrier_faces
) const {
  if (carrier_faces.size() != faces_.size()) {
    return absl::InvalidArgumentError("adaptive support witness requires canonical face extent");
  }
  for (const auto& carrier : carrier_faces) {
    if (carrier.validity > LiquidFaceValidity3D::kLowMass || !std::isfinite(carrier.carrier_velocity)) {
      return absl::InvalidArgumentError("adaptive support witness requires valid finite carriers");
    }
  }
  const auto width = ParticleSupportWidth(particle.world_position);
  if (!width.ok()) {
    return width.status();
  }
  const auto bounds = std::span(face_support_bounds_);
  const auto support = BuildSupport<true>(bounds, faces_, carrier_faces, particle, *width, true);
  if (!support.ok()) {
    return support.status();
  }

  GradedMacSupportMomentWitness3D witness{.leaf_width_m = *width};
  for (std::size_t axis = 0; axis < witness.rows.size(); ++axis) {
    const auto& component = (*support)[axis];
    const auto& covariance = component.covariance;
    Components3D unused_slopes{};
    witness.rows[axis] = {
        .unnormalized_weight_sum = component.weight_sum,
        .mean_offset = Cartesian(component.mean_offset),
        .centered_covariance =
            {
                {covariance[0][0], covariance[1][0], covariance[2][0]},
                {covariance[1][0], covariance[1][1], covariance[2][1]},
                {covariance[2][0], covariance[2][1], covariance[2][2]},
            },
        .full_rank = SolveAffineSlope(component, unused_slopes),
        .uses_low_mass = component.uses_low_mass,
    };
  }

  // Visit positive geometric support before masking so a normalized fit cannot
  // hide missing, incompatible-side, or structurally ineligible carriers.
  const auto measured = VisitSupport<true>(
      bounds,
      faces_,
      carrier_faces,
      particle,
      *width,
      false,
      [&](std::size_t slot, std::size_t axis, const Components3D&, double weight) {
        if (!std::isfinite(weight) || weight <= 0) {
          return absl::OutOfRangeError("adaptive support witness weight is unrepresentable");
        }
        witness.has_width_transition |= faces_[slot].width_m != *width;
        if (!IsEligibleCarrier(carrier_faces[slot], particle.side_tag, true)) {
          witness.has_masked_support = true;
          return absl::OkStatus();
        }
        auto& row = witness.rows[axis];
        ++row.positive_sample_count;
        row.normalized_weight_sum += weight / row.unnormalized_weight_sum;
        return absl::OkStatus();
      }
  );
  if (!measured.ok()) {
    return measured;
  }
  for (const auto& row : witness.rows) {
    const auto& covariance = row.centered_covariance;
    if (!std::isfinite(row.normalized_weight_sum) || !Finite(row.mean_offset) || !Finite(covariance.x_row)
        || !Finite(covariance.y_row) || !Finite(covariance.z_row)) {
      return absl::OutOfRangeError("adaptive support witness moments are nonfinite");
    }
  }
  witness.matches_quadratic_moments = MatchesUniformMoments(witness, LiquidUniformResamplingBasis3D::kQuadratic);
  witness.matches_cubic_normal_moments = MatchesUniformMoments(witness, LiquidUniformResamplingBasis3D::kCubicNormal);
  return witness;
}

absl::Status GradedMacTransfer3D::CheckUniformResamplingSupport(
    const LiquidParticleState3D& particle,
    double requested_leaf_width_m,
    LiquidUniformResamplingBasis3D basis,
    std::span<const LiquidAdaptiveFaceState3D> carrier_faces
) const {
  if (!std::isfinite(requested_leaf_width_m) || requested_leaf_width_m <= 0) {
    return absl::InvalidArgumentError("uniform resampling requires a finite positive leaf width");
  }
  if (basis != LiquidUniformResamplingBasis3D::kQuadratic && basis != LiquidUniformResamplingBasis3D::kCubicNormal) {
    return absl::InvalidArgumentError("uniform resampling requires a recognized basis");
  }
  const auto witness = MeasureSupportMoments(particle, carrier_faces);
  if (!witness.ok()) {
    return witness.status();
  }
  if (!particle.affine_enabled || std::ranges::any_of(witness->rows, [](const auto& row) {
        return !row.full_rank || row.uses_low_mass;
      })) {
    return absl::FailedPreconditionError("uniform resampling requires usable full-rank affine support");
  }
  if (witness->has_masked_support) {
    return absl::FailedPreconditionError("uniform resampling refuses masked graded support");
  }
  if (witness->leaf_width_m != requested_leaf_width_m || witness->has_width_transition) {
    return absl::FailedPreconditionError("uniform resampling refuses a graded width mismatch or transition");
  }
  const bool moment_match = basis == LiquidUniformResamplingBasis3D::kQuadratic ? witness->matches_quadratic_moments
                                                                                : witness->matches_cubic_normal_moments;
  if (!moment_match) {
    return absl::FailedPreconditionError("actual graded transfer does not have the required uniform support moments");
  }
  return absl::FailedPreconditionError("graded radial weights cannot certify complete uniform tensor support");
}

}  // namespace fluidity
