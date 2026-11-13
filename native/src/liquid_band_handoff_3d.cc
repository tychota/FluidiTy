#include "fluidity/host/liquid_band_handoff_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "fluidity/host/graded_mac_transfer_3d.h"
#include "liquid_remap_3d_internal.h"

namespace fluidity {
namespace {

using pressure_internal::arithmetic::CompensatedSum;
using Cartesian = std::array<double, 3>;

constexpr double kRoundoff = 64 * std::numeric_limits<double>::epsilon();

Cartesian Components(Vec3 value) {
  return {value.x, value.y, value.z};
}

Vec3 AsVec3(const Cartesian& value) {
  return {value[0], value[1], value[2]};
}

bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool Finite(Mat3 value) {
  return Finite(value.x_row) && Finite(value.y_row) && Finite(value.z_row);
}

bool PositiveFinite(double value) {
  return value > 0.0 && std::isfinite(value);
}

constexpr double kParticleDepthInLeafWidths = 6;
constexpr double kOverlapInLeafWidths = 2;
// The first binary64 number above the exact half-diagonal of a unit cube.
constexpr double kUnitCubeHalfDiagonalUpper = 0x1.bb67ae8584cabp-1;

// FastTwoSum with the larger magnitude first identifies a rounded-down sum.
// Keep exact sums unchanged so a representable band edge remains a tie.
double AddUpperBound(double first, double second) {
  if (std::abs(first) < std::abs(second)) {
    std::swap(first, second);
  }
  const double sum = first + second;
  if (!std::isfinite(sum)) {
    return sum;
  }
  const double remainder = second - (sum - first);
  return remainder > 0 ? std::nextafter(sum, std::numeric_limits<double>::infinity()) : sum;
}

// The scale is a positive integer, so a nonzero exact product residual is
// at least one binary64 subnormal and its sign survives the fused operation.
double ScaleWidthUpperBound(double width_m, double integer_scale) {
  const double scaled_width_m = width_m * integer_scale;
  if (!std::isfinite(scaled_width_m) || std::fma(width_m, integer_scale, -scaled_width_m) <= 0) {
    return scaled_width_m;
  }
  return std::nextafter(scaled_width_m, std::numeric_limits<double>::infinity());
}

bool ValidInventory(const LiquidRemapInventory3D& inventory) {
  return PositiveFinite(inventory.mass_kg) && PositiveFinite(inventory.reference_volume_m3)
      && PositiveFinite(inventory.current_volume_m3) && Finite(inventory.momentum_kg_m_per_s)
      && std::isfinite(inventory.enthalpy_j) && liquid_remap_internal::ValidConstituentInventory(inventory);
}

// Zero and subnormal inventory lanes must survive exactly. For other lanes
// use a relative bound, with no absolute floor near vacuum or zero momentum.
bool Matches(double expected, double actual) {
  if (expected == actual) {
    return true;
  }
  if (!std::isfinite(actual) || std::abs(expected) < std::numeric_limits<double>::min()) {
    return false;
  }
  return std::abs((actual - expected) / expected) <= kRoundoff;
}

bool Matches(Vec3 expected, Vec3 actual) {
  return Matches(expected.x, actual.x) && Matches(expected.y, actual.y) && Matches(expected.z, actual.z);
}

struct CellBounds {
  Cartesian lower;
  Cartesian center;
  Cartesian upper;
};

absl::StatusOr<CellBounds>
CheckHandoffInputs(const LiquidHandoffCell3D& cell, const std::optional<LiquidHandoffCarrier3D>& carrier) {
  if (cell.occupancy != LiquidHandoffOccupancy3D::kFullLiquid || !carrier) {
    return absl::FailedPreconditionError("Liquid handoff requires full occupancy and a certified destination carrier");
  }
  if (!Finite(cell.world_lower_corner) || !PositiveFinite(cell.cell_width)) {
    return absl::InvalidArgumentError("Invalid liquid handoff cell geometry");
  }
  if (!PositiveFinite(carrier->current_density) || !PositiveFinite(carrier->volume_ratio)
      || !Finite(carrier->cell_center_velocity) || !Finite(carrier->velocity_slopes)) {
    return absl::InvalidArgumentError("Invalid liquid handoff destination carrier or EOS state");
  }

  CellBounds bounds{
      .lower = Components(cell.world_lower_corner),
      .center = {},
      .upper = {},
  };
  for (std::size_t axis = 0; axis < bounds.lower.size(); ++axis) {
    bounds.upper[axis] = bounds.lower[axis] + cell.cell_width;
    bounds.center[axis] = bounds.lower[axis] + cell.cell_width / 2.0;
    if (!std::isfinite(bounds.upper[axis]) || bounds.center[axis] <= bounds.lower[axis]
        || bounds.center[axis] >= bounds.upper[axis]) {
      return absl::OutOfRangeError("Liquid handoff cell bounds or center are unrepresentable");
    }
  }
  return bounds;
}

bool Contains(const CellBounds& bounds, Vec3 world_position) {
  const Cartesian position = Components(world_position);
  for (std::size_t axis = 0; axis < position.size(); ++axis) {
    if (position[axis] < bounds.lower[axis] || position[axis] >= bounds.upper[axis]) {
      return false;
    }
  }
  return true;
}

bool FiniteMomentum(double mass, Vec3 velocity, Vec3& momentum) {
  momentum = {mass * velocity.x, mass * velocity.y, mass * velocity.z};
  return Finite(momentum) && (velocity.x == 0.0 || momentum.x != 0.0) && (velocity.y == 0.0 || momentum.y != 0.0)
      && (velocity.z == 0.0 || momentum.z != 0.0);
}

// The same aggregate checks both retired inputs and the actual stored
// reseed outputs. Compensated summation retains mixed-sign momentum.
absl::StatusOr<LiquidRemapInventory3D> SumBand(
    const CellBounds& bounds,
    std::span<const LiquidParticleState3D> particles,
    double enthalpy_j,
    const LiquidHandoffCarrier3D& carrier
) {
  if (particles.empty() || !std::isfinite(enthalpy_j)) {
    return absl::InvalidArgumentError("Liquid band inventory is empty or has nonfinite enthalpy");
  }

  CompensatedSum mass;
  CompensatedSum reference_volume;
  CompensatedSum current_volume;
  std::array<CompensatedSum, 3> momentum;
  liquid_remap_internal::ConstituentSums constituent_sums;
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const auto& particle = particles[index];
    if (particle.id == 0 || !PositiveFinite(particle.mass) || !PositiveFinite(particle.reference_volume)
        || !PositiveFinite(particle.volume_ratio) || !Finite(particle.world_position)
        || !Finite(particle.world_velocity) || !Finite(particle.velocity_slopes)
        || !Contains(bounds, particle.world_position)) {
      return absl::InvalidArgumentError("Invalid or out-of-cell band particle");
    }
    if (particle.tag != carrier.tag || particle.side_tag != carrier.side_tag || particle.phase_id != carrier.phase_id) {
      return absl::FailedPreconditionError("Band particle material or side differs from the destination");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (particles[prior].id == particle.id) {
        return absl::InvalidArgumentError("Duplicate band particle ID");
      }
    }

    const double particle_current_volume = particle.volume_ratio * particle.reference_volume;
    Vec3 particle_momentum;
    if (!PositiveFinite(particle_current_volume)
        || !FiniteMomentum(particle.mass, particle.world_velocity, particle_momentum)) {
      return absl::OutOfRangeError("Band particle current volume or momentum is unrepresentable");
    }
    if (particle.constituent_count > particle.constituents.size()) {
      return absl::InvalidArgumentError("Band particle constituent count exceeds its capacity");
    }
    const auto composition = std::span(particle.constituents).first(particle.constituent_count);
    if (!ValidLiquidComposition3D(particle.phase_id, composition)) {
      return absl::InvalidArgumentError("Invalid band particle composition");
    }
    for (const auto& constituent : composition) {
      const auto status =
          constituent_sums.Add(constituent.substance_id, constituent.volume_fraction * particle_current_volume);
      if (!status.ok()) {
        return status;
      }
    }
    mass.Add(particle.mass);
    reference_volume.Add(particle.reference_volume);
    current_volume.Add(particle_current_volume);
    momentum[0].Add(particle_momentum.x);
    momentum[1].Add(particle_momentum.y);
    momentum[2].Add(particle_momentum.z);
  }
  LiquidRemapInventory3D inventory{
      .mass_kg = mass.Total(),
      .reference_volume_m3 = reference_volume.Total(),
      .current_volume_m3 = current_volume.Total(),
      .momentum_kg_m_per_s = {momentum[0].Total(), momentum[1].Total(), momentum[2].Total()},
      .enthalpy_j = enthalpy_j,
      .phase_id = carrier.phase_id,
  };
  const auto stored = constituent_sums.Store(inventory);
  if (!stored.ok()) {
    return stored;
  }
  if (!ValidInventory(inventory)) {
    return absl::OutOfRangeError("Summed band inventory is unrepresentable");
  }
  return inventory;
}

absl::Status CheckCarrierInventory(const LiquidRemapInventory3D& inventory, const LiquidHandoffCarrier3D& carrier) {
  if (inventory.phase_id != carrier.phase_id) {
    return absl::FailedPreconditionError("Destination phase differs from owned inventory");
  }
  const double carrier_mass = carrier.current_density * inventory.current_volume_m3;
  const double carrier_reference_volume = inventory.current_volume_m3 / carrier.volume_ratio;
  Vec3 carrier_momentum;
  if (!PositiveFinite(carrier_mass) || !PositiveFinite(carrier_reference_volume)
      || !FiniteMomentum(inventory.mass_kg, carrier.cell_center_velocity, carrier_momentum)) {
    return absl::OutOfRangeError("Destination carrier inventories are unrepresentable");
  }
  if (!Matches(inventory.mass_kg, carrier_mass) || !Matches(inventory.reference_volume_m3, carrier_reference_volume)
      || !Matches(inventory.momentum_kg_m_per_s, carrier_momentum)) {
    return absl::FailedPreconditionError("Destination density, EOS or velocity disagrees with owned inventory");
  }
  return absl::OkStatus();
}

Cartesian SeedOffset(std::size_t particle_count, std::size_t ordinal, double quarter_width) {
  if (particle_count % 2 != 0) {
    if (ordinal == 0) {
      return {};
    }
    --ordinal;
  }
  const std::size_t low_octant = ordinal / 2;
  const std::size_t octant = ordinal % 2 == 0 ? low_octant : 7 - low_octant;
  return {
      (octant & 1u) != 0 ? quarter_width : -quarter_width,
      (octant & 2u) != 0 ? quarter_width : -quarter_width,
      (octant & 4u) != 0 ? quarter_width : -quarter_width,
  };
}

// Refuse loss of a material affine increment at a large carrier velocity.
// Compare with the row's scale, since the three products can cancel.
bool SampleVelocity(double center_velocity, Vec3 slope_row, const Cartesian& offset, double& velocity) {
  const double affine_scale = std::max({std::abs(slope_row.x), std::abs(slope_row.y), std::abs(slope_row.z)});
  const double offset_scale = std::max({std::abs(offset[0]), std::abs(offset[1]), std::abs(offset[2])});
  if (affine_scale == 0.0 || offset_scale == 0.0) {
    velocity = center_velocity;
    return true;
  }
  const double speed_scale = affine_scale * offset_scale;
  const double increment = std::fma(slope_row.x, offset[0], std::fma(slope_row.y, offset[1], slope_row.z * offset[2]));
  velocity = center_velocity + increment;
  const double represented_increment = velocity - center_velocity;
  return PositiveFinite(speed_scale) && std::isfinite(velocity)
      && std::abs((represented_increment - increment) / speed_scale) <= kRoundoff;
}

absl::Status FillSeedGeometry(
    const CellBounds& bounds,
    const LiquidHandoffCarrier3D& carrier,
    const Cartesian& offset,
    LiquidParticleState3D& particle
) {
  Cartesian position{};
  Cartesian represented_offset{};
  for (std::size_t axis = 0; axis < offset.size(); ++axis) {
    position[axis] = bounds.center[axis] + offset[axis];
    represented_offset[axis] = position[axis] - bounds.center[axis];
    if (position[axis] <= bounds.lower[axis] || position[axis] >= bounds.upper[axis]
        || !Matches(offset[axis], represented_offset[axis])) {
      return absl::OutOfRangeError("Liquid reseed cannot represent an interior seed offset");
    }
  }
  particle.world_position = AsVec3(position);
  if (!SampleVelocity(
          carrier.cell_center_velocity.x,
          carrier.velocity_slopes.x_row,
          represented_offset,
          particle.world_velocity.x
      )
      || !SampleVelocity(
          carrier.cell_center_velocity.y,
          carrier.velocity_slopes.y_row,
          represented_offset,
          particle.world_velocity.y
      )
      || !SampleVelocity(
          carrier.cell_center_velocity.z,
          carrier.velocity_slopes.z_row,
          represented_offset,
          particle.world_velocity.z
      )) {
    return absl::OutOfRangeError("Liquid reseed cannot represent the destination affine velocity");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<double> BoundLiquidBandLeafSignedDistance3D(double leaf_width_m, double center_signed_distance_upper_m) {
  if (!PositiveFinite(leaf_width_m) || !std::isfinite(center_signed_distance_upper_m)) {
    return absl::InvalidArgumentError("Liquid band requires a finite leaf width and center signed distance");
  }
  const double half_diagonal_m =
      std::nextafter(leaf_width_m * kUnitCubeHalfDiagonalUpper, std::numeric_limits<double>::infinity());
  const double leaf_signed_distance_upper_m = AddUpperBound(center_signed_distance_upper_m, half_diagonal_m);
  if (!std::isfinite(leaf_signed_distance_upper_m)) {
    return absl::OutOfRangeError("Liquid band leaf signed-distance bound is unrepresentable");
  }
  return leaf_signed_distance_upper_m;
}

absl::StatusOr<LiquidBandPlacement3D> PlanLiquidBandPlacement3D(
    const LiquidHandoffCell3D& cell,
    AdaptiveLiquidMaterialOwner3D committed_owner,
    std::optional<double> leaf_signed_distance_upper_m,
    const LiquidBandMargins3D& margins,
    const std::optional<LiquidHandoffCarrier3D>& destination_carrier
) {
  if (!PositiveFinite(cell.cell_width) || !Finite(cell.world_lower_corner)) {
    return absl::InvalidArgumentError("Invalid liquid band leaf geometry");
  }
  if (cell.occupancy > LiquidHandoffOccupancy3D::kNotLiquid
      || committed_owner > AdaptiveLiquidMaterialOwner3D::kEulerian) {
    return absl::InvalidArgumentError("Invalid liquid band occupancy or material owner");
  }
  if (!leaf_signed_distance_upper_m || cell.occupancy == LiquidHandoffOccupancy3D::kMissing) {
    return absl::FailedPreconditionError("Liquid band placement requires certified occupancy and signed distance");
  }
  const std::array distances{
      margins.particle_support_radius_m,
      margins.accepted_sweep_m,
      margins.reseed_radius_m,
  };
  const bool valid_margins = std::ranges::all_of(distances, [](double distance) {
    return std::isfinite(distance) && distance >= 0;
  });
  if (!std::isfinite(*leaf_signed_distance_upper_m) || !valid_margins) {
    return absl::InvalidArgumentError("Invalid liquid band signed distance or margins");
  }

  const double movement_and_seed_margin_m = AddUpperBound(margins.accepted_sweep_m, margins.reseed_radius_m);
  const double required_depth_m = AddUpperBound(margins.particle_support_radius_m, movement_and_seed_margin_m);
  const double particle_depth_m =
      std::max(ScaleWidthUpperBound(cell.cell_width, kParticleDepthInLeafWidths), required_depth_m);
  const double overlap_width_m =
      std::max(ScaleWidthUpperBound(cell.cell_width, kOverlapInLeafWidths), movement_and_seed_margin_m);
  const double closest_swept_depth_m = -AddUpperBound(*leaf_signed_distance_upper_m, margins.accepted_sweep_m);
  if (!std::isfinite(particle_depth_m) || !std::isfinite(closest_swept_depth_m)) {
    return absl::OutOfRangeError("Liquid band depths or swept margins are unrepresentable");
  }

  LiquidBandPlacement3D placement{
      .widths = {.particle_depth_m = particle_depth_m, .overlap_width_m = overlap_width_m},
      .closest_swept_depth_m = closest_swept_depth_m,
      .action = LiquidBandPlacementAction3D::kKeep,
      .target_owner = committed_owner,
  };
  const bool needs_particles = closest_swept_depth_m <= particle_depth_m;
  const bool owns_particles = committed_owner == AdaptiveLiquidMaterialOwner3D::kParticles;
  if (needs_particles == owns_particles
      || (!needs_particles && cell.occupancy != LiquidHandoffOccupancy3D::kFullLiquid)) {
    return placement;
  }
  const auto bounds = CheckHandoffInputs(cell, destination_carrier);
  if (!bounds.ok()) {
    return bounds.status();
  }
  if (needs_particles) {
    placement.action = LiquidBandPlacementAction3D::kReseed;
    placement.target_owner = AdaptiveLiquidMaterialOwner3D::kParticles;
  } else {
    placement.action = LiquidBandPlacementAction3D::kRetire;
    placement.target_owner = AdaptiveLiquidMaterialOwner3D::kEulerian;
  }
  return placement;
}

absl::StatusOr<LiquidBandCarrierBlend3D> BlendLiquidBandCarrier3D(
    const LiquidBandWidths3D& widths,
    double face_signed_distance_m,
    std::int64_t face_side_tag,
    const LiquidAdaptiveFaceState3D& particle_carrier,
    const LiquidAdaptiveFaceState3D& eulerian_carrier
) {
  if (!PositiveFinite(widths.particle_depth_m) || !PositiveFinite(widths.overlap_width_m)
      || widths.overlap_width_m > widths.particle_depth_m || !std::isfinite(face_signed_distance_m)) {
    return absl::InvalidArgumentError("Invalid liquid band widths or face signed distance");
  }
  if (particle_carrier.validity > LiquidFaceValidity3D::kLowMass
      || eulerian_carrier.validity > LiquidFaceValidity3D::kLowMass) {
    return absl::InvalidArgumentError("Invalid liquid band carrier validity");
  }

  const double inward_face_depth_m = -face_signed_distance_m;
  const double overlap_start_m = widths.particle_depth_m - widths.overlap_width_m;
  double requested_particle_weight = 0;
  if (inward_face_depth_m <= overlap_start_m) {
    requested_particle_weight = 1;
  } else if (inward_face_depth_m < widths.particle_depth_m) {
    requested_particle_weight =
        std::clamp((widths.particle_depth_m - inward_face_depth_m) / widths.overlap_width_m, 0.0, 1.0);
  }
  const bool particle_available = particle_carrier.structurally_eligible && particle_carrier.side_tag == face_side_tag
      && particle_carrier.validity == LiquidFaceValidity3D::kValid;
  const double particle_weight = particle_available ? requested_particle_weight : 0;
  const double eulerian_weight = 1 - particle_weight;
  const bool needs_particle = particle_weight > 0;
  const bool needs_eulerian = eulerian_weight > 0;
  const bool eulerian_available = eulerian_carrier.structurally_eligible && eulerian_carrier.side_tag == face_side_tag
      && (eulerian_carrier.validity == LiquidFaceValidity3D::kValid
          || eulerian_carrier.validity == LiquidFaceValidity3D::kPredicted);
  if (needs_eulerian && !eulerian_available) {
    return absl::FailedPreconditionError("Liquid band blend requires an eligible Eulerian carrier");
  }
  if ((needs_particle && !std::isfinite(particle_carrier.carrier_velocity))
      || (needs_eulerian && !std::isfinite(eulerian_carrier.carrier_velocity))) {
    return absl::InvalidArgumentError("Liquid band blend requires finite carrier velocities");
  }

  double velocity_m_per_s = particle_carrier.carrier_velocity;
  if (!needs_particle) {
    velocity_m_per_s = eulerian_carrier.carrier_velocity;
  } else if (needs_eulerian) {
    velocity_m_per_s = std::lerp(eulerian_carrier.carrier_velocity, particle_carrier.carrier_velocity, particle_weight);
  }
  return LiquidBandCarrierBlend3D{
      .normal_velocity_m_per_s = velocity_m_per_s,
      .particle_weight = particle_weight,
      .eulerian_weight = eulerian_weight,
      .reset_particle_to_pic = requested_particle_weight > 0 && !particle_available,
  };
}

absl::StatusOr<std::optional<LiquidBandCarrierBlend3D>> PrepareLiquidBandOverlapFace3D(
    double fine_width_m,
    double face_signed_distance_m,
    std::int64_t face_side_tag,
    double minimum_face_mass_kg,
    const MacFaceDeposit3D& particle_deposit,
    const LiquidAdaptiveFaceState3D& eulerian_carrier
) {
  if (!PositiveFinite(fine_width_m) || !std::isfinite(face_signed_distance_m) || !std::isfinite(minimum_face_mass_kg)
      || minimum_face_mass_kg < 0) {
    return absl::InvalidArgumentError("invalid measured liquid band overlap");
  }
  const LiquidBandWidths3D widths{.particle_depth_m = 6 * fine_width_m, .overlap_width_m = 2 * fine_width_m};
  if (!std::isfinite(widths.particle_depth_m)) {
    return absl::OutOfRangeError("liquid band overlap width is unrepresentable");
  }
  const double inward_depth_m = -face_signed_distance_m;
  if (inward_depth_m < widths.particle_depth_m - widths.overlap_width_m || inward_depth_m >= widths.particle_depth_m) {
    return std::nullopt;
  }
  if (!eulerian_carrier.structurally_eligible || eulerian_carrier.side_tag != face_side_tag
      || (eulerian_carrier.validity != LiquidFaceValidity3D::kValid
          && eulerian_carrier.validity != LiquidFaceValidity3D::kPredicted)) {
    return absl::FailedPreconditionError("liquid band overlap requires a same-side Eulerian carrier");
  }
  if (!std::isfinite(eulerian_carrier.carrier_velocity)) {
    return absl::InvalidArgumentError("nonfinite liquid band Eulerian carrier");
  }
  const double mass_kg = particle_deposit.deposited_mass;
  const double momentum_kg_m_per_s = particle_deposit.deposited_pure_momentum;
  if (!std::isfinite(mass_kg) || mass_kg < 0 || !std::isfinite(momentum_kg_m_per_s)
      || (mass_kg == 0 && momentum_kg_m_per_s != 0)) {
    return absl::InvalidArgumentError("invalid particle deposit in liquid band overlap");
  }
  LiquidAdaptiveFaceState3D particle_carrier{.side_tag = face_side_tag};
  if (mass_kg > minimum_face_mass_kg) {
    particle_carrier.validity = LiquidFaceValidity3D::kValid;
    particle_carrier.carrier_velocity = momentum_kg_m_per_s / mass_kg;
    if (!std::isfinite(particle_carrier.carrier_velocity)) {
      return absl::OutOfRangeError("liquid band particle velocity is unrepresentable");
    }
  } else if (mass_kg > 0) {
    particle_carrier.validity = LiquidFaceValidity3D::kLowMass;
  }
  const auto blend =
      BlendLiquidBandCarrier3D(widths, face_signed_distance_m, face_side_tag, particle_carrier, eulerian_carrier);
  if (!blend.ok()) {
    return blend.status();
  }
  return *blend;
}

absl::StatusOr<LiquidRemapInventory3D> RetireLiquidBandCell3D(
    const LiquidHandoffCell3D& cell,
    std::span<const LiquidParticleState3D> band_particles,
    double enthalpy_j,
    const std::optional<LiquidHandoffCarrier3D>& destination_carrier
) {
  const auto bounds = CheckHandoffInputs(cell, destination_carrier);
  if (!bounds.ok()) {
    return bounds.status();
  }
  const auto inventory = SumBand(*bounds, band_particles, enthalpy_j, *destination_carrier);
  if (!inventory.ok()) {
    return inventory.status();
  }
  const absl::Status status = CheckCarrierInventory(*inventory, *destination_carrier);
  if (!status.ok()) {
    return status;
  }
  return *inventory;
}

absl::StatusOr<LiquidBandReseed3D> ReseedLiquidCoreCell3D(
    const LiquidHandoffCell3D& cell,
    const LiquidRemapInventory3D& core_inventory,
    const std::optional<LiquidHandoffCarrier3D>& destination_carrier,
    std::size_t particle_count,
    std::size_t available_particle_slots,
    std::uint64_t first_particle_id
) {
  const auto bounds = CheckHandoffInputs(cell, destination_carrier);
  if (!bounds.ok()) {
    return bounds.status();
  }
  if (particle_count == 0 || first_particle_id == 0 || !ValidInventory(core_inventory)) {
    return absl::InvalidArgumentError("Invalid liquid reseed count, first ID or owned inventory");
  }
  if (particle_count > LiquidBandReseed3D::kMaxParticles || particle_count > available_particle_slots) {
    return absl::ResourceExhaustedError("Insufficient liquid reseed particle capacity");
  }
  if (first_particle_id > std::numeric_limits<std::uint64_t>::max() - particle_count) {
    return absl::OutOfRangeError("Liquid reseed exhausts the next-ID counter");
  }

  const auto& carrier = *destination_carrier;
  const absl::Status status = CheckCarrierInventory(core_inventory, carrier);
  if (!status.ok()) {
    return status;
  }

  const double particle_current_volume = core_inventory.current_volume_m3 / static_cast<double>(particle_count);
  const double particle_mass = carrier.current_density * particle_current_volume;
  const double particle_reference_volume = particle_current_volume / carrier.volume_ratio;
  if (!PositiveFinite(particle_current_volume) || !PositiveFinite(particle_mass)
      || !PositiveFinite(particle_reference_volume)) {
    return absl::OutOfRangeError("Liquid reseed mass or volume per particle is unrepresentable");
  }
  const double quarter_width = cell.cell_width / 4.0;
  if (particle_count > 1 && (!PositiveFinite(quarter_width) || quarter_width * 4.0 != cell.cell_width)) {
    return absl::OutOfRangeError("Liquid reseed quarter-cell displacement is unrepresentable");
  }

  std::array<LiquidConstituentFraction3D, kMaximumLiquidParticleConstituents3D> fractions{};
  for (std::size_t index = 0; index < core_inventory.constituent_count; ++index) {
    const auto& constituent = core_inventory.constituents[index];
    fractions[index] = {constituent.substance_id, constituent.volume_m3 / core_inventory.current_volume_m3};
  }
  if (!ValidLiquidComposition3D(
          core_inventory.phase_id,
          std::span(fractions).first(core_inventory.constituent_count)
      )) {
    return absl::OutOfRangeError("Liquid reseed constituent fractions are unrepresentable");
  }
  LiquidBandReseed3D result;
  result.particle_count_ = particle_count;
  result.next_particle_id_ = first_particle_id + particle_count;
  result.enthalpy_j_ = core_inventory.enthalpy_j;
  for (std::size_t ordinal = 0; ordinal < particle_count; ++ordinal) {
    LiquidParticleState3D particle{
        .id = first_particle_id + ordinal,
        .tag = carrier.tag,
        .velocity_slopes = carrier.velocity_slopes,
        .mass = particle_mass,
        .reference_volume = particle_reference_volume,
        .volume_ratio = carrier.volume_ratio,
        .side_tag = carrier.side_tag,
        .phase_id = core_inventory.phase_id,
        .constituent_count = core_inventory.constituent_count,
        .constituents = fractions,
    };
    const auto offset = SeedOffset(particle_count, ordinal, quarter_width);
    const absl::Status seed_status = FillSeedGeometry(*bounds, carrier, offset, particle);
    if (!seed_status.ok()) {
      return seed_status;
    }
    result.particles_[ordinal] = particle;
  }

  const auto represented = SumBand(*bounds, result.particles(), result.enthalpy_j(), carrier);
  if (!represented.ok()) {
    return represented.status();
  }
  if (!Matches(core_inventory.mass_kg, represented->mass_kg)
      || !Matches(core_inventory.reference_volume_m3, represented->reference_volume_m3)
      || !Matches(core_inventory.current_volume_m3, represented->current_volume_m3)
      || !Matches(core_inventory.momentum_kg_m_per_s, represented->momentum_kg_m_per_s)) {
    return absl::OutOfRangeError("Liquid reseed cannot preserve stored cell inventories");
  }
  if (represented->constituent_count != core_inventory.constituent_count) {
    return absl::OutOfRangeError("Liquid reseed lost a constituent");
  }
  for (std::size_t index = 0; index < core_inventory.constituent_count; ++index) {
    const auto& expected = core_inventory.constituents[index];
    const auto& actual = represented->constituents[index];
    if (expected.substance_id != actual.substance_id || !Matches(expected.volume_m3, actual.volume_m3)) {
      return absl::OutOfRangeError("Liquid reseed cannot preserve a constituent volume");
    }
  }
  return result;
}

namespace {

// Match AdaptiveLiquidStep3D::AccumulateInventories in stored particle order.
// The handoff's compensated sum separately checks the full material ledger.
bool MatchesGradedParticleInventory(
    std::span<const LiquidParticleState3D> particles,
    const LiquidRemapInventory3D& owned
) {
  LiquidRemapInventory3D inventory{};
  for (const auto& particle : particles) {
    inventory.mass_kg += particle.mass;
    inventory.reference_volume_m3 += particle.reference_volume;
    inventory.current_volume_m3 += particle.reference_volume * particle.volume_ratio;
    inventory.momentum_kg_m_per_s.x += particle.mass * particle.world_velocity.x;
    inventory.momentum_kg_m_per_s.y += particle.mass * particle.world_velocity.y;
    inventory.momentum_kg_m_per_s.z += particle.mass * particle.world_velocity.z;
  }
  return inventory.mass_kg == owned.mass_kg && inventory.reference_volume_m3 == owned.reference_volume_m3
      && inventory.current_volume_m3 == owned.current_volume_m3
      && Components(inventory.momentum_kg_m_per_s) == Components(owned.momentum_kg_m_per_s);
}

absl::StatusOr<LiquidState3D> PrepareAdaptiveBandHandoff(
    const LiquidState3D& committed,
    LiquidStateCapacities3D capacities,
    AdaptiveMacCellKey3D leaf_key,
    LiquidHandoffOccupancy3D authoritative_occupancy,
    const std::optional<LiquidHandoffCarrier3D>& destination_carrier,
    std::optional<std::size_t> reseed_particle_count
) {
  const auto input_status = ValidateLiquidState3D(committed);
  if (!input_status.ok()) {
    return input_status;
  }
  if (authoritative_occupancy != LiquidHandoffOccupancy3D::kFullLiquid || !destination_carrier) {
    return absl::FailedPreconditionError("Adaptive handoff requires certified full occupancy and a carrier");
  }
  const auto active_cells = committed.adaptive_cells.values();
  const auto target = std::ranges::find_if(active_cells, [&](const auto& cell) {
    const auto lower = cell.key.lower_corner;
    return cell.key.width_exponent == leaf_key.width_exponent && lower.x == leaf_key.lower_corner.x
        && lower.y == leaf_key.lower_corner.y && lower.z == leaf_key.lower_corner.z;
  });
  if (target == active_cells.end()) {
    return absl::FailedPreconditionError("Adaptive handoff leaf is missing");
  }
  const auto source_owner =
      reseed_particle_count ? AdaptiveLiquidMaterialOwner3D::kEulerian : AdaptiveLiquidMaterialOwner3D::kParticles;
  if (target->occupancy != LiquidCellOccupancy3D::kLiquid || target->material_owner != source_owner) {
    return absl::FailedPreconditionError("Adaptive handoff leaf has incompatible occupancy or material ownership");
  }
  if (capacities.particles < committed.particles.size()) {
    return absl::ResourceExhaustedError("Adaptive handoff particle capacity cannot hold the committed state");
  }

  std::vector<AdaptiveMacCellKey3D> keys;
  keys.reserve(active_cells.size());
  for (const auto& cell : active_cells) {
    keys.push_back(cell.key);
  }
  const double finest_width = committed.geometry.cell_width;
  const auto topology = AdaptiveMacTopology3D::Create(keys, finest_width);
  if (!topology.ok()) {
    return topology.status();
  }
  const auto target_ordinal = static_cast<std::size_t>(target - active_cells.begin());
  const auto& leaf = topology->cells()[target_ordinal];
  if (target->inventory.current_volume_m3 != leaf.cell_volume) {
    return absl::FailedPreconditionError("Adaptive handoff inventory does not fill its leaf");
  }

  const auto world_origin = committed.geometry.world_origin;
  const auto transfer = GradedMacTransfer3D::Create(*topology, world_origin);
  if (!transfer.ok()) {
    return transfer.status();
  }

  // The handoff primitive uses lower + width for its half-open box. Require
  // it to describe the same footprint as the fused graded lattice bounds.
  const auto lower = leaf_key.lower_corner;
  const Cartesian lattice_lower{double(lower.x), double(lower.y), double(lower.z)};
  const auto origin = Components(world_origin);
  const double lattice_width = std::ldexp(1.0, leaf_key.width_exponent);
  Cartesian world_lower{};
  for (std::size_t axis = 0; axis < origin.size(); ++axis) {
    world_lower[axis] = std::fma(lattice_lower[axis], finest_width, origin[axis]);
    const double world_upper = std::fma(lattice_lower[axis] + lattice_width, finest_width, origin[axis]);
    if (world_lower[axis] + leaf.cell_width != world_upper) {
      return absl::OutOfRangeError("Adaptive handoff cannot represent the graded leaf bounds");
    }
  }
  const LiquidHandoffCell3D handoff_cell{
      .world_lower_corner = AsVec3(world_lower),
      .cell_width = leaf.cell_width,
      .occupancy = authoritative_occupancy,
  };

  std::vector<LiquidParticleState3D> retiring_particles;
  for (const auto& particle : committed.particles.values()) {
    const auto particle_leaf = transfer->ParticleLeafOrdinal(particle.world_position);
    if (!particle_leaf.ok()) {
      return particle_leaf.status();
    }
    if (active_cells[*particle_leaf].material_owner == AdaptiveLiquidMaterialOwner3D::kEulerian) {
      return absl::FailedPreconditionError("Eulerian adaptive leaf still contains a particle");
    }
    if (*particle_leaf == target_ordinal) {
      retiring_particles.push_back(particle);
    }
  }

  auto candidate = CloneLiquidState3D(committed, capacities);
  if (!candidate.ok()) {
    return candidate.status();
  }
  if (reseed_particle_count) {
    const auto seeds = ReseedLiquidCoreCell3D(
        handoff_cell,
        target->inventory,
        destination_carrier,
        *reseed_particle_count,
        capacities.particles - committed.particles.size(),
        committed.next_particle_id
    );
    if (!seeds.ok()) {
      return seeds.status();
    }
    for (const auto& seed : seeds->particles()) {
      const auto seed_leaf = transfer->ParticleLeafOrdinal(seed.world_position);
      if (!seed_leaf.ok() || *seed_leaf != target_ordinal) {
        return absl::OutOfRangeError("Adaptive handoff seed is outside its destination leaf");
      }
    }
    const auto represented =
        RetireLiquidBandCell3D(handoff_cell, seeds->particles(), seeds->enthalpy_j(), destination_carrier);
    if (!represented.ok()) {
      return represented.status();
    }
    auto particle_owned = *target;
    particle_owned.inventory = *represented;
    if (particle_owned != *target) {
      return absl::OutOfRangeError("Adaptive reseed cannot represent the exact owned inventory");
    }
    if (!MatchesGradedParticleInventory(seeds->particles(), target->inventory)) {
      return absl::OutOfRangeError("Adaptive reseed loses owned inventory in graded accumulation order");
    }

    const auto existing_particle_count = candidate->particles.size();
    const auto resized = candidate->particles.Resize(existing_particle_count + seeds->particles().size());
    if (!resized.ok()) {
      return resized;
    }
    std::ranges::copy(seeds->particles(), candidate->particles.values().subspan(existing_particle_count).begin());
    candidate->next_particle_id = seeds->next_particle_id();
    candidate->adaptive_cells[target_ordinal].material_owner = AdaptiveLiquidMaterialOwner3D::kParticles;
  } else {
    if (retiring_particles.empty()) {
      return absl::FailedPreconditionError("Particle-owned adaptive leaf has no resident particles");
    }
    const auto represented =
        RetireLiquidBandCell3D(handoff_cell, retiring_particles, target->inventory.enthalpy_j, destination_carrier);
    if (!represented.ok()) {
      return represented.status();
    }
    auto particle_owned = *target;
    particle_owned.inventory = *represented;
    if (particle_owned != *target) {
      return absl::FailedPreconditionError("Adaptive band particles disagree with their owned inventory");
    }
    if (!MatchesGradedParticleInventory(retiring_particles, target->inventory)) {
      return absl::FailedPreconditionError("Adaptive band particles disagree with the graded inventory sum");
    }

    // The selected IDs are unique and were collected in physical record order.
    std::size_t retired = 0;
    std::size_t retained = 0;
    for (const auto& particle : committed.particles.values()) {
      if (retired < retiring_particles.size() && particle.id == retiring_particles[retired].id) {
        ++retired;
        continue;
      }
      candidate->particles[retained] = particle;
      ++retained;
    }
    const auto resized = candidate->particles.Resize(retained);
    if (!resized.ok()) {
      return resized;
    }
    candidate->adaptive_cells[target_ordinal].material_owner = AdaptiveLiquidMaterialOwner3D::kEulerian;
  }
  const auto output_status = ValidateLiquidState3D(*candidate);
  if (!output_status.ok()) {
    return output_status;
  }
  return candidate;
}

}  // namespace

absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidBandRetirement3D(
    const LiquidState3D& committed,
    LiquidStateCapacities3D capacities,
    AdaptiveMacCellKey3D leaf_key,
    LiquidHandoffOccupancy3D authoritative_occupancy,
    const std::optional<LiquidHandoffCarrier3D>& destination_carrier
) {
  return PrepareAdaptiveBandHandoff(
      committed,
      capacities,
      leaf_key,
      authoritative_occupancy,
      destination_carrier,
      std::nullopt
  );
}

absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidCoreReseed3D(
    const LiquidState3D& committed,
    LiquidStateCapacities3D capacities,
    AdaptiveMacCellKey3D leaf_key,
    LiquidHandoffOccupancy3D authoritative_occupancy,
    const std::optional<LiquidHandoffCarrier3D>& destination_carrier,
    std::size_t particle_count
) {
  return PrepareAdaptiveBandHandoff(
      committed,
      capacities,
      leaf_key,
      authoritative_occupancy,
      destination_carrier,
      particle_count
  );
}

}  // namespace fluidity
