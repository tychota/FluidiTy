#include "fluidity/host/liquid_boundary_transfer_3d.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "fluidity/host/liquid_boundary_workspace_3d.h"
#include "liquid_boundary_transfer_3d_internal.h"
#include "mac_gather_3d_internal.h"

namespace fluidity {
namespace {
constexpr std::array kAxes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
constexpr std::array kVelocityComponents{&Vec3::x, &Vec3::y, &Vec3::z};
constexpr std::array kVelocitySlopeRows{&Mat3::x_row, &Mat3::y_row, &Mat3::z_row};
// A 4h sphere contains at most 268 staggered anchors of one component.
// The largest count occurs when projecting across two half-cell offsets.
constexpr std::size_t kMaxDonors = 268;
constexpr std::int64_t kMaximumQuantaSum = std::int64_t{1} << 62;

bool IsFinite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double Dot(Vec3 lhs, Vec3 rhs) {
  return (lhs.x * rhs.x + lhs.y * rhs.y) + lhs.z * rhs.z;
}

void AddScaled(Vec3& sum, Vec3 value, double scale) {
  sum.x += value.x * scale;
  sum.y += value.y * scale;
  sum.z += value.z * scale;
}

Vec3 Difference(Vec3 lhs, Vec3 rhs) {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

absl::Status ValidateCapacity(std::size_t face_count, std::size_t particle_count, bool bounded) {
  if (face_count > kLiquidBoundaryFaceCapacity3D || particle_count > kLiquidBoundaryParticleCapacity3D) {
    return absl::Status(
        absl::StatusCode::kResourceExhausted,
        bounded ? "" : "liquid boundary transfer exceeds bounded face/particle capacity"
    );
  }
  return absl::OkStatus();
}

absl::Status ValidateQuantization(MacDepositQuantization3D quantization, std::size_t particle_count, bool bounded) {
  for (double scale : {quantization.mass_per_quantum, quantization.pure_momentum_per_quantum}) {
    int exponent = 0;
    if (!std::isnormal(scale) || std::frexp(scale, &exponent) != 0.5) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          bounded ? "" : "quantization scales must be positive normal powers of two"
      );
    }
  }
  if (quantization.declared_particle_capacity > kLiquidBoundaryParticleCapacity3D
      || particle_count > quantization.declared_particle_capacity) {
    return absl::Status(
        absl::StatusCode::kResourceExhausted,
        bounded ? "" : "particle count or declared capacity exceeds boundary transfer capacity"
    );
  }
  return absl::OkStatus();
}

// The guard bounds this magnitude by 2^62 before conversion. Resolve ties
// explicitly so the ambient floating-point rounding mode cannot change parity.
std::int64_t RoundNearestEven(double scaled_contribution) {
  const double magnitude = std::abs(scaled_contribution);
  const double integral = std::floor(magnitude);
  const double fraction = magnitude - integral;
  auto quanta = static_cast<std::int64_t>(integral);
  if (fraction > 0.5 || (fraction == 0.5 && quanta % 2 != 0)) {
    ++quanta;
  }
  return scaled_contribution < 0.0 ? -quanta : quanta;
}

bool IsUsable(LiquidFaceValidity3D validity) {
  return validity == LiquidFaceValidity3D::kValid || validity == LiquidFaceValidity3D::kPredicted;
}

bool IsCompatible(LiquidBoundaryFaceTag3D tag, std::int64_t side_tag) {
  return tag.structurally_eligible && tag.side_tag == side_tag;
}

absl::Status ValidateHistory(
    const LiquidBoundaryVelocityHistory3D& history,
    std::size_t face_count,
    double stage_time,
    bool bounded
) {
  if (history.velocity_sample_time != stage_time || history.wall_sample_time != stage_time) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        bounded ? "" : "velocity and wall history must match their requested stage time"
    );
  }
  if (history.face_velocities.size() != face_count || history.validity.size() != face_count
      || history.face_tags.size() != face_count || history.wall_samples.size() != face_count) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        bounded ? "" : "each history span must cover every owned face"
    );
  }
  for (std::size_t slot = 0; slot < face_count; ++slot) {
    const auto validity = history.validity[slot];
    if (validity != LiquidFaceValidity3D::kMissing && validity != LiquidFaceValidity3D::kLowMass
        && !IsUsable(validity)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "unknown face validity");
    }
    if (IsUsable(validity) && !std::isfinite(history.face_velocities[slot])) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          bounded ? "" : "valid and predicted face velocities must be finite"
      );
    }
    const auto& wall = history.wall_samples[slot];
    if (wall.available) {
      if (!IsFinite(wall.wall_velocity) || !IsFinite(wall.unit_normal)) {
        return absl::Status(
            absl::StatusCode::kInvalidArgument,
            bounded ? "" : "prescribed wall velocity and normal must be finite"
        );
      }
      if (std::abs(Dot(wall.unit_normal, wall.unit_normal) - 1.0) > 1e-12) {
        return absl::Status(
            absl::StatusCode::kInvalidArgument,
            bounded ? "" : "prescribed wall normal must have unit length"
        );
      }
    }
  }
  return absl::OkStatus();
}

struct OwnedFaceAnchor {
  FaceNormalAxis axis;
  Vec3 anchor_in_cell_widths;
};

// Match SparseMacGrid3D's owned order exactly. Relative lattice differences
// avoid subtracting large translated world anchors and preserve the 2h bound.
template <typename Anchor>
void BuildOwnedAnchors(SparseMacGridView3D grid, std::span<Anchor> anchors) {
  std::size_t slot = 0;
  for (const auto block : grid.active_block_indices()) {
    for (auto axis : kAxes) {
      for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
          for (int x = 0; x < 4; ++x) {
            anchors[slot++] = {
                .axis = axis,
                .anchor_in_cell_widths =
                    {
                        4.0 * block.x + x + (axis == FaceNormalAxis::kX ? 0.0 : 0.5),
                        4.0 * block.y + y + (axis == FaceNormalAxis::kY ? 0.0 : 0.5),
                        4.0 * block.z + z + (axis == FaceNormalAxis::kZ ? 0.0 : 0.5),
                    },
            };
          }
        }
      }
    }
  }
}

struct ContinuedVelocity {
  double velocity;
  bool pic_fallback;
};

// Each component is fit independently on its own staggered donors. The same
// target point is used for all components needed by an oblique wall projection.
template <typename Anchor>
absl::StatusOr<ContinuedVelocity> ContinueComponent(
    const LiquidBoundaryVelocityHistory3D& history,
    std::span<const Anchor> anchors,
    Vec3 target_in_cell_widths,
    std::size_t component_index,
    std::int64_t side_tag,
    const LiquidPrescribedWallSample3D& wall,
    bool bounded,
    double squared_radius
) {
  std::array<internal::LiquidGhostFitSample3D, kMaxDonors> donors;
  std::size_t donor_count = 0;
  std::size_t nearest_slot = anchors.size();
  double nearest_squared_distance = std::numeric_limits<double>::infinity();
  for (std::size_t slot = 0; slot < anchors.size(); ++slot) {
    if (anchors[slot].axis != kAxes[component_index] || !IsCompatible(history.face_tags[slot], side_tag)
        || !IsUsable(history.validity[slot])) {
      continue;
    }
    const Vec3 offset = Difference(anchors[slot].anchor_in_cell_widths, target_in_cell_widths);
    const double squared_distance = Dot(offset, offset);
    if (squared_distance > squared_radius) {
      continue;
    }
    if (donor_count == donors.size()) {
      return absl::Status(
          absl::StatusCode::kResourceExhausted,
          bounded ? "" : "ghost donor count exceeds its bounded local cube"
      );
    }
    donors[donor_count++] = {offset, history.face_velocities[slot]};
    // Strict comparison keeps the first owned slot on exact distance ties.
    if (squared_distance < nearest_squared_distance) {
      nearest_slot = slot;
      nearest_squared_distance = squared_distance;
    }
  }
  const auto fit = internal::FitLiquidGhostAffineCore3D(std::span(donors).first(donor_count));
  if (!fit.ok()) {
    return absl::Status(
        fit.status() == boundary_internal::Error::kInvalid ? absl::StatusCode::kInvalidArgument
                                                           : absl::StatusCode::kOutOfRange,
        bounded ? "" : "unrepresentable ghost fit arithmetic"
    );
  }
  if (fit->full_rank) {
    return ContinuedVelocity{fit->velocity_at_target, false};
  }
  if (nearest_slot != anchors.size()) {
    return ContinuedVelocity{history.face_velocities[nearest_slot], true};
  }
  if (wall.available && wall.side_tag == side_tag) {
    return ContinuedVelocity{wall.wall_velocity.*kVelocityComponents[component_index], true};
  }
  return absl::Status(
      absl::StatusCode::kFailedPrecondition,
      bounded ? "" : "ghost has neither same-side donors nor a compatible prescribed wall"
  );
}

template <typename Anchor>
absl::StatusOr<ContinuedVelocity> FillVelocity(
    const LiquidBoundaryVelocityHistory3D& history,
    std::span<const Anchor> anchors,
    std::size_t target_slot,
    std::size_t component_index,
    std::int64_t side_tag,
    bool bounded,
    double squared_radius = 4.0
) {
  if (IsCompatible(history.face_tags[target_slot], side_tag) && IsUsable(history.validity[target_slot])) {
    return ContinuedVelocity{history.face_velocities[target_slot], false};
  }
  const auto& wall = history.wall_samples[target_slot];
  auto component = ContinueComponent(
      history,
      anchors,
      anchors[target_slot].anchor_in_cell_widths,
      component_index,
      side_tag,
      wall,
      bounded,
      squared_radius
  );
  if (!component.ok()) {
    return component.status();
  }
  const bool compatible_wall = wall.available && wall.side_tag == side_tag;
  const double normal_component = wall.unit_normal.*kVelocityComponents[component_index];
  if (compatible_wall && normal_component != 0.0) {
    Vec3 fitted_vector = wall.wall_velocity;
    fitted_vector.*kVelocityComponents[component_index] = component->velocity;
    for (std::size_t other_component = 0; other_component < 3; ++other_component) {
      if (other_component == component_index || wall.unit_normal.*kVelocityComponents[other_component] == 0.0) {
        continue;
      }
      const auto other = ContinueComponent(
          history,
          anchors,
          anchors[target_slot].anchor_in_cell_widths,
          other_component,
          side_tag,
          wall,
          bounded,
          4.0
      );
      if (!other.ok()) {
        return other.status();
      }
      fitted_vector.*kVelocityComponents[other_component] = other->velocity;
      component->pic_fallback = component->pic_fallback || other->pic_fallback;
    }
    // Only compatible physical affine fields survive this normal replacement.
    // Old/new callers supply their own wall velocity; no wall snapshot is reused.
    const double relative_normal_velocity = Dot(Difference(fitted_vector, wall.wall_velocity), wall.unit_normal);
    component->velocity -= relative_normal_velocity * normal_component;
    if (!std::isfinite(component->velocity)) {
      return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "wall-constrained ghost velocity is nonfinite");
    }
  }
  return component;
}
}  // namespace

namespace internal {
absl::StatusOr<LiquidGhostAffineFit3D> FitLiquidGhostAffine3D(std::span<const LiquidGhostFitSample3D> samples) {
  if (samples.size() > 125) {
    return absl::Status(absl::StatusCode::kResourceExhausted, "affine ghost fit exceeds bounded row count");
  }
  const auto fit = FitLiquidGhostAffineCore3D(samples);
  if (!fit.ok()) {
    if (fit.status() == boundary_internal::Error::kInvalid) {
      return absl::Status(absl::StatusCode::kInvalidArgument, "invalid ghost fit rows");
    }
    return absl::Status(absl::StatusCode::kOutOfRange, "unrepresentable ghost fit arithmetic");
  }
  return *fit;
}

}  // namespace internal

namespace {
template <typename Result>
absl::Status DepositCore(const LiquidBoundaryDepositViewInputs3D& inputs, Result& result, bool bounded) {
  const auto capacity_status = ValidateCapacity(inputs.mac_grid.face_count(), inputs.particles.size(), bounded);
  if (!capacity_status.ok()) {
    return capacity_status;
  }
  const auto quantization_status = ValidateQuantization(inputs.quantization, inputs.particles.size(), bounded);
  if (!quantization_status.ok()) {
    return quantization_status;
  }
  if (inputs.face_tags.size() != inputs.mac_grid.face_count()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "deposit tags must cover every owned face");
  }
  std::ranges::fill(result.face_quanta, MacFaceDepositQuanta3D{});
  std::ranges::fill(result.compatible_stencils, LiquidCompatibleStencils3D{});
  const auto capacity = std::max(std::size_t{1}, inputs.quantization.declared_particle_capacity);
  const int capacity_exponent = static_cast<int>(std::bit_width(capacity - 1));
  const double contribution_limit = std::ldexp(1.0, 62 - capacity_exponent);
  for (std::size_t particle_index = 0; particle_index < inputs.particles.size(); ++particle_index) {
    const auto& particle = inputs.particles[particle_index];
    const auto& transfer = particle.transfer;
    if (!std::isfinite(transfer.particle_mass) || transfer.particle_mass < 0.0
        || !IsFinite(transfer.particle_velocity)) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          bounded ? "" : "particle mass/velocity must be finite with nonnegative mass"
      );
    }
    for (std::size_t component = 0; component < 3; ++component) {
      // Resolve the complete original stencil first: exclusions cannot conceal
      // an unallocated owner, including a node with exactly zero basis weight.
      const auto stencil = BuildOwnedMacFaceStencil3D(
          inputs.mac_grid,
          kAxes[component],
          transfer.sorted_world_position,
          transfer.trial_world_position
      );
      if (!stencil.ok()) {
        return stencil.status();
      }
      const Vec3 affine_row =
          particle.affine_enabled ? transfer.particle_velocity_slopes.*kVelocitySlopeRows[component] : Vec3{};
      if (!IsFinite(affine_row)) {
        return absl::Status(
            absl::StatusCode::kInvalidArgument,
            bounded ? "" : "enabled particle affine slopes must be finite"
        );
      }
      double compatible_weight_sum = 0.0;
      Vec3 compatible_gradient_sum{};
      for (const auto& sample : *stencil) {
        if (IsCompatible(inputs.face_tags[sample.face_slot], particle.side_tag)) {
          compatible_weight_sum += sample.weight;
          AddScaled(compatible_gradient_sum, sample.weight_gradient, 1.0);
        }
      }
      if (!std::isfinite(compatible_weight_sum) || !IsFinite(compatible_gradient_sum)) {
        return absl::Status(
            absl::StatusCode::kOutOfRange,
            bounded ? "" : "compatible weight or derivative sum is nonfinite"
        );
      }
      if (compatible_weight_sum <= 0.0) {
        return absl::Status(
            absl::StatusCode::kFailedPrecondition,
            bounded ? "" : "component has no positive compatible support"
        );
      }
      auto& compatible_stencil = result.compatible_stencils[particle_index][component];
      Vec3 weighted_mean_offset{};
      for (std::size_t sample_index = 0; sample_index < stencil->size(); ++sample_index) {
        const auto& sample = (*stencil)[sample_index];
        auto& compatible_sample = compatible_stencil[sample_index];
        compatible_sample.face_slot = sample.face_slot;
        compatible_sample.compatible = IsCompatible(inputs.face_tags[sample.face_slot], particle.side_tag);
        if (!compatible_sample.compatible) {
          continue;
        }
        compatible_sample.normalized_weight = sample.weight / compatible_weight_sum;
        const double normalized_weight = compatible_sample.normalized_weight;
        compatible_sample.normalized_weight_gradient = {
            (sample.weight_gradient.x - normalized_weight * compatible_gradient_sum.x) / compatible_weight_sum,
            (sample.weight_gradient.y - normalized_weight * compatible_gradient_sum.y) / compatible_weight_sum,
            (sample.weight_gradient.z - normalized_weight * compatible_gradient_sum.z) / compatible_weight_sum,
        };
        if (!std::isfinite(normalized_weight) || !IsFinite(compatible_sample.normalized_weight_gradient)) {
          return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "normalized compatible basis is nonfinite");
        }
        AddScaled(weighted_mean_offset, sample.face_offset_from_particle, normalized_weight);
      }
      if (!IsFinite(weighted_mean_offset)) {
        return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "compatible weighted offset is nonfinite");
      }
      for (std::size_t sample_index = 0; sample_index < stencil->size(); ++sample_index) {
        auto& sample = compatible_stencil[sample_index];
        if (!sample.compatible) {
          continue;
        }
        sample.centered_face_offset =
            Difference((*stencil)[sample_index].face_offset_from_particle, weighted_mean_offset);
        if (!IsFinite(sample.centered_face_offset)) {
          return absl::Status(
              absl::StatusCode::kOutOfRange,
              bounded ? "" : "recentered compatible offset is nonfinite"
          );
        }
        const double mass_contribution = transfer.particle_mass * sample.normalized_weight;
        const double normal_velocity_at_face =
            transfer.particle_velocity.*kVelocityComponents[component] + Dot(affine_row, sample.centered_face_offset);
        const double pure_momentum_contribution = mass_contribution * normal_velocity_at_face;
        const double scaled_mass_contribution = mass_contribution / inputs.quantization.mass_per_quantum;
        const double scaled_pure_momentum_contribution =
            pure_momentum_contribution / inputs.quantization.pure_momentum_per_quantum;
        if (!std::isfinite(scaled_mass_contribution) || std::abs(scaled_mass_contribution) > contribution_limit
            || !std::isfinite(scaled_pure_momentum_contribution)
            || std::abs(scaled_pure_momentum_contribution) > contribution_limit) {
          return absl::Status(
              absl::StatusCode::kOutOfRange,
              bounded ? "" : "actual compatible mass or momentum contribution exceeds quanta guard"
          );
        }
        auto& face_quanta_sums = result.face_quanta[sample.face_slot];
        // One contribution per particle/face; the declared power-of-two
        // capacity bound keeps absolute integer sums <=2^62 in both channels.
        const auto quantized_mass_contribution = RoundNearestEven(scaled_mass_contribution);
        face_quanta_sums.mass_quanta_sum += quantized_mass_contribution;
        const auto quantized_pure_momentum_contribution = RoundNearestEven(scaled_pure_momentum_contribution);
        face_quanta_sums.pure_momentum_quanta_sum += quantized_pure_momentum_contribution;
      }
    }
  }
  return absl::OkStatus();
}

template <typename Result>
absl::Status NormalizeCore(
    std::span<const MacFaceDepositQuanta3D> face_quanta,
    MacDepositQuantization3D quantization,
    double minimum_face_mass,
    Result& result,
    bool bounded
) {
  const auto capacity_status = ValidateCapacity(face_quanta.size(), 0, bounded);
  if (!capacity_status.ok()) {
    return capacity_status;
  }
  const auto quantization_status = ValidateQuantization(quantization, 0, bounded);
  if (!quantization_status.ok()) {
    return quantization_status;
  }
  if (!std::isfinite(minimum_face_mass) || minimum_face_mass < 0.0) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        bounded ? "" : "minimum face mass must be finite and nonnegative"
    );
  }
  std::ranges::fill(result.face_masses, 0.0);
  std::ranges::fill(result.face_velocities, 0.0);
  std::ranges::fill(result.validity, LiquidFaceValidity3D::kMissing);
  for (std::size_t slot = 0; slot < face_quanta.size(); ++slot) {
    const auto& quanta = face_quanta[slot];
    if (quanta.mass_quanta_sum < 0 || quanta.mass_quanta_sum > kMaximumQuantaSum
        || quanta.pure_momentum_quanta_sum < -kMaximumQuantaSum || quanta.pure_momentum_quanta_sum > kMaximumQuantaSum
        || quanta.stress_force_quanta_sum != 0) {
      return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "invalid liquid deposit quanta");
    }
    const double decoded_face_mass = static_cast<double>(quanta.mass_quanta_sum) * quantization.mass_per_quantum;
    const double decoded_face_pure_momentum =
        static_cast<double>(quanta.pure_momentum_quanta_sum) * quantization.pure_momentum_per_quantum;
    if (!std::isfinite(decoded_face_mass) || !std::isfinite(decoded_face_pure_momentum)) {
      return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "decoded face deposit is nonfinite");
    }
    result.face_masses[slot] = decoded_face_mass;
    if (decoded_face_mass == 0.0) {
      continue;
    }
    if (decoded_face_mass <= minimum_face_mass) {
      result.validity[slot] = LiquidFaceValidity3D::kLowMass;
      continue;
    }
    const double normalized_face_velocity = decoded_face_pure_momentum / decoded_face_mass;
    if (!std::isfinite(normalized_face_velocity)) {
      return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "normalized face velocity is nonfinite");
    }
    result.face_velocities[slot] = normalized_face_velocity;
    result.validity[slot] = LiquidFaceValidity3D::kValid;
  }
  return absl::OkStatus();
}

template <typename Result, typename Anchor>
absl::Status GatherCore(
    const LiquidBoundaryGatherViewInputs3D& inputs,
    Result& result,
    std::span<const Anchor> anchors,
    std::span<double> filled_old,
    std::span<double> filled_new,
    bool bounded,
    double squared_radius = 4.0
) {
  const auto capacity_status = ValidateCapacity(inputs.mac_grid.face_count(), inputs.particles.size(), bounded);
  if (!capacity_status.ok()) {
    return capacity_status;
  }
  if (!std::isfinite(inputs.old_time) || !std::isfinite(inputs.new_time) || inputs.new_time <= inputs.old_time) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        bounded ? "" : "gather stage times must be finite and increasing"
    );
  }
  const auto old_status = ValidateHistory(inputs.old_history, inputs.mac_grid.face_count(), inputs.old_time, bounded);
  if (!old_status.ok()) {
    return old_status;
  }
  const auto new_status = ValidateHistory(inputs.new_history, inputs.mac_grid.face_count(), inputs.new_time, bounded);
  if (!new_status.ok()) {
    return new_status;
  }
  std::ranges::fill(result.particles, LiquidBoundaryParticleGather3D{});
  std::ranges::fill(filled_old, 0.0);
  std::ranges::fill(filled_new, 0.0);
  for (std::size_t particle_index = 0; particle_index < inputs.particles.size(); ++particle_index) {
    const auto& particle = inputs.particles[particle_index];
    if (!IsFinite(particle.transfer.particle_velocity)) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          bounded ? "" : "retained particle velocity must be finite"
      );
    }
    auto& gathered = result.particles[particle_index];
    Vec3 old_pic_velocity{};
    for (std::size_t component = 0; component < 3; ++component) {
      const auto stencil = BuildOwnedMacFaceStencil3D(
          inputs.mac_grid,
          kAxes[component],
          particle.transfer.sorted_world_position,
          particle.transfer.trial_world_position
      );
      if (!stencil.ok()) {
        return stencil.status();
      }
      for (const auto& sample : *stencil) {
        const auto old_continued_velocity = FillVelocity(
            inputs.old_history,
            anchors,
            sample.face_slot,
            component,
            particle.side_tag,
            bounded,
            squared_radius
        );
        if (!old_continued_velocity.ok()) {
          return old_continued_velocity.status();
        }
        const auto new_continued_velocity = FillVelocity(
            inputs.new_history,
            anchors,
            sample.face_slot,
            component,
            particle.side_tag,
            bounded,
            squared_radius
        );
        if (!new_continued_velocity.ok()) {
          return new_continued_velocity.status();
        }
        filled_old[sample.face_slot] = old_continued_velocity->velocity;
        filled_new[sample.face_slot] = new_continued_velocity->velocity;
        old_pic_velocity.*kVelocityComponents[component] += sample.weight * old_continued_velocity->velocity;
        gathered.old_pic_fallback = gathered.old_pic_fallback || old_continued_velocity->pic_fallback;
        gathered.new_pic_fallback = gathered.new_pic_fallback || new_continued_velocity->pic_fallback;
        gathered.low_mass_fallback = gathered.low_mass_fallback
            || inputs.old_history.validity[sample.face_slot] == LiquidFaceValidity3D::kLowMass
            || inputs.new_history.validity[sample.face_slot] == LiquidFaceValidity3D::kLowMass;
      }
    }
    MacParticleGatherResult3D complete_gather{};
    for (std::size_t component = 0; component < 3; ++component) {
      const auto stencil = BuildOwnedMacFaceStencil3D(
          inputs.mac_grid,
          kAxes[component],
          particle.transfer.sorted_world_position,
          particle.transfer.trial_world_position
      );
      if (!stencil.ok()) {
        return stencil.status();
      }
      const auto gathered_component = internal::GatherMacVelocityComponent(
          *stencil,
          inputs.mac_grid.geometry().cell_width,
          filled_old,
          filled_new,
          bounded
      );
      if (!gathered_component.ok()) {
        return gathered_component.status();
      }
      complete_gather.new_pic_velocity.*kVelocityComponents[component] = gathered_component->new_pic_velocity;
      complete_gather.grid_velocity_increment.*kVelocityComponents[component] =
          gathered_component->grid_velocity_increment;
      complete_gather.fitted_velocity_slopes.*kVelocitySlopeRows[component] =
          gathered_component->fitted_velocity_slope_row;
      complete_gather.velocity_gradient.*kVelocitySlopeRows[component] = gathered_component->velocity_gradient_row;
    }
    gathered.new_pic_velocity = complete_gather.new_pic_velocity;
    gathered.grid_velocity_increment = complete_gather.grid_velocity_increment;
    gathered.fitted_velocity_slopes = complete_gather.fitted_velocity_slopes;
    gathered.velocity_gradient = complete_gather.velocity_gradient;
    const bool pic_fallback = gathered.old_pic_fallback || gathered.new_pic_fallback || gathered.low_mass_fallback;
    gathered.affine_enabled = particle.affine_enabled && !pic_fallback;
    if (!pic_fallback) {
      gathered.particle_minus_old_pic_velocity = Difference(particle.transfer.particle_velocity, old_pic_velocity);
    }
    if (!IsFinite(gathered.particle_minus_old_pic_velocity)) {
      return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "particle retention residual is nonfinite");
    }
    result.old_pic_fallback_particle_count += gathered.old_pic_fallback ? 1 : 0;
    result.new_pic_fallback_particle_count += gathered.new_pic_fallback ? 1 : 0;
    result.low_mass_fallback_particle_count += gathered.low_mass_fallback ? 1 : 0;
  }
  return absl::OkStatus();
}

struct DepositOutput {
  std::span<MacFaceDepositQuanta3D> face_quanta;
  std::span<LiquidCompatibleStencils3D> compatible_stencils;
};
struct NormalizationOutput {
  std::span<double> face_masses;
  std::span<double> face_velocities;
  std::span<LiquidFaceValidity3D> validity;
};
struct GatherOutput {
  std::span<LiquidBoundaryParticleGather3D> particles;
  std::size_t old_pic_fallback_particle_count = 0;
  std::size_t new_pic_fallback_particle_count = 0;
  std::size_t low_mass_fallback_particle_count = 0;
};
template <typename T>
void Resize(LiquidRecords3D<T>& records, std::size_t size) {
  const auto status = records.Resize(size);
  (void)status;
}
}  // namespace
absl::StatusOr<LiquidBoundaryFaceDeposits3D> DepositCompatibleMacQuanta3D(const LiquidBoundaryDepositInputs3D& inputs) {
  if (const auto status = ValidateCapacity(inputs.mac_grid.face_count(), inputs.particles.size(), false);
      !status.ok()) {
    return status;
  }
  if (const auto status = ValidateQuantization(inputs.quantization, inputs.particles.size(), false); !status.ok()) {
    return status;
  }
  LiquidBoundaryFaceDeposits3D result;
  result.face_quanta.resize(inputs.mac_grid.face_count());
  result.compatible_stencils.resize(inputs.particles.size());
  const auto status =
      DepositCore({inputs.mac_grid.view(), inputs.particles, inputs.face_tags, inputs.quantization}, result, false);
  if (!status.ok()) {
    return status;
  }
  return result;
}
absl::StatusOr<NormalizedLiquidMacFaces3D> NormalizeLiquidMacFaces3D(
    std::span<const MacFaceDepositQuanta3D> quanta,
    MacDepositQuantization3D quantization,
    double minimum_face_mass
) {
  if (const auto status = ValidateCapacity(quanta.size(), 0, false); !status.ok()) {
    return status;
  }
  if (const auto status = ValidateQuantization(quantization, 0, false); !status.ok()) {
    return status;
  }
  NormalizedLiquidMacFaces3D result;
  result.face_masses.resize(quanta.size());
  result.face_velocities.resize(quanta.size());
  result.validity.resize(quanta.size());
  const auto status = NormalizeCore(quanta, quantization, minimum_face_mass, result, false);
  if (!status.ok()) {
    return status;
  }
  return result;
}
namespace {

absl::StatusOr<LiquidBoundaryGatherResults3D> GatherBoundaryParticles(
    const LiquidBoundaryGatherInputs3D& inputs,
    std::vector<double>* completed_new_face_velocities,
    double squared_radius
) {
  if (const auto status = ValidateCapacity(inputs.mac_grid.face_count(), inputs.particles.size(), false);
      !status.ok()) {
    return status;
  }
  std::vector<OwnedFaceAnchor> anchors(inputs.mac_grid.face_count());
  BuildOwnedAnchors(inputs.mac_grid.view(), std::span(anchors));
  std::vector<double> old_values(inputs.mac_grid.face_count());
  std::vector<double> new_values(inputs.mac_grid.face_count());
  LiquidBoundaryGatherResults3D result;
  result.particles.resize(inputs.particles.size());
  const LiquidBoundaryGatherViewInputs3D borrowed_inputs{
      .mac_grid = inputs.mac_grid.view(),
      .particles = inputs.particles,
      .old_time = inputs.old_time,
      .new_time = inputs.new_time,
      .old_history = inputs.old_history,
      .new_history = inputs.new_history,
  };
  const auto status = GatherCore(
      borrowed_inputs,
      result,
      std::span<const OwnedFaceAnchor>(anchors),
      old_values,
      new_values,
      false,
      squared_radius
  );
  if (!status.ok()) {
    return status;
  }
  if (completed_new_face_velocities != nullptr) {
    *completed_new_face_velocities = std::move(new_values);
  }
  return result;
}
}  // namespace

absl::StatusOr<LiquidBoundaryGatherResults3D> GatherLiquidBoundaryParticles3D(const LiquidBoundaryGatherInputs3D& inputs
) {
  return GatherBoundaryParticles(inputs, nullptr, 4.0);
}

absl::StatusOr<LiquidBoundaryGatherResults3D> GatherElasticBoundaryParticle3D(
    const LiquidBoundaryGatherInputs3D& inputs,
    std::vector<double>* completed_new_face_velocities
) {
  if (inputs.particles.size() != 1) {
    return absl::InvalidArgumentError("completed face history requires exactly one particle");
  }
  std::vector<double> local_completed;
  auto* completed = completed_new_face_velocities == nullptr ? nullptr : &local_completed;
  auto gathered = GatherBoundaryParticles(inputs, completed, 4.0);
  // Preserve the most local full-rank fit. A blade can remove the nearer
  // donor plane between time samples, leaving only more distant same-side donors.
  for (const double squared_radius : {9.0, 16.0}) {
    const bool needs_wider_fit =
        absl::IsFailedPrecondition(gathered.status()) || (gathered.ok() && !gathered->particles.front().affine_enabled);
    if (!needs_wider_fit) {
      break;
    }
    gathered = GatherBoundaryParticles(inputs, completed, squared_radius);
  }
  if (gathered.ok() && completed_new_face_velocities != nullptr) {
    *completed_new_face_velocities = std::move(local_completed);
  }
  return gathered;
}

absl::StatusOr<LiquidBoundaryFaceDepositsView3D> LiquidBoundaryWorkspace3D::Deposit(
    const LiquidBoundaryDepositViewInputs3D& inputs
) {
  if (inputs.mac_grid.face_count() > capacities_.faces || inputs.particles.size() > capacities_.particles) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  auto& bank = deposits_[1 - deposit_bank_];
  Resize(bank.quanta, inputs.mac_grid.face_count());
  Resize(bank.stencils, inputs.particles.size());
  RecordHighWater();
  DepositOutput output{bank.quanta.values(), bank.stencils.values()};
  const auto status = DepositCore(inputs, output, true);
  if (!status.ok()) {
    return status;
  }
  deposit_bank_ = 1 - deposit_bank_;
  return LiquidBoundaryFaceDepositsView3D{output.face_quanta, output.compatible_stencils};
}
absl::StatusOr<NormalizedLiquidMacFacesView3D> LiquidBoundaryWorkspace3D::Normalize(
    std::span<const MacFaceDepositQuanta3D> quanta,
    MacDepositQuantization3D quantization,
    double minimum_face_mass
) {
  if (quanta.size() > capacities_.faces) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  auto& bank = normalized_[1 - normalization_bank_];
  Resize(bank.masses, quanta.size());
  Resize(bank.velocities, quanta.size());
  Resize(bank.validity, quanta.size());
  RecordHighWater();
  NormalizationOutput output{bank.masses.values(), bank.velocities.values(), bank.validity.values()};
  const auto status = NormalizeCore(quanta, quantization, minimum_face_mass, output, true);
  if (!status.ok()) {
    return status;
  }
  normalization_bank_ = 1 - normalization_bank_;
  return NormalizedLiquidMacFacesView3D{output.face_masses, output.face_velocities, output.validity};
}
absl::StatusOr<LiquidBoundaryGatherView3D> LiquidBoundaryWorkspace3D::Gather(
    const LiquidBoundaryGatherViewInputs3D& inputs
) {
  if (inputs.mac_grid.face_count() > capacities_.faces || inputs.particles.size() > capacities_.particles) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  Resize(anchors_, inputs.mac_grid.face_count());
  Resize(filled_old_, inputs.mac_grid.face_count());
  Resize(filled_new_, inputs.mac_grid.face_count());
  auto& bank = gathered_[1 - gather_bank_];
  Resize(bank, inputs.particles.size());
  RecordHighWater();
  BuildOwnedAnchors(inputs.mac_grid, anchors_.values());
  GatherOutput output{bank.values()};
  const auto status = GatherCore(
      inputs,
      output,
      std::span<const OwnedFaceAnchor>(anchors_.values()),
      filled_old_.values(),
      filled_new_.values(),
      true
  );
  if (!status.ok()) {
    return status;
  }
  gather_bank_ = 1 - gather_bank_;
  return LiquidBoundaryGatherView3D{
      output.particles,
      output.old_pic_fallback_particle_count,
      output.new_pic_fallback_particle_count,
      output.low_mass_fallback_particle_count,
  };
}
absl::StatusOr<double> LiquidBoundaryWorkspace3D::ContinueFace(
    SparseMacGridView3D grid,
    const LiquidBoundaryVelocityHistory3D& history,
    std::size_t face_slot,
    std::int64_t side_tag
) {
  if (grid.face_count() > capacities_.faces) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  if (face_slot >= grid.face_count()) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  const auto status = ValidateHistory(history, grid.face_count(), history.velocity_sample_time, true);
  if (!status.ok()) {
    return status;
  }
  Resize(anchors_, grid.face_count());
  BuildOwnedAnchors(grid, anchors_.values());
  RecordHighWater();
  const auto value = FillVelocity(
      history,
      std::span<const OwnedFaceAnchor>(anchors_.values()),
      face_slot,
      static_cast<std::size_t>(anchors_[face_slot].axis),
      side_tag,
      true
  );
  if (!value.ok()) {
    return value.status();
  }
  return value->velocity;
}
absl::StatusOr<LiquidBoundaryVelocityHistory3D> LiquidBoundaryWorkspace3D::SaveHistory(
    const LiquidBoundaryVelocityHistory3D& history,
    LiquidVelocityTimeLevel3D time_level
) {
  if (time_level != LiquidVelocityTimeLevel3D::kOld && time_level != LiquidVelocityTimeLevel3D::kNew) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto count = history.face_velocities.size();
  if (count > capacities_.faces) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  if (!std::isfinite(history.velocity_sample_time)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto status = ValidateHistory(history, count, history.velocity_sample_time, true);
  if (!status.ok()) {
    return status;
  }
  const auto level = static_cast<std::size_t>(time_level);
  auto& bank = histories_[level][1 - history_bank_[level]];
  Resize(bank.velocities, count);
  Resize(bank.validity, count);
  Resize(bank.tags, count);
  Resize(bank.walls, count);
  RecordHighWater();
  std::ranges::copy(history.face_velocities, bank.velocities.values().begin());
  std::ranges::copy(history.validity, bank.validity.values().begin());
  std::ranges::copy(history.face_tags, bank.tags.values().begin());
  std::ranges::copy(history.wall_samples, bank.walls.values().begin());
  history_bank_[level] = 1 - history_bank_[level];
  return LiquidBoundaryVelocityHistory3D{
      history.velocity_sample_time,
      history.wall_sample_time,
      bank.velocities.values(),
      bank.validity.values(),
      bank.tags.values(),
      bank.walls.values(),
  };
}
}  // namespace fluidity
