#include <cuda_runtime.h>

#include <cuda/atomic>
#include <type_traits>

#include "fluidity/base/liquid_wall_contact_math_3d.h"
#include "liquid_boundary_donor_slots_3d.h"
#include "liquid_boundary_transfer_3d_bridge.h"
namespace fluidity::cuda_internal {
namespace {
namespace numeric = boundary_internal::numeric;
using boundary_internal::Error;
using boundary_internal::Value;
constexpr std::int64_t kMaximumQuantaSum = std::int64_t{1} << 62;
using GatherStencils = decltype(BoundaryTransferParticle3D::stencils);
constexpr std::size_t kGatherComponentCount = std::extent_v<GatherStencils, 0>;
constexpr std::size_t kGatherSamplesPerComponent = std::extent_v<GatherStencils, 1>;
constexpr std::size_t kGatherSampleCount = kGatherComponentCount * kGatherSamplesPerComponent;
constexpr unsigned kSingleParticleGatherThreads = 96;
constexpr unsigned kParallelTransferThreads = 128;
static_assert(kSingleParticleGatherThreads >= kGatherSampleCount);
__device__ FaceNormalAxis Axis(std::size_t i) {
  return i == 0 ? FaceNormalAxis::kX : (i == 1 ? FaceNormalAxis::kY : FaceNormalAxis::kZ);
}
__device__ double Vec3::* VelocityComponent(std::size_t i) {
  return i == 0 ? &Vec3::x : (i == 1 ? &Vec3::y : &Vec3::z);
}
__device__ Vec3 Mat3::* SlopeRow(std::size_t i) {
  return i == 0 ? &Mat3::x_row : (i == 1 ? &Mat3::y_row : &Mat3::z_row);
}
__device__ bool IsFinite(Vec3 value) {
  return internal::TransferFinite(value);
}
__device__ double Dot(Vec3 a, Vec3 b) {
  return (a.x * b.x + a.y * b.y) + a.z * b.z;
}
__device__ Vec3 Difference(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ void AddScaled(Vec3& sum, Vec3 value, double scale) {
  sum.x += value.x * scale;
  sum.y += value.y * scale;
  sum.z += value.z * scale;
}
__device__ bool IsCompatible(LiquidBoundaryFaceTag3D tag, std::int64_t side) {
  return tag.structurally_eligible && tag.side_tag == side;
}
__device__ bool IsUsable(LiquidFaceValidity3D validity) {
  return validity == LiquidFaceValidity3D::kValid || validity == LiquidFaceValidity3D::kPredicted;
}
// Inputs were checked before upload; each history keeps its own tags and wall.
struct HistoryView {
  const BoundaryTransferFace3D* faces;
  BoundaryHistoryStage3D stage;
  __device__ const BoundaryTransferHistory3D& At(std::size_t slot) const {
    return stage == BoundaryHistoryStage3D::kOld ? faces[slot].old_history : faces[slot].new_history;
  }
  __device__ double Velocity(std::size_t slot) const {
    return At(slot).face_velocity;
  }
  __device__ LiquidFaceValidity3D Validity(std::size_t slot) const {
    return At(slot).validity;
  }
  __device__ LiquidBoundaryFaceTag3D Tag(std::size_t slot) const {
    return At(slot).tag;
  }
  __device__ const LiquidPrescribedWallSample3D& Wall(std::size_t slot) const {
    return At(slot).wall;
  }
};
// Guarded magnitude <=2^62 makes conversion defined. Resolve signed ties explicitly.
__device__ std::int64_t RoundNearestEven(double value) {
  const double magnitude = numeric::abs(value);
  const double integral = numeric::floor(magnitude);
  const double fraction = magnitude - integral;
  auto quanta = static_cast<std::int64_t>(integral);
  if (fraction > 0.5 || (fraction == 0.5 && quanta % 2 != 0)) {
    ++quanta;
  }
  return value < 0 ? -quanta : quanta;
}
struct ContinuedVelocity {
  double velocity;
  bool pic_fallback;
};

// Each component is fit independently on its own staggered donors. The same
// target point is used for all components needed by an oblique wall projection.
__device__ Value<ContinuedVelocity> ContinueComponent(
    const HistoryView& history,
    numeric::span<const BoundaryTransferFace3D> anchors,
    Vec3 target_in_cell_widths,
    std::size_t component_index,
    std::int64_t side_tag,
    const LiquidPrescribedWallSample3D& wall
) {
  numeric::array<internal::LiquidGhostFitSample3D, internal::kMaxRadiusTwoGhostDonors3D> donors{};
  std::size_t donor_count = 0;
  std::size_t nearest_slot = anchors.size();
  double nearest_squared_distance = numeric::numeric_limits<double>::infinity();
  DonorFaceRows3D neighborhood(anchors, target_in_cell_widths, component_index);
  DonorFaceRow3D row;
  while (neighborhood.Next(row)) {
    for (std::size_t donor_face_slot = row.first_face_slot; donor_face_slot < row.past_last_face_slot;
         ++donor_face_slot) {
      if (anchors[donor_face_slot].axis != Axis(component_index)
          || !IsCompatible(history.Tag(donor_face_slot), side_tag) || !IsUsable(history.Validity(donor_face_slot))) {
        continue;
      }
      const Vec3 offset = Difference(anchors[donor_face_slot].anchor_in_cell_widths, target_in_cell_widths);
      const double squared_distance = Dot(offset, offset);
      if (squared_distance > 4.0) {
        continue;
      }
      if (donor_count == donors.size()) {
        return Error::kRange;
      }
      donors[donor_count++] = {offset, history.Velocity(donor_face_slot)};
      // Strict comparison keeps the first owned slot on exact distance ties.
      if (squared_distance < nearest_squared_distance) {
        nearest_slot = donor_face_slot;
        nearest_squared_distance = squared_distance;
      }
    }
  }
  const auto fit = internal::FitBoundedLiquidGhostAffineCore3D<internal::kMaxRadiusTwoGhostDonors3D>(
      numeric::span(donors).first(donor_count)
  );
  if (!fit.ok()) {
    return fit.status();
  }
  if (fit->full_rank) {
    return ContinuedVelocity{fit->velocity_at_target, false};
  }
  if (nearest_slot != anchors.size()) {
    return ContinuedVelocity{history.Velocity(nearest_slot), true};
  }
  if (wall.available && wall.side_tag == side_tag) {
    return ContinuedVelocity{wall.wall_velocity.*VelocityComponent(component_index), true};
  }
  return Error::kPrecondition;
}

__device__ Value<ContinuedVelocity> FillVelocity(
    const HistoryView& history,
    numeric::span<const BoundaryTransferFace3D> anchors,
    std::size_t target_slot,
    std::size_t component_index,
    std::int64_t side_tag
) {
  if (IsCompatible(history.Tag(target_slot), side_tag) && IsUsable(history.Validity(target_slot))) {
    return ContinuedVelocity{history.Velocity(target_slot), false};
  }
  const auto& wall = history.Wall(target_slot);
  auto component =
      ContinueComponent(history, anchors, anchors[target_slot].anchor_in_cell_widths, component_index, side_tag, wall);
  if (!component.ok()) {
    return component.status();
  }
  const bool compatible_wall = wall.available && wall.side_tag == side_tag;
  const double normal_component = wall.unit_normal.*VelocityComponent(component_index);
  if (compatible_wall && normal_component != 0.0) {
    Vec3 fitted_vector = wall.wall_velocity;
    fitted_vector.*VelocityComponent(component_index) = component->velocity;
    for (std::size_t other_component = 0; other_component < 3; ++other_component) {
      if (other_component == component_index || wall.unit_normal.*VelocityComponent(other_component) == 0.0) {
        continue;
      }
      const auto other = ContinueComponent(
          history,
          anchors,
          anchors[target_slot].anchor_in_cell_widths,
          other_component,
          side_tag,
          wall
      );
      if (!other.ok()) {
        return other.status();
      }
      fitted_vector.*VelocityComponent(other_component) = other->velocity;
      component->pic_fallback = component->pic_fallback || other->pic_fallback;
    }
    // Only compatible physical affine fields survive this normal replacement.
    // Old/new callers supply their own wall velocity; no wall snapshot is reused.
    const double relative_normal_velocity = Dot(Difference(fitted_vector, wall.wall_velocity), wall.unit_normal);
    component->velocity -= relative_normal_velocity * normal_component;
    if (!numeric::isfinite(component->velocity)) {
      return Error::kRange;
    }
  }
  return component;
}
// Keep the three stencil passes and all floating arithmetic in node order.
template <bool kParallel>
__device__ Error DepositComponent(
    const BoundaryTransferQuery3D& query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D& staged_particle,
    std::size_t component,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferParticleResult3D& particle_result
) {
  const auto& particle = staged_particle.particle;
  const auto& transfer = particle.transfer;
  const auto& stencil = staged_particle.stencils[component];
  const Vec3 affine_row = particle.affine_enabled ? transfer.particle_velocity_slopes.*SlopeRow(component) : Vec3{};
  double compatible_weight_sum = 0.0;
  Vec3 compatible_gradient_sum{};
  for (const auto& sample : stencil) {
    if (IsCompatible(faces[sample.face_slot].deposit_tag, particle.side_tag)) {
      compatible_weight_sum += sample.weight;
      AddScaled(compatible_gradient_sum, sample.weight_gradient, 1.0);
    }
  }
  if (!numeric::isfinite(compatible_weight_sum) || !IsFinite(compatible_gradient_sum)) {
    return Error::kRange;
  }
  if (compatible_weight_sum <= 0.0) {
    return Error::kPrecondition;
  }
  auto& compatible_stencil = particle_result.stencils[component];
  Vec3 weighted_mean_offset{};
  for (std::size_t sample_index = 0; sample_index < 27; ++sample_index) {
    const auto& sample = stencil[sample_index];
    auto& compatible_sample = compatible_stencil[sample_index];
    compatible_sample = {};
    compatible_sample.face_slot = sample.face_slot;
    compatible_sample.compatible = IsCompatible(faces[sample.face_slot].deposit_tag, particle.side_tag);
    if (!compatible_sample.compatible) {
      continue;
    }
    compatible_sample.normalized_weight = sample.weight / compatible_weight_sum;
    const double weight = compatible_sample.normalized_weight;
    compatible_sample.normalized_weight_gradient = {
        (sample.weight_gradient.x - weight * compatible_gradient_sum.x) / compatible_weight_sum,
        (sample.weight_gradient.y - weight * compatible_gradient_sum.y) / compatible_weight_sum,
        (sample.weight_gradient.z - weight * compatible_gradient_sum.z) / compatible_weight_sum,
    };
    if (!numeric::isfinite(weight) || !IsFinite(compatible_sample.normalized_weight_gradient)) {
      return Error::kRange;
    }
    AddScaled(weighted_mean_offset, sample.face_offset_from_particle, weight);
  }
  if (!IsFinite(weighted_mean_offset)) {
    return Error::kRange;
  }
  for (std::size_t sample_index = 0; sample_index < 27; ++sample_index) {
    auto& sample = compatible_stencil[sample_index];
    if (!sample.compatible) {
      continue;
    }
    sample.centered_face_offset = Difference(stencil[sample_index].face_offset_from_particle, weighted_mean_offset);
    if (!IsFinite(sample.centered_face_offset)) {
      return Error::kRange;
    }
    const double mass_contribution = transfer.particle_mass * sample.normalized_weight;
    const double velocity_at_face =
        transfer.particle_velocity.*VelocityComponent(component) + Dot(affine_row, sample.centered_face_offset);
    const double pure_momentum_contribution = mass_contribution * velocity_at_face;
    const double scaled_mass_contribution = mass_contribution / query.quantization.mass_per_quantum;
    const double scaled_pure_momentum_contribution =
        pure_momentum_contribution / query.quantization.pure_momentum_per_quantum;
    if (!numeric::isfinite(scaled_mass_contribution)
        || numeric::abs(scaled_mass_contribution) > query.scaled_contribution_magnitude_limit
        || !numeric::isfinite(scaled_pure_momentum_contribution)
        || numeric::abs(scaled_pure_momentum_contribution) > query.scaled_contribution_magnitude_limit) {
      return Error::kRange;
    }
    auto& face = face_results[sample.face_slot].face_quanta;
    const auto mass_quanta = RoundNearestEven(scaled_mass_contribution);
    const auto pure_momentum_quanta = RoundNearestEven(scaled_pure_momentum_contribution);
    // An owned stencil visits each face at most once per particle. The
    // declared capacity bounds the sum of absolute rounded quanta by 2^62,
    // so every signed partial sum is representable in any interleaving.
    if constexpr (kParallel) {
      using AtomicQuanta = cuda::atomic_ref<std::int64_t, cuda::thread_scope_device>;
      AtomicQuanta(face.mass_quanta_sum).fetch_add(mass_quanta, cuda::memory_order_relaxed);
      AtomicQuanta(face.pure_momentum_quanta_sum).fetch_add(pure_momentum_quanta, cuda::memory_order_relaxed);
    } else {
      face.mass_quanta_sum += mass_quanta;
      face.pure_momentum_quanta_sum += pure_momentum_quanta;
    }
  }
  return Error::kNone;
}
__device__ Error Deposit(
    const BoundaryTransferQuery3D& query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferParticleResult3D* particle_results
) {
  for (std::size_t particle_index = 0; particle_index < query.particle_count; ++particle_index) {
    for (std::size_t component = 0; component < kGatherComponentCount; ++component) {
      const auto error = DepositComponent<false>(
          query,
          faces,
          particles[particle_index],
          component,
          face_results,
          particle_results[particle_index]
      );
      if (error != Error::kNone) {
        return error;
      }
    }
  }
  return Error::kNone;
}
template <typename FaceResult>
__device__ Error
NormalizeFace(const BoundaryTransferQuery3D& query, const MacFaceDepositQuanta3D& quanta, FaceResult& result) {
  if (quanta.mass_quanta_sum < 0 || quanta.mass_quanta_sum > kMaximumQuantaSum
      || quanta.pure_momentum_quanta_sum < -kMaximumQuantaSum || quanta.pure_momentum_quanta_sum > kMaximumQuantaSum
      || quanta.stress_force_quanta_sum != 0) {
    return Error::kInvalid;
  }
  const double face_mass = static_cast<double>(quanta.mass_quanta_sum) * query.quantization.mass_per_quantum;
  const double face_momentum =
      static_cast<double>(quanta.pure_momentum_quanta_sum) * query.quantization.pure_momentum_per_quantum;
  if (!numeric::isfinite(face_mass) || !numeric::isfinite(face_momentum)) {
    return Error::kRange;
  }
  result.decoded_face_mass = face_mass;
  if (face_mass == 0.0) {
    return Error::kNone;
  }
  if (face_mass <= query.minimum_face_mass) {
    result.validity = LiquidFaceValidity3D::kLowMass;
    return Error::kNone;
  }
  const double velocity = face_momentum / face_mass;
  if (!numeric::isfinite(velocity)) {
    return Error::kRange;
  }
  result.normalized_face_velocity = velocity;
  result.validity = LiquidFaceValidity3D::kValid;
  return Error::kNone;
}
__device__ Error Normalize(
    const BoundaryTransferQuery3D& query,
    const BoundaryTransferFace3D* faces,
    BoundaryTransferFaceResult3D* face_results
) {
  for (std::size_t slot = 0; slot < query.face_count; ++slot) {
    const auto error = NormalizeFace(query, faces[slot].face_quanta, face_results[slot]);
    if (error != Error::kNone) {
      return error;
    }
  }
  return Error::kNone;
}

struct GatherSample {
  ContinuedVelocity old_velocity;
  ContinuedVelocity new_velocity;
  Error old_error;
  Error new_error;
};
static_assert(std::is_trivial_v<GatherSample>);
using GatherSamples = GatherSample[kGatherComponentCount][kGatherSamplesPerComponent];

__device__ GatherSample PrepareGatherSample(
    const HistoryView& old_history,
    const HistoryView& new_history,
    numeric::span<const BoundaryTransferFace3D> anchors,
    std::size_t face_slot,
    std::size_t component,
    std::int64_t side_tag,
    bool same_history_samples
) {
  GatherSample prepared{};
  const auto old_velocity = FillVelocity(old_history, anchors, face_slot, component, side_tag);
  prepared.old_error = old_velocity.status();
  if (!old_velocity.ok()) {
    return prepared;
  }
  prepared.old_velocity = *old_velocity;
  const auto new_velocity =
      same_history_samples ? old_velocity : FillVelocity(new_history, anchors, face_slot, component, side_tag);
  prepared.new_error = new_velocity.status();
  if (new_velocity.ok()) {
    prepared.new_velocity = *new_velocity;
  }
  return prepared;
}

// Consume every node, including zero weights, and finish each component before
// checking the next. The same fold handles on-demand and prepared fills.
template <bool kUsePreparedSamples>
__device__ Error GatherParticle(
    const BoundaryTransferQuery3D& query,
    const HistoryView& old_history,
    const HistoryView& new_history,
    numeric::span<const BoundaryTransferFace3D> anchors,
    const BoundaryTransferParticle3D& transfer_particle,
    const GatherSamples* prepared_samples,
    LiquidBoundaryParticleGather3D& result
) {
  const auto& particle = transfer_particle.particle;
  Vec3 old_pic_velocity_vector{};
  for (std::size_t component = 0; component < kGatherComponentCount; ++component) {
    double old_pic_velocity = 0;
    double new_pic_velocity = 0;
    Vec3 weighted_velocity_offset_moment{};
    Vec3 true_velocity_gradient{};
    for (std::size_t sample_index = 0; sample_index < kGatherSamplesPerComponent; ++sample_index) {
      const auto& sample = transfer_particle.stencils[component][sample_index];
      const auto velocities = kUsePreparedSamples ? (*prepared_samples)[component][sample_index]
                                                  : PrepareGatherSample(
                                                        old_history,
                                                        new_history,
                                                        anchors,
                                                        sample.face_slot,
                                                        component,
                                                        particle.side_tag,
                                                        query.same_history_samples
                                                    );
      if (velocities.old_error != Error::kNone) {
        return velocities.old_error;
      }
      if (velocities.new_error != Error::kNone) {
        return velocities.new_error;
      }
      result.old_pic_fallback = result.old_pic_fallback || velocities.old_velocity.pic_fallback;
      result.new_pic_fallback = result.new_pic_fallback || velocities.new_velocity.pic_fallback;
      result.low_mass_fallback = result.low_mass_fallback
          || old_history.Validity(sample.face_slot) == LiquidFaceValidity3D::kLowMass
          || new_history.Validity(sample.face_slot) == LiquidFaceValidity3D::kLowMass;
      old_pic_velocity += sample.weight * velocities.old_velocity.velocity;
      const double weighted_new_velocity = sample.weight * velocities.new_velocity.velocity;
      new_pic_velocity += weighted_new_velocity;
      const Vec3 normalized_offset{
          sample.face_offset_from_particle.x / query.cell_width,
          sample.face_offset_from_particle.y / query.cell_width,
          sample.face_offset_from_particle.z / query.cell_width,
      };
      AddScaled(weighted_velocity_offset_moment, normalized_offset, weighted_new_velocity);
      AddScaled(true_velocity_gradient, sample.weight_gradient, velocities.new_velocity.velocity);
    }
    const Vec3 slopes{
        (4 * weighted_velocity_offset_moment.x) / query.cell_width,
        (4 * weighted_velocity_offset_moment.y) / query.cell_width,
        (4 * weighted_velocity_offset_moment.z) / query.cell_width,
    };
    const double increment = new_pic_velocity - old_pic_velocity;
    if (!numeric::isfinite(new_pic_velocity) || !numeric::isfinite(increment) || !IsFinite(slopes)
        || !IsFinite(true_velocity_gradient)) {
      return Error::kRange;
    }
    result.new_pic_velocity.*VelocityComponent(component) = new_pic_velocity;
    result.grid_velocity_increment.*VelocityComponent(component) = increment;
    result.fitted_velocity_slopes.*SlopeRow(component) = slopes;
    result.velocity_gradient.*SlopeRow(component) = true_velocity_gradient;
    old_pic_velocity_vector.*VelocityComponent(component) = old_pic_velocity;
  }
  const bool fallback = result.old_pic_fallback || result.new_pic_fallback || result.low_mass_fallback;
  result.affine_enabled = particle.affine_enabled && !fallback;
  if (!fallback) {
    result.particle_minus_old_pic_velocity = Difference(particle.transfer.particle_velocity, old_pic_velocity_vector);
  }
  if (!IsFinite(result.particle_minus_old_pic_velocity)) {
    return Error::kRange;
  }
  return Error::kNone;
}

__device__ Error Gather(
    const BoundaryTransferQuery3D& query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryTransferParticleResult3D* particle_results
) {
  const HistoryView old_history{faces, BoundaryHistoryStage3D::kOld};
  const HistoryView new_history{faces, BoundaryHistoryStage3D::kNew};
  const numeric::span<const BoundaryTransferFace3D> anchors(faces, query.face_count);
  for (std::size_t particle_index = 0; particle_index < query.particle_count; ++particle_index) {
    const Error error = GatherParticle<false>(
        query,
        old_history,
        new_history,
        anchors,
        particles[particle_index],
        nullptr,
        particle_results[particle_index].gathered
    );
    if (error != Error::kNone) {
      return error;
    }
  }
  return Error::kNone;
}
// The same continuation used by gather predicts a missing pressure face. Acceleration
// follows the old-history continuation exactly once and never changes old validity.
__device__ Error
Predict(const BoundaryTransferQuery3D& query, const BoundaryTransferFace3D* faces, BoundaryTransferReport3D& report) {
  const auto axis = faces[query.target_face_slot].axis;
  const std::size_t component = axis == FaceNormalAxis::kX ? 0 : (axis == FaceNormalAxis::kY ? 1 : 2);
  const auto continued = FillVelocity(
      {faces, BoundaryHistoryStage3D::kOld},
      numeric::span<const BoundaryTransferFace3D>(faces, query.face_count),
      query.target_face_slot,
      component,
      query.side_tag
  );
  if (!continued.ok()) {
    return continued.status();
  }
  report.predicted_face_velocity =
      continued->velocity + query.step_duration * query.world_acceleration.*VelocityComponent(component);
  return numeric::isfinite(report.predicted_face_velocity) ? Error::kNone : Error::kRange;
}
__device__ Error ApplyForcesToFace(
    const BoundaryTransferQuery3D& query,
    const BoundaryTransferFace3D& face,
    BoundaryTransferFaceResult3D& result
) {
  const auto axis = face.axis;
  const std::size_t component = axis == FaceNormalAxis::kX ? 0 : (axis == FaceNormalAxis::kY ? 1 : 2);
  const double predicted =
      face.old_history.face_velocity + query.step_duration * query.world_acceleration.*VelocityComponent(component);
  if (!numeric::isfinite(predicted)) {
    return Error::kRange;
  }
  result.normalized_face_velocity = predicted;
  return Error::kNone;
}

__device__ Error Forces(
    const BoundaryTransferQuery3D& query,
    const BoundaryTransferFace3D* faces,
    BoundaryTransferFaceResult3D* results
) {
  for (std::size_t slot = 0; slot < query.face_count; ++slot) {
    const auto error = ApplyForcesToFace(query, faces[slot], results[slot]);
    if (error != Error::kNone) {
      return error;
    }
  }
  return Error::kNone;
}
// J uses the divergence of the selected advection field. Contact constrains the
// advecting and retained velocities independently.
__global__ void LiquidParticleUpdate3DKernel(
    const BoundaryTransferQuery3D* device_query,
    const BoundaryUpdateParticle3D* particles,
    BoundaryUpdateResult3D* results,
    int* compiled_architecture
) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__)
  if (index == 0) {
    *compiled_architecture = __CUDA_ARCH__;
  }
#endif
  const auto& query = *device_query;
  if (index >= query.particle_count) {
    return;
  }
  auto& result = results[index];
  result = {};
  const auto& particle = particles[index];
  const auto& gathered = particle.gathered;
  const Vec3 start = particle.world_position;
  const bool fallback = gathered.old_pic_fallback || gathered.new_pic_fallback || gathered.low_mass_fallback;
  const double alpha = particle.pic_initialized && !fallback ? 0.99 * query.particle_energy_blend : 0;
  Vec3 stored = gathered.new_pic_velocity;
  AddScaled(stored, gathered.particle_minus_old_pic_velocity, alpha);
  Vec3 advecting;
  double divergence;
  if (particle.regular_cell_transport.active) {
    const auto field = internal::EvaluateRegularCellTransport3D(particle.regular_cell_transport, query.cell_width);
    if (!field.ok()) {
      result.error = field.status();
      return;
    }
    advecting = field->advecting_world_velocity;
    divergence = field->divergence;
  } else {
    divergence =
        gathered.velocity_gradient.x_row.x + gathered.velocity_gradient.y_row.y + gathered.velocity_gradient.z_row.z;
    double solid_distance = numeric::numeric_limits<double>::infinity();
    if (query.obstacle.kind == boundary_internal::ObstacleKind::kOpenTank) {
      const auto wall = boundary_internal::SampleTankWall(query.obstacle.tank, start);
      if (!wall.ok()) {
        result.error = wall.status();
        return;
      }
      solid_distance = wall->inward_distance;
    } else if (query.obstacle.kind != boundary_internal::ObstacleKind::kAbsent) {
      const double projected_distance = Dot(query.obstacle.normal, start) - query.obstacle.coordinate;
      solid_distance = query.obstacle.kind == boundary_internal::ObstacleKind::kHalfspace
          ? projected_distance
          : numeric::abs(projected_distance) - query.obstacle.half_width;
      if (!numeric::isfinite(solid_distance)) {
        result.error = Error::kRange;
        return;
      }
    }
    const bool surface_expansion = divergence > 0 && particle.liquid_distance <= 0
        && particle.liquid_distance >= -query.cell_width && solid_distance > query.cell_width;
    const double beta = surface_expansion ? query.particle_energy_blend : 0;
    advecting = gathered.new_pic_velocity;
    AddScaled(advecting, gathered.particle_minus_old_pic_velocity, beta * alpha);
  }
  const auto contact = boundary_internal::ResolveContact(
      {
          start,
          stored,
          advecting,
          query.obstacle,
          query.step_duration,
          query.friction_coefficient,
          query.penetration_tolerance,
      },
      nullptr,
      query.operation == BoundaryTransferOperation3D::kParticleUpdateWithPistonReaction
          ? &result.piston_stored_velocity_change
          : nullptr
  );
  if (!contact.ok()) {
    result.error = contact.status();
    return;
  }
  const Vec3 displacement = Difference(contact->world_position, start);
  const double volume_ratio = particle.volume_ratio * exp(query.step_duration * divergence);
  if (numeric::abs(displacement.x) > 0.5 * query.cell_width || numeric::abs(displacement.y) > 0.5 * query.cell_width
      || numeric::abs(displacement.z) > 0.5 * query.cell_width || !numeric::isfinite(volume_ratio)
      || volume_ratio <= 0) {
    result.retry_step = true;
    return;
  }
  result.world_position = contact->world_position;
  result.stored_world_velocity = contact->stored_world_velocity;
  result.volume_ratio = volume_ratio;
}
__device__ void InitializeTransferOutputs(
    const BoundaryTransferQuery3D& query,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferParticleResult3D* particle_results,
    BoundaryTransferReport3D& report
) {
  report = {};
#if defined(__CUDA_ARCH__)
  report.compiled_architecture = __CUDA_ARCH__;
#endif
  for (std::size_t slot = 0; slot < query.face_count; ++slot) {
    face_results[slot] = {};
  }
  for (std::size_t particle_index = 0; particle_index < query.particle_count; ++particle_index) {
    particle_results[particle_index] = {};
  }
}

__global__ void LiquidBoundaryTransfer3DKernel(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferParticleResult3D* particle_results,
    BoundaryTransferReport3D* report
) {
  InitializeTransferOutputs(*query, face_results, particle_results, *report);
  switch (query->operation) {
    case BoundaryTransferOperation3D::kDeposit:
      report->error = Deposit(*query, faces, particles, face_results, particle_results);
      break;
    case BoundaryTransferOperation3D::kNormalize:
      report->error = Normalize(*query, faces, face_results);
      break;
    case BoundaryTransferOperation3D::kGather:
      report->error = Gather(*query, faces, particles, particle_results);
      break;
    case BoundaryTransferOperation3D::kPredict:
      report->error = Predict(*query, faces, *report);
      break;
    case BoundaryTransferOperation3D::kForces:
      report->error = Forces(*query, faces, face_results);
      break;
    case BoundaryTransferOperation3D::kParticleUpdateWithPistonReaction:
      report->error = Error::kInvalid;
      break;
    case BoundaryTransferOperation3D::kFitGhost: {
      const auto fit = internal::FitLiquidGhostAffineCore3D(numeric::span(query->fit_samples, query->fit_count));
      report->error = fit.status();
      if (fit.ok()) {
        report->fit = *fit;
      }
      break;
    }
  }
}

// Only the earliest failing ordinal can win. Threads write no published data.
__device__ void ReportParallelTransferError(BoundaryTransferReport3D& report, std::size_t ordinal, Error error) {
#if defined(__CUDA_ARCH__)
  if (ordinal == 0) {
    report.compiled_architecture = __CUDA_ARCH__;
  }
#endif
  if (error != Error::kNone) {
    const auto ordered_error = ordinal * kBoundaryTransferErrorCodeCount + static_cast<unsigned>(error);
    atomicMin(&report.first_parallel_error, static_cast<unsigned long long>(ordered_error));
  }
}

__global__ void LiquidBoundaryDeposit3DKernel(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferParticleResult3D* particle_results,
    BoundaryTransferReport3D* report
) {
  const std::size_t ordinal = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ordinal >= query->particle_count * kGatherComponentCount) {
    return;
  }
  const auto particle_index = ordinal / kGatherComponentCount;
  const auto component = ordinal % kGatherComponentCount;
  const auto error = DepositComponent<true>(
      *query,
      faces,
      particles[particle_index],
      component,
      face_results,
      particle_results[particle_index]
  );
  ReportParallelTransferError(*report, ordinal, error);
}

__global__ void LiquidBoundaryNormalize3DKernel(
    const BoundaryTransferQuery3D* query,
    const MacFaceDepositQuanta3D* face_quanta,
    BoundaryNormalizedFace3D* face_results,
    BoundaryTransferReport3D* report
) {
  const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= query->face_count) {
    return;
  }
  auto& result = face_results[slot];
  result = {};
  const auto error = NormalizeFace(*query, face_quanta[slot], result);
  ReportParallelTransferError(*report, slot, error);
}

__global__ void LiquidBoundaryFaceMap3DKernel(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferReport3D* report
) {
  const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= query->face_count) {
    return;
  }
  auto& result = face_results[slot];
  result = {};
  const auto error = ApplyForcesToFace(*query, faces[slot], result);
  ReportParallelTransferError(*report, slot, error);
}

template <bool kGatherRun, typename ParticleResult>
__global__ void LiquidBoundaryParticleGather3DKernel(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryTransferFaceResult3D* face_results,
    ParticleResult* particle_results,
    BoundaryTransferReport3D* report
) {
  const std::size_t particle_index = kGatherRun ? blockIdx.x : 0;
  const auto& particle = particles[particle_index];
  const HistoryView old_history{faces, BoundaryHistoryStage3D::kOld};
  const HistoryView new_history{faces, BoundaryHistoryStage3D::kNew};
  const numeric::span<const BoundaryTransferFace3D> anchors(faces, query->face_count);
  __shared__ GatherSamples prepared_samples;
  const std::size_t sample_lane = threadIdx.x;
  // Fills read immutable histories independently. One consumer keeps the
  // original accumulation and error order, including component range checks.
  if (sample_lane < kGatherSampleCount) {
    const std::size_t component = sample_lane / kGatherSamplesPerComponent;
    const std::size_t sample_index = sample_lane % kGatherSamplesPerComponent;
    const auto& sample = particle.stencils[component][sample_index];
    prepared_samples[component][sample_index] = PrepareGatherSample(
        old_history,
        new_history,
        anchors,
        sample.face_slot,
        component,
        particle.particle.side_tag,
        query->same_history_samples
    );
  }
  __syncthreads();
  if (sample_lane == 0) {
    auto& result = particle_results[particle_index];
    if constexpr (kGatherRun) {
      result = {};
#if defined(__CUDA_ARCH__)
      result.compiled_architecture = __CUDA_ARCH__;
#endif
    } else {
      InitializeTransferOutputs(*query, face_results, particle_results, *report);
    }
    const auto error =
        GatherParticle<true>(*query, old_history, new_history, anchors, particle, &prepared_samples, result.gathered);
    if constexpr (kGatherRun) {
      result.error = error;
    } else {
      report->error = error;
    }
  }
}
}  // namespace
int LaunchLiquidParticleUpdate3D(
    const BoundaryTransferQuery3D* query,
    const BoundaryUpdateParticle3D* particles,
    BoundaryUpdateResult3D* results,
    int* compiled_architecture,
    std::size_t particle_count,
    void* stream
) {
  constexpr unsigned threads = 128;
  const auto block_count = particle_count == 0 ? 1U : static_cast<unsigned>((particle_count + threads - 1) / threads);
  cudaGetLastError();
  LiquidParticleUpdate3DKernel<<<block_count, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      query,
      particles,
      results,
      compiled_architecture
  );
  return static_cast<int>(cudaGetLastError());
}
int LaunchLiquidBoundaryTransfer3D(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferParticleResult3D* particle_results,
    BoundaryTransferReport3D* report,
    void* stream
) {
  cudaGetLastError();
  LiquidBoundaryTransfer3DKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      query,
      faces,
      particles,
      face_results,
      particle_results,
      report
  );
  return static_cast<int>(cudaGetLastError());
}

int LaunchLiquidBoundaryDeposit3D(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferParticleResult3D* particle_results,
    BoundaryTransferReport3D* report,
    std::size_t particle_count,
    void* stream
) {
  const auto component_count = particle_count * kGatherComponentCount;
  const auto blocks =
      static_cast<unsigned>((component_count + kParallelTransferThreads - 1) / kParallelTransferThreads);
  cudaGetLastError();
  LiquidBoundaryDeposit3DKernel<<<blocks, kParallelTransferThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      query,
      faces,
      particles,
      face_results,
      particle_results,
      report
  );
  return static_cast<int>(cudaGetLastError());
}

int LaunchLiquidBoundaryNormalize3D(
    const BoundaryTransferQuery3D* query,
    const MacFaceDepositQuanta3D* face_quanta,
    BoundaryNormalizedFace3D* face_results,
    BoundaryTransferReport3D* report,
    std::size_t face_count,
    void* stream
) {
  const auto blocks = static_cast<unsigned>((face_count + kParallelTransferThreads - 1) / kParallelTransferThreads);
  cudaGetLastError();
  LiquidBoundaryNormalize3DKernel<<<blocks, kParallelTransferThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      query,
      face_quanta,
      face_results,
      report
  );
  return static_cast<int>(cudaGetLastError());
}

int LaunchLiquidBoundaryFaceMap3D(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferReport3D* report,
    std::size_t face_count,
    void* stream
) {
  const auto blocks = static_cast<unsigned>((face_count + kParallelTransferThreads - 1) / kParallelTransferThreads);
  cudaGetLastError();
  LiquidBoundaryFaceMap3DKernel<<<blocks, kParallelTransferThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      query,
      faces,
      face_results,
      report
  );
  return static_cast<int>(cudaGetLastError());
}

int LaunchLiquidBoundarySingleParticleGather3D(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferParticleResult3D* particle_results,
    BoundaryTransferReport3D* report,
    void* stream
) {
  cudaGetLastError();
  LiquidBoundaryParticleGather3DKernel<false>
      <<<1, kSingleParticleGatherThreads, 0, static_cast<cudaStream_t>(stream)>>>(
          query,
          faces,
          particles,
          face_results,
          particle_results,
          report
      );
  return static_cast<int>(cudaGetLastError());
}
int LaunchLiquidBoundaryGatherRun3D(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryGatherRunResult3D* results,
    unsigned particle_count,
    void* stream
) {
  cudaGetLastError();
  LiquidBoundaryParticleGather3DKernel<true>
      <<<particle_count, kSingleParticleGatherThreads, 0, static_cast<cudaStream_t>(stream)>>>(
          query,
          faces,
          particles,
          nullptr,
          results,
          nullptr
      );
  return static_cast<int>(cudaGetLastError());
}
}  // namespace fluidity::cuda_internal
