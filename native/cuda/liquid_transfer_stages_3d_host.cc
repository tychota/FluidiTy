#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

#include "../src/liquid_boundary_backend.h"
#include "liquid_arena_dispatch_3d.h"
#include "liquid_boundary_transfer_3d_bridge.h"
#include "liquid_transfer_stages_3d.h"
namespace fluidity::boundary_internal {
// Only completed CUDA results enter candidate banks. A publication flips the same
// region as its CPU counterpart, so a later failed stage preserves the prior view.
struct BoundedTransferAccess {
  static absl::Status
  Check(const LiquidBoundaryWorkspace3D& workspace, std::size_t face_count, std::size_t particle_count) {
    if (face_count > workspace.capacities_.faces || particle_count > workspace.capacities_.particles) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    return absl::OkStatus();
  }
  static LiquidBoundaryFaceDepositsView3D Deposit(
      LiquidBoundaryWorkspace3D& workspace,
      std::span<const cuda_internal::BoundaryTransferFaceResult3D> faces,
      std::span<const cuda_internal::BoundaryTransferParticleResult3D> particles
  ) {
    auto& bank = workspace.deposits_[1 - workspace.deposit_bank_];
    (void)bank.quanta.Resize(faces.size());
    (void)bank.stencils.Resize(particles.size());
    for (std::size_t slot = 0; slot < faces.size(); ++slot) {
      bank.quanta[slot] = faces[slot].face_quanta;
    }
    for (std::size_t particle = 0; particle < particles.size(); ++particle) {
      for (std::size_t component = 0; component < 3; ++component) {
        std::copy_n(particles[particle].stencils[component], 27, bank.stencils[particle][component].begin());
      }
    }
    workspace.deposit_bank_ = 1 - workspace.deposit_bank_;
    workspace.RecordHighWater();
    return {bank.quanta.values(), bank.stencils.values()};
  }
  static NormalizedLiquidMacFacesView3D
  Normalize(LiquidBoundaryWorkspace3D& workspace, std::span<const cuda_internal::BoundaryTransferFaceResult3D> faces) {
    auto& bank = workspace.normalized_[1 - workspace.normalization_bank_];
    (void)bank.masses.Resize(faces.size());
    (void)bank.velocities.Resize(faces.size());
    (void)bank.validity.Resize(faces.size());
    for (std::size_t slot = 0; slot < faces.size(); ++slot) {
      bank.masses[slot] = faces[slot].decoded_face_mass;
      bank.velocities[slot] = faces[slot].normalized_face_velocity;
      bank.validity[slot] = faces[slot].validity;
    }
    workspace.normalization_bank_ = 1 - workspace.normalization_bank_;
    workspace.RecordHighWater();
    return {bank.masses.values(), bank.velocities.values(), bank.validity.values()};
  }
  static LiquidBoundaryGatherView3D Gather(
      LiquidBoundaryWorkspace3D& workspace,
      std::span<const cuda_internal::BoundaryTransferParticleResult3D> particles
  ) {
    auto& bank = workspace.gathered_[1 - workspace.gather_bank_];
    (void)bank.Resize(particles.size());
    LiquidBoundaryGatherView3D result{bank.values()};
    for (std::size_t index = 0; index < particles.size(); ++index) {
      bank[index] = particles[index].gathered;
      result.old_pic_fallback_particle_count += bank[index].old_pic_fallback ? 1 : 0;
      result.new_pic_fallback_particle_count += bank[index].new_pic_fallback ? 1 : 0;
      result.low_mass_fallback_particle_count += bank[index].low_mass_fallback ? 1 : 0;
    }
    workspace.gather_bank_ = 1 - workspace.gather_bank_;
    workspace.RecordHighWater();
    return result;
  }
};
}  // namespace fluidity::boundary_internal
namespace fluidity::cuda_internal {
namespace {
using Access = boundary_internal::BoundedTransferAccess;
using Range = CudaLiquidRange3D;
#define TRANSFER_TRY(expression) \
  do { \
    const auto status = (expression); \
    if (!status.ok()) { \
      return status; \
    } \
  } while (false)
absl::Status Error(absl::StatusCode code) {
  return absl::Status(code, {});
}
constexpr std::array kAxes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
constexpr std::array kRows{&Mat3::x_row, &Mat3::y_row, &Mat3::z_row};
bool IsFinite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double Dot(Vec3 lhs, Vec3 rhs) {
  return (lhs.x * rhs.x + lhs.y * rhs.y) + lhs.z * rhs.z;
}

absl::Status ValidateCapacity(std::size_t face_count, std::size_t particle_count) {
  if (face_count > kLiquidBoundaryFaceCapacity3D || particle_count > kLiquidBoundaryParticleCapacity3D) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  return absl::OkStatus();
}

absl::Status ValidateQuantization(MacDepositQuantization3D quantization, std::size_t particle_count) {
  for (double scale : {quantization.mass_per_quantum, quantization.pure_momentum_per_quantum}) {
    int exponent = 0;
    if (!std::isnormal(scale) || std::frexp(scale, &exponent) != 0.5) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  if (quantization.declared_particle_capacity > kLiquidBoundaryParticleCapacity3D
      || particle_count > quantization.declared_particle_capacity) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  return absl::OkStatus();
}

bool IsUsable(LiquidFaceValidity3D validity) {
  return validity == LiquidFaceValidity3D::kValid || validity == LiquidFaceValidity3D::kPredicted;
}

absl::Status
ValidateHistory(const LiquidBoundaryVelocityHistory3D& history, std::size_t face_count, double stage_time) {
  if (!std::isfinite(stage_time) || history.velocity_sample_time != stage_time
      || history.wall_sample_time != stage_time) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (history.face_velocities.size() != face_count || history.validity.size() != face_count
      || history.face_tags.size() != face_count || history.wall_samples.size() != face_count) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  for (std::size_t slot = 0; slot < face_count; ++slot) {
    const auto validity = history.validity[slot];
    if (validity != LiquidFaceValidity3D::kMissing && validity != LiquidFaceValidity3D::kLowMass
        && !IsUsable(validity)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    if (IsUsable(validity) && !std::isfinite(history.face_velocities[slot])) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const auto& wall = history.wall_samples[slot];
    if (wall.available) {
      if (!IsFinite(wall.wall_velocity) || !IsFinite(wall.unit_normal)) {
        return absl::Status(absl::StatusCode::kInvalidArgument, {});
      }
      if (std::abs(Dot(wall.unit_normal, wall.unit_normal) - 1.0) > 1e-12) {
        return absl::Status(absl::StatusCode::kInvalidArgument, {});
      }
    }
  }
  return absl::OkStatus();
}

struct StagedTransfer {
  LiquidTypedRange3D<BoundaryTransferQuery3D> query_range;
  LiquidTypedRange3D<BoundaryTransferFace3D> face_range;
  LiquidTypedRange3D<BoundaryTransferParticle3D> particle_range;
  LiquidTypedRange3D<BoundaryTransferFaceResult3D> face_result_range;
  LiquidTypedRange3D<BoundaryTransferParticleResult3D> particle_result_range;
  LiquidTypedRange3D<BoundaryTransferReport3D> report_range;
  std::span<BoundaryTransferFace3D> faces;
  std::span<BoundaryTransferParticle3D> particles;
};
absl::StatusOr<StagedTransfer> Stage(CudaLiquidArena3D& arena, std::size_t face_count, std::size_t particle_count) {
  TRANSFER_TRY(ValidateCapacity(face_count, particle_count));
  if (face_count > arena.capacities().faces || particle_count > arena.capacities().particles) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  const auto query = CheckedLiquidRange3D<BoundaryTransferQuery3D>(arena, Range::kTransferQuery, 1);
  const auto faces = CheckedLiquidRange3D<BoundaryTransferFace3D>(arena, Range::kTransferFaces, face_count);
  const auto particles =
      CheckedLiquidRange3D<BoundaryTransferParticle3D>(arena, Range::kTransferParticles, particle_count);
  const auto face_results =
      CheckedLiquidRange3D<BoundaryTransferFaceResult3D>(arena, Range::kTransferFaceResults, face_count);
  const auto particle_results =
      CheckedLiquidRange3D<BoundaryTransferParticleResult3D>(arena, Range::kTransferParticleResults, particle_count);
  const auto report = CheckedLiquidRange3D<BoundaryTransferReport3D>(arena, Range::kTransferReport, 1);
  if (!query.ok()) {
    return query.status();
  }
  if (!faces.ok()) {
    return faces.status();
  }
  if (!particles.ok()) {
    return particles.status();
  }
  if (!face_results.ok()) {
    return face_results.status();
  }
  if (!particle_results.ok()) {
    return particle_results.status();
  }
  if (!report.ok()) {
    return report.status();
  }
  query->pinned[0] = {};
  query->pinned[0].face_count = face_count;
  query->pinned[0].particle_count = particle_count;
  std::fill(faces->pinned.begin(), faces->pinned.end(), BoundaryTransferFace3D{});
  std::fill(particles->pinned.begin(), particles->pinned.end(), BoundaryTransferParticle3D{});
  return StagedTransfer{
      *query,
      *faces,
      *particles,
      *face_results,
      *particle_results,
      *report,
      faces->pinned,
      particles->pinned,
  };
}
// Never touch pinned storage after a failed asynchronous submission/fence. Every
// CUDA result poisons the arena first; arithmetic rejection is separately reusable.
absl::Status Execute(CudaLiquidArena3D& arena, StagedTransfer& staged) {
  const auto& query = staged.query_range.pinned[0];
  const bool parallel_deposit = query.operation == BoundaryTransferOperation3D::kDeposit && query.particle_count > 0;
  const bool parallel_normalize = query.operation == BoundaryTransferOperation3D::kNormalize && query.face_count > 0;
  if (parallel_deposit || parallel_normalize) {
    staged.report_range.pinned[0] = {};
  }
  TRANSFER_TRY(CopyLiquidRange3D(arena, staged.query_range, true));
  TRANSFER_TRY(CopyLiquidRange3D(arena, staged.face_range, true));
  TRANSFER_TRY(CopyLiquidRange3D(arena, staged.particle_range, true));
  if (parallel_deposit || parallel_normalize) {
    TRANSFER_TRY(CopyLiquidRange3D(arena, staged.report_range, true));
  }
  if (parallel_deposit) {
    TRANSFER_TRY(arena.RecordRuntimeResult(LiquidArenaRuntimeCalls().memset_async(
        staged.face_result_range.device,
        0,
        staged.face_result_range.pinned.size_bytes(),
        static_cast<cudaStream_t>(arena.stream())
    )));
    TRANSFER_TRY(arena.RecordRuntimeResult(LaunchLiquidBoundaryDeposit3D(
        staged.query_range.device,
        staged.face_range.device,
        staged.particle_range.device,
        staged.face_result_range.device,
        staged.particle_result_range.device,
        staged.report_range.device,
        query.particle_count,
        arena.stream()
    )));
  } else if (parallel_normalize) {
    TRANSFER_TRY(arena.RecordRuntimeResult(LaunchLiquidBoundaryNormalize3D(
        staged.query_range.device,
        staged.face_range.device,
        staged.face_result_range.device,
        staged.report_range.device,
        query.face_count,
        arena.stream()
    )));
  } else {
    const bool single_particle_gather =
        query.operation == BoundaryTransferOperation3D::kGather && query.particle_count == 1;
    const auto launch =
        single_particle_gather ? LaunchLiquidBoundarySingleParticleGather3D : LaunchLiquidBoundaryTransfer3D;
    TRANSFER_TRY(arena.RecordRuntimeResult(launch(
        staged.query_range.device,
        staged.face_range.device,
        staged.particle_range.device,
        staged.face_result_range.device,
        staged.particle_result_range.device,
        staged.report_range.device,
        arena.stream()
    )));
  }
  TRANSFER_TRY(CopyLiquidRange3D(arena, staged.face_result_range, false));
  TRANSFER_TRY(CopyLiquidRange3D(arena, staged.particle_result_range, false));
  TRANSFER_TRY(CopyLiquidRange3D(arena, staged.report_range, false));
  TRANSFER_TRY(FinishLiquidDispatch3D(arena));
  const auto& report = staged.report_range.pinned[0];
  if (report.compiled_architecture != 1200) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  if (parallel_deposit || parallel_normalize) {
    if (report.first_parallel_error == kNoParallelTransferError) {
      return absl::OkStatus();
    }
    const auto error =
        static_cast<boundary_internal::Error>(report.first_parallel_error % kBoundaryTransferErrorCodeCount);
    return boundary_internal::ArithmeticStatus(error, true);
  }
  return boundary_internal::ArithmeticStatus(report.error, true);
}
absl::Status
CheckHistory(CudaLiquidArena3D& arena, const LiquidBoundaryVelocityHistory3D& history, std::size_t count, double time) {
  TRANSFER_TRY(ValidateHistory(history, count, time));
  if (AliasesLiquidScratch3D(arena, history.face_velocities) || AliasesLiquidScratch3D(arena, history.validity)
      || AliasesLiquidScratch3D(arena, history.face_tags) || AliasesLiquidScratch3D(arena, history.wall_samples)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  return absl::OkStatus();
}
absl::Status StageAnchors(SparseMacGridView3D grid, StagedTransfer& staged) {
  for (std::size_t slot = 0; slot < staged.faces.size(); ++slot) {
    const auto face = grid.FaceIndex(slot);
    if (!face.ok()) {
      return face.status();
    }
    staged.faces[slot].axis = face->axis;
    staged.faces[slot].anchor_in_cell_widths = {
        face->lattice_index.x + (face->axis == FaceNormalAxis::kX ? 0 : 0.5),
        face->lattice_index.y + (face->axis == FaceNormalAxis::kY ? 0 : 0.5),
        face->lattice_index.z + (face->axis == FaceNormalAxis::kZ ? 0 : 0.5),
    };
  }
  return absl::OkStatus();
}
void StageHistory(StagedTransfer& staged, const LiquidBoundaryVelocityHistory3D& history, bool old) {
  for (std::size_t slot = 0; slot < staged.faces.size(); ++slot) {
    auto& destination = old ? staged.faces[slot].old_history : staged.faces[slot].new_history;
    destination =
        {history.face_velocities[slot], history.validity[slot], history.face_tags[slot], history.wall_samples[slot]};
  }
}
// Resolve every component/node before compatibility filtering, preserving slot order.
absl::Status StageParticles(
    SparseMacGridView3D grid,
    std::span<const LiquidBoundaryParticle3D> particles,
    StagedTransfer& staged,
    bool deposit
) {
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const auto& particle = particles[index];
    if (!IsFinite(particle.transfer.particle_velocity)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    if (deposit) {
      if (!std::isfinite(particle.transfer.particle_mass) || particle.transfer.particle_mass < 0) {
        return absl::Status(absl::StatusCode::kInvalidArgument, {});
      }
      if (particle.affine_enabled) {
        for (auto row : kRows) {
          if (!IsFinite(particle.transfer.particle_velocity_slopes.*row)) {
            return absl::Status(absl::StatusCode::kInvalidArgument, {});
          }
        }
      }
    }
    staged.particles[index].particle = particle;
    for (std::size_t component = 0; component < 3; ++component) {
      const auto stencil = BuildOwnedMacFaceStencil3D(
          grid,
          kAxes[component],
          particle.transfer.sorted_world_position,
          particle.transfer.trial_world_position
      );
      if (!stencil.ok()) {
        return stencil.status();
      }
      std::copy(stencil->begin(), stencil->end(), staged.particles[index].stencils[component]);
    }
  }
  return absl::OkStatus();
}

}  // namespace
absl::StatusOr<LiquidBoundaryFaceDepositsView3D> DepositLiquid3D(
    CudaLiquidArena3D& arena,
    const LiquidBoundaryDepositViewInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace
) {
  TRANSFER_TRY(arena.RecordRuntimeResult(cudaSuccess));
  TRANSFER_TRY(Access::Check(workspace, inputs.mac_grid.face_count(), inputs.particles.size()));
  if (AliasesLiquidScratch3D(arena, inputs.mac_grid.active_block_indices())) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  TRANSFER_TRY(ValidateQuantization(inputs.quantization, inputs.particles.size()));
  if (inputs.face_tags.size() != inputs.mac_grid.face_count() || AliasesLiquidScratch3D(arena, inputs.face_tags)
      || AliasesLiquidScratch3D(arena, inputs.particles)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  auto staged = Stage(arena, inputs.mac_grid.face_count(), inputs.particles.size());
  if (!staged.ok()) {
    return staged.status();
  }
  auto& query = staged->query_range.pinned[0];
  query.quantization = inputs.quantization;
  const auto capacity = std::max(std::size_t{1}, inputs.quantization.declared_particle_capacity);
  query.scaled_contribution_magnitude_limit = std::ldexp(1.0, 62 - static_cast<int>(std::bit_width(capacity - 1)));
  for (std::size_t slot = 0; slot < staged->faces.size(); ++slot) {
    staged->faces[slot].deposit_tag = inputs.face_tags[slot];
  }
  TRANSFER_TRY(StageParticles(inputs.mac_grid, inputs.particles, *staged, true));
  TRANSFER_TRY(Execute(arena, *staged));
  return Access::Deposit(workspace, staged->face_result_range.pinned, staged->particle_result_range.pinned);
}
absl::StatusOr<NormalizedLiquidMacFacesView3D> NormalizeLiquid3D(
    CudaLiquidArena3D& arena,
    std::span<const MacFaceDepositQuanta3D> quanta,
    MacDepositQuantization3D quantization,
    double minimum_face_mass,
    LiquidBoundaryWorkspace3D& workspace
) {
  TRANSFER_TRY(arena.RecordRuntimeResult(cudaSuccess));
  TRANSFER_TRY(Access::Check(workspace, quanta.size(), 0));
  TRANSFER_TRY(ValidateQuantization(quantization, 0));
  if (!std::isfinite(minimum_face_mass) || minimum_face_mass < 0 || AliasesLiquidScratch3D(arena, quanta)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  auto staged = Stage(arena, quanta.size(), 0);
  if (!staged.ok()) {
    return staged.status();
  }
  auto& query = staged->query_range.pinned[0];
  query.operation = BoundaryTransferOperation3D::kNormalize;
  query.quantization = quantization;
  query.minimum_face_mass = minimum_face_mass;
  for (std::size_t slot = 0; slot < quanta.size(); ++slot) {
    staged->faces[slot].face_quanta = quanta[slot];
  }
  TRANSFER_TRY(Execute(arena, *staged));
  return Access::Normalize(workspace, staged->face_result_range.pinned);
}
absl::StatusOr<LiquidBoundaryGatherView3D> GatherLiquid3D(
    CudaLiquidArena3D& arena,
    const LiquidBoundaryGatherViewInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace
) {
  TRANSFER_TRY(arena.RecordRuntimeResult(cudaSuccess));
  TRANSFER_TRY(Access::Check(workspace, inputs.mac_grid.face_count(), inputs.particles.size()));
  if (AliasesLiquidScratch3D(arena, inputs.mac_grid.active_block_indices())) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (!std::isfinite(inputs.old_time) || !std::isfinite(inputs.new_time) || inputs.new_time <= inputs.old_time
      || AliasesLiquidScratch3D(arena, inputs.particles)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  TRANSFER_TRY(CheckHistory(arena, inputs.old_history, inputs.mac_grid.face_count(), inputs.old_time));
  TRANSFER_TRY(CheckHistory(arena, inputs.new_history, inputs.mac_grid.face_count(), inputs.new_time));
  auto staged = Stage(arena, inputs.mac_grid.face_count(), inputs.particles.size());
  if (!staged.ok()) {
    return staged.status();
  }
  auto& query = staged->query_range.pinned[0];
  query.operation = BoundaryTransferOperation3D::kGather;
  query.cell_width = inputs.mac_grid.geometry().cell_width;
  TRANSFER_TRY(StageAnchors(inputs.mac_grid, *staged));
  StageHistory(*staged, inputs.old_history, true);
  StageHistory(*staged, inputs.new_history, false);
  TRANSFER_TRY(StageParticles(inputs.mac_grid, inputs.particles, *staged, false));
  TRANSFER_TRY(Execute(arena, *staged));
  return Access::Gather(workspace, staged->particle_result_range.pinned);
}
absl::StatusOr<double> PredictMissingLiquidFace3D(
    CudaLiquidArena3D& arena,
    const LiquidMissingFacePredictionInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace
) {
  TRANSFER_TRY(arena.RecordRuntimeResult(cudaSuccess));
  TRANSFER_TRY(Access::Check(workspace, inputs.grid.face_count(), 0));
  if (AliasesLiquidScratch3D(arena, inputs.grid.active_block_indices())) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (inputs.face_slot >= inputs.grid.face_count()) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  if (!IsFinite(inputs.world_acceleration) || !std::isfinite(inputs.step_duration) || inputs.step_duration <= 0) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  TRANSFER_TRY(
      CheckHistory(arena, inputs.old_history, inputs.grid.face_count(), inputs.old_history.velocity_sample_time)
  );
  auto staged = Stage(arena, inputs.grid.face_count(), 0);
  if (!staged.ok()) {
    return staged.status();
  }
  auto& query = staged->query_range.pinned[0];
  query.operation = BoundaryTransferOperation3D::kPredict;
  query.target_face_slot = inputs.face_slot;
  query.side_tag = inputs.side_tag;
  query.world_acceleration = inputs.world_acceleration;
  query.step_duration = inputs.step_duration;
  TRANSFER_TRY(StageAnchors(inputs.grid, *staged));
  StageHistory(*staged, inputs.old_history, true);
  TRANSFER_TRY(Execute(arena, *staged));
  return staged->report_range.pinned[0].predicted_face_velocity;
}
absl::Status ApplyLiquidForces3D(
    CudaLiquidArena3D& arena,
    SparseMacGridView3D grid,
    std::span<const double> old_velocities,
    Vec3 acceleration,
    double step_duration,
    std::span<double> predicted
) {
  TRANSFER_TRY(arena.RecordRuntimeResult(cudaSuccess));
  if (old_velocities.size() != grid.face_count() || predicted.size() != grid.face_count()) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (!IsFinite(acceleration) || !std::isfinite(step_duration) || step_duration <= 0) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const auto block_indices = grid.active_block_indices();
  const bool outputs_overlap_inputs = LiquidByteRangesOverlap3D(
                                          old_velocities.data(),
                                          old_velocities.size_bytes(),
                                          predicted.data(),
                                          predicted.size_bytes()
                                      )
      || LiquidByteRangesOverlap3D(block_indices.data(),
                                   block_indices.size_bytes(),
                                   predicted.data(),
                                   predicted.size_bytes());
  if (outputs_overlap_inputs || AliasesLiquidScratch3D(arena, block_indices)
      || AliasesLiquidScratch3D(arena, old_velocities) || AliasesLiquidScratch3D(arena, predicted)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  for (const double velocity : old_velocities) {
    if (!std::isfinite(velocity)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  auto staged = Stage(arena, grid.face_count(), 0);
  if (!staged.ok()) {
    return staged.status();
  }
  auto& query = staged->query_range.pinned[0];
  query.operation = BoundaryTransferOperation3D::kForces;
  query.world_acceleration = acceleration;
  query.step_duration = step_duration;
  TRANSFER_TRY(StageAnchors(grid, *staged));
  for (std::size_t slot = 0; slot < old_velocities.size(); ++slot) {
    staged->faces[slot].old_history.face_velocity = old_velocities[slot];
  }
  TRANSFER_TRY(Execute(arena, *staged));
  for (std::size_t slot = 0; slot < predicted.size(); ++slot) {
    predicted[slot] = staged->face_result_range.pinned[slot].normalized_face_velocity;
  }
  return absl::OkStatus();
}
absl::Status UpdateLiquidParticles3D(
    CudaLiquidArena3D& arena,
    const LiquidParticleUpdateInputs3D& inputs,
    std::span<LiquidParticleState3D> particles
) {
  TRANSFER_TRY(arena.RecordRuntimeResult(cudaSuccess));
  if (inputs.gathered.size() != particles.size() || inputs.liquid_distances.size() != particles.size()) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (!std::isnormal(inputs.cell_width) || inputs.cell_width <= 0 || !std::isfinite(inputs.step_duration)
      || inputs.step_duration <= 0) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (!std::isfinite(inputs.particle_energy_blend) || inputs.particle_energy_blend < 0
      || inputs.particle_energy_blend > 1 || !std::isfinite(inputs.friction_coefficient)
      || inputs.friction_coefficient < 0 || !std::isfinite(inputs.penetration_tolerance)
      || inputs.penetration_tolerance < 0) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const bool outputs_overlap_inputs = LiquidByteRangesOverlap3D(
                                          particles.data(),
                                          particles.size_bytes(),
                                          inputs.gathered.data(),
                                          inputs.gathered.size_bytes()
                                      )
      || LiquidByteRangesOverlap3D(particles.data(),
                                   particles.size_bytes(),
                                   inputs.liquid_distances.data(),
                                   inputs.liquid_distances.size_bytes());
  if (outputs_overlap_inputs || AliasesLiquidScratch3D(arena, particles)
      || AliasesLiquidScratch3D(arena, inputs.gathered) || AliasesLiquidScratch3D(arena, inputs.liquid_distances)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const auto obstacle = boundary_internal::PackObstacle(inputs.obstacle);
  if (obstacle.kind == boundary_internal::ObstacleKind::kOpenTank) {
    if (inputs.friction_coefficient != 0) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    TRANSFER_TRY(
        boundary_internal::ArithmeticStatus(boundary_internal::CheckTankStep(obstacle.tank, inputs.step_duration), true)
    );
  } else if (obstacle.kind != boundary_internal::ObstacleKind::kAbsent) {
    const double normal_length_squared = Dot(obstacle.normal, obstacle.normal);
    if (!IsFinite(obstacle.normal) || !IsFinite(obstacle.velocity) || !std::isfinite(obstacle.coordinate)
        || std::abs(normal_length_squared - 1) > 1e-12) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    if (obstacle.kind == boundary_internal::ObstacleKind::kSlab
        && (!std::isfinite(obstacle.half_width) || obstacle.half_width <= 0)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const auto& particle = particles[index];
    const auto& gather = inputs.gathered[index];
    if (!IsFinite(particle.world_position) || !std::isfinite(particle.volume_ratio) || particle.volume_ratio <= 0
        || !std::isfinite(inputs.liquid_distances[index]) || !IsFinite(gather.new_pic_velocity)
        || !IsFinite(gather.particle_minus_old_pic_velocity)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    for (auto row : kRows) {
      if (!IsFinite(gather.velocity_gradient.*row) || !IsFinite(gather.fitted_velocity_slopes.*row)) {
        return Error(absl::StatusCode::kInvalidArgument);
      }
    }
  }
  TRANSFER_TRY(ValidateCapacity(0, particles.size()));
  if (particles.size() > arena.capacities().particles) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  const auto query = CheckedLiquidRange3D<BoundaryTransferQuery3D>(arena, Range::kTransferQuery, 1);
  if (!query.ok()) {
    return query.status();
  }
  const auto staged_particles =
      CheckedLiquidRange3D<BoundaryUpdateParticle3D>(arena, Range::kTransferParticles, particles.size());
  if (!staged_particles.ok()) {
    return staged_particles.status();
  }
  const auto results =
      CheckedLiquidRange3D<BoundaryUpdateResult3D>(arena, Range::kTransferParticleResults, particles.size());
  if (!results.ok()) {
    return results.status();
  }
  const auto architecture = CheckedLiquidRange3D<int>(arena, Range::kArchitecture, 1);
  if (!architecture.ok()) {
    return architecture.status();
  }
  query->pinned[0] = {
      .particle_count = particles.size(),
      .cell_width = inputs.cell_width,
      .step_duration = inputs.step_duration,
      .obstacle = obstacle,
      .particle_energy_blend = inputs.particle_energy_blend,
      .friction_coefficient = inputs.friction_coefficient,
      .penetration_tolerance = inputs.penetration_tolerance,
  };
  for (std::size_t index = 0; index < particles.size(); ++index) {
    staged_particles->pinned[index] = {
        .world_position = particles[index].world_position,
        .gathered = inputs.gathered[index],
        .liquid_distance = inputs.liquid_distances[index],
        .volume_ratio = particles[index].volume_ratio,
        .pic_initialized = particles[index].pic_initialized,
    };
  }
  TRANSFER_TRY(CopyLiquidRange3D(arena, *query, true));
  TRANSFER_TRY(CopyLiquidRange3D(arena, *staged_particles, true));
  TRANSFER_TRY(arena.RecordRuntimeResult(LaunchLiquidParticleUpdate3D(
      query->device,
      staged_particles->device,
      results->device,
      architecture->device,
      particles.size(),
      arena.stream()
  )));
  TRANSFER_TRY(CopyLiquidRange3D(arena, *results, false));
  TRANSFER_TRY(CopyLiquidRange3D(arena, *architecture, false));
  TRANSFER_TRY(FinishLiquidDispatch3D(arena));
  if (architecture->pinned[0] != 1200) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  // Match the first serial arithmetic failure or timestep retry before
  // publishing any updated particle fields.
  for (const auto& result : results->pinned) {
    TRANSFER_TRY(boundary_internal::ArithmeticStatus(result.error, true));
    if (result.retry_step) {
      return Error(absl::StatusCode::kAborted);
    }
  }
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const auto& result = results->pinned[index];
    particles[index].world_position = result.world_position;
    particles[index].world_velocity = result.stored_world_velocity;
    particles[index].volume_ratio = result.volume_ratio;
    particles[index].velocity_slopes = inputs.gathered[index].fitted_velocity_slopes;
    particles[index].affine_enabled = inputs.gathered[index].affine_enabled;
    particles[index].pic_initialized = true;
  }
  return absl::OkStatus();
}
#undef TRANSFER_TRY
}  // namespace fluidity::cuda_internal
