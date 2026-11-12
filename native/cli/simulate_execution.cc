#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <thread>
#include <vector>

#include "../src/liquid_cache_files_3d_internal.h"
#include "absl/cleanup/cleanup.h"
#include "fluidity/host/effects_cache_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "fluidity/host/motion_capture_3d.h"
#include "fluidity/host/solid_interaction_checkpoint_3d.h"
#include "fluidity/host/whitewater_sphere_source_3d.h"
#include "simulate_job.h"
#if defined(FLUIDITY_SIMULATE_CUDA)
#include "fluidity/host/cuda_liquid_snapshot_3d.h"
#include "fluidity/host/cuda_liquid_substep_3d.h"
#include "simulate_cuda_compatibility.h"
#endif

namespace fluidity::simulate {
namespace {
#define EXEC_TRY(expression) \
  do { \
    const auto execution_status = (expression); \
    if (!execution_status.ok()) { \
      return execution_status; \
    } \
  } while (false)
constexpr double kFramesPerSecond = 24;
// Fixed bounded qualification plan, persisted independently of physical identity.
// No automatic growth is requested by this CLI; capacity errors fail explicitly.
constexpr LiquidStateCapacities3D kCapacities{1024, 128, 4096, 24576, 256};
std::string Number(double value) {
  std::array<char, 64> buffer{};
  auto result = std::to_chars(
      buffer.data(),
      buffer.data() + buffer.size(),
      value,
      std::chars_format::general,
      std::numeric_limits<double>::max_digits10
  );
  return {buffer.data(), result.ptr};
}
std::string OptionalNumber(std::optional<double> value) {
  return value.has_value() ? Number(*value) : "null";
}
std::string ClockMembers(const LiquidClock3D& clock) {
  return "\"frame\":" + std::to_string(clock.frame_ordinal) + ",\"substep\":" + std::to_string(clock.substep_ordinal)
      + ",\"epoch\":" + std::to_string(clock.epoch)
      + ",\"accepted_step\":" + std::to_string(clock.accepted_step_ordinal) + ",\"time_seconds\":" + Number(clock.time);
}
std::string Hex(const LiquidDigest3D& digest) {
  return LiquidDigestHex3D(digest).data();
}
absl::StatusOr<double> CompletedFrameTime(std::uint64_t frame) {
  if (frame > 9007199254740991ULL) {
    return absl::OutOfRangeError("");
  }
  const double time = static_cast<double>(frame) / kFramesPerSecond;
  if (frame > 0 && time <= static_cast<double>(frame - 1) / kFramesPerSecond) {
    return absl::OutOfRangeError("");
  }
  return time;
}
absl::Status ValidateCompletedFrameClock(const LiquidState3D& state) {
  auto time = CompletedFrameTime(state.clock.frame_ordinal);
  if (!time.ok()) {
    return time.status();
  }
  if (state.clock.substep_ordinal != 0 || state.clock.time != *time) {
    return absl::FailedPreconditionError("");
  }
  return absl::OkStatus();
}
absl::StatusOr<std::string> StateMembers(const LiquidState3D& state, const LiquidStepDiagnostics3D& diagnostics) {
  auto bytes = EncodeLiquidState3D(state);
  if (!bytes.ok()) {
    return bytes.status();
  }
  std::string result = ClockMembers(state.clock)
      + ",\"state_digest\":" + JsonString(Hex(LiquidContentDigest3D(bytes->values()))) + ",\"particle_count\":"
      + std::to_string(state.particles.size()) + ",\"block_count\":" + std::to_string(state.blocks.size())
      + ",\"cell_count\":" + std::to_string(state.cells.size()) + ",\"face_count\":"
      + std::to_string(state.faces.size()) + ",\"next_particle_id\":" + std::to_string(state.next_particle_id)
      + ",\"attempts\":" + std::to_string(diagnostics.attempts) + ",\"accepted_attempts\":"
      + std::to_string(diagnostics.accepted_attempts) + ",\"dt_retries\":" + std::to_string(diagnostics.dt_retries)
      + ",\"capacity_replays\":" + std::to_string(diagnostics.capacity_replays) + ",\"source_next_child_ordinals\":[";
  for (std::size_t i = 0; i < state.sources.size(); ++i) {
    if (i != 0) {
      result += ',';
    }
    result += std::to_string(state.sources[i].next_child_ordinal);
  }
  result += "],\"source_fractional_remainders\":[";
  for (std::size_t i = 0; i < state.sources.size(); ++i) {
    if (i != 0) {
      result += ',';
    }
    result += Number(state.sources[i].fractional_remainder);
  }
  return result + ']';
}
// Compare regenerated point bytes with the committed manifest for preview or replay.
// Intermediate frames may lack a publication. Replay requires its final frame.
absl::Status
VerifyCommittedPointFrame(const LiquidCache3D& cache, const LiquidState3D& state, bool require_publication) {
  auto committed_frame = cache.ReadCommittedFrame(state.clock.frame_ordinal);
  if (!committed_frame.ok()) {
    return !require_publication && absl::IsNotFound(committed_frame.status()) ? absl::OkStatus()
                                                                              : committed_frame.status();
  }
  auto points = BuildLiquidPointFrame3D(state);
  if (!points.ok()) {
    return points.status();
  }
  auto bytes = EncodeLiquidPointFrame3D(*points);
  if (!bytes.ok()) {
    return bytes.status();
  }
  return VerifyLiquidContent3D(bytes->values(), committed_frame->manifest.points, LiquidContentKind3D::kPoints);
}
absl::StatusOr<LiquidState3D> ReadInitialState(const Command& command, LiquidResolvedSceneView3D scene) {
  const LiquidUpstreamContent3D* initial_reference = nullptr;
  for (const auto& upstream : scene.upstream) {
    if (upstream.role != LiquidUpstreamRole3D::kInitialParticles || initial_reference != nullptr) {
      return absl::FailedPreconditionError("");
    }
    initial_reference = &upstream;
  }
  if (initial_reference != nullptr && command.initial_state_path.empty()) {
    return absl::NotFoundError("");
  }
  if (initial_reference == nullptr && !command.initial_state_path.empty()) {
    return absl::FailedPreconditionError("");
  }
  LiquidState3D state;
  state.geometry = scene.geometry;
  state.random_seed = scene.random_seed;
  if (initial_reference != nullptr) {
    auto bytes = liquid_cache_files_internal::ReadBytes(command.initial_state_path, {});
    if (!bytes.ok()) {
      return bytes.status();
    }
    EXEC_TRY(VerifyLiquidContent3D(
        bytes->values(),
        {
            .kind = LiquidContentKind3D::kState,
            .byte_length = initial_reference->byte_length,
            .digest = initial_reference->digest,
        },
        LiquidContentKind3D::kState
    ));
    auto decoded = DecodeLiquidState3D(bytes->values());
    if (!decoded.ok()) {
      return decoded.status();
    }
    // Only canonical raw state bytes qualify as upstream, never an envelope.
    auto canonical = EncodeLiquidState3D(*decoded);
    if (!canonical.ok()) {
      return canonical.status();
    }
    if (!std::ranges::equal(bytes->values(), canonical->values())) {
      return absl::FailedPreconditionError("");
    }
    state = std::move(*decoded);
  } else {
    auto sources = LiquidRecords3D<LiquidSourceState3D>::Create(scene.scene.sources.size());
    if (!sources.ok()) {
      return sources.status();
    }
    EXEC_TRY(sources->Resize(scene.scene.sources.size()));
    for (std::size_t i = 0; i < sources->size(); ++i) {
      (*sources)[i].source_ordinal = scene.scene.sources[i].source_ordinal;
    }
    state.sources = std::move(*sources);
  }
  const auto& origin = state.geometry.world_origin;
  const auto& resolved_origin = scene.geometry.world_origin;
  if (state.clock != LiquidClock3D{} || state.random_seed != scene.random_seed
      || state.geometry.cell_width != scene.geometry.cell_width || origin.x != resolved_origin.x
      || origin.y != resolved_origin.y || origin.z != resolved_origin.z
      || state.sources.size() != scene.scene.sources.size()) {
    return absl::FailedPreconditionError("");
  }
  for (std::size_t i = 0; i < state.sources.size(); ++i) {
    if (state.sources[i].source_ordinal != scene.scene.sources[i].source_ordinal
        || state.sources[i].next_child_ordinal != 0 || state.sources[i].fractional_remainder != 0) {
      return absl::FailedPreconditionError("");
    }
  }
  EXEC_TRY(ValidateLiquidState3D(state));
  return state;
}
// Reject unusable diagnostics before serializing a settled physical witness.
// The finite and sign checks do not feed back into the numerical attempt.
// Report only settled lineage losses; rejected physical trials never reach the observer.
absl::StatusOr<std::string> CavityInventoryLossMembers(const LiquidCavityLineageResult3D& lineage) {
  for (const double inventory :
       {lineage.vented_inventory, lineage.subresolution_inventory, lineage.unmatched_inventory}) {
    if (!std::isfinite(inventory) || inventory < 0.0) {
      return absl::DataLossError("");
    }
  }
  return "\"vented_inventory\":" + Number(lineage.vented_inventory) + ",\"subresolution_inventory\":"
      + Number(lineage.subresolution_inventory) + ",\"unmatched_inventory\":" + Number(lineage.unmatched_inventory);
}

absl::StatusOr<std::string>
AcceptedMeasurementMembers(const LiquidState3D& state, const LiquidAcceptedStepMeasurement3D& measurement) {
  const double dt = measurement.accepted_dt_seconds;
  if (measurement.accepted_clock != state.clock || !std::isfinite(dt) || dt <= 0) {
    return absl::DataLossError("");
  }
  const auto capillary_bound = measurement.step_start_capillary_bound_seconds;
  if (capillary_bound && (!std::isfinite(*capillary_bound) || *capillary_bound < dt)) {
    return absl::DataLossError("");
  }
  const auto& pressure = measurement.pressure;
  const double pressure_volume = measurement.step_start_pressure_cell_volume_m3;
  const auto nonnegative = [](double value) {
    return std::isfinite(value) && value >= 0;
  };
  if (!nonnegative(pressure.true_residual_l2) || !nonnegative(measurement.pressure_rhs_l2_m3_per_s)
      || !nonnegative(pressure.max_volume_scaled_residual) || !nonnegative(pressure.max_cell_constraint_defect)
      || !nonnegative(pressure_volume) || state.cells.values().empty() != (pressure_volume == 0)) {
    return absl::DataLossError("");
  }
  const auto viscous_residual = measurement.viscous_max_momentum_residual_kg_m_per_s;
  if (viscous_residual && !nonnegative(*viscous_residual)) {
    return absl::DataLossError("");
  }
  const std::string viscous_member =
      viscous_residual ? ",\"viscous_max_momentum_residual_kg_m_per_s\":" + Number(*viscous_residual) : "";
  const auto& piston = measurement.piston_pose;
  const double inward_flux = measurement.step_start_inward_piston_flux_m3_per_s;
  if (!std::isfinite(inward_flux) || (!piston && inward_flux != 0)
      || (piston && (!std::isfinite(piston->step_start_low_x_m) || !std::isfinite(piston->step_end_low_x_m)))) {
    return absl::DataLossError("");
  }
  const auto cavity_bound = measurement.step_start_cavity_bound_seconds;
  if (cavity_bound && (!std::isfinite(*cavity_bound) || *cavity_bound < dt)) {
    return absl::DataLossError("");
  }
  std::string cavity_member;
  if (const auto& cavity = measurement.cavity) {
    const auto losses = CavityInventoryLossMembers(cavity->lineage);
    if (!losses.ok()) {
      return losses.status();
    }
    cavity_member = ",\"cavity\":{\"active_count\":" + std::to_string(cavity->boundaries.active_cavity_count)
        + ",\"step_start_bound_seconds\":" + OptionalNumber(cavity_bound) + ",\"inventory_loss\":{" + *losses + "}}";
  }
  const std::string piston_start = piston ? Number(piston->step_start_low_x_m) : "null";
  const std::string piston_end = piston ? Number(piston->step_end_low_x_m) : "null";
  return "\"accepted_measurement\":{\"accepted_dt_seconds\":" + Number(dt) + ",\"step_start_capillary_bound_seconds\":"
      + OptionalNumber(capillary_bound) + ",\"pressure_iterations\":" + std::to_string(pressure.iteration_count)
      + ",\"pressure_true_residual_l2_m3_per_s\":" + Number(pressure.true_residual_l2)
      + ",\"pressure_rhs_l2_m3_per_s\":" + Number(measurement.pressure_rhs_l2_m3_per_s)
      + ",\"pressure_max_volume_scaled_residual_per_s\":" + Number(pressure.max_volume_scaled_residual)
      + ",\"pressure_max_constraint_defect_per_s\":" + Number(pressure.max_cell_constraint_defect)
      + ",\"step_start_pressure_cell_volume_m3\":" + Number(pressure_volume)
      + ",\"step_start_inward_piston_flux_m3_per_s\":" + Number(inward_flux) + ",\"step_start_piston_low_x_m\":"
      + piston_start + ",\"step_end_piston_low_x_m\":" + piston_end + viscous_member + cavity_member + '}';
}
std::string GeometryMembers(const LiquidState3D& state) {
  std::string members = "\"geometry\":{\"particles\":[";
  const char* separator = "";
  for (const auto& particle : state.particles.values()) {
    const auto& position = particle.world_position;
    members += separator
        + ("{\"id\":" + std::to_string(particle.id) + ",\"world_position_m\":[" + Number(position.x) + ','
           + Number(position.y) + ',' + Number(position.z) + "]}");
    separator = ",";
  }
  return members + "]}";
}
std::string_view AttemptDispositionName(LiquidAttemptDisposition3D disposition) {
  switch (disposition) {
    case LiquidAttemptDisposition3D::kAccepted:
      return "accepted";
    case LiquidAttemptDisposition3D::kRetryDt:
      return "retry_dt";
    case LiquidAttemptDisposition3D::kCapacity:
      return "capacity";
    case LiquidAttemptDisposition3D::kFatal:
      return "fatal";
  }
  return "fatal";
}
std::string AttemptMembers(const LiquidAttemptMeasurement3D& attempt, bool frame_settled, std::size_t settled_pass) {
  const bool retained = frame_settled && attempt.disposition == LiquidAttemptDisposition3D::kAccepted
      && attempt.capacity_pass == settled_pass;
  const bool restarted_frame =
      attempt.disposition == LiquidAttemptDisposition3D::kCapacity && attempt.capacity_pass < settled_pass;
  return "\"target_frame\":" + std::to_string(attempt.start_clock.frame_ordinal + 1) + ",\"attempt_in_frame\":"
      + std::to_string(attempt.attempt_in_frame) + ",\"capacity_pass\":" + std::to_string(attempt.capacity_pass)
      + ",\"start_clock\":{" + ClockMembers(attempt.start_clock) + '}' + ",\"attempted_dt_seconds\":"
      + Number(attempt.attempted_dt_seconds) + ",\"work_wall_seconds\":" + Number(attempt.work_wall_seconds)
      + ",\"outcome\":" + JsonString(AttemptDispositionName(attempt.disposition))
      + ",\"failure_code\":" + JsonString(absl::StatusCodeToString(attempt.failure_code)) + ",\"retry_dt_seconds\":"
      + OptionalNumber(attempt.retry_dt_seconds) + ",\"retained\":" + (retained ? "true" : "false")
      + ",\"restarted_frame\":" + (restarted_frame ? "true" : "false");
}
enum class JobWallPhase {
  kColdSetup,
  kScheduling,
  kEmitActivationField,
  kDepositNormalization,
  kForcesGeometryPressure,
  kGatherContactAcceptance,
  kSnapshotCache,
  kDrain,
};
constexpr std::array kWorkPhaseNames{
    "emit_activation_field",
    "deposit_normalization",
    "forces_geometry_pressure",
    "gather_contact_acceptance",
};
constexpr std::array kJobWallPhaseNames{
    "cold_setup",
    "scheduling",
    kWorkPhaseNames[0],
    kWorkPhaseNames[1],
    kWorkPhaseNames[2],
    kWorkPhaseNames[3],
    "snapshot_cache",
    "drain",
};
// One monotonic partition for the solver thread. The phase clock also owns
// timing on early exits. CUDA dispatch intervals below are independent.
struct JobMeasurement {
  using Clock = std::chrono::steady_clock;
  const Clock::time_point started = Clock::now();
  Clock::time_point phase_started = started;
  JobWallPhase phase = JobWallPhase::kColdSetup;
  std::array<double, kJobWallPhaseNames.size()> seconds{};
  std::optional<LiquidBackend3D> backend;
  std::uint64_t attempt_count = 0;
  std::uint64_t settled_attempt_count = 0;
#if defined(FLUIDITY_SIMULATE_CUDA)
  std::optional<CudaLiquidArenaMetrics3D> cuda_before_drain;
  std::optional<CudaLiquidArenaMetrics3D> cuda_after_drain;
#endif

  void Enter(JobWallPhase next) {
    const auto now = Clock::now();
    seconds[static_cast<std::size_t>(phase)] += std::chrono::duration<double>(now - phase_started).count();
    phase_started = now;
    phase = next;
  }
  static void EnterWork(void* context, LiquidWorkPhase3D work_phase) {
    constexpr std::array wall_phases{
        JobWallPhase::kEmitActivationField,
        JobWallPhase::kDepositNormalization,
        JobWallPhase::kForcesGeometryPressure,
        JobWallPhase::kGatherContactAcceptance,
        JobWallPhase::kScheduling,
    };
    static_assert(wall_phases.size() == static_cast<std::size_t>(LiquidWorkPhase3D::kCount) + 1);
    static_cast<JobMeasurement*>(context)->Enter(wall_phases[static_cast<std::size_t>(work_phase)]);
  }
  std::string Members(CommandKind kind, const absl::Status& status) {
    Enter(JobWallPhase::kDrain);
    const auto total = std::chrono::duration<double>(phase_started - started).count();
    const auto name = kind == CommandKind::kRun ? "run" : (kind == CommandKind::kReplay ? "replay" : "step");
    const auto backend_member = backend ? JsonString(*backend == LiquidBackend3D::kCuda ? "cuda" : "cpu") : "null";
    std::string members = "\"command\":" + JsonString(name) + ",\"backend\":" + backend_member + ",\"status\":"
        + JsonString(absl::StatusCodeToString(status.code())) + ",\"attempt_count\":" + std::to_string(attempt_count)
        + ",\"settled_attempt_count\":" + std::to_string(settled_attempt_count)
        + ",\"host_wall\":{\"total_seconds\":" + Number(total);
    for (std::size_t index = 0; index < seconds.size(); ++index) {
      members += ',' + JsonString(std::string(kJobWallPhaseNames[index]) + "_seconds") + ':' + Number(seconds[index]);
    }
    members += '}';
#if defined(FLUIDITY_SIMULATE_CUDA)
    if (cuda_before_drain && cuda_after_drain) {
      members += CudaMembers(*cuda_before_drain, *cuda_after_drain);
      return members;
    }
#endif
    return members + ",\"cuda_events\":null,\"cuda_memory\":null";
  }

#if defined(FLUIDITY_SIMULATE_CUDA)
  static std::string CudaMembers(const CudaLiquidArenaMetrics3D& before, const CudaLiquidArenaMetrics3D& after) {
    std::string members = ",\"cuda_events\":{";
    const auto add_events = [&](std::string_view phase_name, const CudaLiquidEventIntervals3D& events) {
      members += JsonString(std::string(phase_name) + "_seconds") + ':' + Number(events.seconds) + ','
          + JsonString(std::string(phase_name) + "_interval_count") + ':' + std::to_string(events.interval_count);
    };
    static_assert(kWorkPhaseNames.size() == static_cast<std::size_t>(LiquidWorkPhase3D::kCount));
    for (std::size_t index = 0; index < kWorkPhaseNames.size(); ++index) {
      add_events(kWorkPhaseNames[index], after.work_phase_events[index]);
      members += ',';
    }
    add_events("pre_topology_field", after.pre_topology_field_events);
    members += ',';
    add_events("snapshot_d2h", after.snapshot_d2h_events);
    members += "},\"cuda_memory\":{\"total_device_bytes\":" + std::to_string(after.total_device_bytes);
    const auto add = [&](std::string_view field, auto value) {
      members += ',' + JsonString(field) + ':' + std::to_string(value);
    };
    add("free_device_bytes_at_start", after.free_device_bytes_at_start);
    add("budget_bytes", after.device_budget_bytes);
    add("device_byte_limit", after.device_byte_limit);
    add("current_device_reserved_bytes", before.live_device_bytes);
    add("peak_device_reserved_bytes", after.peak_device_bytes);
    add("current_pinned_reserved_bytes", before.live_pinned_bytes);
    add("peak_pinned_reserved_bytes", after.peak_pinned_bytes);
    add("current_snapshot_pinned_reserved_bytes", before.live_snapshot_pinned_bytes);
    add("peak_snapshot_pinned_reserved_bytes", after.peak_snapshot_pinned_bytes);
    add("current_physical_state_bytes", before.live_physical_state_bytes);
    add("peak_physical_state_bytes", after.peak_physical_state_bytes);
    add("current_device_scratch_bytes", before.live_device_scratch_bytes);
    add("peak_device_scratch_bytes", after.peak_device_scratch_bytes);
    add("peak_active_device_scratch_bytes", after.peak_active_device_scratch_bytes);
    add("current_snapshot_output_bytes", before.live_snapshot_output_bytes);
    add("peak_snapshot_output_bytes", after.peak_snapshot_output_bytes);
    add("device_reservation_attempts", after.device_reservations);
    add("budget_refusals", after.budget_refusals);
    add("quarantined_owner_count", after.quarantined_storage_count);
    add("remaining_device_reserved_bytes_after_drain", after.live_device_bytes);
    add("remaining_pinned_reserved_bytes_after_drain", after.live_pinned_bytes);
    members += ",\"free_device_bytes_after_drain\":"
        + (after.free_device_bytes_after_drain ? std::to_string(*after.free_device_bytes_after_drain) : "null");
    return members + '}';
  }
#endif
};
#if defined(FLUIDITY_SIMULATE_CUDA)
// Cavity lineage and particle composition are host-owned. Check every field
// in the device ABI against the accepted state before attaching those fields.
bool MatchesCudaSnapshot(const LiquidState3D& device, const LiquidState3D& accepted) {
  const auto& device_origin = device.geometry.world_origin;
  const auto& accepted_origin = accepted.geometry.world_origin;
  if (device_origin.x != accepted_origin.x || device_origin.y != accepted_origin.y
      || device_origin.z != accepted_origin.z || device.geometry.cell_width != accepted.geometry.cell_width
      || device.clock != accepted.clock || device.random_seed != accepted.random_seed
      || device.next_particle_id != accepted.next_particle_id || device.blocks != accepted.blocks
      || device.cells != accepted.cells || device.faces != accepted.faces || device.sources != accepted.sources
      || device.next_cavity_id != 0 || !device.cavity_gas.values().empty()
      || !device.cavity_components.values().empty()) {
    return false;
  }
  return std::ranges::equal(
      device.particles.values(),
      accepted.particles.values(),
      [](const auto& downloaded, auto host) {
        host.phase_id = 0;
        host.constituent_count = 0;
        return downloaded == host;
      }
  );
}
#endif

struct Observations {
  const Command& command;
  const SimulateJob& job;
  const LiquidSceneInputs3D& source_scene;
  const LiquidStepOptions3D& step_options;
  JobMeasurement* measurement;
  std::vector<std::string> trace;
  std::vector<std::string> cavity_losses;
  std::optional<std::string> selected;
  std::vector<LiquidAttemptMeasurement3D> attempts;
  MotionCapture3D* motion_capture = nullptr;
  double previous_accepted_time_seconds = 0;
  static void ObserveAttempt(void* context, const LiquidAttemptMeasurement3D& attempt) {
    auto& self = *static_cast<Observations*>(context);
    if (self.measurement) {
      ++self.measurement->attempt_count;
    }
    if (self.command.trace && self.command.kind == CommandKind::kRun) {
      self.attempts.push_back(attempt);
    }
  }
  static absl::Status
  Observe(void* context, const LiquidState3D& state, const LiquidStepDiagnostics3D& diagnostics, bool rollback) {
    auto& self = *static_cast<Observations*>(context);
    EXEC_TRY(self.job.CheckForJobStopRequest());
    if (self.motion_capture) {
      if (const auto& capture = self.step_options.whitewater_capture; capture && !rollback) {
        const auto births = SampleAcceptedWhitewaterStep3D(
            state,
            self.source_scene,
            self.step_options,
            capture->world_source_grid_origin,
            self.previous_accepted_time_seconds,
            capture->source_options
        );
        if (!births.ok()) {
          return births.status();
        }
        EXEC_TRY(self.motion_capture->Observe(state, false, *births));
      } else {
        EXEC_TRY(self.motion_capture->Observe(state, rollback));
      }
      self.previous_accepted_time_seconds = state.clock.time;
    }
    if (rollback) {
      self.trace.clear();
      self.cavity_losses.clear();
      self.selected.reset();
      return absl::OkStatus();
    }
    if (self.command.kind == CommandKind::kRun && diagnostics.accepted_measurement
        && diagnostics.accepted_measurement->cavity) {
      const auto& lineage = diagnostics.accepted_measurement->cavity->lineage;
      if (lineage.subresolution_inventory > 0.0 || lineage.vented_inventory > 0.0
          || lineage.unmatched_inventory > 0.0) {
        const auto losses = CavityInventoryLossMembers(lineage);
        if (!losses.ok()) {
          return losses.status();
        }
        self.cavity_losses.push_back(ClockMembers(state.clock) + ",\"inventory_loss\":{" + *losses + '}');
      }
    }
    if (!self.command.trace && self.command.kind == CommandKind::kRun) {
      return absl::OkStatus();
    }
    auto members = StateMembers(state, diagnostics);
    if (!members.ok()) {
      return members.status();
    }
    if (self.command.trace) {
      if (!diagnostics.accepted_measurement) {
        return absl::DataLossError("");
      }
      const auto measurement = AcceptedMeasurementMembers(state, *diagnostics.accepted_measurement);
      if (!measurement.ok()) {
        return measurement.status();
      }
      std::string trace = *members + ',' + *measurement;
      if (self.command.trace_geometry) {
        trace += ',' + GeometryMembers(state);
      }
      self.trace.push_back(std::move(trace));
    }
    if (self.command.kind == CommandKind::kStep
        || (self.command.frame == state.clock.frame_ordinal && self.command.substep == state.clock.substep_ordinal)) {
      self.selected = *members;
    }
    return self.command.kind == CommandKind::kStep ? absl::CancelledError("") : absl::OkStatus();
  }
};

absl::StatusOr<std::string> ExecuteCommand(
    const Command& command,
    ProtocolWriter& writer,
    SimulateJob& job,
    Diagnostic* diagnostic,
    JobMeasurement* measurement
) {
  diagnostic->domain = "SIMULATE";
  diagnostic->field = "scene";
  diagnostic->cause = "resolved scene, executable digest or backend is incompatible";
  std::optional<LiquidCache3D> cache;
  const bool replay = command.kind == CommandKind::kReplay;
  const bool step = command.kind == CommandKind::kStep;
  const bool read_only = replay || step || command.preview;
  LiquidResolvedScene3D scene;
  if (command.scene_path.empty()) {
    auto inspection = LiquidCache3D::OpenForInspection(command.cache_directory);
    if (!inspection.ok()) {
      return inspection.status();
    }
    auto resolved = inspection->ReadResolvedScene();
    if (!resolved.ok()) {
      return resolved.status();
    }
    scene = std::move(*resolved);
  } else {
    auto bytes = liquid_cache_files_internal::ReadBytes(command.scene_path, {});
    if (!bytes.ok()) {
      return bytes.status();
    }
    auto resolved = DecodeLiquidResolvedScene3D(bytes->values());
    if (!resolved.ok()) {
      return resolved.status();
    }
    auto canonical = EncodeLiquidResolvedScene3D(resolved->View());
    if (!canonical.ok()) {
      return canonical.status();
    }
    if (!std::ranges::equal(bytes->values(), canonical->values())) {
      return absl::FailedPreconditionError("");
    }
    scene = std::move(*resolved);
  }
  if (measurement) {
    measurement->backend = scene.backend;
  }
  auto digest = ExecutableDigest();
  if (!digest.ok()) {
    return digest.status();
  }
  if (*digest != scene.build_digest || (command.backend && *command.backend != scene.backend)) {
    return absl::FailedPreconditionError("");
  }
#if !defined(FLUIDITY_SIMULATE_CUDA)
  if (scene.backend == LiquidBackend3D::kCuda) {
    return absl::UnavailableError("");
  }
#endif
  for (const auto& upstream : scene.upstream.values()) {
    if (upstream.role != LiquidUpstreamRole3D::kInitialParticles) {
      return absl::FailedPreconditionError("");
    }
  }
  auto fingerprint = ResolvedSceneFingerprint(scene.View());
  if (!fingerprint.ok()) {
    return fingerprint.status();
  }
  std::uint64_t target_frame = replay ? *command.frame : command.frame_end;
  if (replay && *command.substep != 0) {
    if (target_frame == std::numeric_limits<std::uint64_t>::max()) {
      return absl::OutOfRangeError("");
    }
    ++target_frame;
  }
  diagnostic->field = "frame";
  diagnostic->cause = "frame ordinal cannot represent a distinct model-v1 target time";
  auto target_time = CompletedFrameTime(target_frame);
  if (!target_time.ok()) {
    return target_time.status();
  }
  std::optional<LiquidState3D> initial;
  if (!command.initial_state_path.empty()) {
    diagnostic->field = "initial-state";
    diagnostic->cause =
        "initial FLST does not match exact upstream identity, geometry, seed, sources or frame-zero clock";
    auto decoded = ReadInitialState(command, scene.View());
    if (!decoded.ok()) {
      return decoded.status();
    }
    initial = std::move(*decoded);
  }
  std::error_code error;
  const bool exists = std::filesystem::exists(command.cache_directory, error);
  if (error) {
    return absl::UnavailableError("");
  }
  const bool requires_cuda_compatibility = scene.backend == LiquidBackend3D::kCuda;
  diagnostic->field = "cache";
  diagnostic->cause = "cache compatibility or verified retained checkpoint failed";
  if (exists) {
    if (!read_only && !command.resume) {
      return absl::AlreadyExistsError("");
    }
    // Authenticate the cache path and fingerprint without creating lock files
    // while CUDA compatibility admission is still pending.
    auto opened = read_only || requires_cuda_compatibility
        ? LiquidCache3D::OpenForReplay(command.cache_directory, *fingerprint)
        : LiquidCache3D::Open(command.cache_directory, *fingerprint);
    if (!opened.ok()) {
      return opened.status();
    }
    cache = std::move(*opened);
  } else if (replay || command.resume) {
    return absl::NotFoundError("");
  }
#if defined(FLUIDITY_SIMULATE_CUDA)
  std::optional<CudaExecutionIdentity> cuda_identity;
  if (requires_cuda_compatibility && (cache || !read_only)) {
    diagnostic->field = "cuda-compatibility";
    std::optional<std::string> recorded_compatibility;
    if (cache) {
      diagnostic->cause = "missing or malformed CUDA execution compatibility record";
      auto record = ReadCudaCompatibilityRecord(command.cache_directory);
      if (!record.ok()) {
        return record.status();
      }
      recorded_compatibility = std::move(*record);
    }
    diagnostic->cause = "could not identify the selected CUDA device and exact driver/runtime";
    auto actual_identity = QueryCudaExecutionIdentity(command.device_index, *fingerprint);
    if (!actual_identity.ok()) {
      return actual_identity.status();
    }
    if (recorded_compatibility && actual_identity->compatibility_record != *recorded_compatibility) {
      diagnostic->cause = "selected CUDA device or driver/runtime differs from the recorded execution";
      return absl::FailedPreconditionError("");
    }
    cuda_identity = std::move(*actual_identity);
  }
#endif
  diagnostic->field = "cache";
  if (cache && !read_only) {
    if (requires_cuda_compatibility) {
      diagnostic->cause = "could not open the verified cache for publication";
      auto opened = LiquidCache3D::Open(command.cache_directory, *fingerprint);
      if (!opened.ok()) {
        return opened.status();
      }
      cache = std::move(*opened);
    }
    diagnostic->cause = "another process owns the cache job lease or the lease could not be acquired";
    EXEC_TRY(job.AttachCache(command.cache_directory, false));
    if (requires_cuda_compatibility) {
      // A prior writer may have stopped after promotion but before the
      // directory barrier. Finish that barrier before extending this cache.
      diagnostic->cause = "could not flush the verified CUDA compatibility record directory";
      EXEC_TRY(liquid_cache_files_internal::FlushDirectory(command.cache_directory));
    }
  }
  diagnostic->cause = "verified retained checkpoint could not be read";
  LiquidResumeMetadata3D resume{
      .resolved_fingerprint = *fingerprint,
      .execution_id = LiquidExecutionId3D(*fingerprint),
      .capacities = kCapacities,
      .capacity_replay_reservations = {},
  };
  LiquidState3D starting_state;
  bool loaded_checkpoint = false;
  if (cache) {
    std::uint64_t maximum = target_frame;
    if (step) {
      maximum = std::numeric_limits<std::uint64_t>::max();
    } else if (replay) {
      maximum = *command.frame;
    } else if (command.frame_start_explicit) {
      maximum = command.frame_start;
    }
    auto checkpoint = cache->ReadCheckpoint(maximum);
    if (checkpoint.ok()) {
      starting_state = std::move(checkpoint->state);
      resume = checkpoint->resume;
      loaded_checkpoint = true;
    } else if (!absl::IsNotFound(checkpoint.status()) || !initial) {
      return checkpoint.status();
    }
  }
  if (!loaded_checkpoint) {
    if (!initial) {
      diagnostic->field = "initial-state";
      diagnostic->cause = "a verified frame-zero upstream state is required for reconstruction";
      auto decoded = ReadInitialState(command, scene.View());
      if (!decoded.ok()) {
        return decoded.status();
      }
      initial = std::move(*decoded);
    }
    starting_state = std::move(*initial);
  }
  diagnostic->field = "clock";
  diagnostic->cause = "checkpoint clock does not match the model-v1 absolute frame ordinal divided by 24 seconds";
  EXEC_TRY(ValidateCompletedFrameClock(starting_state));
  job.SetSettledClock(scene.backend, starting_state.clock, loaded_checkpoint);
  EXEC_TRY(job.CheckForJobStopRequest());
  if (step) {
    if (starting_state.clock.frame_ordinal == std::numeric_limits<std::uint64_t>::max()) {
      return absl::OutOfRangeError("");
    }
    target_frame = starting_state.clock.frame_ordinal + 1;
    target_time = CompletedFrameTime(target_frame);
    if (!target_time.ok()) {
      return target_time.status();
    }
  }
  const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&scene.scene.obstacle);
  if (tank) {
    diagnostic->field = "frame";
    diagnostic->cause = "the requested time requires a finite open tank with positive width";
    const auto pose = boundary_internal::CheckTankStep(*tank, *target_time);
    if (pose == boundary_internal::Error::kRange) {
      return absl::OutOfRangeError("");
    }
    if (pose != boundary_internal::Error::kNone) {
      return absl::FailedPreconditionError("");
    }
  }
  auto options = scene.options;
  if (options.motion_sample_intervals != 0 && command.frame_start > 1) {
    return absl::InvalidArgumentError("effects capture requires a contiguous point-frame range");
  }
  options.capacities = resume.capacities;
  options.capacity_replay_reservations = resume.capacity_replay_reservations;
  // The backend must outlive scheduler callbacks; the queue must outlive writer
  // borrows and drain its private D2H stream before releasing either DMA side.
#if defined(FLUIDITY_SIMULATE_CUDA)
  std::optional<CudaLiquidSubstepBackend3D> cuda_backend;
  std::optional<CudaLiquidSnapshotQueue3D> queue;
  // Keep a validated host snapshot available until replacement Finish succeeds.
  std::optional<CudaLiquidSnapshotQueue3D> previous_queue;
  const absl::Cleanup drain_cuda = [&] {
    if (measurement && cuda_backend) {
      measurement->Enter(JobWallPhase::kDrain);
      measurement->cuda_before_drain = cuda_backend->usage().arena_metrics;
      previous_queue.reset();
      queue.reset();
      measurement->cuda_after_drain = std::move(*cuda_backend).DrainMetrics();
    }
  };
#endif
  auto operations = CpuLiquidSubstepBackend3D();
  diagnostic->field = "backend";
  diagnostic->cause = "explicit scheduler backend initialization failed";
#if defined(FLUIDITY_SIMULATE_CUDA)
  if (scene.backend == LiquidBackend3D::kCuda) {
    const bool needs_wetted_piston = tank && tank->piston_velocity_x != 0;
    auto backend = CudaLiquidSubstepBackend3D::Create(
        options.capacities,
        command.device_index,
        command.measure_job,
        needs_wetted_piston
    );
    if (!backend.ok()) {
      return backend.status();
    }
    cuda_backend = std::move(*backend);
    operations = cuda_backend->operations(/*enable_gather_runs=*/true, /*enable_gathered_field_batches=*/true);
  }
#endif
  auto scheduler = LiquidSubstepScheduler3D::Create(starting_state, scene.View().scene, options, operations);
  const absl::Cleanup begin_drain = [&] {
    if (measurement) {
      measurement->Enter(JobWallPhase::kDrain);
    }
  };
  if (!scheduler.ok()) {
    return scheduler.status();
  }
  double copy_milliseconds = 0;
#if defined(FLUIDITY_SIMULATE_CUDA)
  if (cuda_backend) {
    auto source = cuda_backend->AcquireCommitted();
    if (!source.ok()) {
      return source.status();
    }
    auto created = CudaLiquidSnapshotQueue3D::Create(*source);
    if (!created.ok()) {
      return created.status();
    }
    queue = std::move(*created);
  }
#endif
  EXEC_TRY(job.CheckForJobStopRequest());
  if (!cache && !read_only) {
    diagnostic->field = "cache";
    diagnostic->cause = "could not create the cache and acquire its whole-job lease";
    auto created = LiquidCache3D::Create(command.cache_directory, scene.View());
    if (!created.ok()) {
      return created.status();
    }
    cache = std::move(*created);
    EXEC_TRY(job.AttachCache(command.cache_directory, true));
#if defined(FLUIDITY_SIMULATE_CUDA)
    if (cuda_identity) {
      diagnostic->field = "cuda-compatibility";
      diagnostic->cause = "could not durably publish the CUDA execution compatibility record";
      EXEC_TRY(PublishCudaCompatibilityRecord(command.cache_directory, cuda_identity->compatibility_record));
    }
#endif
  }
  diagnostic->field = "job-status";
  diagnostic->cause = "could not persist the baking job status";
  EXEC_TRY(job.MarkBaking());
  std::optional<MotionCapture3D> motion_capture;
  std::optional<EffectsCache3D> source_cache;
  if (options.motion_sample_intervals != 0) {
    const auto source_root = command.cache_directory / "effects" / "source";
    const EffectsIdentity3D identity{
        .stage = EffectsStage3D::kSource,
        .upstream_execution = LiquidExecutionId3D(*fingerprint),
        .settings = *fingerprint,
        .build = scene.build_digest,
    };
    auto opened =
        read_only ? EffectsCache3D::OpenForInspection(source_root) : EffectsCache3D::Create(source_root, identity);
    if (!opened.ok()) {
      return opened.status();
    }
    if (opened->identity() != identity) {
      return absl::FailedPreconditionError("source capture identity differs");
    }
    source_cache = std::move(*opened);
  }
  // Synchronous writer boundary over a real asynchronous D2H queue. Each call
  // drains its source before the next whole-frame AdvanceFrame; the host writer
  // lease remains alive through checked serialization and manifest publication.
  auto finish_frame_snapshot = [&](bool publish) -> absl::Status {
    if (measurement) {
      measurement->Enter(JobWallPhase::kSnapshotCache);
    }
    const absl::Cleanup end_publication = [&] {
      if (measurement) {
        measurement->Enter(JobWallPhase::kScheduling);
      }
    };
    const LiquidState3D* state = &scheduler->committed();
#if defined(FLUIDITY_SIMULATE_CUDA)
    LiquidSnapshotLease3D writer_lease;
    if (cuda_backend) {
      auto source = cuda_backend->AcquireCommitted();
      if (!source.ok()) {
        return source.status();
      }
      // Capacity replay may replace the arena layout. All old source copies have
      // drained here; create the replacement queue only outside physical attempts.
      if (scheduler->diagnostics().capacity_replays != 0) {
        auto replacement = CudaLiquidSnapshotQueue3D::Create(*source);
        if (!replacement.ok()) {
          return replacement.status();
        }
        previous_queue = std::move(queue);
        queue = std::move(*replacement);
      }
      const auto identity = source->identity();
      auto ticket = queue->BeginCopy(identity, std::move(*source));
      if (!ticket.ok()) {
        return ticket.status();
      }
      while (true) {
        auto progress = queue->Poll(*ticket);
        if (!progress.ok()) {
          return progress.status();
        }
        if (*progress == LiquidSnapshotProgress3D::kReady) {
          break;
        }
        std::this_thread::yield();
      }
      auto finished = queue->Finish(*ticket);
      if (!finished.ok()) {
        return finished.status();
      }
      writer_lease = std::move(*finished);
      previous_queue.reset();
      const auto& downloaded = *writer_lease.state();
      copy_milliseconds = writer_lease.copy_milliseconds();
      if (downloaded == *state) {
        state = &downloaded;
      } else if (!MatchesCudaSnapshot(downloaded, *state)) {
        return absl::DataLossError("");
      }
      // For host-owned metadata, state remains the same accepted epoch whose
      // entire device payload was just compared against the snapshot lease.
    }
#endif
    if (!cache) {
      return absl::OkStatus();
    }
    if (!publish) {
      EXEC_TRY(VerifyCommittedPointFrame(*cache, *state, replay && state->clock.frame_ordinal == target_frame));
    }
    auto points = BuildLiquidPointFrame3D(*state);
    if (!points.ok()) {
      return points.status();
    }
    if (source_cache) {
      const auto point_bytes = EncodeLiquidPointFrame3D(*points);
      if (!point_bytes.ok()) {
        return point_bytes.status();
      }
      const auto upstream_content = LiquidContentDigest3D(point_bytes->values());
      if (state->clock.frame_ordinal == 0 || motion_capture) {
        auto payload = state->clock.frame_ordinal == 0 ? EncodeMotionSeed3D(*state) : motion_capture->Finish(*state);
        if (!payload.ok()) {
          return payload.status();
        }
        if (publish) {
          LiquidDigest3D previous{};
          if (state->clock.frame_ordinal > 0) {
            const auto predecessor = source_cache->ReadFrame(state->clock.frame_ordinal - 1);
            if (!predecessor.ok()) {
              return predecessor.status();
            }
            previous = predecessor->payload_digest;
          }
          EXEC_TRY(source_cache->Publish(state->clock, upstream_content, previous, payload->values()));
        } else {
          const auto existing = source_cache->ReadFrame(state->clock.frame_ordinal);
          if (!existing.ok()) {
            return existing.status();
          }
          if (existing->upstream_content != upstream_content
              || !std::ranges::equal(existing->payload.values(), payload->values())) {
            return absl::DataLossError("source replay differs from committed interval");
          }
        }
      } else {
        const auto existing = source_cache->ReadFrame(state->clock.frame_ordinal);
        if (!existing.ok()) {
          return existing.status();
        }
        auto decoded = DecodeMotionFrame3D(existing->payload.values());
        if (!decoded.ok()) {
          return decoded.status();
        }
        if (existing->upstream_content != upstream_content || decoded->samples.back() != *state) {
          return absl::DataLossError("source checkpoint endpoint differs");
        }
      }
    }
    if (!publish) {
      return absl::OkStatus();
    }
    const bool checkpoint = state->clock.frame_ordinal % 10 == 0;
    // Record the actual current host reservation plan after any configured replay.
    resume.capacities = {
        state->particles.capacity(),
        state->blocks.capacity(),
        state->cells.capacity(),
        state->faces.capacity(),
        state->sources.capacity(),
    };
    if (resume.capacities == resume.capacity_replay_reservations) {
      resume.capacity_replay_reservations = {};
    }
    // Once the cache is initialized, publish its frame-zero recovery point
    // even if cancel arrives during the initial snapshot. Later copies can
    // drain safely and leave the accepted frame unpublished on cancellation.
    if (state->clock.frame_ordinal != 0) {
      EXEC_TRY(job.CheckForJobStopRequest());
    }
    EXEC_TRY(cache->PublishFrame(*points, checkpoint ? state : nullptr, checkpoint ? &resume : nullptr));
    std::string measurements;
#if defined(FLUIDITY_SIMULATE_CUDA)
    if (cuda_backend) {
      const auto usage = cuda_backend->usage();
      measurements = ",\"compiled_architecture\":" + std::to_string(usage.compiled_architecture)
          + ",\"snapshot_copy_milliseconds\":" + Number(copy_milliseconds);
    }
#endif
    return job.ReportPublishedFrame(state->clock.frame_ordinal, measurements);
  };
  diagnostic->field = "frame";
  diagnostic->cause = "completed state, point publication or deterministic replay verification failed";
  EXEC_TRY(finish_frame_snapshot(!read_only && starting_state.clock.frame_ordinal == 0));
  EXEC_TRY(job.CheckForJobStopRequest());
  const auto source_scene = scene.View().scene;
  Observations observations{
      .command = command,
      .job = job,
      .source_scene = source_scene,
      .step_options = options,
      .measurement = measurement,
      .trace = {},
      .cavity_losses = {},
      .selected = {},
      .attempts = {},
  };
  const LiquidProgressObserver3D observer{
      .context = &observations,
      .observe = &Observations::Observe,
      .observe_attempt =
          measurement || (command.trace && command.kind == CommandKind::kRun) ? &Observations::ObserveAttempt : nullptr,
      .work_phase =
          {
              .context = measurement,
              .enter = measurement ? &JobMeasurement::EnterWork : nullptr,
          },
  };
  for (std::uint64_t frame = starting_state.clock.frame_ordinal; frame < target_frame;) {
    EXEC_TRY(job.CheckForJobStopRequest());
    ++frame;
    auto time = CompletedFrameTime(frame);
    if (!time.ok()) {
      return time.status();
    }
    observations.trace.clear();
    observations.cavity_losses.clear();
    observations.selected.reset();
    observations.attempts.clear();
    if (options.motion_sample_intervals != 0) {
      auto capture = MotionCapture3D::Create(scheduler->committed(), *time, options.motion_sample_intervals);
      if (!capture.ok()) {
        return capture.status();
      }
      motion_capture = std::move(*capture);
      observations.motion_capture = &*motion_capture;
      observations.previous_accepted_time_seconds = scheduler->committed().clock.time;
    }
    const auto accepted_steps_before_frame = scheduler->committed().clock.accepted_step_ordinal;
    const auto advanced = scheduler->AdvanceFrame(*time, observer);
    if (measurement && advanced.ok()) {
      measurement->settled_attempt_count +=
          scheduler->committed().clock.accepted_step_ordinal - accepted_steps_before_frame;
    }
    // The attempt ledger includes work even when the schedule rolls back or
    // cannot finish. Emit it before the terminal error or settled witnesses.
    const auto settled_pass = scheduler->diagnostics().capacity_replays;
    for (const auto& attempt : observations.attempts) {
      EXEC_TRY(writer.Emit("attempt", AttemptMembers(attempt, advanced.ok(), settled_pass)));
    }
    const bool stopped_after_one_substep = step && absl::IsCancelled(advanced) && observations.selected;
    if (!advanced.ok() && !stopped_after_one_substep) {
      return advanced;
    }
    for (const auto& trace : observations.trace) {
      EXEC_TRY(writer.Emit("trace", "\"published\":false," + trace));
    }
    if (step) {
      break;
    }
    // Capacity rollback can discard accepted substeps. Only a complete
    // frame schedule advances the monotonic heartbeat clock.
    job.SetSettledClock(scene.backend, scheduler->committed().clock);
    const bool publish = !read_only && (frame >= command.frame_start || frame % 10 == 0);
    EXEC_TRY(finish_frame_snapshot(publish));
    for (const auto& loss : observations.cavity_losses) {
      EXEC_TRY(writer.Emit("cavity_inventory_loss", loss));
    }
  }
  auto result = StateMembers(scheduler->committed(), scheduler->diagnostics());
  if (!result.ok()) {
    return result.status();
  }
  if (replay && *command.substep != 0) {
    if (!observations.selected) {
      return absl::NotFoundError("");
    }
    result = *observations.selected;
  }
  if (step && observations.selected) {
    result = *observations.selected;
  }
  std::string members = "\"command\":"
      + JsonString(replay     ? "replay"
                       : step ? "step"
                              : "run")
      + ",\"backend\":" + JsonString(scene.backend == LiquidBackend3D::kCuda ? "cuda" : "cpu")
      + ",\"read_only\":" + (read_only ? "true" : "false") + ",\"frames_per_second\":24,\"resolved_fingerprint\":"
      + JsonString(Hex(*fingerprint)) + ",\"execution_id\":" + JsonString(Hex(resume.execution_id)) + ',' + *result;
#if defined(FLUIDITY_SIMULATE_CUDA)
  if (cuda_backend) {
    const auto usage = cuda_backend->usage();
    members += ",\"compiled_architecture\":" + std::to_string(usage.compiled_architecture)
        + ",\"graph_replay_count\":" + std::to_string(usage.graph_replay_count) + ",\"device_reserved_bytes\":"
        + std::to_string(usage.device_reserved_bytes) + ",\"snapshot_copy_milliseconds\":" + Number(copy_milliseconds);
    if (measurement && !read_only && cuda_identity) {
      members += ",\"cuda_execution_device_digest\":" + JsonString(cuda_identity->device_digest)
          + ",\"cuda_execution_device_index\":" + std::to_string(command.device_index);
    }
  }
#else
  static_cast<void>(copy_milliseconds);
#endif
  EXEC_TRY(job.CheckForJobStopRequest());
  return members;
}
}  // namespace

absl::Status ExecuteRigidBoxFrame(
    const Command& liquid,
    const std::filesystem::path& rigid_packet,
    const std::filesystem::path& previous,
    const std::filesystem::path& output,
    ProtocolWriter& writer,
    bool inspect
) {
  namespace files = liquid_cache_files_internal;
  constexpr LiquidCacheLimits3D kPacketLimits{.maximum_bytes = 4096};
  constexpr LiquidCacheLimits3D kCheckpointLimits{.maximum_bytes = 32 * 1024 * 1024};
  const auto scene_bytes = files::ReadBytes(liquid.scene_path, {});
  if (!scene_bytes.ok()) {
    return scene_bytes.status();
  }
  const auto scene = DecodeLiquidResolvedScene3D(scene_bytes->values());
  if (!scene.ok()) {
    return scene.status();
  }
  const auto build = ExecutableDigest();
  if (!build.ok()) {
    return build.status();
  }
  if (scene->build_digest != *build || scene->backend != LiquidBackend3D::kCpu) {
    return absl::FailedPreconditionError("rigid frame requires a CPU scene resolved for this executable");
  }
  const auto packet = files::ReadBytes(rigid_packet, kPacketLimits);
  if (!packet.ok()) {
    return packet.status();
  }
  const auto bytes = packet->values();
  std::istringstream input(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  input.imbue(std::locale::classic());
  std::string version;
  RigidBody3D body{.body_id = 1};
  Vec3 extent{};
  unsigned substeps = 0;
  const auto read_vector = [&](Vec3& value) {
    return static_cast<bool>(input >> value.x >> value.y >> value.z);
  };
  if (!(input >> version >> substeps >> body.mass) || version != "FLUIDITY-RIGID-1" || substeps == 0 || substeps > 256
      || !read_vector(body.principal_inertia) || !read_vector(extent) || !read_vector(body.center)
      || !(input >> body.orientation.w) || !read_vector(body.orientation.vector) || !read_vector(body.linear_velocity)
      || !read_vector(body.angular_velocity) || (input >> version)) {
    return absl::InvalidArgumentError("invalid rigid box packet");
  }
  EXEC_TRY(ValidateRigidBody3D(body));

  std::vector<std::uint8_t> prefix;
  for (const auto& digest : {LiquidContentDigest3D(scene_bytes->values()), LiquidContentDigest3D(bytes), *build}) {
    prefix.insert(prefix.end(), digest.begin(), digest.end());
  }
  auto initial = ReadInitialState(liquid, scene->View());
  if (!initial.ok()) {
    return initial.status();
  }
  if (!previous.empty()) {
    const auto saved = files::ReadBytes(previous, kCheckpointLimits);
    if (!saved.ok()) {
      return saved.status();
    }
    if (saved->values().size() <= prefix.size() || !std::ranges::equal(saved->values().first(prefix.size()), prefix)) {
      return absl::FailedPreconditionError("rigid checkpoint belongs to another scene, packet or executable");
    }
    const auto envelope = saved->values().subspan(prefix.size());
    constexpr auto kDigestSize = std::tuple_size_v<LiquidDigest3D>;
    if (envelope.size() <= kDigestSize) {
      return absl::DataLossError("rigid checkpoint payload digest or payload is missing");
    }
    const auto payload = envelope.subspan(kDigestSize);
    if (!std::ranges::equal(envelope.first(kDigestSize), LiquidContentDigest3D(payload))) {
      return absl::DataLossError("rigid checkpoint payload digest does not match");
    }
    auto restored = DecodeCoupledInteractionCheckpoint3D(payload);
    if (!restored.ok()) {
      return restored.status();
    }
    if (restored->solid.bodies.size() != 1 || !restored->solid.sides.empty()
        || !restored->solid.elastic_particles.empty() || !restored->solid.plastic_particles.empty()
        || restored->liquid.clock.substep_ordinal != 0) {
      return absl::FailedPreconditionError("rigid checkpoint does not contain one completed box frame");
    }
    const auto& saved_body = restored->solid.bodies.front().body;
    if (saved_body.body_id != body.body_id || saved_body.mass != body.mass
        || saved_body.principal_inertia.x != body.principal_inertia.x
        || saved_body.principal_inertia.y != body.principal_inertia.y
        || saved_body.principal_inertia.z != body.principal_inertia.z) {
      return absl::FailedPreconditionError("rigid checkpoint body properties disagree with the packet");
    }
    body = saved_body;
    *initial = std::move(restored->liquid);
  }

  const auto emit_pose = [&](std::string_view event, const LiquidClock3D& clock) {
    const auto vector = [](Vec3 value) {
      return "[" + Number(value.x) + "," + Number(value.y) + "," + Number(value.z) + "]";
    };
    const auto q = body.orientation;
    return writer.Emit(
        event,
        ClockMembers(clock) + ",\"position\":" + vector(body.center) + ",\"orientation\":[" + Number(q.w) + ","
            + Number(q.vector.x) + "," + Number(q.vector.y) + "," + Number(q.vector.z) + "]"
    );
  };
  if (inspect) {
    return emit_pose("rigid_inspect", initial->clock);
  }

  auto options = scene->options;
  options.capacities = kCapacities;
  auto scheduler = LiquidSubstepScheduler3D::Create(*initial, scene->View().scene, options);
  if (!scheduler.ok()) {
    return scheduler.status();
  }
  const auto frame = initial->clock.frame_ordinal;
  const auto frame_start = CompletedFrameTime(frame);
  const auto frame_end = CompletedFrameTime(frame + 1);
  if (!frame_start.ok() || !frame_end.ok() || initial->clock.time != *frame_start) {
    return absl::FailedPreconditionError("rigid checkpoint is off the completed-frame schedule");
  }
  EXEC_TRY(scheduler->BeginFrame(*frame_end));
  RigidBodyPose3D previous_pose{};
  const double nominal_duration = std::min(options.maximum_step_duration, 1.0 / (kFramesPerSecond * substeps));
  while (scheduler->committed().clock.time < *frame_end) {
    double duration = std::min(nominal_duration, *frame_end - scheduler->committed().clock.time);
    for (;;) {
      const auto gravity = scene->scene.world_acceleration;
      body.force = {body.mass * gravity.x, body.mass * gravity.y, body.mass * gravity.z};
      const auto step = scheduler->AttemptWithCoupledOrientedFloatingBox(duration, body, extent);
      if (!step.ok()) {
        return step.status();
      }
      if (step->body) {
        previous_pose = {body.center, body.orientation};
        body = *step->body;
        break;
      }
      if (!step->outcome.retry_duration) {
        return absl::Status(
            scheduler->diagnostics().failure_code,
            "rigid frame attempt failed at stage "
                + std::to_string(static_cast<int>(scheduler->diagnostics().failed_stage))
        );
      }
      duration = *step->outcome.retry_duration;
    }
  }
  EXEC_TRY(scheduler->FinishFrame());
  const auto& state = scheduler->committed();
  if (!output.empty()) {
    const SolidInteractionCheckpoint3D solids{
        .accepted_step = state.clock.accepted_step_ordinal,
        .bodies = {{body, previous_pose}},
    };
    const auto checkpoint = EncodeCoupledInteractionCheckpoint3D(state, solids);
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }
    const auto checkpoint_digest = LiquidContentDigest3D(*checkpoint);
    prefix.insert(prefix.end(), checkpoint_digest.begin(), checkpoint_digest.end());
    prefix.insert(prefix.end(), checkpoint->begin(), checkpoint->end());
    EXEC_TRY(files::PublishBytes(output, prefix, kCheckpointLimits, {}));
  }
  return emit_pose("rigid_frame", state.clock);
}

absl::Status Execute(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic) {
  std::optional<JobMeasurement> measurement;
  if (command.measure_job) {
    measurement.emplace();
  }
  auto result = [&]() -> absl::StatusOr<std::string> {
    SimulateJob job(writer, command.kind == CommandKind::kRun);
    auto executed = ExecuteCommand(command, writer, job, diagnostic, measurement ? &*measurement : nullptr);
    if (measurement) {
      measurement->Enter(JobWallPhase::kDrain);
    }
    const auto final_status = job.Finish(executed.status(), diagnostic);
    if (!final_status.ok()) {
      return final_status;
    }
    return executed;
  }();
  if (measurement) {
    const auto emitted = writer.Emit("job_measurement", measurement->Members(command.kind, result.status()));
    if (result.ok() && !emitted.ok()) {
      return emitted;
    }
  }
  if (!result.ok()) {
    return result.status();
  }
  return writer.Emit("result", *result);
}
#undef EXEC_TRY
}  // namespace fluidity::simulate
