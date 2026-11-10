#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <optional>
#include <thread>
#include <vector>

#include "../src/liquid_cache_files_3d_internal.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
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
absl::StatusOr<double> FrameTime(std::uint64_t frame) {
  if (frame > 9007199254740991ULL) {
    return absl::OutOfRangeError("");
  }
  const double time = static_cast<double>(frame) / kFramesPerSecond;
  if (frame > 0 && time <= static_cast<double>(frame - 1) / kFramesPerSecond) {
    return absl::OutOfRangeError("");
  }
  return time;
}
absl::Status CompatibleClock(const LiquidState3D& state) {
  auto time = FrameTime(state.clock.frame_ordinal);
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
absl::StatusOr<LiquidPointFrame3D> Points(const LiquidState3D& state) {
  LiquidPointFrame3D result;
  result.clock = state.clock;
  auto points = LiquidRecords3D<LiquidPoint3D>::Create(state.particles.size());
  if (!points.ok()) {
    return points.status();
  }
  EXEC_TRY(points->Resize(state.particles.size()));
  for (std::size_t i = 0; i < state.particles.size(); ++i) {
    const auto& particle = state.particles[i];
    // Equivalent-volume spherical radius in meters, using accepted V0 * J.
    const double radius = std::cbrt(3 * particle.reference_volume * particle.volume_ratio / (4 * std::acos(-1.0)));
    (*points)[i] = {
        .id = particle.id,
        .tag = particle.tag,
        .world_position = particle.world_position,
        .world_velocity = particle.world_velocity,
        .radius = radius,
        .volume_ratio = particle.volume_ratio,
        .level = 0,
    };
  }
  result.points = std::move(*points);
  return result;
}
// Verifies existing publication without invoking a writer, including preview and replay.
absl::Status VerifyFrame(const LiquidCache3D& cache, const LiquidState3D& state, bool required) {
  auto existing = cache.ReadCommittedFrame(state.clock.frame_ordinal);
  if (!existing.ok()) {
    return !required && absl::IsNotFound(existing.status()) ? absl::OkStatus() : existing.status();
  }
  auto points = Points(state);
  if (!points.ok()) {
    return points.status();
  }
  auto bytes = EncodeLiquidPointFrame3D(*points);
  if (!bytes.ok()) {
    return bytes.status();
  }
  return VerifyLiquidContent3D(bytes->values(), existing->manifest.points, LiquidContentKind3D::kPoints);
}
absl::StatusOr<LiquidState3D> InitialState(const Command& command, LiquidResolvedSceneView3D scene) {
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
  const auto& piston = measurement.piston_pose;
  const double inward_flux = measurement.step_start_inward_piston_flux_m3_per_s;
  if (!std::isfinite(inward_flux) || (!piston && inward_flux != 0)
      || (piston && (!std::isfinite(piston->step_start_low_x_m) || !std::isfinite(piston->step_end_low_x_m)))) {
    return absl::DataLossError("");
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
      + ",\"step_start_inward_piston_flux_m3_per_s\":" + Number(inward_flux)
      + ",\"step_start_piston_low_x_m\":" + piston_start + ",\"step_end_piston_low_x_m\":" + piston_end + '}';
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
struct Observations {
  const Command& command;
  const SimulateJob& job;
  std::vector<std::string> trace;
  std::optional<std::string> selected;
  std::vector<LiquidAttemptMeasurement3D> attempts;
  static void ObserveAttempt(void* context, const LiquidAttemptMeasurement3D& attempt) {
    static_cast<Observations*>(context)->attempts.push_back(attempt);
  }
  static absl::Status
  Observe(void* context, const LiquidState3D& state, const LiquidStepDiagnostics3D& diagnostics, bool rollback) {
    auto& self = *static_cast<Observations*>(context);
    EXEC_TRY(self.job.CheckForJobStopRequest());
    if (rollback) {
      self.trace.clear();
      self.selected.reset();
      return absl::OkStatus();
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

absl::StatusOr<std::string>
ExecuteCommand(const Command& command, ProtocolWriter& writer, SimulateJob& job, Diagnostic* diagnostic) {
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
  auto target_time = FrameTime(target_frame);
  if (!target_time.ok()) {
    return target_time.status();
  }
  std::optional<LiquidState3D> initial;
  if (!command.initial_state_path.empty()) {
    diagnostic->field = "initial-state";
    diagnostic->cause =
        "initial FLST does not match exact upstream identity, geometry, seed, sources or frame-zero clock";
    auto decoded = InitialState(command, scene.View());
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
  std::optional<std::string> cuda_compatibility;
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
    auto actual_compatibility = CudaCompatibilityRecord(command.device_index, *fingerprint);
    if (!actual_compatibility.ok()) {
      return actual_compatibility.status();
    }
    if (recorded_compatibility && *actual_compatibility != *recorded_compatibility) {
      diagnostic->cause = "selected CUDA device or driver/runtime differs from the recorded execution";
      return absl::FailedPreconditionError("");
    }
    cuda_compatibility = std::move(*actual_compatibility);
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
  LiquidState3D starting;
  bool loaded = false;
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
      starting = std::move(checkpoint->state);
      resume = checkpoint->resume;
      loaded = true;
    } else if (!absl::IsNotFound(checkpoint.status()) || !initial) {
      return checkpoint.status();
    }
  }
  if (!loaded) {
    if (!initial) {
      diagnostic->field = "initial-state";
      diagnostic->cause = "a verified frame-zero upstream state is required for reconstruction";
      auto decoded = InitialState(command, scene.View());
      if (!decoded.ok()) {
        return decoded.status();
      }
      initial = std::move(*decoded);
    }
    starting = std::move(*initial);
  }
  diagnostic->field = "clock";
  diagnostic->cause = "checkpoint clock does not match the model-v1 absolute frame ordinal divided by 24 seconds";
  EXEC_TRY(CompatibleClock(starting));
  job.SetSettledClock(scene.backend, starting.clock, loaded);
  EXEC_TRY(job.CheckForJobStopRequest());
  if (step) {
    if (starting.clock.frame_ordinal == std::numeric_limits<std::uint64_t>::max()) {
      return absl::OutOfRangeError("");
    }
    target_frame = starting.clock.frame_ordinal + 1;
    target_time = FrameTime(target_frame);
    if (!target_time.ok()) {
      return target_time.status();
    }
  }
  if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&scene.scene.obstacle)) {
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
  options.capacities = resume.capacities;
  options.capacity_replay_reservations = resume.capacity_replay_reservations;
  // The backend must outlive scheduler callbacks; the queue must outlive writer
  // borrows and drain its private D2H stream before releasing either DMA side.
#if defined(FLUIDITY_SIMULATE_CUDA)
  std::optional<CudaLiquidSubstepBackend3D> cuda_backend;
  std::optional<CudaLiquidSnapshotQueue3D> queue;
  // Keep a validated host snapshot available until replacement Finish succeeds.
  std::optional<CudaLiquidSnapshotQueue3D> previous_queue;
#endif
  auto operations = CpuLiquidSubstepBackend3D();
  diagnostic->field = "backend";
  diagnostic->cause = "explicit scheduler backend initialization failed";
#if defined(FLUIDITY_SIMULATE_CUDA)
  if (scene.backend == LiquidBackend3D::kCuda) {
    auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, command.device_index);
    if (!backend.ok()) {
      return backend.status();
    }
    cuda_backend = std::move(*backend);
    operations = cuda_backend->operations();
  }
#endif
  auto scheduler = LiquidSubstepScheduler3D::Create(starting, scene.View().scene, options, operations);
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
    if (cuda_compatibility) {
      diagnostic->field = "cuda-compatibility";
      diagnostic->cause = "could not durably publish the CUDA execution compatibility record";
      EXEC_TRY(PublishCudaCompatibilityRecord(command.cache_directory, *cuda_compatibility));
    }
#endif
  }
  diagnostic->field = "job-status";
  diagnostic->cause = "could not persist the baking job status";
  EXEC_TRY(job.MarkBaking());
  // Synchronous writer boundary over a real asynchronous D2H queue. Each call
  // drains its source before the next whole-frame AdvanceFrame; the host writer
  // lease remains alive through checked serialization and manifest publication.
  auto complete_frame = [&](bool publish) -> absl::Status {
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
      state = writer_lease.state();
      copy_milliseconds = writer_lease.copy_milliseconds();
      if (*state != scheduler->committed()) {
        return absl::DataLossError("");
      }
    }
#endif
    if (!cache) {
      return absl::OkStatus();
    }
    if (!publish) {
      return VerifyFrame(*cache, *state, replay && state->clock.frame_ordinal == target_frame);
    }
    auto points = Points(*state);
    if (!points.ok()) {
      return points.status();
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
  EXEC_TRY(complete_frame(!read_only && starting.clock.frame_ordinal == 0));
  EXEC_TRY(job.CheckForJobStopRequest());
  Observations observations{
      .command = command,
      .job = job,
      .trace = {},
      .selected = {},
      .attempts = {},
  };
  const LiquidProgressObserver3D observer{
      .context = &observations,
      .observe = &Observations::Observe,
      .observe_attempt = command.trace && command.kind == CommandKind::kRun ? &Observations::ObserveAttempt : nullptr,
  };
  for (std::uint64_t frame = starting.clock.frame_ordinal; frame < target_frame;) {
    EXEC_TRY(job.CheckForJobStopRequest());
    ++frame;
    auto time = FrameTime(frame);
    if (!time.ok()) {
      return time.status();
    }
    observations.trace.clear();
    observations.selected.reset();
    observations.attempts.clear();
    const auto advanced = scheduler->AdvanceFrame(*time, observer);
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
    EXEC_TRY(complete_frame(publish));
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
  }
#else
  static_cast<void>(copy_milliseconds);
#endif
  EXEC_TRY(job.CheckForJobStopRequest());
  return members;
}
}  // namespace

absl::Status Execute(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic) {
  SimulateJob job(writer, command.kind == CommandKind::kRun);
  auto result = ExecuteCommand(command, writer, job, diagnostic);
  const auto final_status = job.Finish(result.status(), diagnostic);
  if (!final_status.ok()) {
    return final_status;
  }
  return writer.Emit("result", *result);
}
#undef EXEC_TRY
}  // namespace fluidity::simulate
