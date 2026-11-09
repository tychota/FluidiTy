#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <optional>
#include <thread>
#include <vector>

#include "../src/liquid_cache_files_3d_internal.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "fluidity/host/simulate_cli.h"
#if defined(FLUIDITY_SIMULATE_CUDA)
#include "fluidity/host/cuda_liquid_snapshot_3d.h"
#include "fluidity/host/cuda_liquid_substep_3d.h"
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
  const auto& clock = state.clock;
  std::string result = "\"frame\":" + std::to_string(clock.frame_ordinal)
      + ",\"substep\":" + std::to_string(clock.substep_ordinal) + ",\"epoch\":" + std::to_string(clock.epoch)
      + ",\"accepted_step\":" + std::to_string(clock.accepted_step_ordinal) + ",\"time_seconds\":" + Number(clock.time)
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
struct Observations {
  const Command& command;
  std::vector<std::string> trace;
  std::optional<std::string> selected;
  static absl::Status
  Observe(void* context, const LiquidState3D& state, const LiquidStepDiagnostics3D& diagnostics, bool rollback) {
    auto& self = *static_cast<Observations*>(context);
    if (rollback) {
      self.trace.clear();
      self.selected.reset();
      return absl::OkStatus();
    }
    auto members = StateMembers(state, diagnostics);
    if (!members.ok()) {
      return members.status();
    }
    if (self.command.trace) {
      self.trace.push_back(*members);
    }
    if (self.command.kind == CommandKind::kStep
        || (self.command.frame == state.clock.frame_ordinal && self.command.substep == state.clock.substep_ordinal)) {
      self.selected = *members;
    }
    return self.command.kind == CommandKind::kStep ? absl::CancelledError("") : absl::OkStatus();
  }
};
}  // namespace

absl::Status Execute(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic) {
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
  diagnostic->field = "cache";
  diagnostic->cause = "cache compatibility or verified retained checkpoint failed";
  if (exists) {
    if (!read_only && !command.resume) {
      return absl::AlreadyExistsError("");
    }
    auto opened = read_only ? LiquidCache3D::OpenForReplay(command.cache_directory, *fingerprint)
                            : LiquidCache3D::Open(command.cache_directory, *fingerprint);
    if (!opened.ok()) {
      return opened.status();
    }
    cache = std::move(*opened);
  } else if (replay || command.resume) {
    return absl::NotFoundError("");
  }
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
  if (!cache && !read_only) {
    auto created = LiquidCache3D::Create(command.cache_directory, scene.View());
    if (!created.ok()) {
      return created.status();
    }
    cache = std::move(*created);
  }
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
    EXEC_TRY(cache->PublishFrame(*points, checkpoint ? state : nullptr, checkpoint ? &resume : nullptr));
    return writer.Emit("frame", "\"published\":true,\"frame\":" + std::to_string(state->clock.frame_ordinal));
  };
  diagnostic->field = "frame";
  diagnostic->cause = "completed state, point publication or deterministic replay verification failed";
  EXEC_TRY(complete_frame(!read_only && starting.clock.frame_ordinal == 0));
  Observations observations{command, {}, {}};
  for (std::uint64_t frame = starting.clock.frame_ordinal; frame < target_frame;) {
    ++frame;
    auto time = FrameTime(frame);
    if (!time.ok()) {
      return time.status();
    }
    observations.trace.clear();
    observations.selected.reset();
    const auto advanced = scheduler->AdvanceFrame(*time, {&observations, &Observations::Observe});
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
  return writer.Emit("result", members);
}
#undef EXEC_TRY
}  // namespace fluidity::simulate
