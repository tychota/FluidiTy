#include "effects_execution.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <limits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
// psapi.h requires Windows declarations first.
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <sys/resource.h>
#endif

#include "../src/liquid_cache_files_3d_internal.h"
#include "fluidity/host/surface_stage_3d.h"
#include "fluidity/host/volumes_stage_3d.h"
#include "fluidity/host/whitewater_stage_3d.h"
#include "simulate_job.h"

namespace fluidity::simulate {
namespace {
// Process high-water mark, including source decoding, geometry and publication.
// This is cumulative for the process, not per-frame or device allocation bytes.
absl::StatusOr<std::size_t> PeakProcessResidentBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    return absl::UnavailableError("could not measure Mesh process memory");
  }
  return counters.PeakWorkingSetSize;
#else
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss <= 0) {
    return absl::UnavailableError("could not measure Mesh process memory");
  }
#if defined(__APPLE__)
  return static_cast<std::size_t>(usage.ru_maxrss);
#else
  if (static_cast<std::uintmax_t>(usage.ru_maxrss) > std::numeric_limits<std::size_t>::max() / 1024) {
    return absl::OutOfRangeError("Mesh process memory measurement overflow");
  }
  return 1024 * static_cast<std::size_t>(usage.ru_maxrss);
#endif
#endif
}
std::string Number(double value) {
  std::array<char, 64> text{};
  const auto converted = std::to_chars(
      text.data(),
      text.data() + text.size(),
      value,
      std::chars_format::general,
      std::numeric_limits<double>::max_digits10
  );
  return {text.data(), converted.ptr};
}
std::string Digest(const LiquidDigest3D& digest) {
  return JsonString(LiquidDigestHex3D(digest).data());
}

struct FrameDescription {
  LiquidClock3D clock;
  LiquidDigest3D payload_digest;
  std::string measurements;
};

// Both host stages use the same lease, cancellation, prefix restoration and
// reporting protocol. The stage owner supplies its checked frame description.
template <typename Stage, typename DescribeFrame>
absl::StatusOr<std::string> RunStage(
    const Command& command,
    SimulateJob& job,
    Stage& stage,
    std::string_view stage_name,
    DescribeFrame describe_frame
) {
  const bool replay = command.kind == CommandKind::kReplay;
  if (!replay) {
    const auto attached = job.AttachCache(command.output_cache_directory, !command.resume);
    if (!attached.ok()) {
      return attached;
    }
    const auto marked = job.MarkBaking();
    if (!marked.ok()) {
      return marked;
    }
  }
  const auto execution = EffectsExecutionId3D(stage.identity());
  if (!execution.ok()) {
    return execution.status();
  }
  const std::uint64_t target = replay ? *command.frame : command.frame_end;
  std::uint64_t restored_prefix_frames = 0;
  double restore_wall_seconds = 0;
  if (command.resume && target > 0) {
    const auto restore_started = std::chrono::steady_clock::now();
    // Always regenerate the target frame, even if already committed, so the
    // terminal result verifies the requested output and contains its metrics.
    const auto restored = stage.RestoreCommittedPrefix(target - 1, [&] {
      return job.CheckForJobStopRequest();
    });
    if (!restored.ok()) {
      return restored.status();
    }
    restored_prefix_frames = *restored;
    restore_wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - restore_started).count();
  }
  std::string final_members;
  while (stage.next_frame() <= target) {
    const auto stopped = job.CheckForJobStopRequest();
    if (!stopped.ok()) {
      return stopped;
    }
    const auto started = std::chrono::steady_clock::now();
    const LiquidCachePublishControl3D control{
        .check =
            [](void* context, LiquidCachePublishStage3D) {
              return static_cast<SimulateJob*>(context)->CheckForJobStopRequest();
            },
        .context = &job,
    };
    auto result = stage.ProcessNextFrame(control);
    if (!result.ok()) {
      return result.status();
    }
    const double wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto peak_resident_bytes = PeakProcessResidentBytes();
    if (!peak_resident_bytes.ok()) {
      return peak_resident_bytes.status();
    }
    const auto frame = describe_frame(*result);
    const std::string frame_members = "\"command\":" + JsonString(replay ? "replay" : "run")
        + ",\"stage\":" + JsonString(stage_name) + ",\"backend\":\"cpu\",\"read_only\":" + (replay ? "true" : "false")
        + ",\"execution_id\":" + Digest(*execution) + ",\"payload_digest\":" + Digest(frame.payload_digest)
        + frame.measurements + ",\"frame_wall_seconds\":" + Number(wall_seconds)
        + ",\"peak_process_resident_bytes\":" + std::to_string(*peak_resident_bytes);
    final_members = frame_members + ",\"frame\":" + std::to_string(frame.clock.frame_ordinal)
        + ",\"restored_prefix_frames\":" + std::to_string(restored_prefix_frames)
        + ",\"restore_wall_seconds\":" + Number(restore_wall_seconds);
    job.SetSettledClock(LiquidBackend3D::kCpu, frame.clock, true);
    if (!replay && frame.clock.frame_ordinal >= command.frame_start) {
      const auto reported = job.ReportPublishedFrame(frame.clock.frame_ordinal, "," + frame_members);
      if (!reported.ok()) {
        return reported;
      }
    }
  }
  return final_members;
}

absl::StatusOr<VolumesStage3D> OpenVolumes(const Command& command, LiquidDigest3D build) {
  if (command.kind == CommandKind::kReplay) {
    return VolumesStage3D::OpenForReplay(command.cache_directory, command.output_cache_directory, build);
  }
  const LiquidCacheLimits3D limits{.maximum_bytes = 65536};
  const auto bytes = liquid_cache_files_internal::ReadBytes(command.volume_settings_path, limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  const auto settings = DecodeWhitewaterVolumeSettings3D(bytes->values(), limits);
  if (!settings.ok()) {
    return settings.status();
  }
  const auto canonical = EncodeWhitewaterVolumeSettings3D(*settings, limits);
  if (!canonical.ok()) {
    return canonical.status();
  }
  if (!std::ranges::equal(bytes->values(), canonical->values())) {
    return absl::InvalidArgumentError("Volumes settings must be canonical FLVC bytes");
  }
  return VolumesStage3D::OpenForPublication(command.cache_directory, command.output_cache_directory, build, *settings);
}

absl::StatusOr<std::string> RunEffects(const Command& command, SimulateJob& job) {
  const auto build = ExecutableDigest();
  if (!build.ok()) {
    return build.status();
  }
  const bool replay = command.kind == CommandKind::kReplay;
  if (command.stage == CommandStage::kMesh) {
    SurfaceReconstructionOptions3D options;
    options.spacing_m = command.surface_spacing_m;
    options.sigma_m = command.surface_sigma_m;
    auto stage = replay
        ? SurfaceStage3D::OpenForReplay(command.cache_directory, command.output_cache_directory, *build)
        : SurfaceStage3D::OpenForPublication(command.cache_directory, command.output_cache_directory, *build, options);
    if (!stage.ok()) {
      return stage.status();
    }
    return RunStage(command, job, *stage, "mesh", [](const SurfaceStageFrame3D& result) {
      const auto& reconstruction = result.reconstruction;
      const auto& frame = reconstruction.frame;
      double material_volume_m3 = 0;
      for (const auto& volume : frame.volumes) {
        material_volume_m3 += volume.material_m3;
      }
      const double relative_error =
          material_volume_m3 == 0 ? 0 : reconstruction.corrected_metrics.signed_volume / material_volume_m3 - 1;
      return FrameDescription{
          .clock = frame.clock,
          .payload_digest = result.payload_digest,
          .measurements = ",\"vertex_count\":" + std::to_string(frame.mesh.vertices.size())
              + ",\"triangle_count\":" + std::to_string(frame.mesh.triangles.size()) + ",\"retained_ids\":"
              + std::to_string(reconstruction.retained_ids) + ",\"volume_relative_error\":" + Number(relative_error),
      };
    });
  }
  if (command.stage == CommandStage::kVolumes) {
    auto stage = OpenVolumes(command, *build);
    if (!stage.ok()) {
      return stage.status();
    }
    return RunStage(command, job, *stage, "volumes", [](const VolumesStageFrame3D& frame) {
      return FrameDescription{
          .clock = frame.clock,
          .payload_digest = frame.payload_digest,
          .measurements = ",\"source_stage\":\"whitewater\",\"known_voxels\":" + std::to_string(frame.known_voxels)
              + ",\"nonzero_voxels\":" + std::to_string(frame.nonzero_voxels)
              + ",\"foam_inventory_count\":" + Number(frame.foam_inventory_count) + ",\"bubble_inventory_count\":"
              + Number(frame.bubble_inventory_count) + ",\"gas_inventory_m3\":" + Number(frame.gas_inventory_m3),
      };
    });
  }
  if (command.stage != CommandStage::kWhitewater) {
    return absl::InvalidArgumentError("unsupported derived stage");
  }
  auto stage = replay
      ? WhitewaterStage3D::OpenForReplay(command.cache_directory, command.output_cache_directory, *build)
      : WhitewaterStage3D::OpenForPublication(command.cache_directory, command.output_cache_directory, *build);
  if (!stage.ok()) {
    return stage.status();
  }
  const auto source_backend = stage->source_backend();
  return RunStage(command, job, *stage, "whitewater", [source_backend](const WhitewaterStageFrame3D& frame) {
    return FrameDescription{
        .clock = frame.clock,
        .payload_digest = frame.payload_digest,
        .measurements = ",\"source_backend\":" + JsonString(source_backend == LiquidBackend3D::kCuda ? "cuda" : "cpu")
            + ",\"point_count\":" + std::to_string(frame.foam_count + frame.spray_count + frame.bubble_count)
            + ",\"foam_count\":" + std::to_string(frame.foam_count) + ",\"spray_count\":"
            + std::to_string(frame.spray_count) + ",\"bubble_count\":" + std::to_string(frame.bubble_count)
            + ",\"birth_count\":" + std::to_string(frame.birth_count)
            + ",\"expired_count\":" + std::to_string(frame.expired_count)
            + ",\"bubble_fallback_substeps\":" + std::to_string(frame.bubble_fallback_substeps),
    };
  });
}
}  // namespace

absl::Status ExecuteEffects(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic) {
  diagnostic->domain = command.stage == CommandStage::kMesh ? "MESH"
      : command.stage == CommandStage::kWhitewater          ? "WHITEWATER"
                                                            : "VOLUMES";
  diagnostic->field = "cache";
  diagnostic->cause = "accepted source, physical settings, reconstruction or immutable publication failed";
  SimulateJob job(writer, command.kind == CommandKind::kRun);
  auto result = RunEffects(command, job);
  // Derived numerical preconditions carry a specific reason suitable for
  // reporting beside the stable domain error code.
  if (absl::IsFailedPrecondition(result.status()) && !result.status().message().empty()) {
    diagnostic->cause = std::string(result.status().message());
  }
  const auto finalized = job.Finish(result.status(), diagnostic);
  if (!finalized.ok()) {
    return finalized;
  }
  return writer.Emit("result", *result);
}
}  // namespace fluidity::simulate
