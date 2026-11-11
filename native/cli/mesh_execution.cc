#include "mesh_execution.h"

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

#include "fluidity/host/surface_stage_3d.h"
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

absl::StatusOr<std::string> RunMesh(const Command& command, SimulateJob& job) {
  const auto build = ExecutableDigest();
  if (!build.ok()) {
    return build.status();
  }
  const bool replay = command.kind == CommandKind::kReplay;
  SurfaceReconstructionOptions3D options;
  options.spacing_m = command.surface_spacing_m;
  options.sigma_m = command.surface_sigma_m;
  auto stage = replay
      ? SurfaceStage3D::OpenForReplay(command.cache_directory, command.output_cache_directory, *build)
      : SurfaceStage3D::OpenForPublication(command.cache_directory, command.output_cache_directory, *build, options);
  if (!stage.ok()) {
    return stage.status();
  }
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
  const auto execution = EffectsExecutionId3D(stage->identity());
  if (!execution.ok()) {
    return execution.status();
  }
  const std::uint64_t target = replay ? *command.frame : command.frame_end;
  std::string final_members;
  while (stage->next_frame() <= target) {
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
    auto result = stage->ProcessNextFrame(control);
    if (!result.ok()) {
      return result.status();
    }
    const auto& reconstructed = result->reconstruction;
    const auto& frame = reconstructed.frame;
    double material_volume_m3 = 0;
    for (const auto& volume : frame.volumes) {
      material_volume_m3 += volume.material_m3;
    }
    const double relative_error =
        material_volume_m3 == 0 ? 0 : reconstructed.corrected_metrics.signed_volume / material_volume_m3 - 1;
    const double wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto peak_resident_bytes = PeakProcessResidentBytes();
    if (!peak_resident_bytes.ok()) {
      return peak_resident_bytes.status();
    }
    const std::string frame_members = "\"command\":" + JsonString(replay ? "replay" : "run")
        + ",\"stage\":\"mesh\",\"backend\":\"cpu\",\"read_only\":" + (replay ? "true" : "false")
        + ",\"execution_id\":" + Digest(*execution) + ",\"payload_digest\":" + Digest(result->payload_digest)
        + ",\"vertex_count\":" + std::to_string(frame.mesh.vertices.size())
        + ",\"triangle_count\":" + std::to_string(frame.mesh.triangles.size())
        + ",\"retained_ids\":" + std::to_string(reconstructed.retained_ids)
        + ",\"volume_relative_error\":" + Number(relative_error) + ",\"frame_wall_seconds\":" + Number(wall_seconds)
        + ",\"peak_process_resident_bytes\":" + std::to_string(*peak_resident_bytes);
    final_members = frame_members + ",\"frame\":" + std::to_string(frame.clock.frame_ordinal);
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
}  // namespace

absl::Status ExecuteMesh(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic) {
  diagnostic->domain = "MESH";
  diagnostic->field = "cache";
  diagnostic->cause = "accepted source, physical settings, geometry or immutable Mesh publication failed";
  SimulateJob job(writer, command.kind == CommandKind::kRun);
  auto result = RunMesh(command, job);
  const auto finalized = job.Finish(result.status(), diagnostic);
  if (!finalized.ok()) {
    return finalized;
  }
  return writer.Emit("result", *result);
}
}  // namespace fluidity::simulate
