#include "gas_execution.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/host/gas_checkpoint_3d.h"
#include "gas_scene_generated.h"
#include "simulate_job.h"
#if defined(FLUIDITY_SIMULATE_CUDA)
#include "fluidity/host/cuda_mac_pressure_3d.h"
#include "simulate_cuda_compatibility.h"
#endif

namespace fluidity::simulate {
namespace {

namespace fs = std::filesystem;
namespace files = liquid_cache_files_internal;
namespace wire = cache::v1;
constexpr char kSceneFile[] = "gas-scene.flgs";
constexpr LiquidCacheLimits3D kLimits{};
constexpr double kDryAirGasConstant = 287.05;

Vec3 FromWire(const wire::Vector3& vector) {
  return {vector.x(), vector.y(), vector.z()};
}

fs::path CheckpointPath(const fs::path& cache, std::uint64_t frame) {
  return cache / "checkpoints" / files::OrdinalName(frame, ".flgc");
}

absl::StatusOr<GasCheckpoint3D> ReadCheckpoint(
    const fs::path& cache,
    std::uint64_t frame,
    const GasUniformDomain3D& domain,
    const LiquidDigest3D& digest
) {
  const auto contents = files::ReadBytes(CheckpointPath(cache, frame), kLimits);
  if (!contents.ok()) {
    return contents.status();
  }
  const auto bytes = contents->values();
  if (bytes.size() <= digest.size()) {
    return absl::DataLossError("gas checkpoint is truncated");
  }
  const auto payload = bytes.subspan(digest.size());
  if (!std::ranges::equal(bytes.first(digest.size()), LiquidContentDigest3D(payload))) {
    return absl::DataLossError("gas checkpoint digest differs from payload");
  }
  auto checkpoint = DecodeGasCheckpoint3D(payload, domain, digest);
  if (checkpoint.ok() && checkpoint->frame != frame) {
    return absl::DataLossError("gas checkpoint frame differs from its publication");
  }
  return checkpoint;
}

absl::Status PublishCheckpoint(const fs::path& cache, const GasCheckpoint3D& checkpoint, const LiquidDigest3D& digest) {
  const auto payload = EncodeGasCheckpoint3D(checkpoint, digest);
  if (!payload.ok()) {
    return payload.status();
  }
  const auto payload_digest = LiquidContentDigest3D(*payload);
  std::vector<std::uint8_t> bytes(payload_digest.begin(), payload_digest.end());
  bytes.insert(bytes.end(), payload->begin(), payload->end());
  return files::PublishBytes(CheckpointPath(cache, checkpoint.frame), bytes, kLimits, {});
}

std::string Result(std::string_view command, std::uint64_t frame, std::size_t cells) {
  return "\"command\":" + JsonString(command) + ",\"model\":\"gas\",\"frame\":" + std::to_string(frame)
      + ",\"cells\":" + std::to_string(cells);
}

std::string VolumeFields(
    const GasCheckpoint3D& checkpoint,
    const GasUniformDomainOptions3D& options,
    const LiquidDigest3D& digest
) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  const auto& [nx, ny, nz] = options.extent;
  const auto origin = options.geometry.world_origin;
  const double spacing = options.geometry.cell_width;
  output << ",\"scene_digest\":" << JsonString(LiquidDigestHex3D(digest).data()) << ",\"extent\":[" << nx << ',' << ny
         << ',' << nz << "],\"spacing_m\":" << spacing << ",\"origin_m\":[" << origin.x << ',' << origin.y << ','
         << origin.z << "],\"gas_fields\":[";
  for (std::size_t index = 0; index < checkpoint.state.cells.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    const auto& cell = checkpoint.state.cells[index];
    const double temperature = (cell.density.enthalpy / cell.density.carrier) / kGasSpecificHeat;
    output << '[' << cell.density.carrier << ',' << temperature << ',' << cell.density.fuel << ','
           << checkpoint.state.fuel_burn_rate[index] << ',' << cell.density.soot << ']';
  }
  output << ']';
  return output.str();
}

absl::StatusOr<std::string> RunGas(const Command& command, SimulateJob& job, Diagnostic* diagnostic) {
  diagnostic->field = "gas_scene";
  diagnostic->cause = "gas scene, model version, build digest, or backend is incompatible";
  const auto path = command.scene_path.empty() ? command.cache_directory / kSceneFile : command.scene_path;
  const auto contents = files::ReadBytes(path, kLimits);
  if (!contents.ok()) {
    return contents.status();
  }
  const auto bytes = contents->values();
  flatbuffers::Verifier verifier(bytes.data(), bytes.size());
  if (!wire::VerifyGasSceneBuffer(verifier)) {
    return absl::InvalidArgumentError("invalid FLGS buffer");
  }
  const auto& scene = *wire::GetGasScene(bytes.data());
  const auto build_digest = ExecutableDigest();
  if (!build_digest.ok()) {
    return build_digest.status();
  }
  const bool check_cache = command.kind == CommandKind::kCheck && command.scene_path.empty();
  if (scene.schema_version() != 1 || scene.model_version() != 1
      || (!check_cache && !std::ranges::equal(*scene.build_digest()->bytes(), *build_digest))
      || (scene.backend() != wire::Backend::Cpu && scene.backend() != wire::Backend::Cuda)
      || scene.open_boundaries() > 63 || !std::isfinite(scene.frame_duration_s()) || scene.frame_duration_s() <= 0.0
      || scene.substeps_per_frame() == 0 || scene.substeps_per_frame() > 4096) {
    return absl::FailedPreconditionError("gas scene version, build or parameters differ");
  }
  const auto backend = scene.backend() == wire::Backend::Cuda ? LiquidBackend3D::kCuda : LiquidBackend3D::kCpu;
  if (command.backend && *command.backend != backend) {
    return absl::FailedPreconditionError("requested backend differs from gas scene");
  }
  const double pressure = scene.ambient_pressure_pa();
  const double temperature = scene.ambient_temperature_k();
  if (!std::isfinite(temperature) || temperature <= 0.0) {
    return absl::InvalidArgumentError("invalid gas ambient temperature");
  }
  const GasScalars ambient{.carrier = pressure / (kDryAirGasConstant * temperature), .enthalpy = 3.5 * pressure};
  const auto& extent = *scene.extent();
  GasUniformDomainOptions3D options{
      .geometry = {FromWire(*scene.world_origin_m()), scene.cell_width_m()},
      .extent = {extent.x(), extent.y(), extent.z()},
      .open_boundary = {},
      .ambient = ambient,
      .ambient_pressure = pressure,
      .initial_velocity = FromWire(*scene.initial_velocity_m_per_s()),
  };
  for (std::size_t side = 0; side < options.open_boundary.size(); ++side) {
    options.open_boundary[side] = (scene.open_boundaries() & (1u << side)) != 0;
  }
  auto domain = CreateGasUniformDomain3D(options);
  if (!domain.ok()) {
    return domain.status();
  }
  const auto digest = LiquidContentDigest3D(bytes);
  if (check_cache) {
    const auto frame = command.frame.value_or(0);
    const auto checked = ReadCheckpoint(command.cache_directory, frame, *domain, digest);
    if (!checked.ok()) {
      return checked.status();
    }
    return Result("check", frame, checked->state.cells.size()) + VolumeFields(*checked, options, digest);
  }

  std::vector<std::vector<GasSourceKey3D>> key_storage;
  std::vector<GasBoxSource3D> emitters;
  key_storage.reserve(scene.emitters()->size());
  emitters.reserve(scene.emitters()->size());
  for (const auto* emitter : *scene.emitters()) {
    auto& keys = key_storage.emplace_back();
    keys.reserve(emitter->keys()->size());
    for (const auto* key : *emitter->keys()) {
      keys.push_back({key->time_s(), key->fuel_kg_per_m3_s(), key->heat_w_per_m3()});
    }
    emitters.push_back({FromWire(*emitter->lower_m()), FromWire(*emitter->upper_m()), keys});
  }
  std::vector<GasDraftKey3D> draft;
  draft.reserve(scene.draft()->size());
  for (const auto* key : *scene.draft()) {
    draft.push_back({key->time_s(), FromWire(key->acceleration_m_per_s2())});
  }
  const auto& wire_reaction = *scene.reaction();
  const GasReaction3D reaction{
      .ignition_temperature = wire_reaction.ignition_temperature_k(),
      .burn_rate = wire_reaction.burn_rate_per_s(),
      .heat_per_fuel = wire_reaction.heat_per_fuel_j_per_kg(),
      .soot_yield = wire_reaction.soot_yield(),
  };
  const auto gravity = FromWire(*scene.gravity_m_per_s2());
  const auto project = [backend](const MacPressureInputs3D& inputs, MacPressureSolverOptions3D settings)
      -> absl::StatusOr<MacPressureProjection3D> {
    if (backend == LiquidBackend3D::kCpu) {
      return ProjectMacPressure3D(inputs, settings);
    }
#if defined(FLUIDITY_SIMULATE_CUDA)
    auto result = CudaProjectMacPressure3D(inputs, settings);
    if (!result.ok()) {
      return result.status();
    }
    return std::move(result->projection);
#else
    return absl::UnavailableError("gas CUDA backend is unavailable in this executable");
#endif
  };
  if (command.kind == CommandKind::kCheck) {
    const auto validated = AdvanceGasFrame3D(
        *domain,
        ambient,
        pressure,
        gravity,
        0.0,
        scene.frame_duration_s(),
        scene.substeps_per_frame(),
        emitters,
        draft,
        reaction,
        project
    );
    if (!validated.ok()) {
      return validated.status();
    }
    return Result("check", 0, domain->cells.size());
  }

  const bool replay = command.kind == CommandKind::kReplay;
  if ((!replay && command.kind != CommandKind::kRun) || command.initial_state_path.has_filename()
      || (replay && (!command.frame || command.substep.value_or(0) != 0))) {
    return absl::InvalidArgumentError("gas supports completed-frame runs and replay");
  }
  const auto target = replay ? *command.frame : command.frame_end;
  if (target > 1000000) {
    return absl::OutOfRangeError("gas frame exceeds execution limit");
  }
  diagnostic->field = "gas_checkpoint";
  diagnostic->cause = "gas checkpoint, cache or accepted frame could not be verified";
  const bool write = !replay && !command.preview;
  files::NativeFile lease;
  if (write) {
    const auto leased = files::Lease(command.cache_directory, lease, !command.resume);
    if (!leased.ok()) {
      return leased;
    }
    std::error_code path_error;
    const bool exists = fs::exists(command.cache_directory / kSceneFile, path_error);
    if (path_error) {
      return absl::UnavailableError("cannot inspect gas cache scene");
    }
    if (!command.resume && exists) {
      return absl::AlreadyExistsError("gas cache already exists; use resume");
    }
    const auto attached = job.AttachCache(command.cache_directory, !command.resume);
    if (!attached.ok()) {
      return attached;
    }
    const auto published = files::PublishBytes(
        command.cache_directory / kSceneFile,
        bytes,
        kLimits,
        {},
        absl::StatusCode::kFailedPrecondition
    );
    if (!published.ok()) {
      return published;
    }
    const auto baking = job.MarkBaking();
    if (!baking.ok()) {
      return baking;
    }
  }
  if (backend == LiquidBackend3D::kCuda) {
#if defined(FLUIDITY_SIMULATE_CUDA)
    auto identity = QueryCudaExecutionIdentity(0, digest);
    if (!identity.ok()) {
      return identity.status();
    }
    if (write && !command.resume) {
      const auto status = PublishCudaCompatibilityRecord(command.cache_directory, identity->compatibility_record);
      if (!status.ok()) {
        return status;
      }
    } else if (replay || command.resume) {
      const auto stored = ReadCudaCompatibilityRecord(command.cache_directory);
      if (!stored.ok()) {
        return stored.status();
      }
      if (*stored != identity->compatibility_record) {
        return absl::FailedPreconditionError("gas CUDA compatibility record differs");
      }
    }
#else
    return absl::UnavailableError("gas CUDA backend is unavailable in this executable");
#endif
  }

  GasCheckpoint3D checkpoint{
      .frame = 0,
      .time = 0.0,
      .accepted_steps = 0,
      .state =
          {
              .cells = domain->cells,
              .markers = domain->markers,
              .face_velocities = domain->face_velocities,
              .fuel_burn_rate = std::vector<double>(domain->cells.size()),
          },
  };
  if (command.resume || replay) {
    const std::uint64_t last = replay && target > 0 ? target - 1 : target;
    for (std::uint64_t frame = 0; frame <= last; ++frame) {
      auto stored = ReadCheckpoint(command.cache_directory, frame, *domain, digest);
      if (!stored.ok()) {
        if (command.resume && absl::IsNotFound(stored.status())) {
          break;
        }
        return stored.status();
      }
      if (stored->time != static_cast<double>(frame) * scene.frame_duration_s()
          || stored->accepted_steps != frame * scene.substeps_per_frame()) {
        return absl::DataLossError("gas checkpoint clock differs from scene schedule");
      }
      checkpoint = std::move(*stored);
    }
  }
  if (write && checkpoint.frame == 0) {
    const auto status = PublishCheckpoint(command.cache_directory, checkpoint, digest);
    if (!status.ok()) {
      return status;
    }
    job.SetSettledClock(backend, {}, true);
    const auto reported = job.ReportPublishedFrame(0, {});
    if (!reported.ok()) {
      return reported;
    }
  }
  for (auto frame = checkpoint.frame + 1; frame <= target; ++frame) {
    const auto stop = job.CheckForJobStopRequest();
    if (!stop.ok()) {
      return stop;
    }
    domain->cells = checkpoint.state.cells;
    domain->markers = std::move(checkpoint.state.markers);
    domain->face_velocities = std::move(checkpoint.state.face_velocities);
    const double time = static_cast<double>(frame) * scene.frame_duration_s();
    auto advanced = AdvanceGasFrame3D(
        *domain,
        ambient,
        pressure,
        gravity,
        checkpoint.time,
        time,
        scene.substeps_per_frame(),
        emitters,
        draft,
        reaction,
        project
    );
    if (!advanced.ok()) {
      return advanced.status();
    }
    checkpoint = {frame, time, frame * scene.substeps_per_frame(), std::move(*advanced)};
    if (write) {
      const auto status = PublishCheckpoint(command.cache_directory, checkpoint, digest);
      if (!status.ok()) {
        return status;
      }
      job.SetSettledClock(backend, {time, frame, 0, checkpoint.accepted_steps, checkpoint.accepted_steps});
      const auto reported = job.ReportPublishedFrame(frame, {});
      if (!reported.ok()) {
        return reported;
      }
    }
  }
  if (replay) {
    auto expected = ReadCheckpoint(command.cache_directory, target, *domain, digest);
    if (!expected.ok()) {
      return expected.status();
    }
    auto actual_bytes = EncodeGasCheckpoint3D(checkpoint, digest);
    auto expected_bytes = EncodeGasCheckpoint3D(*expected, digest);
    if (!actual_bytes.ok() || !expected_bytes.ok() || *actual_bytes != *expected_bytes) {
      return absl::DataLossError("gas replay differs from committed frame");
    }
  }
  return Result(replay ? "replay" : "run", target, checkpoint.state.cells.size());
}

}  // namespace

bool IsGasCommand(const Command& command) {
  if (!command.scene_path.empty()) {
    return command.scene_path.extension() == ".flgs";
  }
  if (command.cache_directory.empty()) {
    return false;
  }
  std::error_code error;
  return fs::exists(command.cache_directory / kSceneFile, error);
}

absl::Status ExecuteGasCommand(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic) {
  diagnostic->domain = "SIMULATE";
  SimulateJob job(writer, command.kind == CommandKind::kRun);
  const auto result = RunGas(command, job, diagnostic);
  const auto final = job.Finish(result.status(), diagnostic);
  if (!final.ok()) {
    return final;
  }
  return writer.Emit("result", *result);
}

}  // namespace fluidity::simulate
