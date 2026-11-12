#include <Alembic/AbcCoreOgawa/All.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <set>
#include <span>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#include "blender_resolve.h"
#include "fluidity/host/effects_cache_3d.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/surface_cache_3d.h"
#include "fluidity/host/surface_settings_3d.h"
#include "fluidity/host/whitewater_cache_3d.h"
#include "fluidity/host/whitewater_volume_cache_3d.h"
#include "point_samples.h"

namespace fluidity::blender {
namespace {
namespace abc = Alembic::Abc;
namespace geom = Alembic::AbcGeom;
namespace ogawa = Alembic::AbcCoreOgawa;
namespace fs = std::filesystem;

constexpr std::uint64_t kMaximumExportFrame = 10'000;
constexpr double kFramesPerSecond = 24.0;

absl::StatusOr<std::uint64_t> ParseFrame(const fs::path& input) {
  const auto text = input.u8string();
  const auto* begin = reinterpret_cast<const char*>(text.data());
  const auto* end = begin + text.size();
  std::uint64_t frame = 0;
  const auto [next, error] = std::from_chars(begin, end, frame);
  if (error != std::errc{} || next != end || frame > kMaximumExportFrame) {
    return absl::InvalidArgumentError("expected a frame ordinal from 0 through 10000");
  }
  return frame;
}

absl::StatusOr<LiquidCommittedFrame3D> ReadFrame(const LiquidCache3D& cache, std::uint64_t frame_ordinal) {
  auto committed = cache.ReadCommittedFrame(frame_ordinal);
  if (!committed.ok()) {
    return committed.status();
  }
  const auto& clock = committed->manifest.clock;
  if (clock.frame_ordinal != frame_ordinal || clock.substep_ordinal != 0
      || clock.time != static_cast<double>(frame_ordinal) / kFramesPerSecond) {
    return absl::DataLossError("committed frame clock does not match the M1 cadence");
  }
  return committed;
}

void PrintBlenderVector(const Vec3& solver_vector) {
  std::cout << '[' << solver_vector.x << ',' << -solver_vector.z << ',' << solver_vector.y << ']';
}

absl::Status Inspect(const LiquidCache3D& cache, std::uint64_t frame_ordinal) {
  const auto committed = ReadFrame(cache, frame_ordinal);
  if (!committed.ok()) {
    return committed.status();
  }
  std::cout << "{\"frame\":" << frame_ordinal << ",\"time_seconds\":" << committed->manifest.clock.time
            << ",\"execution_id\":\"" << LiquidDigestHex3D(committed->manifest.execution_id).data()
            << "\",\"points_digest\":\"" << LiquidDigestHex3D(committed->manifest.points.digest).data()
            << "\",\"points\":[";
  const char* separator = "";
  for (const auto& point : committed->points.points.values()) {
    std::cout << separator << "{\"id\":" << point.id << ",\"position\":";
    PrintBlenderVector(point.world_position);
    std::cout << ",\"velocity\":";
    PrintBlenderVector(point.world_velocity);
    std::cout << ",\"radius\":" << point.radius << '}';
    separator = ",";
  }
  std::cout << "]}\n";
  return std::cout ? absl::OkStatus() : absl::UnavailableError("could not write inspection result");
}

bool FitsAlembicFloat(double value) {
  return std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max();
}

bool FitsAlembicFloat(const Vec3& value) {
  return FitsAlembicFloat(value.x) && FitsAlembicFloat(value.y) && FitsAlembicFloat(value.z);
}

// Solver and Alembic are both Y-up. Blender applies the inverse axis mapping on import.
abc::V3f Narrow(const Vec3& solver_vector) {
  return {
      static_cast<float>(solver_vector.x),
      static_cast<float>(solver_vector.y),
      static_cast<float>(solver_vector.z),
  };
}

absl::Status WriteArchive(std::ostream& output, const LiquidCache3D& cache, std::uint64_t last_frame) {
  abc::OArchive archive(ogawa::WriteArchive()(&output, abc::MetaData{}));
  const auto sampling = archive.addTimeSampling({1.0 / kFramesPerSecond, 0.0});
  geom::OPoints points_object(archive.getTop(), "FluidiTyPoints", sampling);
  for (std::uint64_t frame = 0; frame <= last_frame; ++frame) {
    const auto committed = ReadFrame(cache, frame);
    if (!committed.ok()) {
      return committed.status();
    }
    const auto points = committed->points.points.values();
    const std::size_t storage_count = points.empty() ? 1 : points.size();
    std::vector<abc::V3f> positions(storage_count);
    std::vector<abc::V3f> velocities(storage_count);
    std::vector<std::uint64_t> ids(storage_count);
    std::vector<float> diameters(storage_count);
    for (std::size_t index = 0; index < points.size(); ++index) {
      const auto& point = points[index];
      const double diameter = 2 * point.radius;
      if (!FitsAlembicFloat(point.world_position) || !FitsAlembicFloat(point.world_velocity)
          || !FitsAlembicFloat(diameter) || static_cast<float>(diameter) <= 0) {
        return absl::OutOfRangeError("committed point cannot be represented in the Alembic carrier");
      }
      positions[index] = Narrow(point.world_position);
      velocities[index] = Narrow(point.world_velocity);
      ids[index] = point.id;
      diameters[index] = static_cast<float>(diameter);
    }
    WritePointSample(
        points_object.getSchema(),
        positions.data(),
        ids.data(),
        velocities.data(),
        diameters.data(),
        points.size()
    );
  }
  return absl::OkStatus();
}

absl::StatusOr<EffectsCommit3D> ReadDerivedFrame(const EffectsCache3D& cache, std::uint64_t frame) {
  auto committed = cache.ReadFrame(frame);
  if (!committed.ok()) {
    return committed.status();
  }
  if (committed->clock.time != static_cast<double>(frame) / kFramesPerSecond) {
    return absl::DataLossError("derived frame clock does not match Blender cache cadence");
  }
  return committed;
}

absl::StatusOr<SurfaceFrame3D> ReadSurfaceFrame(const EffectsCache3D& cache, std::uint64_t ordinal) {
  const auto committed = ReadDerivedFrame(cache, ordinal);
  if (!committed.ok()) {
    return committed.status();
  }
  auto frame = DecodeSurfaceFrame3D(committed->payload.values());
  if (!frame.ok()) {
    return frame.status();
  }
  if (frame->clock != committed->clock) {
    return absl::DataLossError("surface payload clock differs from its manifest");
  }
  return frame;
}

// USDA groups samples by property. Re-read committed frames to bound working
// memory to one decoded mesh plus the sets of published attribute IDs.
template <class WriteValues>
absl::Status WriteSurfaceUsdSamples(
    std::ostream& output,
    const EffectsCache3D& cache,
    std::uint64_t last_frame,
    std::string_view property,
    const WriteValues& write_values
) {
  output << "    " << property << ".timeSamples = {\n";
  for (std::uint64_t ordinal = 0; ordinal <= last_frame; ++ordinal) {
    const auto frame = ReadSurfaceFrame(cache, ordinal);
    if (!frame.ok()) {
      return frame.status();
    }
    output << "        " << ordinal << ": [";
    const auto status = write_values(frame->mesh);
    if (!status.ok()) {
      return status;
    }
    output << "],\n";
    if (!output) {
      return absl::UnavailableError("could not write USD surface samples");
    }
  }
  output << "    }\n";
  return absl::OkStatus();
}

absl::Status WriteSurfaceUsd(std::ostream& output, const EffectsCache3D& cache, std::uint64_t last_frame) {
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<float>::max_digits10) << "#usda 1.0\n(\n"
         << "    defaultPrim = \"FluidiTySurface\"\n"
         << "    startTimeCode = 0\n"
         << "    endTimeCode = " << last_frame << "\n"
         << "    timeCodesPerSecond = " << kFramesPerSecond << "\n"
         << "    framesPerSecond = " << kFramesPerSecond << "\n"
         << "    metersPerUnit = 1\n"
         << "    upAxis = \"Z\"\n)\n"
         << "def Mesh \"FluidiTySurface\"\n{\n"
         << "    uniform token subdivisionScheme = \"none\"\n";
  std::set<std::uint64_t> phase_ids;
  std::set<std::uint64_t> substance_ids;
  auto status = WriteSurfaceUsdSamples(output, cache, last_frame, "point3f[] points", [&](const SurfaceMesh3D& mesh) {
    const char* separator = "";
    for (const auto& vertex : mesh.vertices) {
      if (!FitsAlembicFloat(vertex.position)) {
        return absl::OutOfRangeError("surface position cannot fit a USD float point");
      }
      const auto position = Narrow(vertex.position);
      output << separator << '(' << position.x << ", " << -position.z << ", " << position.y << ')';
      separator = ", ";
      if (vertex.constituent_count > 0) {
        phase_ids.insert(vertex.phase_id);
      }
      for (std::size_t index = 0; index < vertex.constituent_count; ++index) {
        substance_ids.insert(vertex.constituents[index].substance_id);
      }
    }
    return absl::OkStatus();
  });
  if (!status.ok()) {
    return status;
  }
  status = WriteSurfaceUsdSamples(output, cache, last_frame, "int[] faceVertexCounts", [&](const SurfaceMesh3D& mesh) {
    const char* separator = "";
    for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
      output << separator << '3';
      separator = ", ";
    }
    return absl::OkStatus();
  });
  if (!status.ok()) {
    return status;
  }
  status = WriteSurfaceUsdSamples(output, cache, last_frame, "int[] faceVertexIndices", [&](const SurfaceMesh3D& mesh) {
    const char* separator = "";
    for (const auto& triangle : mesh.triangles) {
      output << separator << triangle[0] << ", " << triangle[1] << ", " << triangle[2];
      separator = ", ";
    }
    return absl::OkStatus();
  });
  if (!status.ok()) {
    return status;
  }
  const auto write_attribute = [&](std::string_view prefix, std::uint64_t id, auto sample) {
    const std::string property = "float[] primvars:" + std::string(prefix) + std::to_string(id);
    output << "    " << property << " (\n        interpolation = \"vertex\"\n    )\n";
    return WriteSurfaceUsdSamples(output, cache, last_frame, property, [&](const SurfaceMesh3D& mesh) {
      const char* separator = "";
      for (const auto& vertex : mesh.vertices) {
        output << separator << static_cast<float>(sample(vertex, id));
        separator = ", ";
      }
      return absl::OkStatus();
    });
  };
  for (const auto id : phase_ids) {
    status = write_attribute("fluidity_phase_", id, [](const SurfaceVertex3D& vertex, std::uint64_t phase_id) {
      return vertex.constituent_count > 0 && vertex.phase_id == phase_id;
    });
    if (!status.ok()) {
      return status;
    }
  }
  for (const auto id : substance_ids) {
    status = write_attribute("fluidity_fraction_", id, [](const SurfaceVertex3D& vertex, std::uint64_t substance_id) {
      for (std::size_t index = 0; index < vertex.constituent_count; ++index) {
        const auto& constituent = vertex.constituents[index];
        if (constituent.substance_id == substance_id) {
          return constituent.volume_fraction;
        }
      }
      return 0.0;
    });
    if (!status.ok()) {
      return status;
    }
  }
  output << "}\n";
  return output ? absl::OkStatus() : absl::UnavailableError("could not finish USD surface export");
}
absl::Status WriteSurfaceArchive(std::ostream& output, const EffectsCache3D& cache, std::uint64_t last_frame) {
  abc::OArchive archive(ogawa::WriteArchive()(&output, abc::MetaData{}));
  const auto sampling = archive.addTimeSampling({1.0 / kFramesPerSecond, 0.0});
  geom::OPolyMesh surface(archive.getTop(), "FluidiTySurface", sampling);
  for (std::uint64_t frame_ordinal = 0; frame_ordinal <= last_frame; ++frame_ordinal) {
    const auto committed = ReadDerivedFrame(cache, frame_ordinal);
    if (!committed.ok()) {
      return committed.status();
    }
    const auto frame = DecodeSurfaceFrame3D(committed->payload.values());
    if (!frame.ok()) {
      return frame.status();
    }
    if (frame->clock != committed->clock) {
      return absl::DataLossError("surface payload clock differs from its manifest");
    }
    const auto& mesh = frame->mesh;
    const std::size_t storage_vertices = std::max(std::size_t{1}, mesh.vertices.size());
    std::vector<abc::V3f> positions(storage_vertices);
    std::vector<abc::V3f> velocities(storage_vertices);
    for (std::size_t vertex_index = 0; vertex_index < mesh.vertices.size(); ++vertex_index) {
      const auto& vertex = mesh.vertices[vertex_index];
      if (!FitsAlembicFloat(vertex.position) || !FitsAlembicFloat(vertex.geometric_velocity)) {
        return absl::OutOfRangeError("surface vertex or geometric velocity cannot fit the Alembic carrier");
      }
      positions[vertex_index] = Narrow(vertex.position);
      velocities[vertex_index] = Narrow(vertex.geometric_velocity);
    }
    std::vector<std::int32_t> face_indices(std::max(std::size_t{1}, 3 * mesh.triangles.size()));
    const std::vector<std::int32_t> face_counts(std::max(std::size_t{1}, mesh.triangles.size()), 3);
    for (std::size_t face_index = 0; face_index < mesh.triangles.size(); ++face_index) {
      // Blender reverses Alembic polygon winding on import. Apply the inverse
      // here so the native reader restores the solver's outward orientation.
      const auto& triangle = mesh.triangles[face_index];
      face_indices[3 * face_index] = static_cast<std::int32_t>(triangle[0]);
      face_indices[3 * face_index + 1] = static_cast<std::int32_t>(triangle[2]);
      face_indices[3 * face_index + 2] = static_cast<std::int32_t>(triangle[1]);
    }
    geom::OPolyMeshSchema::Sample sample(
        abc::P3fArraySample(positions.data(), mesh.vertices.size()),
        abc::Int32ArraySample(face_indices.data(), 3 * mesh.triangles.size()),
        abc::Int32ArraySample(face_counts.data(), mesh.triangles.size())
    );
    sample.setVelocities(abc::V3fArraySample(velocities.data(), mesh.vertices.size()));
    surface.getSchema().set(sample);
  }
  return absl::OkStatus();
}

absl::Status
WriteFoamArchive(std::ostream& output, const EffectsCache3D& cache, std::uint64_t last_frame, double radius_m) {
  const double diameter_m = 2 * radius_m;
  if (!FitsAlembicFloat(diameter_m) || static_cast<float>(diameter_m) <= 0) {
    return absl::OutOfRangeError("foam diameter cannot fit the Alembic carrier");
  }
  abc::OArchive archive(ogawa::WriteArchive()(&output, abc::MetaData{}));
  const auto sampling = archive.addTimeSampling({1.0 / kFramesPerSecond, 0.0});
  geom::OPoints foam(archive.getTop(), "FluidiTyFoam", sampling);
  for (std::uint64_t frame_ordinal = 0; frame_ordinal <= last_frame; ++frame_ordinal) {
    const auto committed = ReadDerivedFrame(cache, frame_ordinal);
    if (!committed.ok()) {
      return committed.status();
    }
    const auto frame = DecodeWhitewaterFrame3D(committed->payload.values());
    if (!frame.ok()) {
      return frame.status();
    }
    if (frame->clock != committed->clock) {
      return absl::DataLossError("whitewater payload clock differs from its manifest");
    }
    const auto foam_count = static_cast<std::size_t>(std::ranges::count_if(frame->points, [](const auto& record) {
      return record.point.kind == WhitewaterKind3D::kFoam;
    }));
    const auto storage_count = std::max(std::size_t{1}, foam_count);
    std::vector<abc::V3f> positions(storage_count);
    std::vector<abc::V3f> velocities(storage_count);
    std::vector<std::uint64_t> ids(storage_count);
    const std::vector<float> diameters(storage_count, static_cast<float>(diameter_m));
    std::size_t point_index = 0;
    for (const auto& record : frame->points) {
      if (record.point.kind != WhitewaterKind3D::kFoam) {
        continue;
      }
      const auto& point = record.point.kinematics;
      if (!FitsAlembicFloat(point.position) || !FitsAlembicFloat(point.velocity)) {
        return absl::OutOfRangeError("foam point or velocity cannot fit the Alembic carrier");
      }
      positions[point_index] = Narrow(point.position);
      velocities[point_index] = Narrow(point.velocity);
      ids[point_index] = record.id;
      ++point_index;
    }
    WritePointSample(foam.getSchema(), positions.data(), ids.data(), velocities.data(), diameters.data(), foam_count);
  }
  return absl::OkStatus();
}

// Stage next to the destination and publish without replacing another archive.
struct StagedArchive {
  fs::path directory;
  fs::path file;
  ~StagedArchive() {
    std::error_code error;
    fs::remove(file, error);
    fs::remove(directory, error);
  }
};

absl::Status PublishArchive(const fs::path& destination, absl::FunctionRef<absl::Status(std::ostream&)> write_archive) {
  auto staging = destination;
  staging += ".tmp";
  std::error_code error;
  if (!fs::create_directory(staging, error)) {
    return absl::UnavailableError("cannot exclusively create Alembic staging directory");
  }
  const StagedArchive pending{staging, staging / "points.abc"};
  // Open with a filesystem path rather than passing a narrow filename to Alembic.
  std::ofstream output(pending.file, std::ios::binary);
  if (!output) {
    return absl::UnavailableError("cannot open Alembic staging file");
  }
  const auto wrote = write_archive(output);
  output.close();
  if (!wrote.ok()) {
    return wrote;
  }
  if (!output) {
    return absl::UnavailableError("could not close Alembic archive");
  }
#if defined(_WIN32)
  if (!MoveFileExW(pending.file.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH)) {
    return absl::UnavailableError(
        "could not publish new Alembic archive (Windows error " + std::to_string(GetLastError()) + ")"
    );
  }
#else
  fs::create_hard_link(pending.file, destination, error);
  if (error) {
    return absl::UnavailableError("could not publish new Alembic archive");
  }
#endif
  return absl::OkStatus();
}

absl::Status
ExportEffects(const fs::path& root, std::uint64_t last_frame, const fs::path& destination, bool surface, bool usd) {
  const auto cache = EffectsCache3D::OpenForInspection(root);
  if (!cache.ok()) {
    return cache.status();
  }
  const auto& identity = cache->identity();
  if (identity.stage != (surface ? EffectsStage3D::kMesh : EffectsStage3D::kWhitewater)) {
    return absl::FailedPreconditionError("derived cache stage differs from requested Blender carrier");
  }
  const auto settings = cache->ReadStageSettings();
  if (!settings.ok()) {
    return settings.status();
  }
  const auto execution = EffectsExecutionId3D(identity);
  if (!execution.ok()) {
    return execution.status();
  }
  absl::Status written;
  if (surface) {
    const auto resolved = DecodeSurfaceSettings3D(settings->values());
    if (!resolved.ok()) {
      return resolved.status();
    }
    written = PublishArchive(destination, [&](std::ostream& output) {
      return usd ? WriteSurfaceUsd(output, *cache, last_frame) : WriteSurfaceArchive(output, *cache, last_frame);
    });
  } else {
    const auto resolved = DecodeWhitewaterSettings3D(settings->values());
    if (!resolved.ok()) {
      return resolved.status();
    }
    written = PublishArchive(destination, [&](std::ostream& output) {
      return WriteFoamArchive(output, *cache, last_frame, resolved->transport.classification_radius_m);
    });
  }
  if (!written.ok()) {
    return written;
  }
  std::cout << "{\"last_frame\":" << last_frame << ",\"execution_id\":\"" << LiquidDigestHex3D(*execution).data()
            << "\",\"source_execution_id\":\"" << LiquidDigestHex3D(identity.upstream_execution).data() << "\"}\n";
  return std::cout ? absl::OkStatus() : absl::UnavailableError("could not write Alembic export identity");
}

absl::Status InspectVolume(const fs::path& root, std::uint64_t frame) {
  const auto cache = EffectsCache3D::OpenForInspection(root);
  if (!cache.ok()) {
    return cache.status();
  }
  const auto& identity = cache->identity();
  if (identity.stage != EffectsStage3D::kVolumes) {
    return absl::FailedPreconditionError("expected a Volumes cache");
  }
  const auto settings = cache->ReadStageSettings();
  if (!settings.ok()) {
    return settings.status();
  }
  const auto resolved = DecodeWhitewaterVolumeSettings3D(settings->values());
  if (!resolved.ok()) {
    return resolved.status();
  }
  const auto committed = ReadDerivedFrame(*cache, frame);
  if (!committed.ok()) {
    return committed.status();
  }
  const auto decoded = DecodeWhitewaterVolumeFrame3D(committed->payload.values(), resolved->maximum_voxels);
  if (!decoded.ok()) {
    return decoded.status();
  }
  const auto& volume = decoded->volume;
  const auto& grid = volume.grid;
  const auto& expected = resolved->grid;
  if (decoded->clock != committed->clock || grid.extent != expected.extent || grid.spacing_m != expected.spacing_m
      || grid.origin.x != expected.origin.x || grid.origin.y != expected.origin.y
      || grid.origin.z != expected.origin.z) {
    return absl::DataLossError("volume frame grid or clock differs from its cache");
  }
  const auto execution = EffectsExecutionId3D(identity);
  if (!execution.ok()) {
    return execution.status();
  }
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << "{\"frame\":" << frame
            << ",\"execution_id\":\"" << LiquidDigestHex3D(*execution).data() << "\",\"source_execution_id\":\""
            << LiquidDigestHex3D(identity.upstream_execution).data() << "\",\"payload_digest\":\""
            << LiquidDigestHex3D(committed->payload_digest).data() << "\",\"origin_m\":[" << grid.origin.x << ','
            << grid.origin.y << ',' << grid.origin.z << "],\"spacing_m\":" << grid.spacing_m << ",\"extent\":["
            << grid.extent[0] << ',' << grid.extent[1] << ',' << grid.extent[2] << "],\"known_runs\":[";
  bool first = true;
  for (std::size_t start = 0; start < volume.validity.size();) {
    if (volume.validity[start] == 0) {
      ++start;
      continue;
    }
    std::size_t end = start + 1;
    while (end < volume.validity.size() && volume.validity[end] != 0) {
      ++end;
    }
    std::cout << (first ? "" : ",") << '[' << start << ',' << end - start << ']';
    first = false;
    start = end;
  }
  std::cout << "],\"density\":[";
  first = true;
  for (std::size_t index = 0; index < volume.validity.size(); ++index) {
    const double foam = volume.foam_count_per_m3[index];
    const double bubbles = volume.bubble_count_per_m3[index];
    const double gas = volume.gas_fraction[index];
    if (foam != 0 || bubbles != 0 || gas != 0) {
      std::cout << (first ? "" : ",") << '[' << index << ',' << foam << ',' << bubbles << ',' << gas << ']';
      first = false;
    }
  }
  std::cout << "]}\n";
  return std::cout ? absl::OkStatus() : absl::UnavailableError("could not write volume inspection record");
}

absl::Status Run(std::span<const fs::path> args) {
  if (!args.empty() && args[0] == "resolve") {
    return ResolveBlenderScene(args);
  }
  const bool inspect = args.size() == 3 && args[0] == "inspect";
  const bool inspect_volume = args.size() == 3 && args[0] == "inspect-volume";
  const bool export_points = args.size() == 4 && args[0] == "export";
  const bool export_surface = args.size() == 4 && args[0] == "export-surface";
  const bool export_surface_usd = args.size() == 4 && args[0] == "export-surface-usd";
  const bool export_foam = args.size() == 4 && args[0] == "export-foam";
  if (!inspect && !inspect_volume && !export_points && !export_surface && !export_surface_usd && !export_foam) {
    return absl::InvalidArgumentError(
        "expected a cache and frame for inspect, inspect-volume, export, export-surface, export-surface-usd or "
        "export-foam"
    );
  }
  const auto frame = ParseFrame(args[2]);
  if (!frame.ok()) {
    return frame.status();
  }
  if (inspect_volume) {
    return InspectVolume(args[1], *frame);
  }
  if (export_surface_usd && args[3].extension() != ".usda") {
    return absl::InvalidArgumentError("USD surface export requires a .usda destination");
  }
  if (export_surface || export_surface_usd || export_foam) {
    return ExportEffects(args[1], *frame, args[3], export_surface || export_surface_usd, export_surface_usd);
  }
  const auto cache = LiquidCache3D::OpenForInspection(args[1]);
  if (!cache.ok()) {
    return cache.status();
  }
  if (inspect) {
    return Inspect(*cache, *frame);
  }
  return PublishArchive(args[3], [&](std::ostream& output) {
    return WriteArchive(output, *cache, *frame);
  });
}
}  // namespace

// Contain the third-party Alembic exception boundary in the Blender tool.
int Entry(std::span<const fs::path> args) {
  std::cout.imbue(std::locale::classic());
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
  try {
    const auto result = Run(args);
    if (result.ok()) {
      return 0;
    }
    std::cerr << result << '\n';
  } catch (const std::exception& error) {
    std::cerr << "Alembic bridge: " << error.what() << '\n';
  }
  return 1;
}
}  // namespace fluidity::blender

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  const std::vector<std::filesystem::path> arguments(argv + 1, argv + argc);
  return fluidity::blender::Entry(arguments);
}
