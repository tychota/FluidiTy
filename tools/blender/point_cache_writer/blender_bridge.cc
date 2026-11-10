#include <Alembic/AbcCoreOgawa/All.h>

#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <span>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#include "blender_resolve.h"
#include "fluidity/host/liquid_cache_3d.h"
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

absl::Status Export(const LiquidCache3D& cache, std::uint64_t last_frame, const fs::path& destination) {
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
  const auto wrote = WriteArchive(output, cache, last_frame);
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

absl::Status Run(std::span<const fs::path> args) {
  if (!args.empty() && args[0] == "resolve") {
    return ResolveBlenderScene(args);
  }
  const bool inspect = args.size() == 3 && args[0] == "inspect";
  const bool export_points = args.size() == 4 && args[0] == "export";
  if (!inspect && !export_points) {
    return absl::InvalidArgumentError("expected inspect CACHE FRAME or export CACHE LAST_FRAME OUTPUT");
  }
  const auto frame = ParseFrame(args[2]);
  if (!frame.ok()) {
    return frame.status();
  }
  const auto cache = LiquidCache3D::OpenForInspection(args[1]);
  if (!cache.ok()) {
    return cache.status();
  }
  return inspect ? Inspect(*cache, *frame) : Export(*cache, *frame, args[3]);
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
