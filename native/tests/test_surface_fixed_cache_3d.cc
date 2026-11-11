#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
// psapi.h requires Windows types; preserve this dependency order.
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <sys/resource.h>
#endif

#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/surface_field_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
std::size_t PeakResidentBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    return 0;
  }
  return counters.PeakWorkingSetSize;
#else
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0;
  }
#if defined(__APPLE__)
  return static_cast<std::size_t>(usage.ru_maxrss);
#else
  return 1024 * static_cast<std::size_t>(usage.ru_maxrss);
#endif
#endif
}
std::pair<double, double> RadialError(const SurfaceMesh3D& mesh) {
  double squared_sum = 0;
  double maximum = 0;
  for (const auto& vertex : mesh.vertices) {
    const auto p = vertex.position;
    const double radius = std::hypot(p.x, p.y, p.z);
    const double level = std::hypot(p.x * std::sqrt(1.03) / 0.125, p.y / (0.125 * 1.03), p.z * std::sqrt(1.03) / 0.125);
    const double error = radius * (1 - 1 / level);
    squared_sum += error * error;
    maximum = std::max(maximum, std::abs(error));
  }
  return {std::sqrt(squared_sum / mesh.vertices.size()), maximum};
}
std::string OptionalNumber(double value, bool valid) {
  if (!valid) {
    return "null";
  }
  std::array<char, 64> text{};
  const auto result = std::to_chars(
      text.data(),
      text.data() + text.size(),
      value,
      std::chars_format::general,
      std::numeric_limits<double>::max_digits10
  );
  return {text.data(), result.ptr};
}
// Opt-in historical-data experiment. CI cannot supply the pinned external M1
// archive; skipping without it must never be reported as fixed-cache evidence.
TEST(SurfaceFixedCache3D, MeasuresPinnedParticleCacheWithoutChangingIt) {
  const char* root = std::getenv("FLUIDITY_FIXED_CACHE_ROOT");
  if (root == nullptr) {
    GTEST_SKIP() << "Set FLUIDITY_FIXED_CACHE_ROOT to the authenticated historical M1 drop";
  }
  std::uint64_t ordinal = 0;
  double spacing = 0.0125;
  if (const char* frame = std::getenv("FLUIDITY_FIXED_CACHE_FRAME")) {
    const std::string_view value(frame);
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), ordinal);
    ASSERT_EQ(parsed.ec, std::errc{});
    ASSERT_EQ(parsed.ptr, value.data() + value.size());
  }
  if (const char* input = std::getenv("FLUIDITY_FIXED_CACHE_SPACING")) {
    const std::string_view value(input);
    // This experiment has exactly two pinned resolutions. Avoid a floating
    // from_chars dependency unavailable in the qualified Apple libc++.
    ASSERT_TRUE(value == "0.0125" || value == "0.00625");
    spacing = value == "0.0125" ? 0.0125 : 0.00625;
  }
  ASSERT_TRUE(spacing == 0.0125 || spacing == 0.00625);
  ASSERT_TRUE(ordinal == 0 || ordinal == 48 || ordinal == 96 || ordinal == 144 || ordinal == 192 || ordinal == 216);
  auto cache = LiquidCache3D::OpenForInspection(std::filesystem::path(root));
  ASSERT_TRUE(cache.ok()) << cache.status();
  auto scene = cache->ReadResolvedScene();
  ASSERT_TRUE(scene.ok()) << scene.status();
  auto fingerprint = ResolvedSceneFingerprint(scene->View());
  ASSERT_TRUE(fingerprint.ok()) << fingerprint.status();
  ASSERT_STREQ(
      LiquidDigestHex3D(*fingerprint).data(),
      "f8929d0eac4fc91e745589087773706152691cf51f1bb0e5e114420a7d724afa"
  );
  auto source = cache->ReadCommittedFrame(ordinal);
  ASSERT_TRUE(source.ok()) << source.status();
  ASSERT_EQ(source->points.points.size(), 552);
  std::vector<SurfaceMaterialPoint3D> particles;
  Vec3 lower{-0.125 / std::sqrt(1.03), -0.125 * 1.03, -0.125 / std::sqrt(1.03)};
  Vec3 upper{-lower.x, -lower.y, -lower.z};
  double material_volume = 0;
  for (const auto& point : source->points.points.values()) {
    const double volume = 4 * std::numbers::pi / 3 * std::pow(point.radius, 3);
    particles.push_back({.position = point.world_position, .volume_m3 = volume});
    material_volume += volume;
    lower.x = std::min(lower.x, point.world_position.x);
    lower.y = std::min(lower.y, point.world_position.y);
    lower.z = std::min(lower.z, point.world_position.z);
    upper.x = std::max(upper.x, point.world_position.x);
    upper.y = std::max(upper.y, point.world_position.y);
    upper.z = std::max(upper.z, point.world_position.z);
  }
  const double padding = 0.1 + 2 * spacing;
  const std::array minimum{
      std::floor((lower.x - padding) / spacing),
      std::floor((lower.y - padding) / spacing),
      std::floor((lower.z - padding) / spacing),
  };
  const std::array maximum{
      std::ceil((upper.x + padding) / spacing),
      std::ceil((upper.y + padding) / spacing),
      std::ceil((upper.z + padding) / spacing),
  };
  const SurfaceFieldOptions3D options{
      .origin = {minimum[0] * spacing, minimum[1] * spacing, minimum[2] * spacing},
      .spacing_m = spacing,
      .extent =
          {static_cast<std::size_t>(maximum[0] - minimum[0] + 1),
           static_cast<std::size_t>(maximum[1] - minimum[1] + 1),
           static_cast<std::size_t>(maximum[2] - minimum[2] + 1)},
      .sigma_m = 0.025,
  };
  const auto start = std::chrono::steady_clock::now();
  auto field = BuildSurfaceDensity3D(particles, options, 2000000);
  ASSERT_TRUE(field.ok()) << field.status();
  auto mesh = ExtractSurfaceMesh3D(*field, 100000);
  ASSERT_TRUE(mesh.ok()) << mesh.status();
  auto raw = MeasureSurfaceMesh3D(*mesh);
  ASSERT_TRUE(raw.ok()) << raw.status();
  EXPECT_EQ(raw->components, 1);
  EXPECT_EQ(raw->euler_characteristic, 2);
  EXPECT_EQ(raw->boundary_edges, 0);
  EXPECT_EQ(raw->nonmanifold_edges, 0);
  EXPECT_EQ(raw->nonmanifold_vertices, 0);
  EXPECT_EQ(raw->orientation_defects, 0);
  EXPECT_EQ(raw->self_intersections, 0);
  const auto raw_error = ordinal == 0 ? RadialError(*mesh) : std::pair{0.0, 0.0};
  std::vector<Vec3> raw_positions;
  for (const auto& vertex : mesh->vertices) {
    raw_positions.push_back(vertex.position);
  }
  auto corrected = CorrectSurfaceVolumeWithNormals3D(*mesh, material_volume, scene->geometry.cell_width, 0.005, 32);
  ASSERT_TRUE(corrected.ok()) << corrected.status();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  EXPECT_LE(std::abs(corrected->signed_volume / material_volume - 1), 0.005);
  const auto corrected_error = ordinal == 0 ? RadialError(*mesh) : std::pair{0.0, 0.0};
  double maximum_offset_m = 0;
  for (std::size_t index = 0; index < mesh->vertices.size(); ++index) {
    const auto p = mesh->vertices[index].position;
    const auto raw_position = raw_positions[index];
    maximum_offset_m =
        std::max(maximum_offset_m, std::hypot(p.x - raw_position.x, p.y - raw_position.y, p.z - raw_position.z));
  }
  EXPECT_LE(maximum_offset_m, 0.25 * scene->geometry.cell_width);
  const auto peak_bytes = PeakResidentBytes();
  EXPECT_GT(peak_bytes, 0);
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "native_surface_measurement {\"frame\":" << ordinal << ",\"spacing_m\":" << spacing
            << ",\"source_points_digest\":\"" << LiquidDigestHex3D(source->manifest.points.digest).data() << "\""
            << ",\"material_volume_m3\":" << material_volume << ",\"raw_volume_m3\":" << raw->signed_volume
            << ",\"corrected_volume_m3\":" << corrected->signed_volume
            << ",\"raw_volume_error\":" << raw->signed_volume / material_volume - 1
            << ",\"corrected_volume_error\":" << corrected->signed_volume / material_volume - 1
            << ",\"maximum_offset_m\":" << maximum_offset_m
            << ",\"offset_limit_m\":" << 0.25 * scene->geometry.cell_width << ",\"vertices\":" << mesh->vertices.size()
            << ",\"triangles\":" << mesh->triangles.size() << ",\"components\":" << corrected->components
            << ",\"euler\":" << corrected->euler_characteristic
            << ",\"self_intersections\":" << corrected->self_intersections
            << ",\"raw_radial_rms_m\":" << OptionalNumber(raw_error.first, ordinal == 0)
            << ",\"raw_radial_max_m\":" << OptionalNumber(raw_error.second, ordinal == 0)
            << ",\"corrected_radial_rms_m\":" << OptionalNumber(corrected_error.first, ordinal == 0)
            << ",\"corrected_radial_max_m\":" << OptionalNumber(corrected_error.second, ordinal == 0)
            << ",\"peak_process_rss_bytes\":" << peak_bytes << ",\"runtime_seconds\":" << seconds << "}\n";
}
}  // namespace
}  // namespace fluidity
