#include <cmath>

#include "fluidity/host/surface_tracking_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
SurfaceMesh3D Tetrahedron() {
  SurfaceMesh3D mesh;
  for (const auto p : {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}}) {
    mesh.vertices.push_back({.id = mesh.vertices.size() + 1, .position = p, .component = 7});
  }
  mesh.next_id = 5;
  mesh.triangles = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
  return mesh;
}
TEST(SurfaceTracking3D, UniformTranslationPreservesIdsConnectivityAndFluidVelocity) {
  auto mesh = Tetrahedron();
  const LiquidClock3D first{};
  auto last = first;
  last.frame_ordinal = 1;
  last.time = 1.0 / 24;
  last.epoch = 1;
  const std::array stamps{MotionStamp3D{0, 0}, MotionStamp3D{last.time, 1}};
  const auto interval = MotionInterval3D::Create(first, last, stamps);
  ASSERT_TRUE(interval.ok()) << interval.status();
  const auto carrier = [](std::size_t, Vec3) -> absl::StatusOr<Vec3> {
    return Vec3{0.6, 0.3, -0.2};
  };
  const auto surface = [](Vec3, std::uint64_t component) -> absl::StatusOr<SurfaceProjectionSample3D> {
    return SurfaceProjectionSample3D{
        .value = 0,
        .gradient = {1, 0, 0},
        .component = component,
        .obstacle_distance_m = 1
    };
  };
  auto result = TrackSurfaceMesh3D(mesh, *interval, 0.1, {}, carrier, surface);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->mesh.triangles, mesh.triangles);
  EXPECT_EQ(result->mesh.next_id, mesh.next_id);
  EXPECT_EQ(result->retained_ids, mesh.vertices.size());
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto& vertex = result->mesh.vertices[i];
    EXPECT_EQ(vertex.id, mesh.vertices[i].id);
    EXPECT_NEAR(vertex.position.x - mesh.vertices[i].position.x, 0.025, 1e-14);
    EXPECT_NEAR(vertex.geometric_velocity.x, 0.6, 1e-14);
    EXPECT_EQ(vertex.fluid_velocity.y, 0.3);
    EXPECT_EQ(vertex.age_seconds, last.time);
  }
}
TEST(SurfaceTracking3D, DifferentComponentMissingHaloAndProjectionBudgetFailWithoutOutput) {
  const auto mesh = Tetrahedron();
  const LiquidClock3D first{};
  auto last = first;
  last.frame_ordinal = 1;
  last.time = 1.0 / 24;
  last.epoch = 1;
  const std::array stamps{MotionStamp3D{0, 0}, MotionStamp3D{last.time, 1}};
  const auto interval = MotionInterval3D::Create(first, last, stamps);
  ASSERT_TRUE(interval.ok());
  const auto carrier = [](std::size_t, Vec3) -> absl::StatusOr<Vec3> {
    return Vec3{};
  };
  auto query = SurfaceProjectionSample3D{.value = 0, .gradient = {1, 0, 0}, .component = 8, .obstacle_distance_m = 1};
  const auto surface = [&](Vec3, std::uint64_t) -> absl::StatusOr<SurfaceProjectionSample3D> {
    return query;
  };
  EXPECT_EQ(
      TrackSurfaceMesh3D(mesh, *interval, 0.1, {}, carrier, surface).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  query.component = 7;
  query.value = 1;
  EXPECT_EQ(
      TrackSurfaceMesh3D(mesh, *interval, 0.1, {}, carrier, surface).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  const auto absent = [](std::size_t, Vec3) -> absl::StatusOr<Vec3> {
    return absl::FailedPreconditionError("missing halo");
  };
  EXPECT_EQ(
      TrackSurfaceMesh3D(mesh, *interval, 0.1, {}, absent, surface).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(mesh.vertices.front().position.x, 0);
}
}  // namespace
}  // namespace fluidity
