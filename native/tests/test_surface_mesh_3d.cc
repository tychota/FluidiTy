#include <cmath>
#include <numbers>

#include "fluidity/host/surface_mesh_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
TEST(SurfaceMesh3D, ExtractsClosedSphereAndCorrectsActualVertexVolume) {
  SurfaceLattice3D lattice{.origin = {-1.2, -1.2, -1.2}, .spacing = 0.1, .extent = {25, 25, 25}};
  lattice.values.resize(25 * 25 * 25);
  for (std::size_t x = 0; x < 25; ++x) {
    for (std::size_t y = 0; y < 25; ++y) {
      for (std::size_t z = 0; z < 25; ++z) {
        const Vec3 position{-1.2 + 0.1 * x, -1.2 + 0.1 * y, -1.2 + 0.1 * z};
        lattice.values[(x * 25 + y) * 25 + z] = std::hypot(position.x, position.y, position.z) - 1;
      }
    }
  }
  auto mesh = ExtractSurfaceMesh3D(lattice, 100000);
  ASSERT_TRUE(mesh.ok()) << mesh.status();
  auto metrics = MeasureSurfaceMesh3D(*mesh);
  ASSERT_TRUE(metrics.ok()) << metrics.status();
  EXPECT_EQ(metrics->components, 1);
  EXPECT_EQ(metrics->euler_characteristic, 2);
  EXPECT_EQ(metrics->boundary_edges, 0);
  EXPECT_EQ(metrics->nonmanifold_edges, 0);
  EXPECT_EQ(metrics->orientation_defects, 0);
  EXPECT_EQ(metrics->self_intersections, 0);
  const double target = 4 * std::numbers::pi / 3;
  EXPECT_LT(std::abs(metrics->signed_volume / target - 1), 0.02);
  const auto original = mesh->vertices;
  const auto corrected = CorrectSurfaceVolume3D(*mesh, target, 0.1, 0.005, 24);
  ASSERT_TRUE(corrected.ok()) << corrected.status();
  EXPECT_LT(std::abs(corrected->signed_volume / target - 1), 0.005);
  EXPECT_NE(mesh->vertices.front().position.x, original.front().position.x);
}

TEST(SurfaceMesh3D, RejectsOpenOrNonfiniteLatticeBeforeAccess) {
  SurfaceLattice3D lattice{.origin = {}, .spacing = 1, .extent = {2, 2, 2}, .values = std::vector<double>(8, 1)};
  lattice.values[0] = -1;
  EXPECT_EQ(ExtractSurfaceMesh3D(lattice, 100).status().code(), absl::StatusCode::kFailedPrecondition);
  lattice.values[0] = std::nan("");
  EXPECT_EQ(ExtractSurfaceMesh3D(lattice, 100).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SurfaceMesh3D, DetectsCrossedDisconnectedComponents) {
  SurfaceMesh3D mesh;
  for (const Vec3 position :
       {Vec3{0, 0, 0},
        Vec3{1, 0, 0},
        Vec3{0, 1, 0},
        Vec3{0, 0, 1},
        Vec3{0.2, 0.2, -0.2},
        Vec3{1.2, 0.2, -0.2},
        Vec3{0.2, 1.2, -0.2},
        Vec3{0.2, 0.2, 0.8}}) {
    mesh.vertices.push_back({.id = mesh.vertices.size() + 1, .position = position});
  }
  for (const std::uint32_t base : {0u, 4u}) {
    for (const std::array<std::uint32_t, 3> triangle :
         {std::array{0u, 2u, 1u}, std::array{0u, 1u, 3u}, std::array{0u, 3u, 2u}, std::array{1u, 2u, 3u}}) {
      mesh.triangles.push_back({base + triangle[0], base + triangle[1], base + triangle[2]});
    }
  }
  auto metrics = MeasureSurfaceMesh3D(mesh);
  ASSERT_TRUE(metrics.ok()) << metrics.status();
  EXPECT_EQ(metrics->components, 2);
  EXPECT_GT(metrics->self_intersections, 0);
}

TEST(SurfaceMesh3D, InfeasiblePinnedVolumeLeavesVerticesUnchanged) {
  SurfaceMesh3D mesh;
  for (const auto position : {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}}) {
    mesh.vertices.push_back({.id = mesh.vertices.size() + 1, .position = position, .contact = true});
  }
  mesh.triangles = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
  const auto original = mesh.vertices;
  EXPECT_EQ(
      CorrectSurfaceVolume3D(mesh, 1.0 / 3, 0.1, 0.005, 8).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  for (std::size_t i = 0; i < original.size(); ++i) {
    EXPECT_EQ(mesh.vertices[i].position.x, original[i].position.x);
    EXPECT_EQ(mesh.vertices[i].position.y, original[i].position.y);
    EXPECT_EQ(mesh.vertices[i].position.z, original[i].position.z);
  }
}

TEST(SurfaceMesh3D, MeasuresCoplanarDisjointTouchingAndCrossingTriangles) {
  SurfaceMesh3D mesh;
  for (const auto position :
       {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{1, 1, 0}, Vec3{2, 1, 0}, Vec3{1, 2, 0}}) {
    mesh.vertices.push_back({.position = position});
  }
  mesh.triangles = {{0, 1, 2}, {3, 4, 5}};
  ASSERT_TRUE(MeasureSurfaceMesh3D(mesh).ok());
  EXPECT_EQ(MeasureSurfaceMesh3D(mesh)->self_intersections, 0);
  mesh.vertices[3].position = {0.5, 0.5, 0};
  EXPECT_EQ(MeasureSurfaceMesh3D(mesh)->self_intersections, 1);
  mesh.vertices[3].position = {0.25, 0.25, 0};
  EXPECT_EQ(MeasureSurfaceMesh3D(mesh)->self_intersections, 1);
}

TEST(SurfaceMesh3D, DetectsOverlapBeyondASharedVertexOrEdge) {
  SurfaceMesh3D mesh;
  for (const auto p :
       {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0.2, 0.2, -1}, Vec3{0.2, 0.2, 1}, Vec3{0.3, 0.3, 0}}) {
    mesh.vertices.push_back({.position = p});
  }
  mesh.triangles = {{0, 1, 2}, {0, 3, 4}};
  EXPECT_EQ(MeasureSurfaceMesh3D(mesh)->self_intersections, 1);
  mesh.triangles = {{0, 1, 2}, {0, 1, 5}};
  EXPECT_EQ(MeasureSurfaceMesh3D(mesh)->self_intersections, 1);
  mesh.vertices[5].position.y = -0.3;
  EXPECT_EQ(MeasureSurfaceMesh3D(mesh)->self_intersections, 0);
}

TEST(SurfaceMesh3D, VolumeOffsetCapBoundsTotalDisplacementAcrossIterations) {
  SurfaceMesh3D mesh;
  for (const auto p : {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}}) {
    mesh.vertices.push_back({.id = mesh.vertices.size() + 1, .position = p});
  }
  mesh.triangles = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
  const auto original = mesh.vertices;
  EXPECT_EQ(
      CorrectSurfaceVolume3D(mesh, 1.01 / 6, 0.0001, 1e-8, 128).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(mesh.vertices[0].position.x, original[0].position.x);
  EXPECT_EQ(mesh.vertices[1].position.x, original[1].position.x);
}

TEST(SurfaceMesh3D, StoredNormalCorrectionValidatesIndicesBeforeNormalAccess) {
  SurfaceMesh3D mesh;
  mesh.vertices.push_back({.id = 1, .position = {}, .normal = {1, 0, 0}});
  mesh.triangles.push_back({0, 0, UINT32_MAX});
  EXPECT_EQ(
      CorrectSurfaceVolumeWithNormals3D(mesh, 1, 0.1, 0.005, 4).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}
}  // namespace
}  // namespace fluidity
