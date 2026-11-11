#include <cmath>

#include "fluidity/host/surface_closest_point_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
SurfaceMesh3D Tetrahedron() {
  SurfaceMesh3D mesh;
  for (const auto position : {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}}) {
    mesh.vertices.push_back({.id = mesh.vertices.size() + 1, .position = position, .component = 7});
  }
  mesh.triangles = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
  return mesh;
}
TEST(SurfaceClosestPoint3D, FindsFaceEdgeVertexAndExactCoincidentQueries) {
  const auto mesh = Tetrahedron();
  auto field = SurfaceClosestPoint3D::Create(mesh, 100);
  ASSERT_TRUE(field.ok()) << field.status();
  auto sample = field->Sample({0.2, 0.2, -0.1}, 7);
  ASSERT_TRUE(sample.ok());
  EXPECT_NEAR(sample->distance_m, 0.1, 1e-15);
  EXPECT_NEAR(sample->position.x, 0.2, 1e-15);
  EXPECT_EQ(sample->position.z, 0);
  sample = field->Sample({0.5, -0.1, -0.1}, 7);
  ASSERT_TRUE(sample.ok());
  EXPECT_NEAR(sample->distance_m, std::sqrt(0.02), 1e-15);
  EXPECT_EQ(sample->position.y, 0);
  EXPECT_EQ(sample->position.z, 0);
  sample = field->Sample({-0.1, -0.1, -0.1}, 7);
  ASSERT_TRUE(sample.ok());
  EXPECT_EQ(sample->position.x, 0);
  for (const auto& vertex : mesh.vertices) {
    sample = field->Sample(vertex.position, 7);
    ASSERT_TRUE(sample.ok());
    EXPECT_EQ(sample->distance_m, 0);
    EXPECT_EQ(sample->position.x, vertex.position.x);
    EXPECT_EQ(sample->position.y, vertex.position.y);
    EXPECT_EQ(sample->position.z, vertex.position.z);
  }
  EXPECT_EQ(field->Sample({}, 8).status().code(), absl::StatusCode::kFailedPrecondition);
}
TEST(SurfaceClosestPoint3D, RejectsUnclosedTopologyInvalidQueriesAndBudgets) {
  auto mesh = Tetrahedron();
  EXPECT_EQ(SurfaceClosestPoint3D::Create(mesh, 3).status().code(), absl::StatusCode::kResourceExhausted);
  auto field = SurfaceClosestPoint3D::Create(mesh, 10);
  ASSERT_TRUE(field.ok());
  EXPECT_EQ(field->Sample({std::nan(""), 0, 0}, 7).status().code(), absl::StatusCode::kInvalidArgument);
  mesh.triangles.pop_back();
  EXPECT_EQ(SurfaceClosestPoint3D::Create(mesh, 10).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(SurfaceClosestPoint3D, MedianTreeFindsEverySphereVertexAndAnAnalyticPole) {
  SurfaceLattice3D lattice{.origin = {-1.5, -1.5, -1.5}, .spacing = 0.25, .extent = {13, 13, 13}};
  for (int x = 0; x < 13; ++x) {
    for (int y = 0; y < 13; ++y) {
      for (int z = 0; z < 13; ++z) {
        lattice.values.push_back(std::hypot(-1.5 + 0.25 * x, -1.5 + 0.25 * y, -1.5 + 0.25 * z) - 1);
      }
    }
  }
  auto mesh = ExtractSurfaceMesh3D(lattice, 5000);
  ASSERT_TRUE(mesh.ok());
  ASSERT_GT(mesh->triangles.size(), 8);
  for (auto& vertex : mesh->vertices) {
    vertex.component = 7;
  }
  const auto field = SurfaceClosestPoint3D::Create(*mesh, 10000);
  ASSERT_TRUE(field.ok()) << field.status();
  for (const auto& vertex : mesh->vertices) {
    const auto sample = field->Sample(vertex.position, 7);
    ASSERT_TRUE(sample.ok());
    EXPECT_EQ(sample->distance_m, 0);
  }
  const auto sample = field->Sample({0, 0, 1.5}, 7);
  ASSERT_TRUE(sample.ok());
  EXPECT_NEAR(sample->distance_m, 0.5, 1e-14);
  EXPECT_NEAR(sample->position.z, 1, 1e-14);
}

TEST(SurfaceClosestPoint3D, SkinnyFaceProjectionRetainsTheInteriorFoot) {
  auto mesh = Tetrahedron();
  mesh.vertices[3].position.z = 1e-8;
  const auto field = SurfaceClosestPoint3D::Create(mesh, 10);
  ASSERT_TRUE(field.ok()) << field.status();
  const auto sample = field->Sample({0.25, -1e-9, 3e-9}, 7);
  ASSERT_TRUE(sample.ok());
  EXPECT_NEAR(sample->distance_m, 1e-9, 1e-22);
  EXPECT_NEAR(sample->position.x, 0.25, 1e-15);
  EXPECT_NEAR(sample->position.y, 0, 1e-22);
  EXPECT_NEAR(sample->position.z, 3e-9, 1e-22);
}
}  // namespace
}  // namespace fluidity
