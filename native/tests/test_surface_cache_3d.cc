#include "fluidity/host/surface_cache_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
SurfaceFrame3D Frame() {
  SurfaceFrame3D frame{.cell_width_m = 0.1};
  for (const auto p : {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}}) {
    frame.mesh.vertices.push_back(
        {.id = frame.mesh.vertices.size() + 1, .position = p, .normal = {1, 0, 0}, .component = 7}
    );
  }
  frame.mesh.next_id = 5;
  frame.mesh.triangles = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
  frame.volumes.push_back({.component = 7, .material_m3 = 1.0 / 6, .raw_m3 = 0.16, .corrected_m3 = 1.0 / 6});
  return frame;
}
TEST(SurfaceCache3D, RoundTripsAllPersistentFieldsAndRejectsCorruption) {
  auto frame = Frame();
  auto& vertex = frame.mesh.vertices[2];
  vertex.age_seconds = 0.5;
  vertex.geometric_velocity = {1, 2, 3};
  vertex.fluid_velocity = {4, 5, 6};
  vertex.wetness = 0.8;
  vertex.thickness_m = 0.01;
  vertex.contact = true;
  vertex.parent_id = 1;
  auto bytes = EncodeSurfaceFrame3D(frame);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  auto decoded = DecodeSurfaceFrame3D(bytes->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->mesh.next_id, 5);
  EXPECT_EQ(decoded->mesh.triangles, frame.mesh.triangles);
  EXPECT_EQ(decoded->mesh.vertices[2].fluid_velocity.y, 5);
  EXPECT_EQ(decoded->mesh.vertices[2].geometric_velocity.y, 2);
  EXPECT_EQ(decoded->mesh.vertices[2].parent_id, 1);
  EXPECT_EQ(decoded->mesh.vertices[2].wetness, 0.8);
  EXPECT_TRUE(decoded->mesh.vertices[2].contact);
  auto repeated = EncodeSurfaceFrame3D(*decoded);
  ASSERT_TRUE(repeated.ok());
  EXPECT_TRUE(std::ranges::equal(bytes->values(), repeated->values()));
  bytes->values()[4] ^= 1;
  EXPECT_EQ(DecodeSurfaceFrame3D(bytes->values()).status().code(), absl::StatusCode::kDataLoss);
}
TEST(SurfaceCache3D, RoundTripsOneSharedPhaseSheetAndRejectsBrokenIncidence) {
  auto frame = Frame();
  for (const auto position : {Vec3{0.5, 0, 0}, Vec3{0, 0.5, 0}, Vec3{0, 0, 0.5}}) {
    frame.mesh.vertices.push_back(
        {.id = frame.mesh.next_id++, .position = position, .normal = {1, 0, 0}, .component = 7}
    );
  }
  frame.mesh.triangles = {
      {0, 5, 4},
      {5, 2, 1},
      {5, 1, 4},
      {0, 4, 6},
      {4, 1, 3},
      {4, 3, 6},
      {0, 6, 5},
      {6, 3, 2},
      {6, 2, 5},
      {1, 2, 3},
  };
  frame.phase_interface = SurfacePhaseInterface3D{
      .phase_a = 4,
      .phase_b = 5,
      .exterior_sides = {0, 1, 1, 0, 1, 1, 0, 1, 1, 1},
      .interior_positions = {{0.125, 0.125, 0.25}},
      .triangles = {{4, 5, 7}, {5, 6, 7}, {6, 4, 7}},
  };
  for (const std::uint32_t vertex : {4, 5, 6, 7}) {
    frame.phase_interface->side_compositions.push_back({
        .vertex_ordinal = vertex,
        .phase_id = 4,
        .constituent_count = 1,
        .constituents = {{{8, 1}}},
    });
    frame.phase_interface->side_compositions.push_back({
        .vertex_ordinal = vertex,
        .phase_id = 5,
        .constituent_count = 1,
        .constituents = {{{25, 1}}},
    });
  }
  const auto bytes = EncodeSurfaceFrame3D(frame);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  const auto decoded = DecodeSurfaceFrame3D(bytes->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  ASSERT_TRUE(decoded->phase_interface.has_value());
  EXPECT_EQ(decoded->phase_interface->phase_a, 4);
  EXPECT_EQ(decoded->phase_interface->phase_b, 5);
  EXPECT_EQ(decoded->phase_interface->exterior_sides, frame.phase_interface->exterior_sides);
  EXPECT_EQ(decoded->phase_interface->triangles, frame.phase_interface->triangles);
  EXPECT_EQ(decoded->phase_interface->interior_positions.front().z, 0.25);
  const auto repeated = EncodeSurfaceFrame3D(*decoded);
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  EXPECT_TRUE(std::ranges::equal(bytes->values(), repeated->values()));
  EXPECT_EQ(
      DecodeSurfaceFrame3D(bytes->values(), {.maximum_vertices = 7}).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  ASSERT_EQ(decoded->phase_interface->side_compositions.size(), 8);
  auto invalid_composition = frame;
  invalid_composition.phase_interface->side_compositions.pop_back();
  EXPECT_EQ(EncodeSurfaceFrame3D(invalid_composition).status().code(), absl::StatusCode::kInvalidArgument);
  invalid_composition = frame;
  invalid_composition.phase_interface->side_compositions[1].constituents[0].substance_id = 8;
  EXPECT_EQ(EncodeSurfaceFrame3D(invalid_composition).status().code(), absl::StatusCode::kInvalidArgument);
  auto broken = frame;
  broken.phase_interface->exterior_sides[0] = 1;
  EXPECT_EQ(EncodeSurfaceFrame3D(broken).status().code(), absl::StatusCode::kInvalidArgument);
  broken = frame;
  std::swap(broken.phase_interface->triangles[0][1], broken.phase_interface->triangles[0][2]);
  EXPECT_EQ(EncodeSurfaceFrame3D(broken).status().code(), absl::StatusCode::kInvalidArgument);
  broken = frame;
  broken.phase_interface->triangles[0][2] = 8;
  EXPECT_EQ(EncodeSurfaceFrame3D(broken).status().code(), absl::StatusCode::kInvalidArgument);
}
TEST(SurfaceCache3D, RejectsFalseVolumeMixedComponentsIdentityAndBudgets) {
  auto frame = Frame();
  frame.volumes[0].corrected_m3 *= 2;
  EXPECT_EQ(EncodeSurfaceFrame3D(frame).status().code(), absl::StatusCode::kInvalidArgument);
  frame = Frame();
  frame.mesh.vertices[0].component = 8;
  EXPECT_FALSE(EncodeSurfaceFrame3D(frame).ok());
  frame = Frame();
  frame.mesh.vertices[0].id = 2;
  EXPECT_FALSE(EncodeSurfaceFrame3D(frame).ok());
  frame = Frame();
  EXPECT_EQ(EncodeSurfaceFrame3D(frame, {.maximum_vertices = 3}).status().code(), absl::StatusCode::kResourceExhausted);
}
}  // namespace
}  // namespace fluidity
