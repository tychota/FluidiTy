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
