#include <cmath>

#include "fluidity/host/surface_settings_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
TEST(SurfaceSettings3D, CanonicalPhysicalSettingsExcludeOperationalCapacityLimits) {
  SurfaceReconstructionOptions3D options;
  auto encoded = EncodeSurfaceSettings3D(options);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  auto decoded = DecodeSurfaceSettings3D(encoded->values());
  ASSERT_TRUE(decoded.ok());
  EXPECT_EQ(decoded->spacing_m, options.spacing_m);
  EXPECT_EQ(decoded->sigma_m, options.sigma_m);
  EXPECT_EQ(decoded->tracking.maximum_projection_cells, options.tracking.maximum_projection_cells);
  EXPECT_EQ(decoded->tracking.projection_tolerance_cells, options.tracking.projection_tolerance_cells);
  options.maximum_nodes *= 2;
  options.limits.maximum_vertices *= 2;
  options.maximum_correction_iterations *= 2;
  auto capacity_change = EncodeSurfaceSettings3D(options);
  ASSERT_TRUE(capacity_change.ok());
  EXPECT_TRUE(std::ranges::equal(encoded->values(), capacity_change->values()));
  options.sigma_m *= 2;
  auto physical_change = EncodeSurfaceSettings3D(options);
  ASSERT_TRUE(physical_change.ok());
  EXPECT_NE(LiquidContentDigest3D(encoded->values()), LiquidContentDigest3D(physical_change->values()));
}
TEST(SurfaceSettings3D, RejectsInvalidPhysicalControlsTruncationAndByteBudgets) {
  SurfaceReconstructionOptions3D options;
  options.spacing_m = std::nan("");
  EXPECT_EQ(EncodeSurfaceSettings3D(options).status().code(), absl::StatusCode::kInvalidArgument);
  options.spacing_m = 1;
  options.sigma_m = 1e-6;
  EXPECT_EQ(EncodeSurfaceSettings3D(options).status().code(), absl::StatusCode::kInvalidArgument);
  options = {};
  auto encoded = EncodeSurfaceSettings3D(options);
  ASSERT_TRUE(encoded.ok());
  EXPECT_EQ(DecodeSurfaceSettings3D(encoded->values().first(12)).status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(
      EncodeSurfaceSettings3D(options, {.maximum_bytes = 16}).status().code(),
      absl::StatusCode::kResourceExhausted
  );
}
}  // namespace
}  // namespace fluidity
