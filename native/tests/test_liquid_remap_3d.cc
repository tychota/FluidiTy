#include <algorithm>
#include <array>
#include <limits>

#include "fluidity/host/liquid_remap_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

std::array<double, 7> Lanes(const LiquidRemapInventory3D& inventory) {
  const auto momentum = inventory.momentum_kg_m_per_s;
  return {
      inventory.mass_kg,
      inventory.reference_volume_m3,
      inventory.current_volume_m3,
      momentum.x,
      momentum.y,
      momentum.z,
      inventory.enthalpy_j,
  };
}

// Affine parent-cell integrals in normalized parent-cell coordinates.
LiquidRemapInventory3D AffineInventory(double x, double y, double z) {
  return {
      .mass_kg = 64 + 8 * x - 4 * y + 2 * z,
      .reference_volume_m3 = 8 - 2 * x,
      .current_volume_m3 = 12 + 4 * z,
      .momentum_kg_m_per_s = {-32 + 4 * x, 0 - 8 * y, 16 + 4 * z},
      .enthalpy_j = -8 + x + 2 * y + 4 * z,
  };
}

TEST(LiquidRemap3D, ProlongsAffineInventoriesAndRestrictsExactly) {
  const auto parent = AffineInventory(0, 0, 0);
  const LiquidRemapStencil3D neighbors{{
      {AffineInventory(-1, 0, 0), AffineInventory(1, 0, 0)},
      {AffineInventory(0, -1, 0), AffineInventory(0, 1, 0)},
      {AffineInventory(0, 0, -1), AffineInventory(0, 0, 1)},
  }};
  const auto children = ProlongLiquidInventory3D(parent, neighbors);
  ASSERT_TRUE(children.ok()) << children.status();
  std::size_t index = 0;
  for (const double x : {-0.25, 0.25}) {
    for (const double y : {-0.25, 0.25}) {
      for (const double z : {-0.25, 0.25}) {
        auto expected = Lanes(AffineInventory(x, y, z));
        for (double& lane : expected) {
          lane /= 8;
        }
        EXPECT_EQ(Lanes((*children)[index++]), expected);
      }
    }
  }
  const auto restricted = RestrictLiquidInventory3D(*children);
  ASSERT_TRUE(restricted.ok()) << restricted.status();
  EXPECT_EQ(Lanes(*restricted), Lanes(parent));
}

TEST(LiquidRemap3D, PreservesConstituentsWithLimitedNonuniformChildVolume) {
  const auto mixture = [](double z) {
    auto inventory = AffineInventory(0, 0, z);
    inventory.phase_id = 4;
    inventory.constituent_count = 2;
    inventory.constituents[0] = {8, inventory.current_volume_m3 * 0.25};
    inventory.constituents[1] = {14, inventory.current_volume_m3 * 0.75};
    return inventory;
  };
  const auto parent = mixture(0);
  const LiquidRemapStencil3D neighbors{{{parent, parent}, {parent, parent}, {mixture(-1), mixture(1)}}};
  const auto children = ProlongLiquidInventory3D(parent, neighbors);
  ASSERT_TRUE(children.ok()) << children.status();
  for (std::size_t index = 0; index < children->size(); ++index) {
    const auto& child = (*children)[index];
    EXPECT_DOUBLE_EQ(child.current_volume_m3, index % 2 == 0 ? 1.375 : 1.625);
    EXPECT_DOUBLE_EQ(child.constituents[0].volume_m3, child.current_volume_m3 * 0.25);
    EXPECT_DOUBLE_EQ(child.constituents[1].volume_m3, child.current_volume_m3 * 0.75);
  }
  const auto restricted = RestrictLiquidInventory3D(*children);
  ASSERT_TRUE(restricted.ok()) << restricted.status();
  EXPECT_DOUBLE_EQ(restricted->constituents[0].volume_m3, 3);
  EXPECT_DOUBLE_EQ(restricted->constituents[1].volume_m3, 9);

  auto malformed = parent;
  malformed.constituents[0].volume_m3 = 1;
  EXPECT_EQ(ProlongLiquidInventory3D(malformed, neighbors).status().code(), absl::StatusCode::kInvalidArgument);
  auto excessive = *children;
  for (std::size_t index = 0; index < excessive.size(); ++index) {
    auto& child = excessive[index];
    child.constituents[0].substance_id = 100 + 2 * index;
    child.constituents[1].substance_id = 101 + 2 * index;
  }
  EXPECT_EQ(RestrictLiquidInventory3D(excessive).status().code(), absl::StatusCode::kResourceExhausted);
}
TEST(LiquidRemap3D, LimitsAtExtremaAndPreservesNonnegativeInventories) {
  const LiquidRemapInventory3D parent{.mass_kg = 8, .current_volume_m3 = 0.125};
  LiquidRemapStencil3D neighbors{};
  neighbors[0][1].mass_kg = 24;
  neighbors[1][0].mass_kg = 16;  // Local minimum along y.
  neighbors[1][1].mass_kg = 24;
  neighbors[2][0].current_volume_m3 = 0;
  neighbors[2][1].current_volume_m3 = 2;
  const auto children = ProlongLiquidInventory3D(parent, neighbors);
  ASSERT_TRUE(children.ok()) << children.status();
  for (std::size_t index = 0; index < children->size(); ++index) {
    const auto& child = (*children)[index];
    EXPECT_DOUBLE_EQ(child.mass_kg, index < 4 ? 0.75 : 1.25);
    EXPECT_DOUBLE_EQ(child.reference_volume_m3, 0);
    EXPECT_GT(child.current_volume_m3, 0);
    EXPECT_LE(child.current_volume_m3, 0.25);
  }
  const auto restricted = RestrictLiquidInventory3D(*children);
  ASSERT_TRUE(restricted.ok()) << restricted.status();
  EXPECT_EQ(Lanes(*restricted), Lanes(parent));
}

TEST(LiquidRemap3D, RefusesUnrepresentableSubnormalLossAndMalformedInventory) {
  const double quantum = std::numeric_limits<double>::denorm_min();
  LiquidRemapInventory3D parent{.mass_kg = quantum};
  LiquidRemapStencil3D neighbors{};
  for (auto& pair : neighbors) {
    pair.fill(parent);
  }
  EXPECT_EQ(ProlongLiquidInventory3D(parent, neighbors).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(parent.mass_kg, quantum);
  EXPECT_EQ(neighbors.back().back().mass_kg, quantum);

  parent.mass_kg = 8 * quantum;
  for (auto& pair : neighbors) {
    pair.fill(parent);
  }
  const auto children = ProlongLiquidInventory3D(parent, neighbors);
  ASSERT_TRUE(children.ok()) << children.status();
  EXPECT_EQ(children->back().mass_kg, quantum);
  const auto restricted = RestrictLiquidInventory3D(*children);
  ASSERT_TRUE(restricted.ok()) << restricted.status();
  EXPECT_EQ(restricted->mass_kg, parent.mass_kg);

  neighbors.back().back().reference_volume_m3 = -1;
  EXPECT_EQ(ProlongLiquidInventory3D(parent, neighbors).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(neighbors.back().back().reference_volume_m3, -1);
  LiquidRemapChildren3D invalid_children = *children;
  invalid_children.back().enthalpy_j = std::numeric_limits<double>::infinity();
  EXPECT_EQ(RestrictLiquidInventory3D(invalid_children).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidRemap3D, PreservesSmallSignedSumsAndRejectsOverflow) {
  LiquidRemapChildren3D children{};
  children[0].momentum_kg_m_per_s.x = 1e16;
  children[1].momentum_kg_m_per_s.x = -0.25;
  children[7].momentum_kg_m_per_s.x = -1e16;
  const auto restricted = RestrictLiquidInventory3D(children);
  ASSERT_TRUE(restricted.ok()) << restricted.status();
  EXPECT_EQ(restricted->momentum_kg_m_per_s.x, -0.25);

  const double largest = std::numeric_limits<double>::max();
  children[0].mass_kg = largest;
  children[7].mass_kg = largest;
  EXPECT_EQ(RestrictLiquidInventory3D(children).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(children[0].mass_kg, largest);
  EXPECT_EQ(children[7].mass_kg, largest);
}

TEST(LiquidRemap3D, RestrictsSignedFluxInUniqueSlotOrderAndProlongsInFineOrientation) {
  std::array<LiquidRemapSubface3D, 4> faces{{
      {.slot = {17, -1}, .flux_m3_per_s = 1e16},
      {.slot = {2, 1}, .flux_m3_per_s = 1e16},
      {.slot = {7, -1}, .flux_m3_per_s = 0.75},
      {.slot = {4, 1}, .flux_m3_per_s = 1.25},
  }};
  const auto coarse = RestrictLiquidFaceFlux3D(faces);
  ASSERT_TRUE(coarse.ok()) << coarse.status();
  EXPECT_EQ(*coarse, 0.5);
  std::reverse(faces.begin(), faces.end());
  const auto reordered = RestrictLiquidFaceFlux3D(faces);
  ASSERT_TRUE(reordered.ok()) << reordered.status();
  EXPECT_EQ(*reordered, *coarse);

  const std::array<LiquidRemapFaceSlot3D, 4> slots{{{7, -1}, {0, 1}, {2, -1}, {4, 1}}};
  const auto prolonged = ProlongLiquidFaceFlux3D(-10, slots);
  ASSERT_TRUE(prolonged.ok()) << prolonged.status();
  EXPECT_EQ((*prolonged)[0].slot.owner_slot, 7);
  EXPECT_EQ((*prolonged)[0].flux_m3_per_s, 2.5);
  EXPECT_EQ((*prolonged)[1].flux_m3_per_s, -2.5);
  const auto restricted = RestrictLiquidFaceFlux3D(*prolonged);
  ASSERT_TRUE(restricted.ok()) << restricted.status();
  EXPECT_EQ(*restricted, -10);

  EXPECT_EQ(
      ProlongLiquidFaceFlux3D(std::numeric_limits<double>::denorm_min(), slots).status().code(),
      absl::StatusCode::kOutOfRange
  );
  faces[1].slot.owner_slot = faces[0].slot.owner_slot;
  EXPECT_EQ(RestrictLiquidFaceFlux3D(faces).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(faces[1].slot.owner_slot, faces[0].slot.owner_slot);
  auto invalid_slots = slots;
  invalid_slots[0].normal_sign = 0;
  EXPECT_EQ(ProlongLiquidFaceFlux3D(-10, invalid_slots).status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace fluidity
