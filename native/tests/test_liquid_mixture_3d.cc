#include <array>
#include <limits>

#include "fluidity/host/liquid_mixture_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
constexpr std::array<LiquidConstituentMaterial3D, 3> kMaterials{{
    {8, 1, 1000.0},
    {14, 1, 1020.0},
    {25, 2, 820.0},
}};

TEST(LiquidMixture3D, MatchesUnequalDensityVolumeOracleAndKeepsOilSeparate) {
  constexpr std::array<LiquidConstituentVolume3D, 3> kAqueous{{
      {8, 1.0},
      {14, 0.5},
      {14, 2.5},
  }};
  std::array<LiquidConstituentFraction3D, 2> fractions;
  const auto aqueous = MixLiquidConstituentVolumes3D(1, kMaterials, kAqueous, fractions);
  ASSERT_TRUE(aqueous.ok()) << aqueous.status();
  EXPECT_EQ(aqueous->constituent_count, 2);
  EXPECT_DOUBLE_EQ(aqueous->volume_m3, 4.0);
  EXPECT_DOUBLE_EQ(aqueous->mass_kg, 4060.0);
  EXPECT_DOUBLE_EQ(aqueous->density_kg_per_m3, 1015.0);
  EXPECT_EQ(fractions[0].substance_id, 8);
  EXPECT_DOUBLE_EQ(fractions[0].volume_fraction, 0.25);
  EXPECT_EQ(fractions[1].substance_id, 14);
  EXPECT_DOUBLE_EQ(fractions[1].volume_fraction, 0.75);

  constexpr std::array<LiquidConstituentVolume3D, 1> kOil{{{25, 2.0}}};
  const auto oil = MixLiquidConstituentVolumes3D(2, kMaterials, kOil, fractions);
  ASSERT_TRUE(oil.ok()) << oil.status();
  EXPECT_EQ(oil->constituent_count, 1);
  EXPECT_EQ(fractions[0].substance_id, 25);
  EXPECT_DOUBLE_EQ(fractions[0].volume_fraction, 1.0);
  EXPECT_DOUBLE_EQ(oil->density_kg_per_m3, 820.0);
  EXPECT_EQ(
      MixLiquidConstituentVolumes3D(1, kMaterials, kOil, fractions).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidMixture3D, InvalidCapacityAndArithmeticDoNotPublishFractions) {
  std::array<LiquidConstituentFraction3D, 1> fractions{{{91, 0.5}}};
  constexpr std::array<LiquidConstituentVolume3D, 2> kTwo{{{8, 1.0}, {14, 1.0}}};
  EXPECT_EQ(
      MixLiquidConstituentVolumes3D(1, kMaterials, kTwo, fractions).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  constexpr std::array<LiquidConstituentVolume3D, 1> kOverflow{{
      {8, std::numeric_limits<double>::max()},
  }};
  EXPECT_EQ(
      MixLiquidConstituentVolumes3D(1, kMaterials, kOverflow, fractions).status().code(),
      absl::StatusCode::kOutOfRange
  );
  constexpr std::array<LiquidConstituentVolume3D, 2> kUnordered{{{14, 1.0}, {8, 1.0}}};
  EXPECT_EQ(
      MixLiquidConstituentVolumes3D(1, kMaterials, kUnordered, fractions).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(fractions[0].substance_id, 91);
  EXPECT_DOUBLE_EQ(fractions[0].volume_fraction, 0.5);
}
}  // namespace
}  // namespace fluidity
