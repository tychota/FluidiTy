#include <array>
#include <limits>

#include "fluidity/host/liquid_mixture_3d.h"
#include "fluidity/host/liquid_state_3d.h"
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

TEST(LiquidMixture3D, DepositsCurrentVolumesConservativelyAcrossSignedCells) {
  constexpr UniformMacGeometry3D kGeometry{{0.0, 0.0, 0.0}, 1.0};
  constexpr std::array<CellIndex3D, 2> kCells{{{-1, 0, 0}, {0, 0, 0}}};
  std::array<LiquidParticleState3D, 3> particles{};
  particles[0].world_position = {0.0, 0.5, 0.5};
  particles[0].reference_volume = 0.5;
  particles[0].volume_ratio = 2.0;
  particles[0].phase_id = 1;
  particles[0].constituent_count = 2;
  particles[0].constituents[0] = {8, 0.25};
  particles[0].constituents[1] = {14, 0.75};
  particles[1] = particles[0];
  particles[1].world_position.x = 0.5;
  particles[1].reference_volume = 2.0;
  particles[1].volume_ratio = 0.5;
  particles[1].constituent_count = 1;
  particles[1].constituents[0] = {8, 1.0};
  particles[2] = particles[1];
  particles[2].phase_id = 2;
  particles[2].constituents[0] = {25, 1.0};

  const auto aqueous_materials = std::span(kMaterials).first(2);
  std::array<LiquidConstituentVolume3D, 4> aqueous;
  const auto deposited = DepositLiquidConstituentVolumes3D(kGeometry, 1, kCells, aqueous_materials, particles, aqueous);
  ASSERT_TRUE(deposited.ok()) << deposited;
  EXPECT_DOUBLE_EQ(aqueous[0].volume_m3, 0.125);
  EXPECT_DOUBLE_EQ(aqueous[1].volume_m3, 0.375);
  EXPECT_DOUBLE_EQ(aqueous[2].volume_m3, 1.125);
  EXPECT_DOUBLE_EQ(aqueous[3].volume_m3, 0.375);
  EXPECT_DOUBLE_EQ(aqueous[0].volume_m3 + aqueous[2].volume_m3, 1.25);
  EXPECT_DOUBLE_EQ(aqueous[1].volume_m3 + aqueous[3].volume_m3, 0.75);

  std::array<LiquidConstituentFraction3D, 2> fractions;
  const auto mixture = MixLiquidConstituentVolumes3D(1, aqueous_materials, std::span(aqueous).last(2), fractions);
  ASSERT_TRUE(mixture.ok()) << mixture.status();
  EXPECT_DOUBLE_EQ(fractions[0].volume_fraction, 0.75);
  EXPECT_DOUBLE_EQ(fractions[1].volume_fraction, 0.25);
  EXPECT_DOUBLE_EQ(mixture->density_kg_per_m3, 1005.0);

  std::array<LiquidConstituentVolume3D, 2> oil;
  const auto oil_materials = std::span(kMaterials).last(1);
  ASSERT_TRUE(DepositLiquidConstituentVolumes3D(kGeometry, 2, kCells, oil_materials, particles, oil).ok());
  EXPECT_EQ(oil[0].substance_id, 25);
  EXPECT_DOUBLE_EQ(oil[0].volume_m3, 0.0);
  EXPECT_DOUBLE_EQ(oil[1].volume_m3, 1.0);
  EXPECT_EQ(
      DepositLiquidConstituentVolumes3D(
          kGeometry,
          1,
          std::span(kCells).last(1),
          aqueous_materials,
          particles,
          std::span(aqueous).first(2)
      )
          .code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(LiquidMixture3D, UsesAllEightCellCentersInCanonicalLatticeOrder) {
  constexpr UniformMacGeometry3D kGeometry{{2.0, 4.0, 6.0}, 2.0};
  constexpr std::array<CellIndex3D, 8> kCells{{
      {-1, -1, -1},
      {0, -1, -1},
      {-1, 0, -1},
      {0, 0, -1},
      {-1, -1, 0},
      {0, -1, 0},
      {-1, 0, 0},
      {0, 0, 0},
  }};
  LiquidParticleState3D particle;
  particle.world_position = {2.5, 4.0, 5.5};
  particle.reference_volume = 8.0;
  particle.phase_id = 1;
  particle.constituent_count = 1;
  particle.constituents[0] = {8, 1.0};
  std::array<LiquidConstituentVolume3D, 8> volumes;
  const auto deposited = DepositLiquidConstituentVolumes3D(
      kGeometry,
      1,
      kCells,
      std::span(kMaterials).first(1),
      std::span(&particle, 1),
      volumes
  );
  ASSERT_TRUE(deposited.ok()) << deposited;
  constexpr std::array<double, 8> kExpected{0.75, 2.25, 0.75, 2.25, 0.25, 0.75, 0.25, 0.75};
  for (std::size_t index = 0; index < volumes.size(); ++index) {
    EXPECT_DOUBLE_EQ(volumes[index].volume_m3, kExpected[index]) << index;
  }
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
