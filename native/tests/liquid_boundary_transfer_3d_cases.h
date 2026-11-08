#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

#include "../src/liquid_boundary_transfer_3d_internal.h"
#include "fluidity/host/liquid_boundary_transfer_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace boundary_transfer_test {
constexpr std::array kAxes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
constexpr std::array kBlocks{BlockIndex3D{0, 0, 0}};
constexpr MacDepositQuantization3D kQuanta{0x1p-30, 0x1p-30, 1};

struct BoundaryFixture {
  SparseMacGrid3D grid = *SparseMacGrid3D::Create({{}, 1.0}, kBlocks, 1);
  LiquidBoundaryParticle3D particle{{{1.5, 1.5, 1.5}, {1.5, 1.5, 1.5}, 16.0, {3.0, 4.0, 5.0}, {}}, 7, true};
  std::vector<LiquidBoundaryFaceTag3D> tags = std::vector<LiquidBoundaryFaceTag3D>(grid.face_count(), {7, true});
  std::vector<double> old_values = std::vector<double>(grid.face_count(), 2.0);
  std::vector<double> new_values = std::vector<double>(grid.face_count(), 3.0);
  std::vector<LiquidFaceValidity3D> validity =
      std::vector<LiquidFaceValidity3D>(grid.face_count(), LiquidFaceValidity3D::kValid);
  std::vector<LiquidPrescribedWallSample3D> old_walls = std::vector<LiquidPrescribedWallSample3D>(grid.face_count());
  std::vector<LiquidPrescribedWallSample3D> new_walls = old_walls;

  std::size_t Slot(FaceNormalAxis axis, int x, int y, int z) const {
    return *grid.FaceSlot({axis, {x, y, z}});
  }
  LiquidBoundaryDepositInputs3D DepositInputs() const {
    return {grid, {&particle, 1}, tags, kQuanta};
  }
  LiquidBoundaryGatherInputs3D GatherInputs() const {
    return {
        grid,
        {&particle, 1},
        0.0,
        1.0,
        {0.0, 0.0, old_values, validity, tags, old_walls},
        {1.0, 1.0, new_values, validity, tags, new_walls},
    };
  }
};

void ExpectVec(Vec3 value, Vec3 expected, double tolerance = 2e-12) {
  EXPECT_NEAR(value.x, expected.x, tolerance);
  EXPECT_NEAR(value.y, expected.y, tolerance);
  EXPECT_NEAR(value.z, expected.z, tolerance);
}

class TransferCases {
 public:
  std::function<absl::StatusOr<LiquidBoundaryFaceDeposits3D>(const LiquidBoundaryDepositInputs3D&)> deposit;
  std::function<absl::StatusOr<
      NormalizedLiquidMacFaces3D>(std::span<const MacFaceDepositQuanta3D>, MacDepositQuantization3D, double)>
      normalize;
  std::function<absl::StatusOr<LiquidBoundaryGatherResults3D>(const LiquidBoundaryGatherInputs3D&)> gather;
  std::function<absl::StatusOr<internal::LiquidGhostAffineFit3D>(std::span<const internal::LiquidGhostFitSample3D>)>
      fit_affine;
  void MaskedDepositRecentersMomentumAndQuotientGradient() {
    BoundaryFixture fixture;
    fixture.particle.transfer.particle_velocity_slopes.x_row = {100.0, 2.0, 0.0};
    for (int z = 0; z < 4; ++z) {
      for (int y = 0; y < 4; ++y) {
        fixture.tags[fixture.Slot(FaceNormalAxis::kX, 1, y, z)].side_tag = -7;
      }
    }
    auto deposited = deposit(fixture.DepositInputs());
    ASSERT_TRUE(deposited.ok()) << deposited.status();
    const auto center_slot = fixture.Slot(FaceNormalAxis::kX, 2, 1, 1);
    EXPECT_EQ(deposited->face_quanta[center_slot].mass_quanta_sum, std::int64_t{9} << 30);
    EXPECT_EQ(deposited->face_quanta[center_slot].pure_momentum_quanta_sum, std::int64_t{27} << 30);
    double mass = 0.0;
    double momentum = 0.0;
    Vec3 gradient_sum{};
    Vec3 centered_sum{};
    for (const auto& sample : deposited->compatible_stencils[0][0]) {
      const auto& quanta = deposited->face_quanta[sample.face_slot];
      mass += static_cast<double>(quanta.mass_quanta_sum) * kQuanta.mass_per_quantum;
      momentum += static_cast<double>(quanta.pure_momentum_quanta_sum) * kQuanta.pure_momentum_per_quantum;
      EXPECT_EQ(quanta.stress_force_quanta_sum, 0);
      gradient_sum.x += sample.normalized_weight_gradient.x;
      gradient_sum.y += sample.normalized_weight_gradient.y;
      gradient_sum.z += sample.normalized_weight_gradient.z;
      centered_sum.x += sample.normalized_weight * sample.centered_face_offset.x;
      centered_sum.y += sample.normalized_weight * sample.centered_face_offset.y;
      centered_sum.z += sample.normalized_weight * sample.centered_face_offset.z;
      if (sample.face_slot == fixture.Slot(FaceNormalAxis::kX, 2, 0, 1)) {
        EXPECT_DOUBLE_EQ(sample.normalized_weight, 0.09375);
        ExpectVec(sample.normalized_weight_gradient, {0.0, -0.375, 0.0});
        ExpectVec(sample.centered_face_offset, {0.0, -1.0, 0.0});
      }
    }
    EXPECT_DOUBLE_EQ(mass, 16.0);
    EXPECT_DOUBLE_EQ(momentum, 48.0);
    ExpectVec(gradient_sum, {});
    ExpectVec(centered_sum, {});
    // No prior-mass input exists: a fresh zero-mass field remains eligible.
    auto normalized = normalize(deposited->face_quanta, kQuanta, 0.0);
    ASSERT_TRUE(normalized.ok()) << normalized.status();
    EXPECT_EQ(normalized->validity[center_slot], LiquidFaceValidity3D::kValid);
    EXPECT_DOUBLE_EQ(normalized->face_velocities[center_slot], 3.0);
  }
  void RejectsEmptyZeroSupportAndMissingExcludedOwner() {
    BoundaryFixture fixture;
    for (auto& tag : fixture.tags) {
      tag.structurally_eligible = false;
    }
    EXPECT_EQ(deposit(fixture.DepositInputs()).status().code(), absl::StatusCode::kFailedPrecondition);
    fixture.tags[fixture.Slot(FaceNormalAxis::kX, 3, 1, 1)].structurally_eligible = true;
    EXPECT_EQ(deposit(fixture.DepositInputs()).status().code(), absl::StatusCode::kFailedPrecondition);
    // Keep positive allocated support: only the absent owner may reject this case.
    std::fill(fixture.tags.begin(), fixture.tags.end(), LiquidBoundaryFaceTag3D{7, true});
    fixture.particle.transfer.sorted_world_position = {3.75, 3.75, 3.75};
    fixture.particle.transfer.trial_world_position = {3.75, 3.75, 3.75};
    EXPECT_EQ(deposit(fixture.DepositInputs()).status().code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_EQ(gather(fixture.GatherInputs()).status().code(), absl::StatusCode::kFailedPrecondition);
    auto empty_gather = fixture.GatherInputs();
    empty_gather.particles = {};
    const auto empty_gathered = gather(empty_gather);
    ASSERT_TRUE(empty_gathered.ok()) << empty_gathered.status();
    EXPECT_TRUE(empty_gathered->particles.empty());
    auto empty = fixture.DepositInputs();
    empty.particles = {};
    auto deposited = deposit(empty);
    ASSERT_TRUE(deposited.ok()) << deposited.status();
    for (const auto& face : deposited->face_quanta) {
      EXPECT_EQ(face.mass_quanta_sum, 0);
    }
  }
  void RoundsActualMaskedContributionsAndHonorsAffineOff() {
    BoundaryFixture fixture;
    for (auto& tag : fixture.tags) {
      tag.structurally_eligible = false;
    }
    for (auto axis : kAxes) {
      fixture.tags[fixture.Slot(axis, 1, 1, 1)].structurally_eligible = true;
    }
    fixture.particle.transfer.particle_mass = 0.5;
    fixture.particle.transfer.particle_velocity = {-3.0, 3.0, 5.0};
    auto inputs = fixture.DepositInputs();
    inputs.quantization = {1.0, 1.0, 1};
    auto deposited = deposit(inputs);
    ASSERT_TRUE(deposited.ok()) << deposited.status();
    EXPECT_EQ(deposited->face_quanta[fixture.Slot(FaceNormalAxis::kX, 1, 1, 1)].mass_quanta_sum, 0);
    EXPECT_EQ(deposited->face_quanta[fixture.Slot(FaceNormalAxis::kX, 1, 1, 1)].pure_momentum_quanta_sum, -2);
    EXPECT_EQ(deposited->face_quanta[fixture.Slot(FaceNormalAxis::kZ, 1, 1, 1)].pure_momentum_quanta_sum, 2);
    fixture.tags[fixture.Slot(FaceNormalAxis::kX, 2, 1, 1)].structurally_eligible = true;
    fixture.particle.transfer.particle_mass = 1.0;
    fixture.particle.transfer.particle_velocity_slopes.x_row.x = 0x1p66;
    EXPECT_EQ(deposit(inputs).status().code(), absl::StatusCode::kOutOfRange);
    fixture.particle.affine_enabled = false;
    deposited = deposit(inputs);
    ASSERT_TRUE(deposited.ok()) << deposited.status();
    EXPECT_EQ(deposited->face_quanta[fixture.Slot(FaceNormalAxis::kX, 1, 1, 1)].pure_momentum_quanta_sum, -2);
  }
  void NormalizationSeparatesLowMassFromMissingAndRejectsMalformedQuanta() {
    const std::array quanta{
        MacFaceDepositQuanta3D{0, 17, 0},
        MacFaceDepositQuanta3D{1, 8, 0},
        MacFaceDepositQuanta3D{2, 6, 0}
    };
    auto normalized = normalize(quanta, {0.5, 1.0, 1}, 0.5);
    ASSERT_TRUE(normalized.ok()) << normalized.status();
    EXPECT_EQ(normalized->validity[0], LiquidFaceValidity3D::kMissing);
    EXPECT_EQ(normalized->validity[1], LiquidFaceValidity3D::kLowMass);
    EXPECT_EQ(normalized->validity[2], LiquidFaceValidity3D::kValid);
    EXPECT_DOUBLE_EQ(normalized->face_velocities[1], 0.0);
    EXPECT_DOUBLE_EQ(normalized->face_velocities[2], 6.0);
    auto bad = quanta;
    bad[1].mass_quanta_sum = -1;
    EXPECT_EQ(normalize(bad, kQuanta, 0.0).status().code(), absl::StatusCode::kInvalidArgument);
    bad = quanta;
    bad[1].stress_force_quanta_sum = 1;
    EXPECT_EQ(normalize(bad, kQuanta, 0.0).status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(normalize(quanta, kQuanta, -1.0).status().code(), absl::StatusCode::kInvalidArgument);
  }
  void SameSideAffineGhostsExcludePoisonedDonorsAndSeparateHistories() {
    BoundaryFixture fixture;
    for (std::size_t axis_index = 0; axis_index < kAxes.size(); ++axis_index) {
      auto axis = kAxes[axis_index];
      for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
          for (int x = 0; x < 4; ++x) {
            const Vec3 anchor = *fixture.grid.FaceWorldAnchor({axis, {x, y, z}});
            const auto slot = fixture.Slot(axis, x, y, z);
            fixture.old_values[slot] = 1.0 + anchor.x + 2.0 * anchor.y - anchor.z;
            fixture.new_values[slot] = 4.0 + 2.0 * anchor.x - anchor.y + 3.0 * anchor.z;
          }
        }
      }
      const auto poison = fixture.Slot(axis, 1, 1, 1);
      fixture.tags[poison].side_tag = -7;
      fixture.old_values[poison] = -1e8;
      fixture.new_values[poison] = 1e9;
    }
    auto gathered = gather(fixture.GatherInputs());
    ASSERT_TRUE(gathered.ok()) << gathered.status();
    const auto& result = gathered->particles[0];
    ExpectVec(result.new_pic_velocity, {10.0, 10.0, 10.0});
    ExpectVec(result.grid_velocity_increment, {6.0, 6.0, 6.0});
    ExpectVec(result.particle_minus_old_pic_velocity, {-1.0, 0.0, 1.0});
    ExpectVec(result.fitted_velocity_slopes.x_row, {2.0, -1.0, 3.0});
    ExpectVec(result.fitted_velocity_slopes.y_row, {2.0, -1.0, 3.0});
    ExpectVec(result.velocity_gradient.z_row, {2.0, -1.0, 3.0});
    EXPECT_TRUE(result.affine_enabled);
    EXPECT_EQ(gathered->old_pic_fallback_particle_count, 0);
    EXPECT_EQ(gathered->new_pic_fallback_particle_count, 0);
  }
  void IdenticalHistoriesKeepParticleRetentionDistinctFromGridIncrement() {
    BoundaryFixture fixture;
    fixture.new_values = fixture.old_values;
    auto gathered = gather(fixture.GatherInputs());
    ASSERT_TRUE(gathered.ok()) << gathered.status();
    ExpectVec(gathered->particles[0].grid_velocity_increment, {});
    ExpectVec(gathered->particles[0].particle_minus_old_pic_velocity, {1.0, 2.0, 3.0});
  }
  void WallHistoriesUseSeparateTimesAndConstrainObliqueNormal() {
    BoundaryFixture fixture;
    std::fill(fixture.validity.begin(), fixture.validity.end(), LiquidFaceValidity3D::kMissing);
    for (std::size_t slot = 0; slot < fixture.grid.face_count(); ++slot) {
      fixture.old_walls[slot] = {true, 7, {1.0, 0.0, 0.0}, {2.0, 4.0, 6.0}};
      fixture.new_walls[slot] = {true, 7, {1.0, 0.0, 0.0}, {5.0, 8.0, 9.0}};
    }
    auto gathered = gather(fixture.GatherInputs());
    ASSERT_TRUE(gathered.ok()) << gathered.status();
    ExpectVec(gathered->particles[0].new_pic_velocity, {5.0, 8.0, 9.0});
    ExpectVec(gathered->particles[0].grid_velocity_increment, {3.0, 4.0, 3.0});
    ExpectVec(gathered->particles[0].particle_minus_old_pic_velocity, {});
    EXPECT_FALSE(gathered->particles[0].affine_enabled);
    EXPECT_EQ(gathered->old_pic_fallback_particle_count, 1);
    EXPECT_EQ(gathered->new_pic_fallback_particle_count, 1);
    auto mismatched = fixture.GatherInputs();
    mismatched.old_history.wall_sample_time = 1.0;
    EXPECT_EQ(gather(mismatched).status().code(), absl::StatusCode::kInvalidArgument);
    mismatched.old_history.wall_sample_time = 0.0;
    mismatched.old_history.face_velocities = std::span(fixture.old_values).first(1);
    EXPECT_EQ(gather(mismatched).status().code(), absl::StatusCode::kInvalidArgument);

    // A full-rank vector fit (3,3,3) projected onto n=(1,1,0)/sqrt(2)
    // relative to wall=(1,1,0) yields (1,1,3), before component selection.
    std::fill(fixture.validity.begin(), fixture.validity.end(), LiquidFaceValidity3D::kValid);
    const double n = std::sqrt(0.5);
    for (auto& wall : fixture.new_walls) {
      wall = {true, 7, {n, n, 0.0}, {1.0, 1.0, 0.0}};
    }
    const auto slot = fixture.Slot(FaceNormalAxis::kX, 1, 1, 1);
    fixture.validity[slot] = LiquidFaceValidity3D::kMissing;
    gathered = gather(fixture.GatherInputs());
    ASSERT_TRUE(gathered.ok()) << gathered.status();
    EXPECT_NEAR(gathered->particles[0].new_pic_velocity.x, 3.0 - 2.0 * 0.28125, 2e-12);
    EXPECT_TRUE(gathered->particles[0].affine_enabled);
  }
  void SingularDonorsUseNearestContinuationAndSuppressRetention() {
    BoundaryFixture fixture;
    std::fill(fixture.validity.begin(), fixture.validity.end(), LiquidFaceValidity3D::kMissing);
    for (auto axis : kAxes) {
      const auto first = fixture.Slot(axis, 1, 1, 1);
      const auto second = fixture.Slot(
          axis,
          axis == FaceNormalAxis::kX ? 2 : 1,
          axis == FaceNormalAxis::kY ? 2 : 1,
          axis == FaceNormalAxis::kZ ? 2 : 1
      );
      fixture.validity[first] = LiquidFaceValidity3D::kValid;
      fixture.validity[second] = LiquidFaceValidity3D::kPredicted;
      fixture.old_values[first] = 2.0;
      fixture.old_values[second] = 2.0;
      fixture.new_values[first] = 4.0;
      fixture.new_values[second] = 4.0;
    }
    auto gathered = gather(fixture.GatherInputs());
    ASSERT_TRUE(gathered.ok()) << gathered.status();
    ExpectVec(gathered->particles[0].new_pic_velocity, {4.0, 4.0, 4.0});
    ExpectVec(gathered->particles[0].grid_velocity_increment, {2.0, 2.0, 2.0});
    ExpectVec(gathered->particles[0].particle_minus_old_pic_velocity, {});
    EXPECT_FALSE(gathered->particles[0].affine_enabled);
    EXPECT_EQ(gathered->old_pic_fallback_particle_count, 1);
    EXPECT_EQ(gathered->new_pic_fallback_particle_count, 1);
    std::fill(fixture.validity.begin(), fixture.validity.end(), LiquidFaceValidity3D::kMissing);
    EXPECT_EQ(gather(fixture.GatherInputs()).status().code(), absl::StatusCode::kFailedPrecondition);
  }
  void LowMassDisablesAffineAndRetentionEvenWithFullRankGhost() {
    BoundaryFixture fixture;
    auto old_validity = fixture.validity;
    fixture.validity[fixture.Slot(FaceNormalAxis::kX, 1, 1, 1)] = LiquidFaceValidity3D::kLowMass;
    auto inputs = fixture.GatherInputs();
    inputs.old_history.validity = old_validity;
    auto gathered = gather(inputs);
    ASSERT_TRUE(gathered.ok()) << gathered.status();
    ExpectVec(gathered->particles[0].new_pic_velocity, {3.0, 3.0, 3.0});
    ExpectVec(gathered->particles[0].grid_velocity_increment, {1.0, 1.0, 1.0});
    ExpectVec(gathered->particles[0].particle_minus_old_pic_velocity, {});
    EXPECT_FALSE(gathered->particles[0].affine_enabled);
    EXPECT_EQ(gathered->low_mass_fallback_particle_count, 1);
    EXPECT_EQ(gathered->old_pic_fallback_particle_count, 0);
    EXPECT_EQ(gathered->new_pic_fallback_particle_count, 0);
  }
  void NonaffineCompleteFieldReportsFitSeparatelyFromTrueGradient() {
    BoundaryFixture fixture;
    fixture.particle.transfer.trial_world_position = {1.75, 1.75, 1.75};
    fixture.particle.transfer.sorted_world_position = fixture.particle.transfer.trial_world_position;
    for (auto axis : kAxes) {
      for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
          for (int x = 0; x < 4; ++x) {
            const Vec3 anchor = *fixture.grid.FaceWorldAnchor({axis, {x, y, z}});
            const double coordinate =
                axis == FaceNormalAxis::kX ? anchor.x : (axis == FaceNormalAxis::kY ? anchor.y : anchor.z);
            fixture.new_values[fixture.Slot(axis, x, y, z)] = (coordinate - 1.75) * (coordinate - 1.75);
          }
        }
      }
    }
    auto gathered = gather(fixture.GatherInputs());
    ASSERT_TRUE(gathered.ok()) << gathered.status();
    ExpectVec(gathered->particles[0].new_pic_velocity, {0.25, 0.25, 0.25});
    ExpectVec(gathered->particles[0].fitted_velocity_slopes.x_row, {-0.1875, 0.0, 0.0});
    ExpectVec(gathered->particles[0].velocity_gradient.x_row, {});
  }
  void RankRevealingQrRejectsNearDependentNormalizedDirection() {
    // The uniform public lattice cannot approach singularity continuously within
    // a 2h ball; exercise the same internal QR kernel at its numerical cutoff.
    for (double height : {0.0, 1e-12, 1e-8}) {
      const std::array samples{
          internal::LiquidGhostFitSample3D{{0.0, 0.0, 0.0}, 7.0},
          internal::LiquidGhostFitSample3D{{1.0, 0.0, 0.0}, 9.0},
          internal::LiquidGhostFitSample3D{{0.0, 1.0, 0.0}, 10.0},
          internal::LiquidGhostFitSample3D{{1.0, 1.0, height}, 12.0 + 4.0 * height},
      };
      const auto fit = fit_affine(samples);
      ASSERT_TRUE(fit.ok()) << fit.status();
      EXPECT_EQ(fit->full_rank, height == 1e-8);
      if (fit->full_rank) {
        EXPECT_NEAR(fit->velocity_at_target, 7.0, 1e-12);
        ExpectVec(fit->velocity_slopes_in_cell_widths, {2.0, 3.0, 4.0}, 1e-6);
      }
    }
  }
  void NearestFallbackBreaksDistanceTiesByOwnedSlotOnlyInNewStage() {
    BoundaryFixture fixture;
    fixture.particle.transfer.sorted_world_position.x = 2.0;
    fixture.particle.transfer.trial_world_position.x = 2.0;
    auto new_validity = fixture.validity;
    for (int z = 0; z < 4; ++z) {
      for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
          new_validity[fixture.Slot(FaceNormalAxis::kX, x, y, z)] = LiquidFaceValidity3D::kMissing;
        }
      }
    }
    const auto low_slot = fixture.Slot(FaceNormalAxis::kX, 1, 1, 1);
    const auto high_slot = fixture.Slot(FaceNormalAxis::kX, 3, 1, 1);
    new_validity[low_slot] = LiquidFaceValidity3D::kValid;
    new_validity[high_slot] = LiquidFaceValidity3D::kPredicted;
    fixture.new_values[low_slot] = 2.0;
    fixture.new_values[high_slot] = 8.0;
    auto inputs = fixture.GatherInputs();
    inputs.new_history.validity = new_validity;
    const auto gathered = gather(inputs);
    ASSERT_TRUE(gathered.ok()) << gathered.status();
    EXPECT_NEAR(gathered->particles[0].new_pic_velocity.x, 2.75, 1e-12);
    EXPECT_NEAR(gathered->particles[0].grid_velocity_increment.x, 0.75, 1e-12);
    EXPECT_EQ(gathered->old_pic_fallback_particle_count, 0);
    EXPECT_EQ(gathered->new_pic_fallback_particle_count, 1);
    ExpectVec(gathered->particles[0].particle_minus_old_pic_velocity, {});
    EXPECT_FALSE(gathered->particles[0].affine_enabled);
    // The true full-stencil derivative remains meaningful despite affine-off.
    EXPECT_NEAR(gathered->particles[0].velocity_gradient.x_row.x, 3.0, 1e-12);
  }
  void QuantizesEachParticleBeforeSummingAndIncludesGuardEndpoint() {
    BoundaryFixture fixture;
    std::fill(fixture.tags.begin(), fixture.tags.end(), LiquidBoundaryFaceTag3D{7, false});
    for (auto axis : kAxes) {
      fixture.tags[fixture.Slot(axis, 1, 1, 1)].structurally_eligible = true;
    }
    fixture.particle.transfer.particle_mass = 0.5;
    fixture.particle.transfer.particle_velocity = {-3.0, 3.0, 5.0};
    std::array particles{fixture.particle, fixture.particle};
    auto inputs = fixture.DepositInputs();
    inputs.particles = particles;
    inputs.quantization = {1.0, 1.0, 2};
    auto deposited = deposit(inputs);
    ASSERT_TRUE(deposited.ok()) << deposited.status();
    const auto slot = fixture.Slot(FaceNormalAxis::kX, 1, 1, 1);
    EXPECT_EQ(deposited->face_quanta[slot].mass_quanta_sum, 0);
    EXPECT_EQ(deposited->face_quanta[slot].pure_momentum_quanta_sum, -4);
    for (auto& particle : particles) {
      particle.transfer.particle_mass = 1.0;
      particle.transfer.particle_velocity = {0x1p61, 0.0, 0.0};
    }
    deposited = deposit(inputs);
    ASSERT_TRUE(deposited.ok()) << deposited.status();
    EXPECT_EQ(deposited->face_quanta[slot].pure_momentum_quanta_sum, std::int64_t{1} << 62);
    particles[1].transfer.particle_velocity.x = std::nextafter(0x1p61, std::numeric_limits<double>::infinity());
    EXPECT_EQ(deposit(inputs).status().code(), absl::StatusCode::kOutOfRange);
  }
  void CompleteBoundaryDepositPreservesUnmaskedApiAtSignedTranslatedScale() {
    constexpr std::array blocks{BlockIndex3D{-1, -1, -1}};
    const auto grid = SparseMacGrid3D::Create({{11.0, -17.0, 31.0}, 2.0}, blocks, 1);
    ASSERT_TRUE(grid.ok()) << grid.status();
    const LiquidBoundaryParticle3D particle{
        .transfer =
            {{6.5, -21.5, 26.5}, {6.5, -21.5, 26.5}, 2.0, {1.0, 2.0, 3.0}, {{2.0, -1.0, 3.0}, {}, {1.0, 1.0, 1.0}}},
        .side_tag = -3,
        .affine_enabled = true,
    };
    std::vector<LiquidBoundaryFaceTag3D> tags(grid->face_count(), {-3, true});
    std::vector<MacFaceDepositQuanta3D> unmasked(grid->face_count());
    ASSERT_TRUE(DepositMacQuanta3D(*grid, {&particle.transfer, 1}, kQuanta, unmasked).ok());
    const auto deposited = deposit({*grid, {&particle, 1}, tags, kQuanta});
    ASSERT_TRUE(deposited.ok()) << deposited.status();
    for (std::size_t slot = 0; slot < grid->face_count(); ++slot) {
      EXPECT_EQ(deposited->face_quanta[slot].mass_quanta_sum, unmasked[slot].mass_quanta_sum);
      EXPECT_EQ(deposited->face_quanta[slot].pure_momentum_quanta_sum, unmasked[slot].pure_momentum_quanta_sum);
    }
  }
  void RejectsMalformedHistoryCapacityAndArithmeticWithoutMutatingInputs() {
    BoundaryFixture fixture;
    auto deposit_inputs = fixture.DepositInputs();
    deposit_inputs.quantization.declared_particle_capacity = 0;
    EXPECT_EQ(deposit(deposit_inputs).status().code(), absl::StatusCode::kResourceExhausted);
    deposit_inputs.quantization = {3.0, 1.0, 1};
    EXPECT_EQ(deposit(deposit_inputs).status().code(), absl::StatusCode::kInvalidArgument);
    deposit_inputs.quantization = kQuanta;
    deposit_inputs.face_tags = std::span(fixture.tags).first(1);
    EXPECT_EQ(deposit(deposit_inputs).status().code(), absl::StatusCode::kInvalidArgument);
    std::vector<LiquidBoundaryParticle3D> too_many_particles(kLiquidBoundaryParticleCapacity3D + 1, fixture.particle);
    auto gather_inputs = fixture.GatherInputs();
    gather_inputs.particles = too_many_particles;
    EXPECT_EQ(gather(gather_inputs).status().code(), absl::StatusCode::kResourceExhausted);
    gather_inputs.particles = {&fixture.particle, 1};
    gather_inputs.new_time = 0.0;
    EXPECT_EQ(gather(gather_inputs).status().code(), absl::StatusCode::kInvalidArgument);
    gather_inputs.new_time = 1.0;
    fixture.old_walls[0] = {true, 7, {2.0, 0.0, 0.0}, {}};
    EXPECT_EQ(gather(gather_inputs).status().code(), absl::StatusCode::kInvalidArgument);
    fixture.old_walls[0] = {};
    fixture.old_values[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(gather(gather_inputs).status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(std::isnan(fixture.old_values[0]));
    // Eligibility never hides a nonfinite payload marked valid or predicted.
    fixture.tags[0].side_tag = -7;
    fixture.validity[0] = LiquidFaceValidity3D::kPredicted;
    EXPECT_EQ(gather(gather_inputs).status().code(), absl::StatusCode::kInvalidArgument);
    fixture.validity[0] = LiquidFaceValidity3D::kMissing;
    EXPECT_TRUE(gather(gather_inputs).ok());
    fixture.tags[0].side_tag = 7;
    fixture.validity[0] = LiquidFaceValidity3D::kValid;
    fixture.old_values[0] = 2.0;
    fixture.particle.transfer.particle_velocity.x = std::numeric_limits<double>::max();
    std::fill(fixture.old_values.begin(), fixture.old_values.end(), -std::numeric_limits<double>::max() / 2.0);
    std::fill(fixture.new_values.begin(), fixture.new_values.end(), 0.0);
    EXPECT_EQ(gather(gather_inputs).status().code(), absl::StatusCode::kOutOfRange);
    EXPECT_DOUBLE_EQ(fixture.new_values[0], 0.0);
    EXPECT_DOUBLE_EQ(fixture.particle.transfer.particle_velocity.x, std::numeric_limits<double>::max());
  }
};
}  // namespace boundary_transfer_test
}  // namespace fluidity
