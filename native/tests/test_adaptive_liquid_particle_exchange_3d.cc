#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "fluidity/host/adaptive_liquid_particle_exchange_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr std::array<AdaptiveMacCellKey3D, 3> kLeaves{{
    {{-2, 0, 0}, 1},
    {{0, 0, 0}, 0},
    {{0, 1, 0}, 0},
}};

absl::StatusOr<LiquidState3D> MakeSnapshot() {
  LiquidState3D state;
  state.clock = {.time = 0.5, .frame_ordinal = 1, .accepted_step_ordinal = 1, .epoch = 1};
  state.next_particle_id = 10;
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(3);
  auto cells = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(3);
  if (!particles.ok()) {
    return particles.status();
  }
  if (!cells.ok()) {
    return cells.status();
  }
  const auto topology = AdaptiveMacTopology3D::Create(kLeaves, 1.0);
  if (!topology.ok()) {
    return topology.status();
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  const auto face_count = topology->interior_faces().size() + ownership->exterior_faces.size();
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(face_count);
  if (!faces.ok()) {
    return faces.status();
  }
  EXPECT_TRUE(particles->Resize(3).ok());
  EXPECT_TRUE(cells->Resize(3).ok());
  EXPECT_TRUE(faces->Resize(face_count).ok());
  (*particles)[0] = {
      .id = 8,
      .world_position = {0.25, 0.25, 0.25},
      .world_velocity = {2, 0, 0},
      .mass = 0.125,
      .reference_volume = 0.125,
  };
  (*particles)[1] = {
      .id = 3,
      .world_position = {0.25, 1.25, 0.25},
      .world_velocity = {-2, 1, 0},
      .mass = 0.25,
      .reference_volume = 0.25,
  };
  (*particles)[2] = {
      .id = 1,
      .world_position = {0.75, 0.75, 0.75},
      .world_velocity = {2, 0, 0},
      .mass = 0.125,
      .reference_volume = 0.125,
  };
  (*cells)[0] = {
      .key = kLeaves[0],
      .material_owner = AdaptiveLiquidMaterialOwner3D::kEulerian,
      .inventory = {0.5, 0.5, 0.5, {0.5, 0, 0}, 0},
  };
  (*cells)[1] = {.key = kLeaves[1], .inventory = {0.25, 0.25, 0.25, {0.5, 0, 0}, 0}};
  (*cells)[2] = {.key = kLeaves[2], .inventory = {0.25, 0.25, 0.25, {-0.5, 0.25, 0}, 0}};
  state.particles = std::move(*particles);
  state.adaptive_cells = std::move(*cells);
  state.adaptive_faces = std::move(*faces);
  return state;
}

AdaptiveLiquidOwnerExchangeFace3D Face(double velocity) {
  return {
      .projected_carrier = {.carrier_velocity = velocity, .validity = LiquidFaceValidity3D::kValid},
      .open_area_m2 = 1,
  };
}

struct ExchangeInputs {
  std::array<AdaptiveLiquidOwnerExchangeFace3D, 3> faces{Face(0.125), Face(-0.125), Face(7)};
  std::array<AdaptiveLiquidParticleConstitutiveState3D, 3> states{
      AdaptiveLiquidParticlePrescribedState3D{1, 1},
      AdaptiveLiquidParticlePrescribedState3D{1, 1},
      AdaptiveLiquidParticlePrescribedState3D{1, 1},
  };

  AdaptiveLiquidParticleExchangeInputs3D For(LiquidClock3D source_clock) const {
    return {
        .projected_source_clock = source_clock,
        .projected_interior_faces = faces,
        .step_duration_s = 1,
        .constitutive_states = states,
        .certify_support =
            [](void*, std::size_t, const LiquidParticleState3D&, const LiquidParticleState3D&) {
              return absl::OkStatus();
            },
    };
  }
};

TEST(AdaptiveLiquidParticleExchange3D, ReconcilesSignedMixedFacesAndPreservesParticleIdentity) {
  const auto state = MakeSnapshot();
  ASSERT_TRUE(state.ok()) << state.status();
  const ExchangeInputs fixture;
  const auto inputs = fixture.For(state->clock);
  const auto result = PrepareAdaptiveLiquidParticleExchange3D(*state, inputs);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->source_clock, state->clock);
  EXPECT_EQ(result->next_particle_id, state->next_particle_id);
  ASSERT_EQ(result->exchange.transfers.size(), 2);
  ASSERT_EQ(result->candidate_particles.size(), 3);
  EXPECT_EQ(result->candidate_particles[0], state->particles[0]);
  auto expected_donor = state->particles[1];
  expected_donor.mass = 0.125;
  expected_donor.reference_volume = 0.125;
  auto expected_receiver = state->particles[2];
  expected_receiver.mass = 0.25;
  expected_receiver.reference_volume = 0.25;
  expected_receiver.world_velocity.x = 1.5;
  EXPECT_EQ(result->candidate_particles[1], expected_donor);
  EXPECT_EQ(result->candidate_particles[2], expected_receiver);
  EXPECT_EQ(result->exchange.candidate_cells[1].inventory.current_volume_m3, 0.375);
  EXPECT_EQ(result->exchange.candidate_cells[1].inventory.momentum_kg_m_per_s.x, 0.625);
  EXPECT_EQ(result->exchange.candidate_cells[2].inventory.current_volume_m3, 0.125);
  EXPECT_EQ(result->exchange.candidate_cells[2].inventory.momentum_kg_m_per_s.x, -0.25);
  const auto repeated = PrepareAdaptiveLiquidParticleExchange3D(*state, inputs);
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  EXPECT_EQ(repeated->candidate_particles, result->candidate_particles);
  EXPECT_EQ(repeated->exchange.candidate_cells, result->exchange.candidate_cells);
}

TEST(AdaptiveLiquidParticleExchange3D, RefusesEmptyUnsupportedThermalAndInconsistentConstitutiveState) {
  auto state = MakeSnapshot();
  ASSERT_TRUE(state.ok()) << state.status();
  ExchangeInputs fixture;
  auto inputs = fixture.For(state->clock);
  const auto initial = PrepareAdaptiveLiquidParticleExchange3D(*state, inputs);
  ASSERT_TRUE(initial.ok()) << initial.status();
  const std::vector<LiquidParticleState3D> original(state->particles.values().begin(), state->particles.values().end());
  const auto committed_clock = state->clock;

  inputs.certify_support = nullptr;
  EXPECT_EQ(
      PrepareAdaptiveLiquidParticleExchange3D(*state, inputs).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  inputs = fixture.For(state->clock);
  inputs.certify_support = [](void*, std::size_t leaf, const LiquidParticleState3D&, const LiquidParticleState3D&) {
    return leaf == 2 ? absl::FailedPreconditionError("unsupported resident") : absl::OkStatus();
  };
  EXPECT_EQ(
      PrepareAdaptiveLiquidParticleExchange3D(*state, inputs).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  inputs = fixture.For(state->clock);
  fixture.states[1] = AdaptiveLiquidParticleEosState3D{{1, 1, LiquidEosLaw::kTait}, 1};
  EXPECT_EQ(
      PrepareAdaptiveLiquidParticleExchange3D(*state, inputs).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  fixture.states[1] = AdaptiveLiquidParticleEosState3D{{1, 1, LiquidEosLaw::kTait}, -1};
  EXPECT_EQ(
      PrepareAdaptiveLiquidParticleExchange3D(*state, inputs).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  fixture.states[1] = AdaptiveLiquidParticleEosState3D{{1, 1, LiquidEosLaw::kTait}, 0};
  EXPECT_TRUE(PrepareAdaptiveLiquidParticleExchange3D(*state, inputs).ok());

  state->adaptive_cells[0].inventory.enthalpy_j = 1;
  EXPECT_EQ(
      PrepareAdaptiveLiquidParticleExchange3D(*state, inputs).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  state->adaptive_cells[0].inventory.enthalpy_j = 0;
  ++inputs.projected_source_clock.epoch;
  EXPECT_EQ(
      PrepareAdaptiveLiquidParticleExchange3D(*state, inputs).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(state->clock, committed_clock);
  EXPECT_TRUE(std::ranges::equal(state->particles.values(), original));

  inputs = fixture.For(state->clock);
  state->particles[0] = state->particles[1];
  ASSERT_TRUE(state->particles.Resize(1).ok());
  state->adaptive_cells[1].inventory = {};
  EXPECT_EQ(
      PrepareAdaptiveLiquidParticleExchange3D(*state, inputs).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(state->next_particle_id, 10);
  EXPECT_EQ(state->particles[0], original[1]);
}

TEST(AdaptiveLiquidParticleExchange3D, RefusesLedgerMismatchAndDeletionOfAResident) {
  auto state = MakeSnapshot();
  ASSERT_TRUE(state.ok()) << state.status();
  ExchangeInputs fixture;
  const auto inputs = fixture.For(state->clock);
  state->particles[2].mass += 0.125;
  EXPECT_EQ(
      PrepareAdaptiveLiquidParticleExchange3D(*state, inputs).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  state->particles[2].mass -= 0.125;

  fixture.faces[1] = Face(-0.25);
  const auto drain = PrepareAdaptiveLiquidParticleExchange3D(*state, inputs);
  EXPECT_EQ(drain.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(state->particles[1].mass, 0.25);
  EXPECT_EQ(state->particles[1].id, 3);
  EXPECT_EQ(state->adaptive_cells[2].inventory.current_volume_m3, 0.25);

  fixture.faces[1] = Face(-0.125);
  state->particles[0].world_position.x = -1;
  EXPECT_EQ(
      PrepareAdaptiveLiquidParticleExchange3D(*state, inputs).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(state->particles[0].world_position.x, -1);
}

TEST(AdaptiveLiquidParticleExchange3D, RefusesMomentumBetweenAdjacentRepresentableVelocities) {
  auto state = MakeSnapshot();
  ASSERT_TRUE(state.ok()) << state.status();
  state->particles[0] = state->particles[1];
  state->particles[1] = state->particles[2];
  ASSERT_TRUE(state->particles.Resize(2).ok());
  state->particles[1].mass = 0.09375;
  state->particles[1].reference_volume = 0.09375;
  state->particles[1].world_velocity.x = 1.5;
  state->adaptive_cells[0].inventory = {0.03125, 0.03125, 0.03125, {-0.0546875, 0, 0}, 0};
  state->adaptive_cells[1].inventory = {0.09375, 0.09375, 0.09375, {0.140625, 0, 0}, 0};
  ExchangeInputs fixture;
  fixture.faces = {Face(0.015625), Face(0), Face(0)};

  constexpr double kTargetMass = 0.109375;
  constexpr double kTargetMomentum = 0.11328125;
  const double high_velocity = kTargetMomentum / kTargetMass;
  const double low_velocity = std::nextafter(high_velocity, 0.0);
  ASSERT_LT(kTargetMass * low_velocity, kTargetMomentum);
  ASSERT_GT(kTargetMass * high_velocity, kTargetMomentum);
  const auto result = PrepareAdaptiveLiquidParticleExchange3D(*state, fixture.For(state->clock));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(state->particles[1].world_velocity.x, 1.5);
  EXPECT_EQ(state->particles[1].mass, 0.09375);
}

}  // namespace
}  // namespace fluidity
