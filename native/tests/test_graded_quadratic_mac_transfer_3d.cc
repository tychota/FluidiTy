#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "fluidity/host/graded_quadratic_mac_transfer_3d.h"
#include "fluidity/host/transfers_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr Vec3 kOrigin{4, -2, 8};
constexpr std::int64_t kSide = -17;
constexpr std::array kAxes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};

void ExpectBits(double actual, double expected) {
  EXPECT_EQ(std::bit_cast<std::uint64_t>(actual), std::bit_cast<std::uint64_t>(expected));
}

void ExpectBits(Vec3 actual, Vec3 expected) {
  ExpectBits(actual.x, expected.x);
  ExpectBits(actual.y, expected.y);
  ExpectBits(actual.z, expected.z);
}

void ExpectBits(Mat3 actual, Mat3 expected) {
  ExpectBits(actual.x_row, expected.x_row);
  ExpectBits(actual.y_row, expected.y_row);
  ExpectBits(actual.z_row, expected.z_row);
}

double CarrierVelocity(FaceNormalAxis axis, Vec3 anchor, bool new_stage) {
  const double base = static_cast<double>(axis) + 0.125 * anchor.x * anchor.x - 0.25 * anchor.y + 0.5 * anchor.z;
  return new_stage ? base + 0.0625 * anchor.x * anchor.z + 0.25 : base;
}

std::vector<AdaptiveMacCellKey3D> SameLevelKeys() {
  std::vector<AdaptiveMacCellKey3D> keys;
  for (int z = -3; z < 3; ++z) {
    for (int y = -3; y < 3; ++y) {
      for (int x = -3; x < 3; ++x) {
        keys.push_back({.lower_corner = {2 * x, 2 * y, 2 * z}, .width_exponent = 1});
      }
    }
  }
  return keys;
}

TEST(GradedQuadraticMacTransfer3D, SignedIslandDepositAndGatherMatchUniformBits) {
  auto keys = SameLevelKeys();
  const auto topology = AdaptiveMacTopology3D::Create(keys, 0.5);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto adaptive = GradedQuadraticMacTransfer3D::Create(*topology, kOrigin);
  ASSERT_TRUE(adaptive.ok()) << adaptive.status();
  const auto anchors = GradedMacTransfer3D::Create(*topology, kOrigin);
  ASSERT_TRUE(anchors.ok()) << anchors.status();

  std::array<BlockIndex3D, 8> blocks{};
  std::size_t block = 0;
  for (int z = -1; z < 1; ++z) {
    for (int y = -1; y < 1; ++y) {
      for (int x = -1; x < 1; ++x) {
        blocks[block++] = {x, y, z};
      }
    }
  }
  const auto uniform = SparseMacGrid3D::Create({kOrigin, 1}, blocks, blocks.size());
  ASSERT_TRUE(uniform.ok()) << uniform.status();

  const std::array particles{
      MacTransferParticle3D{
          .sorted_world_position = {3.5, -1.25, 7.75},
          .trial_world_position = {3.5, -1.375, 7.75},
          .particle_mass = 2,
          .particle_velocity = {1, -2, 4},
          .particle_velocity_slopes = {{2, 4, -1}, {-5, 1, 5}, {7, -2, 3}},
      },
      MacTransferParticle3D{
          .sorted_world_position = {3.75, -1.375, 7.5},
          .trial_world_position = {3.875, -1.25, 7.5},
          .particle_mass = 3.25,
          .particle_velocity = {-2, 1, -1},
          .particle_velocity_slopes = {{-4, 1, 2}, {6, 0, -2}, {-3, 5, -1}},
      },
  };
  const std::array graded_particles{
      GradedQuadraticMacParticle3D{.transfer = particles[0], .side_tag = kSide},
      GradedQuadraticMacParticle3D{.transfer = particles[1], .side_tag = kSide},
  };

  std::vector<double> uniform_old(uniform->face_count());
  std::vector<double> uniform_new(uniform->face_count());
  for (std::size_t slot = 0; slot < uniform->face_count(); ++slot) {
    const auto key = uniform->view().FaceIndex(slot);
    ASSERT_TRUE(key.ok()) << key.status();
    const auto anchor = uniform->FaceWorldAnchor(*key);
    ASSERT_TRUE(anchor.ok()) << anchor.status();
    uniform_old[slot] = CarrierVelocity(key->axis, *anchor, false);
    uniform_new[slot] = CarrierVelocity(key->axis, *anchor, true);
  }
  std::vector<double> areas(anchors->faces().size(), 1);
  std::vector<LiquidAdaptiveFaceState3D> old_faces(areas.size());
  std::vector<LiquidAdaptiveFaceState3D> new_faces(areas.size());
  for (std::size_t slot = 0; slot < areas.size(); ++slot) {
    const auto& face = anchors->faces()[slot];
    old_faces[slot] = {
        .carrier_velocity = CarrierVelocity(face.axis, face.world_anchor, false),
        .validity = LiquidFaceValidity3D::kValid,
        .side_tag = kSide,
    };
    new_faces[slot] = old_faces[slot];
    new_faces[slot].carrier_velocity = CarrierVelocity(face.axis, face.world_anchor, true);
  }

  std::vector<MacFaceDeposit3D> uniform_deposits(uniform->face_count());
  const auto deposited = DepositMac3D(*uniform, particles, uniform_deposits);
  ASSERT_TRUE(deposited.ok()) << deposited;
  std::array<MacParticleGatherResult3D, particles.size()> uniform_gathers{};
  const auto gathered = GatherMac3D(*uniform, particles, uniform_old, uniform_new, uniform_gathers);
  ASSERT_TRUE(gathered.ok()) << gathered;

  const auto deposits = adaptive->Deposit(graded_particles, areas, old_faces);
  ASSERT_TRUE(deposits.ok()) << deposits.status();
  ASSERT_EQ(deposits->size(), areas.size());
  std::ranges::reverse(keys);
  const auto reordered_topology = AdaptiveMacTopology3D::Create(keys, 0.5);
  ASSERT_TRUE(reordered_topology.ok()) << reordered_topology.status();
  const auto reordered = GradedQuadraticMacTransfer3D::Create(*reordered_topology, kOrigin);
  ASSERT_TRUE(reordered.ok()) << reordered.status();
  std::vector<bool> touched(areas.size());
  std::size_t zero_weight_slot = areas.size();
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const auto binding = adaptive->Bind(graded_particles[index], areas, old_faces);
    const auto repeated_binding = reordered->Bind(graded_particles[index], areas, old_faces);
    ASSERT_TRUE(binding.ok()) << binding.status();
    ASSERT_TRUE(repeated_binding.ok()) << repeated_binding.status();
    EXPECT_EQ(binding->cell_width_m, 1);
    for (std::size_t axis = 0; axis < kAxes.size(); ++axis) {
      const auto uniform_row = BuildOwnedMacFaceStencil3D(
          *uniform,
          kAxes[axis],
          particles[index].sorted_world_position,
          particles[index].trial_world_position
      );
      ASSERT_TRUE(uniform_row.ok()) << uniform_row.status();
      for (std::size_t node = 0; node < uniform_row->size(); ++node) {
        const auto& actual = binding->rows[axis][node];
        const auto& expected = (*uniform_row)[node];
        EXPECT_EQ(actual.face_slot, repeated_binding->rows[axis][node].face_slot);
        const auto uniform_key = uniform->view().FaceIndex(expected.face_slot);
        ASSERT_TRUE(uniform_key.ok()) << uniform_key.status();
        const auto uniform_anchor = uniform->FaceWorldAnchor(*uniform_key);
        ASSERT_TRUE(uniform_anchor.ok()) << uniform_anchor.status();
        const auto& bound_face = anchors->faces()[actual.face_slot];
        EXPECT_EQ(bound_face.axis, kAxes[axis]);
        ExpectBits(bound_face.world_anchor, *uniform_anchor);
        ExpectBits(actual.weight, expected.weight);
        ExpectBits(actual.face_offset_from_particle, expected.face_offset_from_particle);
        ExpectBits(actual.weight_gradient, expected.weight_gradient);
        ExpectBits((*deposits)[actual.face_slot].deposited_mass, uniform_deposits[expected.face_slot].deposited_mass);
        ExpectBits(
            (*deposits)[actual.face_slot].deposited_pure_momentum,
            uniform_deposits[expected.face_slot].deposited_pure_momentum
        );
        touched[actual.face_slot] = true;
        if (index == 0 && actual.weight == 0) {
          zero_weight_slot = actual.face_slot;
        }
      }
    }
    const auto actual = adaptive->Gather(graded_particles[index], areas, old_faces, new_faces);
    ASSERT_TRUE(actual.ok()) << actual.status();
    const auto& expected = uniform_gathers[index];
    ExpectBits(actual->new_pic_velocity, expected.new_pic_velocity);
    ExpectBits(actual->grid_velocity_increment, expected.grid_velocity_increment);
    ExpectBits(actual->fitted_velocity_slopes, expected.fitted_velocity_slopes);
    ExpectBits(actual->velocity_gradient, expected.velocity_gradient);
  }
  for (std::size_t slot = 0; slot < areas.size(); ++slot) {
    if (!touched[slot]) {
      EXPECT_EQ((*deposits)[slot].deposited_mass, 0);
      EXPECT_EQ((*deposits)[slot].deposited_pure_momentum, 0);
    }
  }

  ASSERT_LT(zero_weight_slot, areas.size());
  new_faces[zero_weight_slot].validity = LiquidFaceValidity3D::kLowMass;
  EXPECT_EQ(
      adaptive->Gather(graded_particles[0], areas, old_faces, new_faces).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  old_faces[zero_weight_slot].structurally_eligible = false;
  EXPECT_EQ(
      adaptive->Deposit(graded_particles, areas, old_faces).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  old_faces[zero_weight_slot].structurally_eligible = true;
  old_faces[zero_weight_slot].side_tag = -kSide;
  EXPECT_EQ(
      adaptive->Bind(graded_particles[0], areas, old_faces).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  old_faces[zero_weight_slot].side_tag = kSide;
  areas[zero_weight_slot] = std::nextafter(1.0, 0.0);
  EXPECT_EQ(
      adaptive->Bind(graded_particles[0], areas, old_faces).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  areas[zero_weight_slot] = -1;
  EXPECT_EQ(adaptive->Bind(graded_particles[0], areas, old_faces).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(GradedQuadraticMacTransfer3D, RefusesFineWidthFaceWithCoarseIncidentLeaf) {
  std::vector<AdaptiveMacCellKey3D> keys;
  for (int z = -4; z < 4; z += 2) {
    for (int y = -4; y < 4; y += 2) {
      keys.push_back({.lower_corner = {-2, y, z}, .width_exponent = 1});
    }
  }
  for (int z = -4; z < 4; ++z) {
    for (int y = -4; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        keys.push_back({.lower_corner = {x, y, z}, .width_exponent = 0});
      }
    }
  }
  const auto topology = AdaptiveMacTopology3D::Create(keys, 1);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto adaptive = GradedQuadraticMacTransfer3D::Create(*topology, {});
  ASSERT_TRUE(adaptive.ok()) << adaptive.status();
  const auto anchors = GradedMacTransfer3D::Create(*topology, {});
  ASSERT_TRUE(anchors.ok()) << anchors.status();
  std::vector<double> areas;
  for (const auto& face : anchors->faces()) {
    areas.push_back(face.width_m * face.width_m);
  }
  std::vector<LiquidAdaptiveFaceState3D> carriers(
      areas.size(),
      {.validity = LiquidFaceValidity3D::kValid, .side_tag = kSide}
  );
  const GradedQuadraticMacParticle3D particle{
      .transfer =
          {
              .sorted_world_position = {0.75, 0.5, 0.5},
              .trial_world_position = {0.75, 0.5, 0.5},
              .particle_mass = 1,
          },
      .side_tag = kSide,
  };
  const auto bound = adaptive->Bind(particle, areas, carriers);
  ASSERT_EQ(bound.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_NE(bound.status().message().find("incident leaves"), std::string_view::npos);
  EXPECT_EQ(
      adaptive->Deposit(std::span(&particle, 1), areas, carriers).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(
      adaptive->Gather(particle, areas, carriers, carriers).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(GradedQuadraticMacTransfer3D, CertifiesPhysicalSplitAndMergeAcrossTwoTopologySnapshots) {
  const auto coarse_keys = SameLevelKeys();
  std::vector<AdaptiveMacCellKey3D> fine_keys;
  for (const auto key : coarse_keys) {
    const auto lower = key.lower_corner;
    if (lower.x < -2 || lower.x > 2 || lower.y < -2 || lower.y > 2 || lower.z < -2 || lower.z > 2) {
      fine_keys.push_back(key);
      continue;
    }
    for (int z = 0; z < 2; ++z) {
      for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
          fine_keys.push_back({
              .lower_corner = {lower.x + x, lower.y + y, lower.z + z},
              .width_exponent = 0,
          });
        }
      }
    }
  }

  const auto coarse_topology = AdaptiveMacTopology3D::Create(coarse_keys, 0.5);
  const auto fine_topology = AdaptiveMacTopology3D::Create(fine_keys, 0.5);
  ASSERT_TRUE(coarse_topology.ok()) << coarse_topology.status();
  ASSERT_TRUE(fine_topology.ok()) << fine_topology.status();
  const auto coarse_transfer = GradedQuadraticMacTransfer3D::Create(*coarse_topology, {});
  const auto fine_transfer = GradedQuadraticMacTransfer3D::Create(*fine_topology, {});
  const auto coarse_anchors = GradedMacTransfer3D::Create(*coarse_topology, {});
  const auto fine_anchors = GradedMacTransfer3D::Create(*fine_topology, {});
  ASSERT_TRUE(coarse_transfer.ok()) << coarse_transfer.status();
  ASSERT_TRUE(fine_transfer.ok()) << fine_transfer.status();
  ASSERT_TRUE(coarse_anchors.ok()) << coarse_anchors.status();
  ASSERT_TRUE(fine_anchors.ok()) << fine_anchors.status();

  const auto full_areas = [](const GradedMacTransfer3D& transfer) {
    std::vector<double> areas;
    for (const auto& face : transfer.faces()) {
      areas.push_back(face.width_m * face.width_m);
    }
    return areas;
  };
  const auto coarse_areas = full_areas(*coarse_anchors);
  const auto fine_areas = full_areas(*fine_anchors);
  const LiquidAdaptiveFaceState3D carrier{
      .validity = LiquidFaceValidity3D::kValid,
      .side_tag = kSide,
  };
  std::vector<LiquidAdaptiveFaceState3D> coarse_faces(coarse_areas.size(), carrier);
  std::vector<LiquidAdaptiveFaceState3D> fine_faces(fine_areas.size(), carrier);
  const GradedQuadraticMacSupportSnapshot3D coarse{*coarse_transfer, coarse_areas, coarse_faces};
  const GradedQuadraticMacSupportSnapshot3D fine{*fine_transfer, fine_areas, fine_faces};
  const LiquidParticleState3D parent{
      .id = 1,
      .world_position = {0.5, 0.5, 0.5},
      .world_velocity = {0.25, -0.125, 0.5},
      .velocity_slopes = {{0.125, 0.25, 0}, {-0.25, 0.125, 0}, {0, 0.25, -0.125}},
      .mass = 2,
      .side_tag = kSide,
      .pic_initialized = true,
  };
  int certified = 0;
  const auto certify = [](const LiquidParticleState3D&, double, LiquidUniformResamplingBasis3D basis, void* context) {
    EXPECT_EQ(basis, LiquidUniformResamplingBasis3D::kQuadratic);
    ++*static_cast<int*>(context);
    return absl::OkStatus();
  };

  const auto split = SplitGradedQuadraticLiquidParticle3D(parent, 2, 1, coarse, fine, certify, &certified);
  ASSERT_TRUE(split.ok()) << split.status();
  EXPECT_EQ(certified, 9);
  EXPECT_EQ(split->next_particle_id, 10);
  EXPECT_EQ(split->kinetic_affine_energy_deficit, 0);
  double split_mass = 0;
  double split_reference_volume = 0;
  Vec3 split_momentum{};
  for (const auto& child : split->children) {
    split_mass += child.mass;
    split_reference_volume += child.reference_volume;
    split_momentum.x += child.mass * child.world_velocity.x;
    split_momentum.y += child.mass * child.world_velocity.y;
    split_momentum.z += child.mass * child.world_velocity.z;
  }
  EXPECT_EQ(split_mass, parent.mass);
  EXPECT_EQ(split_reference_volume, parent.reference_volume);
  EXPECT_NEAR(split_momentum.x, parent.mass * parent.world_velocity.x, 1e-15);
  EXPECT_NEAR(split_momentum.y, parent.mass * parent.world_velocity.y, 1e-15);
  EXPECT_NEAR(split_momentum.z, parent.mass * parent.world_velocity.z, 1e-15);

  auto children = split->children;
  for (std::size_t ordinal = 0; ordinal < children.size(); ++ordinal) {
    children[ordinal].world_position = {
        (ordinal & 1u) == 0 ? 0.25 : 0.75,
        (ordinal & 2u) == 0 ? 0.25 : 0.75,
        (ordinal & 4u) == 0 ? 0.25 : 0.75,
    };
  }
  const auto merge = MergeGradedQuadraticLiquidParticles3D(children, 10, 0.5, fine, coarse, certify, &certified);
  ASSERT_TRUE(merge.ok()) << merge.status();
  EXPECT_EQ(certified, 18);
  EXPECT_GT(merge->covariance_gap_lower_bound, 0);
  EXPECT_EQ(merge->particle.mass, parent.mass);
  EXPECT_EQ(merge->particle.reference_volume, parent.reference_volume);
  ExpectBits(merge->particle.world_position, parent.world_position);

  const auto wrong_width = SplitGradedQuadraticLiquidParticle3D(parent, 2, 1, coarse, coarse, certify, &certified);
  EXPECT_EQ(wrong_width.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(certified, 19);
  std::fill(coarse_faces.begin(), coarse_faces.end(), LiquidAdaptiveFaceState3D{});
  const auto unavailable = MergeGradedQuadraticLiquidParticles3D(children, 10, 0.5, fine, coarse, certify, &certified);
  EXPECT_EQ(unavailable.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(certified, 27);
}

}  // namespace
}  // namespace fluidity
