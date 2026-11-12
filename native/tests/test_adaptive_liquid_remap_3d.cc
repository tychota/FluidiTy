#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#include "fluidity/host/adaptive_liquid_remap_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr AdaptiveMacCellKey3D kParent{{-2, 0, -2}, 1};
constexpr std::array<AdaptiveMacCellKey3D, 9> kLeaves{{
    kParent,
    {{-4, 0, -2}, 1},
    {{0, 0, -2}, 1},
    {{-2, 0, 0}, 1},
    {{0, 0, 0}, 1},
    {{-2, -1, -2}, 0},
    {{-1, -1, -2}, 0},
    {{-2, -1, -1}, 0},
    {{-1, -1, -1}, 0},
}};

LiquidRemapInventory3D Affine(double x, double z) {
  return {
      .mass_kg = 64 + 8 * x - 4 * z,
      .reference_volume_m3 = 8 + x,
      .current_volume_m3 = 6 + z,
      .momentum_kg_m_per_s = {-16 + 4 * x, 32 + 2 * z, 2 * x - 8 * z},
      .enthalpy_j = -32 + 4 * z,
  };
}

auto InventoryValues(const LiquidRemapInventory3D& row) {
  return std::array{
      row.mass_kg,
      row.reference_volume_m3,
      row.current_volume_m3,
      row.momentum_kg_m_per_s.x,
      row.momentum_kg_m_per_s.y,
      row.momentum_kg_m_per_s.z,
      row.enthalpy_j,
  };
}

struct FixtureFields {
  std::vector<LiquidRemapInventory3D> inventories;
  std::vector<double> pressures;
  std::vector<double> fluxes;
  LiquidRemapStencil3D stencil{{
      {Affine(-1, 0), Affine(1, 0)},
      {Affine(0, 0), Affine(0, 0)},
      {Affine(0, -1), Affine(0, 1)},
  }};
  std::size_t parent_ordinal = 0;

  AdaptiveLiquidRemapInput3D View() const {
    return {inventories, pressures, fluxes};
  }
};

FixtureFields MakeFields(const AdaptiveMacTopology3D& topology) {
  FixtureFields fields;
  for (std::size_t ordinal = 0; ordinal < topology.cells().size(); ++ordinal) {
    const auto key = topology.cells()[ordinal].key;
    if (key.lower_corner.x == -2 && key.lower_corner.y == 0 && key.lower_corner.z == -2) {
      fields.parent_ordinal = ordinal;
      fields.inventories.push_back(Affine(0, 0));
    } else {
      fields.inventories.push_back(Affine(ordinal * 0.25, -1));
    }
    fields.pressures.push_back(10 + ordinal);
  }
  for (const auto& face : topology.interior_faces()) {
    const auto index = face.face_index.lattice_index;
    if (face.tail_cell_ordinal != fields.parent_ordinal && face.head_cell_ordinal != fields.parent_ordinal) {
      fields.fluxes.push_back(-3.0);
      continue;
    }
    switch (face.face_index.axis) {
      case FaceNormalAxis::kX:
        fields.fluxes.push_back(index.x == -2 ? 4 : -8);
        break;
      case FaceNormalAxis::kY:
        fields.fluxes.push_back(index.x == -2 ? 1 : -1);
        break;
      case FaceNormalAxis::kZ:
        fields.fluxes.push_back(20);
        break;
    }
  }
  return fields;
}

std::vector<double> IntegratedOutflow(const AdaptiveMacTopology3D& topology, std::span<const double> fluxes) {
  std::vector<double> result(topology.cells().size());
  for (std::size_t face = 0; face < fluxes.size(); ++face) {
    const auto& geometry = topology.interior_faces()[face];
    result[geometry.tail_cell_ordinal] += fluxes[face];
    result[geometry.head_cell_ordinal] -= fluxes[face];
  }
  return result;
}

TEST(AdaptiveLiquidRemap3D, RefinesAffineInventoryAndPreservesSignedDivergence) {
  const auto topology = AdaptiveMacTopology3D::Create(kLeaves, 0.5);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto fields = MakeFields(*topology);
  const auto old_outflow = IntegratedOutflow(*topology, fields.fluxes);
  ASSERT_EQ(old_outflow[fields.parent_ordinal], 8);
  const auto refined = PrepareAdaptiveLiquidRefinement3D(*topology, fields.View(), kParent, fields.stencil);
  ASSERT_TRUE(refined.ok()) << refined.status();
  const auto outflow = IntegratedOutflow(refined->family.topology, refined->interior_fluxes_m3_per_s);
  for (std::size_t ordinal : refined->family.child_ordinals) {
    const auto corner = refined->family.topology.cells()[ordinal].key.lower_corner;
    auto expected = InventoryValues(Affine(corner.x == -2 ? -0.25 : 0.25, corner.z == -2 ? -0.25 : 0.25));
    for (double& amount : expected) {
      amount /= 8;
    }
    EXPECT_EQ(InventoryValues(refined->inventories[ordinal]), expected);
    EXPECT_EQ(refined->family.previous_pressures[ordinal], fields.pressures[fields.parent_ordinal]);
    EXPECT_EQ(outflow[ordinal], 1);
  }
  for (const auto retained : refined->family.retained_leaves) {
    EXPECT_EQ(
        InventoryValues(refined->inventories[retained.new_cell_ordinal]),
        InventoryValues(fields.inventories[retained.old_cell_ordinal])
    );
    EXPECT_EQ(
        refined->family.previous_pressures[retained.new_cell_ordinal],
        fields.pressures[retained.old_cell_ordinal]
    );
    EXPECT_EQ(outflow[retained.new_cell_ordinal], old_outflow[retained.old_cell_ordinal]);
  }

  const auto replay = PrepareAdaptiveLiquidRefinement3D(*topology, fields.View(), kParent, fields.stencil);
  ASSERT_TRUE(replay.ok()) << replay.status();
  EXPECT_EQ(replay->family.child_ordinals, refined->family.child_ordinals);
  EXPECT_EQ(replay->family.previous_pressures, refined->family.previous_pressures);
  EXPECT_EQ(replay->interior_fluxes_m3_per_s, refined->interior_fluxes_m3_per_s);

  const AdaptiveLiquidRemapInput3D fine{
      refined->inventories,
      refined->family.previous_pressures,
      refined->interior_fluxes_m3_per_s
  };
  const auto restored = PrepareAdaptiveLiquidCoarsening3D(refined->family.topology, fine, kParent);
  ASSERT_TRUE(restored.ok()) << restored.status();
  EXPECT_EQ(restored->family.previous_pressures, fields.pressures);
  EXPECT_EQ(restored->interior_fluxes_m3_per_s, fields.fluxes);
  ASSERT_EQ(restored->inventories.size(), fields.inventories.size());
  for (std::size_t index = 0; index < fields.inventories.size(); ++index) {
    EXPECT_EQ(InventoryValues(restored->inventories[index]), InventoryValues(fields.inventories[index]));
  }
}

TEST(AdaptiveLiquidRemap3D, ConservesApertureWeightedExteriorFluxAndChildDivergence) {
  constexpr AdaptiveMacCellKey3D kRetained{{-2, 0, 0}, 1};
  const auto topology = AdaptiveMacTopology3D::Create(std::array{kParent, kRetained}, 0.5);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto fields = MakeFields(*topology);
  const auto coarse_ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  ASSERT_TRUE(coarse_ownership.ok()) << coarse_ownership.status();
  const auto fine_family = PrepareAdaptiveRefinement3D(*topology, fields.pressures, kParent);
  ASSERT_TRUE(fine_family.ok()) << fine_family.status();
  const auto fine_ownership = BuildAdaptiveMacFaceOwnership3D(fine_family->topology);
  ASSERT_TRUE(fine_ownership.ok()) << fine_ownership.status();

  std::vector<double> coarse_areas;
  std::vector<double> coarse_outward;
  for (const auto& face : coarse_ownership->exterior_faces) {
    const bool parent_open = face.cell_ordinal == fields.parent_ordinal && face.face_index.axis == FaceNormalAxis::kX;
    const bool retained_open =
        face.cell_ordinal != fields.parent_ordinal && face.face_index.axis == FaceNormalAxis::kZ && face.cell_is_tail;
    coarse_areas.push_back(parent_open ? 0.5 : (retained_open ? 1 : 0));
    coarse_outward.push_back(parent_open ? (face.cell_is_tail ? 8 : -4) : (retained_open ? 5 : 0));
  }
  std::vector<double> fine_areas;
  for (const auto& face : fine_ownership->exterior_faces) {
    if (fine_family->topology.cells()[face.cell_ordinal].key.width_exponent == kRetained.width_exponent) {
      fine_areas.push_back(face.face_index.axis == FaceNormalAxis::kZ && face.cell_is_tail ? face.face_area : 0);
      continue;
    }
    const auto index = face.face_index.lattice_index;
    const auto quadrant = static_cast<std::size_t>((index.z + 2) * 2 + index.y);
    constexpr std::array apertures{0.25, 0.125, 0.125, 0.0};
    fine_areas.push_back(face.face_index.axis == FaceNormalAxis::kX ? apertures[quadrant] : 0);
  }

  auto committed = fields.View();
  committed.exterior = {coarse_areas, coarse_outward, fine_areas};
  const auto refined = PrepareAdaptiveLiquidRefinement3D(*topology, committed, kParent, fields.stencil);
  ASSERT_TRUE(refined.ok()) << refined.status();
  auto child_outflow = IntegratedOutflow(refined->family.topology, refined->interior_fluxes_m3_per_s);
  ASSERT_EQ(refined->exterior_outward_fluxes_m3_per_s.size(), fine_areas.size());
  for (std::size_t exterior = 0; exterior < fine_areas.size(); ++exterior) {
    const auto& face = fine_ownership->exterior_faces[exterior];
    const double outward = refined->exterior_outward_fluxes_m3_per_s[exterior];
    const bool retained = refined->family.topology.cells()[face.cell_ordinal].key.width_exponent == 1;
    const double flux_per_area = retained ? 5 : (face.cell_is_tail ? 16 : -8);
    EXPECT_EQ(outward, fine_areas[exterior] * flux_per_area);
    child_outflow[face.cell_ordinal] += outward;
  }
  for (const auto child : refined->family.child_ordinals) {
    EXPECT_EQ(child_outflow[child], 3);
  }
  ASSERT_EQ(refined->family.retained_leaves.size(), 1u);
  EXPECT_EQ(child_outflow[refined->family.retained_leaves.front().new_cell_ordinal], -15);

  const AdaptiveLiquidRemapInput3D fine{
      .inventories = refined->inventories,
      .previous_pressures_pa = refined->family.previous_pressures,
      .interior_fluxes_m3_per_s = refined->interior_fluxes_m3_per_s,
      .exterior =
          AdaptiveLiquidExteriorRemapInput3D{fine_areas, refined->exterior_outward_fluxes_m3_per_s, coarse_areas},
  };
  const auto coarsened = PrepareAdaptiveLiquidCoarsening3D(refined->family.topology, fine, kParent);
  ASSERT_TRUE(coarsened.ok()) << coarsened.status();
  EXPECT_EQ(coarsened->exterior_outward_fluxes_m3_per_s, coarse_outward);

  const auto original_fluxes = coarse_outward;
  const auto original_pressures = fields.pressures;
  const auto original_inventory = InventoryValues(fields.inventories[fields.parent_ordinal]);
  committed.exterior->candidate_open_areas_m2 = std::span(fine_areas).first(fine_areas.size() - 1);
  EXPECT_EQ(
      PrepareAdaptiveLiquidRefinement3D(*topology, committed, kParent, fields.stencil).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  committed.exterior->candidate_open_areas_m2 = fine_areas;
  const auto low_open_face = std::ranges::find(fine_areas, 0.25);
  ASSERT_NE(low_open_face, fine_areas.end());
  *low_open_face = 0;
  EXPECT_EQ(
      PrepareAdaptiveLiquidRefinement3D(*topology, committed, kParent, fields.stencil).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(coarse_outward, original_fluxes);
  EXPECT_EQ(fields.pressures, original_pressures);
  EXPECT_EQ(InventoryValues(fields.inventories[fields.parent_ordinal]), original_inventory);
  EXPECT_EQ(topology->cells().size(), 2u);
}

TEST(AdaptiveLiquidRemap3D, PreservesRepresentableTinyExteriorShare) {
  const auto topology = AdaptiveMacTopology3D::Create(std::array{kParent}, 0.5);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto fields = MakeFields(*topology);
  const auto coarse_ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  ASSERT_TRUE(coarse_ownership.ok()) << coarse_ownership.status();
  const auto fine_family = PrepareAdaptiveRefinement3D(*topology, fields.pressures, kParent);
  ASSERT_TRUE(fine_family.ok()) << fine_family.status();
  const auto fine_ownership = BuildAdaptiveMacFaceOwnership3D(fine_family->topology);
  ASSERT_TRUE(fine_ownership.ok()) << fine_ownership.status();

  std::vector<double> coarse_apertures(coarse_ownership->exterior_faces.size());
  coarse_apertures.front() = 0.5;
  std::vector<double> coarse_fluxes(coarse_apertures.size());
  coarse_fluxes.front() = 1;
  std::vector<double> fine_apertures(fine_ownership->exterior_faces.size());
  std::vector<double> expected_fluxes(fine_apertures.size());
  constexpr std::array apertures{0x1p-55, 0.25, 0x1.fffffffffffffp-3, 0.0};
  std::size_t subface = 0;
  for (std::size_t index = 0; index < fine_ownership->exterior_faces.size(); ++index) {
    const auto& face = fine_ownership->exterior_faces[index];
    if (face.face_index.axis == FaceNormalAxis::kX && !face.cell_is_tail) {
      ASSERT_LT(subface, apertures.size());
      fine_apertures[index] = apertures[subface++];
      expected_fluxes[index] = 2 * fine_apertures[index];
    }
  }
  ASSERT_EQ(subface, apertures.size());
  auto committed = fields.View();
  committed.exterior = {coarse_apertures, coarse_fluxes, fine_apertures};
  const auto refined = PrepareAdaptiveLiquidRefinement3D(*topology, committed, kParent, fields.stencil);
  ASSERT_TRUE(refined.ok()) << refined.status();
  EXPECT_EQ(refined->exterior_outward_fluxes_m3_per_s, expected_fluxes);

  const AdaptiveLiquidRemapInput3D fine{
      .inventories = refined->inventories,
      .previous_pressures_pa = refined->family.previous_pressures,
      .interior_fluxes_m3_per_s = refined->interior_fluxes_m3_per_s,
      .exterior =
          AdaptiveLiquidExteriorRemapInput3D{
              fine_apertures,
              refined->exterior_outward_fluxes_m3_per_s,
              coarse_apertures,
          },
  };
  const auto restored = PrepareAdaptiveLiquidCoarsening3D(refined->family.topology, fine, kParent);
  ASSERT_TRUE(restored.ok()) << restored.status();
  EXPECT_EQ(restored->exterior_outward_fluxes_m3_per_s, coarse_fluxes);
}

TEST(AdaptiveLiquidRemap3D, ReplaysConstituentInventoryWithoutCrossPhaseTransfer) {
  const auto topology = AdaptiveMacTopology3D::Create(kLeaves, 0.5);
  ASSERT_TRUE(topology.ok()) << topology.status();
  auto fields = MakeFields(*topology);
  const auto constituent_values = [](const LiquidRemapInventory3D& inventory) {
    std::vector<std::pair<std::uint64_t, double>> values;
    for (const auto& constituent : std::span(inventory.constituents).first(inventory.constituent_count)) {
      values.emplace_back(constituent.substance_id, constituent.volume_m3);
    }
    return values;
  };
  const auto make_aqueous = [](LiquidRemapInventory3D& inventory) {
    inventory.phase_id = 4;
    inventory.constituent_count = 2;
    inventory.constituents[0] = {8, inventory.current_volume_m3 / 4};
    inventory.constituents[1] = {14, inventory.current_volume_m3 * 3 / 4};
  };
  const auto make_oil = [](LiquidRemapInventory3D& inventory) {
    inventory.phase_id = 5;
    inventory.constituent_count = 1;
    inventory.constituents = {};
    inventory.constituents[0] = {25, inventory.current_volume_m3};
  };
  for (auto& inventory : fields.inventories) {
    make_oil(inventory);
  }
  for (auto& axis : fields.stencil) {
    for (auto& neighbor : axis) {
      make_aqueous(neighbor);
    }
  }
  make_aqueous(fields.inventories[fields.parent_ordinal]);
  make_oil(fields.stencil[0][0]);

  const auto refined = PrepareAdaptiveLiquidRefinement3D(*topology, fields.View(), kParent, fields.stencil);
  const auto replay = PrepareAdaptiveLiquidRefinement3D(*topology, fields.View(), kParent, fields.stencil);
  ASSERT_TRUE(refined.ok()) << refined.status();
  ASSERT_TRUE(replay.ok()) << replay.status();
  ASSERT_EQ(refined->family.child_ordinals, replay->family.child_ordinals);
  for (const std::size_t ordinal : refined->family.child_ordinals) {
    const auto& child = refined->inventories[ordinal];
    const auto corner = refined->family.topology.cells()[ordinal].key.lower_corner;
    auto expected = InventoryValues(Affine(0, corner.z == -2 ? -0.25 : 0.25));
    for (double& amount : expected) {
      amount /= 8;
    }
    EXPECT_EQ(InventoryValues(child), expected);
    EXPECT_EQ(child.phase_id, 4u);
    ASSERT_EQ(child.constituent_count, 2u);
    EXPECT_EQ(child.constituents[0].substance_id, 8u);
    EXPECT_DOUBLE_EQ(child.constituents[0].volume_m3, child.current_volume_m3 / 4);
    EXPECT_EQ(child.constituents[1].substance_id, 14u);
    EXPECT_DOUBLE_EQ(child.constituents[1].volume_m3, child.current_volume_m3 * 3 / 4);
    EXPECT_EQ(constituent_values(child), constituent_values(replay->inventories[ordinal]));
  }
  for (const auto retained : refined->family.retained_leaves) {
    const auto& inventory = refined->inventories[retained.new_cell_ordinal];
    EXPECT_EQ(inventory.phase_id, 5u);
    EXPECT_EQ(constituent_values(inventory), constituent_values(fields.inventories[retained.old_cell_ordinal]));
  }

  auto candidate_inventories = refined->inventories;
  const AdaptiveLiquidRemapInput3D fine{
      candidate_inventories,
      refined->family.previous_pressures,
      refined->interior_fluxes_m3_per_s,
  };
  const auto restored = PrepareAdaptiveLiquidCoarsening3D(refined->family.topology, fine, kParent);
  ASSERT_TRUE(restored.ok()) << restored.status();
  ASSERT_EQ(restored->inventories.size(), fields.inventories.size());
  for (std::size_t ordinal = 0; ordinal < fields.inventories.size(); ++ordinal) {
    EXPECT_EQ(restored->inventories[ordinal].phase_id, fields.inventories[ordinal].phase_id);
    EXPECT_EQ(constituent_values(restored->inventories[ordinal]), constituent_values(fields.inventories[ordinal]));
  }

  const auto child = refined->family.child_ordinals.front();
  make_oil(candidate_inventories[child]);
  const auto oil = constituent_values(candidate_inventories[child]);
  const auto rejected = PrepareAdaptiveLiquidCoarsening3D(refined->family.topology, fine, kParent);
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(constituent_values(candidate_inventories[child]), oil);
  EXPECT_EQ(refined->inventories[child].phase_id, 4u);
}

TEST(AdaptiveLiquidRemap3D, CoarsensMixedSubfacesAndRejectsLossySums) {
  const auto topology = AdaptiveMacTopology3D::Create(kLeaves, 0.5);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto fields = MakeFields(*topology);
  auto refined = PrepareAdaptiveLiquidRefinement3D(*topology, fields.View(), kParent, fields.stencil);
  ASSERT_TRUE(refined.ok()) << refined.status();
  std::vector<std::size_t> high_x_faces;
  const auto fine_faces = refined->family.topology.interior_faces();
  for (std::size_t ordinal = 0; ordinal < fine_faces.size(); ++ordinal) {
    const auto index = fine_faces[ordinal].face_index.lattice_index;
    if (fine_faces[ordinal].face_index.axis == FaceNormalAxis::kX && index.x == 0 && index.z < 0 && index.y >= 0) {
      high_x_faces.push_back(ordinal);
    }
  }
  ASSERT_EQ(high_x_faces.size(), 4);
  constexpr std::array mixed_fluxes{2.0, -3.0, 1.0, -8.0};
  for (std::size_t slot = 0; slot < high_x_faces.size(); ++slot) {
    refined->interior_fluxes_m3_per_s[high_x_faces[slot]] = mixed_fluxes[slot];
  }
  for (std::size_t child = 0; child < refined->family.child_ordinals.size(); ++child) {
    refined->family.previous_pressures[refined->family.child_ordinals[child]] = 2.0 * child;
  }
  const AdaptiveLiquidRemapInput3D fine{
      refined->inventories,
      refined->family.previous_pressures,
      refined->interior_fluxes_m3_per_s,
  };
  const auto coarsened = PrepareAdaptiveLiquidCoarsening3D(refined->family.topology, fine, kParent);
  ASSERT_TRUE(coarsened.ok()) << coarsened.status();
  EXPECT_EQ(coarsened->interior_fluxes_m3_per_s, fields.fluxes);
  EXPECT_EQ(coarsened->family.previous_pressures[coarsened->family.parent_ordinal], 7);

  constexpr std::array lossy_fluxes{1e16, 0.25, 0.0, 0.0};
  for (std::size_t slot = 0; slot < high_x_faces.size(); ++slot) {
    refined->interior_fluxes_m3_per_s[high_x_faces[slot]] = lossy_fluxes[slot];
  }
  const auto committed = refined->interior_fluxes_m3_per_s;
  EXPECT_EQ(
      PrepareAdaptiveLiquidCoarsening3D(refined->family.topology, fine, kParent).status().code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(refined->interior_fluxes_m3_per_s, committed);
}

TEST(AdaptiveLiquidRemap3D, RejectsLossyCandidateWithoutMutatingCommittedFields) {
  const auto topology = AdaptiveMacTopology3D::Create(kLeaves, 0.5);
  ASSERT_TRUE(topology.ok()) << topology.status();
  auto fields = MakeFields(*topology);
  const auto initial_pressures = fields.pressures;
  const auto initial_fluxes = fields.fluxes;
  const double quantum = std::numeric_limits<double>::denorm_min();
  fields.inventories[fields.parent_ordinal].mass_kg = quantum;
  EXPECT_EQ(
      PrepareAdaptiveLiquidRefinement3D(*topology, fields.View(), kParent, fields.stencil).status().code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(fields.inventories[fields.parent_ordinal].mass_kg, quantum);
  EXPECT_EQ(fields.pressures, initial_pressures);
  EXPECT_EQ(fields.fluxes, initial_fluxes);

  fields.inventories[fields.parent_ordinal] = Affine(0, 0);
  for (std::size_t face = 0; face < fields.fluxes.size(); ++face) {
    if (topology->interior_faces()[face].face_index.axis == FaceNormalAxis::kX && fields.fluxes[face] == 4) {
      fields.fluxes[face] = quantum;
    }
  }
  const auto lossy_fluxes = fields.fluxes;
  EXPECT_EQ(
      PrepareAdaptiveLiquidRefinement3D(*topology, fields.View(), kParent, fields.stencil).status().code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(fields.fluxes, lossy_fluxes);
  EXPECT_EQ(fields.pressures, initial_pressures);
  EXPECT_EQ(topology->cells().size(), kLeaves.size());
  auto missing_flux = fields.View();
  missing_flux.interior_fluxes_m3_per_s = missing_flux.interior_fluxes_m3_per_s.first(1);
  EXPECT_EQ(
      PrepareAdaptiveLiquidRefinement3D(*topology, missing_flux, kParent, fields.stencil).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

}  // namespace
}  // namespace fluidity
