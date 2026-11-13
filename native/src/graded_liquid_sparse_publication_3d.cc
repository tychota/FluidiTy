#include "graded_liquid_sparse_publication_3d.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

namespace fluidity {
namespace {

bool SameIndex(CellIndex3D left, CellIndex3D right) {
  return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool Overlaps(std::span<const std::byte> left, std::span<const std::byte> right) {
  if (left.empty() || right.empty()) {
    return false;
  }
  const std::less<const std::byte*> before;
  return before(left.data(), right.data() + right.size()) && before(right.data(), left.data() + left.size());
}

bool EmptyInventory(const LiquidAdaptiveCellState3D& leaf) {
  auto empty = leaf;
  empty.inventory = {};
  return leaf == empty;
}

// Build and validate the inverse row map before using any graph incidence.
absl::Status CheckRows(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    const GradedWetPressureSubset3D& subset,
    std::span<const LiquidAdaptiveCellState3D> leaves,
    std::span<std::optional<std::size_t>> row_by_leaf
) {
  const auto pressure_cells = subset.system.cells();
  std::size_t row = 0;
  for (std::size_t leaf = 0; leaf < leaves.size(); ++leaf) {
    const auto& canonical = geometry.topology().cells()[leaf];
    const auto& measured = geometry.leaves()[leaf];
    const auto& record = leaves[leaf];
    if (!SameIndex(record.key.lower_corner, canonical.key.lower_corner)
        || record.key.width_exponent != canonical.key.width_exponent) {
      return absl::InvalidArgumentError("Sparse pressure publication leaf key differs from frozen topology");
    }
    if (measured.fluid_volume_m3 == 0) {
      if (!measured.known_air || record.occupancy != LiquidCellOccupancy3D::kAir || !EmptyInventory(record)) {
        return absl::FailedPreconditionError("Sparse pressure publication requires certified empty omitted leaves");
      }
      continue;
    }
    if (record.occupancy != LiquidCellOccupancy3D::kLiquid) {
      return absl::FailedPreconditionError("Sparse pressure publication requires liquid in every pressure row");
    }
    if (row >= pressure_cells.size() || subset.leaf_ordinals[row] != leaf
        || !SameIndex(pressure_cells[row].cell_index, canonical.key.lower_corner)
        || pressure_cells[row].fluid_volume != std::min(measured.fluid_volume_m3, canonical.cell_volume)) {
      return absl::InvalidArgumentError("Sparse pressure publication row mapping differs from frozen wet leaves");
    }
    row_by_leaf[leaf] = row++;
  }
  if (row != pressure_cells.size()) {
    return absl::InvalidArgumentError("Sparse pressure publication has extraneous pressure rows");
  }
  return absl::OkStatus();
}

// Walk canonical slots even when a slot has no pressure link. This detects
// skipped wet links, inserted air links, and ordinal/slot confusion.
absl::Status CheckLinks(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const pressure_internal::FrozenMacPressureLink3D> links,
    std::span<const std::optional<std::size_t>> row_by_leaf,
    std::span<const LiquidAdaptiveFaceState3D> faces
) {
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(geometry.topology());
  if (!ownership.ok()) {
    return ownership.status();
  }
  const auto interior_faces = geometry.topology().interior_faces();
  if (interior_faces.size() + ownership->exterior_faces.size() != faces.size()) {
    return absl::InvalidArgumentError("Sparse pressure publication face roster differs from frozen topology");
  }
  std::size_t interior = 0;
  std::size_t exterior = 0;
  std::size_t link_ordinal = 0;
  for (std::size_t slot = 0; slot < faces.size(); ++slot) {
    const auto& predictor = faces[slot];
    if (predictor.validity > LiquidFaceValidity3D::kLowMass || !std::isfinite(predictor.carrier_velocity)) {
      return absl::InvalidArgumentError("Sparse pressure publication received an invalid predictor record");
    }
    std::optional<std::size_t> tail;
    std::optional<std::size_t> head;
    const bool is_interior = interior < interior_faces.size() && ownership->interior_face_slots[interior] == slot;
    double full_area_m2 = 0;
    if (is_interior) {
      const auto& face = interior_faces[interior++];
      tail = row_by_leaf[face.tail_cell_ordinal];
      head = row_by_leaf[face.head_cell_ordinal];
      full_area_m2 = face.face_area;
    } else {
      const auto& face = ownership->exterior_faces[exterior++];
      auto& row = face.cell_is_tail ? tail : head;
      row = row_by_leaf[face.cell_ordinal];
      full_area_m2 = face.face_area;
    }
    if (!tail && !head) {
      continue;
    }
    if (link_ordinal >= links.size()) {
      return absl::InvalidArgumentError("Sparse pressure publication is missing a wet incident link");
    }
    const auto& link = links[link_ordinal++];
    const auto& measured = geometry.subfaces()[slot];
    const double wet_open_area_m2 = std::min(measured.wet_open_area_m2, full_area_m2);
    const bool must_be_sealed = is_interior && (!tail || !head);
    const bool aperture_matches = link.open_area == wet_open_area_m2 || (!is_interior && link.open_area == 0);
    if (link.face_slot != slot || link.tail_cell_ordinal != tail || link.head_cell_ordinal != head
        || link.prescribed_wall_flux != measured.wall_normal_flux_m3_per_s || !aperture_matches
        || (must_be_sealed && wet_open_area_m2 != 0)) {
      return absl::InvalidArgumentError(
          "Sparse pressure publication link differs from frozen slot incidence or measures"
      );
    }
    if (link.open_area > 0
        && (!predictor.structurally_eligible || predictor.validity == LiquidFaceValidity3D::kMissing
            || predictor.carrier_velocity != link.predicted_normal_velocity)) {
      return absl::FailedPreconditionError("Sparse pressure publication requires a matching open predictor carrier");
    }
    if (link.open_area == 0 && wet_open_area_m2 > 0
        && (predictor.carrier_velocity != 0
            || (predictor.structurally_eligible && predictor.validity == LiquidFaceValidity3D::kMissing))) {
      return absl::FailedPreconditionError("Sparse pressure publication requires a qualified zero sealed carrier");
    }
  }
  if (link_ordinal != links.size()) {
    return absl::InvalidArgumentError("Sparse pressure publication has extraneous pressure links");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status PublishGradedSparsePressure3D(
    LiquidState3D& trial,
    LiquidClock3D expected_trial_clock,
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    const GradedWetPressureSubset3D& subset,
    const absl::StatusOr<GradedSparsePressureResultView3D>& projection
) {
  if (!projection.ok()) {
    return projection.status();
  }
  if (trial.clock != expected_trial_clock) {
    return absl::FailedPreconditionError("Sparse pressure publication trial clock is stale");
  }
  auto leaves = trial.adaptive_cells.values();
  auto faces = trial.adaptive_faces.values();
  const auto pressure_cells = subset.system.cells();
  const auto links = subset.system.links();
  const auto pressures = projection->cell_pressures;
  const auto velocities = projection->corrected_face_velocities;
  if (!subset.system.is_explicit_graph() || pressure_cells.empty() || leaves.size() != geometry.leaves().size()
      || leaves.size() != geometry.topology().cells().size() || faces.size() != geometry.subfaces().size()
      || subset.leaf_ordinals.size() != pressure_cells.size() || pressures.size() != pressure_cells.size()
      || velocities.size() != links.size()) {
    return absl::InvalidArgumentError("Sparse pressure publication extent differs from frozen topology or graph");
  }
  const auto origin = geometry.world_origin_m();
  const auto trial_origin = trial.geometry.world_origin;
  if (trial.geometry.cell_width != geometry.topology().finest_cell_width() || trial_origin.x != origin.x
      || trial_origin.y != origin.y || trial_origin.z != origin.z) {
    return absl::InvalidArgumentError("Sparse pressure publication lattice differs from frozen geometry");
  }
  for (const auto input : {std::as_bytes(pressures), std::as_bytes(velocities)}) {
    if (Overlaps(input, std::as_bytes(leaves)) || Overlaps(input, std::as_bytes(faces))) {
      return absl::InvalidArgumentError("Sparse pressure publication inputs overlap mutable trial records");
    }
  }
  const auto finite = [](double value) {
    return std::isfinite(value);
  };
  if (!std::ranges::all_of(pressures, finite) || !std::ranges::all_of(velocities, finite)) {
    return absl::InvalidArgumentError("Sparse pressure publication result is nonfinite");
  }
  std::vector<std::optional<std::size_t>> row_by_leaf(leaves.size());
  if (const auto status = CheckRows(geometry, subset, leaves, row_by_leaf); !status.ok()) {
    return status;
  }
  if (const auto status = CheckLinks(geometry, links, row_by_leaf, faces); !status.ok()) {
    return status;
  }

  for (std::size_t leaf = 0; leaf < leaves.size(); ++leaf) {
    const auto row = row_by_leaf[leaf];
    leaves[leaf].previous_pressure = row ? pressures[*row] : 0;
  }
  std::size_t link_ordinal = 0;
  for (std::size_t slot = 0; slot < faces.size(); ++slot) {
    auto& face = faces[slot];
    if (link_ordinal < links.size() && links[link_ordinal].face_slot == slot) {
      const std::size_t current_link = link_ordinal++;
      if (links[current_link].open_area > 0) {
        face.carrier_velocity = velocities[current_link];
        continue;
      }
      if (geometry.subfaces()[slot].wet_open_area_m2 > 0) {
        continue;
      }
    }
    face.carrier_velocity = 0;
    face.validity = LiquidFaceValidity3D::kMissing;
  }
  return absl::OkStatus();
}

}  // namespace fluidity
