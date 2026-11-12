#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <span>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_air_components.h"

namespace fluidity {

struct LiquidCavityInventory3D {
  double initial_volume;  // Positive m³.
  double initial_absolute_pressure;  // Positive Pa.
};

struct LiquidCavityInventoryShare3D {
  LiquidCavityInventory3D parent;
  double fraction;  // Positive fraction of the parent's gas inventory, at most one.
};

// Combines shares in caller order into one equivalent polytropic inventory.
// V_initial * p_initial^(1/1.4) is additive for this topology transfer.
// The caller assigns fractions summing to one across the retained children
// of each parent, accounting separately for any discarded share. Empty input
// and invalid shares return InvalidArgument. An unrepresentable reference
// volume returns OutOfRange.
absl::StatusOr<LiquidCavityInventory3D> CombineLiquidCavityInventoryShares3D(
    std::span<const LiquidCavityInventoryShare3D> shares
);

struct LiquidCavityPressureState3D {
  double absolute_pressure;  // Pa.
  double gas_gauge_pressure;  // Pa relative to atmosphere, before the capillary lift.
  double boundary_gauge_pressure;  // Pa, including the signed liquid-side Laplace jump.
  double equivalent_radius;  // m.
  double maximum_step_duration;  // s, from the radial cavity frequency.
  bool resolved;  // Equivalent radius is at least twice the cell width.
};

// Initializes the absolute inventory of a submerged spherical cavity in
// mechanical equilibrium. Volume is positive m³, atmospheric absolute
// pressure is positive Pa, hydrostatic gauge pressure is nonnegative Pa,
// and surface tension is nonnegative N/m. Nonfinite or invalid inputs return
// InvalidArgument. Unrepresentable inventory returns OutOfRange.
absl::StatusOr<LiquidCavityInventory3D> InitializeSubmergedLiquidCavity3D(
    double volume,
    double atmospheric_absolute_pressure,
    double hydrostatic_gauge_pressure,
    double surface_tension
);

// Evaluates a fixed inventory with polytropic exponent 1.4. Volume is positive
// m³, atmospheric pressure is positive absolute Pa, liquid density is positive
// kg/m³, surface tension is nonnegative N/m, curvature is signed outward from
// liquid in m⁻¹, and cell width is positive m. A spherical cavity has negative
// liquid curvature. The result is a single boundary pressure lift. Invalid
// inputs return InvalidArgument and unrepresentable results return OutOfRange.
absl::StatusOr<LiquidCavityPressureState3D> EvaluateLiquidCavity3D(
    LiquidCavityInventory3D inventory,
    double volume,
    double atmospheric_absolute_pressure,
    double liquid_density,
    double surface_tension,
    double liquid_curvature,
    double cell_width
);

struct LiquidCavityLineageState3D {
  std::uint64_t id = 0;  // Zero when there is no inventory.
  std::optional<LiquidCavityInventory3D> inventory;
};

struct LiquidCavityLineageCandidate3D {
  LiquidCavityLineageState3D state;
  double inherited_id_overlap = 0.0;  // m³ of overlap that nominated the inherited ID.
};

struct LiquidCavityLineageResult3D {
  std::uint64_t next_id;
  double vented_inventory = 0.0;  // m³ Pa^(1/1.4).
  double subresolution_inventory = 0.0;  // m³ Pa^(1/1.4).
  double unmatched_inventory = 0.0;  // m³ Pa^(1/1.4).
};

struct LiquidCavityLineageInputs3D {
  std::span<const LiquidCavityLineageState3D> previous;  // Indexed by previous component, with unique nonzero IDs.
  std::span<const LiquidAirComponent3D> current;
  std::span<const LiquidAirComponentOverlap3D> overlaps;  // Unique edges in (previous,current) order.
  std::span<const std::optional<LiquidCavityInventory3D>> initial_inventories;  // One per current component.
  std::uint64_t next_id;  // First available ID, larger than every previous ID.
  double cell_width;  // m.
};

// Prepares a candidate lineage for the next accepted step. Each tracked parent
// apportions its inventory by its outgoing overlap weights. Its largest
// resolved enclosed recipient may inherit the ID. At a merge, the strongest
// nomination wins. Ties use canonical child order or the lower parent ID.
// New resolved enclosed cavities with no inherited inventory require an
// explicit initialization. Unknown recipients of tracked inventory return
// FailedPrecondition. Lost inventory is reported by cause. The caller supplies
// disjoint candidate scratch, which may change on error, and commits it only
// after accepting the step. InvalidArgument covers invalid inputs and aliasing,
// and OutOfRange covers inventory arithmetic and ID exhaustion.
absl::StatusOr<LiquidCavityLineageResult3D> PrepareLiquidCavityLineage3D(
    const LiquidCavityLineageInputs3D& inputs,
    std::span<LiquidCavityLineageCandidate3D> candidates
);

struct LiquidCavityBoundaryInputs3D {
  std::span<const LiquidAirVolumeCell3D> gas_cells;  // Sorted gas nodes from the sampled domain.
  std::span<const std::size_t> labels;  // Component index for each gas cell.
  std::span<const LiquidAirComponent3D> components;
  std::span<const std::optional<LiquidCavityInventory3D>> inventories;  // Matched by component.
  std::span<const LiquidCenterDistance3D> center_distances;  // Frozen pressure sample order (z,y,x).
  double atmospheric_absolute_pressure;  // Pa.
  double liquid_density;  // kg/m³.
  double cell_width;  // m.
};

struct LiquidCavityBoundaryComponent3D {
  bool active = false;
  double gas_gauge_pressure = 0.0;  // Pa before capillarity.
};

struct LiquidCavityBoundaryPreparation3D {
  std::size_t air_pressure_count = 0;
  std::size_t active_cavity_count = 0;
  std::size_t subresolution_inventory_count = 0;
  double maximum_step_duration = std::numeric_limits<double>::infinity();  // s.
};

// Prepares the gas-only pressure lift and minimum radial step bound for
// enclosed, resolved components with matched inventories. Unclassified
// and vented components supply no override. Emits only air centers present
// in the frozen pressure samples. Scratch spans are disjoint from all inputs
// and each other. InvalidArgument covers invalid inputs, ResourceExhausted
// covers insufficient pressure output, and cavity arithmetic errors propagate.
// Scratch may change on failure. No partially prepared result is returned.
absl::StatusOr<LiquidCavityBoundaryPreparation3D> PrepareLiquidCavityBoundaries3D(
    const LiquidCavityBoundaryInputs3D& inputs,
    std::span<LiquidCavityBoundaryComponent3D> component_scratch,
    std::span<LiquidAirCellPressure3D> air_pressures
);

struct LiquidCavityStepInputs3D {
  UniformMacGeometry3D geometry;
  const LiquidObstacle3D& obstacle;
  std::span<const CellIndex3D> sampled_cells;
  std::span<const LiquidCornerDistance3D> corners;
  std::span<const LiquidCenterDistance3D> centers;
  std::span<const FaceIndex3D> atmosphere_outlets;
  LiquidAirComponentSnapshot3D previous_snapshot;
  std::span<const LiquidCavityLineageState3D> previous_lineage;
  std::uint64_t next_id;
  double atmospheric_absolute_pressure;  // Pa.
  double liquid_density;  // kg/m³.
  LiquidAirExterior3D exterior = LiquidAirExterior3D::kExplicitOutlets;
  void* initialization_context = nullptr;
  // Called only for newly detected resolved enclosed components without tracked overlap.
  absl::StatusOr<LiquidCavityInventory3D> (*initialize)(void*, const LiquidAirComponent3D&) = nullptr;
};

struct LiquidCavityStepScratch3D {
  std::span<LiquidAirVolumeCell3D> gas_cells;
  std::span<std::size_t> labels;
  std::span<std::size_t> queue;
  std::span<LiquidAirComponent3D> components;
  std::span<LiquidAirComponentOverlap3D> overlaps;
  std::span<std::optional<LiquidCavityInventory3D>> inventories;
  std::span<LiquidCavityLineageCandidate3D> lineage;
  std::span<LiquidCavityBoundaryComponent3D> boundary_components;
  std::span<LiquidAirCellPressure3D> air_pressures;
};

struct LiquidCavityStepPreparation3D {
  std::size_t gas_cell_count;
  std::size_t component_count;
  LiquidCavityLineageResult3D lineage;
  LiquidCavityBoundaryPreparation3D boundaries;
};

// Prepares one candidate from sampled geometry and the prior accepted
// snapshot. Sample and outlet ordering follows BuildLiquidAirDomain3D.
// The caller commits the populated gas, labels and lineage only on step
// acceptance. Scratch is mutually disjoint and disjoint from input storage.
// Scratch may change on failure. InvalidArgument covers invalid extents or
// aliasing, ResourceExhausted covers short scratch, and stage errors propagate.
absl::StatusOr<LiquidCavityStepPreparation3D>
PrepareLiquidCavityStep3D(const LiquidCavityStepInputs3D& inputs, const LiquidCavityStepScratch3D& scratch);

}  // namespace fluidity
