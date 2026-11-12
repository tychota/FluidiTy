#pragma once

#include <cstdint>
#include <span>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity::liquid_step_internal {

// Proposes one aligned family change for the stationary planar tank. Copies
// the committed clock, IDs and particle records, remaps pressure and signed
// volumetric flux in canonical order. Particle-owned mass, volumes,
// momentum and one miscible phase's constituent volumes must exactly
// match the remapped inventory in each leaf.
// FailedPrecondition denotes a change that requires certified resampling or
// incompatible face history. Capacity and checked topology errors propagate.
// The input remains unchanged on every result.
absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidFamilyCandidate3D(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    AdaptiveMacCellKey3D parent,
    bool refine
);

// Stage a closure by remapping coarser neighbors first. The topology belongs
// to the committed state and must use its canonical leaf keys and order.
absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidRefinementClosureCandidate3D(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    const AdaptiveMacTopology3D& topology,
    std::span<const AdaptiveMacCellKey3D> requested_parents
);

// Refine a graded closure and replace one parent particle by its certified
// quadratic children. The selected family must be the final closure edit.
// Every child and remapped extensive leaf lane must be representable before
// returning a detached state. Exact closure requires all implied edits once.
// No record or ID in committed is modified.
absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidQuadraticSplitCandidate3D(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    std::span<const AdaptiveMacCellKey3D> requested_parents,
    AdaptiveMacCellKey3D split_parent,
    std::uint64_t split_particle_id,
    bool require_exact_closure = false
);

// Coarsen the requested full families and replace the selected fine roster
// by its certified quadratic F4 particle. The selected family must be the
// final edit. IDs must identify two to eight distinct committed particles in
// that family. Exact closure forbids duplicate family requests. The source,
// including its next ID, is read-only on all results.
absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidQuadraticMergeCandidate3D(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    std::span<const AdaptiveMacCellKey3D> requested_parents,
    AdaptiveMacCellKey3D merge_parent,
    std::span<const std::uint64_t> children,
    bool require_exact_closure = false
);

}  // namespace fluidity::liquid_step_internal
