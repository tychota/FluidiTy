#pragma once

#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity {

// Advects implicit single-phase inventory between liquid, Eulerian-owned
// leaves. Cell records and integrated subface fluxes in m³/s cover the topology
// in canonical cell and interior face order. Positive flux goes from tail to
// head. Supply a finite positive duration in seconds. The signed swept volume
// on each subface transfers the corresponding fraction of step-start donor
// mass, reference volume, momentum, and signed enthalpy. Gross donor outflow
// is limited by its step-start current volume. Final current volume is limited
// by geometric leaf capacity. The final face of an exact drain carries the
// donor's rounded remainder. Conservation is subject to binary64 roundoff.
//
// Exterior fluxes are outside this operation. Interior faces wholly outside
// the Eulerian liquid region are ignored. Nonzero flux between the region
// and any other leaf, explicit phase inventory in the region, gross overdraw,
// and final overfill return FailedPrecondition. Invalid dimensions, keys,
// metadata, or numerical inputs return InvalidArgument. Unrepresentable
// arithmetic returns OutOfRange. Returns independent candidate records with
// updated inventories on success. All inputs are read-only on every path.
absl::StatusOr<std::vector<LiquidAdaptiveCellState3D>> PrepareAdaptiveLiquidCoreTransport3D(
    const AdaptiveMacTopology3D& topology,
    std::span<const LiquidAdaptiveCellState3D> committed_cells,
    std::span<const double> interior_fluxes_m3_per_s,
    double step_duration_s
);

}  // namespace fluidity
