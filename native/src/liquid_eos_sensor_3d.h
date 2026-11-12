#pragma once

#include <span>

#include "fluidity/host/liquid_boundary_geometry_3d.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity::liquid_step_internal {

struct CellVolumeRatioSensor3D {
  double sampled_reference_volume = 0.0;  // Sum of tent-weighted reference volumes in m³.
  double mean_volume_ratio = 0.0;  // Meaningful only when sampled_reference_volume is positive.
};

// Scatters an eight-point cell-centered linear tent from each same-side
// particle in input order to selected cells. Cells are ordered by (z,y,x)
// and geometry and particle records have already been validated. The scratch
// span matches cell count and is disjoint from inputs. A cell with no donor
// has zero sampled reference volume. Invalid extents return InvalidArgument;
// unrepresentable contributions return OutOfRange. Scratch may change on error.
absl::Status SampleLiquidCellVolumeRatios3D(
    UniformMacGeometry3D geometry,
    std::span<const FrozenLiquidCell3D> cells,
    std::span<const LiquidParticleState3D> particles,
    std::int64_t liquid_side_tag,
    std::span<CellVolumeRatioSensor3D> sensors
);

// Computes the clamped source in s⁻¹ from a finite target log J. A cell with
// no sensor returns zero. Correction time is positive seconds and the maximum
// rate is finite nonnegative s⁻¹. Invalid inputs return InvalidArgument.
absl::StatusOr<double> LiquidVolumeDriftRate3D(
    CellVolumeRatioSensor3D sensor,
    double target_log_volume_ratio,
    double correction_time,
    double maximum_rate
);

struct SealedLiquidDriftCell3D {
  std::size_t cell_ordinal;  // Caller identity, retained across sorting.
  double fluid_volume;  // Positive m³.
  double raw_rate;  // Unclamped drift in s⁻¹.
  double corrected_rate = 0.0;  // Balanced, clamped drift in s⁻¹.
};

// Balances one sealed component's sensed cells by a common shift followed
// by clamping to ±maximum_rate. Core cells and physical sources are omitted.
// Sorts the scratch by raw rate and cell ordinal. Invalid inputs return
// InvalidArgument. An unrepresentable balance returns OutOfRange and may
// reorder scratch, with corrected rates unchanged.
absl::Status BalanceSealedLiquidDrift3D(std::span<SealedLiquidDriftCell3D> sensed_cells, double maximum_rate);

}  // namespace fluidity::liquid_step_internal
