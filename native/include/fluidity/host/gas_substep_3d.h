#pragma once

#include <span>
#include <vector>

#include "absl/functional/function_ref.h"
#include "fluidity/host/gas_transport.h"
#include "fluidity/host/mac_pressure_3d.h"

namespace fluidity {

struct GasCell3D {
  CellIndex3D cell_index;
  double volume;  // Fixed gas volume in m³.
  GasScalars density;
  double absolute_pressure;  // Pa, identical throughout a sealed component.
  double gauge_pressure = 0.0;  // Previous projected gauge pressure in Pa.
};

struct GasCellSource3D {
  GasScalars rates;  // Signed density sources per second. Includes external and net conductive heat.
  double burn_rate = 0.0;  // First-order rate in s⁻¹, zero when unignited.
  double heat_per_fuel = 0.0;  // J/kg of consumed fuel.
  double soot_yield = 0.0;  // kg of soot per kg of consumed fuel.
};

struct GasSubstep3D {
  std::vector<GasCell3D> cells;
  MacPressureProjection3D projection;
  GasScalars outward_rate;  // Signed boundary budgets in kg/s and J/s.
  std::vector<double> fuel_burn_rate;  // kg/(m³ s) in cell order, available for flame visualization.
};

// Advances fixed gas cells on the owned uniform MAC faces. Cell and face
// ordering follows MacPressureInputs3D. Faces supply apertures, distances,
// boundary types and predicted m/s before buoyancy. Walls must be stationary
// and pressure anchors use zero gauge. Face mobility is derived from carrier
// density, never from marker weights. Ambient scalars supply inflow and
// ambient temperature. Gravity is a world vector in m/s².
// Returns InvalidArgument for malformed inputs, FailedPrecondition for
// incompatible topology or rejected scalar transport, and OutOfRange for
// nonrepresentable updates. Pressure backend failures propagate without
// publishing a result. The callback consumes the same input as the CPU solve.
absl::StatusOr<GasSubstep3D> AdvanceGasSubstep3D(
    const SparseMacGrid3D& grid,
    double step_duration,
    std::span<const GasCell3D> cells,
    std::span<const MacPressureFace3D> faces,
    std::span<const GasCellSource3D> sources,
    GasScalars ambient,
    double ambient_pressure,
    Vec3 gravity,
    absl::FunctionRef<absl::StatusOr<MacPressureProjection3D>(const MacPressureInputs3D&, MacPressureSolverOptions3D)>
        project = ProjectMacPressure3D
);

}  // namespace fluidity
