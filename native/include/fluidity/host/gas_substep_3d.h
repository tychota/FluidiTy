#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
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

struct GasApicMarker3D {
  Vec3 world_position;  // Meters.
  double weight;  // Positive transfer weight, independent of physical carrier mass.
  Vec3 velocity;  // World m/s.
  Mat3 velocity_slopes;  // APIC affine rows in s⁻¹.
};

struct GasApicSubstep3D {
  GasSubstep3D gas;
  std::vector<GasApicMarker3D> markers;
  std::vector<double> face_velocities;  // Projected field including prior values at ghost slots.
};

// Deposits and gathers APIC markers around the fixed-grid gas step. The old
// owned-slot field supplies velocities at slots without marker support and
// at ghosts. Optional accelerations (m/s²) are in pressure-face order.
// Markers begin inside a gas cell and need complete owned quadratic support.
// Each axis moves at most half a cell per step. A stationary sealed face
// stops normal marker motion and an open exterior face removes the marker.
// Errors propagate without changing any inputs or publishing an update.
absl::StatusOr<GasApicSubstep3D> AdvanceGasApicSubstep3D(
    const SparseMacGrid3D& grid,
    double step_duration,
    std::span<const GasCell3D> cells,
    std::span<const MacPressureFace3D> faces,
    std::span<const GasCellSource3D> sources,
    GasScalars ambient,
    double ambient_pressure,
    Vec3 gravity,
    std::span<const GasApicMarker3D> markers,
    std::span<const double> old_face_velocities,
    std::span<const double> face_accelerations = {},
    absl::FunctionRef<absl::StatusOr<MacPressureProjection3D>(const MacPressureInputs3D&, MacPressureSolverOptions3D)>
        project = ProjectMacPressure3D
);

struct GasUniformDomainOptions3D {
  UniformMacGeometry3D geometry;
  std::array<std::int32_t, 3> extent;  // Cell counts in x, y, z order, starting at lattice zero.
  std::array<bool, 6> open_boundary;  // Low/high x, low/high y, low/high z.
  GasScalars ambient;  // Initial cell and inflow densities.
  double ambient_pressure;  // Absolute Pa.
  Vec3 initial_velocity;  // World m/s before enforcing stationary walls.
};

struct GasUniformDomain3D {
  SparseMacGrid3D grid;
  std::vector<GasCell3D> cells;
  std::vector<MacPressureFace3D> faces;
  std::vector<GasApicMarker3D> markers;
  std::vector<double> face_velocities;  // All owned slots, including the transfer halo.
};

// Constructs a rectangular fixed domain with the owned quadratic transfer halo,
// unit-weight markers at cell centers, and zero-gauge open boundaries. Cells and
// faces are in pressure solver order. Invalid extents, ambient state or velocity
// return InvalidArgument, domains over 4096 cells return ResourceExhausted, and
// unrepresentable geometry returns OutOfRange. Errors carry no domain.
absl::StatusOr<GasUniformDomain3D> CreateGasUniformDomain3D(const GasUniformDomainOptions3D& options);

inline constexpr double kGasSpecificHeat = 3.5 * 287.05;  // Dry air cp in J/(kg K).

struct GasSourceKey3D {
  double time;  // Seconds from frame zero.
  double fuel_rate;  // kg/(m³ s).
  double heat_rate;  // W/m³.
};

struct GasBoxSource3D {
  Vec3 lower;  // Inclusive world-space bounds in meters, evaluated at cell centers.
  Vec3 upper;  // Exclusive world-space bounds in meters.
  std::span<const GasSourceKey3D> keys;  // Borrowed, in strictly increasing time order.
};

struct GasDraftKey3D {
  double time;  // Seconds from frame zero.
  Vec3 acceleration;  // World-space uniform field in m/s².
};

struct GasReaction3D {
  double ignition_temperature = 300.0;  // K.
  double burn_rate = 0.0;  // s⁻¹ above the ignition threshold.
  double heat_per_fuel = 0.0;  // J/kg of consumed fuel.
  double soot_yield = 0.0;  // kg/kg of consumed fuel.
};

struct GasFrame3D {
  std::vector<GasCell3D> cells;
  std::vector<GasApicMarker3D> markers;
  std::vector<double> face_velocities;
  std::vector<double> fuel_burn_rate;  // Final substep flame in kg/(m³ s), in cell order.
};

// Advances a fixed gas domain over [start_time, end_time] in seconds. Sources
// and the draft use linear interpolation with endpoint hold at the midpoint of
// each of 1–4096 equal steps. Overlapping source rates add. The input domain is
// borrowed and remains valid on any failure. Invalid settings return
// InvalidArgument, and rejected solver steps propagate their status.
absl::StatusOr<GasFrame3D> AdvanceGasFrame3D(
    const GasUniformDomain3D& domain,
    GasScalars ambient,
    double ambient_pressure,
    Vec3 gravity,
    double start_time,
    double end_time,
    std::size_t substeps,
    std::span<const GasBoxSource3D> emitters,
    std::span<const GasDraftKey3D> draft,
    GasReaction3D reaction,
    absl::FunctionRef<absl::StatusOr<MacPressureProjection3D>(const MacPressureInputs3D&, MacPressureSolverOptions3D)>
        project = ProjectMacPressure3D
);

}  // namespace fluidity
