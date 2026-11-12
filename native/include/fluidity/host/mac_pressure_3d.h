#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/sparse_mac_grid_3d.h"

namespace fluidity {

struct MacPressureCell3D {
  CellIndex3D cell_index;
  double fluid_volume;  // Positive fluid volume in m³, at most the grid cell volume.
  double target_divergence;  // Prescribed divergence in s⁻¹.
  double bulk_modulus;  // Positive Pa. Positive infinity selects zero compliance.
  double previous_pressure;  // Pressure before the step in Pa.
  // Flux through a solid boundary inside the cell, positive into the liquid.
  // In m³/s. Excludes any contribution already represented by an incident MAC face.
  double integrated_inward_wall_flux = 0.0;
};

// The jump in Pa enters the pressure correction as p_tail − p_head + pressure_jump.
struct InternalMacPressureLink3D {
  double pressure_jump;
};

struct DirichletMacPressureLink3D {
  double exterior_pressure;  // Pressure in Pa at the single missing endpoint.
};

struct SealedMacPressureLink3D {};

struct MacPressureFace3D {
  FaceIndex3D face_index;  // The lattice index names the head; the normal points from tail to head.
  double open_area;  // Fluid aperture in m², between zero and the full face area.
  double inverse_density;  // Face mobility in m³/kg.
  double pressure_gradient_distance;  // Center-to-center or center-to-boundary distance in meters.
  double predicted_normal_velocity;  // Fluid velocity along the positive normal in m/s.
  double wall_normal_velocity;  // Wall velocity along the positive normal in m/s.
  // Internal faces have two pressure endpoints and may have zero area.
  // Dirichlet faces have one pressure endpoint and positive area. Sealed
  // faces have zero area. At zero area, mobility and distance need only be finite.
  std::variant<InternalMacPressureLink3D, DirichletMacPressureLink3D, SealedMacPressureLink3D> pressure_link;
  // Optional integrated wall flux in m³/s along the positive face normal.
  // If absent, the entire blocked area moves at wall_normal_velocity. An
  // explicit flux must be finite and must be zero on a fully open face.
  std::optional<double> prescribed_wall_flux = std::nullopt;
};

// Borrowed read-only inputs. Supply 1–4096 distinct pressure cells in (z,y,x)
// order and their exact union of incident faces (at most 24576) in owned slot
// order. Even a sealed face must have an allocated owner.
struct MacPressureInputs3D {
  const SparseMacGrid3D& mac_grid;
  double step_duration;  // Positive finite time step in seconds.
  std::span<const MacPressureCell3D> pressure_cells;
  std::span<const MacPressureFace3D> pressure_faces;
};

struct MacPressureDiagnostics3D {
  std::size_t iteration_count = 0;
  std::size_t multigrid_level_count = 0;
  double true_residual_l2 = 0.0;  // L2 norm of the original integrated residual in m³/s.
  double max_volume_scaled_residual = 0.0;  // Maximum |b − Ap| / V in s⁻¹.
  double max_cell_constraint_defect = 0.0;  // Maximum absolute physical defect in s⁻¹.
};

// Owns rows in input cell order and velocities in input face order. Both
// original-system diagnostics include the rows selected as zero-pressure pins.
struct MacPressureProjection3D {
  std::vector<double> cell_pressures;  // Pa.
  std::vector<double> corrected_face_velocities;  // Positive normal, m/s.
  std::vector<double> integrated_rhs;  // m³/s.
  std::vector<double> true_residuals;  // b − Ap in m³/s.
  std::vector<double> cell_constraint_defects;  // Corrected outward flux − source + compliance, divided by V, in s⁻¹.
  std::vector<std::size_t> gauge_cell_ordinals;  // In increasing cell order.
  MacPressureDiagnostics3D diagnostics;
};

struct MacPressureSolverOptions3D {
  std::size_t max_iterations = 256;  // Inclusive budget in [1, 256].
};

// Project frozen MAC velocities using mixed-precision multigrid PCG. The
// previous pressure initializes each anchored component; subtract the prior
// pin pressure on unanchored components. Success requires the original FP64
// residual and independent physical defect gates, including every pin row:
// ||b-Ap||₂ <= 2e-6 m³/s + 2e-6 ||b||₂ and both maximum |r|/V and physical
// constraint defect <= 2e-5 s⁻¹. A nonzero internal pressure jump additionally
// caps the global tolerance at 1e-3 ||b||₂ when the RHS norm is positive.
// Returns kInvalidArgument for invalid data, face topology, or an iteration
// budget outside [1,256]. Missing face ownership or incompatible unanchored
// components return kFailedPrecondition. Extents above capacity return
// kResourceExhausted. Unrepresentable indices or arithmetic return kOutOfRange,
// and a residual/defect identity mismatch returns kInternal. Nonpositive
// curvature or failure to converge returns kAborted. Errors carry no projection.
absl::StatusOr<MacPressureProjection3D>
ProjectMacPressure3D(const MacPressureInputs3D& inputs, MacPressureSolverOptions3D options = {});

}  // namespace fluidity
