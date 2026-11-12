#pragma once

#include <array>
#include <cstddef>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/host/grid.h"
#include "fluidity/host/small_solve.h"

namespace fluidity {

struct MacStrainExtents {
  std::size_t strain_row_count;  // Strain samples, including both normal components and interior shear.
  std::size_t face_velocity_count;  // Face velocities in MAC face order.
  // Number of entries in the dense row-major strain operator.
  std::size_t coefficient_count;
};

// Returns the extents of the dense strain operator and its diagonal quadrature.
// Propagates checked grid errors. Returns kUnimplemented for more than 64 velocity
// columns. The strain row count can exceed 64.
absl::StatusOr<MacStrainExtents> MacStrainExtents2D(const Grid2D& grid);

// Assembles the symmetric-gradient operator E (inverse length) and its diagonal
// quadrature W for a constant dynamic viscosity mu (Pa s). Columns are x-normal
// faces followed by y-normal faces in the FaceGeometry order. Rows are all du/dx
// cells, then all dv/dy cells, then Dxy at interior vertices, each in row-major order.
// W contains 2*mu*h^2 for normal rows and 4*mu*h^2 for shear rows at unit depth.
//
// The caller supplies exact MacStrainExtents2D extents. Outputs must be disjoint
// from each other and the grid and are preserved on error. Returns kInvalidArgument
// for invalid geometry, viscosity, extents, or overlap, kUnimplemented for unsupported
// size, and kOutOfRange for unrepresentable geometry or coefficients. A nonzero
// quadrature must be a positive normal double.
absl::Status AssembleMacStrain2D(
    const Grid2D& grid,
    double dynamic_viscosity,
    std::span<double> strain_operator,
    std::span<double> strain_quadrature_weights
);

enum class ViscosityMode {
  kAuto,
  kExplicit,
  kImplicit,
};

enum class ViscosityMethod {
  kExplicit,
  kImplicit,
};

struct ViscosityInputsView {
  // Face velocity in length/time. Its order defines the columns of E.
  std::span<const double> face_velocities;
  // Positive face inertia in mass units, supplied by the matching pressure geometry.
  std::span<const double> face_inertias;
  // Dense row-major E in inverse length, with one column per velocity unknown.
  std::span<const double> strain_operator;
  // Number of strain rows and entries in both strain-space vectors.
  std::size_t strain_row_count;
  // Diagonal strain quadrature W in mass*length^2/time.
  std::span<const double> strain_quadrature_weights;
  // Prescribed wall strain in inverse time, in the same row order as E.
  std::span<const double> boundary_strain;
  // Nonnegative step duration in time units.
  double dt;
  ViscosityMode mode = ViscosityMode::kAuto;
  // Fractional explicit stability margin in [0, 1).
  double explicit_stability_margin = 0.01;
};

struct ViscosityDiagnostics {
  ViscosityMethod method;
  // Outward-rounded trace bound for the exact mass-scaled stiffness, in inverse
  // time. Positive infinity means no finite certificate was representable.
  double spectral_bound;
  // Mass-metric kinetic energies, in energy units.
  double energy_before;
  double energy_after;
  // Wall work in energy units. Uses old velocity for forward Euler and new
  // velocity for backward Euler.
  double wall_work;
  // Half the squared mass norm of the velocity increment, in energy units.
  double numerical_dissipation;
};

struct ViscosityOutputsView {
  // Caller-owned span in the same units, order, and extent as the input velocity.
  std::span<double> updated_face_velocities;
  ViscosityDiagnostics& diagnostics;
};

struct ViscosityWorkspace {
  // Dense K during assembly, then the mass-scaled system for an implicit step.
  std::array<double, kMaxDenseUnknowns * kMaxDenseUnknowns> matrix;
  std::array<double, kMaxDenseUnknowns> wall_load;  // E^T W boundary_strain in force units.
  std::array<double, kMaxDenseUnknowns> sqrt_mass;  // Facewise square roots of inertia.
  std::array<double, kMaxDenseUnknowns> solve_rhs;  // Implicit RHS with mass and power-of-two scaling.
  // Holds the scaled implicit solution until physical face velocities are restored.
  std::array<double, kMaxDenseUnknowns> next_velocity;
  SmallSolveWorkspace solve;
};

// Advances the entire system using K = E^T W E and wall load E^T W boundary_strain.
// Auto mode selects explicit only when an outward-rounded comparison certifies
// dt * spectral_bound <= 2 * (1 - explicit_stability_margin). The trace bound
// encloses the exact operator represented by the input coefficients. The wall
// strain is an algebraic lift.
//
// Uses 1 to 64 velocity unknowns and any representable strain row count, including
// zero. All inputs must be finite, masses positive and weights nonnegative.
// Read-only spans may alias. Velocity output, diagnostics, and the entire workspace
// must be disjoint from each other, the input view object, and its referenced spans.
// Outputs are written only on success. Scratch may change on error and is nonpersistent.
//
// Returns kInvalidArgument for invalid inputs, extents, or overlap and kUnimplemented
// for more than 64 unknowns. Returns kFailedPrecondition for an unsafe forced explicit
// step or unsafe SPD solve. Returns kOutOfRange for unrepresentable sizes, nonfinite
// arithmetic, or nonzero stiffness, load, or scaled matrix terms below normal double range.
// The implicit solve also requires a common power-of-two scale that keeps the nonzero
// right-hand-side terms and weighted solution components normal, with room for sums.
// Range loss detected in the solve or when restoring physical velocities also returns
// kOutOfRange.
absl::Status
ViscousStep(const ViscosityInputsView& inputs, ViscosityOutputsView outputs, ViscosityWorkspace& workspace);

}  // namespace fluidity
