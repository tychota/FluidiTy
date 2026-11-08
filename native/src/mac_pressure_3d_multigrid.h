#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/mac_pressure_3d.h"
#include "mac_pressure_3d_system.h"

namespace fluidity::pressure_internal {

inline constexpr std::size_t kNoPressureUnknown3D = std::numeric_limits<std::size_t>::max();

// One positive coefficient in (m³/s)/Pa, shared by both endpoint rows.
struct MacPressureEdge3D {
  std::size_t lower_unknown_ordinal;
  std::size_t higher_unknown_ordinal;
  double conductance;
};

struct MacPressureNeighbor3D {
  std::size_t unknown_ordinal;
  std::size_t edge_ordinal;  // Index of the shared coefficient; never a separately rounded mirror.
};

// A connected terminal block with at most 32 unknowns. The row-major lower
// Cholesky factor has units sqrt((m³/s)/Pa); upper entries are zero.
struct MacPressureCoarseBlock3D {
  std::vector<std::size_t> unknown_ordinals;
  std::vector<double> cholesky_lower;
};

struct MacPressureLevel3D {
  std::vector<CellIndex3D> coordinates;  // Signed aggregate coordinates, not world positions.
  std::vector<std::size_t> component_ordinals;  // Lowest unknown in each connected graph block.
  std::vector<double> anchor_diagonal;  // Compliance, Dirichlet and eliminated-pin conductance, (m³/s)/Pa.
  std::vector<double> diagonal;  // Anchor plus all incident conductances, (m³/s)/Pa.
  std::vector<MacPressureEdge3D> edges;
  std::vector<std::size_t> row_offsets;  // CSR offsets into neighbors, including final exclusive end.
  std::vector<MacPressureNeighbor3D> neighbors;
  // Piecewise-constant prolongation: fine correction i receives coarse correction aggregate_ordinals[i].
  // Restriction is the unnormalized transpose sum. Empty on the terminal level.
  std::vector<std::size_t> aggregate_ordinals;
  std::vector<MacPressureCoarseBlock3D> terminal_blocks;
};

// Shared, owning upload data for CPU/CUDA. Fine edges and anchors are rounded
// once to FP32 and stored exactly in FP64. Fine diagonals are their FP64 sums
// and convert to FP32 when used by the fine smoother. Galerkin coefficients,
// coarse operations and factors use FP64. The fine operator is matrix-free.
struct MacPressureHierarchy3D {
  std::vector<std::size_t> unknown_to_cell;
  std::vector<std::size_t> cell_to_unknown;  // kNoPressureUnknown3D marks eliminated zero pins.
  std::vector<MacPressureLevel3D> levels;
};

// Build the gauge-restricted SPD graph, retaining edges to pins as anchors.
// Aggregate within connected blocks after pin removal using signed floor / 2;
// fall back to deterministic adjacent pairs if coordinates do not shrink a
// block. At least one coarsening is performed for every nonsingleton fine
// block. Returns kOutOfRange for nonrepresentable coefficients or factors.
// An error publishes no hierarchy.
absl::StatusOr<MacPressureHierarchy3D> BuildMacPressureHierarchy3D(const PreparedMacPressureSystem3D& system);

// Initialize FP64 pressure from previous pressure, subtracting the prior pin
// pressure only in its unanchored component. Returns kOutOfRange on overflow.
absl::StatusOr<std::vector<double>> InitialMacPressure3D(const PreparedMacPressureSystem3D& system);

struct MacPressureStoppingCriteria3D {
  double integrated_residual_limit;  // 2e-6 m³/s + 2e-6 ||b||₂.
  double volume_scaled_limit = 2e-5;  // s⁻¹, for both algebraic residual and physical defect.
};

// Freeze the original-RHS global tolerance with an overflow-safe FP64 norm.
// Returns kOutOfRange when the norm or tolerance cannot be represented.
absl::StatusOr<MacPressureStoppingCriteria3D> MacPressureStoppingCriteria(const PreparedMacPressureSystem3D& system);

// All three original-system metrics must be finite and satisfy their bounds.
// The evaluator, separately, enforces the signed residual/defect identity.
bool MeetsMacPressureStoppingCriteria3D(
    const MacPressureDiagnostics3D& diagnostics,
    const MacPressureStoppingCriteria3D& criteria
);

}  // namespace fluidity::pressure_internal
