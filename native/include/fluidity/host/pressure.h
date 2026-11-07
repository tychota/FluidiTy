#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "fluidity/host/grid.h"
#include "fluidity/host/small_solve.h"

namespace fluidity {

inline constexpr std::size_t kMaxPressureCells = 64;
// The face count is 2*nx*ny + nx + ny, maximized by a 1-by-64 grid.
inline constexpr std::size_t kMaxPressureFaces = 3 * kMaxPressureCells + 1;

// Every face field has FaceCount(grid) entries in MAC face order. Every cell field
// has CellCount(grid) entries in row-major order. Callers supply all fields.
// Velocities and jumps follow the oriented MAC face normal. dt, mobility, distance
// and fluid volume are finite and positive. Aperture is the open area fraction in
// [0, 1]. A mask value of 1 selects a boundary pressure and requires zero jump.
// Bulk modulus is positive, with +infinity selecting incompressibility. All other
// floating-point fields are finite.
struct PressureInputsView {
  Grid2D grid;
  double dt;  // Step duration.
  std::span<const double> velocity;  // Predicted fluid face velocity.
  std::span<const double> aperture;
  std::span<const double> mobility;  // Face inverse density.
  std::span<const double> distance;  // Physical pressure-gradient distance.
  std::span<const double> wall_velocity;
  std::span<const std::uint8_t> dirichlet_mask;
  std::span<const double> dirichlet_pressure;
  std::span<const double> jump;  // Oriented pressure jump added to tail pressure minus head pressure.
  std::span<const double> source;  // Target divergence in inverse time.
  std::span<const double> fluid_volume;
  std::span<const double> bulk_modulus;
  std::span<const double> previous_pressure;
};

struct PressureOutputsView {
  std::span<double> pressure;
  std::span<double> velocity;
  // Original integrated system before gauge constraints, stored row-major.
  std::span<double> original_matrix;
  std::span<double> rhs;
  // Integrated residual b - A*p, including the original equation at each gauge.
  std::span<double> residual;
  // Corrected outward flux per fluid volume, minus source, plus the compliance term. Units are inverse time.
  std::span<double> constraint_defect;
  // Cell indices of the pins in ascending order. Only the first gauge_count entries are written.
  std::span<std::size_t> gauges;
  std::size_t& gauge_count;
};

// Caller-owned staging for the original system, gauged solve and corrected fields.
// Scratch may change on failure and has no persistent state.
struct PressureWorkspace {
  // Cached pressure coefficients for one oriented face. A missing cell index is -1.
  struct FaceLink {
    std::int64_t tail;
    std::int64_t head;
    double open_area;
    double wall_area;
    double velocity_scale;  // dt * mobility / distance, for the face velocity correction.
    double conductance;  // Open area times velocity_scale, for the integrated system.
    // Exterior pressure multiplied by the adjacent cell's outward incidence.
    double signed_boundary_pressure;
    bool active;
  };

  std::array<FaceLink, kMaxPressureFaces> faces;
  // Bulk modulus times dt, infinite for an incompressible cell.
  std::array<double, kMaxPressureCells> bulk_time;
  std::array<double, kMaxPressureCells * kMaxPressureCells> original_matrix;
  std::array<double, kMaxPressureCells> rhs;
  std::array<double, kMaxPressureCells * kMaxPressureCells> solve_matrix;
  std::array<double, kMaxPressureCells> solve_rhs;
  SmallSolveWorkspace solve_workspace;
  std::array<double, kMaxPressureCells> pressure;
  std::array<double, kMaxPressureFaces> velocity;
  std::array<double, kMaxPressureCells> residual;
  // Integrated corrected flux during accumulation, then the staged per-volume defect.
  std::array<double, kMaxPressureCells> constraint_defect;
  std::array<std::size_t, kMaxPressureCells> gauges;
};

// Assembles the integrated pressure system and corrects open internal and Dirichlet face velocities.
// Each compatible unanchored component pins the pressure at its lowest cell index to zero.
// The original matrix has n*n entries, velocity has f entries, and all other output spans have
// n entries (n = CellCount(grid), f = FaceCount(grid)). Read-only spans may alias. Outputs,
// including gauge_count, and the entire workspace must be disjoint from the input view,
// its referenced spans, and each other.
// All outputs are preserved on failure.
//
// Returns kInvalidArgument for invalid inputs, extents or overlap. Returns
// kUnimplemented for grids over 64 cells and kFailedPrecondition for incompatible
// unanchored components or unsafe SPD solves. Returns kOutOfRange for grid or
// arithmetic overflow, other nonfinite computed arithmetic, or a positive
// active-face open area or velocity scale rounded to zero. Active conductance
// and finite-modulus compliance must be positive normal doubles, at least
// std::numeric_limits<double>::min(), or the solve returns kOutOfRange.
absl::Status
SolvePressure2D(const PressureInputsView& inputs, PressureOutputsView outputs, PressureWorkspace& workspace);

}  // namespace fluidity
