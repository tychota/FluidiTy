#pragma once

#include <cstddef>
#include <type_traits>

namespace fluidity::cuda_internal {

// POD staging uses explicit counts and offsets into checked device allocations.
// Missing original endpoints use cell_count; graph indices are level-local.
struct PressureCell3D {
  double fluid_volume;  // m³.
  double integrated_target_flux;  // V s, m³/s.
  double compliance;  // (m³/s)/Pa.
  double previous_pressure;
  double integrated_rhs;
  double rhs_absolute_term_sum;
  double initial_pressure;  // Gauge-adjusted warm start in Pa.
};
struct PressureLink3D {
  std::size_t tail_cell_ordinal;
  std::size_t head_cell_ordinal;
  double open_area;
  double blocked_area;
  double pressure_to_velocity_scale;
  double conductance;  // Shared physical open-area coefficient, (m³/s)/Pa.
  double predicted_normal_velocity;
  double prescribed_wall_flux;  // Checked positive-normal wall flux, m³/s.
  double jump;
  double exterior_pressure;
};
// Offsets address the packed coefficient or index allocation. CSR row ranges
// reference a single conductance for both mirrored endpoint rows. The scratch
// offset counts unknowns across all preceding levels.
struct PressureLevel3D {
  std::size_t count;
  std::size_t scratch_offset;
  std::size_t anchor_offset;
  std::size_t diagonal_offset;
  std::size_t edge_offset;
  std::size_t row_offset;
  std::size_t neighbor_offset;
  std::size_t neighbor_edge_offset;
  std::size_t aggregate_offset;
  std::size_t block_offset;
  std::size_t block_count;
};
// One terminal block, at most 32 unknowns. The factor offset names count²
// row-major FP64 Cholesky values in the coefficient allocation.
struct PressureBlock3D {
  std::size_t count;
  std::size_t unknown_offset;
  std::size_t factor_offset;
};
struct PressureSum3D {
  double sum;
  double correction;
};
struct PressureRow3D {
  PressureSum3D flux;
  PressureSum3D absolute_terms;
};
enum class PressureError3D {
  kNone,
  kRange,
  kIdentity,
  kCurvature,
  kExhausted
};
struct PressureState3D {
  PressureError3D error;
  int compiled_architecture;
  std::size_t iterations;
  bool converged;
  bool restart;  // Restart direction after true-residual replacement.
  double previous_preconditioned_residual_inner_product;
  double residual_l2;
  double maximum_scaled_residual;
  double maximum_defect;
};
struct PressureDevice3D {
  const PressureCell3D* cells;
  const PressureLink3D* links;
  const PressureLevel3D* levels;
  const PressureBlock3D* blocks;
  const std::size_t* indices;
  const double* coefficients;
  std::size_t cell_count;
  std::size_t face_count;
  std::size_t level_count;
  std::size_t unknown_count;
  std::size_t level_unknown_count;
  std::size_t unknown_map_offset;
  std::size_t max_iterations;
  double integrated_limit;  // L2 limit for the original integrated residual, m³/s.
  double local_limit;  // Limit for the volume-scaled residual and physical defect, s⁻¹.
  // Six fine vectors and three coarse vectors; coarse slots include unused
  // fine-level positions to keep every level's explicit scratch offset stable.
  float* fine_scratch;
  double* coarse_scratch;
  PressureRow3D* rows;  // Original operator rows followed by independent physical rows.
  // Full-cell base, candidate, residual and defect, then corrected face velocities.
  double* output;
  PressureState3D* state;
};
static_assert(std::is_trivially_copyable_v<PressureDevice3D>);
static_assert(std::is_trivially_copyable_v<PressureCell3D>);
static_assert(std::is_trivially_copyable_v<PressureLink3D>);
static_assert(std::is_trivially_copyable_v<PressureLevel3D>);
static_assert(std::is_trivially_copyable_v<PressureBlock3D>);
static_assert(std::is_trivially_copyable_v<PressureState3D>);

// Initialize and evaluate the warm start, or execute exactly one PCG iteration.
// Arithmetic and convergence decisions stay on device. Returns the launch error.
int LaunchMacPressure3D(PressureDevice3D device, bool initialize, void* stream);

// The arena batches a bounded number of complete iterations between status
// reads. The device stops at its first convergence or numerical failure.
inline constexpr std::size_t kMacPressureIterationsPerBatch3D = 8;
int LaunchMacPressureBatch3D(PressureDevice3D device, bool initialize, std::size_t iteration_count, void* stream);
}  // namespace fluidity::cuda_internal
