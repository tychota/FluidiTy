#pragma once

#include <cstddef>
#include <cstdint>

namespace fluidity::cuda_internal {

inline constexpr std::size_t kMaxCudaPressureCells = 8;
// A one-cell-wide strip maximizes face count for the given number of cells.
inline constexpr std::size_t kMaxCudaPressureFaces = 3 * kMaxCudaPressureCells + 1;

struct PressureFaceInput {
  // The tail has outward incidence +1, the head -1. A missing cell has index -1.
  std::int64_t tail_cell_index;
  std::int64_t head_cell_index;
  double face_area;  // At unit depth.
  double predicted_face_velocity;
  double open_area_fraction;
  double inverse_density;
  double pressure_gradient_distance;
  double wall_face_velocity;
  double dirichlet_pressure;
  double oriented_pressure_jump;  // Added to tail pressure minus head pressure.
  bool has_dirichlet_pressure;
};

struct PressureCellInput {
  double target_cell_divergence;
  double fluid_volume;
  double bulk_modulus;  // Positive infinity gives zero compliance.
  double previous_pressure;
};

struct PreparedPressureInputs {
  std::size_t cell_count;
  std::size_t face_count;
  double step_duration;
  PressureFaceInput faces[kMaxCudaPressureFaces];
  PressureCellInput cells[kMaxCudaPressureCells];
};

enum class PressureKernelError : int {
  kNone,
  kNonrepresentableArithmetic,
  kIncompatibleComponent,
  kUnsafeCholeskyPivot,
};

struct PressureKernelResult {
  int compiled_architecture;
  PressureKernelError error;
  std::size_t gauge_count;
  std::size_t gauge_cell_indices[kMaxCudaPressureCells];
  double cell_pressures[kMaxCudaPressureCells];
  double corrected_face_velocities[kMaxCudaPressureFaces];
  // Original integrated system, with the actual cell count as the row stride.
  double original_matrix[kMaxCudaPressureCells * kMaxCudaPressureCells];
  double original_rhs[kMaxCudaPressureCells];
  double original_residuals[kMaxCudaPressureCells];
  double cell_constraint_defects[kMaxCudaPressureCells];
};

// Launches the complete solve with one thread. Both buffers outlive the stream
// work, and the result starts zeroed. The return value is the CUDA launch code.
int LaunchPressure(
    const PreparedPressureInputs* device_pressure_inputs,
    PressureKernelResult* device_pressure_result,
    void* stream
);

}  // namespace fluidity::cuda_internal
