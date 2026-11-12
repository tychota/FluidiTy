#pragma once

#include <cstddef>

namespace fluidity::cuda_internal {

struct ViscousVelocityTerm3D {
  std::size_t face_slot;
  double weighted_coefficient;
};

struct ViscousVelocityFace3D {
  double inertia;
  double diagonal;
  double rhs;
};

enum class ViscousVelocityError3D {
  kNone,
  kRange,
  kCurvature,
  kExhausted
};

struct ViscousVelocityState3D {
  int compiled_architecture;
  ViscousVelocityError3D error;
  std::size_t iterations;
  double max_residual;
};

struct ViscousVelocityDevice3D {
  const ViscousVelocityFace3D* faces;
  const ViscousVelocityTerm3D* terms;
  const std::size_t* row_offsets;
  std::size_t face_count;
  std::size_t row_count;
  std::size_t max_iterations;
  double max_rhs;
  double tolerance;
  double* velocity;
  double* residual;
  double* preconditioned;
  double* direction;
  double* applied;
  ViscousVelocityState3D* state;
};

}  // namespace fluidity::cuda_internal
