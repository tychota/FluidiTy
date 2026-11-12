#pragma once

#include <cstddef>

#include "absl/status/status.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity::liquid_state_internal {

// A checked source of canonical adaptive records. Read functions borrow the
// context for the call and return one record by value for an in-range ordinal.
struct AdaptiveRecordView3D {
  const void* context;
  double finest_cell_width;
  std::size_t cell_count;
  std::size_t face_count;
  LiquidAdaptiveCellState3D (*read_cell)(const void*, std::size_t);
  LiquidAdaptiveFaceState3D (*read_face)(const void*, std::size_t);
};

// Validates geometric grading, canonical leaf order, complete face count and
// finite physical values. Uses only stack storage and leaves input untouched.
absl::Status ValidateAdaptiveRecordView3D(const AdaptiveRecordView3D& records);

}  // namespace fluidity::liquid_state_internal
