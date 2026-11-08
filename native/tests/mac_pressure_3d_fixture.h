#pragma once

#include <string_view>
#include <vector>

#include "fluidity/host/mac_pressure_3d.h"

namespace fluidity::pressure_test {

// Owns one decoded independent oracle case, including all allocated face
// owners. Expected rows and face velocities retain the fixture's input order.
struct MacPressureFixtureCase3D {
  std::string_view name;  // Refers to static case-name storage.
  SparseMacGrid3D grid;
  double step_duration;  // Seconds.
  std::vector<MacPressureCell3D> cells;
  std::vector<MacPressureFace3D> faces;
  MacPressureProjection3D expected_projection;
  double absolute_tolerance;
  double relative_tolerance;
};

// Decode all eight uniform_mac_pressure_3d cases, asserting their array types,
// shapes, units, finite values, index bounds and link encodings. Clears cases
// before loading. Call with ASSERT_NO_FATAL_FAILURE before consuming cases.
void LoadMacPressureFixtureCases3D(std::vector<MacPressureFixtureCase3D>& cases);

// Compare all pressure, velocity, RHS, residual, defect and gauge outputs.
// Callers supply tolerances explicitly so evaluator, CPU and CUDA tests retain
// their own numerical acceptance policy. Does not invoke a solver or evaluator.
void ExpectMacPressureFixtureProjection3D(
    const MacPressureProjection3D& actual,
    const MacPressureProjection3D& expected,
    double absolute_tolerance,
    double relative_tolerance
);

}  // namespace fluidity::pressure_test
