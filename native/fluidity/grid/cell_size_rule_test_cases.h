// Test-only: the 10,000 fact sets of the cell size rule's tests.
//
// Purpose: give every test of the cell size rule the same many fact sets and reference speeds, drawn from an integer
//   by fixed formulas, so a test can compare two ways of computing the rule over them.
//
// Idea: fact set i takes each fact from i modulo a different prime, so the 10,000 sets i = 0..9999 mix depths and
//   distances inside and beyond each reach, infinite ones, velocity change sizes of 0 and above, and both flags.
//     depth                 (i % 97) * 0.013 m, +infinity when i % 11 == 0
//     distance to a solid   (i % 89) * 0.011 m, +infinity when i % 7 == 0
//     velocity change size  (i % 83) * 0.37 1/s
//     touches an air pocket when i % 31 == 0; wants the smallest cells when i % 37 == 0
//     reference speed       0.5 + (i % 13) * 0.25 m/s
//   Example: i = 12 gives depth 0.156 m, distance 0.132 m, velocity change size 4.44 1/s, no flag, and 3.5 m/s.
//
// Owns:    kTestFactSetCount, BuildTestCellSizeFacts, ComputeTestReferenceSpeed
// Uses:    CPU-only guard (base/cpu_only.h), math::kPositiveInfinity (base/scalar_math.h), CellSizeFacts
//          (grid/cell_size_rule.h)
// Used by: the tests of the cell size rule
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-size-rule>

#pragma once

#include "fluidity/base/cpu_only.h"

#include "fluidity/base/scalar_math.h"
#include "fluidity/grid/cell_size_rule.h"

namespace fluidity::grid {

// The fact sets i = 0..kTestFactSetCount - 1 are "the 10,000 fact sets".
inline constexpr int kTestFactSetCount = 10000;

// Builds fact set i = fact_set_index with the given cell size factor (no unit). The merge test's two flags stay false.
//
// Example: BuildTestCellSizeFacts(0, 1.0) has depth and distance +infinity, velocity change size 0 and both flags
//   set.
// Requires: fact_set_index is 0 or more.
inline CellSizeFacts BuildTestCellSizeFacts(int fact_set_index, double cell_size_factor) {
  const bool has_no_surface_near = fact_set_index % 11 == 0;
  const bool has_no_solid_near = fact_set_index % 7 == 0;
  const auto depth_steps = static_cast<double>(fact_set_index % 97);
  const auto distance_steps = static_cast<double>(fact_set_index % 89);
  return {
      .depth_below_surface = has_no_surface_near ? math::kPositiveInfinity : depth_steps * 0.013,
      .distance_to_solid = has_no_solid_near ? math::kPositiveInfinity : distance_steps * 0.011,
      .velocity_change_size = static_cast<double>(fact_set_index % 83) * 0.37,
      .cell_size_factor = cell_size_factor,
      .touches_air_pocket = fact_set_index % 31 == 0,
      .wants_smallest_cells = fact_set_index % 37 == 0,
  };
}

// Computes the reference speed of fact set i = fact_set_index, in m/s: 0.5 + (i % 13) * 0.25.
//
// Example: ComputeTestReferenceSpeed(12) is 3.5.
// Requires: fact_set_index is 0 or more.
inline double ComputeTestReferenceSpeed(int fact_set_index) {
  const double speed_steps = static_cast<double>(fact_set_index % 13) * 0.25;
  return 0.5 + speed_steps;
}

}  // namespace fluidity::grid
