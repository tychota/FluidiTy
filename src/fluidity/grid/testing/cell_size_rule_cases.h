// Test-only: the 10,000 fact sets of the cell size rule's tests, and the refinement volumes of its factor tests.
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
//   A refinement volume of a test comes from its layer-0 codes: each layer above holds the smallest code over 2 x 2 x 2
//   entries of the layer below, and the layers own their storage. The factor fixture is one such volume, which the
//   tests of the factor code on the CPU and on the GPU share.
//
// Owns:    kTestFactSetCount, BuildTestCellSizeFacts, ComputeTestReferenceSpeed, TestRefinementVolume,
//          BuildTestRefinementFactors, kFactorFixtureSmallestSizeLevel, kFactorFixtureCountX, kFactorFixtureCountY,
//          kFactorFixtureCountZ, kFactorFixtureSpacing, BuildFactorFixtureVolume
// Uses:    CPU-only guard (base/cpu_only.h), math::kPositiveInfinity (base/scalar_math.h), CellSizeFacts,
//          RefinementFactors (grid/cell_size_rule.h), SizeLevel (grid/grid_ids.h)
// Used by: the tests of the cell size rule
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-size-rule>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/scalar_math.h"
#include "fluidity/grid/cell_size_rule.h"
#include "fluidity/grid/grid_ids.h"

namespace fluidity::grid {

// The fact sets i = 0..kTestFactSetCount - 1 are "the 10,000 fact sets".
inline constexpr int kTestFactSetCount = 10000;

// Builds fact set i = fact_set_index with the given cell size factor (no unit). The merge test's two flags stay false.
//
// Example: BuildTestCellSizeFacts(0, 1.0) has depth and distance +infinity, velocity change size 0 and both flags set.
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

// A refinement volume for a test: the layers it reads from, and the RefinementFactors that point into them. Moving
// it keeps the pointers valid, since each layer's array stays where it is; copying would not, so it is not copyable.
struct TestRefinementVolume {
  TestRefinementVolume() = default;
  TestRefinementVolume(const TestRefinementVolume&) = delete;
  TestRefinementVolume& operator=(const TestRefinementVolume&) = delete;
  TestRefinementVolume(TestRefinementVolume&&) noexcept = default;
  TestRefinementVolume& operator=(TestRefinementVolume&&) noexcept = default;
  ~TestRefinementVolume() = default;

  std::vector<std::vector<std::int8_t>> layers;  // layers[0] is the given layer-0 codes.
  RefinementFactors factors;
};

// Makes a refinement volume from the codes of layer 0, with layer_count layers: layer L has ceil(count / 2^L)
// entries per axis, and each of its entries is the smallest code over the 2 x 2 x 2 entries of layer L - 1 that exist.
// The layers hold the smallest code over the entries that exist; ComputeCellSizeFactorCode, not the layers, accounts
// for a box that reaches outside the volume.
//
// Example: 8 x 4 x 4 entries and 3 layers give layers of 8 x 4 x 4, 4 x 2 x 2 and 2 x 1 x 1 entries; layer 2's entry
//   x = 1 is the smallest code over x = 4..7, y = 0..3, z = 0..3 of layer 0.
// Requires: count_x, count_y and count_z are 1 or more; layer_count is 1..kMaxSizeLevels; layer_0_codes holds
//   count_x * count_y * count_z codes, x fastest, then y, then z.
// clang-tidy: three counts and two more integers; the swappable-parameters check flags them.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
inline TestRefinementVolume BuildTestRefinementFactors(
    int count_x,
    int count_y,
    int count_z,
    int entry_spacing_log2,
    int layer_count,
    std::span<const std::int8_t> layer_0_codes
) {
  TestRefinementVolume volume;
  volume.factors.count_x = count_x;
  volume.factors.count_y = count_y;
  volume.factors.count_z = count_z;
  volume.factors.entry_spacing_log2 = entry_spacing_log2;
  volume.factors.layer_count = layer_count;
  volume.layers.emplace_back(layer_0_codes.begin(), layer_0_codes.end());
  int below_x = count_x;
  int below_y = count_y;
  int below_z = count_z;
  for (int layer = 1; layer < layer_count; ++layer) {
    const int layer_x = (below_x + 1) / 2;
    const int layer_y = (below_y + 1) / 2;
    const int layer_z = (below_z + 1) / 2;
    const std::vector<std::int8_t>& below = volume.layers[static_cast<std::size_t>(layer) - 1];
    std::vector<std::int8_t> codes(static_cast<std::size_t>(layer_x) * layer_y * layer_z, std::int8_t{127});
    for (int z = 0; z < below_z; ++z) {
      for (int y = 0; y < below_y; ++y) {
        for (int x = 0; x < below_x; ++x) {
          const std::size_t below_index = (static_cast<std::size_t>(z) * below_y + y) * below_x + x;
          const std::size_t index = (static_cast<std::size_t>(z / 2) * layer_y + y / 2) * layer_x + x / 2;
          codes[index] = std::min(codes[index], below[below_index]);
        }
      }
    }
    volume.layers.push_back(std::move(codes));
    below_x = layer_x;
    below_y = layer_y;
    below_z = layer_z;
  }
  for (int layer = 0; layer < layer_count; ++layer) {
    volume.factors.layer_codes[layer] = volume.layers[static_cast<std::size_t>(layer)].data();
  }
  return volume;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// The factor fixture: biggest cells 1 m wide, 5 size levels, smallest size level 4, one entry per 4 smallest cells,
// 2 x 1 x 1 size-level-0 cells. Layer 0 is 8 x 4 x 4 entries; code -8 on the entries x = 5, 0 elsewhere.
inline constexpr SizeLevel kFactorFixtureSmallestSizeLevel{4};
inline constexpr int kFactorFixtureCountX = 8;
inline constexpr int kFactorFixtureCountY = 4;
inline constexpr int kFactorFixtureCountZ = 4;
inline constexpr int kFactorFixtureSpacing = 2;

// Builds the refinement volume of the factor fixture with layer_count layers.
//
// Example: 3 layers give layers of 8 x 4 x 4, 4 x 2 x 2 and 2 x 1 x 1 entries, enough for every size level 0..4.
// Requires: layer_count is 1..kMaxSizeLevels.
inline TestRefinementVolume BuildFactorFixtureVolume(int layer_count) {
  const std::size_t entry_count =
      static_cast<std::size_t>(kFactorFixtureCountX) * kFactorFixtureCountY * kFactorFixtureCountZ;
  std::vector<std::int8_t> codes(entry_count, 0);
  for (int z = 0; z < kFactorFixtureCountZ; ++z) {
    for (int y = 0; y < kFactorFixtureCountY; ++y) {
      codes[(static_cast<std::size_t>(z) * kFactorFixtureCountY + y) * kFactorFixtureCountX + 5] = -8;
    }
  }
  return BuildTestRefinementFactors(
      kFactorFixtureCountX,
      kFactorFixtureCountY,
      kFactorFixtureCountZ,
      kFactorFixtureSpacing,
      layer_count,
      codes
  );
}

}  // namespace fluidity::grid
