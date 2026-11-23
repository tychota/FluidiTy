// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on a machine with a GPU.
//
// Runs the cell size rule and the size level of a new cell on the GPU and compares every result with the CPU's, bit
// for bit: the method paper promises the same bytes on both processors (<sec-cell-size-rule>). Every GPU result goes
// to WriteResultDigest under a unique name, so the repeat run shows the GPU gave the same bytes twice.

#include "fluidity/grid/cell_size_rule.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <random>
#include <span>
#include <string_view>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_size_rule_cases.h"
#include "fluidity/grid/testing/cell_size_rule_runner_gpu.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// The frame of the factor fixture: biggest cells 1 m wide and 5 size levels.
constexpr GridFrame kFactorFixtureFrame = {.origin = {}, .biggest_cell_width = 1.0, .size_level_count = 5};

// The case of fact set i: its facts with the factor table.factor[i % 49], its reference speed, and for the factor code
// the cell of size level i % 5 at the size-level-4 coordinate {i % 41, i % 23, 0} shifted right by 4 - i % 5.
CellSizeRuleCase BuildTestRuleCase(int fact_set_index, const CellSizeFactorTable& table) {
  const int size_level = fact_set_index % 5;
  const auto shift = static_cast<unsigned>(4 - size_level);
  // Both remainders are 0 or more, so the shifts act on unsigned values.
  const auto coord_x = static_cast<std::int32_t>(static_cast<unsigned>(fact_set_index % 41) >> shift);
  const auto coord_y = static_cast<std::int32_t>(static_cast<unsigned>(fact_set_index % 23) >> shift);
  const CellCoord cell_coord = {.x = coord_x, .y = coord_y, .z = 0};
  return {
      .facts = BuildTestCellSizeFacts(fact_set_index, table.factor[fact_set_index % 49]),
      .reference_speed = ComputeTestReferenceSpeed(fact_set_index),
      .cell = ToCellId(static_cast<SizeLevel>(size_level), cell_coord),
  };
}

// Says whether two tables of factors hold the same bits in every entry.
bool ComputeIsSameTableBits(const CellSizeFactorTable& first, const CellSizeFactorTable& second) {
  for (int factor_index = 0; factor_index <= 2 * kMaxCellSizeFactorCode; ++factor_index) {
    if (std::bit_cast<std::uint64_t>(first.factor[factor_index])
        != std::bit_cast<std::uint64_t>(second.factor[factor_index])) {
      return false;
    }
  }
  return true;
}

// Says whether two results of a case hold the same bits: the width's bits, the size level and the factor code.
bool ComputeIsSameResultBits(const CellSizeRuleResult& first, const CellSizeRuleResult& second) {
  const bool is_same_width =
      std::bit_cast<std::uint64_t>(first.wanted_width) == std::bit_cast<std::uint64_t>(second.wanted_width);
  return is_same_width && first.wanted_size_level == second.wanted_size_level
      && first.factor_code == second.factor_code;
}

// The 10,000 fact sets with default settings but the fixture's smallest size level 4, in the fixture's frame, and the
// factor table, on both processors.
TEST(ComputeWantedCellWidth, OnGpu_SameBytesAsCpu) {
  const CellSizeFactorTable table = ComputeCellSizeFactorTable();
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kFactorFixtureFrame);
  const CellSizeSettings settings = {.smallest_size_level = kFactorFixtureSmallestSizeLevel};
  const TestRefinementVolume volume = BuildFactorFixtureVolume(3);
  std::vector<CellSizeRuleCase> cases;
  cases.reserve(kTestFactSetCount);
  for (int fact_set_index = 0; fact_set_index < kTestFactSetCount; ++fact_set_index) {
    cases.push_back(BuildTestRuleCase(fact_set_index, table));
  }

  std::vector<CellSizeRuleResult> on_gpu(cases.size());
  CellSizeFactorTable table_on_gpu{};
  const RunOnCallStatus status = ComputeCellSizeRuleOnGpu(
      cases.data(),
      kTestFactSetCount,
      widths,
      settings,
      volume.factors,
      on_gpu.data(),
      &table_on_gpu
  );
  ASSERT_EQ(status, RunOnCallStatus::kDone);
  std::vector<CellSizeRuleResult> on_cpu;
  on_cpu.reserve(cases.size());
  for (const CellSizeRuleCase& rule_case : cases) {
    on_cpu.push_back(ComputeCellSizeRuleResult(rule_case, widths, settings, volume.factors));
  }

  EXPECT_TRUE(ComputeIsSameTableBits(table_on_gpu, table));
  for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
    EXPECT_TRUE(ComputeIsSameResultBits(on_gpu[case_index], on_cpu[case_index]))
        << "fact set " << case_index << ": gpu width " << on_gpu[case_index].wanted_width << ", cpu width "
        << on_cpu[case_index].wanted_width;
  }
  WriteResultDigest("ComputeCellSizeFactorTable on the GPU", std::span<const CellSizeFactorTable>(&table_on_gpu, 1));
  WriteResultDigest(
      "ComputeWantedCellWidth on the GPU, 10,000 fact sets: widths, size levels and factor codes",
      std::span<const CellSizeRuleResult>(on_gpu)
  );
}

// The brief's 10,000 positions of the new-cell test: x, y and z each -1 + 6 u, u from ToUnitDouble of the next raw
// number of one std::mt19937_64 of seed 20261115, so they fill -1..5 m on each axis, the same on every machine.
std::vector<Vec3> BuildTestPositions(int position_count) {
  std::mt19937_64 generator(20261115);  // NOLINT(bugprone-random-generator-seed): a fixed sequence is the point.
  std::vector<Vec3> positions;
  positions.reserve(static_cast<std::size_t>(position_count));
  for (int position_index = 0; position_index < position_count; ++position_index) {
    const double x = -1.0 + (6.0 * ToUnitDouble(generator()));
    const double y = -1.0 + (6.0 * ToUnitDouble(generator()));
    const double z = -1.0 + (6.0 * ToUnitDouble(generator()));
    positions.push_back({.x = x, .y = y, .z = z});
  }
  return positions;
}

// 10,000 positions in -2..6 m on each axis, two biggest widths beyond the cells: x, y and z each -2 + 8 u, from one
// std::mt19937_64 of seed 20261116. Searched positions of the outer ones find no cell.
std::vector<Vec3> BuildWidePositions(int position_count) {
  std::mt19937_64 generator(20261116);  // NOLINT(bugprone-random-generator-seed): a fixed sequence is the point.
  std::vector<Vec3> positions;
  positions.reserve(static_cast<std::size_t>(position_count));
  for (int position_index = 0; position_index < position_count; ++position_index) {
    const double x = -2.0 + (8.0 * ToUnitDouble(generator()));
    const double y = -2.0 + (8.0 * ToUnitDouble(generator()));
    const double z = -2.0 + (8.0 * ToUnitDouble(generator()));
    positions.push_back({.x = x, .y = y, .z = z});
  }
  return positions;
}

// 10,000 positions on the lattice of a quarter smallest width, 0.0625 m, in -2..6 m: each axis -2 + 0.0625 n, n from
// ToUniformIndex(raw, 129) of one std::mt19937_64 of seed 20261117. Faces, edges, corners and centres, so exact ties
// between cells of two sizes occur.
std::vector<Vec3> BuildLatticePositions(int position_count) {
  std::mt19937_64 generator(20261117);  // NOLINT(bugprone-random-generator-seed): a fixed sequence is the point.
  constexpr std::uint64_t kLatticeStepCount = 129;
  constexpr double kLatticeStep = 0.0625;
  const auto to_lattice_coord = [&generator] {
    return -2.0 + (static_cast<double>(ToUniformIndex(generator(), kLatticeStepCount)) * kLatticeStep);
  };
  std::vector<Vec3> positions;
  positions.reserve(static_cast<std::size_t>(position_count));
  for (int position_index = 0; position_index < position_count; ++position_index) {
    const double x = to_lattice_coord();
    const double y = to_lattice_coord();
    const double z = to_lattice_coord();
    positions.push_back({.x = x, .y = y, .z = z});
  }
  return positions;
}

// NaN, +infinity, -infinity, -0 and 1e300 on each axis in turn, the other two axes at 0.5 m: 15 positions.
std::vector<Vec3> BuildSpecialPositions() {
  const std::array<double, 5> special_values = {
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -0.0,
      1e300,
  };
  std::vector<Vec3> positions;
  for (const double value : special_values) {
    positions.push_back({.x = value, .y = 0.5, .z = 0.5});
    positions.push_back({.x = 0.5, .y = value, .z = 0.5});
    positions.push_back({.x = 0.5, .y = 0.5, .z = value});
  }
  return positions;
}

// Says whether the nearest centre among the cells that hold the 27 searched positions is shared by two cells of two
// sizes. Computed in m with ComputeCellCenter: exact for lattice positions in a frame of origin 0 and biggest cells
// 1 m wide, where every centre and every difference has few bits.
bool ComputeIsTwoSizeTie(const GridTables& tables, Vec3 position) {
  double nearest_squared_distance = std::numeric_limits<double>::infinity();
  std::vector<CellPlaceIndex> nearest_places;
  for (int offset_z = -1; offset_z <= 1; ++offset_z) {
    for (int offset_y = -1; offset_y <= 1; ++offset_y) {
      for (int offset_x = -1; offset_x <= 1; ++offset_x) {
        const Vec3 offset = {
            .x = static_cast<double>(offset_x),
            .y = static_cast<double>(offset_y),
            .z = static_cast<double>(offset_z),
        };
        const CellPlaceIndex found = FindCellAt(tables, position + offset);
        if (found.value == kNoIndex) {
          continue;
        }
        const Vec3 center_offset = ComputeCellCenter(tables, found) - position;
        const double squared_distance = math::Dot(center_offset, center_offset);
        if (squared_distance < nearest_squared_distance) {
          nearest_squared_distance = squared_distance;
          nearest_places.clear();
        }
        if (squared_distance == nearest_squared_distance) {
          nearest_places.push_back(found);
        }
      }
    }
  }
  return std::ranges::any_of(nearest_places, [&](CellPlaceIndex place) {
    return ToSizeLevel(tables, ToCellBlockIndex(place)) != ToSizeLevel(tables, ToCellBlockIndex(nearest_places[0]));
  });
}

// Expects each GPU result to equal the CPU's result of the same position in `cpu_tables`.
void CheckSameNewCellResults(
    const GridTables& cpu_tables,
    std::span<const Vec3> positions,
    SizeLevel smallest_size_level,
    std::span<const NewCellResult> on_gpu
) {
  for (std::size_t position_index = 0; position_index < positions.size(); ++position_index) {
    const NewCellResult on_cpu = ComputeNewCellResult(cpu_tables, positions[position_index], smallest_size_level);
    EXPECT_EQ(on_gpu[position_index].nearest_place, on_cpu.nearest_place) << "position " << position_index;
    EXPECT_EQ(on_gpu[position_index].size_level, on_cpu.size_level) << "position " << position_index;
  }
}

// Runs one set of positions on the GPU, expects the CPU's results and writes the GPU results' digest under `name`.
void CheckGpuSetEqualsCpu(
    const TestGrid& gpu_grid,
    const GridTables& cpu_tables,
    std::span<const Vec3> positions,
    std::string_view name
) {
  constexpr SizeLevel kSmallestSizeLevel{2};
  std::vector<NewCellResult> on_gpu(positions.size());
  const RunOnCallStatus status = ComputeNewCellSizeLevelsOnGpu(
      gpu_grid.grid.tables(RunOn::kGpu),
      positions.data(),
      static_cast<int>(positions.size()),
      kSmallestSizeLevel,
      on_gpu.data()
  );
  ASSERT_EQ(status, RunOnCallStatus::kDone) << name;
  CheckSameNewCellResults(cpu_tables, positions, kSmallestSizeLevel, on_gpu);
  WriteResultDigest(name, std::span<const NewCellResult>(on_gpu));
}

// BuildRandomGridCells(7, 3, 4) built on both processors, in a frame of origin 0, biggest cells 1 m wide and 3 size
// levels, so its cells fill 0..4 m on each axis. Four sets of positions. The brief's 10,000 in -1..5 m all find a cell
// and hold no tie: every position there has a searched position inside the cells. The wide set holds positions that
// find no cell, the lattice set exact ties between cells of two sizes, and the special set positions that are not
// finite or lie far away. The CPU checks that the sets hold those cases, so the coverage cannot drop unseen.
TEST(ComputeNewCellSizeLevel, OnGpu_SameBytesAsCpu) {
  constexpr GridFrame kFrame = {.origin = {}, .biggest_cell_width = 1.0, .size_level_count = 3};
  constexpr int kPositionCount = 10000;
  const std::vector<CellId> cells = BuildRandomGridCells(7, 3, 4);
  const GridInputs inputs = {.frame = kFrame, .active_cells = cells, .version = {.value = 1}};
  const std::expected<TestGrid, GridFailure> cpu_grid = BuildTestGrid(inputs);
  ASSERT_TRUE(cpu_grid.has_value());
  const std::expected<TestGrid, GridFailure> gpu_grid = BuildTestGrid(inputs, {.run_on = RunOn::kGpu});
  ASSERT_TRUE(gpu_grid.has_value());
  const GridTables cpu_tables = cpu_grid->grid.tables(RunOn::kCpu);

  const std::vector<Vec3> wide_positions = BuildWidePositions(kPositionCount);
  const std::vector<Vec3> lattice_positions = BuildLatticePositions(kPositionCount);
  const auto finds_no_cell = [&cpu_tables](Vec3 position) {
    return FindNearestCommittedCell(cpu_tables, position).value == kNoIndex;
  };
  EXPECT_TRUE(std::ranges::any_of(wide_positions, finds_no_cell));
  EXPECT_TRUE(std::ranges::any_of(lattice_positions, [&cpu_tables](Vec3 position) {
    return ComputeIsTwoSizeTie(cpu_tables, position);
  }));

  CheckGpuSetEqualsCpu(
      *gpu_grid,
      cpu_tables,
      BuildTestPositions(kPositionCount),
      "ComputeNewCellSizeLevel on the GPU, 10,000 positions: nearest places and size levels"
  );
  CheckGpuSetEqualsCpu(
      *gpu_grid,
      cpu_tables,
      wide_positions,
      "ComputeNewCellSizeLevel on the GPU, 10,000 positions in -2..6 m: nearest places and size levels"
  );
  CheckGpuSetEqualsCpu(
      *gpu_grid,
      cpu_tables,
      lattice_positions,
      "ComputeNewCellSizeLevel on the GPU, 10,000 lattice positions: nearest places and size levels"
  );
  CheckGpuSetEqualsCpu(
      *gpu_grid,
      cpu_tables,
      BuildSpecialPositions(),
      "ComputeNewCellSizeLevel on the GPU, 15 special positions: nearest places and size levels"
  );
}

}  // namespace
}  // namespace fluidity::grid
