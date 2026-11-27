// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on a machine with a GPU.
//
// Runs the cell size rule on the GPU and compares every result with the CPU's, bit for bit: the method paper promises
// the same bytes on both processors (<sec-cell-size-rule>). Every GPU result goes to WriteResultDigest under a unique
// name, so the repeat run shows the GPU gave the same bytes twice.

#include "fluidity/grid/cell_size_rule.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/cell_size_rule_cases.h"
#include "fluidity/grid/testing/cell_size_rule_runner_gpu.h"

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

}  // namespace
}  // namespace fluidity::grid
