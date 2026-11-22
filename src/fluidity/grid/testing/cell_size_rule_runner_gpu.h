// The cell size rule evaluated for one case, and its GPU runners, for cell_size_rule.gpu.test.cc, private to grid/.
//
// Purpose: run the cell size rule and the size level of a new cell on the GPU with fixed inputs, so the test can
//   compare every result with the CPU's, byte for byte.
//
// Idea: ComputeCellSizeRuleResult computes, for one case, the wanted cell width, its wanted size level and the
//   cell size factor code of the case's cell; ComputeNewCellResult computes, for one position, the nearest committed
//   cell and the size level of a new cell there. Both are marked FLUIDITY_CPU_AND_GPU, so the GPU functions and the
//   CPU loop of the test call the same code. The results hold no padding, so equal results are equal bytes. The
//   runners copy the inputs to the GPU, run one thread per input, and copy the results back.
//   ComputeCellSizeRuleOnGpu also computes the table of cell size factors on the GPU and copies the refinement
//   volume's layers there.
//
// Owns:    CellSizeRuleCase, CellSizeRuleResult, NewCellResult, ComputeCellSizeRuleResult, ComputeNewCellResult,
//          ComputeCellSizeRuleOnGpu, ComputeNewCellSizeLevelsOnGpu
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), RunOnCallStatus (base/run_on_call.h), Vec3 (base/vec3.h),
//          CellSizeFacts, CellSizeSettings, CellWidthsBySizeLevel, CellSizeFactorTable, RefinementFactors,
//          ComputeWantedCellWidth, ComputeWantedSizeLevel, ComputeCellSizeFactorCode, FindNearestCommittedCell,
//          ComputeNewCellSizeLevel (grid/cell_size_rule.h), CellId, SizeLevel (grid/grid_ids.h),
//          GridTables (grid/grid_tables.h)
// Used by: cell_size_rule.gpu.test.cc, and testing/cell_size_rule_runner_gpu.cu, which defines the two runners
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-size-rule>

#pragma once

#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/cell_size_rule.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// One case of the cell size rule: the facts and reference speed of one cell, and the cell whose factor code is read.
struct CellSizeRuleCase {
  CellSizeFacts facts{};
  double reference_speed = 0.0;  // m/s.
  CellId cell{};  // The cell whose cell size factor code is read from the refinement volume.
};

// The results of one case. 16 bytes and no padding, so equal results are equal bytes.
struct CellSizeRuleResult {
  double wanted_width = 0.0;  // m. ComputeWantedCellWidth of the case.
  std::int32_t wanted_size_level = 0;  // ComputeWantedSizeLevel of wanted_width.
  std::int32_t factor_code = 0;  // ComputeCellSizeFactorCode of the case's cell.
};
static_assert(sizeof(CellSizeRuleResult) == 16, "a double and two 32-bit integers, with no padding");

// The results of one position. 8 bytes and no padding.
struct NewCellResult {
  std::uint32_t nearest_place = 0;  // FindNearestCommittedCell, kNoIndex when none.
  std::uint32_t size_level = 0;  // ComputeNewCellSizeLevel.
};
static_assert(sizeof(NewCellResult) == 8, "two 32-bit integers, with no padding");

// Computes the results of one case with the given widths, settings and refinement volume. The factor code is read
// with settings.smallest_size_level.
FLUIDITY_CPU_AND_GPU inline CellSizeRuleResult ComputeCellSizeRuleResult(
    CellSizeRuleCase rule_case,
    const CellWidthsBySizeLevel& widths,
    CellSizeSettings settings,
    const RefinementFactors& volume
) {
  const double wanted_width = ComputeWantedCellWidth(rule_case.facts, rule_case.reference_speed, widths, settings);
  const SizeLevel wanted_size_level = ComputeWantedSizeLevel(wanted_width, widths);
  const int factor_code = ComputeCellSizeFactorCode(volume, settings.smallest_size_level, rule_case.cell);
  return {
      .wanted_width = wanted_width,
      .wanted_size_level = static_cast<std::int32_t>(wanted_size_level),
      .factor_code = factor_code,
  };
}

// Computes the results of one position in a committed grid, on the processor of its tables.
FLUIDITY_CPU_AND_GPU inline NewCellResult
ComputeNewCellResult(const GridTables& committed, Vec3 position, SizeLevel smallest_size_level) {
  return {
      .nearest_place = FindNearestCommittedCell(committed, position).value,
      .size_level = static_cast<std::uint32_t>(ComputeNewCellSizeLevel(committed, position, smallest_size_level)),
  };
}

// Computes ComputeCellSizeRuleResult(cases[i], widths, settings, volume) on the GPU for i below `count`, into the CPU
// array `results`, and ComputeCellSizeFactorTable() on the GPU into the CPU value `table`. `volume` is a refinement
// volume in CPU memory: its layers below layer_count are copied to the GPU first, with the entry counts its counts say.
// Writes `results` and `table` once the GPU work is done; when a later step fails (freeing the GPU memory), the status
// says so and they may already hold the values.
// Requires: count is 1 or more; volume's counts are 1..2^20 and its layer arrays hold the entries its counts say.
// Fails: kOutOfMemory  the GPU had no memory for the inputs, the layers or the results
//        kCallFailed   another GPU call failed
RunOnCallStatus ComputeCellSizeRuleOnGpu(
    const CellSizeRuleCase* cases,
    int count,
    const CellWidthsBySizeLevel& widths,
    CellSizeSettings settings,
    const RefinementFactors& volume,
    CellSizeRuleResult* results,
    CellSizeFactorTable* table
);

// Computes ComputeNewCellResult(committed, positions[i], smallest_size_level) on the GPU for i below `count`, into the
// CPU array `results`. `committed` holds the GPU tables of a built grid (StaggeredGrid::tables(RunOn::kGpu)); the
// positions are in CPU memory and are copied to the GPU first. Writes `results` as ComputeCellSizeRuleOnGpu does.
// Requires: count is 1 or more.
// Fails: kOutOfMemory  the GPU had no memory for the positions or the results
//        kCallFailed   another GPU call failed
RunOnCallStatus ComputeNewCellSizeLevelsOnGpu(
    const GridTables& committed,
    const Vec3* positions,
    int count,
    SizeLevel smallest_size_level,
    NewCellResult* results
);

}  // namespace fluidity::grid
