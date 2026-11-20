// Proves that the shared headers grid/grid_build_status.h, grid/grid_ids.h, grid/grid_tables.h and
// grid/weight_curve.h compile as GPU code:
// label no_gpu.
//
// The GPU function below is never launched. Building it in the CUDA container makes the CUDA compiler generate GPU
// code for every function of grid_ids.h and weight_curve.h, the helpers those functions call included, and for
// CellBlockRange, GridFrame, CellState, FacePieceKind, WeightCurveValue and GridBuildStatus read and written by GPU
// code. The integer results go to `results`, and the double results of weight_curve.h go to `double_results`. The CUDA
// compiler generates GPU code for a shared function only where GPU code calls it, so each function is called here once.
// A shared function that calls a CPU-only function fails this build. That build is the proof; the test only takes the
// function's address, which needs no GPU.

#include <gtest/gtest.h>

#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/weight_curve.h"

namespace fluidity::grid {
namespace {

// Reads its inputs from `values` (4 integers), the range, the frame and the two tables, and writes at least one
// integer per shared function to `results` and the doubles of weight_curve.h to `double_results`, so the compiler
// cannot drop any call.
__global__ void ComputeWithEverySharedGridFunction(
    const std::int32_t* values,
    CellBlockRange range,
    GridFrame frame,
    const CellState* cell_states,
    const FacePieceKind* face_piece_kinds,
    GridBuildStatus* build_status,
    std::uint64_t* results,
    double* double_results
) {
  const auto size_level = static_cast<SizeLevel>(values[0] & 7);
  const CellCoord cell_coord = {.x = values[1], .y = values[2], .z = values[3]};
  const auto axis = static_cast<Axis>(values[0] % 3);

  int result_index = 0;
  results[result_index++] = ComputeIsCoordInRange(cell_coord) ? 1 : 0;
  const CellId cell = ToCellId(size_level, cell_coord);
  results[result_index++] = cell.value;
  results[result_index++] = static_cast<std::uint64_t>(ToSizeLevel(cell));
  results[result_index++] = static_cast<std::uint64_t>(ToCellCoord(cell).z);
  const CellBlockId block = ToCellBlockId(cell);
  results[result_index++] = block.value;
  results[result_index++] = ToFacePieceId(axis, size_level, cell_coord).value;
  const CellId small_cell = ComputeSmallCell(cell, values[1] & 7);
  results[result_index++] = small_cell.value;
  results[result_index++] = ComputeBigCell(small_cell).value;
  results[result_index++] = static_cast<std::uint64_t>(ComputeCellEighth(small_cell));
  results[result_index++] = ComesBefore(cell, small_cell) ? 1 : 0;
  results[result_index++] = ComesBefore(block, ToCellBlockId(small_cell)) ? 1 : 0;
  const auto place = static_cast<std::uint32_t>(values[2]);
  results[result_index++] = ToCellBlockIndex(CellPlaceIndex{place}).value - range.first_block;
  results[result_index++] = ToCellBlockIndex(FacePlaceIndex{place}).value + range.block_count;
  // grid_tables.h: its types, read by GPU code.
  results[result_index++] = static_cast<std::uint64_t>(frame.size_level_count);
  results[result_index++] = frame.biggest_cell_width > frame.origin.z ? 1 : 0;
  results[result_index++] = cell_states[place] == CellState::kActive ? 1 : 0;
  results[result_index++] = face_piece_kinds[place] == FacePieceKind::kBetweenCells ? 1 : 0;
  // grid_build_status.h: the record, read and written by GPU code.
  build_status->failure_key = values[0] == 0 ? kNoFailureKey : build_status->failure_key;
  build_status->error = GridError::kBadInput;
  build_status->cell = cell;
  build_status->active_cells_per_size_level[values[0] & 15] = build_status->face_pieces + build_status->cell_blocks;
  results[result_index++] = build_status->failure_key + static_cast<std::uint64_t>(build_status->error);
  // weight_curve.h: the curve, the spread and the constants.
  // weight_curve.h writes doubles, so they go to their own buffer and no double is converted to an integer.
  int double_result_index = 0;
  const WeightCurveValue curve_value = ComputeWeightCurve(frame.origin.x);
  double_results[double_result_index++] = curve_value.weight;
  double_results[double_result_index++] = curve_value.slope;
  double_results[double_result_index++] = ComputeCompleteWeightedSpread(frame.biggest_cell_width);
  double_results[double_result_index++] = kWeightCurveReach;
  double_results[double_result_index++] = kMinWeightSum;
  double_results[double_result_index++] = kMinPivotShare;
  results[result_index++] = static_cast<std::uint64_t>(kMaxNearbyFaces);
}

TEST(SharedGridFunctions, BuiltForTheGpu_HaveAnAddress) {
  const auto function_address = &ComputeWithEverySharedGridFunction;
  EXPECT_NE(function_address, nullptr);
}

}  // namespace
}  // namespace fluidity::grid
