// Proves that the shared headers grid/grid_ids.h and grid/grid_tables.h compile as GPU code: label no_gpu.
//
// The GPU function below is never launched. Building it in the CUDA container makes the CUDA compiler generate GPU
// code for every function of grid_ids.h, the helpers the header's functions call included, and for CellBlockRange,
// GridFrame, CellState and FacePieceKind read and written by GPU code. The CUDA compiler generates GPU code for a
// shared function only where GPU code calls it, so each function is called here once. A shared function that calls
// a CPU-only function fails this build. That build is the proof; the test only takes the function's address, which
// needs no GPU.

#include <gtest/gtest.h>

#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {
namespace {

// Reads its inputs from `values` (4 integers), the range, the frame and the two tables, and writes one integer per
// shared function to `results`, so the compiler cannot drop any call.
__global__ void ComputeWithEveryGridIdFunction(
    const std::int32_t* values,
    CellBlockRange range,
    GridFrame frame,
    const CellState* cell_states,
    const FacePieceKind* face_piece_kinds,
    std::uint64_t* results
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
}

TEST(CellBlockRange, BuiltForTheGpu_HasAnAddress) {
  const auto function_address = &ComputeWithEveryGridIdFunction;
  EXPECT_NE(function_address, nullptr);
}

}  // namespace
}  // namespace fluidity::grid
