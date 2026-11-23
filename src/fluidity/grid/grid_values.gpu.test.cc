// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.

#include "fluidity/grid/grid_values.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

// Writes 0.5, 1.5, ... into the bound part on the GPU, reads it back, and writes the digest of what was read.
template <int kPlacesPerBlock>
void CheckWriteThenReadOnGpuReturnsTheSameValues(GridValues<double, kPlacesPerBlock>& values, std::string_view name) {
  std::vector<double> written(std::size_t{values.block_count()} * kPlacesPerBlock);
  for (std::size_t index = 0; index < written.size(); ++index) {
    written[index] = 0.5 + static_cast<double>(index);
  }
  ASSERT_TRUE(WriteValues<double>(written, &values).has_value());
  std::vector<double> read(written.size(), -1.0);
  ASSERT_TRUE(ReadValues(values, std::span<double>(read)).has_value());
  EXPECT_EQ(read, written);
  WriteResultDigest(name, std::span<const double>(read));
}

TEST(WriteValues, OnGpuThenReadValues_ReturnsTheSameValues) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(3);
  const std::expected<TestGrid, GridFailure> test_grid =
      BuildTestGrid(ToInputs(ToUnitFrame(3), cells), {.max_cell_blocks = 0, .run_on = RunOn::kGpu});
  ASSERT_TRUE(test_grid.has_value());
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, std::size_t{1048576});
  ASSERT_TRUE(run_memory.has_value());
  std::expected<CellValues<double>, GridFailure> cell_values = BuildCellValues<double>(test_grid->grid, *run_memory);
  std::expected<FaceValues<double>, GridFailure> face_values = BuildFaceValues<double>(test_grid->grid, *run_memory);
  std::expected<CornerValues<double>, GridFailure> corner_values =
      BuildCornerValues<double>(test_grid->grid, *run_memory);
  std::expected<CellBlockValues<double>, GridFailure> block_values =
      BuildCellBlockValues<double>(test_grid->grid, *run_memory);
  ASSERT_TRUE(cell_values.has_value());
  ASSERT_TRUE(face_values.has_value());
  ASSERT_TRUE(corner_values.has_value());
  ASSERT_TRUE(block_values.has_value());
  CheckWriteThenReadOnGpuReturnsTheSameValues(*cell_values, "WriteValues on the GPU, cells");
  CheckWriteThenReadOnGpuReturnsTheSameValues(*face_values, "WriteValues on the GPU, faces");
  CheckWriteThenReadOnGpuReturnsTheSameValues(*corner_values, "WriteValues on the GPU, corners");
  CheckWriteThenReadOnGpuReturnsTheSameValues(*block_values, "WriteValues on the GPU, blocks");
}

}  // namespace
}  // namespace fluidity::grid
