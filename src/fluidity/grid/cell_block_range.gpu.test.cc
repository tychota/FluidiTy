// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.

#include "fluidity/grid/cell_block_range.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <expected>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

// Pins what holds until the GPU form of the buffer search exists: a grid on the GPU is not searched. The test that
// replaces this one expects the CPU's list.
TEST(FindBorderBlocks, StackedBufferFormOnGpuGrid_IsRunOnCallFailed) {
  const std::vector<CellId> cells = {ToCell(0, 3, 0, 0), ToCell(0, 4, 0, 0)};
  const std::expected<TestGrid, GridFailure> test_grid =
      BuildTestGrid(ToInputs(ToUnitFrame(1), cells), {.run_on = RunOn::kGpu});
  ASSERT_TRUE(test_grid.has_value());
  const std::uint32_t block_count = test_grid->grid.counts().cell_blocks;
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(
      RunOn::kGpu,
      ComputeRunMemoryBytes(
          BuildBorderSearchMemoryEstimate(RunOn::kGpu, block_count),
          ToMemoryCounts(test_grid->grid.counts(), 0),
          MemoryWhere::kRunOn
      )
  );
  ASSERT_TRUE(run_memory.has_value());
  std::expected<BorderSearchBuffers, GridFailure> search_buffers =
      BuildBorderSearchBuffers(RunOn::kGpu, block_count, *run_memory);
  ASSERT_TRUE(search_buffers.has_value());
  std::expected<Buffer<CellBlockIndex>, BufferError> border_so_far =
      BuildBuffer<CellBlockIndex>(RunOn::kGpu, block_count);
  std::expected<Buffer<CellBlockIndex>, BufferError> border_blocks =
      BuildBuffer<CellBlockIndex>(RunOn::kGpu, block_count);
  ASSERT_TRUE(border_so_far.has_value());
  ASSERT_TRUE(border_blocks.has_value());
  const std::expected<std::uint32_t, GridFailure> count = FindBorderBlocks(
      test_grid->grid,
      {.first_block = 0, .block_count = 1},
      *border_so_far,
      0,
      BorderWidth::kTouchingCells,
      *search_buffers,
      *border_blocks
  );
  ASSERT_FALSE(count.has_value());
  EXPECT_EQ(count.error().error, GridError::kRunOnCallFailed);
}

}  // namespace
}  // namespace fluidity::grid
