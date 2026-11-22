#include "fluidity/grid/staggered_grid.h"

#include <gtest/gtest.h>

#include <expected>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// A machine without a GPU cannot reserve the GPU memory of a GPU grid: the first call that needs the GPU fails, and no
// grid is returned.
TEST(BuildTestGrid, OnGpuWithoutAGpu_IsRunOnCallFailed) {
  const std::vector<CellId> cells = {ToCellId(SizeLevel{0}, {.x = 0, .y = 0, .z = 0})};
  const GridInputs inputs = {
      .frame = {.origin = {.x = 0, .y = 0, .z = 0}, .biggest_cell_width = 1.0, .size_level_count = 3},
      .active_cells = cells,
      .version = {.value = 1},
  };
  const std::expected<TestGrid, GridFailure> test_grid =
      BuildTestGrid(inputs, {.max_cell_blocks = 0, .run_on = RunOn::kGpu});
  ASSERT_FALSE(test_grid.has_value());
  EXPECT_EQ(test_grid.error().error, GridError::kRunOnCallFailed);
}

}  // namespace
}  // namespace fluidity::grid
