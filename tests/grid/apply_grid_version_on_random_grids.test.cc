// Binds value buffers to 100 random grids in turn and counts the allocations of each bind.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/grid_values.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/allocation_counter.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

constexpr std::uint32_t kRoom = 4096;
constexpr std::size_t kGridCount = 100;

// The bytes of the four value buffers of `Value = double` with room for kRoom blocks, rounded for the pieces.
constexpr std::size_t kRunMemoryBytes =
    4 * std::size_t{kRoom} * (kCellPlacesPerBlock + kFacePlacesPerBlock + kCornerPlacesPerBlock + 1) * sizeof(double);

// Binds `values` to `grid` and checks the version, the block count and that this one call allocated nothing.
template <int kPlacesPerBlock>
void CheckBindAllocatesNothingAndFollowsTheGrid(
    GridValues<double, kPlacesPerBlock>& values,
    const StaggeredGrid& grid
) {
  const std::uint64_t count_before_bind = ComputeAllocationCount();
  const std::expected<void, GridFailure> bound = ApplyGridVersion(values, grid);
  const std::uint64_t count_after_bind = ComputeAllocationCount();
  ASSERT_TRUE(bound.has_value());
  EXPECT_EQ(count_after_bind - count_before_bind, 0U);
  EXPECT_EQ(values.grid_version().value, grid.version().value);
  EXPECT_EQ(values.block_count(), grid.counts().cell_blocks);
}

// The first kGridCount random lists, built as grids of versions 1, 2, 3, ...
std::vector<TestGrid> BuildGridsOfRandomLists() {
  std::vector<TestGrid> grids;
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  for (std::size_t grid = 0; grid < kGridCount; ++grid) {
    const GridInputs inputs = {
        .frame = frame,
        .active_cells = BuildShuffledRandomGrids().at(grid),
        .version = {.value = grid + 1},
    };
    std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(inputs);
    EXPECT_TRUE(test_grid.has_value());
    if (test_grid.has_value()) {
      grids.push_back(std::move(*test_grid));
    }
  }
  return grids;
}

TEST(ApplyGridVersion, HundredRandomGrids_RebindsWithoutAllocating) {
  const std::vector<TestGrid> grids = BuildGridsOfRandomLists();
  ASSERT_EQ(grids.size(), kGridCount);

  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, kRunMemoryBytes);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<CellValues<double>, GridFailure> cell_values = BuildCellValues<double>(RunOn::kCpu, kRoom, *run_memory);
  std::expected<FaceValues<double>, GridFailure> face_values = BuildFaceValues<double>(RunOn::kCpu, kRoom, *run_memory);
  std::expected<CornerValues<double>, GridFailure> corner_values =
      BuildCornerValues<double>(RunOn::kCpu, kRoom, *run_memory);
  std::expected<CellBlockValues<double>, GridFailure> block_values =
      BuildCellBlockValues<double>(RunOn::kCpu, kRoom, *run_memory);
  ASSERT_TRUE(cell_values.has_value());
  ASSERT_TRUE(face_values.has_value());
  ASSERT_TRUE(corner_values.has_value());
  ASSERT_TRUE(block_values.has_value());

  for (const TestGrid& test_grid : grids) {
    SCOPED_TRACE(test_grid.grid.version().value);
    CheckBindAllocatesNothingAndFollowsTheGrid(*cell_values, test_grid.grid);
    CheckBindAllocatesNothingAndFollowsTheGrid(*face_values, test_grid.grid);
    CheckBindAllocatesNothingAndFollowsTheGrid(*corner_values, test_grid.grid);
    CheckBindAllocatesNothingAndFollowsTheGrid(*block_values, test_grid.grid);
  }
}

}  // namespace
}  // namespace fluidity::grid
