#include "fluidity/grid/staggered_grid.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <new>
#include <span>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/allocation_counter.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

TEST(BuildTestGrid, DefaultSettings_SizedToTheList) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(9);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), cells));
  ASSERT_TRUE(test_grid.has_value());
  EXPECT_EQ(test_grid->grid.max_cell_blocks(), 9U);
  EXPECT_EQ(test_grid->grid.counts().cell_blocks, 9U);
}

TEST(StaggeredGrid, CpuGrid_HasNoGpuTables) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(2);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables gpu_tables = test_grid->grid.tables(RunOn::kGpu);
  EXPECT_EQ(gpu_tables.version.value, 0U);
  EXPECT_EQ(gpu_tables.block_ids, nullptr);
  EXPECT_EQ(gpu_tables.same_size_level_blocks, nullptr);
  EXPECT_EQ(gpu_tables.bigger_block, nullptr);
  EXPECT_EQ(gpu_tables.smaller_blocks, nullptr);
  EXPECT_EQ(gpu_tables.cell_states, nullptr);
  EXPECT_EQ(gpu_tables.face_piece_kinds, nullptr);
}

// A request whose aligned block would need more than SIZE_MAX bytes fails as its form says, counted, rather than
// wrapping the size to a small block: null from the nothrow form, std::bad_alloc from the throwing one.
TEST(ComputeAllocationCount, AlignedRequestNearSizeMax_FailsWithoutWrapping) {
  // Read through a volatile, so the compiler sees no constant size to warn about.
  const volatile std::size_t near_size_max = std::numeric_limits<std::size_t>::max() - 8;
  const std::size_t byte_count = near_size_max;
  constexpr std::align_val_t kAlignment{64};
  const std::uint64_t allocation_count_before = ComputeAllocationCount();
  void* const memory = ::operator new(byte_count, kAlignment, std::nothrow);
  EXPECT_EQ(memory, nullptr);
  EXPECT_THROW(::operator delete(::operator new(byte_count, kAlignment), kAlignment), std::bad_alloc);
  EXPECT_EQ(ComputeAllocationCount() - allocation_count_before, 2U);
}

}  // namespace
}  // namespace fluidity::grid
