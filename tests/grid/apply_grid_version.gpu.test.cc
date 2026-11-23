// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Binds value buffers to 100 grids built on the GPU in turn: a bind launches and copies nothing, so the values read
// back are the ones written before, each at its own place. The values read back go to WriteResultDigest.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/grid_values.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

constexpr std::uint32_t kRoom = 512;
constexpr std::size_t kGridCount = 100;

// The first kGridCount random lists, built on the GPU as grids of versions 1, 2, 3, ...
std::vector<TestGrid> BuildGpuGridsOfRandomLists() {
  std::vector<TestGrid> grids;
  const GridFrame frame = {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = kRandomGridSizeLevelCount,
  };
  for (std::size_t grid = 0; grid < kGridCount; ++grid) {
    const GridInputs inputs = {
        .frame = frame,
        .active_cells = BuildShuffledRandomGrids().at(grid),
        .version = {.value = grid + 1},
    };
    std::expected<TestGrid, GridFailure> test_grid =
        BuildTestGrid(inputs, {.max_cell_blocks = 0, .run_on = RunOn::kGpu});
    EXPECT_TRUE(test_grid.has_value());
    if (test_grid.has_value()) {
      grids.push_back(std::move(*test_grid));
    }
  }
  return grids;
}

// Binds the values to every grid in turn.
void CheckBindsFollowEveryGrid(CellValues<double>& values, const std::vector<TestGrid>& grids) {
  for (const TestGrid& test_grid : grids) {
    ASSERT_TRUE(ApplyGridVersion(values, test_grid.grid).has_value());
    EXPECT_EQ(values.grid_version().value, test_grid.grid.version().value);
    EXPECT_EQ(values.block_count(), test_grid.grid.counts().cell_blocks);
  }
}

TEST(ApplyGridVersion, OnGpuGrids_KeepEveryValueAtItsPlace) {
  const std::vector<TestGrid> grids = BuildGpuGridsOfRandomLists();
  ASSERT_EQ(grids.size(), kGridCount);
  std::expected<RunMemory, BufferError> run_memory =
      BuildRunMemory(RunOn::kGpu, (std::size_t{kRoom} * kCellPlacesPerBlock * sizeof(double)) + 256);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<CellValues<double>, GridFailure> values = BuildCellValues<double>(RunOn::kGpu, kRoom, *run_memory);
  ASSERT_TRUE(values.has_value());
  std::vector<double> written(std::size_t{kRoom} * kCellPlacesPerBlock);
  for (std::size_t place = 0; place < written.size(); ++place) {
    written[place] = static_cast<double>(place) + 0.5;  // Each place has its own value, exact in a double.
  }
  ASSERT_TRUE(WriteBuffer<double>(written, values->buffer()).has_value());

  CheckBindsFollowEveryGrid(*values, grids);

  std::vector<double> read(written.size(), 0.0);
  ASSERT_TRUE(ReadBuffer<double>(values->buffer(), read).has_value());
  EXPECT_EQ(read, written);
  WriteResultDigest("ApplyGridVersion on 100 GPU grids", std::span<const double>(read));
}

}  // namespace
}  // namespace fluidity::grid
