#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/allocation_counter_test_support.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/grid_test_support.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {
namespace {

// The room of the buffers the random grids are counted and built in: they have at most kMaxRandomGridCellBlocks (252)
// blocks.
constexpr std::uint32_t kRandomGridRoom = 4096;

GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

// The inputs of a CPU list of version 1.
GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

// The bytes of one whole array of grid buffers, as kept in memory: the address and the byte count.
struct ArrayBytes {
  const std::byte* first = nullptr;
  std::size_t byte_count = 0;
};

template <typename Value>
ArrayBytes FindArrayBytes(const Buffer<Value>& array) {
  return {.first = reinterpret_cast<const std::byte*>(array.data()), .byte_count = array.size() * sizeof(Value)};
}

// The six arrays of the grid buffers, whole: every byte of the room, used or not. Their memory stays where it is while
// a grid holds the buffers, so the addresses found before a build still name its arrays after it.
std::vector<ArrayBytes> FindWholeArrays(const GridBuffers& buffers) {
  const GridBuffersArrays& arrays = ToGridBuffersArrays(buffers);
  return {
      FindArrayBytes(arrays.block_ids),
      FindArrayBytes(arrays.same_size_level_neighbour_blocks),
      FindArrayBytes(arrays.bigger_and_smaller_blocks),
      FindArrayBytes(arrays.cell_states),
      FindArrayBytes(arrays.face_piece_kinds),
      FindArrayBytes(arrays.piece_map),
  };
}

// A copy of the bytes of the given arrays, one after the other.
std::vector<std::byte> BuildBytesOfArrays(const std::vector<ArrayBytes>& arrays) {
  std::vector<std::byte> bytes;
  for (const ArrayBytes& array : arrays) {
    bytes.insert(bytes.end(), array.first, array.first + array.byte_count);
  }
  return bytes;
}

// Size-level-0 cells {4 i, 0, 0} for i = 0..count - 1: each alone in a block of its own.
std::vector<CellId> BuildCellsInBlocksOfTheirOwn(std::int32_t count) {
  std::vector<CellId> cells;
  cells.reserve(static_cast<std::size_t>(count));
  for (std::int32_t index = 0; index < count; ++index) {
    cells.push_back(ToCellId(SizeLevel{0}, {.x = 4 * index, .y = 0, .z = 0}));
  }
  return cells;
}

// ---------------------------------------------------------------------------------------------------------------------
// ComputeGridCounts on buffers
// ---------------------------------------------------------------------------------------------------------------------

TEST(ComputeGridCounts, IntoBuffers_EqualsSpanForm) {
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(frame, {}), {.max_cell_blocks = kRandomGridRoom});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    SCOPED_TRACE(grid);
    const GridInputs inputs = ToInputs(frame, BuildShuffledRandomGrids().at(grid));
    const std::expected<GridCounts, GridFailure> span_counts = ComputeGridCounts(inputs);
    const std::expected<GridCounts, GridFailure> buffers_counts = ComputeGridCounts(inputs, buffers);
    ASSERT_TRUE(span_counts.has_value());
    ASSERT_TRUE(buffers_counts.has_value());
    EXPECT_EQ(ToGridCountFields(*buffers_counts), ToGridCountFields(*span_counts));
  }
}

// 4,097 cells 4 apart along x, each alone in its block, into buffers for 4,096 blocks: the blocks are counted against
// the room and none is named. The one-argument form has no room and counts them all.
TEST(ComputeGridCounts, MoreBlocksThanTheRoom_IsTooManyCellBlocksOnBuffers) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(4097);
  const GridInputs inputs = ToInputs(ToUnitFrame(3), cells);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 4096});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);

  const std::expected<GridCounts, GridFailure> buffers_counts = ComputeGridCounts(inputs, buffers);
  ASSERT_FALSE(buffers_counts.has_value());
  EXPECT_EQ(buffers_counts.error().error, GridError::kTooManyCellBlocks);
  EXPECT_EQ(buffers_counts.error().cell.value, 0U);
  EXPECT_EQ(buffers_counts.error().other_cell.value, 0U);

  const std::expected<GridCounts, GridFailure> span_counts = ComputeGridCounts(inputs);
  ASSERT_TRUE(span_counts.has_value());
  EXPECT_EQ(span_counts->cell_blocks, 4097U);
}

// The failed call leaves the buffers with their arrays and their loan: they count and build the first 4,096 cells.
TEST(ComputeGridCounts, FailedOnBuffers_LeavesBuffersUsable) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(4097);
  const GridFrame frame = ToUnitFrame(3);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(frame, {}), {.max_cell_blocks = 4096});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ASSERT_FALSE(ComputeGridCounts(ToInputs(frame, cells), buffers).has_value());

  const GridInputs fitting_inputs = ToInputs(frame, std::span<const CellId>(cells).first(4096));
  const std::expected<GridCounts, GridFailure> counts = ComputeGridCounts(fitting_inputs, buffers);
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(counts->cell_blocks, 4096U);
  const std::expected<StaggeredGrid, GridFailure> grid = BuildGrid(fitting_inputs, buffers);
  ASSERT_TRUE(grid.has_value());
  EXPECT_EQ(ToGridCountFields(grid->counts()), ToGridCountFields(*counts));
}

// Lists with their own corners of the count: an empty list, one block, both coordinate limits on the x axis and a chain
// down to size level 15, each with the frame it needs.
struct Fixture {
  GridFrame frame;
  std::vector<CellId> cells;
};

std::vector<CellId> BuildSmallCellsOfCell(CellId big_cell) {
  const CellCoord coord = ToCellCoord(big_cell);
  const auto small_size_level = static_cast<SizeLevel>(static_cast<int>(ToSizeLevel(big_cell)) + 1);
  std::vector<CellId> small_cells;
  for (std::int32_t dz = 0; dz < 2; ++dz) {
    for (std::int32_t dy = 0; dy < 2; ++dy) {
      for (std::int32_t dx = 0; dx < 2; ++dx) {
        small_cells.push_back(
            ToCellId(small_size_level, {.x = (2 * coord.x) + dx, .y = (2 * coord.y) + dy, .z = (2 * coord.z) + dz})
        );
      }
    }
  }
  return small_cells;
}

// Size level 0 {3,3,3} split down its eighth 7 to size level 15: at each size level 1..14 the 7 small cells other than
// eighth 7, at size level 15 all 8, and one far cell of size level 0.
std::vector<CellId> BuildChainToSizeLevelFifteen() {
  std::vector<CellId> cells;
  CellId cell = ToCellId(SizeLevel{0}, {.x = 3, .y = 3, .z = 3});
  for (int size_level = 0; size_level < 15; ++size_level) {
    const std::vector<CellId> small_cells = BuildSmallCellsOfCell(cell);
    for (std::size_t cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      if (cell_eighth != 7 || size_level == 14) {
        cells.push_back(small_cells.at(cell_eighth));
      }
    }
    cell = small_cells.at(7);
  }
  cells.push_back(ToCellId(SizeLevel{0}, {.x = 262139, .y = 262140, .z = 262140}));
  return cells;
}

std::vector<Fixture> BuildFixtures() {
  std::vector<CellId> limit_cells = {
      ToCellId(SizeLevel{0}, {.x = -kMaxCoord, .y = 0, .z = 0}),
      ToCellId(SizeLevel{0}, {.x = kMaxCoord, .y = 0, .z = 0}),
  };
  const std::vector<CellId> small_cells = BuildSmallCellsOfCell(ToCellId(SizeLevel{0}, {.x = 0, .y = 0, .z = 0}));
  limit_cells.insert(limit_cells.end(), small_cells.begin(), small_cells.end());
  return {
      {.frame = ToUnitFrame(3), .cells = {}},
      {.frame = ToUnitFrame(3), .cells = {ToCellId(SizeLevel{0}, {.x = 0, .y = 0, .z = 0})}},
      {.frame = ToUnitFrame(2), .cells = limit_cells},
      {.frame = ToUnitFrame(16), .cells = BuildChainToSizeLevelFifteen()},
  };
}

TEST(ComputeGridCounts, IntoBuffers_FixturesEqualSpanForm) {
  const std::vector<Fixture> fixtures = BuildFixtures();
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 64});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  for (std::size_t fixture = 0; fixture < fixtures.size(); ++fixture) {
    SCOPED_TRACE(fixture);
    const GridInputs inputs = ToInputs(fixtures.at(fixture).frame, fixtures.at(fixture).cells);
    const std::expected<GridCounts, GridFailure> span_counts = ComputeGridCounts(inputs);
    const std::expected<GridCounts, GridFailure> buffers_counts = ComputeGridCounts(inputs, buffers);
    ASSERT_TRUE(span_counts.has_value());
    ASSERT_TRUE(buffers_counts.has_value());
    EXPECT_EQ(ToGridCountFields(*buffers_counts), ToGridCountFields(*span_counts));
  }
}

// Buffers that were moved from have no arrays and no loan: the lent builder buffers are checked before anything is
// written, as for BuildGrid and CheckSizeStepRule.
TEST(ComputeGridCounts, MovedFromBuffers_IsBadInput) {
  const std::vector<CellId> cells = {ToCellId(SizeLevel{0}, {.x = 0, .y = 0, .z = 0})};
  const GridInputs inputs = ToInputs(ToUnitFrame(3), cells);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  const GridBuffers moved_to = std::move(buffers);

  const std::expected<GridCounts, GridFailure> counts = ComputeGridCounts(inputs, buffers);
  ASSERT_FALSE(counts.has_value());
  EXPECT_EQ(counts.error().error, GridError::kBadInput);
  EXPECT_EQ(counts.error().cell.value, 0U);
  EXPECT_EQ(counts.error().other_cell.value, 0U);
}

// What counting the random grids in turn into one set of buffers found.
struct CountsInTurn {
  std::size_t failed_count = 0;
  std::size_t different_count = 0;  // Grids whose counts differ from the reference counts.
};

// Counts the random grid of each entry into `buffers`, then checks it against the size step rule, in turn, and adds
// what it finds to `found`. Allocates nothing.
void ApplyCountsInTurn(
    const std::vector<std::size_t>& grids,
    const std::vector<GridCounts>& reference_counts,
    GridBuffers& buffers,
    CountsInTurn& found
) {
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  for (std::size_t position = 0; position < grids.size(); ++position) {
    const GridInputs inputs = ToInputs(frame, BuildShuffledRandomGrids()[grids[position]]);
    const std::expected<GridCounts, GridFailure> counts = ComputeGridCounts(inputs, buffers);
    if (!counts.has_value()) {
      ++found.failed_count;
      continue;
    }
    found.different_count += ToGridCountFields(*counts) == ToGridCountFields(reference_counts[position]) ? 0 : 1;
    found.failed_count += CheckSizeStepRule(inputs, buffers).has_value() ? 0 : 1;
  }
}

// The counts of the random grid of each entry by the one-argument form, which needs no buffers.
std::vector<GridCounts> BuildReferenceCounts(const std::vector<std::size_t>& grids) {
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  std::vector<GridCounts> reference_counts;
  reference_counts.reserve(grids.size());
  for (const std::size_t grid : grids) {
    const std::expected<GridCounts, GridFailure> counts =
        ComputeGridCounts(ToInputs(frame, BuildShuffledRandomGrids().at(grid)));
    EXPECT_TRUE(counts.has_value()) << grid;
    if (counts.has_value()) {
      reference_counts.push_back(*counts);
    }
  }
  return reference_counts;
}

// 30 random grids spread by size (FindRandomGridsSpreadBySize: at evenly spaced ranks of the size order, from the grid
// with the most cells to the one with the fewest, the spread that staggered_grid_test.cc uses in place of all 200),
// counted into buffers for 4,096 blocks and checked against the size step rule, one after the other. The reference
// counts are found first, outside the counted part.
TEST(ComputeGridCounts, IntoBuffers_AllocatesNothing) {
  constexpr std::size_t kGridCount = 30;
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  const std::vector<std::size_t> grids = FindRandomGridsSpreadBySize(kGridCount);
  const std::vector<GridCounts> reference_counts = BuildReferenceCounts(grids);
  ASSERT_EQ(reference_counts.size(), kGridCount);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(frame, {}), {.max_cell_blocks = kRandomGridRoom});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);

  const std::uint64_t allocation_count_before = ComputeAllocationCount();
  CountsInTurn counts_in_turn;
  ApplyCountsInTurn(grids, reference_counts, buffers, counts_in_turn);
  const std::uint64_t allocation_count_after = ComputeAllocationCount();

  EXPECT_EQ(allocation_count_after - allocation_count_before, 0U);
  EXPECT_EQ(counts_in_turn.failed_count, 0U);
  EXPECT_EQ(counts_in_turn.different_count, 0U);
}

// 9 random grids spread by size (FindRandomGridsSpreadBySize), from the one with the most cells to the one with the
// fewest.
TEST(ComputeGridCounts, SpanForm_AllocatesOneArray) {
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  for (const std::size_t grid : FindRandomGridsSpreadBySize(9)) {
    SCOPED_TRACE(grid);
    const GridInputs inputs = ToInputs(frame, BuildShuffledRandomGrids().at(grid));
    const std::uint64_t allocation_count_before = ComputeAllocationCount();
    const std::expected<GridCounts, GridFailure> counts = ComputeGridCounts(inputs);
    const std::uint64_t allocation_count_after = ComputeAllocationCount();
    ASSERT_TRUE(counts.has_value());
    EXPECT_EQ(allocation_count_after - allocation_count_before, 1U);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// One set of builder buffers shared by two grids
// ---------------------------------------------------------------------------------------------------------------------

// Two sets of grid buffers of kRandomGridRoom blocks that lend one set of builder buffers, with their memory. Members
// are destroyed in reverse order: the buffers, then the builder buffers they borrow, then the memory.
struct SharedBuffers {
  std::unique_ptr<RunMemory> run_memory;
  std::unique_ptr<GridBuilderBuffers> builder_buffers;
  GridBuffers first;
  GridBuffers second;
};

// Builds the memory of exactly the estimate of two grids that share one set of builder buffers, then the two buffers.
std::optional<SharedBuffers> BuildSharedBuffers() {
  const std::vector<MemoryEstimateEntry> estimate =
      BuildGridMemoryEstimate(RunOn::kCpu, 2, kRandomGridRoom, BuilderBuffersSharing::kOneForAllGrids);
  GridCounts room_counts{};
  room_counts.cell_blocks = kRandomGridRoom;
  const MemoryCounts memory_counts = ToMemoryCounts(room_counts, 0);
  const std::size_t reserved_bytes = ComputeRunMemoryBytes(estimate, memory_counts, MemoryWhere::kRunOn)
      + ComputeRunMemoryBytes(estimate, memory_counts, MemoryWhere::kCpu);
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, reserved_bytes);
  if (!run_memory.has_value()) {
    ADD_FAILURE() << "BuildRunMemory failed";
    return std::nullopt;
  }
  auto shared_run_memory = std::make_unique<RunMemory>(std::move(*run_memory));
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, kRandomGridRoom, *shared_run_memory);
  if (!builder_buffers.has_value()) {
    ADD_FAILURE() << "BuildGridBuilderBuffers failed";
    return std::nullopt;
  }
  auto shared_builder_buffers = std::make_unique<GridBuilderBuffers>(std::move(*builder_buffers));
  const GridSettings settings = {.max_cell_blocks = kRandomGridRoom, .run_on = RunOn::kCpu};
  std::expected<GridBuffers, GridFailure> first =
      BuildGridBuffers(settings, *shared_builder_buffers, *shared_run_memory);
  std::expected<GridBuffers, GridFailure> second =
      BuildGridBuffers(settings, *shared_builder_buffers, *shared_run_memory);
  if (!first.has_value() || !second.has_value()) {
    ADD_FAILURE() << "BuildGridBuffers failed";
    return std::nullopt;
  }
  return SharedBuffers{
      .run_memory = std::move(shared_run_memory),
      .builder_buffers = std::move(shared_builder_buffers),
      .first = std::move(*first),
      .second = std::move(*second),
  };
}

// Counts and builds a list with a repeated cell into `buffers`: both fail, and the buffers stay as they were. Adds the
// calls that did not fail to `wrong_count`.
void ApplyFailedCallsOnBuffers(const GridInputs& inputs, GridBuffers& buffers, std::size_t& wrong_count) {
  std::vector<CellId> repeated_cells(inputs.active_cells.begin(), inputs.active_cells.end());
  repeated_cells.push_back(repeated_cells.front());
  const GridInputs repeated_inputs = ToInputs(inputs.frame, repeated_cells);
  wrong_count += ComputeGridCounts(repeated_inputs, buffers).has_value() ? 1 : 0;
  std::expected<StaggeredGrid, GridFailure> built = BuildGrid(repeated_inputs, buffers);
  if (built.has_value()) {
    ++wrong_count;
    buffers = TakeBuffers(std::move(*built));
  }
}

// In turn for the random grid of each entry, into `buffers`: a failed count and a failed build, then a count, a check
// and a build that must equal its reference grid (same position), then the buffers taken back. Adds the steps that went
// wrong to `wrong_count`.
void ApplyBuildsInTurn(
    const std::vector<std::size_t>& grids,
    const std::vector<TestGrid>& reference_grids,
    GridBuffers& buffers,
    std::size_t& wrong_count
) {
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  for (std::size_t position = 0; position < reference_grids.size(); ++position) {
    const GridInputs inputs = ToInputs(frame, BuildShuffledRandomGrids().at(grids.at(position)));
    ApplyFailedCallsOnBuffers(inputs, buffers, wrong_count);
    wrong_count += ComputeGridCounts(inputs, buffers).has_value() ? 0 : 1;
    wrong_count += CheckSizeStepRule(inputs, buffers).has_value() ? 0 : 1;
    std::expected<StaggeredGrid, GridFailure> built = BuildGrid(inputs, buffers);
    if (!built.has_value()) {
      ++wrong_count;
      continue;
    }
    wrong_count += ComputeIsSameGrid(*built, reference_grids.at(position).grid) ? 0 : 1;
    buffers = TakeBuffers(std::move(*built));
  }
}

// Which of the two shared buffers holds grid A.
enum class BuffersOfGridA : std::uint8_t {
  kFirst,
  kSecond,
};

// The buffers that hold grid A, and the other ones.
GridBuffers& FindBuffersOfGridA(SharedBuffers& shared, BuffersOfGridA buffers_of_grid_a) {
  return buffers_of_grid_a == BuffersOfGridA::kFirst ? shared.first : shared.second;
}

GridBuffers& FindOtherBuffers(SharedBuffers& shared, BuffersOfGridA buffers_of_grid_a) {
  return buffers_of_grid_a == BuffersOfGridA::kFirst ? shared.second : shared.first;
}

// Grid A is built into the buffers `buffers_of_grid_a` names and kept. The other buffers lend the same builder buffers:
// 30 random grids spread by size (FindRandomGridsSpreadBySize, the spread that staggered_grid_test.cc uses in place of
// all 200) are each counted, checked against the size step rule, built and taken back in turn, with a failed count and
// a failed build before each. The builder buffers hold nothing between builds, so every byte of A's six arrays, the
// unused room included, and A's counts are the same before and after, and each grid built into the other buffers equals
// the same grid built with memory of its own.
void ExpectOtherGridUnchanged(SharedBuffers& shared, BuffersOfGridA buffers_of_grid_a) {
  GridBuffers& a_buffers = FindBuffersOfGridA(shared, buffers_of_grid_a);
  GridBuffers& other_buffers = FindOtherBuffers(shared, buffers_of_grid_a);
  constexpr std::size_t kGridCount = 30;
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  const std::vector<std::size_t> grids = FindRandomGridsSpreadBySize(kGridCount);
  const std::vector<TestGrid> reference_grids = BuildReferenceGrids(grids);
  ASSERT_EQ(reference_grids.size(), kGridCount);

  // Grid A: the biggest random grid.
  const std::vector<ArrayBytes> a_arrays = FindWholeArrays(a_buffers);
  std::expected<StaggeredGrid, GridFailure> grid_a =
      BuildGrid(ToInputs(frame, BuildShuffledRandomGrids().at(grids.front())), a_buffers);
  ASSERT_TRUE(grid_a.has_value());
  const std::vector<std::byte> bytes_before = BuildBytesOfArrays(a_arrays);
  const auto counts_before = ToGridCountFields(grid_a->counts());
  ASSERT_FALSE(bytes_before.empty());

  std::size_t wrong_count = 0;
  ApplyBuildsInTurn(grids, reference_grids, other_buffers, wrong_count);

  EXPECT_EQ(wrong_count, 0U);
  EXPECT_TRUE(BuildBytesOfArrays(a_arrays) == bytes_before);
  EXPECT_TRUE(ToGridCountFields(grid_a->counts()) == counts_before);
}

TEST(BuildGrid, SharedBuilderBuffers_OtherLiveGridUnchanged) {
  std::optional<SharedBuffers> shared = BuildSharedBuffers();
  ASSERT_TRUE(shared.has_value());
  ASSERT_EQ(shared->builder_buffers->lent_count(), 2U);
  ExpectOtherGridUnchanged(*shared, BuffersOfGridA::kFirst);
}

// The same with the two buffers swapped: whichever buffers hold the live grid, the other's work leaves it alone.
TEST(BuildGrid, SharedBuilderBuffers_OtherLiveGridUnchangedInReverseOrder) {
  std::optional<SharedBuffers> shared = BuildSharedBuffers();
  ASSERT_TRUE(shared.has_value());
  ExpectOtherGridUnchanged(*shared, BuffersOfGridA::kSecond);
}

}  // namespace
}  // namespace fluidity::grid
