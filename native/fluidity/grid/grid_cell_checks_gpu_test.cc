// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Checks lists of active cells on the GPU (<proc-build-grid-in-parallel> P1 to P6 and the failure half of P12) and
// compares every result with the CPU's: the same failure (error, cell, other_cell) or the same pass, and the same
// sorted list. The lists are put on the GPU with WriteBuffer before each check, so the check reads them there. Every
// GPU result goes to WriteResultDigest under a unique name, so the repeat run shows the GPU gave the same results
// twice: the failure the call returns, followed, for every call that ran on the GPU, by the status record the GPU wrote
// (its key, error, cell and other_cell), read back by the test; and the sorted lists.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/processor_description.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/test_support.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_cell_checks.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/grid_test_support.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/staggered_grid_gpu.h"

namespace fluidity::grid {
namespace {

// The room of the buffers of every test here but the one of 1 block: 262,144 cell places, more than any list here.
constexpr std::uint32_t kListRoom = 4096;

GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// GPU grid buffers with the builder buffers lent to them and a GPU buffer for a list of up to 64 cells per block, in a
// RunMemory of exactly their estimate plus the list. Kept behind pointers, since lent builder buffers must not move;
// the members are destroyed in reverse order, the RunMemory last.
struct GpuGridBuffers {
  std::unique_ptr<RunMemory> run_memory;
  std::unique_ptr<GridBuilderBuffers> builder_buffers;
  std::unique_ptr<GridBuffers> buffers;
  Buffer<CellId> cell_list;
};

// Builds GpuGridBuffers of max_cell_blocks blocks. Null, with a test failure, when a builder fails.
std::unique_ptr<GpuGridBuffers> BuildGpuGridBuffers(std::uint32_t max_cell_blocks) {
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kGpu, 1, max_cell_blocks);
  GridCounts room_counts = {};
  room_counts.cell_blocks = max_cell_blocks;
  const MemoryCounts memory_counts = ToMemoryCounts(room_counts, 0);
  const std::size_t list_count = std::size_t{kCellPlacesPerBlock} * max_cell_blocks;
  const std::size_t byte_count = ComputeRunMemoryBytes(estimate, memory_counts, MemoryWhere::kRunOn)
      + ComputeRunMemoryPieceBytes(list_count * sizeof(CellId));
  auto built = std::make_unique<GpuGridBuffers>();
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, byte_count);
  if (!run_memory.has_value()) {
    ADD_FAILURE() << "BuildRunMemory failed";
    return nullptr;
  }
  built->run_memory = std::make_unique<RunMemory>(std::move(*run_memory));
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kGpu, max_cell_blocks, *built->run_memory);
  if (!builder_buffers.has_value()) {
    ADD_FAILURE() << "BuildGridBuilderBuffers failed";
    return nullptr;
  }
  built->builder_buffers = std::make_unique<GridBuilderBuffers>(std::move(*builder_buffers));
  const GridSettings settings = {.max_cell_blocks = max_cell_blocks, .run_on = RunOn::kGpu};
  std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers(settings, *built->builder_buffers, *built->run_memory);
  if (!buffers.has_value()) {
    ADD_FAILURE() << "BuildGridBuffers failed";
    return nullptr;
  }
  built->buffers = std::make_unique<GridBuffers>(std::move(*buffers));
  std::expected<Buffer<CellId>, BufferError> cell_list =
      BuildBuffer<CellId>(RunOn::kGpu, list_count, *built->run_memory);
  if (!cell_list.has_value()) {
    ADD_FAILURE() << "BuildBuffer of the list failed";
    return nullptr;
  }
  built->cell_list = std::move(*cell_list);
  return built;
}

// The inputs of a list in the GPU buffer of `gpu_buffers`, after writing `cells` there: the test's setup, not part of
// the check.
GridInputs ToGpuInputs(GridFrame frame, std::span<const CellId> cells, GpuGridBuffers& gpu_buffers) {
  if (!cells.empty()) {
    const std::expected<void, BufferError> written = WriteBuffer(cells, gpu_buffers.cell_list, 0);
    EXPECT_TRUE(written.has_value());
  }
  return {
      .frame = frame,
      .active_cells = {},
      .active_cell_buffer = &gpu_buffers.cell_list,
      .active_cell_buffer_count = static_cast<std::uint32_t>(cells.size()),
      .version = {.value = 1},
  };
}

// The failure of a result as the four words the digest keeps: error, cell, other_cell, magnitude (zeros on success).
std::array<std::uint64_t, 4> ToFailureWords(const std::expected<void, GridFailure>& result) {
  const GridFailure failure = result.has_value() ? GridFailure{} : result.error();
  return {static_cast<std::uint64_t>(failure.error), failure.cell.value, failure.other_cell.value, failure.magnitude};
}

// Appends the failure words of `result` to digest_words, then the key, error, cell and other_cell of the status record
// the GPU wrote into `buffers` during the call.
void AppendResultWords(
    const std::expected<void, GridFailure>& result,
    const GridBuffers& buffers,
    std::vector<std::uint64_t>& digest_words
) {
  const std::array<std::uint64_t, 4> words = ToFailureWords(result);
  digest_words.insert(digest_words.end(), words.begin(), words.end());
  GridBuildStatus status{};
  const std::expected<void, BufferError> read =
      ReadBuffer(ToGridBuffersArrays(buffers).build_status, 0, std::span<GridBuildStatus>(&status, 1));
  EXPECT_TRUE(read.has_value());
  digest_words.push_back(status.failure_key);
  digest_words.push_back(static_cast<std::uint64_t>(status.error));
  digest_words.push_back(status.cell.value);
  digest_words.push_back(status.other_cell.value);
}

// A CPU grid of no cells in a room of kListRoom blocks and `size_level_count` size levels, whose buffers
// TakeEmptyGridBuffers hands to a test as the CPU's side of a comparison.
std::expected<TestGrid, GridFailure> BuildCpuBuffersOwner(int size_level_count, std::uint32_t max_cell_blocks) {
  return BuildTestGrid(
      {.frame = ToUnitFrame(size_level_count), .active_cells = {}, .version = {.value = 1}},
      {.max_cell_blocks = max_cell_blocks}
  );
}

std::expected<void, GridFailure> ToCheckResult(const std::expected<std::uint32_t, GridFailure>& built) {
  if (!built.has_value()) {
    return std::unexpected(built.error());
  }
  return {};
}

// The three lists two size levels apart: size level 0 {0,0,0} and a size-level-2 cell beyond its high corner, beyond
// its edge and beyond its side.
std::vector<std::vector<CellId>> BuildTwoSizeLevelsApartLists() {
  return {
      {ToCell(0, 0, 0, 0), ToCell(2, 4, 4, 4)},
      {ToCell(0, 0, 0, 0), ToCell(2, 4, 4, 0)},
      {ToCell(0, 0, 0, 0), ToCell(2, 4, 0, 0)},
  };
}

TEST(CheckSizeStepRule, OnGpu_EqualsCpu) {
  const std::unique_ptr<GpuGridBuffers> gpu_buffers = BuildGpuGridBuffers(kListRoom);
  ASSERT_NE(gpu_buffers, nullptr);
  std::vector<std::pair<GridFrame, std::vector<CellId>>> lists;
  for (std::vector<CellId>& cells : BuildTwoSizeLevelsApartLists()) {
    lists.emplace_back(ToUnitFrame(3), std::move(cells));
  }
  // One size level apart, siblings complete with the size step rule broken (BuildSizeStepBreakWithSiblings), and the
  // empty list.
  lists.emplace_back(ToUnitFrame(3), std::vector<CellId>{ToCell(0, 0, 0, 0), ToCell(1, 2, 0, 0)});
  lists.emplace_back(ToUnitFrame(3), BuildSizeStepBreakWithSiblings());
  lists.emplace_back(ToUnitFrame(3), std::vector<CellId>{});
  for (const std::vector<CellId>& cells : BuildShuffledRandomGrids()) {
    lists.emplace_back(ToUnitFrame(4), cells);
  }
  std::vector<std::uint64_t> digest_words;
  for (std::size_t list = 0; list < lists.size(); ++list) {
    const auto& [frame, cells] = lists.at(list);
    const std::expected<void, GridFailure> gpu_result =
        CheckSizeStepRule(ToGpuInputs(frame, cells, *gpu_buffers), *gpu_buffers->buffers);
    EXPECT_EQ(ToFailureWords(gpu_result), ToFailureWords(CheckSizeStepRule(frame, cells))) << list;
    AppendResultWords(gpu_result, *gpu_buffers->buffers, digest_words);
  }
  WriteResultDigest(
      "CheckSizeStepRule on the GPU, fixed and random lists",
      std::span<const std::uint64_t>(digest_words)
  );
}

// Bits 61..62 are never set in a cell id: step 1 names the id as given.
TEST(CheckSizeStepRule, OnGpuIdWithBitSixtyOneSet_IsBadInputNamingIt) {
  const std::unique_ptr<GpuGridBuffers> gpu_buffers = BuildGpuGridBuffers(kListRoom);
  ASSERT_NE(gpu_buffers, nullptr);
  std::vector<CellId> cells = BuildShuffledRandomGrids().at(0);
  const CellId bad_id = {.value = cells.at(cells.size() / 2).value | (std::uint64_t{1} << 61U)};
  cells.at(cells.size() / 2) = bad_id;
  const std::expected<void, GridFailure> gpu_result =
      CheckSizeStepRule(ToGpuInputs(ToUnitFrame(4), cells, *gpu_buffers), *gpu_buffers->buffers);
  ASSERT_FALSE(gpu_result.has_value());
  EXPECT_EQ(gpu_result.error().error, GridError::kBadInput);
  EXPECT_EQ(gpu_result.error().cell.value, bad_id.value);
  EXPECT_EQ(ToFailureWords(gpu_result), ToFailureWords(CheckSizeStepRule(ToUnitFrame(4), cells)));
  std::vector<std::uint64_t> digest_words;
  AppendResultWords(gpu_result, *gpu_buffers->buffers, digest_words);
  WriteResultDigest("CheckSizeStepRule on the GPU, id with bit 61 set", std::span<const std::uint64_t>(digest_words));
}

// 65 cells do not fit the 64 cell places of 1 block, whatever their ids: the length is an input check, so the bad id at
// index 0 is never read and nothing runs on the GPU. The list lies in a GPU buffer of its own, 65 entries.
TEST(CheckSizeStepRule, OnGpuTooLongListWithBadId_SameFailureAsCpu) {
  const std::unique_ptr<GpuGridBuffers> gpu_buffers = BuildGpuGridBuffers(1);
  ASSERT_NE(gpu_buffers, nullptr);
  std::vector<CellId> cells;
  cells.reserve(65);
  for (std::int32_t x = 0; x < 65; ++x) {
    cells.push_back(ToCell(0, x, 0, 0));
  }
  cells.at(0) = {.value = cells.at(0).value & ~(std::uint64_t{1} << 63U)};
  const Buffer<CellId> long_list = BuildBufferHolding(RunOn::kGpu, cells);
  ASSERT_EQ(long_list.size(), cells.size());
  const GridInputs gpu_inputs = {
      .frame = ToUnitFrame(1),
      .active_cells = {},
      .active_cell_buffer = &long_list,
      .active_cell_buffer_count = static_cast<std::uint32_t>(cells.size()),
      .version = {.value = 1},
  };
  const std::expected<void, GridFailure> gpu_result = CheckSizeStepRule(gpu_inputs, *gpu_buffers->buffers);
  const std::array<std::uint64_t, 4> too_many_cell_blocks = {
      static_cast<std::uint64_t>(GridError::kTooManyCellBlocks),
      0,
      0,
      0,
  };
  EXPECT_EQ(ToFailureWords(gpu_result), too_many_cell_blocks);
  std::expected<TestGrid, GridFailure> cpu_grid = BuildCpuBuffersOwner(1, 1);
  GridBuffers cpu_buffers = TakeEmptyGridBuffers(cpu_grid);
  const GridInputs cpu_inputs = {.frame = ToUnitFrame(1), .active_cells = cells, .version = {.value = 1}};
  EXPECT_EQ(ToFailureWords(gpu_result), ToFailureWords(CheckSizeStepRule(cpu_inputs, cpu_buffers)));
  // Nothing ran on the GPU, so the status record holds nothing of this call and is not in the digest.
  const std::array<std::uint64_t, 4> words = ToFailureWords(gpu_result);
  WriteResultDigest("CheckSizeStepRule on the GPU, too long list", std::span<const std::uint64_t>(words));
}

// Checks the 500 mutated lists on the GPU with both sibling checks and returns the failure words in order, expecting
// each to equal the CPU's.
std::vector<std::uint64_t> ComputeMutatedListFailuresOnGpu(GpuGridBuffers& gpu_buffers) {
  std::expected<TestGrid, GridFailure> cpu_grid = BuildCpuBuffersOwner(4, kListRoom);
  GridBuffers cpu_buffers = TakeEmptyGridBuffers(cpu_grid);
  std::vector<std::uint64_t> digest_words;
  for (std::uint64_t seed = 1; seed <= kMutatedGridCount; ++seed) {
    const std::vector<CellId> cells = BuildMutatedGridCells(seed);
    for (const SiblingCheck sibling_check : {SiblingCheck::kRequire, SiblingCheck::kSkip}) {
      const std::expected<void, GridFailure> gpu_result = ToCheckResult(
          BuildSortedCellList(ToGpuInputs(ToUnitFrame(4), cells, gpu_buffers), *gpu_buffers.buffers, sibling_check)
      );
      const GridInputs cpu_inputs = {.frame = ToUnitFrame(4), .active_cells = cells, .version = {.value = 1}};
      const std::expected<void, GridFailure> cpu_result =
          ToCheckResult(BuildSortedCellList(cpu_inputs, cpu_buffers, sibling_check));
      EXPECT_EQ(ToFailureWords(gpu_result), ToFailureWords(cpu_result)) << seed;
      AppendResultWords(gpu_result, *gpu_buffers.buffers, digest_words);
    }
  }
  return digest_words;
}

TEST(CheckSizeStepRule, OnGpuSeveralFailures_SameFirstFailureAsCpu) {
  const std::unique_ptr<GpuGridBuffers> gpu_buffers = BuildGpuGridBuffers(kListRoom);
  ASSERT_NE(gpu_buffers, nullptr);
  const std::vector<std::uint64_t> digest_words = ComputeMutatedListFailuresOnGpu(*gpu_buffers);
  WriteResultDigest("CheckSizeStepRule on the GPU, mutated lists", std::span<const std::uint64_t>(digest_words));
}

TEST(CheckSizeStepRule, OnGpuTwice_GivesIdenticalFailure) {
  const std::unique_ptr<GpuGridBuffers> gpu_buffers = BuildGpuGridBuffers(kListRoom);
  ASSERT_NE(gpu_buffers, nullptr);
  const std::vector<std::uint64_t> first_words = ComputeMutatedListFailuresOnGpu(*gpu_buffers);
  const std::vector<std::uint64_t> second_words = ComputeMutatedListFailuresOnGpu(*gpu_buffers);
  EXPECT_EQ(first_words, second_words);
  WriteResultDigest("CheckSizeStepRule on the GPU, mutated lists twice", std::span<const std::uint64_t>(second_words));
}

// Checks `inputs` call_count times and returns the words AppendResultWords gives every call, in order. Each call must
// pass. The words are reserved first, so the loop allocates nothing on the CPU either.
std::vector<std::uint64_t> ComputeRepeatedCheckWords(const GridInputs& inputs, GridBuffers& buffers, int call_count) {
  std::vector<std::uint64_t> digest_words;
  digest_words.reserve(8 * static_cast<std::size_t>(call_count));
  for (int call = 0; call < call_count; ++call) {
    const std::expected<void, GridFailure> result = CheckSizeStepRule(inputs, buffers);
    EXPECT_TRUE(result.has_value()) << call;
    AppendResultWords(result, buffers, digest_words);
  }
  return digest_words;
}

TEST(CheckSizeStepRule, OnGpu_AllocatesNothing) {
  const std::unique_ptr<GpuGridBuffers> gpu_buffers = BuildGpuGridBuffers(kListRoom);
  ASSERT_NE(gpu_buffers, nullptr);
  const std::vector<CellId>& cells = BuildShuffledRandomGrids().at(0);
  const GridInputs inputs = ToGpuInputs(ToUnitFrame(4), cells, *gpu_buffers);
  ASSERT_TRUE(CheckSizeStepRule(inputs, *gpu_buffers->buffers).has_value());
  const std::expected<MemoryAvailable, BufferError> before = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(before.has_value());
  const std::vector<std::uint64_t> digest_words = ComputeRepeatedCheckWords(inputs, *gpu_buffers->buffers, 100);
  const std::expected<MemoryAvailable, BufferError> after = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(after->free_bytes, before->free_bytes);
  WriteResultDigest("CheckSizeStepRule on the GPU, 100 calls", std::span<const std::uint64_t>(digest_words));
}

// The grid's memory estimate has no entry held by the driver: every GPU function of the check keeps under the
// driver's default stack of 1,024 bytes per thread. localSizeBytes counts a function's own frame; it covers the shared
// functions it calls because the compiler inlines them all (ptxas of the release build reports no separate device
// function), which a debug build with -G does not do.
TEST(CheckSizeStepRule, OnGpuFunctions_KeepUnderTheDefaultStack) {
  std::uint64_t largest_bytes = 0;
  ASSERT_EQ(ComputeLargestStackOnGpu(&largest_bytes), RunOnCallStatus::kDone);
  RecordProperty("largest_stack_bytes", static_cast<int>(largest_bytes));
  EXPECT_LT(largest_bytes, 1024U);
  WriteResultDigest(
      "CheckSizeStepRule GPU functions, largest stack",
      std::span<const std::uint64_t>(&largest_bytes, 1)
  );
}

// Reads the first `count` cell ids of the lent builder buffers of `buffers` back to the CPU, as values.
std::vector<std::uint64_t> ReadSortedValues(const GridBuffers& buffers, std::size_t count) {
  const Buffer<CellId>& cell_ids = ToGridBuilderBuffersArrays(*ToGridBuffersArrays(buffers).builder_buffers).cell_ids;
  std::vector<CellId> sorted_cells(count);
  const std::expected<void, BufferError> read = ReadBuffer(cell_ids, 0, std::span<CellId>(sorted_cells));
  EXPECT_TRUE(read.has_value());
  std::vector<std::uint64_t> values;
  values.reserve(count);
  for (const CellId cell : sorted_cells) {
    values.push_back(cell.value);
  }
  return values;
}

// Checks `cells` on both processors with siblings required and expects them to pass with the same sorted list, which
// it appends to digest_words.
void ExpectSameSortedList(
    GridFrame frame,
    std::span<const CellId> cells,
    GpuGridBuffers& gpu_buffers,
    GridBuffers& cpu_buffers,
    std::vector<std::uint64_t>& digest_words
) {
  const std::expected<std::uint32_t, GridFailure> gpu_result =
      BuildSortedCellList(ToGpuInputs(frame, cells, gpu_buffers), *gpu_buffers.buffers, SiblingCheck::kRequire);
  const std::expected<std::uint32_t, GridFailure> cpu_result = BuildSortedCellList(
      {.frame = frame, .active_cells = cells, .version = {.value = 1}},
      cpu_buffers,
      SiblingCheck::kRequire
  );
  ASSERT_TRUE(gpu_result.has_value());
  ASSERT_TRUE(cpu_result.has_value());
  const std::vector<std::uint64_t> gpu_sorted = ReadSortedValues(*gpu_buffers.buffers, cells.size());
  EXPECT_EQ(gpu_sorted, ReadSortedValues(cpu_buffers, cells.size())) << cells.size();
  digest_words.insert(digest_words.end(), gpu_sorted.begin(), gpu_sorted.end());
}

// Checks `cells`, a list that fails in a frame of 3 size levels, on the GPU and expects it to fail, so the GPU buffers
// then hold the values of a failed call.
void ApplyFailingCheck(std::span<const CellId> cells, GpuGridBuffers& gpu_buffers) {
  const std::expected<void, GridFailure> result =
      CheckSizeStepRule(ToGpuInputs(ToUnitFrame(3), cells, gpu_buffers), *gpu_buffers.buffers);
  EXPECT_FALSE(result.has_value());
}

// The GPU's copy into sorted order, read back: the 200 random grids and the valid box of 103,823 cells, each after a
// list that fails, so the cell ids start with the values of another call.
TEST(BuildSortedCellList, OnGpu_SortedListEqualsCpu) {
  const std::unique_ptr<GpuGridBuffers> gpu_buffers = BuildGpuGridBuffers(kListRoom);
  ASSERT_NE(gpu_buffers, nullptr);
  std::expected<TestGrid, GridFailure> cpu_grid = BuildCpuBuffersOwner(4, kListRoom);
  GridBuffers cpu_buffers = TakeEmptyGridBuffers(cpu_grid);
  const std::vector<CellId> failing_cells = BuildTwoSizeLevelsApartLists().front();
  std::vector<std::uint64_t> digest_words;
  for (const std::vector<CellId>& cells : BuildShuffledRandomGrids()) {
    ApplyFailingCheck(failing_cells, *gpu_buffers);
    ExpectSameSortedList(ToUnitFrame(4), cells, *gpu_buffers, cpu_buffers, digest_words);
  }
  ApplyFailingCheck(failing_cells, *gpu_buffers);
  ExpectSameSortedList(ToUnitFrame(3), BuildLargeBoxOfCells({}), *gpu_buffers, cpu_buffers, digest_words);
  WriteResultDigest("BuildSortedCellList on the GPU, sorted lists", std::span<const std::uint64_t>(digest_words));
}

// A large list unlike the boxes of BuildLargeBoxOfCells: the size-level-1 cells 0..46 on each axis, 103,823 cells,
// and size level 1 {10,10,10} a second time, shuffled with ApplyFixedShuffle(cells, 4712). Checked, it fails at step 2,
// overlaps, so it leaves the key of step 2 at the position of the repeat in the status record. Every id differs from
// the boxes' ids in its size level, so its sorted cell ids, sort keys and orders differ from theirs at almost every
// place.
std::vector<CellId> BuildStaleLargeList() {
  constexpr std::int32_t kBoxWidth = 47;
  std::vector<CellId> cells = {ToCell(1, 10, 10, 10)};
  cells.reserve(1 + (std::size_t{kBoxWidth} * kBoxWidth * kBoxWidth));
  for (std::int32_t z = 0; z < kBoxWidth; ++z) {
    for (std::int32_t y = 0; y < kBoxWidth; ++y) {
      for (std::int32_t x = 0; x < kBoxWidth; ++x) {
        cells.push_back(ToCell(1, x, y, z));
      }
    }
  }
  ApplyFixedShuffle(cells, 4712);
  return cells;
}

// The box of 103,823 cells with size level 2 {188,188,188} fails only at its last sorted position, 103,823: the one
// item of the last, partial thread block (thread block 405, thread 143). Its valid twin passes. Each runs with both
// sibling checks, right after a check of BuildStaleLargeList on the GPU. That call leaves in the status record a key of
// step 2, below every key of the boxes, which a missing reset would report as the box's failure; and in the cell ids,
// the sort keys and both orders, 103,824 entries of a list of another size level, which a skipped or partial write
// would leave in the box's sorted list or order.
TEST(CheckSizeStepRule, OnGpuLargeListFailingAtLastPosition_SameFailureAsCpu) {
  const std::unique_ptr<GpuGridBuffers> gpu_buffers = BuildGpuGridBuffers(kListRoom);
  ASSERT_NE(gpu_buffers, nullptr);
  std::expected<TestGrid, GridFailure> cpu_grid = BuildCpuBuffersOwner(3, kListRoom);
  GridBuffers cpu_buffers = TakeEmptyGridBuffers(cpu_grid);
  const std::vector<CellId> stale_cells = BuildStaleLargeList();
  const std::vector<CellId> failing_cells = BuildLargeBoxOfCells({ToCell(2, 188, 188, 188)});
  const std::vector<CellId> valid_cells = BuildLargeBoxOfCells({});
  std::vector<std::uint64_t> digest_words;
  for (const std::vector<CellId>* cells : {&failing_cells, &valid_cells}) {
    for (const SiblingCheck sibling_check : {SiblingCheck::kRequire, SiblingCheck::kSkip}) {
      ApplyFailingCheck(stale_cells, *gpu_buffers);
      const std::expected<void, GridFailure> gpu_result = ToCheckResult(
          BuildSortedCellList(ToGpuInputs(ToUnitFrame(3), *cells, *gpu_buffers), *gpu_buffers->buffers, sibling_check)
      );
      const std::expected<void, GridFailure> cpu_result = ToCheckResult(BuildSortedCellList(
          {.frame = ToUnitFrame(3), .active_cells = *cells, .version = {.value = 1}},
          cpu_buffers,
          sibling_check
      ));
      EXPECT_EQ(ToFailureWords(gpu_result), ToFailureWords(cpu_result)) << cells->size();
      AppendResultWords(gpu_result, *gpu_buffers->buffers, digest_words);
    }
  }
  WriteResultDigest(
      "CheckSizeStepRule on the GPU, large box failing at its last position",
      std::span<const std::uint64_t>(digest_words)
  );
}

}  // namespace
}  // namespace fluidity::grid
