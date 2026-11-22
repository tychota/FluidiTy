// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Builds grids on the GPU (<proc-build-grid-in-parallel> P1 to P12) and compares each with the CPU's build of the same
// list: the same failure (error, cell, other_cell), or the same frame, version, counts and bytes in the used part of
// the five tables, read back to the CPU with ReadGrid. The lists are put on the GPU with WriteBuffer in each test's
// setup, so the build reads them there. Every GPU result goes to WriteResultDigest under a unique name, so the repeat
// run shows the GPU gave the same results twice: the failure words of each call, then, for each call that ran on the
// GPU, the status record it wrote (its key, error, cell, other_cell and counts), and for a grid its version, counts and
// one digest of the used part of each table. This program lists the allocation counter, so a test can show that a
// build on the GPU allocates nothing on the CPU.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
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
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/staggered_grid_gpu.h"
#include "fluidity/grid/testing/allocation_counter.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// The room of the buffers of every test here but the room checks: the random grids have at most 252 blocks.
constexpr std::uint32_t kRoom = 4096;

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

// The inputs of a CPU list of version 1.
GridInputs ToCpuInputs(GridFrame frame, std::span<const CellId> cells) {
  return {.frame = frame, .active_cells = cells, .version = {.value = 1}};
}

// The cells a list buffer of a room holds: one more than the room's cell places, so a list that fails the length check
// of P1 can lie on the GPU too.
std::size_t ComputeListRoom(std::uint32_t max_cell_blocks) {
  return (std::size_t{kCellPlacesPerBlock} * max_cell_blocks) + 1;
}

// GPU grid buffers that lend one set of builder buffers, and a GPU buffer for a list, in a RunMemory of exactly their
// estimate plus the list. Kept behind pointers, since lent builder buffers must not move; the members are destroyed in
// reverse order, the RunMemory last. `buffers` holds one set, or two for builder buffers shared by two grids.
struct GpuGridBuffers {
  std::unique_ptr<RunMemory> run_memory;
  std::unique_ptr<GridBuilderBuffers> builder_buffers;
  std::vector<GridBuffers> buffers;
  Buffer<CellId> cell_list;
};

// Builds GpuGridBuffers of max_cell_blocks blocks with grid_count sets of grid buffers that share the builder buffers.
// Null, with a test failure, when a builder fails.
//
// clang-tidy: a room and a count of grids; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::unique_ptr<GpuGridBuffers> BuildGpuGridBuffers(std::uint32_t max_cell_blocks, std::uint32_t grid_count = 1) {
  const std::vector<MemoryEstimateEntry> estimate =
      BuildGridMemoryEstimate(RunOn::kGpu, grid_count, max_cell_blocks, BuilderBuffersSharing::kOneForAllGrids);
  GridCounts room_counts = {};
  room_counts.cell_blocks = max_cell_blocks;
  const MemoryCounts memory_counts = ToMemoryCounts(room_counts, 0);
  const std::size_t list_count = ComputeListRoom(max_cell_blocks);
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
  built->buffers.reserve(grid_count);
  for (std::uint32_t grid = 0; grid < grid_count; ++grid) {
    std::expected<GridBuffers, GridFailure> buffers =
        BuildGridBuffers(settings, *built->builder_buffers, *built->run_memory);
    if (!buffers.has_value()) {
      ADD_FAILURE() << "BuildGridBuffers failed";
      return nullptr;
    }
    built->buffers.push_back(std::move(*buffers));
  }
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
// the build.
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

// CPU buffers of a room, taken from grids of no cells that keep their memory: one set to build the CPU's grid into and
// one to read the GPU's grid into. The owners are declared first, so they are built first and outlive the buffers.
struct CpuBuffers {
  explicit CpuBuffers(std::uint32_t max_cell_blocks)
      : build_owner(BuildTestGrid(ToCpuInputs(ToUnitFrame(4), {}), {.max_cell_blocks = max_cell_blocks})),
        read_owner(BuildTestGrid(ToCpuInputs(ToUnitFrame(4), {}), {.max_cell_blocks = max_cell_blocks})),
        build_buffers(TakeEmptyGridBuffers(build_owner)),
        read_buffers(TakeEmptyGridBuffers(read_owner)) {
  }

  std::expected<TestGrid, GridFailure> build_owner;
  std::expected<TestGrid, GridFailure> read_owner;
  GridBuffers build_buffers;
  GridBuffers read_buffers;
};

std::unique_ptr<CpuBuffers> BuildCpuBuffers(std::uint32_t max_cell_blocks) {
  return std::make_unique<CpuBuffers>(max_cell_blocks);
}

// The failure of a result as the four words the digest keeps: error, cell, other_cell, magnitude (zeros on success).
template <typename Value>
std::array<std::uint64_t, 4> ToFailureWords(const std::expected<Value, GridFailure>& result) {
  const GridFailure failure = result.has_value() ? GridFailure{} : result.error();
  return {static_cast<std::uint64_t>(failure.error), failure.cell.value, failure.other_cell.value, failure.magnitude};
}

// Appends every count of `counts`: the blocks, the active cells per size level, the active cells and the face pieces.
void ApplyCountWordsToDigest(const GridCounts& counts, std::vector<std::uint64_t>& digest_words) {
  digest_words.push_back(counts.cell_blocks);
  for (const std::uint64_t size_level_count : counts.active_cells_per_size_level) {
    digest_words.push_back(size_level_count);
  }
  digest_words.push_back(counts.active_cells);
  digest_words.push_back(counts.face_pieces);
}

// Appends the key, error, cell and other_cell of the status record the GPU wrote into `buffers`, and its counts.
void ApplyStatusWordsToDigest(const GridBuffers& buffers, std::vector<std::uint64_t>& digest_words) {
  GridBuildStatus status{};
  const std::expected<void, BufferError> read =
      ReadBuffer(ToGridBuffersArrays(buffers).build_status, 0, std::span<GridBuildStatus>(&status, 1));
  EXPECT_TRUE(read.has_value());
  digest_words.push_back(status.failure_key);
  digest_words.push_back(static_cast<std::uint64_t>(status.error));
  digest_words.push_back(status.cell.value);
  digest_words.push_back(status.other_cell.value);
  digest_words.push_back(status.cell_blocks);
  digest_words.push_back(status.split_cells);
  digest_words.push_back(status.active_cells);
  digest_words.push_back(status.face_pieces);
}

// The digest of the bytes of `count` values from `first`.
template <typename Value>
std::uint64_t ComputeTableDigest(const Value* first, std::size_t count) {
  return ComputeResultDigest(std::as_bytes(std::span<const Value>(first, count)));
}

// Appends the version, the counts and one digest of the used part of each table of a grid on the CPU.
void ApplyGridWordsToDigest(const StaggeredGrid& cpu_grid, std::vector<std::uint64_t>& digest_words) {
  const GridTables tables = cpu_grid.tables(RunOn::kCpu);
  const std::size_t block_count = tables.block_count;
  digest_words.push_back(cpu_grid.version().value);
  ApplyCountWordsToDigest(cpu_grid.counts(), digest_words);
  digest_words.push_back(ComputeTableDigest(tables.block_ids, block_count));
  digest_words.push_back(ComputeTableDigest(tables.same_size_level_blocks, 27 * block_count));
  digest_words.push_back(ComputeTableDigest(tables.bigger_block, block_count));
  digest_words.push_back(ComputeTableDigest(tables.smaller_blocks, 8 * block_count));
  digest_words.push_back(ComputeTableDigest(tables.cell_states, kCellPlacesPerBlock * block_count));
  digest_words.push_back(ComputeTableDigest(tables.face_piece_kinds, kFacePlacesPerBlock * block_count));
}

// Reads the GPU's grid back into cpu.read_buffers and expects it to equal the CPU's grid (frame, counts, version and
// used tables); appends its words, then takes cpu.read_buffers back.
void CheckReadBackEqualsCpu(
    const StaggeredGrid& gpu_grid,
    const std::expected<StaggeredGrid, GridFailure>& cpu_grid,
    CpuBuffers& cpu,
    std::vector<std::uint64_t>& digest_words
) {
  std::expected<StaggeredGrid, GridFailure> read_grid = ReadGrid(gpu_grid, cpu.read_buffers);
  ASSERT_TRUE(read_grid.has_value());
  EXPECT_TRUE(cpu_grid.has_value() && ComputeIsSameGrid(*read_grid, *cpu_grid));
  EXPECT_EQ(read_grid->version().value, 1U);
  ApplyGridWordsToDigest(*read_grid, digest_words);
  cpu.read_buffers = TakeBuffers(std::move(*read_grid));
}

// Builds `cells` on the GPU into `gpu_buffers` and on the CPU into cpu.build_buffers and expects the same failure, or,
// on success, the same grid (CheckReadBackEqualsCpu). Appends the failure words, the status record when the build ran
// on the GPU, and on success the grid's words. Every set of buffers is taken back, so the next call starts from
// buffers that held this grid.
void CheckGpuBuildEqualsCpu(
    GridFrame frame,
    std::span<const CellId> cells,
    GridBuffers& gpu_buffers,
    GpuGridBuffers& gpu_memory,
    CpuBuffers& cpu,
    std::vector<std::uint64_t>& digest_words
) {
  SCOPED_TRACE(cells.size());
  std::expected<StaggeredGrid, GridFailure> gpu_grid = BuildGrid(ToGpuInputs(frame, cells, gpu_memory), gpu_buffers);
  std::expected<StaggeredGrid, GridFailure> cpu_grid = BuildGrid(ToCpuInputs(frame, cells), cpu.build_buffers);
  EXPECT_EQ(ToFailureWords(gpu_grid), ToFailureWords(cpu_grid));
  const std::array<std::uint64_t, 4> failure_words = ToFailureWords(gpu_grid);
  digest_words.insert(digest_words.end(), failure_words.begin(), failure_words.end());
  if (gpu_grid.has_value()) {
    CheckReadBackEqualsCpu(*gpu_grid, cpu_grid, cpu, digest_words);
    gpu_buffers = TakeBuffers(std::move(*gpu_grid));
  }
  if (cpu_grid.has_value()) {
    cpu.build_buffers = TakeBuffers(std::move(*cpu_grid));
  }
  // A list longer than the room's cell places fails before anything runs on the GPU: the record holds nothing of it.
  if (cells.size() <= std::size_t{kCellPlacesPerBlock} * gpu_buffers.max_cell_blocks()) {
    ApplyStatusWordsToDigest(gpu_buffers, digest_words);
  }
}

// The lists every machine builds alike: the empty list, one cell and the two-size fixture, in a frame of 3 size levels,
// then the 200 random grids in a frame of 4.
struct FrameAndCells {
  GridFrame frame;
  std::vector<CellId> cells;
};

std::vector<FrameAndCells> BuildFixedAndRandomLists() {
  std::vector<FrameAndCells> lists = {
      {.frame = ToUnitFrame(3), .cells = {}},
      {.frame = ToUnitFrame(3), .cells = {ToCell(0, 1, -2, 3)}},
      {.frame = ToUnitFrame(3), .cells = BuildTwoSizeFixtureCells()},
  };
  for (const std::vector<CellId>& cells : BuildShuffledRandomGrids()) {
    lists.push_back({.frame = ToUnitFrame(kRandomGridSizeLevelCount), .cells = cells});
  }
  return lists;
}

// Builds every list of BuildFixedAndRandomLists into GPU buffers for 4,096 blocks, each after the one before, and
// returns the digest words, expecting each build to equal the CPU's.
std::vector<std::uint64_t> ComputeFixedAndRandomBuildWords() {
  const std::unique_ptr<GpuGridBuffers> gpu_memory = BuildGpuGridBuffers(kRoom);
  const std::unique_ptr<CpuBuffers> cpu = BuildCpuBuffers(kRoom);
  std::vector<std::uint64_t> digest_words;
  if (gpu_memory == nullptr) {
    return digest_words;
  }
  for (const FrameAndCells& list : BuildFixedAndRandomLists()) {
    CheckGpuBuildEqualsCpu(list.frame, list.cells, gpu_memory->buffers.front(), *gpu_memory, *cpu, digest_words);
  }
  return digest_words;
}

TEST(BuildGrid, OnGpu_EqualsCpuTablesBitForBit) {
  const std::vector<std::uint64_t> digest_words = ComputeFixedAndRandomBuildWords();
  ASSERT_FALSE(digest_words.empty());
  WriteResultDigest("BuildGrid on the GPU, fixed and random lists", std::span<const std::uint64_t>(digest_words));
}

// Builds `cells` into GPU and CPU buffers of max_cell_blocks blocks, after a valid grid that fits them, and expects the
// same failure; appends the failure words and, when the GPU ran, the status record.
//
// clang-tidy: the failing list and the grid before it; the swappable-parameters check flags the two spans.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void CheckSameRoomFailure(
    std::span<const CellId> cells,
    std::span<const CellId> grid_before,
    std::uint32_t max_cell_blocks,
    std::vector<std::uint64_t>& digest_words
) {
  const std::unique_ptr<GpuGridBuffers> gpu_memory = BuildGpuGridBuffers(max_cell_blocks);
  ASSERT_NE(gpu_memory, nullptr);
  const std::unique_ptr<CpuBuffers> cpu = BuildCpuBuffers(max_cell_blocks);
  GridBuffers& gpu_buffers = gpu_memory->buffers.front();
  CheckGpuBuildEqualsCpu(ToUnitFrame(4), grid_before, gpu_buffers, *gpu_memory, *cpu, digest_words);
  CheckGpuBuildEqualsCpu(ToUnitFrame(4), cells, gpu_buffers, *gpu_memory, *cpu, digest_words);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// The 500 mutated lists, each after the one before in buffers for 4,096 blocks, then the room checks, each after a
// valid grid in the same buffers: 9 size-level-0 blocks into buffers for 8 (step 6, the block count), 64 x 8 + 1 cells
// into the same (the length check of P1, before anything runs on the GPU), and the 64 size-level-1 cells of 8
// size-level-0 cells into room for 1 (step 5 at P7, the active and split cells against the cell places).
TEST(BuildGrid, OnGpuFailures_EqualCpuFailures) {
  const std::unique_ptr<GpuGridBuffers> gpu_memory = BuildGpuGridBuffers(kRoom);
  ASSERT_NE(gpu_memory, nullptr);
  const std::unique_ptr<CpuBuffers> cpu = BuildCpuBuffers(kRoom);
  std::vector<std::uint64_t> digest_words;
  for (std::uint64_t seed = 1; seed <= kMutatedGridCount; ++seed) {
    const std::vector<CellId> cells = BuildMutatedGridCells(seed);
    CheckGpuBuildEqualsCpu(ToUnitFrame(4), cells, gpu_memory->buffers.front(), *gpu_memory, *cpu, digest_words);
  }
  const std::vector<CellId> eight_blocks = BuildLoneCellsAlongX(8);
  const std::vector<CellId> one_block = {ToCell(0, 0, 0, 0), ToCell(0, 1, 0, 0), ToCell(0, 2, 0, 0)};
  CheckSameRoomFailure(BuildLoneCellsAlongX(9), eight_blocks, 8, digest_words);
  CheckSameRoomFailure(BuildLoneCellsAlongX((64 * 8) + 1), eight_blocks, 8, digest_words);
  CheckSameRoomFailure(BuildSmallCellsOfEightBigCells(), one_block, 1, digest_words);
  WriteResultDigest("BuildGrid on the GPU, failures", std::span<const std::uint64_t>(digest_words));
}

// The first mutated list whose CPU build fails, by seed.
std::vector<CellId> FindFailingMutatedList() {
  const std::unique_ptr<CpuBuffers> cpu = BuildCpuBuffers(kRoom);
  for (std::uint64_t seed = 1; seed <= kMutatedGridCount; ++seed) {
    std::vector<CellId> cells = BuildMutatedGridCells(seed);
    std::expected<StaggeredGrid, GridFailure> built = BuildGrid(ToCpuInputs(ToUnitFrame(4), cells), cpu->build_buffers);
    if (!built.has_value()) {
      return cells;
    }
    cpu->build_buffers = TakeBuffers(std::move(*built));
  }
  ADD_FAILURE() << "no mutated list fails";
  return {};
}

// Builds `cells` into `gpu_buffers` and reads the grid back into read_buffers, which the returned grid then holds.
// Fails the calling test, and gives nothing, when either call fails; the GPU buffers are taken back either way.
std::optional<StaggeredGrid> BuildGridReadBack(
    std::span<const CellId> cells,
    GridBuffers& gpu_buffers,
    GpuGridBuffers& gpu_memory,
    GridBuffers& read_buffers
) {
  std::expected<StaggeredGrid, GridFailure> gpu_grid =
      BuildGrid(ToGpuInputs(ToUnitFrame(kRandomGridSizeLevelCount), cells, gpu_memory), gpu_buffers);
  EXPECT_TRUE(gpu_grid.has_value());
  if (!gpu_grid.has_value()) {
    return std::nullopt;
  }
  std::expected<StaggeredGrid, GridFailure> read_grid = ReadGrid(*gpu_grid, read_buffers);
  gpu_buffers = TakeBuffers(std::move(*gpu_grid));
  EXPECT_TRUE(read_grid.has_value());
  if (!read_grid.has_value()) {
    return std::nullopt;
  }
  return std::move(*read_grid);
}

// A failing list leaves the GPU buffers with what its steps wrote; grid B built into them right after equals grid B
// built into fresh buffers. B is the largest random grid.
TEST(BuildGrid, OnGpuFailedBuildThenRebuild_EqualsFreshBuffers) {
  const std::vector<CellId> failing_cells = FindFailingMutatedList();
  ASSERT_FALSE(failing_cells.empty());
  const std::vector<CellId>& grid_b_cells = BuildShuffledRandomGrids().at(FindRandomGridsSpreadBySize(1).front());
  const std::unique_ptr<GpuGridBuffers> used_memory = BuildGpuGridBuffers(kRoom);
  const std::unique_ptr<GpuGridBuffers> fresh_memory = BuildGpuGridBuffers(kRoom);
  ASSERT_NE(used_memory, nullptr);
  ASSERT_NE(fresh_memory, nullptr);
  const std::unique_ptr<CpuBuffers> cpu = BuildCpuBuffers(kRoom);
  GridBuffers& used_buffers = used_memory->buffers.front();

  const std::expected<StaggeredGrid, GridFailure> failed =
      BuildGrid(ToGpuInputs(ToUnitFrame(4), failing_cells, *used_memory), used_buffers);
  ASSERT_FALSE(failed.has_value());
  std::vector<std::uint64_t> digest_words;
  const std::array<std::uint64_t, 4> failure_words = ToFailureWords(failed);
  digest_words.insert(digest_words.end(), failure_words.begin(), failure_words.end());
  const std::optional<StaggeredGrid> after_failure =
      BuildGridReadBack(grid_b_cells, used_buffers, *used_memory, cpu->read_buffers);
  const std::optional<StaggeredGrid> from_fresh =
      BuildGridReadBack(grid_b_cells, fresh_memory->buffers.front(), *fresh_memory, cpu->build_buffers);
  ASSERT_TRUE(after_failure.has_value());
  ASSERT_TRUE(from_fresh.has_value());
  EXPECT_TRUE(ComputeIsSameGrid(*after_failure, *from_fresh));
  ApplyGridWordsToDigest(*after_failure, digest_words);
  WriteResultDigest("BuildGrid on the GPU, rebuild after a failure", std::span<const std::uint64_t>(digest_words));
}

// Random grid 1 built into GPU buffers and taken back, then the two-size fixture into the same buffers: it equals the
// fixture built into fresh buffers, so no entry of the bigger grid is left in the used part of the smaller one.
TEST(BuildGrid, RebuildIntoUsedGpuBuffers_EqualsFreshBuffers) {
  const std::unique_ptr<GpuGridBuffers> used_memory = BuildGpuGridBuffers(kRoom);
  const std::unique_ptr<GpuGridBuffers> fresh_memory = BuildGpuGridBuffers(kRoom);
  ASSERT_NE(used_memory, nullptr);
  ASSERT_NE(fresh_memory, nullptr);
  const std::unique_ptr<CpuBuffers> cpu = BuildCpuBuffers(kRoom);
  GridBuffers& used_buffers = used_memory->buffers.front();
  std::optional<StaggeredGrid> grid_one =
      BuildGridReadBack(BuildShuffledRandomGrids().at(0), used_buffers, *used_memory, cpu->read_buffers);
  ASSERT_TRUE(grid_one.has_value());
  cpu->read_buffers = TakeBuffers(std::move(*grid_one));
  const std::vector<CellId> two_size_cells = BuildTwoSizeFixtureCells();
  const std::optional<StaggeredGrid> after_grid_one =
      BuildGridReadBack(two_size_cells, used_buffers, *used_memory, cpu->read_buffers);
  const std::optional<StaggeredGrid> from_fresh =
      BuildGridReadBack(two_size_cells, fresh_memory->buffers.front(), *fresh_memory, cpu->build_buffers);
  ASSERT_TRUE(after_grid_one.has_value());
  ASSERT_TRUE(from_fresh.has_value());
  EXPECT_TRUE(ComputeIsSameGrid(*after_grid_one, *from_fresh));
  std::vector<std::uint64_t> digest_words;
  ApplyGridWordsToDigest(*after_grid_one, digest_words);
  WriteResultDigest("BuildGrid on the GPU, rebuild into used buffers", std::span<const std::uint64_t>(digest_words));
}

// What the builds in turn of IntoGpuBuffers_AllocatesNothing gave: the counts of each grid, the builds that failed
// and the CPU allocations made between the start of a build and the return of its buffers.
struct BuildsInTurn {
  std::vector<std::uint64_t> digest_words;  // Reserved before the first build.
  std::size_t failed_build_count = 0;
  std::uint64_t allocation_count = 0;
};

// Writes `cells` to the GPU, the test's setup, then builds them into `gpu_buffers`, appends the counts and takes the
// buffers back; only the build, the append and the take-back are counted in builds.allocation_count.
void ApplyBuildInTurn(
    std::span<const CellId> cells,
    GridBuffers& gpu_buffers,
    GpuGridBuffers& gpu_memory,
    BuildsInTurn& builds
) {
  const GridInputs inputs = ToGpuInputs(ToUnitFrame(kRandomGridSizeLevelCount), cells, gpu_memory);
  const std::uint64_t allocation_count_before = ComputeAllocationCount();
  std::expected<StaggeredGrid, GridFailure> built = BuildGrid(inputs, gpu_buffers);
  if (built.has_value()) {
    ApplyCountWordsToDigest(built->counts(), builds.digest_words);
    gpu_buffers = TakeBuffers(std::move(*built));
  } else {
    ++builds.failed_build_count;
  }
  builds.allocation_count += ComputeAllocationCount() - allocation_count_before;
}

// The 100 random grids spread by size (FindRandomGridsSpreadBySize, from the most cells to the fewest), built into GPU
// buffers and taken back in turn. After the first build, GPU free bytes stay the same and no build allocates on the
// CPU, the digest words included, which are reserved first: each build writes only into memory sized once for the
// room, over small, middling and large grids. The test writes each list to the GPU before its build starts.
TEST(BuildGrid, IntoGpuBuffers_AllocatesNothing) {
  constexpr std::size_t kGridCount = 100;
  const std::vector<std::size_t> grids = FindRandomGridsSpreadBySize(kGridCount);
  const std::unique_ptr<GpuGridBuffers> gpu_memory = BuildGpuGridBuffers(kRoom);
  ASSERT_NE(gpu_memory, nullptr);
  GridBuffers& gpu_buffers = gpu_memory->buffers.front();
  BuildsInTurn builds;
  builds.digest_words.reserve(kGridCount * (kMaxSizeLevels + 3));
  const std::vector<std::vector<CellId>>& random_grids = BuildShuffledRandomGrids();
  ApplyBuildInTurn(random_grids.at(grids.front()), gpu_buffers, *gpu_memory, builds);
  builds.allocation_count = 0;
  const std::expected<MemoryAvailable, BufferError> before = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(before.has_value());
  for (std::size_t position = 1; position < kGridCount; ++position) {
    ApplyBuildInTurn(random_grids.at(grids.at(position)), gpu_buffers, *gpu_memory, builds);
  }
  const std::expected<MemoryAvailable, BufferError> after = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(builds.allocation_count, 0U);
  EXPECT_EQ(after->free_bytes, before->free_bytes);
  EXPECT_EQ(builds.failed_build_count, 0U);
  WriteResultDigest("BuildGrid on the GPU, 100 grids in turn", std::span<const std::uint64_t>(builds.digest_words));
}

// The memory counts of a room of max_cell_blocks blocks, as the estimate reads them.
MemoryCounts ToRoomMemoryCounts(std::uint32_t max_cell_blocks) {
  GridCounts room_counts = {};
  room_counts.cell_blocks = max_cell_blocks;
  return ToMemoryCounts(room_counts, 0);
}

// At 1,000 blocks: reserving the estimate of one GPU grid takes at least its reserved bytes from the GPU's free bytes;
// building the builder buffers and the grid buffers into it takes no more, and allocates one thing on the CPU, the
// 8-byte count of loans of the builder buffers: no CPU array. The estimate is built before the counted part.
TEST(ComputeMemoryBytes, MatchesAllocatedGpuBytes) {
  constexpr std::uint32_t kThousandBlocks = 1000;
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kGpu, 1, kThousandBlocks);
  const std::size_t byte_count =
      ComputeRunMemoryBytes(estimate, ToRoomMemoryCounts(kThousandBlocks), MemoryWhere::kRunOn);
  const std::expected<MemoryAvailable, BufferError> before_reserve = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(before_reserve.has_value());
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, byte_count);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<MemoryAvailable, BufferError> after_reserve = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(after_reserve.has_value());
  EXPECT_GE(before_reserve->free_bytes - after_reserve->free_bytes, run_memory->reserved_bytes());

  const std::uint64_t allocation_count_before = ComputeAllocationCount();
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffersFromEntries(RunOn::kGpu, kThousandBlocks, estimate, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const GridSettings settings = {.max_cell_blocks = kThousandBlocks, .run_on = RunOn::kGpu};
  const std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffersFromEntries(settings, *builder_buffers, estimate, *run_memory);
  const std::uint64_t allocation_count_after = ComputeAllocationCount();
  ASSERT_TRUE(buffers.has_value());
  const std::expected<MemoryAvailable, BufferError> after_build = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(after_build.has_value());
  EXPECT_EQ(after_build->free_bytes, after_reserve->free_bytes);
  EXPECT_EQ(allocation_count_after - allocation_count_before, 1U);
  const std::array<std::uint64_t, 2> words = {run_memory->reserved_bytes(), run_memory->handed_out_bytes()};
  WriteResultDigest("ComputeMemoryBytes of a GPU grid of 1,000 blocks", std::span<const std::uint64_t>(words));
}

// Builds grid_count GPU grid buffers of 1,000 blocks and their builder buffers, shared as `builder_buffers_sharing`
// says, in a GPU RunMemory of exactly their estimate, and expects the block used to the byte.
void CheckGpuBuildersHandOutTheEstimate(BuilderBuffersSharing builder_buffers_sharing, std::uint32_t grid_count) {
  constexpr std::uint32_t kThousandBlocks = 1000;
  const std::vector<MemoryEstimateEntry> estimate =
      BuildGridMemoryEstimate(RunOn::kGpu, grid_count, kThousandBlocks, builder_buffers_sharing);
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(
      RunOn::kGpu,
      ComputeRunMemoryBytes(estimate, ToRoomMemoryCounts(kThousandBlocks), MemoryWhere::kRunOn)
  );
  ASSERT_TRUE(run_memory.has_value());
  const std::uint32_t builder_buffers_count =
      builder_buffers_sharing == BuilderBuffersSharing::kOnePerGrid ? grid_count : 1;
  std::vector<std::unique_ptr<GridBuilderBuffers>> builder_buffers;
  for (std::uint32_t builder = 0; builder < builder_buffers_count; ++builder) {
    std::expected<GridBuilderBuffers, GridFailure> built =
        BuildGridBuilderBuffers(RunOn::kGpu, kThousandBlocks, *run_memory);
    ASSERT_TRUE(built.has_value());
    builder_buffers.push_back(std::make_unique<GridBuilderBuffers>(std::move(*built)));
  }
  std::vector<GridBuffers> grid_buffers;
  for (std::uint32_t grid = 0; grid < grid_count; ++grid) {
    GridBuilderBuffers& lent = *builder_buffers.at(grid % builder_buffers_count);
    std::expected<GridBuffers, GridFailure> buffers =
        BuildGridBuffers({.max_cell_blocks = kThousandBlocks, .run_on = RunOn::kGpu}, lent, *run_memory);
    ASSERT_TRUE(buffers.has_value());
    grid_buffers.push_back(std::move(*buffers));
  }
  EXPECT_EQ(run_memory->handed_out_bytes(), run_memory->reserved_bytes()) << grid_count;
  const std::array<std::uint64_t, 1> words = {run_memory->handed_out_bytes()};
  WriteResultDigest(
      grid_count == 1 ? "BuildGridBuffers on the GPU, one grid" : "BuildGridBuffers on the GPU, two grids sharing",
      std::span<const std::uint64_t>(words)
  );
}

TEST(BuildGridBuffers, OnGpuIntoRunMemory_HandsOutExactlyTheEstimate) {
  CheckGpuBuildersHandOutTheEstimate(BuilderBuffersSharing::kOnePerGrid, 1);
  CheckGpuBuildersHandOutTheEstimate(BuilderBuffersSharing::kOneForAllGrids, 2);
}

// The counts of the 200 random grids on the GPU, each after the one before, equal the CPU's.
TEST(ComputeGridCounts, OnGpu_EqualsCpu) {
  const std::unique_ptr<GpuGridBuffers> gpu_memory = BuildGpuGridBuffers(kRoom);
  ASSERT_NE(gpu_memory, nullptr);
  const std::unique_ptr<CpuBuffers> cpu = BuildCpuBuffers(kRoom);
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  std::vector<std::uint64_t> digest_words;
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    const std::vector<CellId>& cells = BuildShuffledRandomGrids().at(grid);
    const std::expected<GridCounts, GridFailure> gpu_counts =
        ComputeGridCounts(ToGpuInputs(frame, cells, *gpu_memory), gpu_memory->buffers.front());
    const std::expected<GridCounts, GridFailure> cpu_counts =
        ComputeGridCounts(ToCpuInputs(frame, cells), cpu->build_buffers);
    ASSERT_TRUE(gpu_counts.has_value()) << grid;
    ASSERT_TRUE(cpu_counts.has_value()) << grid;
    EXPECT_EQ(ToGridCountFields(*gpu_counts), ToGridCountFields(*cpu_counts)) << grid;
    ApplyCountWordsToDigest(*gpu_counts, digest_words);
  }
  WriteResultDigest("ComputeGridCounts on the GPU, random grids", std::span<const std::uint64_t>(digest_words));
}

// The fixed and random lists built twice, each time into new GPU buffers: the same words, tables included.
TEST(BuildGrid, OnGpuTwice_GivesIdenticalBytes) {
  const std::vector<std::uint64_t> first_words = ComputeFixedAndRandomBuildWords();
  const std::vector<std::uint64_t> second_words = ComputeFixedAndRandomBuildWords();
  ASSERT_FALSE(first_words.empty());
  EXPECT_EQ(first_words, second_words);
  WriteResultDigest("BuildGrid on the GPU, fixed and random lists twice", std::span<const std::uint64_t>(second_words));
}

// Counts and builds a list with a repeated cell into `buffers`: both must fail. Adds the calls that did not to
// wrong_count; a build that did not is taken back.
void ApplyFailedCallsOnGpuBuffers(
    std::span<const CellId> cells,
    GridBuffers& buffers,
    GpuGridBuffers& gpu_memory,
    std::size_t& wrong_count
) {
  std::vector<CellId> repeated_cells(cells.begin(), cells.end());
  repeated_cells.push_back(repeated_cells.front());
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  wrong_count += ComputeGridCounts(ToGpuInputs(frame, repeated_cells, gpu_memory), buffers).has_value() ? 1 : 0;
  std::expected<StaggeredGrid, GridFailure> built = BuildGrid(ToGpuInputs(frame, repeated_cells, gpu_memory), buffers);
  if (built.has_value()) {
    ++wrong_count;
    buffers = TakeBuffers(std::move(*built));
  }
}

// Counts, checks, builds and takes back `cells` in other_buffers, after a failed count and a failed build, and expects
// the grid it builds, read back into cpu.read_buffers, to equal `reference`. Returns how many of these went wrong, and
// appends the grid's words.
std::size_t ComputeWrongCallsInTurn(
    std::span<const CellId> cells,
    const StaggeredGrid& reference,
    GridBuffers& other_buffers,
    GpuGridBuffers& gpu_memory,
    CpuBuffers& cpu,
    std::vector<std::uint64_t>& digest_words
) {
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  std::size_t wrong_count = 0;
  ApplyFailedCallsOnGpuBuffers(cells, other_buffers, gpu_memory, wrong_count);
  wrong_count += ComputeGridCounts(ToGpuInputs(frame, cells, gpu_memory), other_buffers).has_value() ? 0 : 1;
  wrong_count += CheckSizeStepRule(ToGpuInputs(frame, cells, gpu_memory), other_buffers).has_value() ? 0 : 1;
  std::optional<StaggeredGrid> read_grid = BuildGridReadBack(cells, other_buffers, gpu_memory, cpu.read_buffers);
  if (!read_grid.has_value()) {
    return wrong_count + 1;
  }
  wrong_count += ComputeIsSameGrid(*read_grid, reference) ? 0 : 1;
  ApplyGridWordsToDigest(*read_grid, digest_words);
  cpu.read_buffers = TakeBuffers(std::move(*read_grid));
  return wrong_count;
}

// ComputeWrongCallsInTurn for the random grid of each entry of `grids`, in turn, against the reference grid at the same
// position. Returns the total of what went wrong.
std::size_t ComputeWrongCallsOfGrids(
    const std::vector<std::size_t>& grids,
    const std::vector<TestGrid>& reference_grids,
    GridBuffers& other_buffers,
    GpuGridBuffers& gpu_memory,
    CpuBuffers& cpu,
    std::vector<std::uint64_t>& digest_words
) {
  std::size_t wrong_count = 0;
  for (std::size_t position = 0; position < grids.size(); ++position) {
    wrong_count += ComputeWrongCallsInTurn(
        BuildShuffledRandomGrids().at(grids.at(position)),
        reference_grids.at(position).grid,
        other_buffers,
        gpu_memory,
        cpu,
        digest_words
    );
  }
  return wrong_count;
}

// Builds grid A, the largest random grid, into one of two sets of GPU buffers that share builder buffers, and reads it
// back. The other set then counts, checks, builds and takes back 30 random grids spread by size, with a failed count
// and a failed build before each (ComputeWrongCallsInTurn); each grid it builds, read back, equals the same grid built
// on the CPU with memory of its own. A read back again equals A's first read: the shared builder buffers hold nothing
// of A between builds. Appends the digest words.
void CheckOtherGpuGridUnchanged(std::size_t buffers_of_grid_a, std::vector<std::uint64_t>& digest_words) {
  constexpr std::size_t kGridCount = 30;
  const std::vector<std::size_t> grids = FindRandomGridsSpreadBySize(kGridCount);
  const std::vector<TestGrid> reference_grids = BuildReferenceGrids(grids);
  const std::unique_ptr<GpuGridBuffers> gpu_memory = BuildGpuGridBuffers(kRoom, 2);
  ASSERT_TRUE(reference_grids.size() == kGridCount && gpu_memory != nullptr);
  const std::unique_ptr<CpuBuffers> cpu = BuildCpuBuffers(kRoom);
  GridBuffers& a_buffers = gpu_memory->buffers.at(buffers_of_grid_a);
  GridBuffers& other_buffers = gpu_memory->buffers.at(1 - buffers_of_grid_a);
  const GridInputs a_inputs =
      ToGpuInputs(ToUnitFrame(kRandomGridSizeLevelCount), BuildShuffledRandomGrids().at(grids.front()), *gpu_memory);
  const std::expected<StaggeredGrid, GridFailure> grid_a = BuildGrid(a_inputs, a_buffers);
  ASSERT_TRUE(grid_a.has_value());
  const std::expected<StaggeredGrid, GridFailure> a_before = ReadGrid(*grid_a, cpu->build_buffers);
  ASSERT_TRUE(a_before.has_value());
  EXPECT_EQ(ComputeWrongCallsOfGrids(grids, reference_grids, other_buffers, *gpu_memory, *cpu, digest_words), 0U);
  const std::expected<StaggeredGrid, GridFailure> a_after = ReadGrid(*grid_a, cpu->read_buffers);
  ASSERT_TRUE(a_after.has_value());
  EXPECT_TRUE(ComputeIsSameGrid(*a_after, *a_before));
  ApplyGridWordsToDigest(*a_after, digest_words);
}

// Whichever of the two sets holds grid A, the other set's work leaves it alone.
TEST(BuildGrid, OnGpuSharedBuilderBuffers_OtherLiveGridUnchanged) {
  std::vector<std::uint64_t> first_words;
  CheckOtherGpuGridUnchanged(0, first_words);
  WriteResultDigest("BuildGrid on the GPU, shared builder buffers", std::span<const std::uint64_t>(first_words));
  std::vector<std::uint64_t> second_words;
  CheckOtherGpuGridUnchanged(1, second_words);
  WriteResultDigest(
      "BuildGrid on the GPU, shared builder buffers in reverse order",
      std::span<const std::uint64_t>(second_words)
  );
}

// The grid's memory estimate has no entry held by the driver: every GPU function of the grid's GPU half, the build's
// and the check's, keeps at most the driver's default stack of 1,024 bytes per thread. localSizeBytes counts a
// function's own frame; it covers the shared functions it calls because the compiler inlines them all (ptxas of the
// release preset reports no separate device function), which a debug compile with -G does not do.
TEST(BuildGrid, OnGpuFunctions_KeepUnderTheDefaultStack) {
  std::uint64_t largest_bytes = 0;
  ASSERT_EQ(ComputeLargestBuildStackOnGpu(&largest_bytes), RunOnCallStatus::kDone);
  RecordProperty("largest_stack_bytes", static_cast<int>(largest_bytes));
  EXPECT_LE(largest_bytes, 1024U);
  WriteResultDigest("BuildGrid GPU functions, largest stack", std::span<const std::uint64_t>(&largest_bytes, 1));
}

}  // namespace
}  // namespace fluidity::grid
