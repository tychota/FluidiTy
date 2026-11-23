// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Builds grid pieces of a GPU whole grid with the buffer form of BuildGridPiece (six GPU functions: a reset, one thread
// for each border entry, each loaded item, each piece block, each cell place and each face place) and expects the CPU's
// piece, table for table, by bits: the version, the counts, the start of the range in the piece, the piece map, the
// block ids, the 36 neighbour entries of each block, the cell states and the face piece kinds. Every grid is built on
// both processors from one list, and each processor finds its own border with the buffer form of FindBorderBlocks, so
// the GPU's border list goes from the border search to the piece build without leaving the GPU. A comparison that fails
// names the piece, the table, the place and both values. The tables and counts read back go to WriteResultDigest, so
// the repeat run shows the GPU gave the same bytes twice.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_error_message.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece.h"
#include "fluidity/grid/grid_piece_gpu.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/allocation_counter.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_functions_gpu_cases.h"
#include "fluidity/grid/testing/grid_functions_result_words.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

// The random grids built, and the blocks of a range of each kind of grid.
constexpr int kGridCount = 10;
constexpr std::uint32_t kRangeBlocks = 4;
constexpr std::uint32_t kNestedBallRangeBlocks = 16;

// A piece as a comparison sees it: everything the piece holds, read back from the processor it was built on.
struct PieceReadBack {
  std::uint32_t version = 0;
  std::uint32_t block_count = 0;
  std::uint32_t range_first_block = 0;  // The range as a run of the piece.
  std::uint32_t range_first_block_in_whole_grid = 0;
  std::uint32_t range_block_count = 0;
  std::uint64_t active_cells = 0;
  std::uint64_t face_pieces = 0;
  std::array<std::uint64_t, kMaxSizeLevels> active_cells_per_size_level{};
  std::vector<std::uint32_t> map;  // block_count entries.
  std::vector<std::uint64_t> block_ids;  // block_count entries.
  std::vector<std::uint32_t> neighbours;  // For each block: 27 of its size level, 1 bigger, 8 smaller.
  std::vector<std::uint8_t> cell_states;  // 64 for each block.
  std::vector<std::uint8_t> face_piece_kinds;  // 240 for each block.
};

void ApplyWordsOf(const PieceReadBack& piece, ResultWords& words) {
  words.Add("version", piece.version);
  words.Add("block_count", piece.block_count);
  words.Add("range_first_block", piece.range_first_block);
  words.Add("range_first_block_in_whole_grid", piece.range_first_block_in_whole_grid);
  words.Add("range_block_count", piece.range_block_count);
  words.Add("active_cells", piece.active_cells);
  words.Add("face_pieces", piece.face_pieces);
  std::size_t entry = 0;
  for (const std::uint64_t count : piece.active_cells_per_size_level) {
    words.AddEntry("active_cells_per_size_level", entry++, count);
  }
  entry = 0;
  for (const std::uint32_t block : piece.map) {
    words.AddEntry("map", entry++, block);
  }
  entry = 0;
  for (const std::uint64_t id : piece.block_ids) {
    words.AddEntry("block_ids", entry++, id);
  }
  entry = 0;
  for (const std::uint32_t neighbour : piece.neighbours) {
    words.AddEntry("neighbours", entry++, neighbour);
  }
  entry = 0;
  for (const std::uint8_t state : piece.cell_states) {
    words.AddEntry("cell_states", entry++, state);
  }
  entry = 0;
  for (const std::uint8_t kind : piece.face_piece_kinds) {
    words.AddEntry("face_piece_kinds", entry++, kind);
  }
}

// The tables of a piece on the CPU, with its map and ranges, as a comparison sees them.
PieceReadBack ToReadBack(const GridPiece& piece, const StaggeredGrid& cpu_grid, std::span<const CellBlockIndex> map) {
  const GridTables tables = cpu_grid.tables(RunOn::kCpu);
  const std::uint32_t block_count = tables.block_count;
  PieceReadBack read_back;
  read_back.version = tables.version.value;
  read_back.block_count = block_count;
  read_back.range_first_block = piece.range().first_block;
  read_back.range_first_block_in_whole_grid = piece.range_in_whole_grid().first_block;
  read_back.range_block_count = piece.range().block_count;
  const GridCounts& counts = cpu_grid.counts();
  read_back.active_cells = counts.active_cells;
  read_back.face_pieces = counts.face_pieces;
  std::ranges::copy(counts.active_cells_per_size_level, read_back.active_cells_per_size_level.begin());
  for (const CellBlockIndex block : map.first(block_count)) {
    read_back.map.push_back(block.value);
  }
  for (std::uint32_t block = 0; block < block_count; ++block) {
    read_back.block_ids.push_back(tables.block_ids[block].value);
    for (std::size_t entry = 0; entry < 27; ++entry) {
      read_back.neighbours.push_back(tables.same_size_level_blocks[(std::size_t{block} * 27) + entry]);
    }
    read_back.neighbours.push_back(tables.bigger_block[block]);
    for (std::size_t entry = 0; entry < 8; ++entry) {
      read_back.neighbours.push_back(tables.smaller_blocks[(std::size_t{block} * 8) + entry]);
    }
  }
  for (std::size_t place = 0; place < std::size_t{block_count} * kCellPlacesPerBlock; ++place) {
    read_back.cell_states.push_back(static_cast<std::uint8_t>(tables.cell_states[place]));
  }
  for (std::size_t place = 0; place < std::size_t{block_count} * kFacePlacesPerBlock; ++place) {
    read_back.face_piece_kinds.push_back(static_cast<std::uint8_t>(tables.face_piece_kinds[place]));
  }
  return read_back;
}

// Reads a piece of the GPU back: its tables with ReadGrid into CPU buffers, and its piece map with ReadBuffer.
std::optional<PieceReadBack> ReadBackOfGpuPiece(const GridPiece& piece) {
  const std::uint32_t block_count = piece.grid().counts().cell_blocks;
  std::optional<PieceBuffersRoom> cpu_room = BuildPieceBuffersRoom(RunOn::kCpu, block_count);
  EXPECT_TRUE(cpu_room.has_value());
  if (!cpu_room.has_value()) {
    return std::nullopt;
  }
  const std::expected<StaggeredGrid, GridFailure> cpu_grid = ReadGrid(piece.grid(), cpu_room->buffers);
  EXPECT_TRUE(cpu_grid.has_value());
  std::vector<CellBlockIndex> map(block_count);
  const std::expected<void, BufferError> read = ReadBuffer<CellBlockIndex>(piece.block_in_whole_grid_buffer(), 0, map);
  EXPECT_TRUE(read.has_value());
  if (!cpu_grid.has_value() || !read.has_value()) {
    return std::nullopt;
  }
  return ToReadBack(piece, *cpu_grid, map);
}

// The processors' own borders of one range: the CPU's list, and the count of the GPU's, whose list stays in the buffer
// of the GPU's rig.
struct BordersOfBothProcessors {
  std::vector<CellBlockIndex> on_cpu;
  std::uint32_t gpu_count = 0;
};

// Finds the stacked border of `range` with `layer_widths` on both rigs, each layer on the list of the layer before: the
// last layer of the GPU's rig stays in its border buffer.
BordersOfBothProcessors FindBordersOnBothProcessors(
    BorderSearchRig& on_cpu,
    BorderSearchRig& on_gpu,
    CellBlockRange range,
    std::span<const BorderWidth> layer_widths
) {
  BordersOfBothProcessors borders;
  std::vector<CellBlockIndex> gpu_so_far;
  for (const BorderWidth width : layer_widths) {
    borders.on_cpu = FindLayerOrFail(on_cpu, range, borders.on_cpu, width);
    gpu_so_far = FindLayerOrFail(on_gpu, range, gpu_so_far, width);
  }
  borders.gpu_count = static_cast<std::uint32_t>(gpu_so_far.size());
  EXPECT_EQ(borders.gpu_count, borders.on_cpu.size());
  return borders;
}

// The pieces of the ranges of one grid, on both processors.
struct PiecesOfBothProcessors {
  std::vector<PieceReadBack> on_cpu;
  std::vector<PieceReadBack> on_gpu;
};

// The memory that builds the pieces of one grid on both processors, made once and used for every piece: the border
// search buffers and the room of the pieces.
struct RoomsOfBothProcessors {
  BorderSearchRig& search_on_cpu;
  BorderSearchRig& search_on_gpu;
  PieceBuffersRoom& room_on_cpu;
  PieceBuffersRoom& room_on_gpu;
};

// Builds the piece of `range` on both processors with the border of `layer_widths`, appends what each holds to `pieces`
// and takes the buffers back for the next piece.
void ApplyPiecesOfRange(
    const GridOnBothProcessors& grid,
    CellBlockRange range,
    std::span<const BorderWidth> layer_widths,
    RoomsOfBothProcessors rooms,
    PiecesOfBothProcessors& pieces
) {
  const BordersOfBothProcessors borders =
      FindBordersOnBothProcessors(rooms.search_on_cpu, rooms.search_on_gpu, range, layer_widths);
  std::expected<GridPiece, GridFailure> piece_on_cpu =
      BuildGridPiece(grid.on_cpu.grid, range, borders.on_cpu, rooms.room_on_cpu.buffers);
  std::expected<GridPiece, GridFailure> piece_on_gpu = BuildGridPiece(
      grid.on_gpu.grid,
      range,
      rooms.search_on_gpu.border_buffer,
      borders.gpu_count,
      rooms.room_on_gpu.buffers
  );
  ASSERT_TRUE(piece_on_cpu.has_value());
  ASSERT_TRUE(piece_on_gpu.has_value()) << ToName(piece_on_gpu.error().error);
  EXPECT_EQ(piece_on_gpu->grid().run_on(), RunOn::kGpu);
  pieces.on_cpu.push_back(ToReadBack(*piece_on_cpu, piece_on_cpu->grid(), piece_on_cpu->block_in_whole_grid()));
  const std::optional<PieceReadBack> read_back = ReadBackOfGpuPiece(*piece_on_gpu);
  ASSERT_TRUE(read_back.has_value());
  pieces.on_gpu.push_back(*read_back);
  rooms.room_on_cpu.buffers = TakeBuffers(std::move(*piece_on_cpu));
  rooms.room_on_gpu.buffers = TakeBuffers(std::move(*piece_on_gpu));
}

// Builds the memory of both processors for the pieces of `grid`, and calls `apply_pieces` with it.
template <typename ApplyPieces>
void ApplyWithRooms(const GridOnBothProcessors& grid, ApplyPieces apply_pieces) {
  std::expected<BorderSearchRig, GridFailure> search_on_cpu = BuildBorderSearchRig(grid.on_cpu.grid);
  std::expected<BorderSearchRig, GridFailure> search_on_gpu = BuildBorderSearchRig(grid.on_gpu.grid);
  const std::uint32_t block_count = grid.on_cpu.grid.counts().cell_blocks;
  std::optional<PieceBuffersRoom> room_on_cpu = BuildPieceBuffersRoom(RunOn::kCpu, block_count);
  std::optional<PieceBuffersRoom> room_on_gpu = BuildPieceBuffersRoom(RunOn::kGpu, block_count);
  ASSERT_TRUE(search_on_cpu.has_value());
  ASSERT_TRUE(search_on_gpu.has_value());
  ASSERT_TRUE(room_on_cpu.has_value());
  ASSERT_TRUE(room_on_gpu.has_value());
  apply_pieces(
      RoomsOfBothProcessors{
          .search_on_cpu = *search_on_cpu,
          .search_on_gpu = *search_on_gpu,
          .room_on_cpu = *room_on_cpu,
          .room_on_gpu = *room_on_gpu,
      }
  );
}

// The pieces of `ranges` of one grid, with each border kind: GPU against CPU, by bits.
void CheckPiecesOfRanges(
    const GridOnBothProcessors& grid,
    std::span<const CellBlockRange> ranges,
    const std::string& name
) {
  const std::array<BorderWidth, 1> near_border = {BorderWidth::kTouchingCells};
  PiecesOfBothProcessors pieces;
  ApplyWithRooms(grid, [&](RoomsOfBothProcessors rooms) {
    for (const CellBlockRange range : ranges) {
      SCOPED_TRACE(testing::Message() << name << ", range from block " << range.first_block);
      ApplyPiecesOfRange(grid, range, near_border, rooms, pieces);
      ApplyPiecesOfRange(grid, range, kNestedBallLayerWidths, rooms, pieces);
    }
  });
  CheckSameResults(std::span<const PieceReadBack>(pieces.on_cpu), std::span<const PieceReadBack>(pieces.on_gpu), name);
  WriteDigestOfResults("grid pieces on the GPU, " + name, std::span<const PieceReadBack>(pieces.on_gpu));
}

// The random grids, the tested ranges of each (first, middle, last, whole grid), a near border and a stacked border.
TEST(BuildGridPiece, OnGpuRandomGrids_EqualCpuPiece) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  for (std::size_t grid = 0; grid < grids.size(); ++grid) {
    SCOPED_TRACE(grid);
    const std::vector<CellBlockRange> ranges = BuildTestedRanges(grids[grid].on_cpu.grid, kRangeBlocks);
    CheckPiecesOfRanges(grids[grid], ranges, "random grid " + std::to_string(grid));
  }
}

// The nested-ball grid, ranges of 16 blocks: the tested ranges, a near border and a stacked border.
TEST(BuildGridPiece, OnGpuNestedBallRanges_EqualCpuPiece) {
  const std::optional<GridOnBothProcessors> grid = BuildGridOnBothProcessors(ToUnitFrame(4), BuildNestedBallCells());
  ASSERT_TRUE(grid.has_value());
  const std::vector<CellBlockRange> ranges = BuildTestedRanges(grid->on_cpu.grid, kNestedBallRangeBlocks);
  CheckPiecesOfRanges(*grid, ranges, "nested ball");
}

// Says whether `entries` holds `wanted`.
template <typename Entry>
bool ComputeHasEntry(const std::vector<Entry>& entries, Entry wanted) {
  return std::ranges::find(entries, wanted) != entries.end();
}

// Which kinds of entry some piece of a set holds.
struct EntryKinds {
  bool is_border_on_both_sides = false;
  bool is_outside_piece = false;
  bool is_neighbour_none = false;
};

// Adds what `piece` holds to `kinds`.
void ApplyEntryKinds(const PieceReadBack& piece, EntryKinds& kinds) {
  const bool is_before = piece.range_first_block > 0;
  const bool is_after = piece.range_first_block + piece.range_block_count < piece.block_count;
  kinds.is_border_on_both_sides = kinds.is_border_on_both_sides || (is_before && is_after);
  kinds.is_outside_piece = kinds.is_outside_piece
      || ComputeHasEntry(piece.face_piece_kinds, static_cast<std::uint8_t>(FacePieceKind::kOutsidePiece));
  kinds.is_neighbour_none = kinds.is_neighbour_none || ComputeHasEntry(piece.neighbours, kNoIndex);
}

// The pieces that a comparison of the tests above is made of are not trivial: some piece of the CPU has a border block
// before its range and one after it, a face piece outside the piece and a neighbour entry that reads as none.
TEST(BuildGridPiece, OnGpuRandomGrids_ComparePiecesWithAllKindsOfEntries) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  EntryKinds kinds;
  const std::array<BorderWidth, 1> near_border = {BorderWidth::kTouchingCells};
  for (const GridOnBothProcessors& grid : grids) {
    PiecesOfBothProcessors pieces;
    ApplyWithRooms(grid, [&](RoomsOfBothProcessors rooms) {
      for (const CellBlockRange range : BuildTestedRanges(grid.on_cpu.grid, kRangeBlocks)) {
        ApplyPiecesOfRange(grid, range, near_border, rooms, pieces);
      }
    });
    for (const PieceReadBack& piece : pieces.on_cpu) {
      ApplyEntryKinds(piece, kinds);
    }
  }
  EXPECT_TRUE(kinds.is_border_on_both_sides);
  EXPECT_TRUE(kinds.is_outside_piece);
  EXPECT_TRUE(kinds.is_neighbour_none);
}

// What one failed call returned.
struct FailedBuild {
  GridError error = GridError::kNone;
  std::uint64_t cell = 0;
  std::uint64_t other_cell = 0;
};

// Builds a border list of `blocks` in a buffer on the processor of `grid`.
Buffer<CellBlockIndex> BuildBorderBuffer(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks) {
  std::expected<Buffer<CellBlockIndex>, BufferError> buffer =
      BuildBuffer<CellBlockIndex>(grid.run_on(), std::max<std::size_t>(blocks.size(), 1));
  EXPECT_TRUE(buffer.has_value());
  EXPECT_TRUE(blocks.empty() || WriteBuffer<CellBlockIndex>(blocks, *buffer, 0).has_value());
  return std::move(*buffer);
}

// The failure of the buffer form for `range` and `border` into buffers of room `room_blocks`, on the processor of
// `grid`.
FailedBuild FindFailureOf(
    const StaggeredGrid& grid,
    CellBlockRange range,
    std::span<const CellBlockIndex> border,
    std::uint32_t room_blocks
) {
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(grid.run_on(), room_blocks);
  EXPECT_TRUE(room.has_value());
  if (!room.has_value()) {
    return {};
  }
  const Buffer<CellBlockIndex> border_buffer = BuildBorderBuffer(grid, border);
  const std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(grid, range, border_buffer, static_cast<std::uint32_t>(border.size()), room->buffers);
  EXPECT_FALSE(piece.has_value());
  if (piece.has_value()) {
    return {};
  }
  EXPECT_EQ(room->buffers.max_cell_blocks(), std::max(room_blocks, 1U));
  return {.error = piece.error().error, .cell = piece.error().cell.value, .other_cell = piece.error().other_cell.value};
}

// Expects the GPU's failure for the case to be the CPU's, and names the case when it is not.
void CheckFailureIsSame(
    const GridOnBothProcessors& grid,
    CellBlockRange range,
    std::span<const CellBlockIndex> border,
    std::uint32_t room_blocks,
    const std::string& name
) {
  SCOPED_TRACE(name);
  const FailedBuild on_cpu = FindFailureOf(grid.on_cpu.grid, range, border, room_blocks);
  const FailedBuild on_gpu = FindFailureOf(grid.on_gpu.grid, range, border, room_blocks);
  EXPECT_NE(on_cpu.error, GridError::kNone);
  EXPECT_EQ(on_gpu.error, on_cpu.error);
  EXPECT_EQ(on_gpu.cell, on_cpu.cell);
  EXPECT_EQ(on_gpu.other_cell, on_cpu.other_cell);
}

// An unsorted border, a border that repeats a block, a border block inside the range, a border index beyond the whole
// grid, a range beyond it and too little room: the GPU's error is the CPU's and its cells are zero; the room is the one
// asked for after each failure.
TEST(BuildGridPiece, OnGpuFailures_EqualCpuFailures) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(1);
  ASSERT_EQ(grids.size(), 1U);
  const GridOnBothProcessors& grid = grids.front();
  const std::uint32_t block_count = grid.on_cpu.grid.counts().cell_blocks;
  ASSERT_GE(block_count, 12U);
  const CellBlockRange range = {.first_block = 4, .block_count = 2};
  CheckFailureIsSame(grid, range, ToBlocks({7, 6}), block_count, "unsorted border");
  CheckFailureIsSame(grid, range, ToBlocks({7, 7}), block_count, "border that repeats a block");
  CheckFailureIsSame(grid, range, ToBlocks({1, 5}), block_count, "border block inside the range");
  CheckFailureIsSame(grid, range, ToBlocks({1, block_count}), block_count, "border index beyond the whole grid");
  CheckFailureIsSame(
      grid,
      {.first_block = block_count - 1, .block_count = 2},
      {},
      block_count,
      "range beyond the grid"
  );
  CheckFailureIsSame(grid, range, ToBlocks({1, 2, 8}), 4, "too little room");
}

// After a failed build into buffers, a good build into the same buffers gives the CPU's piece.
TEST(BuildGridPiece, OnGpuAfterFailure_BuildsTheCpuPiece) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(1);
  ASSERT_EQ(grids.size(), 1U);
  const GridOnBothProcessors& grid = grids.front();
  const std::uint32_t block_count = grid.on_cpu.grid.counts().cell_blocks;
  ASSERT_GE(block_count, 12U);
  const CellBlockRange range = {.first_block = 4, .block_count = 2};
  const std::vector<CellBlockIndex> good_border = ToBlocks({1, 2, 8});
  std::optional<PieceBuffersRoom> room_on_cpu = BuildPieceBuffersRoom(RunOn::kCpu, block_count);
  std::optional<PieceBuffersRoom> room_on_gpu = BuildPieceBuffersRoom(RunOn::kGpu, block_count);
  ASSERT_TRUE(room_on_cpu.has_value());
  ASSERT_TRUE(room_on_gpu.has_value());
  const std::vector<CellBlockIndex> bad_border = ToBlocks({7, 6});
  const Buffer<CellBlockIndex> bad_buffer = BuildBorderBuffer(grid.on_gpu.grid, bad_border);
  const std::expected<GridPiece, GridFailure> failed = BuildGridPiece(
      grid.on_gpu.grid,
      range,
      bad_buffer,
      static_cast<std::uint32_t>(bad_border.size()),
      room_on_gpu->buffers
  );
  ASSERT_FALSE(failed.has_value());
  const Buffer<CellBlockIndex> good_buffer = BuildBorderBuffer(grid.on_gpu.grid, good_border);
  const std::expected<GridPiece, GridFailure> piece_on_gpu = BuildGridPiece(
      grid.on_gpu.grid,
      range,
      good_buffer,
      static_cast<std::uint32_t>(good_border.size()),
      room_on_gpu->buffers
  );
  const std::expected<GridPiece, GridFailure> piece_on_cpu =
      BuildGridPiece(grid.on_cpu.grid, range, good_border, room_on_cpu->buffers);
  ASSERT_TRUE(piece_on_gpu.has_value());
  ASSERT_TRUE(piece_on_cpu.has_value());
  const std::optional<PieceReadBack> read_back = ReadBackOfGpuPiece(*piece_on_gpu);
  ASSERT_TRUE(read_back.has_value());
  const std::vector<PieceReadBack> on_cpu = {
      ToReadBack(*piece_on_cpu, piece_on_cpu->grid(), piece_on_cpu->block_in_whole_grid()),
  };
  const std::vector<PieceReadBack> on_gpu = {*read_back};
  CheckSameResults(std::span<const PieceReadBack>(on_cpu), std::span<const PieceReadBack>(on_gpu), "after a failure");
}

// One build on the GPU allocates nothing on the CPU: the calls are counted apart.
void CheckBuildAllocatesNothing(
    const GridOnBothProcessors& grid,
    CellBlockRange range,
    const Buffer<CellBlockIndex>& border,
    std::uint32_t border_count
) {
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(RunOn::kGpu, grid.on_gpu.grid.counts().cell_blocks);
  ASSERT_TRUE(room.has_value());
  const std::uint64_t before = ComputeAllocationCount();
  const std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(grid.on_gpu.grid, range, border, border_count, room->buffers);
  const std::uint64_t after = ComputeAllocationCount();
  EXPECT_TRUE(piece.has_value());
  EXPECT_EQ(after - before, 0U) << range.first_block;
}

// Each build on the GPU allocates nothing on the CPU. One call is made first and not counted: CUDA loads its code for
// the GPU functions at the first call, and may allocate on the CPU. The grid build's allocation test does the same.
TEST(BuildGridPiece, OnGpu_AllocatesNothing) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(1);
  ASSERT_EQ(grids.size(), 1U);
  const GridOnBothProcessors& grid = grids.front();
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(grid.on_gpu.grid);
  ASSERT_TRUE(rig.has_value());
  const std::uint32_t block_count = grid.on_gpu.grid.counts().cell_blocks;
  const std::array<CellBlockRange, 3> ranges = {
      CellBlockRange{.first_block = 0, .block_count = 1},
      CellBlockRange{.first_block = 3, .block_count = 2},
      CellBlockRange{.first_block = 0, .block_count = block_count},
  };
  const std::vector<CellBlockIndex> warm_up_border =
      FindLayerOrFail(*rig, ranges.front(), {}, BorderWidth::kTouchingCells);
  // An empty border would leave some GPU functions unloaded for the counted calls.
  ASSERT_FALSE(warm_up_border.empty());
  {
    std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(RunOn::kGpu, block_count);
    ASSERT_TRUE(room.has_value());
    const std::expected<GridPiece, GridFailure> warm_up = BuildGridPiece(
        grid.on_gpu.grid,
        ranges.front(),
        rig->border_buffer,
        static_cast<std::uint32_t>(warm_up_border.size()),
        room->buffers
    );
    ASSERT_TRUE(warm_up.has_value());
  }
  for (const CellBlockRange range : ranges) {
    const std::vector<CellBlockIndex> border = FindLayerOrFail(*rig, range, {}, BorderWidth::kFourCellWidths);
    CheckBuildAllocatesNothing(grid, range, rig->border_buffer, static_cast<std::uint32_t>(border.size()));
  }
}

// Builds the piece of `range` on the GPU into `room`, reads it back and takes the buffers back for the next piece.
std::optional<PieceReadBack> ReadBackAndTakeBuffers(
    const StaggeredGrid& grid,
    CellBlockRange range,
    const BorderSearchRig& rig,
    std::pair<std::uint32_t, PieceBuffersRoom*> border_count_and_room
) {
  PieceBuffersRoom& room = *border_count_and_room.second;
  std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(grid, range, rig.border_buffer, border_count_and_room.first, room.buffers);
  EXPECT_TRUE(piece.has_value());
  if (!piece.has_value()) {
    return std::nullopt;
  }
  std::optional<PieceReadBack> read_back = ReadBackOfGpuPiece(*piece);
  EXPECT_TRUE(read_back.has_value());
  room.buffers = TakeBuffers(std::move(*piece));
  return read_back;
}

// The pieces of every tested range of one grid on the GPU, built twice into buffers that are used again for each piece.
std::vector<PieceReadBack> BuildPiecesOnGpuTwice(const GridOnBothProcessors& grid) {
  std::vector<PieceReadBack> pieces;
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(grid.on_gpu.grid);
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(RunOn::kGpu, grid.on_gpu.grid.counts().cell_blocks);
  EXPECT_TRUE(rig.has_value());
  EXPECT_TRUE(room.has_value());
  if (!rig.has_value() || !room.has_value()) {
    return pieces;
  }
  for (int pass = 0; pass < 2; ++pass) {
    for (const CellBlockRange range : BuildTestedRanges(grid.on_cpu.grid, kRangeBlocks)) {
      const std::vector<CellBlockIndex> border = FindLayerOrFail(*rig, range, {}, BorderWidth::kFourCellWidths);
      std::optional<PieceReadBack> read_back =
          ReadBackAndTakeBuffers(grid.on_gpu.grid, range, *rig, {static_cast<std::uint32_t>(border.size()), &*room});
      if (!read_back.has_value()) {
        return pieces;
      }
      pieces.push_back(std::move(*read_back));
    }
  }
  return pieces;
}

// The same pieces built twice into the same buffers, the second time over the tables of the pieces before: the bytes of
// every piece are equal, so nothing depends on what the buffers held or on the order of threads.
TEST(BuildGridPiece, OnGpuTwice_GivesIdenticalBytes) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(2);
  ASSERT_EQ(grids.size(), 2U);
  for (const GridOnBothProcessors& grid : grids) {
    const std::vector<PieceReadBack> both_passes = BuildPiecesOnGpuTwice(grid);
    ASSERT_FALSE(both_passes.empty());
    ASSERT_EQ(both_passes.size() % 2, 0U);
    const std::size_t pass_size = both_passes.size() / 2;
    const std::vector<PieceReadBack> first(
        both_passes.begin(),
        both_passes.begin() + static_cast<std::ptrdiff_t>(pass_size)
    );
    const std::vector<PieceReadBack> second(
        both_passes.begin() + static_cast<std::ptrdiff_t>(pass_size),
        both_passes.end()
    );
    EXPECT_EQ(
        ToResultWords(std::span<const PieceReadBack>(first)),
        ToResultWords(std::span<const PieceReadBack>(second))
    );
    WriteDigestOfResults("grid pieces on the GPU, twice", std::span<const PieceReadBack>(second));
  }
}

// The threads of the largest launch: 65,535 thread blocks of 256 (ComputeLaunchSize in grid/grid_piece_gpu.cu), the
// number a thread's entries repeat after in a grid-stride loop.
constexpr std::uint64_t kLargestLaunchThreadCount = 16776960;

// Expects two big pieces to be equal. A failure names the table, the range, the number of differing entries and where
// the first and the last differ (FindEntryDifferences): the comparison of words that CheckSameResults makes would hold
// hundreds of megabytes here, and thousands of megabytes of names on a failure.
template <typename Entry>
void CheckSameEntries(
    const std::vector<Entry>& on_cpu,
    const std::vector<Entry>& on_gpu,
    const EntryTablePlaces& places,
    CellBlockRange range
) {
  const std::optional<std::string> difference =
      FindEntryDifferences(std::span<const Entry>(on_cpu), std::span<const Entry>(on_gpu), places, range);
  EXPECT_FALSE(difference.has_value()) << difference.value_or("");
}

// Checks that two pieces have the same scalars and the same entry in every table, for the piece built for `range`.
void CheckSameBigPiece(const PieceReadBack& on_cpu, const PieceReadBack& on_gpu, CellBlockRange range) {
  EXPECT_EQ(on_gpu.version, on_cpu.version);
  EXPECT_EQ(on_gpu.block_count, on_cpu.block_count);
  EXPECT_EQ(on_gpu.range_first_block, on_cpu.range_first_block);
  EXPECT_EQ(on_gpu.active_cells, on_cpu.active_cells);
  EXPECT_EQ(on_gpu.face_pieces, on_cpu.face_pieces);
  EXPECT_EQ(on_gpu.active_cells_per_size_level, on_cpu.active_cells_per_size_level);
  const auto places_of = [](std::string_view name, std::uint64_t places_per_block) {
    return EntryTablePlaces{
        .name = name,
        .places_per_block = places_per_block,
        .thread_count = kLargestLaunchThreadCount,
    };
  };
  CheckSameEntries(on_cpu.map, on_gpu.map, places_of("map", 1), range);
  CheckSameEntries(on_cpu.block_ids, on_gpu.block_ids, places_of("block ids", 1), range);
  CheckSameEntries(on_cpu.neighbours, on_gpu.neighbours, places_of("neighbours", 36), range);
  CheckSameEntries(on_cpu.cell_states, on_gpu.cell_states, places_of("cell states", kCellPlacesPerBlock), range);
  CheckSameEntries(
      on_cpu.face_piece_kinds,
      on_gpu.face_piece_kinds,
      places_of("face piece kinds", kFacePlacesPerBlock),
      range
  );
}

// A grid of more blocks than one wave of threads has places for: 72,000 blocks have 17,280,000 face places, above the
// 16,776,960 threads of the largest launch (65,535 thread blocks of 256), so each thread of the face launch has more
// than one place. The piece of the whole grid and the piece of its middle with its border are the CPU's.
TEST(BuildGridPiece, OnGpuMoreFacePlacesThanThreads_EqualCpuPiece) {
  constexpr std::int32_t kBlockCount = 72000;
  const std::optional<GridOnBothProcessors> grid =
      BuildGridOnBothProcessors(ToUnitFrame(3), BuildTouchingCellsInManyBlocks(kBlockCount));
  ASSERT_TRUE(grid.has_value());
  ASSERT_EQ(grid->on_cpu.grid.counts().cell_blocks, static_cast<std::uint32_t>(kBlockCount));
  const std::array<CellBlockRange, 2> ranges = {
      CellBlockRange{.first_block = 0, .block_count = static_cast<std::uint32_t>(kBlockCount)},
      CellBlockRange{.first_block = 1000, .block_count = 70000},
  };
  const std::array<BorderWidth, 1> near_border = {BorderWidth::kTouchingCells};
  PiecesOfBothProcessors pieces;
  ApplyWithRooms(*grid, [&](RoomsOfBothProcessors rooms) {
    for (const CellBlockRange range : ranges) {
      SCOPED_TRACE(range.first_block);
      ApplyPiecesOfRange(*grid, range, near_border, rooms, pieces);
    }
  });
  ASSERT_EQ(pieces.on_cpu.size(), ranges.size());
  ASSERT_EQ(pieces.on_gpu.size(), ranges.size());
  for (std::size_t piece = 0; piece < ranges.size(); ++piece) {
    SCOPED_TRACE(piece);
    CheckSameBigPiece(pieces.on_cpu[piece], pieces.on_gpu[piece], ranges[piece]);
  }
  EXPECT_GT(pieces.on_cpu.back().range_first_block, 0U);
}

// The largest local memory of the GPU functions is at most the default stack of 1,024 bytes per thread, so the build
// needs no stack of the driver's beyond it, and at most the bytes of the Stack sentence of grid_piece_gpu.h.
TEST(BuildGridPiece, OnGpuFunctions_KeepUnderTheDefaultStack) {
  std::uint64_t bytes_per_thread = 0;
  ASSERT_EQ(ComputeLargestGridPieceStackOnGpu(&bytes_per_thread), RunOnCallStatus::kDone);
  EXPECT_LE(bytes_per_thread, 1024U);
  const std::array<std::uint64_t, 1> words = {bytes_per_thread};
  WriteResultDigest("grid piece stack bytes per thread on the GPU", std::span<const std::uint64_t>(words));
}

}  // namespace
}  // namespace fluidity::grid
