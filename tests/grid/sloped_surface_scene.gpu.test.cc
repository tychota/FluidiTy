// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Runs the whole grid module on the scene of grid/testing/sloped_surface_scene.h, a box of liquid with a sloped
// surface, on the GPU, and expects the CPU's results with the same bits: the formulas use only + - * /, sqrt and floor,
// the GPU code is built with no fused multiply-add, and the method paper asks for equal bits then (<sec-run-on>). The
// grid is built on both processors from one list; the GPU's grid is read back and compared with the CPU's tables. For
// every range of 8 blocks, the GPU's stacked touching border (one call per layer) and the GPU's grid piece, read back,
// are compared with the CPU's. The weights of the three axes at the 1,000 positions are computed by the GPU function on
// the GPU's whole grid and, for the positions whose cell lies in a range, on the GPU's piece of that range, and are
// compared with the CPU's by bits. Everything read back goes to WriteResultDigest, so the repeat run shows the GPU gave
// the same bytes twice. A comparison that fails names the field and both values.

#include "fluidity/grid/testing/sloped_surface_scene.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_error_message.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_functions_answers.h"
#include "fluidity/grid/testing/grid_functions_gpu_cases.h"
#include "fluidity/grid/testing/grid_functions_result_words.h"
#include "fluidity/grid/testing/grid_functions_runner_gpu.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

constexpr std::array<Axis, 3> kAxes = {Axis::kX, Axis::kY, Axis::kZ};

// One layer of a stacked border as a comparison sees it: the count and the entries that were written.
struct BorderLayerResult {
  std::uint32_t count = 0;
  std::vector<std::uint32_t> blocks;
};

void ApplyWordsOf(const BorderLayerResult& result, ResultWords& words) {
  words.Add("count", result.count);
  std::size_t entry = 0;
  for (const std::uint32_t block : result.blocks) {
    words.AddEntry("blocks", entry++, block);
  }
}

BorderLayerResult ToLayerResult(std::span<const CellBlockIndex> layer) {
  BorderLayerResult result;
  result.count = static_cast<std::uint32_t>(layer.size());
  for (const CellBlockIndex block : layer) {
    result.blocks.push_back(block.value);
  }
  return result;
}

// A grid piece as a comparison sees it: its counts, its ranges, and one digest of the bytes of its piece map and of
// each of its tables, read back from the processor it was built on.
struct PieceDigest {
  std::uint64_t version = 0;
  std::uint32_t block_count = 0;
  std::uint32_t range_first_block = 0;  // The range as a run of the piece.
  std::uint32_t range_first_block_in_whole_grid = 0;
  std::uint32_t range_block_count = 0;
  std::uint64_t active_cells = 0;
  std::uint64_t face_pieces = 0;
  std::uint64_t map_digest = 0;
  std::uint64_t block_ids_digest = 0;
  std::uint64_t same_size_level_blocks_digest = 0;
  std::uint64_t bigger_block_digest = 0;
  std::uint64_t smaller_blocks_digest = 0;
  std::uint64_t cell_states_digest = 0;
  std::uint64_t face_piece_kinds_digest = 0;
};

void ApplyWordsOf(const PieceDigest& piece, ResultWords& words) {
  words.Add("version", piece.version);
  words.Add("block_count", piece.block_count);
  words.Add("range_first_block", piece.range_first_block);
  words.Add("range_first_block_in_whole_grid", piece.range_first_block_in_whole_grid);
  words.Add("range_block_count", piece.range_block_count);
  words.Add("active_cells", piece.active_cells);
  words.Add("face_pieces", piece.face_pieces);
  words.Add("map_digest", piece.map_digest);
  words.Add("block_ids_digest", piece.block_ids_digest);
  words.Add("same_size_level_blocks_digest", piece.same_size_level_blocks_digest);
  words.Add("bigger_block_digest", piece.bigger_block_digest);
  words.Add("smaller_blocks_digest", piece.smaller_blocks_digest);
  words.Add("cell_states_digest", piece.cell_states_digest);
  words.Add("face_piece_kinds_digest", piece.face_piece_kinds_digest);
}

// The digest of the bytes of `count` values from `first`.
template <typename Value>
std::uint64_t ComputeTableDigest(const Value* first, std::size_t count) {
  return ComputeResultDigest(std::as_bytes(std::span<const Value>(first, count)));
}

// The digest of a piece whose tables are on the CPU in `cpu_grid`, with the piece map `map` and the ranges of `piece`.
PieceDigest
BuildPieceDigest(const GridPiece& piece, const StaggeredGrid& cpu_grid, std::span<const CellBlockIndex> map) {
  const GridTables tables = cpu_grid.tables(RunOn::kCpu);
  const std::size_t block_count = tables.block_count;
  return {
      .version = tables.version.value,
      .block_count = tables.block_count,
      .range_first_block = piece.range().first_block,
      .range_first_block_in_whole_grid = piece.range_in_whole_grid().first_block,
      .range_block_count = piece.range().block_count,
      .active_cells = cpu_grid.counts().active_cells,
      .face_pieces = cpu_grid.counts().face_pieces,
      .map_digest = ComputeTableDigest(map.data(), block_count),
      .block_ids_digest = ComputeTableDigest(tables.block_ids, block_count),
      .same_size_level_blocks_digest = ComputeTableDigest(tables.same_size_level_blocks, 27 * block_count),
      .bigger_block_digest = ComputeTableDigest(tables.bigger_block, block_count),
      .smaller_blocks_digest = ComputeTableDigest(tables.smaller_blocks, 8 * block_count),
      .cell_states_digest = ComputeTableDigest(tables.cell_states, kCellPlacesPerBlock * block_count),
      .face_piece_kinds_digest = ComputeTableDigest(tables.face_piece_kinds, kFacePlacesPerBlock * block_count),
  };
}

// What every range needs on both processors: the border search memory, the memory of the pieces, and the CPU memory
// that a GPU piece is read back into.
struct MemoryOfBothProcessors {
  BorderSearchRig& search_on_cpu;
  BorderSearchRig& search_on_gpu;
  PieceBuffersRoom& room_on_cpu;
  PieceBuffersRoom& room_on_gpu;
  PieceBuffersRoom& read_back_room;
};

// What the loop over the ranges collects, for the comparisons after it.
struct ResultsOfBothProcessors {
  std::vector<BorderLayerResult> layers_on_cpu;
  std::vector<BorderLayerResult> layers_on_gpu;
  std::vector<PieceDigest> pieces_on_cpu;
  std::vector<PieceDigest> pieces_on_gpu;
  std::vector<NearbyFaceResult> weights_on_cpu;
  std::vector<NearbyFaceResult> weights_on_gpu;
};

// Finds every layer of the stacked border of `range` on the rig, each on the list of the layer before, and appends
// each layer to `layers`. The last layer is the rig's border buffer; the result is its list.
std::vector<CellBlockIndex>
FindStackedBorder(BorderSearchRig& rig, CellBlockRange range, std::vector<BorderLayerResult>& layers) {
  std::vector<CellBlockIndex> border_so_far;
  for (const BorderWidth width : kLayerWidths) {
    border_so_far = FindLayerOrFail(rig, range, border_so_far, width);
    layers.push_back(ToLayerResult(border_so_far));
  }
  return border_so_far;
}

// Reads a piece of the GPU back: its tables with ReadGrid into CPU buffers, and its piece map with ReadBuffer. The
// tables must equal the CPU piece's. Gives the digest of what was read.
std::optional<PieceDigest>
ReadGpuPieceDigest(const GridPiece& gpu_piece, const GridPiece& cpu_piece, PieceBuffersRoom& read_back_room) {
  const std::uint32_t block_count = gpu_piece.grid().counts().cell_blocks;
  std::expected<StaggeredGrid, GridFailure> read_grid = ReadGrid(gpu_piece.grid(), read_back_room.buffers);
  EXPECT_TRUE(read_grid.has_value());
  std::vector<CellBlockIndex> map(block_count);
  const std::expected<void, BufferError> read_map =
      ReadBuffer<CellBlockIndex>(gpu_piece.block_in_whole_grid_buffer(), 0, map);
  EXPECT_TRUE(read_map.has_value());
  if (!read_grid.has_value() || !read_map.has_value()) {
    return std::nullopt;
  }
  EXPECT_TRUE(ComputeIsSameGrid(*read_grid, cpu_piece.grid()));
  const PieceDigest digest = BuildPieceDigest(gpu_piece, *read_grid, map);
  read_back_room.buffers = TakeBuffers(std::move(*read_grid));
  return digest;
}

// The queries of the positions at `entries`, three axes each, every face kept, in the order of the entries.
std::vector<NearbyFaceQuery> BuildWeightQueries(std::span<const Vec3> positions, std::span<const std::size_t> entries) {
  const NearbyFaceMask all_faces = {
      .bits = {~std::uint64_t{0}, ~std::uint64_t{0}, ~std::uint64_t{0}, ~std::uint64_t{0}},
  };
  std::vector<NearbyFaceQuery> queries;
  for (const std::size_t entry : entries) {
    for (const Axis axis : kAxes) {
      queries.push_back({.position = positions[entry], .axis = axis, .drawn_mask = all_faces});
    }
  }
  return queries;
}

// Asks the weights of the queries on the CPU's tables and on the GPU's, and appends both answers.
void ApplyWeightResults(
    const GridTables& cpu_tables,
    const GridTables& gpu_tables,
    std::span<const NearbyFaceQuery> queries,
    ResultsOfBothProcessors& results
) {
  if (queries.empty()) {
    return;
  }
  std::vector<NearbyFaceResult> on_gpu(queries.size());
  const RunOnCallStatus status =
      ComputeNearbyFaceResultsOnGpu(gpu_tables, queries.data(), static_cast<int>(queries.size()), on_gpu.data());
  ASSERT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
  for (const NearbyFaceQuery& query : queries) {
    results.weights_on_cpu.push_back(ComputeNearbyFaceResult(cpu_tables, query));
  }
  results.weights_on_gpu.insert(results.weights_on_gpu.end(), on_gpu.begin(), on_gpu.end());
}

// What the loop over the ranges works on: the grid on both processors, the positions, the memory and the results.
struct RangeRun {
  const GridOnBothProcessors& grid;
  std::span<const Vec3> positions;
  MemoryOfBothProcessors memory;
  ResultsOfBothProcessors& results;
};

// Builds the piece of `range` on both processors from the stacked border, compares the GPU's piece read back with the
// CPU's, and asks the weights of the range's positions, the entries of `position_entries`, in both pieces.
void ApplyRange(RangeRun& run, CellBlockRange range, std::span<const std::size_t> position_entries) {
  MemoryOfBothProcessors& memory = run.memory;
  ResultsOfBothProcessors& results = run.results;
  const std::vector<CellBlockIndex> border_on_cpu =
      FindStackedBorder(memory.search_on_cpu, range, results.layers_on_cpu);
  const std::vector<CellBlockIndex> border_on_gpu =
      FindStackedBorder(memory.search_on_gpu, range, results.layers_on_gpu);
  std::expected<GridPiece, GridFailure> piece_on_cpu =
      BuildGridPiece(run.grid.on_cpu.grid, range, border_on_cpu, memory.room_on_cpu.buffers);
  std::expected<GridPiece, GridFailure> piece_on_gpu = BuildGridPiece(
      run.grid.on_gpu.grid,
      range,
      memory.search_on_gpu.border_buffer,
      static_cast<std::uint32_t>(border_on_gpu.size()),
      memory.room_on_gpu.buffers
  );
  ASSERT_TRUE(piece_on_cpu.has_value());
  ASSERT_TRUE(piece_on_gpu.has_value()) << ToName(piece_on_gpu.error().error);
  EXPECT_EQ(piece_on_gpu->grid().run_on(), RunOn::kGpu);
  results.pieces_on_cpu.push_back(
      BuildPieceDigest(*piece_on_cpu, piece_on_cpu->grid(), piece_on_cpu->block_in_whole_grid())
  );
  const std::optional<PieceDigest> read_back = ReadGpuPieceDigest(*piece_on_gpu, *piece_on_cpu, memory.read_back_room);
  ASSERT_TRUE(read_back.has_value());
  results.pieces_on_gpu.push_back(*read_back);
  const std::vector<NearbyFaceQuery> queries = BuildWeightQueries(run.positions, position_entries);
  ApplyWeightResults(
      piece_on_cpu->grid().tables(RunOn::kCpu),
      piece_on_gpu->grid().tables(RunOn::kGpu),
      queries,
      results
  );
  memory.room_on_cpu.buffers = TakeBuffers(std::move(*piece_on_cpu));
  memory.room_on_gpu.buffers = TakeBuffers(std::move(*piece_on_gpu));
}

// The entries of the positions whose range is `range_entry`.
std::vector<std::size_t> FindPositionsOfRange(std::span<const std::size_t> range_of_position, std::size_t range_entry) {
  std::vector<std::size_t> entries;
  for (std::size_t entry = 0; entry < range_of_position.size(); ++entry) {
    if (range_of_position[entry] == range_entry) {
      entries.push_back(entry);
    }
  }
  return entries;
}

// Checks that the GPU's whole grid, read back, equals the CPU's: frame, counts, version and used tables.
void CheckWholeGridReadBack(const GridOnBothProcessors& grid, PieceBuffersRoom& read_back_room) {
  std::expected<StaggeredGrid, GridFailure> read_grid = ReadGrid(grid.on_gpu.grid, read_back_room.buffers);
  ASSERT_TRUE(read_grid.has_value());
  EXPECT_TRUE(ComputeIsSameGrid(*read_grid, grid.on_cpu.grid));
  read_back_room.buffers = TakeBuffers(std::move(*read_grid));
}

// Asks the weights of every position on the GPU's whole grid and on the CPU's, and expects the same bits.
void CheckWeightsInWholeGrid(const GridOnBothProcessors& grid, std::span<const Vec3> positions) {
  std::vector<std::size_t> every_position(positions.size());
  for (std::size_t entry = 0; entry < every_position.size(); ++entry) {
    every_position[entry] = entry;
  }
  ResultsOfBothProcessors results;
  ApplyWeightResults(
      grid.on_cpu.grid.tables(RunOn::kCpu),
      grid.on_gpu.grid.tables(RunOn::kGpu),
      BuildWeightQueries(positions, every_position),
      results
  );
  CheckSameResults(
      std::span<const NearbyFaceResult>(results.weights_on_cpu),
      std::span<const NearbyFaceResult>(results.weights_on_gpu),
      "weights of the 1,000 positions in the whole grid"
  );
  WriteDigestOfResults(
      "sloped surface scene on the GPU, weights in the whole grid",
      std::span<const NearbyFaceResult>(results.weights_on_gpu)
  );
}

// Compares what the loop over the ranges collected, and writes the GPU's side to the digest.
void CheckResultsOfBothProcessors(const ResultsOfBothProcessors& results, std::size_t position_count) {
  CheckSameResults(
      std::span<const BorderLayerResult>(results.layers_on_cpu),
      std::span<const BorderLayerResult>(results.layers_on_gpu),
      "layers of the stacked borders"
  );
  CheckSameResults(
      std::span<const PieceDigest>(results.pieces_on_cpu),
      std::span<const PieceDigest>(results.pieces_on_gpu),
      "grid pieces"
  );
  CheckSameResults(
      std::span<const NearbyFaceResult>(results.weights_on_cpu),
      std::span<const NearbyFaceResult>(results.weights_on_gpu),
      "weights of the positions in the piece of their range"
  );
  EXPECT_EQ(results.weights_on_gpu.size(), position_count * kAxes.size());
  WriteDigestOfResults(
      "sloped surface scene on the GPU, stacked borders",
      std::span<const BorderLayerResult>(results.layers_on_gpu)
  );
  WriteDigestOfResults(
      "sloped surface scene on the GPU, grid pieces",
      std::span<const PieceDigest>(results.pieces_on_gpu)
  );
  WriteDigestOfResults(
      "sloped surface scene on the GPU, weights in the pieces",
      std::span<const NearbyFaceResult>(results.weights_on_gpu)
  );
}

// Builds the stacked borders and the pieces of every range on both processors and compares them, then the weights of
// the positions in the piece of the range that holds each.
void CheckEveryRange(const GridOnBothProcessors& grid, std::span<const Vec3> positions) {
  const std::expected<std::vector<CellBlockRange>, GridFailure> ranges =
      BuildRanges(grid.on_cpu.grid, kSlopedSurfaceRangeBlocks);
  ASSERT_TRUE(ranges.has_value());
  const std::vector<std::size_t> range_of_position =
      FindRangeOfEachPosition(grid.on_cpu.grid.tables(RunOn::kCpu), *ranges, positions);
  const std::uint32_t block_count = grid.on_cpu.grid.counts().cell_blocks;
  std::expected<BorderSearchRig, GridFailure> search_on_cpu = BuildBorderSearchRig(grid.on_cpu.grid);
  std::expected<BorderSearchRig, GridFailure> search_on_gpu = BuildBorderSearchRig(grid.on_gpu.grid);
  std::optional<PieceBuffersRoom> room_on_cpu = BuildPieceBuffersRoom(RunOn::kCpu, block_count);
  std::optional<PieceBuffersRoom> room_on_gpu = BuildPieceBuffersRoom(RunOn::kGpu, block_count);
  std::optional<PieceBuffersRoom> read_back_room = BuildPieceBuffersRoom(RunOn::kCpu, block_count);
  ASSERT_TRUE(search_on_cpu.has_value());
  ASSERT_TRUE(search_on_gpu.has_value());
  ASSERT_TRUE(room_on_cpu.has_value());
  ASSERT_TRUE(room_on_gpu.has_value());
  ASSERT_TRUE(read_back_room.has_value());
  CheckWholeGridReadBack(grid, *read_back_room);
  ResultsOfBothProcessors results;
  RangeRun run = {
      .grid = grid,
      .positions = positions,
      .memory =
          {
              .search_on_cpu = *search_on_cpu,
              .search_on_gpu = *search_on_gpu,
              .room_on_cpu = *room_on_cpu,
              .room_on_gpu = *room_on_gpu,
              .read_back_room = *read_back_room,
          },
      .results = results,
  };
  for (std::size_t range_entry = 0; range_entry < ranges->size(); ++range_entry) {
    SCOPED_TRACE(testing::Message() << "range " << range_entry << " from block " << (*ranges)[range_entry].first_block);
    ApplyRange(run, (*ranges)[range_entry], FindPositionsOfRange(range_of_position, range_entry));
  }
  CheckResultsOfBothProcessors(results, positions.size());
}

// The scene on the GPU: the grid, the stacked touching borders and the pieces of the ranges of 8 blocks, and the
// weights at the 1,000 positions, in the whole grid and in the piece of the range that holds each, all equal to the
// CPU's by bits.
TEST(StaggeredGrid, SlopedSurfaceInBoxOnGpu_EqualsCpu) {
  const std::optional<GridOnBothProcessors> grid =
      BuildGridOnBothProcessors(BuildSlopedSurfaceFrame(), BuildSlopedSurfaceCells());
  ASSERT_TRUE(grid.has_value());
  const std::vector<Vec3> positions = BuildSlopedSurfacePositions();
  CheckWeightsInWholeGrid(*grid, positions);
  CheckEveryRange(*grid, positions);
}

}  // namespace
}  // namespace fluidity::grid
