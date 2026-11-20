#include "fluidity/grid/grid_memory_estimate.h"

#include <cstdint>
#include <string_view>
#include <vector>

#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {
namespace {

// Bytes of one item of each array: a block id is 8 bytes and a block index 4. The 27 same-size-level neighbours include
// the block itself.
constexpr std::uint32_t kBlockIdBytes = 8;
constexpr std::uint32_t kSameSizeLevelNeighbourBlocksBytes = 27 * 4;
constexpr std::uint32_t kBiggerAndSmallerBlocksBytes = (1 + 8) * 4;  // The bigger block, then the 8 smaller blocks.
constexpr std::uint32_t kCellStateBytes = 1;
constexpr std::uint32_t kFacePieceKindBytes = 1;
constexpr std::uint32_t kCellIdBytes = 8;
constexpr std::uint32_t kSortKeyBytes = 8;
constexpr std::uint32_t kPieceMapBytes = 4;
// Bytes of one item of the two orders of the GPU build (a 32-bit index each).
constexpr std::uint32_t kOrderItemBytes = 4;

// The entry of an array counted per cell block, cell place or face place.
MemoryEstimateEntry BuildCountedEntry(
    std::string_view name,
    CountedPer counted_per,
    std::uint32_t bytes_each,
    std::uint32_t copies,
    MemoryWhere where
) {
  return {
      .name = name,
      .counted_per = counted_per,
      .given_count = 0,
      .bytes_each = bytes_each,
      .copies = copies,
      .kept_for = KeptFor::kWholeRun,
      .where = where,
  };
}

// The entry of an array whose item count is its own: bytes_each x given_count are its exact bytes.
MemoryEstimateEntry
BuildGivenEntry(std::string_view name, std::uint64_t given_count, std::uint32_t bytes_each, std::uint32_t copies) {
  return {
      .name = name,
      .counted_per = CountedPer::kGivenCount,
      .given_count = given_count,
      .bytes_each = bytes_each,
      .copies = copies,
      .kept_for = KeptFor::kWholeRun,
      .where = MemoryWhere::kRunOn,
  };
}

}  // namespace

MemoryCounts ToMemoryCounts(const GridCounts& counts, std::uint64_t particles) {
  const std::uint64_t cell_blocks = counts.cell_blocks;
  return {
      .cell_blocks = cell_blocks,
      .cell_places = std::uint64_t{kCellPlacesPerBlock} * cell_blocks,
      .face_places = std::uint64_t{kFacePlacesPerBlock} * cell_blocks,
      .corner_places = std::uint64_t{kCornerPlacesPerBlock} * cell_blocks,
      .particles = particles,
  };
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters): as the declaration.
std::vector<MemoryEstimateEntry> BuildGridMemoryEstimate(
    RunOn run_on,
    std::uint32_t grid_count,
    std::uint32_t max_cell_blocks,
    BuilderBuffersSharing builder_buffers_sharing
) {
  // One set of builder buffers is lent to all grids, or each grid has its own.
  const std::uint32_t builder_buffers_copies =
      builder_buffers_sharing == BuilderBuffersSharing::kOneForAllGrids ? 1 : grid_count;

  // The tables come first on both processors, always in the run's memory: 456 bytes per block.
  std::vector<MemoryEstimateEntry> estimate = {
      BuildCountedEntry("grid: block ids", CountedPer::kCellBlock, kBlockIdBytes, grid_count, MemoryWhere::kRunOn),
      BuildCountedEntry(
          "grid: same-size-level neighbour blocks",
          CountedPer::kCellBlock,
          kSameSizeLevelNeighbourBlocksBytes,
          grid_count,
          MemoryWhere::kRunOn
      ),
      BuildCountedEntry(
          "grid: bigger and smaller blocks",
          CountedPer::kCellBlock,
          kBiggerAndSmallerBlocksBytes,
          grid_count,
          MemoryWhere::kRunOn
      ),
      BuildCountedEntry("grid: cell states", CountedPer::kCellPlace, kCellStateBytes, grid_count, MemoryWhere::kRunOn),
      BuildCountedEntry(
          "grid: face piece kinds",
          CountedPer::kFacePlace,
          kFacePieceKindBytes,
          grid_count,
          MemoryWhere::kRunOn
      ),
  };

  if (run_on == RunOn::kCpu) {
    // The CPU builder sorts the cell ids with std::sort in CPU memory and needs no OrderBuffers.
    estimate.push_back(BuildCountedEntry(
        "grid: builder buffers: cell ids",
        CountedPer::kCellPlace,
        kCellIdBytes,
        builder_buffers_copies,
        MemoryWhere::kCpu
    ));
    estimate.push_back(
        BuildCountedEntry("grid: piece map", CountedPer::kCellBlock, kPieceMapBytes, grid_count, MemoryWhere::kCpu)
    );
    return estimate;
  }

  // Each array below is one piece of the run's memory, so an array with a per-item part and a fixed part is one entry
  // of its exact bytes: a per-item entry and a fixed entry would round up twice. A prefix sum writes its total after
  // the last entry, so the orders and the order buffers hold one item more than the cell places.
  const std::uint64_t order_item_count = (std::uint64_t{kCellPlacesPerBlock} * max_cell_blocks) + 1;
  const OrderBuffersSize order_buffers_size = ComputeOrderBuffersSize(RunOn::kGpu);
  const std::uint64_t order_buffers_bytes =
      (std::uint64_t{order_buffers_size.bytes_per_item} * order_item_count) + order_buffers_size.fixed_bytes;
  estimate.push_back(BuildCountedEntry(
      "grid: builder buffers: cell ids",
      CountedPer::kCellPlace,
      kCellIdBytes,
      builder_buffers_copies,
      MemoryWhere::kRunOn
  ));
  estimate.push_back(BuildCountedEntry(
      "grid: builder buffers: sort keys",
      CountedPer::kCellPlace,
      kSortKeyBytes,
      builder_buffers_copies,
      MemoryWhere::kRunOn
  ));
  estimate.push_back(
      BuildGivenEntry("grid: builder buffers: order", kOrderItemBytes * order_item_count, 1, builder_buffers_copies)
  );
  estimate.push_back(BuildGivenEntry(
      "grid: builder buffers: second order",
      kOrderItemBytes * order_item_count,
      1,
      builder_buffers_copies
  ));
  estimate.push_back(
      BuildGivenEntry("grid: builder buffers: order buffers", order_buffers_bytes, 1, builder_buffers_copies)
  );
  estimate.push_back(
      BuildCountedEntry("grid: piece map", CountedPer::kCellBlock, kPieceMapBytes, grid_count, MemoryWhere::kRunOn)
  );
  estimate.push_back(BuildGivenEntry("grid: build status", 1, sizeof(GridBuildStatus), grid_count));
  return estimate;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

}  // namespace fluidity::grid
