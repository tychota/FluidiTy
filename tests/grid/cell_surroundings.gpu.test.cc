// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on a machine with a GPU.
//
// Asks the shared functions that find what lies around a cell or a face piece on the GPU, over every active cell and
// every face piece of 10 random grids, and expects the CPU's results: the touching cells, the face pieces of a cell,
// the corner places of cells and faces, the positions of corners, the cells around the edges, the cells near a face
// piece and the walks of a face search. The places are compared as integers and the corner positions bit for bit. The
// grids are built on the GPU, so this also shows that their tables answer as the CPU's do (corners have no table of
// their own). Every GPU result goes to WriteResultDigest under a unique name.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/grid_functions_answers.h"
#include "fluidity/grid/testing/grid_functions_gpu_cases.h"
#include "fluidity/grid/testing/grid_functions_result_words.h"
#include "fluidity/grid/testing/grid_functions_runner_gpu.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {
namespace {

constexpr int kGridCount = 10;
constexpr double kLargerRadius = 2.5;  // Widths of a cell of the start's size level.

// The places a runner is asked about, one list per grid, and the grids on both processors.
struct GridsAndPlaces {
  std::vector<GridOnBothProcessors> grids;
  std::vector<std::vector<CellPlaceIndex>> cell_places;  // The active cells of each grid.
  std::vector<std::vector<FacePlaceIndex>> face_places;  // The face pieces of each grid.
};

// Builds the 10 random grids on both processors and lists their active cells and face pieces, from the CPU tables.
GridsAndPlaces BuildGridsAndPlaces() {
  GridsAndPlaces built = {.grids = BuildRandomGridsOnBothProcessors(kGridCount), .cell_places = {}, .face_places = {}};
  for (const GridOnBothProcessors& grid : built.grids) {
    const GridTables cpu_tables = grid.on_cpu.grid.tables(RunOn::kCpu);
    built.cell_places.push_back(FindActiveCellPlaces(cpu_tables, grid.cells));
    built.face_places.push_back(FindFacePiecePlaces(cpu_tables));
  }
  return built;
}

// How one shared function is asked on each processor: its answer function, which the CPU loop calls, and its runner,
// which asks the same queries on the GPU.
template <typename Result, typename Query>
struct ProcessorsToAsk {
  Result (*answer_on_cpu)(const GridTables&, const Query&);
  RunOnCallStatus (*answer_on_gpu)(const GridTables&, const Query*, int, Result*);
};

// Asks every query on the GPU and expects the CPU's results of the same queries; writes the digest of the GPU results
// under `name`. Fails the calling test when the runner fails.
template <typename Result, typename Query>
void CheckGpuEqualsCpu(
    const GridOnBothProcessors& grid,
    std::span<const Query> queries,
    const ProcessorsToAsk<Result, Query>& processors,
    std::string_view name
) {
  ASSERT_FALSE(queries.empty());
  const GridTables cpu_tables = grid.on_cpu.grid.tables(RunOn::kCpu);
  std::vector<Result> on_cpu;
  on_cpu.reserve(queries.size());
  for (const Query& query : queries) {
    on_cpu.push_back(processors.answer_on_cpu(cpu_tables, query));
  }
  std::vector<Result> on_gpu(queries.size());
  const RunOnCallStatus status = processors.answer_on_gpu(
      grid.on_gpu.grid.tables(RunOn::kGpu),
      queries.data(),
      static_cast<int>(queries.size()),
      on_gpu.data()
  );
  ASSERT_EQ(status, RunOnCallStatus::kDone) << name << ": " << ReadLastRunnerFailure();
  CheckSameResults(std::span<const Result>(on_cpu), std::span<const Result>(on_gpu), name);
  WriteDigestOfResults(name, std::span<const Result>(on_gpu));
}

// The name of a result set of one grid, unique among the gpu tests.
std::string ToResultName(std::string_view function_name, int grid_entry) {
  return std::string(function_name) + " on the GPU, every place of grid " + std::to_string(grid_entry);
}

// The corner places of every active cell, and of every face piece, as the GPU finds them: the CPU's.
TEST(FindCornerPlaces, OnGpu_EqualsCpu) {
  const GridsAndPlaces built = BuildGridsAndPlaces();
  ASSERT_EQ(built.grids.size(), static_cast<std::size_t>(kGridCount));
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const auto grid_index = static_cast<std::size_t>(grid_entry);
    CheckGpuEqualsCpu<CellCorners, CellPlaceIndex>(
        built.grids.at(grid_index),
        built.cell_places.at(grid_index),
        {.answer_on_cpu = FindCellCornersOf, .answer_on_gpu = FindCellCornersOnGpu},
        ToResultName("FindCornerPlaces of cells", grid_entry)
    );
    CheckGpuEqualsCpu<FaceCorners, FacePlaceIndex>(
        built.grids.at(grid_index),
        built.face_places.at(grid_index),
        {.answer_on_cpu = FindFaceCornersOf, .answer_on_gpu = FindFaceCornersOnGpu},
        ToResultName("FindCornerPlaces of face pieces", grid_entry)
    );
  }
}

// The position of every corner place that an active cell holds, as the GPU computes it: the CPU's bits. The corner
// places come from the CPU's FindCornerPlaces, each once, in ascending order.
TEST(ComputeCornerPosition, OnGpu_SameBitsAsCpu) {
  const GridsAndPlaces built = BuildGridsAndPlaces();
  ASSERT_EQ(built.grids.size(), static_cast<std::size_t>(kGridCount));
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const auto grid_index = static_cast<std::size_t>(grid_entry);
    const GridOnBothProcessors& grid = built.grids.at(grid_index);
    const GridTables cpu_tables = grid.on_cpu.grid.tables(RunOn::kCpu);
    std::vector<CornerPlaceIndex> corner_places;
    for (const CellPlaceIndex cell : built.cell_places.at(grid_index)) {
      const CellCorners cell_corners = FindCornerPlaces(cpu_tables, cell);
      for (const CornerPlaceIndex corner : cell_corners.corners) {
        corner_places.push_back(corner);
      }
    }
    std::ranges::sort(corner_places, {}, &CornerPlaceIndex::value);
    const auto [removed_begin, removed_end] = std::ranges::unique(corner_places, {}, &CornerPlaceIndex::value);
    corner_places.erase(removed_begin, removed_end);
    CheckGpuEqualsCpu<Vec3, CornerPlaceIndex>(
        grid,
        corner_places,
        {.answer_on_cpu = ComputeCornerPositionOf, .answer_on_gpu = ComputeCornerPositionsOnGpu},
        ToResultName("ComputeCornerPosition", grid_entry)
    );
  }
}

// The cells around the three pairs of axes and the four edges of every active cell, as the GPU finds them: the CPU's.
TEST(FindCellsAroundEdge, OnGpu_EqualsCpu) {
  const GridsAndPlaces built = BuildGridsAndPlaces();
  ASSERT_EQ(built.grids.size(), static_cast<std::size_t>(kGridCount));
  constexpr std::array<std::array<Axis, 2>, 3> kAxisPairs = {
      std::array{Axis::kX, Axis::kY},
      std::array{Axis::kX, Axis::kZ},
      std::array{Axis::kY, Axis::kZ},
  };
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const auto grid_index = static_cast<std::size_t>(grid_entry);
    std::vector<EdgeQuery> queries;
    for (const CellPlaceIndex cell : built.cell_places.at(grid_index)) {
      for (const std::array<Axis, 2>& axis_pair : kAxisPairs) {
        for (int quarter = 0; quarter < 4; ++quarter) {
          queries.push_back(
              {.cell = cell, .first_axis = axis_pair[0], .second_axis = axis_pair[1], .quarter = quarter}
          );
        }
      }
    }
    CheckGpuEqualsCpu<CellsAroundEdge, EdgeQuery>(
        built.grids.at(grid_index),
        queries,
        {.answer_on_cpu = FindCellsAroundEdgeOf, .answer_on_gpu = FindCellsAroundEdgesOnGpu},
        ToResultName("FindCellsAroundEdge", grid_entry)
    );
  }
}

// The touching cells of every active cell, as the GPU finds them: the CPU's.
TEST(FindTouchingCells, OnGpu_EqualsCpu) {
  const GridsAndPlaces built = BuildGridsAndPlaces();
  ASSERT_EQ(built.grids.size(), static_cast<std::size_t>(kGridCount));
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const auto grid_index = static_cast<std::size_t>(grid_entry);
    CheckGpuEqualsCpu<TouchingCells, CellPlaceIndex>(
        built.grids.at(grid_index),
        built.cell_places.at(grid_index),
        {.answer_on_cpu = FindTouchingCellsOf, .answer_on_gpu = FindTouchingCellsOnGpu},
        ToResultName("FindTouchingCells", grid_entry)
    );
  }
}

// The face pieces of every active cell, as the GPU finds them: the CPU's.
TEST(FindFacePieces, OnGpu_EqualsCpu) {
  const GridsAndPlaces built = BuildGridsAndPlaces();
  ASSERT_EQ(built.grids.size(), static_cast<std::size_t>(kGridCount));
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const auto grid_index = static_cast<std::size_t>(grid_entry);
    CheckGpuEqualsCpu<CellFacePieces, CellPlaceIndex>(
        built.grids.at(grid_index),
        built.cell_places.at(grid_index),
        {.answer_on_cpu = FindFacePiecesOf, .answer_on_gpu = FindFacePiecesOnGpu},
        ToResultName("FindFacePieces", grid_entry)
    );
  }
}

// The cells near every face piece, as the GPU finds them: the CPU's.
TEST(FindCellsNearFace, OnGpu_EqualsCpu) {
  const GridsAndPlaces built = BuildGridsAndPlaces();
  ASSERT_EQ(built.grids.size(), static_cast<std::size_t>(kGridCount));
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const auto grid_index = static_cast<std::size_t>(grid_entry);
    CheckGpuEqualsCpu<CellsNearFace, FacePlaceIndex>(
        built.grids.at(grid_index),
        built.face_places.at(grid_index),
        {.answer_on_cpu = FindCellsNearFaceOf, .answer_on_gpu = FindCellsNearFacesOnGpu},
        ToResultName("FindCellsNearFace", grid_entry)
    );
  }
}

// The walk of a face search from every face piece, at the radii 1.5 and 2.5 cell widths, as the GPU makes it: the CPU's
// faces in the CPU's order. The CPU checks that no walk is cut at kMaxFaceSearchWalkFaces places.
TEST(BuildFaceSearch, OnGpu_WalksLikeCpu) {
  const GridsAndPlaces built = BuildGridsAndPlaces();
  ASSERT_EQ(built.grids.size(), static_cast<std::size_t>(kGridCount));
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const auto grid_index = static_cast<std::size_t>(grid_entry);
    std::vector<FaceSearchQuery> queries;
    for (const FacePlaceIndex face : built.face_places.at(grid_index)) {
      queries.push_back({.face = face, .radius_in_cell_widths = 1.5});
      queries.push_back({.face = face, .radius_in_cell_widths = kLargerRadius});
    }
    const GridOnBothProcessors& grid = built.grids.at(grid_index);
    const GridTables cpu_tables = grid.on_cpu.grid.tables(RunOn::kCpu);
    // The walk of the radius 1.5 lies inside that of 2.5 (the odd queries), so the cut check needs the bigger only.
    for (std::size_t query_index = 1; query_index < queries.size(); query_index += 2) {
      ASSERT_LE(FindFaceSearchWalk(cpu_tables, queries[query_index]).faces_found, kMaxFaceSearchWalkFaces);
    }
    CheckGpuEqualsCpu<FaceSearchWalk, FaceSearchQuery>(
        grid,
        queries,
        {.answer_on_cpu = FindFaceSearchWalk, .answer_on_gpu = WalkFaceSearchesOnGpu},
        ToResultName("BuildFaceSearch and FindNextFace", grid_entry)
    );
  }
}

}  // namespace
}  // namespace fluidity::grid
