#include <gtest/gtest.h>

#include <cstdint>
#include <expected>
#include <random>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/testing/face_search_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/nearby_face_weights_cases.h"
#include "fluidity/grid/testing/nearby_faces_cases.h"

namespace fluidity::grid {
namespace {

// The 4^3 cells in the middle of an 8^3 grid: coordinates 2..5 on each axis.
std::vector<CellId> BuildCentralCells() {
  std::vector<CellId> cells;
  for (std::int32_t cell_z = 2; cell_z <= 5; ++cell_z) {
    for (std::int32_t cell_y = 2; cell_y <= 5; ++cell_y) {
      for (std::int32_t cell_x = 2; cell_x <= 5; ++cell_x) {
        cells.push_back(ToCell(0, cell_x, cell_y, cell_z));
      }
    }
  }
  return cells;
}

// Uniform 8^3 grid, cells 0.25 m wide, 1,000 random positions inside the central 4^3 cells: on every axis the 27 faces
// give W = 1, a mean offset of 0 and a weighted spread of h^2 / 4 times the identity, to within 1e-14.
TEST(ComputeNearbyFaceWeights, CompleteSameSize_MatchesClosedForm) {
  const GridFrame frame = {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 0.25,
      .size_level_count = 1,
  };
  const std::vector<CellId> grid_cells = BuildBoxOfCells(0, 7);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, grid_cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::vector<CellId> central_cells = BuildCentralCells();
  std::mt19937_64 generator(20261121);  // NOLINT(bugprone-random-generator-seed)
  for (int draw = 0; draw < 1000; ++draw) {
    const PlacedPosition placed = FindDrawnPosition(generator, frame, central_cells, DrawnPlace::kAnywhere);
    for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
      SCOPED_TRACE(draw);
      CheckCompleteClosedForm(FindWeightsOfAllFaces(ToTablesOf(test_grid), placed.position, axis));
    }
  }
}

// How far W may be from 1 in the non-dyadic frame. A face centre there is a double near 18.6 m, so it is rounded by up
// to half a step of 3.6e-15 m, which is 4.7e-14 of a 0.0375 m cell width, and the product that places it adds about
// 3e-15 of the width. Each plane of faces then has a distance off by about 5e-14 of the width, and W changes by that
// times the sum of |Q'| over the planes of faces (Q' is the slope of the weight curve). That sum gets to 2 at a cell
// centre, where |Q'| is 1 at the two distances of 0.5, so W is off by up to about 1e-13, and adding the 27 products
// adds about 3e-15. The scaled weights still add up to 1 within kWeightSumTolerance. A bound of 2e-13 keeps that
// worst case inside, and a wrong formula is off by more than 1e-3.
constexpr double kNonDyadicWeightSumTolerance = 2e-13;

// The 1,000 positions of the non-dyadic frame, each on a cell side or centre moved by up to 3 rounding steps: every
// axis is complete, every weight is at least 0, the scaled weights add up to 1 within 1e-14 and W is 1 within 2e-13.
TEST(ComputeNearbyFaceWeights, NonDyadicFrameAtSideOrCentre_IsComplete) {
  const GridFrame frame = ToNonDyadicFrame();
  const std::vector<CellId> cells = BuildNonDyadicCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  std::mt19937_64 generator(kNonDyadicSeed);  // NOLINT(bugprone-random-generator-seed)
  for (int draw = 0; draw < 1000; ++draw) {
    const Vec3 position = FindDrawnNonDyadicPosition(generator, frame);
    for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
      SCOPED_TRACE(draw);
      const NearbyFaceWeights weights = FindWeightsOfAllFaces(ToTablesOf(test_grid), position, axis);
      CheckIsCompleteWithUnitWeightSum(weights, kNonDyadicWeightSumTolerance);
      CheckKeptWeightsAreNotNegative(ToTablesOf(test_grid), weights);
      EXPECT_NEAR(ComputeKeptWeightTotal(ToTablesOf(test_grid), weights), 1.0, kWeightSumTolerance);
    }
  }
}

}  // namespace
}  // namespace fluidity::grid
