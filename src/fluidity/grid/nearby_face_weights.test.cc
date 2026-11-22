#include "fluidity/grid/nearby_face_weights.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <limits>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/face_search_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/nearby_faces_cases.h"
#include "fluidity/grid/weight_curve.h"

namespace fluidity::grid {
namespace {

static_assert(sizeof(NearbyFaceList) == 888);
static_assert(sizeof(NearbyFaceMask) == 32);

// The offsets from `position` of the faces that FindNearbyFaces lists for one axis.
std::vector<Vec3> FindOffsets(const GridTables& tables, Vec3 position, Axis axis) {
  GridError error = GridError::kInternalFault;
  const NearbyFaceList list = FindNearbyFaces(tables, position, axis, CellBlockIndex{kNoIndex}, &error);
  EXPECT_EQ(error, GridError::kNone);
  return ToFaceOffsets(tables, list, position);
}

// Counts the faces with an offset of +1.5 widths on one of the two axes across `axis`, and checks that none has -1.5.
int ComputeCountAtPlusOneAndAHalfAcross(const std::vector<Vec3>& offsets, Axis axis) {
  int count = 0;
  for (const Vec3 offset : offsets) {
    bool is_at_plus_one_and_a_half = false;
    for (const Axis cross_axis : {Axis::kX, Axis::kY, Axis::kZ}) {
      const double cross_offset = ComputeValueAlong(offset, cross_axis);
      EXPECT_NE(cross_offset, -1.5);
      is_at_plus_one_and_a_half = is_at_plus_one_and_a_half || (cross_axis != axis && cross_offset == 1.5);
    }
    count += is_at_plus_one_and_a_half ? 1 : 0;
  }
  return count;
}

TEST(FindNearbyFaces, AllNeighboursSplit_Has76) {
  const std::vector<CellId> cells = BuildCellWithEveryNeighbourSplit(1, 1, 1);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  CheckEveryAxisHasCount(ToTablesOf(test_grid), {.x = 1.5, .y = 1.5, .z = 1.5}, 76);
}

TEST(FindNearbyFaces, TwoSizesWorstCase_Has124) {
  const std::vector<CellId> cells = BuildTwoSizesWorstCaseCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToTwoSizesWorstCaseFrame(), cells));
  ASSERT_TRUE(test_grid.has_value());
  GridError error = GridError::kInternalFault;
  const NearbyFaceList list =
      FindNearbyFaces(ToTablesOf(test_grid), kTwoSizesWorstCasePosition, Axis::kY, CellBlockIndex{kNoIndex}, &error);
  EXPECT_EQ(error, GridError::kNone);
  EXPECT_EQ(list.count, 124);
  EXPECT_EQ(list.axis, Axis::kY);
  EXPECT_EQ(list.cell_width, 2.0);
  CheckFacesAreAscending(list);
}

TEST(FindNearbyFacesWithRoom, MoreFacesThanRoom_IsInternalFault) {
  const std::vector<CellId> cells = BuildTwoSizesWorstCaseCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToTwoSizesWorstCaseFrame(), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  GridError error = GridError::kNone;
  const NearbyFaceList too_small =
      FindNearbyFacesWithRoom(tables, kTwoSizesWorstCasePosition, Axis::kY, CellBlockIndex{kNoIndex}, 100, &error);
  EXPECT_EQ(error, GridError::kInternalFault);
  EXPECT_EQ(too_small.count, 0);
  CheckEntriesFromAreZero(too_small, 100);
  const NearbyFaceList just_enough =
      FindNearbyFacesWithRoom(tables, kTwoSizesWorstCasePosition, Axis::kY, CellBlockIndex{kNoIndex}, 124, &error);
  EXPECT_EQ(error, GridError::kNone);
  EXPECT_EQ(just_enough.count, 124);
  const NearbyFaceList one_short =
      FindNearbyFacesWithRoom(tables, kTwoSizesWorstCasePosition, Axis::kY, CellBlockIndex{kNoIndex}, 123, &error);
  EXPECT_EQ(error, GridError::kInternalFault);
  EXPECT_EQ(one_short.count, 0);
}

TEST(ComputeCandidateWindowQuarter, AnyRounding_StaysInZeroToThree) {
  constexpr double kOrigin = -0.5;
  constexpr double kCellWidth = 0.0625;
  constexpr std::int32_t kCellCoord = 34;
  constexpr double kLowCorner = 1.625;
  constexpr double kHighCorner = 1.6875;
  EXPECT_EQ(ComputeCandidateWindowQuarter(std::nextafter(kLowCorner, 0.0), kOrigin, kCellWidth, kCellCoord), 0);
  EXPECT_EQ(ComputeCandidateWindowQuarter(kLowCorner - 0.01, kOrigin, kCellWidth, kCellCoord), 0);
  EXPECT_EQ(ComputeCandidateWindowQuarter(kHighCorner, kOrigin, kCellWidth, kCellCoord), 3);
  EXPECT_EQ(ComputeCandidateWindowQuarter(kHighCorner + 0.01, kOrigin, kCellWidth, kCellCoord), 3);
}

TEST(ComputeCandidateWindowQuarter, PositionsInsideTheCell_GiveTheirQuarter) {
  EXPECT_EQ(ComputeCandidateWindowQuarter(2.1, 0.0, 1.0, 2), 0);
  EXPECT_EQ(ComputeCandidateWindowQuarter(2.3, 0.0, 1.0, 2), 1);
  EXPECT_EQ(ComputeCandidateWindowQuarter(2.5, 0.0, 1.0, 2), 2);
  EXPECT_EQ(ComputeCandidateWindowQuarter(2.9, 0.0, 1.0, 2), 3);
}

TEST(ComputeCandidateWindowQuarter, NotANumber_IsZero) {
  EXPECT_EQ(ComputeCandidateWindowQuarter(std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0, 2), 0);
}

TEST(ComputeIsInCandidateWindow, TwelveQuarterSteps_LowEndOpenHighEndClosed) {
  EXPECT_FALSE(ComputeIsInCandidateWindow(-6, 1));
  EXPECT_TRUE(ComputeIsInCandidateWindow(-4, 1));
  EXPECT_TRUE(ComputeIsInCandidateWindow(7, 1));
  EXPECT_FALSE(ComputeIsInCandidateWindow(8, 1));
  EXPECT_TRUE(ComputeIsInCandidateWindow(-5, 0));
  EXPECT_TRUE(ComputeIsInCandidateWindow(6, 0));
}

// The origin and the width have no short binary fraction, so a test of float offsets would list 36 faces here.
TEST(FindNearbyFaces, CellCentreInNonDyadicFrame_Has27) {
  const GridFrame frame = {
      .origin = {.x = -17.616723516683763, .y = 0, .z = 0},
      .biggest_cell_width = 0.0375,
      .size_level_count = 1,
  };
  const std::vector<CellId> cells = BuildFiveByFiveBoxAround(-28, 2, 2);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  const double centre_x = -18.647973516683763;
  const double position_y_and_z = 2.3 * 0.0375;
  for (const int steps : {-3, -1, 0, 1, 3}) {
    const double position_x = ComputeMovedByRoundingSteps(centre_x, steps);
    const Vec3 position = {.x = position_x, .y = position_y_and_z, .z = position_y_and_z};
    CheckEveryAxisHasCount(ToTablesOf(test_grid), position, 27);
  }
}

// A test of float offsets would list 18 faces on the Y and Z axes here.
TEST(FindNearbyFaces, OneStepBelowCellSide_Has27) {
  const GridFrame frame = {
      .origin = {.x = -0.5, .y = -0.5, .z = -0.5},
      .biggest_cell_width = 0.0625,
      .size_level_count = 1,
  };
  const std::vector<CellId> cells = BuildFiveByFiveBoxAround(34, 10, 10);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  const double side_x = 1.625;
  const double position_y_and_z = -0.5 + (10.3 * 0.0625);
  for (const double position_x : {std::nextafter(side_x, 0.0), side_x, std::nextafter(side_x, 2.0)}) {
    const Vec3 position = {.x = position_x, .y = position_y_and_z, .z = position_y_and_z};
    CheckEveryAxisHasCount(ToTablesOf(test_grid), position, 27);
  }
}

// No offset is 1.5 widths. Uniform 5^3 grid, origin 0, width 1 m; the middle cell is {2,2,2}.
TEST(FindNearbyFaces, UniformGridOffCentre_Has27) {
  const std::vector<CellId> cells = BuildBoxOfCells(0, 4);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  CheckEveryAxisHasCount(ToTablesOf(test_grid), {.x = 2.6, .y = 2.7, .z = 2.8}, 27);
}

// For each axis the list holds the 9 faces at +1.5 widths along that axis and none at -1.5.
TEST(FindNearbyFaces, UniformGridAtCellCentre_Has27) {
  const std::vector<CellId> cells = BuildBoxOfCells(0, 4);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 2.5, .y = 2.5, .z = 2.5};
  CheckEveryAxisHasCount(tables, position, 27);
  for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
    const std::vector<Vec3> offsets = FindOffsets(tables, position, axis);
    EXPECT_EQ(ComputeOffsetCountAt(offsets, axis, 1.5), 9) << static_cast<int>(axis);
    EXPECT_EQ(ComputeOffsetCountAt(offsets, axis, -1.5), 0) << static_cast<int>(axis);
  }
}

// On the low X side, which belongs to the cell: the X list has no offset of 1.5 widths, and the Y and Z lists each hold
// 9 faces at +1.5 widths along x.
TEST(FindNearbyFaces, UniformGridOnCellSide_Has27) {
  const std::vector<CellId> cells = BuildBoxOfCells(0, 4);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 2.0, .y = 2.6, .z = 2.7};
  CheckEveryAxisHasCount(tables, position, 27);
  EXPECT_EQ(ComputeOffsetCountAt(FindOffsets(tables, position, Axis::kX), Axis::kX, 1.5), 0);
  EXPECT_EQ(ComputeOffsetCountAt(FindOffsets(tables, position, Axis::kY), Axis::kX, 1.5), 9);
  EXPECT_EQ(ComputeOffsetCountAt(FindOffsets(tables, position, Axis::kZ), Axis::kX, 1.5), 9);
}

// Each list holds 15 faces with an offset of +1.5 widths on one of its two cross axes (9 + 9 - 3), and none at -1.5.
TEST(FindNearbyFaces, UniformGridAtCellCorner_Has27) {
  const std::vector<CellId> cells = BuildBoxOfCells(0, 4);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 2.0, .y = 2.0, .z = 2.0};
  CheckEveryAxisHasCount(tables, position, 27);
  for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
    EXPECT_EQ(ComputeCountAtPlusOneAndAHalfAcross(FindOffsets(tables, position, axis), axis), 15)
        << static_cast<int>(axis);
  }
}

TEST(FindNearbyFaces, Outside_IsPositionOutsideGrid) {
  const std::vector<CellId> cells = BuildBoxOfCells(0, 4);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  GridError error = GridError::kNone;
  const NearbyFaceList list = FindNearbyFaces(
      ToTablesOf(test_grid),
      {.x = -0.5, .y = 2.5, .z = 2.5},
      Axis::kX,
      CellBlockIndex{kNoIndex},
      &error
  );
  EXPECT_EQ(error, GridError::kPositionOutsideGrid);
  EXPECT_EQ(list.count, 0);
}

TEST(ToMaskOfAll, OneHundredTwentyFourFaces_SetsExactlyThoseBits) {
  NearbyFaceList list{};
  list.count = 124;
  const NearbyFaceMask mask = ToMaskOfAll(list);
  EXPECT_EQ(mask.bits[0], ~std::uint64_t{0});
  EXPECT_EQ(mask.bits[1], (std::uint64_t{1} << 60U) - 1);
  EXPECT_EQ(mask.bits[2], 0U);
  EXPECT_EQ(mask.bits[3], 0U);
}

TEST(ToMaskOfAll, TwoHundredSixteenFaces_SetsBitsZeroTo215) {
  NearbyFaceList list{};
  list.count = kMaxNearbyFaces;
  const NearbyFaceMask mask = ToMaskOfAll(list);
  EXPECT_EQ(mask.bits[0], ~std::uint64_t{0});
  EXPECT_EQ(mask.bits[1], ~std::uint64_t{0});
  EXPECT_EQ(mask.bits[2], ~std::uint64_t{0});
  EXPECT_EQ(mask.bits[3], (std::uint64_t{1} << 24U) - 1);
}

TEST(ToMaskOfAll, NoFaces_SetsNoBit) {
  const NearbyFaceMask mask = ToMaskOfAll(NearbyFaceList{});
  EXPECT_EQ(mask.bits[0] | mask.bits[1] | mask.bits[2] | mask.bits[3], 0U);
}

}  // namespace
}  // namespace fluidity::grid
