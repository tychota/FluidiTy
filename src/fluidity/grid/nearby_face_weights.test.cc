#include "fluidity/grid/nearby_face_weights.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <iomanip>
#include <ios>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/face_search_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/nearby_face_weights_cases.h"
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

// Checks that the faces of `list` are the faces of the checker's candidates for the size-level-1 cell {2,2,2} in the
// window (1, 0, 1), on the Y axis, as places in ascending order.
void CheckListIsTheCheckersList(const GridTables& tables, const GeometricGrid& checker, const NearbyFaceList& list) {
  std::vector<std::uint32_t> listed_places(static_cast<std::size_t>(list.count));
  for (int entry = 0; entry < list.count; ++entry) {
    listed_places[static_cast<std::size_t>(entry)] = list.faces[entry].value;
  }
  const std::vector<GeometricFacePiece> candidates = checker.NearbyFaces(ToCell(1, 2, 2, 2), {1, 0, 1}, Axis::kY);
  std::vector<std::uint32_t> checker_places(candidates.size());
  std::ranges::transform(candidates, checker_places.begin(), [&tables](const GeometricFacePiece& piece) {
    return FindFacePlace(tables, piece.id).value;
  });
  std::ranges::sort(checker_places);
  EXPECT_EQ(listed_places, checker_places);
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
  CheckListIsTheCheckersList(ToTablesOf(test_grid), GeometricGrid(ToTwoSizesWorstCaseFrame(), cells), list);
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

// The quotient is -(0.25 + 2^-54) in the cell with coordinate -1, so the position is 3/4 of a quarter below the cell's
// first quarter step: the quarter is 2. Subtracting the coordinate from the quotient first rounds to 0.75 and gives 3.
TEST(ComputeCandidateWindowQuarter, JustBelowAQuarterInCellMinusOne_RoundsDown) {
  const double quotient = -(0.25 + std::ldexp(1.0, -54));
  ASSERT_EQ(quotient, -std::nextafter(0.25, 1.0));
  EXPECT_EQ(ComputeCandidateWindowQuarter(quotient, 0.0, 1.0, -1), 2);
  EXPECT_EQ(ComputeCandidateWindowQuarter(-0.25, 0.0, 1.0, -1), 3);
  EXPECT_EQ(ComputeCandidateWindowQuarter(-0.25 - std::ldexp(1.0, -53), 0.0, 1.0, -1), 2);
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

// Sorts the X faces that FindNearbyFaces lists at a position and checks that their centres, in m, are exactly
// `expected_centres`, written in ascending order of x, then y, then z.
void CheckXFaceCentresAre(
    const GridTables& tables,
    Vec3 position,
    std::vector<std::array<double, 3>> expected_centres
) {
  GridError error = GridError::kInternalFault;
  const NearbyFaceList list = FindNearbyFaces(tables, position, Axis::kX, CellBlockIndex{kNoIndex}, &error);
  ASSERT_EQ(error, GridError::kNone);
  std::vector<std::array<double, 3>> listed_centres;
  for (int entry = 0; entry < list.count; ++entry) {
    const Vec3 centre = ComputeFacePosition(tables, list.faces[entry]);
    listed_centres.push_back({centre.x, centre.y, centre.z});
  }
  std::ranges::sort(listed_centres);
  std::ranges::sort(expected_centres);
  EXPECT_EQ(listed_centres, expected_centres) << "position x " << position.x;
}

// The cell {1,0,0} of size level 1 is 0.5 m wide, its +X neighbour is one cell of 1 m and the other neighbours are
// cells of 0.5 m, so no cell lies beyond the neighbour. Along X the faces within reach lie at x = 0 (n_f = -4 in
// quarter widths of the cell), 0.5 and 1 (n_f = 0 and 4), four each, at y and z of 0.25 or 0.75 m; the big cell's far
// side at x = 2 (n_f = 12) is out of every window. The window of a position in quarter n_p starts at n_f = n_p - 5, so
// the plane x = 0 leaves the list at quarter 2: x = 0.7 (quarter 1) lists 12 faces, x = 0.8 (quarter 2) lists 8. A
// window off by one quarter changes the list at one of the two positions.
TEST(FindNearbyFaces, CoarserNeighbour_ListsThePlaneOfTheWindowQuarter) {
  std::vector<CellId> cells = {ToCell(0, 1, 0, 0)};
  ApplySmallCellsToList(ToCell(0, 0, 0, 0), cells);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  std::vector<std::array<double, 3>> expected_centres;
  for (const double plane : {0.0, 0.5, 1.0}) {
    for (const double centre_y : {0.25, 0.75}) {
      for (const double centre_z : {0.25, 0.75}) {
        expected_centres.push_back({plane, centre_y, centre_z});
      }
    }
  }
  CheckXFaceCentresAre(tables, {.x = 0.7, .y = 0.25, .z = 0.25}, expected_centres);
  std::erase_if(expected_centres, [](const std::array<double, 3>& centre) {
    return centre[0] == 0.0;
  });
  CheckXFaceCentresAre(tables, {.x = 0.8, .y = 0.25, .z = 0.25}, expected_centres);
}

// Checks that FindNearbyFaces puts a position in `expected_cell` and in quarter `expected_quarter_x` of it along X,
// the window the 27 faces are counted in.
void CheckPositionIsInCellAndQuarter(
    const GridTables& tables,
    Vec3 position,
    CellId expected_cell,
    int expected_quarter_x
) {
  GridError error = GridError::kInternalFault;
  const NearbyFaceList list = FindNearbyFaces(tables, position, Axis::kX, CellBlockIndex{kNoIndex}, &error);
  ASSERT_EQ(error, GridError::kNone);
  EXPECT_EQ(ToCellId(tables, list.cell).value, expected_cell.value);
  const std::int32_t cell_coord_x = ToCellCoord(expected_cell).x;
  EXPECT_EQ(
      ComputeCandidateWindowQuarter(position.x, tables.frame.origin.x, list.cell_width, cell_coord_x),
      expected_quarter_x
  );
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
    CheckPositionIsInCellAndQuarter(ToTablesOf(test_grid), position, ToCell(0, -28, 2, 2), steps < 0 ? 1 : 2);
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
    // x - origin of the position one step below the side rounds to the side (2.125), so FindCellAt gives all three
    // positions to cell 34, in its first quarter.
    CheckPositionIsInCellAndQuarter(ToTablesOf(test_grid), position, ToCell(0, 34, 10, 10), 0);
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

// Builds the mask that keeps exactly the faces of a list whose offset from the position along `axis` is `offset`, in m.
NearbyFaceMask
BuildMaskOfFacesAt(const GridTables& tables, const NearbyFaceWeights& weights, Axis axis, double offset) {
  NearbyFaceMask mask{};
  const std::vector<Vec3> offsets = ToFaceOffsets(tables, weights.list, weights.position);
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (ComputeValueAlong(offsets[static_cast<std::size_t>(position_in_list)], axis) == offset) {
      mask.bits[position_in_list / 64] |= std::uint64_t{1} << static_cast<unsigned>(position_in_list % 64);
    }
  }
  return mask;
}

// Writes a number with 3 digits after the point and an exponent, for RecordProperty.
std::string ToScientificText(double value) {
  std::ostringstream text;
  text << std::scientific << std::setprecision(3) << value;
  return text.str();
}

// Says whether an offset in m is within 1e-9 m of `expected`.
bool ComputeIsNearOffset(double actual, double expected) {
  return std::abs(actual - expected) < 1e-9;
}

// Builds the mask that keeps exactly the faces of a list whose offset from the position, in m, `is_kept_offset`
// accepts.
template <typename IsKeptOffset>
NearbyFaceMask
BuildMaskOfFacesWhere(const GridTables& tables, const NearbyFaceWeights& weights, IsKeptOffset is_kept_offset) {
  NearbyFaceMask mask{};
  const std::vector<Vec3> offsets = ToFaceOffsets(tables, weights.list, weights.position);
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (is_kept_offset(offsets[static_cast<std::size_t>(position_in_list)])) {
      mask.bits[position_in_list / 64] |= std::uint64_t{1} << static_cast<unsigned>(position_in_list % 64);
    }
  }
  return mask;
}

// Builds the mask that keeps only face `position_in_list` of a list.
NearbyFaceMask BuildMaskOfOneFace(int position_in_list) {
  NearbyFaceMask mask{};
  mask.bits[position_in_list / 64] = std::uint64_t{1} << static_cast<unsigned>(position_in_list % 64);
  return mask;
}

// Checks that every face of weight 0 lies exactly 1.5 widths (1.5 m, for the unit grids of these tests) away on some
// axis, and has a weight change of exactly 0.
void CheckZeroWeightFacesAreOneAndAHalfAway(const GridTables& tables, const NearbyFaceWeights& weights) {
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    const FaceWeight face_weight = ComputeFaceWeight(tables, weights, position_in_list);
    if (face_weight.weight == 0.0) {
      const Vec3 offset = face_weight.offset;
      EXPECT_TRUE(offset.x == 1.5 || offset.y == 1.5 || offset.z == 1.5) << position_in_list;
      CheckVectorIsNear(face_weight.weight_change, {.x = 0.0, .y = 0.0, .z = 0.0}, 0.0);
    }
  }
}

// Finds the weight a face has in a list, or -1 when the list does not hold the face.
double FindWeightOfFace(const GridTables& tables, const NearbyFaceWeights& weights, FacePlaceIndex face) {
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (weights.list.faces[position_in_list].value == face.value) {
      return ComputeFaceWeight(tables, weights, position_in_list).weight;
    }
  }
  return -1.0;
}

// What two lists share: how many faces, and the largest difference of the weight a shared face has in each.
struct SharedWeights {
  int count;
  double largest_difference;
};

// Compares the weights that the faces held by both lists get in each.
SharedWeights ComputeSharedWeights(
    const GridTables& tables,
    const NearbyFaceWeights& first_weights,
    const NearbyFaceWeights& second_weights
) {
  SharedWeights shared = {.count = 0, .largest_difference = 0.0};
  for (int position_in_list = 0; position_in_list < first_weights.list.count; ++position_in_list) {
    const double second_weight = FindWeightOfFace(tables, second_weights, first_weights.list.faces[position_in_list]);
    if (second_weight >= 0.0) {
      const double first_weight = ComputeFaceWeight(tables, first_weights, position_in_list).weight;
      ++shared.count;
      shared.largest_difference = std::max(shared.largest_difference, std::abs(second_weight - first_weight));
    }
  }
  return shared;
}

// Finds the position in the list of the kept face with the smallest weight above 0, or -1 when no face has weight.
int FindSmallestPositiveWeight(const GridTables& tables, const NearbyFaceWeights& weights) {
  int smallest_position = -1;
  double smallest_weight = std::numeric_limits<double>::infinity();
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    const double weight = ComputeFaceWeight(tables, weights, position_in_list).weight;
    if (weight > 0.0 && weight < smallest_weight) {
      smallest_weight = weight;
      smallest_position = position_in_list;
    }
  }
  return smallest_position;
}

// The uniform 5^3 grid of 1 m cells, size-level-0 cells {0..4}^3, whose middle cell is {2,2,2}.
std::expected<TestGrid, GridFailure> BuildUniformFiveByFiveGrid() {
  const std::vector<CellId> cells = BuildBoxOfCells(0, 4);
  return BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
}

// Checks one axis at the centre of the middle cell: complete, 9 faces of weight 0 along the axis, the others above 0.
void CheckCellCentreWeights(const GridTables& tables, Axis axis) {
  const NearbyFaceWeights weights = FindWeightsOfAllFaces(tables, {.x = 2.5, .y = 2.5, .z = 2.5}, axis);
  CheckIsCompleteWithUnitWeightSum(weights, kWeightSumTolerance);
  EXPECT_EQ(ComputeZeroWeightCount(tables, weights), 9);
  EXPECT_EQ(ComputePositiveWeightCount(tables, weights), 18);
  EXPECT_EQ(ComputeOffsetCountAt(ToFaceOffsets(tables, weights.list, weights.position), axis, 1.5), 9);
  CheckZeroWeightFacesAreOneAndAHalfAway(tables, weights);
}

// Checks one axis at a corner of the middle cell: complete, 15 faces of weight 0, 12 above 0.
void CheckCellCornerWeights(const GridTables& tables, Axis axis) {
  const NearbyFaceWeights weights = FindWeightsOfAllFaces(tables, {.x = 2.0, .y = 2.0, .z = 2.0}, axis);
  CheckIsCompleteWithUnitWeightSum(weights, kWeightSumTolerance);
  EXPECT_EQ(ComputeZeroWeightCount(tables, weights), 15);
  EXPECT_EQ(ComputePositiveWeightCount(tables, weights), 12);
  CheckZeroWeightFacesAreOneAndAHalfAway(tables, weights);
}

// Checks one axis of a position on the low X side of the middle cell: complete, `zero_count` faces of weight 0.
void CheckCellSideWeights(const GridTables& tables, Axis axis, int zero_count) {
  const NearbyFaceWeights weights = FindWeightsOfAllFaces(tables, {.x = 2.0, .y = 2.6, .z = 2.7}, axis);
  CheckIsCompleteWithUnitWeightSum(weights, kWeightSumTolerance);
  EXPECT_EQ(ComputeZeroWeightCount(tables, weights), zero_count);
  CheckZeroWeightFacesAreOneAndAHalfAway(tables, weights);
}

// Checks that the mean offset the particle's pass made is, bit for bit, what the reads of its faces add up to.
void CheckMeanOffsetIsReadBack(const GridTables& tables, const NearbyFaceWeights& weights) {
  Vec3 mean_offset = {.x = 0.0, .y = 0.0, .z = 0.0};
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    const FaceWeight face_weight = ComputeFaceWeight(tables, weights, position_in_list);
    EXPECT_EQ(face_weight.face.value, weights.list.faces[position_in_list].value);
    mean_offset.x += face_weight.weight * face_weight.offset.x;
    mean_offset.y += face_weight.weight * face_weight.offset.y;
    mean_offset.z += face_weight.weight * face_weight.offset.z;
  }
  CheckVectorIsNear(mean_offset, weights.mean_offset, 0.0);
}

// Checks that the spread entry zy the particle's pass made is, bit for bit, what the reads of its faces add up to.
void CheckSpreadEntryIsReadBack(const GridTables& tables, const NearbyFaceWeights& weights) {
  double spread_zy = 0.0;
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    const FaceWeight face_weight = ComputeFaceWeight(tables, weights, position_in_list);
    spread_zy += (face_weight.weight * face_weight.offset_from_mean.z) * face_weight.offset_from_mean.y;
  }
  EXPECT_EQ(spread_zy, weights.weighted_spread.zy);
}

// Checks that every kept face has weight 0 and a weight change of 0, 1.5 m away along X, and counts the kept faces.
int CheckKeptFacesHaveNoWeight(const GridTables& tables, const NearbyFaceWeights& weights) {
  int kept_count = 0;
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (ComputeIsKept(weights.kept, position_in_list)) {
      ++kept_count;
      const FaceWeight face_weight = ComputeFaceWeight(tables, weights, position_in_list);
      EXPECT_EQ(face_weight.weight, 0.0);
      EXPECT_EQ(face_weight.offset.x, 1.5);
      CheckVectorIsNear(face_weight.weight_change, {.x = 0.0, .y = 0.0, .z = 0.0}, 0.0);
      CheckVectorIsNear(face_weight.offset_from_mean, {.x = 0.0, .y = 0.0, .z = 0.0}, 0.0);
    }
  }
  return kept_count;
}

TEST(NearbyFaceWeights, OneAxis_Is1152Bytes) {
  EXPECT_EQ(sizeof(NearbyFaceWeights), 1152U);
}

// The position is the centre of the middle cell.
TEST(ComputeNearbyFaceWeights, AtExactCellCentre_IsCompleteWithNineZeroWeights) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
    SCOPED_TRACE(static_cast<int>(axis));
    CheckCellCentreWeights(ToTablesOf(test_grid), axis);
  }
}

// On the low X side of the cell: the X list has no face 1.5 widths away, the Y and Z lists have 9 each.
TEST(ComputeNearbyFaceWeights, OnCellSide_IsCompleteWithNineZeroWeights) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  CheckCellSideWeights(ToTablesOf(test_grid), Axis::kX, 0);
  CheckCellSideWeights(ToTablesOf(test_grid), Axis::kY, 9);
  CheckCellSideWeights(ToTablesOf(test_grid), Axis::kZ, 9);
}

// At a corner of the cell each axis has 15 faces 1.5 widths away on a cross axis or along its own, and 12 with weight.
TEST(ComputeNearbyFaceWeights, AtCellCorner_IsCompleteWithFifteenZeroWeights) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
    SCOPED_TRACE(static_cast<int>(axis));
    CheckCellCornerWeights(ToTablesOf(test_grid), axis);
  }
}

// The frame, the cells and the position of the worst case: 124 faces on the Y axis, with two sizes of face.
TEST(ComputeNearbyFaceWeights, TwoSizesWorstCase_WeightsSumToOne) {
  const std::vector<CellId> cells = BuildTwoSizesWorstCaseCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToTwoSizesWorstCaseFrame(), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
    SCOPED_TRACE(static_cast<int>(axis));
    const NearbyFaceWeights weights = FindWeightsOfAllFaces(tables, kTwoSizesWorstCasePosition, axis);
    EXPECT_FALSE(weights.is_complete);
    CheckWeightsAddUp(tables, weights);
  }
  // All 124 entries of the Y list are read back; the sums of what they give are the particle's own, bit for bit.
  const NearbyFaceWeights y_weights = FindWeightsOfAllFaces(tables, kTwoSizesWorstCasePosition, Axis::kY);
  ASSERT_EQ(y_weights.list.count, 124);
  EXPECT_EQ(y_weights.fit, NearbyFaceFit::kFull);
  CheckMeanOffsetIsReadBack(tables, y_weights);
  CheckSpreadEntryIsReadBack(tables, y_weights);
}

// Uniform grid; the mask keeps the 9 faces of one plane across X, so the kept faces have no spread along X.
TEST(ComputeNearbyFaceWeights, FacesOnOnePlane_IsMeanOnly) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 2.6, .y = 2.7, .z = 2.8};
  const NearbyFaceWeights all_kept = FindWeightsOfAllFaces(tables, position, Axis::kX);
  const NearbyFaceMask plane_mask = BuildMaskOfFacesAt(tables, all_kept, Axis::kX, 3.0 - position.x);
  const NearbyFaceWeights weights = ComputeNearbyFaceWeights(tables, all_kept.list, position, plane_mask);
  EXPECT_EQ(weights.fit, NearbyFaceFit::kMeanOnly);
  EXPECT_FALSE(weights.is_complete);
  EXPECT_GT(weights.weight_sum, kMinWeightSum);
  CheckMatrixIsNear(weights.inverse_weighted_spread, Mat3{}, 0.0);
  CheckWeightsAddUp(tables, weights);
}

// One kept face has too little weight to fit anything, though it has some: the weights still add up to 1.
TEST(ComputeNearbyFaceWeights, WeightSumBelowTheMinimum_IsMeanOnly) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 2.6, .y = 2.7, .z = 2.8};
  const NearbyFaceWeights all_kept = FindWeightsOfAllFaces(tables, position, Axis::kX);
  const int smallest_position = FindSmallestPositiveWeight(tables, all_kept);
  ASSERT_GE(smallest_position, 0);
  const NearbyFaceMask one_face = BuildMaskOfOneFace(smallest_position);
  const NearbyFaceWeights weights = ComputeNearbyFaceWeights(tables, all_kept.list, position, one_face);
  EXPECT_GT(weights.weight_sum, 0.0);
  EXPECT_LT(weights.weight_sum, kMinWeightSum);
  EXPECT_EQ(weights.fit, NearbyFaceFit::kMeanOnly);
  CheckMatrixIsNear(weights.inverse_weighted_spread, Mat3{}, 0.0);
  CheckWeightsAddUp(tables, weights);
}

// The 8 faces at the corners of the 3 x 3 x 3 lattice of an X list, at the position {2.6, 2.7, 2.8} of the unit grid:
// offsets x in {-0.6, 1.4}, y in {-1.2, 0.8} and z in {-1.3, 0.7}. The weight curve gives, per axis,
//   x: Q(0.6) + Q(1.4) = 0.5 x 0.9^2 + 0.5 x 0.1^2 = 0.405 + 0.005 = 0.41,
//   y: Q(1.2) + Q(0.8) = 0.5 x 0.3^2 + 0.5 x 0.7^2 = 0.045 + 0.245 = 0.29,
//   z: Q(1.3) + Q(0.7) = 0.5 x 0.2^2 + 0.5 x 0.8^2 = 0.02 + 0.32 = 0.34,
// so W = 0.41 x 0.29 x 0.34 = 0.040426, below the 0.1 minimum. The weights are products of the axes' weights, so the
// spread is diagonal; with p the scaled weight of the low face of an axis and 2 the distance between its two faces,
// each entry is 4 p (1 - p): x 4 x (0.405 / 0.41) (0.005 / 0.41) = 0.0482, y 0.5244, z 0.2215. Every pivot is far
// above the floor of 1e-12 times the trace, so the spread alone would be invertible; only W keeps the fit mean-only.
TEST(ComputeNearbyFaceWeights, SmallWeightSumWithFullRankSpread_IsMeanOnly) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 2.6, .y = 2.7, .z = 2.8};
  const NearbyFaceWeights all_kept = FindWeightsOfAllFaces(tables, position, Axis::kX);
  const NearbyFaceMask corner_mask = BuildMaskOfFacesWhere(tables, all_kept, [](Vec3 offset) {
    return (ComputeIsNearOffset(offset.x, -0.6) || ComputeIsNearOffset(offset.x, 1.4))
        && (ComputeIsNearOffset(offset.y, -1.2) || ComputeIsNearOffset(offset.y, 0.8))
        && (ComputeIsNearOffset(offset.z, -1.3) || ComputeIsNearOffset(offset.z, 0.7));
  });
  const NearbyFaceWeights weights = ComputeNearbyFaceWeights(tables, all_kept.list, position, corner_mask);
  EXPECT_NEAR(weights.weight_sum, 0.040426, 1e-12);
  EXPECT_LT(weights.weight_sum, kMinWeightSum);
  const double spread_diagonal_x = 4.0 * (0.405 / 0.41) * (0.005 / 0.41);
  CheckMatrixIsNear(
      weights.weighted_spread,
      {.xx = spread_diagonal_x, .xy = 0, .xz = 0, .yx = 0, .yy = 0.5244, .yz = 0, .zx = 0, .zy = 0, .zz = 0.2215},
      1e-3
  );
  EXPECT_TRUE(ComputeCholeskyInverse(weights.weighted_spread).is_invertible);
  EXPECT_EQ(weights.fit, NearbyFaceFit::kMeanOnly);
  CheckMatrixIsNear(weights.inverse_weighted_spread, Mat3{}, 0.0);
  CheckWeightsAddUp(tables, weights);
}

// The 8 faces in the corners of the lattice at x offsets -0.6 and 0.4 of the position {2.6, 2.7, 2.8} of the unit grid,
// as a list and a mask. Moving the position within its quarter of the cell, here to {x, y, 2.8} with x in [2.5, 2.75)
// and y in [2.5, 2.75), leaves the list as it is.
struct CornerFaces {
  NearbyFaceMask mask;
  NearbyFaceList list;
};

CornerFaces BuildCornerFacesOfTheBasePosition(const GridTables& tables) {
  const Vec3 base_position = {.x = 2.6, .y = 2.7, .z = 2.8};
  const NearbyFaceWeights all_kept = FindWeightsOfAllFaces(tables, base_position, Axis::kX);
  const NearbyFaceMask mask = BuildMaskOfFacesWhere(tables, all_kept, [](Vec3 offset) {
    return (ComputeIsNearOffset(offset.x, -0.6) || ComputeIsNearOffset(offset.x, 0.4))
        && (ComputeIsNearOffset(offset.y, -1.2) || ComputeIsNearOffset(offset.y, 0.8))
        && (ComputeIsNearOffset(offset.z, -1.3) || ComputeIsNearOffset(offset.z, 0.7));
  });
  return {.mask = mask, .list = all_kept.list};
}

// Computes the weights of the corner faces at the position {x, y, 2.8}.
NearbyFaceWeights
ComputeCornerFaceWeightsAt(const GridTables& tables, const CornerFaces& corner, double position_x, double position_y) {
  const Vec3 position = {.x = position_x, .y = position_y, .z = 2.8};
  return ComputeNearbyFaceWeights(tables, corner.list, position, corner.mask);
}

// Finds the smallest double y, in [2.70, 2.75], at which the weight sum of the corner faces at {2.6, y, 2.8} is at
// least kMinWeightSum. The sum grows with y: it is 0.0981 at y = 2.70 and 0.1057 at y = 2.75.
double FindYWhereWeightSumIsAtLeastTheMinimum(const GridTables& tables, const CornerFaces& corner) {
  double below = 2.70;
  double upper = 2.75;
  for (int step = 0; step < 60; ++step) {
    const double middle = 0.5 * (below + upper);
    if (ComputeCornerFaceWeightsAt(tables, corner, 2.6, middle).weight_sum < kMinWeightSum) {
      below = middle;
    } else {
      upper = middle;
    }
  }
  return upper;
}

// Finds the first x, going up from 2.6 a double at a time, at which the weight sum of the corner faces at {x, y, 2.8}
// equals `target`, or 0 when none of the next 1,000 doubles does. The sum falls by about 0.01 per metre of x, so a step
// of x moves it by less than a step of the sum near 0.1, and the doubles of the sum there are met one after the
// other.
double FindXWithWeightSum(const GridTables& tables, const CornerFaces& corner, double position_y, double target) {
  double position_x = 2.6;
  for (int step = 0; step < 1000; ++step) {
    if (ComputeCornerFaceWeightsAt(tables, corner, position_x, position_y).weight_sum == target) {
      return position_x;
    }
    position_x = std::nextafter(position_x, 3.0);
  }
  return 0.0;
}

// The weight sum exactly equal to kMinWeightSum fits in full, and the double just below it keeps the mean only: the
// test is "at least the minimum", so the sum on the line is on the fitted side.
TEST(ComputeNearbyFaceWeights, WeightSumExactlyTheMinimum_IsFull) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const CornerFaces corner = BuildCornerFacesOfTheBasePosition(tables);
  const double position_y = FindYWhereWeightSumIsAtLeastTheMinimum(tables, corner);
  const double just_below_minimum = std::nextafter(kMinWeightSum, 0.0);
  const double x_at_minimum = FindXWithWeightSum(tables, corner, position_y, kMinWeightSum);
  ASSERT_GT(x_at_minimum, 0.0);
  const NearbyFaceWeights at_minimum = ComputeCornerFaceWeightsAt(tables, corner, x_at_minimum, position_y);
  EXPECT_EQ(at_minimum.weight_sum, kMinWeightSum);
  EXPECT_EQ(at_minimum.fit, NearbyFaceFit::kFull);
  const double x_below_minimum = FindXWithWeightSum(tables, corner, position_y, just_below_minimum);
  ASSERT_GT(x_below_minimum, 0.0);
  const NearbyFaceWeights below_minimum = ComputeCornerFaceWeightsAt(tables, corner, x_below_minimum, position_y);
  EXPECT_EQ(below_minimum.weight_sum, just_below_minimum);
  EXPECT_EQ(below_minimum.fit, NearbyFaceFit::kMeanOnly);
}

// The two-plane scenario of the fit's limit: a unit grid, X axis, position {2.5 - 2e-6, 2.5, 2.5}, only the 9 faces at
// x = 2 and the 9 at x = 1 kept. The faces at x = 1 weigh about 2e-12 of the others, so S_xx is about 4e-12 against a
// floor of 5e-13: the fit is kFull with a condition number near 6e10. The field is 1 + b . (x_f - x_p). The row fitted
// from the values has an error of about 8e-6; fitted from the values minus their weighted mean it is right to 1e-13.
TEST(ComputeNearbyFaceWeights, TwoPlanesOneNearlyWeightless_CentredValuesFitTightly) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 2.5 - 2e-6, .y = 2.5, .z = 2.5};
  const NearbyFaceWeights all_kept = FindWeightsOfAllFaces(tables, position, Axis::kX);
  const NearbyFaceMask plane_mask = BuildMaskOfFacesWhere(tables, all_kept, [](Vec3 offset) {
    return ComputeIsNearOffset(offset.x, -1.5 + 2e-6) || ComputeIsNearOffset(offset.x, -0.5 + 2e-6);
  });
  const NearbyFaceWeights weights = ComputeNearbyFaceWeights(tables, all_kept.list, position, plane_mask);
  ASSERT_EQ(weights.fit, NearbyFaceFit::kFull);
  EXPECT_NEAR(weights.weight_sum, 0.5, 1e-5);
  const double condition_number = weights.weighted_spread.yy / weights.weighted_spread.xx;
  EXPECT_GT(condition_number, 1e10);
  constexpr LinearField kField = {.base_value = 1.0, .slope = {.x = 0.3, .y = -1.1, .z = 2.0}};
  const Vec3 centred_row = ComputeFittedMatrixRow(tables, weights, kField, ValueCentring::kCentred);
  const Vec3 uncentred_row = ComputeFittedMatrixRow(tables, weights, kField, ValueCentring::kUncentred);
  RecordProperty("condition_number", ToScientificText(condition_number));
  RecordProperty("uncentred_row_error", ToScientificText(math::Length(uncentred_row - kField.slope)));
  RecordProperty("centred_row_error", ToScientificText(math::Length(centred_row - kField.slope)));
  CheckVectorIsNear(centred_row, kField.slope, 1e-13);
}

TEST(ComputeNearbyFaceWeights, EmptyMask_IsNone) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 2.6, .y = 2.7, .z = 2.8};
  const NearbyFaceWeights all_kept = FindWeightsOfAllFaces(tables, position, Axis::kX);
  CheckHasNoWeight(ComputeNearbyFaceWeights(tables, all_kept.list, position, NearbyFaceMask{}));
}

// The position is the centre of the cell, and the mask keeps only the 9 faces 1.5 widths away along X: each has a
// weight of exactly 0, so W is exactly 0.
TEST(ComputeNearbyFaceWeights, OnlyZeroWeightFacesKept_IsNone) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 2.5, .y = 2.5, .z = 2.5};
  const NearbyFaceWeights all_kept = FindWeightsOfAllFaces(tables, position, Axis::kX);
  const NearbyFaceMask far_mask = BuildMaskOfFacesAt(tables, all_kept, Axis::kX, 1.5);
  const NearbyFaceWeights weights = ComputeNearbyFaceWeights(tables, all_kept.list, position, far_mask);
  CheckHasNoWeight(weights);
  EXPECT_EQ(CheckKeptFacesHaveNoWeight(tables, weights), 9);
}

// Two positions 1e-9 m apart on either side of the side between a 1 m cell and its 0.5 m neighbours, the Y list: a
// shared face gets a different weight, because the width, the candidates and the area factors follow the cell. This is
// the known open question of the force between cells of different sizes; a fix must change this test.
TEST(ComputeNearbyFaceWeights, CrossingSizeChange_WeightsJump) {
  const std::vector<CellId> cells = BuildCellWithEveryNeighbourSplit(1, 1, 1);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const NearbyFaceWeights in_big_cell = FindWeightsOfAllFaces(tables, {.x = 2.0 - 5e-10, .y = 1.6, .z = 1.6}, Axis::kY);
  const NearbyFaceWeights in_small_cell =
      FindWeightsOfAllFaces(tables, {.x = 2.0 + 5e-10, .y = 1.6, .z = 1.6}, Axis::kY);
  EXPECT_EQ(in_big_cell.list.cell_width, 1.0);
  EXPECT_EQ(in_small_cell.list.cell_width, 0.5);
  const SharedWeights shared = ComputeSharedWeights(tables, in_big_cell, in_small_cell);
  EXPECT_GT(shared.count, 0);
  EXPECT_GT(shared.largest_difference, 1e-3);
}

TEST(ComputeCholeskyInverse, DiagonalMatrix_GivesTheReciprocals) {
  const Mat3 matrix =
      {.xx = 2.0, .xy = 0.0, .xz = 0.0, .yx = 0.0, .yy = 4.0, .yz = 0.0, .zx = 0.0, .zy = 0.0, .zz = 8.0};
  const CholeskyInverse result = ComputeCholeskyInverse(matrix);
  ASSERT_TRUE(result.is_invertible);
  EXPECT_NEAR(result.inverse.xx, 0.5, 1e-16);
  EXPECT_NEAR(result.inverse.yy, 0.25, 1e-16);
  EXPECT_NEAR(result.inverse.zz, 0.125, 1e-16);
  EXPECT_EQ(result.inverse.xy, 0.0);
  EXPECT_EQ(result.inverse.zx, 0.0);
}

// The matrix {{4, 2, 0}, {2, 5, 1}, {0, 1, 3}} has determinant 4 (15 - 1) - 2 (6) = 44; its inverse is
// {{14, -6, 2}, {-6, 12, -4}, {2, -4, 16}} / 44.
TEST(ComputeCholeskyInverse, FullMatrix_GivesTheKnownInverse) {
  const Mat3 matrix =
      {.xx = 4.0, .xy = 2.0, .xz = 0.0, .yx = 2.0, .yy = 5.0, .yz = 1.0, .zx = 0.0, .zy = 1.0, .zz = 3.0};
  const CholeskyInverse result = ComputeCholeskyInverse(matrix);
  ASSERT_TRUE(result.is_invertible);
  EXPECT_NEAR(result.inverse.xx, 14.0 / 44.0, 1e-15);
  EXPECT_NEAR(result.inverse.xy, -6.0 / 44.0, 1e-15);
  EXPECT_NEAR(result.inverse.xz, 2.0 / 44.0, 1e-15);
  EXPECT_NEAR(result.inverse.yy, 12.0 / 44.0, 1e-15);
  EXPECT_NEAR(result.inverse.yz, -4.0 / 44.0, 1e-15);
  EXPECT_NEAR(result.inverse.zz, 16.0 / 44.0, 1e-15);
  EXPECT_EQ(result.inverse.yx, result.inverse.xy);
  EXPECT_EQ(result.inverse.zx, result.inverse.xz);
  EXPECT_EQ(result.inverse.zy, result.inverse.yz);
}

// A pivot at or below 1e-12 of the trace is too small to divide by: here the second one is 0.
TEST(ComputeCholeskyInverse, FlatMatrix_IsNotInvertible) {
  const Mat3 matrix =
      {.xx = 1.0, .xy = 1.0, .xz = 0.0, .yx = 1.0, .yy = 1.0, .yz = 0.0, .zx = 0.0, .zy = 0.0, .zz = 1.0};
  const CholeskyInverse result = ComputeCholeskyInverse(matrix);
  EXPECT_FALSE(result.is_invertible);
  EXPECT_EQ(result.inverse.xx, 0.0);
}

// Builds a diagonal matrix whose pivot number `pivot_entry` (0, 1 or 2, the entries xx, yy and zz) is exactly the
// limit, 1e-12 of the trace as the factorisation computes it, and whose other two entries are 1. The pivot is found by
// repeating pivot = 1e-12 x ((xx + yy) + zz), which settles on a double that equals its own limit.
Mat3 BuildDiagonalMatrixWithPivotAtTheLimit(int pivot_entry, double pivot) {
  std::array<double, 3> diagonal = {1.0, 1.0, 1.0};
  diagonal.at(static_cast<std::size_t>(pivot_entry)) = pivot;
  return {
      .xx = diagonal[0],
      .xy = 0.0,
      .xz = 0.0,
      .yx = 0.0,
      .yy = diagonal[1],
      .yz = 0.0,
      .zx = 0.0,
      .zy = 0.0,
      .zz = diagonal[2],
  };
}

double FindPivotEqualToItsLimit(int pivot_entry) {
  double pivot = 2e-12;
  for (int step = 0; step < 8; ++step) {
    const Mat3 matrix = BuildDiagonalMatrixWithPivotAtTheLimit(pivot_entry, pivot);
    pivot = kMinPivotShare * ((matrix.xx + matrix.yy) + matrix.zz);
  }
  return pivot;
}

// A pivot exactly equal to its limit is too small ("at most" the limit), and the next double above it is not, for the
// first, second and third pivot in turn.
TEST(ComputeCholeskyInverse, PivotExactlyAtTheLimit_IsNotInvertible) {
  for (const int pivot_entry : {0, 1, 2}) {
    SCOPED_TRACE(pivot_entry);
    const double pivot = FindPivotEqualToItsLimit(pivot_entry);
    const Mat3 at_limit = BuildDiagonalMatrixWithPivotAtTheLimit(pivot_entry, pivot);
    ASSERT_EQ(pivot, kMinPivotShare * ((at_limit.xx + at_limit.yy) + at_limit.zz));
    EXPECT_FALSE(ComputeCholeskyInverse(at_limit).is_invertible);
    const Mat3 above_limit = BuildDiagonalMatrixWithPivotAtTheLimit(pivot_entry, std::nextafter(pivot, 1.0));
    EXPECT_TRUE(ComputeCholeskyInverse(above_limit).is_invertible);
  }
}

TEST(ComputeCholeskyInverse, PivotJustAboveTheLimit_IsInvertible) {
  const double pivot = 2e-12 * 3.0;
  const Mat3 matrix =
      {.xx = 1.0, .xy = 0.0, .xz = 0.0, .yx = 0.0, .yy = 1.0, .yz = 0.0, .zx = 0.0, .zy = 0.0, .zz = pivot};
  EXPECT_TRUE(ComputeCholeskyInverse(matrix).is_invertible);
  const Mat3 flatter =
      {.xx = 1.0, .xy = 0.0, .xz = 0.0, .yx = 0.0, .yy = 1.0, .yz = 0.0, .zx = 0.0, .zy = 0.0, .zz = 1e-12 * 2.0};
  EXPECT_FALSE(ComputeCholeskyInverse(flatter).is_invertible);
}

}  // namespace
}  // namespace fluidity::grid
