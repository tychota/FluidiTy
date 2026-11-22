#include "fluidity/grid/nearby_face_weights.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <limits>
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
  EXPECT_GT(weights.weight_sum, kMinWeightSum);
  EXPECT_EQ(weights.inverse_weighted_spread.xx, 0.0);
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
  CheckWeightsAddUp(tables, weights);
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
