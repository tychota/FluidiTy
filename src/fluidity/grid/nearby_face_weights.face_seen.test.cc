#include "fluidity/grid/nearby_face_weights.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/face_search_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/nearby_face_weights_cases.h"
#include "fluidity/grid/testing/nearby_faces_cases.h"

namespace fluidity::grid {
namespace {

// The uniform 5^3 grid of 1 m cells, size-level-0 cells {0..4}^3, whose middle cell is {2,2,2}.
std::expected<TestGrid, GridFailure> BuildUniformFiveByFiveGrid() {
  const std::vector<CellId> cells = BuildBoxOfCells(0, 4);
  return BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
}

// Finds the list of one axis at a position and says whether it holds `face`.
bool ComputeIsListed(const GridTables& tables, Vec3 position, Axis axis, FacePlaceIndex face) {
  GridError error = GridError::kInternalFault;
  const NearbyFaceList list = FindNearbyFaces(tables, position, axis, CellBlockIndex{kNoIndex}, &error);
  EXPECT_EQ(error, GridError::kNone);
  for (int position_in_list = 0; position_in_list < list.count; ++position_in_list) {
    if (list.faces[position_in_list].value == face.value) {
      return true;
    }
  }
  return false;
}

// Checks that a face outside the list of the position gets weight 0 and zero vectors, whatever the sums say.
void CheckFaceIsNotInList(const GridTables& tables, FacePlaceIndex face, const NearbyFaceWeights& weights) {
  const FaceWeight face_weight =
      ComputeFaceWeight(tables, face, weights.position, weights.list.cell, ToNearbyFaceSums(weights));
  const Vec3 zero_vector = {.x = 0.0, .y = 0.0, .z = 0.0};
  EXPECT_EQ(face_weight.face.value, face.value);
  EXPECT_EQ(face_weight.weight, 0.0);
  CheckVectorIsNear(face_weight.offset, zero_vector, 0.0);
  CheckVectorIsNear(face_weight.offset_from_mean, zero_vector, 0.0);
  CheckVectorIsNear(face_weight.weight_change, zero_vector, 0.0);
}

// A face of the X list of a position near the corner of the grid lies 3 cells from a position near the other corner:
// it is outside the window, so it gets nothing from the particle.
TEST(ComputeFaceWeight, FaceNotInList_IsZero) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 near_position = {.x = 2.6, .y = 2.7, .z = 2.8};
  const Vec3 far_position = {.x = 0.6, .y = 0.7, .z = 0.8};
  GridError error = GridError::kInternalFault;
  const NearbyFaceList far_list = FindNearbyFaces(tables, far_position, Axis::kX, CellBlockIndex{kNoIndex}, &error);
  ASSERT_EQ(error, GridError::kNone);
  ASSERT_GT(far_list.count, 0);
  const FacePlaceIndex far_face = far_list.faces[0];
  ASSERT_FALSE(ComputeIsListed(tables, near_position, Axis::kX, far_face));
  CheckFaceIsNotInList(tables, far_face, FindWeightsOfAllFaces(tables, near_position, Axis::kX));
}

// Finds the position in a list of the face whose offset is exactly `offset`, or -1; `offsets` are the offsets of the
// list's faces.
int FindPositionOfOffset(const std::vector<Vec3>& offsets, Vec3 offset) {
  for (std::size_t position_in_list = 0; position_in_list < offsets.size(); ++position_in_list) {
    const Vec3 listed = offsets[position_in_list];
    if (listed.x == offset.x && listed.y == offset.y && listed.z == offset.z) {
      return static_cast<int>(position_in_list);
    }
  }
  return -1;
}

// A face of the list gets the weight the list gives it: 0.5 x 0.75 x 0.75 at offset {-0.5, 0, 0}, as the example of
// ComputeFaceWeight says.
TEST(ComputeFaceWeight, FaceInList_GetsItsWeight) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 2.5, .y = 2.5, .z = 2.5};
  const NearbyFaceWeights weights = FindWeightsOfAllFaces(tables, position, Axis::kX);
  const NearbyFaceSums sums = ToNearbyFaceSums(weights);
  EXPECT_EQ(sums.weight_sum, weights.weight_sum);
  const int position_in_list =
      FindPositionOfOffset(ToFaceOffsets(tables, weights.list, position), {.x = -0.5, .y = 0.0, .z = 0.0});
  ASSERT_GE(position_in_list, 0);
  const FaceWeight from_list = ComputeFaceWeight(tables, weights, position_in_list);
  const FaceWeight from_face = ComputeFaceWeight(tables, from_list.face, position, weights.list.cell, sums);
  EXPECT_EQ(from_face.weight, from_list.weight);
  EXPECT_NEAR(from_face.weight, 0.28125, 1e-15);
}

// Finds the face place of `axis` whose centre is exactly `center`, or kNoIndex. `kind` is kNotAPiece to look for a
// place that holds no face piece, and any other kind to look for a face piece.
FacePlaceIndex FindPlaceAt(const GridTables& tables, Axis axis, Vec3 center, FacePieceKind kind) {
  const bool wants_piece = kind != FacePieceKind::kNotAPiece;
  for (std::uint32_t place = 0; place < tables.block_count * static_cast<std::uint32_t>(kFacePlacesPerBlock); ++place) {
    const FacePlaceIndex face = {.value = place};
    const bool is_piece = tables.face_piece_kinds[place] != FacePieceKind::kNotAPiece;
    const Vec3 face_center = ComputeFacePosition(tables, face);
    if (ToAxis(face) == axis && is_piece == wants_piece && face_center.x == center.x && face_center.y == center.y
        && face_center.z == center.z) {
      return face;
    }
  }
  return {.value = kNoIndex};
}

// Finds a face piece of `axis` at size level `level` whose centre lies within 1.4 cell widths of `position` on each
// axis (1 m widths), or kNoIndex.
FacePlaceIndex FindPieceNear(const GridTables& tables, Axis axis, int level, Vec3 position) {
  for (std::uint32_t place = 0; place < tables.block_count * static_cast<std::uint32_t>(kFacePlacesPerBlock); ++place) {
    const FacePlaceIndex face = {.value = place};
    if (ToAxis(face) != axis || tables.face_piece_kinds[place] == FacePieceKind::kNotAPiece
        || static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(face))) != level) {
      continue;
    }
    const Vec3 offset = ComputeFacePosition(tables, face) - position;
    if (std::abs(offset.x) <= 1.4 && std::abs(offset.y) <= 1.4 && std::abs(offset.z) <= 1.4) {
      return face;
    }
  }
  return {.value = kNoIndex};
}

// Checks that the face form gives weight 0 to `face` for the cell {1,1,1} of a grid whose 26 places around it are all
// split, at `position`, X axis, with every list face kept.
void CheckFaceOfSplitNeighbourIsZero(const GridTables& tables, FacePlaceIndex face, Vec3 position) {
  ASSERT_NE(face.value, kNoIndex);
  const NearbyFaceWeights weights = FindWeightsOfAllFaces(tables, position, Axis::kX);
  ASSERT_FALSE(ComputeIsListed(tables, position, Axis::kX, face));
  CheckFaceIsNotInList(tables, face, weights);
}

// <sec-nearby-faces>: the cell {1,1,1} has all 26 places around it split. The face at (1.0, 0.25, 1.25) lies between
// two small cells, {1,0,2} and {2,0,2}, that stop 0.5 m short of the cell: it is in the window of the position but is
// no face piece of the cell or of a touching cell, so the list does not hold it and it gets nothing.
TEST(ComputeFaceWeight, FaceOfFarSmallCells_IsZero) {
  const std::expected<TestGrid, GridFailure> test_grid =
      BuildTestGrid(ToInputs(ToUnitFrame(2), BuildCellWithEveryNeighbourSplit(1, 1, 1)));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const FacePlaceIndex face =
      FindPlaceAt(tables, Axis::kX, {.x = 1.0, .y = 0.25, .z = 1.25}, FacePieceKind::kBetweenCells);
  CheckFaceOfSplitNeighbourIsZero(tables, face, {.x = 1.1, .y = 1.1, .z = 1.6});
}

// <sec-nearby-faces>: the size-level-1 block over the cell {1,1,1} has places on the cell's middle plane x = 1.5, and
// none holds a face piece. One of them lies in the window of the position and still gets nothing.
TEST(ComputeFaceWeight, PlaceThatIsNotAPiece_IsZero) {
  const std::expected<TestGrid, GridFailure> test_grid =
      BuildTestGrid(ToInputs(ToUnitFrame(2), BuildCellWithEveryNeighbourSplit(1, 1, 1)));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const FacePlaceIndex place =
      FindPlaceAt(tables, Axis::kX, {.x = 1.5, .y = 1.25, .z = 1.25}, FacePieceKind::kNotAPiece);
  CheckFaceOfSplitNeighbourIsZero(tables, place, {.x = 1.6, .y = 1.6, .z = 1.6});
}

// <sec-nearby-faces>: a face piece two size levels below the cell's is in no list, even when its centre lies within
// 1.5 widths of the position.
TEST(ComputeFaceWeight, FaceTwoSizeLevelsFromCell_IsZero) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), BuildThreeSizeCells()));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 0.9, .y = 0.5, .z = 0.5};
  const FacePlaceIndex face = FindPieceNear(tables, Axis::kX, 2, position);
  ASSERT_NE(face.value, kNoIndex);
  ASSERT_FALSE(ComputeIsListed(tables, position, Axis::kX, face));
  CheckFaceIsNotInList(tables, face, FindWeightsOfAllFaces(tables, position, Axis::kX));
}

}  // namespace
}  // namespace fluidity::grid
