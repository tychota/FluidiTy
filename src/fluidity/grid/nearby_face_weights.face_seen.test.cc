#include "fluidity/grid/nearby_face_weights.h"

#include <gtest/gtest.h>

#include <cstddef>
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

// The sums are those of the X list and the face is one of the Y list of the same position: of another axis, it is
// never in the X list.
TEST(ComputeFaceWeight, FaceOfAnotherAxis_IsZero) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildUniformFiveByFiveGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const Vec3 position = {.x = 2.6, .y = 2.7, .z = 2.8};
  GridError error = GridError::kInternalFault;
  const NearbyFaceList y_list = FindNearbyFaces(tables, position, Axis::kY, CellBlockIndex{kNoIndex}, &error);
  ASSERT_EQ(error, GridError::kNone);
  ASSERT_GT(y_list.count, 0);
  CheckFaceIsNotInList(tables, y_list.faces[0], FindWeightsOfAllFaces(tables, position, Axis::kX));
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
  EXPECT_EQ(sums.axis, Axis::kX);
  EXPECT_EQ(sums.fit, NearbyFaceFit::kFull);
  EXPECT_EQ(sums.weight_sum, weights.weight_sum);
  const int position_in_list =
      FindPositionOfOffset(ToFaceOffsets(tables, weights.list, position), {.x = -0.5, .y = 0.0, .z = 0.0});
  ASSERT_GE(position_in_list, 0);
  const FaceWeight from_list = ComputeFaceWeight(tables, weights, position_in_list);
  const FaceWeight from_face = ComputeFaceWeight(tables, from_list.face, position, weights.list.cell, sums);
  EXPECT_EQ(from_face.weight, from_list.weight);
  EXPECT_NEAR(from_face.weight, 0.28125, 1e-15);
}

}  // namespace
}  // namespace fluidity::grid
