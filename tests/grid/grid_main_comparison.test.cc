#include <gtest/gtest.h>

#include <array>
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
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/testing/face_search_cases.h"
#include "fluidity/grid/testing/grid_main_comparison_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/nearby_face_weights_cases.h"

namespace fluidity::grid {
namespace {

// How far a new value may be from the old one: 1e-15 + 1e-13 |value|. A plain relative bound fails where the old value
// is exactly 0, as the slope at position index 0, where the position lies on face planes.
constexpr double kAbsoluteBound = 1e-15;
constexpr double kRelativeBound = 1e-13;

// Says whether `actual` is within the bound of `expected`, the bound scaled by `scale`.
bool ComputeIsWithinBound(double actual, double expected, double scale) {
  return std::abs(actual - expected) <= (kAbsoluteBound + (kRelativeBound * std::abs(expected))) * scale;
}

// The new axis of a face of the old axis: the old X, Y and Z are the new X, Z and Y.
Axis ToNewAxis(int old_axis) {
  constexpr std::array<Axis, 3> kNewAxes = {Axis::kX, Axis::kZ, Axis::kY};
  return kNewAxes.at(static_cast<std::size_t>(old_axis));
}

// The position, in the old Y-up axes and in m, of the face of an old entry: (lattice index + shift) h, with the shift 0
// along the face's axis and 0.5 along the two others.
Vec3 ToOldFacePosition(const MainEntry& entry) {
  const auto along = [&entry](int old_axis) {
    const double shift = entry.old_axis == old_axis ? 0.0 : 0.5;
    return (static_cast<double>(entry.old_lattice_index.at(static_cast<std::size_t>(old_axis))) + shift)
        * kMainCellWidth;
  };
  return {.x = along(0), .y = along(1), .z = along(2)};
}

// Finds the place in the list of weights of the face at `face_position`: the face the old entry names, found by its
// position. Gives -1 when no face of the list lies there.
int FindListPositionAt(const GridTables& tables, const NearbyFaceWeights& weights, Vec3 face_position) {
  const double tolerance = kMainCellWidth * 1e-6;
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    const Vec3 listed = ComputeFacePosition(tables, weights.list.faces[position_in_list]);
    const bool is_there = std::abs(listed.x - face_position.x) <= tolerance
        && std::abs(listed.y - face_position.y) <= tolerance && std::abs(listed.z - face_position.z) <= tolerance;
    if (is_there) {
      return position_in_list;
    }
  }
  return -1;
}

// Checks that a new vector is within the bound of the old one on each axis, scaled by `scale`.
void CheckVectorIsWithinBound(Vec3 actual, Vec3 expected, double scale) {
  EXPECT_TRUE(ComputeIsWithinBound(actual.x, expected.x, scale)) << actual.x << " against " << expected.x;
  EXPECT_TRUE(ComputeIsWithinBound(actual.y, expected.y, scale)) << actual.y << " against " << expected.y;
  EXPECT_TRUE(ComputeIsWithinBound(actual.z, expected.z, scale)) << actual.z << " against " << expected.z;
}

// What the old code gave the face of `entry`, against what the new code gives: a face of weight 0 may be left out of
// the list or have a weight within the absolute bound; any other face is in the list with the weight, the weight
// change and the offset within the bound (the offset in m, so the bound is scaled by h).
void CheckEntryMatches(
    const GridTables& tables,
    const NearbyFaceWeights& weights,
    const MainEntry& entry,
    Vec3 position
) {
  const Vec3 face_position = ToMainZUp(ToOldFacePosition(entry));
  const int position_in_list = FindListPositionAt(tables, weights, face_position);
  if (entry.weight == 0.0) {
    if (position_in_list >= 0) {
      EXPECT_LE(std::abs(ComputeFaceWeight(tables, weights, position_in_list).weight), kAbsoluteBound);
    }
    return;
  }
  ASSERT_GE(position_in_list, 0) << "the face of old weight " << entry.weight << " is not in the list";
  const FaceWeight face_weight = ComputeFaceWeight(tables, weights, position_in_list);
  EXPECT_TRUE(ComputeIsWithinBound(face_weight.weight, entry.weight, 1.0))
      << face_weight.weight << " against " << entry.weight;
  CheckVectorIsWithinBound(face_weight.weight_change, ToMainZUp(entry.weight_change), 1.0);
  CheckVectorIsWithinBound(face_weight.offset, face_position - position, kMainCellWidth);
}

// The faces of one position and axis in the old output.
constexpr std::size_t kMainFacesPerGroup = 27;

// The 16^3 cells of the new grid, 0.25 m wide, origin {0, -4, 0}: the old positions turned to Z-up lie inside it.
std::vector<CellId> BuildMainComparisonCells() {
  return BuildBoxOfCells(0, 15);
}

// Checks the 27 entries of the old code that start at `first_table_index`: one position and one old axis. The faces of
// the new code are listed once for them.
void CheckGroupMatches(const GridTables& tables, std::size_t first_table_index) {
  const MainEntry first_entry = ToMainEntry(kMainOldEntries.at(first_table_index));
  const Vec3 position = ToMainZUp(ToMainPosition(first_entry.position_index));
  const NearbyFaceWeights weights = FindWeightsOfAllFaces(tables, position, ToNewAxis(first_entry.old_axis));
  for (std::size_t table_index = first_table_index; table_index < first_table_index + kMainFacesPerGroup;
       ++table_index) {
    const MainEntry entry = ToMainEntry(kMainOldEntries.at(table_index));
    EXPECT_EQ(entry.position_index, first_entry.position_index);
    EXPECT_EQ(entry.old_axis, first_entry.old_axis);
    SCOPED_TRACE(
        testing::Message() << "position index " << entry.position_index << ", old axis " << entry.old_axis
                           << ", table index " << table_index
    );
    CheckEntryMatches(tables, weights, entry, position);
  }
}

// Uniform 16^3 grid at width 0.25 m, origin {0, -4, 0}. The 8,100 entries of the old code (100 positions, 3 axes, 27
// faces): every face the old code gave a weight is in the list of the new code, with the same weight, weight change and
// offset within 1e-15 + 1e-13 |value|; every face of old weight 0 is absent from the list or has weight within 1e-15.
TEST(ComputeNearbyFaceWeights, CompleteCase_MatchesMainWithinTolerance) {
  const GridFrame frame = {
      .origin = {.x = 0, .y = -4, .z = 0},
      .biggest_cell_width = kMainCellWidth,
      .size_level_count = 1,
  };
  const std::vector<CellId> cells = BuildMainComparisonCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  std::size_t visited_count = 0;
  for (std::size_t first_table_index = 0; first_table_index < kMainEntryCount;
       first_table_index += kMainFacesPerGroup) {
    CheckGroupMatches(tables, first_table_index);
    visited_count += kMainFacesPerGroup;
  }
  EXPECT_EQ(visited_count, kMainEntryCount);
}

// The old junction MakeJunction(3, true) of the old topology test (lines 22 to 37), turned to Z-up:
// three big cells and the 8 small cells of a corner. The old lattice counts small cells (0.5 m) from -4. A cell of
// low corner (x, y, z) and width w (1 for a small cell, 2 for a big one) in that lattice has the Z-up low corner
// (x, -(z + w), y), by (x, y, z) -> (x, -z, y), and every coordinate is moved up by 4 small cells on x and z so that no
// coordinate is negative. A big cell is size level 0 (1 m) at half that corner, a small cell size level 1.
CellId ToMainJunctionCell(int size_level, std::array<std::int32_t, 3> old_low_corner) {
  constexpr std::int32_t kMove = 4;
  const std::int32_t width = size_level == 0 ? 2 : 1;
  const std::int32_t cell_x = old_low_corner[0] + kMove;
  const std::int32_t cell_y = -(old_low_corner[2] + width);
  const std::int32_t cell_z = old_low_corner[1] + kMove;
  return ToCell(size_level, cell_x / width, cell_y / width, cell_z / width);
}

std::vector<CellId> BuildMainJunctionCells() {
  constexpr std::int32_t kLowCorner = -4;
  std::vector<CellId> cells;
  for (std::size_t plane_axis = 0; plane_axis < 3; ++plane_axis) {
    std::array<std::int32_t, 3> low_corner = {kLowCorner, kLowCorner, kLowCorner};
    low_corner.at(plane_axis) += 2;
    cells.push_back(ToMainJunctionCell(0, low_corner));
  }
  for (std::int32_t old_z = kLowCorner; old_z < kLowCorner + 2; ++old_z) {
    for (std::int32_t old_y = kLowCorner; old_y < kLowCorner + 2; ++old_y) {
      for (std::int32_t old_x = kLowCorner; old_x < kLowCorner + 2; ++old_x) {
        cells.push_back(ToMainJunctionCell(1, {old_x, old_y, old_z}));
      }
    }
  }
  return cells;
}

// What the face pieces between cells of a grid are: how many, whether every one is 0.25 m2, and how many have each
// centre distance.
struct BetweenCellsCounts {
  int piece_count = 0;
  int wrong_area_count = 0;
  int near_distance_count = 0;  // Centre distance 0.5 m.
  int far_distance_count = 0;  // Centre distance 0.75 m.
};

BetweenCellsCounts ComputeBetweenCellsCounts(const GridTables& tables) {
  BetweenCellsCounts counts;
  const std::size_t face_place_count = std::size_t{tables.block_count} * kFacePlacesPerBlock;
  for (std::size_t place = 0; place < face_place_count; ++place) {
    if (tables.face_piece_kinds[place] != FacePieceKind::kBetweenCells) {
      continue;
    }
    const FacePlaceIndex face = {.value = static_cast<std::uint32_t>(place)};
    ++counts.piece_count;
    counts.wrong_area_count += ComputeFaceArea(tables, face) == 0.25 ? 0 : 1;
    const double centre_distance = ComputeCellCenterDistance(tables, face);
    counts.near_distance_count += centre_distance == 0.5 ? 1 : 0;
    counts.far_distance_count += centre_distance == 0.75 ? 1 : 0;
  }
  return counts;
}

// The face pieces of the old junction, the numbers of the old test's own assertions (lines 166 to 177): 24 face pieces
// between cells, each 0.25 m2, 12 with a centre distance of 0.75 m (a big cell and a small one) and 12 with 0.5 m
// (two small cells).
TEST(BuildGrid, MainJunctionFixture_HasSamePiecesAsMain) {
  const GridFrame frame = {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = 2,
  };
  const std::vector<CellId> cells = BuildMainJunctionCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  const BetweenCellsCounts counts = ComputeBetweenCellsCounts(ToTablesOf(test_grid));
  EXPECT_EQ(counts.piece_count, 24);
  EXPECT_EQ(counts.wrong_area_count, 0);
  EXPECT_EQ(counts.far_distance_count, 12);
  EXPECT_EQ(counts.near_distance_count, 12);
}

}  // namespace
}  // namespace fluidity::grid
