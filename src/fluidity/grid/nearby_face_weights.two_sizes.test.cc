#include "fluidity/grid/nearby_face_weights.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <expected>
#include <string_view>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/face_search_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/nearby_face_weights_cases.h"

// The weight of a face whose size differs from the particle's cell, checked against numbers worked out by hand from the
// method paper (<eq:weight-curve>, <eq:scaled-weights>, <eq:weighted-spread>), not from the code under test.
//
// Why these tests exist: the weights are divided by their sum, so the other mixed-size tests (weights add up to 1, a
// linear field is reproduced, the change is the derivative) pass for any positive raw weight used consistently. An area
// factor turned upside down, or a distance divided by the face's own width, passes them all. Here the sum W itself is
// checked, and W is the raw scale: it is exact, since every raw weight on this grid is a short binary fraction.
//
// The grid: the two-size fixture, the 1 m cell [0,1]^3 (size level 0) beside the 8 cells of 0.5 m that fill
// [1,2] x [0,1] x [0,1] (size level 1), 41 face pieces. A face piece between the two sizes has the area of the small
// cell, 0.25 m2 (<sec-face-pieces>).
//
// The formulas, h the width of the cell that holds the position x_p and x_f the centre of a face piece f:
//   d = (x_f - x_p) / h on each axis
//   Q(t) = 3/4 - t^2 for |t| < 1/2, (3/2 - |t|)^2 / 2 for 1/2 <= |t| < 3/2, else 0
//   Q'(t) = -2 t for |t| < 1/2, -(3/2 - |t|) sign(t) for 1/2 <= |t| < 3/2, else 0
//   w_f = Q(d_x) Q(d_y) Q(d_z) area_f / h^2
//   g_f = -(area_f / h^3) (Q'(d_x) Q(d_y) Q(d_z), Q(d_x) Q'(d_y) Q(d_z), Q(d_x) Q(d_y) Q'(d_z))
//   W = sum of w_f, G = sum of g_f, over every face of the list (all kept)
//   weight = w_f / W, weight_change = (g_f - weight G) / W, m = sum of weight (x_f - x_p), offset_from_mean = x_f - x_p
//   - m
//
// Worked in full: the position (0.75, 0.375, 0.625) in the 1 m cell, X axis, the face of 0.5 m centred at (1, 0.25,
// 0.75) on the side between the two sizes.
//   h = 1, so d = offset = (1/4, -1/8, 1/8).
//   Q(1/4) = 3/4 - 1/16 = 11/16; Q(-1/8) = Q(1/8) = 3/4 - 1/64 = 47/64.
//   Q'(1/4) = -1/2; Q'(-1/8) = 1/4; Q'(1/8) = -1/4.
//   area_f / h^2 = 0.25 / 1 = 1/4.
//   w_f = 11/16 x 47/64 x 47/64 x 1/4 = 24299/262144 = 0.09269332885742188.
//   g_f = -(1/4) (-1/2 x 47/64 x 47/64, 11/16 x 1/4 x 47/64, 11/16 x (-1/4) x 47/64)
//       = (2209/32768, -517/16384, 517/16384) = (0.067413330078125, -0.03155517578125, 0.03155517578125).
//   The list holds 9 faces of the X axis: the 1 m face at x = 0, the 4 faces of 0.5 m at x = 1 and the 4 at x = 1.5.
//   The 4 faces at x = 2 lie in the window but belong to small cells that do not touch the 1 m cell.
//   W = 4825/8192 = 0.5889892578125; G = (1309/8192, 439/2048, -439/2048).
//   weight = (24299/262144) / (4825/8192) = 24299/154400 = 0.15737694300518135.
//   weight_change = ((2209/32768) - (24299/154400)(1309/8192), ...) / (4825/8192)
//                 = (53460009/744980000, -20645361/186245000, 20645361/186245000).
//   m = (15479/154400, 8317/77200, -8317/77200), so offset_from_mean = (1/4 - 15479/154400, ...).
// The other faces, as exact fractions, sit beside their literals below.
//
// Read: docs/method-paper/fluidity-method.typ <sec-nearby-faces>, <eq:scaled-weights>, <eq:weighted-spread>

namespace fluidity::grid {
namespace {

// The relative tolerance of a value that is not a short binary fraction: a few roundings of a double.
constexpr double kRelativeTolerance = 1e-15;

// What one face gets from the position, worked from the paper.
struct PaperFaceWeight {
  std::string_view name;  // Which size relation the face shows.
  Vec3 center{};  // m, world. The face piece's centre.
  double raw_weight = 0;  // No unit. w_f, exact.
  Vec3 raw_weight_change{};  // 1/m. g_f, exact.
  double weight = 0;  // No unit. w_f / W.
  Vec3 weight_change{};  // 1/m. (g_f - weight G) / W.
  Vec3 offset_from_mean{};  // m. x_f - x_p - m.
};

// One position and one axis, with every face of its list kept, worked from the paper.
struct PaperWeights {
  Vec3 position{};  // m, world.
  Axis axis{};
  double cell_width = 0;  // m. h, the width of the cell that holds the position.
  int face_count = 0;  // The faces of the list.
  double weight_sum = 0;  // No unit. W, exact.
  Vec3 raw_weight_change_sum{};  // 1/m. G, exact.
  Vec3 mean_offset{};  // m. m.
  std::array<PaperFaceWeight, 3> faces{};
};

// The position in the 1 m cell, the X axis, across the change of size: a face of the cell's size, a smaller face beside
// the cell and a smaller face between two small cells.
// W = 4825/8192, G = (1309/8192, 439/2048, -439/2048), m = (15479/154400, 8317/77200, -8317/77200).
constexpr PaperWeights kInBigCell = {
    .position = {.x = 0.75, .y = 0.375, .z = 0.625},
    .axis = Axis::kX,
    .cell_width = 1.0,
    .face_count = 9,
    .weight_sum = 0.5889892578125,
    .raw_weight_change_sum = {.x = 0.1597900390625, .y = 0.21435546875, .z = -0.21435546875},
    .mean_offset = {.x = 0.10025259067357513, .y = 0.10773316062176166, .z = -0.10773316062176166},
    .faces = {
        {
            // d = (-3/4, 1/8, -1/8), area share 1: w_f = 9/32 x 47/64 x 47/64 = 19881/131072,
            // g_f = (-6627/16384, 423/8192, -423/8192), weight = 19881/77200,
            // weight_change = (-281826429/372490000, -563859/93122500, 563859/93122500).
            {
                .name = "the cell's size, 1 m at x = 0",
                .center = {.x = 0.0, .y = 0.5, .z = 0.5},
                .raw_weight = 0.15167999267578125,
                .raw_weight_change = {.x = -0.40447998046875, .y = 0.0516357421875, .z = -0.0516357421875},
                .weight = 0.25752590673575132,
                .weight_change = {.x = -0.75660132889473541, .y = -0.0060550242959542542, .z = 0.0060550242959542542},
                .offset_from_mean = {.x = -0.85025259067357517, .y = 0.017266839378238343, .z = -0.017266839378238343},
            },
            // The face worked in full in the file's comment.
            {
                .name = "smaller, 0.5 m at x = 1 beside the cell",
                .center = {.x = 1.0, .y = 0.25, .z = 0.75},
                .raw_weight = 0.09269332885742188,
                .raw_weight_change = {.x = 0.067413330078125, .y = -0.03155517578125, .z = 0.03155517578125},
                .weight = 0.15737694300518135,
                .weight_change = {.x = 0.071760327794034739, .y = -0.11085055169266289, .z = 0.11085055169266289},
                .offset_from_mean = {.x = 0.14974740932642486, .y = -0.23273316062176166, .z = 0.23273316062176166},
            },
            // d = (3/4, -1/8, 1/8), area share 1/4: w_f = 9/32 x 47/64 x 47/64 / 4 = 19881/524288,
            // g_f = (6627/65536, -423/32768, 423/32768), weight = 19881/308800,
            // weight_change = (229777971/1489960000, -16891659/372490000, 16891659/372490000).
            {
                .name = "smaller, 0.5 m at x = 1.5 between small cells",
                .center = {.x = 1.5, .y = 0.25, .z = 0.75},
                .raw_weight = 0.03791999816894531,
                .raw_weight_change = {.x = 0.1011199951171875, .y = -0.012908935546875, .z = 0.012908935546875},
                .weight = 0.06438147668393783,
                .weight_change = {.x = 0.15421754342398453, .y = -0.045347952965180273, .z = 0.045347952965180273},
                .offset_from_mean = {.x = 0.64974740932642483, .y = -0.23273316062176166, .z = 0.23273316062176166},
            },
        },
    },
};

// The position in the 0.5 m cell [1,1.5] x [0,0.5] x [0.5,1], the Y axis, along the change of size: a face of the
// cell's size and the two Y faces of the 1 m cell, which are bigger. 14 faces: 12 of 0.5 m and the 2 of 1 m.
// W = 24543/32768, G = (1873/2048, -141/1024, -209/1024), m = (21557/196344, -611/43632, -3199/392688).
constexpr PaperWeights kInSmallCell = {
    .position = {.x = 1.125, .y = 0.3125, .z = 0.5625},
    .axis = Axis::kY,
    .cell_width = 0.5,
    .face_count = 14,
    .weight_sum = 0.748992919921875,
    .raw_weight_change_sum = {.x = 0.91455078125, .y = -0.1376953125, .z = -0.2041015625},
    .mean_offset = {.x = 0.10979199771829035, .y = -0.014003483681701503, .z = -0.0081464164935012018},
    .faces = {
        {
            // d = (1/4, 3/8, 3/8), area share 1: w_f = 11/16 x 39/64 x 39/64 = 16731/65536,
            // g_f = (1521/4096, 1287/2048, 1287/2048), weight = 1859/5454,
            // weight_change = (5326880/66928761, 20116096/22309587, 62370880/66928761).
            {
                .name = "the cell's size, 0.5 m at y = 0.5",
                .center = {.x = 1.25, .y = 0.5, .z = 0.75},
                .raw_weight = 0.2552947998046875,
                .raw_weight_change = {.x = 0.371337890625, .y = 0.62841796875, .z = 0.62841796875},
                .weight = 0.34085075174184087,
                .weight_change = {.x = 0.079590297510512706, .y = 0.90167944390902444, .z = 0.93189951626326983},
                .offset_from_mean = {.x = 0.015208002281709652, .y = 0.20150348368170151, .z = 0.1956464164935012},
            },
            // d = (-5/4, -5/8, -1/8), area share 1 / 0.25 = 4: w_f = 1/32 x 49/128 x 47/64 x 4 = 2303/65536,
            // g_f = (-2303/4096, -329/2048, -49/2048), weight = 2303/49086,
            // weight_change = (-486688384/602358849, -41332928/200786283, -11540480/602358849).
            {
                .name = "bigger, 1 m at y = 0",
                .center = {.x = 0.5, .y = 0.0, .z = 0.5},
                .raw_weight = 0.0351409912109375,
                .raw_weight_change = {.x = -0.562255859375, .y = -0.16064453125, .z = -0.02392578125},
                .weight = 0.046917654728435808,
                .weight_change = {.x = -0.80797083799461211, .y = -0.20585533723934718, .z = -0.0191588120920923},
                .offset_from_mean = {.x = -0.7347919977182904, .y = -0.29849651631829849, .z = -0.054353583506498798},
            },
            // d = (-5/4, 11/8, -1/8), area share 4: w_f = 1/32 x 1/128 x 47/64 x 4 = 47/65536,
            // g_f = (-47/4096, 47/2048, -1/2048), weight = 47/49086,
            // weight_change = (-9932416/602358849, 6187456/200786283, -235520/602358849).
            {
                .name = "bigger, 1 m at y = 1",
                .center = {.x = 0.5, .y = 1.0, .z = 0.5},
                .raw_weight = 0.0007171630859375,
                .raw_weight_change = {.x = -0.011474609375, .y = 0.02294921875, .z = -0.00048828125},
                .weight = 0.00095750315772317974,
                .weight_change = {.x = -0.016489200775400246, .y = 0.030816129008175326, .z = -0.00039099616514474083},
                .offset_from_mean = {.x = -0.7347919977182904, .y = 0.70150348368170146, .z = -0.054353583506498798},
            },
        },
    },
};

std::expected<TestGrid, GridFailure> BuildTwoSizeFixtureGrid() {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  return BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
}

// Checks that a value is within kRelativeTolerance of `expected`, relative to `expected`.
void CheckIsNearRelative(double actual, double expected) {
  EXPECT_NEAR(actual, expected, kRelativeTolerance * std::abs(expected));
}

void CheckVectorIsNearRelative(Vec3 actual, Vec3 expected) {
  CheckIsNearRelative(actual.x, expected.x);
  CheckIsNearRelative(actual.y, expected.y);
  CheckIsNearRelative(actual.z, expected.z);
}

// Checks a weight change, (g - weight G) / W, against the paper's. The subtraction can cancel: for the 1 m face of the
// big-cell position, g_y = 0.0516 and weight G_y = 0.0552 leave 0.0036. Rounding scales with the two terms, not with
// what is left, so each entry's tolerance is kRelativeTolerance times (|g| + |weight G|) / W.
void CheckWeightChangeIsNear(Vec3 actual, const PaperWeights& paper, const PaperFaceWeight& face) {
  const Vec3 raw_change = face.raw_weight_change;
  const Vec3 scaled_change_sum = paper.raw_weight_change_sum * face.weight;
  const double tolerance_per_term = kRelativeTolerance / paper.weight_sum;
  EXPECT_NEAR(
      actual.x,
      face.weight_change.x,
      tolerance_per_term * (std::abs(raw_change.x) + std::abs(scaled_change_sum.x))
  );
  EXPECT_NEAR(
      actual.y,
      face.weight_change.y,
      tolerance_per_term * (std::abs(raw_change.y) + std::abs(scaled_change_sum.y))
  );
  EXPECT_NEAR(
      actual.z,
      face.weight_change.z,
      tolerance_per_term * (std::abs(raw_change.z) + std::abs(scaled_change_sum.z))
  );
}

// Finds the position in the list of the face whose centre is exactly `center`, or -1.
int FindPositionOfCenter(const GridTables& tables, const NearbyFaceList& list, Vec3 center) {
  for (int position_in_list = 0; position_in_list < list.count; ++position_in_list) {
    const Vec3 face_center = ComputeFacePosition(tables, list.faces[position_in_list]);
    if (face_center.x == center.x && face_center.y == center.y && face_center.z == center.z) {
      return position_in_list;
    }
  }
  return -1;
}

// Checks one face's weight against the paper's numbers. The raw weight and its change are not public, so they are
// rebuilt from what is: w_f = weight W, and g_f = W weight_change + weight G.
void CheckIsPaperFaceWeight(const FaceWeight& face_weight, const PaperWeights& paper, const PaperFaceWeight& face) {
  CheckIsNearRelative(face_weight.weight, face.weight);
  CheckWeightChangeIsNear(face_weight.weight_change, paper, face);
  CheckVectorIsNearRelative(face_weight.offset_from_mean, face.offset_from_mean);
  // The offset is exact: every centre and the position are short binary fractions.
  const Vec3 offset = face.center - paper.position;
  CheckVectorIsNear(face_weight.offset, offset, 0.0);
  CheckIsNearRelative(face_weight.weight * paper.weight_sum, face.raw_weight);
  const Vec3 rebuilt_change =
      (face_weight.weight_change * paper.weight_sum) + (paper.raw_weight_change_sum * face_weight.weight);
  CheckVectorIsNearRelative(rebuilt_change, face.raw_weight_change);
}

// Checks the list, the sums and the mean offset of one position against the paper's numbers.
void CheckHasPaperSums(const NearbyFaceWeights& weights, const PaperWeights& paper) {
  EXPECT_EQ(weights.list.count, paper.face_count);
  EXPECT_EQ(weights.list.cell_width, paper.cell_width);
  EXPECT_EQ(weights.fit, NearbyFaceFit::kFull);
  // W and G are sums of short binary fractions, exact in a double whatever the order.
  EXPECT_EQ(weights.weight_sum, paper.weight_sum);
  CheckVectorIsNear(weights.raw_weight_change_sum, paper.raw_weight_change_sum, 0.0);
  CheckVectorIsNearRelative(weights.mean_offset, paper.mean_offset);
}

// Checks the three named faces through the list form: ComputeFaceWeight by their position in the list.
void CheckListGivesPaperWeights(const GridTables& tables, const PaperWeights& paper) {
  const NearbyFaceWeights weights = FindWeightsOfAllFaces(tables, paper.position, paper.axis);
  for (const PaperFaceWeight& face : paper.faces) {
    SCOPED_TRACE(face.name);
    const int position_in_list = FindPositionOfCenter(tables, weights.list, face.center);
    ASSERT_GE(position_in_list, 0);
    CheckIsPaperFaceWeight(ComputeFaceWeight(tables, weights, position_in_list), paper, face);
  }
}

// Checks the three named faces through the face form: ComputeFaceWeight from the face, with the particle's sums.
void CheckFaceGivesPaperWeights(const GridTables& tables, const PaperWeights& paper) {
  const NearbyFaceWeights weights = FindWeightsOfAllFaces(tables, paper.position, paper.axis);
  const NearbyFaceSums sums = ToNearbyFaceSums(weights);
  for (const PaperFaceWeight& face : paper.faces) {
    SCOPED_TRACE(face.name);
    const int position_in_list = FindPositionOfCenter(tables, weights.list, face.center);
    ASSERT_GE(position_in_list, 0);
    const FacePlaceIndex place = weights.list.faces[position_in_list];
    const FaceWeight face_weight = ComputeFaceWeight(tables, place, paper.position, weights.list.cell, sums);
    EXPECT_EQ(face_weight.face.value, place.value);
    CheckIsPaperFaceWeight(face_weight, paper, face);
  }
}

TEST(ComputeNearbyFaceWeights, BigCellBesideSmallCells_HasThePapersSums) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTwoSizeFixtureGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  CheckHasPaperSums(FindWeightsOfAllFaces(tables, kInBigCell.position, kInBigCell.axis), kInBigCell);
}

TEST(ComputeNearbyFaceWeights, SmallCellBesideBigCell_HasThePapersSums) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTwoSizeFixtureGrid();
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  CheckHasPaperSums(FindWeightsOfAllFaces(tables, kInSmallCell.position, kInSmallCell.axis), kInSmallCell);
}

TEST(ComputeFaceWeight, ListOfBigCellBesideSmallCells_GivesThePapersWeights) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTwoSizeFixtureGrid();
  ASSERT_TRUE(test_grid.has_value());
  CheckListGivesPaperWeights(ToTablesOf(test_grid), kInBigCell);
}

TEST(ComputeFaceWeight, ListOfSmallCellBesideBigCell_GivesThePapersWeights) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTwoSizeFixtureGrid();
  ASSERT_TRUE(test_grid.has_value());
  CheckListGivesPaperWeights(ToTablesOf(test_grid), kInSmallCell);
}

TEST(ComputeFaceWeight, FaceBesideBigCellPosition_GivesThePapersWeights) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTwoSizeFixtureGrid();
  ASSERT_TRUE(test_grid.has_value());
  CheckFaceGivesPaperWeights(ToTablesOf(test_grid), kInBigCell);
}

TEST(ComputeFaceWeight, FaceBesideSmallCellPosition_GivesThePapersWeights) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTwoSizeFixtureGrid();
  ASSERT_TRUE(test_grid.has_value());
  CheckFaceGivesPaperWeights(ToTablesOf(test_grid), kInSmallCell);
}

}  // namespace
}  // namespace fluidity::grid
