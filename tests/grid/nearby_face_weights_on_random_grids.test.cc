#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <random>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/run_on.h"
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

// Builds a mask that keeps each face of a list with the chance 2 in 4, 3 in 4 or 4 in 4, the chance drawn per mask.
NearbyFaceMask BuildRandomMask(std::mt19937_64& generator, const NearbyFaceList& list) {
  const std::uint64_t kept_quarters = 2 + ToUniformIndex(generator(), 3);
  NearbyFaceMask mask{};
  for (int position_in_list = 0; position_in_list < list.count; ++position_in_list) {
    if (ToUniformIndex(generator(), 4) < kept_quarters) {
      mask.bits[position_in_list / 64] |= std::uint64_t{1} << static_cast<unsigned>(position_in_list % 64);
    }
  }
  return mask;
}

// What one random position gives on one axis: the weights of a random mask, or no weight at all.
struct RandomWeights {
  NearbyFaceWeights weights;
  bool has_weight;  // False when no kept face has weight.
};

// Finds the list of one axis at a position and the weights of a random mask.
RandomWeights
FindRandomWeights(const GridTables& tables, const PlacedPosition& placed, Axis axis, std::mt19937_64& generator) {
  GridError error = GridError::kInternalFault;
  const NearbyFaceList list = FindNearbyFaces(tables, placed.position, axis, CellBlockIndex{kNoIndex}, &error);
  EXPECT_EQ(error, GridError::kNone);
  const NearbyFaceMask mask = BuildRandomMask(generator, list);
  const NearbyFaceWeights weights = ComputeNearbyFaceWeights(tables, list, placed.position, mask);
  return {.weights = weights, .has_weight = weights.fit != NearbyFaceFit::kNone};
}

// Calls `check` for each of the 3 axes at each of `positions_per_grid` positions of each of the 200 random grids, with
// a random mask, when some kept face has weight. Gives how many positions were checked.
template <typename Check>
int ApplyToRandomWeights(int positions_per_grid, Check check) {
  const RandomGridsWithCheckers& random_grids = FindRandomGridsWithCheckers();
  const std::vector<std::vector<CellId>>& cell_lists = BuildShuffledRandomGrids();
  EXPECT_EQ(random_grids.grids.size(), kRandomGridCount);
  int checked_count = 0;
  for (std::size_t grid_entry = 0; grid_entry < random_grids.grids.size(); ++grid_entry) {
    const GridTables tables = random_grids.grids[grid_entry].grid.tables(RunOn::kCpu);
    std::mt19937_64 generator(grid_entry + 7001);
    for (int draw = 0; draw < positions_per_grid; ++draw) {
      const PlacedPosition placed =
          FindDrawnPosition(generator, tables.frame, cell_lists[grid_entry], DrawnPlace::kAnywhere);
      for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
        const RandomWeights random_weights = FindRandomWeights(tables, placed, axis, generator);
        if (random_weights.has_weight) {
          SCOPED_TRACE(grid_entry * 1000 + static_cast<std::size_t>(draw));
          check(tables, random_weights.weights);
          ++checked_count;
        }
      }
    }
  }
  return checked_count;
}

// The 200 random grids, 50 positions, random masks with at least one kept face of weight: the kept weights are never
// negative, add up to 1 within 1e-14, and their changes add up to 0 within 1e-13 / h.
TEST(ComputeNearbyFaceWeights, AnyKeptSet_WeightsSumToOne) {
  const int checked_count = ApplyToRandomWeights(50, [](const GridTables& tables, const NearbyFaceWeights& weights) {
    CheckWeightsAddUp(tables, weights);
  });
  RecordProperty("checked_lists", checked_count);
  EXPECT_GT(checked_count, 20000);
}

// Checks that a full fit reproduces the paper's linear field a + b . (x_f - x_p): the value at the particle, and the
// matrix row fitted from the values.
//
// A fit with a nearly flat spread (a condition number of 1e6, say) is still kFull, and rounding in the moment is
// magnified by that number in the inverse: the matrix row is off by about eps cond |value| / h, and the value at the
// particle by eps cond |value|, as the documentation of ComputeNearbyFaceWeights says. The condition number is at most
// trace(S) trace(S^-1), and the extra tolerance is 16 rounding steps of it, so a wrong fit, which is wrong by O(1),
// still fails. The field is centred on the position, so its values stay at its base value a plus b h and do not grow
// with the distance of the grid from the origin.
void CheckLinearFieldIsReproduced(const GridTables& tables, const NearbyFaceWeights& weights) {
  constexpr LinearField kField = {.base_value = 0.7, .slope = {.x = 0.3, .y = -1.1, .z = 2.0}};
  if (weights.fit != NearbyFaceFit::kFull) {
    return;
  }
  double weighted_value_sum = 0.0;
  double largest_value = 0.0;
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (!ComputeIsKept(weights.kept, position_in_list)) {
      continue;
    }
    const FaceWeight face_weight = ComputeFaceWeight(tables, weights, position_in_list);
    const double value = ComputeLinearValue(kField, face_weight.offset);
    largest_value = std::max(largest_value, std::abs(value));
    weighted_value_sum += face_weight.weight * value;
  }
  const Vec3 fitted_matrix_row = ComputeFittedMatrixRow(tables, weights, kField, ValueCentring::kUncentred);
  const double value_at_particle = weighted_value_sum - math::Dot(fitted_matrix_row, weights.mean_offset);
  const double slope_size = math::Length(kField.slope);
  const double spread_trace = weights.weighted_spread.xx + weights.weighted_spread.yy + weights.weighted_spread.zz;
  const double inverse_trace =
      weights.inverse_weighted_spread.xx + weights.inverse_weighted_spread.yy + weights.inverse_weighted_spread.zz;
  const double conditioning_error = 16.0 * std::numeric_limits<double>::epsilon() * spread_trace * inverse_trace;
  const double cell_width = weights.list.cell_width;
  const double value_tolerance =
      (1e-13 * (kField.base_value + (slope_size * cell_width))) + (conditioning_error * largest_value);
  EXPECT_NEAR(value_at_particle, kField.base_value, value_tolerance);
  const double matrix_row_tolerance = (1e-11 * slope_size) + (conditioning_error * largest_value / cell_width);
  EXPECT_LE(math::Length(fitted_matrix_row - kField.slope), matrix_row_tolerance);
}

// The value a + b . (x_f - x_p) on every face, the 200 random grids with random masks, only where the fit is kFull: the
// value at the particle and the fitted matrix row are those of the field.
TEST(ComputeNearbyFaceWeights, LinearField_IsReproduced) {
  const int checked_count = ApplyToRandomWeights(20, CheckLinearFieldIsReproduced);
  RecordProperty("checked_lists", checked_count);
  EXPECT_GT(checked_count, 5000);
}

// Checks that a face asking the particle's sums for its own weight gets the same bytes as the particle's list gives,
// for every kept face: weight, offset, offset from the mean and weight change.
void CheckFaceFormMatchesListForm(const GridTables& tables, const NearbyFaceWeights& weights) {
  const NearbyFaceSums sums = ToNearbyFaceSums(weights);
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (!ComputeIsKept(weights.kept, position_in_list)) {
      continue;
    }
    const FaceWeight from_list = ComputeFaceWeight(tables, weights, position_in_list);
    const FaceWeight from_face = ComputeFaceWeight(tables, from_list.face, weights.position, weights.list.cell, sums);
    CheckFaceWeightHasSameBits(from_face, from_list);
  }
}

// The 200 random grids, 50 positions, random masks with a kept face of weight: the face form with the particle's sums
// returns the same bytes as the list form for every kept face.
TEST(ComputeFaceWeight, SeenFromFace_IsBitIdentical) {
  const int checked_count = ApplyToRandomWeights(50, CheckFaceFormMatchesListForm);
  RecordProperty("checked_lists", checked_count);
  EXPECT_GT(checked_count, 20000);
}

// Says whether the list holds the face.
bool ComputeListHolds(const NearbyFaceList& list, FacePlaceIndex face) {
  for (int position_in_list = 0; position_in_list < list.count; ++position_in_list) {
    if (list.faces[position_in_list].value == face.value) {
      return true;
    }
  }
  return false;
}

// How many lists of each fit the face-form check has seen.
struct FitCounts {
  int none = 0;
  int mean_only = 0;
  int full = 0;
};

// Adds one list to the count of its fit.
void ApplyFitCount(NearbyFaceFit fit, FitCounts& counts) {
  if (fit == NearbyFaceFit::kNone) {
    ++counts.none;
  } else if (fit == NearbyFaceFit::kMeanOnly) {
    ++counts.mean_only;
  } else {
    ++counts.full;
  }
}

// Checks, for one face place and the particle's sums, that the face form gives nothing to a face the list does not
// hold, and for a listed face its own offset and, when the particle kept it and has weight, the list form's bits.
void CheckFaceFormOfPlace(const GridTables& tables, const NearbyFaceWeights& weights, FacePlaceIndex face) {
  const NearbyFaceSums sums = ToNearbyFaceSums(weights);
  const FaceWeight from_face = ComputeFaceWeight(tables, face, weights.position, weights.list.cell, sums);
  const Vec3 zero_vector = {.x = 0.0, .y = 0.0, .z = 0.0};
  if (!ComputeListHolds(weights.list, face)) {
    EXPECT_EQ(from_face.weight, 0.0);
    CheckVectorIsNear(from_face.offset, zero_vector, 0.0);
    CheckVectorIsNear(from_face.offset_from_mean, zero_vector, 0.0);
    CheckVectorIsNear(from_face.weight_change, zero_vector, 0.0);
    return;
  }
  EXPECT_GE(from_face.weight, 0.0);
  CheckVectorIsNear(from_face.offset, ComputeFacePosition(tables, face) - weights.position, 0.0);
  if (weights.fit == NearbyFaceFit::kNone) {
    EXPECT_EQ(from_face.weight, 0.0);
    return;
  }
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (weights.list.faces[position_in_list].value == face.value && ComputeIsKept(weights.kept, position_in_list)) {
      CheckFaceWeightHasSameBits(from_face, ComputeFaceWeight(tables, weights, position_in_list));
    }
  }
}

// Calls CheckFaceFormOfPlace for every place of one axis whose centre lies within 3 widths of the position on each
// axis, in every block of the grid: every face piece near the cell and every place that holds none.
void CheckFaceFormOfPlacesNear(const GridTables& tables, const NearbyFaceWeights& weights) {
  const double search_distance = 3.0 * weights.list.cell_width;
  const std::uint32_t place_count = tables.block_count * static_cast<std::uint32_t>(kFacePlacesPerBlock);
  for (std::uint32_t place = 0; place < place_count; ++place) {
    const FacePlaceIndex face = {.value = place};
    if (ToAxis(face) != weights.list.axis) {
      continue;
    }
    const Vec3 offset = ComputeFacePosition(tables, face) - weights.position;
    if (std::abs(offset.x) <= search_distance && std::abs(offset.y) <= search_distance
        && std::abs(offset.z) <= search_distance) {
      CheckFaceFormOfPlace(tables, weights, face);
    }
  }
}

// The random numbers and the fit counts of a run of the face-form check.
struct FaceFormRun {
  // clang-tidy: the seed is set again for each grid; this one only gives the member a state.
  std::mt19937_64 generator{1};  // NOLINT(bugprone-random-generator-seed)
  FitCounts counts;
};

// The masks tried at each position and axis: every face, a random mask, one face (a mean-only fit or none), no face.
void CheckFaceFormOfMasks(const GridTables& tables, Vec3 position, Axis axis, FaceFormRun& run) {
  GridError error = GridError::kInternalFault;
  const NearbyFaceList list = FindNearbyFaces(tables, position, axis, CellBlockIndex{kNoIndex}, &error);
  ASSERT_EQ(error, GridError::kNone);
  NearbyFaceMask one_face{};
  one_face.bits[0] =
      std::uint64_t{1} << ToUniformIndex(run.generator(), static_cast<std::uint64_t>(std::min(list.count, 64)));
  const std::array<NearbyFaceMask, 4> masks =
      {ToMaskOfAll(list), BuildRandomMask(run.generator, list), one_face, NearbyFaceMask{}};
  for (const NearbyFaceMask& mask : masks) {
    const NearbyFaceWeights weights = ComputeNearbyFaceWeights(tables, list, position, mask);
    ApplyFitCount(weights.fit, run.counts);
    CheckFaceFormOfPlacesNear(tables, weights);
  }
}

// The 200 random grids, 3 positions each, and the cell with every neighbour split at each of its 64 window quarters,
// each axis, four masks: the face form gives a face weight only when the list holds it, and then the list's own bits.
// Every fit comes up.
TEST(ComputeFaceWeight, FacesNearCell_AreWeightedOnlyWhenListed) {
  FaceFormRun run;
  const RandomGridsWithCheckers& random_grids = FindRandomGridsWithCheckers();
  const std::vector<std::vector<CellId>>& cell_lists = BuildShuffledRandomGrids();
  for (std::size_t grid_entry = 0; grid_entry < random_grids.grids.size(); ++grid_entry) {
    const GridTables tables = random_grids.grids[grid_entry].grid.tables(RunOn::kCpu);
    run.generator.seed(grid_entry + 9001);
    for (int draw = 0; draw < 3; ++draw) {
      const PlacedPosition placed =
          FindDrawnPosition(run.generator, tables.frame, cell_lists[grid_entry], DrawnPlace::kAnywhere);
      SCOPED_TRACE(grid_entry * 1000 + static_cast<std::size_t>(draw));
      for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
        CheckFaceFormOfMasks(tables, placed.position, axis, run);
      }
    }
  }
  const std::vector<CellId> split_cells = BuildCellWithEveryNeighbourSplit(1, 1, 1);
  const std::expected<TestGrid, GridFailure> split_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), split_cells));
  ASSERT_TRUE(split_grid.has_value());
  const GridTables split_tables = ToTablesOf(split_grid);
  run.generator.seed(4242);
  for (int window = 0; window < 64; ++window) {
    const PlacedPosition placed = ToWindowMiddlePosition(split_tables.frame, ToCell(0, 1, 1, 1), window);
    SCOPED_TRACE(window);
    for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
      CheckFaceFormOfMasks(split_tables, placed.position, axis, run);
    }
  }
  RecordProperty("lists_with_no_weight", run.counts.none);
  RecordProperty("lists_with_mean_only", run.counts.mean_only);
  RecordProperty("lists_with_full_fit", run.counts.full);
  EXPECT_GT(run.counts.none, 0);
  EXPECT_GT(run.counts.mean_only, 0);
  EXPECT_GT(run.counts.full, 0);
}

// Says whether a coordinate along one axis lies at least 0.01 widths from the two sides of its cell.
bool ComputeIsAwayFromCellSides(double position_on_axis, double origin_on_axis, double cell_width) {
  const double cells_from_origin = (position_on_axis - origin_on_axis) / cell_width;
  const double fraction_in_cell = cells_from_origin - std::floor(cells_from_origin);
  return fraction_in_cell >= 0.01 && fraction_in_cell <= 0.99;
}

// Says whether a distance in cell widths lies at least 0.01 from the two bends of the weight curve, 0.5 and 1.5.
bool ComputeIsAwayFromBends(double distance_in_cell_widths) {
  const double absolute_distance = std::abs(distance_in_cell_widths);
  return std::abs(absolute_distance - 0.5) >= 0.01 && std::abs(absolute_distance - 1.5) >= 0.01;
}

// Says whether the position lies at least 0.01 widths from every side of its cell and from the bends of the curve of
// every listed face, on each axis.
bool ComputeIsAwayFromBendsOfEveryFace(const GridTables& tables, const NearbyFaceWeights& weights) {
  const Vec3 position = weights.position;
  const double cell_width = weights.list.cell_width;
  const GridFrame frame = tables.frame;
  if (!ComputeIsAwayFromCellSides(position.x, frame.origin.x, cell_width)
      || !ComputeIsAwayFromCellSides(position.y, frame.origin.y, cell_width)
      || !ComputeIsAwayFromCellSides(position.z, frame.origin.z, cell_width)) {
    return false;
  }
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    const Vec3 offset = ComputeFacePosition(tables, weights.list.faces[position_in_list]) - position;
    if (!ComputeIsAwayFromBends(offset.x / cell_width) || !ComputeIsAwayFromBends(offset.y / cell_width)
        || !ComputeIsAwayFromBends(offset.z / cell_width)) {
      return false;
    }
  }
  return true;
}

// The weight of a face for the same list and mask at a moved position, or 0 when no kept face has weight there.
double
ComputeWeightAt(const GridTables& tables, const NearbyFaceWeights& weights, Vec3 moved_position, int position_in_list) {
  const NearbyFaceWeights moved = ComputeNearbyFaceWeights(tables, weights.list, moved_position, weights.kept);
  return moved.fit == NearbyFaceFit::kNone ? 0.0 : ComputeFaceWeight(tables, moved, position_in_list).weight;
}

// A vector of length `step` along an axis.
Vec3 ToStepAlong(Axis axis, double step) {
  if (axis == Axis::kX) {
    return {.x = step, .y = 0.0, .z = 0.0};
  }
  return axis == Axis::kY ? Vec3{.x = 0.0, .y = step, .z = 0.0} : Vec3{.x = 0.0, .y = 0.0, .z = step};
}

// The centred difference of the weight of one face along one axis, with a step of 1e-6 widths.
double ComputeWeightSlope(const GridTables& tables, const NearbyFaceWeights& weights, Axis axis, int position_in_list) {
  const Vec3 step = ToStepAlong(axis, 1e-6 * weights.list.cell_width);
  const Vec3 forward = weights.position + step;
  const Vec3 backward = weights.position - step;
  const double actual_distance = ComputeValueAlong(forward - backward, axis);
  const double weight_forward = ComputeWeightAt(tables, weights, forward, position_in_list);
  const double weight_backward = ComputeWeightAt(tables, weights, backward, position_in_list);
  return (weight_forward - weight_backward) / actual_distance;
}

// Checks the weight change of every kept face against the centred difference, within 1e-7 / h, at a position away from
// every bend.
void CheckWeightChangeMatchesDifference(const GridTables& tables, const NearbyFaceWeights& weights) {
  if (!ComputeIsAwayFromBendsOfEveryFace(tables, weights)) {
    return;
  }
  const double tolerance = 1e-7 / weights.list.cell_width;
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (!ComputeIsKept(weights.kept, position_in_list)) {
      continue;
    }
    const Vec3 weight_change = ComputeFaceWeight(tables, weights, position_in_list).weight_change;
    EXPECT_NEAR(weight_change.x, ComputeWeightSlope(tables, weights, Axis::kX, position_in_list), tolerance);
    EXPECT_NEAR(weight_change.y, ComputeWeightSlope(tables, weights, Axis::kY, position_in_list), tolerance);
    EXPECT_NEAR(weight_change.z, ComputeWeightSlope(tables, weights, Axis::kZ, position_in_list), tolerance);
  }
}

// The weight change of each kept face is the derivative of its weight: a centred difference with a step of 1e-6 widths,
// for the same list and mask, at positions 0.01 widths or more from every cell side and every bend of a curve.
TEST(ComputeNearbyFaceWeights, WeightChange_MatchesFiniteDifference) {
  const int checked_count = ApplyToRandomWeights(10, CheckWeightChangeMatchesDifference);
  RecordProperty("checked_lists", checked_count);
  EXPECT_GT(checked_count, 1000);
}

}  // namespace
}  // namespace fluidity::grid
