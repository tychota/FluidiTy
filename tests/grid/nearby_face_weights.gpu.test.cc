// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on a machine with a GPU.
//
// Asks FindNearbyFaces, ComputeNearbyFaceWeights and both forms of ComputeFaceWeight on the GPU for 10,000 positions of
// random grids, on all three axes with random masks, and expects the CPU's results with the same bits in every field:
// the formulas use only + - * /, sqrt and floor, the GPU code is built with no fused multiply-add, and the method paper
// asks for equal bits then (<sec-run-on>). The face form is asked about four faces: a face of the list (the same bits
// as the list form), a face on the same axis that may be outside the list, a face of the next axis with that axis's
// sums, and, on a grid whose neighbours are all split, faces that pass the size-level and window tests of the
// candidates and are still not in the list (weight 0). Every GPU result goes to WriteResultDigest under a unique name.

#include "fluidity/grid/nearby_face_weights.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/face_search_cases.h"
#include "fluidity/grid/testing/grid_functions_answers.h"
#include "fluidity/grid/testing/grid_functions_gpu_cases.h"
#include "fluidity/grid/testing/grid_functions_result_words.h"
#include "fluidity/grid/testing/grid_functions_runner_gpu.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/nearby_face_weights_cases.h"
#include "fluidity/grid/testing/nearby_faces_cases.h"

namespace fluidity::grid {
namespace {

constexpr int kGridCount = 10;
constexpr int kPositionsPerGrid = 1000;
constexpr std::size_t kAxisCount = 3;
constexpr int kAxesPerPosition = static_cast<int>(kAxisCount);

// Makes the mask a query may keep: all faces, or a mask that keeps each face with the chance 1 in 2, 3 in 4 or 1 in 4,
// the kind drawn per mask. The answer keeps only the faces of the list whose bit is set.
NearbyFaceMask BuildDrawnMask(std::mt19937_64& generator) {
  const std::uint64_t kind = ToUniformIndex(generator(), 4);
  NearbyFaceMask mask{};
  for (std::uint64_t& word : mask.bits) {
    const std::uint64_t first_draw = generator();
    const std::uint64_t second_draw = generator();
    if (kind == 0) {
      word = ~std::uint64_t{0};
    } else if (kind == 1) {
      word = first_draw;
    } else if (kind == 2) {
      word = first_draw | second_draw;
    } else {
      word = first_draw & second_draw;
    }
  }
  return mask;
}

// The 3,000 queries of one grid: 1,000 positions anywhere inside the cells of the list, from one std::mt19937_64 of
// seed grid_entry + 7001, each on the axes x, y and z with its own mask.
std::vector<NearbyFaceQuery> BuildNearbyFaceQueries(const GridOnBothProcessors& grid, int grid_entry) {
  const GridTables tables = grid.on_cpu.grid.tables(RunOn::kCpu);
  std::mt19937_64 generator(static_cast<std::uint64_t>(grid_entry) + 7001);  // NOLINT(bugprone-random-generator-seed)
  std::vector<NearbyFaceQuery> queries;
  queries.reserve(static_cast<std::size_t>(kPositionsPerGrid) * kAxesPerPosition);
  for (int draw = 0; draw < kPositionsPerGrid; ++draw) {
    const PlacedPosition placed = FindDrawnPosition(generator, tables.frame, grid.cells, DrawnPlace::kAnywhere);
    for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
      queries.push_back({.position = placed.position, .axis = axis, .drawn_mask = BuildDrawnMask(generator)});
    }
  }
  return queries;
}

// Answers the queries on the CPU, in order.
std::vector<NearbyFaceResult>
ComputeNearbyFaceResultsOnCpu(const GridTables& tables, std::span<const NearbyFaceQuery> queries) {
  std::vector<NearbyFaceResult> results;
  results.reserve(queries.size());
  for (const NearbyFaceQuery& query : queries) {
    results.push_back(ComputeNearbyFaceResult(tables, query));
  }
  return results;
}

// Counts of what the results cover, so a drop in the coverage fails the test.
struct WeightCoverage {
  int lists = 0;
  int full_fits = 0;
  int complete_lists = 0;
};

// Adds the results of one grid to the coverage.
void ApplyToCoverage(std::span<const NearbyFaceResult> results, WeightCoverage& coverage) {
  for (const NearbyFaceResult& result : results) {
    coverage.lists += result.error == GridError::kNone ? 1 : 0;
    coverage.full_fits += result.weights.fit == NearbyFaceFit::kFull ? 1 : 0;
    coverage.complete_lists += result.weights.is_complete ? 1 : 0;
  }
}

// 10 random grids, 1,000 positions each, all three axes, random masks: the weights of the GPU have the CPU's bits in
// every field, the lists, masks, fits and completeness included.
TEST(ComputeNearbyFaceWeights, OnGpu_SameBitsAsCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  WeightCoverage coverage;
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const GridOnBothProcessors& grid = grids.at(static_cast<std::size_t>(grid_entry));
    const std::vector<NearbyFaceQuery> queries = BuildNearbyFaceQueries(grid, grid_entry);
    const std::vector<NearbyFaceResult> on_cpu =
        ComputeNearbyFaceResultsOnCpu(grid.on_cpu.grid.tables(RunOn::kCpu), queries);
    std::vector<NearbyFaceResult> on_gpu(queries.size());
    const RunOnCallStatus status = ComputeNearbyFaceResultsOnGpu(
        grid.on_gpu.grid.tables(RunOn::kGpu),
        queries.data(),
        static_cast<int>(queries.size()),
        on_gpu.data()
    );
    ASSERT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
    CheckSameResults(
        std::span<const NearbyFaceResult>(on_cpu),
        std::span<const NearbyFaceResult>(on_gpu),
        "ComputeNearbyFaceWeights, grid " + std::to_string(grid_entry)
    );
    ApplyToCoverage(on_cpu, coverage);
    WriteDigestOfResults(
        "ComputeNearbyFaceWeights on the GPU, 1,000 positions on 3 axes, grid " + std::to_string(grid_entry),
        std::span<const NearbyFaceResult>(on_gpu)
    );
  }
  RecordProperty("checked_lists", coverage.lists);
  EXPECT_EQ(coverage.lists, kGridCount * kPositionsPerGrid * kAxesPerPosition);
  EXPECT_GT(coverage.full_fits, coverage.lists / 4);
  EXPECT_GT(coverage.complete_lists, 0);
}

// The same queries run twice on the GPU: the bytes are the same. The words of each result are compared, since a
// result struct holds padding bytes that no function writes.
TEST(ComputeNearbyFaceWeights, OnGpuTwice_GivesIdenticalBytes) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const GridOnBothProcessors& grid = grids.at(static_cast<std::size_t>(grid_entry));
    const std::vector<NearbyFaceQuery> queries = BuildNearbyFaceQueries(grid, grid_entry);
    std::vector<NearbyFaceResult> first_run(queries.size());
    std::vector<NearbyFaceResult> second_run(queries.size());
    const GridTables gpu_tables = grid.on_gpu.grid.tables(RunOn::kGpu);
    const auto query_count = static_cast<int>(queries.size());
    ASSERT_EQ(
        ComputeNearbyFaceResultsOnGpu(gpu_tables, queries.data(), query_count, first_run.data()),
        RunOnCallStatus::kDone
    ) << ReadLastRunnerFailure();
    ASSERT_EQ(
        ComputeNearbyFaceResultsOnGpu(gpu_tables, queries.data(), query_count, second_run.data()),
        RunOnCallStatus::kDone
    ) << ReadLastRunnerFailure();
    CheckSameResults(
        std::span<const NearbyFaceResult>(first_run),
        std::span<const NearbyFaceResult>(second_run),
        "ComputeNearbyFaceWeights, second run against the first, grid " + std::to_string(grid_entry)
    );
  }
}

// Says whether the face is a candidate of the particle by the two tests that come before the ownership test of
// ComputeIsCandidateFace: its size level is within one of the cell's, and its centre is in the window of the position
// on each axis. The same arithmetic as ComputeIsCandidateFace, on the pieces it is built from.
bool ComputeIsCandidateByLevelAndWindow(
    const GridTables& tables,
    FacePlaceIndex face,
    const NearbyFaceWeights& weights
) {
  const CellId cell_id = ToCellId(tables, weights.list.cell);
  const auto cell_size_level = static_cast<int>(ToSizeLevel(cell_id));
  const auto face_size_level = static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(face)));
  const int size_level_difference = face_size_level - cell_size_level;
  if (size_level_difference < -1 || size_level_difference > 1) {
    return false;
  }
  const CellCoord cell_coord = ToCellCoord(cell_id);
  const double cell_width = weights.list.cell_width;
  const Vec3 position = weights.position;
  const GridFrame frame = tables.frame;
  const int quarter_x = ComputeCandidateWindowQuarter(position.x, frame.origin.x, cell_width, cell_coord.x);
  const int quarter_y = ComputeCandidateWindowQuarter(position.y, frame.origin.y, cell_width, cell_coord.y);
  const int quarter_z = ComputeCandidateWindowQuarter(position.z, frame.origin.z, cell_width, cell_coord.z);
  const QuarterWidthPoint center = ComputeFaceCenterInQuarterWidths(tables, face, cell_size_level);
  return ComputeIsInCandidateWindow(center.x - (4 * std::int64_t{cell_coord.x}), quarter_x)
      && ComputeIsInCandidateWindow(center.y - (4 * std::int64_t{cell_coord.y}), quarter_y)
      && ComputeIsInCandidateWindow(center.z - (4 * std::int64_t{cell_coord.z}), quarter_z);
}

// Says whether the list of the weights holds the face.
bool ComputeIsListed(const NearbyFaceWeights& weights, FacePlaceIndex face) {
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (weights.list.faces[position_in_list].value == face.value) {
      return true;
    }
  }
  return false;
}

// The place of the first face piece on the axis of the weights that passes the size-level and window tests of the
// particle and is not in its list, or kNoIndex when there is none: a face of far small cells beside a split place.
FacePlaceIndex FindUnlistedCandidateFace(const GridTables& tables, const NearbyFaceWeights& weights) {
  for (const FacePlaceIndex face : FindFacePiecePlaces(tables)) {
    const bool is_on_axis = ToAxis(face) == weights.list.axis;
    if (is_on_axis && !ComputeIsListed(weights, face) && ComputeIsCandidateByLevelAndWindow(tables, face, weights)) {
      return face;
    }
  }
  return {.value = kNoIndex};
}

// The kept position in the list of the weights at (index) mod (kept count), or -1 when no face is kept.
int FindKeptPosition(const NearbyFaceWeights& weights, std::size_t index) {
  std::vector<int> kept_positions;
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (ComputeIsKept(weights.kept, position_in_list)) {
      kept_positions.push_back(position_in_list);
    }
  }
  return kept_positions.empty() ? -1 : kept_positions[index % kept_positions.size()];
}

// The queries of ComputeFaceWeight for the results of a grid, whose entries come three to a position on the axes x, y
// and z (BuildNearbyFaceQueries). For the result i with a kept face: that face at (i) mod (kept count) among the kept
// faces; as the unlisted face the first face of the list of the same axis at the next position; and as the face of
// another axis the first face of the list of the next axis at the same position, with that list's sums. A result
// whose lists have no face, or whose list failed, gives no query.
std::vector<FaceWeightQuery> BuildFaceWeightQueries(std::span<const NearbyFaceResult> results) {
  std::vector<FaceWeightQuery> queries;
  for (std::size_t result_index = 0; result_index < results.size(); ++result_index) {
    const NearbyFaceResult& result = results[result_index];
    const std::size_t axis_index = result_index % kAxisCount;
    const std::size_t next_axis_at_position = result_index - axis_index + ((axis_index + 1) % kAxisCount);
    const NearbyFaceResult& next_axis_result = results[next_axis_at_position];
    const NearbyFaceResult& next_position_result = results[(result_index + kAxisCount) % results.size()];
    const int kept_position = FindKeptPosition(result.weights, result_index);
    const bool has_faces = result.error == GridError::kNone && next_axis_result.error == GridError::kNone
        && next_position_result.error == GridError::kNone && kept_position >= 0
        && next_axis_result.weights.list.count > 0 && next_position_result.weights.list.count > 0;
    if (!has_faces) {
      continue;
    }
    queries.push_back({
        .weights = result.weights,
        .position_in_list = kept_position,
        .unlisted_face = next_position_result.weights.list.faces[0],
        .other_axis_face = next_axis_result.weights.list.faces[0],
        .other_axis_sums = ToNearbyFaceSums(next_axis_result.weights),
    });
  }
  return queries;
}

// Answers the queries of ComputeFaceWeight on the CPU, in order.
std::vector<FaceWeightResult>
ComputeFaceWeightResultsOnCpu(const GridTables& tables, std::span<const FaceWeightQuery> queries) {
  std::vector<FaceWeightResult> results;
  results.reserve(queries.size());
  for (const FaceWeightQuery& query : queries) {
    results.push_back(ComputeFaceWeightResult(tables, query));
  }
  return results;
}

// Expects the face form to give a listed face the bits of the list form, as the CPU does and as the GPU must: the
// weights of the two forms of every result are compared as two result lists.
void CheckFormsAgree(std::span<const FaceWeightResult> results, std::string_view processor) {
  std::vector<FaceWeight> from_list;
  std::vector<FaceWeight> seen_from_face;
  for (const FaceWeightResult& result : results) {
    from_list.push_back(result.from_list);
    seen_from_face.push_back(result.seen_from_face);
  }
  CheckSameResults(
      std::span<const FaceWeight>(from_list),
      std::span<const FaceWeight>(seen_from_face),
      "ComputeFaceWeight on the " + std::string(processor) + ", face form against list form"
  );
}

// Counts the results whose listed face has a weight above 0 and whose face of the other axis has one too: the cases
// whose bits mean something, not two zeros.
struct FaceWeightCoverage {
  int listed_with_weight = 0;
  int other_axis_with_weight = 0;
};

// Adds the results of one grid to the coverage.
void ApplyToCoverage(std::span<const FaceWeightResult> results, FaceWeightCoverage& coverage) {
  for (const FaceWeightResult& result : results) {
    coverage.listed_with_weight += result.from_list.weight > 0.0 ? 1 : 0;
    coverage.other_axis_with_weight += result.seen_from_other_axis_face.weight > 0.0 ? 1 : 0;
  }
}

// Asks both forms of ComputeFaceWeight on the GPU for the weights the CPU computed on one random grid, and expects the
// CPU's bits. Adds the results to the coverage and returns the number of queries.
std::size_t CheckFaceWeightsOfGrid(const GridOnBothProcessors& grid, int grid_entry, FaceWeightCoverage& coverage) {
  const GridTables cpu_tables = grid.on_cpu.grid.tables(RunOn::kCpu);
  const std::vector<NearbyFaceQuery> weight_queries = BuildNearbyFaceQueries(grid, grid_entry);
  const std::vector<NearbyFaceResult> weight_results = ComputeNearbyFaceResultsOnCpu(cpu_tables, weight_queries);
  const std::vector<FaceWeightQuery> queries = BuildFaceWeightQueries(weight_results);
  EXPECT_FALSE(queries.empty());
  const std::vector<FaceWeightResult> on_cpu = ComputeFaceWeightResultsOnCpu(cpu_tables, queries);

  std::vector<FaceWeightResult> on_gpu(queries.size());
  const RunOnCallStatus status = ComputeFaceWeightResultsOnGpu(
      grid.on_gpu.grid.tables(RunOn::kGpu),
      queries.data(),
      static_cast<int>(queries.size()),
      on_gpu.data()
  );
  EXPECT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
  CheckSameResults(
      std::span<const FaceWeightResult>(on_cpu),
      std::span<const FaceWeightResult>(on_gpu),
      "ComputeFaceWeight, grid " + std::to_string(grid_entry)
  );
  CheckFormsAgree(on_gpu, "GPU");
  ApplyToCoverage(on_cpu, coverage);
  WriteDigestOfResults(
      "ComputeFaceWeight on the GPU, both forms, grid " + std::to_string(grid_entry),
      std::span<const FaceWeightResult>(on_gpu)
  );
  return queries.size();
}

// 10 random grids: the list form and the face form of ComputeFaceWeight on the GPU, for the weights the CPU computed,
// have the CPU's bits for a listed face, the first face of the next position and the first face of the next axis.
TEST(ComputeFaceWeight, OnGpu_SameBitsAsCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  std::size_t checked_count = 0;
  FaceWeightCoverage coverage;
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    checked_count += CheckFaceWeightsOfGrid(grids.at(static_cast<std::size_t>(grid_entry)), grid_entry, coverage);
  }
  RecordProperty("checked_faces", static_cast<int>(checked_count));
  EXPECT_GT(checked_count, static_cast<std::size_t>(kGridCount * kPositionsPerGrid));
  EXPECT_GT(coverage.listed_with_weight, static_cast<int>(checked_count / 2));
  EXPECT_GT(coverage.other_axis_with_weight, 0);
}

// The queries of ComputeFaceWeight on a grid whose cell {1,1,1} has every neighbour split, at the middle of each of the
// 64 windows of the cell, on each axis, with every face of the list kept. The unlisted face is a face that passes the
// size-level and window tests and is not in the list; a window with none gives no query. The face of another axis is
// the first face of the next axis's list.
std::vector<FaceWeightQuery> BuildSplitNeighbourQueries(const GridTables& tables) {
  std::vector<FaceWeightQuery> queries;
  const std::array<Axis, 3> axes = {Axis::kX, Axis::kY, Axis::kZ};
  for (int window = 0; window < 64; ++window) {
    const PlacedPosition placed = ToWindowMiddlePosition(tables.frame, ToCell(0, 1, 1, 1), window);
    for (std::size_t axis_index = 0; axis_index < axes.size(); ++axis_index) {
      const NearbyFaceWeights weights = FindWeightsOfAllFaces(tables, placed.position, axes[axis_index]);
      const NearbyFaceWeights next_axis_weights =
          FindWeightsOfAllFaces(tables, placed.position, axes[(axis_index + 1) % axes.size()]);
      const FacePlaceIndex unlisted_face = FindUnlistedCandidateFace(tables, weights);
      if (unlisted_face.value == kNoIndex || weights.list.count == 0 || next_axis_weights.list.count == 0) {
        continue;
      }
      queries.push_back({
          .weights = weights,
          .position_in_list = 0,
          .unlisted_face = unlisted_face,
          .other_axis_face = next_axis_weights.list.faces[0],
          .other_axis_sums = ToNearbyFaceSums(next_axis_weights),
      });
    }
  }
  return queries;
}

// Expects the weight the face form gives an unlisted face on the CPU to be what ComputeFaceWeight says for a face that
// no list holds: weight 0 and zero vectors, with the face echoed. A face that gets weight here would make the check of
// the GPU's zero meaningless.
void CheckUnlistedFaceHasNoWeight(const FaceWeight& weight, FacePlaceIndex face) {
  EXPECT_EQ(weight.face.value, face.value);
  EXPECT_EQ(weight.weight, 0.0);
  EXPECT_EQ(weight.weight_change.x, 0.0);
  EXPECT_EQ(weight.offset_from_mean.x, 0.0);
}

// Expects each unlisted face of the queries to have no weight on the CPU, one query after the other.
void CheckUnlistedFacesHaveNoWeight(
    std::span<const FaceWeightQuery> queries,
    std::span<const FaceWeightResult> on_cpu
) {
  ASSERT_EQ(queries.size(), on_cpu.size());
  for (std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
    SCOPED_TRACE(query_index);
    CheckUnlistedFaceHasNoWeight(on_cpu[query_index].seen_from_unlisted_face, queries[query_index].unlisted_face);
  }
}

// The cell {1,1,1} with all 26 places around it split (the fixture of the CPU tests of ComputeFaceWeight): faces of far
// small cells pass the level and window tests of the particle and are in no list. The GPU gives them weight 0 and the
// CPU's bits, and a listed face and a face of another axis still get theirs.
TEST(ComputeFaceWeight, OnGpuOnSplitNeighbours_SameBitsAsCpu) {
  const std::optional<GridOnBothProcessors> built =
      BuildGridOnBothProcessors(ToUnitFrame(2), BuildCellWithEveryNeighbourSplit(1, 1, 1));
  ASSERT_TRUE(built.has_value());
  const GridOnBothProcessors& grid = *built;
  const GridTables cpu_tables = grid.on_cpu.grid.tables(RunOn::kCpu);
  const std::vector<FaceWeightQuery> queries = BuildSplitNeighbourQueries(cpu_tables);
  ASSERT_GT(queries.size(), 0U);
  const std::vector<FaceWeightResult> on_cpu = ComputeFaceWeightResultsOnCpu(cpu_tables, queries);
  CheckUnlistedFacesHaveNoWeight(queries, on_cpu);

  std::vector<FaceWeightResult> on_gpu(queries.size());
  const RunOnCallStatus status = ComputeFaceWeightResultsOnGpu(
      grid.on_gpu.grid.tables(RunOn::kGpu),
      queries.data(),
      static_cast<int>(queries.size()),
      on_gpu.data()
  );
  ASSERT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
  CheckSameResults(
      std::span<const FaceWeightResult>(on_cpu),
      std::span<const FaceWeightResult>(on_gpu),
      "ComputeFaceWeight on split neighbours"
  );
  CheckFormsAgree(on_gpu, "GPU");
  FaceWeightCoverage coverage;
  ApplyToCoverage(on_cpu, coverage);
  RecordProperty("unlisted_candidate_faces", static_cast<int>(queries.size()));
  EXPECT_GT(coverage.listed_with_weight, 0);
  WriteDigestOfResults("ComputeFaceWeight on the GPU, split neighbours", std::span<const FaceWeightResult>(on_gpu));
}

}  // namespace
}  // namespace fluidity::grid
