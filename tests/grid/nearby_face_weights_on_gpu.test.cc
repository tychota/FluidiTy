// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Asks FindNearbyFaces, ComputeNearbyFaceWeights and both forms of ComputeFaceWeight on the GPU for 10,000 positions of
// random grids, on all three axes with random masks, and compares each result with the CPU's. Places, counts, masks
// and the fit are equal; the weights and sums are within the tolerance of the method paper, with the CPU value the
// reference: 1e-15 + 1e-13 |value|, the absolute term scaled by the cell width h (by h^2, or by 1 / h, 1 / h^2) for a
// value whose unit has it (<sec-run-on>). Every GPU result goes to WriteResultDigest under a unique name.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/testing/grid_functions_gpu_cases.h"
#include "fluidity/grid/testing/grid_functions_result_words_runner_gpu.h"
#include "fluidity/grid/testing/grid_functions_runner_gpu.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/nearby_faces_cases.h"

namespace fluidity::grid {
namespace {

constexpr int kGridCount = 10;
constexpr int kPositionsPerGrid = 1000;
constexpr int kAxesPerPosition = 3;
constexpr double kAbsoluteTerm = 1e-15;
constexpr double kRelativeTerm = 1e-13;

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

// Expects the GPU's value within the tolerance of the CPU's, whose absolute term is kAbsoluteTerm times `unit_scale`.
// clang-tidy: the CPU's value and the GPU's are both doubles; the swappable-parameters check flags them.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void CheckNearCpuValue(double cpu_value, double gpu_value, double unit_scale) {
  const Tolerance tolerance = {.absolute_floor = kAbsoluteTerm * unit_scale, .relative_term = kRelativeTerm};
  EXPECT_TRUE(CheckWithinTolerance(cpu_value, gpu_value, tolerance));
}

// Expects the three numbers of a vector within the tolerance of the CPU's, as CheckNearCpuValue.
void CheckNearCpuVector(Vec3 cpu_vector, Vec3 gpu_vector, double unit_scale) {
  CheckNearCpuValue(cpu_vector.x, gpu_vector.x, unit_scale);
  CheckNearCpuValue(cpu_vector.y, gpu_vector.y, unit_scale);
  CheckNearCpuValue(cpu_vector.z, gpu_vector.z, unit_scale);
}

// Expects the nine numbers of a matrix within the tolerance of the CPU's, as CheckNearCpuValue.
void CheckNearCpuMatrix(const Mat3& cpu_matrix, const Mat3& gpu_matrix, double unit_scale) {
  CheckNearCpuValue(cpu_matrix.xx, gpu_matrix.xx, unit_scale);
  CheckNearCpuValue(cpu_matrix.xy, gpu_matrix.xy, unit_scale);
  CheckNearCpuValue(cpu_matrix.xz, gpu_matrix.xz, unit_scale);
  CheckNearCpuValue(cpu_matrix.yx, gpu_matrix.yx, unit_scale);
  CheckNearCpuValue(cpu_matrix.yy, gpu_matrix.yy, unit_scale);
  CheckNearCpuValue(cpu_matrix.yz, gpu_matrix.yz, unit_scale);
  CheckNearCpuValue(cpu_matrix.zx, gpu_matrix.zx, unit_scale);
  CheckNearCpuValue(cpu_matrix.zy, gpu_matrix.zy, unit_scale);
  CheckNearCpuValue(cpu_matrix.zz, gpu_matrix.zz, unit_scale);
}

// The words of what the weights hold that is not a sum: the list with its cell, width and axis, the mask, the fit,
// the completeness and the position. Two weights agree there when these words are equal.
std::vector<std::uint64_t> ToDiscreteWords(const NearbyFaceWeights& weights) {
  std::vector<std::uint64_t> words;
  words.push_back(weights.list.cell.value);
  words.push_back(ToBitsOf(weights.list.cell_width));
  words.push_back(static_cast<std::uint64_t>(weights.list.axis));
  words.push_back(static_cast<std::uint64_t>(weights.list.count));
  for (const FacePlaceIndex face : weights.list.faces) {
    words.push_back(face.value);
  }
  for (const std::uint64_t mask_word : weights.kept.bits) {
    words.push_back(mask_word);
  }
  words.push_back(static_cast<std::uint64_t>(weights.fit));
  words.push_back(weights.is_complete ? 1 : 0);
  AppendWords(weights.position, words);
  return words;
}

// Expects one result of the GPU to match the CPU's: the discrete parts equal, the sums and matrices within tolerance.
// The absolute term of a value in m is scaled by h, of one in m2 by h^2, of one in 1/m by 1/h and of one in 1/m2 by
// 1/h^2.
void CheckSameNearbyFaceResult(const NearbyFaceResult& cpu_result, const NearbyFaceResult& gpu_result) {
  ASSERT_EQ(gpu_result.error, cpu_result.error);
  const NearbyFaceWeights& cpu_weights = cpu_result.weights;
  const NearbyFaceWeights& gpu_weights = gpu_result.weights;
  EXPECT_EQ(ToDiscreteWords(gpu_weights), ToDiscreteWords(cpu_weights));
  const double cell_width = cpu_weights.list.cell_width;
  if (cell_width == 0.0) {
    return;
  }
  CheckNearCpuValue(cpu_weights.weight_sum, gpu_weights.weight_sum, 1.0);
  CheckNearCpuVector(cpu_weights.mean_offset, gpu_weights.mean_offset, cell_width);
  CheckNearCpuVector(cpu_weights.raw_weight_change_sum, gpu_weights.raw_weight_change_sum, 1.0 / cell_width);
  CheckNearCpuMatrix(cpu_weights.weighted_spread, gpu_weights.weighted_spread, cell_width * cell_width);
  CheckNearCpuMatrix(
      cpu_weights.inverse_weighted_spread,
      gpu_weights.inverse_weighted_spread,
      1.0 / (cell_width * cell_width)
  );
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

// 10 random grids, 1,000 positions each, all three axes, random masks: lists, masks, fits and completeness are the
// CPU's, the weights and sums are within the tolerance.
TEST(ComputeNearbyFaceWeights, OnGpu_MatchesCpu) {
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
    ASSERT_EQ(status, RunOnCallStatus::kDone);
    for (std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
      SCOPED_TRACE(query_index);
      CheckSameNearbyFaceResult(on_cpu[query_index], on_gpu[query_index]);
    }
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
    );
    ASSERT_EQ(
        ComputeNearbyFaceResultsOnGpu(gpu_tables, queries.data(), query_count, second_run.data()),
        RunOnCallStatus::kDone
    );
    EXPECT_EQ(
        ToResultWords(std::span<const NearbyFaceResult>(first_run)),
        ToResultWords(std::span<const NearbyFaceResult>(second_run))
    );
  }
}

// The queries of ComputeFaceWeight for the results of a grid: for each result with a kept face, the kept face at
// (result index) mod (kept count) among its kept faces, and as the other face the first face of the next result's list.
// A result with no kept face, or whose next list is empty, gives no query.
std::vector<FaceWeightQuery> BuildFaceWeightQueries(std::span<const NearbyFaceResult> results) {
  std::vector<FaceWeightQuery> queries;
  for (std::size_t result_index = 0; result_index < results.size(); ++result_index) {
    const NearbyFaceWeights& weights = results[result_index].weights;
    if (results[result_index].error != GridError::kNone) {
      continue;
    }
    std::vector<int> kept_positions;
    for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
      if (ComputeIsKept(weights.kept, position_in_list)) {
        kept_positions.push_back(position_in_list);
      }
    }
    const NearbyFaceWeights& other_weights = results[(result_index + 1) % results.size()].weights;
    if (kept_positions.empty() || other_weights.list.count == 0) {
      continue;
    }
    queries.push_back({
        .weights = weights,
        .other_face = other_weights.list.faces[0],
        .position_in_list = kept_positions[result_index % kept_positions.size()],
    });
  }
  return queries;
}

// Expects the weight a face gets, as the CPU computed it, within the tolerance: the face place equal, the weight with
// the absolute term 1, the offsets with h and the weight change with 1 / h.
void CheckSameFaceWeight(const FaceWeight& cpu_weight, const FaceWeight& gpu_weight, double cell_width) {
  EXPECT_EQ(gpu_weight.face.value, cpu_weight.face.value);
  CheckNearCpuValue(cpu_weight.weight, gpu_weight.weight, 1.0);
  CheckNearCpuVector(cpu_weight.offset, gpu_weight.offset, cell_width);
  CheckNearCpuVector(cpu_weight.offset_from_mean, gpu_weight.offset_from_mean, cell_width);
  CheckNearCpuVector(cpu_weight.weight_change, gpu_weight.weight_change, 1.0 / cell_width);
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

// Expects the three weights of one result to match the CPU's within the tolerance. On the GPU itself, a face that asks
// for its own weight gets the words of the first form.
void CheckSameFaceWeightResult(
    const FaceWeightResult& cpu_result,
    const FaceWeightResult& gpu_result,
    double cell_width
) {
  CheckSameFaceWeight(cpu_result.from_list, gpu_result.from_list, cell_width);
  CheckSameFaceWeight(cpu_result.seen_from_face, gpu_result.seen_from_face, cell_width);
  CheckSameFaceWeight(cpu_result.seen_from_other_face, gpu_result.seen_from_other_face, cell_width);
  std::vector<std::uint64_t> from_list_words;
  std::vector<std::uint64_t> seen_from_face_words;
  AppendWords(gpu_result.from_list, from_list_words);
  AppendWords(gpu_result.seen_from_face, seen_from_face_words);
  EXPECT_EQ(from_list_words, seen_from_face_words);
}

// 10 random grids: both forms of ComputeFaceWeight on the GPU, for the weights the CPU computed, give the CPU's faces
// and weights.
TEST(ComputeFaceWeight, OnGpu_MatchesCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  std::size_t checked_count = 0;
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const GridOnBothProcessors& grid = grids.at(static_cast<std::size_t>(grid_entry));
    const GridTables cpu_tables = grid.on_cpu.grid.tables(RunOn::kCpu);
    const std::vector<NearbyFaceQuery> weight_queries = BuildNearbyFaceQueries(grid, grid_entry);
    const std::vector<NearbyFaceResult> weight_results = ComputeNearbyFaceResultsOnCpu(cpu_tables, weight_queries);
    const std::vector<FaceWeightQuery> queries = BuildFaceWeightQueries(weight_results);
    ASSERT_FALSE(queries.empty());
    const std::vector<FaceWeightResult> on_cpu = ComputeFaceWeightResultsOnCpu(cpu_tables, queries);

    std::vector<FaceWeightResult> on_gpu(queries.size());
    const RunOnCallStatus status = ComputeFaceWeightResultsOnGpu(
        grid.on_gpu.grid.tables(RunOn::kGpu),
        queries.data(),
        static_cast<int>(queries.size()),
        on_gpu.data()
    );
    ASSERT_EQ(status, RunOnCallStatus::kDone);
    for (std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
      SCOPED_TRACE(query_index);
      CheckSameFaceWeightResult(on_cpu[query_index], on_gpu[query_index], queries[query_index].weights.list.cell_width);
    }
    checked_count += queries.size();
    WriteDigestOfResults(
        "ComputeFaceWeight on the GPU, both forms, grid " + std::to_string(grid_entry),
        std::span<const FaceWeightResult>(on_gpu)
    );
  }
  RecordProperty("checked_faces", static_cast<int>(checked_count));
  EXPECT_GT(checked_count, static_cast<std::size_t>(kGridCount * kPositionsPerGrid));
}

// The GPU function of the weights keeps a list and its weights per thread, so its frame holds at least the weights: the
// bytes the compiler gave it are recorded for the modules that call the weights.
TEST(ComputeNearbyFaceWeights, OnGpuRunner_ReportsItsStack) {
  std::uint64_t bytes_per_thread = 0;
  ASSERT_EQ(ComputeWeightsRunnerStackOnGpu(&bytes_per_thread), RunOnCallStatus::kDone);
  EXPECT_GE(bytes_per_thread, sizeof(NearbyFaceWeights));
  RecordProperty("weights_runner_stack_bytes", std::to_string(bytes_per_thread));
}

}  // namespace
}  // namespace fluidity::grid
