// The result words of the results of the GPU runners: how two results are compared and digested, for the GPU tests of
// the shared grid functions, private to grid/.
//
// Purpose: turn each result of grid_functions_runner_gpu.h into a list of 64-bit words, so a test compares a result of
//   the GPU with the CPU's and digests it, written once.
//
// Idea: each integer of a result goes in as it is and each double as its bits, in the order of the struct, whatever
//   padding the struct holds. Two results are equal when their words are equal. The words of every GPU result go to
//   WriteResultDigest, so the repeat run shows the GPU gave the same bytes twice.
//
// Words:
//   result words  the fields of one result as 64-bit numbers, in the order of the struct
//
// Owns:    ToBitsOf, AppendWords (for each result of grid_functions_runner_gpu.h), ToResultWords, CheckSameResults,
//          WriteDigestOfResults
// Uses:    Mat3 (base/mat3.h), Vec3 (base/vec3.h), WriteResultDigest (base/testing/base_support.h), CellCorners,
//          FaceCorners, CellsAroundEdge (grid/cell_corners.h), CellsNearFace (grid/face_search.h), the results of
//          grid/testing/grid_functions_runner_gpu.h, the id types (grid/grid_ids.h), FaceWeight, NearbyFaceWeights
//          (grid/nearby_face_weights.h), TouchingCells, CellFacePieces (grid/touching_cells.h), GoogleTest
// Used by: tests/grid/find_cell_at_on_gpu.test.cc, tests/grid/nearby_face_weights_on_gpu.test.cc and
//          tests/grid/cell_surroundings_on_gpu.test.cc
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/testing/grid_functions_runner_gpu.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {

// The 64 bits of a double.
inline std::uint64_t ToBitsOf(double value) {
  return std::bit_cast<std::uint64_t>(value);
}

inline void AppendWords(Vec3 vector, std::vector<std::uint64_t>& words) {
  words.push_back(ToBitsOf(vector.x));
  words.push_back(ToBitsOf(vector.y));
  words.push_back(ToBitsOf(vector.z));
}

inline void AppendWords(const Mat3& matrix, std::vector<std::uint64_t>& words) {
  for (const double entry :
       {matrix.xx, matrix.xy, matrix.xz, matrix.yx, matrix.yy, matrix.yz, matrix.zx, matrix.zy, matrix.zz}) {
    words.push_back(ToBitsOf(entry));
  }
}

inline void AppendWords(const FoundCells& found, std::vector<std::uint64_t>& words) {
  words.push_back(found.without_start.value);
  words.push_back(found.with_start.value);
}

inline void AppendWords(const CellCorners& corners, std::vector<std::uint64_t>& words) {
  for (const CornerPlaceIndex corner : corners.corners) {
    words.push_back(corner.value);
  }
}

inline void AppendWords(const FaceCorners& corners, std::vector<std::uint64_t>& words) {
  for (const CornerPlaceIndex corner : corners.corners) {
    words.push_back(corner.value);
  }
}

// Not the padding after is_same_size: the struct says no reader writes it.
inline void AppendWords(const CellsAroundEdge& around, std::vector<std::uint64_t>& words) {
  for (const CellPlaceIndex cell : around.cells) {
    words.push_back(cell.value);
  }
  words.push_back(around.is_same_size ? 1 : 0);
}

inline void AppendWords(const TouchingCells& touching, std::vector<std::uint64_t>& words) {
  words.push_back(static_cast<std::uint64_t>(touching.count));
  for (const CellPlaceIndex cell : touching.cells) {
    words.push_back(cell.value);
  }
}

inline void AppendWords(const CellFacePieces& face_pieces, std::vector<std::uint64_t>& words) {
  words.push_back(static_cast<std::uint64_t>(face_pieces.count));
  for (const FacePlaceIndex face : face_pieces.faces) {
    words.push_back(face.value);
  }
}

inline void AppendWords(const CellsNearFace& near_face, std::vector<std::uint64_t>& words) {
  words.push_back(static_cast<std::uint64_t>(near_face.count));
  for (const CellPlaceIndex cell : near_face.cells) {
    words.push_back(cell.value);
  }
}

// The count, and the places the walk kept: the entries past kMaxFaceSearchWalkFaces are not written.
inline void AppendWords(const FaceSearchWalk& walk, std::vector<std::uint64_t>& words) {
  words.push_back(static_cast<std::uint64_t>(walk.faces_found));
  const int kept_count = walk.faces_found < kMaxFaceSearchWalkFaces ? walk.faces_found : kMaxFaceSearchWalkFaces;
  for (int entry = 0; entry < kept_count; ++entry) {
    words.push_back(walk.faces[entry].value);
  }
}

inline void AppendWords(const FaceWeight& face_weight, std::vector<std::uint64_t>& words) {
  words.push_back(face_weight.face.value);
  words.push_back(ToBitsOf(face_weight.weight));
  AppendWords(face_weight.offset, words);
  AppendWords(face_weight.offset_from_mean, words);
  AppendWords(face_weight.weight_change, words);
}

inline void AppendWords(const FaceWeightResult& result, std::vector<std::uint64_t>& words) {
  AppendWords(result.from_list, words);
  AppendWords(result.seen_from_face, words);
  AppendWords(result.seen_from_other_face, words);
}

// Every field of the weights, the whole list array included: its entries from count on are fixed too.
inline void AppendWords(const NearbyFaceResult& result, std::vector<std::uint64_t>& words) {
  const NearbyFaceWeights& weights = result.weights;
  words.push_back(static_cast<std::uint64_t>(result.error));
  words.push_back(weights.list.cell.value);
  words.push_back(ToBitsOf(weights.list.cell_width));
  words.push_back(static_cast<std::uint64_t>(weights.list.axis));
  words.push_back(static_cast<std::uint64_t>(weights.list.count));
  for (const FacePlaceIndex face : weights.list.faces) {
    words.push_back(face.value);
  }
  for (const std::uint64_t word : weights.kept.bits) {
    words.push_back(word);
  }
  AppendWords(weights.position, words);
  words.push_back(ToBitsOf(weights.weight_sum));
  AppendWords(weights.mean_offset, words);
  AppendWords(weights.raw_weight_change_sum, words);
  AppendWords(weights.weighted_spread, words);
  AppendWords(weights.inverse_weighted_spread, words);
  words.push_back(static_cast<std::uint64_t>(weights.fit));
  words.push_back(weights.is_complete ? 1 : 0);
}

// The result words of every result, one after another.
template <typename Result>
std::vector<std::uint64_t> ToResultWords(std::span<const Result> results) {
  std::vector<std::uint64_t> words;
  for (const Result& result : results) {
    AppendWords(result, words);
  }
  return words;
}

// Expects each GPU result to have the words of the CPU's result at the same index; the first 5 differences are named.
// `what` names the results in the failure messages.
template <typename Result>
void CheckSameResults(std::span<const Result> on_cpu, std::span<const Result> on_gpu, std::string_view what) {
  ASSERT_EQ(on_cpu.size(), on_gpu.size());
  int difference_count = 0;
  std::vector<std::uint64_t> cpu_words;
  std::vector<std::uint64_t> gpu_words;
  for (std::size_t index = 0; index < on_cpu.size(); ++index) {
    cpu_words.clear();
    gpu_words.clear();
    AppendWords(on_cpu[index], cpu_words);
    AppendWords(on_gpu[index], gpu_words);
    if (cpu_words != gpu_words) {
      ++difference_count;
      if (difference_count <= 5) {
        ADD_FAILURE() << what << " " << index << ": the GPU's result differs from the CPU's";
      }
    }
  }
  EXPECT_EQ(difference_count, 0) << what << ": results that differ";
}

// Writes the digest of the result words of the GPU results under `name`, unique among the gpu tests.
template <typename Result>
void WriteDigestOfResults(std::string_view name, std::span<const Result> on_gpu) {
  const std::vector<std::uint64_t> words = ToResultWords(on_gpu);
  WriteResultDigest(name, std::span<const std::uint64_t>(words));
}

}  // namespace fluidity::grid
