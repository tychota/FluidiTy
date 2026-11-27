// The result words of the results of the GPU runners: how two results are compared and digested, for the GPU tests of
// the shared grid functions, private to grid/.
//
// Purpose: turn each result of grid_functions_answers.h into a list of 64-bit words, so a test compares a result of
//   the GPU with the CPU's, bit for bit, and digests it, written once. A comparison that fails says which query and
//   which field differed and shows both values, so one failing run on a GPU is enough to see what is wrong.
//
// Idea: each integer of a result goes in as it is and each double as its bits, in the order of the struct, whatever
//   padding the struct holds. Two results are equal when their words are equal; for doubles that is equal bits, which
//   is what the method paper asks of a formula built from + - * /, sqrt and floor (<sec-run-on>). A list of words may
//   also carry the name of each word ("weights.position.x"); it is built only for a result that differs, so the
//   comparison of the other results does not pay for names. The words of every GPU result go to WriteResultDigest, so
//   the repeat run shows the GPU gave the same bytes twice.
//
// Words:
//   result words  the fields of one result as 64-bit numbers, in the order of the struct
//
// Owns:    ResultWords, WordNames, ToBitsOf, ApplyWordsOf (for each result of grid_functions_answers.h),
//          ToResultWords, kMostNamedDifferences, DifferenceReport, CheckNamedDifferences,
//          CheckSameResults, WriteDigestOfResults
// Uses:    Mat3 (base/mat3.h), Vec3 (base/vec3.h), WriteResultDigest (base/testing/base_support.h), CellCorners,
//          FaceCorners, CellsAroundEdge (grid/cell_corners.h), CellsNearFace (grid/face_search.h), the results of
//          grid/testing/grid_functions_answers.h, the id types (grid/grid_ids.h), NewCellNeighbors
//          (grid/new_cell_rule.h), FaceWeight, NearbyFaceWeights
//          (grid/nearby_face_weights.h), TouchingCells, CellFacePieces (grid/touching_cells.h), GoogleTest
// Used by: the GPU tests of the grid in tests/grid/ that compare a runner with the CPU
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/new_cell_rule.h"
#include "fluidity/grid/testing/grid_functions_answers.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {

// The 64 bits of a double.
inline std::uint64_t ToBitsOf(double value) {
  return std::bit_cast<std::uint64_t>(value);
}

// Whether a list of words keeps the name of each word.
enum class WordNames : std::uint8_t {
  kOmitted,  // Only the values.
  kKept,  // The values and the name of each.
};

// The words of one result, in the order of its struct. With names, each word also keeps the name of its field.
class ResultWords {
 public:
  explicit ResultWords(WordNames names) : has_names_(names == WordNames::kKept) {
  }

  // Adds an integer field.
  void Add(std::string_view field, std::uint64_t value) {
    values_.push_back(value);
    if (has_names_) {
      names_.push_back(prefix_ + std::string(field));
      is_double_.push_back(false);
    }
  }

  // Adds a double field as its bits.
  void AddDouble(std::string_view field, double value) {
    values_.push_back(ToBitsOf(value));
    if (has_names_) {
      names_.push_back(prefix_ + std::string(field));
      is_double_.push_back(true);
    }
  }

  // Adds the entry `index` of an array of integers.
  // clang-tidy: an index and a word are both 64-bit integers; the swappable-parameters check flags them.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  void AddEntry(std::string_view field, std::size_t index, std::uint64_t value) {
    values_.push_back(value);
    if (has_names_) {
      names_.push_back(prefix_ + std::string(field) + "[" + std::to_string(index) + "]");
      is_double_.push_back(false);
    }
  }

  // Makes the names of the fields added from now on start with `scope` and a point, until the matching Leave.
  void Enter(std::string_view scope) {
    if (has_names_) {
      scope_lengths_.push_back(prefix_.size());
      prefix_ += std::string(scope) + ".";
    }
  }

  // Ends the scope of the last Enter.
  void Leave() {
    if (has_names_) {
      prefix_.resize(scope_lengths_.back());
      scope_lengths_.pop_back();
    }
  }

  void Clear() {
    values_.clear();
    names_.clear();
    is_double_.clear();
  }

  [[nodiscard]] const std::vector<std::uint64_t>& values() const {
    return values_;
  }

  // The name of the word `index`; empty when the list has no names.
  [[nodiscard]] std::string NameOf(std::size_t index) const {
    return index < names_.size() ? names_[index] : std::string();
  }

  // The word `index` as hexadecimal, and for a double also as a decimal with 17 digits: "0x3ff0000000000000 (1)".
  [[nodiscard]] std::string ToTextOf(std::size_t index) const {
    std::ostringstream text;
    text << "0x" << std::hex << std::setw(16) << std::setfill('0') << values_[index];
    if (index < is_double_.size() && is_double_[index]) {
      text << " (" << std::dec << std::setprecision(17) << std::bit_cast<double>(values_[index]) << ")";
    }
    return text.str();
  }

 private:
  bool has_names_;
  std::vector<std::uint64_t> values_;
  std::vector<std::string> names_;
  std::vector<bool> is_double_;
  std::string prefix_;
  std::vector<std::size_t> scope_lengths_;
};

inline void ApplyWordsOf(Vec3 vector, ResultWords& words) {
  words.AddDouble("x", vector.x);
  words.AddDouble("y", vector.y);
  words.AddDouble("z", vector.z);
}

inline void ApplyWordsOf(const Mat3& matrix, ResultWords& words) {
  words.AddDouble("xx", matrix.xx);
  words.AddDouble("xy", matrix.xy);
  words.AddDouble("xz", matrix.xz);
  words.AddDouble("yx", matrix.yx);
  words.AddDouble("yy", matrix.yy);
  words.AddDouble("yz", matrix.yz);
  words.AddDouble("zx", matrix.zx);
  words.AddDouble("zy", matrix.zy);
  words.AddDouble("zz", matrix.zz);
}

inline void ApplyWordsOf(const FoundCells& found, ResultWords& words) {
  words.Add("without_start", found.without_start.value);
  words.Add("with_start", found.with_start.value);
}

inline void ApplyWordsOf(const CellCorners& corners, ResultWords& words) {
  std::size_t entry = 0;
  for (const CornerPlaceIndex corner : corners.corners) {
    words.AddEntry("corners", entry++, corner.value);
  }
}

inline void ApplyWordsOf(const FaceCorners& corners, ResultWords& words) {
  std::size_t entry = 0;
  for (const CornerPlaceIndex corner : corners.corners) {
    words.AddEntry("corners", entry++, corner.value);
  }
}

// Not the padding after is_same_size: the struct says no reader writes it.
inline void ApplyWordsOf(const CellsAroundEdge& around, ResultWords& words) {
  std::size_t entry = 0;
  for (const CellPlaceIndex cell : around.cells) {
    words.AddEntry("cells", entry++, cell.value);
  }
  words.Add("is_same_size", around.is_same_size ? 1 : 0);
}

inline void ApplyWordsOf(const TouchingCells& touching, ResultWords& words) {
  words.Add("count", static_cast<std::uint64_t>(touching.count));
  std::size_t entry = 0;
  for (const CellPlaceIndex cell : touching.cells) {
    words.AddEntry("cells", entry++, cell.value);
  }
}

inline void ApplyWordsOf(const NewCellNeighbors& neighbors, ResultWords& words) {
  words.Add("touches_much_smaller_cell", neighbors.touches_much_smaller_cell ? 1 : 0);
  words.Add("touches_bigger_cell", neighbors.touches_bigger_cell ? 1 : 0);
}

inline void ApplyWordsOf(const CellFacePieces& face_pieces, ResultWords& words) {
  words.Add("count", static_cast<std::uint64_t>(face_pieces.count));
  std::size_t entry = 0;
  for (const FacePlaceIndex face : face_pieces.faces) {
    words.AddEntry("faces", entry++, face.value);
  }
}

inline void ApplyWordsOf(const CellsNearFace& near_face, ResultWords& words) {
  words.Add("count", static_cast<std::uint64_t>(near_face.count));
  std::size_t entry = 0;
  for (const CellPlaceIndex cell : near_face.cells) {
    words.AddEntry("cells", entry++, cell.value);
  }
}

// The count, and the places the walk kept: the entries past kMaxFaceSearchWalkFaces are not written.
inline void ApplyWordsOf(const FaceSearchWalk& walk, ResultWords& words) {
  words.Add("faces_found", static_cast<std::uint64_t>(walk.faces_found));
  const int kept_count = walk.faces_found < kMaxFaceSearchWalkFaces ? walk.faces_found : kMaxFaceSearchWalkFaces;
  for (int entry = 0; entry < kept_count; ++entry) {
    words.AddEntry("faces", static_cast<std::size_t>(entry), walk.faces[entry].value);
  }
}

inline void ApplyWordsOf(const FaceWeight& face_weight, ResultWords& words) {
  words.Add("face", face_weight.face.value);
  words.AddDouble("weight", face_weight.weight);
  words.Enter("offset");
  ApplyWordsOf(face_weight.offset, words);
  words.Leave();
  words.Enter("offset_from_mean");
  ApplyWordsOf(face_weight.offset_from_mean, words);
  words.Leave();
  words.Enter("weight_change");
  ApplyWordsOf(face_weight.weight_change, words);
  words.Leave();
}

inline void ApplyWordsOf(const FaceWeightResult& result, ResultWords& words) {
  words.Enter("from_list");
  ApplyWordsOf(result.from_list, words);
  words.Leave();
  words.Enter("seen_from_face");
  ApplyWordsOf(result.seen_from_face, words);
  words.Leave();
  words.Enter("seen_from_unlisted_face");
  ApplyWordsOf(result.seen_from_unlisted_face, words);
  words.Leave();
  words.Enter("seen_from_other_axis_face");
  ApplyWordsOf(result.seen_from_other_axis_face, words);
  words.Leave();
}

// Every field of the weights, the whole list array included: its entries from count on are fixed too.
inline void ApplyWordsOf(const NearbyFaceResult& result, ResultWords& words) {
  const NearbyFaceWeights& weights = result.weights;
  words.Add("error", static_cast<std::uint64_t>(result.error));
  words.Add("list.cell", weights.list.cell.value);
  words.AddDouble("list.cell_width", weights.list.cell_width);
  words.Add("list.axis", static_cast<std::uint64_t>(weights.list.axis));
  words.Add("list.count", static_cast<std::uint64_t>(weights.list.count));
  std::size_t entry = 0;
  for (const FacePlaceIndex face : weights.list.faces) {
    words.AddEntry("list.faces", entry++, face.value);
  }
  entry = 0;
  for (const std::uint64_t word : weights.kept.bits) {
    words.AddEntry("kept", entry++, word);
  }
  words.Enter("position");
  ApplyWordsOf(weights.position, words);
  words.Leave();
  words.AddDouble("weight_sum", weights.weight_sum);
  words.Enter("mean_offset");
  ApplyWordsOf(weights.mean_offset, words);
  words.Leave();
  words.Enter("raw_weight_change_sum");
  ApplyWordsOf(weights.raw_weight_change_sum, words);
  words.Leave();
  words.Enter("weighted_spread");
  ApplyWordsOf(weights.weighted_spread, words);
  words.Leave();
  words.Enter("inverse_weighted_spread");
  ApplyWordsOf(weights.inverse_weighted_spread, words);
  words.Leave();
  words.Add("fit", static_cast<std::uint64_t>(weights.fit));
  words.Add("is_complete", weights.is_complete ? 1 : 0);
}

// Every width of the table, those past size_level_count included: they are 0 on both sides.
inline void ApplyWordsOf(const FormulaResult& result, ResultWords& words) {
  words.AddDouble("complete_weighted_spread", result.complete_weighted_spread);
  words.AddDouble("reference_speed", result.reference_speed);
  std::size_t entry = 0;
  for (const double width : result.widths.width) {
    words.AddDouble("widths.width[" + std::to_string(entry++) + "]", width);
  }
  words.Add("widths.size_level_count", result.widths.size_level_count);
  words.Add("sort_count", result.sort_count);
  words.Add("small_cell", result.small_cell.value);
}

inline void ApplyWordsOf(const CellPlaceFacts& facts, ResultWords& words) {
  words.Enter("center");
  ApplyWordsOf(facts.center, words);
  words.Leave();
  words.Add("id", facts.id.value);
  words.Add("place_of_id", facts.place_of_id.value);
}

inline void ApplyWordsOf(const BorderBlockResult& result, ResultWords& words) {
  words.Add("is_border", result.is_border);
}

inline void ApplyWordsOf(const PieceStepResult& result, ResultWords& words) {
  words.Add("piece_place", result.piece_place);
  words.Add("piece_neighbour", result.piece_neighbour);
  words.Add("face_kind", result.face_kind);
}

inline void ApplyWordsOf(const ReadMarksResult& result, ResultWords& words) {
  int block = 0;
  for (const std::uint8_t mark : result.marks) {
    words.Add("marks[" + std::to_string(block++) + "]", mark);
  }
}

inline void ApplyWordsOf(const FacePlaceFacts& facts, ResultWords& words) {
  words.AddDouble("center_distance", facts.center_distance);
  words.Add("id", facts.id.value);
  words.Add("id_from_parts", facts.id_from_parts.value);
  words.Add("place_of_id", facts.place_of_id.value);
}

// The result words of every result, one after another.
template <typename Result>
std::vector<std::uint64_t> ToResultWords(std::span<const Result> results) {
  ResultWords words(WordNames::kOmitted);
  for (const Result& result : results) {
    ApplyWordsOf(result, words);
  }
  return words.values();
}

// The most differences CheckSameResults names.
inline constexpr int kMostNamedDifferences = 10;

// What CheckSameResults has found so far.
struct DifferenceReport {
  std::string_view what;  // Names the results in the failure messages.
  int differing_query_count = 0;
  int differing_word_count = 0;
  int named_count = 0;  // The failures added so far, at most kMostNamedDifferences.
};

// Adds a failure for each word of one query that differs, until the report has named kMostNamedDifferences words, and
// counts the words that differ. The words are rebuilt here with their names.
// clang-tidy: the CPU's result and the GPU's have one type; the swappable-parameters check flags them.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <typename Result>
void CheckNamedDifferences(
    const Result& on_cpu,
    const Result& on_gpu,
    std::size_t query_index,
    DifferenceReport& report
) {
  ResultWords cpu_words(WordNames::kKept);
  ResultWords gpu_words(WordNames::kKept);
  ApplyWordsOf(on_cpu, cpu_words);
  ApplyWordsOf(on_gpu, gpu_words);
  const std::size_t cpu_size = cpu_words.values().size();
  const std::size_t gpu_size = gpu_words.values().size();
  const std::size_t word_count = cpu_size > gpu_size ? cpu_size : gpu_size;
  for (std::size_t word = 0; word < word_count; ++word) {
    const bool is_on_both = word < cpu_size && word < gpu_size;
    if (is_on_both && cpu_words.values()[word] == gpu_words.values()[word]) {
      continue;
    }
    ++report.differing_word_count;
    if (report.named_count >= kMostNamedDifferences) {
      continue;
    }
    ++report.named_count;
    const ResultWords& longer = cpu_size > gpu_size ? cpu_words : gpu_words;
    const std::string cpu_text = word < cpu_size ? cpu_words.ToTextOf(word) : "none";
    const std::string gpu_text = word < gpu_size ? gpu_words.ToTextOf(word) : "none";
    ADD_FAILURE() << report.what << ", query " << query_index << ", field " << longer.NameOf(word) << ": CPU "
                  << cpu_text << ", GPU " << gpu_text;
  }
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Expects each GPU result to have the words of the CPU's result at the same index: the same bits in every field. Names
// the first kMostNamedDifferences words that differ, each with its query index, its field and both values in
// hexadecimal, then the count of the queries and of the words. `what` names the results in the failure messages.
template <typename Result>
void CheckSameResults(std::span<const Result> on_cpu, std::span<const Result> on_gpu, std::string_view what) {
  ASSERT_EQ(on_cpu.size(), on_gpu.size()) << what;
  ResultWords cpu_words(WordNames::kOmitted);
  ResultWords gpu_words(WordNames::kOmitted);
  DifferenceReport report = {.what = what};
  for (std::size_t index = 0; index < on_cpu.size(); ++index) {
    cpu_words.Clear();
    gpu_words.Clear();
    ApplyWordsOf(on_cpu[index], cpu_words);
    ApplyWordsOf(on_gpu[index], gpu_words);
    if (cpu_words.values() != gpu_words.values()) {
      ++report.differing_query_count;
      CheckNamedDifferences(on_cpu[index], on_gpu[index], index, report);
    }
  }
  if (report.differing_query_count > 0) {
    ADD_FAILURE() << what << ": " << report.differing_query_count << " of " << on_cpu.size() << " queries differ, "
                  << report.differing_word_count << " words in all; the first " << report.named_count
                  << " are named above";
  }
}

// Writes the digest of the result words of the GPU results under `name`, unique among the gpu tests.
template <typename Result>
void WriteDigestOfResults(std::string_view name, std::span<const Result> on_gpu) {
  const std::vector<std::uint64_t> words = ToResultWords(on_gpu);
  WriteResultDigest(name, std::span<const std::uint64_t>(words));
}

}  // namespace fluidity::grid
