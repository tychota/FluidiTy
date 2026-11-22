#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "fluidity/base/scalar_math.h"
#include "fluidity/grid/cell_size_rule.h"
#include "fluidity/grid/cell_size_rule_test_cases.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_test_support.h"

namespace fluidity::grid {
namespace {

// The table, written once at compile time: the checks of the doubles' bit patterns are static_asserts.
constexpr CellSizeFactorTable kTable = ComputeCellSizeFactorTable();

constexpr bool ComputeIsStrictlyIncreasing() {
  for (int factor_index = 0; factor_index < 48; ++factor_index) {
    if (!(kTable.factor[factor_index] < kTable.factor[factor_index + 1])) {
      return false;
    }
  }
  return true;
}

constexpr bool ComputeEveryFourthFactorDoubles() {
  for (int factor_index = 0; factor_index <= 44; ++factor_index) {
    if (kTable.factor[factor_index + 4] != 2 * kTable.factor[factor_index]) {
      return false;
    }
  }
  return true;
}

static_assert(kCellSizeFactorStepsPerDoubling == 4);
static_assert(kMaxCellSizeFactorCode == 24);
static_assert(sizeof(kTable.factor) == 49 * sizeof(double));
static_assert(kTable.factor[24] == 1.0);
static_assert(kTable.factor[28] == 2.0);
static_assert(kTable.factor[20] == 0.5);
static_assert(kTable.factor[0] == 0x1p-6);
static_assert(kTable.factor[48] == 64.0);
static_assert(kTable.factor[25] == 0x1.306fe0a31b715p+0);
// NOLINTNEXTLINE(modernize-use-std-numbers): the literal is the specification.
static_assert(kTable.factor[26] == 0x1.6a09e667f3bcdp+0);
static_assert(kTable.factor[27] == 0x1.ae89f995ad3adp+0);
static_assert(kTable.factor[1] == 0x1.306fe0a31b715p-6);
static_assert(kTable.factor[47] == 0x1.ae89f995ad3adp+5);
static_assert(ComputeEveryFourthFactorDoubles());
static_assert(ComputeIsStrictlyIncreasing());

// The fixture of cell_size_rule_test_cases.h, which the GPU test shares.
constexpr SizeLevel kSmallestSizeLevel = kFactorFixtureSmallestSizeLevel;
constexpr int kFixtureCountX = kFactorFixtureCountX;
constexpr int kFixtureCountY = kFactorFixtureCountY;
constexpr int kFixtureCountZ = kFactorFixtureCountZ;
constexpr int kFixtureSpacing = kFactorFixtureSpacing;

TestRefinementVolume BuildFixtureVolume(int layer_count) {
  return BuildFactorFixtureVolume(layer_count);
}

CellId ToTestCell(int size_level, int x, int y, int z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// The factor code of the cell of `size_level` at (x, y, z), with the fixture's smallest size level.
// clang-tidy: a size level and three coordinates; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int ComputeFixtureCode(const RefinementFactors& factors, int size_level, int x, int y, int z) {
  return ComputeCellSizeFactorCode(factors, kSmallestSizeLevel, ToTestCell(size_level, x, y, z));
}

TEST(ComputeCellSizeFactorTable, QuarterDoublings_AreExact) {
  // The static_asserts above prove the bit patterns at compile time; this repeats them where a failure names a line.
  EXPECT_EQ(kTable.factor[24], 1.0);
  // NOLINTNEXTLINE(modernize-use-std-numbers): the literal is the specification.
  EXPECT_EQ(kTable.factor[26], 0x1.6a09e667f3bcdp+0);
  EXPECT_EQ(kTable.factor[47], 0x1.ae89f995ad3adp+5);
  EXPECT_EQ(kTable.factor[48], 64.0);
}

TEST(ComputeCellSizeFactorTable, MiddleRoot_IsTheSquareRootOfTwo) {
  EXPECT_EQ(kTable.factor[26], math::Sqrt(2.0));
}

TEST(ComputeCellSizeFactorCode, NoVolume_IsZero) {
  const RefinementFactors no_volume{};
  for (const CellId cell : BuildRandomGridCells(7, 5, 4)) {
    EXPECT_EQ(ComputeCellSizeFactorCode(no_volume, kSmallestSizeLevel, cell), 0);
  }
}

TEST(ComputeCellSizeFactorCode, SmallCell_ReadsItsEntry) {
  const TestRefinementVolume volume = BuildFixtureVolume(3);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, 20, 3, 3), -8);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, 19, 3, 3), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, 23, 0, 0), -8);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, 24, 0, 0), 0);
}

TEST(ComputeCellSizeFactorCode, BigCell_IsSmallestCodeOverItsBox) {
  const TestRefinementVolume volume = BuildFixtureVolume(3);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 0, 1, 0, 0), -8);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 0, 0, 0, 0), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 2, 5, 0, 0), -8);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 2, 4, 0, 0), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 1, 2, 1, 1), -8);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 1, 1, 1, 1), 0);
}

TEST(ComputeCellSizeFactorCode, OutsideTheVolume_IsZero) {
  const TestRefinementVolume volume = BuildFixtureVolume(3);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, 40, 0, 0), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, -1, 0, 0), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, 20, -1, 3), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, 20, 3, -1), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, 20, 16, 3), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, 20, 3, 16), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, kMaxCoord, kMaxCoord, kMaxCoord), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 4, -kMaxCoord, 0, 0), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 0, 2, 0, 0), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 0, 1, 1, 0), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 0, -1, 0, 0), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 0, kMaxCoord + 1, 0, 0), 0);
}

TEST(ComputeCellSizeFactorCode, MoreSizeLevelsThanLayers_IsZero) {
  const TestRefinementVolume volume = BuildFixtureVolume(2);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 0, 1, 0, 0), 0);
  EXPECT_EQ(ComputeFixtureCode(volume.factors, 1, 2, 0, 0), -8);
  const TestRefinementVolume only_layer_zero = BuildFixtureVolume(1);
  EXPECT_EQ(ComputeFixtureCode(only_layer_zero.factors, 1, 2, 0, 0), 0);
  EXPECT_EQ(ComputeFixtureCode(only_layer_zero.factors, 2, 5, 0, 0), -8);
}

TEST(ComputeCellSizeFactorCode, LayerAtLayerCount_IsNotRead) {
  // layer_count is 2, yet a third layer's array is set: the lookup must not read it, since the volume says it holds
  // no data.
  const TestRefinementVolume three_layers = BuildFixtureVolume(3);
  RefinementFactors factors = three_layers.factors;
  factors.layer_count = 2;
  ASSERT_NE(factors.layer_codes[2], nullptr);
  EXPECT_EQ(ComputeFixtureCode(three_layers.factors, 0, 1, 0, 0), -8);
  EXPECT_EQ(ComputeFixtureCode(factors, 0, 1, 0, 0), 0);
}

TEST(ComputeCellSizeFactorCode, SmallestSizeLevelFarBelowTheCell_ReadsNoLayerPastTheLayers) {
  const TestRefinementVolume volume = BuildFixtureVolume(3);
  for (const int smallest : {5, 8, 15, 100, 255}) {
    const CellId cell = ToTestCell(0, 1, 0, 0);
    EXPECT_EQ(ComputeCellSizeFactorCode(volume.factors, static_cast<SizeLevel>(smallest), cell), 0) << smallest;
  }
}

// 2^exponent as a signed integer, shifted as an unsigned value so no signed shift arises. Requires exponent 0..62.
std::int64_t ComputePowerOfTwo(int exponent) {
  return static_cast<std::int64_t>(std::uint64_t{1} << static_cast<unsigned>(exponent));
}

// The code a one-entry volume (one size-level-4 cell wide) gives for the stored code.
int ComputeCodeOfStoredCode(int stored_code) {
  const std::vector<std::int8_t> codes = {static_cast<std::int8_t>(stored_code)};
  const TestRefinementVolume volume = BuildTestRefinementFactors(1, 1, 1, 0, 1, codes);
  return ComputeFixtureCode(volume.factors, 4, 0, 0, 0);
}

TEST(ComputeCellSizeFactorCode, StoredCodeAtTheLimits_IsClampedToTheTable) {
  for (const int stored : {-24, 24, -23, 23, -25, 25, 127, -128, 0}) {
    EXPECT_EQ(ComputeCodeOfStoredCode(stored), std::clamp(stored, -24, 24)) << stored;
  }
}

TEST(ComputeCellSizeFactorCode, EveryStoredCode_IndexesTheTable) {
  for (int stored = -128; stored <= 127; ++stored) {
    const int table_index = ComputeCodeOfStoredCode(stored) + kMaxCellSizeFactorCode;
    EXPECT_GE(table_index, 0) << stored;
    EXPECT_LE(table_index, 48) << stored;
  }
}

// The probe: a 1 x 1 x 1 volume of `layer_count` layers, spacing 0, every entry code -5, so any read of a layer shows
// as -5 and a lookup that reads nothing shows as 0. The counts, spacing and layer count are then changed one at a time.
TestRefinementVolume BuildProbeVolume(int layer_count) {
  const std::vector<std::int8_t> codes = {-5};
  return BuildTestRefinementFactors(1, 1, 1, 0, layer_count, codes);
}

// The code of the probe's size-level-4 cell {0, 0, 0} in `factors`, whose smallest size level is 4 (layer 0).
int ComputeProbeCode(const RefinementFactors& factors) {
  return ComputeFixtureCode(factors, 4, 0, 0, 0);
}

TEST(ComputeCellSizeFactorCode, ProbeVolume_ReadsItsCode) {
  const TestRefinementVolume volume = BuildProbeVolume(3);
  EXPECT_EQ(ComputeProbeCode(volume.factors), -5);
}

TEST(ComputeCellSizeFactorCode, CountOutsideOneToTwoToTheTwenty_IsZeroOnEveryAxis) {
  const TestRefinementVolume volume = BuildProbeVolume(3);
  for (const int bad_count :
       {-1, 0, 1'048'577, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max()}) {
    RefinementFactors bad_x = volume.factors;
    bad_x.count_x = bad_count;
    EXPECT_EQ(ComputeProbeCode(bad_x), 0) << "x " << bad_count;
    RefinementFactors bad_y = volume.factors;
    bad_y.count_y = bad_count;
    EXPECT_EQ(ComputeProbeCode(bad_y), 0) << "y " << bad_count;
    RefinementFactors bad_z = volume.factors;
    bad_z.count_z = bad_count;
    EXPECT_EQ(ComputeProbeCode(bad_z), 0) << "z " << bad_count;
  }
}

// A volume of `count` entries on `axis` (0 for x, 1 for y, 2 for z) and 1 on the others, spacing 0, 3 layers, layer 0
// holding `code` at every entry except those of `coded_entries`, which hold -7.
// clang-tidy: three integers of different meaning; the swappable-parameters check flags them.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
TestRefinementVolume BuildLongVolume(int axis, int count, std::int8_t code, const std::vector<int>& coded_entries) {
  std::vector<std::int8_t> codes(static_cast<std::size_t>(count), code);
  for (const int coded_entry : coded_entries) {
    codes[static_cast<std::size_t>(coded_entry)] = -7;
  }
  return BuildTestRefinementFactors(axis == 0 ? count : 1, axis == 1 ? count : 1, axis == 2 ? count : 1, 0, 3, codes);
}

// The code of the size-level-`size_level` cell at coordinate `coord` on `axis` and 0 on the others, in `volume`.
// clang-tidy: three integers of different meaning; the swappable-parameters check flags them.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int ComputeLongVolumeCode(const TestRefinementVolume& volume, int axis, int size_level, int coord) {
  const int x = axis == 0 ? coord : 0;
  const int y = axis == 1 ? coord : 0;
  const int z = axis == 2 ? coord : 0;
  return ComputeFixtureCode(volume.factors, size_level, x, y, z);
}

// The next two tests: 2^20 entries on one axis, each array as long as its counts say, codes 0 except -7 at entries
// kMaxCoord = 2^18 - 1 and 2^20 - 1, with smallest size level 4 and spacing 0. A cell coordinate is at most kMaxCoord.
constexpr int kLongCount = 1'048'576;

TEST(ComputeCellSizeFactorCode, CountOfTwoToTheTwenty_ReadsLayerZeroAtTheLargestCoordinate) {
  // A size-level-4 cell reads layer 0 at its coordinate: entry kMaxCoord, and the entry before it holds 0.
  for (int axis = 0; axis < 3; ++axis) {
    const TestRefinementVolume volume = BuildLongVolume(axis, kLongCount, 0, {kMaxCoord, kLongCount - 1});
    EXPECT_EQ(ComputeLongVolumeCode(volume, axis, 4, kMaxCoord), -7) << axis;
    EXPECT_EQ(ComputeLongVolumeCode(volume, axis, 4, kMaxCoord - 1), 0) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, CountOfTwoToTheTwenty_ReadsItsLastEntry) {
  // A size-level-2 cell is 4 smallest cells wide and reads layer 2 at index kMaxCoord: the smallest code over layer-0
  // entries 4 * kMaxCoord = 2^20 - 4 up to 2^20 - 1, so the last entry is reached this way. The cell before it covers
  // entries 2^20 - 8 up to 2^20 - 5, which hold 0.
  for (int axis = 0; axis < 3; ++axis) {
    const TestRefinementVolume volume = BuildLongVolume(axis, kLongCount, 0, {kMaxCoord, kLongCount - 1});
    EXPECT_EQ(ComputeLongVolumeCode(volume, axis, 2, kMaxCoord), -7) << axis;
    EXPECT_EQ(ComputeLongVolumeCode(volume, axis, 2, kMaxCoord - 1), 0) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, SmallCounts_AreReadOnEveryAxis) {
  // Real volumes of 1 and of 2 entries on one axis, each array as long as its counts say, every entry -5.
  for (int axis = 0; axis < 3; ++axis) {
    for (const int count : {1, 2}) {
      const TestRefinementVolume volume = BuildLongVolume(axis, count, -5, {});
      EXPECT_EQ(ComputeProbeCode(volume.factors), -5) << axis << " " << count;
    }
  }
}

TEST(ComputeCellSizeFactorCode, SpacingOutsideZeroToThirtyOne_IsZeroWithMoreThanOneLayer) {
  const TestRefinementVolume volume = BuildProbeVolume(3);
  for (const int bad_spacing :
       {-1, 32, 33, 1000, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max()}) {
    RefinementFactors factors = volume.factors;
    factors.entry_spacing_log2 = bad_spacing;
    EXPECT_EQ(ComputeProbeCode(factors), 0) << bad_spacing;
    EXPECT_EQ(ComputeFixtureCode(factors, 0, 0, 0, 0), 0) << bad_spacing;
  }
}

TEST(ComputeCellSizeFactorCode, SpacingZeroToThirtyOne_IsRead) {
  const TestRefinementVolume volume = BuildProbeVolume(3);
  for (const int good_spacing : {0, 1, 30, 31}) {
    RefinementFactors factors = volume.factors;
    factors.entry_spacing_log2 = good_spacing;
    EXPECT_EQ(ComputeProbeCode(factors), -5) << good_spacing;
  }
  RefinementFactors big_cell_factors = volume.factors;
  big_cell_factors.entry_spacing_log2 = 31;
  // A size-level-0 cell is 16 smallest cells wide, which is inside one entry of 2^31 smallest cells.
  EXPECT_EQ(ComputeFixtureCode(big_cell_factors, 0, 0, 0, 0), -5);
}

TEST(ComputeCellSizeFactorCode, LayerCountAboveSixteen_ReadsAsSixteen) {
  const TestRefinementVolume volume = BuildProbeVolume(16);
  const CellId deepest_cell = ToTestCell(0, 0, 0, 0);
  for (const int layer_count : {16, 17, 1000, std::numeric_limits<std::int32_t>::max()}) {
    RefinementFactors factors = volume.factors;
    factors.layer_count = layer_count;
    EXPECT_EQ(ComputeCellSizeFactorCode(factors, SizeLevel{15}, deepest_cell), -5) << layer_count;
  }
}

TEST(ComputeCellSizeFactorCode, LayerCountBelowTheLayerRead_IsZero) {
  const TestRefinementVolume volume = BuildProbeVolume(16);
  const CellId deepest_cell = ToTestCell(0, 0, 0, 0);
  for (const int layer_count : {15, 1, 0, -1, std::numeric_limits<std::int32_t>::min()}) {
    RefinementFactors factors = volume.factors;
    factors.layer_count = layer_count;
    EXPECT_EQ(ComputeCellSizeFactorCode(factors, SizeLevel{15}, deepest_cell), 0) << layer_count;
  }
  RefinementFactors only_layer_zero = volume.factors;
  only_layer_zero.layer_count = 1;
  EXPECT_EQ(ComputeCellSizeFactorCode(only_layer_zero, SizeLevel{15}, ToTestCell(15, 0, 0, 0)), -5);
}

// The factors followed by pointers to a layer holding -5, so a read past the factors' 16 layer pointers shows as -5.
struct FactorsWithGuardPointers {
  RefinementFactors factors;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): the pointers must follow the factors in memory.
  const std::int8_t* guard_layers[8] = {};
};

TEST(ComputeCellSizeFactorCode, LayerBeyondSixteen_IsZeroWhateverFollowsTheLayerPointers) {
  const TestRefinementVolume volume = BuildProbeVolume(16);
  const std::vector<std::int8_t> guard_codes = {-5};
  FactorsWithGuardPointers guarded{};
  guarded.factors = volume.factors;
  guarded.factors.layer_count = 1000;
  for (const std::int8_t*& guard_layer : guarded.guard_layers) {
    guard_layer = guard_codes.data();
  }
  // Smallest size level 20 and a size-level-0 cell: 20 doublings, layer 20, past the 16 layer pointers.
  for (const int smallest : {16, 17, 20, 255}) {
    const CellId cell = ToTestCell(0, 0, 0, 0);
    EXPECT_EQ(ComputeCellSizeFactorCode(guarded.factors, static_cast<SizeLevel>(smallest), cell), 0) << smallest;
  }
}

TEST(ComputeCellSizeFactorCode, NullLayerZero_IsZero) {
  const TestRefinementVolume volume = BuildProbeVolume(3);
  RefinementFactors factors = volume.factors;
  factors.layer_codes[0] = nullptr;
  EXPECT_EQ(ComputeProbeCode(factors), 0);
}

TEST(ComputeCellSizeFactorCode, CellSmallerThanTheSmallestSizeLevel_IsZero) {
  // The probe's entry 0 holds -5; a size-level-4 cell {0, 0, 0} with smallest size level 3 would read it if the lookup
  // did not stop, since its coordinate is 0.
  const TestRefinementVolume volume = BuildProbeVolume(3);
  EXPECT_EQ(ComputeCellSizeFactorCode(volume.factors, SizeLevel{3}, ToTestCell(4, 0, 0, 0)), 0);
  EXPECT_EQ(ComputeCellSizeFactorCode(volume.factors, SizeLevel{0}, ToTestCell(15, 0, 0, 0)), 0);
  EXPECT_EQ(ComputeCellSizeFactorCode(volume.factors, SizeLevel{4}, ToTestCell(4, 0, 0, 0)), -5);
}

TEST(ComputeCellSizeFactorCode, NullLayerBeyondLayerZero_IsZero) {
  const TestRefinementVolume volume = BuildFixtureVolume(3);
  RefinementFactors factors = volume.factors;
  factors.layer_count = 1000;
  EXPECT_EQ(ComputeFixtureCode(factors, 0, 1, 0, 0), -8);
  EXPECT_EQ(ComputeFixtureCode(factors, 4, 20, 3, 3), -8);
  factors.layer_codes[2] = nullptr;
  EXPECT_EQ(ComputeFixtureCode(factors, 0, 1, 0, 0), 0);
}

// The entries of layer 0 that the box of a cell covers on one axis, as the first and last index, cut to the volume, and
// whether the box reaches outside the volume. The last is below the first when no entry of the volume is covered.
// Signed 64-bit arithmetic; no shifts, so a negative coordinate divides exactly.
struct EntryRange {
  std::int64_t first;
  std::int64_t last;
  bool reaches_outside;
};

// The cells of the box of a cell along one axis are coord * 2^cell_width_log2 .. (coord + 1) * 2^cell_width_log2 - 1 in
// smallest cells; an entry is 2^spacing smallest cells, and floor division gives its index.
// clang-tidy: a coordinate and three counts; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
EntryRange ComputeEntryRange(std::int64_t axis_coord, int cell_width_log2, int entry_spacing_log2, int count) {
  const auto cells_per_cell = ComputePowerOfTwo(cell_width_log2);
  const auto cells_per_entry = ComputePowerOfTwo(entry_spacing_log2);
  const std::int64_t first_cell = axis_coord * cells_per_cell;
  const std::int64_t last_cell = first_cell + cells_per_cell - 1;
  const auto floor_divide = [&](std::int64_t value) {
    return value >= 0 ? value / cells_per_entry : -((-value + cells_per_entry - 1) / cells_per_entry);
  };
  const std::int64_t uncut_first = floor_divide(first_cell);
  const std::int64_t uncut_last = floor_divide(last_cell);
  EntryRange range{};
  range.first = std::max<std::int64_t>(uncut_first, 0);
  range.last = std::min<std::int64_t>(uncut_last, count - 1);
  range.reaches_outside = uncut_first < 0 || uncut_last > count - 1;
  return range;
}

// The smallest code over the box of `cell`, found by looking at every layer-0 entry of the box, where outside the
// volume the code is 0, clamped to the table's limits. A cell wholly outside gives 0.
int ComputeBruteForceCode(const TestRefinementVolume& volume, SizeLevel smallest_size_level, CellId cell) {
  const RefinementFactors& factors = volume.factors;
  const int cell_width_log2 = static_cast<int>(smallest_size_level) - static_cast<int>(ToSizeLevel(cell));
  const CellCoord coord = ToCellCoord(cell);
  const int spacing = factors.entry_spacing_log2;
  const EntryRange range_x = ComputeEntryRange(coord.x, cell_width_log2, spacing, factors.count_x);
  const EntryRange range_y = ComputeEntryRange(coord.y, cell_width_log2, spacing, factors.count_y);
  const EntryRange range_z = ComputeEntryRange(coord.z, cell_width_log2, spacing, factors.count_z);
  constexpr int kNoCode = 1000;
  int smallest = kNoCode;
  for (std::int64_t z = range_z.first; z <= range_z.last; ++z) {
    for (std::int64_t y = range_y.first; y <= range_y.last; ++y) {
      for (std::int64_t x = range_x.first; x <= range_x.last; ++x) {
        const auto index = static_cast<std::size_t>((z * factors.count_y + y) * factors.count_x + x);
        smallest = std::min<int>(smallest, volume.layers[0][index]);
      }
    }
  }
  const bool reaches_outside = range_x.reaches_outside || range_y.reaches_outside || range_z.reaches_outside;
  if (smallest == kNoCode || reaches_outside) {
    smallest = std::min(smallest == kNoCode ? 0 : smallest, 0);
  }
  return std::clamp(smallest, -kMaxCellSizeFactorCode, kMaxCellSizeFactorCode);
}

// Fills layer 0 with codes from -30 to 30 drawn from a fixed seed, so some are beyond the table's limits.
// clang-tidy: a count and a seed; the swappable-parameters check flags the two integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<std::int8_t> BuildDrawnCodes(std::size_t code_count, std::uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::vector<std::int8_t> codes(code_count);
  for (std::int8_t& code : codes) {
    code = static_cast<std::int8_t>(static_cast<int>(generator() % 61U) - 30);
  }
  return codes;
}

// The number of cells of one size level along an axis of `count` entries, one cell past the volume at each end.
// clang-tidy: two integers of different meaning; the swappable-parameters check flags them.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int ComputeCoordEnd(const RefinementFactors& factors, int count, int cell_width_log2) {
  const std::int64_t smallest_cells = static_cast<std::int64_t>(count) * ComputePowerOfTwo(factors.entry_spacing_log2);
  return static_cast<int>(smallest_cells / ComputePowerOfTwo(cell_width_log2)) + 2;
}

// Compares every cell of one size level, with coordinates from -1 to one past the volume, against brute force. Returns
// how many cells gave a code other than 0.
int ComputeNonzeroCodeCountOfSizeLevel(
    const TestRefinementVolume& volume,
    SizeLevel smallest_size_level,
    int size_level
) {
  const RefinementFactors& factors = volume.factors;
  const int cell_width_log2 = static_cast<int>(smallest_size_level) - size_level;
  int nonzero = 0;
  for (int z = -1; z <= ComputeCoordEnd(factors, factors.count_z, cell_width_log2); ++z) {
    for (int y = -1; y <= ComputeCoordEnd(factors, factors.count_y, cell_width_log2); ++y) {
      for (int x = -1; x <= ComputeCoordEnd(factors, factors.count_x, cell_width_log2); ++x) {
        const CellId cell = ToTestCell(size_level, x, y, z);
        const int expected = ComputeBruteForceCode(volume, smallest_size_level, cell);
        const int code = ComputeCellSizeFactorCode(factors, smallest_size_level, cell);
        EXPECT_EQ(code, expected) << "size level " << size_level << " cell {" << x << ", " << y << ", " << z << "}";
        nonzero += code != 0 ? 1 : 0;
      }
    }
  }
  return nonzero;
}

// Every size level from 0 to smallest_size_level against brute force, and enough nonzero codes to mean something.
void CheckEveryCellAgainstBruteForce(const TestRefinementVolume& volume, SizeLevel smallest_size_level) {
  int nonzero = 0;
  for (int size_level = 0; size_level <= static_cast<int>(smallest_size_level); ++size_level) {
    nonzero += ComputeNonzeroCodeCountOfSizeLevel(volume, smallest_size_level, size_level);
  }
  EXPECT_GT(nonzero, 10);
}

TEST(ComputeCellSizeFactorCode, EverySizeLevelOverTheFixture_MatchesTheBruteForceSmallestCode) {
  // 8 x 4 x 4 entries.
  const std::vector<std::int8_t> codes = BuildDrawnCodes(128U, 11);
  const TestRefinementVolume volume =
      BuildTestRefinementFactors(kFixtureCountX, kFixtureCountY, kFixtureCountZ, kFixtureSpacing, 3, codes);
  CheckEveryCellAgainstBruteForce(volume, kSmallestSizeLevel);
}

TEST(ComputeCellSizeFactorCode, SpacingZero_MatchesTheBruteForceSmallestCode) {
  // 4 x 2 x 2.
  const std::vector<std::int8_t> codes = BuildDrawnCodes(16U, 12);
  const TestRefinementVolume volume = BuildTestRefinementFactors(4, 2, 2, 0, 3, codes);
  CheckEveryCellAgainstBruteForce(volume, SizeLevel{2});
}

TEST(ComputeCellSizeFactorCode, CountsNotMultiplesOfTheLayerFactor_MatchTheBruteForceSmallestCode) {
  // 5 x 3 x 2 entries: layers of 5 x 3 x 2, 3 x 2 x 1, 2 x 1 x 1 and 1 x 1 x 1 entries, each a ceiling.
  const std::vector<std::int8_t> codes = BuildDrawnCodes(30U, 13);
  const TestRefinementVolume volume = BuildTestRefinementFactors(5, 3, 2, 1, 4, codes);
  ASSERT_EQ(volume.layers.size(), 4U);
  ASSERT_EQ(volume.layers[1].size(), 3U * 2U * 1U);
  ASSERT_EQ(volume.layers[2].size(), 2U * 1U * 1U);
  ASSERT_EQ(volume.layers[3].size(), 1U);
  CheckEveryCellAgainstBruteForce(volume, SizeLevel{4});
  // 7 x 1 x 3 entries.
  const std::vector<std::int8_t> odd_codes = BuildDrawnCodes(21U, 14);
  const TestRefinementVolume odd = BuildTestRefinementFactors(7, 1, 3, 0, 4, odd_codes);
  CheckEveryCellAgainstBruteForce(odd, SizeLevel{3});
}

TEST(ComputeCellSizeFactorCode, CountsThatAreMultiplesOfTheLayerFactor_MatchTheBruteForceSmallestCode) {
  // 4 x 4 x 4 entries: layers of 4 x 4 x 4, 2 x 2 x 2 and 1 x 1 x 1 entries; no box reaches outside.
  const std::vector<std::int8_t> codes = BuildDrawnCodes(64U, 15);
  const TestRefinementVolume volume = BuildTestRefinementFactors(4, 4, 4, 0, 3, codes);
  CheckEveryCellAgainstBruteForce(volume, SizeLevel{2});
}

TEST(ComputeCellSizeFactorCode, LastEntryOfAnOddCount_IsRead) {
  // 5 entries on x, all code -8 except entry 4, the last, which is -9: layer 1 has 3 entries on x.
  std::vector<std::int8_t> codes(5, -8);
  codes[4] = -9;
  const TestRefinementVolume volume = BuildTestRefinementFactors(5, 1, 1, 0, 3, codes);
  EXPECT_EQ(ComputeCellSizeFactorCode(volume.factors, SizeLevel{1}, ToTestCell(0, 2, 0, 0)), -9);
  EXPECT_EQ(ComputeCellSizeFactorCode(volume.factors, SizeLevel{0}, ToTestCell(0, 4, 0, 0)), -9);
}

TEST(ComputeCellSizeFactorCode, EntryBeyondAnOddCount_IsZero) {
  std::vector<std::int8_t> codes(5, -8);
  const TestRefinementVolume volume = BuildTestRefinementFactors(5, 1, 1, 0, 3, codes);
  EXPECT_EQ(ComputeCellSizeFactorCode(volume.factors, SizeLevel{1}, ToTestCell(0, 3, 0, 0)), 0);
  EXPECT_EQ(ComputeCellSizeFactorCode(volume.factors, SizeLevel{0}, ToTestCell(0, 5, 0, 0)), 0);
}

// The code, with every entry `code` and spacing 0, of the size-level-`size_level` cell at coordinate `coord` on `axis`
// (0 for x, 1 for y, 2 for z) and 0 on the other axes, in a volume of `count` entries on that axis and 8 on the others
// (so the others never reach outside), with `layer_count` layers and smallest size level `smallest`. clang-tidy: six
// integers of different meaning; the swappable-parameters check flags them.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
int ComputeEdgeCode(int axis, int count, int layer_count, int smallest, int size_level, int coord, std::int8_t code) {
  const std::array<int, 3> counts = {axis == 0 ? count : 8, axis == 1 ? count : 8, axis == 2 ? count : 8};
  const std::vector<std::int8_t> codes(static_cast<std::size_t>(counts[0]) * counts[1] * counts[2], code);
  const TestRefinementVolume volume =
      BuildTestRefinementFactors(counts[0], counts[1], counts[2], 0, layer_count, codes);
  const int x = axis == 0 ? coord : 0;
  const int y = axis == 1 ? coord : 0;
  const int z = axis == 2 ? coord : 0;
  return ComputeCellSizeFactorCode(volume.factors, static_cast<SizeLevel>(smallest), ToTestCell(size_level, x, y, z));
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// The next tests: 5 entries on one axis, 4 layers (5, 3, 2, 1), smallest size level 3, spacing 0, code +3. Layer
// L is read by the size-level-(3 - L) cell. Layer 1 index 2 covers entries 4..5; layer 2 index 1 covers 4..7; layer 3
// index 0 covers 0..7.
TEST(ComputeCellSizeFactorCode, PositiveCodeAtTheLastEntryOfLayerZero_IsRead) {
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(ComputeEdgeCode(axis, 5, 4, 3, 3, 4, 3), 3) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, PositiveCodeBeyondTheLastEntryOfLayerZero_IsZero) {
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(ComputeEdgeCode(axis, 5, 4, 3, 3, 5, 3), 0) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, PositiveCodeOfLayerOneInsideTheVolume_IsRead) {
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(ComputeEdgeCode(axis, 5, 4, 3, 2, 1, 3), 3) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, PositiveCodeOfLayerOneReachingOutside_IsZero) {
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(ComputeEdgeCode(axis, 5, 4, 3, 2, 2, 3), 0) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, PositiveCodeOneOfACellReachingOutside_IsZero) {
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(ComputeEdgeCode(axis, 5, 4, 3, 2, 2, 1), 0) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, PositiveCodeOfLayerTwoInsideTheVolume_IsRead) {
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(ComputeEdgeCode(axis, 5, 4, 3, 1, 0, 3), 3) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, PositiveCodeOfLayerTwoReachingOutside_IsZero) {
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(ComputeEdgeCode(axis, 5, 4, 3, 1, 1, 3), 0) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, PositiveCodeOfLayerThreeReachingOutside_IsZero) {
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(ComputeEdgeCode(axis, 5, 4, 3, 0, 0, 3), 0) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, NegativeCodeAtTheHighEdge_IsKeptForACellThatReachesOutside) {
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(ComputeEdgeCode(axis, 5, 4, 3, 2, 2, -3), -3) << axis;
    EXPECT_EQ(ComputeEdgeCode(axis, 5, 4, 3, 0, 0, -3), -3) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, PositiveCodeWithCountsThatAreMultiples_IsKeptWithoutAStraddle) {
  // 4 entries, 3 layers (4, 2, 1), smallest size level 2: no box reaches outside, so every code is read as stored.
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(ComputeEdgeCode(axis, 4, 3, 2, 2, 3, 3), 3) << axis;  // layer 0, last entry.
    EXPECT_EQ(ComputeEdgeCode(axis, 4, 3, 2, 1, 1, 3), 3) << axis;  // layer 1 index 1: entries 2..3.
    EXPECT_EQ(ComputeEdgeCode(axis, 4, 3, 2, 0, 0, 3), 3) << axis;  // layer 2 index 0: entries 0..3.
  }
}

}  // namespace
}  // namespace fluidity::grid
