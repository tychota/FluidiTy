#include "fluidity/grid/grid_piece_steps.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "fluidity/grid/grid_ids.h"

namespace fluidity::grid {
namespace {

// The worked example of the header: the whole grid of blocks 0..9, the range {4, 3} and the border {1, 2, 8}.
constexpr std::array<CellBlockIndex, 3> kBorderBlocks = {{{.value = 1}, {.value = 2}, {.value = 8}}};
constexpr CellBlockRange kRange = {.first_block = 4, .block_count = 3};

TEST(ToPiecePlace, RangeBetweenBorderEntries_GivesAscendingMap) {
  const std::array<std::uint32_t, 6> expected_places = {0, 1, 5, 2, 3, 4};
  std::array<std::uint32_t, 6> map = {};
  for (std::uint32_t loaded_item = 0; loaded_item < 6; ++loaded_item) {
    const CellBlockIndex place = ToPiecePlace(kBorderBlocks.data(), 3, kRange, loaded_item);
    EXPECT_EQ(place.value, expected_places.at(loaded_item)) << "loaded item " << loaded_item;
    const std::uint32_t whole_block = loaded_item < 3 ? kBorderBlocks.at(loaded_item).value : 4 + (loaded_item - 3);
    map.at(place.value) = whole_block;
  }
  EXPECT_EQ(map, (std::array<std::uint32_t, 6>{1, 2, 4, 5, 6, 8}));
}

TEST(ToPiecePlace, NoBorder_GivesTheRangeFromPlaceZero) {
  EXPECT_EQ(ToPiecePlace(nullptr, 0, kRange, 0).value, 0U);
  EXPECT_EQ(ToPiecePlace(nullptr, 0, kRange, 2).value, 2U);
}

TEST(ToPieceNeighbour, BlockNotLoaded_IsNoIndex) {
  const std::array<CellBlockIndex, 6> map = {{{1}, {2}, {4}, {5}, {6}, {8}}};
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 7), kNoIndex);
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 0), kNoIndex);
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 9), kNoIndex);
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, kNoIndex), kNoIndex);
}

TEST(ToPieceNeighbour, BlockLoaded_IsItsPlaceInTheMap) {
  const std::array<CellBlockIndex, 6> map = {{{1}, {2}, {4}, {5}, {6}, {8}}};
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 1), 0U);
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 5), 3U);
  EXPECT_EQ(ToPieceNeighbour(map.data(), 6, 8), 5U);
}

}  // namespace
}  // namespace fluidity::grid
