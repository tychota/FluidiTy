#include "fluidity/grid/testing/grid_piece_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>

#include "fluidity/grid/grid_ids.h"

namespace fluidity::grid {
namespace {

constexpr CellBlockRange kRange = {.first_block = 1000, .block_count = 70000};

TEST(FindEntryDifferences, EqualTables_GiveNothing) {
  const std::array<std::uint32_t, 4> entries = {1, 2, 3, 4};
  const EntryTablePlaces places = {.name = "map", .places_per_block = 1, .thread_count = 3};
  EXPECT_FALSE(FindEntryDifferences(
                   std::span<const std::uint32_t>(entries),
                   std::span<const std::uint32_t>(entries),
                   places,
                   kRange
  )
                   .has_value());
}

// Entries 1 and 3 of 5 differ: 2 places per block and 4 threads put them at block 0 place 1 thread 1, and block 1
// place 1 thread 3. The values 0x2 and 0xff show in hexadecimal.
TEST(FindEntryDifferences, TwoDifferences_ReportCountFirstLastPlaces) {
  const std::array<std::uint32_t, 5> on_cpu = {7, 2, 7, 255, 7};
  const std::array<std::uint32_t, 5> on_gpu = {7, 9, 7, 16, 7};
  const EntryTablePlaces places = {.name = "neighbours", .places_per_block = 2, .thread_count = 4};
  const std::optional<std::string> difference = FindEntryDifferences(
      std::span<const std::uint32_t>(on_cpu),
      std::span<const std::uint32_t>(on_gpu),
      places,
      kRange
  );
  ASSERT_TRUE(difference.has_value());
  EXPECT_EQ(
      *difference,
      "neighbours (range from block 1000, 70000 blocks): 2 of 5 entries differ; first index 1 (block 0, place 1, "
      "thread 1: CPU 0x2, GPU 0x9); last index 3 (block 1, place 1, thread 3: CPU 0xff, GPU 0x10)"
  );
}

TEST(FindEntryDifferences, OneDifferenceAboveTheThreadCount_ShowsTheThreadItWrapsTo) {
  std::array<std::uint8_t, 10> on_cpu{};
  std::array<std::uint8_t, 10> on_gpu{};
  on_gpu[9] = 1;
  const EntryTablePlaces places = {.name = "cell states", .places_per_block = 5, .thread_count = 4};
  const std::optional<std::string> difference = FindEntryDifferences(
      std::span<const std::uint8_t>(on_cpu),
      std::span<const std::uint8_t>(on_gpu),
      places,
      kRange
  );
  ASSERT_TRUE(difference.has_value());
  EXPECT_NE(
      difference->find(
          "1 of 10 entries differ; first index 9 (block 1, place 4, thread 1: CPU 0x0, GPU 0x1); last index 9"
      ),
      std::string::npos
  ) << *difference;
}

TEST(FindEntryDifferences, TablesOfDifferentSizes_NameBothSizes) {
  const std::array<std::uint32_t, 3> on_cpu = {1, 2, 3};
  const std::array<std::uint32_t, 2> on_gpu = {1, 2};
  const EntryTablePlaces places = {.name = "map", .places_per_block = 1, .thread_count = 1};
  const std::optional<std::string> difference = FindEntryDifferences(
      std::span<const std::uint32_t>(on_cpu),
      std::span<const std::uint32_t>(on_gpu),
      places,
      kRange
  );
  ASSERT_TRUE(difference.has_value());
  EXPECT_NE(difference->find("the CPU table has 3 entries and the GPU table 2"), std::string::npos) << *difference;
}

}  // namespace
}  // namespace fluidity::grid
