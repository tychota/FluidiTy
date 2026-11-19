#include "fluidity/grid/grid_error.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string_view>
#include <type_traits>

#include "fluidity/grid/grid_error_message.h"

namespace fluidity::grid {
namespace {

struct ErrorNameEntry {
  GridError error;
  std::string_view name;
  std::string_view message_phrase;  // A phrase only this error's message contains: a swap of two messages fails.
};

// The enumerator spelling without the k, in snake case, typed by hand: the test checks the table, not a formula.
constexpr std::array kErrorNames = {
    ErrorNameEntry{
        .error = GridError::kNone,
        .name = "none",
        .message_phrase = "no grid error",
    },
    ErrorNameEntry{
        .error = GridError::kBadInput,
        .name = "bad_input",
        .message_phrase = "finite positive number",
    },
    ErrorNameEntry{
        .error = GridError::kCoordOutOfRange,
        .name = "coord_out_of_range",
        .message_phrase = "too far from the grid origin",
    },
    ErrorNameEntry{
        .error = GridError::kCellsOverlap,
        .name = "cells_overlap",
        .message_phrase = "overlap",
    },
    ErrorNameEntry{
        .error = GridError::kSiblingCellMissing,
        .name = "sibling_cell_missing",
        .message_phrase = "8 small cells is missing",
    },
    ErrorNameEntry{
        .error = GridError::kSizeStepRuleBroken,
        .name = "size_step_rule_broken",
        .message_phrase = "more than one size level",
    },
    ErrorNameEntry{
        .error = GridError::kTooManyCellBlocks,
        .name = "too_many_cell_blocks",
        .message_phrase = "more cell blocks than the limit",
    },
    ErrorNameEntry{
        .error = GridError::kRunOnCallFailed,
        .name = "run_on_call_failed",
        .message_phrase = "cannot run here",
    },
    ErrorNameEntry{
        .error = GridError::kWrongGridVersion,
        .name = "wrong_grid_version",
        .message_phrase = "another grid version",
    },
    ErrorNameEntry{
        .error = GridError::kPositionOutsideGrid,
        .name = "position_outside_grid",
        .message_phrase = "No active cell contains",
    },
    ErrorNameEntry{
        .error = GridError::kInternalFault,
        .name = "internal_fault",
        .message_phrase = "fault in FluidiTy, not in the scene",
    },
    ErrorNameEntry{
        .error = GridError::kOutOfMemory,
        .name = "out_of_memory",
        .message_phrase = "not enough memory",
    },
};

// The table holds one entry for each value up to kOutOfMemory, the last value. A value added before kOutOfMemory moves
// it and fails here. A value added after it leaves these asserts true; the test below that tries every byte fails
// when that value has a name, and -Wswitch fails the build when it has no case in grid_error_message.cc.
static_assert(static_cast<std::size_t>(GridError::kOutOfMemory) + 1 == kErrorNames.size());
static_assert(kErrorNames.size() == 12);

TEST(ToName, EveryError_IsStableSnakeCase) {
  std::set<std::string_view> names;
  std::uint8_t expected_value = 0;
  for (const ErrorNameEntry& entry : kErrorNames) {
    EXPECT_EQ(static_cast<std::uint8_t>(entry.error), expected_value) << "the table follows the enumerator order";
    EXPECT_EQ(ToName(entry.error), entry.name);
    names.insert(ToName(entry.error));
    ++expected_value;
  }
  EXPECT_EQ(names.size(), kErrorNames.size());
  EXPECT_EQ(ToName(GridError::kTooManyCellBlocks), "too_many_cell_blocks");
  EXPECT_EQ(ToName(GridError::kNone), "none");
}

TEST(ToName, OutOfRangeError_IsUnknown) {
  EXPECT_EQ(ToName(static_cast<GridError>(255)), "unknown");
}

TEST(ToMessage, EveryError_HasADistinctNonEmptyMessage) {
  std::set<std::string_view> messages;
  for (const ErrorNameEntry& entry : kErrorNames) {
    EXPECT_FALSE(ToMessage(entry.error).empty()) << entry.name;
    messages.insert(ToMessage(entry.error));
  }
  EXPECT_EQ(messages.size(), kErrorNames.size());
}

TEST(ToMessage, EveryError_HasItsOwnPhrase) {
  for (const ErrorNameEntry& entry : kErrorNames) {
    EXPECT_NE(ToMessage(entry.error).find(entry.message_phrase), std::string_view::npos) << entry.name;
  }
}

TEST(ToName, EveryNamedValue_IsInTheTable) {
  for (int number = 0; number <= 255; ++number) {
    const auto error = static_cast<GridError>(number);
    if (ToName(error) == "unknown") {
      continue;
    }
    EXPECT_LT(static_cast<std::size_t>(number), kErrorNames.size()) << ToName(error) << " is missing from kErrorNames";
  }
}

TEST(ToMessage, TooManyCellBlocks_SaysWhatTheUserCanDo) {
  EXPECT_EQ(
      ToMessage(GridError::kTooManyCellBlocks),
      "The grid needs more cell blocks than the limit. Raise max_cell_blocks or use fewer small cells."
  );
}

TEST(ToMessage, InternalFault_AsksForTheScene) {
  EXPECT_EQ(
      ToMessage(GridError::kInternalFault),
      "The grid code reached a state it should never reach. This is a fault in FluidiTy, not in the scene. Please "
      "report it and send the scene."
  );
}

TEST(ToMessage, OutOfRangeError_HasItsOwnNonEmptyMessage) {
  const std::string_view unknown_message = ToMessage(static_cast<GridError>(255));
  EXPECT_FALSE(unknown_message.empty());
  for (const ErrorNameEntry& entry : kErrorNames) {
    EXPECT_NE(unknown_message, ToMessage(entry.error)) << entry.name;
  }
}

TEST(GridError, OutOfMemory_IsTheLastValue) {
  // No byte after kOutOfMemory has a name.
  for (int number = static_cast<int>(GridError::kOutOfMemory) + 1; number <= 255; ++number) {
    EXPECT_EQ(ToName(static_cast<GridError>(number)), "unknown") << number;
  }
  EXPECT_EQ(static_cast<int>(GridError::kOutOfMemory), 11);
  EXPECT_EQ(ToName(GridError::kOutOfMemory), "out_of_memory");
  EXPECT_EQ(GridFailure{}.magnitude, 0U);
}

TEST(GridFailure, Default_IsNoneWithZeroMagnitude) {
  const GridFailure failure;
  EXPECT_EQ(failure.error, GridError::kNone);
  EXPECT_EQ(failure.magnitude, 0U);
}

TEST(GridFailure, Default_NamesNoCell) {
  // CellId{0} is never a valid id, so a default failure names no cell.
  const GridFailure failure;
  EXPECT_EQ(failure.cell.value, 0U);
  EXPECT_EQ(failure.other_cell.value, 0U);
}

TEST(GridFailure, IsTriviallyCopyable) {
  EXPECT_TRUE(std::is_trivially_copyable_v<GridFailure>);
}

TEST(GridFailure, FieldOrder_IsErrorCellsThenMagnitude) {
  // The error, padded to 8 bytes, the two cell ids and the magnitude: the CPU and the GPU write one layout.
  EXPECT_EQ(offsetof(GridFailure, error), 0U);
  EXPECT_EQ(offsetof(GridFailure, cell), 8U);
  EXPECT_EQ(offsetof(GridFailure, other_cell), 16U);
  EXPECT_EQ(offsetof(GridFailure, magnitude), 24U);
  EXPECT_EQ(sizeof(GridFailure), 32U);
}

}  // namespace
}  // namespace fluidity::grid
