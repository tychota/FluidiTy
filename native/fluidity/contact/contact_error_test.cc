#include "fluidity/contact/contact_error.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <set>
#include <string_view>
#include <type_traits>

#include "fluidity/contact/contact_error_message.h"

namespace fluidity::contact {
namespace {

struct ErrorNameEntry {
  ContactError error;
  std::string_view name;
};

struct CloseCallNameEntry {
  ContactCloseCall close_call;
  std::string_view name;
};

// The enumerator spelling without the k, in snake case, typed by hand: the test checks the table, not a formula.
constexpr std::array kErrorNames = {
    ErrorNameEntry{.error = ContactError::kNone, .name = "none"},
    ErrorNameEntry{.error = ContactError::kBadInput, .name = "bad_input"},
    ErrorNameEntry{.error = ContactError::kStartsInsideSolid, .name = "starts_inside_solid"},
    ErrorNameEntry{.error = ContactError::kTooCloseToCall, .name = "too_close_to_call"},
    ErrorNameEntry{.error = ContactError::kOnWrongSideOfPlate, .name = "on_wrong_side_of_plate"},
    ErrorNameEntry{.error = ContactError::kSolidTooThin, .name = "solid_too_thin"},
    ErrorNameEntry{.error = ContactError::kBuffersTooSmall, .name = "buffers_too_small"},
    ErrorNameEntry{.error = ContactError::kRunOnCallFailed, .name = "run_on_call_failed"},
    ErrorNameEntry{.error = ContactError::kOutOfMemory, .name = "out_of_memory"},
};

constexpr std::array kCloseCallNames = {
    CloseCallNameEntry{.close_call = ContactCloseCall::kNone, .name = "none"},
    CloseCallNameEntry{.close_call = ContactCloseCall::kTooManyHits, .name = "too_many_hits"},
    CloseCallNameEntry{.close_call = ContactCloseCall::kSqueezedBetweenSolids, .name = "squeezed_between_solids"},
    CloseCallNameEntry{
        .close_call = ContactCloseCall::kSlowApproachToTurningSolid,
        .name = "slow_approach_to_turning_solid",
    },
    CloseCallNameEntry{
        .close_call = ContactCloseCall::kTurningSolidNearOtherSolid,
        .name = "turning_solid_near_other_solid",
    },
    CloseCallNameEntry{.close_call = ContactCloseCall::kSecondReactingSolid, .name = "second_reacting_solid"},
    CloseCallNameEntry{.close_call = ContactCloseCall::kSolidTurnsTooFar, .name = "solid_turns_too_far"},
    CloseCallNameEntry{.close_call = ContactCloseCall::kEndsTooCloseToSolid, .name = "ends_too_close_to_solid"},
};

// kOutOfMemory is the last value, so it is the count minus one: a value added after it fails here.
static_assert(static_cast<std::uint8_t>(ContactError::kOutOfMemory) == 8);
static_assert(kErrorNames.size() == 9);
static_assert(static_cast<std::uint8_t>(ContactCloseCall::kEndsTooCloseToSolid) == 7);
static_assert(kCloseCallNames.size() == 8);

TEST(ToName, EveryContactError_IsSnakeCaseOfEnumerator) {
  std::set<std::string_view> names;
  std::uint8_t expected_value = 0;
  for (const ErrorNameEntry& entry : kErrorNames) {
    EXPECT_EQ(static_cast<std::uint8_t>(entry.error), expected_value) << "the table follows the enumerator order";
    EXPECT_EQ(ToName(entry.error), entry.name);
    names.insert(ToName(entry.error));
    ++expected_value;
  }
  EXPECT_EQ(names.size(), kErrorNames.size());
}

TEST(ToName, EveryContactCloseCall_IsSnakeCaseOfEnumerator) {
  std::set<std::string_view> names;
  std::uint8_t expected_value = 0;
  for (const CloseCallNameEntry& entry : kCloseCallNames) {
    EXPECT_EQ(static_cast<std::uint8_t>(entry.close_call), expected_value) << "the table follows the enumerator order";
    EXPECT_EQ(ToName(entry.close_call), entry.name);
    names.insert(ToName(entry.close_call));
    ++expected_value;
  }
  EXPECT_EQ(names.size(), kCloseCallNames.size());
}

TEST(ToMessage, EveryContactError_HasADistinctNonEmptySentence) {
  std::set<std::string_view> messages;
  for (const ErrorNameEntry& entry : kErrorNames) {
    EXPECT_FALSE(ToMessage(entry.error).empty()) << entry.name;
    messages.insert(ToMessage(entry.error));
  }
  EXPECT_EQ(messages.size(), kErrorNames.size());
}

TEST(ToMessage, EveryContactCloseCall_HasADistinctNonEmptySentence) {
  std::set<std::string_view> messages;
  for (const CloseCallNameEntry& entry : kCloseCallNames) {
    EXPECT_FALSE(ToMessage(entry.close_call).empty()) << entry.name;
    messages.insert(ToMessage(entry.close_call));
  }
  EXPECT_EQ(messages.size(), kCloseCallNames.size());
}

TEST(ToMessage, SolidTooThin_SaysWhatTheUserCanDo) {
  EXPECT_EQ(
      ToMessage(ContactError::kSolidTooThin),
      "A solid is thinner than 1024 contact tolerances. Make it thicker or move the scene closer to the "
      "origin."
  );
}

TEST(ToName, OutOfRangeValues_AreUnknown) {
  EXPECT_EQ(ToName(static_cast<ContactError>(255)), "unknown");
  EXPECT_EQ(ToName(static_cast<ContactCloseCall>(255)), "unknown");
}

TEST(ToMessage, OutOfRangeContactError_HasItsOwnNonEmptySentence) {
  const std::string_view unknown_message = ToMessage(static_cast<ContactError>(255));
  EXPECT_FALSE(unknown_message.empty());
  for (const ErrorNameEntry& entry : kErrorNames) {
    EXPECT_NE(unknown_message, ToMessage(entry.error)) << entry.name;
  }
}

TEST(ToMessage, OutOfRangeContactCloseCall_HasItsOwnNonEmptySentence) {
  const std::string_view unknown_message = ToMessage(static_cast<ContactCloseCall>(255));
  EXPECT_FALSE(unknown_message.empty());
  for (const CloseCallNameEntry& entry : kCloseCallNames) {
    EXPECT_NE(unknown_message, ToMessage(entry.close_call)) << entry.name;
  }
}

TEST(ContactFailure, Default_IsNoneWithZeroContext) {
  const ContactFailure failure;
  EXPECT_EQ(failure.error, ContactError::kNone);
  EXPECT_EQ(failure.close_call, ContactCloseCall::kNone);
  EXPECT_EQ(failure.particle_id, 0U);
  EXPECT_EQ(failure.solid_id, 0U);
  EXPECT_EQ(failure.magnitude, 0.0);
}

TEST(ContactFailure, IsTriviallyCopyable) {
  EXPECT_TRUE(std::is_trivially_copyable_v<ContactFailure>);
}

}  // namespace
}  // namespace fluidity::contact
