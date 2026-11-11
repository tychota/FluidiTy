#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>

#include "fluidity/host/effects_cache_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

EffectsIdentity3D Identity() {
  EffectsIdentity3D identity{.stage = EffectsStage3D::kSource};
  identity.upstream_execution[0] = 1;
  identity.settings[0] = 2;
  identity.build[0] = 3;
  return identity;
}

TEST(EffectsCache3D, ExecutionIdentityBindsStageSettingsBuildAndUpstreamWithoutPaths) {
  const auto identity = Identity();
  const auto original = EffectsExecutionId3D(identity);
  ASSERT_TRUE(original.ok());
  EXPECT_EQ(EffectsExecutionId3D(identity).value(), *original);
  for (int field = 0; field < 4; ++field) {
    auto changed = identity;
    if (field == 0) {
      changed.stage = EffectsStage3D::kMesh;
    } else if (field == 1) {
      ++changed.settings[0];
    } else if (field == 2) {
      ++changed.build[0];
    } else {
      ++changed.upstream_execution[0];
    }
    auto digest = EffectsExecutionId3D(changed);
    ASSERT_TRUE(digest.ok());
    EXPECT_NE(*digest, *original);
  }
  EXPECT_EQ(EffectsExecutionId3D({}).status().code(), absl::StatusCode::kInvalidArgument);
}

std::filesystem::path Scratch() {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::path(FLUIDITY_CACHE_SCRATCH_PATH) / ("effects-" + std::to_string(stamp));
}

TEST(EffectsCache3D, CommitsExactContentAndRejectsConflictingReplay) {
  const auto root = Scratch();
  const auto cache = EffectsCache3D::Create(root, Identity());
  ASSERT_TRUE(cache.ok()) << cache.status();
  constexpr std::array<std::uint8_t, 4> payload{1, 2, 3, 4};
  LiquidDigest3D upstream{};
  upstream[0] = 4;
  ASSERT_TRUE(cache->Publish({}, upstream, {}, payload).ok());
  const auto read_only = EffectsCache3D::OpenForInspection(root);
  ASSERT_TRUE(read_only.ok()) << read_only.status();
  const auto committed = read_only->ReadFrame(0);
  ASSERT_TRUE(committed.ok()) << committed.status();
  EXPECT_EQ(committed->identity, Identity());
  EXPECT_EQ(committed->upstream_content, upstream);
  EXPECT_TRUE(std::ranges::equal(committed->payload.values(), payload));
  EXPECT_EQ(read_only->Publish({}, upstream, {}, payload).code(), absl::StatusCode::kFailedPrecondition);
  const auto timestamp = std::filesystem::last_write_time(root / "frames/00000000000000000000.flem");
  ASSERT_TRUE(cache->Publish({}, upstream, {}, payload).ok());
  EXPECT_EQ(timestamp, std::filesystem::last_write_time(root / "frames/00000000000000000000.flem"));
  constexpr std::array<std::uint8_t, 4> changed{1, 2, 3, 5};
  EXPECT_EQ(cache->Publish({}, upstream, {}, changed).code(), absl::StatusCode::kAlreadyExists);
  const LiquidClock3D clock{.time = 1.0 / 24, .frame_ordinal = 1, .accepted_step_ordinal = 1, .epoch = 1};
  EXPECT_EQ(cache->Publish(clock, upstream, {}, payload).code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(cache->Publish(clock, upstream, committed->payload_digest, payload).ok());
  auto incompatible = Identity();
  incompatible.settings[0] += 1;
  EXPECT_EQ(EffectsCache3D::Create(root, incompatible).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(EffectsCache3D, DerivedSettingsAreImmutableAuthenticatedAndReadOnlyOnInspection) {
  const auto root = Scratch();
  constexpr std::array<std::uint8_t, 4> settings{1, 2, 3, 4};
  auto identity = Identity();
  identity.stage = EffectsStage3D::kMesh;
  identity.settings = LiquidContentDigest3D(settings);
  auto cache = EffectsCache3D::Create(root, identity);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(cache->PublishStageSettings(settings).ok());
  const auto timestamp = std::filesystem::last_write_time(root / "settings.flec");
  ASSERT_TRUE(cache->PublishStageSettings(settings).ok());
  EXPECT_EQ(std::filesystem::last_write_time(root / "settings.flec"), timestamp);
  const auto read = cache->ReadStageSettings();
  ASSERT_TRUE(read.ok());
  EXPECT_TRUE(std::ranges::equal(read->values(), settings));
  constexpr std::array<std::uint8_t, 4> changed{4, 3, 2, 1};
  EXPECT_EQ(cache->PublishStageSettings(changed).code(), absl::StatusCode::kFailedPrecondition);
  auto inspection = EffectsCache3D::OpenForInspection(root);
  ASSERT_TRUE(inspection.ok());
  EXPECT_EQ(inspection->PublishStageSettings(settings).code(), absl::StatusCode::kFailedPrecondition);
  std::ofstream stream(root / "settings.flec", std::ios::binary | std::ios::trunc);
  stream << "corrupt";
  stream.close();
  EXPECT_EQ(inspection->ReadStageSettings().status().code(), absl::StatusCode::kDataLoss);
}

TEST(EffectsCache3D, InterruptedManifestAndCorruptionFailClosed) {
  const auto root = Scratch();
  const auto cache = EffectsCache3D::Create(root, Identity());
  ASSERT_TRUE(cache.ok()) << cache.status();
  constexpr std::array<std::uint8_t, 3> payload{1, 2, 3};
  LiquidDigest3D upstream{};
  upstream[0] = 4;
  const LiquidCachePublishControl3D stop{
      .check =
          [](void*, LiquidCachePublishStage3D stage) {
            return stage == LiquidCachePublishStage3D::kBeforeManifest ? absl::CancelledError("injected")
                                                                       : absl::OkStatus();
          },
  };
  EXPECT_EQ(cache->Publish({}, upstream, {}, payload, stop).code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(cache->ReadFrame(0).status().code(), absl::StatusCode::kNotFound);
  ASSERT_TRUE(cache->Publish({}, upstream, {}, payload).ok());
  const auto commit = cache->ReadFrame(0);
  ASSERT_TRUE(commit.ok()) << commit.status();
  const auto hex = LiquidDigestHex3D(commit->payload_digest);
  std::ofstream corrupt(root / "blobs" / (std::string(hex.data()) + ".flef"), std::ios::binary);
  corrupt << "corrupt";
  corrupt.close();
  EXPECT_EQ(cache->ReadFrame(0).status().code(), absl::StatusCode::kDataLoss);
}

}  // namespace
}  // namespace fluidity
