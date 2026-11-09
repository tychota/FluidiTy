#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "checkpoint_reference_generated.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
namespace fs = std::filesystem;
class LiquidCacheStore3D : public testing::Test {
 protected:
  void SetUp() override {
    const std::string name = testing::UnitTest::GetInstance()->current_test_info()->name();
    const auto digest = LiquidContentDigest3D({reinterpret_cast<const std::uint8_t*>(name.data()), name.size()});
    root = fs::path(FLUIDITY_CACHE_SCRATCH_PATH) / std::string(LiquidDigestHex3D(digest).data(), 16);
    std::error_code error;
    fs::remove_all(root, error);
    ASSERT_FALSE(error);
    ASSERT_TRUE(fs::create_directories(root, error));
    ASSERT_FALSE(error);
    auto fingerprint = ResolvedSceneFingerprint(scene);
    ASSERT_TRUE(fingerprint.ok());
    resume.resolved_fingerprint = *fingerprint;
    resume.execution_id = LiquidExecutionId3D(*fingerprint);
  }
  void SetFrame(std::uint64_t ordinal) {
    points.clock = {static_cast<double>(ordinal), ordinal, 0, ordinal, ordinal};
    state.clock = points.clock;
  }
  absl::Status Publish(LiquidCache3D& cache, std::uint64_t ordinal, LiquidCachePublishControl3D control = {}) {
    SetFrame(ordinal);
    if (ordinal % 10 == 0) {
      return cache.PublishFrame(points, &state, &resume, control);
    }
    return cache.PublishFrame(points, nullptr, nullptr, control);
  }
  std::string Bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }
  void Write(const fs::path& path, const std::string& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(output.good());
  }
  fs::path FramePath(std::uint64_t ordinal, const char* suffix = ".flmf") {
    auto digits = std::to_string(ordinal);
    return root / "frames" / (std::string(20 - digits.size(), '0') + digits + suffix);
  }
  fs::path root;
  LiquidResolvedSceneView3D scene;
  LiquidResumeMetadata3D resume;
  LiquidPointFrame3D points;
  LiquidState3D state;
};
TEST_F(LiquidCacheStore3D, PublishesInitialFrameResolvedInputsAndPortableInspection) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok()) << cache.status();
  ASSERT_TRUE(Publish(*cache, 0).ok());
  auto frame = cache->ReadCommittedFrame(0);
  ASSERT_TRUE(frame.ok()) << frame.status();
  EXPECT_EQ(frame->points.clock, points.clock);
  auto checkpoint = cache->ReadCheckpoint();
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  EXPECT_EQ(checkpoint->state, state);
  EXPECT_EQ(checkpoint->resume, resume);
  auto resolved = cache->ReadResolvedScene();
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(resolved->random_seed, 0);
  auto json = Bytes(FramePath(0, ".json"));
  EXPECT_NE(json.find("\"status\":\"complete\""), std::string::npos);
  EXPECT_NE(json.find("\"frame\":0"), std::string::npos);
  EXPECT_NE(json.find("\"time_seconds\":0"), std::string::npos);
  EXPECT_NE(json.find("\"relative_name\":\"blobs/"), std::string::npos);
  EXPECT_EQ(json.find(root.string()), std::string::npos);
  EXPECT_EQ(json.find("heartbeat"), std::string::npos);
  EXPECT_EQ(json.find("process"), std::string::npos);
  EXPECT_TRUE(fs::exists(FramePath(0)));
}
TEST_F(LiquidCacheStore3D, IdenticalReplayPreservesBytesAndModificationTimes) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  std::vector<std::pair<fs::path, fs::file_time_type>> times;
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file() && entry.path().filename() != ".writer.lock") {
      times.emplace_back(entry.path(), entry.last_write_time());
    }
  }
  auto manifest = Bytes(FramePath(0));
  ASSERT_TRUE(Publish(*cache, 0).ok());
  EXPECT_EQ(Bytes(FramePath(0)), manifest);
  for (const auto& [path, time] : times) {
    EXPECT_EQ(fs::last_write_time(path), time);
  }
}
TEST_F(LiquidCacheStore3D, ConflictingFrameCannotOverwriteCommit) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  ASSERT_TRUE(Publish(*cache, 1).ok());
  auto before = Bytes(FramePath(1));
  points.clock.time = 1.5;
  EXPECT_EQ(cache->PublishFrame(points).code(), absl::StatusCode::kAlreadyExists);
  EXPECT_EQ(Bytes(FramePath(1)), before);
  EXPECT_TRUE(cache->ReadCommittedFrame(0).ok());
}
TEST_F(LiquidCacheStore3D, CorruptReferencedContentAndInspectionFailClosed) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  auto frame = cache->ReadCommittedFrame(0);
  ASSERT_TRUE(frame.ok());
  auto name = LiquidContentName3D(frame->manifest.points);
  ASSERT_TRUE(name.ok());
  auto path = root / name->data();
  auto original = Bytes(path);
  auto corrupt = original;
  corrupt.back() ^= 1;
  Write(path, corrupt);
  EXPECT_EQ(cache->ReadCommittedFrame(0).status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(Publish(*cache, 0).code(), absl::StatusCode::kDataLoss);
  Write(path, original + "x");
  EXPECT_EQ(cache->ReadCommittedFrame(0).status().code(), absl::StatusCode::kDataLoss);
  Write(path, original);
  Write(FramePath(0, ".json"), "{}");
  EXPECT_EQ(cache->ReadCommittedFrame(0).status().code(), absl::StatusCode::kDataLoss);
}
TEST_F(LiquidCacheStore3D, RejectsJobClockCheckpointAndPathMismatches) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  EXPECT_EQ(cache->PublishFrame(points).code(), absl::StatusCode::kInvalidArgument);
  auto bad_resume = resume;
  bad_resume.resolved_fingerprint[0] ^= 1;
  bad_resume.execution_id = LiquidExecutionId3D(bad_resume.resolved_fingerprint);
  EXPECT_EQ(cache->PublishFrame(points, &state, &bad_resume).code(), absl::StatusCode::kFailedPrecondition);
  state.clock.time = 1;
  EXPECT_EQ(cache->PublishFrame(points, &state, &resume).code(), absl::StatusCode::kInvalidArgument);
  SetFrame(0);
  state.random_seed = 1;
  EXPECT_EQ(cache->PublishFrame(points, &state, &resume).code(), absl::StatusCode::kInvalidArgument);
  auto wrong_fingerprint = resume.resolved_fingerprint;
  wrong_fingerprint[0] ^= 1;
  EXPECT_EQ(LiquidCache3D::Open(root, wrong_fingerprint).status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(LiquidCache3D::Create({}, scene).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      LiquidCache3D::Create(fs::path(std::string("cache\0escape", 12)), scene).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}
struct StopAt {
  LiquidCachePublishStage3D stage;
  static absl::Status Check(void* context, LiquidCachePublishStage3D stage) {
    return stage == static_cast<StopAt*>(context)->stage ? absl::CancelledError("interrupted") : absl::OkStatus();
  }
};
TEST_F(LiquidCacheStore3D, InterruptedStagesPreserveEarlierCommitsAndAllowExactRetry) {
  constexpr LiquidCachePublishStage3D stages[] = {
      LiquidCachePublishStage3D::kTemporaryWritten,
      LiquidCachePublishStage3D::kTemporaryFlushed,
      LiquidCachePublishStage3D::kContentPublished,
      LiquidCachePublishStage3D::kInspectionPublished,
      LiquidCachePublishStage3D::kBeforeManifest,
      LiquidCachePublishStage3D::kManifestPublished,
      LiquidCachePublishStage3D::kCheckpointPublished,
      LiquidCachePublishStage3D::kBeforePrune,
  };
  for (auto stage : stages) {
    auto directory = root / std::to_string(static_cast<int>(stage));
    auto cache = LiquidCache3D::Create(directory, scene);
    ASSERT_TRUE(cache.ok());
    ASSERT_TRUE(Publish(*cache, 0).ok());
    auto prior = Bytes(directory / "frames/00000000000000000000.flmf");
    StopAt stop{stage};
    EXPECT_EQ(Publish(*cache, 10, {StopAt::Check, &stop}).code(), absl::StatusCode::kCancelled);
    EXPECT_EQ(Bytes(directory / "frames/00000000000000000000.flmf"), prior);
    EXPECT_TRUE(cache->ReadCommittedFrame(0).ok());
    auto reopened = LiquidCache3D::Open(directory, resume.resolved_fingerprint);
    ASSERT_TRUE(reopened.ok());
    auto latest = reopened->ReadCheckpoint();
    ASSERT_TRUE(latest.ok()) << latest.status();
    const bool checkpoint_committed =
        stage == LiquidCachePublishStage3D::kCheckpointPublished || stage == LiquidCachePublishStage3D::kBeforePrune;
    EXPECT_EQ(latest->state.clock.frame_ordinal, checkpoint_committed ? 10 : 0);
    ASSERT_TRUE(Publish(*reopened, 10).ok());
    auto recovered = reopened->ReadCheckpoint();
    ASSERT_TRUE(recovered.ok());
    EXPECT_EQ(recovered->state.clock.frame_ordinal, 10);
  }
}
TEST_F(LiquidCacheStore3D, RetainsTwoCheckpointsWithoutInvalidatingOldPointFrames) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  for (std::uint64_t frame = 0; frame <= 30; ++frame) {
    ASSERT_TRUE(Publish(*cache, frame).ok()) << frame;
  }
  EXPECT_EQ(cache->ReadCheckpoint(19).status().code(), absl::StatusCode::kNotFound);
  auto previous = cache->ReadCheckpoint(29);
  ASSERT_TRUE(previous.ok());
  EXPECT_EQ(previous->state.clock.frame_ordinal, 20);
  auto current = cache->ReadCheckpoint();
  ASSERT_TRUE(current.ok());
  EXPECT_EQ(current->state.clock.frame_ordinal, 30);
  std::size_t reference_count = 0;
  for (const auto& entry : fs::directory_iterator(root / "checkpoints")) {
    if (entry.path().extension() == ".flcr") {
      ++reference_count;
    }
  }
  EXPECT_EQ(reference_count, 4);
  std::size_t state_blob_count = 0;
  for (const auto& entry : fs::directory_iterator(root / "blobs")) {
    if (entry.path().extension() == ".flst") {
      ++state_blob_count;
    }
  }
  EXPECT_EQ(state_blob_count, 2);
  SetFrame(0);
  state.next_particle_id = 2;
  EXPECT_EQ(cache->PublishCheckpoint(state, resume).code(), absl::StatusCode::kAlreadyExists);
  for (std::uint64_t frame = 0; frame <= 30; ++frame) {
    EXPECT_TRUE(cache->ReadCommittedFrame(frame).ok()) << frame;
  }
  auto opened = LiquidCache3D::Open(root, resume.resolved_fingerprint);
  ASSERT_TRUE(opened.ok());
  EXPECT_EQ(opened->ReadCheckpoint()->state.clock.frame_ordinal, 30);
}
TEST_F(LiquidCacheStore3D, ValidOrphanIsNeverResumedAndCanBeReused) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  StopAt stop{LiquidCachePublishStage3D::kBeforeManifest};
  EXPECT_EQ(Publish(*cache, 10, {StopAt::Check, &stop}).code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(cache->ReadCommittedFrame(10).status().code(), absl::StatusCode::kNotFound);
  auto latest = cache->ReadCheckpoint();
  ASSERT_TRUE(latest.ok());
  EXPECT_EQ(latest->state.clock.frame_ordinal, 0);
  ASSERT_TRUE(Publish(*cache, 10).ok());
  EXPECT_EQ(cache->ReadCheckpoint()->state.clock.frame_ordinal, 10);
}
TEST_F(LiquidCacheStore3D, NativeLeaseRejectsCompetingPublisherWithoutOverwriting) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  struct Contender {
    LiquidCache3D* cache;
    LiquidPointFrame3D* points;
    bool checked = false;
    static absl::Status Check(void* context, LiquidCachePublishStage3D stage) {
      auto& self = *static_cast<Contender*>(context);
      if (stage == LiquidCachePublishStage3D::kBeforeManifest) {
        self.checked = true;
        EXPECT_EQ(self.cache->PublishFrame(*self.points).code(), absl::StatusCode::kAborted);
      }
      return absl::OkStatus();
    }
  } contender{&*cache, &points};
  ASSERT_TRUE(Publish(*cache, 1, {Contender::Check, &contender}).ok());
  EXPECT_TRUE(contender.checked);
  EXPECT_TRUE(cache->ReadCommittedFrame(1).ok());
}

TEST_F(LiquidCacheStore3D, CheckpointReferencesResolveToExactPublishedContent) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  auto reference = Bytes(root / "checkpoints/00000000000000000000.flcr");
  const auto* record = cache::v1::GetCheckpointReference(reference.data());
  const auto* manifest = record->frame_manifest();
  const auto manifest_path = root / manifest->relative_name()->str();
  EXPECT_TRUE(fs::exists(manifest_path));
  EXPECT_EQ(Bytes(manifest_path), Bytes(FramePath(0)));
  EXPECT_EQ(Bytes(manifest_path).size(), manifest->byte_length());
}
TEST_F(LiquidCacheStore3D, RequiresInitialFrameBeforePublishingLaterFrames) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  EXPECT_EQ(Publish(*cache, 1).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_FALSE(fs::exists(FramePath(1)));
  EXPECT_EQ(Publish(*cache, 10).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_FALSE(fs::exists(FramePath(10)));
}
TEST_F(LiquidCacheStore3D, EveryIoBoundaryLeavesOnlyVerifiedRestartState) {
  struct StopAfter {
    std::size_t event = 0;
    std::size_t stop = 0;
    static absl::Status Check(void* context, LiquidCachePublishStage3D) {
      auto& self = *static_cast<StopAfter*>(context);
      ++self.event;
      return self.event == self.stop ? absl::CancelledError("interrupted") : absl::OkStatus();
    }
  };
  auto baseline = LiquidCache3D::Create(root / "baseline", scene);
  ASSERT_TRUE(baseline.ok());
  ASSERT_TRUE(Publish(*baseline, 0).ok());
  StopAfter count;
  ASSERT_TRUE(Publish(*baseline, 10, {StopAfter::Check, &count}).ok());
  EXPECT_GT(count.event, 15);
  for (std::size_t event = 1; event <= count.event; ++event) {
    SCOPED_TRACE(event);
    auto directory = root / std::to_string(event);
    auto cache = LiquidCache3D::Create(directory, scene);
    ASSERT_TRUE(cache.ok());
    ASSERT_TRUE(Publish(*cache, 0).ok());
    StopAfter failure{0, event};
    ASSERT_EQ(Publish(*cache, 10, {StopAfter::Check, &failure}).code(), absl::StatusCode::kCancelled);
    auto reopened = LiquidCache3D::Open(directory, resume.resolved_fingerprint);
    ASSERT_TRUE(reopened.ok());
    ASSERT_TRUE(reopened->ReadCommittedFrame(0).ok());
    auto last_checkpoint = reopened->ReadCheckpoint();
    ASSERT_TRUE(last_checkpoint.ok());
    const auto latest_frame = last_checkpoint->state.clock.frame_ordinal;
    EXPECT_TRUE(latest_frame == 0 || latest_frame == 10);
    if (latest_frame == 10) {
      EXPECT_TRUE(reopened->ReadCommittedFrame(10).ok());
    }
    ASSERT_TRUE(Publish(*reopened, 10).ok());
    EXPECT_TRUE(reopened->ReadCommittedFrame(10).ok());
  }
}
TEST_F(LiquidCacheStore3D, CheckpointReferenceRejectsTraversalKindLengthDigestClockAndIdentity) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  const auto path = root / "checkpoints/00000000000000000000.flcr";
  const auto original = Bytes(path);
  for (int mutation = 0; mutation < 7; ++mutation) {
    auto changed = original;
    auto* record = cache::v1::GetMutableCheckpointReference(changed.data());
    auto* checkpoint = record->mutable_checkpoint();
    switch (mutation) {
      case 0:
        checkpoint->mutable_relative_name()->Mutate(0, '.');
        break;
      case 1:
        checkpoint->mutate_kind(cache::v1::ContentKind::Points);
        break;
      case 2:
        checkpoint->mutate_byte_length(checkpoint->byte_length() + 1);
        break;
      case 3:
        checkpoint->mutable_digest()->mutable_bytes()->Mutate(0, 123);
        break;
      case 4:
        record->mutable_clock()->mutate_frame(10);
        break;
      case 5:
        record->mutable_resolved_fingerprint()->mutable_bytes()->Mutate(0, 123);
        break;
      case 6:
        record->mutable_frame_manifest()->mutable_digest()->mutable_bytes()->Mutate(0, 123);
        break;
    }
    Write(path, changed);
    EXPECT_FALSE(cache->ReadCheckpoint().ok()) << mutation;
    EXPECT_TRUE(cache->ReadCommittedFrame(0).ok()) << mutation;
  }
  Write(path, original);
  EXPECT_TRUE(cache->ReadCheckpoint().ok());
}
TEST_F(LiquidCacheStore3D, CorruptCheckpointPreventsPruningEarlierVerifiedHistory) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  ASSERT_TRUE(Publish(*cache, 10).ok());
  const auto reference_path = root / "checkpoints/00000000000000000010.flcr";
  const auto reference = Bytes(reference_path);
  const auto* record = cache::v1::GetCheckpointReference(reference.data());
  const auto state_path = root / record->checkpoint()->relative_name()->str();
  const auto original = Bytes(state_path);
  auto corrupted = original;
  corrupted.back() ^= 1;
  Write(state_path, corrupted);
  EXPECT_EQ(Publish(*cache, 20).code(), absl::StatusCode::kDataLoss);
  EXPECT_TRUE(fs::exists(root / "checkpoints/00000000000000000000.flcr"));
  EXPECT_TRUE(cache->ReadCommittedFrame(0).ok());
  EXPECT_TRUE(cache->ReadCommittedFrame(20).ok());
  Write(state_path, original);
  ASSERT_TRUE(Publish(*cache, 20).ok());
  EXPECT_EQ(cache->ReadCheckpoint(0).status().code(), absl::StatusCode::kNotFound);
}
TEST_F(LiquidCacheStore3D, SeparateCheckpointRepairRequiresMatchingCommittedFrame) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  SetFrame(10);
  EXPECT_EQ(cache->PublishCheckpoint(state, resume).code(), absl::StatusCode::kNotFound);
  StopAt stop{LiquidCachePublishStage3D::kManifestPublished};
  EXPECT_EQ(Publish(*cache, 10, {StopAt::Check, &stop}).code(), absl::StatusCode::kCancelled);
  ASSERT_TRUE(cache->PublishCheckpoint(state, resume).ok());
  EXPECT_EQ(cache->ReadCheckpoint()->state.clock.frame_ordinal, 10);
  state.next_particle_id = 2;
  EXPECT_EQ(cache->PublishCheckpoint(state, resume).code(), absl::StatusCode::kAlreadyExists);
  EXPECT_EQ(cache->ReadCheckpoint()->state.next_particle_id, 1);
}
TEST_F(LiquidCacheStore3D, CheckpointMustMatchExportedStableParticleFields) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(1);
  auto point_records = LiquidRecords3D<LiquidPoint3D>::Create(1);
  ASSERT_TRUE(particles.ok());
  ASSERT_TRUE(point_records.ok());
  ASSERT_TRUE(particles->Resize(1).ok());
  ASSERT_TRUE(point_records->Resize(1).ok());
  state.particles = std::move(*particles);
  points.points = std::move(*point_records);
  state.particles[0].id = 1;
  points.points[0].id = 1;
  state.next_particle_id = 2;
  resume.capacities.particles = 1;
  for (int mutation = 0; mutation < 5; ++mutation) {
    auto saved = points.points[0];
    switch (mutation) {
      case 0:
        points.points[0].id = 2;
        break;
      case 1:
        points.points[0].tag = 1;
        break;
      case 2:
        points.points[0].world_position.x = 1;
        break;
      case 3:
        points.points[0].world_velocity.y = 1;
        break;
      case 4:
        points.points[0].volume_ratio = 2;
        break;
    }
    EXPECT_EQ(cache->PublishFrame(points, &state, &resume).code(), absl::StatusCode::kInvalidArgument);
    points.points[0] = saved;
  }
  ASSERT_TRUE(cache->PublishFrame(points, &state, &resume).ok());
  EXPECT_EQ(cache->ReadCheckpoint()->state, state);
}
TEST_F(LiquidCacheStore3D, MissingInspectionAndOversizedResolvedDataFailClosed) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  std::error_code error;
  fs::remove(FramePath(0, ".json"), error);
  EXPECT_FALSE(cache->ReadCommittedFrame(0).ok());
  auto resolved = Bytes(root / "resolved.flrs");
  Write(root / "resolved.flrs", resolved + "trailing");
  EXPECT_EQ(cache->ReadResolvedScene().status().code(), absl::StatusCode::kDataLoss);
  EXPECT_FALSE(LiquidCache3D::Open(root, resume.resolved_fingerprint).ok());
}

TEST_F(LiquidCacheStore3D, AbandonedTemporaryAndReleasedLeaseAreNotRestartRecords) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  Write(root / "frames/.tmp-999-1", "incomplete manifest");
  Write(root / "checkpoints/.tmp-999-2", "incomplete reference");
  Write(root / "blobs/.tmp-999-3", "incomplete blob");
  Write(root / ".writer.lock", "stale operational bytes are not immutable identity");
  auto opened = LiquidCache3D::Open(root, resume.resolved_fingerprint);
  ASSERT_TRUE(opened.ok()) << opened.status();
  EXPECT_EQ(opened->ReadCheckpoint()->state.clock.frame_ordinal, 0);
  EXPECT_TRUE(Publish(*opened, 1).ok());
  EXPECT_EQ(Bytes(root / "frames/.tmp-999-1"), "incomplete manifest");
}
TEST_F(LiquidCacheStore3D, InvalidDirectoryAndDestinationObjectsPreserveExistingFrames) {
  Write(root / "blobs", "not a directory");
  EXPECT_FALSE(LiquidCache3D::Create(root, scene).ok());
  std::error_code error;
  fs::remove(root / "blobs", error);
  ASSERT_FALSE(error);
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  const auto original = Bytes(FramePath(0));
  ASSERT_TRUE(fs::create_directory(FramePath(1, ".json"), error));
  ASSERT_FALSE(error);
  EXPECT_FALSE(Publish(*cache, 1).ok());
  EXPECT_FALSE(fs::exists(FramePath(1)));
  EXPECT_EQ(Bytes(FramePath(0)), original);
  EXPECT_TRUE(cache->ReadCommittedFrame(0).ok());
}
#if !defined(_WIN32)
TEST_F(LiquidCacheStore3D, PosixSymbolicLinksCannotRedirectBlobOrDirectoryReads) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  auto frame = cache->ReadCommittedFrame(0);
  ASSERT_TRUE(frame.ok());
  auto name = LiquidContentName3D(frame->manifest.points);
  ASSERT_TRUE(name.ok());
  const auto content_path = root / name->data();
  const auto saved = Bytes(content_path);
  const auto outside = root / "outside-sentinel";
  Write(outside, saved);
  std::error_code error;
  fs::remove(content_path, error);
  ASSERT_FALSE(error);
  fs::create_symlink(outside, content_path, error);
  ASSERT_FALSE(error);
  EXPECT_EQ(cache->ReadCommittedFrame(0).status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(Bytes(outside), saved);
  fs::rename(root / "frames", root / "saved-frames", error);
  ASSERT_FALSE(error);
  fs::create_directory_symlink(root / "saved-frames", root / "frames", error);
  ASSERT_FALSE(error);
  EXPECT_EQ(cache->ReadCommittedFrame(0).status().code(), absl::StatusCode::kInvalidArgument);
}
#endif

TEST_F(LiquidCacheStore3D, PortableInspectionCannotPublishOrAuthorizeResume) {
  {
    auto cache = LiquidCache3D::Create(root, scene);
    ASSERT_TRUE(cache.ok());
    ASSERT_TRUE(Publish(*cache, 0).ok());
  }
  std::vector<std::pair<fs::path, fs::file_time_type>> times;
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file()) {
      times.emplace_back(entry.path(), entry.last_write_time());
    }
  }
  auto inspection = LiquidCache3D::OpenForInspection(root);
  ASSERT_TRUE(inspection.ok()) << inspection.status();
  auto resolved = inspection->ReadResolvedScene();
  ASSERT_TRUE(resolved.ok());
  auto fingerprint = ResolvedSceneFingerprint(resolved->View());
  ASSERT_TRUE(fingerprint.ok());
  EXPECT_EQ(*fingerprint, resume.resolved_fingerprint);
  EXPECT_TRUE(inspection->ReadCommittedFrame(0).ok());
  EXPECT_EQ(Publish(*inspection, 1).code(), absl::StatusCode::kFailedPrecondition);
  SetFrame(0);
  EXPECT_EQ(inspection->PublishCheckpoint(state, resume).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(inspection->ReadCheckpoint().status().code(), absl::StatusCode::kFailedPrecondition);
  for (const auto& [path, time] : times) {
    EXPECT_EQ(fs::last_write_time(path), time);
  }
  EXPECT_FALSE(fs::exists(FramePath(1)));
  auto compatible = LiquidCache3D::Open(root, resume.resolved_fingerprint);
  ASSERT_TRUE(compatible.ok());
  EXPECT_TRUE(compatible->ReadCheckpoint().ok());
}
TEST_F(LiquidCacheStore3D, InspectionDoesNotCreateAMissingDirectoryOrLease) {
  EXPECT_EQ(LiquidCache3D::OpenForInspection(root / "absent").status().code(), absl::StatusCode::kNotFound);
  EXPECT_FALSE(fs::exists(root / "absent"));
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  std::error_code error;
  fs::remove(root / ".writer.lock", error);
  ASSERT_FALSE(error);
  EXPECT_EQ(LiquidCache3D::OpenForInspection(root).status().code(), absl::StatusCode::kNotFound);
  EXPECT_FALSE(fs::exists(root / ".writer.lock"));
}

TEST_F(LiquidCacheStore3D, InterruptedPruneCannotReenableAnOlderCheckpoint) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(Publish(*cache, 0).ok());
  ASSERT_TRUE(Publish(*cache, 10).ok());
  StopAt stop{LiquidCachePublishStage3D::kBeforePrune};
  EXPECT_EQ(Publish(*cache, 20, {StopAt::Check, &stop}).code(), absl::StatusCode::kCancelled);
  std::size_t state_blobs = 0;
  for (const auto& entry : fs::directory_iterator(root / "blobs")) {
    if (entry.path().extension() == ".flst") {
      ++state_blobs;
    }
  }
  EXPECT_EQ(state_blobs, 3);
  auto reopened = LiquidCache3D::Open(root, resume.resolved_fingerprint);
  ASSERT_TRUE(reopened.ok());
  EXPECT_EQ(reopened->ReadCheckpoint(0).status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(reopened->ReadCheckpoint(10)->state.clock.frame_ordinal, 10);
  EXPECT_EQ(reopened->ReadCheckpoint()->state.clock.frame_ordinal, 20);
  EXPECT_TRUE(reopened->ReadCommittedFrame(0).ok());
  EXPECT_TRUE(Publish(*reopened, 20).ok());
}
TEST_F(LiquidCacheStore3D, InitialCheckpointMetadataMustCompleteBeforeLaterFrames) {
  auto cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(cache.ok());
  StopAt stop{LiquidCachePublishStage3D::kManifestPublished};
  EXPECT_EQ(Publish(*cache, 0, {StopAt::Check, &stop}).code(), absl::StatusCode::kCancelled);
  EXPECT_TRUE(cache->ReadCommittedFrame(0).ok());
  EXPECT_EQ(Publish(*cache, 1).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_FALSE(fs::exists(FramePath(1)));
  SetFrame(0);
  ASSERT_TRUE(cache->PublishCheckpoint(state, resume).ok());
  EXPECT_TRUE(Publish(*cache, 1).ok());
}

TEST_F(LiquidCacheStore3D, RelativeRootRemainsAnchoredAfterWorkingDirectoryChanges) {
  auto original_cache = LiquidCache3D::Create(root, scene);
  ASSERT_TRUE(original_cache.ok());
  ASSERT_TRUE(Publish(*original_cache, 0).ok());
  const auto original_directory = fs::current_path();
  struct RestoreDirectory {
    fs::path original;
    ~RestoreDirectory() {
      std::error_code ignored;
      fs::current_path(original, ignored);
    }
  } restore{original_directory};
  fs::current_path(root.parent_path());
  auto created = LiquidCache3D::Create(root.filename(), scene);
  auto opened = LiquidCache3D::Open(root.filename(), resume.resolved_fingerprint);
  auto inspected = LiquidCache3D::OpenForInspection(root.filename());
  fs::current_path(original_directory);
  ASSERT_TRUE(created.ok());
  ASSERT_TRUE(opened.ok());
  ASSERT_TRUE(inspected.ok());
  EXPECT_TRUE(created->ReadCommittedFrame(0).ok());
  EXPECT_TRUE(opened->ReadCheckpoint().ok());
  EXPECT_TRUE(inspected->ReadResolvedScene().ok());
  EXPECT_TRUE(Publish(*created, 1).ok());
}
}  // namespace
}  // namespace fluidity
