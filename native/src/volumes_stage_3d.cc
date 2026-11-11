#include "fluidity/host/volumes_stage_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include "fluidity/host/effects_source_3d.h"
#include "fluidity/host/whitewater_cache_3d.h"

namespace fluidity {
namespace {
struct PointsSource {
  EffectsCache3D cache;
  LiquidDigest3D execution_id;
  double represented_count_per_point;
};

absl::StatusOr<PointsSource> OpenPointsSource(const std::filesystem::path& root) {
  auto cache = EffectsCache3D::OpenForInspection(root);
  if (!cache.ok()) {
    return cache.status();
  }
  if (cache->identity().stage != EffectsStage3D::kWhitewater) {
    return absl::FailedPreconditionError("Volumes requires a committed Whitewater cache");
  }
  const auto execution = EffectsExecutionId3D(cache->identity());
  if (!execution.ok()) {
    return execution.status();
  }
  const auto settings = cache->ReadStageSettings();
  if (!settings.ok()) {
    return settings.status();
  }
  const auto decoded = DecodeWhitewaterSettings3D(settings->values());
  if (!decoded.ok()) {
    return decoded.status();
  }
  return PointsSource{
      .cache = std::move(*cache),
      .execution_id = *execution,
      .represented_count_per_point = decoded->represented_count_per_point,
  };
}

struct CompensatedSum {
  double value = 0;
  double correction = 0;
  void Add(double contribution) {
    const double corrected_contribution = contribution - correction;
    const double sum = value + corrected_contribution;
    correction = (sum - value) - corrected_contribution;
    value = sum;
  }
};

bool InventoryMatches(double measured, double expected) {
  return std::isfinite(measured) && std::isfinite(expected)
      && (expected == 0 ? measured == 0 : std::abs(measured / expected - 1) <= 1e-10);
}

absl::StatusOr<VolumesStageFrame3D> MeasureInventory(
    const WhitewaterVolumeFrame3D& frame,
    const WhitewaterFrame3D& points,
    const WhitewaterVolumeSettings3D& settings,
    double represented_count_per_point
) {
  VolumesStageFrame3D measured{.clock = frame.clock};
  const auto& volume = frame.volume;
  CompensatedSum foam;
  CompensatedSum bubble;
  CompensatedSum gas;
  for (std::size_t index = 0; index < volume.validity.size(); ++index) {
    foam.Add(volume.foam_count_per_m3[index]);
    bubble.Add(volume.bubble_count_per_m3[index]);
    gas.Add(volume.gas_fraction[index]);
    measured.known_voxels += volume.validity[index] == 1;
    measured.nonzero_voxels += volume.foam_count_per_m3[index] != 0 || volume.bubble_count_per_m3[index] != 0
        || volume.gas_fraction[index] != 0;
  }
  const double voxel_volume_m3 = std::pow(volume.grid.spacing_m, 3);
  measured.foam_inventory_count = foam.value * voxel_volume_m3;
  measured.bubble_inventory_count = bubble.value * voxel_volume_m3;
  measured.gas_inventory_m3 = gas.value * voxel_volume_m3;
  std::size_t foam_points = 0;
  std::size_t bubble_points = 0;
  for (const auto& record : points.points) {
    foam_points += record.point.kind == WhitewaterKind3D::kFoam;
    bubble_points += record.point.kind == WhitewaterKind3D::kBubble;
  }
  const double expected_foam = static_cast<double>(foam_points) * represented_count_per_point;
  const double expected_bubble = static_cast<double>(bubble_points) * represented_count_per_point;
  const double expected_gas_m3 =
      expected_bubble * (4 * std::numbers::pi / 3) * std::pow(settings.physical_bubble_radius_m, 3);
  if (!InventoryMatches(measured.foam_inventory_count, expected_foam)
      || !InventoryMatches(measured.bubble_inventory_count, expected_bubble)
      || !InventoryMatches(measured.gas_inventory_m3, expected_gas_m3)) {
    return absl::DataLossError("volume deposition did not preserve the Whitewater inventory");
  }
  return measured;
}

bool SameGrid(const WhitewaterVolumeGrid3D& a, const WhitewaterVolumeGrid3D& b) {
  return a.origin.x == b.origin.x && a.origin.y == b.origin.y && a.origin.z == b.origin.z && a.spacing_m == b.spacing_m
      && a.extent == b.extent;
}
}  // namespace

VolumesStage3D::VolumesStage3D(
    EffectsCache3D source,
    EffectsCache3D cache,
    WhitewaterVolumeSettings3D settings,
    double represented_count_per_point,
    bool read_only
)
    : source_(std::move(source)),
      cache_(std::move(cache)),
      settings_(settings),
      represented_count_per_point_(represented_count_per_point),
      read_only_(read_only) {
}

absl::StatusOr<VolumesStage3D> VolumesStage3D::OpenForPublication(
    const std::filesystem::path& whitewater_root,
    const std::filesystem::path& output_root,
    LiquidDigest3D build,
    const WhitewaterVolumeSettings3D& settings
) {
  const auto root = ValidateEffectsOutputRoot3D(whitewater_root, output_root, EffectsStage3D::kVolumes);
  if (!root.ok()) {
    return root;
  }
  const auto encoded = EncodeWhitewaterVolumeSettings3D(settings);
  if (!encoded.ok()) {
    return encoded.status();
  }
  auto source = OpenPointsSource(whitewater_root);
  if (!source.ok()) {
    return source.status();
  }
  const EffectsIdentity3D identity{
      .stage = EffectsStage3D::kVolumes,
      .upstream_execution = source->execution_id,
      .settings = LiquidContentDigest3D(encoded->values()),
      .build = build,
  };
  auto cache = EffectsCache3D::Create(output_root, identity);
  if (!cache.ok()) {
    return cache.status();
  }
  const auto published = cache->PublishStageSettings(encoded->values());
  if (!published.ok()) {
    return published;
  }
  return VolumesStage3D(
      std::move(source->cache),
      std::move(*cache),
      settings,
      source->represented_count_per_point,
      false
  );
}

absl::StatusOr<VolumesStage3D> VolumesStage3D::OpenForReplay(
    const std::filesystem::path& whitewater_root,
    const std::filesystem::path& output_root,
    LiquidDigest3D build
) {
  const auto root = ValidateEffectsOutputRoot3D(whitewater_root, output_root, EffectsStage3D::kVolumes);
  if (!root.ok()) {
    return root;
  }
  auto source = OpenPointsSource(whitewater_root);
  if (!source.ok()) {
    return source.status();
  }
  auto cache = EffectsCache3D::OpenForInspection(output_root);
  if (!cache.ok()) {
    return cache.status();
  }
  const auto& identity = cache->identity();
  if (identity.stage != EffectsStage3D::kVolumes || identity.upstream_execution != source->execution_id
      || identity.build != build) {
    return absl::FailedPreconditionError("Volumes identity differs from Whitewater source or executable");
  }
  const auto settings = cache->ReadStageSettings();
  if (!settings.ok()) {
    return settings.status();
  }
  const auto decoded = DecodeWhitewaterVolumeSettings3D(settings->values());
  if (!decoded.ok()) {
    return decoded.status();
  }
  return VolumesStage3D(
      std::move(source->cache),
      std::move(*cache),
      *decoded,
      source->represented_count_per_point,
      true
  );
}

std::uint64_t VolumesStage3D::next_frame() const {
  return next_frame_;
}
const EffectsIdentity3D& VolumesStage3D::identity() const {
  return cache_.identity();
}

absl::StatusOr<std::uint64_t>
VolumesStage3D::RestoreCommittedPrefix(std::uint64_t maximum_frame, absl::FunctionRef<absl::Status()> check_stop) {
  if (read_only_ || next_frame_ != 0) {
    return absl::FailedPreconditionError("Volumes restore requires a fresh publication owner");
  }
  if (maximum_frame == std::numeric_limits<std::uint64_t>::max()) {
    return absl::OutOfRangeError("Volumes restore frame bound exhausts cursor");
  }
  std::optional<EffectsCommit3D> last;
  std::uint64_t restored_count = 0;
  for (; restored_count <= maximum_frame; ++restored_count) {
    const auto stopped = check_stop();
    if (!stopped.ok()) {
      return stopped;
    }
    auto committed = cache_.ReadFrameIfCommitted(restored_count);
    if (!committed.ok()) {
      return committed.status();
    }
    if (!committed->has_value()) {
      break;
    }
    const auto upstream = source_.ReadFrame(restored_count);
    if (!upstream.ok()) {
      return absl::IsNotFound(upstream.status()) ? absl::DataLossError("committed Volumes source is missing")
                                                 : upstream.status();
    }
    const auto expected_previous = last ? last->payload_digest : LiquidDigest3D{};
    if ((**committed).clock != upstream->clock || (**committed).upstream_content != upstream->payload_digest
        || (**committed).previous_payload != expected_previous) {
      return absl::DataLossError("Volumes checkpoint source or predecessor differs");
    }
    last = std::move(**committed);
  }
  if (!last) {
    return 0;
  }
  const auto frame = DecodeWhitewaterVolumeFrame3D(last->payload.values(), settings_.maximum_voxels);
  if (!frame.ok()) {
    return frame.status();
  }
  if (frame->clock != last->clock || !SameGrid(frame->volume.grid, settings_.grid)
      || std::ranges::any_of(frame->volume.validity, [](std::uint8_t value) {
           return value != 1;
         })) {
    return absl::DataLossError("Volumes checkpoint clock or declared box differs");
  }
  previous_payload_ = last->payload_digest;
  next_frame_ = restored_count;
  return restored_count;
}

absl::StatusOr<VolumesStageFrame3D> VolumesStage3D::ProcessNextFrame(LiquidCachePublishControl3D control) {
  if (next_frame_ == std::numeric_limits<std::uint64_t>::max()) {
    return absl::OutOfRangeError("Volumes frame cursor exhausted");
  }
  const auto upstream = source_.ReadFrame(next_frame_);
  if (!upstream.ok()) {
    return upstream.status();
  }
  const auto points = DecodeWhitewaterFrame3D(upstream->payload.values());
  if (!points.ok()) {
    return points.status();
  }
  if (points->clock != upstream->clock) {
    return absl::DataLossError("Whitewater point clock differs from its manifest");
  }
  const auto frame = BuildWhitewaterVolumeFrame3D(*points, settings_, represented_count_per_point_);
  if (!frame.ok()) {
    return frame.status();
  }
  auto measured = MeasureInventory(*frame, *points, settings_, represented_count_per_point_);
  if (!measured.ok()) {
    return measured.status();
  }
  const auto payload = EncodeWhitewaterVolumeFrame3D(*frame, settings_.maximum_voxels);
  if (!payload.ok()) {
    return payload.status();
  }
  measured->payload_digest = LiquidContentDigest3D(payload->values());
  if (read_only_) {
    const auto committed = cache_.ReadFrame(next_frame_);
    if (!committed.ok()) {
      return committed.status();
    }
    if (committed->clock != frame->clock || committed->upstream_content != upstream->payload_digest
        || committed->previous_payload != previous_payload_ || committed->payload_digest != measured->payload_digest
        || !std::ranges::equal(committed->payload.values(), payload->values())) {
      return absl::DataLossError("Volumes replay differs from committed source, lineage or voxels");
    }
  } else {
    const auto published =
        cache_.Publish(frame->clock, upstream->payload_digest, previous_payload_, payload->values(), control);
    if (!published.ok()) {
      return published;
    }
  }
  previous_payload_ = measured->payload_digest;
  ++next_frame_;
  return measured;
}
}  // namespace fluidity
