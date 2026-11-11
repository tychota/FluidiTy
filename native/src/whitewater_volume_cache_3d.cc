#include "fluidity/host/whitewater_volume_cache_3d.h"

#include <numbers>

#include "liquid_cache_3d_internal.h"
#include "volume_frame_generated.h"
#include "volume_settings_generated.h"

namespace fluidity {
namespace {
namespace detail = liquid_cache_internal;
namespace wire = effects::v1;

absl::StatusOr<std::size_t> VoxelCount(const WhitewaterVolumeGrid3D& grid, std::size_t maximum_voxels) {
  if (!detail::Finite(grid.origin) || !detail::Positive(grid.spacing_m) || *std::ranges::min_element(grid.extent) < 2) {
    return absl::InvalidArgumentError("invalid volume box geometry");
  }
  maximum_voxels = std::min(maximum_voxels, std::size_t{2000000});
  if (grid.extent[0] > maximum_voxels / grid.extent[1] / grid.extent[2]) {
    return absl::ResourceExhaustedError("volume box exceeds voxel budget");
  }
  return grid.extent[0] * grid.extent[1] * grid.extent[2];
}

bool ValidFrameClock(LiquidClock3D clock) {
  return detail::ValidClock(clock) && clock.substep_ordinal == 0
      && (clock.frame_ordinal == 0 ? clock == LiquidClock3D{} : clock.time > 0);
}

bool ValidDensities(double foam, double bubble, double gas) {
  return detail::Nonnegative(foam) && detail::Nonnegative(bubble) && detail::Nonnegative(gas);
}

bool NonzeroDensities(double foam, double bubble, double gas) {
  return foam != 0 || bubble != 0 || gas != 0;
}

template <typename Root>
WhitewaterVolumeGrid3D UnpackGrid(const Root& root) {
  return {
      .origin = detail::Unpack(root.origin()),
      .spacing_m = root.spacing_m(),
      .extent = {root.nx(), root.ny(), root.nz()}
  };
}
}  // namespace

absl::StatusOr<WhitewaterVolumeFrame3D> BuildWhitewaterVolumeFrame3D(
    const WhitewaterFrame3D& points,
    const WhitewaterVolumeSettings3D& settings,
    double represented_count_per_point
) {
  const auto count = VoxelCount(settings.grid, settings.maximum_voxels);
  if (!count.ok()) {
    return count.status();
  }
  if (points.points.size() > (std::size_t{1} << 20)) {
    return absl::ResourceExhaustedError("volume input point budget exceeded");
  }
  if (!ValidFrameClock(points.clock) || !detail::Positive(settings.physical_bubble_radius_m)
      || !detail::Positive(represented_count_per_point)) {
    return absl::InvalidArgumentError("invalid volume frame clock or gas inventory settings");
  }
  const double radius_m = settings.physical_bubble_radius_m;
  const double represented_gas_volume_m3 =
      represented_count_per_point * (4 * std::numbers::pi / 3) * radius_m * radius_m * radius_m;
  if (!detail::Positive(represented_gas_volume_m3)) {
    return absl::OutOfRangeError("represented gas volume is unrepresentable");
  }
  std::vector<WhitewaterVolumePoint3D> deposits;
  deposits.reserve(points.points.size());
  for (const auto& record : points.points) {
    deposits.push_back({
        .kind = record.point.kind,
        .position = record.point.kinematics.position,
        .represented_count = represented_count_per_point,
        .gas_volume_m3 = record.point.kind == WhitewaterKind3D::kBubble ? represented_gas_volume_m3 : 0,
    });
  }
  const std::vector<std::uint8_t> known(*count, 1);
  auto volume = RasterizeWhitewaterVolume3D(deposits, settings.grid, known, settings.maximum_voxels);
  if (!volume.ok()) {
    return volume.status();
  }
  return WhitewaterVolumeFrame3D{.clock = points.clock, .volume = std::move(*volume)};
}

absl::StatusOr<LiquidCacheBytes3D> EncodeWhitewaterVolumeFrame3D(
    const WhitewaterVolumeFrame3D& frame,
    std::size_t maximum_voxels,
    LiquidCacheLimits3D limits
) {
  const auto& volume = frame.volume;
  const auto& grid = volume.grid;
  const auto count = VoxelCount(grid, maximum_voxels);
  if (!count.ok()) {
    return count.status();
  }
  if (!ValidFrameClock(frame.clock) || volume.validity.size() != *count || volume.foam_count_per_m3.size() != *count
      || volume.bubble_count_per_m3.size() != *count || volume.gas_fraction.size() != *count) {
    return absl::InvalidArgumentError("volume frame clock or voxel array extents differ");
  }
  std::size_t nonzero_count = 0;
  for (std::size_t index = 0; index < *count; ++index) {
    const double foam = volume.foam_count_per_m3[index];
    const double bubble = volume.bubble_count_per_m3[index];
    const double gas = volume.gas_fraction[index];
    const bool nonzero = NonzeroDensities(foam, bubble, gas);
    if (!ValidDensities(foam, bubble, gas) || volume.validity[index] > 1
        || (nonzero && (volume.validity[index] == 0 || frame.clock.frame_ordinal == 0))) {
      return absl::InvalidArgumentError("invalid volume density or known-domain mask");
    }
    nonzero_count += nonzero;
  }
  const auto record_bytes = nonzero_count * sizeof(wire::VolumeVoxel) + volume.validity.size();
  const auto table_budget = 2 + (record_bytes + 511) / 512;
  const auto budget = detail::Budget(table_budget, limits);
  if (!budget.ok()) {
    return budget.status();
  }
  std::vector<wire::VolumeVoxel> voxels;
  voxels.reserve(nonzero_count);
  for (std::size_t index = 0; index < *count; ++index) {
    const double foam = volume.foam_count_per_m3[index];
    const double bubble = volume.bubble_count_per_m3[index];
    const double gas = volume.gas_fraction[index];
    if (NonzeroDensities(foam, bubble, gas)) {
      voxels.emplace_back(
          static_cast<std::uint32_t>(index),
          detail::Canonical(foam),
          detail::Canonical(bubble),
          detail::Canonical(gas)
      );
    }
  }
  return detail::Build<wire::VolumeFrame>(
      table_budget,
      limits,
      "FLVO",
      [&](auto& builder) -> absl::StatusOr<flatbuffers::Offset<wire::VolumeFrame>> {
        const auto validity = builder.CreateVector(volume.validity);
        const auto records = builder.CreateVectorOfStructs(voxels);
        const auto clock = detail::Pack(frame.clock);
        const auto origin = detail::PackCanonical(grid.origin);
        return wire::CreateVolumeFrame(
            builder,
            1,
            0,
            &clock,
            &origin,
            grid.spacing_m,
            static_cast<std::uint32_t>(grid.extent[0]),
            static_cast<std::uint32_t>(grid.extent[1]),
            static_cast<std::uint32_t>(grid.extent[2]),
            validity,
            records
        );
      }
  );
}

absl::StatusOr<WhitewaterVolumeFrame3D> DecodeWhitewaterVolumeFrame3D(
    std::span<const std::uint8_t> bytes,
    std::size_t maximum_voxels,
    LiquidCacheLimits3D limits
) {
  const auto verified = detail::Verify<wire::VolumeFrame>(bytes, "FLVO", limits);
  if (!verified.ok()) {
    return verified.status();
  }
  const auto* source = *verified;
  const auto grid = UnpackGrid(*source);
  const auto count = VoxelCount(grid, maximum_voxels);
  if (!count.ok()) {
    return absl::IsResourceExhausted(count.status()) ? count.status() : absl::DataLossError(count.status().message());
  }
  const auto clock = detail::Unpack(source->clock());
  if (!ValidFrameClock(clock) || source->validity()->size() != *count || source->voxels()->size() > *count
      || (clock.frame_ordinal == 0 && source->voxels()->size() != 0)
      || std::ranges::any_of(*source->validity(), [](std::uint8_t value) {
           return value > 1;
         })) {
    return absl::DataLossError("invalid volume frame clock, extents or mask");
  }
  std::optional<std::uint32_t> previous_index;
  for (const auto* voxel : *source->voxels()) {
    const auto index = voxel->linear_index();
    if (index >= *count || (previous_index && index <= *previous_index) || source->validity()->Get(index) != 1
        || !ValidDensities(voxel->foam_count_per_m3(), voxel->bubble_count_per_m3(), voxel->gas_fraction())
        || !NonzeroDensities(voxel->foam_count_per_m3(), voxel->bubble_count_per_m3(), voxel->gas_fraction())) {
      return absl::DataLossError("invalid or unordered sparse volume voxel");
    }
    previous_index = index;
  }
  WhitewaterVolumeFrame3D frame{
      .clock = clock,
      .volume =
          {
              .grid = grid,
              .foam_count_per_m3 = std::vector<double>(*count),
              .bubble_count_per_m3 = std::vector<double>(*count),
              .gas_fraction = std::vector<double>(*count),
              .validity = {source->validity()->begin(), source->validity()->end()},
          },
  };
  for (const auto* voxel : *source->voxels()) {
    const auto index = voxel->linear_index();
    frame.volume.foam_count_per_m3[index] = voxel->foam_count_per_m3();
    frame.volume.bubble_count_per_m3[index] = voxel->bubble_count_per_m3();
    frame.volume.gas_fraction[index] = voxel->gas_fraction();
  }
  return frame;
}

absl::StatusOr<LiquidCacheBytes3D>
EncodeWhitewaterVolumeSettings3D(const WhitewaterVolumeSettings3D& settings, LiquidCacheLimits3D limits) {
  const auto& grid = settings.grid;
  const auto count = VoxelCount(grid, settings.maximum_voxels);
  if (!count.ok()) {
    return count.status();
  }
  if (!detail::Positive(settings.physical_bubble_radius_m)) {
    return absl::InvalidArgumentError("invalid physical bubble radius");
  }
  return detail::Build<wire::VolumeSettings>(
      2,
      limits,
      "FLVC",
      [&](auto& builder) -> absl::StatusOr<flatbuffers::Offset<wire::VolumeSettings>> {
        const auto origin = detail::PackCanonical(grid.origin);
        return wire::CreateVolumeSettings(
            builder,
            1,
            0,
            1,
            &origin,
            grid.spacing_m,
            static_cast<std::uint32_t>(grid.extent[0]),
            static_cast<std::uint32_t>(grid.extent[1]),
            static_cast<std::uint32_t>(grid.extent[2]),
            settings.physical_bubble_radius_m
        );
      }
  );
}

absl::StatusOr<WhitewaterVolumeSettings3D>
DecodeWhitewaterVolumeSettings3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits) {
  const auto verified = detail::Verify<wire::VolumeSettings>(bytes, "FLVC", limits);
  if (!verified.ok()) {
    return verified.status();
  }
  const auto* source = *verified;
  if (source->algorithm_version() != 1) {
    return absl::FailedPreconditionError("unsupported volume deposition algorithm");
  }
  WhitewaterVolumeSettings3D settings{
      .grid = UnpackGrid(*source),
      .physical_bubble_radius_m = source->physical_bubble_radius_m(),
  };
  const auto count = VoxelCount(settings.grid, settings.maximum_voxels);
  if (!count.ok()) {
    return absl::IsResourceExhausted(count.status()) ? count.status() : absl::DataLossError(count.status().message());
  }
  if (!detail::Positive(settings.physical_bubble_radius_m)) {
    return absl::DataLossError("invalid physical bubble radius");
  }
  return settings;
}
}  // namespace fluidity
