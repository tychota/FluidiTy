#include "fluidity/host/surface_settings_3d.h"

#include "liquid_cache_3d_internal.h"
#include "mesh_settings_generated.h"

namespace fluidity {
namespace {
namespace detail = liquid_cache_internal;
namespace wire = effects::v1;
// Match density's finite, resolved Gaussian support before allocating a field.
bool ValidPhysicalControls(const SurfaceReconstructionOptions3D& options) {
  if (!detail::Positive(options.spacing_m) || !detail::Positive(options.sigma_m)
      || !detail::Positive(options.tracking.maximum_projection_cells) || options.tracking.maximum_projection_cells > 0.5
      || !detail::Positive(options.tracking.projection_tolerance_cells)
      || options.tracking.projection_tolerance_cells >= options.tracking.maximum_projection_cells) {
    return false;
  }
  const double support_radius_nodes = std::round(4 * options.sigma_m / options.spacing_m);
  return std::isfinite(support_radius_nodes) && support_radius_nodes >= 1 && support_radius_nodes <= 4096;
}
}  // namespace

absl::StatusOr<LiquidCacheBytes3D>
EncodeSurfaceSettings3D(const SurfaceReconstructionOptions3D& options, LiquidCacheLimits3D limits) {
  if (!ValidPhysicalControls(options)) {
    return absl::InvalidArgumentError("invalid physical surface settings");
  }
  return detail::Build<wire::MeshSettings>(
      2,
      limits,
      "FLMC",
      [&](auto& builder) -> absl::StatusOr<flatbuffers::Offset<wire::MeshSettings>> {
        return wire::CreateMeshSettings(
            builder,
            1,
            0,
            1,
            options.spacing_m,
            options.sigma_m,
            options.tracking.maximum_projection_cells,
            options.tracking.projection_tolerance_cells
        );
      }
  );
}

absl::StatusOr<SurfaceReconstructionOptions3D>
DecodeSurfaceSettings3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits) {
  const auto verified = detail::Verify<wire::MeshSettings>(bytes, "FLMC", limits);
  if (!verified.ok()) {
    return verified.status();
  }
  const auto* source = *verified;
  if (source->algorithm_version() != 1) {
    return absl::FailedPreconditionError("unsupported surface reconstruction algorithm");
  }
  SurfaceReconstructionOptions3D options;
  options.spacing_m = source->spacing_m();
  options.sigma_m = source->sigma_m();
  options.tracking.maximum_projection_cells = source->maximum_projection_cells();
  options.tracking.projection_tolerance_cells = source->projection_tolerance_cells();
  if (!ValidPhysicalControls(options)) {
    return absl::DataLossError("invalid physical surface settings");
  }
  return options;
}
}  // namespace fluidity
