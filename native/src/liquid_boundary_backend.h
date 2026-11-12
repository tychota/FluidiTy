#pragma once
#include "fluidity/base/liquid_boundary_geometry_math_3d.h"
#include "fluidity/base/liquid_boundary_math.h"
#include "fluidity/host/liquid_boundary_workspace_3d.h"
#include "fluidity/host/liquid_wall_contact_3d.h"
namespace fluidity::boundary_internal {
// Partitions the frozen liquid volume into cell octants. Scheduler callers
// use code-only errors so a rejected physical attempt allocates nothing.
absl::StatusOr<std::array<double, 8>> MeasureLiquidCellOctants(
    UniformMacGeometry3D geometry,
    CellIndex3D cell_index,
    const LiquidObstacle3D& obstacle,
    std::span<const double, 8> liquid_corner_distances,
    bool detailed_errors
);

// Owns no storage. The callback fills the first count results in query order
// and returns a dispatch-level status. Each result has its own arithmetic status.
struct GeometryBatch {
  std::span<GeometryInput> queries;
  std::span<GeometryBatchResult> results;
  absl::Status (*compute)(void*, std::size_t);
  void* context;
};
// The orchestrator validates sample ownership and order, then delegates every
// geometry calculation. CUDA supplies this callback explicitly; it never falls back.
struct GeometryBackend {
  absl::StatusOr<GeometryOutput> (*compute)(void*, const GeometryInput&);
  void* context;
  GeometryBatch* batch = nullptr;
};
// Internal typed seam preserves the bounded workspace publication and validation.
struct BoundedGeometryAccess {
  static absl::StatusOr<FrozenLiquidBoundaryGeometryView3D> Freeze(
      LiquidBoundaryWorkspace3D& workspace,
      const LiquidBoundaryGeometryViewInputs3D& inputs,
      GeometryBackend backend
  ) {
    return workspace.FreezeWithBackend(inputs, backend);
  }
  static absl::StatusOr<LiquidPressureProblemView3D> BuildLinks(
      LiquidBoundaryWorkspace3D& workspace,
      const FrozenLiquidBoundaryGeometryView3D& geometry,
      const LiquidPressureFields3D& fields,
      GeometryBackend backend
  ) {
    return workspace.BuildLinksWithBackend(geometry, fields, backend);
  }
};
absl::StatusOr<FrozenLiquidBoundaryGeometry3D> FreezeGeometry(const LiquidBoundaryGeometryInputs3D&, GeometryBackend);
absl::StatusOr<LiquidPressureProblem3D>
BuildLinks(const FrozenLiquidBoundaryGeometry3D&, const LiquidPressureFields3D&, GeometryBackend);
inline Obstacle PackObstacle(const LiquidObstacle3D& obstacle) {
  if (const auto* halfspace = std::get_if<TranslatingHalfspace3D>(&obstacle)) {
    return {
        ObstacleKind::kHalfspace,
        halfspace->unit_normal,
        halfspace->start_plane_coordinate,
        0,
        halfspace->wall_velocity
    };
  }
  if (const auto* slab = std::get_if<TranslatingSlab3D>(&obstacle)) {
    return {
        ObstacleKind::kSlab,
        slab->unit_normal,
        slab->start_center_coordinate,
        slab->half_width,
        slab->wall_velocity
    };
  }
  if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle)) {
    return {.kind = ObstacleKind::kOpenTank, .tank = *tank};
  }
  return {};
}
inline ContactInput PackContact(const LiquidWallContactInputs3D& inputs) {
  return {
      inputs.start_world_position,
      inputs.stored_world_velocity,
      inputs.advecting_world_velocity,
      PackObstacle(inputs.obstacle),
      inputs.step_duration,
      inputs.friction_coefficient,
      inputs.penetration_tolerance,
  };
}
inline absl::Status ArithmeticStatus(Error error, bool bounded = false) {
  if (bounded) {
    switch (error) {
      case Error::kNone:
        return absl::OkStatus();
      case Error::kInvalid:
        return absl::Status(absl::StatusCode::kInvalidArgument, {});
      case Error::kPrecondition:
        return absl::Status(absl::StatusCode::kFailedPrecondition, {});
      case Error::kRange:
        return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    return absl::Status(absl::StatusCode::kInternal, {});
  }
  switch (error) {
    case Error::kNone:
      return absl::OkStatus();
    case Error::kInvalid:
      return absl::InvalidArgumentError("Invalid boundary arithmetic input");
    case Error::kPrecondition:
      return absl::FailedPreconditionError("Uncertified boundary geometry or contact");
    case Error::kRange:
      return absl::OutOfRangeError("Unrepresentable boundary arithmetic");
  }
  return absl::InternalError("Unknown boundary arithmetic status");
}
inline LiquidWallContactResult3D UnpackContact(const ContactOutput& result) {
  return {
      result.world_position,
      result.stored_world_velocity,
      result.advecting_world_velocity,
      result.contact_fraction.has_value() ? std::optional<double>(*result.contact_fraction) : std::nullopt,
      result.outward_normal,
      result.incoming_stored_relative_normal_speed,
      result.incoming_advecting_relative_normal_speed,
      static_cast<LiquidWallContactDiagnostic3D>(result.diagnostic),
  };
}
}  // namespace fluidity::boundary_internal
