#include <algorithm>
#include <cmath>

#include "../src/liquid_boundary_backend.h"
#include "liquid_arena_dispatch_3d.h"
#include "liquid_boundary_3d_bridge.h"
#include "liquid_geometry_stages_3d.h"
#include "liquid_geometry_stages_3d_bridge.h"
namespace fluidity::cuda_internal {
namespace {
using Range = CudaLiquidRange3D;
#define STAGE_TRY(expression) \
  do { \
    const auto stage_status = (expression); \
    if (!stage_status.ok()) { \
      return stage_status; \
    } \
  } while (false)
absl::Status Error(absl::StatusCode code) {
  return absl::Status(code, {});
}
bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
absl::Status Architecture(CudaLiquidArena3D& arena, const LiquidTypedRange3D<int>& architecture) {
  STAGE_TRY(CopyLiquidRange3D(arena, architecture, false));
  STAGE_TRY(FinishLiquidDispatch3D(arena));
  return architecture.pinned[0] == 1200 ? absl::OkStatus() : Error(absl::StatusCode::kFailedPrecondition);
}
absl::StatusOr<boundary_internal::GeometryOutput>
ComputeGeometry(void* context, const boundary_internal::GeometryInput& input) {
  auto& arena = *static_cast<CudaLiquidArena3D*>(context);
  const auto query = CheckedLiquidRange3D<boundary_internal::GeometryInput>(arena, Range::kGeometryInput, 1);
  const auto report = CheckedLiquidRange3D<BoundaryReport3D>(arena, Range::kBoundaryReport, 1);
  if (!query.ok()) {
    return query.status();
  }
  if (!report.ok()) {
    return report.status();
  }
  query->pinned[0] = input;
  STAGE_TRY(CopyLiquidRange3D(arena, *query, true));
  STAGE_TRY(arena.RecordRuntimeResult(LaunchLiquidBoundaryGeometry3D(query->device, report->device, arena.stream())));
  STAGE_TRY(CopyLiquidRange3D(arena, *report, false));
  STAGE_TRY(FinishLiquidDispatch3D(arena));
  if (report->pinned[0].compiled_architecture != 1200) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  STAGE_TRY(boundary_internal::ArithmeticStatus(report->pinned[0].error, true));
  return report->pinned[0].geometry;
}

Vec3 PackedFieldSample(Vec3 world_position) {
  return world_position;
}
Vec3 PackedFieldSample(double lattice_y) {
  return {0, lattice_y, 0};
}

// World and lattice queries share the checked arena ranges and one fence. The
// lattice form packs y in cell widths and resolves its distance only on device.
template <typename Sample>
absl::Status SampleFieldOnDevice(
    CudaLiquidArena3D& arena,
    const LiquidSceneField3D& field,
    std::span<const LiquidParticleState3D> particles,
    std::span<const Sample> positions,
    std::span<double> distances,
    const UniformMacGeometry3D* lattice_geometry
) {
  STAGE_TRY(arena.RecordRuntimeResult(cudaSuccess));
  if (positions.size() != distances.size()) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&field);
  const auto* particle_union = std::get_if<LiquidParticleSphereUnion3D>(&field);
  const auto* plane = std::get_if<LiquidHorizontalPlaneSurface3D>(&field);
  const double radius = sphere ? sphere->radius : (particle_union ? particle_union->radius : 0);
  if ((plane && !std::isfinite(plane->world_surface_height_y)) || (!plane && (!std::isfinite(radius) || radius <= 0))
      || (sphere && !Finite(sphere->world_center))) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (lattice_geometry
      && (!plane || !std::isfinite(lattice_geometry->world_origin.y) || !std::isfinite(lattice_geometry->cell_width)
          || lattice_geometry->cell_width <= 0)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (particles.size() > arena.capacities().particles
      || positions.size() > arena.range(Range::kSamplePositions).pinned.size() / sizeof(Vec3)) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  if (AliasesLiquidScratch3D(arena, particles) || AliasesLiquidScratch3D(arena, positions)
      || LiquidByteRangesOverlap3D(distances.data(), distances.size_bytes(), &field, sizeof(field))
      || AliasesLiquidScratch3D(arena, distances)
      || LiquidByteRangesOverlap3D(distances.data(), distances.size_bytes(), positions.data(), positions.size_bytes())
      || LiquidByteRangesOverlap3D(
          distances.data(),
          distances.size_bytes(),
          particles.data(),
          particles.size_bytes()
      )) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  for (const auto position : positions) {
    if (!Finite(PackedFieldSample(position))) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  for (const auto& particle : particles) {
    if (!Finite(particle.world_position)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  if (positions.empty()) {
    return absl::OkStatus();
  }
  const auto query = CheckedLiquidRange3D<LiquidSampleQuery3D>(arena, Range::kSampleQuery, 1);
  const auto samples = CheckedLiquidRange3D<Vec3>(arena, Range::kSamplePositions, positions.size());
  const auto centers = CheckedLiquidRange3D<Vec3>(arena, Range::kSampleParticles, particles.size());
  const auto output = CheckedLiquidRange3D<double>(arena, Range::kSampleDistances, distances.size());
  const auto architecture = CheckedLiquidRange3D<int>(arena, Range::kArchitecture, 1);
  if (!query.ok()) {
    return query.status();
  }
  if (!samples.ok()) {
    return samples.status();
  }
  if (!centers.ok()) {
    return centers.status();
  }
  if (!output.ok()) {
    return output.status();
  }
  if (!architecture.ok()) {
    return architecture.status();
  }
  auto field_kind = LiquidSampleFieldKind3D::kAnalyticSphere;
  if (plane) {
    field_kind =
        lattice_geometry ? LiquidSampleFieldKind3D::kLatticeHorizontalPlane : LiquidSampleFieldKind3D::kHorizontalPlane;
  } else if (particle_union) {
    field_kind = LiquidSampleFieldKind3D::kParticleUnion;
  }
  query->pinned[0] = {
      .sphere_center = sphere ? sphere->world_center : Vec3{},
      .radius = radius,
      .world_surface_height_y = plane ? plane->world_surface_height_y : 0,
      .field_kind = field_kind,
      .particle_count = particles.size(),
      .sample_count = positions.size(),
      .geometry = lattice_geometry ? *lattice_geometry : UniformMacGeometry3D{},
  };
  for (std::size_t index = 0; index < positions.size(); ++index) {
    samples->pinned[index] = PackedFieldSample(positions[index]);
  }
  for (std::size_t index = 0; index < particles.size(); ++index) {
    centers->pinned[index] = particles[index].world_position;
  }
  STAGE_TRY(CopyLiquidRange3D(arena, *query, true));
  STAGE_TRY(CopyLiquidRange3D(arena, *samples, true));
  STAGE_TRY(CopyLiquidRange3D(arena, *centers, true));
  STAGE_TRY(arena.RecordRuntimeResult(LiquidArenaRuntimeCalls().launch_field_samples(
      query->device,
      centers->device,
      samples->device,
      output->device,
      architecture->device,
      positions.size(),
      arena.stream()
  )));
  STAGE_TRY(CopyLiquidRange3D(arena, *output, false));
  STAGE_TRY(Architecture(arena, *architecture));
  for (double value : output->pinned) {
    if (!std::isfinite(value)) {
      return Error(absl::StatusCode::kOutOfRange);
    }
  }
  std::ranges::copy(output->pinned, distances.begin());
  return absl::OkStatus();
}
}  // namespace

absl::Status SampleLiquidField3D(
    CudaLiquidArena3D& arena,
    const LiquidSceneField3D& field,
    std::span<const LiquidParticleState3D> particles,
    std::span<const Vec3> positions,
    std::span<double> distances
) {
  return SampleFieldOnDevice(arena, field, particles, positions, distances, nullptr);
}
absl::StatusOr<double> SampleLiquidField3D(
    CudaLiquidArena3D& arena,
    const LiquidSceneField3D& field,
    std::span<const LiquidParticleState3D> particles,
    Vec3 position
) {
  double distance = 0;
  STAGE_TRY(SampleLiquidField3D(arena, field, particles, {&position, 1}, {&distance, 1}));
  return distance;
}
absl::Status SampleLiquidLatticePlane3D(
    CudaLiquidArena3D& arena,
    const LiquidHorizontalPlaneSurface3D& plane,
    UniformMacGeometry3D geometry,
    std::span<const double> sample_lattice_ordinates_y,
    std::span<double> distances
) {
  STAGE_TRY(arena.RecordRuntimeResult(cudaSuccess));
  if (sample_lattice_ordinates_y.size() != distances.size()
      || LiquidByteRangesOverlap3D(distances.data(), distances.size_bytes(), &plane, sizeof(plane))) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (sample_lattice_ordinates_y.size() > kLiquidFieldBatchCapacity3D) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  return SampleFieldOnDevice(arena, plane, {}, sample_lattice_ordinates_y, distances, &geometry);
}
absl::Status SampleLiquidFaces3D(
    CudaLiquidArena3D& arena,
    SparseMacGridView3D grid,
    const LiquidObstacle3D& obstacle,
    std::int64_t side,
    std::span<LiquidBoundaryFaceTag3D> tags,
    std::span<LiquidPrescribedWallSample3D> walls
) {
  STAGE_TRY(arena.RecordRuntimeResult(cudaSuccess));
  if ((!tags.empty() && tags.size() != grid.face_count()) || (!walls.empty() && walls.size() != grid.face_count())) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (grid.face_count() > arena.capacities().faces) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  if (AliasesLiquidScratch3D(arena, tags) || AliasesLiquidScratch3D(arena, walls)
      || LiquidByteRangesOverlap3D(tags.data(), tags.size_bytes(), walls.data(), walls.size_bytes())) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const auto packed_obstacle = boundary_internal::PackObstacle(obstacle);
  if (packed_obstacle.kind == boundary_internal::ObstacleKind::kOpenTank) {
    if (side != 0) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    STAGE_TRY(boundary_internal::ArithmeticStatus(boundary_internal::CheckTankStep(packed_obstacle.tank, 0), true));
  } else if (packed_obstacle.kind != boundary_internal::ObstacleKind::kAbsent) {
    if (packed_obstacle.kind == boundary_internal::ObstacleKind::kSlab && !walls.empty() && side != -1 && side != 1) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    const auto normal = packed_obstacle.normal;
    const double length_squared = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
    if (!Finite(normal) || !Finite(packed_obstacle.velocity) || !std::isfinite(packed_obstacle.coordinate)
        || std::abs(length_squared - 1) > 1e-12) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    if (packed_obstacle.kind == boundary_internal::ObstacleKind::kSlab
        && (!std::isfinite(packed_obstacle.half_width) || packed_obstacle.half_width <= 0)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  const auto query = CheckedLiquidRange3D<LiquidSampleQuery3D>(arena, Range::kSampleQuery, 1);
  const auto positions = CheckedLiquidRange3D<Vec3>(arena, Range::kSamplePositions, grid.face_count());
  const auto output =
      CheckedLiquidRange3D<LiquidSampleFaceResult3D>(arena, Range::kSampleFaceResults, grid.face_count());
  const auto architecture = CheckedLiquidRange3D<int>(arena, Range::kArchitecture, 1);
  if (!query.ok()) {
    return query.status();
  }
  if (!positions.ok()) {
    return positions.status();
  }
  if (!output.ok()) {
    return output.status();
  }
  if (!architecture.ok()) {
    return architecture.status();
  }
  for (std::size_t index = 0; index < grid.face_count(); ++index) {
    const auto face = grid.FaceIndex(index);
    if (!face.ok()) {
      return face.status();
    }
    const auto anchor = grid.FaceWorldAnchor(*face);
    if (!anchor.ok()) {
      return anchor.status();
    }
    positions->pinned[index] = packed_obstacle.kind == boundary_internal::ObstacleKind::kOpenTank
        ? boundary_internal::TankFaceAnchorInCellWidths(*face)
        : *anchor;
  }
  query->pinned[0] = {
      .sample_count = grid.face_count(),
      .obstacle = packed_obstacle,
      .geometry = grid.geometry(),
      .requested_side = side,
  };
  STAGE_TRY(CopyLiquidRange3D(arena, *query, true));
  STAGE_TRY(CopyLiquidRange3D(arena, *positions, true));
  STAGE_TRY(arena.RecordRuntimeResult(LaunchLiquidFaceSamples3D(
      query->device,
      positions->device,
      output->device,
      architecture->device,
      grid.face_count(),
      arena.stream()
  )));
  STAGE_TRY(CopyLiquidRange3D(arena, *output, false));
  STAGE_TRY(Architecture(arena, *architecture));
  for (const auto& result : output->pinned) {
    STAGE_TRY(boundary_internal::ArithmeticStatus(result.error, true));
  }
  for (std::size_t index = 0; index < grid.face_count(); ++index) {
    if (!tags.empty()) {
      tags[index] = output->pinned[index].tag;
    }
    if (!walls.empty()) {
      walls[index] = output->pinned[index].wall;
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<FrozenLiquidBoundaryGeometryView3D> FreezeLiquidGeometry3D(
    CudaLiquidArena3D& arena,
    const LiquidBoundaryGeometryViewInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace
) {
  STAGE_TRY(arena.RecordRuntimeResult(cudaSuccess));
  return boundary_internal::BoundedGeometryAccess::Freeze(workspace, inputs, {ComputeGeometry, &arena});
}
absl::StatusOr<LiquidPressureProblemView3D> BuildLiquidLinks3D(
    CudaLiquidArena3D& arena,
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    const LiquidPressureFields3D& fields,
    LiquidBoundaryWorkspace3D& workspace
) {
  STAGE_TRY(arena.RecordRuntimeResult(cudaSuccess));
  return boundary_internal::BoundedGeometryAccess::BuildLinks(workspace, geometry, fields, {ComputeGeometry, &arena});
}
#undef STAGE_TRY
}  // namespace fluidity::cuda_internal
