#pragma once
#include "fluidity/host/liquid_substep_backend_3d.h"
namespace fluidity::liquid_step_internal {
inline absl::Status Error(absl::StatusCode code) {
  return absl::Status(code, {});
}
struct Staging {
  LiquidRecords3D<CellIndex3D> candidates;
  LiquidRecords3D<LiquidCenterDistance3D> centers;
  LiquidRecords3D<LiquidCornerDistance3D> corners;
  LiquidRecords3D<LiquidBoundaryParticle3D> particles;
  LiquidRecords3D<LiquidBoundaryFaceTag3D> tags;
  LiquidRecords3D<LiquidBoundaryFaceTag3D> new_tags;
  LiquidRecords3D<double> old_velocities;
  LiquidRecords3D<double> new_velocities;
  LiquidRecords3D<LiquidFaceValidity3D> validity;
  LiquidRecords3D<LiquidPrescribedWallSample3D> old_walls;
  LiquidRecords3D<LiquidPrescribedWallSample3D> new_walls;
  LiquidRecords3D<LiquidPressureCellFields3D> pressure_fields;
  LiquidRecords3D<double> incident_velocities;
  LiquidRecords3D<LiquidBoundaryParticleGather3D> gathered;
  LiquidRecords3D<double> particle_distances;
  std::size_t reserved_bytes = 0;
  static absl::StatusOr<Staging> Create(LiquidStateCapacities3D capacities);
};
bool Finite(Vec3 value);
double Component(Vec3 value, FaceNormalAxis axis);
double MaximumComponentMagnitude(Vec3 value);
double Dot(Vec3 a, Vec3 b);
Vec3 Add(Vec3 a, Vec3 b);
Vec3 Scale(Vec3 a, double factor);
LiquidObstacle3D ObstacleAtTime(const LiquidObstacle3D& obstacle, double time);
Vec3 WallVelocity(const LiquidObstacle3D& obstacle);
double SolidDistance(const LiquidObstacle3D& obstacle, Vec3 position);
double FieldDistance(const LiquidSceneField3D& field, std::span<const LiquidParticleState3D> particles, Vec3 position);
absl::Status
ValidateScene(const LiquidState3D& initial, const LiquidSceneInputs3D& scene, const LiquidStepOptions3D& options);
absl::Status Emit(const LiquidSceneInputs3D& scene, double duration, LiquidState3D& trial);
absl::Status PrepareField(
    const LiquidSceneField3D& field,
    const LiquidState3D& trial,
    Staging& staging,
    const LiquidSubstepBackend3D& backend
);
absl::Status PrepareTransfers(const LiquidState3D& trial, SparseMacGridView3D grid, Staging& staging);
absl::Status
FillFaceTags(SparseMacGridView3D grid, const LiquidObstacle3D& obstacle, std::span<LiquidBoundaryFaceTag3D> tags);
absl::Status FillWalls(
    SparseMacGridView3D grid,
    const LiquidObstacle3D& obstacle,
    std::int64_t side,
    std::span<LiquidPrescribedWallSample3D> walls
);
}  // namespace fluidity::liquid_step_internal
