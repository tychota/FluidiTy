#pragma once
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "liquid_isolated_drop_3d.h"
namespace fluidity::liquid_step_internal {
inline absl::Status Error(absl::StatusCode code) {
  return absl::Status(code, {});
}
class OceanInletParticles3D {
 public:
  static absl::StatusOr<OceanInletParticles3D> Create(const OceanBoundarySettings3D& settings, double density);
  // Reconciles the trial inlet strip with the prescribed surface. The
  // transaction discards all particle and ID changes on rejected attempts.
  absl::Status
  Apply(const ResolvedOceanBoundary3D& boundary, const LiquidObstacle3D& obstacle, double time_s, LiquidState3D& trial);
  std::size_t reserved_bytes() const;

 private:
  Vec3 lower_m_{};
  Vec3 upper_m_{};
  std::array<std::size_t, 3> extent_{};
  double spacing_m_ = 0;
  double particle_volume_m3_ = 0;
  double particle_mass_kg_ = 0;
  LiquidRecords3D<std::uint8_t> occupied_;
  LiquidRecords3D<LiquidParticleState3D> births_;
};
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
  LiquidRecords3D<MacPressureCell3D> pressure_cell_scratch;  // Trial interior wall flux rows.
  LiquidRecords3D<double> incident_velocities;
  LiquidRecords3D<LiquidBoundaryParticleGather3D> gathered;
  LiquidRecords3D<double> particle_distances;
  // Reserved only for eligible stationary tank scenes.
  std::optional<LiquidRecords3D<LiquidRegularCellTransport3D>> regular_transport;
  // Reserved only for scenes with a translating piston.
  std::optional<WettedPistonWorkspace3D> wetted_piston;
  // New-history donor eligibility for pressure-supported particles.
  std::optional<LiquidRecords3D<LiquidFaceValidity3D>> projected_gather_validity;
  // Rebuilt from the current attempt's particles and used through gather.
  std::optional<PreparedIsolatedDrop3D> isolated_drop;
  // Keep the restricted old run across workspace reuse for the full and projected gathers.
  std::optional<LiquidRecords3D<LiquidBoundaryParticleGather3D>> isolated_drop_restricted_gathers;
  // Separate owned link copy used by pressure and velocity correction.
  std::optional<LiquidRecords3D<MacPressureFace3D>> pressure_face_scratch;
  std::size_t reserved_bytes = 0;
  static absl::StatusOr<Staging> Create(
      LiquidStateCapacities3D capacities,
      bool needs_regular_transport = false,
      bool needs_wetted_piston = false,
      bool needs_isolated_drop = false
  );
};
bool Finite(Vec3 value);
// Check finite carrier data and positive start-time J in the checked particle
// order. Inputs must have one gathered value and distance per particle.
absl::Status ValidateParticleUpdateRecords(
    const LiquidParticleUpdateInputs3D& inputs,
    std::span<const LiquidParticleState3D> particles
);
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
    const LiquidObstacle3D& obstacle,
    const LiquidState3D& trial,
    Staging& staging,
    const LiquidSubstepBackend3D& backend
);
// Copy the assembled faces into staging and replace the isolated drop air
// lifts with ambient + sigma times analytic curvature. On success the
// pressure problem borrows that copy for both solve and correction.
absl::Status SetIsolatedDropPressureLifts(
    UniformMacGeometry3D geometry,
    double surface_tension,
    double ambient_pressure,
    Staging& staging,
    LiquidPressureProblemView3D& pressure
);
// Return a certified half-open tank lattice cell, or no selection when the
// containing cell and its one-cell halo cannot be represented.
std::optional<CellIndex3D> TankContainingCell(UniformMacGeometry3D geometry, Vec3 world_position);
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
