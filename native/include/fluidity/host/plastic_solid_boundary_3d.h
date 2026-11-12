#pragma once

#include <array>

#include "fluidity/host/liquid_boundary_geometry_3d.h"
#include "fluidity/host/liquid_boundary_workspace_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/plastic_solid_checkpoint_3d.h"

namespace fluidity {

struct PlasticSolidBoundarySamples3D {
  LiquidRecords3D<SolidCornerDistance3D> corners;
  LiquidRecords3D<std::uint64_t> particle_ids;  // Owner of each corner, in the same order.
  LiquidRecords3D<Vec3> particle_offsets;  // Corner offsets from each owner in world metres.

  // Borrows the sampled signed distances and nodal velocities for pressure freeze.
  SampledSolidBoundary3D View() const;
};

// Evaluates a single body's accepted particle sphere union at the liquid
// corner lattice keys. Uses an explicit common support radius in metres
// and the winning particle's affine velocity at each corner. Smaller IDs
// break distance ties. The input particle IDs and (z,y,x) corner keys must
// be strictly ordered. Liquid sample values are unused. Missing body
// returns FailedPrecondition, invalid inputs return InvalidArgument, and
// unrepresentable arithmetic returns OutOfRange. No partial samples escape.
absl::StatusOr<PlasticSolidBoundarySamples3D> SamplePlasticSolidBoundary3D(
    SparseMacGridView3D grid,
    std::span<const LiquidCornerDistance3D> liquid_corner_keys,
    std::span<const PlasticSolidCheckpointParticle3D> accepted_particles,
    std::uint64_t body_id,
    double support_radius
);

// One corner's impulse and lever arm for its owning particle. The APIC
// moment increment is the outer product of impulse and world_offset.
struct PlasticSolidPressureImpulse3D {
  std::uint64_t particle_id;
  Vec3 impulse;  // World N s.
  Vec3 world_offset;  // Corner minus accepted particle centre, metres.
};

// Stages the transpose of one sampled wall face's pressure flux. Samples
// must be the unmodified output used to freeze this face. The supplied
// physical pressures in Pa use a common gauge and tail-to-head order.
// Supply the same frozen halfspace used for the forward wall flux.
// Impulses are returned in transverse corner-bit order. Invalid arguments,
// missing ownership or a mismatched frozen face, and unrepresentable
// arithmetic return status without changing the inputs.
absl::StatusOr<std::array<PlasticSolidPressureImpulse3D, 4>> StagePlasticSolidPressureFace3D(
    SparseMacGridView3D grid,
    const PlasticSolidBoundarySamples3D& samples,
    const FrozenLiquidFace3D& face,
    double tail_pressure,
    double head_pressure,
    double step_duration,
    const LiquidObstacle3D& obstacle = {}
);

// Restores a single snow body's trial particles and adds the staged linear
// and affine pressure impulses. The subsequent dry step must use complete
// quadratic MAC support at this cell width. Loads are applied in supplied
// order. Unknown particle IDs and invalid inputs return InvalidArgument.
// Nonfinite resulting velocities or slopes return OutOfRange. The accepted
// records are immutable and failure returns no trial.
absl::StatusOr<std::vector<DrySolidParticle3D>> StagePressureLoadedSnowParticles3D(
    std::span<const PlasticSolidCheckpointParticle3D> accepted,
    std::uint64_t body_id,
    std::uint64_t material_id,
    const SnowParameters3D& snow,
    double cell_width,
    std::span<const PlasticSolidPressureImpulse3D> loads
);

// Stages pressure impulses for all blocked faces in frozen slot order.
// Solved pressures are in frozen cell order, in Pa. Missing fluid
// endpoints use the supplied physical exterior pressure, in the same
// gauge. An inconsistent face or pressure extent rejects the entire batch.
absl::StatusOr<LiquidRecords3D<PlasticSolidPressureImpulse3D>> StagePlasticSolidPressureLoads3D(
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    const PlasticSolidBoundarySamples3D& samples,
    std::span<const double> solved_cell_pressures,
    double exterior_pressure,
    const LiquidObstacle3D& obstacle = {}
);

// Frozen pressure-to-wall mobility B M^-1 B^T. Each face uses the same
// corner coefficients as pressure traction. The affine mass metric uses
// h^2 / 4 for each slope coordinate. Prescribed analytic wall motion is fixed.
class PlasticSolidPressureMobility3D {
 public:
  // Freezes the unit-pressure coefficients in face slot order. The samples
  // and obstacle must match the frozen geometry. Accepted particles are
  // strictly ordered by ID and supply finite positive masses. The result
  // owns all coefficients. Failure publishes no partial operator.
  static absl::StatusOr<PlasticSolidPressureMobility3D> Create(
      const FrozenLiquidBoundaryGeometryView3D& geometry,
      const PlasticSolidBoundarySamples3D& samples,
      std::span<const PlasticSolidCheckpointParticle3D> accepted,
      const LiquidObstacle3D& obstacle = {}
  );

  // Applies one pressure increment in frozen cell order with zero exterior
  // increment. Returns induced wall flux in frozen face order in m³/s.
  // The operator includes the frozen step duration. Failure returns no flux.
  absl::StatusOr<std::vector<double>> Apply(std::span<const double> pressure_increment) const;

 private:
  struct Corner {
    std::size_t owner = 0;
    std::array<double, 4> coefficients{};
  };
  struct Face {
    std::size_t axis = 0;
    std::size_t tail_cell = 0;
    std::size_t head_cell = 0;
    std::array<Corner, 4> corners{};
  };
  std::size_t particle_count_ = 0;
  std::size_t cell_count_ = 0;
  std::vector<Face> faces_;
};

}  // namespace fluidity
