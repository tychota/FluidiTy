#pragma once

#include <array>

#include "fluidity/host/liquid_boundary_geometry_3d.h"
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
// Impulses are returned in transverse corner-bit order. Invalid arguments,
// missing ownership or a mismatched frozen face, and unrepresentable
// arithmetic return status without changing the inputs.
absl::StatusOr<std::array<PlasticSolidPressureImpulse3D, 4>> StagePlasticSolidPressureFace3D(
    SparseMacGridView3D grid,
    const PlasticSolidBoundarySamples3D& samples,
    const FrozenLiquidFace3D& face,
    double tail_pressure,
    double head_pressure,
    double step_duration
);

}  // namespace fluidity
