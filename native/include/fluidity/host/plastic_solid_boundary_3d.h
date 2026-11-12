#pragma once

#include "fluidity/host/liquid_boundary_geometry_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/plastic_solid_checkpoint_3d.h"

namespace fluidity {

struct PlasticSolidBoundarySamples3D {
  LiquidRecords3D<SolidCornerDistance3D> corners;
  LiquidRecords3D<std::uint64_t> particle_ids;  // Owner of each corner, in the same order.

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

}  // namespace fluidity
