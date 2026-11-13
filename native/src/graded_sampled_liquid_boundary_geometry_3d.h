#pragma once

#include <span>

#include "fluidity/host/liquid_substep_3d.h"
#include "graded_liquid_boundary_geometry_3d.h"

namespace fluidity {

// Freeze the piecewise-affine geometry on the canonical tetrahedra and
// subface diagonals. Supply unique finite corner samples ordered by signed
// (z,y,x), covering every leaf and canonical subface corner, with no unrelated
// samples. The leaf-order certificate is 1 only when the caller has certified
// the entire leaf is dry and nonsolid, and 0 otherwise. A zero-volume leaf
// without that certificate fails closed, as does a conflicting certificate.
// A subdivided coarse side requires strictly uniform signs for each field
// across its coarse and subface corners. This excludes unresolved cuts and
// zero plateaus at graded interfaces. Walls are stationary. Pressure
// distances remain explicit caller inputs under the existing graph contract.
// Invalid input returns InvalidArgument, missing support or unsupported
// geometry FailedPrecondition, and unrepresentable measures OutOfRange.
// No input changes and no geometry is returned on failure.
absl::StatusOr<FrozenGradedPlanarLiquidGeometry3D> FreezeGradedSampledLiquidGeometry3D(
    const AdaptiveMacTopology3D& topology,
    Vec3 world_origin_m,
    double step_duration_s,
    std::span<const GradedLiquidCornerSample3D> samples,
    std::span<const std::uint8_t> certified_air_leaves
);

// Evaluate the same tetrahedral liquid trace at a world position inside the
// named canonical leaf. The corner roster has passed the sampled freeze.
absl::StatusOr<double> EvaluateGradedSampledLiquidDistance3D(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const GradedLiquidCornerSample3D> samples,
    std::size_t leaf_ordinal,
    Vec3 position_m
);

}  // namespace fluidity
