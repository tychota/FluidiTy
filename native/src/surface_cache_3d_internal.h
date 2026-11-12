#pragma once

#include "fluidity/host/surface_cache_3d.h"

namespace fluidity::surface_cache_internal {

// Checks closed exterior and phase-region topology before retessellation can
// transfer IDs. Both frames must carry the same phase pair. current_exterior
// contains the measurements for current.mesh. A topology change that needs
// lineage returns FailedPrecondition. Neither input frame is modified.
absl::Status CheckSharedSurfaceTopology3D(
    const SurfaceFrame3D& previous,
    const SurfaceFrame3D& current,
    const SurfaceMetrics3D& current_exterior,
    SurfaceCacheLimits3D limits
);

}  // namespace fluidity::surface_cache_internal
