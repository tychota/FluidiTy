#pragma once

#include "fluidity/host/motion_capture_3d.h"
#include "fluidity/host/surface_cache_3d.h"
#include "fluidity/host/surface_tracking_3d.h"

namespace fluidity {
struct SurfaceReconstructionOptions3D {
  double spacing_m = 0.0125;  // Fixed world-grid spacing, independent of camera.
  double sigma_m = 0.025;  // Fixed Gaussian scale in world metres.
  std::size_t maximum_nodes = 2000000;
  std::size_t maximum_correction_iterations = 32;
  SurfaceCacheLimits3D limits{};
  SurfaceTrackingOptions3D tracking{};
};
struct SurfaceReconstructionResult3D {
  SurfaceFrame3D frame;
  SurfaceMetrics3D raw_metrics;
  SurfaceMetrics3D corrected_metrics;
  bool used_tracker = false;
  std::size_t retained_ids = 0;
  std::size_t deep_liquid_fills = 0;  // Density sign conflicts repaired by authoritative deep occupancy.
  std::size_t deep_air_fills = 0;
  double maximum_source_field_distance_m = 0;  // Final vertices' discrepancy to Simulate's geometric field.
};

// Connects the density reference, accepted carrier, persistent tracking and
// measured correction for a closed material component. Supports stationary
// halfspace contact and a stationary open tank with a planar liquid field.
// Frame zero can also seed up to 16 separate sphere-union components, sharing
// the lattice budget evenly. Nested or intersecting render surfaces fail closed.
// World lattice bounds snap to fixed spacing; they do not change the query
// origin. Projection uses a measured, volume-qualified geometric scaffold rather
// than repeatedly undoing/reapplying the raw field's volume deficit. Uses the
// complete Simulate field/particles to preserve known signs
// farther than 2h from the physical interface. Cached resolved cavities use
// the complete source field and require one inward mesh boundary per cavity,
// one outward boundary, and stable cavity IDs across a tracked interval.
// Does not infer occupancy from
// missing samples. Records field discrepancy rather than certifying it.
// Seed requires one zero-clock sample and no predecessor. Later intervals must
// match the supplied checked SurfaceFrame clock and preserve raw component count
// and Euler characteristic. Moving obstacles and multi-component tracking or repair
// return Unimplemented/FailedPrecondition until their repair adapters qualify.
// Missing support propagates; volume or unsafe geometry failure publishes no
// result. Inputs are borrowed and never modified. At most 65 source samples and
// the supplied source/cache record and geometry work bounds apply.
absl::StatusOr<SurfaceReconstructionResult3D> ReconstructSurfaceFrame3D(
    const MotionFrame3D& source,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& step_options,
    const SurfaceFrame3D* previous,
    const SurfaceReconstructionOptions3D& options
);
}  // namespace fluidity
