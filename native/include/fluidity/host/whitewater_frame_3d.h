#pragma once

#include "fluidity/host/motion_capture_3d.h"
#include "fluidity/host/whitewater_transport_3d.h"

namespace fluidity {
struct WhitewaterPointRecord3D {
  std::uint64_t id = 0;
  WhitewaterTransportPoint3D point{};
};
struct WhitewaterFrame3D {
  LiquidClock3D clock{};
  WhitewaterEmissionCheckpoint3D emission{};
  std::vector<WhitewaterPointRecord3D> points{};  // Increasing nonzero persistent point ID.
};
struct WhitewaterFrameOptions3D {
  double emission_rate_per_m3_second = 1000;
  double represented_count_per_point = 1;
  WhitewaterTransportOptions3D transport{};  // Cell width and gravity are supplied by Simulate.
  std::size_t maximum_points = 65536;
  std::size_t maximum_source_cells = 4096;
};
struct WhitewaterFrameResult3D {
  WhitewaterFrame3D frame{};
  std::size_t birth_count = 0;
  std::size_t expired_count = 0;
  std::size_t transport_substeps = 0;
  std::size_t bubble_fallback_substeps = 0;
};

// Evolves a persistent dispersed point frame from checked accepted motion and
// birth records for a sphere field. Frame zero has no points or
// remainder. Later frames require the exact preceding clock and emission cursor.
// Incoming plus newborn point count is bounded before transport. Newborns are
// classified at their birth position using ±r, then all points use accepted-time
// transport, hysteresis and foam expiry. Source states/geometry are read-only.
// A stationary halfspace constrains contact. Material sides and other obstacles
// require a different adapter. Unsupported scenes
// return Unimplemented, invalid clocks/records InvalidArgument or
// FailedPrecondition, excess work ResourceExhausted. Sampling errors propagate.
// No partial frame on failure; the supplied checkpoint is unchanged.
absl::StatusOr<WhitewaterFrameResult3D> ReconstructWhitewaterFrame3D(
    const MotionFrame3D& source,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& step_options,
    const WhitewaterFrame3D* previous,
    const WhitewaterFrameOptions3D& options
);
}  // namespace fluidity
