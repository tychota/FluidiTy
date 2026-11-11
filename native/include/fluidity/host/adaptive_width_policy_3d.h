#pragma once

#include <cstdint>
#include <limits>
#include <span>

#include "absl/status/statusor.h"

namespace fluidity {

// Physical upper bounds in meters. Positive infinity is an unconstrained
// demand except for maximum_width, which must be finite.
struct AdaptiveWidthDemand3D {
  double physical_width = std::numeric_limits<double>::infinity();
  double interface_width = std::numeric_limits<double>::infinity();
  double support_width = std::numeric_limits<double>::infinity();
  double camera_width = std::numeric_limits<double>::infinity();
  double maximum_width = 0;
};

struct AdaptiveCoarsenHaloSample3D {
  AdaptiveWidthDemand3D demand;
  double cavity_width = std::numeric_limits<double>::infinity();  // Smallest local cavity, in m.
  double sheet_width = std::numeric_limits<double>::infinity();  // Smallest local sheet, in m.
  double gap_width = std::numeric_limits<double>::infinity();  // Smallest local gap, in m.
  bool touches_cut_or_contact = false;
  bool support_certified = true;  // Support is complete and its moments are nonsingular.
};

enum class AdaptiveWidthChange3D : std::uint8_t {
  kKeep,
  kRefine,
  kCoarsen,
};

enum class AdaptiveCoarsenVeto3D : std::uint8_t {
  kNone,
  kIncompleteSiblings,
  kIncompleteHalo,
  kWidth,
  kCutOrContact,
  kUncertifiedSupport,
  kUnresolvedFeature,
};

struct AdaptiveWidthDecision3D {
  AdaptiveWidthChange3D change = AdaptiveWidthChange3D::kKeep;
  AdaptiveCoarsenVeto3D veto = AdaptiveCoarsenVeto3D::kNone;
  double eligible_duration = 0.0;  // Seconds to checkpoint after a successful adaptation transaction.
};

// Refine immediately when the cell exceeds its tightest demand. Coarsen
// only a complete sibling family whose parent and actual support/movement
// halo satisfy 2h <= 0.8 h_req continuously for hysteresis_duration seconds.
// Any veto resets eligible_duration. Pass zero accepted_step_duration to
// evaluate without crediting time. A returned refinement or coarsening
// starts a fresh counter when accepted. Rejected adaptation keeps the
// caller's committed counter. Invalid controls or data return InvalidArgument.
// Unrepresentable proposed width returns OutOfRange. Inputs are read-only.
absl::StatusOr<AdaptiveWidthDecision3D> DecideAdaptiveWidth3D(
    double cell_width,
    const AdaptiveWidthDemand3D& cell_demand,
    std::span<const AdaptiveCoarsenHaloSample3D> parent_and_halo,
    bool complete_siblings,
    bool complete_halo,
    double prior_eligible_duration,
    double accepted_step_duration,
    double hysteresis_duration
);

}  // namespace fluidity
