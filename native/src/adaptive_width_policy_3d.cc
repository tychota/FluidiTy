#include "fluidity/host/adaptive_width_policy_3d.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>

#include "absl/status/status.h"

namespace fluidity {
namespace {

std::optional<double> RequestedWidth(const AdaptiveWidthDemand3D& demand) {
  const auto widths = {
      demand.physical_width,
      demand.interface_width,
      demand.support_width,
      demand.camera_width,
      demand.maximum_width,
  };
  const bool valid_widths = std::ranges::all_of(widths, [](double width) {
    return width > 0.0;
  });
  if (!valid_widths || !std::isfinite(demand.maximum_width)) {
    return std::nullopt;
  }
  return std::ranges::min(widths);
}

bool ValidDuration(double duration) {
  return std::isfinite(duration) && duration >= 0.0;
}

AdaptiveCoarsenVeto3D HaloVeto(const AdaptiveCoarsenHaloSample3D& sample, double requested_width, double parent_width) {
  if (parent_width > 0.8 * requested_width) {
    return AdaptiveCoarsenVeto3D::kWidth;
  }
  if (sample.touches_cut_or_contact) {
    return AdaptiveCoarsenVeto3D::kCutOrContact;
  }
  if (!sample.support_certified) {
    return AdaptiveCoarsenVeto3D::kUncertifiedSupport;
  }
  const double smallest_feature_width = std::min({sample.cavity_width, sample.sheet_width, sample.gap_width});
  if (smallest_feature_width / 4 < parent_width) {
    return AdaptiveCoarsenVeto3D::kUnresolvedFeature;
  }
  return AdaptiveCoarsenVeto3D::kNone;
}

}  // namespace

absl::StatusOr<AdaptiveWidthDecision3D> DecideAdaptiveWidth3D(
    double cell_width,
    const AdaptiveWidthDemand3D& cell_demand,
    std::span<const AdaptiveCoarsenHaloSample3D> parent_and_halo,
    bool complete_siblings,
    bool complete_halo,
    double prior_eligible_duration,
    double accepted_step_duration,
    double hysteresis_duration
) {
  if (!std::isfinite(cell_width) || cell_width <= 0.0) {
    return absl::InvalidArgumentError("adaptive cell width must be finite and positive");
  }
  if (!ValidDuration(prior_eligible_duration) || !ValidDuration(accepted_step_duration)
      || !ValidDuration(hysteresis_duration) || prior_eligible_duration > hysteresis_duration) {
    return absl::InvalidArgumentError("adaptive hysteresis durations are invalid");
  }
  const auto cell_requested_width = RequestedWidth(cell_demand);
  if (!cell_requested_width) {
    return absl::InvalidArgumentError("adaptive cell width demands are invalid");
  }
  if (cell_width > *cell_requested_width) {
    return AdaptiveWidthDecision3D{.change = AdaptiveWidthChange3D::kRefine};
  }
  if (!complete_siblings) {
    return AdaptiveWidthDecision3D{.veto = AdaptiveCoarsenVeto3D::kIncompleteSiblings};
  }
  if (!complete_halo || parent_and_halo.empty()) {
    return AdaptiveWidthDecision3D{.veto = AdaptiveCoarsenVeto3D::kIncompleteHalo};
  }
  const double parent_width = 2 * cell_width;
  if (!std::isfinite(parent_width)) {
    return absl::OutOfRangeError("adaptive parent width is unrepresentable");
  }
  auto veto = parent_width > 0.8 * *cell_requested_width ? AdaptiveCoarsenVeto3D::kWidth : AdaptiveCoarsenVeto3D::kNone;
  for (const auto& sample : parent_and_halo) {
    const auto requested_width = RequestedWidth(sample.demand);
    if (!requested_width) {
      return absl::InvalidArgumentError("adaptive halo width demands are invalid");
    }
    const bool valid_features = sample.cavity_width > 0.0 && sample.sheet_width > 0.0 && sample.gap_width > 0.0;
    if (!valid_features) {
      return absl::InvalidArgumentError("adaptive halo feature widths are invalid");
    }
    if (veto == AdaptiveCoarsenVeto3D::kNone) {
      veto = HaloVeto(sample, *requested_width, parent_width);
    }
  }
  if (veto != AdaptiveCoarsenVeto3D::kNone) {
    return AdaptiveWidthDecision3D{.veto = veto};
  }
  const double remaining_duration = hysteresis_duration - prior_eligible_duration;
  if (accepted_step_duration >= remaining_duration) {
    return AdaptiveWidthDecision3D{.change = AdaptiveWidthChange3D::kCoarsen};
  }
  return AdaptiveWidthDecision3D{.eligible_duration = prior_eligible_duration + accepted_step_duration};
}

}  // namespace fluidity
