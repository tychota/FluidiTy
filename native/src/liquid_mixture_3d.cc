#include "fluidity/host/liquid_mixture_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "fluidity/host/views.h"

namespace fluidity {
namespace {
absl::Status Error(absl::StatusCode code) {
  return absl::Status(code, {});
}

// Contributions share an ID within a group. Keep their incoming order so
// repeated reconstruction of a sample has the same reduction order.
double SumConstituentVolume(std::span<const LiquidConstituentVolume3D>& remaining) {
  const auto substance_id = remaining.front().substance_id;
  std::size_t count = 0;
  double volume = 0.0;
  while (count < remaining.size() && remaining[count].substance_id == substance_id) {
    volume += remaining[count].volume_m3;
    ++count;
  }
  remaining = remaining.subspan(count);
  return volume;
}
}  // namespace

absl::StatusOr<LiquidMixture3D> MixLiquidConstituentVolumes3D(
    std::uint64_t phase_id,
    std::span<const LiquidConstituentMaterial3D> materials,
    std::span<const LiquidConstituentVolume3D> contributions,
    std::span<LiquidConstituentFraction3D> fractions
) {
  const auto disjoint =
      CheckWritableViews({std::as_bytes(fractions)}, {std::as_bytes(materials), std::as_bytes(contributions)});
  if (!disjoint.ok()) {
    return disjoint;
  }
  for (std::size_t index = 0; index < materials.size(); ++index) {
    const auto& material = materials[index];
    if (!std::isfinite(material.density_kg_per_m3) || material.density_kg_per_m3 <= 0.0
        || (index > 0 && materials[index - 1].substance_id >= material.substance_id)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  for (std::size_t index = 0; index < contributions.size(); ++index) {
    const auto& contribution = contributions[index];
    if (!std::isfinite(contribution.volume_m3) || contribution.volume_m3 < 0.0
        || (index > 0 && contributions[index - 1].substance_id > contribution.substance_id)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }

  LiquidMixture3D result{};
  double minimum_positive_volume = std::numeric_limits<double>::infinity();
  auto remaining = contributions;
  auto material = materials.begin();
  while (!remaining.empty()) {
    const auto substance_id = remaining.front().substance_id;
    while (material != materials.end() && material->substance_id < substance_id) {
      ++material;
    }
    if (material == materials.end() || material->substance_id != substance_id || material->phase_id != phase_id) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    const double constituent_volume = SumConstituentVolume(remaining);
    const double constituent_mass = constituent_volume * material->density_kg_per_m3;
    if (constituent_volume > 0.0) {
      if (constituent_mass == 0.0) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      minimum_positive_volume = std::min(minimum_positive_volume, constituent_volume);
    }
    ++result.constituent_count;
    result.volume_m3 += constituent_volume;
    result.mass_kg += constituent_mass;
  }
  if (result.volume_m3 == 0.0) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  result.density_kg_per_m3 = result.mass_kg / result.volume_m3;
  if (!std::isfinite(result.volume_m3) || !std::isfinite(result.mass_kg) || !std::isfinite(result.density_kg_per_m3)
      || result.density_kg_per_m3 <= 0.0 || minimum_positive_volume / result.volume_m3 == 0.0) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  if (fractions.size() < result.constituent_count) {
    return Error(absl::StatusCode::kResourceExhausted);
  }

  remaining = contributions;
  for (auto& fraction : fractions.first(result.constituent_count)) {
    const auto substance_id = remaining.front().substance_id;
    const double constituent_volume = SumConstituentVolume(remaining);
    fraction = {.substance_id = substance_id, .volume_fraction = constituent_volume / result.volume_m3};
  }
  return result;
}

}  // namespace fluidity
