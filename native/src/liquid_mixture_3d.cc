#include "fluidity/host/liquid_mixture_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

#include "fluidity/host/liquid_state_3d.h"
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

absl::Status ValidateLiquidPhasePairs3D(
    std::span<const LiquidPhasePair3D> pairs,
    std::span<const LiquidConstituentMaterial3D> materials
) {
  if (pairs.empty()) {
    return absl::OkStatus();
  }
  auto phases = LiquidRecords3D<std::uint64_t>::Create(materials.size());
  if (!phases.ok()) {
    return phases.status();
  }
  const auto resized = phases->Resize(materials.size());
  if (!resized.ok()) {
    return resized;
  }
  std::ranges::transform(materials, phases->values().begin(), &LiquidConstituentMaterial3D::phase_id);
  std::ranges::sort(phases->values());
  const auto has_phase = [&](std::uint64_t phase) {
    return std::ranges::binary_search(phases->values(), phase);
  };
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    const auto& pair = pairs[index];
    if (pair.phase_a >= pair.phase_b || !std::isfinite(pair.surface_tension) || pair.surface_tension < 0.0
        || !has_phase(pair.phase_a) || !has_phase(pair.phase_b)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    if (index > 0) {
      const auto& previous = pairs[index - 1];
      if (std::pair{previous.phase_a, previous.phase_b} >= std::pair{pair.phase_a, pair.phase_b}) {
        return Error(absl::StatusCode::kInvalidArgument);
      }
    }
  }
  return absl::OkStatus();
}

const LiquidPhasePair3D*
FindLiquidPhasePair3D(std::span<const LiquidPhasePair3D> pairs, std::uint64_t first_phase, std::uint64_t second_phase) {
  if (first_phase == second_phase) {
    return nullptr;
  }
  const std::pair key{std::min(first_phase, second_phase), std::max(first_phase, second_phase)};
  const auto found = std::ranges::lower_bound(pairs, key, {}, [](const LiquidPhasePair3D& pair) {
    return std::pair{pair.phase_a, pair.phase_b};
  });
  return found != pairs.end() && std::pair{found->phase_a, found->phase_b} == key ? &*found : nullptr;
}

bool ValidLiquidComposition3D(std::uint64_t phase_id, std::span<const LiquidConstituentFraction3D> fractions) {
  if (fractions.empty()) {
    return phase_id == 0;
  }
  double sum = 0.0;
  for (std::size_t index = 0; index < fractions.size(); ++index) {
    const auto& entry = fractions[index];
    if (!std::isfinite(entry.volume_fraction) || entry.volume_fraction <= 0.0 || entry.volume_fraction > 1.0
        || (index > 0 && fractions[index - 1].substance_id >= entry.substance_id)) {
      return false;
    }
    sum += entry.volume_fraction;
  }
  constexpr double kTolerance = 8.0 * std::numeric_limits<double>::epsilon();
  return std::abs(sum - 1.0) <= kTolerance;
}

bool ValidLiquidConstituentMaterials3D(std::span<const LiquidConstituentMaterial3D> materials) {
  for (std::size_t index = 0; index < materials.size(); ++index) {
    const auto& material = materials[index];
    if (!std::isfinite(material.density_kg_per_m3) || material.density_kg_per_m3 <= 0.0
        || (material.dynamic_viscosity_pa_s
            && (!std::isfinite(*material.dynamic_viscosity_pa_s) || *material.dynamic_viscosity_pa_s < 0.0))
        || (index > 0 && materials[index - 1].substance_id >= material.substance_id)) {
      return false;
    }
  }
  return true;
}

bool LiquidParticleUsesRegisteredConstituents3D(
    const LiquidParticleState3D& particle,
    std::span<const LiquidConstituentMaterial3D> materials
) {
  if (!ValidLiquidParticleComposition3D(particle)) {
    return false;
  }
  for (const auto& constituent : std::span(particle.constituents).first(particle.constituent_count)) {
    const auto found =
        std::ranges::lower_bound(materials, constituent.substance_id, {}, &LiquidConstituentMaterial3D::substance_id);
    if (found == materials.end() || found->substance_id != constituent.substance_id
        || found->phase_id != particle.phase_id) {
      return false;
    }
  }
  return true;
}

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
  if (!ValidLiquidConstituentMaterials3D(materials)) {
    return Error(absl::StatusCode::kInvalidArgument);
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

absl::Status DepositLiquidConstituentVolumes3D(
    UniformMacGeometry3D geometry,
    std::uint64_t phase_id,
    std::span<const CellIndex3D> cells,
    std::span<const LiquidConstituentMaterial3D> materials,
    std::span<const LiquidParticleState3D> particles,
    std::span<LiquidConstituentVolume3D> cell_volumes
) {
  const auto extent = CheckedExtentProduct(cells.size(), materials.size());
  if (!extent.ok()) {
    return extent.status();
  }
  const auto origin = geometry.world_origin;
  if (cell_volumes.size() != *extent || !std::isfinite(geometry.cell_width) || geometry.cell_width <= 0.0
      || !std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const auto disjoint = CheckWritableViews(
      {std::as_bytes(cell_volumes)},
      {std::as_bytes(cells), std::as_bytes(materials), std::as_bytes(particles)}
  );
  if (!disjoint.ok()) {
    return disjoint;
  }
  const auto cell_key = [](CellIndex3D index) {
    return std::tuple{index.z, index.y, index.x};
  };
  for (std::size_t index = 1; index < cells.size(); ++index) {
    if (cell_key(cells[index - 1]) >= cell_key(cells[index])) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  for (std::size_t index = 0; index < materials.size(); ++index) {
    const auto& material = materials[index];
    if (material.phase_id != phase_id || !std::isfinite(material.density_kg_per_m3) || material.density_kg_per_m3 <= 0.0
        || (material.dynamic_viscosity_pa_s
            && (!std::isfinite(*material.dynamic_viscosity_pa_s) || *material.dynamic_viscosity_pa_s < 0.0))
        || (index > 0 && materials[index - 1].substance_id >= material.substance_id)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    for (std::size_t material = 0; material < materials.size(); ++material) {
      cell_volumes[cell * materials.size() + material] = {materials[material].substance_id, 0.0};
    }
  }

  for (const auto& particle : particles) {
    if (!ValidLiquidParticleComposition3D(particle)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    if (particle.constituent_count == 0 || particle.phase_id != phase_id) {
      continue;
    }
    if (!std::isfinite(particle.reference_volume) || particle.reference_volume <= 0.0
        || !std::isfinite(particle.volume_ratio) || particle.volume_ratio <= 0.0) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    const double current_volume = particle.reference_volume * particle.volume_ratio;
    if (!std::isfinite(current_volume) || current_volume <= 0.0) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    std::array<std::size_t, kMaximumLiquidParticleConstituents3D> material_ordinals;
    for (std::size_t index = 0; index < particle.constituent_count; ++index) {
      const auto id = particle.constituents[index].substance_id;
      const auto found = std::ranges::lower_bound(materials, id, {}, &LiquidConstituentMaterial3D::substance_id);
      if (found == materials.end() || found->substance_id != id) {
        return Error(absl::StatusCode::kInvalidArgument);
      }
      material_ordinals[index] = static_cast<std::size_t>(found - materials.begin());
    }

    const auto position = particle.world_position;
    const std::array coordinates{
        (position.x - origin.x) / geometry.cell_width - 0.5,
        (position.y - origin.y) / geometry.cell_width - 0.5,
        (position.z - origin.z) / geometry.cell_width - 0.5,
    };
    constexpr auto kMinimum = std::numeric_limits<std::int32_t>::min();
    constexpr auto kMaximum = std::numeric_limits<std::int32_t>::max();
    std::array<std::int64_t, 3> lower;
    std::array<double, 3> fraction;
    for (std::size_t axis = 0; axis < coordinates.size(); ++axis) {
      if (!std::isfinite(coordinates[axis]) || coordinates[axis] < kMinimum || coordinates[axis] > kMaximum) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      const double floor = std::floor(coordinates[axis]);
      lower[axis] = static_cast<std::int64_t>(floor);
      fraction[axis] = coordinates[axis] - floor;
    }
    for (std::size_t corner = 0; corner < 8; ++corner) {
      const bool high_x = (corner & 1) != 0;
      const bool high_y = (corner & 2) != 0;
      const bool high_z = (corner & 4) != 0;
      const double x_weight = high_x ? fraction[0] : 1.0 - fraction[0];
      const double y_weight = high_y ? fraction[1] : 1.0 - fraction[1];
      const double z_weight = high_z ? fraction[2] : 1.0 - fraction[2];
      if (x_weight == 0.0 || y_weight == 0.0 || z_weight == 0.0) {
        continue;
      }
      const CellIndex3D index{
          static_cast<std::int32_t>(lower[0] + high_x),
          static_cast<std::int32_t>(lower[1] + high_y),
          static_cast<std::int32_t>(lower[2] + high_z),
      };
      const auto key = cell_key(index);
      const auto cell = std::ranges::lower_bound(cells, key, {}, cell_key);
      if (cell == cells.end() || cell_key(*cell) != key) {
        return Error(absl::StatusCode::kFailedPrecondition);
      }
      const auto row = static_cast<std::size_t>(cell - cells.begin()) * materials.size();
      const double weighted_volume = x_weight * y_weight * z_weight * current_volume;
      for (std::size_t constituent = 0; constituent < particle.constituent_count; ++constituent) {
        const double contribution = particle.constituents[constituent].volume_fraction * weighted_volume;
        auto& volume = cell_volumes[row + material_ordinals[constituent]].volume_m3;
        volume += contribution;
        if (contribution == 0.0 || !std::isfinite(volume)) {
          return Error(absl::StatusCode::kOutOfRange);
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace fluidity
