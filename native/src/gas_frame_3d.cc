#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "fluidity/host/gas_substep_3d.h"

namespace fluidity {
namespace {

bool IsFiniteVector(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool IsNonnegativeFinite(double value) {
  return std::isfinite(value) && value >= 0.0;
}

template <class Key>
bool HasOrderedKeyTimes(std::span<const Key> keys) {
  double previous_time = -1.0;
  for (const auto& key : keys) {
    if (!IsNonnegativeFinite(key.time) || key.time <= previous_time) {
      return false;
    }
    previous_time = key.time;
  }
  return true;
}

template <class Key>
struct KeyInterval {
  const Key& left;
  const Key& right;
  double fraction;
};

// Keys are checked and nonempty. Endpoint hold applies before and after them.
template <class Key>
KeyInterval<Key> FindKeyInterval(std::span<const Key> keys, double time) {
  const auto right = std::ranges::upper_bound(keys, time, {}, &Key::time);
  if (right == keys.begin()) {
    return {keys.front(), keys.front(), 0.0};
  }
  if (right == keys.end()) {
    return {keys.back(), keys.back(), 0.0};
  }
  const auto left = std::prev(right);
  return {*left, *right, (time - left->time) / (right->time - left->time)};
}

bool IsInside(const GasBoxSource3D& emitter, Vec3 center) {
  return center.x >= emitter.lower.x && center.x < emitter.upper.x && center.y >= emitter.lower.y
      && center.y < emitter.upper.y && center.z >= emitter.lower.z && center.z < emitter.upper.z;
}

}  // namespace

absl::StatusOr<GasFrame3D> AdvanceGasFrame3D(
    const GasUniformDomain3D& domain,
    GasScalars ambient,
    double ambient_pressure,
    Vec3 gravity,
    double start_time,
    double end_time,
    std::size_t substeps,
    std::span<const GasBoxSource3D> emitters,
    std::span<const GasDraftKey3D> draft,
    GasReaction3D reaction,
    absl::FunctionRef<absl::StatusOr<MacPressureProjection3D>(const MacPressureInputs3D&, MacPressureSolverOptions3D)>
        project
) {
  if (!IsNonnegativeFinite(start_time) || !std::isfinite(end_time) || end_time <= start_time || substeps == 0
      || substeps > 4096) {
    return absl::InvalidArgumentError("invalid gas frame time or substep count");
  }
  const double step_duration = (end_time - start_time) / static_cast<double>(substeps);
  if (!std::isfinite(step_duration) || step_duration <= 0.0) {
    return absl::InvalidArgumentError("gas frame step duration is unrepresentable");
  }
  if (!IsNonnegativeFinite(reaction.ignition_temperature) || !IsNonnegativeFinite(reaction.burn_rate)
      || !IsNonnegativeFinite(reaction.heat_per_fuel) || !IsNonnegativeFinite(reaction.soot_yield)
      || reaction.soot_yield > 1.0) {
    return absl::InvalidArgumentError("invalid gas reaction settings");
  }
  for (const auto& emitter : emitters) {
    if (!IsFiniteVector(emitter.lower) || !IsFiniteVector(emitter.upper) || emitter.keys.empty()
        || emitter.lower.x >= emitter.upper.x || emitter.lower.y >= emitter.upper.y
        || emitter.lower.z >= emitter.upper.z || !HasOrderedKeyTimes(emitter.keys)) {
      return absl::InvalidArgumentError("invalid gas emitter bounds or key times");
    }
    for (const auto& key : emitter.keys) {
      if (!IsNonnegativeFinite(key.fuel_rate) || !IsNonnegativeFinite(key.heat_rate)) {
        return absl::InvalidArgumentError("gas emitter rates must be nonnegative and finite");
      }
    }
  }
  if (!HasOrderedKeyTimes(draft) || !std::ranges::all_of(draft, [](const auto& key) {
        return IsFiniteVector(key.acceleration);
      })) {
    return absl::InvalidArgumentError("invalid gas draft key times or accelerations");
  }

  std::vector<std::vector<std::size_t>> emitter_cells(emitters.size());
  for (std::size_t cell = 0; cell < domain.cells.size(); ++cell) {
    const auto center = domain.grid.CellWorldCenter(domain.cells[cell].cell_index);
    if (!center.ok()) {
      return center.status();
    }
    for (std::size_t emitter = 0; emitter < emitters.size(); ++emitter) {
      if (IsInside(emitters[emitter], *center)) {
        emitter_cells[emitter].push_back(cell);
      }
    }
  }
  GasFrame3D frame{
      .cells = domain.cells,
      .markers = domain.markers,
      .face_velocities = domain.face_velocities,
      .fuel_burn_rate = {},
  };
  std::vector<GasCellSource3D> sources(frame.cells.size());
  std::vector<double> face_accelerations(domain.faces.size());
  for (std::size_t substep = 0; substep < substeps; ++substep) {
    const double midpoint = start_time + (static_cast<double>(substep) + 0.5) * step_duration;
    for (std::size_t cell = 0; cell < frame.cells.size(); ++cell) {
      const auto& density = frame.cells[cell].density;
      const double temperature = (density.enthalpy / density.carrier) / kGasSpecificHeat;
      sources[cell] = {
          .rates = {},
          .burn_rate = temperature >= reaction.ignition_temperature ? reaction.burn_rate : 0.0,
          .heat_per_fuel = reaction.heat_per_fuel,
          .soot_yield = reaction.soot_yield,
      };
    }
    for (std::size_t emitter = 0; emitter < emitters.size(); ++emitter) {
      const auto [left, right, fraction] = FindKeyInterval(emitters[emitter].keys, midpoint);
      const double fuel_rate = std::lerp(left.fuel_rate, right.fuel_rate, fraction);
      const double heat_rate = std::lerp(left.heat_rate, right.heat_rate, fraction);
      for (const std::size_t cell : emitter_cells[emitter]) {
        sources[cell].rates.fuel += fuel_rate;
        sources[cell].rates.enthalpy += heat_rate;
      }
    }
    if (!draft.empty()) {
      const auto [left, right, fraction] = FindKeyInterval(draft, midpoint);
      const std::array acceleration{
          std::lerp(left.acceleration.x, right.acceleration.x, fraction),
          std::lerp(left.acceleration.y, right.acceleration.y, fraction),
          std::lerp(left.acceleration.z, right.acceleration.z, fraction),
      };
      for (std::size_t face = 0; face < domain.faces.size(); ++face) {
        const auto axis = static_cast<std::size_t>(domain.faces[face].face_index.axis);
        if (axis >= acceleration.size()) {
          return absl::InvalidArgumentError("invalid gas draft face axis");
        }
        face_accelerations[face] = acceleration[axis];
      }
    }
    auto advanced = AdvanceGasApicSubstep3D(
        domain.grid,
        step_duration,
        frame.cells,
        domain.faces,
        sources,
        ambient,
        ambient_pressure,
        gravity,
        frame.markers,
        frame.face_velocities,
        face_accelerations,
        project
    );
    if (!advanced.ok()) {
      return advanced.status();
    }
    frame.cells = std::move(advanced->gas.cells);
    frame.markers = std::move(advanced->markers);
    frame.face_velocities = std::move(advanced->face_velocities);
    frame.fuel_burn_rate = std::move(advanced->gas.fuel_burn_rate);
  }
  return frame;
}

}  // namespace fluidity
