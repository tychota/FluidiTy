#include "fluidity/host/liquid_remap_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "absl/status/status.h"
#include "liquid_remap_3d_internal.h"

namespace fluidity::liquid_remap_internal {

bool ValidConstituentInventory(const LiquidRemapInventory3D& inventory) {
  if (inventory.constituent_count == 0) {
    return inventory.phase_id == 0;
  }
  if (inventory.phase_id == 0 || inventory.constituent_count > inventory.constituents.size()
      || !std::isfinite(inventory.current_volume_m3) || inventory.current_volume_m3 <= 0) {
    return false;
  }
  pressure_internal::arithmetic::CompensatedSum volume;
  std::uint64_t previous_id = 0;
  for (const auto& constituent : std::span(inventory.constituents).first(inventory.constituent_count)) {
    if (constituent.substance_id <= previous_id || !std::isfinite(constituent.volume_m3)
        || constituent.volume_m3 <= 0) {
      return false;
    }
    previous_id = constituent.substance_id;
    volume.Add(constituent.volume_m3);
  }
  const double total = volume.Total();
  constexpr double tolerance = 64 * std::numeric_limits<double>::epsilon();
  return std::isfinite(total)
      && (total == inventory.current_volume_m3
          || (inventory.current_volume_m3 >= std::numeric_limits<double>::min()
              && std::abs((total - inventory.current_volume_m3) / inventory.current_volume_m3) <= tolerance));
}

absl::Status ConstituentSums::Add(std::uint64_t substance_id, double volume_m3) {
  if (!std::isfinite(volume_m3) || volume_m3 <= 0) {
    return absl::OutOfRangeError("liquid remap constituent contribution is unrepresentable");
  }
  const auto end = sums_.begin() + static_cast<std::ptrdiff_t>(count_);
  const auto found = std::lower_bound(sums_.begin(), end, substance_id, [](const Entry& entry, std::uint64_t id) {
    return entry.substance_id < id;
  });
  if (found != end && found->substance_id == substance_id) {
    found->volume.Add(volume_m3);
  } else {
    if (count_ == sums_.size()) {
      return absl::ResourceExhaustedError("liquid remap constituent capacity exceeded");
    }
    std::move_backward(found, end, end + 1);
    *found = {.substance_id = substance_id, .volume = {}};
    found->volume.Add(volume_m3);
    ++count_;
  }
  return absl::OkStatus();
}

absl::Status ConstituentSums::Store(LiquidRemapInventory3D& inventory) const {
  auto candidate = inventory;
  candidate.constituent_count = 0;
  for (const auto& [substance_id, sum] : std::span(sums_).first(count_)) {
    candidate.constituents[candidate.constituent_count++] = {substance_id, sum.Total()};
  }
  if (!ValidConstituentInventory(candidate)) {
    return absl::OutOfRangeError("liquid remap constituent sums disagree with total volume");
  }
  inventory = candidate;
  return absl::OkStatus();
}

}  // namespace fluidity::liquid_remap_internal
namespace fluidity {
namespace {

using pressure_internal::arithmetic::CompensatedSum;
using InventoryLanes = std::array<double, 7>;

constexpr double kConservationRoundoff = 64 * std::numeric_limits<double>::epsilon();

InventoryLanes Pack(const LiquidRemapInventory3D& inventory) {
  const Vec3 momentum = inventory.momentum_kg_m_per_s;
  return {
      inventory.mass_kg,
      inventory.reference_volume_m3,
      inventory.current_volume_m3,
      momentum.x,
      momentum.y,
      momentum.z,
      inventory.enthalpy_j,
  };
}

LiquidRemapInventory3D Unpack(const InventoryLanes& lanes) {
  return {
      .mass_kg = lanes[0],
      .reference_volume_m3 = lanes[1],
      .current_volume_m3 = lanes[2],
      .momentum_kg_m_per_s = {lanes[3], lanes[4], lanes[5]},
      .enthalpy_j = lanes[6],
  };
}

bool Valid(const InventoryLanes& lanes) {
  return lanes[0] >= 0 && lanes[1] >= 0 && lanes[2] >= 0 && std::all_of(lanes.begin(), lanes.end(), [](double value) {
           return std::isfinite(value);
         });
}

double Minmod(double left, double right) {
  if (left > 0 && right > 0) {
    return std::min(left, right);
  }
  if (left < 0 && right < 0) {
    return std::max(left, right);
  }
  return 0;
}

// A nonzero subnormal parent inventory must survive exactly, even when it is
// the small net result of large signed children.
bool Conserves(double parent, double child_sum, double largest_child_magnitude) {
  if (child_sum == parent) {
    return true;
  }
  const double parent_magnitude = std::abs(parent);
  const double scale = std::max(parent_magnitude, largest_child_magnitude);
  constexpr double kSmallestNormal = std::numeric_limits<double>::min();
  if (scale < kSmallestNormal || (parent_magnitude > 0 && parent_magnitude < kSmallestNormal)) {
    return false;
  }
  return std::abs(child_sum - parent) <= kConservationRoundoff * scale;
}

absl::StatusOr<std::array<std::size_t, 4>> OrderFaceSlots(const std::array<LiquidRemapFaceSlot3D, 4>& slots) {
  for (const auto slot : slots) {
    if (slot.normal_sign != -1 && slot.normal_sign != 1) {
      return absl::InvalidArgumentError("subface normal sign must be +1 or -1");
    }
  }
  std::array<std::size_t, 4> order{0, 1, 2, 3};
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return slots[a].owner_slot < slots[b].owner_slot;
  });
  for (std::size_t index = 1; index < order.size(); ++index) {
    if (slots[order[index]].owner_slot == slots[order[index - 1]].owner_slot) {
      return absl::InvalidArgumentError("subfaces must have distinct owner slots");
    }
  }
  return order;
}

}  // namespace

absl::StatusOr<LiquidRemapInventory3D> RestrictLiquidInventory3D(const LiquidRemapChildren3D& children) {
  std::array<CompensatedSum, InventoryLanes{}.size()> sums{};
  liquid_remap_internal::ConstituentSums constituent_sums;
  std::optional<std::uint64_t> wet_phase;
  for (const auto& child : children) {
    const auto child_lanes = Pack(child);
    if (!Valid(child_lanes) || !liquid_remap_internal::ValidConstituentInventory(child)) {
      return absl::InvalidArgumentError("invalid child liquid inventory");
    }
    if (child.current_volume_m3 > 0) {
      if (wet_phase && *wet_phase != child.phase_id) {
        return absl::FailedPreconditionError("cannot restrict liquid inventories across phases");
      }
      wet_phase = child.phase_id;
    }
    for (const auto& constituent : std::span(child.constituents).first(child.constituent_count)) {
      const auto status = constituent_sums.Add(constituent.substance_id, constituent.volume_m3);
      if (!status.ok()) {
        return status;
      }
    }
    for (std::size_t lane = 0; lane < sums.size(); ++lane) {
      sums[lane].Add(child_lanes[lane]);
    }
  }
  InventoryLanes parent_lanes{};
  for (std::size_t lane = 0; lane < sums.size(); ++lane) {
    parent_lanes[lane] = sums[lane].Total();
  }
  if (!Valid(parent_lanes)) {
    return absl::OutOfRangeError("restricted liquid inventory is unrepresentable");
  }
  auto parent = Unpack(parent_lanes);
  parent.phase_id = wet_phase.value_or(0);
  const auto status = constituent_sums.Store(parent);
  if (!status.ok()) {
    return status;
  }
  return parent;
}

absl::StatusOr<LiquidRemapChildren3D>
ProlongLiquidInventory3D(const LiquidRemapInventory3D& parent, const LiquidRemapStencil3D& neighbors) {
  const auto parent_lanes = Pack(parent);
  if (!Valid(parent_lanes) || !liquid_remap_internal::ValidConstituentInventory(parent)) {
    return absl::InvalidArgumentError("invalid parent liquid inventory");
  }
  InventoryLanes minima = parent_lanes;
  InventoryLanes maxima = parent_lanes;
  std::array<InventoryLanes, 3> slopes{};
  for (std::size_t axis = 0; axis < neighbors.size(); ++axis) {
    auto low = Pack(neighbors[axis][0]);
    auto high = Pack(neighbors[axis][1]);
    if (!Valid(low) || !Valid(high) || !liquid_remap_internal::ValidConstituentInventory(neighbors[axis][0])
        || !liquid_remap_internal::ValidConstituentInventory(neighbors[axis][1])) {
      return absl::InvalidArgumentError("invalid neighboring liquid inventory");
    }
    if (neighbors[axis][0].current_volume_m3 > 0 && neighbors[axis][0].phase_id != parent.phase_id) {
      low = parent_lanes;
    }
    if (neighbors[axis][1].current_volume_m3 > 0 && neighbors[axis][1].phase_id != parent.phase_id) {
      high = parent_lanes;
    }
    for (std::size_t lane = 0; lane < parent_lanes.size(); ++lane) {
      minima[lane] = std::min({minima[lane], low[lane], high[lane]});
      maxima[lane] = std::max({maxima[lane], low[lane], high[lane]});
      const double low_difference = parent_lanes[lane] - low[lane];
      const double high_difference = high[lane] - parent_lanes[lane];
      if (!std::isfinite(low_difference) || !std::isfinite(high_difference)) {
        return absl::OutOfRangeError("liquid remap slope is unrepresentable");
      }
      slopes[axis][lane] = Minmod(low_difference, high_difference);
    }
  }

  // Symmetric octants cancel slope offsets before rounding. Use a common
  // limiter for all lanes, then verify the inventories actually stored.
  std::array<InventoryLanes, 8> offsets{};
  double limiter = 1;
  for (std::size_t lane = 0; lane < parent_lanes.size(); ++lane) {
    const double lower_room = parent_lanes[lane] - minima[lane];
    const double upper_room = maxima[lane] - parent_lanes[lane];
    for (std::size_t index = 0; index < offsets.size(); ++index) {
      const double dx = index & 4 ? 0.25 : -0.25;
      const double dy = index & 2 ? 0.25 : -0.25;
      const double dz = index & 1 ? 0.25 : -0.25;
      const double offset = (dx * slopes[0][lane] + dy * slopes[1][lane]) + dz * slopes[2][lane];
      offsets[index][lane] = offset;
      if (offset > upper_room) {
        limiter = std::min(limiter, upper_room / offset);
      }
      if (offset < -lower_room) {
        limiter = std::min(limiter, -lower_room / offset);
      }
    }
  }

  LiquidRemapChildren3D children{};
  InventoryLanes largest_child_magnitude{};
  for (std::size_t index = 0; index < children.size(); ++index) {
    InventoryLanes child_lanes{};
    for (std::size_t lane = 0; lane < child_lanes.size(); ++lane) {
      const double child_inventory = parent_lanes[lane] / 8 + (limiter / 8) * offsets[index][lane];
      child_lanes[lane] = std::clamp(child_inventory, minima[lane] / 8, maxima[lane] / 8);
      largest_child_magnitude[lane] = std::max(largest_child_magnitude[lane], std::abs(child_lanes[lane]));
    }
    auto& child = children[index];
    child = Unpack(child_lanes);
    if (parent.constituent_count > 0 && child.current_volume_m3 > 0) {
      child.phase_id = parent.phase_id;
      child.constituent_count = parent.constituent_count;
      const double volume_share = child.current_volume_m3 / parent.current_volume_m3;
      for (std::size_t constituent = 0; constituent < parent.constituent_count; ++constituent) {
        const auto& owned = parent.constituents[constituent];
        child.constituents[constituent] = {owned.substance_id, owned.volume_m3 * volume_share};
      }
      if (!liquid_remap_internal::ValidConstituentInventory(child)) {
        return absl::OutOfRangeError("liquid prolongation cannot represent child constituent inventory");
      }
    }
  }
  const auto restricted = RestrictLiquidInventory3D(children);
  if (!restricted.ok()) {
    return restricted.status();
  }
  const auto restricted_lanes = Pack(*restricted);
  for (std::size_t lane = 0; lane < parent_lanes.size(); ++lane) {
    if (!Conserves(parent_lanes[lane], restricted_lanes[lane], largest_child_magnitude[lane])) {
      return absl::OutOfRangeError("prolongation cannot conserve the liquid inventory");
    }
  }
  if (parent.constituent_count != restricted->constituent_count) {
    return absl::OutOfRangeError("liquid prolongation lost a constituent");
  }
  for (std::size_t index = 0; index < parent.constituent_count; ++index) {
    const double expected = parent.constituents[index].volume_m3;
    const double actual = restricted->constituents[index].volume_m3;
    if (!Conserves(expected, actual, expected)) {
      return absl::OutOfRangeError("liquid prolongation cannot conserve a constituent volume");
    }
  }
  return children;
}

absl::StatusOr<double> RestrictLiquidFaceFlux3D(const std::array<LiquidRemapSubface3D, 4>& subfaces) {
  std::array<LiquidRemapFaceSlot3D, 4> slots{};
  for (std::size_t index = 0; index < slots.size(); ++index) {
    if (!std::isfinite(subfaces[index].flux_m3_per_s)) {
      return absl::InvalidArgumentError("subface flux must be finite");
    }
    slots[index] = subfaces[index].slot;
  }
  const auto order = OrderFaceSlots(slots);
  if (!order.ok()) {
    return order.status();
  }
  CompensatedSum coarse_flux;
  for (const std::size_t index : *order) {
    coarse_flux.Add(slots[index].normal_sign * subfaces[index].flux_m3_per_s);
  }
  const double flux_m3_per_s = coarse_flux.Total();
  if (!std::isfinite(flux_m3_per_s)) {
    return absl::OutOfRangeError("restricted liquid face flux is unrepresentable");
  }
  return flux_m3_per_s;
}

absl::StatusOr<std::array<LiquidRemapSubface3D, 4>>
ProlongLiquidFaceFlux3D(double coarse_flux_m3_per_s, const std::array<LiquidRemapFaceSlot3D, 4>& slots) {
  if (!std::isfinite(coarse_flux_m3_per_s)) {
    return absl::InvalidArgumentError("coarse face flux must be finite");
  }
  const auto order = OrderFaceSlots(slots);
  if (!order.ok()) {
    return order.status();
  }
  const double subface_flux_m3_per_s = coarse_flux_m3_per_s / 4;
  if (4 * subface_flux_m3_per_s != coarse_flux_m3_per_s) {
    return absl::OutOfRangeError("prolongation cannot conserve the liquid face flux");
  }
  std::array<LiquidRemapSubface3D, 4> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = {
        .slot = slots[index],
        .flux_m3_per_s = slots[index].normal_sign * subface_flux_m3_per_s,
    };
  }
  return result;
}

}  // namespace fluidity
