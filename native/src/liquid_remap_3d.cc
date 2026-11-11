#include "fluidity/host/liquid_remap_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "absl/status/status.h"
#include "mac_pressure_3d_arithmetic.h"

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
  for (const auto& child : children) {
    const auto child_lanes = Pack(child);
    if (!Valid(child_lanes)) {
      return absl::InvalidArgumentError("invalid child liquid inventory");
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
  return Unpack(parent_lanes);
}

absl::StatusOr<LiquidRemapChildren3D>
ProlongLiquidInventory3D(const LiquidRemapInventory3D& parent, const LiquidRemapStencil3D& neighbors) {
  const auto parent_lanes = Pack(parent);
  if (!Valid(parent_lanes)) {
    return absl::InvalidArgumentError("invalid parent liquid inventory");
  }
  InventoryLanes minima = parent_lanes;
  InventoryLanes maxima = parent_lanes;
  std::array<InventoryLanes, 3> slopes{};
  for (std::size_t axis = 0; axis < neighbors.size(); ++axis) {
    const auto low = Pack(neighbors[axis][0]);
    const auto high = Pack(neighbors[axis][1]);
    if (!Valid(low) || !Valid(high)) {
      return absl::InvalidArgumentError("invalid neighboring liquid inventory");
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
    children[index] = Unpack(child_lanes);
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
