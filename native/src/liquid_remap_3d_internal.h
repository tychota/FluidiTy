#pragma once

#include "fluidity/host/liquid_remap_3d.h"
#include "mac_pressure_3d_arithmetic.h"

namespace fluidity::liquid_remap_internal {

bool ValidConstituentInventory(const LiquidRemapInventory3D& inventory);

// Accumulates positive volumes in donor order and publishes the stable-ID
// prefix only when all constituent sums and the total are representable.
class ConstituentSums {
 public:
  absl::Status Add(std::uint64_t substance_id, double volume_m3);
  absl::Status Store(LiquidRemapInventory3D& inventory) const;

 private:
  struct Entry {
    std::uint64_t substance_id;
    pressure_internal::arithmetic::CompensatedSum volume;
  };
  std::array<Entry, kMaximumLiquidParticleConstituents3D> sums_{};
  std::size_t count_ = 0;
};

}  // namespace fluidity::liquid_remap_internal
