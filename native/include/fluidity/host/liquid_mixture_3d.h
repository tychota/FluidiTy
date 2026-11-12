#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {

struct LiquidConstituentMaterial3D {
  std::uint64_t substance_id;
  std::uint64_t phase_id;  // Shared by miscible substances.
  double density_kg_per_m3;  // Positive density at the supplied material state.
};

struct LiquidConstituentVolume3D {
  std::uint64_t substance_id;
  double volume_m3;  // Nonnegative current volume, including the donor's transfer weight.
};

struct LiquidConstituentFraction3D {
  std::uint64_t substance_id;
  double volume_fraction;  // Constituent current volume divided by total current volume.
  bool operator==(const LiquidConstituentFraction3D&) const = default;
};

struct LiquidMixture3D {
  std::size_t constituent_count;
  double volume_m3;
  double mass_kg;
  double density_kg_per_m3;
};

// Mixes one phase under additive volumes. The material table is strictly
// ordered by substance ID. Volume contributions are ordered by substance ID,
// with duplicates summed in supplied order. Every contribution must refer to
// a material in phase_id. Emits one fraction per distinct contributing ID,
// including zero-volume entries, in ascending order. The output must not
// alias inputs and remains untouched on failure. InvalidArgument covers
// malformed input, cross-phase contributions and zero total volume,
// ResourceExhausted covers short output, and OutOfRange covers arithmetic.
absl::StatusOr<LiquidMixture3D> MixLiquidConstituentVolumes3D(
    std::uint64_t phase_id,
    std::span<const LiquidConstituentMaterial3D> materials,
    std::span<const LiquidConstituentVolume3D> contributions,
    std::span<LiquidConstituentFraction3D> fractions
);

struct LiquidParticleState3D;

// Deposits f * J * V0 with an eight-point cell-centered linear tent on the
// supplied MAC lattice. Cells are unique in (z,y,x) order, and materials are
// unique in substance ID order and belong to phase_id. Implicit particles and
// explicit particles in other phases contribute nothing. Every positive
// stencil weight needs a represented cell. Each output row contains one
// record per material in material order, including zeros. Output is scratch
// disjoint from all inputs and may change on error. InvalidArgument covers
// malformed input, incompatible materials, aliasing, or a wrong extent,
// FailedPrecondition covers missing support, and OutOfRange covers arithmetic.
absl::Status DepositLiquidConstituentVolumes3D(
    UniformMacGeometry3D geometry,
    std::uint64_t phase_id,
    std::span<const CellIndex3D> cells,
    std::span<const LiquidConstituentMaterial3D> materials,
    std::span<const LiquidParticleState3D> particles,
    std::span<LiquidConstituentVolume3D> cell_volumes
);

}  // namespace fluidity
