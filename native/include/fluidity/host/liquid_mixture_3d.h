#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {

inline constexpr std::size_t kMaximumLiquidParticleConstituents3D = 8;

struct LiquidPhasePair3D {
  std::uint64_t phase_a;
  std::uint64_t phase_b;  // Strictly greater than phase_a.
  double surface_tension;  // Nonnegative N/m at their shared interface.
};

struct LiquidConstituentMaterial3D {
  std::uint64_t substance_id;
  std::uint64_t phase_id;  // Shared by miscible substances.
  double density_kg_per_m3;  // Positive density at the supplied material state.
  std::optional<double> dynamic_viscosity_pa_s{};  // Nonnegative. Absent uses the scene coefficient.
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

// Checks a canonical, unit-sum fraction span. Empty means implicit legacy
// liquid and requires phase ID zero. Active fractions are strictly positive.
bool ValidLiquidComposition3D(std::uint64_t phase_id, std::span<const LiquidConstituentFraction3D> fractions);

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

// Checks canonical pair order, finite nonnegative tension and membership of
// both endpoints in the supplied substance registry. Invalid records return
// InvalidArgument. Temporary phase ordering may return ResourceExhausted.
absl::Status ValidateLiquidPhasePairs3D(
    std::span<const LiquidPhasePair3D> pairs,
    std::span<const LiquidConstituentMaterial3D> materials
);

// Looks up a distinct phase pair in either orientation. Returns nullptr when
// no pair is configured. The sorted table must outlive the returned borrow.
const LiquidPhasePair3D*
FindLiquidPhasePair3D(std::span<const LiquidPhasePair3D> pairs, std::uint64_t first_phase, std::uint64_t second_phase);

// Validates positive finite densities and strictly ordered substance IDs.
bool ValidLiquidConstituentMaterials3D(std::span<const LiquidConstituentMaterial3D> materials);

// Checks that every active constituent resolves to its particle's phase in
// a previously validated registry. Implicit legacy particles need no entry.
bool LiquidParticleUsesRegisteredConstituents3D(
    const LiquidParticleState3D& particle,
    std::span<const LiquidConstituentMaterial3D> materials
);

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
