#pragma once

#include <array>
#include <cstdint>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_mixture_3d.h"

namespace fluidity {

struct LiquidRemapInventory3D {
  double mass_kg = 0;
  double reference_volume_m3 = 0;
  double current_volume_m3 = 0;
  Vec3 momentum_kg_m_per_s{};
  double enthalpy_j = 0;  // Signed relative to the material's enthalpy reference.
  std::uint64_t phase_id = 0;  // Zero for empty inventory or legacy implicit liquid.
  std::uint8_t constituent_count = 0;
  // Positive extensive current volumes, in strictly increasing substance ID order.
  std::array<LiquidConstituentVolume3D, kMaximumLiquidParticleConstituents3D> constituents{};
};

using LiquidRemapChildren3D = std::array<LiquidRemapInventory3D, 8>;
// Same-size neighboring cells, axes x/y/z, with each pair in low/high order.
using LiquidRemapStencil3D = std::array<std::array<LiquidRemapInventory3D, 2>, 3>;

struct LiquidRemapFaceSlot3D {
  std::uint64_t owner_slot = 0;  // Unique among the four owned subfaces, including slot zero.
  int normal_sign = 0;  // +1 for agreement with the coarse normal, -1 for opposition.
};
struct LiquidRemapSubface3D {
  LiquidRemapFaceSlot3D slot{};
  double flux_m3_per_s = 0;  // Integrated flux in the fine face's own orientation.
};

// Sums eight extensive child inventories in lexicographic xyz order (z changes
// fastest). All lanes must be finite and mass and both volumes nonnegative.
// Wet child inventories must agree on explicit phase or legacy mode.
// Invalid data returns InvalidArgument, incompatible phases FailedPrecondition,
// excess constituent IDs ResourceExhausted, and unrepresentable sums OutOfRange.
absl::StatusOr<LiquidRemapInventory3D> RestrictLiquidInventory3D(const LiquidRemapChildren3D& children);

// Prolongs one parent into eight equal-volume Cartesian child cells at offsets
// ±1/4 of a parent cell width on each axis, in the same order as restriction.
// The limited parent-scale reconstruction stays within stencil extrema. Child
// extensive values divide it by eight. Incompatible wet neighbors use the
// parent state at the phase boundary. Constituent volumes use the parent
// fractions and the limited child current volumes. Uses the same input rules as restriction
// and returns OutOfRange for unrepresentable arithmetic or inventory loss.
// These operations return new values. Inputs are read-only on every path.
absl::StatusOr<LiquidRemapChildren3D>
ProlongLiquidInventory3D(const LiquidRemapInventory3D& parent, const LiquidRemapStencil3D& neighbors);

// Restricts four integrated subface fluxes to the coarse normal, summing in
// ascending unique owner-slot order. Nonfinite input, duplicate slots or
// invalid signs return InvalidArgument. Nonfinite sums return OutOfRange.
absl::StatusOr<double> RestrictLiquidFaceFlux3D(const std::array<LiquidRemapSubface3D, 4>& subfaces);

// Distributes a finite integrated coarse-normal flux over four equal-area
// subfaces, in the supplied slot order and each slot's own normal. Inexact
// quartering returns OutOfRange. Other input rules match restriction.
absl::StatusOr<std::array<LiquidRemapSubface3D, 4>>
ProlongLiquidFaceFlux3D(double coarse_flux_m3_per_s, const std::array<LiquidRemapFaceSlot3D, 4>& slots);

}  // namespace fluidity
