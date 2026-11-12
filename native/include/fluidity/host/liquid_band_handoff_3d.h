#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_remap_3d.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity {

enum class LiquidHandoffOccupancy3D {
  kMissing,
  kPartial,
  kFullLiquid,
  kNotLiquid,
};

struct LiquidHandoffCell3D {
  Vec3 world_lower_corner{};  // Meters, Cartesian and axis-aligned.
  double cell_width = 0;  // Meters.
  LiquidHandoffOccupancy3D occupancy = LiquidHandoffOccupancy3D::kMissing;  // Authoritative simulation occupancy.
};

// Caller-certified destination state for one phase throughout this cell.
// A supplied carrier asserts complete admissible support and an affine
// continuation on that support, separate from the reconstructed derivative L.
struct LiquidHandoffCarrier3D {
  double current_density = 0;  // kg/m³.
  double volume_ratio = 0;  // Dimensionless J from the destination EOS.
  Vec3 cell_center_velocity{};  // m/s.
  Mat3 velocity_slopes{};  // APIC C, inverse seconds.
  std::uint32_t tag = 0;  // Packed substance/phase/flags.
  std::int64_t side_tag = 0;  // Material/body side.
  std::uint64_t phase_id = 0;  // Stable explicit phase, or zero for implicit liquid.
};

class LiquidBandReseed3D {
 public:
  static constexpr std::size_t kMaxParticles = 8;

  // The sole mass, volume and momentum owner after this proposal replaces its
  // source core inventory. The view borrows this proposal.
  std::span<const LiquidParticleState3D> particles() const {
    return {particles_.data(), particle_count_};
  }

  std::uint64_t next_particle_id() const {
    return next_particle_id_;
  }

  // Signed enthalpy remains in the unchanged cell ledger. Particles have no enthalpy lane.
  double enthalpy_j() const {
    return enthalpy_j_;
  }

 private:
  LiquidBandReseed3D() = default;

  friend absl::StatusOr<LiquidBandReseed3D> ReseedLiquidCoreCell3D(
      const LiquidHandoffCell3D&,
      const LiquidRemapInventory3D&,
      const std::optional<LiquidHandoffCarrier3D>&,
      std::size_t,
      std::size_t,
      std::uint64_t
  );

  std::array<LiquidParticleState3D, kMaxParticles> particles_{};
  std::size_t particle_count_ = 0;
  std::uint64_t next_particle_id_ = 0;
  double enthalpy_j_ = 0;
};

// Proposes retirement of all band-owned particles in one full liquid cell.
// The nonempty input has unique nonzero IDs and positions in the half-open cell.
// The result sums their kg, V0, J*V0 and kg*m/s in input order, carrying signed
// enthalpy from the cell ledger. Explicit fractions become extensive
// current constituent volumes in stable substance order. The caller installs this inventory in place
// of the retiring particles, without advancing its ID counter.
//
// Both handoffs require a present certified carrier. FailedPrecondition means
// occupancy, material/side, carrier availability or inventory consistency
// refused the handoff. InvalidArgument means malformed input. OutOfRange
// means unrepresentable arithmetic. Inputs are untouched on every path.
absl::StatusOr<LiquidRemapInventory3D> RetireLiquidBandCell3D(
    const LiquidHandoffCell3D& cell,
    std::span<const LiquidParticleState3D> band_particles,
    double enthalpy_j,
    const std::optional<LiquidHandoffCarrier3D>& destination_carrier
);

// Proposes a replacement band owner for one full core cell. The source ledger
// provides V, and the carrier must agree with its mass, V0 and momentum.
// Each particle receives m=ρV/n and V0=V/(nJ), and v and C from the carrier.
// Explicit constituents receive the same fractions derived from the core inventory.
// Seeds are the center (for odd counts), then opposite quarter-cell octants in
// pairs, using ascending low octant ordinal with x changing fastest. New
// records use consecutive IDs from the caller's first unused ID and start PIC.
//
// ResourceExhausted means the count exceeds either bound (including available
// slots). OutOfRange also covers the next-ID counter or any reseed inventory
// that cannot be represented within 64 binary64 epsilons relative to its
// source lane (zero and subnormal source lanes must match exactly).
// The core input and the caller's ID counter remain unchanged. The caller
// installs the band in place of the core owner only after success.
absl::StatusOr<LiquidBandReseed3D> ReseedLiquidCoreCell3D(
    const LiquidHandoffCell3D& cell,
    const LiquidRemapInventory3D& core_inventory,
    const std::optional<LiquidHandoffCarrier3D>& destination_carrier,
    std::size_t particle_count,
    std::size_t available_particle_slots,
    std::uint64_t first_particle_id
);

// Prepare one unpublished replacement for a canonical adaptive leaf. The
// caller supplies authoritative occupancy and certifies the carrier over the
// full leaf. Retirement requires particle ownership and exact agreement of
// the leaf ledger with its resident particles. Both directions require the
// graded scheduler's ordered mass, volume and momentum sums to agree exactly.
// Retained particles keep order
// and IDs. Eulerian leaves must have no resident particles. The supplied
// capacities must accommodate the input and result. Errors preserve input.
absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidBandRetirement3D(
    const LiquidState3D& committed,
    LiquidStateCapacities3D capacities,
    AdaptiveMacCellKey3D leaf_key,
    LiquidHandoffOccupancy3D authoritative_occupancy,
    const std::optional<LiquidHandoffCarrier3D>& destination_carrier
);

// Prepare the reverse change. The Eulerian inventory must be exactly
// represented by the new particles. Append them in certified seed order
// and advance the particle ID counter by particle_count. Both operations
// preserve the clock, face records, unrelated leaves, and cell enthalpy.
// FailedPrecondition denotes missing/incompatible ownership, occupancy or
// carrier. ResourceExhausted denotes inadequate capacity. Validation,
// topology and arithmetic errors propagate without a result.
absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidCoreReseed3D(
    const LiquidState3D& committed,
    LiquidStateCapacities3D capacities,
    AdaptiveMacCellKey3D leaf_key,
    LiquidHandoffOccupancy3D authoritative_occupancy,
    const std::optional<LiquidHandoffCarrier3D>& destination_carrier,
    std::size_t particle_count
);

}  // namespace fluidity
