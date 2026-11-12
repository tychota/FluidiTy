#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity {

// Reuses fixed-capacity balances and output records for one host worker.
// The workspace owns all storage. Moves invalidate any borrowed result.
class AdaptiveLiquidCoreTransportWorkspace3D {
 public:
  AdaptiveLiquidCoreTransportWorkspace3D(AdaptiveLiquidCoreTransportWorkspace3D&&) noexcept;
  AdaptiveLiquidCoreTransportWorkspace3D& operator=(AdaptiveLiquidCoreTransportWorkspace3D&&) noexcept;
  ~AdaptiveLiquidCoreTransportWorkspace3D();

  // Reserve room for at most cell_capacity canonical leaves. Returns
  // ResourceExhausted for a capacity above the adaptive state limit or an
  // allocation failure.
  static absl::StatusOr<AdaptiveLiquidCoreTransportWorkspace3D> Create(std::size_t cell_capacity);

  // Dynamic bytes owned by this workspace, or zero after move.
  std::size_t reserved_bytes() const;

  // Uses the same inputs and transport checks as the owning convenience
  // function. The inputs must not overlap workspace output storage. A
  // successful result borrows storage until the next successful Prepare,
  // move, or destruction. A failed call preserves the last published records.
  // Returns ResourceExhausted if the topology exceeds capacity and
  // InvalidArgument on overlap. Preparation allocates nothing and returns
  // code-only errors.
  absl::StatusOr<std::span<const LiquidAdaptiveCellState3D>> Prepare(
      const AdaptiveMacTopology3D& topology,
      std::span<const LiquidAdaptiveCellState3D> committed_cells,
      std::span<const double> interior_fluxes_m3_per_s,
      double step_duration_s
  );

 private:
  struct Storage;
  explicit AdaptiveLiquidCoreTransportWorkspace3D(std::unique_ptr<Storage> storage) noexcept;

  std::unique_ptr<Storage> storage_;
};

// Advects implicit single-phase inventory between liquid, Eulerian-owned
// leaves. Cell records and integrated subface fluxes in m³/s cover the topology
// in canonical cell and interior face order. Positive flux goes from tail to
// head. Supply a finite positive duration in seconds. The signed swept volume
// on each subface transfers the corresponding fraction of step-start donor
// mass, reference volume, momentum, and signed enthalpy. Gross donor outflow
// is limited by its step-start current volume. Final current volume is limited
// by geometric leaf capacity. The final face of an exact drain carries the
// donor's rounded remainder. Conservation is subject to binary64 roundoff.
//
// Exterior fluxes are outside this operation. Interior faces wholly outside
// the Eulerian liquid region are ignored. Nonzero flux between the region
// and any other leaf, explicit phase inventory in the region, gross overdraw,
// and final overfill return FailedPrecondition. Invalid dimensions, keys,
// metadata, or numerical inputs return InvalidArgument. Unrepresentable
// arithmetic returns OutOfRange. Returns independent candidate records with
// updated inventories on success. All inputs are read-only on every path.
absl::StatusOr<std::vector<LiquidAdaptiveCellState3D>> PrepareAdaptiveLiquidCoreTransport3D(
    const AdaptiveMacTopology3D& topology,
    std::span<const LiquidAdaptiveCellState3D> committed_cells,
    std::span<const double> interior_fluxes_m3_per_s,
    double step_duration_s
);

}  // namespace fluidity
