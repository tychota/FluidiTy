#pragma once

#include "fluidity/base/wetted_piston_math_3d.h"
#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity {
namespace wetted_piston_internal {
struct WorkspaceAccess;
}

// One signed lattice patch in (z,y) order, clipped by the fixed tank walls.
struct WettedPistonPatch3D {
  std::int32_t lattice_y = 0;
  std::int32_t lattice_z = 0;
  WettedPistonBounds3D area{};  // Patch wet area in m².
  // Ordinal in the frozen face span. Empty for dry patches without a receiver.
  std::optional<std::size_t> receiver_face_ordinal = std::nullopt;
  // Assigned area in m², populated only on the represented patch. It can
  // exceed one nominal face because it includes the immediately higher patch.
  WettedPistonBounds3D receiver_area{};
  WettedPistonBounds3D receiver_flux{};  // Assigned signed flux enclosure, m³/s.
};

struct WettedPistonInputs3D {
  const LiquidSceneField3D& field;  // Actual liquid field at the freeze time.
  std::span<const LiquidParticleState3D> particles;  // Frozen world centers for the particle union.
  const FrozenLiquidBoundaryGeometryView3D& frozen;  // Selected pressure cells and incident faces at step start.
  AxisAlignedLiquidTank3D tank;  // Exact start-time tank used to freeze geometry.
};

struct WettedPistonView3D {
  std::span<const WettedPistonPatch3D> patches;  // Clipped patches sorted by (z, y), including dry receivers.
  // One entry per frozen face, in frozen face order. Flux in m³/s is signed
  // along +x, positive into the tank. A dry receiver has explicit zero.
  // Faces without a pressure receiver have no value.
  std::span<const std::optional<double>> face_fluxes;
  WettedPistonBounds3D whole_area;  // m².
  WettedPistonBounds3D whole_flux;  // m³/s.
  std::size_t attached_patch_count = 0;  // Wet patches assigned to the selected row directly below.
  int compiled_architecture = 0;  // 0 on CPU or no device dispatch.
};

class WettedPistonWorkspace3D {
 public:
  // Reserve outside attempts. Patch queries and results are each bounded by
  // face capacity, and source and interval counts by max(1, particle capacity).
  // Exhausted or overflowing storage returns ResourceExhausted.
  static absl::StatusOr<WettedPistonWorkspace3D> Create(LiquidStateCapacities3D capacities);
  // Counts every reserved dynamic host byte, excluding allocator bookkeeping.
  std::size_t reserved_bytes() const;

 private:
  friend struct wetted_piston_internal::WorkspaceAccess;
  struct Publication {
    LiquidRecords3D<WettedPistonPatch3D> patches;
    LiquidRecords3D<std::optional<double>> face_fluxes;
  };
  std::array<Publication, 2> publications_;
  LiquidRecords3D<wetted_piston_internal::PatchIndex> indices_;
  LiquidRecords3D<wetted_piston_internal::PatchMeasure> measures_;
  LiquidRecords3D<Vec3> centers_;
  LiquidRecords3D<wetted_piston_internal::Disk> disks_;
  LiquidRecords3D<wetted_piston_internal::Interval> intervals_;
  LiquidStateCapacities3D capacities_{};
  std::size_t reserved_bytes_ = 0;
  std::size_t live_publication_ = 0;
};

// Compute bounded wet area and signed flux at the low-x wall, keeping the
// selected pressure topology fixed. Extra support can use only the highest
// selected row immediately below, with certified direct contact across the
// lower-y edge for each potentially wet source. Wall area must be certified
// zero or have a positive lower bound with upper minus lower at most 0.001 times lower.
// Malformed inputs return InvalidArgument, range/uncertifiable arithmetic
// OutOfRange, unsupported ownership/area width FailedPrecondition, and scratch
// limits ResourceExhausted. Empty particle union is dry. The single-threaded
// stage allocates nothing. Inputs must not alias workspace storage. Results
// borrow it until the next successful call or workspace replacement. Failure
// preserves the previous result and all inputs.
absl::StatusOr<WettedPistonView3D>
ComputeWettedPiston3D(const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& workspace);

}  // namespace fluidity
