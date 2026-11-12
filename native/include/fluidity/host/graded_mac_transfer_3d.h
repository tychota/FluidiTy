#pragma once

#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_transfer_types_3d.h"
#include "fluidity/host/adaptive_mac_topology_3d.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity {

// Physical staggered anchor at the center of a canonical owned subface.
struct GradedMacFace3D {
  FaceNormalAxis axis;
  Vec3 world_anchor;  // m.
  double width_m;
};

struct GradedMacParticleGather3D {
  Vec3 velocity;  // Cartesian m/s evaluated at the particle.
  Mat3 velocity_slopes;  // Cartesian affine slopes, inverse seconds.
  bool affine_enabled;  // False means PIC on all three components and zero slopes.
};

class GradedMacTransfer3D {
 public:
  // Freeze the physical leaf boxes and all owned subface anchors. world_origin
  // is the world position in m of the finest lattice's zero corner. Returns
  // InvalidArgument for invalid topology or origin, OutOfRange if world
  // geometry cannot be represented, and ResourceExhausted above the state cap.
  static absl::StatusOr<GradedMacTransfer3D> Create(const AdaptiveMacTopology3D& topology, Vec3 world_origin);

  // One geometry record per canonical global slot. The span borrows this owner.
  std::span<const GradedMacFace3D> faces() const {
    return faces_;
  }

  // Deposit mass in kg and positive-normal momentum in kg*m/s. Results are in
  // canonical global slot order. Each particle must lie in a half-open leaf
  // box. Only same-side, structurally eligible subfaces participate. Incoming
  // carrier velocity and validity are not deposition inputs. Weight sums are
  // normalized per component, and APIC uses centered offsets when all three
  // components have affine support. Particle order and global slot order fix
  // the reduction order. InvalidArgument denotes malformed extents or inputs,
  // FailedPrecondition denotes missing support, and OutOfRange denotes
  // nonrepresentable arithmetic. Failure returns no deposit owner.
  absl::StatusOr<std::vector<MacFaceDeposit3D>> Deposit(
      std::span<const LiquidParticleState3D> particles,
      std::span<const LiquidAdaptiveFaceState3D> carrier_faces
  ) const;

  // Gather a world-space velocity and weighted affine fit from the canonical
  // global slots. Missing, ineligible and different-side carriers are masked.
  // Low mass or rank loss on any component selects normalized PIC and clears
  // all slopes. FailedPrecondition denotes missing support on any component.
  // InvalidArgument denotes malformed extents, position or carrier data.
  // OutOfRange denotes nonrepresentable arithmetic. Failure returns no result.
  absl::StatusOr<GradedMacParticleGather3D>
  Gather(const LiquidParticleState3D& particle, std::span<const LiquidAdaptiveFaceState3D> carrier_faces) const;

 private:
  struct LeafBox {
    Vec3 lower;
    Vec3 upper;
    double width_m;
  };

  GradedMacTransfer3D() = default;
  absl::StatusOr<double> ParticleSupportWidth(Vec3 world_position) const;

  std::vector<LeafBox> leaves_;
  std::vector<GradedMacFace3D> faces_;
};

}  // namespace fluidity
