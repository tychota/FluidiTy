#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "fluidity/host/graded_mac_transfer_3d.h"
#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity {

struct MotionCarrierSample3D {
  Vec3 velocity;  // World m/s.
  bool continued = false;  // Sampling used one-sided support or a fallback.
  bool rank_fallback = false;  // Nearest donor on uniform grids or PIC on graded grids.
};

// Owning spatial carrier for one accepted snapshot. Valid/predicted same-side
// faces supply the original quadratic interpolation. An incomplete stencil uses
// the existing deterministic radius-2h affine ghost fit at the query, then its
// nearest compatible donor on rank loss. Graded snapshots reuse the canonical
// subface affine fit and its normalized PIC fallback. Both layouts refuse
// queries without usable same-side support.
class FrozenMotionCarrier3D {
 public:
  // Copies a checked state's grid and face data. The zero-clock initial state
  // may omit faces; in that case it constructs the initial P2G carrier with the
  // existing compatible fixed-point deposit and normalization. No force, pressure,
  // particle advance or simulation mutation occurs. Graded snapshots copy
  // their stored canonical subface carriers. Returns state/transfer errors.
  static absl::StatusOr<FrozenMotionCarrier3D>
  Create(const LiquidState3D& state, const LiquidSceneInputs3D& scene, const LiquidStepOptions3D& options);
  // Samples a finite world position in metres on the indicated material side.
  // Solid-interior or absent support returns FailedPrecondition. Invalid positions
  // return InvalidArgument and nonfinite arithmetic returns OutOfRange.
  // On or within half a finest-cell width above a graded stationary planar
  // tank, continue the interior face fit from the open top. A full-rank fit
  // extends affinely and the PIC fallback extends constantly. The result
  // reports one-sided continuation and rank fallback.
  absl::StatusOr<MotionCarrierSample3D> Sample(Vec3 world_position, std::int64_t side_tag = 0) const;

 private:
  struct Face {
    Vec3 position;
    FaceNormalAxis axis;
    double velocity;
    LiquidFaceValidity3D validity;
    LiquidBoundaryFaceTag3D tag;
  };
  struct UniformCarrier {
    SparseMacGrid3D grid;
    std::vector<Face> faces;
  };
  struct PlanarTop {
    double height_m;
    double continuation_width_m;
  };
  struct AdaptiveCarrier {
    GradedMacTransfer3D transfer;
    std::vector<LiquidAdaptiveFaceState3D> faces;
    std::optional<PlanarTop> open_top;
  };
  FrozenMotionCarrier3D(LiquidObstacle3D obstacle, std::variant<UniformCarrier, AdaptiveCarrier> carrier);
  LiquidObstacle3D obstacle_;
  std::variant<UniformCarrier, AdaptiveCarrier> carrier_;
};
}  // namespace fluidity
