#pragma once

#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity {

struct MotionCarrierSample3D {
  Vec3 velocity;  // World m/s.
  bool continued = false;  // Some original quadratic support was unavailable.
  bool rank_fallback = false;  // Continuation used the nearest compatible donor.
};

// Owning spatial carrier for one accepted snapshot. Valid/predicted same-side
// faces supply the original quadratic interpolation. An incomplete stencil uses
// the existing deterministic radius-2h affine ghost fit at the query, then its
// nearest compatible donor on rank loss. No donors means missing support.
class FrozenMotionCarrier3D {
 public:
  // Copies a checked state's grid and face data. The zero-clock initial state
  // may omit faces; in that case it constructs the initial P2G carrier with the
  // existing compatible fixed-point deposit and normalization. No force, pressure,
  // particle advance or simulation mutation occurs. Returns state/transfer errors.
  static absl::StatusOr<FrozenMotionCarrier3D>
  Create(const LiquidState3D& state, const LiquidSceneInputs3D& scene, const LiquidStepOptions3D& options);
  // Samples a finite world position in metres on the indicated material side.
  // Solid-interior or absent support returns FailedPrecondition. Invalid positions
  // return InvalidArgument and nonfinite arithmetic returns OutOfRange.
  // The result reports continuation/fallback independently of numerical velocity.
  absl::StatusOr<MotionCarrierSample3D> Sample(Vec3 world_position, std::int64_t side_tag = 0) const;

 private:
  struct Face {
    Vec3 position;
    FaceNormalAxis axis;
    double velocity;
    LiquidFaceValidity3D validity;
    LiquidBoundaryFaceTag3D tag;
  };
  FrozenMotionCarrier3D(SparseMacGrid3D grid, LiquidObstacle3D obstacle, std::vector<Face> faces);
  SparseMacGrid3D grid_;
  LiquidObstacle3D obstacle_;
  std::vector<Face> faces_;
};
}  // namespace fluidity
