#pragma once
#include "fluidity/base/mac_transfer_types_3d.h"
namespace fluidity {
struct LiquidBoundaryParticle3D {
  MacTransferParticle3D transfer;
  std::int64_t side_tag = 0;  // Equality identifies the same material/body side.
  bool affine_enabled = true;  // False makes deposition consume a zero affine row.
};
struct LiquidBoundaryFaceTag3D {
  std::int64_t side_tag = 0;
  bool structurally_eligible = true;  // Independent of allocation and deposited mass.
};
struct CompatibleMacFaceSample3D {
  std::size_t face_slot;  // Original owned slot, retained even for excluded pairs.
  bool compatible;  // Structural eligibility and side equality, independent of mass.
  double normalized_weight;  // N/S, dimensionless; zero on excluded pairs.
  Vec3 normalized_weight_gradient;  // Quotient-rule derivative, inverse meters.
  Vec3 centered_face_offset;  // r minus the compatible weighted mean, meters.
};
struct LiquidPrescribedWallSample3D {
  bool available = false;
  std::int64_t side_tag = 0;  // Wall continuation is usable only by this side.
  Vec3 unit_normal{};  // World Cartesian normal, unit length when available.
  Vec3 wall_velocity{};  // World m/s at this history's wall_sample_time.
};
struct LiquidBoundaryParticleGather3D {
  Vec3 new_pic_velocity{};  // m/s from the complete new filled stencil.
  Vec3 grid_velocity_increment{};  // new PIC minus old PIC, m/s.
  Vec3 particle_minus_old_pic_velocity{};  // Retained particle residual, m/s.
  Mat3 fitted_velocity_slopes{};  // Complete second-moment fit, inverse seconds.
  Mat3 velocity_gradient{};  // True original-basis derivative, inverse seconds.
  bool affine_enabled = true;  // False on input affine-off or any stage fallback.
  bool old_pic_fallback = false;  // Any old-stage ghost required rank fallback.
  bool new_pic_fallback = false;  // Any new-stage ghost required rank fallback.
  bool low_mass_fallback = false;  // Either complete history touches low mass.
};
enum class LiquidFaceValidity3D : std::uint8_t {
  kMissing,
  kValid,
  kPredicted,  // Explicit finite continuation eligible as a donor, without mass.
  kLowMass,  // Positive mass at or below the minimum; forces PIC retention off.
};
}  // namespace fluidity
