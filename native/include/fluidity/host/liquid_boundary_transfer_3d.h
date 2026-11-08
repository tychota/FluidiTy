#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/base/liquid_boundary_transfer_types_3d.h"
#include "fluidity/host/transfers_3d.h"

namespace fluidity {

// Bounded CPU qualification capacities for subsequent CUDA staging.
inline constexpr std::size_t kLiquidBoundaryParticleCapacity3D = 1024;
inline constexpr std::size_t kLiquidBoundaryFaceCapacity3D = 24576;

using CompatibleMacFaceStencil3D = std::array<CompatibleMacFaceSample3D, 27>;

struct LiquidBoundaryDepositInputs3D {
  const SparseMacGrid3D& mac_grid;
  std::span<const LiquidBoundaryParticle3D> particles;
  std::span<const LiquidBoundaryFaceTag3D> face_tags;  // One per owned slot.
  MacDepositQuantization3D quantization;
};

struct LiquidBoundaryFaceDeposits3D {
  std::vector<MacFaceDepositQuanta3D> face_quanta;
  // Particle order, then x/y/z components; samples retain original stencil
  // (z,y,x) order, with x fastest. Ghost donors separately use owned-slot order.
  std::vector<std::array<CompatibleMacFaceStencil3D, 3>> compatible_stencils;
};

// Resolves every original sample, including excluded/zero-weight pairs, before
// masking by structural eligibility and side. Uses normalized, recentered affine
// momentum and nearest-even per-contribution quanta; liquid stress is zero.
// No previous face mass participates. Empty batches produce zero deposits.
// Scales are positive normal powers of two. Each scaled contribution is bounded
// by 2^(62-ceil(log2(max(1,declared_particle_capacity)))); count and declared
// capacity cannot exceed 1024. Disabled affine rows are ignored, even if stale.
// Returns InvalidArgument for invalid extents/data/scales, FailedPrecondition
// for missing ownership/motion or nonpositive compatible support, OutOfRange for
// arithmetic/range failure, ResourceExhausted above the bounded capacities.
// Inputs are borrowed read-only and may alias; errors publish no result.
absl::StatusOr<LiquidBoundaryFaceDeposits3D> DepositCompatibleMacQuanta3D(const LiquidBoundaryDepositInputs3D& inputs);

struct NormalizedLiquidMacFaces3D {
  std::vector<double> face_masses;  // Kilograms.
  std::vector<double> face_velocities;  // m/s; zero for missing/low-mass slots.
  std::vector<LiquidFaceValidity3D> validity;
};

// Decodes signed quanta with the deposit scales. Divides momentum by mass only
// for strictly positive mass greater than finite nonnegative minimum_face_mass.
// Zero mass is missing; positive mass at/below the threshold is low-mass.
// Rejects negative/out-of-bound mass, nonzero stress, invalid scales/capacity
// (InvalidArgument/ResourceExhausted), or nonfinite arithmetic (OutOfRange).
// Results own their arrays; errors publish nothing and preserve inputs.
absl::StatusOr<NormalizedLiquidMacFaces3D> NormalizeLiquidMacFaces3D(
    std::span<const MacFaceDepositQuanta3D> face_quanta,
    MacDepositQuantization3D quantization,
    double minimum_face_mass
);

struct LiquidBoundaryVelocityHistory3D {
  double velocity_sample_time;  // Seconds, must match the requested stage time.
  double wall_sample_time;  // Seconds, must match velocity_sample_time.
  std::span<const double> face_velocities;
  std::span<const LiquidFaceValidity3D> validity;
  std::span<const LiquidBoundaryFaceTag3D> face_tags;
  std::span<const LiquidPrescribedWallSample3D> wall_samples;
};
struct LiquidBoundaryGatherInputs3D {
  const SparseMacGrid3D& mac_grid;
  std::span<const LiquidBoundaryParticle3D> particles;
  double old_time;  // Finite pre-force snapshot time, seconds.
  double new_time;  // Finite and strictly greater than old_time.
  LiquidBoundaryVelocityHistory3D old_history;
  LiquidBoundaryVelocityHistory3D new_history;
};

struct LiquidBoundaryGatherResults3D {
  std::vector<LiquidBoundaryParticleGather3D> particles;
  std::size_t old_pic_fallback_particle_count = 0;
  std::size_t new_pic_fallback_particle_count = 0;
  std::size_t low_mass_fallback_particle_count = 0;
};

// Uses original complete quadratic weights/derivatives and separate old/new
// ghost histories. Missing/incompatible samples use finite same-side donors of
// the same MAC component within Euclidean distance 2h, in owned-slot order.
// A deterministic column-pivoted QR in h units accepts pivots >=1e-10 of the
// largest. At a compatible wall, project the fitted vector onto its prescribed
// relative-normal constraint before selecting the staggered component.
// Rank failure uses the nearest eligible donor (ties by slot), then a compatible
// prescribed wall, otherwise fails. Any rank/low-mass fallback disables affine
// use and suppresses retained delta, while preserving diagnostic fit/gradient.
// Valid and predicted samples are donors; missing/low-mass samples are not.
// Histories each span all owned slots; all active velocities/walls must be
// finite. Missing/low-mass payload velocities are ignored. History mismatches
// and malformed inputs return InvalidArgument; unavailable support returns
// FailedPrecondition, range failure OutOfRange, excess capacity ResourceExhausted.
// Read-only inputs may alias; owning outputs are published only on success.
absl::StatusOr<LiquidBoundaryGatherResults3D> GatherLiquidBoundaryParticles3D(const LiquidBoundaryGatherInputs3D& inputs
);

}  // namespace fluidity
