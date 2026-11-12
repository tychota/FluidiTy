#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "fluidity/base/mac_transfer_types_3d.h"
#include "fluidity/host/sparse_mac_grid_3d.h"

namespace fluidity {

// Deposits mass and pure normal momentum to one record per owned face slot.
// For each particle, the velocity at a face is its normal velocity plus the
// corresponding APIC slope row dotted with the face offset from the particle.
//
// Empty batches clear every face. The particle and output views
// must be disjoint. Mass must be finite and nonnegative, velocity and affine
// slopes must be finite, and every component needs complete owned support.
// Returns kInvalidArgument for invalid views or data, kFailedPrecondition for
// unsupported motion or ownership, and kOutOfRange for unrepresentable
// geometry or arithmetic. On failure the output is preserved.
absl::Status DepositMac3D(
    const SparseMacGrid3D& mac_grid,
    std::span<const MacTransferParticle3D> particles,
    std::span<MacFaceDeposit3D> face_deposits
);

// Deposits the same liquid mass and momentum as DepositMac3D, rounding each
// contribution to nearest-even quanta before summation. Scales must be
// positive normal powers of two. If cap is the declared capacity, each
// scaled magnitude must be at most 2^(62 - ceil(log2(max(1, cap)))).
// The stress-force channel is zero. Extents, aliasing and other failure
// rules match DepositMac3D. Returns kResourceExhausted if cap exceeds
// 2^20 or the batch exceeds cap. Range failures preserve the output.
absl::Status DepositMacQuanta3D(
    const SparseMacGrid3D& mac_grid,
    std::span<const MacTransferParticle3D> particles,
    MacDepositQuantization3D quantization,
    std::span<MacFaceDepositQuanta3D> face_deposits
);

// Interpolates particle-in-cell (PIC) velocity from the new face field. The
// grid increment is the difference between the new and old interpolations.
// Fits APIC velocity slopes and differentiates the interpolant separately.
// The fit and derivative can differ for a non-affine face field.
//
// Reads only positions from the particles. Each finite face field
// must span every owned face, and the output must have one record per
// particle. Read-only views may alias each other. The output must be
// disjoint from all input views. Returns kInvalidArgument for invalid
// views or data, kFailedPrecondition for unsupported motion or ownership,
// and kOutOfRange for unrepresentable geometry or arithmetic. On failure
// the output is preserved.
absl::Status GatherMac3D(
    const SparseMacGrid3D& mac_grid,
    std::span<const MacTransferParticle3D> particles,
    std::span<const double> old_face_velocities,
    std::span<const double> new_face_velocities,
    std::span<MacParticleGatherResult3D> particle_gather_results
);

// Computes compressible Neo-Hookean Kirchhoff stress in pascals from a
// dimensionless deformation gradient with positive determinant. The shear
// modulus must be positive and the first Lame parameter nonnegative (pascals).
// Returns InvalidArgument for invalid inputs or OutOfRange for nonfinite arithmetic.
absl::StatusOr<Mat3>
NeoHookeanKirchhoffStress3D(Mat3 deformation_gradient, double shear_modulus, double first_lame_parameter);

// Replaces the owned normal face forces (newtons) with one elastic particle's
// uncut stress contribution, using true physical basis derivatives. Positions
// and support rules match BuildOwnedMacFaceStencil3D. Reference volume is in
// cubic meters and must be finite and positive. Stress is in pascals.
// Invalid extent or inputs return InvalidArgument. Unsupported support and
// nonfinite arithmetic return FailedPrecondition and OutOfRange respectively.
// Every failure preserves the output.
absl::Status DepositElasticStress3D(
    const SparseMacGrid3D& mac_grid,
    Vec3 sorted_particle_world_position,
    Vec3 trial_particle_world_position,
    double reference_volume,
    Mat3 kirchhoff_stress,
    std::span<double> normal_face_forces
);

}  // namespace fluidity
