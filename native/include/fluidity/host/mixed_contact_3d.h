#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/contact_face_bodies_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/mac_pressure_3d.h"
#include "fluidity/host/plastic_solid_checkpoint_3d.h"

namespace fluidity {

// Applies detected contact to one liquid body's and any plastic bodies'
// candidate stored velocities in world m/s. Particle IDs must be globally
// unique across the supplied states. Liquid uses liquid_body_id. Each
// plastic record retains its own body ID, distinct from the liquid body.
// The two writable spans must be disjoint from each other and all face data.
// Returns face impulse receipts in face-slot order. Failure leaves both spans
// unchanged, with the same status categories as StageContactFaceBatch3D.
// An unrepresentable final velocity returns OutOfRange. The caller owns the
// surrounding accepted step and handles pressure and particle advection.
absl::StatusOr<std::vector<DistinctBodyContact3D>> ApplyMixedContactFaces3D(
    std::span<LiquidParticleState3D> liquid,
    std::uint64_t liquid_body_id,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    std::span<const DetectedContactFace3D> faces
);

struct PlanarMixedContactFace3D {
  DetectedContactFace3D pair;  // Candidate, admitted only when the sweep hits.
  Vec3 first_interface_world_position;  // Sample on the first body's interface, meters.
  Vec3 first_interface_world_velocity;  // Constant interface velocity in m/s over the attempt.
  double second_plane_start_coordinate;  // Dot(pair.normal_toward_first, second plane point), meters.
  Vec3 second_interface_world_velocity;  // Constant plane velocity in world m/s.
};

// Reconstructs an axis-aligned planar envelope from contributors on one
// face. The first body's minimum projection and the second body's maximum
// projection face each other along the supplied signed Cartesian normal.
// Nonnegative interface offsets extend the fronts toward each other in
// meters. The contributors must be ordered by unique ID and match the
// supplied particle records. The two front velocities come from the face
// samples. Conflicting normal velocities at tied extrema or missing fronts
// return FailedPrecondition. Malformed inputs return InvalidArgument.
// Nonfinite offset arithmetic returns OutOfRange. The result borrows the
// original contributors. This model assumes locally parallel interfaces.
absl::StatusOr<PlanarMixedContactFace3D> ReconstructAxisAlignedMixedContactFace3D(
    std::span<const LiquidParticleState3D> liquid,
    std::uint64_t liquid_body_id,
    std::span<const PlasticSolidCheckpointParticle3D> plastic,
    const DetectedContactFace3D& pair,
    double first_interface_offset,
    double second_interface_offset
);

struct PlanarMixedContactReceipt3D {
  std::size_t face_slot;
  double contact_fraction;  // Earliest hit fraction in [0, 1].
  DistinctBodyContact3D contact;  // Two-body response using supplied face contributors.
};

// Sweeps supplied locally planar interfaces before applying their detected
// face pairs. Uses certified constant-velocity halfspace contact with zero
// initial penetration tolerance. Candidate slots must be strictly ordered.
// Only hits contribute impulses. The velocities used for the sweep are
// reconstructed interface motion; impulses use the supplied face samples.
// Invalid duration, geometry, ordering or overlapping writable views return
// InvalidArgument or the underlying sweep status. Unresolvable contact or
// scatter errors propagate. Any failure leaves both material spans unchanged.
// This updates candidate velocities only. Geometry reconstruction, position
// advancement after impact and accepted-step publication belong to the caller.
absl::StatusOr<std::vector<PlanarMixedContactReceipt3D>> ApplyPlanarMixedContactFaces3D(
    std::span<LiquidParticleState3D> liquid,
    std::uint64_t liquid_body_id,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    std::span<const PlanarMixedContactFace3D> candidates,
    double duration
);

// Advances a bounded piecewise-constant trial with a common hit fraction
// across all detected faces. The incoming and outgoing stored velocities
// are also the advection velocities for this trial. With no hit, advection
// uses the incoming velocity for the entire duration. The plastic trial
// position is advanced while its sorted position remains the last bin pose.
// The caller supplies the certified geometry and a positive path length
// limit in meters. Different hit fractions or an exceeded bound return
// FailedPrecondition. Nonfinite motion returns OutOfRange. Other errors
// follow ApplyPlanarMixedContactFaces3D. No failure changes either span.
absl::StatusOr<std::vector<PlanarMixedContactReceipt3D>> AdvancePlanarMixedContact3D(
    std::span<LiquidParticleState3D> liquid,
    std::uint64_t liquid_body_id,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    std::span<const PlanarMixedContactFace3D> candidates,
    double duration,
    double maximum_path_length
);

struct PlanarSnowPressureTrial3D {
  MacPressureProjection3D liquid_projection;
  double solved_wall_normal_velocity;  // Along the positive MAC face axis, m/s.
  double pressure_impulse_on_snow;  // Along the same axis, N s.
  double pressure_work_on_snow;  // Signed J at the solved wall velocity.
};

using MixedPressureProjector3D = absl::StatusOr<MacPressureProjection3D> (*)(const MacPressureInputs3D&);

// Solve a frozen MAC pressure system with one translating full sealed face
// supported by a finite snow mass. The wall has one incident pressure cell. The projector must implement the MAC
// pressure contract. Supply the wall ordinal in pressure face order and
// the incoming snow speed along its positive MAC axis in m/s. The solve
// balances the pressure impulse with the wall momentum and checks that
// balance after reprojection. Invalid geometry or parameters return
// InvalidArgument, invalid pressure response or balance returns
// FailedPrecondition, and projector failures propagate. No input is modified.
absl::StatusOr<PlanarSnowPressureTrial3D> SolvePlanarSnowPressureTrial3D(
    const MacPressureInputs3D& inputs,
    std::size_t wall_face_ordinal,
    double snow_mass,
    double incoming_snow_normal_velocity,
    MixedPressureProjector3D project
);

struct SnowPressureContributor3D {
  std::uint64_t particle_id;
  double area_fraction;  // Positive fraction of the selected wall's area.
};

// Interpolates wall speed by area fraction and distributes the pressure
// impulse using the transpose of those weights. Particle and contributor
// IDs must be strictly increasing and the contributor fractions must sum
// to one within 1e-12. Contributors must belong to one body. Plastic
// records are candidate state. Their span must
// be disjoint from contributors and pressure inputs. On success, only the
// participating particles' stored normal velocities change. Failure
// leaves the entire plastic span unchanged.
absl::StatusOr<PlanarSnowPressureTrial3D> ApplyPlanarSnowPressure3D(
    const MacPressureInputs3D& inputs,
    std::size_t wall_face_ordinal,
    std::span<const SnowPressureContributor3D> contributors,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    MixedPressureProjector3D project
);

struct FrozenPlanarSnowPressureCell3D {
  MacPressureCell3D cell;
  std::array<MacPressureFace3D, 6> faces;
  std::size_t wall_face_ordinal;
  double wall_plane_coordinate;  // Signed coordinate along the normal into liquid, m.
};

// Reconstruct one axis-aligned snow front from the outermost supplied
// particle and a nonnegative outward offset in meters. The front cuts
// the selected cell with positive liquid thickness on the opposite side.
// Produces a sealed snow wall, stationary sealed transverse faces and a
// full ambient-pressure outlet. All records must belong to one body.
absl::StatusOr<FrozenPlanarSnowPressureCell3D> FreezePlanarSnowPressureCell3D(
    const SparseMacGrid3D& grid,
    CellIndex3D cell,
    FaceNormalAxis axis,
    bool wall_on_low_face,
    std::span<const PlasticSolidCheckpointParticle3D> snow_front_particles,
    double surface_offset,
    double inverse_liquid_density,
    double outlet_pressure
);

// Stages pressure traction, regular-cell liquid advection and snow
// contributor drift in a frozen pressure domain. Non-wall faces must be fully open or stationary
// sealed. Liquid can cross one open face into an adjacent pressure cell.
// Both materials stay on their
// respective sides of the translating wall without a second impulse. Position increments are limited
// by maximum_path_length in meters. A supplied wall coordinate clips the
// incident cell along the wall axis and must match its wet volume.
// Advances volume from the same
// regular-cell field. On success both spans receive the candidate state;
// otherwise neither span changes. The caller owns rebinning and acceptance.
absl::StatusOr<PlanarSnowPressureTrial3D> AdvancePlanarSnowPressure3D(
    const MacPressureInputs3D& inputs,
    std::size_t wall_face_ordinal,
    std::span<const SnowPressureContributor3D> contributors,
    std::span<LiquidParticleState3D> liquid,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    double maximum_path_length,
    MixedPressureProjector3D project,
    std::optional<double> wall_plane_coordinate = std::nullopt
);

struct PlanarSnowPressureAttempt3D {
  LiquidAttemptOutcome3D outcome;
  std::optional<PlanarSnowPressureTrial3D> pressure;  // Present only on acceptance.
};

// Runs a coupled pressure candidate inside an active liquid frame transaction.
// The committed liquid topology must match the supplied pressure grid, rows
// and faces. Starts pressure from committed row history. A recoverable motion
// or pressure precondition failure consumes a timestep retry. Other errors
// terminate the attempt. Liquid topology, pressure, carriers, particles and
// clock commit before the staged snow records are published. No rejected
// attempt modifies either committed material. The caller owns frame setup,
// subcell geometry reconstruction and any outer joint checkpoint.
absl::StatusOr<PlanarSnowPressureAttempt3D> AttemptPlanarSnowPressureStep3D(
    LiquidFrameTransaction3D& transaction,
    const MacPressureInputs3D& inputs,
    std::size_t wall_face_ordinal,
    std::span<const SnowPressureContributor3D> contributors,
    std::span<PlasticSolidCheckpointParticle3D> snow,
    double maximum_path_length,
    MixedPressureProjector3D project,
    std::optional<double> wall_plane_coordinate = std::nullopt
);

}  // namespace fluidity
