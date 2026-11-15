# Contact module design

Status: updated 2026-11-14 with the owner's answers (`owner-answers.md`) and `changes-to-apply.md`; sphere and
cylinder added 2026-11-15 (A74: 3.2, 5.7, 6, 7, 8, Q18); aligned 2026-11-15 with the method paper's contact section
after its review (5.1, 5.3, 5.5, 5.7, 8: cylinder gaps `wall_distance`/`cap_distance`, reaction from the slide, no
touching rule on a turning solid, rim move for closed-form hits only, section order). Module 1
of the rebuild. Names shared between modules are fixed in `shared-interface.md`; where it still says
`state`/`travel`, A1 replaced them by `for_next_step`/`for_this_step`.

## 1. Purpose

A particle travels in a straight line during one step. Contact moves one particle for one step without letting
it enter a solid: it finds the first time the line meets a solid (the hit), removes the speed into the solid,
takes away some sideways speed by friction, and spends the rest of the step sliding along the surface. The same
code serves particles of every material; only the friction number and the plate side they carry differ.

**Owns:** the analytic solid shapes (flat solid, plate, box, sphere and cylinder (NEW, A74), the sides of the domain box) with their pose and
velocity during one step; distance to a solid and velocity of its surface; swept hit detection; the friction
response; the slide; the contact tolerance; which side of a plate a point is on; the per-particle contact record.

**Does not own:** how a solid moves from step to step, the force and torque on a solid, the wave paddle's drive
law, cutting, contact between two materials on grid faces (`CollideSolids`), building `contact::Solid` values
from saved solid settings for render output (`BuildContactSolids`) (all in the solids design), the pressure
solve's solid flux (the pressure design); mixing the two particle velocities (the transfer design); removing
particles that left through an open domain side (the particles design, called by the step); the limit on how far
a particle may travel per step and the retry (the step design).

## 2. What exists today

All paths are on `main`. Y-up: gravity is `-y`, the tank floor is `floor_y`, the tank has no `y` maximum.

| Where | What it does | What is wrong |
|---|---|---|
| `native/include/fluidity/base/liquid_wall_contact_math_3d.h:365-554` `ResolveContact` | Shared CPU/GPU sweep of a point against one translating half space or slab: start classification, linear hit time, response, slide. | Every sign and distance goes through `ProjectionBounds`; an unresolved sign returns `kRange` (naming review A9). |
| same file `:101-149` `ApplyOneWayResponse` | Removes approach speed in world coordinates, then Coulomb friction toward the solid's velocity, per velocity. **This maths is ported.** | Ends with `CertifyContactSide`, which nudges the velocity by one representable step (`NudgeOutward`, `:57-99`). |
| same file `:154-363` `SweepTankAxis`, `ResolveTankContact` | Open tank with a moving low-x side (the piston), solved one axis at a time. | Axis-rewriting trick: the floor is built with normal `{1,0,0}` (A7). Works only with zero friction. The piston is fused into the tank. |
| `native/include/fluidity/base/liquid_boundary_math.h:85-249` | `AddDown`, `AddUp`, `ProjectionBounds`, `ProjectedSum`: error-free sums that enclose a dot product. | The rounding enclosure the owner removed. `:385-418` `ContactInput`/`ContactOutput` mirror the host types; `int diagnostic` duplicates an enum (A5). |
| `native/src/liquid_wall_contact_3d.cc:24-245` | Translating box: picks the entering face with bounded cross products, then calls the plane sweep; returns the impulse on the box. | A second sweep algorithm for one shape; `absl::Status` with empty messages. |
| `native/src/liquid_wall_contact_3d.cc:249-510`, `native/src/rigid_body_3d.cc:833-859, 883-1136` | Turning box: `SweepRigidBoxPoint3D` brackets the entry by bisection with a bound on gap travel (42-interval stack, 4096 probes, turn at most pi/2); the slide is carried in the box frame. | A third sweep. Box plus tank is a special function that returns `Aborted` when both could be hit. Host only. |
| `native/src/liquid_wall_contact_3d.cc:511-683` | Plate ("elastic blade") contact with a committed side `{blade_id, sign}`, initial side classification, face tags `-1/0/+1`. | One plate per particle. Face tagging needs the grid, so it belongs to the solids design. |
| `native/src/liquid_substep_3d_cpu.cc:190-239, 316-376`; `native/cuda/liquid_boundary_transfer_3d.cu:606` | The step calls contact per particle and sums box and piston impulses in particle order. | Four call shapes chosen by scene kind. The GPU path knows only half space, slab and tank. |
| `native/src/dry_solid_step_3d.cc:93-118` | Snow (and sand, which moves to the sand module): floor with friction, then a frictionless moving platen, as two calls. | Hard-wired `{0,1,0}` floor; a second hit is silently ignored. |
| `native/cuda/liquid_wall_contact_3d.cu`, `liquid_wall_contact_3d_host.cc` | One-particle kernel that uploads, runs and downloads one record. | A test harness shaped as a public function (`CudaResolveLiquidWallContact3D`). |
| `native/tests/test_liquid_wall_contact_3d.cc`, `liquid_wall_contact_3d_cases.h` | 27 tests; the closed-form cases are reused in section 7. | Several test only the rounding machinery. |

Report findings fixed here: tolerance must follow scale and must refuse features it cannot protect (numerics
10.4); pressure and contact impulses must not be counted twice and the record must separate the velocity for the
next step, the velocity for this step, hit time, normal and impulse (10.3); one motion description instead of
piston, box and plane paths (10.6, programme row 7); `side_tag` is an identity, not a three-value flag
(architecture, "Identity is not an index"); contact ties need a stable order (architecture line 227).

## 3. Public interface

Folder `native/fluidity/contact/`, namespace `fluidity::contact`. Headers marked *shared* compile for CPU and
GPU (C++20 subset, every function `FLUIDITY_CPU_AND_GPU`, `constexpr` where possible). Tags: D = DECIDED,
N = NEW.

### 3.1 `contact/contact_error.h` (shared) and `contact/contact_error_message.h` (CPU only)

```cpp
enum class ContactError : std::uint8_t {  // D (name and first four values)
  kNone,
  kBadInput,             // A value is not finite or is out of range.
  kStartsInsideSolid,    // The start is deeper inside a solid than start_overlap_tolerance allows.
  kTooCloseToCall,       // No safe answer for this step length. The caller should shorten the step.
  kOnWrongSideOfPlate,   // N. The start is on the other side of the plate the particle is committed to.
  kSolidTooThin,         // N. A solid or cell is thinner than 1024 contact tolerances: not enough resolution.
  kBuffersTooSmall,      // N. More particles or solids than the contact buffers were built for.
  kRunOnCallFailed,      // N. A call to the chosen processor failed.
  kOutOfMemory,          // NEW (A58; reconcile item 31). Last value. Only BuildContactBuffers: the run's RunMemory
                         // or the processor could not hold a buffer; magnitude = bytes asked.
};
enum class ContactCloseCall : std::uint8_t {  // N. Why kTooCloseToCall was returned.
  kNone, kTooManyHits, kSqueezedBetweenSolids, kSlowApproachToTurningSolid, kTurningSolidNearOtherSolid,
  kSecondReactingSolid, kSolidTurnsTooFar, kEndsTooCloseToSolid,
};
struct ContactFailure {  // N. Compact context; never allocates.
  ContactError error = ContactError::kNone;
  ContactCloseCall close_call = ContactCloseCall::kNone;
  std::uint64_t particle_id = 0;  // 0 when the failure is not about one particle.
  std::uint32_t solid_id = 0;     // 0 when the failure is not about one solid.
  double magnitude = 0;           // m. Overlap depth, or the offending length; bytes asked for kOutOfMemory (A58).
                                  // 0 when not relevant.
};
// contact_error_message.h. ToMessage: one English sentence. ToName (N): the stable code, the enumerator's
// name without the `k` in lower snake case (kSolidTooThin -> "solid_too_thin"); the CLI and the add-on use it.
std::string_view ToMessage(ContactError error);
std::string_view ToMessage(ContactCloseCall close_call);
std::string_view ToName(ContactError error);              // N
std::string_view ToName(ContactCloseCall close_call);     // N
```

### 3.2 `contact/solid_shapes.h` (shared)

```cpp
enum class SolidShape : std::uint8_t { kFlatSolid, kPlate, kBox, kDomainBox,  // N
                                      kSphere, kCylinder };                   // NEW (A74), appended
enum class SolidFace : std::uint8_t {   // N. In the solid's own axes.
  kLowX, kHighX, kLowY, kHighY, kLowZ, kHighZ,
  kCurved,    // NEW (A74). The round surface of a sphere, or the wall of a cylinder.
  kLowRim,    // NEW (A74). The round edge where a cylinder's wall meets its low-z cap.
  kHighRim,   // NEW (A74). The round edge where a cylinder's wall meets its high-z cap.
};
enum class SolidSide : std::int8_t { kNoPlate = 0, kAgainstNormal = -1, kAlongNormal = +1 };  // D name; N values
struct PlateSide {             // N. Stable identity of "one side of one plate". 8 bytes.
  std::uint32_t plate_id = 0;  // solid_id of the plate. 0 means no plate.
  SolidSide side = SolidSide::kNoPlate;
  friend constexpr bool operator==(PlateSide, PlateSide) = default;  // same_side is this equality.
};

struct FlatSolid {                      // D
  Vec3 point_on_surface{};              // m, world, at the start of the step.
  Vec3 normal_out_of_solid{0, 0, 1};    // Unit vector, world.
};
struct Plate {                          // D. Unbounded sideways (as on main); a finite thin solid is a Box.
  Vec3 center{};                        // m, world. Any point of the middle plane.
  Vec3 normal{0, 0, 1};                 // Unit vector, world. Points to the kAlongNormal side.
  double half_thickness = 0;            // m, positive.
};
struct Box {                            // D
  Vec3 center{};                        // m, world.
  Mat3 axes = kIdentity;                // Rotation. Columns are the box's x, y, z axes in world.
  Vec3 half_size{};                     // m, positive, along the box's axes.
};
struct Sphere {                         // NEW (A74)
  Vec3 center{};                        // m, world.
  double radius = 0;                    // m, positive.
};
struct Cylinder {                       // NEW (A74). Finite and solid, with two flat caps.
  Vec3 center{};                        // m, world. Midpoint of the axis.
  Vec3 axis{0, 0, 1};                   // Unit vector, world. The solid's own z axis; caps are at z = +-half_height.
  double radius = 0;                    // m, positive.
  double half_height = 0;               // m, positive, along the axis.
};
struct ClosedSides {                    // N. true: particles cannot cross that domain side.
  bool left = true;  bool right = true;  bool front = true;  bool back = true;  bool bottom = true;
  bool top = false;
};
struct DomainBox {                      // D. Fixed, aligned with the world axes.
  Vec3 low_corner{};                    // m. side_left (-X), side_front (-Y), side_bottom (-Z).
  Vec3 high_corner{};                   // m. side_right (+X), side_back (+Y), side_top (+Z).
  ClosedSides closed{};
};
struct SolidMotion {                    // N. Constant during one step.
  Vec3 linear_velocity{};               // m/s, world, of the solid's origin.
  Vec3 angular_velocity{};              // rad/s, world, about the solid's origin.
};
struct SolidPose {                      // N
  Vec3 origin{};                        // m, world.
  Mat3 axes = kIdentity;                // Rotation. Columns are the solid's x, y, z axes in world.
};
struct Solid {                          // D name. What the sweep consumes: shape, pose and velocity in a step.
  SolidShape shape = SolidShape::kFlatSolid;
  std::uint32_t solid_id = 0;           // Stable, nonzero, unique in the scene.
  SolidPose pose_at_step_start{};
  SolidMotion motion{};
  Vec3 half_size{};                     // m. Box, DomainBox: along the axes. Plate: z is half_thickness.
                                        // NEW (A74): Sphere {r, r, r}; Cylinder {r, r, half_height}.
  ClosedSides closed{};                 // DomainBox only.
  double friction_scale = 1;            // No unit, zero or more. Multiplies the particle's friction.
  bool reports_reaction = false;        // true for driven and free solids: hits fill SolidReaction.
};
inline constexpr std::uint32_t kNoSolid = 0xFFFFFFFFu;  // N. "No solid" as an index into the solids array.

// Shape structs, SolidMotion and SolidPose are passed by value in shared functions (AGENTS.md: small structs by
// value); Solid stays const Solid&.
Solid ToSolid(FlatSolid shape, SolidMotion motion, std::uint32_t solid_id);
Solid ToSolid(Plate shape, SolidMotion motion, std::uint32_t solid_id);
Solid ToSolid(Box shape, SolidMotion motion, std::uint32_t solid_id);
Solid ToSolid(DomainBox shape, std::uint32_t solid_id);
Solid ToSolid(Sphere shape, SolidMotion motion, std::uint32_t solid_id);    // NEW (A74)
Solid ToSolid(Cylinder shape, SolidMotion motion, std::uint32_t solid_id);  // NEW (A74)
inline constexpr double kMaxTurnPerStep = 1.5707963267948966;  // rad. N. A quarter turn, as on main. Declared here
                                                               // (Task 3), not in solid_contact.h.
// NEW (A74). The part of solid.motion.angular_velocity that changes the shape, rad/s, world: zero for a sphere, the
// part across the cylinder's own axis (omega - Dot(omega, e_z) e_z) for a cylinder, all of it for the others.
Vec3 ComputeShapeTurningAngularVelocity(const Solid& solid);

// Fails: kBadInput (not finite, axes not a rotation, size not positive, id 0, a moving DomainBox, a DomainBox whose
//        axes are not exactly the identity; NEW (A74):
//        a sphere whose three half_size values differ, a cylinder whose half_size.x != half_size.y),
//        kTooCloseToCall/kSolidTurnsTooFar (turn during the step above kMaxTurnPerStep; NEW (A74): for a sphere
//        and a cylinder only the shape-changing turn counts, see 5.7: never for a sphere, the turn across the
//        axis for a cylinder, so a fast spin about the axis is allowed).
//        The size limits come from CheckContactTolerance: 2 * radius and 2 * half_height enter thinnest_solid.
ContactError CheckSolid(const Solid& solid, double step_duration);
// Pose `elapsed` seconds after the start of the step, 0 <= elapsed <= step_duration.
SolidPose ComputeSolidPose(const Solid& solid, double elapsed);
// Velocity of the solid's surface at a world point, m/s.            D word: solid_velocity
Vec3 ComputeSolidVelocity(const Solid& solid, SolidPose pose, Vec3 point);
struct DistanceToSolid {                // N
  double distance = 0;                  // m. Positive outside the solid, negative inside.
  Vec3 normal_out_of_solid{};           // Unit vector, world, of the nearest face.
  SolidFace face = SolidFace::kHighZ;
};
DistanceToSolid ComputeDistanceToSolid(const Solid& solid, SolidPose pose, Vec3 point);
// Side of a plate a point is on. kNoPlate when the solid is not a plate or the point is inside its thickness.
SolidSide FindSolidSide(const Solid& solid, SolidPose pose, Vec3 point);

// N (added for pressure's open-sea settings). The Blender names of the six domain sides, in SolidFace order:
// kLeft = kLowX, kRight = kHighX, kFront = kLowY, kBack = kHighY, kBottom = kLowZ, kTop = kHighZ.
enum class DomainSide : std::uint8_t { kLeft, kRight, kFront, kBack, kBottom, kTop };
constexpr SolidFace ToSolidFace(DomainSide side);

// D name, N here (added for transfer and viscosity). The nearest solid surface seen from one point.
struct SolidVelocitySample {
  bool has_solid = false;               // A solid surface lies within `reach` of the point.
  std::uint32_t solid_index = kNoSolid; // Index into the solids array.
  PlateSide side{};                     // The plate side that surface faces. {0, kNoPlate} for any other shape.
  Vec3 normal_out_of_solid{};           // Unit vector, world. Zero without a solid.
  Vec3 solid_velocity{};                // m/s, world, of the surface point nearest to the sample point.
  double distance = 0;                  // m. Positive outside the solid, negative inside.
};
// Samples every solid `elapsed` seconds into the step and keeps the one with the smallest distance; ties keep
// the lower solid index. has_solid is false when that distance is larger than `reach`.
SolidVelocitySample ComputeSolidVelocitySample(const Solid* solids, std::uint32_t solid_count, double elapsed,
                                               Vec3 point, double reach);
```

Solid-frame conventions (stated in the header, drawn in the paper): a flat solid fills `z <= 0` and its only
face is `kHighZ`; a plate fills `|z| <= half_thickness`, `kHighZ` is the `kAlongNormal` face; a box fills
`|x|,|y|,|z| <= half_size`; for a domain box the liquid is inside and each closed side is an unbounded flat
face whose normal points into the box. `ToSolid(FlatSolid)` and `ToSolid(Plate)` build the axes with z along
the normal and x along `Cross(world axis with the smallest |normal part|, normal)`, normalised.
NEW (A74): a sphere fills `|p| <= radius` around its origin; `ToSolid(Sphere)` sets the axes to `kIdentity`
(a sphere's axes matter only through its turn, which moves its surface but not its shape). A cylinder fills
`rho <= radius` and `|p.z| <= half_height`, `rho = Sqrt(p.x^2 + p.y^2)`; `ToSolid(Cylinder)` builds the axes from
`axis` by the rule of the flat solid and the plate (z along the axis). Its surface has three regions, each a
`SolidFace`: the wall (`kCurved`), the caps (`kLowZ`, `kHighZ`) and the two rims (`kLowRim`, `kHighRim`), the
round edges where the nearest surface point is on the cap's circle.

### 3.3 `contact/contact_tolerance.h` (shared)

```cpp
inline constexpr double kContactToleranceRelative = 0x1p-40;          // N. Share of the largest coordinate.
inline constexpr double kContactToleranceFloorInCellWidths = 0x1p-30; // N
inline constexpr double kMinFeatureInTolerances = 1024;               // N
struct ContactScale {                   // N
  double largest_coordinate = 0;        // m. Largest |x|, |y| or |z| a solid origin takes during the step, plus
                                        // each bounded solid's reach Length(half_size) (plan P8); the domain box
                                        // brings the particles' extent.
  double smallest_cell_width = 0;       // m, positive.
  double thinnest_solid = 0;            // m, positive. Smallest plate thickness, box size or domain width;
                                        // NEW (A74): 2 * radius (sphere), min(2 * radius, 2 * half_height)
                                        // (cylinder).
};
ContactScale ComputeContactScale(const Solid* solids, std::uint32_t solid_count, double step_duration,
                                 double smallest_cell_width);
double ComputeContactTolerance(const ContactScale& scale);                      // m
ContactError CheckContactTolerance(const ContactScale& scale, double contact_tolerance);  // kSolidTooThin
```

### 3.4 `contact/solid_contact.h` (shared)

```cpp
enum class ContactKind : std::uint8_t { kNone, kHit, kStartedInsideSolid };  // D
inline constexpr int kMaxHitsPerStep = 4;        // N. Three sides of a corner plus one.
inline constexpr int kMaxAdvanceCount = 128;     // N. Search steps toward a turning solid.
// kMaxTurnPerStep (a quarter turn) lives in solid_shapes.h (3.2).

struct SolidContactInputs {              // D (first six fields), N (the rest)
  Vec3 start_position{};                 // m, world, at the start of the step.
  PerParticleVelocity<Vec3> velocity{};  // m/s, world, before the response. for_this_step moves the particle.
  const Solid* solids = nullptr;         // solid_count solids, in ascending solid_id order.
  std::uint32_t solid_count = 0;
  double step_duration = 0;              // s, positive.
  double friction = 0;                   // No unit, zero or more. Papers: Coulomb coefficient.
  double start_overlap_tolerance = 0;    // m, zero or more.
  double contact_tolerance = 0;          // m, positive. From ComputeContactTolerance.
  PlateSide plate_side{};                // The side this particle is committed to, or {0, kNoPlate}.
};
struct SolidReaction {                   // N. What a solid needs to compute the force on it.
  std::uint32_t solid_index = kNoSolid;  // Index into solids. kNoSolid: no reacting solid was hit.
  Vec3 hit_position{};                   // m, world, where the hit happened.
  Vec3 velocity_for_next_step_change{};  // m/s. velocity.for_next_step after minus before, caused by this solid.
};
struct SolidContactResult {              // D (position..kind), N (the rest)
  Vec3 position{};                       // m, world, at the end of the step.
  PerParticleVelocity<Vec3> velocity{};  // m/s, world, after the response.
  ContactKind kind = ContactKind::kNone;
  double hit_fraction = 0;               // When the first hit happens, 0..1 of the step. 0 without a hit.
  Vec3 normal_out_of_solid{};            // Unit vector at the first hit. Zero without a hit.
  PerParticleVelocity<double> approach_speed{};  // m/s at the first hit. Negative means approaching.
  std::uint32_t solid_index = kNoSolid;  // Solid of the first hit.
  SolidFace face = SolidFace::kHighZ;    // Its face.
  SolidSide side_at_hit = SolidSide::kNoPlate;  // Side of the plate that was hit, if it is a plate.
  std::uint8_t hit_count = 0;            // Hits during the step, 0..kMaxHitsPerStep.
  SolidReaction reaction{};
};
// Moves one particle for one step and stops it at solid surfaces. Docstring as naming review 6.3.
// On failure *result is unchanged and *failure (may be null) says which solid and how far.
ContactError MoveParticle(const SolidContactInputs& inputs, SolidContactResult* result,
                          ContactFailure* failure);

struct FirstHit {                        // N
  double elapsed = 0;                    // s since the start of the sweep.
  std::uint32_t solid_index = kNoSolid;  // kNoSolid: the path is free.
  DistanceToSolid at_hit{};              // Distance, normal and face at the hit.
};
// First time the straight path start + velocity * t, 0 <= t <= duration, comes within contact_tolerance of a
// solid. Solids are sampled `start_elapsed` seconds into the step. Fails with kTooCloseToCall.
ContactError FindFirstHit(const SolidContactInputs& inputs, Vec3 start, Vec3 velocity_for_this_step,
                          double start_elapsed, double duration, FirstHit* first_hit);
// One velocity after meeting a surface: no speed into the solid, sideways speed reduced by friction.
Vec3 ComputeContactResponse(Vec3 velocity, Vec3 solid_velocity, Vec3 normal_out_of_solid, double friction);
```

Deviations from the naming review's sketch, all forced by the shared-header rule: `hit_fraction` is a plain
`double` read together with `kind` (no `std::optional` on the device); `MoveParticle` returns the enum and
writes `ContactFailure` through a pointer (no `std::expected` in a shared header; the shared-header form of
A2); `solids` is pointer plus count.

### 3.5 `contact/move_particles.h` (CPU only), private `move_particles_cpu.cc`, `move_particles_gpu.cu`

```cpp
struct ParticleContact {                   // N. One per particle, kept where the simulation runs. 64 bytes.
  ContactKind kind = ContactKind::kNone;
  std::uint8_t hit_count = 0;
  SolidFace face = SolidFace::kHighZ;
  SolidSide side_at_hit = SolidSide::kNoPlate;
  std::uint32_t solid_index = kNoSolid;
  double hit_fraction = 0;
  Vec3 normal_out_of_solid{};
  PerParticleVelocity<double> approach_speed{};  // m/s
};
class ContactBuffers;  // N. Preallocated: a copy of the solids and one failure code per particle. Members:
// run_on(), max_particle_count(), max_solid_count(), and NEW (2026-11-15, sand; reconcile item 33)
// std::span<const Solid> solids_copy() const: the solids the last MoveParticles call wrote, on run_on() (empty
// before the first call); sand's GPU hit split reads it instead of copying the solids per call.
// Changed (A58): builds every buffer as a piece of `run_memory` (the run's one reserved block). Requires:
// run_on == run_memory.run_on() (debug assertion, A30). Fails kOutOfMemory (magnitude = bytes asked; pieces
// already taken stay in the block), kRunOnCallFailed.
std::expected<ContactBuffers, ContactFailure> BuildContactBuffers(RunOn run_on, std::size_t max_particle_count,
                                                                 std::size_t max_solid_count,
                                                                 RunMemory& run_memory);
struct MoveParticlesInputs {             // N. Borrowed pointers on the processor `buffers` was built for.
  std::size_t particle_count = 0;        // Particles of one material in one range of cell blocks.
  const std::uint64_t* particle_id = nullptr;
  const Vec3* start_position = nullptr;  // m
  PerParticleVelocity<const Vec3*> velocity{};  // m/s, before the response.
  const double* friction = nullptr;      // No unit, per particle (a mixture blends its ingredients; gas 0).
  const PlateSide* plate_side = nullptr; // Null: no particle of this material is committed to a plate.
  std::span<const Solid> solids;         // CPU memory, ascending solid_id. A few hundred bytes per solid.
  double step_duration = 0;              // s
  double start_overlap_tolerance = 0;    // m
  double contact_tolerance = 0;          // m
};
struct MovedParticles {                  // N. Trial outputs, same indexes as the inputs, same processor.
  Vec3* position = nullptr;              // m
  Vec3* velocity_for_next_step = nullptr;  // m/s, after the response.
  ParticleContact* record = nullptr;
  SolidReaction* reaction = nullptr;     // Null when no solid has reports_reaction.
};
struct MoveParticlesResult {             // N
  std::uint64_t hit_count = 0;             // Particles with kind == kHit.
  std::uint64_t started_inside_count = 0;  // Particles with kind == kStartedInsideSolid.
};
// Runs MoveParticle for each particle of the range. On failure the trial outputs hold unspecified values and
// must not be committed; the failure reported is the one with the lowest particle_id.
std::expected<MoveParticlesResult, ContactFailure> MoveParticles(const MoveParticlesInputs& inputs,
                                                                 MovedParticles& moved, ContactBuffers& buffers);
// N. The rows BuildContactBuffers allocates from, for the sizing estimate (base/memory_count.h). Changed (A58):
// one row per buffer, all kRunOn, kWholeRun: "contact: solids copy", "contact: failure per particle", "contact:
// call result" (1 MoveParticlesResult), "contact: first failure" (1 ContactFailure), so ComputeRunMemoryBytes of
// the rows equals the bytes BuildContactBuffers hands out.
std::vector<MemoryEstimateEntry> BuildContactMemoryEstimate(std::size_t max_particle_count, std::size_t max_solid_count);
```

The pointer fields are how the cycle with the particles module (built later) is broken: contact includes no
particles header. The step fills `MoveParticlesInputs` from `particles::ParticleFields` (`id`, `position`,
`velocity`, `plate_side`) and `MovedParticles` from `particles::ParticleFieldsToChange` (`position.write`,
`velocity.for_next_step.write`), one call per material and range of cell blocks. `record` and `reaction` point
into `Buffer<ParticleContact>` and `Buffer<SolidReaction>` that the step owns as working buffers. `BuildContactBuffers`
maps a `BufferError` to `kBuffersTooSmall` (`kSizeMismatch`), `kOutOfMemory` (`kOutOfMemory`, `kRunMemoryTooSmall`;
changed 2026-11-15, reconcile item 31: it was `kBuffersTooSmall`) or `kRunOnCallFailed`. Gas marker
particles are moved by the same call with `friction` 0; contact has no gas-specific case.

Not declared here: `math::Length`, `math::Normalized`, `math::IsFinite`,
`math::Rotation(Vec3 rotation_vector) -> Mat3` (turn by `|v|` rad about `v`),
`PerParticleVelocity<T> { T for_next_step; T for_this_step; }` (A1) and `Buffer<T>`. They are Foundation
additions in `base/`, declared once in `shared-interface.md` section (a).

## 4. Consumes from other modules

Each item is marked EXISTS (Foundation, including the `base/` additions) or PROVIDED (by the named module,
with the name as that module's section 3 has it). Contact includes headers from `base/` only; every PROVIDED
item below is reached through plain values or pointers the step passes, never through an include.

**Foundation (EXISTS):** `Vec3`, `Mat3`, `kIdentity`, `math::Dot`, `math::Cross`, `math::Transpose`,
`math::Determinant`, `Axis`, `kUp`, `math::UnitVector`, `RunOn`, `CheckCanRunOn(RunOn)`,
`FLUIDITY_CPU_AND_GPU`, `base/cpu_only.h`, the gate, `figure_style.typ`, `pseudocode.typ`.

**Foundation `base/` additions (EXISTS; `shared-interface.md` section (a) with A1 applied; they land before
this module starts):**

```cpp
template <typename T> struct PerParticleVelocity { T for_next_step; T for_this_step; };  // per_particle_velocity.h
namespace math { double Length(Vec3); Vec3 Normalized(Vec3); bool IsFinite(double); bool IsFinite(Vec3);
                 bool IsFinite(const Mat3&); double Sqrt(double); double Abs(double); double Sin(double);
                 double Cos(double); Mat3 Rotation(Vec3 rotation_vector); }    // base/scalar_math.h, vec3.h, mat3.h
template <typename Value> class Buffer;                                         // base/buffer.h
template <typename Value> std::expected<Buffer<Value>, BufferError> BuildBuffer(RunOn run_on, std::size_t size);
template <typename Value> std::expected<void, BufferError> WriteBuffer(std::span<const Value>, Buffer<Value>&);
template <typename Value> std::expected<void, BufferError> ReadBuffer(const Buffer<Value>&, std::span<Value>);
struct MemoryEstimateEntry;                                                               // base/memory_count.h
// EXISTS (A58; Foundation Tasks 17 and 18, base/run_memory.h): the run's one reserved block.
class RunMemory;  // run_on(), reserved_bytes(), handed_out_bytes(). Move-only.
template <typename Value>
std::expected<Buffer<Value>, BufferError> BuildBuffer(RunOn run_on, std::size_t size, RunMemory& run_memory);
std::uint64_t ComputeRunMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts,
                                    MemoryWhere where);  // Sums the entries kept `where` (item-3 correction).
// BufferError::kRunMemoryTooSmall (last value): a piece does not fit; contact maps it to kOutOfMemory.
```

The other `base/` additions of `changes-to-apply.md` (`CopyBuffer`, slice forms, `CpuMemory`,
`ReadMemoryAvailable`, `ReadProcessorDescription`, `math::Expm1`) are not used by contact.

**Grid (PROVIDED, grid 3.2):** `double grid::ComputeCellWidth(const GridFrame&, SizeLevel)`. The step
calls it for the smallest size level present and passes the result as `smallest_cell_width`. Contact reads no
cell and includes no grid header.

**Particles (PROVIDED, particles 3.1, 3.3, 3.5):** particles grouped so that a range of cell blocks
is one contiguous index range per material (`BuildFieldsToRead(particles_try, material_index, cell_blocks)`);
`ParticleFields::id` (stable `std::uint64_t`), `position`, `velocity` (`for_next_step` and `for_this_step`),
`plate_side` (`contact::PlateSide`, carried through `SplitParticle`, `MergeParticles` (merge only when
`same_side`), regrouping and snapshots); `ParticleFieldsToChange::position` and `velocity.for_next_step` as
the trial copies; `RemoveParticles(..., RemoveReason::kLeftDomain)`, called by the step for particles that end
outside an open domain side.

**Transfer (PROVIDED, transfer 3, `mix_particle_velocity.h`):** `MixResult::velocity`
(`for_next_step` and `for_this_step`) from `MixParticleVelocity`, written by the step into the particles' try
before `MoveParticles`. Transfer gates particle-face exchange on `PlateSide` equality (`same_side`) through
`transfer::SolidsAtFaces::side`.

**Materials (PROVIDED, materials 3):** `MaterialBuffers::friction(material_index)`, one `double` per
particle in grouped order on the simulation's processor, filled by `ComputeParticleFriction` before stage
`MoveParticles`; it is `MoveParticlesInputs::friction`. The gas material's friction is 0.

**Solids (PROVIDED, solids 3.4, 3.5, 3.7, 3.8):** `SolidsTry::solids()`, the `Solid` array of each
try (ascending `solid_id`, pose at step start, motion constant during the step; `BuildTry` fails with
`kPathJumpInsideStep` instead of crossing a jump of a driven path), with `friction_scale` copied from
`SolidSettings::friction_scale` and `reports_reaction` set for driven and free solids. `ApplyContactReactions`
sums `mass * SolidReaction::velocity_for_next_step_change` and its moment per solid in ascending `particle_id`
(`ReactionFields`). `ComputeSolidsAtFaces` and `ComputeSolidCoverage` use `FindSolidSide`,
`ComputeSolidVelocitySample`, `ComputeDistanceToSolid` and `ComputeSolidVelocity`; `ApplyPlateSides` and
`ApplyPlateRemoval` assign and clear `PlateSide` (cutting); `CollideSolids` owns contact between two materials;
`BuildContactSolids` builds `contact::Solid` values for render output;
`CheckSolidsCanRun(settings, smallest_cell_width, largest_cell_width)` checks the scene rules beside contact's
`CheckSolid`.

**Gas (PROVIDED, gas 5):** gas marker particles are moved by `MoveParticles` like every particle,
with friction 0. Gas's closed sides are `DomainBox::closed`.

**Step (PROVIDED, step 3.1 and 5.5):** calls `ComputeContactScale`, `ComputeContactTolerance`,
`CheckContactTolerance` and `CheckSolid` in its capability check and once per step (`kSolidTooThin` becomes
`CapabilityReason::kSolidThinnerThanTolerance`); owns the `Buffer<ParticleContact>` and `Buffer<SolidReaction>`
working buffers; fills `MoveParticlesInputs` and `MovedParticles` from the particles' try, one call per material and
range; runs `MoveParticles` in stage `Stage::kMoveParticles`; turns `kTooCloseToCall` into
`RejectReason::kContactTooCloseToCall` and retries with a shorter step; owns the travel limit per step.

**For the sand module:** sand reads `ParticleContact` (`kind`, `hit_fraction`, `normal_out_of_solid`,
`approach_speed`) to split its deformation at a hit (today `ProjectStationaryFloorVelocityGradient3D`,
`dry_solid_step_3d.cc:134`). Contact provides the record as declared in 3.5; nothing sand-specific is added.

Per material, contact carries: liquids and mixtures, a blended `friction` (often 0) and two different
velocities; snow, `friction` and the record; cut solids, `PlateSide`; moving solids, `SolidReaction`; gas
markers, friction 0 and nothing else.

## 5. Method

Notation for this section only: `T` step duration, `tol` contact tolerance, `d` distance to a solid.

**5.1 Shapes (new, replaces three shape-specific paths).** `ComputeSolidPose`: `origin = origin0 +
linear_velocity * t`, `axes = math::Rotation(angular_velocity * t) * axes0` (the paper's `A(t) = Rot(omega t) A`).
`ComputeSolidVelocity`:
`linear_velocity + Cross(angular_velocity, point - origin)`. `ComputeDistanceToSolid`, with `p` the point in
the solid's axes: flat solid `d = p.z`; plate `d = |p.z| - half_thickness`; box `d = max over axes of
(|p.axis| - half_size.axis)` (the gap to the nearest face plane, positive exactly when outside); domain box
`d = min over closed sides of (half_size.axis - s * p.axis)`, `s = -1` for the low side. Ties take the first
face in the order `kLowX, kHighX, kLowY, kHighY, kLowZ, kHighZ`; so a box reports its low face on its middle plane
(`p.axis <= 0` is `kLow`), a plate and a cylinder's caps report `kHighZ` at `p.z = 0`, and an inside tie in a cylinder
goes to the wall. Sphere and cylinder (NEW, A74): see 5.7; their
distance is the true (Euclidean) signed distance to the surface, not a largest face gap.

**5.2 Tolerance (new, replaces `ProjectionBounds`, `NudgeOutward`, `CertifyContactSide`).**

```text
contact_tolerance = kContactToleranceFloorInCellWidths * smallest_cell_width
                  + kContactToleranceRelative * largest_coordinate
```

Evaluating `d` with plain doubles is wrong by at most about `32 * epsilon * largest_coordinate` (a subtraction,
a rotation, a three-term dot product). `kContactToleranceRelative` is `4096 * epsilon`, so `tol` is at least
128 times that error. The floor keeps `tol` positive for a scene at the origin. Three rules use it:

- *The particle keeps a rest gap.* Every sweep is run against the solid grown by `tol`: a hit is `d == tol`,
  and a particle in contact rests at `d = tol`, never at `d = 0`. A later evaluation of `d` by any module,
  with its own rounding, is therefore positive. This is the guarantee the nudging gave.
- *The end is checked.* After the move, `d >= tol / 2` at the end pose for every solid, else
  `kTooCloseToCall/kEndsTooCloseToSolid`.
- *The scale is checked.* `CheckContactTolerance` returns `kSolidTooThin` unless
  `kMinFeatureInTolerances * tol <= min(thinnest_solid, smallest_cell_width)`. With the defaults this fails
  only when a feature is about `1e9` times smaller than its distance from the origin. The tolerance is never
  enlarged to make contact "work" (report 10.4).

`start_overlap_tolerance` stays a separate user value (default 0): how deep a start may be before it is an
error. Unit vectors and rotations are accepted when their squared length is within `0x1p-40` of 1 (was `1e-12`).

**5.3 `FindFirstHit`.** For each solid in array order (ascending `solid_id`), with gap `g = d - tol`:

1. *Solid that does not turn (closed form, exact first hit).* Each face gap is linear in time:
   `g_face(t) = g_face(0) + rate * t`, `rate = Dot(normal, velocity_for_this_step - linear_velocity)`. *Touching
   rule* (plan P2): when the start's gap is in `[-tol, 0]` for the surface that gives `d` (a plate's or box's face
   with the largest gap, the one face of a flat solid or domain side), that surface is not hit when
   `rate * duration >= -tol` and is hit at `t = 0` otherwise; a start gap below `-tol` is hit at `t = 0`. Flat solid and each closed domain
   side: hit at `t = g(0) / -rate` when `rate < 0` and `t <= duration`. Plate and box (convex): entry time is the
   largest `g_face(0) / -rate` over faces with `g_face(0) > 0, rate < 0`; exit time is the smallest
   `-g_face(0) / rate` over faces with `g_face(0) <= 0, rate > 0`; a face with `g_face(0) > 0, rate >= 0` means
   no hit; hit when `entry <= exit` and `entry <= duration`. The entering face is reported; equal entry times
   keep the first face in face order. A plate uses only the face of `plate_side` when the particle is committed.
2. *Turning solid (ported idea of `SweepRigidBoxPoint3D`, simplified to stepping forward).* The gap cannot
   shrink faster than `bound = |velocity_for_this_step - linear_velocity| + |angular_velocity| * radius`, `radius` the
   larger of the particle's distances to the solid's origin at both ends of the path. Start at `t = 0`; repeat
   `t += g(t) / bound`; stop with a hit when `g(t) <= tol / 4`, with no hit when `t >= duration`. More than
   `kMaxAdvanceCount` steps returns `kTooCloseToCall/kSlowApproachToTurningSolid`. No stack, no bisection.
   There is no touching rule here (amended 2026-11-15, paper review): it belongs to the closed forms of step 1. A
   particle resting on a turning solid at `d = tol` starts with `g(0) = 0 <= tol / 4` and is hit at `t = 0` in every
   step; the response and the slide in the solid's axes (5.5 step 5) carry it with the solid for the rest of the
   step. This is intended: the face it rests on turns, so the plane argument of the touching rule does not hold.
3. Keep the smallest hit time. Equal times keep the lower solid index.

Sphere and cylinder (NEW, A74): step 1 has a closed form for each (5.7); step 2 serves a cylinder whose axis turns.

**5.4 `ComputeContactResponse` (ported unchanged from `ApplyOneWayResponse`, minus the nudge).** With
`relative = velocity - solid_velocity` and `approach_speed = Dot(normal_out_of_solid, relative)`:

1. If `approach_speed >= 0` the velocity is returned unchanged (a solid only blocks motion into it).
2. Remove the approach in world coordinates: `velocity -= approach_speed * normal`. Recompute the remaining
   approach once and remove it too. (World coordinates, because a round trip through a fast solid's frame
   would erase a small unchanged sideways velocity. Kept as a body comment.)
3. Friction, when `friction > 0`: `sideways = velocity - solid_velocity`, `sideways_speed = |sideways|`,
   `into_solid_speed = -approach_speed`. If `sideways_speed == 0` or
   `friction >= sideways_speed / into_solid_speed`, the particle sticks: `velocity = solid_velocity`. Otherwise
   `velocity += (friction * into_solid_speed / sideways_speed) * (solid_velocity - velocity)`. The comparison is
   written as a division so a large friction cannot overflow.

The velocity for the next step and the velocity for this step each get the response with their own approach
speed.

**5.5 `MoveParticle`.** Steps: 1 check inputs, 2 classify the start, 3 find the first hit, 4 respond, 5 slide,
6 settle and check the end.

1. *Check.* Finite values, `step_duration > 0`, `friction >= 0`, tolerances in range, solids ordered: else
   `kBadInput`.
2. *Classify the start* against every solid at `t = 0`, in array order, each from the start as moved by the earlier
   solids; the first failing solid decides (amended 2026-11-15, plan P3, P4, P12, as the paper). Committed to this
   plate and on its other side by more than `tol`: `kOnWrongSideOfPlate` (checked first). `d <
   -start_overlap_tolerance`: `kStartsInsideSolid`. `d < tol`: the start is moved along the normal to `d = tol`;
   then, by `d` before the move: `d < 0`: both velocities get the response, `kind = kStartedInsideSolid`,
   `hit_fraction = 0`. `0 <= d < tol` and either velocity approaches: same, with `kind = kHit`. `0 <= d < tol` and
   neither approaches: the move is silent (velocities, kind, hit count and record unchanged; the solid does not join
   the surfaces, so a touching particle that leaves is free).
3. *Find the first hit* of the straight path along `velocity.for_this_step` over the time left.
4. *Respond* (5.4) to both velocities with `friction * solid.friction_scale`, after moving a rim hit of the
   cylinder's closed form to `d = tol` (5.7); record the first hit; add the change of `velocity.for_next_step` to
   `reaction` when the solid has `reports_reaction`. A second reacting solid in one step is `kSecondReactingSolid`.
5. *Slide.* The solid joins the list of surfaces the particle is on. For each earlier surface the new velocities
   still approach, that approach is removed without friction; if the newest surface is then approached again (a
   corner sharper than a right angle) the velocities are set to the newest solid's velocity; if an earlier one is
   still approached the solids are closing on the particle: `kSqueezedBetweenSolids`. Every change these make to
   `velocity.for_next_step` counts in `reaction` like the response's (amended 2026-11-15, plan P5): a removal is
   charged to the earlier surface's solid, the corner case to the newest solid, each only when that solid has
   `reports_reaction`. On a solid that does not
   turn, go to step 3 with the hit position, the new `velocity.for_this_step` and the time left; a fifth hit is
   `kTooManyHits`. On a turning solid the rest of the step is carried in the solid's axes, as on `main`: the
   relative `velocity.for_this_step` at the hit is turned into the solid's axes, its part into the face is clamped
   to zero, the particle advances in those axes and is placed with the end pose; if any other solid is within the
   distance both could travel, `kTurningSolidNearOtherSolid` (plan P7's bound). Expected behaviour (amended
   2026-11-15): since a particle resting on a turning solid is hit at `t = 0` in every step (5.3 step 2), it runs this
   test over the whole step, so a turning solid close to another one (a paddle near the domain floor) can ask the step
   for shorter steps more often than its own speed alone suggests; that is the price of carrying resting particles
   safely (plan Task 13 test).
6. *Settle and check.* For each surface the particle is on with `d < tol` at the end pose, move it along that
   normal to `d = tol`. Then apply the end check of 5.2 to all solids.

Without a hit the end position is `start_position + velocity.for_this_step * step_duration`, one multiply and one
add per axis, and both velocities are returned unchanged.

**5.6 What changes against `main`.** Unchanged: the response formula, linear hit time, start rules, the separate
treatment of the two velocities, only the velocity for the next step counted as reaction, turn limit of a quarter
turn. Fixed: friction on domain sides (the per-axis tank is gone); a hit followed by another hit in the same step
is handled instead of ignored or aborted; every shape may translate and turn through one `SolidMotion`; the wave
paddle is its own driven flat solid; one sweep for all shapes on both processors. New: the tolerance of 5.2. Z-up:
a domain box open at `+Z`; face and tie order is x, y, z of the solid's axes, so the bottom is now last where the
floor was second; a value from `main` enters through `(x, y, z) -> (x, -z, y)`, which sends the old floor to
`side_bottom`, old low-z to `side_back` and old high-z to `side_front`.

Contact makes no sums. Order matters in two places only: solids are visited in ascending `solid_id`, faces in
face order.

**5.7 Sphere and cylinder (NEW, A74).** In the solid's axes, `p` the point, `tol` the contact tolerance, `u =
velocity_for_this_step - linear_velocity` turned into the solid's axes, `p0` the path start in those axes.

*Distance and normal, sphere.* `L = Sqrt(Dot(p, p))`, `d = L - radius`, normal `p / L`, face `kCurved`. At the
centre (`Dot(p, p) == 0` as computed) the normal is the solid's own -x axis, the normal of `kLowX`, the first face
in face order (degenerate-normal rule, the box's tie rule; as the paper states).

*Distance and normal, cylinder: one formula for the three regions.* `rho = Sqrt(p.x^2 + p.y^2)`,
`wall_distance = rho - radius` (the paper's `d_wall`: the signed distance to the wall's surface, taken as unbounded
in height), `cap_distance = |p.z| - half_height` (`d_cap`: the signed distance to the plane of the nearer cap),
`cap_sign = +1` when `p.z >= 0` else `-1` (`sigma_z`), `radial = (p.x / rho, p.y / rho, 0)` (`e_rho`), `e_z` the
cylinder's own z axis:

```text
if wall_distance > 0 and cap_distance > 0:
    rim   d = Sqrt(wall_distance^2 + cap_distance^2)
          normal = (wall_distance * radial + cap_distance * cap_sign * e_z) / d    face kHighRim (cap_sign +1) / kLowRim
else if wall_distance >= cap_distance:
    wall  d = wall_distance    normal = radial            face kCurved
else:
    cap   d = cap_distance     normal = cap_sign * e_z    face kHighZ (cap_sign +1) / kLowZ
```

The distance is evaluated as one expression for every region (the paper's `<eq:cylinder-distance>`):
`d = Sqrt(max(wall_distance, 0)^2 + max(cap_distance, 0)^2) + min(max(wall_distance, cap_distance), 0)`; the
branches above choose only the normal and the face. They meet continuously (`wall_distance = 0` or
`cap_distance = 0` gives the same `d` from both sides). Inside (both `<= 0`) the nearer of wall and cap wins, a tie
takes the wall. On the axis (`p.x^2 + p.y^2 == 0` as computed) in the wall branch, `radial` is the solid's own -x
axis (degenerate-normal rule, as for the sphere; reached only when `|p.z| <= half_height - radius`, so only inside a
cylinder at least as long as it is wide). Every operation is `+ - * /`, `Abs` or `Sqrt`.

*Surface velocity.* Unchanged: `linear_velocity + Cross(angular_velocity, point - origin)`.

*First hit, sphere (closed form).* With `R = radius + tol`, `L0 = Length(p0)`, `g0 = L0 - radius - tol`:
`A = Dot(u, u)`, `B = Dot(p0, u)`, `C = g0 * (L0 + R)` (equal to `Dot(p0, p0) - R^2`, written from `g0` so its sign
is exactly the sign of the gap the distance gives). No hit when `A == 0`, when `B >= 0` (the approach condition:
not moving toward the centre), or when `B^2 - A * C < 0` (a miss). Otherwise
`t = C / (-B + Sqrt(B^2 - A * C))`, the smaller root written without cancellation, a hit when `t <= duration`.

*First hit, cylinder (closed form).* The cylinder is grown by `tol` with sharp edges: radius `R = radius + tol`,
half height `H = half_height + tol`. Candidates, each accepted only inside the other's range:
(1) wall: the quadratic of the sphere in the `x, y` components (`A = u.x^2 + u.y^2`, `B = p0.x u.x + p0.y u.y`,
`C = (rho0 - radius - tol) * (rho0 + R)`), tried when the start is outside the grown radius (`rho0 > R`), `A > 0`,
`B < 0` and the root exists (`B^2 >= A * C`), its first root `t_w` accepted when `|p0.z + u.z t_w| <= H`;
(2) the cap the particle is beyond (`|p0.z| > H`, `s0` its sign), when `s0 * u.z < 0`:
`t_c = (|p0.z| - H) / (-s0 * u.z)`, accepted when `(p0.x + u.x t_c)^2 + (p0.y + u.y t_c)^2 <= R^2`.
The earliest accepted candidate is the hit; this is the entry into the intersection of the grown slab and the grown
infinite cylinder, so the rim is covered by the two ranges' edges. A path parallel to the axis (`A == 0`) has no
wall candidate; a path parallel to the caps (`u.z == 0`) no cap candidate. The reported face is the region
`ComputeDistanceToSolid` gives at the hit position. At a hit of this closed form in the rim region the true distance
is between `tol` and `Sqrt(2) * tol` (the sharp corner of the grown shape); `MoveParticle` moves that hit position
along the rim normal to `d = tol` before the response, a move of at most `(Sqrt(2) - 1) * tol`, never toward the
inside. The move applies only to hits of this closed form (amended 2026-11-15, plan P15): a hit of the stepping
search toward a turning cylinder stops at `d <= 1.25 tol` in any region and is never moved, even when its reported
face is a rim.

*Starts within the grown shape.* For both shapes the start gap `g0 = d - tol` uses the true distance:
`g0 < -tol` is a hit at `t = 0` (defensive, as P2); `-tol <= g0 <= 0` is the touching rule below; for a cylinder,
`g0 > 0` inside the grown sharp shape (`rho0 <= R` and `|p0.z| <= H`: a rim corner, up to rounding, since these
tests and the gap are computed separately) is a hit at `t = 0` when `rate < 0` and free otherwise.

*Touching rule on a curved surface (counterpart of P2).* `rate = Dot(normal at the start, u)`. A sphere or cylinder
with `-tol <= g0 <= 0` is not hit during this sweep when `rate * duration >= -tol`, and is hit at `t = 0`
otherwise. Why it is safe: the true signed distance of a convex solid is a convex function of position, so along the
straight path `d(t) >= d(0) + rate * t >= d(0) - tol`. After a hit the particle starts at `d = tol` (the hit rule,
the rim move above, P12), so the path never comes closer than `0` up to rounding, and the settle of 5.5 step 6
puts it back at `tol` (the solid is on the surface list). The same tolerance as P2; no curvature term. Like P2 it
belongs to the closed forms only: a cylinder turning across its axis has no touching rule (5.3 step 2).

*The slide on a curved surface.* After a hit the response leaves a velocity tangent to the surface. The re-sweep
follows that tangent in a straight line: on a sphere or a cylinder wall it leaves the surface (the distance grows
as `(speed * time)^2 / (2 * radius)`); on a cap it stays on the plane as on a flat face. By the convexity argument
the straight tangent path cannot re-enter the same solid in this step, so each convex solid is hit at most once per
sweep and `kMaxHitsPerStep` is not reached by sliding round a sphere. The particle comes back to the surface in a
later step under gravity or pressure; over many steps it follows the curve (test: leaving a frictionless sphere at
`cos theta = 2/3`).

*Spin about a symmetry axis.* A turn about a sphere's centre or about a cylinder's own axis moves the surface but
not the shape. The *shape-changing turn* is: none for a sphere; for a cylinder `omega_across = omega - Dot(omega,
e_z) e_z`, `e_z` its own z axis (its `axis`) at the start of the sweep. Code computes it with the public
`ComputeShapeTurningAngularVelocity` (3.2); no other helper does. A sphere is always swept and slid as a solid that does not turn
(closed form); a cylinder is too when `Length(omega_across) * step_duration * Length(half_size) <= tol / 8` (the
shape moves less than `tol / 8` during the step: a spin given along the axis, up to rounding), and otherwise as a turning solid (5.3 step 2, P7 slide in its axes; the `kMaxTurnPerStep`
limit applies to `Length(omega_across) * step_duration`). In every case `ComputeSolidVelocity` uses the full
`omega`, so friction feels the spin and a frictionless particle is not dragged by it.

*Turning cylinder.* The stepping search of 5.3 step 2 with the cylinder's true distance; nothing else changes.

*Response.* Unchanged (5.4): it works on a normal, whichever shape gave it.

## 6. CPU and GPU

**Shared:** everything in 3.1 to 3.4. `MoveParticle` is the same code on both sides and uses only `+ - * /`,
`sqrt`, comparisons, and `sin`/`cos` inside `math::Rotation` (turning solids only).

Same bits (NEW, A74): the sphere and cylinder branches of `ComputeDistanceToSolid` (any pose given), of
`ComputeSolidVelocitySample` and `FindFirstHit` (solids with `angular_velocity` exactly zero), the rim move and the
touching rule use only `+ - * /`, `math::Abs` and `math::Sqrt` (the same-bits group of `base/scalar_math.h`), so
they give the same bits on CPU and GPU unless NaN, like the other shapes. None calls a function of the last-bits
group (`Sin`, `Cos`, `Exp`, `Log`, `Pow`, `Cbrt`, `Expm1`); there is no `acos` or `atan` anywhere in contact (tests
may use `std::asin` and `std::acos` on the CPU for expected values). A spinning sphere or cylinder (angular velocity
not zero, shape not turning) and a turning cylinder reach `math::Rotation` through `ComputeSolidPose` and agree
within the tolerance below.

**CPU (`move_particles_cpu.cc`):** one loop over the index range calling `MoveParticle`. Particles are
independent, so the range may be cut into fixed chunks for threads. Each thread keeps its own failure with the
lowest `particle_id`; the chunks are merged in chunk order. Counts are integer sums.

**GPU (`move_particles_gpu.cu`):** one thread per particle. Arrays (owner in brackets): `particle_id` `uint64`,
`start_position` `Vec3`, `velocity.for_next_step` and `velocity.for_this_step` `Vec3`, `friction` `double`,
`plate_side` `PlateSide` (particles, read only); trial `position`, trial `velocity_for_next_step` (particles,
written); `ParticleContact`, optional `SolidReaction` (step working buffers, written, read later by materials and solids on
the device); the solids copy and one `ContactFailure` per particle (`ContactBuffers`). All stay on the device. Per
call the CPU sends the solids (`solid_count * sizeof(Solid)`, well under a megabyte) and the scalar inputs, and
receives one `MoveParticlesResult` and one `ContactFailure`. A second function reduces the failures to the one
with the lowest `particle_id` (an integer minimum, the same on every run) and counts kinds with integer sums. No
floating-point value is ever summed, so there is no order to fix.

**Range of cell blocks:** the range is the particle index range of those blocks. Contact reads no cell, so it
needs no border cells. A particle may end outside the range's blocks; regrouping is the particles module's (module
3). In a batched run the solids are the same for every batch.

**Fusing:** `MoveParticle` may also be called from inside another module's per-particle function (as `main`
does after its grid-to-particle read); the batch operation is the form this module ships and tests.

**CPU versus GPU tolerance (proposal, calibrated by the hardware run):** `kind`, `solid_index`, `face`,
`side_at_hit`, `hit_count` and the error are compared exactly. A floating-point value `a` against `b` passes
when `|a - b| <= floor + 0x1p-40 * max(|a|, |b|)` with `floor = contact_tolerance / 4` for positions,
`0x1p-40 * largest input speed` for velocities and speeds, `0x1p-40` for `hit_fraction`. Without a turning
solid the expected difference is zero, and the test prints the largest difference seen.

## 7. Tests

`*_test.cc` beside the code, `TEST(FunctionOrType, Situation_Outcome)`. Unless stated, the tolerance is
`2 * contact_tolerance` on positions and `0x1p-44` relative on velocities and fractions. Numbers are in Z-up.

Analytic, shapes: distance, normal and face of each shape at points inside, outside and on a tie; pose after a
quarter turn about Z sends the solid's x axis to world y; surface velocity equals
`linear + Cross(angular, arm)`; `FindSolidSide` on both sides and inside the thickness.

Analytic, response: head-on, `velocity` becomes `solid_velocity`; slip, incoming `(sideways 3, into 1)` with
friction 0.5 leaves sideways 2.5; stick at and above `friction = 3`; leaving and sideways motion unchanged;
fixed solid never raises `|velocity.for_next_step|`; friction 0 keeps the sideways part bit for bit, including a
sideways `1e-9` against a floor moving sideways at `1e8` m/s (amended 2026-11-15: velocity `{1e-9,0,-1}`, solid
velocity `{1e8,0,0}`, normal `{0,0,1}` gives exactly `{1e-9,0,0}`; a response computed in the solid's frame gives
`{0,0,0}`, so the test catches that bug; a solid moving along the normal could not).

Analytic, sweep and move: no solid, end is `start + velocity.for_this_step * T` exactly; the docstring example
(`x = 0.9`, `+1` m/s, face at `1.0`, `T = 0.2`: `hit_fraction` 0.5, end `x = 1 - tol`, speed 0, `approach_speed` -1);
flat solid pushing a resting particle at `V` (particle ends carried, normal speed `V`); plate crossed from each
side on all three axes and with reversed normal, with the three rows of the old reference fixture as closed forms
(start -1, end +1, centre 0, half thickness 1/32: fraction `(1 - 1/32) / 2`); plate with a committed side; box
face hit, corner tie reports `kLowX`, edge graze stays free, hit exactly at the end of the step; turning box, a
resting particle at a known radius is reached at the angle given by trigonometry (tolerance `0x1p-30`); domain box
corner with three sides hit in one step; friction on `side_bottom`; exit through the open top is free; two solids,
lower `solid_id` wins a tie; one error test per `ContactError` and per `ContactCloseCall`;
`ToName, EveryValue_IsSnakeCaseOfEnumerator` for both enums (`kSolidTooThin` gives `solid_too_thin`, every
name unique, `kNone` gives `none`).

Analytic, sphere and cylinder (NEW, A74; `tol = 0x1p-30`, positions within `1e-15` absolute where the inputs are
exact binary fractions, fractions and velocities as above unless stated):

- Sphere centre `{1,2,3}`, radius 0.5. Distance and normal: `{1,2,4}` gives 0.5, `{0,0,1}`; `{1.375,2,3.5}` gives
  0.125, `{0.6,0,0.8}`; inside `{1,2,3.25}` gives -0.25, `{0,0,1}`; the centre gives -0.5 and the solid's -x axis `{-1,0,0}`
  (degenerate normal). Face `kCurved` each time.
- Cylinder at the origin, axis z, radius 1, half height 0.5. Wall `{3,0,0.25}`: 2, `{1,0,0}`, `kCurved`; high cap
  `{0.5,0,2}`: 1.5, `{0,0,1}`, `kHighZ`; low cap `{0,0.5,-1.5}`: 1, `{0,0,-1}`, `kLowZ`; high rim `{1.375,0,1}`
  (`wall_distance` 0.375, `cap_distance` 0.5): 0.625, `{0.6,0,0.8}`, `kHighRim`; low rim `{0,-1.375,-1}`: 0.625, `{0,-0.6,-0.8}`,
  `kLowRim`; inside near the wall `{0.9,0,0}`: -0.1, `{1,0,0}`, `kCurved`; inside near a cap `{0,0,0.375}`: -0.125,
  `{0,0,1}`, `kHighZ`; inside tie `{0.75,0,0.25}` (`wall_distance = cap_distance = -0.25`): wall, -0.25, `{1,0,0}`; on the axis of a
  cylinder radius 0.25, half height 2, at the origin: -0.25 and the solid's -x axis `{-1,0,0}` (degenerate normal); a cylinder
  with axis `{1,0,0}` at `{0.25,3,0}`: 2, normal `{0,1,0}` (solid axes x `{0,0,-1}`, y `{0,1,0}`, z `{1,0,0}`).
- First hit, sphere at the origin radius 0.5, `duration 2`: head-on from `{-2,0,0}` at `{1,0,0}`: `1.5 - tol`,
  normal `{-1,0,0}`; grazing from `{-2,0.3,0}`: `2 - Sqrt((0.5 + tol)^2 - 0.09)` (= 1.5999999988358469), normal
  about `{-0.8,0.6,0}`; tangent within rounding from `{-2, 0.5 + tol, 0}`: no error, either free or a hit within
  `0x1p-20` of 2 (both are safe; the move test checks the end distance); a miss from `{-2,0.6,0}`; moving away
  (`{-1,0,0}`): free; a path starting at `d = tol` along `{0,1,0}`: free (touching rule), along `{1,0,0}`: hit at 0;
  sphere centre `{2,0,0}` moving `{-1,0,0}` toward a particle at rest at the origin: `1.5 - tol` (relative
  velocity); a sphere spinning `{0,0,5}` rad/s: the same times by the closed form (not stepped).
- First hit, cylinder at the origin, axis z, radius 0.5, half height 1: through the wall from `{-2,0,0.25}` at
  `{1,0,0}`, `duration 2`: `1.5 - tol`, `kCurved`; through the cap from `{0.1,0.2,3}` at `{0,0,-1}`, `duration 4`:
  `2 - tol`, `kHighZ`; oblique wall hit from `{-2,0,1.5}` at `{1,0,-0.5}`, `duration 4`: `1.5 - tol`, `kCurved`
  (the cap plane is crossed at radius `1 + 2 tol`, rejected); over the rim into the wall from `{-2,0,2}` at
  `{1,0,-1}`: `1.5 - tol`, `kCurved`; parallel to the axis outside, from `{0.6,0,3}` at `{0,0,-1}`: free; parallel to
  the axis inside the grown radius, from `{0.5 + tol/2, 0, 3}`: `2 - tol`, `kHighRim`, true distance at the hit
  `Sqrt(1.25) * tol` within `0x1p-40`; parallel to the caps above them, from `{-2,0,1.5}` at `{1,0,0}`: free; a
  cylinder spinning `{0,0,5}` rad/s about its axis, `step 1`: closed form, same times, and `CheckSolid` passes; a
  cylinder with axis `{1,0,0}` turning `{0,0,1}` rad/s across its axis, particle at rest at
  `0.6 * {cos 1.2, sin 1.2, 0}`, `duration 1`: `1.2 - asin((0.5 + tol) / 0.6)` (= 0.21488921385421111) within
  `0x1p-30`, `kCurved` (stepping search).
- `CheckSolid`: radius 0, a negative half height, a sphere with unequal half sizes, a cylinder with
  `half_size.x != half_size.y`: `kBadInput`; a cylinder tumbling at 2 rad/s across its axis for 1 s:
  `kSolidTurnsTooFar`; a sphere or a cylinder spinning 100 rad/s about its axis for 0.1 s: passes.
- Move, sphere at the origin radius 0.5, friction 0, `T 2`: head-on stops at `x = -(0.5 + tol)` with velocity 0;
  oblique from `{-2,0.3,0}` at `{1,0,0}`: one hit (`hit_count` 1: the re-sweep along the tangent is not a hit),
  velocity `{0.36,0.48,0}` and end `{-0.256,0.492,0}` within `1e-8`; the tangent-within-rounding path: no error,
  every end distance at least `tol / 2`, velocity x within `0x1p-40` of 1.
- Move, cylinder rim: from `{0.5 + tol/2, 0, 3}` at `{0,0,-1}`, `T 4`, reacting: `kHit`, `kHighRim`, the hit
  position's distance within `0x1p-40` of `tol` (moved along the rim normal), velocity
  `{0,0,-1} + (2 / Sqrt(5)) * n` with `n = {1,0,2} / Sqrt(5)`, that is `{0.4,0,-0.2}`, within `1e-8`; `hit_count` 1.
- Move, a particle resting on top of a sphere stays: sphere at the origin radius 1, particle at `{0,0,1 + tol}`,
  each of 1000 steps of `T 1e-3` adds gravity `{0,0,-9.81 T}` to both velocities and feeds the end back: x and y
  stay exactly 0, both velocities exactly `{0,0,0}` after every step, z within `2 tol` of `1 + tol`.
- Move, a particle slides off a frictionless sphere at the analytic angle: same sphere, start `{0,0,1 + tol}` with
  velocity `{0.01,0,0}`, gravity added by the test each step (`v.z -= 9.81 T`, then `MoveParticle`, end fed back),
  `T = 1e-3`, run until the end distance exceeds `0.01`: no error in any step; the polar angle of the last step with
  `kind == kHit` is within `0.01` rad of `acos(2/3 + 0.01^2 / (3 * 9.81))` = 0.84106 rad. The design's model of the
  scheme gives 0.8369 (error 0.004 rad, from removing the inward speed at each hit and from the last contacts being
  intermittent); the error falls with `T`. Contact alone shows this: the test applies gravity itself, as
  `SlidesTenThousandSteps` does; no later module is needed. (The scene version through the step is not required.)
- Move, a cylinder turning about its own axis moves a resting particle only through friction: cylinder axis z,
  radius 1, half height 0.5, spinning `{0,0,2}` rad/s; particle at `{0.5,0,0.5 + tol}` with both velocities
  `{0,0,-1}` (pressed onto the cap), `T 0.1`: friction 0 gives velocity `{0,0,0}` and end `{0.5,0,0.5 + tol}`;
  friction 0.25 gives velocity `{0,0.25,0}` (slip: the cap moves `{0,1,0}` there) and end `{0.5,0.025,0.5 + tol}`;
  friction 1 gives `{0,1,0}` (stick) and end `{0.5,0.1,0.5 + tol}` (within `2 tol`). No turning-solid close call.

Analytic, tolerance (the tests that replace the enclosures):

- `MoveParticle, ObliqueMovingSolidFarFromOrigin_NeverEndsInside`: 10,000 cases from a fixed integer formula:
  flat solids with oblique normals, moving, coordinates up to `1e3` m, particles hitting and sliding. For each,
  `ComputeDistanceToSolid` at the end pose is at least `tol / 2`. In the same test the naive end
  `hit_position + velocity.for_this_step * time_left`, built from the public helpers with no rest gap and no settling,
  must be inside the solid (`d < 0`) in at least one case. That control is what proves the test fails without
  the tolerance.
- `MoveParticle, SlidesTenThousandSteps_NeverStartsInside`: one particle pressed onto an oblique moving solid,
  each end fed back as the next start with `start_overlap_tolerance = 0`: no error, `d` stays in
  `[tol / 2, 2 * tol]`.
- `MoveParticle, FastParticleThinPlate_IsStopped`: plate of thickness `cell_width / 100`, travel of 1000 cell
  widths in one step: hit reported.
- `ComputeContactTolerance, SceneMovedAMillionMeters_ScalesWithIt`; `CheckContactTolerance,
  PlateThinnerThan1024Tolerances_IsSolidTooThin`.
- `MoveParticle, SceneTurnedRigidly_ResultTurnsWithIt`: every case above under a fixed oblique rotation and
  under the Y-up to Z-up quarter turn (report 10.5), relative `0x1p-40`.

Conservation: `reaction.velocity_for_next_step_change` equals after minus before; for two particles of equal mass
hitting a reacting box symmetrically the summed change has zero moment.

Reference and proofs: none. Every expected value is a closed form, so a Python reference would repeat the test;
the only proofs touching contact were about the removed bounds.

Comparison with `main`: about twenty rows of `liquid_wall_contact_3d_cases.h` (half space and slab, fixed and
moving, with friction) are run once on `main`, turned into Z-up and stored as expected values. Positions agree
within `2 * tol` (the new end rests `tol` off the surface), velocities within `0x1p-40` relative. Supporting
evidence only; tank rows are not compared because the method changed.

GPU (label `gpu`, compiled here, run by the owner): the sweep-and-move cases and the 10,000-case set through
`MoveParticles` with `RunOn::kGpu`, against `RunOn::kCpu` within section 6; `MoveParticles,
SameInputTwice_SameBytes` over positions, velocities, records and reactions; `MoveParticles,
TwoFailures_ReportsLowestParticleId`. Container without a GPU: all shared-header tests run there too, built
inside the CUDA configuration (label `no_gpu`), which proves the shared headers compile for the device and give
the CPU results in that build; `BuildContactBuffers` with `kGpu` returns `kRunOnCallFailed` there (amended A58:
`BuildRunMemory(kGpu, ...)` already returns it, so the builder is not reached).

Memory (NEW, A58; label `cpu`): `BuildContactBuffers, IntoRunMemory_HandsOutExactlyTheEstimate` (block of
`ComputeRunMemoryBytes(BuildContactMemoryEstimate(1000, 4), {}, kRunOn)`: `handed_out_bytes() == reserved_bytes()`,
exact); `BuildContactBuffers, RunMemoryOnePieceShort_IsOutOfMemory` (block 256 B short, release builds:
`kOutOfMemory`, magnitude = bytes the failing piece asked).

## 8. Method paper section

`= Contact with solids <sec-contact>`, with `== Solid shapes <sec-solid-shapes>`, `== The contact tolerance
<sec-contact-tolerance>`, `== Finding the first hit <sec-first-hit>`, `== Response and slide
<sec-contact-response>`. Equations `<eq:solid-distance>`, `<eq:contact-tolerance>`, `<eq:hit-time>`,
`<eq:contact-response>`.

Figures: `<fig-solid-shapes>` the four shapes with their own axes, `normal_out_of_solid`, the two plate sides and
the Blender names of the domain sides; `<fig-sweep-hit-slide>` side view of one step against a moving flat solid:
sweep, hit, response, slide, with the two velocities (`for_next_step`, `for_this_step`) drawn separately;
`<fig-contact-response>` one velocity split into approach and sideways parts, the slip and stick cases;
`<fig-contact-tolerance>` a zoom on the surface showing the rounding band, the rest gap, the end check and a
feature 1024 tolerances thick.

Pseudocode: `<proc-move-particle>` (the six steps of 5.5) and `<proc-find-first-hit>` (the three of 5.3).

NEW (A74), sphere and cylinder: subsection `== Round solids: sphere and cylinder <sec-round-solids>` (amended
2026-11-15: after `<sec-first-hit>` and before `<sec-contact-response>`, since it uses the tolerance, the gap, the
touching rule and the stepping search; the subsections are then solid shapes, tolerance, first hit, round solids,
response and slide), equations `<eq:sphere-distance>` (`d = |x - c| - r`), `<eq:cylinder-distance>` (the
three-region formula of 5.7, with `d_wall` and `d_cap`), `<eq:sphere-hit-time>` (the quadratic and its cancellation-free root),
`<eq:cylinder-hit-time>` (wall root restricted in height, cap time restricted to the disc); figures
`<fig-round-solids>` (sphere with centre, radius and outward normal; cylinder with its own z axis, radius, half
height and the wall, cap and rim regions shaded in a side cut, with the degenerate-normal points marked) and
`<fig-round-first-hit>` (side cut of the grown shapes: the sphere grown by `tol`, the cylinder grown with sharp
edges, a path through the wall, one through the cap and one entering the rim corner, with the rim move to
`d = tol`); pseudocode `<proc-sphere-first-hit>` and `<proc-cylinder-first-hit>`; worked examples (amended
2026-11-15, the paper's own: the sphere of radius 0.5 grazed from `(-2, 0.3, 0)` at `(2, 0, 0)`, hit at
`0.8 - 0.625 tol` with normal `(-0.8, 0.6, 0)`; the cylinder rim hit `2 - tol` with the rim move (radius 0.5, half
height 1, from `(0.5 + tol/2, 0, 3)`); for radius 0.5 and half height 0.25 the cap hit, the wall hit and the miss over
the rim; the distances of `(0.8, 0, 0.65)` and `(0.1, 0, 0.2)`; the touching starts on a sphere and the spinning
cylinder's response); the curved-surface touching rule with its convexity argument, the tangent slide, and the spin
rule, stated in plain words. The paper writes the rotation `Rot(...)` and the cylinder's own axis `e_z`.

Sources from the old reference list: [19] Stomakhin 2013 (the friction response on a collision surface), [3]
Hu 2018 (moving solids coupled to particles and the two sides of a thin solid), [21] Batty 2007 (why the
reaction uses the same geometry as the motion; one sentence pointing to the solids section). The stepping
search toward a turning solid has no entry in the old list: listed as "citation needed", not written from
memory.

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| `ResolveLiquidWallContact3D`, `boundary_internal::ResolveContact` | `MoveParticle` | DECIDED |
| `LiquidWallContactInputs3D`, `ContactInput` | `SolidContactInputs` | DECIDED |
| `LiquidWallContactResult3D`, `ContactOutput` | `SolidContactResult` | DECIDED |
| `LiquidWallContactDiagnostic3D`, `int diagnostic` | `ContactKind` | DECIDED |
| `start_world_position`, `world_position` | `start_position`, `position` | DECIDED |
| `stored_world_velocity`, `advecting_world_velocity` | `velocity.for_next_step`, `velocity.for_this_step` | DECIDED (A1) |
| `friction_coefficient`, `penetration_tolerance` | `friction`, `start_overlap_tolerance` | DECIDED |
| `contact_fraction`, `outward_normal` | `hit_fraction`, `normal_out_of_solid` | DECIDED |
| `incoming_*_relative_normal_speed` | `approach_speed` | DECIDED |
| `wall_velocity` | `solid_velocity`, `SolidMotion::linear_velocity` | DECIDED / NEW |
| `LiquidObstacle3D`, `Obstacle`, `ObstacleKind` | `Solid`, `SolidShape` | DECIDED / NEW |
| `TranslatingHalfspace3D`, `start_plane_coordinate` | `FlatSolid`, `point_on_surface` | DECIDED / NEW |
| `TranslatingSlab3D`, `half_width` | `Plate`, `half_thickness` | DECIDED / NEW |
| `TranslatingContactBox3D`, `body_half_extent` | `Box`, `half_size` | DECIDED / NEW |
| `AxisAlignedLiquidTank3D`, `floor_y`, piston | `DomainBox`, `low_corner.z`, a separate driven `FlatSolid` | DECIDED |
| `RigidBodyStep3D drift`, `RigidBodyPose3D` | `SolidMotion`, `SolidPose` | NEW |
| `body_axis` + `outward_sign` | `SolidFace` | NEW |
| `ElasticBladeSide3D {blade_id, sign}`, `side_tag` | `PlateSide {plate_id, side}`, `SolidSide` | DECIDED name; VERIFIED, see below |
| `ClassifyInitialElasticBladeSide3D` | `FindSolidSide` | NEW |
| `DistanceAlongPath`, `SolidDistance` | `ComputeDistanceToSolid`, `DistanceToSolid` | NEW |
| `ApplyOneWayResponse` | `ComputeContactResponse` | NEW |
| `SelectBoxImpactFace`, `SweepRigidBoxPoint3D`, `RigidBoxSweepHit3D` | `FindFirstHit`, `FirstHit` | NEW |
| `body_impulse`, `body_angular_impulse`, `piston_stored_velocity_change` | `SolidReaction`, `velocity_for_next_step_change` | NEW; field word from A1 |
| `boundary_internal::Error`, `absl::StatusCode` | `ContactError`, `ContactCloseCall`, `ContactFailure` | DECIDED / NEW (A2) |
| empty status messages | `ToMessage`, `ToName` (stable code) | DECIDED (A13) / NEW |
| `CudaResolveLiquidWallContact3D`, particle-update call sites | `MoveParticles` | DECIDED verb, NEW operation |
| `kUnitNormalTolerance`, `kRelativeResolution`, "certified" | `contact_tolerance` and the constants of 3.3 | NEW |

Words marked Verify or Open that this module met:

- `side_tag` (O2). VERIFIED by reading `base/liquid_boundary_transfer_types_3d.h:5,10,21`,
  `liquid_wall_contact_3d.cc:675-679`, `elastic_blade_deformation_3d.cc:50,336` and every consumer found by
  search: production code writes only `-1`, `0`, `+1` (the sign of one plate); consumers test equality only;
  tests write arbitrary integers (`-42`, `23`), so the transfer contract is "any integer, equal means same
  side". The value is an identity. `PlateSide` keeps that (plate id plus side, 8 bytes, the size of the old
  field) and `SolidSide` is the geometric answer for one plate.
- `structurally_eligible` (first drafted `can_hold_liquid`, Verify). In contact's files its one meaning is "the
  face centre is outside the plate's thickness" (`liquid_wall_contact_3d.cc:674-680`). That fits the final name
  `transfer::SolidsAtFaces::outside_solid` (transfer's decision 6).
- `wall`: in these files always the surface of a solid or a tank side, never clock time.
- "scratch" (step-owned contact records): replaced by "working buffers" (owner answer A31).

## 10. Not ported

- `ProjectionBounds`, `ProjectedSum`, `AddWithResidual`, `AddDown`, `AddUp`, `magnitude_is_resolved`,
  `HasRepresentablePlanes`, `NudgeOutward`, `CertifyContactSide`, `UncertifiedContact`
  (`liquid_boundary_math.h`, `liquid_wall_contact_math_3d.h`): replaced by 5.2.
- `CompareBoxSweepTimes`, `SelectBoxImpactFace` (`liquid_wall_contact_3d.cc:24-131`) and the bisection of
  `SweepRigidBoxPoint3D` with its curvature analysis of a touching start: replaced by 5.3.
- `SweepTankAxis`, `ResolveTankContact`, `CheckTankStep`, the piston inside the tank, the zero-friction rule.
- `ResolveBoxInTank`, `TankCorrectedPathClearsBox`, `ResolveRotatingBoxInTankContact3D`: the list of solids
  and the hit loop replace each pairing.
- `Maybe`, `Value`, `PackContact`, `UnpackContact`, `ArithmeticStatus`, the `bounded` flag and its empty
  messages, `ContactInput`/`ContactOutput` as mirrors.
- `CudaResolveLiquidWallContact3D`, `CudaOperatorMeasurement`, the one-particle kernel.
- The scene check that calls contact with an invented particle (`liquid_substep_3d_staging.cc:476-486`):
  `CheckSolid` does it.
- Tests of the rounding machinery (`RejectsUnrepresentableTrajectoryAndWallPose`,
  `CoMovingRoundedEndpoint...`); `reference/.../liquid_boundary_3d_fixtures.py` sweep rows.
- Left for the solids design, not contact: `BuildElasticBladeFaceTags3D`, `ResolveElasticBladeContact3D`'s work and
  angular impulse, `distinct_body_contact_3d.h`, `contact_face_bodies_3d.h`, `mixed_contact_3d.cc`.
- Left for the step design: the "more than half a cell width" travel limit (`liquid_substep_3d_cpu.cc:377-384`).

## 11. Decisions

The questions were merged into `owner-questions.md` (numbers in brackets) and are now decided.

**Q1. `SolidSide` values and a separate stable side identity (O2).** Reading the code settles the facts (section
9). Options: (a) `enum class SolidSide { kNoPlate, kAgainstNormal, kAlongNormal }` for the geometry plus
`struct PlateSide { std::uint32_t plate_id; SolidSide side; }` for identity, compared with `==`; (b) keep one
`std::int64_t side_id` with free values; (c) the three-value enum alone, which allows one plate per scene.
Recommendation: (a). It still allows one plate per particle, as on `main`.

Decision: (a) (owner-questions Q3). Recommended option (owner's standing instruction, 2026-11-14).

**Q2. Where `PerParticleVelocity` lives, and the word `state` (O3).** Contact is its first user, before
particles and transfer exist. Options for the home: `base/per_particle_velocity.h` (recommended, added by this
module) or `particles/`. Options for the field: `state` (decided), `carried`, `kept`:
`velocity.carried` / `velocity.travel`. Recommendation: `carried`, because `state` already means two other
things.

Decision: the fields are `for_next_step` and `for_this_step` (owner answer A1, owner-questions Q1); the home is
`base/per_particle_velocity.h` (owner-questions Q13, recommended option, owner's standing instruction,
2026-11-14).

**Q3. A device buffer type is missing from Foundation, and who owns the batch operation.** `MoveParticles`
cannot keep data on the GPU without a type such as `Buffer<T>` (section 4; the reconciliation chose (a)).
Options: (a) add `base/buffer.h` to Foundation before contact starts (recommended; every later module needs it); (b) contact defines it; (c)
contact ships only the shared `MoveParticle` and a private test kernel, and the batch operation waits for the
particles module. Also confirm that `contact::MoveParticles` is the body of step stage 13.

Decision: (a), `base/buffer.h` lands in Foundation before contact starts; `contact::MoveParticles` is the body of
the step stage `MoveParticles` (stage 16 in the step design). Owner-questions Q45, recommended option (owner's
standing instruction, 2026-11-14).

**Q4. Friction per hit.** Options: (a) `friction` per particle from its material times `friction_scale` per
solid (default 1; 0 for a smooth solid such as the old platen), as designed; (b) a table per material and
solid; (c) per solid only. Recommendation: (a): it keeps the decided `friction` field and reproduces every
case on `main`.

Decision: (a) (owner-questions Q35). Recommended option (owner's standing instruction, 2026-11-14).

**Q5. Tolerance values and the rest gap (part of O1).** Proposed: `0x1p-40` of the largest coordinate plus
`0x1p-30` of the smallest cell width; a particle in contact rests one tolerance outside the surface instead of
on it; `kSolidTooThin` below 1024 tolerances; CPU versus GPU as in section 6. The rest gap is a visible change
(about `1e-11` m in a metre-sized scene). Alternative: rest on the surface and accept `d` down to `-tol`
everywhere, which pushes a tolerance into every other module that asks "is this point inside a solid".
Recommendation: the rest gap.

Decision: the values as proposed and the rest gap (owner-questions Q36). Recommended option (owner's standing
instruction, 2026-11-14).

**Q6. One motion for every shape, and the plate's extent.** Proposed: any shape may translate and turn (at most
a quarter turn per step); a plate is unbounded sideways as on `main`, and a finite thin solid is a `Box`, which
has no plate sides. Alternative: give `Plate` a half size so a finite blade keeps its two sides; that changes
`FindSolidSide` near the rim and module 7's cutting. Recommendation: keep the unbounded plate now and decide
the finite one in the solids design.

Decision: every shape translates and turns through `SolidMotion`; the plate stays unbounded, one per scene, as
the solids design keeps it (owner-questions Q37). Recommended option (owner's standing instruction,
2026-11-14).

**Q7. New names to confirm.** `half_thickness`, `half_size`, `SolidMotion`, `SolidPose`, `SolidFace`
(`kLowX`...`kHighZ` in the solid's own axes), `ClosedSides`, `DistanceToSolid`, `FirstHit`, `ParticleContact`,
`SolidReaction`, `reports_reaction`, `friction_scale`, `ContactBuffers`, `ContactFailure` (an enum plus context
as the `std::expected` error; other modules would follow the pattern), `ContactCloseCall`, `kSolidTooThin`,
`kOnWrongSideOfPlate`, `kRunOnCallFailed`, and in `math::`: `Length`, `Normalized`, `IsFinite`, `Rotation`.

Decision: all confirmed as listed (owner-questions Q13, Q14: recommended option, owner's standing instruction,
2026-11-14). The error pattern `ContactFailure { ContactError error; ... }` through
`std::expected<T, ContactFailure>` is owner answer A2; short `math::` nouns are owner answer A11; `ToMessage`
is owner answer A13; `ToName` was added from `changes-to-apply.md`.

**Q8. Contact between two materials.** `distinct_body_contact_3d.h` and `mixed_contact_3d.cc` (liquid against
snow on shared faces) are left to module 7 as `CollideSolids`. Confirm they are not contact's.

Decision: they are not contact's; `solids::CollideSolids` owns them, applied between material sets (owner
answer A8 for the sets; owner-questions Q46, recommended option, owner's standing instruction, 2026-11-14).

**Q9. Working memory (owner answer A31, 2026-11-14).** The word "scratch" is not used; reusable working memory is
"buffers". Decision: the step's `Buffer<ParticleContact>` and `Buffer<SolidReaction>` are called working buffers;
contact holds no `OrderBuffers` or `SumBuffers`, so no declaration changes. Source: `owner-answers.md` A31.

**Q10. One reserved memory block per run (owner answer A58).** Decision: `BuildContactBuffers` takes `RunMemory&
run_memory` last and builds its four buffers as pieces of it; `BuildContactMemoryEstimate` has one row per buffer
(`"contact: call result"` split, `"contact: first failure"` added), so `ComputeRunMemoryBytes` of the rows is what
the build hands out (test `BuildContactBuffers, IntoRunMemory_HandsOutExactlyTheEstimate`). Source: owner answer
A58; `plan-decisions.md` "Owner answers A40–A64 applied", item 3.

**Q11. Out of memory has its own error (reconcile item 31).** Decision: `ContactError::kOutOfMemory` appended as the
ninth and last value, returned only by `BuildContactBuffers` for `BufferError::kOutOfMemory` and
`kRunMemoryTooSmall`, `magnitude` = bytes asked. The earlier mapping of `kOutOfMemory` to `kBuffersTooSmall`
(3.5) is **replaced**; `kBuffersTooSmall` keeps `kSizeMismatch` and the count checks of `MoveParticles`. Source:
`plan-decisions.md` "Owner answers A40–A64 applied", item 11 (reconcile item 31).

**Q12. Other answers of the A40–A64 pass.** A64 (particles everywhere), the shared cell size rule and items 4 to 10
change nothing in contact: it reads no cell size and moves gas markers and every particle alike. Source:
`plan-decisions.md` "Owner answers A40–A64 applied".

**Q13. The solids copy for sand (reconcile item 33).** Decision: `ContactBuffers::solids_copy()` (NEW) exposes the
copy `MoveParticles` writes each call, so sand's GPU hit split needs no per-call allocation (sand design decision
8). Test `ContactBuffers, SolidsCopy_IsTheLastCallsSolids`. Source: `plan-decisions.md` "Owner answers A40–A64
applied", item 11.

**Q14. Particle size follows the cell (A65).** Every liquid cell holds about eight particles of its own size: a
particle's `size_level` is its cell's level, its rest volume `(w/2)³`. Decision: no change; contact moves points and
reads no particle size or radius (its tolerance is `0x1p-30` of the smallest cell width for every particle). Source:
`plan-decisions.md` "Owner answers A65–A68 applied", A65.

**Q15. Start cells (A66).** Decision: no change; seeded points inside a solid are dropped by I/O
(`io::BuildStartParticles` keeps a point outside every solid, as before), using contact's `ComputeDistanceToSolid`.
Source: `plan-decisions.md` A66.

**Q16. Drifted particles (A67).** Decision: a split child that lands inside a solid is handled by the next
`MoveParticles` like any particle that starts inside: kind `kStartedInsideSolid`, counted in
`MoveParticlesResult::started_inside_count` (3.5); no new contact rule or name. Source: `plan-decisions.md` A67.

**Q17. Later work (A68).** Decision: no change; contact works per particle and per call, whatever the batch shape.
Source: `plan-decisions.md` A68.

**Q18. Sphere and cylinder (A74).** The owner added both shapes to contact now, beside boxes, plates and the domain
box: distance, normal, surface velocity and first hit, each with tests; they unlock the round-body checks
(settling sphere, torque between rotating cylinders) and give round colliders. Decision: `Sphere`, `Cylinder`,
`ToSolid` for each, `SolidShape::kSphere`, `kCylinder` and `SolidFace::kCurved`, `kLowRim`, `kHighRim` appended
(earlier values keep their numbers); true signed distances (5.7) with the three-region cylinder formula and the
degenerate-normal rule (sphere centre and cylinder axis: the solid's own -x axis, as a box tie takes `kLowX`); closed-form first hits (sphere
quadratic with the approach condition; cylinder grown with sharp edges, wall root restricted in height, cap
restricted to the disc, the rim hit moved to `d = tol`); the touching rule on a curved surface is P2's rule with the
same tolerance, justified by convexity; the slide follows the tangent and re-sweeps; a spin about the symmetry axis
does not count as a turn of the shape (nor toward `kMaxTurnPerStep`), a cylinder turning across its axis uses the
stepping search. A plate stays the only shape with sides (P3). Source: owner answer A74; `plan-decisions.md`
"Sphere and cylinder in contact (A74)"; contact plan decisions P13 to P17.
