# Rebuild Contact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `native/fluidity/contact/`: the solid shapes, the contact tolerance, the swept first hit, the
friction response, the slide, `MoveParticle` (shared by CPU and GPU) and the batch operation `MoveParticles`
on both processors, with the method-paper section that explains them. **Amended 2026-11-15 (A74):** the shapes
include the sphere and the finite cylinder (distance, normal, surface velocity, first hit, slide), beside the flat
solid, plate, box and domain box.

**Architecture:** Shared headers (`contact_error.h`, `solid_shapes.h`, `contact_tolerance.h`, `solid_contact_types.h`,
`solid_contact_steps.h`, `solid_contact.h`; see the header layout below) hold every formula, inline and `FLUIDITY_CPU_AND_GPU`, so the CPU loop and the GPU thread run
the same `MoveParticle`. One CPU-only header, `move_particles.h`, declares the batch operation on plain pointers;
`move_particles_cpu.cc` holds the public function and the CPU loop, `move_particles_gpu.cu` one thread per particle
and an integer reduction of the failures. Contact includes only `base/`. **Amended 2026-11-15 (A74):** the sphere
and cylinder are two more `SolidShape` values handled by the same shared functions (a branch per shape in
`ComputeDistanceToSolid` and in the closed-form first hit); no new header, no new public operation.

**Header layout (as built through Task 8):** `contact/solid_contact_types.h` holds the design 3.4 types, the constants
and `ComputeContactResponse` / `ComputeLargestApproachLeftAfterResponse`; `contact/solid_contact_steps.h` holds the
private per-step functions and includes the types header; `contact/solid_contact.h` includes both and holds the public
functions (`FindFirstHit` now, `MoveParticle` later). No forward declarations. New private steps go to
`solid_contact_steps.h`, new public functions to `solid_contact.h`, new types and constants to
`solid_contact_types.h`.

**Tech Stack:** as Foundation: C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2 (GoogleTest), uv,
clang-format and clang-tidy 23.1 (the versions `pyproject.toml` pins), Docker via colima, Typst with CeTZ for the method paper. Nothing is added.

**Spec:** `docs/superpowers/specs/modules/01-contact-design.md` (the contract; "the design" below) and
`docs/superpowers/specs/2026-11-13-production-rebuild-design.md` (the programme design), both in the `main`
checkout `/Users/tychota/Documents/Code/openSource/FluidiTy`, untracked. Owner decisions that override them:
`docs/superpowers/specs/modules/owner-answers.md`. Cross-module names: `.../modules/shared-interface.md`.

All paths below are relative to the worktree `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`
(branch `rebuild`) unless they start with `main:` (read with `git show main:<path>` from the worktree).

## Global Constraints

Copied from Foundation:

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- **Amended 2026-11-15 (Task 8a review): closed-form first hits.** A closed-form first hit is an intersection of
  entry/exit intervals or a single root checked only against the duration; never a candidate accepted or rejected
  by a second, separately rounded test. Each closed form has a generated test aimed at its edges, compared with a
  reference that cannot miss.
- Old code is read with `git show main:<path>` from the worktree. Nothing is checked out wholesale.
- Commit rule (spec section 6): one thing per commit; strict CPU build and all tests pass; CUDA target compiles
  in the container; clang-format 23.1 clean; clang-tidy 23.1 clean on host sources; every touched file passes the
  `AGENTS.md` checklist; new public names listed in the commit message with a one-line reason; tests arrive with
  the code. A commit must pass every part of the gate that exists at that commit.
- **Amended 2026-11-14 (Foundation as built):** Foundation's structure patterns apply to every task (`AGENTS.md`
  "Structure"; `specs/modules/plan-decisions.md` "Foundation after the final review"). Two halves: the public
  function lives in `<name>_cpu.cc` (checks, `RunOn` dispatch, the CPU loop, the GPU half called under
  `#if defined(FLUIDITY_HAS_GPU)`); the GPU half `<name>_gpu.cu` is declared in the private `<name>_gpu.h`, returns
  `RunOnCallStatus`, which the CPU half maps with `ToBufferError` and then into `ContactFailure` through one mapping
  function, and launches, waits and holds temporary GPU memory only through `base/gpu_launch.cuh`
  (`ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`, `GpuArray`). Every GPU function is a thin loop over
  `FLUIDITY_CPU_AND_GPU` per-item functions in a shared header, and `cpu` tests run those functions for every
  thread block and thread, in launch order and in reverse, against the CPU loop. Working buffers (A31): an operation's reused
  arrays are buffers built once and held by the module's buffers (contact needs none). Errors:
  a failed check writes nothing; after `kRunOnCallFailed` the outputs are unspecified and the run stops;
  `kOutOfMemory` comes only from building forms. Exactness: discrete outputs equal on CPU and GPU, floating-point
  results within the module's tolerance unless a header promises the same bits unless NaN. Tests link
  `fluidity_base_test_support` (`ToBits`, `BuildBufferHolding`, `ReadAll`, `CheckWithinTolerance`,
  `CheckSameBitsOrBothNaN`) and any module helper library made with `fluidity_add_test_library` (named in
  `LINKS`); every `gpu` test passes its main outputs to `WriteResultDigest`. Labels: `cpu` everywhere, `no_gpu` only
  for behaviour on a machine without a GPU, `gpu` only for tests that need hardware. Every target is created through
  the helpers of `cmake/fluidity_targets.cmake` and every source file is listed in a CMake file.
- **Amended 2026-11-15 (A58; reconcile item 31):** one reserved block per run (Foundation Tasks 17 to 19,
  `base/run_memory.h`; `plan-decisions.md` "Owner answers A40–A64 applied", items 3 and 11). `BuildContactBuffers`
  takes `RunMemory& run_memory` as its last parameter (every other parameter stays); `run_on ==
  run_memory.run_on()` is a `Requires:` checked by a debug assertion (A30). Its buffers (all `where = kRunOn`) are
  pieces built with `BuildBuffer(run_on, size, run_memory)`; plain `BuildBuffer` stays for tests and tools. Each
  estimate entry stands for whole buffers (`copies` = the number of buffers, each `count x bytes_each` bytes), so
  `ComputeRunMemoryBytes(BuildContactMemoryEstimate(...))` equals the bytes the build hands out.
  `ContactError::kOutOfMemory` is appended as the last value (earlier values keep their numbers) and is returned only
  by `BuildContactBuffers`, for `BufferError::kOutOfMemory` and `kRunMemoryTooSmall`, with `magnitude` = the bytes
  asked. Every test that builds `ContactBuffers` first builds a `RunMemory` of exactly
  `ComputeRunMemoryBytes(BuildContactMemoryEstimate(...), {}, kRunOn)` on the same processor. A `no_gpu` test that asked
  for GPU buffers without a GPU now gets `kRunOnCallFailed` from `BuildRunMemory(kGpu, ...)` and then checks that
  `BuildContactBuffers` was not reached (plan decision: no GPU builder can be called without a GPU block).
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the CPU build in the debug and the
  release presets, and the CUDA build compiled in the container (its `cuda` step fails, naming the file, for a
  tracked `.cc` or `.cu` file that no build compiles). clang-format and clang-tidy are the pinned 23.1.
- History on `rebuild` may be rewritten. A defect found later is fixed in the commit that introduced it.
- Commit messages carry no AI trailer (no `Co-Authored-By`, no `Assisted-by`, no session link); the owner is the
  author. After committing, write the commit's note with `uv run python tools/add_session_note.py --session
  <session link> --models <implementer's model id>,claude-fable-5-2`
  (a git note in `refs/notes/claude-session`; owner answer A55, 2026-11-15).
- Device code is "compiled, not run" until `docs/gpu-runs.md` records the owner's run. The owner runs on
  hardware once, after all modules are written. Never claim otherwise.
- Host C++23, exceptions and RTTI off in the library; device and shared headers C++20 subset; a shared header
  never includes `<expected>`, `<string>` or anything that allocates. No Abseil.
- Device numerical flags: `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. Host: `-ffp-contract=off`,
  `/fp:precise`. No fast-math.
- Z-up, right-handed. Gravity points along `{0, 0, -1}`.
- Naming: no `3D` suffix, no `Cuda`/`Gpu`/`Device`/`host` in public names, no bare `bool` parameters, no
  abbreviations, no milestone codes. Function verbs: `Compute`, `Build`, `Check`, `Find`, `Apply`, `Solve`,
  `Read`, `Write`, the stage verbs (`Move`), and `To<Type>`; anything else is listed for the owner's review.
- 120 columns, two-space indent, `//` comments, braces on every control-flow body.
- Tests are named `TEST(FunctionOrType, Situation_Outcome)`, one behaviour each, in `*_test.cc` beside the code.
- Deletion is the default: add a dependency, file or tool only when a task here names it.

Contact's own:

- Folder `native/fluidity/contact/`, namespace `fluidity::contact`. Contact includes headers from `base/` only.
  Design section 3 is the contract: names, fields, defaults and signatures are copied from it exactly.
- Every function of the shared headers is `inline FLUIDITY_CPU_AND_GPU` (and `constexpr` where the body allows),
  defined in the header so device code sees it without relocatable device code.
- Constants (design 3.2, 3.3, 3.4, 5.2): `kContactToleranceRelative = 0x1p-40`, `kContactToleranceFloorInCellWidths =
  0x1p-30`, `kMinFeatureInTolerances = 1024`, `kMaxHitsPerStep = 4`, `kMaxAdvanceCount = 128`, `kMaxTurnPerStep =
  1.5707963267948966` (declared in `solid_shapes.h` since Task 3), `kNoSolid = 0xFFFFFFFFu`. A unit vector or rotation column is accepted when its squared
  length is within `0x1p-40` of 1.
- Order: solids are visited in array order (ascending `solid_id`), faces in `kLowX, kHighX, kLowY, kHighY, kLowZ,
  kHighZ` (**amended 2026-11-15, A74:** then `kCurved, kLowRim, kHighRim`; a round solid's face comes from its region,
  P14). Equal hit times keep the lower solid index, then the first face. Contact makes no floating-point sum
  across particles; the GPU reduction is an integer minimum and integer counts.
- On failure `MoveParticle` leaves `*result` unchanged; `MoveParticles` leaves the trial outputs unspecified.
- Contact has no material-specific code. Every material (liquid, snow, springy solid, gas markers with friction
  0) goes through the same call; the step makes one call per material and range of cell blocks.
- A value from `main` enters through `(x, y, z) -> (x, -z, y)` (old floor `-y` becomes `side_bottom`).
- Test tolerances (design 7): positions `2 * contact_tolerance` absolute; velocities and fractions
  `|a - b| <= 0x1p-44 * max(|a|, |b|, reference)` with `reference` the largest input speed of the test (1 for
  fractions). CPU versus GPU: design section 6. No test in this module is known-open.

Plan decisions (the design left these open; every task respects them, the paper states them):

- **P1 Test tolerance.** Analytic tests pass `contact_tolerance = kTestContactTolerance = 0x1p-30` m (declared in
  the test-only header `contact/contact_test_cases.h`) unless the test is about the tolerance itself. Expected
  values include the rest gap: the docstring example ends at `x = 1 - tol` and its `hit_fraction` is
  `(0.1 - tol) / 0.2`, compared within the relative tolerance above.
- **P2 Touching face.** In `FindFirstHit`, a face whose gap `g = d - tol` at the path start is in `[-tol, 0]` and
  is the solid's largest face gap is the face the particle rests on. It is not hit when `rate * duration >= -tol`
  (the path cannot enter a convex solid, a flat solid or a domain side without crossing that plane); otherwise it
  is hit at `t = 0`. A start with `g < -tol` is hit at `t = 0` (defensive; `MoveParticle` never sweeps from there).
  **Amended 2026-11-15 (paper review):** this is the paper's "touching rule". It belongs to the closed forms of a
  solid that does not turn; a turning solid (P7, P17) has no touching rule. A particle resting on a turning solid at
  `d = tol` starts with `g(0) = 0 <= tol / 4` and is hit at `t = 0` in every step; the response and the slide in the
  solid's axes (P7) carry it with the solid for the rest of the step. This is intended: the face it rests on turns,
  so the plane argument above does not hold. Consequence, also intended: such a particle runs P7's
  `kTurningSolidNearOtherSolid` test over the whole step, so a turning solid (a paddle) close to another solid (the
  domain floor) can ask for shorter steps (Task 13 test).
  **Amended 2026-11-15 (Task 10 as built, review ruling 18 and re-review I3): a touched solid has a stricter rule.**
  `MoveParticle` keeps a touched-surface list, one entry per hit: the solid index, the normal recorded at the hit and
  the allowance `ComputeLargestApproachLeftAfterResponse` computed from the velocity passed to that response. Every
  re-sweep passes the list to the private `FindFirstHitOfSweep`; the public `FindFirstHit` passes an empty list and
  behaves as before. For a listed solid that does not turn, a start gap in `[-tol, tol / 16]`
  (`kTouchedSolidRestingGapShare = 2^-4`) counts as resting, and the solid is hit at `t = 0` only when the approach
  rate is below minus the largest allowance of ALL entries so far (not its own entry's: the next sweep's velocity
  comes from the latest response, wherever it was); otherwise the whole sweep is free of it. An unlisted solid keeps
  P2 as it was. A listed solid whose start gap is above `tol / 16` takes its closed form. A turning solid has no
  touching rule. The touched flag is per solid: a domain box touched on one side is strict on every side. Why: at a
  hit, rounding can leave the gap a hair positive and the response's leftover approach a hair negative, so the solid
  just hit was hit again (found in Task 10's review). Proof, in one sentence: each entry asserts `allowance * time
  left <= tol / 16` at its hit and the time left only shrinks, so a free path from a touched solid comes closer to it
  by at most `tol / 16`. The allowance is the one proved for the velocity the response received (carried, not
  recomputed from the velocity after it).
- **P3 Committed plate.** For a particle committed to `{plate_id, s}`, that plate is the half space on side `s`:
  `d = s * p.z - half_thickness`, face `kHighZ` for `kAlongNormal`, `kLowZ` for `kAgainstNormal`. This distance
  is used in classification, `FindFirstHit`, settling and the end check. "On the other side by more than `tol`"
  (`kOnWrongSideOfPlate`) means `s * p.z < -(half_thickness + tol)`; a start inside the thickness is pushed to the
  committed side. A `plate_side` whose `plate_id` names no plate in `solids` is `kBadInput`. **Amended 2026-11-15
  (paper review):** for a committed plate, `kOnWrongSideOfPlate` is tested before `kStartsInsideSolid` (a particle
  on the other side is not "inside" its half space; the specific diagnosis wins); solids are still visited in index
  order and the first failing solid decides.
- **P4 Start hits.** Step 2 visits solids in index order; each qualifying solid gets the response, joins the
  surface list and counts as one hit. The first one fills the record. `kind` is `kStartedInsideSolid` when any
  start distance was negative. Each solid is classified from the start as moved by the earlier solids (P12).
- **P5 Reaction.** Every change of `velocity.for_next_step` made against a solid with `reports_reaction` is added to
  `reaction.velocity_for_next_step_change`: the response at a hit (step 2 or step 4) and **each removal during the
  slide** (step 5). A removal of the approach to an earlier surface is charged to that surface's solid. **Amended
  2026-11-15 (paper review):** the corner case of step 5 (the newest surface approached again, both velocities set
  to the newest solid's velocity) is charged to the **newest** solid. `hit_position` is the position of the first
  such change. A second, different reacting solid is `kTooCloseToCall/kSecondReactingSolid`. Tests: Task 10
  (response), Task 12 (a slide removal, the corner case).
  **Amended 2026-11-15 (Task 10 as built, review minor M3; TO CONFIRM WITH THE OWNER):** a change of exactly zero
  (compared exactly) at a solid neither claims that solid's reaction record nor conflicts with it, so a second
  reacting solid whose change is zero is not `kSecondReactingSolid`, and a solid's `hit_count` (module 8) counts only
  particles whose `for_next_step` changed at it. Tests: `ReactingSolidWithZeroChange_LeavesReactionEmpty`,
  `SecondReactingSolidWithZeroChange_DoesNotConflict`. The plan has no open-points list; this note and the design
  (3.4, 5.5) carry the mark.
- **P6 Failure context.** `MoveParticle` fills `ContactFailure::solid_id` (the solid's `solid_id`, never its
  index) and `magnitude` (`-d` for `kStartsInsideSolid`, `-(s * p.z + half_thickness)` for `kOnWrongSideOfPlate`,
  `d` for `kEndsTooCloseToSolid`, else 0); `particle_id` stays 0 and `MoveParticles` fills it.
  **Amended 2026-11-15 (Task 10 as built, review ruling 19):** on `kTooCloseToCall`, `FindFirstHit` (and the private
  `FindFirstHitOfSweep`) writes `*first_hit = FirstHit{}` with `solid_index` naming the turning solid that could not
  be searched, so `MoveParticle` can fill `solid_id`; `kBadInput` leaves `*first_hit` unchanged.
- **P7 Turning slide.** A solid turns when `angular_velocity != {0, 0, 0}` exactly (amended 2026-11-15: a flat solid
  or a plate when the part across its normal is not exactly zero, P17). After a hit on a turning
  solid the rest of the step is carried in its axes and ends the hit loop. `kTurningSolidNearOtherSolid` when, for
  any other solid `j`, `d_j - tol <= time_left * (Length(v_local) + Length(linear_velocity) + Length(angular_velocity)
  * r + Length(linear_velocity_j) + Length(angular_velocity_j) * r_j)` at the hit, `r` and `r_j` the distances of
  the hit position to each solid's origin.
- **P8 Scale.** `ComputeContactScale`: `largest_coordinate` is the largest `|component|` of every solid origin at
  step start and at step end, plus `Length(half_size)` for a box or domain box; `thinnest_solid` is the smallest
  of `2 * half_thickness` (plate), `2 * min(half_size)` (box, domain box), or `smallest_cell_width` when there is
  none. The caller passes particles' extent through the domain box; a scene with no domain box relies on solids.
  **Amended 2026-11-15 (A74):** a sphere and a cylinder also add `Length(half_size)` to `largest_coordinate`; they
  give `thinnest_solid` candidates `2 * radius` (sphere) and `min(2 * radius, 2 * half_height)` (cylinder).
  Confirmed for the paper: a domain box's origin is its centre and its `half_size` is half its extent (Task 3 test
  `ToSolid, DomainBox_OriginIsCentreAndHalfSizeIsHalfExtent`), and its smallest width `2 * min(half_size)` counts
  as a `thinnest_solid` candidate.
- **P9 Unchecked by `MoveParticle`.** `MoveParticle` does not call `CheckSolid` (the step calls it once per step).
  `FindFirstHit` does not validate its inputs; it accepts `contact_tolerance >= 0` (tests use 0 for the naive
  control). A domain box with no closed side is `kBadInput` in `CheckSolid`.
- **P10 Sample.** `ComputeSolidVelocitySample` evaluates `solid_velocity` at `point - distance * normal`; when
  `has_solid` is false every other field keeps its default; a plate gives `{solid_id, FindSolidSide(...)}` and
  `{0, kNoPlate}` when the point is inside the thickness.
- **P11 Generated cases.** Every generated test set uses one integer generator, typed in
  `contact_test_cases.h`: `state` starts at `0x2545F4914F6CDD1D`; `CaseGenerator::Next()` sets `state = state *
  6364136223846793005 + 1442695040888963407` (wrapping `std::uint64_t`) and returns `(state >> 11) * 0x1p-53`,
  in `[0, 1)`; `NextSigned()` is `2 * Next() - 1`. Draw order is part of each set's definition.
- **P12 Start inside the tolerance (coordinator, 2026-11-14).** A start with
  `-start_overlap_tolerance <= d < tol` is moved along that solid's normal to `d = tol` before the sweep, whatever
  its velocity. With `d < 0` this is `kStartedInsideSolid` with the response (design 5.5 step 2). With
  `0 <= d < tol` and a velocity approaching the solid it is `kHit` with the response. With `0 <= d < tol` and no
  approaching velocity the move is silent: velocities, `kind`, `hit_count` and the record are unchanged, and the
  solid does not join the surface list. So a particle placed near a solid (a source, a resize) never fails the
  end check for that reason; `kEndsTooCloseToSolid` comes only from solids that close on the particle during the
  step faster than it can leave.

Plan decisions added 2026-11-15 for the sphere and the cylinder (A74; design 5.7 and Q18; the paper states them):

- **P13 Degenerate normals.** Sphere: when `Dot(p, p) == 0` as computed (the point is the centre), the normal is
  minus the pose's x column (the normal of `kLowX`, the first face in face order: the box's tie rule, as the paper
  says). Cylinder: in the wall branch, when `p.x * p.x + p.y * p.y == 0` as computed (the point is on the axis),
  `radial` is minus the pose's x column. Both are tested with exact equality, never with a threshold, so CPU
  and GPU take the same branch.
- **P14 Cylinder regions (the rim formula).** In the solid's axes, `rho = Sqrt(p.x * p.x + p.y * p.y)`,
  `wall_distance = rho - radius` (paper `d_wall`: the signed distance to the wall's surface, taken as unbounded in
  height), `cap_distance = Abs(p.z) - half_height` (paper `d_cap`: the signed distance to the plane of the nearer
  cap), `cap_sign = p.z >= 0 ? +1 : -1` (paper `sigma_z`). The distance is the one expression
  `d = Sqrt(max(wall_distance, 0)^2 + max(cap_distance, 0)^2) + min(max(wall_distance, cap_distance), 0)` (the
  paper's `<eq:cylinder-distance>`, the same bits in every region); the branches give the normal and face, with
  `radial = (p.x, p.y, 0) / rho` (paper `e_rho`) and `e_z` the cylinder's own z axis. Branch order: rim when
  `wall_distance > 0 && cap_distance > 0` (`d = Sqrt(wall_distance^2 + cap_distance^2)`, normal
  `(wall_distance * radial + cap_distance * cap_sign * e_z) / d`, face `kHighRim` or `kLowRim` by `cap_sign`); else
  wall when `wall_distance >= cap_distance` (`d = wall_distance`, normal `radial`, `kCurved`); else cap
  (`d = cap_distance`, normal `cap_sign * e_z`, `kHighZ` or `kLowZ`). So an inside tie takes the wall, and a point
  exactly on a region boundary gets the same `d` from either branch.
- **P15 Cylinder first hit.** Grown with sharp edges (`R = radius + tol`, `H = half_height + tol`). **Amended
  2026-11-15 (Task 8a as built):** the first hit is an intersection of two intervals, not two candidates. With
  `wall_gap = (rho0 - radius) - tol` and `cap_gap = (Abs(p0.z) - half_height) - tol` (the distance's own
  expressions minus `tol`; where this plan wrote `rho0 > R` or `Abs(p0.z) > H` it now means `wall_gap > 0` or
  `cap_gap > 0`): the times the path spends inside the grown wall (unbounded along the axis) and between the grown
  caps' planes are each one interval. Wall: with `A = u.x^2 + u.y^2`, `B = p0.x u.x + p0.y u.y`,
  `C = wall_gap * (rho0 + radius + tol)`, when `wall_gap > 0` it is empty unless `A != 0`, `B < 0` and
  `B^2 - A C >= 0`, otherwise from `C / (-B + Sqrt(B^2 - A C))` to `(-B + Sqrt(B^2 - A C)) / A`; when
  `wall_gap <= 0` it runs from 0 to the larger root (`(-B + Sqrt(...)) / A` for `B <= 0`, `-C / (B + Sqrt(...))`
  for `B > 0`, for ever when `A == 0`). Caps: with `s0 = sign(p0.z)`, when `cap_gap > 0` it is empty unless
  `-s0 u.z > 0`, otherwise from `cap_gap / (-s0 u.z)` to `(Abs(p0.z) + half_height + tol) / (-s0 u.z)`; when
  `cap_gap <= 0` it runs from 0 to `-(sign(u.z) p0.z - half_height - tol) / Abs(u.z)`, for ever when `u.z == 0`.
  The hit is the later of the two entries when neither interval is empty and that entry is no later than either
  exit and no later than `duration` (both limits count). The reported face, normal and distance are
  `ComputeDistanceToSolid` at the hit position, so the face comes from P14, not from an interval. A start in the
  rim corner (`wall_gap <= 0` and `cap_gap <= 0` with a positive distance gap) is a hit at 0 only when
  `rate < 0`, free otherwise. Why intervals: a candidate validated by a second, separately rounded test let paths
  through the grown rim edge pass unseen (found in Task 8a's review). A hit position that **this closed form** finds in
  the rim region is moved by `MoveParticle` along its normal to `d = tol` before the response (step 4; the move is
  at most `(Sqrt(2) - 1) * tol`). **Amended 2026-11-15 (paper review):** the rim move never applies to a hit found
  by the stepping search toward a turning cylinder (P17), even when the reported face is `kLowRim` or `kHighRim`:
  that search stops at `d <= 1.25 tol` in any region and its hit is used as found (Task 12a test).
- **P16 Touching rule on a curved surface (counterpart of P2).** For a sphere or cylinder, `g0 = d - tol` with the
  true distance at the path start and `rate = Dot(normal at the start, u)`, `u` the velocity relative to the solid.
  `-tol <= g0 <= 0`: not hit in this sweep when `rate * duration >= -tol`, hit at `t = 0` otherwise; `g0 < -tol`:
  hit at `t = 0` (defensive). Same tolerance as P2. Safe because the true signed distance of a convex solid is
  convex along a line: `d(t) >= d(0) + rate * t`, and `MoveParticle` starts every sweep at `d = tol` (a hit, the rim
  move or the start move put it there), so the window below `d = 0` only absorbs rounding. Like P2, it belongs to
  the closed forms only: a cylinder turning across its axis (P17) has no touching rule, and a particle resting on it
  is hit at `t = 0` in every step and carried by the slide in its axes. In the sphere and wall quadratics `C` is
  computed as `g0 * (L0 + R)` (and `(rho0 - radius - tol) * (rho0 + R)`), never as `Dot(p0, p0) - R * R`, so the
  sign of `C` is exactly the sign of the gap this rule reads.
- **P17 Spin about a symmetry axis.** The shape-changing turn is zero for a sphere and `omega_across = omega -
  Dot(omega, e_z) * e_z` for a cylinder (`e_z` its own z axis, the pose's z column at the sweep's start). A sphere is always swept and
  slid as a solid that does not turn (closed form, Task 8a). A cylinder is too when `Length(omega_across) *
  step_duration * Length(half_size) <= contact_tolerance / 8` (the step's duration, not the sweep's; it turns only
  when `Abs(omega_across) * step_duration * Length(half_size) > tol / 8`, and below that bound it uses the closed
  form); otherwise it is a turning solid (Task 9 stepping,
  Task 13 slide, P7). `CheckSolid`'s `kSolidTurnsTooFar` uses `Length(omega_across) * step_duration` for a cylinder
  and never fires for a sphere. P7's "turns when `angular_velocity != 0`" stays for the other shapes.
  `ComputeSolidVelocity` always uses the full `omega`. **Amended 2026-11-15 (controller rulings 17 and 17 revised,
  Task 9b):** a flat solid and a plate spinning about their own normal do not turn either: both are unbounded in their
  own x and y (checked in every function that reads a plate: `ComputeDistanceToSolid`, `FindSolidSide`, the
  closed-form first hit with the `kLowZ`/`kHighZ` planes, the committed face of P3, `CheckSolid`'s sizes and
  `ComputeContactScale` read only `p.z` and `half_size.z`), so a spin about the normal leaves the shape unchanged.
  `ComputeShapeTurningAngularVelocity` gives both `omega - Dot(omega, e_z) * e_z`, `e_z` the normal (the pose's z
  column at the step's start), as for a cylinder, and each turns when that is not exactly `{0, 0, 0}` (P7's exact test,
  applied to the part across the normal; a spin given about a tilted normal may leave a part of a few rounding units,
  which makes it turning: safe, it then takes the stepping search). The closed form stays exact for such a spin: the
  surface velocity `omega x r` has no part along the normal. `CheckSolid`'s `kSolidTurnsTooFar` uses that part, so
  both may spin about their normal at any rate. Friction still sees the full spin through `ComputeSolidVelocity`.
  Task 9b makes the change and its tests.
  **Amended 2026-11-15 (Task 9b as built, review I1, replaces "a few rounding units ... makes it turning"):** a flat
  solid or a plate whose angular velocity has a part across its normal of length at most `kNegligibleTurnShare =
  0x1p-46` of `|omega|` does not count as turning (`ComputeShapeTurningAngularVelocity` returns `{0, 0, 0}`): a spin
  given about a tilted normal leaves rounding bits across, and those no longer send the solid to the stepping search.
  Its face then moves by at most `0.907 * 2^-43 * L` (`L` the largest coordinate), within `tol / 8`, since `tol >=
  2^-40 L`. A cylinder keeps its own bound (`Abs(omega_across) * step_duration * Length(half_size) <= tol / 8`).

Plan decision added 2026-11-15 for the second-order advance (controller ruling 16; Task 9b; the paper states it):

- **P18 Second-order advance toward a convex turning solid.** The stepping search (design 5.3 step 2) advances by the
  larger of two safe advances when the solid's distance is convex in its own axes, which holds for every turning
  shape of module 1: flat solid `p.z`, plate `|p.z| - half_thickness`, committed plate `s * p.z - half_thickness`
  (P3), box `max_k (|p_k| - half_size_k)`, cylinder the true signed distance (P14). With the sample at `t`, its gap
  `g = d - tol > tol / 4`, `n` its `normal_out_of_solid`, `v_rel = v - linear_velocity` and Task 9's `bound` and
  `r_max`:
  `gap_rate = Dot(n, v - ComputeSolidVelocity(solid, pose(t), x(t)))` (`gdot`, m/s),
  `gap_curvature_bound = Length(omega) * (2 * Length(v_rel) + Length(omega) * r_max)` (`K`, m/s^2),
  first-order advance `g / bound` (Task 9), second-order advance the positive root of `g + gdot * s - K s^2 / 2 = 0`:
  `2 * g / (Sqrt(gdot^2 + 2 * K * g) - gdot)` when `gdot <= 0`, `(gdot + Sqrt(gdot^2 + 2 * K * g)) / K` when
  `gdot > 0` and `K > 0`, infinite when `gdot > 0` and `K == 0`. Each form adds two terms of the same sign, so it has
  no cancellation; the sum under the root adds non-negative terms, so rounding cannot make it negative; the only zero
  denominator, `gdot == 0` with `K == 0`, is written as its own branch (an infinite advance: the path is free), as is
  `K == 0` with `gdot > 0`. The advance taken is the larger of the two. Why it is safe: (1) the reported normal,
  written in own axes, is a subgradient of the convex distance (a unit vector), so `d(p(s)) >= d(p(t)) +
  n_own . (p(s) - p(t))` for every `s`; this holds at an edge or corner, where the reported face is one active plane
  of the max, at the box's `p_k = 0` tie and the plate's middle plane (both `-1` and `+1` lie in the subdifferential
  of `|.|` at 0), and at a cylinder's rim (the true distance, differentiable outside); (2) with `A' = [omega]x A`,
  `o' = linear_velocity` and `omega` constant in the step, `p'' = A^T (-2 omega x v_rel + omega x (omega x (x - o)))`,
  so `|p''| <= K` while `|x - o| <= r_max`, with no small-angle step, so up to `kMaxTurnPerStep` and beyond; a moving
  origin enters through `v_rel` and the moving origins of `r_max`; (3) `n_own . p'(t) = gdot`. So `g(s) >= g + gdot
  (s - t) - K (s - t)^2 / 2`, the root is safe, and the larger of two safe advances is safe. The search advances only
  from `d > 1.25 tol > 0`, outside the solid, and the inequality holds along the whole line, so nothing inside the
  tolerance band or inside the solid is needed. A committed plate is linear in the whole space, so its "wrong side"
  breaks nothing: a particle there has `d < 0` and is hit at 0 before any advance; an uncommitted plate is convex on
  both sides. `r_max` must bound `|x(s) - o(s)|` over every time the advance can reach: Task 9's `r_max` (both path
  ends, origins at the matching times) does, because the length of the straight line `x - o` is convex in time; an
  underestimate of either end puts particles inside (measured: `K` from the start alone, 3.8 cm inside a quarter-turn
  arm; from the end alone, 1.8 mm), the lesson of Task 9's M7 and M10. Rounding margin: none beyond Task 9's. The
  target is still `g = 0` (`d = tol`); the distance is off by at most `tol / 128`, `gdot` by about `2^-50 (|v| +
  |omega| r_max)`, which over one advance moves the model by below `2^-40` of a cell for a step that moves the
  particle or the surface at most 1000 cells, and the root and `K` carry relative rounding only: the path stays
  farther than `tol - tol / 64` before the hit (the tests' allowance). The hit band and the reported hit are Task 9's:
  a hit when `g <= kTurningSolidHitGapShare * tol`, `at_hit` as found, so after an advance `d` is in `[tol - tol / 64,
  1.25 tol + tol / 128]` (a second-order advance usually lands near `d = tol`); P15's "no rim move for a
  stepping-search hit" and Task 12a are unchanged. `kMaxAdvanceCount` stays 128 and a close call stays possible (a
  particle hovering with no normal motion where `K` is far above the path's true curvature), but must be rare: at most
  1 per 1000 paths on every generated set of Tasks 9 and 9b and at most 1 % per kind on the paddle set (measured: 0
  in 30,000 mirror paths of seven generated sets, 1 in 8,000 broad paths, 0 to 0.1 % in the paddle set with gaps down to the hit band),
  pinned by the ceilings of `ExpectTurningSweepCounts` and `GeneratedPaddleSweeps_AgreeWithFacePlaneReference`; the
  remaining close call is pinned by `TurningBoxHoverOnSpinAxisBeyondAdvanceLimit_IsCloseCall`. A shape whose
  distance is not convex (the domain box's smallest wall gap; a 1b mesh, the smallest triangle distance) uses the
  first-order advance alone, chosen by the private `HasConvexDistance(SolidShape)`: measured on a test-only turning
  room, the second-order advance misses walls on 27.5 % of paths, ending up to 0.5 m outside. Also (review m4):
  `FindFirstHit` searches the turning solids after the others, each with `duration` capped at the earliest hit found
  so far, so a turning solid's close call after an earlier hit is not reached.

## Before you start

Foundation (`docs/superpowers/plans/2026-11-13-rebuild-foundation.md` in the `main` checkout) is complete on
`rebuild`: Tasks 1 to 16 and 11, including `base/vec3.h`, `mat3.h`, `axis.h`, `scalar_math.h` (`math::Length`,
`Normalized`, `IsFinite`, `Sqrt`, `Abs`, `Rotation`), `cpu_and_gpu.h`, `cpu_only.h`, `run_on.h`,
`per_particle_velocity.h`, `buffer.h`, `memory_count.h`; `cmake/fluidity_targets.cmake` with
`fluidity_add_test`; the method paper with `figure_style.typ`, `pseudocode.typ` and `references.typ`;
`docs/gpu-runs.md`. No other module exists. **Amended 2026-11-14 (Foundation as built):** Foundation as built also
provides `base/run_on_call.h` (`RunOnCallStatus`, `ComputeLaunchSize`), `base/gpu_launch.cuh`, the sums with their
`SumBuffers` and the stable order and grouping with their `OrderBuffers` (one form each, A31), `base/test_support.h` (library
`fluidity_base_test_support`) and `fluidity_add_test_library`; read `AGENTS.md` "Structure" before the first task.
Prove it from the worktree:

```bash
uv run python tools/check_commit.py
```

Expected: seven `PASS` lines (format, rules, cpu debug, cpu release, tidy, paper, cuda). If any step fails, stop and report; do not start contact on a red gate.

## Review Focus

1. **A particle resting on a face is swept again after every hit.** If the touching-face rule (P2) is wrong,
   each re-sweep hits the same face at `t = 0` and a plain slide ends in `kTooManyHits`. Pinned in Task 8,
   `FindFirstHit, SlidingOnTouchedFace_IsNotAHit`.
2. **Index versus id.** `SolidContactResult::solid_index` and `SolidReaction::solid_index` are array indexes;
   `ContactFailure::solid_id` is the stable id. Pinned in Task 11,
   `MoveParticle, StartsInsideSecondSolid_FailureNamesItsId` (ids 7 and 9).
3. **Committed plate side with a start inside the thickness.** The nearest face is on the wrong side; the particle
   must go to its committed side (P3). Pinned in Task 11,
   `MoveParticle, CommittedSideInsideThickness_PushedToCommittedSide`.
4. **Friction 0 at a fast solid.** A response computed in the solid's frame erases a small sideways velocity.
   Pinned in Task 7, `ComputeContactResponse, FrictionZeroFastSolid_KeepsSidewaysBitForBit`.
5. **The lowest failing particle on the GPU.** Two failures in different thread blocks, the lower id at the higher
   index. Pinned in Task 16, `MoveParticles, TwoFailures_ReportsLowestParticleId`.
6. **Amended 2026-11-15 (A74): a curved surface after a hit, and the cylinder's rim.** A particle that hit a sphere
   re-sweeps along the tangent from `d = tol`; a quadratic whose `C` is rounded to the wrong sign, or a touching
   rule missing for curved shapes, makes it hit the same sphere at `t = 0` again (a slide ends in `kTooManyHits`),
   and a path tangent within rounding must give either answer without error. At the rim the sweep's sharp corner
   leaves the hit up to `Sqrt(2) * tol` away and must be moved to `tol`. Pinned in Task 12a,
   `MoveParticle, SphereOblique_SlidesOffAlongTheTangent` (`hit_count` 1), `MoveParticle,
   SphereTangentWithinRounding_NeverEndsInside` and `MoveParticle, CylinderRim_HitPositionMovedToOneTolerance`.

---

### Task 1: Method paper section "Contact with solids"

**Kind:** mechanical

Load the `typesetting-with-typst` and `drawing-with-cetz` skills first. Figures use `figure_style.typ`,
procedures use `pseudocode.typ`.

**Files:**
- Modify: `docs/method-paper/fluidity-method.typ` (new section and notation rows)
- Create: `docs/method-paper/figures/contact_solid_shapes.typ`, `contact_sweep_hit_slide.typ`,
  `contact_response.typ`, `contact_tolerance.typ`

**Interfaces:**
- Produces labels (design 8): `<sec-contact>`, `<sec-solid-shapes>`, `<sec-contact-tolerance>`, `<sec-first-hit>`,
  `<sec-contact-response>`; `<eq:solid-distance>`, `<eq:contact-tolerance>`, `<eq:hit-time>`,
  `<eq:contact-response>`; `<fig-solid-shapes>`, `<fig-sweep-hit-slide>`, `<fig-contact-response>`,
  `<fig-contact-tolerance>`; `<proc-move-particle>`, `<proc-find-first-hit>`.

**Amended 2026-11-15 (A74):** the section also covers the sphere and the cylinder (design 5.7, 8). Added labels:
`<sec-round-solids>` (subsection after `<sec-first-hit>`, amended 2026-11-15: it needs the tolerance, the gap, the
touching rule and the stepping search first); `<eq:sphere-distance>` (`d = |x - c| - r`, normal
`(x - c) / |x - c|`), `<eq:cylinder-distance>` (the three regions of P14 with `d_wall = rho - r` and
`d_cap = |p_z| - h`: wall `d = d_wall`, cap `d = d_cap`, rim `d = Sqrt(d_wall^2 + d_cap^2)`),
`<eq:sphere-hit-time>` (`t = C / (-B + Sqrt(B^2 - A C))`, the approach condition `B < 0`), `<eq:cylinder-hit-time>`
(wall root restricted to `|z| <= h + tol`, cap time restricted to the disc by squared distances `<= (r + tol)^2`); `<fig-round-solids>`, `<fig-round-first-hit>`; `<proc-sphere-first-hit>`,
`<proc-cylinder-first-hit>`. Added file: `docs/method-paper/figures/contact_round_solids.typ` (both figures may live
in it). Content: sphere described by centre and radius; cylinder by centre, its own z axis (the `axis` field),
radius and half height; the wall, cap and rim regions with the branch order and the inside tie of P14; the
degenerate normals of P13; the two first-hit procedures (P15 for the cylinder: grown with sharp edges, the rim hit
moved to `d = tol`); the curved touching rule P16 with its one-line convexity argument; the tangent slide (it leaves a
convex surface and is not re-hit in the step, so `kMaxHitsPerStep` is not reached); the spin rule P17. One worked
example each (amended 2026-11-15: the paper's own sphere example replaced the head-on one below: radius 0.5, path
from `(-2, 0.3, 0)` at `(2, 0, 0)`, hit at `0.8 - 0.625 tol`): sphere radius 0.5 at the origin, path from
`(-2, 0, 0)` at `(1, 0, 0)`: hit at `1.5 - tol`; cylinder
radius 0.5, half height 1, path from `(0.5 + tol/2, 0, 3)` at `(0, 0, -1)`: hit at `2 - tol` in the high rim corner,
true distance `Sqrt(1.25) tol`, moved along the rim normal `(1, 0, 2) / Sqrt(5)` to `tol`. `<fig-round-solids>`: the
sphere with centre, radius and an outward normal; a side cut of the cylinder with its z axis, radius, half height
and the three regions shaded, the two degenerate-normal points marked. `<fig-round-first-hit>`: side cut of the
grown shapes (sphere grown by `tol`; cylinder grown with sharp edges, exaggerated), a path through the wall, one
through the cap, one entering the rim corner with the rim move drawn. State P13 to P17 in plain words where they
belong. If the paper was already written without these items (Task 1 in progress on 2026-11-15), add them in the
same commit before it lands.

**Amended 2026-11-15 (A74, after reading commit `920b063a`):** the paper's sphere and cylinder text agrees with the
design on: the names `radius`, `half_height`; the one cylinder expression (written after the review as
`Sqrt(max(d_wall,0)^2 + max(d_cap,0)^2) + min(max(d_wall,d_cap),0)`, code names `wall_distance`, `cap_distance`)
with a wall/cap tie going to the wall; the degenerate normal = the own -x axis (adopted as P13);
the stable sphere root; the cylinder sweep against a sharp-edged grown cylinder with no third (rim) equation; P2 on a
curved surface justified by the tangent plane at the nearest point, and the domain box read side by side. Its own
worked examples are kept (they replace the examples listed above where they differ). Fold these changes into that
commit (`git commit --fixup 920b063a`):
1. **Rim move (P15).** A hit in the rim corner of the sharp grown cylinder is at true distance `tol` to
   `Sqrt(2) tol`; its gap `d - tol` is then in `[0, 0.414 tol]`, which neither the touching rule (`[-tol, 0]`) nor
   the start-inside rule (`< -tol`) covers, so the re-sweep would start "inside the grown cylinder" with no rule.
   `MoveParticle` therefore moves a rim hit along the rim normal to `d = tol` (at most `(Sqrt(2) - 1) tol`). The end
   check (`d >= tol / 2`) never fires wrongly there: before the move `d >= tol`, after it `d = tol`. Add one sentence
   after "never closer than tol", and the start rule: a start inside the sharp grown cylinder with a positive true
   gap (rim corner only, up to rounding) is hit at `t = 0` when it approaches, free otherwise. The rim move applies
   only to hits of the cylinder's closed form, never to hits of the stepping search (P15).
2. **`gamma` from the gap (P16).** `gamma` is evaluated as `g0 * (|q| + R)` with `g0 = |q| - radius - tol` (equal to
   `q.q - R^2`), so its sign is exactly the sign of the gap the touching rule reads; one clause.
3. **Spin is not a turn (P17).** Replace "Every shape that turns, the sphere and the cylinder included, uses this
   search" by: a turn about a sphere's centre or a cylinder's own axis moves the surface but not the shape, so those
   are swept and slid as solids that do not turn (closed form), the spin entering only the surface velocity, and the
   quarter-turn limit does not apply to it; a cylinder whose axis itself turns uses the search. Reason: the turning
   slide carries the particle in the solid's axes, which would drag a frictionless particle round a spinning roller.
4. **Faces.** Name the faces once: sphere surface and cylinder wall `kCurved`, caps `kLowZ`/`kHighZ`, rims
   `kLowRim`/`kHighRim`.
5. **Start order (P3 amended):** for a committed plate, "on the wrong side" is checked before "too deep inside".

**Amended 2026-11-15 (paper review, round 2; folded as `ad1d88ae`):** the paper now writes the cylinder's gaps
`d_wall`, `d_cap` (P14), the rotation `Rot(...)` and the cylinder's own axis `e_z`; states P5 in full (slide removals
and the corner case count, P5), the touching rule's absence on a turning solid (P2, P16) and the rim move's limit to
closed-form hits (P15). Its five subsections are, in order, `<sec-solid-shapes>`, `<sec-contact-tolerance>`,
`<sec-first-hit>`, `<sec-round-solids>`, `<sec-contact-response>`.

- [ ] **Step 1: Text.** `= Contact with solids <sec-contact>` after the existing sections, with the five
  subsections of design 8 (four, plus `<sec-round-solids>` after `<sec-first-hit>`). Content: design 5.1 (shapes, solid-frame conventions of 3.2, `<eq:solid-distance>`),
  5.2 (`<eq:contact-tolerance>`, the three rules, the rest gap, why `4096 * epsilon`), 5.3 (`<eq:hit-time>`, the
  convex entry and exit times, the stepping search toward a turning solid), 5.4 (`<eq:contact-response>`, why the
  approach is removed in world coordinates, why the stick test is a division) and 5.5 (the slide). State the plan
  decisions P2, P3, P4, P5, P7 and P12 in plain words where they belong. Decided vocabulary only; `Coulomb friction`
  appears once in parentheses.
- [ ] **Step 2: Citations.** Cite by number from `references.typ`: [19] Stomakhin 2013 (friction response),
  [3] Hu 2018 (moving solids, two sides of a thin solid), [21] Batty 2007 (one sentence: the reaction uses the same
  geometry as the motion). The stepping search toward a turning solid carries the text "citation needed". Never
  write a DOI or URL; check each number against `references.typ`.
- [ ] **Step 3: Figures** (design 8): `<fig-solid-shapes>` the four shapes with their own axes,
  `normal_out_of_solid`, the two plate sides and the Blender names of the domain sides; `<fig-sweep-hit-slide>`
  side view of one step against a moving flat solid, sweep, hit, response and slide, `for_next_step` and
  `for_this_step` drawn separately; `<fig-contact-response>` one velocity split into approach and sideways parts,
  slip and stick; `<fig-contact-tolerance>` a zoom on the surface: rounding band, rest gap at `tol`, end check at
  `tol / 2`, a feature 1024 tolerances thick. Each caption states the one thing to see; each is referenced.
- [ ] **Step 4: Pseudocode.** `<proc-move-particle>` with the six steps of design 5.5 (1 check inputs, 2 classify
  the start, 3 find the first hit, 4 respond, 5 slide, 6 settle and check the end) and `<proc-find-first-hit>` with
  the three of 5.3 (1 solid that does not turn, 2 turning solid, 3 keep the earliest). Each lists inputs, outputs and
  "fails when". Code comments `// Step N:` in Tasks 8 to 13 follow these numbers.
- [ ] **Step 5: Notation rows** for the symbols this section uses: `T` step duration, `tol` contact tolerance,
  `d` distance to a solid, `g` gap (`d - tol`), `mu` friction.
- [ ] **Step 6: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`.
  Expected: exit 0, no warning. Read every new page; fix overlaps, clipped labels and inconsistent sizes.
  Run the gate.
- [ ] **Step 7: Commit.** Subject: `Explain contact with solids in the method paper`.

### Task 2: Contact folder, errors and their names

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** `fluidity_add_shared_test` retired; tests use `fluidity_add_test(... LABEL cpu)`, which the container's `cuda` step also runs.

**Amended 2026-11-15 (reconcile item 31):** `ContactError` gains `kOutOfMemory` as its ninth and last value.

**Files:**
- Create: `native/fluidity/contact/contact_error.h`, `contact_error_message.h`, `contact_error_message.cc`,
  `contact_error_test.cc`
- Modify: `CMakeLists.txt` (contact sources join the `fluidity` target as `base/` sources do; the folder's own
  `native/fluidity/contact/CMakeLists.txt` lists headers with `fluidity_add_headers`, sources with
  `target_sources(fluidity PRIVATE ...)` and tests with `fluidity_add_test`), `docs/glossary.md`

**Interfaces:**
- Consumes: `FLUIDITY_CPU_AND_GPU`, `base/cpu_only.h`.
- Produces: design 3.1 exactly: `ContactError` (9 values, `kOutOfMemory` last), `ContactCloseCall` (8 values), `ContactFailure`,
  `ToMessage(ContactError)`, `ToMessage(ContactCloseCall)`, `ToName(ContactError)`, `ToName(ContactCloseCall)`.
  CMake: shared-header tests are `fluidity_add_test(<name> SOURCES <files> LABEL cpu)`; the container's `cuda` step
  runs every `cpu` test, so they run in the CUDA build too (design 7). No new CMake helper.

- [ ] **Step 1: Failing tests** (`contact_error_test.cc`, `fluidity_add_test(contact_error_test ... LABEL cpu)`):

  ```cpp
  TEST(ToName, EveryContactError_IsSnakeCaseOfEnumerator)   // kSolidTooThin -> "solid_too_thin"; kNone -> "none";
                                                            // kOutOfMemory -> "out_of_memory"; all nine unique;
                                                            // static_assert(ContactError::kOutOfMemory == 8): last
  TEST(ToName, EveryContactCloseCall_IsSnakeCaseOfEnumerator)  // kSqueezedBetweenSolids -> "squeezed_between_solids"
  TEST(ToMessage, EveryContactError_HasADistinctNonEmptySentence)
  TEST(ToMessage, EveryContactCloseCall_HasADistinctNonEmptySentence)
  TEST(ContactFailure, Default_IsNoneWithZeroContext)       // error kNone, close_call kNone, ids 0, magnitude 0
  ```
  The snake-case tests compare against a table typed in the test (enumerator spelling), not against a formula.
- [ ] **Step 2: Run.** Expected: compile error, headers missing.
- [ ] **Step 3: Implement.** `contact_error.h` is shared (no `<string_view>`); `contact_error_message.h` is CPU
  only and includes `base/cpu_only.h` first. Messages say what the user can do
  ("A solid is thinner, or a grid cell narrower, than 1024 contact tolerances. Make the solid thicker or the cells larger, or move the scene closer to the origin.").
  Glossary rows: `close call` (a step contact cannot answer safely; the step is shortened).
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Add contact errors and their stable names`. New names: the 3.1 list.

### Task 3: Solid shapes, `ToSolid` and `CheckSolid`

**Kind:** numerics

**Files:**
- Create: `native/fluidity/contact/solid_shapes.h`, `solid_shapes_test.cc`, `contact_test_cases.h` (test-only)
- Modify: `CMakeLists.txt`, `docs/glossary.md`

**Interfaces:**
- Consumes: `Vec3`, `Mat3`, `kIdentity`, `math::Cross`, `math::Dot`, `math::Transpose`, `math::Determinant`,
  `math::Length`, `math::Normalized`, `math::IsFinite`; `ContactError`, `ContactCloseCall` (Task 2).
- Produces: from design 3.2: `SolidShape`, `SolidFace`, `SolidSide`, `PlateSide`, `FlatSolid`, `Plate`, `Box`,
  `ClosedSides`, `DomainBox`, `SolidMotion`, `SolidPose`, `Solid`, `kNoSolid`, the four `ToSolid`, `CheckSolid`,
  `DomainSide`, `ToSolidFace`. Test-only: `kTestContactTolerance`, `ExpectVec3Near(actual, expected, tolerance)`.
  `CheckSolid` returns `ContactError` only; the close call is written through an overload decided here:
  `ContactError CheckSolid(const Solid& solid, double step_duration, ContactCloseCall* close_call)` (may be null)
  beside the design's two-argument form, which calls it with null (plan decision: the step needs the reason).
- **Amended 2026-11-15 (A74):** also produces `Sphere`, `Cylinder`, `ToSolid(const Sphere&, ...)`,
  `ToSolid(const Cylinder&, ...)` (six `ToSolid` in all), `SolidShape::kSphere`, `kCylinder`, `SolidFace::kCurved`,
  `kLowRim`, `kHighRim` (appended; `static_assert` that `kDomainBox == 3`, `kHighZ == 5`). `CheckSolid` adds design
  3.2's rules for the two shapes and P17's turn limit.

- [ ] **Step 1: Failing tests.** Amended 2026-11-15 (A74): the second block below joins the first.

  ```cpp
  // A74:
  TEST(ToSolid, Sphere_HalfSizeIsRadiusAndAxesAreIdentity)   // centre {1,2,3}, r 0.5: origin {1,2,3}, half_size
                                                             // {0.5,0.5,0.5}, axes kIdentity exactly, kSphere
  TEST(ToSolid, Cylinder_ZAxisIsAxisAndHalfSizeIsRadiusRadiusHalfHeight)  // axis {1,0,0}, r 0.5, h 1: columns
                                                             // x {0,0,-1}, y {0,1,0}, z {1,0,0} exactly;
                                                             // half_size {0.5,0.5,1}, kCylinder
  TEST(CheckSolid, ZeroRadiusSphere_IsBadInput)              // and r -1, r NaN
  TEST(CheckSolid, SphereUnequalHalfSizes_IsBadInput)        // half_size {0.5,0.5,0.6} written by hand
  TEST(CheckSolid, CylinderNegativeHalfHeight_IsBadInput)    // and half_size.x != half_size.y
  TEST(CheckSolid, CylinderTumblingTwoRadiansInOneStep_IsSolidTurnsTooFar)  // axis z, angular {2,0,0}, step 1
  TEST(CheckSolid, SpinningFastAboutOwnAxis_IsNone)          // sphere and cylinder (axis z), angular {0,0,100},
                                                             // step 0.1 (10 rad, P17)
  ```

  ```cpp
  TEST(ToSolid, FlatSolid_ZAxisIsNormalAndAxesAreARotation)  // normal Normalized({1,2,3}): column z == normal,
                                                             // Determinant 1 within 1e-15, x along
                                                             // Cross({1,0,0}, normal) normalised (x is smallest)
  TEST(ToSolid, Plate_HalfSizeZIsHalfThickness)              // half_thickness 0.25 -> half_size {0,0,0.25}
  TEST(ToSolid, DomainBox_OriginIsCentreAndHalfSizeIsHalfExtent)  // {0,0,0}..{2,4,6} -> {1,2,3}, {1,2,3}
  TEST(ToSolidFace, EveryDomainSide_MatchesSolidFaceOrder)   // static_assert all six
  TEST(PlateSide, SamePlateAndSide_AreEqual)                 // and {1,kAlongNormal} != {1,kAgainstNormal}
  TEST(CheckSolid, ValidBox_IsNone)
  TEST(CheckSolid, NonFiniteOrigin_IsBadInput)
  TEST(CheckSolid, AxesNotARotation_IsBadInput)              // a reflection (swap two columns) and a scaled column
  TEST(CheckSolid, ZeroHalfSizeBox_IsBadInput)
  TEST(CheckSolid, IdZero_IsBadInput)
  TEST(CheckSolid, MovingDomainBox_IsBadInput)
  TEST(CheckSolid, DomainBoxWithNoClosedSide_IsBadInput)     // P9
  TEST(CheckSolid, NegativeFrictionScale_IsBadInput)
  TEST(CheckSolid, TwoRadiansInOneStep_IsSolidTurnsTooFar)   // |angular| 2 rad/s, step 1 s: kTooCloseToCall and
                                                             // close_call kSolidTurnsTooFar
  TEST(CheckSolid, QuarterTurnExactly_IsNone)                // |angular| * step == kMaxTurnPerStep
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement** the declarations of design 3.2 exactly, with the solid-frame conventions in the header
  (design 3.2, last paragraph). `ToSolid(FlatSolid)` and `ToSolid(Plate)`: z column = normal; x column =
  `Normalized(Cross(UnitVector(axis with smallest |normal part|, first in x, y, z order on ties), normal))`;
  y column = `Cross(z, x)`. `static_assert(sizeof(PlateSide) == 8)`. Rotation check: each column's squared length
  within `0x1p-40` of 1, each pair's `|Dot| <= 0x1p-40`, determinant positive.
  Glossary rows: `solid`, `flat solid`, `plate`, `box`, `domain box`, `plate side`, `solid face`, `pose`,
  `motion`; never say: `obstacle`, `halfspace`, `slab`, `blade`, `tank`, `side_tag`.
  Amended 2026-11-15 (A74): `ToSolid(Cylinder)` builds the axes from `axis` by the same rule as the plate (z column =
  axis); `ToSolid(Sphere)` sets `kIdentity`. `CheckSolid`: sphere needs `half_size.x == half_size.y == half_size.z
  > 0`; cylinder `half_size.x == half_size.y > 0` and `half_size.z > 0` (exact equality: `ToSolid` writes them so);
  turn limit per P17. Glossary rows: `sphere`, `cylinder`, `wall` (of a cylinder), `cap`, `rim`, `spin` (a turn about
  a symmetry axis, which moves the surface but not the shape).
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Describe solids by shape, pose and motion`. New names: the 3.2 names above
  (with `Sphere`, `Cylinder`, `kSphere`, `kCylinder`, `kCurved`, `kLowRim`, `kHighRim`).

### Task 4: Pose, surface velocity and distance to a solid

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_shapes.h`, `solid_shapes_test.cc`

**Interfaces:**
- Consumes: Task 3; `math::Rotation`.
- Produces: `ComputeSolidPose`, `ComputeSolidVelocity`, `DistanceToSolid`, `ComputeDistanceToSolid` (design 3.2,
  formulas design 5.1). Port note: there is no single source on `main`; the pose follows
  `main:native/src/rigid_body_3d.cc:833-859` (`SampleRigidBodyPose3D`) without its status checks.

- [ ] **Step 1: Failing tests** (exact unless stated):

  ```cpp
  TEST(ComputeSolidPose, QuarterTurnAboutZ_SendsSolidXToWorldY)   // angular {0,0,pi/2}, elapsed 1: column x
                                                                  // {0,1,0} within 1e-15
  TEST(ComputeSolidPose, Translating_MovesOriginOnly)             // origin {1,2,3} + {0.5,0,0} * 2 == {2,2,3}
  TEST(ComputeSolidVelocity, Turning_IsLinearPlusAngularCrossArm) // linear {1,0,0}, angular {0,0,2}, origin
                                                                  // {0,0,0}, point {0,3,0}: {-5,0,0}
  TEST(ComputeDistanceToSolid, FlatSolidAboveAndBelow_IsSignedHeight)  // tilted normal; +0.5 and -0.25, face kHighZ
  TEST(ComputeDistanceToSolid, PlateEitherSide_IsGapToNearerFace)      // h 0.1: p.z 0.3 -> 0.2 kHighZ;
                                                                       // p.z -0.3 -> 0.2 kLowZ; p.z 0 -> -0.1 kHighZ
  TEST(ComputeDistanceToSolid, BoxOutsideOneFace_IsGapToThatFace)      // half {1,1,1}, p {3,0,0}: 2, kHighX, {1,0,0}
  TEST(ComputeDistanceToSolid, BoxInside_IsMinusDepthOfNearestFace)    // p {0.9,0,0.5}: -0.1, kHighX
  TEST(ComputeDistanceToSolid, BoxCornerTie_TakesFirstFaceInOrder)     // p {-2,-2,0}: 1, kLowX
  TEST(ComputeDistanceToSolid, TurnedBox_UsesSolidAxes)                // box turned pi/2 about z, p {0,3,0}: 2, kHighX
  TEST(ComputeDistanceToSolid, DomainBoxInside_IsGapToNearestClosedSide)  // {0,0,0}..{1,1,1}, p {0.5,0.5,0.1}:
                                                                          // 0.1, kLowZ, normal {0,0,1}
  TEST(ComputeDistanceToSolid, DomainBoxOpenTop_IgnoresTop)               // p {0.5,0.5,0.95}: 0.5, kLowX (four
                                                                          // sides tie); 0.05, kHighZ if top counted
  TEST(ComputeDistanceToSolid, DomainBoxTie_TakesLeftBeforeFront)         // p {0.2,0.2,0.5}: kLowX
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement** design 5.1. `p = Transpose(axes) * (point - origin)`. Plate face: `kHighZ` when
  `p.z >= 0`; box face on an axis: `kLow` when `p.axis <= 0`, else `kHigh` (a box reports its low face on its middle plane; the
  face-order tie rule covers the box and the domain box; a plate and a cylinder's caps take `kHighZ` at `p.z = 0`;
  an inside tie in a cylinder goes to the wall). Domain box sides point into the box.
  Each formula carries the matching `<eq:solid-distance>` line in `Read:`.
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Compute a solid's pose, surface velocity and distance`. New names:
  `ComputeSolidPose`, `ComputeSolidVelocity`, `DistanceToSolid`, `ComputeDistanceToSolid`.

**Amended 2026-11-15 (A74):** Task 4 leaves the `kSphere` and `kCylinder` branches of `ComputeDistanceToSolid`
to Task 4a; no test builds those shapes before then, and the branch holds a debug assertion until Task 4a fills it.

### Task 4a: Distance and normal of a sphere and a cylinder (A74)

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_shapes.h`, `solid_shapes_test.cc`, `docs/glossary.md`

**Interfaces:**
- Consumes: Tasks 3 and 4; `math::Sqrt`, `math::Abs`.
- Produces: the `kSphere` and `kCylinder` branches of `ComputeDistanceToSolid` (design 5.7, P13, P14). No new name.
  No port: `main` has no round solid.

- [ ] **Step 1: Failing tests** (design 7 numbers; within `1e-15` absolute):

  ```cpp
  TEST(ComputeDistanceToSolid, SphereOutside_IsDistanceMinusRadius)  // centre {1,2,3}, r 0.5: {1,2,4} -> 0.5,
                                                                      // {0,0,1}; {1.375,2,3.5} -> 0.125, {0.6,0,0.8};
                                                                      // face kCurved
  TEST(ComputeDistanceToSolid, SphereInside_IsNegative)               // {1,2,3.25} -> -0.25, {0,0,1}
  TEST(ComputeDistanceToSolid, SphereCentre_NormalIsMinusSolidXAxis)  // {1,2,3} -> -0.5, {-1,0,0} (P13)
  TEST(ComputeDistanceToSolid, CylinderWall_IsRadialGap)              // axis z, r 1, h 0.5: {3,0,0.25} -> 2,
                                                                      // {1,0,0}, kCurved
  TEST(ComputeDistanceToSolid, CylinderCaps_AreHeightGaps)            // {0.5,0,2} -> 1.5, {0,0,1}, kHighZ;
                                                                      // {0,0.5,-1.5} -> 1, {0,0,-1}, kLowZ
  TEST(ComputeDistanceToSolid, CylinderRims_AreDistanceToTheCircle)   // {1.375,0,1} -> 0.625, {0.6,0,0.8},
                                                                      // kHighRim; {0,-1.375,-1} -> 0.625,
                                                                      // {0,-0.6,-0.8}, kLowRim
  TEST(ComputeDistanceToSolid, CylinderInside_NearerOfWallAndCap)     // {0.9,0,0} -> -0.1, {1,0,0}, kCurved;
                                                                      // {0,0,0.375} -> -0.125, {0,0,1}, kHighZ
  TEST(ComputeDistanceToSolid, CylinderInsideTie_TakesWall)           // {0.75,0,0.25} -> -0.25, {1,0,0}, kCurved
  TEST(ComputeDistanceToSolid, CylinderOnAxis_NormalIsMinusSolidXAxis)  // r 0.25, h 2, {0,0,0} -> -0.25,
                                                                      // {0,1,0} (P13: minus the pose's x column,
                                                                      // {0,-1,0} for the cylinder ToSolid builds
                                                                      // along z; the test checks the x column first)
  TEST(ComputeDistanceToSolid, CylinderRegionBoundaries_AreContinuous)  // {1,0,0.75} (wall_distance 0,
                                                                      // cap_distance 0.25): 0.25 kHighZ;
                                                                      // {1.25,0,0.5} (cap_distance 0): 0.25 kCurved
  TEST(ComputeDistanceToSolid, TurnedCylinder_UsesSolidAxes)          // axis {1,0,0}: {0.25,3,0} -> 2, {0,1,0}
  TEST(ComputeSolidVelocity, TurningSphere_IsAngularCrossArm)         // origin 0, angular {0,0,2}, {0.5,0,0}: {0,1,0}
  ```
- [ ] **Step 2: Run.** Expected: the new tests fail.
- [ ] **Step 3: Implement** design 5.7 "Distance and normal" with P13 and P14 exactly (branch order rim, wall,
  cap; locals `wall_distance`, `cap_distance`, `cap_sign`, as the paper's `d_wall`, `d_cap`, `sigma_z`). Only
  `+ - * /`, `math::Abs`, `math::Sqrt`. `Read:` lines name `<eq:sphere-distance>` and
  `<eq:cylinder-distance>`. Glossary rows: `degenerate normal` (the normal chosen where every direction is nearest).
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Compute the distance to a sphere and a cylinder`.

### Task 5: Plate side and the nearest solid surface

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_shapes.h`, `solid_shapes_test.cc`, `docs/glossary.md`

**Interfaces:**
- Consumes: Task 4.
- Produces: `FindSolidSide`, `SolidVelocitySample`, `ComputeSolidVelocitySample` (design 3.2, P10). Port note:
  `FindSolidSide` replaces `main:native/src/liquid_wall_contact_3d.cc:595-628`
  (`ClassifyInitialElasticBladeSide3D`) without its status handling.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(FindSolidSide, AboveAlongNormal_IsAlongNormal)          // plate h 0.1, normal {0,0,1}, p.z 0.5
  TEST(FindSolidSide, BelowAgainstNormal_IsAgainstNormal)      // p.z -0.5
  TEST(FindSolidSide, InsideThickness_IsNoPlate)               // p.z 0.05 and p.z exactly 0.1
  TEST(FindSolidSide, NotAPlate_IsNoPlate)                     // a box
  TEST(ComputeSolidVelocitySample, NoSolids_HasNoSolid)
  TEST(ComputeSolidVelocitySample, TwoSolids_KeepsNearest)     // distances 0.3 and 0.1: index 1
  TEST(ComputeSolidVelocitySample, EqualDistances_KeepsLowerIndex)
  TEST(ComputeSolidVelocitySample, BeyondSearchDistance_HasNoSolid)  // distance 0.3, search distance 0.2: defaults (P10)
  TEST(ComputeSolidVelocitySample, AtSearchDistance_HasSolid)  // distance equal to the search distance: has_solid
  TEST(ComputeSolidVelocitySample, Plate_ReportsSideWithPlateId)  // solid_id 5, above: {5, kAlongNormal}
  TEST(ComputeSolidVelocitySample, TurningBox_VelocityAtNearestSurfacePoint)  // box half 1 at origin, angular
                                                               // {0,0,1}, point {3,0,0}: velocity at {1,0,0} ==
                                                               // {0,1,0}; elapsed 0
  // Amended 2026-11-15 (A74):
  TEST(FindSolidSide, SphereOrCylinder_IsNoPlate)              // P3: only a plate has sides
  TEST(ComputeSolidVelocitySample, TurningSphere_VelocityAtNearestSurfacePoint)  // sphere r 1 at origin, angular
                                                               // {0,0,1}, point {3,0,0}: distance 2, normal {1,0,0},
                                                               // velocity at {1,0,0} == {0,1,0}; side {0, kNoPlate}
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** Glossary rows: `side of a plate` (`kAlongNormal`, `kAgainstNormal`), `nearest solid
  surface`.
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Find the plate side and the nearest solid surface of a point`.
  New names: `FindSolidSide`, `SolidVelocitySample`, `ComputeSolidVelocitySample`.

### Task 6: The contact tolerance

**Kind:** numerics

**Files:**
- Create: `native/fluidity/contact/contact_tolerance.h`, `contact_tolerance_test.cc`
- Modify: `CMakeLists.txt`, `docs/glossary.md`

**Interfaces:**
- Consumes: `Solid`, `ComputeSolidPose` (Task 4).
- Produces: design 3.3 exactly: the three constants, `ContactScale`, `ComputeContactScale`,
  `ComputeContactTolerance`, `CheckContactTolerance`. Rules P8.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(ComputeContactTolerance, SceneAtOrigin_IsTheCellWidthFloor)   // largest 0, cell 0.01: 0x1p-30 * 0.01 exactly
  TEST(ComputeContactTolerance, SceneMovedAMillionMeters_ScalesWithIt)  // box at {1,2,3} half 0.5, step 0.01, cell
                                                                     // 0.01; then moved by {0,0,1e6}: difference
                                                                     // of tolerances == 0x1p-40 * 1e6 within 1e-12
                                                                     // relative; largest_coordinate grew by 1e6
                                                                     // (z is the largest coordinate of {1,2,3}, so
                                                                     // only a move along z grows the scale by 1e6)
  TEST(ComputeContactScale, BoxAndPlate_TakesThinnestAndLargest)     // plate h 0.002, box half {1,2,3} at {10,0,0}:
                                                                     // thinnest 0.004, largest 10 + Length({1,2,3})
  TEST(ComputeContactScale, MovingSolid_UsesEndOfStepOrigin)         // origin {0,0,0}, v {0,0,-5}, step 2: largest 10
  TEST(ComputeContactScale, NoThinSolid_ThinnestIsCellWidth)         // flat solids only
  TEST(ComputeContactScale, SphereAndCylinder_GiveTheirThinnestSize)  // Amended 2026-11-15 (A74, P8): sphere r 0.3
                                                                     // at {5,0,0}, cylinder r 1 h 0.2 at origin:
                                                                     // thinnest 0.4 (2 h), largest 5 +
                                                                     // Length({0.3,0.3,0.3})
  TEST(CheckContactTolerance, PlateThinnerThan1024Tolerances_IsSolidTooThin)  // thinnest 1023 * tol
  TEST(CheckContactTolerance, Exactly1024Tolerances_IsNone)
  TEST(CheckContactTolerance, CellNarrowerThan1024Tolerances_IsSolidTooThin)
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement** design 5.2's formula; the header explains with numbers why `0x1p-40` is
  `4096 * epsilon` and 128 times the evaluation error, and that the tolerance is never enlarged to make contact
  pass (report 10.4). Glossary rows: `contact tolerance`, `rest gap`; never say: `certified`, `nudge`,
  `penetration tolerance`, `resolution` (for this).
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Add the contact tolerance that follows the scene's scale`. New names: the 3.3 list.

### Task 7: The contact response

**Kind:** numerics

**Files:**
- Create: `native/fluidity/contact/solid_contact_types.h` (built; the plan first said `solid_contact.h`), `contact_response_test.cc`
- Modify: `CMakeLists.txt`, `docs/glossary.md`

**Interfaces:**
- Consumes: `Vec3`, `math::Dot`, `math::Length`; `kMaxTurnPerStep` from Task 3 (`solid_shapes.h`, not declared here).
- Produces: `ComputeContactResponse` (design 3.4), and in `solid_contact_types.h` the declarations `ContactKind`,
  `kMaxHitsPerStep`, `kMaxAdvanceCount`, `SolidContactInputs`, `SolidReaction`,
  `SolidContactResult`, `FirstHit` exactly as design 3.4 (bodies of `FindFirstHit` and `MoveParticle` arrive in
  Tasks 8 to 13).
- Port: `main:native/include/fluidity/base/liquid_wall_contact_math_3d.h:101-149` (`ApplyOneWayResponse`),
  without `CertifyContactSide` (`:83-99`), `NudgeOutward` (`:57-81`) and the `ProjectionBounds` estimate; the
  normal is a unit vector, so the division by `normal_length_squared` goes. The blend is written
  `velocity + fraction * (solid_velocity - velocity)` (not `std::lerp`).

- [ ] **Step 1: Failing tests** (floor normal `{0,0,1}` unless stated):

  ```cpp
  TEST(ComputeContactResponse, HeadOn_BecomesSolidVelocity)       // v {0,0,-2}, solid {0,0,0.5}, friction 0.3:
                                                                  // == {0,0,0.5} exactly (stick assignment)
  TEST(ComputeContactResponse, Slip_ReducesSidewaysByFrictionTimesApproach)  // v {3,0,-1}, fixed, 0.5: {2.5,0,0}
  TEST(ComputeContactResponse, StickAtThreshold_IsSolidVelocity)  // same v, friction 3: == {0,0,0}
  TEST(ComputeContactResponse, StickAboveThreshold_IsSolidVelocity)  // friction 4
  TEST(ComputeContactResponse, Leaving_IsUnchangedBitForBit)      // v {1,0,2}
  TEST(ComputeContactResponse, SidewaysOnly_IsUnchangedBitForBit) // v {3,0,0}
  TEST(ComputeContactResponse, FixedSolid_NeverRaisesSpeed)       // 1000 cases (P11; draw normal {w,w,w} redrawn
                                                                  // while Length < 0.25, velocity 10 * {w,w,w},
                                                                  // friction 2 * next()): Length(after) <=
                                                                  // Length(before) * (1 + 0x1p-50)
  TEST(ComputeContactResponse, FrictionZeroFastSolid_KeepsSidewaysBitForBit)  // amended 2026-11-15: the solid moves
                                                                  // fast SIDEWAYS: v {1e-9,0,-1}, solid {1e8,0,0},
                                                                  // normal {0,0,1}, friction 0: result bits ==
                                                                  // {1e-9, 0, +0.0} exactly (ToBits). A response in the
                                                                  // solid's frame gives x == 0 (1e-9 - 1e8 + 1e8), so
                                                                  // this fails for the bug Review Focus 4 names
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement** design 5.4 steps 1 to 3. Body comment kept: why the approach is removed in world
  coordinates. Glossary rows: `approach speed`, `sideways speed`, `stick`, `slip`, `friction` (no unit, the
  Coulomb coefficient); never say: `outward_normal`, `impulse` (for a velocity change).
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Remove the approach and apply friction at a solid surface`. New names:
  `ComputeContactResponse`, `ContactKind`, `SolidContactInputs`, `SolidReaction`, `SolidContactResult`,
  `FirstHit`, `kMaxHitsPerStep`, `kMaxAdvanceCount`.

### Task 8: First hit on solids that do not turn

**Kind:** numerics

**Files:**
- Create: `native/fluidity/contact/solid_contact_steps.h` (shared, private to `contact/`), `find_first_hit_test.cc`
- Modify: `native/fluidity/contact/solid_contact.h`, `docs/glossary.md`

**Interfaces:**
- Consumes: Tasks 4, 7.
- Produces: `FindFirstHit` (design 3.4) for solids with zero angular velocity; turning solids return
  `kTooCloseToCall/kSlowApproachToTurningSolid` until Task 9 (no test relies on that). Private:
  `ComputeDistanceForParticle(const SolidContactInputs& inputs, std::uint32_t solid_index, SolidPose pose,
  Vec3 point, PlateSide plate_side)` in `solid_contact_steps.h` (P3 distance; A80: `inputs` and `solid_index`, not a
  bare `Solid`). `FindFirstHit` loops over the solids and calls one private `FindFirstHitOnSolid(inputs,
  solid_index, start, velocity, start_elapsed, duration, FirstHit*)`.

**Room for module 1b (mesh solids) (A80):** 1b adds a mesh table to `inputs` and a `case` to one `switch`; these
three habits mean no call site and no loop is rewritten. (1) Every private helper of `solid_contact_steps.h` that
evaluates a solid takes `const SolidContactInputs& inputs` and `solid_index`, never a bare `const Solid&`. (2) The
`switch (solid.shape)` of `FindFirstHitOnSolid` writes every shape out and has no `default` that computes; an
unknown shape asserts in debug and returns `kBadInput`. (3) The touching rule lives inside each shape's branch,
never in the loop.
- Replaces: `main:native/src/liquid_wall_contact_3d.cc:24-131` (`CompareBoxSweepTimes`, `SelectBoxImpactFace`)
  and the linear hit time of `liquid_wall_contact_math_3d.h:420-480`.

- [ ] **Step 1: Failing tests** (`tol = kTestContactTolerance`, start elapsed 0 unless stated):

  ```cpp
  TEST(FindFirstHit, NoSolids_IsFree)                          // solid_index kNoSolid
  TEST(FindFirstHit, FlatSolidAhead_HitsAtGapOverApproach)     // face x = 1 (point {1,0,0}, normal {-1,0,0}),
                                                               // start {0.9,0,0}, v {1,0,0}, 0.2 s:
                                                               // elapsed 0.1 - tol
  TEST(FindFirstHit, BoxFace_ReportsEnteringFace)              // box half 0.5 at origin, start {-1,0.1,0.2},
                                                               // v {2,0,0}, 1 s: (0.5 - tol) / 2, kLowX
  TEST(FindFirstHit, BoxCornerTieAtEndOfStep_ReportsLowX)      // box centre {1.5,1.5,0.5} half 0.5, start
                                                               // {-tol,-tol,0.5}, v {1,1,0}, 1 s: elapsed exactly 1,
                                                               // kLowX
  TEST(FindFirstHit, BoxEdgeGraze_StaysFree)                   // same box, start {0,1 + 4 * tol,0.5}, v {1,1,0}
  TEST(FindFirstHit, TwoSolidsSameTime_LowerIndexWins)         // two flat solids at the same plane, ids 3 and 4
  TEST(FindFirstHit, CommittedPlate_UsesOnlyCommittedFace)     // plate h 0.01, start p.z -0.005, v {0,0,1},
                                                               // both are hits at 0 (P3's half space holds the
                                                               // whole thickness): committed kAlongNormal gives
                                                               // kHighZ, distance -0.015; uncommitted kLowZ, -0.005
  TEST(FindFirstHit, DomainBoxOpenTop_ExitIsFree)              // {0,0,0}..{1,1,1}, start {0.5,0.5,0.9}, v {0,0,1}
  TEST(FindFirstHit, MovingSolid_UsesRelativeVelocity)         // plate h 1/32, centre {-1,0,0} moving {2,0,0},
                                                               // particle at rest at origin, 1 s:
                                                               // elapsed (1 - 1/32 - tol) / 2, kHighZ
  TEST(FindFirstHit, StartElapsed_SamplesSolidLater)           // solid moving {-1,0,0}, start_elapsed 0.5: gap
                                                               // measured at the later pose
  TEST(FindFirstHit, SlidingOnTouchedFace_IsNotAHit)           // floor, start {0,0,tol}, v {5,0,-tol/4}, 1 s: free
  TEST(FindFirstHit, PushingIntoTouchedFace_HitsAtZero)        // same, v {5,0,-1}: elapsed 0
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement** design 5.3 step 1 and step 3 with P2 and P3; `// Step N:` follows
  `<proc-find-first-hit>`. Glossary rows: `sweep`, `hit`, `hit fraction`, `gap`; never say: `contact_fraction`,
  `impact`, `swept impact`.
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Find the first hit of a straight path on solids that do not turn`.
  New names: `FindFirstHit`.

**Amended 2026-11-15 (A74):** `FindFirstHit` dispatches per shape (through `FindFirstHitOnSolid`, A80); the `kSphere` and `kCylinder` branches arrive in
Task 8a (until then they hold a debug assertion; no Task 8 test builds them). The order of step 3 (smallest time,
then lower solid index) covers every shape.

### Task 8a: First hit on a sphere and a cylinder (A74)

**As built (2026-11-15), where this task's text below differs:** the cylinder's first hit is the interval form of P15 (not two candidates each checked at its hit point: that form let paths through the grown rim edge pass unseen, found in this task's review); `FindSphereHit(const SolidContactInputs&, std::uint32_t solid_index, OwnPath own_path, double duration)` and `FindCylinderHit` likewise read the tolerance from `inputs.contact_tolerance`; normals at a rim hit on a tilted cylinder are tested to 1e-5; the turning bound of P17 uses the step's duration (this also corrects the comment of Task 12a's `TurningCylinderRimHit_IsNotMoved`).

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact_steps.h` (new private steps), `find_first_hit_test.cc`; `solid_contact.h` only if a public function changes

**Interfaces:**
- Consumes: Tasks 4a, 8; `ComputeShapeTurningAngularVelocity` (Task 3) for P17's `omega_across`: use it, write no
  second helper.
- Produces: the closed-form `kSphere` and `kCylinder` branches of `FindFirstHit` (design 5.7) with P15, P16 and P17
  (a sphere never turns; a cylinder whose `omega_across` is below P17's bound is swept here; a cylinder turning
  across its axis goes to the Task 9 search). Private helpers in `solid_contact_steps.h`, names decided here:
  `FindSphereHit(...)` and `FindCylinderHit(...)`, each taking the start and relative velocity in the solid's axes,
  `inputs` and `solid_index` (A80, not a bare `Solid`), `contact_tolerance` and `duration`, returning the hit time
  or "free". No port: `main` has none.

**Room for module 1b (mesh solids) (A80):** the three habits of Task 8 apply here unchanged: the two branches are
`case`s of `FindFirstHitOnSolid`'s `switch` with no computing `default`, and the touching rule (P16) stays inside
each branch.

- [ ] **Step 1: Failing tests** (`tol = kTestContactTolerance`; expected times within `0x1p-44` relative unless
  stated; normals within `1e-8`):

  ```cpp
  // sphere at the origin, r 0.5, duration 2
  TEST(FindFirstHit, SphereHeadOn_HitsAtGapOverSpeed)          // {-2,0,0}, v {1,0,0}: 1.5 - tol, {-1,0,0}, kCurved
  TEST(FindFirstHit, SphereGrazing_HitsAtQuadraticRoot)        // {-2,0.3,0}: 2 - Sqrt((0.5 + tol)^2 - 0.09)
                                                               // (1.5999999988358469), normal near {-0.8,0.6,0}
  TEST(FindFirstHit, SphereTangentWithinRounding_FreeOrHitNearTwo)  // {-2, 0.5 + tol, 0}: no error; free, or hit
                                                               // with elapsed within 0x1p-20 of 2 and normal x
                                                               // within 0x1p-20 of 0
  TEST(FindFirstHit, SphereMiss_IsFree)                        // {-2,0.6,0}
  TEST(FindFirstHit, SphereMovingAway_IsFree)                  // {-2,0,0}, v {-1,0,0} (B >= 0)
  TEST(FindFirstHit, SphereTouchingTangentPath_IsNotAHit)      // start {-(0.5 + tol),0,0}, v {0,1,0}: free (P16)
  TEST(FindFirstHit, SphereTouchingPushingIn_HitsAtZero)       // same start, v {1,0,0}: elapsed 0
  TEST(FindFirstHit, MovingSphere_UsesRelativeVelocity)        // centre {2,0,0} moving {-1,0,0}, particle at rest at
                                                               // origin: 1.5 - tol
  TEST(FindFirstHit, SpinningSphere_IsClosedForm)              // head-on case, sphere angular {0,0,5}: same time
                                                               // within 0x1p-44 relative (no stepping, P17)
  // cylinder at the origin, axis z, r 0.5, h 1
  TEST(FindFirstHit, CylinderWall_HitsAtRadialGap)             // {-2,0,0.25}, v {1,0,0}, 2 s: 1.5 - tol, kCurved
  TEST(FindFirstHit, CylinderCap_HitsAtHeightGap)              // {0.1,0.2,3}, v {0,0,-1}, 4 s: 2 - tol, kHighZ
  TEST(FindFirstHit, CylinderObliqueWall_CapPlaneOutsideDiscRejected)  // {-2,0,1.5}, v {1,0,-0.5}, 4 s: 1.5 - tol,
                                                               // kCurved
  TEST(FindFirstHit, CylinderOverRimIntoWall_HitsWall)         // {-2,0,2}, v {1,0,-1}, 4 s: 1.5 - tol, kCurved
  TEST(FindFirstHit, CylinderParallelToAxisOutside_IsFree)     // {0.6,0,3}, v {0,0,-1}, 4 s
  TEST(FindFirstHit, CylinderParallelToAxisInGrownRadius_HitsRim)  // {0.5 + tol/2,0,3}, v {0,0,-1}, 4 s: 2 - tol,
                                                               // kHighRim, at_hit.distance == Sqrt(1.25) * tol within
                                                               // 0x1p-40 (the sharp corner, P15)
  TEST(FindFirstHit, CylinderParallelToCapsAbove_IsFree)       // {-2,0,1.5}, v {1,0,0}
  TEST(FindFirstHit, CylinderTouchingWallTangentPath_IsNotAHit)  // start {-(0.5 + tol),0,0}, v {0,1,0}: free (P16)
  TEST(FindFirstHit, CylinderRimCornerStartMovingAway_IsFree)  // start {0.5 + tol/2, 0, 1 + tol/2} (true gap
                                                               // (Sqrt(0.5) - 1) tol < 0: touching) v {1,0,1}: free;
                                                               // start {0.5 + 0.9 tol, 0, 1 + 0.9 tol} (true gap
                                                               // > 0, inside the sharp corner), v {-1,0,0}: hit at 0;
                                                               // v {1,0,0}: free (P15; each start's sign holds up
                                                               // to rounding, see P15)
  TEST(FindFirstHit, SpinningCylinderAboutAxis_IsClosedForm)   // wall case, angular {0,0,5}: same time, no stepping
  ```
  The rim-corner starts are far from a rounding boundary (`0.9 tol` against `tol`), so they hold whether the inside
  test is `rho0 <= R && Abs(p0.z) <= H` or, if the code shows the need, the same computed gaps that set the start's
  sign (P15).
- [ ] **Step 2: Run.** Expected: the new tests fail.
- [ ] **Step 3: Implement** design 5.7 "First hit": sphere `A = Dot(u,u)`, `B = Dot(p0,u)`, `C = g0 * (L0 + R)`,
  free when `A == 0`, `B >= 0` or `B * B - A * C < 0`, else `t = C / (-B + Sqrt(B * B - A * C))`; cylinder with
  P15's two candidates in order; starts per P15 and P16. Only `+ - * /`, `math::Abs`, `math::Sqrt`. `// Step 1:`
  comments keep `<proc-find-first-hit>` numbering, and the branches cite `<proc-sphere-first-hit>` and
  `<proc-cylinder-first-hit>` in `Read:`.
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Find the first hit of a straight path on a sphere and a cylinder`.

### Task 9: First hit on turning solids

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact_steps.h` (new private steps), `solid_contact.h` (public functions only), `find_first_hit_test.cc`

**Interfaces:**
- Consumes: Task 8, `ComputeSolidPose`; the seven-argument form `FindFirstHit(inputs, start, velocity_for_this_step,
  start_elapsed, duration, FirstHit* first_hit, ContactCloseCall* close_call)` already exists since Task 8 (the
  six-argument form passes `nullptr`) and maps `kTooCloseToCall` to `kSlowApproachToTurningSolid`; Task 9 fills in the
  turning case behind `FindFirstHitOnSolid` and must keep the rule "writes `kNone` whenever the pointer is not null and
  the failure is not a close call". Preconditions to respect: `FindFirstHit` requires `duration <= step_duration - start_elapsed` (asserted); the start rule and the touching rule use the solid's pose at `start_elapsed`; a path that meets a grown edge exactly is a hit; the domain box's touching rule is applied per closed side.
  Also `ComputeShapeTurningAngularVelocity` (Task 3) to decide whether a cylinder's
  shape turns (P17): use it, write no second helper. **Amended 2026-11-15 (Task 8a as built):** a cylinder counts
  as turning only when `Abs(omega_across) * step_duration * Length(half_size) > tol / 8` (the step's duration, not
  the sweep's); below that bound it uses the closed form.
- Produces: `FindFirstHit` for turning solids (design 5.3 step 2). **Room for module 1b (mesh solids) (A80):** the
  stepping search reads `g(t)` only through `ComputeDistanceForParticle` and decides "turning" only through
  `ComputeShapeTurningAngularVelocity`; it must not assume that the distance belongs to one face (1b gives it a
  distance that comes from a triangle). A second close-call reason would add a `ContactCloseCall*` parameter to
  `FindFirstHitOnSolid`; today every close call from it is `kSlowApproachToTurningSolid`, including a tilting
  cylinder above the P17 bound.
- Port the idea only: `main:native/src/rigid_body_3d.cc:861-881` (`RigidBodySweepDistanceBound3D`, the bound
  `translation + radius * angle`) and `:883-1130` (`SweepRigidBoxPoint3D`); not its bisection, stack or rounding
  slack.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(FindFirstHit, TurningBoxReachesRestingParticle_AtTrigonometricAngle)
      // box half 0.5 at origin, angular {0,0,1}, particle at rest at 0.6 * {cos 0.3, sin 0.3, 0}, 1 s:
      // elapsed == 0.3 + acos((0.5 + tol) / 0.6) within 0x1p-30, face kHighX
  TEST(FindFirstHit, TurningBoxNotReachedInStep_IsFree)             // angular {0,0,-1}, same particle, 0.2 s
                                                                    // (local angle 0.5 at the end, gap 0.027)
  TEST(FindFirstHit, GrazingTurningBox_IsSlowApproachToTurningSolid)
      // particle at rest at {0.5 + 2 * tol, 0, 0}, angular {0,0,1}, 0.1 s: kTooCloseToCall
  // Amended 2026-11-15 (paper review, P2): no touching rule on a turning solid
  TEST(FindFirstHit, RestingOnTurningBox_IsHitAtZero)
      // box half 0.5 at origin, angular {0,0,1}; start {0.5 + tol, 0, 0} (d == tol exactly, g(0) == 0),
      // v {0,0,1} (along the face), 0.1 s: elapsed 0, solid_index 0, face kHighX. Control in the same test: the
      // same box with angular {0,0,0} gives kNoSolid (P2: rate 0, the touching rule frees it)
  // Amended 2026-11-15 (A74, P17): a cylinder turning across its axis uses the same search with its true distance
  TEST(FindFirstHit, CylinderTurningAcrossAxis_ReachesRestingParticleAtTrigonometricAngle)
      // cylinder at origin, axis {1,0,0}, r 0.5, h 1, angular {0,0,1}; particle at rest at 0.6 * {cos 1.2, sin 1.2,
      // 0}, 1 s: elapsed == 1.2 - asin((0.5 + tol) / 0.6) (0.21488921385421111) within 0x1p-30, face kCurved
  ```
  **Amended 2026-11-15 (A74):** a sphere never enters this search, and a cylinder only when P17 says its shape
  turns; `ComputeDistanceToSolid` (Task 4a) gives `g(t)`, the bound is unchanged.
  The third test needs the close call: `FindFirstHit` writes it through a pointer decided here,
  `FindFirstHit(..., FirstHit* first_hit, ContactCloseCall* close_call)` (plan decision; may be null), the
  design's six-argument form forwarding null. The close-call output follows `CheckSolid`'s rule: whenever the
  pointer is not null and the failure is not a close call, `kNone` is written.
- [ ] **Step 2: Run.** Expected: the new tests fail.
- [ ] **Step 3: Implement** design 5.3 step 2: `bound = Length(velocity_for_this_step - linear_velocity) +
  Length(angular_velocity) * radius`, `radius` the larger of the path ends' distances to the solid's origin at the
  matching times; `t += g(t) / bound`; hit when `g <= tol / 4`; free when `t >= duration`; more than
  `kMaxAdvanceCount` advances is the close call. The bound's derivation goes in a body comment, and a comment says
  why there is no touching rule here (P2: the face turns, so the plane argument fails; a resting particle is hit at
  0 and carried by the slide in the solid's axes).
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Step toward a turning solid until the path reaches it`.

### Task 9b: Second-order advance toward a turning solid (P18)

**As built (2026-11-15), where this task's text below differs:**
- Commit order: this task was implemented after Task 10 (commits 9, 10, 9b), not before it; its brief predates Task 10.
- Private names beyond the brief: `ComputeIsTurningSolid`, `ComputeEarlierHit`, `FindEarliestHitWithTurningSolids`,
  `GapLowerModel`, `ComputeSecondOrderAdvance`, `HasConvexDistance(SolidShape)` (constexpr, the one dispatch point
  for module 1b).
- P17 amended by review I1: a flat solid or a plate whose angular velocity has a part across its normal of length at
  most `kNegligibleTurnShare = 0x1p-46` of `|omega|` (in `solid_shapes.h`) does not count as turning, instead of "not
  exactly zero"; the face then moves by at most `0.907 * 2^-43 * L`, within `tol / 8`. A cylinder keeps its own bound.
  Tests: `SpinAboutTiltedNormal_IsNoTurn`, `PartAcrossJustBelowShare_IsNoTurn`, `PartAcrossJustAboveShare_IsTurn`.
- Task 10's scene `TurningBoxSlowApproach` is now `TurningBoxHoverOnSpinAxis` (start `{0, 0, 0.5 + 2^-17}`, the hover);
  its test is `MoveParticle.TurningBoxHoverOnSpinAxis_PassesOnCloseCall`. `CloseCall_LeavesFirstHitUnchanged` is now
  `CloseCall_NamesTurningSolid` (Task 10's ruling 19: a close call writes `FirstHit{}` with `solid_index`), with a 1 s
  step and sweep so the hover stays a close call.
- The inward set redraws starts that fall inside the arm (`StartsAwayFromSolid`), as the outward set does.
- The close-call counts of the tables below are the ones measured before and after this task; the counts of the
  other tasks' generated sets that were measured before it are stale (see Tasks 12 to 14).

Added 2026-11-15 (controller ruling 16, from Task 9's review, "Close calls: recommendation", and minors m4 and m7).
Task 9 as built is safe but answers "too close to call" too often: 10.75 % of its 17,000 generated paths and 50 to
62 % of particles near a paddle, and halving the step removes about 5 points of that; the step allows two retries and
then stops the run. This task keeps Task 9's search as the floor and adds P18's second-order advance; it also makes
the controller's flat-solid and plate ruling (P17 amended) and review m4. Run it after Task 9 and before Task 10.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact_steps.h` (`FindHitOnTurningSolid`, `FindFirstHitOnSolid`, new private
  `HasConvexDistance`), `solid_contact.h` (`FindFirstHit`'s loop, m4; docstrings), `solid_shapes.h`
  (`ComputeShapeTurningAngularVelocity` and `CheckSolid` docstrings for the flat solid and the plate), `solid_contact_types.h`
  (the `kMaxAdvanceCount` comment), `find_first_hit_test.cc`, `solid_shapes_test.cc`,
  `docs/method-paper/fluidity-method.typ`, `docs/glossary.md`

**Interfaces:**
- Consumes: Task 9 as built after its fix round (`FindHitOnTurningSolid`, `kTurningSolidHitGapShare`,
  `kMaxAdvanceCount`, the lower-bound reference `IsOutsideGrownSolidThroughout`, `CountTurningSweep`,
  `ExpectTurningSweepCounts`, `BuildQuarterTurnLongBox`, `BuildOutwardPath`); `ComputeSolidVelocity`,
  `ComputeShapeTurningAngularVelocity` (Task 3).
- Produces: P18 inside `FindHitOnTurningSolid`; private `constexpr bool HasConvexDistance(SolidShape shape)` (true for
  flat solid, plate, box, sphere, cylinder; false for the domain box); `FindHitOnTurningSolid` gains a last parameter
  `int* advance_count` (may be null; written on every return with the advances made) so tests can read the cost;
  `FindFirstHitOnSolid` passes null. `FindFirstHit`'s public form and `FirstHit` do not change. P17's flat-solid
  and plate rule in `ComputeShapeTurningAngularVelocity`. m4 in `FindFirstHit`: first every solid that does not turn, in index
  order; then every turning solid in index order, each searched with `duration` replaced by the earliest hit time
  found so far when that is smaller; a turning solid's hit replaces the kept one when its time is smaller, or equal
  with a lower index; a turning search that reaches the cap is free, and the kept hit stands (safe: the path stayed at
  `d >= tol` up to then). The close call is returned only when a turning search fails inside its capped duration.
- **Room for module 1b (mesh solids) (A80):** the choice of advance is `HasConvexDistance(solid.shape)` only, read
  inside `FindHitOnTurningSolid`; the advance still reads the gap and the normal only through
  `ComputeDistanceForParticle`. A mesh is not convex: 1b adds `case kMesh: return false;` (first order) unless its
  design chooses per-triangle second order with measurements (`changes-to-apply.md`, section 01b). Nothing assumes
  the normal belongs to one face.
- The plate is unbounded in its own x and y in every function that reads it (P17 amended lists them), so ruling 17
  revised frees its spin about the normal like the flat solid's. Task 9's `TurningPlateCommitted_SearchUsesCommittedHalfSpace`
  used a plate spinning about its normal, which is now a closed form: give it angular `{1, 0, 0}` (a turn about an
  axis in its plane) so it still exercises the search; its expected values (both forms hit at 0, kLowZ at -0.005,
  committed kHighZ at -0.015) are unchanged.

Checked before writing (mirror of `FindHitOnTurningSolid` in Python, bit-faithful to Task 9; reference that cannot
miss: Task 9's branch and bound with its own loose rate bound, and for the paddle a face-plane reference whose bound
`|f''| <= |omega|^2 (r0 + |v_rel| T) + 2 |omega| |v_rel|` on each smooth face-plane gap is derived without convexity;
scripts in the session scratchpad `task9b/`):

| set | paths | close calls, Task 9 | close calls, 9b | mistakes, 9b | advances, 9b: mean / p99 / max |
|---|---|---|---|---|---|
| Task 9's 17,000 (random 5 solids, nearly tight, farthest at end, edges and corners, rims) | 17,000 | 1,828 (10.75 %) | 0 | 0 | 3.5 to 6.6 / 4 to 13 / 27 |
| outward radial (Task 9 I1 geometry) | 3,000 | 102 per 2,746 (review) | 0 | 0 | 4.5 / 6 / 6 |
| inward radial, long box | 3,000 | - | 0 | 0 | 5.4 / 7 / 7 |
| inward through the origin, four shapes, turn up to pi/2 | 4,000 | - | 0 | 0 | - |
| slide past the radial end face of a quarter-turn arm | 3,000 | - | 0 | 0 | 5.8 / 18 / 86 |
| broad: faces, edges, corners, rims, random axes, 4 seeds | 8,000 | 72.5 % (review) | 1 | 0 | 10.2 to 10.8 / 35 to 38 / 128 |
| paddle, gaps 1e-7 to 1e-2 m, three kinds, steps 1/120 to 1/480 | 18,000 | 42 to 62 % | 0 | 0 | 2.0 to 4.4 / 9 to 20 / 38 |
| paddle, gaps 0.3 tol to 1e-2 m, normal speed down to 1.5e-9 m/s | 18,000 | 69 to 79 % | 0 approach, 0 separate, 0.1 % co-moving | 0 (3,600 checked) | 4.2 to 8.2 / 25 to 54 / 128 |

Mutants of the formula on these sets: `K` from the start's distance alone, 149 mistakes in 3,000 outward paths (3.8 cm
deep); from the end's alone, 136 in 3,000 end-face paths (5 mm); with the Coriolis term once instead of twice, 197
(8.7 mm); without it, 1,515 outward; without the centripetal term or halved, almost every hit wrong; `gdot` with the
wrong sign, from the world velocity, or without the turn's surface velocity, almost every hit wrong; Task 9's M7 (the
bound from the start alone) is still caught through the first-order floor (283 in 3,000). Two mutants found no
mistake anywhere: `K` from the distance at the current sample (unsafe in principle: the distance can grow over the
advance) and the second-order advance without the first-order floor (both advances are safe; the floor changes the
mean count from 5.41 to 5.4). On a test-only turning room (the domain box's distance, not convex) the second-order
advance made 275 mistakes in 1,000 paths and on an L of two boxes 14 in 923; first order made none.

- [ ] **Step 1: Failing tests** (`tol = kTestContactTolerance` unless stated; hit times "within 2^-30" use
  `kTurningHitTimeTolerance`; "in the band" means `at_hit.distance` in `[tol - tol / 64, (1 + kTurningSolidHitGapShare)
  * tol]`; the quarter-turn long box is `BuildQuarterTurnLongBox({})`, half sizes {0.5, 4, 0.5}, world axes, turning
  `{0, 0, pi / 2}` rad/s, step 1 s; every value below comes from the mirror and was checked against the true crossing
  time found by bisection on the distance):

  ```cpp
  // Task 9 tests that change (m7: a close call that the second-order advance finds in one advance gets a new path)
  TEST(FindFirstHit, SecondOrderSearchStoppingTimes_PinAdvanceCounts)   // replaces FirstOrderSearchStoppingTimes_...
      // same cases, stopping times within 1e-13 s (mirror; the implementer records the C++ values and pins them,
      // explaining any difference above 1e-13 by the order of operations in gdot):
      // trig box 3 advances, 0.8856855405066031 (Task 9: 30, 0.88568554023775237);
      // cylinder 3, 0.21488921385342513 (Task 9: 25); slow approach r 0.505859375: 2, 0.15235117303165863 (113);
      // slow approach r 0.5029296875: 2, 0.1079898627864529 (157, a close call in Task 9);
      // TurningLongBoxOutwardPath's path: 6, 0.9562428207946203 (32); StartElapsed_MovingTurningBox_...'s: 5,
      // 0.7981334038631699. Also through FindHitOnTurningSolid's advance_count: 3, 3, 2, 2, 6, 5 exactly
  TEST(FindFirstHit, TurningBoxSlowApproachBeyondFirstOrderLimit_IsHit)  // replaces ...BeyondAdvanceLimit_IsCloseCall:
      // at rest at {0.5029296875, 0, 0}, box half 0.5 turning {0,0,1}, 1 s: hit at acos((0.5 + tol) / 0.5029296875)
      // = 0.10798986286079153 within 4 * 2^-30, kHighX, in the band
  TEST(FindFirstHit, GrazingTurningBox_HitWhereFaceTurnsOntoParticle)    // replaces GrazingTurningBox_IsSlow...:
      // at rest at {0.5 + 2 tol, 0, 0}, 0.1 s: hit at acos((0.5 + tol) / (0.5 + 2 tol)) = 6.103515625947391e-05
      // within 2^-30 (mirror 6.103515613631316e-05, one advance; Task 9 needed 43,154), kHighX, in the band
  TEST(FindFirstHit, CloseCall_LeavesFirstHitUnchanged)                 // path changed to the hover below at
      // z = 0.5 + 2^-17; the rest unchanged
  // The remaining close call, and the advance limit from both sides (libm-free: on the spin axis the own point is
  // {0, 0, z} exactly, the distance z - 0.5, gdot exactly 0, K = z, so every advance is 2 g / Sqrt(2 K g))
  TEST(FindFirstHit, TurningBoxHoverOnSpinAxisWithinAdvanceLimit_IsFree)
      // box half 0.5 at the origin turning {0,0,1}; at rest at {0, 0, 0.5 + 2^-15}, 1 s: free (91 advances; Task 9
      // needs about 16,400)
  TEST(FindFirstHit, TurningBoxHoverOnSpinAxisBeyondAdvanceLimit_IsCloseCall)
      // at rest at {0, 0, 0.5 + 2^-17}: kTooCloseToCall / kSlowApproachToTurningSolid (182 needed; 1 / advance =
      // 181.02, far from an integer, so the count does not depend on rounding)
  // The radius bound inside K (P18): each end of r_max is load-bearing
  TEST(FindFirstHit, TurningLongBoxOutwardPathFromNearOrigin_FarEndOfRadiusBoundsCurvature)
      // quarter-turn long box; start 0.6 * {sin 1.68125, -cos 1.68125, 0}, velocity 3 * {sin 1.68125, -cos 1.68125,
      // 0}: hit at 0.98010536766574 within 2^-30 (crossing 0.9801053676657359), kHighX, in the band. K from the
      // start's distance alone: "hit" 3.8 cm inside after 2 advances
  TEST(FindFirstHit, TurningLongBoxSlidePastEndFace_CoriolisAndNearEndBoundCurvature)
      // quarter-turn long box; start {-0.4, 4.05, 0}, velocity {8 - 4.05 * pi / 2, -0.4 * pi / 2, 0} (the end face's
      // velocity there plus 8 m/s along own x): hit at 0.0815739451161 within 2^-30 (crossing 0.08157394511611948),
      // kHighY, in the band, 3 advances. K from the end's distance alone: 1.8 mm inside; Coriolis term once: 8.7 mm;
      // without it: 2.5 cm
  // Not convex: first order only (P18)
  TEST(HasConvexDistance, TurningShapesYes_DomainBoxNo)                 // flat solid, plate, box, sphere, cylinder
                                                                        // true; domain box false
  TEST(FindHitOnTurningSolid, TurningRoom_NotConvexSoFirstOrder)
      // test-only Solid of shape kDomainBox built by hand: origin 0, axes kIdentity, half_size {1,1,1}, every side
      // closed, angular {0, 0, pi / 2} (CheckSolid would refuse it; FindHitOnTurningSolid is called directly);
      // at {0.5, 0.7, 0} moving {1, -1, 0}, 1 s: hit at 0.66938054133 within 2^-30 (crossing 0.6693805413275088), in
      // the band. With the second-order advance the search says free after 2 advances and the path ends 0.5 m beyond
      // a wall
  // P17 amended (controller rulings 17 and 17 revised): a flat solid or a plate spinning about its normal does not
  // turn (both are unbounded sideways); a turn about an axis in its plane still does
  TEST(ComputeShapeTurningAngularVelocity, FlatSolidAndPlateSpinAboutNormal_IsNoTurn)  // normal {0,0,1}: {2,0,100}
                                                                        // gives {2,0,0} exactly for both; a box gives
                                                                        // {2,0,100}
  TEST(CheckSolid, FlatSolidAndPlateSpinningFastAboutNormal_AreAccepted) // {0,0,100} rad/s, step 0.1 s: kNone for
                                                                        // both (was kSolidTurnsTooFar); {100,0,0}:
                                                                        // still kSolidTurnsTooFar
  TEST(FindFirstHit, SpinningFlatSolid_IsClosedForm)                     // floor at the origin, normal {0,0,1},
      // angular {0,0,5}; start {0.3,0,0.5}, v {0,0,-1}, 1 s: elapsed == 0.5 - tol exactly, the same bits as the floor
      // without the spin, kHighZ
  TEST(FindFirstHit, RestingOnSpinningFlatSolid_TouchingRuleFrees)       // same floor; start {0.3,0,tol}, v {1,0,0},
      // 0.1 s: free (P2); Task 9 gave a hit at 0
  TEST(FindFirstHit, SpinningPlateAboutNormal_IsClosedForm)              // plate h 0.01 at the origin, normal
      // {0,0,1}, angular {0,0,5}; start {0.3,0,0.5}, v {0,0,-1}, 1 s: elapsed == (0.5 - 0.01) - tol exactly, the same
      // bits as the plate without the spin, kHighZ
  TEST(FindFirstHit, RestingOnSpinningPlate_TouchingRuleFrees)           // same plate; start {0.3,0,0.01 + tol},
      // v {1,0,0}, 0.1 s: free (P2); Task 9 gave a hit at 0
  TEST(ComputeSolidVelocity, SpinningPlate_SurfaceVelocityKeepsTheSpin)  // same plate, pose at 0, point {0.3,0,0.01}:
      // {0,1.5,0} exactly, so friction feels the spin (the MoveParticle test is Task 12a's
      // SpinningPlatePressedOn_MovesOnlyThroughFriction)
  TEST(FindFirstHit, PlateTurningAboutAxisInItsPlane_IsTurning)          // same plate with angular {1,0,0}; start
      // {0.3,0,0.01 + tol}, v {1,0,0}, 0.1 s: hit at 0, kHighZ (a turning solid has no touching rule, P2)
  // Review m4: turning solids after the others, capped at the earliest hit
  TEST(FindFirstHit, TurningSolidCloseCallAfterEarlierHit_IsHit)
      // solid 0: box half 0.5 turning {0,0,1}; solid 1: flat solid point {0,0,0.6}, normal {0,0,-1}, moving {0,0,-1};
      // at rest at {0, 0, 0.5 + 2^-17}, 1 s: hit on solid 1 at (0.6 - (0.5 + 2^-17)) - tol = 0.09999236967414615
      // (closed form), kHighZ; the box, searched up to that time, is free in 19 advances. Without m4: kTooCloseToCall
  TEST(FindFirstHit, TurningSolidHitBeforeClosedFormHit_Wins)
      // solid 0: flat solid point {0,0,0.95}, normal {0,0,-1}, moving {0,0,-1}; solid 1: the trig box of
      // TurningBoxReachesRestingParticle; that particle, 1 s: solid_index 1 at 0.3 + acos((0.5 + tol) / 0.6) within
      // 2^-30 (the flat solid would hit at 0.95 - tol)
  // Generated sets (P11; reference IsOutsideGrownSolidThroughout unless stated). Every set: 0 mistakes, at most 1
  // close call per 1000 paths, mean advances <= 10 and 99th percentile <= 30 (from advance_count; recorded with
  // RecordProperty), except where stated
  TEST(FindFirstHit, GeneratedTurningSweeps_AgreeWithLowerBoundReference)          // ceiling 85 -> 1 per solid
  TEST(FindFirstHit, GeneratedNearlyTightBoundSweeps_AgreeWithLowerBoundReference) // ceiling 10 -> 2
  TEST(FindFirstHit, GeneratedOutwardSweeps_AgreeWithLowerBoundReference)          // ceiling 180 -> 3 (118 today)
  TEST(FindFirstHit, GeneratedTurningBoxEdgeAndCornerSweeps_AgreeWithLowerBoundReference)  // 1200 -> 4
  TEST(FindFirstHit, GeneratedTiltingCylinderRimSweeps_AgreeWithLowerBoundReference)       // 440 -> 4
  TEST(FindFirstHit, GeneratedEndFaceSlideSweeps_AgreeWithLowerBoundReference)     // new, 3000 paths, below
  TEST(FindFirstHit, GeneratedInwardSweeps_AgreeWithLowerBoundReference)           // new, 3000 paths, below
  TEST(FindFirstHit, GeneratedPaddleSweeps_AgreeWithFacePlaneReference)            // new, 6000 paths, below
  ```
  End-face slide set (kills the end-only `K` and the Coriolis mutants): the quarter-turn long box; per path, in this
  order: gap `exp(log(1e-6) + Next() * (log(0.5) - log(1e-6)))` m; own start `{0.6 * NextSigned(), 4 + tol + gap,
  0.4 * NextSigned()}` (equal to world at the step's start); a relative velocity `{w, -Next() * c, 0.3 * NextSigned()}`
  with `w = 8 * Next()` times -1 when `Next() < 1/3`, and `c` 0, 0.01, 0.1 or 1 by `floor(4 * Next())`; velocity = the
  surface velocity at the start plus that; sweep 1 s from 0. Measured with the mirror's draws: 0 close calls, mean 5.8,
  99th percentile 18; the ceilings above hold.
  Inward set: the quarter-turn long box; per path: start_elapsed 0 when `Next() < 0.5`, else `0.25 * Next()`; r0
  `1.5 + 2.4 * Next()`; speed `1 + 5 * Next()`; angle `0.2 + 1.8 * Next()`; the particle starts r0 out along the own
  direction `{sin angle, -cos angle, 0}` at the pose of start_elapsed and moves toward the origin at that speed, to
  the end of the step (an `OutwardPath` with a negative outward speed and box speed 0). Measured: 0 close calls, mean
  5.4, 99th percentile 7.
  Paddle set: a box of half sizes {0.25, 0.01, 0.1} m at the origin, world axes, turning `{0, 0, 6}` rad/s;
  `contact_tolerance = 0x1p-30 * 0.01` (a 1 cm cell); `step_duration = 1 / 120` s; sweeps from 0 over the whole step;
  2000 paths per kind, kinds approaching, separating, co-moving in turn (`path_index % 3`). Per path: own x `0.24 *
  NextSigned()`; own z `0.09 * NextSigned()`; side `+1` when `Next() < 0.5`; gap `exp(log(0.3 tol) + Next() * (log(0.01)
  - log(0.3 tol)))`; start `{x, side * (0.01 + tol + gap), z}`; slide `0.45 * NextSigned()` m/s along x; normal speed
  `0.15 * 10^(-8 * Next())` m/s, into the face, out of it, or 0 by kind; velocity = surface velocity at the start +
  slide + normal speed times `{0, side, 0}`. Reference: `IsOutsideBoxThroughout`, test-only, the face-plane reference
  (the box distance is the largest of six face-plane gaps, each smooth in time with `|f''| <= |omega|^2 (r0 + |v_rel|
  T) + 2 |omega| |v_rel|`; on `[a, b]` each is at least its chord minus `M (b - a)^2 / 8`; cut until certain), because
  the plain lower-bound reference needs about `gap / bound` per sample along a particle hovering a few tolerances off
  the face and does not finish. Allowance `tol / 64`. Criteria: 0 mistakes; close calls at most 20 per kind (1 %);
  mean advances <= 12 and 99th percentile <= 80 per kind (measured with the mirror's draws: 0, 0 and 2 close calls
  per 2000; mean 5.9, 7.0, 8.2; 99th percentile 42, 49, 54; the reference took at most 62 samples per path).
- [ ] **Step 2: Run.** Expected failures: the stopping-time and advance-count test, the slow approach beyond the old
  limit and the graze (close calls under Task 9), the hover within the limit (about 16,400 first-order advances), the
  generated sets' new ceilings and counts, `HasConvexDistance` (does not exist), the flat-solid and spinning-plate tests, and
  `TurningSolidCloseCallAfterEarlierHit_IsHit`. Passing already: the hover beyond the limit, the two radius tests,
  the turning room, the plate turning about an axis in its plane, the plate's surface velocity, and `TurningSolidHitBeforeClosedFormHit_Wins` (first order is safe; these pin mutants).
- [ ] **Step 3: Implement** P18 in `FindHitOnTurningSolid`: compute `K` once beside `bound`; per sample, after the
  hit and limit checks, `gdot` from `at_position.normal_out_of_solid` and `ComputeSolidVelocity(solid, pose, position)`,
  the two advances and their maximum, when `HasConvexDistance(solid.shape)`; first order alone otherwise. Body
  comments: the convexity of each turning shape's distance and the subgradient case by case (ties, the plate's middle
  plane, the committed plate, the rim), `|p''| <= K`, the lower model, why the larger of two safe advances is safe,
  the cancellation-free root and the `K == 0` branch, the rounding budget, and why the hit band is unchanged. The
  docstring's Cost paragraph and example change (3 advances; close calls are now rare: a hover with no normal motion).
  P17 amended: the flat solid's and the plate's branch of `ComputeShapeTurningAngularVelocity` (one helper for the
  cylinder's, the flat solid's and the plate's "part across the own z axis", written once). m4 in `FindFirstHit`, with a comment on the cap and the
  tie order. `kMaxAdvanceCount`'s comment: "3 in the paper's example; first order needed 30".
- [ ] **Step 4: Mutants** (debug build, full test binary; each must be killed or argued in the report):
  N1 `K` from the start's distance alone (outward-from-near-origin test, generated outward); N2 from the end's alone
  (end-face test, generated end-face); N3 from the distance at the current sample (expected to survive: argue that it
  is unsafe in principle and keep Task 9's `r_max`); N4 without the Coriolis term; N5 with it once (end-face test);
  N6 without the centripetal term; N7 `K / 2`; N8 `gdot` with its sign flipped; N9 `gdot` from the world velocity;
  N10 `gdot` from `v - linear_velocity`, without `omega x r`; N11 `HasConvexDistance` true for every shape (turning
  room); N12 the second-order advance without the first-order floor (expected to survive: both are safe; say so);
  N13 the first-order advance alone (Task 9: the failing tests); N14 the bound of Task 9 from the start alone and N15
  its end origin at `duration` (Task 9's M7 and M10: still killed by `TurningLongBoxOutwardPath_...` and
  `StartElapsed_MovingTurningBox_...`, through the floor); N16 the `gdot <= 0` form used for `gdot > 0` (the
  cancelling form: argue or kill with the separating paddle kind); N17 no m4 cap; N18 m4 with the turning solids
  searched first; N19 the flat solid's or the plate's spin about its normal counted as a turn (the old rule; the two
  touching-rule tests); N20 the plate's whole angular velocity ignored (`PlateTurningAboutAxisInItsPlane_IsTurning`); N21 limits 64 and 256 (the hover pair).
- [ ] **Step 5: Paper and glossary.** `<sec-first-hit>`, "A turning solid": keep the first-order paragraph as the
  floor and as what a shape whose distance is not convex uses; add a paragraph with P18's lower model, the three
  reasons it is safe (convexity with the normal as a subgradient; `|p''| <= K` with the radius over the rest of the
  sweep; `gdot`), the root in its two forms and "the larger of the two advances"; say that a close call is now rare
  and when it remains (a hover with no normal motion). Worked example: same box and particle, `K = 0.6` m/s^2, 3
  advances (Task 9: 30), stopping within the test's `2^(-30)` (this paper's count: give the C++ stopping time).
  `<proc-find-first-hit>` "Turning solid": "advance by the larger of `g / bound` and the positive root of `g + gdot
  s - K s^2 / 2` when the distance is convex (every turning shape here), by `g / bound` otherwise"; the m4 order
  ("solids that do not turn first; then turning solids, each up to the earliest hit so far"); the flat solid's and the
  plate's spin in "Solid that does not turn". Glossary rows: `second-order advance`, `gap rate` (`gdot`), `gap curvature bound`
  (`K`), `HasConvexDistance`; the `stepping search, advance` row says "the larger of the first-order and the
  second-order advance"; the `turning solid` row adds the flat solid's and the plate's spin. Never say: `Newton step`, `certified`.
- [ ] **Step 6: Run** build, tests, gate (`uv run python tools/check_commit.py`); record the generated sets' counts and
  run time in the report.
- [ ] **Step 7: Commit.** Subject: `Advance toward a convex turning solid by a second-order bound`. New names:
  `HasConvexDistance` (private; listed because 1b extends it).

### Task 10: `MoveParticle` with hits on solids that do not turn

**As built (2026-11-15), where this task's text below differs:**
- Commit order: the commits are 9, 10, 9b (this task was implemented before Task 9b). The module-end fold will
  reorder or keep that order, to be decided there. Later tasks: 11, 12, 13, 12a, 14 to 18.
- The hit loop lives in `MoveParticle` itself (it calls the public `FindFirstHit` family and the layout forbids
  forward declarations); every other step is a private function of `solid_contact_steps.h`: `TouchedSurface`,
  `TouchedSurfaces`, `ComputeTouchedSolid`, `FindFirstHitOfSweep`, `ParticleMove`, `CheckSolidContactInputs` (step 1,
  with the plate-side validation: `{0, side != kNoPlate}` is `kBadInput`), `CheckStartOutsideSolids` (the placeholder
  step 2), `ApplyHit`, `ApplyResponseAtSurface`, `ApplyReactionChange`, `ToSideAtHit`, `MoveToRestGap(inputs,
  solid_index, pose, point)`, `ApplySettleAtEnd`, `CheckEndOutsideSolids`. `MoveToRestGap` exists since this task:
  Task 11 reuses it. Test-only: `ToSolidContactInputs` and `FindMoveParticleCase` beside `BuildMoveParticleCases()`.
- Touched surfaces (P2 amended, ruling 18 and I3): every re-sweep passes the list of touched surfaces (solid index,
  normal at the hit, allowance) to `FindFirstHitOfSweep`; a touched solid is hit at 0 only when it is approached
  beyond the largest allowance of all entries. `TiltedUprightPushesBackToFloor` hits the floor again (`hit_count` 3).
- P6 amended (ruling 19): a close call from `FindFirstHit` names its solid in `first_hit->solid_index`; `MoveParticle`
  maps it to `solid_id`.
- P5 amended (minor M3, to confirm with the owner): an exactly zero change of `for_next_step` neither claims a
  solid's reaction record nor conflicts with it. Tests `ReactingSolidWithZeroChange_LeavesReactionEmpty`,
  `SecondReactingSolidWithZeroChange_DoesNotConflict`.
- A hit on a turning solid is re-swept and ends in `kTooManyHits` until Task 13 (the docstring says so).
- `MoveParticle`'s `Requires:` states `(sum of the absolute axes of for_this_step and of the solid velocity) * time
  left <= 2^42 tol` at every hit; it is asserted in debug builds only (see Task 15).
- The settle asserts `d >= -2 tol` at the end pose (see Task 13).
- Tests already present from Task 11's list, with its names: `StartsInsideSecondSolid_FailureNamesItsId` and
  `CeilingClosingFasterThanParticleLeaves_IsEndsTooCloseToSolid`; `PlateSideOfUnknownPlate_IsBadInput` exists as
  `PlateSideNamingNoSolid_IsBadInput`. 58 tests in `move_particle_test.cc` after review round 2.
- The analytic `TiltedUprightOverRisingFloor_MovingTouchedFloorIsHitAgain` succeeds with 3 hits; its scene
  `TiltedUprightOverRisingFloor` stays out of the all-scenes generated set (see Task 12).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/contact/move_particle_test.cc`
- Modify: `native/fluidity/contact/solid_contact.h` (`MoveParticle`, public), `solid_contact_steps.h` (new private steps), `contact_test_cases.h`

**Room for module 1b (mesh solids) (A80):** a mesh can be hit several times in one sweep, so the surface list of
the slide stores, per entry, the solid index and the normal recorded at the hit (it has `kMaxHitsPerStep`
entries), and nothing assumes a solid appears once. "A convex solid is hit once per sweep" stays a consequence of
the touching rule; no line of code skips "the solid just hit". Step 6's move to `d = tol` is Task 11's
`MoveToRestGap`: call it, write no second copy.

**Interfaces:**
- Consumes: Tasks 6 to 9. Preconditions to respect: `FindFirstHit` requires `duration <= step_duration - start_elapsed` (asserted); the start rule and the touching rule use the solid's pose at `start_elapsed`; a path that meets a grown edge exactly is a hit; the domain box's touching rule is applied per closed side.
- Produces: `MoveParticle` (design 3.4) with steps 1, 3, 4, 5 (re-sweep with the time left, no earlier-surface
  rule yet) and 6; step 2 only returns `kStartsInsideSolid` for `d < 0` and treats `0 <= d <= tol` as free
  (Task 11 replaces it). **Amended 2026-11-15 (Task 8a as built):** `MoveParticle` passes `duration = step_duration - elapsed`, computed fresh before every re-sweep (never a running time-left reduced by each hit time: rounding would trip `FindFirstHit`'s assertion), and stops the hit loop when that is `<= 0`. Docstring as the naming review
  6.3 with the design's example and P1's numbers. Test-only: `struct MoveParticleCase { std::string_view name;
  std::vector<Solid> solids; SolidContactInputs inputs; }` and `std::vector<MoveParticleCase>
  BuildMoveParticleCases()` in `contact_test_cases.h`; every `MoveParticle` test from here on adds its inputs
  there, so Tasks 14 and 16 reuse them.
- Port: start rules and the free path of `main:native/include/fluidity/base/liquid_wall_contact_math_3d.h:365-554`
  (`ResolveContact`) without `ProjectionBounds`.

- [ ] **Step 1: Failing tests** (`T` the step, both velocities equal unless stated):

  ```cpp
  TEST(MoveParticle, NoSolid_EndIsStartPlusVelocityTimesStep)  // EXPECT_EQ per axis to start + v * T
  TEST(MoveParticle, DocstringExample_StopsOneToleranceShort)  // face x = 1 as in Task 8, T 0.2: hit_fraction
                                                               // (0.1 - tol) / 0.2, x 1 - tol, speed 0,
                                                               // approach_speed {-1,-1}, kind kHit, kHighZ
  TEST(MoveParticle, PlateRowsOnEveryAxisAndNormal_MatchClosedForms)  // below
  TEST(MoveParticle, FrictionOnSideBottom_LeavesSideways2Point5)  // domain {0,0,0}..{1,1,1}, start {0.2,0.5,0.1},
                                                               // v {3,0,-1}, friction 0.5, T 0.2: velocity
                                                               // {2.5,0,0}, z tol, x 0.2 + 3 t + 2.5 (0.2 - t),
                                                               // t = 0.1 - tol, face kLowZ
  TEST(MoveParticle, DomainCorner_ThreeSidesHitInOneStep)      // same box, start {0.1,0.2,0.3}, v {-1,-1,-1}, T 0.5:
                                                               // end {tol,tol,tol}, velocities 0, hit_count 3,
                                                               // face kLowX, hit_fraction (0.1 - tol) / 0.5
  TEST(MoveParticle, OctagonRoom_FifthHitIsTooManyHits)        // below
  TEST(MoveParticle, ReactingSolid_ReactionIsChangeOfNextStepVelocity)  // docstring example, reports_reaction:
                                                               // reaction.solid_index 0, change == after - before
                                                               // bit for bit, hit_position x 1 - tol
  TEST(MoveParticle, TwoReactingSolids_IsSecondReactingSolid)  // floor and wall x = 1 (normal {-1,0,0}) both
                                                               // reacting, start {0.8,0,0.1}, v {2,0,-2}, T 0.2
  TEST(MoveParticle, DifferentVelocities_EachGetsItsOwnApproach)  // for_this_step {0,0,-1}, for_next_step {1,0,-3},
                                                               // floor, friction 0.5: next {0,0,0} (stick at 1/3),
                                                               // this {0,0,0}; approach_speed {-3,-1}
  TEST(MoveParticle, NonFiniteStart_IsBadInput)                // and step 0, friction -1, contact_tolerance 0,
  TEST(MoveParticle, SolidsOutOfIdOrder_IsBadInput)            // ids {2,1}; result unchanged in each
  ```
  Plate rows (old fixture `main:reference/src/fluidity_reference/liquid_boundary_3d_fixtures.py:40-80`, as closed
  forms; plate half thickness `1/32`, `T = 1`, run for each axis `e` and normal `sigma * e`, `sigma = +-1`):
  row 0: start `-e`, velocity `2e`, plate fixed at 0: `hit_fraction (1 - 1/32 - tol) / 2`, end
  `-(1/32 + tol) e`, velocities 0, normal `-e`, `approach_speed` -2, `side_at_hit` `kAgainstNormal` for
  `sigma = +1`, `kAlongNormal` for -1. Row 1: particle at rest at 0, plate centre from `-e` moving `2e`: same
  fraction, end `(1 + 1/32 + tol) e`, velocities `2e`, normal `+e`. Row 2: particle from 0 at `e`, plate centre
  from `-e` at `e`: free, end `e`, kind `kNone`.
  Octagon (friction 0): eight flat solids `k = 0..7`, ids `k + 1`, `point_on_surface =
  (cos k pi/4, sin k pi/4, 0)`, `normal_out_of_solid = -(cos k pi/4, sin k pi/4, 0)`; start `{0,0,0}`,
  velocity `{100,20,0}`, `T = 1`: the particle slides along walls 0, 1, 2, 3 and the fifth hit (wall 4) returns
  `kTooCloseToCall/kTooManyHits` before `t = 0.3`.
- [ ] **Step 2: Run.** Expected: the new tests fail.
- [ ] **Step 3: Implement** `<proc-move-particle>` steps 1, 3, 4, 5 (loop), 6 with P5 and P6; `// Step N:`
  comments follow the paper. Without a hit the end is one multiply and one add per axis.
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Move one particle for one step and stop it at solid surfaces`.
  New names: `MoveParticle`.

### Task 11: `MoveParticle` start rules

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact.h` (`MoveParticle`, public), `solid_contact_steps.h` (new private steps), `move_particle_test.cc`,
  `contact_test_cases.h`

**Interfaces:**
- Consumes: Task 10. Preconditions to respect: `FindFirstHit` requires `duration <= step_duration - start_elapsed` (asserted); the start rule and the touching rule use the solid's pose at `start_elapsed`; a path that meets a grown edge exactly is a hit; the domain box's touching rule is applied per closed side.
- Produces: step 2 of design 5.5 in full, with P3, P4, P6 and P12.
- **Room for module 1b (mesh solids) (A80):** "move the point to `d = tol`" is one private
  `MoveToRestGap(inputs, solid_index, pose, point)` in `solid_contact_steps.h`, used by this task's step 2 and by
  step 6 of `MoveParticle` (Task 10's settle calls it too). For the six shapes it is one move along the normal;
  1b adds the mesh case inside it.
- **Amended 2026-11-15 (Task 10 as built):** `MoveToRestGap(inputs, solid_index, pose, point)` already exists
  (`solid_contact_steps.h`, Task 10's settle calls it): reuse it, write no second copy. It moves unconditionally; the
  caller tests `d < tol`. Two tests of the list below already exist and are not added again:
  `StartsInsideSecondSolid_FailureNamesItsId` and `CeilingClosingFasterThanParticleLeaves_IsEndsTooCloseToSolid`
  (`PlateSideOfUnknownPlate_IsBadInput` exists as `PlateSideNamingNoSolid_IsBadInput`); plate-side validation is
  already in step 1 (`{0, side != kNoPlate}` is `kBadInput`). Required new test: one that fails if the settle call of
  `MoveParticle` (step 6) is removed. Mutant M47 (the settle call removed) survives after Task 10, because with the
  touched-surface rule a touched solid ends at least `tol - tol / 16` away and no Task 10 path needs the settle; this
  task's start rules (the start moved to `d = tol`) make the settle reachable again, so write the test on a path
  whose end is closer than `tol` to a touched surface and check the end position against `d = tol`.

- [ ] **Step 1: Failing tests** (fixed floor at `z = 0` unless stated):

  ```cpp
  TEST(MoveParticle, FlatSolidPushesRestingParticle_CarriesIt)  // floor moving {0,0,2}, start {0,0,tol}, at rest,
                                                                // T 0.1: kind kHit, hit_fraction 0, velocities
                                                                // {0,0,2}, end z 0.2 + tol, approach_speed {-2,-2}
  TEST(MoveParticle, TouchingAndLeaving_IsFree)                 // start {0,0,tol}, v {0,0,1}: kind kNone
  TEST(MoveParticle, ShallowOverlap_StartedInsideSolidAndMovedOut)  // start z -0.0005, overlap tolerance 0.001:
                                                                // kStartedInsideSolid, start moved to z tol
  TEST(MoveParticle, DeepOverlap_IsStartsInsideSolid)           // z -0.01, overlap tolerance 0.001: magnitude 0.01
  TEST(MoveParticle, StartsInsideSecondSolid_FailureNamesItsId) // two solids ids 7 and 9, inside id 9: solid_id 9
  TEST(MoveParticle, CommittedSideInsideThickness_PushedToCommittedSide)  // plate h 0.01, start p.z -0.001,
                                                                // committed kAlongNormal, overlap 0.05: end p.z
                                                                // 0.01 + tol, face kHighZ, kStartedInsideSolid;
                                                                // uncommitted: -(0.01 + tol), kLowZ
  TEST(MoveParticle, CommittedBeyondOtherFace_IsOnWrongSideOfPlate)  // p.z -0.02: magnitude 0.01 (P6)
  TEST(MoveParticle, PlateSideOfUnknownPlate_IsBadInput)        // plate_id 99 with no such solid
  TEST(MoveParticle, StartsWithinToleranceMovingAway_IsMovedOutToOneTolerance)  // P12: start {0,0,tol / 4},
                                                                // both velocities {1,0,2}, T 0.1: no error, kind
                                                                // kNone, hit_count 0, velocities unchanged bit for
                                                                // bit, end {0.1, 0, tol + 0.2} (x exact, z within
                                                                // 2 tol)
  TEST(MoveParticle, CeilingClosingFasterThanParticleLeaves_IsEndsTooCloseToSolid)
      // floor id 1 fixed at z 0; ceiling id 2, point {0,0,3 tol}, normal {0,0,-1}, linear_velocity
      // {0,0,-1.75 tol}; particle at rest at {0,0,tol}; T 1, friction 0. The ceiling hits at t 4/7 and sets the
      // velocity to {0,0,-1.75 tol}; over the 3/7 left the floor's rate * time_left is -0.75 tol, above -tol, so
      // the floor is not hit (P2) and the particle ends at z 0.25 tol with the ceiling at 1.25 tol:
      // kTooCloseToCall/kEndsTooCloseToSolid, solid_id 1, magnitude 0.25 tol within 0x1p-40 * tol
  ```
- [ ] **Step 2: Run.** Expected: the new tests fail.
- [ ] **Step 3: Implement.** Every error branch says in its comment why the case can occur and what the caller
  does (the step turns `kTooCloseToCall` into a shorter step; the others are scene errors).
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Classify a particle's start against every solid`.

### Task 12: `MoveParticle` slide rules between surfaces

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact.h` (`MoveParticle`, public), `solid_contact_steps.h` (new private steps), `move_particle_test.cc`,
  `contact_test_cases.h`, `docs/glossary.md`

**Interfaces:**
- Consumes: Task 11; `ComputeLargestApproachLeftAfterResponse` (Task 7, `solid_contact_types.h`). Preconditions to respect: `FindFirstHit` requires `duration <= step_duration - start_elapsed` (asserted); the start rule and the touching rule use the solid's pose at `start_elapsed`; a path that meets a grown edge exactly is a hit; the domain box's touching rule is applied per closed side. The rule "approaches again" is
  `approach >= -bound` means no longer approaching, never `approach >= 0`: `bound` is the function's result for the
  velocity passed to that response (or to that frictionless removal), and the approach is computed as
  `Dot(normal, result - solid_velocity)`.
- Produces: design 5.5 step 5's earlier-surface rule (removal without friction, the sharp-corner stop,
  `kSqueezedBetweenSolids`), with P5's reaction for each change it makes (amended 2026-11-15: a removal is charged
  to the earlier surface's solid, the corner case to the newest solid). **Amended 2026-11-15 (Task 8a as built):**
  every re-sweep keeps Task 10's rule: `MoveParticle` passes `duration = step_duration - elapsed`, computed fresh before every re-sweep (never a running time-left reduced by each hit time: rounding would trip `FindFirstHit`'s assertion), and stops the hit loop when that is `<= 0`.
- **Amended 2026-11-15 (Task 10 as built; reads the touched-surface list `TouchedSurface`, P2 amended):**
  - "Still approaching" at an earlier surface is tested against the largest allowance of all entries so far, which is
    what `ComputeTouchedSolid(list, solid).approach_allowance` returns, not that entry's own bound: the velocity
    comes from the latest response, wherever it was, and its rounding lies along every normal. Each frictionless
    removal adds its own bound, `ComputeLargestApproachLeftAfterResponse(velocity passed to the removal, solid
    velocity)`, under the same assertion (`allowance * time left <= tol / 16`), by updating the entry it acted on or
    by adding to a running largest; either keeps P2's proof. The charged solid is `entry.solid_index`, through
    `ApplyReactionChange`; the newest entry is tested last against its own allowance (the corner case).
  - True squeezes found by the generated tests end in `kTooManyHits` today and must become
    `kSqueezedBetweenSolids`. The rising-floor scene is `TiltedUprightOverRisingFloor` (`contact_test_cases.h`): its
    generated paths ping-pong into `kTooManyHits` today, so it stays out of the all-scenes generated set until this
    task; its analytic test `TiltedUprightOverRisingFloor_MovingTouchedFloorIsHitAgain` succeeds with 3 hits and
    must keep doing so. Then add the scene to the generated set.
  - The settle can hide a wrong hit by moving a particle back to the rest gap from inside a touched surface (Task 10
    found it: mutant M19, a wrong hit position, was invisible in the end position). The tests of this task therefore
    check the reaction's position (`hit_position`) and the hit time (`hit_fraction`), not only the end position. The
    one-pass settle can also push a particle back toward the first of two acute touched surfaces, and the end check
    then fails, safely; `AcuteWedge` will meet it.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(MoveParticle, AcuteWedge_StopsWithoutError)      // floor, and flat solid point {1,0,0} normal
                                                        // Normalized({-1,0,-1}); start {0.5,0,tol}, v {2,0,0},
                                                        // T 0.5, friction 0: no error, velocities {0,0,0},
                                                        // hit_count 2, end d >= tol / 2 for both solids
  TEST(MoveParticle, ClosingCeiling_IsSqueezedBetweenSolids)  // floor fixed; ceiling point {0,0,0.1} normal
                                                        // {0,0,-1} moving {0,0,-1}; at rest at z 0.05, T 0.1
  TEST(MoveParticle, RightAngleCorner_SlidesAlongSecondSurface)  // floor and wall x = 1: start {0.5,0,0.2},
                                                        // v {2,1,-4}, T 0.5: end on the corner line x 1 - tol,
                                                        // z tol, velocity {0,1,0}, hit_count 2
  // Amended 2026-11-15 (paper review, P5): the slide's changes count in the reaction
  TEST(MoveParticle, SlideRemovalOnReactingFloor_CountsInReaction)
      // floor id 1 at z 0, reports_reaction, friction_scale 0; wall id 2 point {1,0,0} normal {-1,0,0}, linear
      // velocity {0,0,-1} (moving along its own surface), not reacting, friction_scale 1; friction 0.5; start
      // {0.5,0,0.2}, both velocities {1,0,-2}, T 1. Floor hit at 0.1 - tol/2 at {0.6 - tol/2, 0, tol}: velocity
      // {1,0,0}, floor change {0,0,2}. Wall hit at 0.5 - tol: approach -1, sideways speed 1 against the wall's
      // {0,0,-1}, slip leaves {0,0,-0.5}. The slide removes the floor approach -0.5: {0,0,0}, a change of {0,0,0.5}
      // charged to the floor; the wall is not approached again. Expected: no error, hit_count 2, velocities
      // {0,0,0}, end {1 - tol, 0, tol} within 2 tol, reaction.solid_index 0, velocity_for_next_step_change
      // {0,0,2.5} exactly (every value is a binary fraction; without the slide's removal it is {0,0,2}),
      // hit_position {0.6 - tol/2, 0, tol} within 2 tol
  TEST(MoveParticle, AcuteWedgeOnReactingFloor_CornerChargedToNewestSolid)
      // AcuteWedge's scene with the floor (id 1, index 0) reacting and the wedge (id 2) not. The particle touches
      // the floor (not hit) and hits the wedge first at x 1 - (1 + Sqrt(2)) tol: {1,0,-1} (not counted). The
      // re-sweep hits the floor at 0: {1,0,0}, floor change {0,0,1}. The slide removes the wedge approach
      // -Sqrt(0.5): {0.5,0,-0.5} (charged to the wedge, not counted); the floor, the newest, is approached again,
      // so both velocities become its velocity {0,0,0}: a change of {-0.5,0,0.5} charged to the floor. Expected:
      // hit_count 2, velocities {0,0,0}, reaction.solid_index 0, velocity_for_next_step_change {-0.5,0,1.5}
      // within 1e-15 (it is {0,0,1} if the corner case is left out or charged to the wedge, {-1,0,1} if the whole
      // slide is charged to the floor), hit_position {1 - (1 + Sqrt(2)) tol, 0, tol} within 2 tol
  // Rounding residue of the response (design 5.4): the test fails if the comparison is with 0
  TEST(MoveParticle, ObliqueCornerResidue_SlidesOnWithoutSqueezeOrStop)
      // two fixed flat solids, friction 0: plate A point {0,0,0} normal {0.6,0,0.8}; wall B point {-0.2,0,0}
      // normal {1,0,0}; start {0.3,0,0.4}, both velocities {-2,0,-2}, T 1. A's first response of that velocity
      // leaves an approach of exactly -2^-55 (pinned in contact_response_test.cc), below 0 but within the bound
      // 4.0e-14. The particle slides along A at {-0.32,0,0.24}, hits B, and must slide on along the corner line.
      // Expected: no error (not kSqueezedBetweenSolids), hit_count 2, velocities {0,0,0.24} within 1e-15 (not
      // {0,0,0}), end x -0.2 + tol within 2 tol, end z 0.24 within 1e-3 (a stop at the first hit would end near
      // z 0.043). Fails if "approaches again" tests `approach >= 0` instead of `approach >= -bound`
  ```
- [ ] **Step 2: Run.** Expected: the first two fail (`ClosingCeiling` passes through the floor or loops), and the
  two reaction tests fail until step 5 adds its changes to the reaction.
- [ ] **Step 3: Implement.** Glossary row: `slide` (the rest of the step spent moving along a surface).
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Slide along several surfaces and refuse a squeeze`.

**Amended 2026-11-15 (A74):** the earlier-surface rule of Task 12 applies to every shape unchanged (it reads the
normal recorded at each hit). Round solids get their own move rules in Task 12a.

### Task 12a: `MoveParticle` on a sphere and a cylinder (A74)

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact.h` (`MoveParticle`, public), `solid_contact_steps.h` (new private steps), `move_particle_test.cc`,
  `contact_test_cases.h`

**Interfaces:**
- Consumes: Tasks 8a, 11, 12; preconditions to respect: `FindFirstHit` requires `duration <= step_duration - start_elapsed` (asserted); the start rule and the touching rule use the solid's pose at `start_elapsed`; a path that meets a grown edge exactly is a hit; the domain box's touching rule is applied per closed side. (Task 12's rule follows, including its rule that "approaches again" is `approach >= -bound` means no longer
  approaching, never `approach >= 0`, with `bound` from `ComputeLargestApproachLeftAfterResponse`); `ComputeShapeTurningAngularVelocity` (Task 3) for P17 in step 5, no second helper; and (amended 2026-11-15) Task 13 for the test
  `TurningCylinderRimHit_IsNotMoved`: it needs the slide on a turning solid to finish the step, so **run Task 12a
  after Task 13**. **Amended 2026-11-15 (Task 9b):** and Task 9b (P18). The task order is 9, 9b, 10, 11, 12, 13,
  12a, 14 to 18. What changes here: a stepping-search hit is still in Task 9's band and still never moved, but a
  second-order advance usually stops near `d = tol`, so the old path of `TurningCylinderRimHit_IsNotMoved` (start
  `{1,-1,1}`, velocity `{-1,2,-1}`) now stops at 1.0000001 tol after 5 advances, where a rim move could not be seen; the
  test gets the path below, which stops at 1.11 tol under P18 (1.20 tol under first order).
- Produces: `MoveParticle` step 4's rim move (P15: a hit position that the cylinder's closed form (Task 8a) finds in
  the rim region is moved along its normal to `d = tol` before the response; the reaction's `hit_position` is the
  moved position). **Amended 2026-11-15 (paper review):** the move is decided by where the hit came from, not by
  its face: a hit of the stepping search toward a turning cylinder (Task 9, P17) is never moved, even when
  `ComputeDistanceToSolid` reports `kLowRim` or `kHighRim` there (that search stops at `d <= 1.25 tol` in any
  region). The helper that finds the hit tells `MoveParticle` which path produced it (a private flag in
  `solid_contact_steps.h`, name decided here; `FirstHit` is unchanged). Also the use of P17 in step 5
  (a sphere, or a cylinder spinning about its axis, slides as a solid that does not turn; the response uses
  `ComputeSolidVelocity` with the full `omega`). Every case is added to `BuildMoveParticleCases()`.

- [ ] **Step 1: Failing tests** (`tol = kTestContactTolerance`, friction 0 unless stated, both velocities equal):

  ```cpp
  TEST(MoveParticle, SphereHeadOn_StopsOneToleranceShort)      // sphere origin r 0.5, {-2,0,0}, v {1,0,0}, T 2:
                                                               // kHit, x -(0.5 + tol) within 2 tol, velocities 0,
                                                               // approach_speed {-1,-1}, kCurved
  TEST(MoveParticle, SphereOblique_SlidesOffAlongTheTangent)   // {-2,0.3,0}, v {1,0,0}, T 2: hit_count 1 (the
                                                               // tangent re-sweep is not a hit), velocity
                                                               // {0.36,0.48,0} and end {-0.256,0.492,0} within 1e-8,
                                                               // hit_fraction (2 - Sqrt((0.5 + tol)^2 - 0.09)) / 2
  TEST(MoveParticle, SphereTangentWithinRounding_NeverEndsInside)  // {-2, 0.5 + tol, 0}, v {1,0,0}, T 4: no error,
                                                               // end distance >= tol / 2, velocity x within 0x1p-40
                                                               // relative of 1
  TEST(MoveParticle, SphereStartWithinToleranceMovingAway_MovedOutSilently)  // P12: start {-(0.5 + tol/4),0,0},
                                                               // v {0,1,0}, T 0.1: kNone, hit_count 0, velocities bit
                                                               // for bit, end distance >= tol
  TEST(MoveParticle, CylinderRim_HitPositionMovedToOneTolerance)  // cylinder origin axis z r 0.5 h 1, reacting;
                                                               // {0.5 + tol/2,0,3}, v {0,0,-1}, T 4: kHit, kHighRim,
                                                               // hit_count 1, ComputeDistanceToSolid(reaction
                                                               // .hit_position) within 0x1p-40 of tol, velocity
                                                               // {0.4,0,-0.2} within 1e-5 (see the note below)
  TEST(MoveParticle, CylinderCap_SlidesOnThePlane)             // {0.1,0,3}, v {0.1,0,-1}, T 3: hit_fraction
                                                               // (2 - tol) / 3, kHighZ, velocity {0.1,0,0}, end
                                                               // {0.4,0,1 + tol} within 2 tol, hit_count 1
  TEST(MoveParticle, SpinningCylinderPressedOnCap_MovesOnlyThroughFriction)
      // axis z, r 1, h 0.5, angular {0,0,2}; start {0.5,0,0.5 + tol}, v {0,0,-1}, T 0.1: friction 0 -> velocity
      // {0,0,0}, end {0.5,0,0.5 + tol}; friction 0.25 -> {0,0.25,0}, end {0.5,0.025,0.5 + tol}; friction 1 ->
      // {0,1,0} (stick), end {0.5,0.1,0.5 + tol}; positions within 2 tol; no close call (P17: spin, not a turn)
  // Amended 2026-11-15 (Task 9b, P17 amended): a plate spinning about its normal slides as a solid that does not turn
  TEST(MoveParticle, SpinningPlatePressedOn_MovesOnlyThroughFriction)
      // plate at the origin, normal {0,0,1}, half thickness 0.5, angular {0,0,2}; the same start, velocity, T and
      // frictions as the cylinder above, the same expected velocities and ends (the surface moves at {0,1,0} there)
  // Amended 2026-11-15 (paper review, P15): the rim move is for closed-form hits only
  TEST(MoveParticle, TurningCylinderRimHit_IsNotMoved)
      // cylinder at origin, axis z, r 0.5, h 0.5, angular {1e-6,0,0} (across the axis: |omega_across| T
      // |half_size| = 8.7e-7 > tol / 8, so P17 makes it turning; the slow turn keeps the corner within 1e-13 of
      // its start), reacting; amended 2026-11-15 (Task 9b): both velocities 1.5 * Normalized({-cos 0.05, 1,
      // -sin 0.05}) and start {0.5 + tol / 2, 0, 0.5 + tol / 2} - 0.5 * velocity (aimed just beyond the corner at
      // t 0.5, obliquely, so the search stops away from d = tol), T 1, friction 0.
      // First, FindFirstHit with the same inputs: elapsed about 0.49999999958 (4 advances under P18, mirror),
      // at_hit.face kHighRim, at_hit.distance about 1.110 tol; ASSERT face is a rim and
      // at_hit.distance > tol + tol / 16 (the case's premise). Then MoveParticle: no error, face kHighRim,
      // reaction.hit_position within tol / 1024 of start + velocity * elapsed (on the path: not moved), and
      // ComputeDistanceToSolid(cylinder, ComputeSolidPose(cylinder, elapsed), reaction.hit_position).distance
      // within tol / 1024 of at_hit.distance (a rim move would give tol, 0.2 tol away)
  ```
  **Amended 2026-11-15 (Task 8a as built):** an expected normal (and a velocity or position computed from it) at a
  hit within about the tolerance of an edge or rim is tested to 1e-5, not 1e-8: 1e-15 m of rounding at a distance
  of the tolerance turns the normal by about 1e-6 (measured worst 2.07e-6, tilted cylinder's rim hits).
- [ ] **Step 2: Run.** Expected: the new tests fail.
- [ ] **Step 3: Implement.** The rim move goes between "find the first hit" and "respond", applied only to a hit
  of the closed form (P15), with a comment on why the sharp grown corner leaves that hit up to `Sqrt(2) * tol` away
  and why a stepping-search hit is left where the search stopped; the slide comment says why the straight tangent path
  cannot re-enter a convex solid in this step (P16's convexity line) and so `kMaxHitsPerStep` is not reached.
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Move a particle against a sphere and a cylinder`.

### Task 13: `MoveParticle` slide on a turning solid

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact.h` (`MoveParticle`, public), `solid_contact_steps.h` (new private steps), `move_particle_test.cc`,
  `contact_test_cases.h`

**Interfaces:**
- Consumes: Tasks 9, 9b, 12 (amended 2026-11-15, Task 9b: the search behind every turning hit now uses P18, so far
  fewer sweeps end in `kSlowApproachToTurningSolid`; hit times and faces of this task's tests are unchanged within
  their tolerances, and `TurningBoxSlide_KeepsSidewaysPartInSolidAxes` and
  `RestingOnTurningBoxNearFloor_StepLengthDecidesCloseCall` start at `d == tol`, a hit at 0 with no advance, so they do
  not depend on the advance at all. The second is P7's near-other-solid close call, which Task 9b does not touch: its
  numbers stand. A flat solid or a plate spinning about its normal no longer turns (P17 amended): it slides as a solid
  that does not turn, with the full `omega` in the response); preconditions to respect: `FindFirstHit` requires `duration <= step_duration - start_elapsed` (asserted); the start rule and the touching rule use the solid's pose at `start_elapsed`; a path that meets a grown edge exactly is a hit; the domain box's touching rule is applied per closed side. `ComputeShapeTurningAngularVelocity` (Task 3) for P17's choice of slide, no second helper (amended 2026-11-15: a cylinder counts as turning only when `Abs(omega_across) * step_duration * Length(half_size) > tol / 8`, the step's duration, not the sweep's; below the bound it uses the closed form);
  `ComputeLargestApproachLeftAfterResponse` (Task 7, `solid_contact_types.h`), through Task 12's rule: "approaches again" is
  `approach >= -bound` means no longer approaching, never `approach >= 0`.
- **Amended 2026-11-15 (Task 10 as built):**
  - The settle asserts `d >= -2 tol` at the end pose (`ApplySettleAtEnd`): the turning carry must revisit that
    assertion and keep it or change it with its reason in the body comment.
  - A hit on a turning solid is re-swept in Task 10 and ends in `kTooManyHits` (touching rule absent, hit at 0 again
    each sweep). This task removes that behaviour and the sentence in `MoveParticle`'s docstring that says so.
  - The close call left in the turning search is a particle hovering near or grazing a turning face
    (`TurningBoxHoverOnSpinAxis`). Just above `kNegligibleTurnShare` a fast-spinning flat floor makes slow sliders
    close calls: a known cost, deferred.
- Produces: design 5.5 step 5 for a turning solid, with P7. **Room for module 1b (mesh solids) (A80):** the carry
  in the solid's axes is one private function of `solid_contact_steps.h` taking `inputs` and `solid_index`, so 1b
  can sweep that remainder against a mesh.
- Port: `main:native/src/liquid_wall_contact_3d.cc:349-400` (from the comment "Carry the corrected relative
  remainder in the moving face coordinates"), without the rounding bound.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(MoveParticle, TurningBoxHitsRestingParticle_EndsOnFaceAtEndPose)
      // Task 9's case, T 1, friction 0: hit_fraction 0.3 + acos((0.5 + tol) / 0.6) within 0x1p-30;
      // ComputeDistanceToSolid at the end pose in [tol / 2, 2 tol], face kHighX
  TEST(MoveParticle, TurningBoxNearFloor_IsTurningSolidNearOtherSolid)
      // same box and particle (particle z 0), plus a fixed floor at z -0.05: kTooCloseToCall
  TEST(MoveParticle, TurningBoxSlide_KeepsSidewaysPartInSolidAxes)
      // particle at {0.5 + tol, 0, 0} moving {0,0,1} on the box turning {0,0,1}, T 0.1, friction 0: the face moves
      // {0,0.5,0} there, so the relative velocity in the box's axes is {0,-0.5,1}; end position ==
      // origin + axes * {0.5 + tol, -0.05, 0.1} at ComputeSolidPose(box, 0.1), within 2 tol; hit_fraction 0
      // (the particle rests at d == tol: no touching rule on a turning solid, so it is hit at 0 and carried, P2)
  // Amended 2026-11-15 (paper review, P2 and P7): a particle resting on a turning solid runs the near-other-solid
  // test over the whole step, so the step length decides
  TEST(MoveParticle, RestingOnTurningBoxNearFloor_StepLengthDecidesCloseCall)
      // box half 0.5 at origin, angular {0,0,1}; fixed floor id 2 at z -0.5 (touching the box's bottom); particle
      // at rest at {0.5 + tol, 0, -0.45} (d == tol from kHighX, 0.05 above the floor), friction 0. Hit at 0
      // (time_left = T); v_local = {0, -(0.5 + tol), 0}, r = Sqrt((0.5 + tol)^2 + 0.45^2) = 0.67268120, so P7's
      // swept distance is T * (0.5 + tol + 0.67268120) = 1.17268120 T against d_floor - tol = 0.05 - tol; it fires for
      // T >= 0.0426373. T 0.1 (swept distance 0.1172681): kTooCloseToCall/kTurningSolidNearOtherSolid. T 0.04 (swept
      // distance 0.0469072): no error, hit_fraction 0, face kHighX, end distance to the box at
      // ComputeSolidPose(box, 0.04) in [tol / 2, 2 tol], end z -0.45 within 2 tol (floor not hit)
  // Amended 2026-11-15 (A74, P17): a cylinder turning across its axis slides as a turning solid
  TEST(MoveParticle, CylinderTurningAcrossAxisHitsRestingParticle_EndsOnWallAtEndPose)
      // Task 9's cylinder case, T 1: hit_fraction 0.21488921385421111 within 0x1p-30; ComputeDistanceToSolid at
      // ComputeSolidPose(cylinder, 1) in [tol / 2, 2 tol], face kCurved
  ```
- [ ] **Step 2: Run.** Expected: the new tests fail.
- [ ] **Step 3: Implement.** Relative `for_this_step` at the hit into the solid's axes, the face component
  clamped to zero when it points into the face, advance by the time left, place with
  `ComputeSolidPose(solid, step_duration)`; then step 6.
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Carry a slide on a turning solid in its own axes`.

### Task 14: Tests that replace the rounding enclosures

**Kind:** numerics

**Files:**
- Create: `native/fluidity/contact/move_particle_tolerance_test.cc`
- Modify: `native/fluidity/contact/contact_test_cases.h`, `CMakeLists.txt`

**Interfaces:**
- Consumes: `MoveParticle`, `FindFirstHit`, `ComputeContactResponse`, `ComputeDistanceToSolid`,
  `ComputeContactTolerance`, `BuildMoveParticleCases`; Task 9b (amended 2026-11-15): every `BuildMoveParticleCases()`
  case with a turning solid runs through P18's search, so `SceneTurnedRigidly_ResultTurnsWithIt` may see different
  last bits of a turning hit time than before 9b (still within its tolerance) and no case may rely on a close call of
  the old search; the oblique and round sets have no turning solid and do not change.
- **Amended 2026-11-15 (Tasks 9b and 10 as built):** the scene `TurningBoxSlowApproach` is now
  `TurningBoxHoverOnSpinAxis`. Generated-set counts in this plan that were measured before Task 9b (close calls on
  turning solids, and the `kSlowApproachToTurningSolid` and `kTooManyHits` shares of the all-scenes generated test
  of `move_particle_test.cc`) are stale: to re-measure when this task runs, not to copy.
- Produces: test-only `std::vector<MoveParticleCase> BuildObliqueCases(int count)` (used again in Task 16).
  **Room for module 1b (mesh solids) (A80):** the check "every end distance is at least `tol / 2`, and the naive
  control enters" is one helper of `contact_test_cases.h` that takes any list of solids (the test below calls it
  for the oblique set and for the round set); 1b runs it on meshes. **Amended 2026-11-15 (Task 8a as built):** an
  expected normal at a hit within about the tolerance of an edge or rim is tested to 1e-5, not 1e-8 (1e-15 m of
  rounding at a distance of the tolerance turns the normal by about 1e-6; measured worst 2.07e-6).

Oblique set (generator P11; draw in this order). Per case: normal `{NextSigned(), NextSigned(), NextSigned()}` redrawn while `Length < 0.25`,
then `Normalized`; `point_on_surface = 1e3 * {NextSigned(), NextSigned(), NextSigned()}`; `linear_velocity = 10 * {NextSigned(), NextSigned(), NextSigned()}`; `gap = 0.01 * Next() + 2 * tol`; tangent
`Normalized(Cross(normal, {NextSigned(), NextSigned(), NextSigned()}))`, redrawn while the cross product's length is below 0.25; `sideways = 5 * NextSigned()`;
`hit_share = 0.1 + 0.8 * Next()`; `for_this_step = linear_velocity + sideways * tangent - gap / (T * hit_share) *
normal`; `for_next_step = for_this_step + 0.5 * {NextSigned(), NextSigned(), NextSigned()}`; `start = point_on_surface + gap * normal + 50 * NextSigned() *
tangent`; `friction = Next()`. `T = 0x1p-6` s, one flat solid id 1, `tol = ComputeContactTolerance({1100, 0.01,
0.01})`, `start_overlap_tolerance = 0`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(MoveParticle, ObliqueMovingSolidFarFromOrigin_NeverEndsInside)
      // 10,000 cases: no error; ComputeDistanceToSolid at ComputeSolidPose(solid, T) >= tol / 2 for each.
      // Control in the same test: FindFirstHit with contact_tolerance 0, ComputeContactResponse on
      // for_this_step, naive end = hit position + response * (T - elapsed): d < 0 in at least one case
      // (print the count).
  TEST(MoveParticle, SlidesTenThousandSteps_NeverStartsInside)
      // flat solid point {0.3,-0.2,0.1}, normal Normalized({1,2,3}), velocity {0.5,-0.25,0.125}; start
      // point + tol * normal; both velocities reset each step to the solid velocity + {0.2,0.1,-0.3} - 0.5 *
      // normal; friction 0.3; T 1/240; tol = ComputeContactTolerance({100, 0.01, 0.01}); each end is the next
      // start and the solid's pose_at_step_start advances by ComputeSolidPose(solid, T): no error and d in
      // [tol / 2, 2 tol] after every step
  TEST(MoveParticle, FastParticleThinPlate_IsStopped)
      // cell 0.01, plate half thickness 5e-5 at origin, start {0,0,5}, v {0,0,-1000}, T 0.01, tol from
      // ComputeContactScale and CheckContactTolerance (passes): kHit, hit_fraction (5 - 5e-5 - tol) / 10
  TEST(MoveParticle, SceneTurnedRigidly_ResultTurnsWithIt)
      // every BuildMoveParticleCases() case without a DomainBox (world-aligned by definition), under
      // math::Rotation({0.3,-0.5,0.7}) and under the Y-up to Z-up rows {1,0,0},{0,0,-1},{0,1,0}: positions,
      // velocities, solid origins, axes and motions turned; discrete fields equal; floating values within
      // 0x1p-40 * max(|a|, |b|, case scale), scale the largest coordinate (positions) or speed (velocities)
  // Amended 2026-11-15 (A74): the round set below
  TEST(MoveParticle, RoundSolidsFarFromOrigin_NeverEndInside)
      // BuildRoundCases(10000): no error; ComputeDistanceToSolid at ComputeSolidPose(solid, T) >= tol / 2 for each;
      // at least 4500 cases are hits (every cylinder case aims inside; print the count)
  ```
  **Amended 2026-11-15 (A74):** also produces test-only `std::vector<MoveParticleCase> BuildRoundCases(int count)`
  (used again in Task 16). Draw order (P11): `is_sphere = Next() < 0.5`; `centre = 1e3 * {NextSigned(), NextSigned(), NextSigned()}`; `radius = 0.05 +
  0.45 * Next()`; `half_height = 0.05 + 0.45 * Next()` (drawn for both, used by the cylinder); `axis = {NextSigned(), NextSigned(), NextSigned()}`
  redrawn while `Length < 0.25`, then `Normalized` (drawn for both); `linear_velocity = 10 * {NextSigned(), NextSigned(), NextSigned()}`; `direction`
  drawn as the axis was; `start = centre + (Length(half_size) + 0.01 + 0.05 * Next()) * direction`; `target = centre +
  axes * (0.9 * half_size` times `{NextSigned(), NextSigned(), NextSigned()}` component by component`)`; `hit_share = 0.1 + 0.8 * Next()`;
  `for_this_step = linear_velocity + (target - start) / (T * hit_share)`; `for_next_step = for_this_step + 0.5 *
  {NextSigned(), NextSigned(), NextSigned()}`; `friction = Next()`. One solid id 1 at rest in pose (only `linear_velocity`, no turn), `T = 0x1p-6`,
  `tol = ComputeContactTolerance({1100, 0.01, 0.01})`, `start_overlap_tolerance = 0`. The control of the oblique
  set is not repeated (the rest gap's proof is shape-free).
- [ ] **Step 2: Run.** Expected: they pass if Tasks 10 to 13 (and 12a) are right. Then prove the control: temporarily set
  the settle of step 6 to no-op and the hit at `d == 0`; `NeverEndsInside` must fail; restore. Record the failing
  count in the commit message.
- [ ] **Step 3: Run** the gate.
- [ ] **Step 4: Commit.** Subject: `Show the rest gap keeps particles outside without enclosures`.

### Task 15: `MoveParticles` on the CPU

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** private GPU header is `move_particles_gpu.h` returning `RunOnCallStatus`; the GPU call's status goes through `ToBufferError` and one mapping into `ContactFailure`; failure wording and test support adopted.

**Amended 2026-11-15 (A58; reconcile item 31):** `BuildContactBuffers(run_on, max_particle_count, max_solid_count, RunMemory& run_memory)` builds its buffers as pieces of the run's block; the `"contact: call result"` entry is split into one entry per buffer; out of memory is `kOutOfMemory` (was `kBuffersTooSmall`); new test `BuildContactBuffers, IntoRunMemory_HandsOutExactlyTheEstimate`.

**Files:**
- Create: `native/fluidity/contact/move_particles.h`, `move_particles_cpu.cc`, `move_particles_gpu.h`
  (private), `move_particles_test.cc`
- Modify: `native/fluidity/contact/CMakeLists.txt` (`move_particles_cpu.cc` in `target_sources`, the test with
  `LINKS fluidity_base_test_support`), `docs/glossary.md`

**Interfaces:**
- Consumes: `MoveParticle` (`solid_contact.h`); `RunOn`, `CheckCanRunOn`; `Buffer`, `BuildBuffer` (both forms), `WriteBuffer`,
  `ReadBuffer`, `BufferError`, `ToBufferError`, `RunOnCallStatus`; `RunMemory`, `BuildRunMemory`,
  `ComputeRunMemoryBytes` (Foundation Task 17); `MemoryEstimateEntry`, `CountedPer`, `KeptFor`,
  `ComputeMemoryBytes`; `PerParticleVelocity`; test support `ToBits`, `BuildBufferHolding`, `ReadAll`.
- Produces: design 3.5 exactly: `ParticleContact`, `ContactBuffers`, `BuildContactBuffers`, `MoveParticlesInputs`,
  `MovedParticles`, `MoveParticlesResult`, `MoveParticles`, `BuildContactMemoryEstimate`. Plan decisions:
  - `ContactBuffers` is move-only with `RunOn run_on() const`, `std::size_t max_particle_count() const`,
    `std::size_t max_solid_count() const`; it holds `Buffer<Solid>` (max solids), `Buffer<ContactFailure>`
    (max particles) and `Buffer<ContactFailure>` plus `Buffer<MoveParticlesResult>` of size 1 for the GPU reduction.
  - `BuildContactMemoryEstimate` rows, all `kGivenCount`, `kWholeRun`, `where = kRunOn`, `copies = 1`, one row per
    buffer (amended 2026-11-15, A58): `"contact: solids copy"` (max solids x `sizeof(Solid)`), `"contact: failure
    per particle"` (max particles x `sizeof(ContactFailure)`), `"contact: call result"` (1 x
    `sizeof(MoveParticlesResult)`), `"contact: first failure"` (1 x `sizeof(ContactFailure)`; was part of `"call
    result"`).
  - **NEW (2026-11-15, for sand; reconcile item 33):** `ContactBuffers` has `std::span<const Solid> solids_copy()
    const`: the solids the last `MoveParticles` call wrote into its `"contact: solids copy"` buffer, on `run_on()`
    (`solids.size()` entries; empty before the first call). Sand's hit split on the GPU reads it, so materials
    needs no per-call copy. Test `ContactBuffers, SolidsCopy_IsTheLastCallsSolids` (CPU: after a call with the
    four solids of `ThousandParticlesFourSolids`, the span has 4 entries equal byte for byte to the input).
  - `BuildContactBuffers` builds the four buffers in row order with `BuildBuffer(run_on, size, run_memory)`; on a
    failure it returns, the pieces already taken stay in the block (a piece is never freed alone) and the step stops.
  - `MoveParticles` (defined in `move_particles_cpu.cc`) checks, in order: `particle_count` or `solids.size()`
    above the buffers' sizes is `kBuffersTooSmall`; a null required pointer with `particle_count > 0`, or
    `moved.reaction` null while a solid has `reports_reaction`, is `kBadInput`. A failed check writes nothing.
    Then `kCpu` runs one loop (one thread for now: the lowest-id rule makes later chunking give the same result);
    `kGpu` calls `RunOnCallStatus MoveParticlesOneThreadPerParticleOnGpu(inputs, moved, buffers)` declared in the
    private `move_particles_gpu.h` (defined in Task 16), under `#if defined(FLUIDITY_HAS_GPU)`; without it the call
    site returns `kRunOnCallFailed`. A status other than `kDone` goes through `ToBufferError` and the one private
    mapping `ToContactFailure(BufferError)` of `move_particles_cpu.cc` (`kSizeMismatch` gives `kBuffersTooSmall`;
    `kRunOnCallFailed` gives `kRunOnCallFailed`; amended 2026-11-15, reconcile item 31: `kOutOfMemory` and
    `kRunMemoryTooSmall` give `kOutOfMemory`, which only `BuildContactBuffers` can meet), which `BuildContactBuffers`
    also uses. After
    `kRunOnCallFailed` the outputs are unspecified and the run stops.
  - **Amended 2026-11-15 (Task 10 as built):** outside debug builds nothing checks the precondition in
    `MoveParticle`'s `Requires:`, `(sum of the absolute axes of for_this_step and of the solid velocity) * time left <=
    2^42 tol` (asserted in debug only; it keeps a touched solid's allowance within `tol / 16`). `MoveParticles` relies
    on the step's travel limit for it. Required: a test at the `MoveParticles` level, or a checked failure there
    (`kBadInput` or a close call, decided here), that pins what happens when a particle's speed breaks it; the
    decision is stated in the header.
  - `ParticleContact` is copied field by field from `SolidContactResult`; `static_assert(sizeof(ParticleContact) ==
    56)` (the design says 64; its fields total 56, see the hand-over).

**Amended 2026-11-15 (A74):** `EveryCase_SameAsMoveParticle` covers the sphere and cylinder cases of Tasks 8a,
12a and 13 through `BuildMoveParticleCases()` (no new test name); `BuildContactMemoryEstimate` is unchanged
(`sizeof(Solid)` does not change: the round shapes reuse `half_size`).

**Room for module 1b (mesh solids) (A80):** `solid_shapes.h` has no assertion on `sizeof(Solid)` yet. Computed from
its fields, it is 200 bytes: `shape` at 0, `solid_id` at 4, `pose_at_step_start` (96) at 8, `motion` (48) at 104,
`half_size` (24) at 152, `closed` (6) at 176, `friction_scale` at 184, `reports_reaction` at 192, then 7 bytes of
trailing padding. Add `static_assert(sizeof(Solid) == 200, ...)` in `solid_shapes.h` after `Solid`, beside the
memory estimate's use of `sizeof(Solid)`, so that 1b's 4-byte `mesh_index` is proven to fit: it goes at offset 196,
inside that padding, and the size stays 200.

- [ ] **Step 1: Failing tests** (`move_particles_test.cc`, label `cpu`):

  ```cpp
  TEST(MoveParticles, EveryCase_SameAsMoveParticle)        // each BuildMoveParticleCases() case as a one-particle
                                                           // call: position, velocity, record bit for bit
                                                           // (ToBits); a
                                                           // failing case gives the same error and close call
  TEST(MoveParticles, CountsHitsAndStartsInside)           // three particles: free, hit, started inside: {1, 1}
  TEST(MoveParticles, TwoRanges_SameAsOneCall)             // particles of the first 100 oblique cases against the
                                                           // solid of case 0, split 0..37 and 37..100: same bytes
  TEST(MoveParticles, TwoFailuresOnCpu_ReportsLowestParticleId)  // ids 50 at index 2 and 9 at index 7 start 0.01
                                                           // deep in a floor, overlap tolerance 0: particle_id 9
  TEST(MoveParticles, MoreParticlesThanBuffers_IsBuffersTooSmall)
  TEST(MoveParticles, MoreSolidsThanBuffers_IsBuffersTooSmall)
  TEST(MoveParticles, ReactingSolidWithoutReactionArray_IsBadInput)
  TEST(BuildContactBuffers, GpuInBuildWithoutGpuCode_IsRunOnCallFailed)  // only without FLUIDITY_HAS_GPU
  TEST(BuildContactMemoryEstimate, ThousandParticlesFourSolids_MatchesBufferSizes)  // ComputeMemoryBytes ==
                                                           // 1000 * sizeof(ContactFailure) + 4 * sizeof(Solid) +
                                                           // sizeof(MoveParticlesResult) + sizeof(ContactFailure)
  TEST(BuildContactBuffers, IntoRunMemory_HandsOutExactlyTheEstimate)  // BuildRunMemory(kCpu,
      // ComputeRunMemoryBytes(BuildContactMemoryEstimate(1000, 4), {}, kRunOn)), then BuildContactBuffers(kCpu, 1000, 4,
      // run_memory): handed_out_bytes() == reserved_bytes(), exact (four pieces, each rounded up to 256 B)
  TEST(ContactBuffers, SolidsCopy_IsTheLastCallsSolids)     // NEW (sand, 2026-11-15): 4 entries, same bytes
  TEST(BuildContactBuffers, RunMemoryOnePieceShort_IsOutOfMemory)  // the same block less 256 B (GTEST_SKIP in
      // debug builds, which assert, Foundation Task 17): kOutOfMemory, magnitude == bytes the failing piece asked
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** `BuildContactBuffers` maps `BufferError::kOutOfMemory` and `kRunMemoryTooSmall` to
  `kOutOfMemory` (magnitude = bytes asked), `kSizeMismatch` to `kBuffersTooSmall`, `kRunOnCallFailed` to
  `kRunOnCallFailed` (design 3.5) through `ToContactFailure`. The header explains how the step fills
  the pointers from `particles::ParticleFields` without contact including a particles header (design 3.5, last
  paragraph, in plain words). Glossary rows: `contact record`, `reaction` (for `SolidReaction`).
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Move a range of particles on the CPU`. New names: the 3.5 list.

### Task 16: `MoveParticles` on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** two-halves pattern: GPU half `move_particles_gpu.cu` returning `RunOnCallStatus` through `base/gpu_launch.cuh`; kernel bodies moved to the shared `move_particles_steps.h`, run by a new `cpu` test over every thread block and thread; the shared-formula GPU test follows Foundation's `.cc` test plus `_test_gpu.cu` runner, with a deliberate same-bits promise for solids that do not turn; digests.

**Files:**
- Create: `native/fluidity/contact/move_particles_gpu.cu`, `move_particles_steps.h` (shared, private to
  `contact/`), `move_particles_steps_test.cc`, `move_particles_gpu_test.cc`, `move_particles_no_gpu_test.cc`,
  `shared_formulas_gpu_test.cc`, `shared_formulas_test_gpu.h`, `shared_formulas_test_gpu.cu` (test-only GPU runner,
  as `base/shared_math_test_gpu.h`)
- Modify: `native/fluidity/contact/CMakeLists.txt` (`move_particles_steps.h` in `fluidity_add_headers`;
  `fluidity_add_gpu_sources(move_particles_gpu.cu)`; `move_particles_steps_test` `LABEL cpu`;
  `move_particles_gpu_test` and `shared_formulas_gpu_test` (`SOURCES shared_formulas_gpu_test.cc
  shared_formulas_test_gpu.cu`) `LABEL gpu`; `move_particles_no_gpu_test` `LABEL no_gpu`; every test
  `LINKS fluidity_base_test_support`), `solid_shapes.h`, `solid_contact_types.h`, `contact_tolerance.h` (the same-bits
  sentence below)

**Interfaces:**
- Consumes: `move_particles_gpu.h`, `ContactBuffers` (Task 15), `BuildObliqueCases`, `BuildMoveParticleCases`,
  the shared headers; `ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`, `GpuArray`, `ComputeGridStride`;
  test support `CheckWithinTolerance`, `CheckSameBitsOrBothNaN`, `ReadAll`, `WriteResultDigest`.
- Produces: `MoveParticlesOneThreadPerParticleOnGpu` (private, returns `RunOnCallStatus`). Shared per-item
  functions in `move_particles_steps.h` (plan decisions; `inline FLUIDITY_CPU_AND_GPU`):
  - `struct MoveParticlesView` — the plain pointers and counts of `MoveParticlesInputs` and `MovedParticles` and the
    buffers' per-particle failure array, copied by the CPU half (the shared header cannot include the CPU-only
    `move_particles.h`). **Room for module 1b (mesh solids) (A80):** its scalar inputs (step duration, contact
    tolerance, start overlap tolerance, solid count) are fields of this one struct too, and
    `ApplyMoveParticleAt` takes it, never a positional scalar list, so 1b's `MeshShapes` field is one line. Leave
    about 400 B of thread stack unassumed (the tree search's 48 indexes and the surface list).
  - `void ApplyMoveParticleAt(MoveParticlesView view, std::size_t index)` — runs `MoveParticle` on particle
    `index`, writes its outputs, record and reaction, and its `ContactFailure` (`kNone` on success, `particle_id`
    filled).
  - `std::uint64_t ComputeFailureKey(ContactFailure failure)` — the failing `particle_id`, or
    `0xFFFFFFFFFFFFFFFF` for `kNone`; the minimum key names the reported failure.
  - `struct MoveParticleCounts { std::uint64_t hit_count; std::uint64_t started_inside_count; }` (the fields of
    `MoveParticlesResult`, which lives in the CPU-only header) and `MoveParticleCounts
    ComputeParticleCounts(ContactKind kind)` — the counts one particle adds (0 or 1 each).
  Kernels (each a grid-stride loop over these, launched with `ApplyGpuFunction(..., ComputeLaunchSize(count),
  ...)`): kernel 1 calls `ApplyMoveParticleAt`; kernel 2 does integer `atomicMin` of `ComputeFailureKey` and
  integer `atomicAdd` of `ComputeParticleCounts` (integer atomics: order-free, so discrete results are the same on
  every run); kernel 3 copies the failure whose key equals the minimum into the result slot. The CPU writes the
  solids with `WriteBuffer` (slice form) and reads one `MoveParticlesResult` and one `ContactFailure` with
  `ReadBuffer`; `CheckGpuWork` before the read. No floating-point value is summed. Every per-call array is a
  `GpuArray`; `ContactBuffers` already holds the per-step arrays.
- Same-bits promise (deliberate, plan decision): the headers of `solid_shapes.h`, `contact_tolerance.h` and
  `solid_contact_types.h` state that `ComputeDistanceToSolid`, `FindSolidSide`, `ComputeSolidVelocitySample`,
  `ComputeContactResponse` and `ComputeContactTolerance` give the same bits on CPU and GPU, unless NaN, when every
  angular velocity is zero, because they use only `+`, `-`, `*`, `/` and `math::Sqrt`, each correctly rounded on both with
  `--fmad=false`; with a nonzero angular velocity they call `math::Rotation` (may differ in the last bits) and agree within design 6.
  **Amended 2026-11-15 (Task 8a as built, controller ruling 15):** a spinning sphere or cylinder is posed with its
  spin through `ComputeSolidPose`, which uses `Sin`, so spinning round solids are compared within the design 6
  tolerance, not by bits; revisit only if the hardware run needs exact parity.
- **Amended 2026-11-15 (A74):** the promise covers the sphere and cylinder branches too (design 6): distance and
  normal for any pose, `FindFirstHit`'s closed forms, the rim move and the touching rule use only `+ - * /`,
  `math::Abs` and `math::Sqrt`; no last-bits function. A spinning sphere or cylinder (`angular_velocity` not zero)
  reaches `math::Rotation` through `ComputeSolidPose` and is compared within design 6. The 32 fixed inputs of
  `SharedFormulas, OnGpu_MatchCpu` include the twelve distance points of Task 4a (sphere, cylinder regions, both
  degenerate normals) and the sphere and cylinder hits of Task 8a; `ApplyMoveParticleAt,
  EveryThreadBlockAndThread_SameAsCpuLoop` and `MoveParticles, ObliqueCasesOnGpu_MatchCpu` also run
  `BuildRoundCases(10000)` (one call per case on the GPU; one packed call per shape kind is not possible since each
  case has its own solid).

- [ ] **Step 1: Failing tests.**

  ```cpp
  // move_particles_steps_test.cc, label cpu: the main evidence until the hardware run
  TEST(ApplyMoveParticleAt, EveryThreadBlockAndThread_SameAsCpuLoop)
      // BuildObliqueCases(10000) packed as one call of 10,000 particles against the solid of case 0, plus the
      // BuildMoveParticleCases() cases as one-particle calls: for every thread block and thread of
      // ComputeLaunchSize(count), in launch order and in reverse, run ApplyMoveParticleAt over the thread's
      // grid-stride items; outputs, records and reactions equal the CPU loop's (ToBits)
  TEST(ComputeFailureKey, TwoFailuresAnyOrder_MinimumIsLowestParticleId)
      // the Task 15 two-failure case, keys reduced with a plain minimum in launch order and in reverse: 9
  TEST(ComputeParticleCounts, SummedOverThreads_EqualsCpuCounts)  // the three-particle case of Task 15: {1, 1}
  // move_particles_gpu_test.cc, label gpu (compiled here, run by the owner)
  TEST(MoveParticles, SweepAndMoveCasesOnGpu_MatchCpu)      // every BuildMoveParticleCases() case
  TEST(MoveParticles, ObliqueCasesOnGpu_MatchCpu)           // BuildObliqueCases(10000), one call per case
  TEST(MoveParticles, SameInputTwice_SameBytes)             // positions, velocities, records, reactions
  TEST(MoveParticles, TwoFailures_ReportsLowestParticleId)  // 1000 particles; ids 50 at index 5 and 9 at index 700
                                                            // (another thread block) start 0.01 deep in a floor: 9
  // shared_formulas_gpu_test.cc with the runner shared_formulas_test_gpu.cu, label gpu
  TEST(SharedFormulas, OnGpu_MatchCpu)                      // ComputeDistanceToSolid, FindSolidSide,
                                                            // ComputeSolidVelocitySample, ComputeContactResponse,
                                                            // ComputeContactTolerance on 32 fixed inputs:
                                                            // CheckSameBitsOrBothNaN when every angular velocity
                                                            // is zero (the promise above), design 6 tolerance
                                                            // otherwise (spinning round solids included)
  // move_particles_no_gpu_test.cc, label no_gpu
  TEST(BuildContactBuffers, GpuWithoutAGpu_IsRunOnCallFailed)
  ```
  CPU versus GPU comparison (design 6): `kind`, `solid_index`, `face`, `side_at_hit`, `hit_count`, error exact;
  floating values with `CheckWithinTolerance(cpu, gpu, {floor, 0x1p-40})`, the CPU value the reference (the method
  paper's form of design 6's `floor + 0x1p-40 * max(|a|, |b|)`), floor `contact_tolerance / 4` (positions),
  `0x1p-40 * largest input speed` (velocities, speeds), `0x1p-40` (`hit_fraction`). Each test prints the largest
  difference seen. Every `gpu` test passes its GPU outputs (positions, velocities, records, reactions, the result
  and the reported failure; the formula results) to `WriteResultDigest` with a unique name, so the hardware run's
  repeat check covers them.
- [ ] **Step 2: Run** the gate. Expected before implementing: `FAIL cpu` (the steps test does not compile), then
  `FAIL cuda` (link error).
- [ ] **Step 3: Implement** `move_particles_steps.h`, then `move_particles_gpu.cu`: three thin kernels over the
  per-item functions, launched with `ApplyGpuFunction`, statuses returned as `RunOnCallStatus`. It includes only the
  shared headers, `move_particles_gpu.h`, `move_particles_steps.h`, `base/gpu_launch.cuh` and `base/`. Free nothing
  that `ContactBuffers` owns.
- [ ] **Step 4: Run** the gate. Expected: seven `PASS`; in the container the `cpu` tests (including every
  shared-header test and `move_particles_steps_test`) and the `no_gpu` test pass.
- [ ] **Step 5: Commit.** Subject: `Move a range of particles on the GPU, one thread each`. Body: `CUDA: compiled,
  not run`, the private name `MoveParticlesOneThreadPerParticleOnGpu`, and the new names `MoveParticlesView`,
  `ApplyMoveParticleAt`, `ComputeFailureKey`, `MoveParticleCounts`, `ComputeParticleCounts`.

### Task 17: Scene-level tests: conservation and comparison with `main`

**Kind:** integration

**Files:**
- Create: `native/fluidity/contact/move_particles_scene_test.cc`, `contact_main_comparison_cases.h` (test-only)
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `MoveParticles`, `MoveParticle`, `ComputeSolidVelocity`.

- [ ] **Step 1: Conservation tests** (label `cpu`):

  ```cpp
  TEST(MoveParticles, SymmetricHitsOnReactingBox_SummedChangeHasZeroMoment)
      // reacting box half {0.5,0.5,0.5} at origin; two particles of equal mass at {-1, 0.25, 0} and
      // {-1, -0.25, 0}, both velocity {2,0,0}, T 1: sum of velocity_for_next_step_change is {-4,0,0} within
      // 0x1p-44 relative and sum of Cross(hit_position - origin, change) is {0,0,0} within 1e-15
  TEST(MoveParticles, ReactionSum_EqualsSumOfVelocityChanges)  // particles of the first 100 oblique cases against
      // the solid of case 0 with reports_reaction: per particle, change == after - before within 0x1p-44 relative
  ```
- [ ] **Step 1a: Round-solid scenes** (amended 2026-11-15, A74; label `cpu`; `tol = kTestContactTolerance`; the
  test adds gravity itself, so contact alone shows these):

  ```cpp
  TEST(MoveParticles, FrictionlessHitsOnReactingSphere_ChangeHasNoMomentAboutCentre)
      // reacting sphere r 0.5 at {1,2,3}; 100 particles (P11, in this order per particle: direction {NextSigned(), NextSigned(), NextSigned()}
      // redrawn while Length < 0.25 then Normalized; jitter 0.3 * {NextSigned(), NextSigned(), NextSigned()}): start centre + 2 * direction, both
      // velocities -direction + jitter, T 3, friction 0: per particle with a hit,
      // |Cross(hit_position - centre, change)| <= 1e-12 * |change| (every normal passes through the centre)
  TEST(MoveParticle, RestingOnTopOfSphere_Stays)
      // sphere r 1 at origin; particle {0,0,1 + tol}, velocities 0; 1000 steps of T 1e-3, each adds {0,0,-9.81 T}
      // to both velocities, calls MoveParticle and feeds the end back: no error, x and y exactly 0, both velocities
      // exactly {0,0,0} after every step, z within 2 tol of 1 + tol
  TEST(MoveParticle, SlidingOffFrictionlessSphere_LeavesAtAnalyticAngle)
      // same sphere, start {0,0,1 + tol}, velocities {0.01,0,0}, friction 0, T 1e-3; each step v.z -= 9.81 T on
      // both velocities, MoveParticle, end fed back; stop when the end distance exceeds 0.01 (about 2100 steps):
      // no error in any step, hit_count <= 1 in every step; the polar angle acos(z / |x|) of the end of the last
      // step with kind == kHit is within 0.01 rad of acos(2/3 + 0.01^2 / (3 * 9.81)) = 0.8410641118063884
      // (the design's model of the scheme gives 0.8369; the gap comes from removing the inward speed at each hit
      // and from intermittent last contacts, and shrinks with T). Print the measured angle.
  ```
- [ ] **Step 2: Produce the old numbers.** On `main` (read-only; build it in a detached temporary worktree:
  `git worktree add --detach <temp-dir>/fluidity-main main`, then its own presets as `main:AGENTS.md` and
  `main:docs/native/toolchains.md` say), write a throwaway program that calls `ResolveLiquidWallContact3D` for every
  half-space and slab call in `main:native/tests/liquid_wall_contact_3d_cases.h:209-398` and `:441-487` (not the
  tank rows `:38-144`, not the rounding-machinery rows `:399-440`), about twenty rows, and prints inputs and
  outputs as hex floats. Turn each into Z-up with `(x, y, z) -> (x, -z, y)` and store it in
  `contact_main_comparison_cases.h` with the source line of each row. Remove the temporary worktree afterwards
  (`git worktree remove`). If `main` does not build here, stop and tell the owner; the conservation tests still
  land.
- [ ] **Step 3: Comparison test.**

  ```cpp
  TEST(MoveParticle, MainHalfSpaceAndSlabRows_AgreeAfterTurningToZUp)
      // each stored row: positions within 2 * tol (the new end rests tol off the surface), velocities within
      // 0x1p-40 relative, hit or no hit equal
  ```
- [ ] **Step 4: Run** build, tests, gate.
- [ ] **Step 5: Commit.** Subject: `Check contact conserves the reaction and agrees with the old sweep`. Body names
  the `main` commit the old numbers came from.

### Task 18: Comment pass and hand-over

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** hand-over records the Task 16 same-bits promise instead of `fluidity_add_shared_test`; the GPU tests write result digests.

**Files:**
- Modify: any file under `native/fluidity/contact/`, `docs/method-paper/`, `docs/glossary.md`,
  `docs/gpu-runs.md` that the pass changes

- [ ] **Step 1: Comment pass.** Reread every contact file and the paper section from top to bottom as a C++
  programmer who has never read a fluid paper. For each: every comment true of the final code; says why, not what;
  a worked example with numbers on every non-trivial public function (`MoveParticle`, `FindFirstHit`,
  `ComputeContactResponse`, `ComputeContactTolerance`, `MoveParticles`); every word in the glossary; `Words`,
  `Uses`, `Used by` (the step, solids, transfer, particles, materials, render output through solids), `Read:` lines
  resolve; `// Step N:` matches `<proc-move-particle>` and `<proc-find-first-hit>`. Amended 2026-11-15 (A74): the
  worked examples include a sphere and a cylinder rim hit (the paper's two examples), and the round branches' `Read:`
  lines resolve to `<proc-sphere-first-hit>`, `<proc-cylinder-first-hit>`, `<eq:sphere-distance>`,
  `<eq:cylinder-distance>`.
- [ ] **Step 2: Fold each change into the commit that owns the file** (`git commit --fixup <hash>`, then
  `git rebase --autosquash <first contact commit>~1`). Run
  `git rebase <first contact commit>~1 --exec "uv run python tools/check_commit.py --only format" --exec
  "uv run python tools/check_commit.py --only rules"` and the full gate on the tip. Expected: all pass.
- [ ] **Step 3: Check the design section by section** (3.1 to 3.5, 5, 6, 7, 8, 10) against the branch; record in
  the hand-over every plan decision P1 to P17 (P13 to P17 added 2026-11-15 for A74), the two signature additions (`CheckSolid` and `FindFirstHit` with a
  `ContactCloseCall*`), the same-bits promise of Task 16, and `sizeof(ParticleContact) == 56` versus the design's 64.
- [ ] **Step 4: `docs/gpu-runs.md`.** Add the row: module `contact`, the tip commit, result `compiled, not run`,
  notes listing the `gpu` tests: `MoveParticles.SweepAndMoveCasesOnGpu_MatchCpu`,
  `MoveParticles.ObliqueCasesOnGpu_MatchCpu`, `MoveParticles.SameInputTwice_SameBytes`,
  `MoveParticles.TwoFailures_ReportsLowestParticleId`, `SharedFormulas.OnGpu_MatchCpu`. Fold into Task 16's commit.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the contact commits; the gate's output; the
  new public names with their reasons; the deviations of Step 3; the statement that no contact device code has run.
