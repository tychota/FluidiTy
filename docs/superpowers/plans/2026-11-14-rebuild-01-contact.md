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
  **Amended 2026-11-16 (P21, Task 12b):** the face that gives the distance keeps this rule. Every other face plane of
  a box, and a cylinder's wall and cap for a start beside a rim, has the touching rule of a bounding surface: a start
  gap in `[-tol / 8, tol / 16]` that sinks at most `tol / 64` during the sweep rests on it, and the sweep is free of
  the solid (P21 items 1 and 2). It is what lets a particle cross the seam of two flush solids.
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
  **Amended 2026-11-15 (P19, Task 11 review I1; built in Task 11b):** step 2 no longer moves the start one solid at
  a time, so no solid is classified from a start that a later solid's move pushed toward it (the sentence above is
  withdrawn). Order: (1) the failing checks of every solid, in array order, at the start as given: the wrong side of
  the committed plate, then the depth; the first that fails decides. (2) The near surfaces of every solid
  (`FindSurfacesWithinTolerance`: the solid's one surface, or each closed side of a domain box, with `d < tol`), and
  one move of the start to the rest gap of all of them (P19 item 1). (3) The responses, solid by solid in array
  order, each near surface with its distance and normal before the move, as Task 11 built them: a hit at time 0, or
  a silent-move entry. Each velocity is compared with its own allowance so far (P19 item 3), 0 before any response.
  From Task 12 on, each start response is followed by the slide rule, as a hit is (P19 item 3). A start with no point
  at the rest gap of its near surfaces within P19's move cap fails with the scene error `kStartsInNarrowGap` (P19
  item 6: two solids facing each other less than 2 tol apart, or within rounding of a tangent contact point).
- **P5 Reaction.** Every change of `velocity.for_next_step` made against a solid with `reports_reaction` is added to
  `reaction.velocity_for_next_step_change`: the response at a hit (step 2 or step 4) and **each removal during the
  slide** (step 5). A removal of the approach to an earlier surface is charged to that surface's solid. **Amended
  2026-11-15 (paper review):** the corner case of step 5 (the newest surface approached again, both velocities set
  to the newest solid's velocity) is charged to the **newest** solid. `hit_position` is the position of the first
  such change. A second, different reacting solid is `kTooCloseToCall/kSecondReactingSolid`. Tests: Task 10
  (response), Task 12 (a slide removal, the corner case).
  **Amended 2026-11-15 (P19 items 3 and 4, Task 11 review I1; coordinator ruling: accepted; TO CONFIRM WITH THE
  OWNER, beside the zero-change rule above):** what counts in a solid's reaction is now, in full:
  1. *A response* (a hit, or a start response): the change of `for_next_step` it makes, charged to the solid
     responded to, as before.
  2. *The slide's change* (P19 item 3): one frictionless change over the active surfaces,
     `w = sum m_i n_i` with multipliers `m_i >= 0`. Each used surface's push `m_i n_i` of the change of
     `for_next_step` is charged to that surface's solid, at the particle's position. Each solid so receives an
     impulse along its own normal, which is what two-way coupling needs: a frictionless surface cannot push sideways.
     A removal at one earlier surface is the one-surface case and is charged as before.
  3. *A sliver's stop* (P19 item 3: a pair below the velocity conditioning limit): the whole change, the newest
     solid's velocity minus `for_next_step`, is charged to the newest solid, which the particle sticks to.
  4. *Second reacting solid:* unchanged. A push or a change that is not exactly `{0, 0, 0}` at a second reacting
     solid is `kSecondReactingSolid`, whether it comes from a response or from a push of the slide's change; a push of
     exactly zero (a surface the change did not use, or a multiplier of 0) neither claims nor conflicts, as the
     zero-change rule above says.
  5. *`hit_position`*: the position of the first non-zero change charged to the solid, response or push.
  What changes against the paper review's amendment: the corner case is no longer charged whole to the newest solid,
  which gave the newest solid the other surface's push. In Task 12's `AcuteWedgeOnReactingFloor` the floor would get
  `{-0.5, 0, 1.5}`, a sideways push from a frictionless corner; per surface it gets `{0, 0, 2}`. Tests: Task 12
  (`AcuteWedgeOnReactingFloor_EachSurfaceChargedItsOwnPush`, `SlideRemovalOnReactingFloor_CountsInReaction`,
  `SliverStop_ChargedToNewestSolid`, `CornerPushOnSecondReactingSolid_IsSecondReactingSolid`).
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
  **Amended 2026-11-15 (P19, Task 11 review I1):** with several near surfaces the start moves once, by the smallest
  move that leaves every one at `d >= tol` (P19 item 1); with one it is the move along that solid's normal, as
  before. The promise above now holds in wedges, corners and the cusps of tangent contacts (a ball resting on a
  floor); the only start without a rest point is `kStartsInNarrowGap` (P19 item 6). The settle also moves the end
  off every near solid, touched or not (P19 item 7).

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
  **Amended 2026-11-16 (P21, Task 12b; coordinator ruling 44):** the entry into the cylinder grown with sharp edges is
  the hit only where it lies on the wall or on a cap proper. Beside a rim (both distances positive at the entry) the
  hit is found on the true distance by the rim search from that entry (P21 items 3 and 4), so a rim is hit at `d <=
  tol + tol / 1024`. The rim move is withdrawn, and with it the sentence above on the stepping search, which had
  nothing left to be exempt from; the rule for a start in the rim corner ("a hit at 0 only when `rate < 0`") is
  withdrawn too: such a start has the entry 0 and the rim search decides, after the touching rule of the wall and the
  cap as bounding surfaces (P21 item 1). The two intervals are unchanged.
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
  **Amended 2026-11-16 (P21):** a sweep after a rim hit starts within `tol / 1024` of `d = tol`, inside the touched
  window, not at `d = tol` by a rim move; "(a hit, the rim move or the start move put it there)" reads "(a hit or the
  start move put it there)".
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
  stepping-search hit" and Task 12a are unchanged. (Amended 2026-11-16, P21: no hit is moved any more; a turning
  cylinder's hit is used where the search stopped, as before.) `kMaxAdvanceCount` stays 128 and a close call stays possible (a
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

Plan decision added 2026-11-15 for wedges, corners and tangent contacts (Task 11 review I1; built in Tasks 11b and
12; the paper states it):

- **P19 Several near surfaces at once: wedges, corners and tangent contacts.** Words: the *wedge angle* of two
  surfaces is 180 degrees minus the angle between their normals (an acute wedge has `Dot(n_A, n_B) < 0`); a surface's
  *shortfall* at a point is `tol - d`; a *tangent contact* is two convex solids touching at a point or along a line (a
  ball or a cylinder resting on a floor, two balls touching), around which the free space is a *cusp* whose angle
  shrinks to 0. Why: a start within tol of two surfaces of an acute wedge failed at every step length (Task 11 review
  I1: at rest on both surfaces, 2000 of 2000 starts `kEndsTooCloseToSolid` at 60 degrees, `kTooManyHits` at 30),
  because each solid's move to its rest gap pushed the particle toward the other. A particle that slides down a
  V-groove, or into the cusp under a ball resting on a floor, rests at the rest gap of both surfaces, so the next
  step starts there. Amended the same day (coordinator ruling): a first version refused wedges sharper than
  0.4476 degrees, which made every ball resting on a floor fail; the limit below protects only what it must.
  1. **Position: one move off every near surface** (private `MoveToRestGapOfNearSurfaces`, used by the start rules
     and by the settle). Given a point `p0`, a pose time and the solids (every solid, at the start and at the end),
     it moves `p0` to a point at `d >= tol` from every surface, as near to `p0` as it can find, by *sequential
     linearisation*: iterate `x_1, x_2, ...` from `x_0 = p0`; `x_(k+1) = p0 + w` with `w` the smallest change such
     that `d_i(x_k) + Dot(n_i(x_k), p0 + w - x_k) >= tol` for every near surface `i` (`SolveSmallestCorrection`), the
     distances and normals read afresh at `x_k` (`FindSurfacesWithinTolerance` of each solid: its one surface, or each
     closed side of a domain box). Why it works: every shape of module 1 has a convex distance per surface, so
     `d_i(y) >= d_i(x) + Dot(n_i(x), y - x)` (P18's subgradient inequality) and each iterate after the first that
     includes every near surface is at the rest gap of all of them up to rounding; and `x_k` satisfies the
     linearisation made at itself, so `|x_(k+1) - p0| <= |x_k - p0|`: the iterates come nearer to `p0`, never
     farther. Two flat faces: exact in one iteration (2 tol at 60 degrees from a start on both, `tol / sin(a / 2)`
     in general). A cusp: the first linearisation from deep in the cusp overshoots out along the floor (its tangent
     planes meet at about half the distance), and the next ones come back to the rest point: for a ball of radius
     `R` on a floor, the point `{2 Sqrt(R tol), 0, tol}` from the contact (both distances exactly tol), reached in
     about 5 iterations at `R / tol = 2^29` and to within `1e-4` of it in 8 at `R / tol = 2^40`. Stop: when an
     iterate moves less than `tol / 64` and is at `d >= tol - tol / 1024` from every surface, or after
     `kMaxRestGapIterations = 8`; the last iterate is returned when it is at the rest gap of every surface (it is,
     for convex shapes, from the first iterate that includes them all), else no move is found. A surface the move
     brings within tol joins the next iteration; at most `kMaxConstrainingSurfaces = 4` surfaces. Each iterate's
     change is clipped to the move cap (item 5), which keeps the first, overshooting iterate of a deep cusp near; the
     next iteration repairs it. A move longer than the thinnest solid could pass through one: so the segment from
     `p0` to each iterate is swept against the solids that are not near (the closed forms of `FindFirstHitOfSweep` at
     the fixed pose, no touched list), and a solid it would enter joins the next iteration.
  2. **`SolveSmallestCorrection`** (private, shape-free): up to `kMaxConstrainingSurfaces` unit normals `n_i`,
     targets `s_i` and a slack. It tries each single surface, each pair and each triple as equalities, in that order
     with ascending indices: the Gram system `Dot(n_i, n_j) m_j = s_i` by Cramer's rule, multipliers `m >= 0`,
     `w = sum m_i n_i`, and with `refine` up to `kMaxCorrectionRefinements = 4` more solves on the same subset for
     the residual, until it is within a quarter of the slack. It keeps the candidate with
     `Dot(n_i, w) >= s_i - slack` for every `i` and the smallest `Length(w)`, the first in order on a tie, or reports
     none. Fixed loops (at most 14 subsets), no allocation, + - * / and Sqrt only, the same bits on CPU and GPU. The
     exact minimum of a convex problem with at most three active surfaces, enumerated. A pair is skipped when
     `1 - c^2 <= 0` as computed (`c = Dot(n_A, n_B)`), a triple when `Dot(n_1, Cross(n_2, n_3)) == 0`; for a velocity
     also below the conditioning limit of item 5. Slack: for a position `tol / 1024` plus the rounding of composing
     `w`, `2^-49 sum |m_i|` (a cusp's multipliers are large and the next iteration corrects); for a velocity the
     allowance of item 3.
  3. **Velocity** (Task 12's slide rule, also run after each start response). The *active* surfaces are the near
     surfaces of the touched solids at the current position and pose, read afresh (`FindSurfacesWithinTolerance`,
     `d - tol <= tol / 16`), with their normals there, not the normals recorded at the hits: on a ball the normal of
     the first hit is stale once the particle has slid along it (in the model, the stale normal beside the fresh one
     made a corner of three surfaces with no solution). A surface the particle has left is not active (a ceiling hit
     0.1 m above a touched floor must not stop the particle). For each velocity `v` with its own allowance `b_v` (for
     `for_this_step` the largest allowance of the touched list, as in P2; for `for_next_step` a running largest of its
     own, from the `for_next_step` passed to each response, read only by approach tests), with `u_i` the velocity of
     surface `i`'s solid at the particle: when some `-Dot(n_i, v - u_i)` exceeds `b_v`, `v` changes by
     `SolveSmallestCorrection` with those targets, `refine` on. One surface approached: its approach is removed
     (design 5.5's removal). Two fixed surfaces: the part along their common line stays, so a particle in a V-groove
     slides on along it, and one pushed into a cusp stops while it can still leave along the floor. Three
     independent fixed surfaces: it stops. Moving solids: the velocity that approaches none, which can be faster than
     either solid (a floor rising at V under a fixed wall at 60 degrees gives `{V / tan 60, 0, V}`: carried along the
     apex). Friction acts only in each surface's own response, never in this change. This replaces design 5.5's
     sequence (remove each earlier approach, then the newest solid's velocity in a corner, then squeeze if an earlier
     one is still approached): the newest solid's velocity stopped a particle that should slide along a groove. The
     change's allowance is `kApproachLeftAfterResponseShare * (|v|_1 + |w|_1 + sum |u_i|_1)`; it raises the newest
     entry's allowance (for `for_this_step`) or the running one (for `for_next_step`). *Newest*: the last entry of the
     touched list; at the start, the last surface that responded, in array order and a domain box's face order.
     *Sliver* (a pair active and approached whose `1 + c` is below the conditioning limit, a wedge under 1.35e-6 rad:
     a plank lying on a floor with a rounding-level tilt): the frictionless change cannot be computed within the
     allowance, so the velocity takes the newest solid's velocity (the particle sticks), charged to that solid; the
     other surfaces are then checked against the allowance as a squeeze.
  4. **Reaction** (P5 amended): the change of `for_next_step` is split into the used surfaces' pushes `m_i n_i`,
     each charged to its own solid through `ApplyReactionChange`, at the particle's position.
  5. **What the limits protect, and their values.** *The size of a move.* The particle is displaced without a matching
     velocity, so the move must be small against the cell width `h` (what the grid and the step's travel limit resolve),
     not against tol. `kMaxRestGapMoveInTolerances = 2^22`: with `tol = 2^-30 h + 2^-40 L` (`ComputeContactTolerance`),
     `2^22 tol = h / 256 + L / 2^18`, a 256th of a cell plus 4 micrometres in a one-metre scene. It covers every cusp:
     the rest point of a ball of radius `R <= L` (P8 counts `Length(half_size)` in `L`) is `2 Sqrt(R tol) <= 2
     Sqrt(2^40) tol = 2^21 tol` from the contact (measured largest start move in the tangent sets: 45,241 tol at `R /
     tol = 2^29`). A flat wedge needs more only below about `2^-21` rad (`2 shortfall / angle > 2^22 tol`).
     *Conditioning*, for velocities only: the pair solve leaves `2.5 eps / (1 + c)` of the target as residual (measured,
     `eps = 2^-52`; `1 + c` from `2^-40` to `2^-4`) and each refinement multiplies it by about `eps / (1 + c)`, so a
     pair is used for a velocity when `1 + c >= kVelocityPairLimitShare = 2^-40`: a wedge angle of at least `2
     asin(2^-20.5) = 1.35e-6` rad (7.7e-5 degrees), where at most 3 refinements leave at most `16 eps` of the summed
     speeds (measured), a quarter of the share. Every cusp a particle can rest in is above it: at the rest point the
     angle is about `2 Sqrt(tol / R) >= 2^-19` rad, so `1 + c >= 2^-39`. Positions need no such limit: an imprecise
     iterate is corrected by the next one and the final check reads the true distances (at `1 + c = 2^-40` the residual
     is `tol / 2000` anyway). A triple is used for a velocity when `Abs(Dot(n_1, Cross(n_2, n_3))) >= 2^-20`. *Speed of
     an escape* (coordinator ruling: capped). A wedge closing on a particle at a small angle pushes it out at the
     closing speed over the angle (measured without a cap: a ball sinking at 1e-3 m/s into a floor, 12.6 m/s; a ball
     driven into a tilted plane at up to 0.6 m/s, 5,600 m/s; a 1e-5 rad wedge closing at 1 m/s, 1e5 m/s, a kilometre in
     a step of 0.01 s). A particle thrown that far is worse than a reported squeeze, so the change is refused when the
     velocity it gives is faster than `kMaxSqueezeOutSpeedFactor = 64` times the largest speed involved: the particle's
     two velocities before the change and the velocities of the active surfaces' solids at the particle (`Length` each).
     Which closing angles this is: one face closing at speed V on a fixed one gives `V / sin a`, refused below
     `asin(1/64) = 0.895` degrees; two faces closing symmetrically, each at V along its own normal, give `V / sin(a /
     2)`, refused below `2 asin(1/64) = 1.79` degrees. The geometry and the speeds decide, not the step length, so a
     shorter step does not cure it (item 6). Separately, the change is also refused when its allowance times the time
     left exceeds `tol / 64`, `MoveParticle`'s own Requires (the summed speeds times the time left at most `2^40 tol`)
     checked instead of asserted; with the cap this needs speeds near `2^34 tol` per second of step and a shorter step
     cures it.
  6. **What still fails, and why that is acceptable.** `kTooCloseToCall/kSqueezedBetweenSolids`: no velocity change
     exists because two active surfaces close on the particle, `(n_i - n_j) . (u_i - u_j) / 2 > b_v` (compared with the
     allowance, not with 0: two solids moving together have 0), for example two parallel plates closing; or the escape
     of item 5 is above the cap (one face closing on another at under 0.895 degrees, two at under 1.79). Both name the
     newest surface's solid, and neither is cured by a shorter step: `MoveParticle`'s docstring lists this squeeze among
     the failures a shorter step cannot fix, and the step module decides what a persistent squeeze does
     (`changes-to-apply.md`, 10 Step, for the owner). Only the allowance check of item 5 is cured by a shorter step.
     Measured with the cap, moving sets: flat wedges of 5 to 175 degrees, 0 of 35,000 squeezed; flat wedges of 0.0001 to
     5 degrees with both faces moving at random up to 1 m/s per axis, 4,344 of 21,000 (from 0 at 2 to 5 degrees and 0.6
     % at 1 to 2 degrees to 53 % under 0.01 degrees); tangent contacts, 1,211 of 20,000, all but one a ball driven into
     a tilted plane it penetrates at up to 0.56 m/s (30 % of those draws), the other a ball sinking into a floor.
     `kStartsInNarrowGap`, a new scene error (`ContactError`, appended after `kOutOfMemory`, name
     `starts_in_narrow_gap`), replacing the `kWedgeTooSharp` of this decision's first version: the start has no point at
     the rest gap of every near surface within the move cap. It happens only (a) between two solids whose facing
     surfaces are parallel to the last bit (`1 + c` computes to 0) and less than 2 tol apart, which a sweep never
     reaches, since the grown solids overlap there: two objects placed nearly touching face to face (a box 1e-12 m from
     a wall), or a start within about `1.8e-8 R` of the exact contact point of a ball on a floor (about 1 % of the
     cusp's radius at `R / tol = 2^40`, 0.02 % at `2^29`; a source would have to place a particle in a disc of that
     radius and 2 tol high); (b) more than four near surfaces; (c) a start deep at the apex of a flat wedge sharper than
     about `2^-21` rad; (d) a pocket too small to hold a point at tol from each of its sides (a domain corner cut off by
     a solid 4 tol out; a sliver wedge closed by a plate nearer than its rest point). Each is a place no particle can
     rest in, and a sweep stops a moving particle before it. It names the near solid latest in array order, magnitude 0.
     Scene check: module 10 may name case (a) between flat faces before the run ("solids 4 and 7 face each other less
     than two contact tolerances apart"); it is not needed for correctness and is left to module 10's design. Nothing
     that a sweep can produce fails without a cure by a shorter step.
  7. **Settle** (amended with this decision): one `MoveToRestGapOfNearSurfaces` over every solid at the end pose,
     not only the touched ones. With one move off several surfaces, leaving out a near solid that was not touched
     pushes the particle into it (in a cusp under a ball touching a wall, measured: the settle off the touched ball
     ended 0.43 tol from the untouched wall). It also lifts a particle that P2's touching rule let drift toward an
     untouched surface (up to tol in a step; the model's slow paths failed the end check that way 1 to 2 % of the
     time) back to the rest gap, which P12 promises. The end check keeps its role: when no move exists (two
     surfaces less than 2 tol apart, `CeilingClosingFasterThanParticleLeaves`), the settle leaves the particle and the
     end check fails. The settle's `d >= -2 tol` assertion stays. *Cost per particle:* the settle reads one distance
     per solid at the end pose, the reads the end check makes anyway: when no solid is nearer than tol (almost every
     particle) nothing moves and the end check reuses those distances, so the cost is unchanged; a particle that is
     moved pays at most `kMaxRestGapIterations` more reads per solid, plus one closed-form sweep of the move per solid
     that is not near. *Proof for touched surfaces:* unchanged, since it bounds the sweeps (a touched solid approached
     by at most `tol / 16` before the settle); the settle only moves the particle away from every near surface (each
     iterate is at the rest gap of all of them, item 1), so it leaves a touched surface at `d >= tol - tol / 1024`.
     *End check:* unchanged, every solid at `d >= tol / 2` at the end pose, read after the settle; it fails only when
     the settle found no move, when a solid that is not near after the settle closed on the particle faster than the
     sweeps could keep it out, or after a refused move.
  8. **Touched list and its proof.** The position move adds no entry: each near solid keeps Task 11's response entry
     or silent-move entry (`touched_surface_count`, `kMaxTouchedSurfaces = 8`). The velocity change adds no entry; its
     allowance raises the newest entry's, made at the same time. So entries still come at most at `kMaxHitsPerStep`
     times, each asserting `allowance * time left <= tol / 64`, and `ComputeTouchedSolid`'s proof (a touched solid
     approached by at most `tol / 16` in the step) holds unchanged. `for_next_step`'s allowance is not in the proof:
     that velocity does not move the particle. `kMaxHitsPerStep = 4` stays enough: a start in a three-sided corner
     makes at most three hit entries, and the change then leaves the velocity at the solids' velocity. Two convex
     solids that both curve away (two touching balls) can make a fast particle zig-zag into their neck, hit after
     hit, and end in `kTooManyHits`; measured, only for steps that move the particle 1 to 29 cm, and always cured by a
     shorter step (31 of 4000 two-ball draws; every one succeeds by 1/16 of the step).
  9. **Rejected.** (a) Repeated passes of the one-solid move: each multiplies the deficit by `cos^2` of the wedge
     angle; from a start on both surfaces reaching `tol - tol / 64` takes
     `ceil(ln(64 (1 + cos a) / cos a) / (-2 ln cos a))` passes: 4 at 60 degrees, 18 at 30, 159 at 10, 15,928 at 1
     (measured, as predicted); never exact, and in a cusp no faster than its local angle. (b) One linear solve from
     the normals at the start: exact for flat faces, but in a cusp it overshoots far out along the floor (to about
     `2 tol R / x` from a start at `x`). (c) An angle limit (this decision's first version, 0.4476 degrees): it
     refused tangent contacts, which are among the most ordinary scenes. (d) A closed form for ball and plane: exact
     but shape-specific, and 1b's meshes need the general rule anyway.
  10. **Measured** with a double-precision Python model of `MoveParticle` with this rule (flat solids, plates,
     spheres, a cylinder lying along x, domain boxes; the start, the slide rule of Task 12, the settle). Flat wedges,
     5 to 175 degrees, 1000 draws per 5 degrees, fixed and moving solids (up to 1 m/s per axis), starts within, on
     or up to 2 tol inside each surface, velocities at rest, slow, falling or up to 2 m/s: 70,000 of 70,000 succeed
     (the code as built fails 8 % of the fixed draws at 85 degrees, 49 % at 60, 98 % at 5). Flat wedges of 0.0001
     to 0.6 degrees: 18,000 fixed draws succeed; the moving ones are in item 6 (squeezed by the speed cap). Tangent contacts, two seeds of 2000
     draws per scene and motion: ball on a floor, cylinder lying on a floor, ball in a domain corner (touching three
     sides), two touching balls, ball on a tilted flat solid; fixed, and moving (solids sliding along each other at up
     to 1 m/s, balls sinking or rising at 1e-3 m/s, a ball driven into the tilted plane); starts in the cusp from
     0.02 to 1 times the rest point's distance, and slides into it from up to 5 cm at up to 2 m/s, with gravity:
     39,969 of 40,000 succeed without the speed cap, the other 31 being the two-ball `kTooManyHits` above; with the
     cap, 1,211 of the 20,000 moving draws are squeezed (item 6); no `kStartsInNarrowGap`. A ball
     with `R / tol = 2^40`: starts from 0.01 to 1 times the rest point's distance succeed, within 1e-4 of it.

Plan decision added 2026-11-16 for the response at several surfaces (Task 12 re-review 2, coordinator ruling 40;
built in Task 12's third fix round; the paper states it):

- **P20 One joint response at several surfaces.** Words: a *hit* is one moment, at the start or along the path, when
  the particle is on one or more surfaces and both velocities get the response; the *joint response* is the response
  of a hit over every surface the particle is on; a surface's *push* is how far the frictionless change goes along
  its normal (the multiplier of `SolveSmallestCorrection`, m/s); its *friction limit* is friction times its
  `friction_scale` times its push, the most sideways speed its friction can take in this response (m/s). Why:
  Task 12 as built applied friction by one response per surface, in turn, then one frictionless change. The result
  depended on the order of those responses, and the rounding of the start point decided the order: at the rest point
  of two faces moved by one unit in the last place, 16.0 % of 100,000 draws jumped (48 % when the velocity pushes
  into both faces), by a median of 0.049 and up to 0.42 of the speed; without friction 17.5 %, up to 0.50, above 90
  degrees. Two rounds of repairs on the sequence (repeated neighbour passes, a rule for coincident twins) each added
  findings (re-review 2: twin entries overflow the touched-surface list; a copy whose velocity differs by one
  rounding unit takes a second response; more `kTooManyHits` at moving corners). This decision replaces the sequence.
  It amends P2 (the touched-surface list), P4 (the start's responses), P5 (what is charged), P12 (the normal a start
  response reads) and P19 items 3, 4, 6 and 8.
  1. **The surfaces of a hit.** The *active surfaces*: every surface of every solid (its one surface, or each closed
     side of a domain box) within `tol + tol / 16` of the particle (`kTouchedSolidRestingGapShare`), read afresh at
     the particle's position and the pose of the hit, with the normal there and the solid's velocity at the particle
     (`ComputeSolidVelocity`); and the surface the sweep hit, whatever its distance (a turning solid is hit up to
     1.25 tol away), with the hit's own normal. Touched or not: a face at the rest gap that a velocity approaches
     counts (ruling 37, kept). At the start they are read at the moved start, not at the start as given (P12's
     "normal before the move" is withdrawn for the response: the sweep that follows reads the normal at the moved
     start, and the response must leave no approach along that one; `CuspStartIntoBallOnly` now keeps 4.3e-5 m/s
     along the ball where it stopped). A surface that is the same as one already listed, bit for bit, in normal,
     solid velocity, friction scale and `reports_reaction` is dropped (an exact copy adds nothing, so 17 or 40
     coincident floors take one place). The rest are sorted by value: normal x, y, z, then solid velocity x, y, z,
     then friction scale, then solid index. Surfaces whose normals agree within `kSameNormalAxisDifference` on every
     axis share one *direction*, whose normal is that of its first surface in the sorted order. Limits, unchanged:
     `kMaxTrackedSurfaces = 16` surfaces and `kMaxSlideDirections = 8` directions, beyond which the hit is
     `kTooManyHits`. Why sorted: every sum and every tie then reads the surfaces in an order that their values
     decide, so the velocities are the same bits for every order of the solids array (measured: 0 differences over
     every permutation of 119,999 corners, and of 40,000 whole moves; unsorted, 554 of 39,999 differ, by at most
     1.7e-13 of the speed).
     **Amended 2026-11-17 (P21 item 16, Task 12b review I1):** a solid that does not turn and that the particle passes
     at a bounding surface's level (level with a box's face plane other than the one that gives its distance, or
     beside a cylinder's rim and level with its wall or its cap's plane: a gap in `[-tol / 8, tol / 16]`, the gap
     alone) contributes no active surface, unless it is the surface the sweep hit. Its distance there is a face plane
     continued past the edge, or a rim whose normal leans across a seam, and the sweep is free of it while the
     particle rests at that level. The test reads the position only, so the order-free and copy-free properties stand.
  2. **One active surface.** Each velocity gets `ComputeContactResponse` with that surface, and nothing else: the
     single-surface response of Task 7, the same bits as before this decision (measured: 120,000 of 120,000).
  3. **Several: the frictionless change.** For each velocity `v`, each direction's target is the largest
     `-Dot(n_i, v - u_i)` of its surfaces, each measured along its own normal from its own solid. When no target is
     above the velocity's allowance so far (0 before the first response of the step), `v` is unchanged, bit for bit.
     Otherwise `SolveSmallestCorrection` (velocity; refinements and conditioning limits as in P19 item 5) gives the
     change `w = sum push_k n_k` over at most three directions and `v0 = v + w`. Two changes to the solver, for a
     velocity only. (a) "No change" is the answer only when every target is within the slack passed, the allowance
     so far, not within the candidate's own allowance `2^-46 (speeds)`: a slow real approach beside a fast solid is
     removed, as one surface's response removes it (`SlidingFloorThenSlowUpright`: 1e-13 m/s beside a floor sliding
     at 100 m/s). (b) Ties. Where normals depend on each other (three faces sharing one line, four at a point) the
     same smallest change has several sets of pushes, and friction, which reads the pushes, differed by which set
     rounding picked: with the first subset in order, 15,059 of 20,000 coplanar triples jumped, by up to 0.70 of the
     speed, for a 1e-13 change of the velocity. So among the candidates no longer than the shortest plus its slack,
     the solver keeps the one with the largest sum of friction limits (`sum friction_k push_k`, the friction of each
     direction's pushing surface, item 4), then the first in subset order. That sum is the largest value of a
     linear function over the admissible pushes, which varies continuously with the velocity (measured: 0 of 20,000,
     largest difference 2.6e-13). Coincident surfaces are the same case and get the same answer: the rougher acts.
  4. **The pushing surface of a direction** carries the direction's friction and reaction. Among its surfaces whose
     target is within `2^-40 (|v|_1 + |u_i|_1)` of the direction's (they push; the bound is what normals
     `kSameNormalAxisDifference` apart can differ by), it is the one with the largest friction scale, then one whose
     solid reports a reaction, then the first in the sorted order. So a copy that is as rough or smoother, with the
     same solid velocity, changes nothing (measured: 0 of 30,000 scenes for an exact copy and for a smoother one, in
     every place of the array); a rougher one acts in its twin's place (60 degree wedge, `{-0.2, 1, -1}`, friction
     0.2: 0.7307 along the edge; with a ground of friction scale 2 on the floor, in any order, 0.5076); a copy whose
     solid velocity differs by one rounding unit changes the result by at most 3.8e-15 of the speed (re-review 2's
     I-1 scene: the same bits as without the copy). A coincident surface that moves differently along the surface (a
     conveyor on a floor) is not a copy: with equal friction scales the first in the sorted order acts, the fixed
     floor; this is a choice, pinned by a test.
  5. **Friction**, from `v0`, the pushes and each used direction's pushing surface (normal `n_k`, solid velocity
     `u_k`, friction limit `L_k`):
     - *one used direction:* the velocity is `ComputeContactResponse(v, u_k, n_k, friction_k)` itself, so a hit on
       one surface beside others that are not approached is still the single-surface response, bit for bit.
     - *several, all with one solid velocity `u`, bit for bit* (fixed solids, or one moving solid's faces): the
       surfaces act as one. With `s = v0 - u` and `L = sum L_k`: when `L >= Length(s)` the velocity is `u` exactly
       (stick), else `v0 + (L / Length(s)) (u - v0)` (slip). This is the exact answer of friction with these pushes
       (checked against a numerical minimisation: at most 1.3e-8 of the speed, its own tolerance, over 2,028 draws).
       A wedge amplifies friction, as it should: in a 30 degree groove a push of 1 m/s into the apex makes pushes of
       2 and 2.73 m/s.
     - *several, moving differently:* the mean, over every order of the used directions (2 or 6), of the surfaces'
       own frictions applied one after the other from `v0`: each takes `min(L_k, sideways speed)` off the sideways
       velocity relative to its own solid at that moment. Every order is a sequence of single-surface frictions, so
       within one order no step reverses the sliding relative to its own solid, and the mean of all orders does
       not depend on any order. The mean itself can end past sticking to one of the solids (as built, measured in
       the task's review: 1,331 of 24,401 such responses, 5.5 %, by up to 0.22 of the speeds); this is a known
       limit of the approximation, pinned by a test, and part of the owner's question on an exact rule. When the solid velocities
       differ by rounding only, it agrees with the case above to 1.7e-14 of the speed (20,000 wedges), so nothing
       jumps between the two cases. It is an approximation: against the exact answer of the same model, median
       0.046, p90 0.19, p99 0.37 and at worst 0.66 of the largest speed, over 2,418 draws with friction up to 2 and
       solids up to 1 m/s apart. More passes per order do not help (median 0.034 with 8). The re-review's sketch,
       every surface's own friction from the same `v0`, added, is rejected: with two velocities one rounding unit
       apart it reverses the sliding in 13,218 of 20,000 draws, by up to 9.9 times the speed, and it is twice as far
       from the exact answer (0.097, 0.51, 1.02, 1.52).
     The friction change is `F`, with a share for each used direction (`(L_k / L) F`, or the mean of its own steps).
  6. **No approach afterwards.** With one used direction or one solid velocity, `F` has no part along a used normal,
     and `v0 + F` is the result when it approaches no active direction beyond the response's allowance (the first
     two cases of item 5 keep their bits); when it does (a surface of another solid that friction turns the velocity
     into, or rounding: 53 of 19,625 fixed or shared cases), one more frictionless solve is made. When the used
     solids move differently, that second solve is always made. It starts from `v + F`, not from `v0 + F`: the
     result is `v + F + w2`. From `v + F` the pushes are found afresh, so friction lifts a particle off a surface
     only when it overcomes what pressed it there: beside a particle falling at 1 m/s onto a floor and pushed at
     1 m/s into a wall whose surface rises at 2 m/s, friction 0.5 leaves `{0, 0, 0}` and friction 1.5 leaves
     `{0, 0, 0.5}`; from `v0 + F` friction 0.5 would lift it at 0.5 m/s. The second solve's pushes add no friction.
  7. **Squeezes, the sliver and the cap** (P19 items 3, 5 and 6, without "newest"). The cap reads each solve's
     result: faster than `kMaxSqueezeOutSpeedFactor = 64` times the largest `Length` among the two velocities before
     the response and the active solids' velocities is `kSqueezedBetweenSolids`. The speeds are now those before any
     friction, so the limits are the closed forms: one face closing on a fixed one below `asin(1 / 64) = 0.8953`
     degrees, two closing faces below `2 asin(1 / 64) = 1.7906` degrees (measured: 0.8953 and 1.7906; Task 12 as
     built measured 1.7889, after the side's own response). No change found (a pair below the conditioning limit,
     or solids closing): the particle sticks to the active surface it approaches fastest (the largest target; a
     tie: the first in the sorted order), the whole change is charged to that solid, and any active surface that
     velocity still approaches beyond the allowance is `kSqueezedBetweenSolids`; the separate test of the closing
     speed is dropped (it never decided an outcome). For `for_this_step`, an allowance times the time left above
     `tol / 64` is `kSqueezedBetweenSolids` as before, cured by a shorter step. A squeeze names the solid of the
     surface that the failing velocity approaches fastest (the lowest solid index on a tie), no longer "the newest".
  8. **Allowance.** One active surface: `ComputeLargestApproachLeftAfterResponse` of the velocity passed, as before.
     Several: `2^-46 (|v|_1 + |change|_1 + sum |u_i|_1)` over the sorted active surfaces, for the solve that gave
     the result (and the first when there were two). Measured: what a joint response leaves along any active normal
     is at most 41 rounding units (`2^-52`) of those speeds, against the 64 allowed; p99 1.0.
  9. **Bookkeeping.** `hit_count` counts hits, so joint responses, not surfaces: a wedge, a corner of three sides
     or a corner with two chamfers is one hit. `kMaxHitsPerStep = 4` is unchanged and bounds the responses and the
     sweeps of a step; a fifth is `kTooManyHits`. The record is that of the first hit: along the path, the surface
     the sweep hit; at the start, the first in array order (solid, then face) among the active surfaces that the
     start was inside or that a velocity approaches. The touched-surface list is removed. After the first response
     of a step every solid has the touched rule of P2 (a start gap in `[-tol, tol / 16]` rests; hit at 0 only when
     the path approaches faster than the allowance), with the largest allowance of the responses so far, one number
     per velocity. Why this is the same rule: a sweep starts where a response was made, and a solid within that
     window at the sweep's start was an active surface of that response, so the response left its approach within
     the allowance; a solid outside the window takes its closed form, listed or not. So nothing is listed, nothing
     can overflow (re-review 2's C1: eight, 17 or 40 coincident floors and a wall give 2 hits), and the silent
     entries, the twin entries and "a fifth solid moved silently is `kTooManyHits`" are gone. P2's proof is
     unchanged: responses come at most at `kMaxHitsPerStep` times, each with `allowance * time left <= tol / 64`,
     so a free path comes closer to a solid by at most `tol / 16` in the step. Before any response the plain
     touching rule holds, as P12 asks. Nothing replaces "a frictionless change once a step has four hits": one
     response reads every surface, uses at most three, and is one hit.
  10. **The start** (P4 amended). After the failing checks and the one move off near surfaces (P19 item 1), the
     start has a response, at time 0, when it was inside a solid (`d < 0` as given) or when either velocity
     approaches an active surface (below 0, or below the rounding of a spinning solid's surface velocity, as
     before). It is one joint response over the active surfaces at the moved start, which replaces the per-solid
     start responses and the slide after them. `kind` is `kStartedInsideSolid` when the start was inside a solid,
     else `kHit`. Otherwise the move is silent: no hit, no record, and the plain touching rule. A surface that was
     near at the start but is farther than `tol + tol / 16` after the move is not in the response (of two parallel
     floors tol / 4 apart, only the upper one, which the particle rests on, responds and is charged; as built the
     lower one, behind it, took the response and the upper one's reaction was lost).
  11. **Along the path.** A fifth hit fails; the particle moves to the hit; one joint response. The neighbour
     passes, the rule for twins and the response-then-change sequence are removed entirely.
  12. **Reaction** (P5, decision unchanged). What one response charges, for `velocity.for_next_step`: with one
     active or one used direction, the whole change to the pushing surface's solid; otherwise each used direction's
     push `push_k n_k` and its share of `F` to its pushing surface's solid (after a second solve, the second
     solve's pushes and the first's friction shares). The charges of one response are applied in ascending solid
     index, so two reacting solids pushed by one response fail with `kSecondReactingSolid` naming the higher index.
     What P5 implies here, stated, not changed: a particle in the crease of two reacting solids that pushes into
     both fails at every step length (the owner's open question); a direction has one pushing surface, so two
     coincident reacting solids never conflict in one response, and of the two the rougher, then the first in the
     sorted order, is charged; a reacting solid lying exactly on a rougher solid that does not report gets no
     reaction. In a narrow wedge the pushes grow as 1 over the angle (P19), and so does what a reacting face is
     charged.
  13. **What holds, and why.** (a) *No approach:* the result passed the test of item 6 or is a solve's answer, so
     every direction's target is within the allowance. (b) *No kinetic energy from fixed solids:* with every active
     solid fixed the allowed velocities form a cone through 0, the smallest change is the projection on it, which
     does not lengthen a velocity, and friction shortens the result toward 0 (measured: 0 gains above 1e-12 in
     51,069 corners and in 253,237 moves of re-review 2's draws). (c) *Friction opposes sliding and never reverses
     it when the solids share one velocity (for solids moving differently see item 6's known limit):* with one solid velocity the result is `u + c (v0 - u)`, `0 <= c <= 1`; in re-review 2's 98,316 two-face
     draws the part along the edge is never reversed or grown. (d) *One surface:* the same function. (e) *Order
     and copies:* item 1's sort and drop, and formulas that are sums or largest values over the surfaces.
     (f) *Continuity:* the projection, the stick-or-slip formula and the mean over orders are continuous in the
     velocity, and the start point enters only through the set of active surfaces, the normals and the solid
     velocities read there. Measured: the 27 starts and both array orders of re-review 2's test give the same bits
     in 100,000 of 100,000 draws, with friction and without; over 120,000 corners a 1e-9 change of the velocity
     changes the result by at most 25 times as much (in a 2.4 degree wedge; median 0.84 to 1.0).
  14. **Where it is not continuous.** (a) *The edge of the window:* a surface `tol + tol / 16` away is active or
     not. Inside, the two faces respond together; outside, the second is hit an instant later, a sequence, as two
     separate hits are. Measured across the edge: 35.9 % of draws differ without friction (median 0.12, at most
     0.49 of the speed) and 30.0 % with it (0.058, 0.44). It is the difference between meeting two faces at once
     and one after the other, which no rule removes; what matters is that rounding cannot cross it: a particle at
     rest sits at `tol`, `tol / 16` inside the window, and rounding moves it by about 1e-7 tol. (b) The
     conditioning limits of P19 (a pair below `1 + c = 2^-40`, a triple below `2^-20`), which the geometry decides.
     (c) Normals that agree within `2^-40` but not bit for bit, in a change that uses another direction too: the
     shared normal misses the other surface by up to `2^-40` of the change, which the next sweep reads as an
     approach: one more hit at 0 (93 of 30,000 scenes with such a copy; velocities within 9.5e-12 of the speed),
     as in Task 12 as built.
  15. **Cost per particle.** No solid near: one distance per solid for the active surfaces replaces the start
     rules' read and the first neighbour pass: 5 distance reads and one sweep per solid in a step, where the code
     as built makes 6. Per hit: one distance per solid (as built: 1 to `kMaxHitsPerStep + 1` neighbour passes and
     one read for the active surfaces), and per velocity at most 2 solves of at most `kMaxCorrectionSubsets = 92`
     subsets (1, 3 or 7 for one, two or three directions; a velocity's solve walks them twice, for the tie rule),
     and at most `kMaxFrictionOrders = 6` orders of 3 frictions. At most `kMaxHitsPerStep` hits. The move's state
     loses the list of 8 touched surfaces.
  16. **What it does not fix.** The neck of two touching balls still needs shorter steps (the same 108 calls of
     the tangent set end in `kTooManyHits`, cured by the same halvings): each zig-zag is a separate hit. Four
     moving faces within a few tol of each other but not at one rest point are hit one after the other within
     nanoseconds: 13 of re-review 2's 283,929 moving corners still fail at every step length (as built: 348). The
     record's solid and the solid a failure names are indexes, so they follow the array. End positions differ
     between array orders by rounding (208 of 40,000 scenes, at most 3.4e-9 tol): the position move of P19 reads
     the solids in array order. P5's crease, and turning solids (Task 13), are as before.
  17. **Measured** with the Python model of P19 item 10, extended with this rule and with the start move as built,
     double precision, the operation order above. Against the code as built on the tests' own inputs: every-scene
     paths (241,046 modelled) `kTooManyHits` 478 to 476, `kSecondReactingSolid` 109 to 109, squeezed 2,991 to
     2,916; every-scene near starts (163,395) 462 to 458, 338 to 338, 3,809 to 3,290; free paths and kinds
     identical, which checks the model; the other generated sets (188,709 calls) keep every outcome. Re-review 2's
     corners: fixed, `kTooManyHits` 16 to 5, all cured by two halvings; moving, 376 to 18. Step then step: 60,000
     wedges, slivers, corners, cusps and moving corners, up to four steps each, 0 failures after the first step.
     Of 166 analytic tests the model gives the built result in 112 and a different one in 42 (29 in the hit count
     or the named solid only, 3 in the record or the reaction only, 10 in a velocity or the outcome); 7 are not
     modelled (boxes, spin), and the 5 of many steps hold with one hit per step.

Plan decision added 2026-11-16 for a cylinder's rim and for flush solids (Task 12a review I2, coordinator ruling 44;
built in Task 12b; the paper states it):

- **P21 A cylinder's rim at its true distance, and the touching rule of every bounding surface.** Words: a *bounding
  surface* of a convex solid is a surface the whole solid lies behind: a face plane of a box, a cylinder's wall taken
  as unbounded along its axis, the plane of a cap; a particle *rests on* a bounding surface when its gap to it is in a
  window around 0 and its path does not sink into it; the *rim search* is how a sweep finds its first hit beside a rim
  of a cylinder that does not turn. Why: P15 finds a cylinder's first hit on the cylinder grown with sharp edges,
  whose corner beside a rim lies up to `Sqrt(2) tol` from the rim, while the distance (P14) is the true one. Task
  12a's review (I2) measured what follows from the two shapes: a particle that slides down a drum at the rest gap and
  reaches the seam with a second drum of the same radius is hit on the lower drum's rim 1.41 tol from it, with a
  normal at 45 degrees, and leaves at `{0.5, 0, -0.5}` with half its kinetic energy; so is a particle that slides
  along a floor onto a puck lying flush in it; and the rim move that brings such a hit to the rest gap (P15, Task 12a)
  drags the particle to 0.586 tol of a floor beside the rim. On 220,000 generated paths past a cylinder (item 15) the
  rule as built reports 48.9 % of its hits farther than `tol + tol / 1024` from the cylinder and 13.9 % on paths that
  never come within tol of it, where the response removes 0.24 of the speed on average and up to 0.71. A box has the
  seam's half of this (item 8). This decision amends P2, P15 and P16. The rim move, its exception for the stepping
  search and the wider window of the active surfaces after it (Task 12a as built, in no decision) are withdrawn.
  1. **The touching rule of a bounding surface.** When the touching rule on the distance (P2 for a box, P16 for a
     cylinder, both unchanged) finds the start outside, every bounding surface other than the one that gives the
     distance is read, in the solid's own axes at the pose of the sweep's start (`p0`, `u` of `ComputeOwnPath`). Box:
     each face plane other than the face of the start's distance, `gap = sign * p0.axis - offset - tol` and `gap_rate
     = sign * u.axis` (the expressions of `FindEntryThroughFacePlanes`). Cylinder: only when the start's distance
     reports a rim (`kLowRim` or `kHighRim`, so `wall_distance > 0` and `cap_distance > 0`; elsewhere the wall or a
     cap gives the distance and P16 has ruled on it): the wall, `gap = rho0 - radius - tol` and `gap_rate = (p0.x *
     u.x + p0.y * u.y) / rho0`, and the cap on the start's side, `gap = Abs(p0.z) - half_height - tol` and `gap_rate =
     cap_sign * u.z`. The start rests on that bounding surface, and the whole sweep is free of the solid, when `-tol /
     8 <= gap <= tol / 16` and `gap_rate * duration >= -tol / 64` (`kBoundingSurfaceRestingDepthShare = 0x1p-3`,
     `kTouchedSolidRestingGapShare`, `kBoundingSurfaceSinkShare = 0x1p-6`), the same before and after a response. In
     P15's words: the time a path spends behind a bounding surface it rests on is empty, so the intersection is. Why
     these numbers. Below, `tol / 8`: a particle at rest is within `tol / 128` of the rest gap after the settle and
     comes at most `tol / 16` nearer after a response (P20 item 9); an eighth holds both. Above, `tol / 16`: the
     touched window. The sink, `tol / 64`: what each response may leave (`allowance * time left <= tol / 64`, P20 item
     7), so a velocity that a response left never fails it, and neither do coplanar faces whose normals differ by
     rounding, or by up to `tol / (64 * path length)` (1.5e-10 rad over 0.1 m at `tol = 2^-30` m). The surface that
     gives the distance keeps its own rule so that nothing changes for a flat solid, a plate, a domain side, a sphere,
     or a start on a box's or a cylinder's face (their tests keep their bits).
     **Amended 2026-11-17 (item 16 b):** this test now comes before the touching rule on the distance, whatever the
     sign of the distance's gap.
  2. **Why it is safe.** The solid lies behind `b`, so `d >= gap_b + tol` at every point: a box's distance is its
     largest face gap, and a cylinder's is at least `Max(wall_distance, cap_distance)`. Along the path `gap_b(t) >=
     gap_b(0) + gap_rate * t`: a face plane's gap is linear in time; `rho` is the length of a point moving on a
     straight line, convex in time; `Abs(z) >= cap_sign * z`, linear. So through a sweep that rests, `gap_b >= -tol /
     8 - tol / 64` and the path stays `55 / 64 tol` or more from the solid. Every sweep reads its start afresh, so
     this holds for the step whatever the number of sweeps; the settle then lifts the particle to the rest gap, and
     the end check (`tol / 2`) is not at stake. What it gives up: a path that the true distance would stop up to `9 /
     64 tol` inside the rest gap is not hit, and a ledge lower than `tol / 8` between two solids is not seen.
     Measured: 67,288 of 202,997 swept paths rest on a bounding surface; the smallest true gap along them is `-0.126
     tol`.
  3. **A cylinder's first hit** (a cylinder that does not turn; own axes at the pose of the sweep's start; P15's two
     intervals unchanged). (a) The touching rule on the true distance (P16): a hit at 0, free, or outside. (b) Outside
     and beside a rim: item 1 on the wall and on the cap; free when the start rests on either. (c) The entry into the
     cylinder grown with sharp edges: the later of the two intervals' entries, when it is no later than either exit
     and no later than `duration`; otherwise free, as built. A start inside both intervals (P15's rim corner) has the
     entry 0: the branch that read its rate is withdrawn, (e) does its work. (d) The entry is the hit when its
     position `p0 + u * entry` is not beside a rim (`wall_distance <= 0` or `cap_distance <= 0`, as
     `ComputeDistanceToCylinder` computes them there): on the wall and on a cap the sharp grown surface is the true
     one, and the hit time keeps the bits it has today. (e) Beside a rim, the rim search from the entry (item 4). The
     reported distance, normal and face are still `ComputeDistanceToSolid` at the hit position and the pose of the
     hit's time (`ToFirstHit`).
     **Amended 2026-11-17 (item 16 b):** (b) comes before (a).
  4. **The rim search.** Samples at `elapsed`, the first at the entry, each `ComputeDistanceToCylinder` at `p0 + u *
     elapsed` in own axes: `gap = d - tol`, `gap_rate = Dot(n, u)` with `n` the normal of that read. In this order:
     `gap <= 0` is a hit at `elapsed`; `gap_rate >= 0` is free (the gap is convex along the path and never closes
     again); `advance = gap / -gap_rate`, and `elapsed + advance > duration` is free (closing at its present rate,
     which only slows, the gap does not reach 0 in the sweep); `gap <= kRimHitGapShare * tol` (`0x1p-10`) is a hit at
     `elapsed`; after `kMaxRimAdvanceCount = 16` advances, a hit at `elapsed`; otherwise `elapsed += advance`. The
     advance is P18's second-order advance with a gap curvature bound of 0, which is what a solid that does not turn
     has; it is written as the quotient, not through `ComputeSecondOrderAdvance`, whose square of a gap rate below
     1e-154 m/s underflows to an advance of 0. No pose is computed in the search and no `Sin`. Why `2^-10`: a hit is
     reported while the true distance is at most `tol + tol / 1024`; a path whose line stays outside is hit only when
     it passes within `tol / 1024`, and then loses at most `Sqrt(2 * 2^-10) = 0.044` of its speed (measured 0.0443); a
     finer share is below the rounding of a distance where the tolerance is at its floor of `2^-40` of the scene (one
     rounding unit of a coordinate is `2^-12 tol` there). Measured on 60,000 paths per share: `2^-6` gives 1.15
     advances on average and at most 3, a hit up to 0.016 tol out and up to 0.17 of the speed removed on a graze;
     `2^-10` gives 2.0, at most 5, 0.0011 and 0.044; `2^-14` gives 2.8, at most 8, 0.0011 (rounding) and 0.011. At a
     graze: a path tangent to the round rim closes the gap by about 4 per advance (the gap rate halves), so from the
     corner's `0.414 tol` it is within the hit share after 5 advances, or the gap has stopped closing and the path is
     free; the limit of 16 is three times that and is never reached (item 15). At the limit the search reports the hit
     where it stands, the safe side: no path is let through.
  5. **Why no hit is missed and none is reported far from the surface.** `g(t) = d(p0 + u t) - tol`, with `d` the true
     signed distance to a convex solid, is convex in `t`. (i) The cylinder grown with sharp edges contains the true
     grown shape `{g <= 0}`, so before the entry of item 3 (c) `g > 0`, and a path that never enters is free: this is
     P15's interval form, the only place where a path is declared free of the sharp shape, as built. (ii) At the
     entry, on the wall or a cap proper, `g = 0`: the hit, as built. (iii) Beside a rim `0 <= g <= (Sqrt(2) - 1) tol`
     at the entry, and from a sample with `g > 0` and `g' < 0` convexity gives `g(s) >= g + g' (s - t) > 0` for every
     `s` before `t + g / -g'`: no advance passes a hit. With `g' >= 0`, `g(s) >= g > 0` for every later `s`: free,
     exactly. (iv) A hit is reported at a sample only when `g <= 0` there, or `0 < g <= tol / 1024` with the gap
     closing fast enough to reach 0 within the sweep. Rounding: a computed distance is off by at most `tol / 128`
     (`contact/contact_tolerance.h`), so the path stays farther than `tol - tol / 64` before the hit, P18's wording
     and the tests' allowance; measured at the tolerance's floor the true gap at a hit is between `-0.0023 tol` and
     `0.0020 tol`. A gap rate that rounding turns from just below 0 to 0 frees a path whose gap could close by at most
     `2^-50 * |u| * duration <= 2^-10 tol` (within `MoveParticle`'s Requires): still outside. The module's rule "never
     find a candidate and then validate it; compute intervals and intersect" holds in this form: no hit is ever
     rejected by a second, separately rounded test. The intervals alone decide that a path misses the sharp grown
     shape; after the entry the path is either hit there or searched onward from there, each advance proved free by
     convexity, and the search ends free only by the sign of the gap rate or the end of the sweep. The test of item 3
     (d) chooses between two ways of going on, both of which keep the path: on the edge between them (`cap_distance` a
     rounding unit either side of 0) the first gives the entry and the second a gap of rounding at its first sample,
     the same hit.
  6. **What a rim hit reports.** `at_hit` is P14's rim branch at the hit position: face `kHighRim` or `kLowRim`,
     normal `(wall_distance * radial + cap_distance * cap_sign * e_z) / d`, distance in `[tol, tol + tol / 1024]` up
     to rounding. `ApplyHit` moves the particle and the clock by `ToFirstHit`'s own expressions and nothing moves it
     afterwards, so the hit surface that `FindActiveSurfaces` lists and the surface the next sweep's touching rule
     reads are `ComputeDistanceToSolid` at one point and one pose, for a rim as for a wall: step 2 of
     `FindFirstHitOfSweep`'s proof (Task 12a: the sweep's rate equals minus the response's own target, bit for bit)
     holds without its two clauses on the rim move. The hit lies inside the touched window (`tol / 1024 < tol / 16`),
     so the next sweep rests on it by the touched rule; as built the rim move put it there. The normal: the sample of
     the hit is at most `tol / 1024` short of the contact along the normal, which turns the normal by at most 0.044
     rad on a graze and by 1e-6 to 2e-4 rad on the tests' paths (velocities within 1e-5 of the exact ones).
  7. **What goes, entirely.** The rim move (`ApplyHit`'s step 3: the move, the second read of `at_hit`, its pose);
     `ComputeIsRimHitToMove`; the `window_extension` parameter of `FindActiveSurfaces` and the sum behind it; the rim
     corner's branch of `FindCylinderHit` with its read of the surface's velocity; P15's sentence that a
     stepping-search hit is not moved, and with it the question of how `ApplyHit` knows which search found a hit
     (review M6) and the missing test of a rim move between tol and `tol + tol / 16` (M3); the order dependence the
     review could not rule out (a rim hit tied in time with another solid's hit moved the particle in one array order
     and not in the other): no hit moves the particle. Nothing of the three stays. What else gets simpler: `ApplyHit`
     has three steps (limit, move to the hit, respond); a rim hit beside a floor needs no rule of its own (item 9);
     the tests' reference bound for a cylinder hit falls from `Sqrt(2) tol` to `tol + tol / 1024`; the glossary loses
     `rim move`, and `rim corner` becomes a word of the rim search only.
  8. **Boxes.** A box's distance is its largest face gap, so the box grown by tol has sharp edges by definition, its
     first hit and its distance agree, and an edge hit reports the face whose plane the path crosses last, at `d =
     tol`: there is no rim move and nothing to reconcile. What it has, measured: (a) *The seam.* At a flush seam (two
     boxes stacked into one side, boxes side by side in one floor, a box set flush in a floor) the particle's gap to the next box's
     coplanar face is rounding of either sign. When it is 0 or less the path counts as behind that plane, the entry is
     the crossing of the next box's side plane, and the hit has the side's normal: the particle stops dead (`{0, 0,
     -1}` to `{0, 0, 0}` down two stacked boxes; 2,281 of 4,000 generated seams of boxes side by side, 57 %). Item 1 on every face
     plane other than the one that gives the distance removes it: 0 of 4,000, each velocity kept. (b) *Boxes side by side,
     misaligned* by 1e-15 to 1e-9 rad about the seam: 1,610 of 4,000 lose speed as built, 163 with item 1, those that
     climb more than `tol / 64` during the sweep: a real, if tiny, ramp met on its side face. (c) *The edge itself.* A
     path that clips a grown edge is hit with a face normal although it never comes within tol of the box in a
     straight line: 2,719 of 71,286 paths aimed within 2.5 tol of an edge (3.8 %), which lose 0.53 of their speed on
     average; and a step up of `tol / 4` between two such boxes stops a particle. (b) and (c) are what a round edge would
     cure. For that the box's distance must become the straight-line distance beyond its edges and corners, in
     `ComputeDistanceToSolid`, which other modules read; the rim search then applies as it is (that distance is
     convex). It is not decided here: an owner question.
  9. **P17, P20 and the end check.** *P17.* The intervals, item 1 and the rim search read the cylinder at the pose of
     the sweep's start, as P17's bound allows; the true shape differs by less than `tol / 4` over the step (Task 12a
     review I1, ruling 45). A rim hit then reports a distance within `tol / 1024` of tol plus that slack, as a wall or
     cap hit of a tilting cylinder already does (as built the rim move hid the slack for rim hits alone, by moving to
     tol at the true pose). The gap rates of item 1 and of the search are measured from `linear_velocity`, in own
     axes: the tilt's surface speed, at most `tol / (8 T)`, is part of the same slack. A start in the touched window
     still reads the surface's true velocity (`ComputeStartCase`, unchanged), which is what Task 12a's fix of the
     sinking cap needs. One test changes for this reason: `SlowlyTiltingCylinderRimCornerStart`, a particle at rest
     1.27 tol from a rim that rises 0.019 tol during the step, is no longer hit at 0; its true distance stays above
     1.25 tol. A sweep that rests on a bounding surface of a tilting cylinder stays `55 / 64 - 1 / 4 = 39 / 64 tol` or
     more from the true shape. The model has no tilt: Task 12b's generated set with tilts up to 0.99 of the bound must
     still show 0 repeat hits and 0 failures (a rim hit lands like a wall hit, whose argument is unchanged: a start
     gap above `tol / 16` from the slack needs more than half the step to build, and less than half is left to close
     it). *P20.* The window of the active surfaces is `tol + tol / 16` at every hit again. A rim hit is at most `tol +
     tol / 1024` away, inside it, and a surface the particle rests on beside the rim stays in the response because the
     particle is not moved: in Task 12a's `RimOnFloorSweepingParticle` with the cylinder rising at 9e-4 m/s the rim is
     hit 1.0002 tol away with the floor at exactly tol, one joint response over both, and the particle stays on the
     floor (as built: lifted to 1.11 tol). With the scene's own 1e-3 m/s the round rim passes 1.069 tol from the
     particle and does not hit it: the hit as built was the sharp corner's. *End check.* Nothing new ends below `tol /
     2`: a rim hit is at tol or more up to rounding, a resting sweep at `55 / 64 tol` or more, and the settle is
     unchanged.
     **Amended 2026-11-17 (item 16 a):** a solid passed at a bounding surface's level is not an active surface.
  10. **Turning solids.** A turning cylinder (above P17's bound) keeps the stepping search with P18's advance on the
     true distance, and its hit, on a rim or not, is used where the search stopped, between tol and 1.25 tol:
     unchanged, only there is no rim move left for it to be exempt from. The two searches now differ in three numbers:
     the hit gap share (a quarter against `2^-10`: Task 13's carry, not the touched window, keeps a turning solid from
     being hit again, so its hit may land anywhere up to 1.25 tol), the gap curvature bound (`K` against 0) and the
     limit (128 advances and a close call against 16 and a hit). A turning solid has no touching rule, so none for its
     bounding surfaces either.
  11. **Module 1b (mesh solids).** Its plan already meets a triangle at its true distance, by closed forms: a face
     piece, three edge pieces (cylinders of radius tol) and three vertex pieces (spheres), each an interval; that is
     two quadratics where a cylinder's rim is a quartic, so 1b needs no search, and nothing here changes its pieces.
     Three things do change for it. (a) Its row H6 ("step 4 has one place where a hit position may be moved before the
     response: the rim move") no longer holds: module 1 moves no hit. Its resting sweep (half the radius, then
     `RestingSweepHit_IsMovedToTheRestGapBeforeTheResponse`) is the pattern this decision withdraws, with its measured
     cost: a move before the response leans away from a surface the particle rests on, which then falls out of the
     window and is hit again (Task 12a: `kTooManyHits` under a rim sliding over a floor, cured only by widening the
     window by the move). 1b should instead give each triangle's plane the touching rule of a bounding surface, which
     is what its Review Focus 2 asks for (coplanar triangles of one flat face are flush seams by construction) and
     needs no second radius and no move; `ComputeIsRestingOnBoundingSurface` takes a gap, a gap rate and a duration,
     no shape, for that reason. A mesh is not convex, but each triangle lies in its plane, so the rule holds per
     triangle. (b) If 1b ever wants a search (a shape with no closed form), the rim search is safe for any convex
     piece from any point outside it; what it must measure is the limit: the entry into a triangle's sharp grown prism
     lies up to `tol / sin(a / 2)` from a vertex of angle `a`. (c) Until the owner question of item 8 is answered, a
     primitive box has sharp grown edges and the same box as a mesh has round ones.
  12. **Cost per sweep.** A cylinder the path does not reach (nearly every sweep): as built, plus item 1's two tests
     (three comparisons and one product each, one division for the wall's rate) for a start beside a rim only, minus
     the rim corner's branch (a pose and a solid velocity). A hit on the wall or a cap: one read of the two distances
     at the entry (one square root) more. Beside a rim: one distance read in own axes per sample (two square roots,
     four divisions, no pose), 1 + the advances: 2.0 advances on average, at most 5 measured, at most
     `kMaxRimAdvanceCount = 16`. A rim hit no longer pays the rim move's pose and distance read. A box: one test of
     item 1 per face plane, at most five.
  13. **The flush scenes, as tests** (`tol = 2^-30` m, friction 0, both velocities equal; Task 12b gives the starts).
     Two stacked drums of one radius, down the seam at the rest gap: no hit, velocity `{0, 0, -1}` bit for bit, end at
     tol from the lower drum (as built: one rim hit, `{0.5, 0, -0.5}`); the same from one rounding unit and from `tol
     / 32` inside the rest gap. A puck flush in a floor, onto it and off it: no hit, `{-1, 0, 0}` and `{1, 0, 0}` bit
     for bit, end at tol from both (as built: `{-0.5, 0, 0.5}`). A drum standing on a floor: along the floor into the
     groove, one hit on the wall at `0.5 - tol` s, velocity 0, end at tol from both; down the wall into the groove,
     one hit on the floor at `0.7 - tol` s, velocity 0; around the groove, no hit: all three as built, guards. Over a
     rim: falling `tol / 2` beyond the wall, one hit `Sqrt(0.75) tol` above the cap's plane, velocity `{Sqrt(3) / 4,
     0, -1 / 4}` within 1e-6 (as built `{0.4, 0, -0.2}`, from the corner at 1.118 tol); from the side `tol / 2` beyond
     the cap, `{-1 / 4, 0, Sqrt(3) / 4}`; sliding off the cap past the rim and up the wall past the rim, one hit at 0
     on the face, then free, end `Sqrt(65) tol` from the rim; carried around the rim in 64 steps of `tol / 2`, cap to
     wall and wall to cap: one hit per step, faces in the order cap, rim (3 steps), wall, end distances between tol
     and 1.13 tol. A rim crest 0.41 tol below a floor's surface: no hit (as built: a rim hit at 1.41 tol and a rim
     move of 0.41 tol toward the floor).
  14. **What it does not fix.** A box's edges, its steps between `tol / 8` and tol and its misaligned seams (item 8 b
     and c). Seams between solids that turn together: a turning solid has no touching rule. Flush faces of two solids
     whose normals differ by more than `tol / (64 * path length)`: a cylinder is then hit at its true distance, with a
     small lean; a box on its side face. A graze within `tol / 1024` of a rim is a hit that removes up to 0.044 of the
     speed, where an exact rule would remove nothing at exactly tol. P17's slack of `tol / 4` for a cylinder that
     tilts within its bound, now the same on a rim as on a wall. The settle's dependence on array order (review M2)
     and the neck of two balls (P20 item 16) are as before.
  15. **Measured** with a Python model in double precision, the operation order above: the cylinder's distance (P14),
     the first hit as built and by this decision, the box's first hit, and for whole moves the model of P20 with both.
     Reference: the gap along the path in 60-digit arithmetic, exact for the doubles of the path, its smallest value
     by golden section on the convex gap. (a) 220,000 paths past one cylinder in its own axes, where a moving cylinder
     is the same sweep at the relative velocity: `radius / tol` and `half_height / tol` each from `2^4` to `2^40`, the
     larger about 1 m; 35 % through the rim's neighbourhood at every angle, 15 % down the wall at the rest gap and 15
     % in along the cap's plane (offsets 0, a few rounding units, up to `tol / 8`, tilts to 1e-3), 15 % tangent to the
     round rim or the wall, 10 % starting in the rim corner, 10 % anywhere; 30 % with the touched rule. 17,003 start
     deeper than tol inside and are not swept. This decision: 102,403 hits and 100,594 free; missed hits 0; hits
     reported farther than `tol / 1024` plus rounding 0, later than the surface 0; 57,850 rim searches with 0 to 5
     advances (7,827, 14,796, 15,990, 9,315, 8,287, 1,635; mean 2.0), none at the limit; of the 56,840 hits as built
     that enter on the wall or a cap, 56,726 have the same time bit for bit and 114 now rest on a bounding surface. As
     built: 140,412 hits, 68,603 farther than `tol + tol / 1024` (up to 1.4143 tol), 19,534 on paths that never come
     within tol. (b) The test `GeneratedRimEdgeSweeps`' own 50,000 draws for each of its two cylinders along z: 50,000
     hits, 1 to 3 advances (7,189, 32,918, 9,893), the true gap at the hit at most `0.00098 tol`; as built all 50,000
     are at `Sqrt(2) tol`. (c) Whole moves, 4,000 per set, fixed and moving, world axes and tilted: 0 failures as
     built and by this decision in every set. Speed lost at a seam, as built against this decision: boxes side by side 2,281 to
     0; the same misaligned 1,610 to 163; stacked drums 1,731 to 0; a puck in a floor 1,173 to 0. A lone cylinder with
     half the paths aimed within 2 tol of a rim: 3,224 hits to 3,156, rim hits 258 to 190, no hit farther than `tol +
     tol / 512`, no second hit. A rim touching, over or sunk into a floor, at rest and sweeping: same failures (none),
     31 rim hits to 17. Six steps in a row across each seam, 500 runs per set: 0 failures, 0 runs lose speed (as built
     282, 217 and 149 of 500). (d) Not modelled: spin, tilt within P17's bound, turning solids, friction between
     differently moving solids at a rim.
  16. **Amended 2026-11-17 (Task 12b review I1): a solid the particle is level with is read by the response as the
      sweep reads it.** Words: a point is *level with* a bounding surface when its gap to it is in the window of item
      1, `-tol / 8 <= gap <= tol / 16`, the gap alone, whatever the velocity; a solid that does not turn *is passed at
      a bounding surface's level* when the point is level with one of its bounding surfaces other than the one that
      gives the distance (a box: any other face plane; a cylinder: its distance reports a rim and the point is level
      with the wall or with the cap's plane). The gap in this decision as built: items 1 to 3 free a sweep, but the
      start's response and the joint response (P20 item 1) still listed such a solid by its distance, and the touched
      rule read that distance before item 1. Beside a box the distance is a side face's plane continued above the top,
      1.03 tol away while the edge is 1.43 tol away; beside a cylinder it is the rim, whose normal leans across the
      seam. So a step that starts within `tol + tol / 16` of the next solid loses its speed: a dead stop for boxes, a
      deflection for drums and pucks (stacked drums from 0.2 tol above the seam: `{0.19, 0, -0.96}`), in a share of
      crossings equal to the window over the step length (measured in the model, as in the review: 205 of 400 at steps
      of 0.7 tol for drums and pucks, 35 of 400 for boxes). Item 15's six steps in a row could not see it: each step
      was 0.05 to 0.1 m long, about 10^8 tol, and each run started 0.01 to 0.15 m before the seam, so a step began
      inside a window of 0.06 to 0.36 tol about once in 10^9 crossings. Two changes. (a) **Active surfaces (amends P20
      item 1 and item 9 above).** A solid that is passed at a bounding surface's level contributes no active surface,
      at the start and at a hit on another solid; the surface a sweep hit is active as before, wherever the particle
      is. Not the bounding surface in its place, as the review suggested: beside a lone box, level with its top, that
      plane has nothing under it, and a particle falling there would be held up at every step for as long as it stays
      in the sixteenth of a tolerance beyond the grown edge (with the solid left out it falls, `{0, 0, -1}` kept:
      measured). The test reads gaps only, so the active surfaces still depend on the position alone, sorted by value
      and free of the array's order, and nothing new is charged (P5 unchanged: a solid left out is charged nothing;
      coincident surfaces of other solids are listed and dropped as before). (b) **The sweep (amends items 1 and 3).**
      Item 1's test of the other bounding surfaces comes before the touching rule on the distance, not after that rule
      has found the start outside, and whatever the sign of the distance's gap: a start that rests on such a surface
      is free of the solid; otherwise P2 or P16 rules as before, the touched rule included. Why before: after the
      start's response on a floor, the touched rule read a box's side plane 1.03 tol away as rested on and approached,
      a hit at 0 and a dead stop; with (a) alone the pressed rows of the multi-step runs stay at 205 and 35 of 400.
      Why whatever the sign: a particle riding `tol / 32` under the rest gap met the same continued plane as the
      distance's face inside its own window (a dead stop as built). Safety is item 2's, which never used the
      distance's gap: a sweep that rests stays `55 / 64 tol` or more from the solid. Step 2 of `FindFirstHitOfSweep`'s
      proof becomes: a solid within the touched window at a sweep's start was an active surface of the latest
      response, or is passed at a bounding surface's level; the second is free when the sweep rests on that surface,
      and otherwise may be hit at 0 on its distance's face, one more hit, never a missed one (the response then lists
      it as the hit surface). The bit-for-bit step is unchanged for every surface that is listed; a bounding surface
      is never listed, and its sink test has the room of `tol / 64`, which no response is asked to keep. The start
      move and the settle (P19) read the distance alone and move only from `d < tol`: a particle 1.03 tol from a
      continued plane is not moved, as built; ruling 37 is not touched (the start rules reject on `d < tol` alone).
      What changes besides the seams: a start on a box's grown edge pushing along the other face's level is free
      (`BoxRestingOnEdge`: it was a hit at 0 on `kHighX`); a particle pressed slowly onto a rim near the wall's or the
      cap's level gets no response while it sinks less than tol in the step (carried around the rim in steps of `tol /
      8`: 62 hits in 64 steps, least distance before the settle 0.946 tol; in steps of `tol / 2`, unchanged); a
      particle falling beside a rim, level with the cap's plane, is hit at the rim's true distance and no longer
      answered at the start 1.048 tol away (`{0.2969, 0, -0.0976}`, the exact value, where the start's response gave
      `{0.2847, 0, -0.0890}`). Measured in the model, the decision as built at `9792eae3` against this amendment: the
      review's six starts and their pressed and rough variants: velocity kept bit for bit, or `{1 - 0.3 * 0.16, 0, 0}`
      with friction 0.3; five kinds of seam, steps of 0.3, 0.7 and 3 tol, pushes 0 and 0.16 m/s, friction 0 and 0.3,
      400 offsets in each of 45 rows: runs that lose speed as built 400, 205 to 224 and 48 of 400 for drums and pucks
      and 83, 35 and 8 for boxes, amended 0 of 400 in every row, with exactly one hit per pressed step and none
      otherwise; 1,000 generated multi-step runs per set with starts 1.5 to 6 tol before the seam, steps from `tol /
      4` to `2^10 tol`, fixed or moving together, tilted axes: 36, 165 and 69 runs lose speed as built (boxes, drums,
      pucks), 0 amended, 0 failures; the single-step sets of item 15 (c) and 100,000 first-hit paths of item 15 (a)
      with the new order: the same counts, 0 missed hits, smallest true gap of a resting path `-0.125 tol`. **Not
      fixed by this amendment.** A rim that is active while the particle is level with neither the wall nor the cap
      (both distances below `7 / 8 tol`, the rim within `tol + tol / 16`) is still answered up to `tol / 16` early,
      with a normal that turns over one tolerance: of 600 tangent passes at 1.0005 to 1.0625 tol in steps of 0.3 to 3
      tol, 188 lose speed, at most 8.3 %. Narrowing a rim's window to the rim search's `tol / 1024` leaves 3 of 600
      and at most 0.05 %, but it meets the touched rule and P17's slack, which the model cannot measure: an owner
      question. A box beside a box that rises or sinks relative to it is still met on its side face (a real step). A
      turning solid has no bounding surface (item 10). Items 8 b, 8 c and 14 stand.
      **As built (review of the amendment, 2026-11-17), two consequences this item did not list, accepted and
      pinned by tests named as known choices.** (a) A particle resting on the last `tol / 8` of a box's grown top
      (its centre at least `7 / 8 tol` beyond the real edge) falls off: it is level with the side plane, so the box
      is left out. (b) An approach that sinks less than tol in the step, at a box's edge strip or at a rim's rest
      gap, gets no response and no reaction; before the settle the particle can be as close as 0.001 tol, never
      inside the solid, and the settle lifts it back. Measured in compiled code: 420,000 pressed seam steps with
      one hit each and a least distance of 1.0 tol; around a rim at most one hit per step and 0.930 tol before the
      settle; 650 near starts beside lone cylinders changed (602 from a hit to free, 48 lose one hit), none hiding
      a real approach. Owner question: whether a solid left out whose sweep does not rest should get the touched
      rule in the first sweep.

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

### Task 11b: `MoveParticle` start and settle near several solids (P19)

**As built (2026-11-16), where this task's text below differs:** surfaces whose normals agree within 2^-40 on every axis share one target (the larger), so colliders lying exactly on domain faces do not use up the four directions; up to 16 coincident surfaces count once, and more than four distinct directions (or a seventeenth coincident surface) is `kStartsInNarrowGap` with a message that says so. After a crossing the move is shortened to where it first comes within tol of the crossed surface, which then constrains the next iterate (the model's fresh start from the first point lost cusp scenes); surfaces that already constrain the move are swept again, grown by tol/2. The model's candidate rule and final cap test are not in the code (both redundant); a single near solid keeps the uncapped `MoveToRestGap`; the last iterate is accepted at tol - tol/128. Silent entries keep allowance 0. Known limits, each pinned by a test: a wall within the rounding window of a cusp's rest point (about 0.0014 tol for a 0.5 m ball, 128 tol at 1 km) gives a false `kStartsInNarrowGap`; nothing bounds the composing slack, so no iterate is trusted until its true distances are read. For Task 12: the solver has no empty subset, and a zero change must be a valid answer for velocities.

**Order:** after Task 11 as built (with its fix round), before Task 12. The task order becomes 9, 9b, 10, 11, 11b,
12, 13, 12a, 14 to 18. Task 12 is split (coordinator ruling): this task moves the particle (the start and the settle,
P19 items 1, 2, 5, 6 for the start, 7 and 8) and fixes the start's approach test; Task 12 changes its velocity (P19
items 3 and 4, and the squeeze of item 6).

**Kind:** numerics

**Why (Task 11 review I1):** a start within tol of two solids that meet at less than a right angle fails at every
step length, because each solid's start rule moves the particle to its rest gap along its own normal, toward the
other. On the generated two-surface set (P19 item 10) the code as built fails 8 % of the fixed draws at 85 degrees, 49 %
at 60 and 98 % at 5, as `kEndsTooCloseToSolid`, `kTooManyHits` or `kStartsInsideSolid`; the step retries twice and
stops the run. The same holds under a ball resting on a floor, where the angle shrinks to 0.

**Files:**
- Modify: `native/fluidity/contact/solid_contact_types.h` (constants `kMaxRestGapMoveInTolerances = 0x1p22`,
  `kMaxRestGapIterations = 8`, `kMaxConstrainingSurfaces = 4`, `kMaxCorrectionRefinements = 4`,
  `kVelocityPairLimitShare = 0x1p-40`, `kVelocityTripleLimit = 0x1p-20`, the last three read by Task 12; the result
  type of the solver, name decided here), `solid_contact_steps.h` (new `SolveSmallestCorrection`,
  `MoveToRestGapOfNearSurfaces`; changed `ParticleMove`, `ComputeIsApproachingSolid`, `ApplyStartRulesOfSolid`,
  `ApplyStartRules`, `ApplySettleAtEnd`), `solid_contact.h` (`MoveParticle` steps 2 and 6 and Fails),
  `contact_error.h` (`kStartsInNarrowGap` appended after `kOutOfMemory`), `contact_error_message.cc` and
  `contact_error_test.cc` (its name `starts_in_narrow_gap` and message), `move_particle_test.cc`,
  `contact_test_cases.h`, `docs/method-paper/fluidity-method.typ` (<proc-move-particle> steps 2 and 6, and a
  paragraph with a figure of the move off two faces and of the iterates in a cusp), `docs/glossary.md` (`wedge angle`,
  `shortfall`, `tangent contact`, `cusp`).

**Interfaces:**
- Consumes: Task 11 as built after its fix round: `FindSurfacesWithinTolerance` and `NearSurfaces` (one surface, or
  each closed side of a domain box), `ApplyStartResponses`, `ApplyTouchedSurfaceOfSilentMove` (a silent move is a
  touched entry without a hit, kept only when some solid responded; `touched_surface_count`,
  `kMaxTouchedSurfaces = 8`), `MoveToRestGap` (the one-surface move, kept), `CheckStartOnCommittedSide`,
  `CheckEndOutsideSolids`, `FindFirstHitOfSweep`'s closed forms (for the crossing check of P19 item 1); P18's
  subgradient argument for convex distances. Preconditions to respect: as Task 11.
- Produces: P4 and P12 amended and P19 items 1, 2, 5, 6 (start), 7 and 8:
  - `SolveSmallestCorrection` (P19 item 2), with a `refine` choice and the velocity limits for Task 12.
  - `MoveToRestGapOfNearSurfaces` (P19 items 1 and 5): the solids, the pose time, the point; it returns the moved
    point and whether a move within the cap was found.
  - Step 2 in the order of P4 amended: every solid's failing checks at the start as given; one move off the near
    surfaces of all solids; the responses solid by solid with `ApplyStartResponses` and the silent-move entries as
    built. No solid is classified twice. No move found: `kStartsInNarrowGap` naming the near solid latest in array
    order, magnitude 0; `*result` unchanged.
  - Each velocity's own allowance (P19 item 3; folded in by the coordinator, the bug found in Task 11 as built):
    `ParticleMove` keeps the running largest allowance of `for_next_step` (`ComputeLargestApproachLeftAfterResponse`
    of the `for_next_step` passed to each response, and of the velocity of each silent-move entry);
    `ComputeIsApproachingSolid` compares `for_next_step` with it and `for_this_step` with the touched list's, as now.
    Why: as built, a `for_this_step` at rest gives an allowance of 0, and the `-2^-55` m/s the first response leaves
    in a moving `for_next_step` is taken as an approach to a parallel second solid (a spurious second hit, and a
    reaction of `2.8e-17` m/s on it).
  - Step 6: `ApplySettleAtEnd` calls `MoveToRestGapOfNearSurfaces` over every solid at the end pose (P19 item 7); no
    move found leaves the position (the end check decides). Its `d >= -2 tol` assertion stays; its comment loses the
    wedge case.
  - The comments that today describe the wedge limit (`ApplyStartRules`, `ApplySettleAtEnd`, `ComputeStartCase`'s
    "deeper than one tolerance", `MoveParticle` step 2, the Idea of `solid_contact_steps.h`) say what now happens.
- **Room for module 1b (mesh solids) (A80):** the shape switch stays in `FindSurfacesWithinTolerance` (which surfaces
  of one solid are near) and in `ComputeDistanceForParticle`; `SolveSmallestCorrection` and
  `MoveToRestGapOfNearSurfaces` read only distances and normals and have no shape code. 1b adds the mesh case to
  `FindSurfacesWithinTolerance`: each triangle within tol is one surface, with the normal of its nearest feature.
  What 1b inherits: (1) the distance to one triangle is convex, so the iterates are at the rest gap from the first
  that includes every near triangle; (2) at most `kMaxConstrainingSurfaces` surfaces at once, so triangles that
  share their nearest feature (an edge or a vertex, the same normal) must be merged before the solve, or a particle
  near a vertex with six triangles fails; (3) a mesh resting on a surface makes cusps like a ball's, resolved the
  same way. 1b re-measures its generated sets against this.

- [ ] **Step 1: Failing tests** (`tol = kTestContactTolerance`, fixed solids through the origin, friction 0, both
  velocities equal, overlap 2 tol, unless stated; the wedge side is `BuildWedgeSide(normal angle, id 2)` beside the
  floor id 1, so a wedge angle `a` is the normal angle `pi - a`; "at the rest gap" is a distance within `tol / 1024`
  of tol):

  ```cpp
  // The two scenes pinned "to change in plan Task 12" by Task 11's fix round
  TEST(MoveParticle, SixtyDegreeWedgeRestingOnBoth_EndsAtRestGapOfBoth)
      // replaces SixtyDegreeWedgeRestingOnBoth_TodayEndsOffTheRestGap; scene as built (start {0,0,0} at rest,
      // T 0.1): no error, kind kNone, hit_count 0, end {Sqrt(3) tol, 0, tol} within tol / 1024 (a move of
      // tol / sin 30 deg = 2 tol along the bisector), both at the rest gap. Today: 0.8125 tol from the floor
  TEST(MoveParticle, ThirtyDegreeWedgeRestingOnBoth_EndsAtRestGapOfBoth)
      // replaces ..._TodayIsTooManyHits: kind kNone, hit_count 0, end {(2 + Sqrt(3)) tol, 0, tol} (move 3.8637 tol)
  TEST(MoveParticle, TenDegreeWedgeFallingOnBoth_EndsAtRestGapOfBoth)
      // normal angle 17 pi / 18, start {0,0,0}, both velocities {0,0,-0.981}, T 0.1: hit_count 1 (the floor; the
      // fall leaves the side), velocities {0,0,0}, end {tol / tan(5 deg), 0, tol} = {11.430052 tol, 0, tol}
  TEST(MoveParticle, VerySharpFlatWedgeRestingOnBoth_EndsAtRestGapOfBoth)
      // wedge 0.01 deg, start {0,0,0} at rest: kind kNone, end {tol / tan(0.005 deg), 0, tol} = {11459.156 tol, 0,
      // tol} within tol / 256, both at the rest gap. P19's first version refused it, below its 0.4476 deg limit
  TEST(MoveParticle, FlatWedgeNeedingMoveBeyondCap_IsStartsInNarrowGap)
      // wedge 6e-7 rad, start {0,0,0} at rest: end x 3333942 tol (within 2e-4 of tol / tan(3e-7)), both at the
      // rest gap; wedge 3e-7 rad: the rest point is 6.67e6 tol away, beyond 2^22 tol = 4.19e6 tol:
      // kStartsInNarrowGap, solid_id 2, magnitude 0, result unchanged
  // The as-built wedge scenes of contact_test_cases.h, new outcomes (wedge angle 36.87 deg, normals -0.8 apart)
  TEST(MoveParticle, WedgeStartPushedTowardFloor_MovesOffBothAtOnce)
      // replaces ..._SettleLiftsItToRestGap. Shortfalls 0.5 tol (floor) and 0.1 tol (wedge): one move of
      // {-5/6 tol, 0, 0.5 tol}; the floor responds (hit_count 1), velocities {-1,0,0}, end {-0.1 - 5/6 tol, 0, tol}
      // within tol / 1024
  TEST(MoveParticle, WedgeStartPushedIntoFloor_MovesOffBothAtOnce)
      // replaces ..._EndsInTooManyHitsOnFloor: kind kStartedInsideSolid, hit_count 2 (the floor approached, the
      // wedge started inside), velocities {-1,0,0}, end {-0.1 - 0.0005 * 5/3 - 5/6 tol, 0, tol} within tol / 1024
  // Corners and pockets
  TEST(MoveParticle, ThreeSolidAcuteCornerFallingOnAll_EndsAtRestGapOfAll)
      // floor id 1, id 2 normal {Sqrt(3)/2, 0, -1/2}, id 3 normal {0, Sqrt(3)/2, -1/2}; start {0,0,0}, both
      // velocities {0,0,-0.0981}, T 0.01: hit_count 1, velocities {0,0,0}, end {Sqrt(3) tol, Sqrt(3) tol, tol}
      // (move Sqrt(7) tol), all three at the rest gap
  TEST(MoveParticle, ThirdSolidNearOnlyAfterFirstMove_JoinsNextIteration)
      // floor id 1, the 60 deg side id 2, and id 3 through {2 tol, 0, 0} with normal -{1,1,0} / Sqrt(2): 1.414 tol
      // from the start {0,0,0} (at rest, T 0.01), 0.19 tol after the pair's move. End {Sqrt(3) tol, -1.146264 tol,
      // tol} within tol / 1024 (move 2.3052 tol), all three at the rest gap, kind kNone
  TEST(MoveParticle, DomainCornerCutBySolid_HoldsParticleOrIsStartsInNarrowGap)
      // unit domain box id 1 and a flat solid id 2 through {L tol, 0, 0} with normal -{1,1,1} / Sqrt(3); start
      // {0,0,0}, both velocities {0,0,-1}, T 0.01. L = 6: end {tol, tol, tol}, hit_count 1. L = 4: the pocket's
      // inscribed radius L / (3 + Sqrt(3)) = 0.845 tol is below tol: kStartsInNarrowGap, solid_id 2
  TEST(MoveParticle, ParallelFacesLessThanTwoTolerancesApart_IsStartsInNarrowGap)
      // floor id 1 and a ceiling id 2 at z 1.5 tol with normal {0,0,-1}; start {0,0,0.75 tol} at rest, overlap 0:
      // kStartsInNarrowGap, solid_id 2. With the ceiling at 2.5 tol and the start at tol / 2: end z tol, kind kNone
  // Tangent contacts (P19 item 1): rest points from the closed form 2 Sqrt(R tol)
  TEST(MoveParticle, BallRestingOnFloorStartInCusp_MovesToRestPoint)
      // floor id 1; sphere id 2, centre {0,0,0.5}, radius 0.5. Start {Sqrt(0.5 tol), 0, tol / 4} (0.25 tol from
      // both), at rest, T 0.01: no error, kind kNone, end {2 Sqrt(0.5 tol), 0, tol} = {46340.95 tol, 0, tol} with x
      // within tol / 256 and z within tol / 1024, both at the rest gap
  TEST(MoveParticle, BallRestingOnFloorDeepStart_MovesToRestPoint)
      // the same at production scale: sphere centre {0,0,1024}, radius 1024 (R / tol = 2^40), x* = 2 Sqrt(1024 tol)
      // = 0.001953125. Starts {f x*, 0, (f x*)^2 / 4096} (half the gap) for f = 0.1 and 0.01, at rest: end x within
      // 1e-4 of x* (the model: 0.999994 and 1.0000002, 8 iterations), both at the rest gap; f = 0.0005 (within
      // 1.8e-8 R of the contact, where 1 + Dot of the normals rounds to 0): kStartsInNarrowGap, solid_id 2
  TEST(MoveParticle, TwoBallsTouchingStartInNeck_MovesToRestRing)
      // spheres id 1 centre {0,0,0} and id 2 centre {1,0,0}, radius 0.5; start {0.5, rho / 2, 0}, rho =
      // Sqrt(tol (1 + tol)), at rest, T 0.01: kind kNone, end {0.5, rho, 0} = {0.5, 32768.0 tol, 0} within tol / 256,
      // both at the rest gap
  TEST(MoveParticle, PlateAndSphereRing_MovesOffBothAlongBisector)
      // plate id 1 at the origin, normal {0,0,1}, half thickness 0.01; sphere id 2, centre {0,0,0.26}, radius 0.5:
      // they cross on the ring r = Sqrt(0.1875) at z = 0.01 at 60 deg. Start: ring point + 0.5 tol * {Sqrt(3)/2, 0,
      // 1/2} (0.25 tol from both), both velocities {0,0,-0.0981}, T 0.01: hit_count 1, velocities {0,0,0}, end ring
      // point + 2 tol * {Sqrt(3)/2, 0, 1/2} within tol / 1024, plate at the rest gap, sphere distance in
      // [tol - tol / 1024, tol + tol / 64]
  TEST(MoveParticle, ParallelSolidAfterStartResponse_NextStepRoundingIsNotAnApproach)
      // the scene of ParallelSolidAfterStartResponse with for_this_step {0,0,0} and for_next_step {-2,0,-2}:
      // hit_count 1, reaction.solid_index kNoSolid. As built: hit_count 2 and a 2.8e-17 m/s reaction
  // The settle
  TEST(ApplySettleAtEnd, SixtyDegreeWedgeBothNearAtEnd_EndsAtRestGapOfBoth)
      // a ParticleMove with the floor and the 60 deg side touched, at {Sqrt(3)/2 tol, 0, tol / 2} (tol / 2 from
      // both): end {Sqrt(3) tol, 0, tol}. One move per surface in turn (as built) leaves the floor at 0.625 tol
  TEST(ApplySettleAtEnd, UntouchedNearSolid_IsMovedOffToo)
      // the same position with only the floor touched: end {Sqrt(3) tol, 0, tol}. Settling the touched floor alone
      // (to z tol) leaves the side at 0.25 tol
  TEST(MoveParticle, CeilingClosingFasterThanParticleLeaves_IsEndsTooCloseToSolid)
      // as built, unchanged: at the end the floor is 0.25 tol and the ceiling 1.25 tol away, 1.5 tol apart, so the
      // settle finds no move and the end check fails, solid_id 1, magnitude 0.25 tol
  // The private steps
  TEST(SolveSmallestCorrection, OneTwoAndThreeSurfaces_MatchClosedForms)
      // slack 0x1p-40. {0,0,1} and {Sqrt(3)/2,0,-1/2} (60 deg wedge), targets {1, 1}: change {Sqrt(3), 0, 1},
      // multipliers {2, 2}; targets {1, 0}: change {1/Sqrt(3), 0, 1}, multipliers {4/3, 2/3}; {0,0,1} and {1,0,0},
      // targets {1, -1}: change {0,0,1}, one surface; the three axes, targets {1,1,1}: change {1,1,1}; {0,0,1} and
      // {Sqrt(3)/2,0,1/2} (an obtuse wedge), targets {1, 1}: change {1/Sqrt(3), 0, 1}, multipliers {2/3, 2/3};
      // normals {sin 3deg cos p, sin 3deg sin p, cos 3deg}, p = 0, 120, 240 deg (a shallow pit), targets {1,1,1}:
      // change {0, 0, 1 / cos 3deg} = {0, 0, 1.0013723} within 1e-12
  TEST(SolveSmallestCorrection, PairResidualWithoutRefinement_IsAboutEpsOverOnePlusC)
      // 3000 pairs at each 1 + c = 2^-40, 2^-28, 2^-16, 2^-4, targets in [0, 1]: |Dot(n_i, w) - s_i| at most
      // 4 * 2^-52 / (1 + c) (the model: 2.0 to 2.5); with one refinement at most 2^-23 at 2^-40
  TEST(MoveParticle, PlateAcrossSharpWedge_IsNotCrossedByTheMove)
      // floor id 1, the 0.03 deg side id 2 (from a start on both the pair's rest point is 3819.7 tol out along the
      // bisector b = {cos 0.015deg, 0, sin 0.015deg}), and a plate id 3 with normal b, half thickness 600 tol (at
      // least the 1024 tol feature), centre 700 tol * b: its faces 100 and 1300 tol out. Start {0,0,0} at rest,
      // T 0.1. The segment to the pair's point crosses the plate, which joins; no point between the apex and the
      // plate is at tol from all three: kStartsInNarrowGap. Without the crossing check the particle ends past the
      // plate, 2520 tol beyond its far face
  // Generated
  TEST(MoveParticle, GeneratedTwoSurfaceStarts_SucceedDownToFiveDegrees)
      // replaces GeneratedRightAndObtuseWedgesAndDomainCorners_StartsSucceed: BuildTwoSurfaceWedge draws phi (the
      // angle between the normals) from 5 to 175 degrees instead of 10 to 90, at the same place in the draw order.
      // Expected: no failure in right and obtuse wedges and domain corners; in acute wedges no kEndsTooCloseToSolid,
      // kStartsInsideSolid or kStartsInNarrowGap, and kTooManyHits only for velocities that approach both surfaces
      // (the ping-pong Task 12's change ends; the model gives 30 % of acute draws), printed; every success at least
      // tol / 2 from both
  TEST(MoveParticle, GeneratedTangentContactStarts_Succeed)
      // generator P11, 2000 draws per scene, in this order per draw: scene (ball on floor, cylinder lying on floor
      // from Task 12a on, ball in the unit domain box's corner touching three sides, two touching balls, ball on a
      // flat solid tilted by 0.6 Next() rad), R = 0.05 + 0.45 Next() (and the second ball's), the contact's offset
      // in [-1, 1]^2, the radial direction, x = x* (0.02 + 0.98 Next()) with x* = 2 Sqrt(R_eff tol), the height in the
      // gap ((1.5 Next() - 0.5) of it, so some starts are up to half the gap inside the floor), T from {1e-3, 0.01,
      // 0.1}, velocity at rest, slow, falling, inward at up to 2 m/s, or three NextSigned * 2 m/s: no failure but
      // kTooManyHits from velocities that approach both surfaces (the ping-pong Task 12's change ends), counted;
      // every success at least tol / 2 from every solid (the model with Task 12's change: every start succeeds)
  ```
  The expected values were computed with the Python model of P19 item 10 and checked against closed forms (the
  apex point `tol / tan(a / 2)`; the cusp rest point `2 Sqrt(R tol)`, both distances exactly tol; the neck ring
  `Sqrt(tol (2 R + tol))`; the pair solve by hand for the 36.87-degree scenes; the three-plane point; the inscribed
  radius of the pocket).
- [ ] **Step 2: Run.** Expected: the new tests fail; the replaced tests are removed with their scenes' names kept.
- [ ] **Step 3: Implement** P19 items 1, 2, 5, 6 (start), 7 and 8 and P4 amended. Comments: why one move and not
  one per solid (the push toward the other solid, with the measured failures); why the iterates are at the rest gap
  and come nearer (P18's subgradient inequality; `x_k` satisfies its own linearisation); why the cap is `2^22 tol`
  (a 256th of a cell, and every cusp's rest point); why positions need no conditioning limit and velocities do; why
  the settle reads every solid; why `kStartsInNarrowGap` is a scene error that no sweep can reach; why each velocity
  has its own allowance. The paper's step 2 and 6 text and a figure (two faces at 60 degrees with the start on both
  and the move to the point at tol from both, the old one-surface moves drawn faint; the iterates in the cusp under a
  ball) follow.
- [ ] **Step 4: Run** build, tests, gate. Mutants, each killed by the named test:
  - one move per surface in turn (the rule as built): `SixtyDegree...`, `ThirtyDegree...`, `ThreeSolidAcuteCorner...`;
  - one linear solve, no further iteration: `BallRestingOnFloorStartInCusp` (the point overshoots far along the
    floor), `ThirdSolidNearOnlyAfterFirstMove`;
  - the stop on a small step without the rest-gap test: `BallRestingOnFloorDeepStart` (in the model a start
    0.014 tol short of one surface stopped after one iterate and left the other 0.0124 tol short);
  - each iterate measured from the previous one instead of from the start: `BallRestingOnFloorStartInCusp` (the
    iterates drift away from the start; x beyond tol / 256 of the rest point);
  - the iterates not clipped to the cap: `BallRestingOnFloorDeepStart` (from 0.01 x* the first iterate goes about
    `2 tol R / x = 100 x*` out, and 8 iterations end 9 % beyond the rest point, measured);
  - an angle limit for positions (the first version, 0.4476 degrees): `VerySharpFlatWedge...`,
    `BallRestingOnFloorStartInCusp`;
  - the composing slack left out for positions: `BallRestingOnFloorDeepStart` (at 0.1 x* the pair's rounding,
    0.01 tol, refuses the candidate);
  - the cap `2^21` or `2^23`: `FlatWedgeNeedingMoveBeyondCap` (6e-7 rad refused at `2^21`; 3e-7 rad accepted at
    `2^23`);
  - no crossing check: `PlateAcrossSharpWedge_IsNotCrossedByTheMove`;
  - the depth tested after the move: `DeepOverlap_IsStartsInsideSolid` (as built);
  - one allowance for both velocities: `ParallelSolidAfterStartResponse_NextStepRoundingIsNotAnApproach`;
  - the settle moving off one surface at a time: `SixtyDegreeWedgeBothNearAtEnd`;
  - the settle over the touched solids only: `UntouchedNearSolid_IsMovedOffToo`.
  Expected survivors, with their reason in the report: the settle call removed (M47): with one move at the start no
  path of the tests ends nearer than tol to a solid by more than rounding (`SixtyDegreeWedgeBothNearAtEnd` covers the
  settle at the function level); the responses reading the normal after the move (the same for flat faces).
- [ ] **Step 5: Commit.** Subject: `Move a start or an end off several near solids at once`. New names:
  `SolveSmallestCorrection`, `MoveToRestGapOfNearSurfaces`, the solver's result type, `kMaxRestGapMoveInTolerances`,
  `kMaxRestGapIterations`, `kMaxConstrainingSurfaces`, `kMaxCorrectionRefinements`, `kVelocityPairLimitShare`,
  `kVelocityTripleLimit`, `ContactError::kStartsInNarrowGap`.

### Task 12: `MoveParticle` slide rules between surfaces

**As built, and superseded by P20 (2026-11-16), where this task's text below differs:** the task was built as the
text says (one response per surface, then one frictionless change over the active surfaces), and two fix rounds
added the neighbours' own responses in repeated passes, a rule for coincident twins with touched entries of their
own, and a frictionless change once a step has four hits. Plan decision P20 replaces all of that by one joint
response per hit; the third fix round builds it (brief: `task-12-round3-brief.md` in the task's working folder).
What no longer holds below:
- *"Each start response is followed by it"*, and the private step "called after `ApplyResponseAtSurface` by
  `ApplyHit` and by the start responses": the start has one joint response over the active surfaces at the moved
  start, and a hit along the path has one (P20 items 9 to 11). There is no response per surface followed by a change.
- *The touched-surface list* (the bullets "Amended 2026-11-15 (Task 10 as built ...)": the largest allowance of all
  entries, the entry a removal raises, `entry.solid_index`): the list is removed; after the first response of a step
  every solid has the touched rule with the largest allowance of the responses so far (P20 item 9).
- *"Newest"*: a sliver sticks to the active surface approached fastest, a squeeze names the solid approached
  fastest, and an allowance belongs to the response, not to a newest entry (P20 items 7 and 8).
- *"Friction acts only in each surface's own response"* (P19 item 3): friction is part of the joint response
  (P20 item 5).
- *"No change found: `kSqueezedBetweenSolids` when a pair of active surfaces closes"*: the test of the closing speed
  is dropped; the stick's own check decides (P20 item 7).
- *The hit counts of step 1's tests*: a hit is one joint response, so the tests that expect 2 or 3 hits for one
  wedge or corner expect 1 (`AcuteWedge`, `AcuteWedgeOnReactingFloor`, `GrooveSlideAlongEdge`,
  `NarrowWedgeAtRestPoint`, `SliverStop`, `ThreeSolidAcuteCornerInto`, the cusp, neck and ring tests, the closing
  wedges, and "from step 12 on hit_count 2 per step" of the V-groove tests, now 1). `RightAngleCorner`,
  `SlideRemovalOnReactingFloor`, `ObliqueCornerResidue` and `CeilingHitFarAboveTouchedFloor` keep theirs: their hits
  are at different moments. `CornerPushOnSecondReactingSolid` names solid_id 2 (the higher index of the two that one
  response pushes), and its second part has 1 hit. `SliverStop_ChargedToNewestSolid` becomes
  `SliverStop_ChargedToSolidApproachedFastest`: with `{-2, 1, 0}` the side is approached fastest and the floor is
  not charged; with `{-2, 1, -0.001}` the reacting floor is, `{2, -1, 0.001}`. The closing tests name the solid
  approached fastest: `OneFaceClosingAtPointEightDegrees` 1, `TwoFacesClosing...` at 1.6 degrees 1,
  `GrazingWedgeClosingFast` 2, `ClosingCeiling` 2. "1.79 degrees (1.7889 measured)" is 1.7906, the closed form.
- *Step 4's mutants* that name the newest solid, the touched entries or the start responses: the brief lists the
  mutants of the joint response.
What still holds: the active surfaces read afresh and near only, the smallest frictionless change and its
conditioning limits, each push charged to its own solid, the speed cap of 64 and the escape check, and every
velocity of the analytic tests of step 1 without friction.

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
    then fails, safely; `AcuteWedge` will meet it. (Task 11b replaced that settle with one move off every touched
    surface.)
- **Amended 2026-11-15 (P19 items 3, 4, 5 and 6, Task 11 review I1, coordinator rulings; consumes Task 11b):** the
  earlier-surface rule is one frictionless change over the active surfaces, which replaces the three-part sequence
  of design 5.5 step 5 (remove each earlier approach; the newest solid's velocity in a corner; squeeze when an
  earlier one is still approached). The bullets above still hold, read this way:
  - *Which surfaces.* The active surfaces: the near surfaces of the touched solids (hits and silent moves) at the
    current position and pose, read afresh with `FindSurfacesWithinTolerance` (`d - tol <= tol / 16`), with their
    normals there, not the normals recorded at the hits (on a ball the first hit's normal is stale after a slide
    along it). A surface the particle has left is not in the change.
  - *"Still approaching"* is tested per velocity against that velocity's own allowance (Task 11b): `for_this_step`
    against the largest of the touched list (the rule above), `for_next_step` against its running largest.
  - *The change.* `SolveSmallestCorrection` (Task 11b) with the targets `-Dot(n_i, v - u_i)`, `refine` on (up to
    `kMaxCorrectionRefinements`), pairs used when `1 + c >= kVelocityPairLimitShare`, slack the allowance. One
    surface: the removal of the bullets above. Two: the part along the common line stays (a groove), or the particle
    stops while it can still leave (a cusp). Three: the solids' velocity. Moving solids: the velocity that approaches
    none (carried along a closing wedge's apex).
  - *A sliver* (an active pair below the limit, under 1.35e-6 rad): the newest solid's velocity, a stick, charged to
    that solid (P5 amended, item 3); then the other surfaces are checked as below.
  - *Its allowance* ("each frictionless removal passes the tol/64 assertion"): `kApproachLeftAfterResponseShare`
    times `|v|_1 + |w|_1 + sum |u_i|_1`, raising the newest entry's allowance (for `for_this_step`) or the running
    one (for `for_next_step`). The assertion becomes a check: above `tol / 64` times the time left, the escape is too
    fast for the step, `kSqueezedBetweenSolids`, which a shorter step cures (P19 item 5).
  - *The speed cap* (coordinator ruling, P19 item 5): a change whose velocity is faster than
    `kMaxSqueezeOutSpeedFactor = 64` (new, `solid_contact_types.h`) times the largest of `Length` of the two
    velocities before the change and of the active surfaces' solid velocities at the particle is refused:
    `kSqueezedBetweenSolids` naming the newest surface's solid. A shorter step does not cure it: `MoveParticle`'s
    Fails lists it with the start-only failures that a shorter step cannot fix, and the step module decides what a
    persistent squeeze does (`changes-to-apply.md`, 10 Step, for the owner).
  - *The reaction* (P5 amended, to confirm with the owner): each used surface's push `m_i n_i` of the change of
    `for_next_step` is charged to its own solid; a non-zero push at a second reacting solid is `kSecondReactingSolid`.
  - *No change found:* `kSqueezedBetweenSolids` when a pair of active surfaces closes, `(n_i - n_j) . (u_i - u_j) / 2`
    above the allowance (parallel plates closing); otherwise the sliver stop. Squeezes name the newest surface's solid.
    No failure here is a scene error: `kStartsInNarrowGap` belongs to the start (Task 11b).
  - *The start* uses the same rule: each start response (Task 11b's step 2, part 3) is followed by it, with the
    active surfaces read at the moved start; newest is the last surface that responded, in array order and a domain
    box's face order. The start's ping-pong in acute wedges (Task 11b's `kTooManyHits` count) ends here.
  - *Hand-over from Task 11's review (coordinator):* a flat solid or plate spinning fast about its normal (not
    turning, P17) and moved silently at the start can be hit at time 0 by rounding in the sweep that follows, a
    harmless extra hit: its silent entry's allowance uses `ComputeLargestApproachLeftAfterResponse` of
    `for_this_step` and the solid velocity at the particle, while the touching rule's rate is measured against
    `linear_velocity` alone (`ComputeStartCase`), so the two disagree by the rounding of the spin's `omega x r`. Make
    them agree: measure the touching rule's rate against `ComputeSolidVelocity` at the sweep's start (the velocity the
    start rule uses), or give the silent entry the allowance `2^-46 (|v|_1 + |linear_velocity|_1 + |omega x r|_1)`
    (0 without spin). Prefer the first when it keeps P2's closed form unchanged (the spin's `omega x r` has no part
    along the normal, so the rate differs only by rounding); state the choice in `ComputeStartCase`'s comment.
  - The private step is one function of `solid_contact_steps.h` called after `ApplyResponseAtSurface` by `ApplyHit`
    and by the start responses (name decided here, for example `ApplySlideRule`); no shape code in it (1b inherits it
    unchanged, with Task 11b's merge of triangles that share a nearest feature).

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
  // Amended 2026-11-15 (P19 item 4, P5 amended; to confirm with the owner): each surface is charged its own push
  TEST(MoveParticle, AcuteWedgeOnReactingFloor_EachSurfaceChargedItsOwnPush)
      // AcuteWedge's scene with the floor (id 1, index 0) reacting and the wedge (id 2) not. The particle touches
      // the floor (not hit) and hits the wedge first at x 1 - (1 + Sqrt(2)) tol: {1,0,-1} (not counted). The
      // re-sweep hits the floor at 0: {1,0,0}, floor change {0,0,1}. The wedge is approached at -Sqrt(0.5) and both
      // surfaces are active, so the change is the pair's: {-1,0,0} = 1 * {0,0,1} + Sqrt(2) * Normalized({-1,0,-1}),
      // the floor's push {0,0,1}, the wedge's {-1,0,-1} (not counted). Expected: hit_count 2, velocities {0,0,0}
      // within 1e-15, reaction.solid_index 0, velocity_for_next_step_change {0,0,2} within 1e-15 (the newest-solid
      // rule gives {-0.5,0,1.5}; the whole change to the floor {-1,0,1}; the corner left out {0,0,1}), hit_position
      // {1 - (1 + Sqrt(2)) tol, 0, tol} within 2 tol
  // Rounding residue of the response (design 5.4): the test fails if the comparison is with 0
  TEST(MoveParticle, ObliqueCornerResidue_SlidesOnWithoutSqueezeOrStop)
      // two fixed flat solids, friction 0: plate A point {0,0,0} normal {0.6,0,0.8}; wall B point {-0.2,0,0}
      // normal {1,0,0}; start {0.3,0,0.4}, both velocities {-2,0,-2}, T 1. A's first response of that velocity
      // leaves an approach of exactly -2^-55 (pinned in contact_response_test.cc), below 0 but within the bound
      // 4.0e-14. The particle slides along A at {-0.32,0,0.24}, hits B, and must slide on along the corner line.
      // Expected: no error (not kSqueezedBetweenSolids), hit_count 2, velocities {0,0,0.24} within 1e-15 (not
      // {0,0,0}), end x -0.2 + tol within 2 tol, end z 0.24 within 1e-3 (a stop at the first hit would end near
      // z 0.043). Fails if "approaches again" tests `approach >= 0` instead of `approach >= -bound`
  // Amended 2026-11-15 (P19 items 3 to 6): wedges, grooves, corners and tangent contacts. Scenes of Task 11b
  // unless stated: floor id 1 and BuildWedgeSide(pi - a, 2) for a wedge angle a; tol = kTestContactTolerance;
  // friction 0
  TEST(MoveParticle, GrooveSlideAlongEdge_KeepsSpeedAlongEdge)
      // a = 60 deg, start {0,0,0}, both velocities {-1,1,-1}, T 0.01: both surfaces respond at the start; the
      // change keeps the part along the common line: hit_count 2, velocities {0,1,0} within 1e-15, end
      // {Sqrt(3) tol, 0.01, tol} within tol / 1024. The newest solid's velocity (design 5.5 as it was) gives {0,0,0}
  TEST(MoveParticle, VGrooveSlideIntoApex_ComesToRestThere)
      // faces through the origin, normals {-cos 15deg, 0, sin 15deg} (id 1) and {cos 15deg, 0, sin 15deg} (id 2): a
      // 30 deg groove opening upward. Start 0.05 {sin 15deg, 0, cos 15deg} + tol {-cos 15deg, 0, sin 15deg} (on face
      // 1) at rest; 1000 steps of T 0.01, both velocities += {0,0,-9.81 T} before each, each end the next start;
      // friction 0, then 0.5. Expected: no error in any step; from step 12 on (the model: step 9 with friction 0,
      // 10 with 0.5) hit_count 2 per step and the end {0, 0, tol / sin 15deg} = {0, 0, 3.8637033 tol} within
      // tol / 1024 in x and z, both at the rest gap; velocities after the last step 0 within 1e-15
  TEST(MoveParticle, TenDegreeVGrooveSlideIntoApex_ComesToRestThere)
      // the same with normals at 5 deg from the horizontal (a 10 deg groove): end z tol / sin 5deg = 11.473713 tol
  TEST(MoveParticle, NarrowWedgeAtRestPoint_KeepsSpeedAlongEdge)
      // a = 2e-6 rad (above the velocity limit); start at its rest point {tol / tan(1e-6), 0, tol} =
      // {999999.99 tol, 0, tol}, both velocities {-2,1,0}, T 0.01: hit_count 2, velocities {0,1,0} within 1e-9,
      // both at the rest gap
  TEST(MoveParticle, SliverStop_ChargedToNewestSolid)
      // a = 1e-7 rad (a sliver: 1 + c = 5e-15, below 2^-40); the floor reacting. Start at the rest point
      // {tol / tan(5e-8), 0, tol} = {2e7 tol, 0, tol}, both velocities {-2,1,0}, T 0.01: the side responds at the
      // start, the floor is hit at 0, and the pair is a sliver: the particle sticks to the floor, the newest solid.
      // No error, hit_count 2, velocities {0,0,0}, end at the rest point within tol / 1024, reaction.solid_index 0,
      // velocity_for_next_step_change {2, -1, 2e-7} within 1e-15 (the floor's response {0, 0, 2e-7} and the stop
      // {2, -1, 0})
  TEST(MoveParticle, FloorRisingUnderSixtyDegreeWall_CarriesParticleAlongApex)
      // a = 60 deg, the floor moving {0,0,1}; start at the apex rest point {Sqrt(3) tol, 0, tol} at rest, T 0.1:
      // no error, hit_count 2, velocities {1 / Sqrt(3), 0, 1} within 1e-15, end {Sqrt(3) tol + 0.1 / Sqrt(3), 0,
      // 0.1 + tol} within tol / 1024, both at the rest gap at the end pose. Not a squeeze
  TEST(MoveParticle, BothFacesClosingAtSixtyDegrees_CarriesParticleOut)
      // a = 60 deg, each surface moving 0.5 m/s along its own normal; the same start: velocities
      // {Sqrt(3) / 2, 0, 1/2} within 1e-15, no error
  TEST(MoveParticle, OneFaceClosingAtOneDegree_CarriesParticleOut)
      // a = 1 deg, the floor moving {0,0,1} (its normal), the side fixed; start at the rest point {tol / tan(0.5 deg),
      // 0, tol} at rest, T 0.01: no error, hit_count 2, velocities {1 / tan(1 deg), 0, 1} = {57.28996, 0, 1} within
      // 1e-9 relative, speed 1 / sin(1 deg) = 57.2987, below 64 times the largest speed involved (1 m/s)
  TEST(MoveParticle, OneFaceClosingAtPointEightDegrees_IsSqueezedBetweenSolids)
      // the same at a = 0.8 deg: the change would give 1 / sin(0.8 deg) = 71.622 m/s, above 64:
      // kTooCloseToCall/kSqueezedBetweenSolids, solid_id 2 (the side, the newest surface: it responds after the
      // floor at the start), result unchanged; the same at T 0.001 (a shorter step does not cure it)
  TEST(MoveParticle, TwoFacesClosingSymmetrically_CapAtOnePointSevenNineDegrees)
      // floor moving {0,0,1} and side moving 1 m/s along its own normal, start at the rest point at rest, T 0.01:
      // a = 2 deg: no error, speed 1 / sin(1 deg) = 57.2987 m/s; a = 1.6 deg: kSqueezedBetweenSolids
      // (1 / sin(0.8 deg) = 71.62)
  TEST(MoveParticle, GrazingWedgeClosingFast_IsSqueezedAtEveryStepLength)
      // a = 1e-5 rad, the side moving 1 m/s along its own normal; start {4 tol / tan(1e-5), 0, tol} at rest: the
      // escape would be 1 / tan(1e-5) = 1e5 m/s: kSqueezedBetweenSolids at T 0.1, 0.01 and 0.001 (before the cap the
      // model launched it 1 km in a step of 0.01 s)
  TEST(MoveParticle, ClosingCeiling_IsSqueezedBetweenSolids)  // as listed above; solid_id 1, the newest surface
  TEST(MoveParticle, BallRestingOnFloorSlidingIntoCusp_ComesToRestThere)
      // floor id 1, sphere id 2 centre {0,0,0.5} radius 0.5; start {0.05, 0, tol}, both velocities {-1,0,0},
      // T 0.1: hit_count 2 (the sphere, then the floor at 0), velocities 0 within 1e-15, end {2 Sqrt(0.5 tol), 0, tol}
      // = {46340.95 tol, 0, tol} with x within tol / 256, both at the rest gap. The first version of P19 failed it
  TEST(MoveParticle, BallRestingOnFloorRestsInCuspForThousandSteps)
      // the same scene; start {0.05, 0, tol}, both velocities {-0.5,0,0}; 1000 steps of T 0.01, both velocities
      // += {0,0,-9.81 T} before each, friction 0: no error in any step; the rest point reached in step 9 (the model),
      // then hit_count 1 per step (the floor), the end within tol / 256 of {2 Sqrt(0.5 tol), 0, tol}
  TEST(MoveParticle, TwoBallsTouchingSlidingIntoNeck_ComesToRest)
      // spheres id 1 centre {0,0,0} and id 2 centre {1,0,0}, radius 0.5; start {0.5, 0.05, 0}, both velocities
      // {0,-1,0}, T 0.1: hit_count 2, velocities 0 within 1e-15, end {0.5, Sqrt(tol (1 + tol)), 0} =
      // {0.5, 32768.0 tol, 0} within tol / 256
  TEST(MoveParticle, ThreeSolidAcuteCornerInto_StopsInCorner)
      // Task 11b's three-solid corner; start {0,0,0}, both velocities {-1,-1,-1}, friction 0.5, T 0.01: hit_count
      // 3, velocities {0,0,0} within 1e-15, end {Sqrt(3) tol, Sqrt(3) tol, tol} within tol / 1024
  TEST(MoveParticle, PlateAndSphereSlidingIn_ComesToRestAtRing)
      // Task 11b's plate and sphere; start {0.5, 0, 0.01 + tol}, both velocities {-1,0,0}, T 0.1: the record is the
      // sphere (solid_index 1, face kCurved) at hit_fraction (0.5 - Sqrt(0.1875) - Sqrt(3) tol) / 0.1 =
      // 0.669872965 within 1e-9; then the plate at 0; hit_count 2, velocities 0 within 1e-15, end ring point +
      // 2 tol {Sqrt(3)/2, 0, 1/2} within tol / 1024
  TEST(MoveParticle, CeilingHitFarAboveTouchedFloor_SlidesDownToApex)
      // floor id 1; ceiling id 2 point {0,0,0.1}, normal {-0.6,0,-0.8}; start {0,0,-tol / 2} (overlap tol), both
      // velocities {5,0,1}, T 0.2: the floor responds at the start (inside), the ceiling is hit at x 0.10526 and
      // leaves {2.72,0,-2.04}, which approaches the floor 0.021 m below; the floor is not active there, so the
      // particle slides down the ceiling to the apex and stops: kind kStartedInsideSolid, hit_count 3, velocities
      // 0 within 1e-15, end {(0.08 - 1.8 tol) / 0.6, 0, tol} = {0.1333333305, 0, tol} within tol / 1024. With the
      // floor in the change it stops at the ceiling hit, {0.10526, 0, 0.02105}
  TEST(MoveParticle, CornerPushOnSecondReactingSolid_IsSecondReactingSolid)
      // the 60 deg side id 1 (index 0) and the floor id 2 (index 1), both reacting; start 0.999 {Sqrt(3) tol, 0,
      // tol} (0.999 tol from both), both velocities {-1,0,-2}, T 0.01: the side is left at the start (silent, an
      // entry), the floor responds ({0,0,2}), and the change pushes the side by {1, 0, -1/Sqrt(3)}:
      // kTooCloseToCall/kSecondReactingSolid, solid_id 1. With the side not reacting: hit_count 1, velocities 0
      // within 1e-15, reaction.solid_index 1, velocity_for_next_step_change {0, 0, 2 + 1/Sqrt(3)} within 1e-15
  TEST(MoveParticle, SpinningFloorMovedSilently_RoundingIsNotAHit)
      // the hand-over above: the case of Task 11's review mirror (a flat solid spinning about its normal at a
      // surface speed of 100 m/s at the particle, moved silently while a second solid responds at the start):
      // hit_count 1 (the responding solid only); as built, rounding counts the spinning solid as a hit at 0
  TEST(SolveSmallestCorrection, SharpPairsAndCornersNearLimit_ApproachLeftWithinShare)
      // 20,000 cases, generator P11: a pair with 1 + c = 2^(-40 + 36 Next()), or a triple with triple product at
      // least 2^-20, by Next() < 0.5; v and each u_i 2^(60 Next() - 30) times three NextSigned; refine on.
      // Expected: for every surface, -Dot(n_i, v + w - u_i) <= 2^-46 (|v|_1 + |w|_1 + sum |u_i|_1), and at most 3
      // refinements (the model: at most 16 * 2^-52 of that sum with the refinements; 177 for pairs near 0.45 deg and
      // 2697 for triples without them)
  TEST(MoveParticle, GeneratedTwoSurfaceStarts_SucceedDownToFiveDegrees)
      // Task 11b's test, now: no failure at all (the model: 0 of 10,001 acute draws)
  TEST(MoveParticle, GeneratedMovingTwoSurfaceStarts_SucceedDownToFiveDegrees)
      // the same draws, then each wedge solid's linear_velocity three NextSigned m/s (solid A, then B): no failure
      // (two independent surfaces always leave a velocity; the model: 0 of 35,000)
  TEST(MoveParticle, GeneratedTangentContactSlides_FailOnlyWhereAShorterStepCures)
      // Task 11b's tangent set with slides added (by Next() < 1/3: start on the floor at x* + (0.05 - x*) Next() from
      // the contact, inward at up to 2 m/s with gravity), fixed and moving (each solid sliding along the other at up
      // to 1 m/s per axis; a ball sinking or rising at 1e-3 m/s). Expected: no failure but kTooManyHits and
      // kSqueezedBetweenSolids, and each of those succeeds when the step is shortened by a factor of up to 1/16
      // (rerun in the test); count them (the model: 31 two-ball kTooManyHits in 40,000 draws, all cured by 1/16;
      // no squeeze; every success at least tol / 2 from every solid)
  ```
  The expected values were computed with the Python model of P19 item 10 and checked by hand against closed forms
  (the apex point; the cusp rest point `2 Sqrt(R tol)`; the neck ring `Sqrt(tol (2 R + tol))`; the escape velocity
  `{V / tan a, 0, V}`; the pair's multipliers; the hit time on the sphere).
- [ ] **Step 2: Run.** Expected: the first two fail (`ClosingCeiling` passes through the floor or loops), the
  reaction tests fail until step 5 adds its changes to the reaction, and every P19 test fails (the groove ends in
  `kTooManyHits`, the closing 60-degree wedges too).
- [ ] **Step 3: Implement.** Glossary row: `slide` (the rest of the step spent moving along a surface). Comments:
  why one change over the active surfaces and not the sequence of design 5.5 (the groove); why the active surfaces
  are read afresh and only near ones (the ball's stale normal; the far floor); why the refinement solves and the
  velocity limit (the measured rounding); why a sliver sticks; why each push is charged to its own solid; how the
  squeeze test tells a closing wedge that carries the particle from one that crushes it, why the speed cap is 64 and
  which angles it refuses (0.895 degrees for one closing face, 1.79 for two), and why too fast an escape is
  a squeeze a shorter step cures. The paper's step 5 and <proc-move-particle> follow.
- [ ] **Step 4: Run** build, tests, gate. Mutants, each killed by the named test:
  - the newest solid's velocity in a corner (design 5.5 as it was): `GrooveSlideAlongEdge`,
    `CeilingHitFarAboveTouchedFloor`, `AcuteWedgeOnReactingFloor`, `NarrowWedgeAtRestPoint`;
  - every touched entry in the change, active or not: `CeilingHitFarAboveTouchedFloor`;
  - the normals recorded at the hits instead of read afresh: `GeneratedTangentContactSlides` (the model: one draw
    in 2000 of a ball in a domain corner, a slide along the ball between two contacts with it, ends in a corner of
    three surfaces with no change);
  - the whole change charged to the newest solid: `AcuteWedgeOnReactingFloor` (`{-1,0,1}`);
  - pushes not charged (only responses): `CornerPushOnSecondReactingSolid` (no conflict; the floor `{0,0,2}`);
  - no refinement solve: `SharpPairsAndCornersNearLimit_ApproachLeftWithinShare`;
  - the velocity limit `2^-30` instead of `2^-40`: `NarrowWedgeAtRestPoint` (2e-6 rad, `1 + c = 2e-12`, would stick);
  - the sliver failing instead of sticking: `SliverStop_ChargedToNewestSolid`;
  - the speed cap removed: `OneFaceClosingAtPointEightDegrees` (launched at 71.6 m/s), `GrazingWedgeClosingFast`;
    the factor 32: `OneFaceClosingAtOneDegree` (squeezed); 128: `OneFaceClosingAtPointEightDegrees` (carried);
    the largest speed taken from the particle alone (0 here): every closing test squeezed;
  - the escape check left out (an assertion only): with the cap also removed, a debug build stops on
    `GrazingWedgeClosingFast` (a release build goes on with an allowance outside the proof);
  - the squeeze test on the closing speed dropped: `ClosingCeiling`;
  - the start responses without the change: `GrooveSlideAlongEdge` (ping-pong to `kTooManyHits`).
  Expected survivor, with its reason in the report: the closing speed compared with 0 instead of the allowance (two
  solids of module 1 whose velocities differ by less than the allowance differ by exactly 0 in every scene the tests
  can build).
- [ ] **Step 5: Commit.** Subject: `Slide along several surfaces and refuse a squeeze`.

**Amended 2026-11-15 (A74):** the earlier-surface rule of Task 12 applies to every shape unchanged (it reads the
normal recorded at each hit). Round solids get their own move rules in Task 12a. **Amended 2026-11-15 (P19):** the
same for Task 11b's position move and Task 12's change: they read distances and normals only, and a round surface's
convex distance makes one solve enough (the plate-and-sphere tests of Tasks 11b and 12 run before Task 12a, since
the sphere's distance and first hit exist since Tasks 4a and 8a).

### Task 12a: `MoveParticle` on a sphere and a cylinder (A74)

**As built (2026-11-16, commit `60b9901b`), and superseded in part by P21 (Task 12b), where this task's text below
differs:** the task was built before Task 13, so `TurningCylinderRimHit_IsNotMoved` is a test of `ApplyHit`; the rim
move applies only above `d = tol`, and the window of the active surfaces grows by its length (neither in a decision).
Task 12b removes the rim move, `ComputeIsRimHitToMove` and that wider window: every sentence below on the rim move,
and the expected values `{0.4, 0, -0.2}` and "within 0x1p-40 of tol" of `CylinderRim_HitPositionMovedToOneTolerance`,
are replaced by Task 12b's.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact.h` (`MoveParticle`, public), `solid_contact_steps.h` (new private steps), `move_particle_test.cc`,
  `contact_test_cases.h`

**Interfaces:**
- Consumes: Tasks 8a, 11, 12; preconditions to respect: `FindFirstHit` requires `duration <= step_duration - start_elapsed` (asserted); the start rule and the touching rule use the solid's pose at `start_elapsed`; a path that meets a grown edge exactly is a hit; the domain box's touching rule is applied per closed side. (Task 12's rule follows, including its rule that "approaches again" is `approach >= -bound` means no longer
  approaching, never `approach >= 0`, with `bound` from `ComputeLargestApproachLeftAfterResponse`); `ComputeShapeTurningAngularVelocity` (Task 3) for P17 in step 5, no second helper; and (amended 2026-11-15) Task 13 for the test
  `TurningCylinderRimHit_IsNotMoved`: it needs the slide on a turning solid to finish the step, so **run Task 12a
  after Task 13**. **Amended 2026-11-15 (Task 9b):** and Task 9b (P18). The task order is 9, 9b, 10, 11, 12, 13,
  12a, 14 to 18 (amended 2026-11-15, P19: 11b after 11). **Amended 2026-11-15 (P19):** the rim move of step 4 is a
  move off one surface; when the moved hit is also within tol of another surface (a cylinder's rim on a floor), the
  joint response (P20; it was the slide's change of Task 12) reads the active surfaces at the moved position. A
  cylinder lying on a floor touches it
  along a line: a cusp, resolved like a ball's (P19 items 1 and 3); the cylinder joins Task 11b's generated tangent
  set here. Add `CylinderLyingOnFloorSlidingIntoCusp_ComesToRestThere` (axis along world x, radius 0.5, half height
  1, centre {0,0,0.5}; start {0, 0.05, tol}, both velocities {0,-1,0}, T 0.1: no error, hit_count 1 (amended
  2026-11-16, P20: one joint response over the wall and the floor; it was 2), velocities 0
  within 1e-15, end {0, 2 Sqrt(0.5 tol), tol} with y within tol / 256, both at the rest gap) and
  `CylinderSunkIntoFloorSlidingIn_ComesToRest` (centre {0,0,0.25}: the wall meets the floor at y = Sqrt(0.1875) at 60
  degrees; start {0, 0.5, tol}, both velocities {0,-1,0}, T 0.1: no error, hit_count 1 (P20, as above), velocities 0
  within 1e-15,
  end {0, Sqrt(0.1875) + Sqrt(3) tol, tol} within tol / 1024). Both are the cross-sections of the ball scenes of
  Tasks 11b and 12 (`BallRestingOnFloorSlidingIntoCusp`, `PlateAndSphereSlidingIn`); the model's cylinder (infinite
  along x) gave the same numbers.
  What changes here (Task 9b): a stepping-search hit is still in Task 9's band and still never moved, but a
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

### Task 12b: A cylinder's rim at its true distance

**Kind:** numerics

Added 2026-11-16 (Task 12a review I2, coordinator ruling 44, plan decision P21). Run it after Task 12a and before Task
13. A cylinder that does not turn is hit beside a rim where its true distance reaches tol, by the rim search; a start
that rests on a bounding surface of a box or a cylinder is free of that solid, so flush drums, pucks and boxes no
longer stop or deflect a sliding particle; the rim move, `ComputeIsRimHitToMove` and the window extension of Task 12a
are removed. P21 is the specification: read its items 1 to 7 before the code. The step on boxes (item 8 a) can be
built as a commit of its own if the coordinator prefers; it shares `ComputeIsRestingOnBoundingSurface` and nothing
else.

**Files:**
- Modify: `native/fluidity/contact/solid_contact_types.h` (four constants), `solid_contact_steps.h` (private steps),
  `solid_contact.h` (docstrings of `FindFirstHit` and `MoveParticle` only: no public signature changes),
  `find_first_hit_test.cc`, `move_particle_test.cc`, `contact_test_cases.h`, `docs/glossary.md`,
  `docs/method-paper/fluidity-method.typ` (and its figure of the round solids), `tools/check_rules_allowed_names.txt`
  only if the rules check asks.

**Interfaces:**
- Consumes: Tasks 8, 8a (P15's intervals: `ComputeTimeInsideGrownWall`, `ComputeTimeBetweenGrownCaps`, `OwnPath`,
  `HitTime`, `ToFirstHit`, `FindEntryThroughFacePlanes`), 4a (`ComputeDistanceToCylinder(own_point, half_size,
  solid_axes)`; with the identity as `solid_axes` its normal is in own axes), 12 and 12a (`ApplyHit`,
  `FindActiveSurfaces`, the touched rule of `ComputeStartCase`, the proof comment of `FindFirstHitOfSweep`).
  Preconditions to respect: `FindFirstHit` accepts `contact_tolerance == 0` (P9, the naive control): the windows of
  item 1 then shrink to the single value 0 and the hit gap of the rim search is 0; nothing divides by the tolerance.
  The domain box, the flat solid, the plate and the sphere keep their results bit for bit.
- Produces, in `solid_contact_types.h`:

  ```cpp
  // No unit. How far below its rest gap, as a share of the contact tolerance, a start may be and still rest on a
  // bounding surface that does not give its distance (plan decision P21, item 1).
  inline constexpr double kBoundingSurfaceRestingDepthShare = 0x1p-3;
  // No unit. The most a path may sink toward a bounding surface it rests on during one sweep, as a share of the
  // contact tolerance: what a response may leave, allowance * time left <= tol / 64.
  inline constexpr double kBoundingSurfaceSinkShare = 0x1p-6;
  // No unit. The rim search stops with a hit once the gap is at most this share of the contact tolerance.
  inline constexpr double kRimHitGapShare = 0x1p-10;
  // The most advances of the rim search in one sweep; at the limit it reports the hit where it stands.
  inline constexpr int kMaxRimAdvanceCount = 16;
  ```

  and in `solid_contact_steps.h` (private; names decided here, comments in the house style with Method, Units,
  Example, Requires, Read):

  ```cpp
  // True when a sweep's start rests on one bounding surface: -depth share * tol <= start_gap <=
  // kTouchedSolidRestingGapShare * tol and gap_rate * duration >= -sink share * tol. Only + - * and comparisons.
  FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsRestingOnBoundingSurface(
      double contact_tolerance, double start_gap, double gap_rate, double duration);

  // The rim search (P21 items 3 d, 3 e and 4): from the entry into the cylinder grown with sharp edges. The first
  // sample's face decides: not a rim, the entry is the hit. *advance_count, when not null, holds the advances made.
  FLUIDITY_CPU_AND_GPU inline HitTime FindCylinderRimHit(
      OwnPath own_path, Vec3 half_size, double contact_tolerance, double entry_elapsed, double duration,
      int* advance_count);

  // One more parameter: the face that gives the start's distance, whose touching rule the caller has applied; every
  // other face plane gets ComputeIsRestingOnBoundingSurface before its gap is read as in front or behind.
  FLUIDITY_CPU_AND_GPU inline FirstHit FindEntryThroughFacePlanes(
      const SolidContactInputs& inputs, std::uint32_t solid_index, Sweep sweep, const FacePlane* face_planes,
      int face_plane_count, SolidFace face_of_start_distance);

  // Back to three parameters: the window is tol + tol / 16 at every hit.
  FLUIDITY_CPU_AND_GPU inline ActiveSurfaces FindActiveSurfaces(
      const SolidContactInputs& inputs, const ParticleMove& move, const FirstHit* hit);
  ```

  `FindCylinderHit(inputs, solid_index, sweep, touched)` and `ApplyHit(inputs, first_hit, move, failure)` keep their
  signatures. `FindHitOnConvexSolid` passes `at_start.face`; `FindHitOnDomainBox` passes the side's own face, so its
  one plane is never tested (a domain side keeps P2 alone).
- **Room for module 1b (mesh solids) (A80):** `ComputeIsRestingOnBoundingSurface` reads no shape, so 1b can give each
  triangle's plane the same rule; `ApplyHit` has no place left where a hit position is moved before the response (1b's
  row H6 must be rewritten, P21 item 11); the rim search's one shape-specific line is its distance read.

- [ ] **Step 1: Failing tests** (`tol = kTestContactTolerance`; friction 0; both velocities equal; "the cylinder" is
  `BuildCylinder({}, kAxisZ, kNoMotion)`: origin, axis z, radius 0.5, half height 1; "the box" is half sizes 0.5 at
  the origin with identity axes). First hits, in `find_first_hit_test.cc`:

  ```cpp
  // --- the rim search
  TEST(FindFirstHit, CylinderParallelToAxisInGrownRadius_HitsRim)   // existing, new values: {0.5 + tol/2, 0, 3},
      // v {0,0,-1}, T 4: elapsed within tol / 1024 s of 2 - Sqrt(0.75) tol (1.999999999193451); kHighRim; normal
      // {0.5, 0, Sqrt(0.75)} within 1e-5; at_hit.distance in [tol - 0x1p-40, tol + tol / 1024 + 0x1p-40]
      // (was 2 - tol s at Sqrt(1.25) tol)
  TEST(FindFirstHit, CylinderThroughGrownRimEdge_IsHit)             // existing, new values: elapsed within
      // tol / 1024 s of 0.45 + (8 - Sqrt(24)) / 20 tol (0.4500000001444026); kHighRim; distance as above
  TEST(FindFirstHit, CylinderRimHeadOnAtFortyFiveDegrees_HitsAtOneTolerance)   // {1.5, 0, 2}, v {-1,0,-1}, T 2:
      // elapsed within tol / 1024 s of 1 - Sqrt(0.5) tol (0.9999999993414554); kHighRim; normal {Sqrt(0.5), 0,
      // Sqrt(0.5)} within 1e-5; distance within 0x1p-40 of tol (was 1 - tol s at Sqrt(2) tol)
  TEST(FindFirstHit, CylinderRimPassedBeyondOneTolerance_IsFree)    // {0.75 + a, 0, 0.75 + a}, v {-1,0,1}, T 0.5,
      // a = 25/32 tol: the path passes the rim at 1.105 tol, through the sharp corner: free (was a hit at
      // 0.2499999997962732). a = 23/32 tol (1.016 tol): free. a = 22/32 tol (0.972 tol): a hit, kHighRim, distance
      // in [tol - 0x1p-40, tol + tol / 1024 + 0x1p-40]
  TEST(FindFirstHit, CylinderRimCornerStartMovingAway_IsFree)       // existing; its approaching case changes:
      // start {0.5 + 0.9 tol, 0, 1 + 0.9 tol}, v {-1,0,-1}, T 1: elapsed within tol / 1024 s of
      // (0.9 - Sqrt(0.5)) tol (1.796458e-10); kHighRim; normal {Sqrt(0.5), 0, Sqrt(0.5)} within 1e-5.
      // v {-1,0,0}: free now (it rests on the cap's plane, 0.1 tol inside its rest gap: P21 item 1)
  TEST(FindFirstHit, StartElapsed_CylinderRimCornerUsesLaterPose)   // existing; start {1 + 0.9 tol, 0, 1 + 0.5 tol}
      // at rest, start_elapsed 0.5, T 0.5, cylinder moving {1,0,0}: elapsed within tol / 1024 s of
      // (0.9 - Sqrt(0.75)) tol (3.1637e-11); kHighRim. ..._MovingWithCylinderIsFree takes the same start: free
  TEST(FindCylinderRimHit, AdvanceCounts)   // own paths of the tests above, entry from the two intervals:
      // tol/2 beyond the wall: 2; through the grown rim edge: 2; head on at 45 degrees: 1; the 0.9, 0.9 start
      // along {-1,0,-1} (entry 0): 1; {0.5 + 13/16 tol, 0, 3}, v {0,0,-1}: 2, elapsed within tol / 1024 s of
      // 2 - Sqrt(1 - (13/16)^2) tol; {0.5 + tol, 0, 3}, v {-tol/128, 0, -1}, T 4: 4, kHighRim at
      // cap_distance 0.1763 tol within tol / 512; an entry on the wall proper ({-2, 0, 0}, v {1,0,0}): 0, and
      // the same elapsed as the entry, bit for bit
  TEST(FindCylinderRimHit, GapRateTooSmallToSquare_IsFreeNotStalled)  // own start {0.5 + 0.9 tol, 0,
      // 1 + 0.9 tol}, own velocity {-1e-160, 0, -1e-160}, entry 0, duration 1: free with 0 advances (the
      // advance is the quotient, 1.6e150 s; through ComputeSecondOrderAdvance it would be 0 and the search
      // would report a hit 0.27 tol out at the limit)
  // --- the touching rule of a bounding surface, cylinder
  TEST(FindFirstHit, CylinderDownItsGrownWallPastTheRim_IsFree)     // replaces CylinderGrownDiscEdgeMetExactly_
      // IsHit: {0.5 + tol, 0, 3}, v {0,0,-1}, T 4: free (was a hit at 2 - tol s, kHighRim, Sqrt(2) tol away)
  TEST(FindFirstHit, CylinderAlongItsGrownCapPlanePastTheRim_IsFree)  // replaces CylinderGrownWallTopMetExactly_
      // IsHit: {-2, 0, 1 + tol}, v {1,0,0}, T 2: free
  TEST(FindFirstHit, CylinderBoundingSurfaceWindowAndSink)   // v {0,0,-1}, T 4, start {x, 0, 3}:
      // x = 0.5 + tol - tol/8: free; x = 0.5 + tol - 3 tol/16: a hit (elapsed as in AdvanceCounts);
      // x = 0.5 + tol, v {-tol/512, 0, -1} (sinks tol/128): free; v {-tol/128, 0, -1} (sinks tol/32): a hit;
      // x = 0.5 + tol + tol/128, v {-tol/128, 0, -2}, T 2 (above the rest gap, sinks tol/64): free (was a hit
      // on the wall at 1 s); x = 0.5 + tol + tol/8, same velocity: free by the intervals, as built
  TEST(FindFirstHit, CylinderWallThatGivesTheDistance_KeepsItsOwnTouchingRule)  // {0.5 + tol + tol/128, 0, 0},
      // v {-tol/128, 0, 1}, T 2: a hit at 1 s within 1e-9, kCurved (the wall gives the distance: P16 alone)
  // --- the touching rule of a bounding surface, box
  TEST(FindFirstHit, BoxDownItsGrownSidePlanePastTheEdge_IsFree)    // {0.5 + tol, 0, 3}, v {0,0,-1}, T 4: free
      // (was a hit on kHighZ at 2.5 - tol)
  TEST(FindFirstHit, BoxBoundingSurfaceWindowAndSink)   // v {0,0,-1}, T 4, start {x, 0, 3}:
      // x = 0.5 + tol - tol/8: free; x = 0.5 + tol - 3 tol/16: kHighZ at 2.5 - tol exactly;
      // x = 0.5 + tol, v {-tol/512, 0, -1}: free; v {-tol/128, 0, -1}: kHighZ at 2.5 - tol;
      // x = 0.5 + tol + tol/128, v {-tol/128, 0, -2}, T 2: free (was kHighZ at 1.25 - tol/2)
  TEST(FindFirstHit, BoxFaceThatGivesTheDistance_KeepsItsOwnTouchingRule)   // {0, 0, 0.5 + tol + tol/128},
      // v {0.25, 0, -tol/128}, T 1: kHighZ at elapsed 1 exactly (the top gives the distance: P2 alone)
  ```

  Unchanged, and run as guards: `BoxRestingOnEdge_UsesTheFaceThatGivesTheDistance`,
  `BoxRestingOnTop_SlidingOffTheEdgeIsFree`, `BoxEdgeMetExactly_IsHit`, `BoxEdgeMissedByOneUnit_IsFree`,
  `BoxEdgeGraze_StaysFree`, `CylinderRimCornerStartAlongTangent_IsFree`, `CylinderHitExactlyAtEnd_IsHit`,
  `CylinderTangentToWall_FreeOrHitNearTwo`, every domain box, plate, flat solid and sphere test.
  `MovingCylinderEachFace_HitsThatFace` and `TiltedMovingCylinderEachFace_HitsThatFace`: the two rim rows take
  `distance_at_hit = tol` (it was `Sqrt(2) tol`), compared within `tol / 1024`; the other rows keep their bits. Moves,
  in `move_particle_test.cc` with their scenes in `contact_test_cases.h` (registered last, so earlier scenes keep
  their generated draws):

  ```cpp
  // --- flush scenes (P21 item 13)
  TEST(MoveParticle, StackedDrumsOfOneRadius_SlideDownTheSeamKeepsItsVelocity)
      // cylinders axis z, r 0.5, h 0.5, centres {0,0,1.5} (id 1) and {0,0,0.5} (id 2); start {0.5 + tol, 0, 1.8},
      // v {0,0,-1}, T 1: no error, kNone, hit_count 0, both velocities {0,0,-1} bit for bit, end {0.5 + tol, 0,
      // 0.8} within 0x1p-50, distance to the lower drum within 0x1p-50 of tol. Again from x = 0.5 + tol -
      // 0x1p-53 and from x = 0.5 + tol - tol/32: the same. (As built: one hit on kHighRim, {0.5, 0, -0.5}.)
  TEST(MoveParticle, PuckFlushInFloor_SlideOntoItAndOffKeepsItsVelocity)
      // BuildFloor(1); cylinder id 2 centre {0,0,-0.1}, axis z, r 0.5, h 0.1. Start {0.5 + 3 tol, 0, tol},
      // v {-1,0,0}, T 0.1: kNone, hit_count 0, velocities bit for bit, end {0.4 + 3 tol, 0, tol}, both
      // distances within tol / 1024 of tol. Start {0.45, 0, tol}, v {1,0,0}: the same, end {0.55, 0, tol}.
      // (As built, onto it: kHighRim, {-0.5, 0, 0.5}.)
  TEST(MoveParticle, DrumStandingOnFloor_GrooveStopsAndLetsSlideAround)   // guards: as built
      // BuildFloor(1); cylinder id 2 centre {0,0,0.5}, axis z, r 0.5, h 0.5. {1, 0, tol}, v {-1,0,0}, T 1:
      // hit_count 1, solid_index 1, kCurved, hit_fraction 0.5 - tol, velocities {0,0,0}, end {0.5 + tol, 0, tol}.
      // {0.5 + tol, 0, 0.7}, v {0,0,-1}, T 1: hit_count 1, solid_index 0, kHighZ, hit_fraction 0.7 - tol,
      // velocities {0,0,0}, the same end. {0.5 + tol, 0, tol}, v {0,1,0}, T 0.25: hit_count 0, end y 0.25
  TEST(MoveParticle, RimCrestUnderFloor_IsNotHit)
      // BuildFloor(1); cylinder id 2, axis {Sqrt(0.5), 0, Sqrt(0.5)}, r 0.5, h 0.25, centre {0, 0, -(0.25 + 0.5)
      // Sqrt(0.5) - 0.41 tol} (its high rim's crest 0.41 tol under the floor's surface, at x = -0.25 Sqrt(0.5));
      // start {-0.25 Sqrt(0.5) - 0.25, 0, tol}, v {1,0,0}, T 0.5: hit_count 0, velocities bit for bit, end z tol.
      // (As built: kHighRim 1.41 tol away, moved 0.41 tol toward the floor.)
  // --- a rim at its true distance
  TEST(MoveParticle, CylinderRim_HitAtItsTrueDistance)   // replaces CylinderRim_HitPositionMovedToOneTolerance,
      // scene CylinderRim: kHit, kHighRim, hit_count 1, hit_fraction (2 - Sqrt(0.75) tol) / 4, velocities
      // {Sqrt(3)/4, 0, -1/4} = {0.4330127, 0, -0.25} within 1e-6; reaction.hit_position x == 0.5 + tol/2 bit for
      // bit (on the path: not moved) and z within tol / 512 of 1 + Sqrt(0.75) tol; ComputeDistanceToSolid there
      // in [tol - 0x1p-40, tol + tol / 1024 + 0x1p-40]. (Was {0.4, 0, -0.2}.)
  TEST(MoveParticle, CylinderRimFromTheSide_HitAtItsTrueDistance)   // {2, 0, 1 + tol/2}, v {-1,0,0}, T 4:
      // hit_count 1, kHighRim, velocities {-1/4, 0, Sqrt(3)/4} within 1e-6, hit x within tol / 512 of
      // 0.5 + Sqrt(0.75) tol
  TEST(MoveParticle, CylinderRimNearRestGap_HitAtItsTrueDistance)   // scene CylinderRimNearRestGap: velocities
      // {Sqrt(15)/16, 0, -1/16} = {0.2420615, 0, -0.0625} within 2e-5 (the search stops 3e-5 tol out; was
      // {4/17, 0, -1/17})
  TEST(MoveParticle, RisingCylinderRim_HitReadAtThePoseOfTheHit)    // existing: {Sqrt(3)/4, 0, 3/4} within 1e-6
  TEST(MoveParticle, SlowlyTiltingCylinderRim_HitAtItsTrueDistance) // replaces ..._IsMoved: {Sqrt(3)/4, 0, -1/4}
      // within 1e-5; distance at the pose of the hit within tol / 16 of tol (P17's slack: 0.024 tol here)
  // ExpectRimHitMovedToRestGap becomes ExpectRimHitAtTrueDistance: after ApplyHit the position is
  // start + velocity * first_hit.elapsed bit for bit, and its distance is first_hit.at_hit.distance bit for bit.
  TEST(MoveParticle, SlideOffCapPastRim_AndUpWallPastRim_OneHitThenFree)
      // {0.5 - 8 tol, 0, 1 + tol}, v {1, 0, -0.5}, T 16 tol: hit_count 1, hit_fraction 0, kHighZ, velocities
      // {1,0,0}, end distance Sqrt(65) tol within tol / 1024. {0.5 + tol, 0, 1 - 8 tol}, v {-0.5, 0, 1}, T 16 tol:
      // hit_count 1, kCurved, velocities {0,0,1}, the same end distance. (As built: the same; guards.)
  TEST(MoveParticle, ParticleCarriedAroundRim_OneHitPerStep)
      // 64 steps of T 1 from {0.5 - 2 tol, 0, 1 + tol}; before each step read n = the normal of
      // ComputeDistanceToSolid at the position, t = {n.z, 0, -n.x}, both velocities tol/2 * t - tol/4 * n. Each
      // step: no error, hit_count 1, end distance in [tol - tol / 1024, 1.13 tol]; the faces at the ends are
      // kHighZ 4 times, kHighRim 3 times, kCurved 57 times, in that order; the last end is within tol / 512 of
      // tol beyond the wall and 28.39 tol below the cap's plane within 0.01 tol. Then the other way, t negated,
      // from {0.5 + tol, 0, 1 - 2 tol}: kCurved 4, kHighRim 3, kHighZ 57, mirrored end.
  // --- beside another surface
  TEST(MoveParticle, RimOnFloorSweepingParticle_RimPassesOutsideItsTrueDistance)   // replaces ..._FloorStaysIn
      // TheResponse, same scene: hit_count 1 (the start's), velocities {0,0,0} within 1e-15, end distance to the
      // floor within tol / 1024 of tol: the round rim passes 1.069 tol from the particle. (Was 2 hits.)
  TEST(MoveParticle, RimOnFloorSweepingParticleLower_RimAndFloorRespondTogether)
      // new scene: the same with the cylinder's linear_velocity.z 0.0009 (was 0.001): hit_count 2, second face
      // kLowRim; for_this_step {0.875932, 5.139e-5, 0} within 1e-5; the position before the settle
      // (ComputePositionBeforeSettle) is within tol / 1024 of tol from the floor (as built: 1.106 tol, lifted by
      // the rim move) and at least tol / 2 from the cylinder at the end pose
  TEST(MoveParticle, SlowlyTiltingCylinderRimCornerStart_RisingRimIsNotHit)   // replaces ..._IsHitAtStart:
      // kNone, hit_count 0, velocities {0,0,0} bit for bit, end distance at the end pose above 1.2 tol
  TEST(ApplyHit, TurningCylinderRimHit_StaysWhereTheSearchStopped)   // rename of ..._IsNotMoved; same checks
  // --- boxes (P21 item 8)
  TEST(MoveParticle, StackedBoxes_SlideDownTheSeamKeepsItsVelocity)
      // boxes half 0.5, centres {0,0,1.5} (id 1) and {0,0,0.5} (id 2); start {0.5 + tol, 0, 1.8}, v {0,0,-1},
      // T 1: hit_count 0, velocities bit for bit, end {0.5 + tol, 0, 0.8}. (As built: kHighZ of the lower box,
      // velocities {0,0,0}.)
  TEST(MoveParticle, BoxTiles_SlideAcrossTheSeamKeepsItsVelocity)
      // boxes half 0.5, centres {0,0,-0.5} (id 1) and {1,0,-0.5} (id 2); start {0.2, 0, tol}, T 0.6. v {1,0,0}:
      // hit_count 0, velocities bit for bit, end {0.8, 0, tol}. v {1, 0, -0.16}: hit_count 1 (the start's, kHighZ,
      // solid_index 0), velocities {1,0,0}, the same end. (As built: stopped on kLowX of the second box at
      // 0.3 - tol s, velocities {0,0,0}, 1 and 2 hits.)
  TEST(MoveParticle, BoxFlushInFloor_SlideOntoItKeepsItsVelocity)
      // BuildFloor(1); box id 2 half 0.5 centre {0,0,-0.5}; start {0.5 + 3 tol, 0, tol}, v {-1,0,0}, T 0.1:
      // hit_count 0, velocities bit for bit. (As built: kHighX, velocities {0,0,0}.)
  TEST(MoveParticle, BoxTilesStepUp_ASixteenthIsNotSeenAQuarterStops)   // the limit of P21 item 14, pinned
      // the tiles above with the second centre at z -0.5 + tol/32: hit_count 0, the position before the settle
      // 31/32 tol from the second box within tol / 1024, the end at tol. At z -0.5 + tol/4: hit_count 1, kLowX,
      // solid_index 1, velocities {0,0,0}, end x 0.5 - tol
  ```
- [ ] **Step 2: Run.** Expected: every new or changed test above fails except those marked guards and
  `GapRateTooSmallToSquare` (which fails only for an implementation through `ComputeSecondOrderAdvance`); the two
  `KeepsItsOwnTouchingRule` tests pass before and must pass after.
- [ ] **Step 3: Implement**, in this order, each with its tests green before the next. (1) The four constants. (2)
  `ComputeIsRestingOnBoundingSurface`. (3) `FindEntryThroughFacePlanes`: per plane, when its face is not
  `face_of_start_distance` and the start rests on it, return free, before the gap's sign is read; the two callers. (4)
  `FindCylinderRimHit`, a loop of at most `kMaxRimAdvanceCount + 1` samples in the order of P21 item 4, the first
  sample's face test of item 3 (d) before it; samples are `own_path.start + own_path.velocity * elapsed` and
  `ComputeDistanceToCylinder` with the identity. (5) `FindCylinderHit`: after the start case, for a start whose
  `at_start.face` is a rim, the two tests of item 1 with `wall_gap` and `cap_gap` as they are computed today; the rim
  corner's branch is deleted; the entry of the two intervals goes to `FindCylinderRimHit`. (6) `ApplyHit` and
  `FindActiveSurfaces` as in Step 4. Comments: the file header's Idea (the cylinder is swept grown with sharp edges
  and, beside a rim, searched on to its true distance; a start resting on a bounding surface), Words (`bounding
  surface`, `rim search`; `rim corner` reworded; `rim move` removed; `active surface` and `touching rule` updated) and
  Owns; `FindCylinderHit`'s Method and example (the fall `tol / 2` beyond the wall now ends at `{0.5 + tol / 2, 0, 1 +
  Sqrt(0.75) tol}`); `FindFirstHitOfSweep`'s proof: step 2 loses "(after a rim move, at the moved position, ApplyHit)"
  and "and a rim move reads them again where it ends", step 3 gains one sentence (a sweep that rests on a bounding
  surface stays `55 / 64 tol` or more from that solid, P21 item 2); `ApplySettleAtEnd`'s comment (`tol - tol / 16`
  becomes `55 / 64 tol` for a solid the path rested beside); `FindFirstHit`'s and `MoveParticle`'s docstrings in
  `solid_contact.h` (step 1 and step 4: no rim move; the convexity line stays). State in `FindCylinderRimHit`'s
  comment why the advance is the quotient and not `ComputeSecondOrderAdvance`, and why the limit reports a hit.
- [ ] **Step 4: Remove.** `ComputeIsRimHitToMove`; `ApplyHit`'s step 3 (the move, `rim_move_length`, the second read
  of `at_hit`) and its docstring's step 3 and second example; the fourth parameter of `FindActiveSurfaces` with
  `active_distance`'s sum, its assertion and the paragraph on the rim move in its comment; in `FindCylinderHit` the
  `is_in_rim_corner` branch with its `ComputeVelocityFromSurface` read; the tests' helper
  `ExpectRimHitMovedToRestGap`; `std::numbers::sqrt2 * kTolerance` as a bound of a cylinder hit in
  `find_first_hit_test.cc` (lines near 1280, 1354, 1539, 1781, 1790, 2644, 2695: each becomes `kTolerance + kTolerance
  / 1024`, with `kReferenceAllowance` where it has one). `grep -n "rim move\|RimHitToMove\|window_extension\|rim_move"
  native docs/glossary.md docs/method-paper` must print nothing afterwards.
- [ ] **Step 5: Glossary and method paper.** Glossary: remove the row `rim move`; `rim corner`: "The part of a
  cylinder grown by the contact tolerance with sharp edges that lies farther than one tolerance from the rim. A sweep
  that enters the grown cylinder there goes on by the rim search"; new row `bounding surface`: "A surface that a whole
  convex solid lies behind: a face plane of a box, a cylinder's wall taken as unbounded along its axis, the plane of a
  cap. A sweep whose start rests on one that does not give its distance (a gap between minus an eighth and plus a
  sixteenth of the contact tolerance, sinking at most a sixty-fourth of it during the sweep) is free of the solid:
  `ComputeIsRestingOnBoundingSurface`, `kBoundingSurfaceRestingDepthShare`, `kBoundingSurfaceSinkShare`. It is what
  lets a particle slide across the seam of two flush solids"; new row `rim search`, `kRimHitGapShare`,
  `kMaxRimAdvanceCount`: "How a sweep finds its first hit beside a rim of a cylinder that does not turn: from where
  the path enters the cylinder grown with sharp edges, advances of the gap divided by the speed at which it closes
  (the second-order advance of a solid that does not turn), on the true distance. A hit once the gap is at most
  `kRimHitGapShare` (2^-10) of the tolerance; free when the gap stops closing or would not close within the sweep;
  after `kMaxRimAdvanceCount` (16) advances a hit where it stands"; `active surface`: drop "(plus the length of a rim
  move, so a surface it was on before that move stays)"; `touching rule`: add "A bounding surface that does not give
  the distance has a rule of its own (`bounding surface`)"; `stepping search`: unchanged. Paper: `<sec-round-solids>`
  (the cylinder's paragraph: grown with sharp edges for the entry, then the rim search, with `g(t)` convex and the
  advance; the paragraph and the figure panel of the rim move are replaced by the rim search, drawn as the sharp
  corner, the round rim and two advances), `<eq:cylinder-hit-time>` unchanged, `<proc-cylinder-first-hit>` (steps:
  start case; resting on the wall or a cap's plane; the two intervals; the entry not beside a rim is the hit; the rim
  search), `<sec-first-hit>` (one paragraph on bounding surfaces after the touching rule, with the two flush examples;
  the sentence "`MoveParticle` starts every sweep at `d = tol` (a hit, the rim move or the start move put it there)"
  loses the rim move, twice, lines near 905 and 1213), `<proc-move-particle>` step 4 and the active-surface paragraph
  (lines near 1462 and 1635: no rim move, no added length). The notation table gains no symbol.
- [ ] **Step 6: Generated sets**, each re-measured and its numbers written in the test's comment and the report. First
  hits: `GeneratedCylinderSweeps_AgreeWithSlowReference` with the bound `tol + tol / 1024` (hit count still between
  100 and 900); `GeneratedRimEdgeSweeps_AgreeWithSlowReference` with `is_at_grown_surface` up to `tol + tol / 1024`:
  predicted for the two cylinders along z, from the model on the test's own draws, 50,000 hits each, 0 mistakes,
  advances 1, 2 or 3 in 7,189, 32,918 and 9,893 paths (a different `cos` may move a few between neighbours; assert at
  most 5); `GeneratedGrazingRoundSweeps_NeverHitBehindStart` unchanged (the wall proper); the turning sets unchanged.
  New: `GeneratedPathsPastACylinderRim_AgreeWithSlowReference`, 20,000 paths by P11's generator in the six kinds of
  P21 item 15 (a), for the four cylinders of the rim-edge test: a free path that rests on a bounding surface keeps the
  reference's smallest distance at `tol - 9 tol / 64 - kReferenceAllowance` or more, any other free path at `tol -
  kReferenceAllowance` or more; a hit after time 0 is between `tol - kReferenceAllowance` and `tol + tol / 1024 +
  kReferenceAllowance` at the hit, with the path before it at `tol - kReferenceAllowance` or more; advances at most 8
  (model: at most 5, mean 2.0). Moves: `GeneratedRoundSolidPaths_NeverHitAgain` (as built: hits 10,490, rim hits 548,
  repeat hits 0, failures 0): predicted repeat hits 0 and failures 0, tilted cylinders included; rim hits and hits
  lower by the same number, about a quarter of the rim hits (the model's own set: 258 to 190);
  `GeneratedCylinderRimPaths_NeverHitItAgain` 0 repeats; the cylinder tangent starts 6,000 of 6,000 and slides 12,000
  of 12,000 predicted unchanged; the every-scene paths and near starts: per scene identical except `CylinderRim`,
  `CylinderRimNearRestGap`, `RisingCylinderRim`, `SlowlyTiltingCylinderRim`, `SlowlyTiltingCylinderRimCornerStart`,
  `RimOnFloorSweepingParticle` and the new scenes; the `hit_paths` threshold is not lowered. New:
  `GeneratedFlushSeams_KeepTheSpeedAlongTheSeam`, 2,000 each of box tiles, stacked drums and a puck in a floor, fixed
  or moving together, identity or `kTiltedAxes`, a start 0.01 to 0.15 m before the seam at the rest gap, a push into
  the surface of 0, 0.16 or 1 m/s: no error, at most one hit, and the speed along the path, measured from the solids,
  at least `1 - 1e-6` of its start (model, its own draws: 0 of 4,000 per set lose speed; as built 57 %, 43 % and 29
  %).
- [ ] **Step 7: Mutants** (scratch copies, both contact test binaries, asserts on). Each must be killed by a named
  test, or argued in the report: no bounding-surface rule on cylinders (`StackedDrums`, `PuckFlushInFloor`); none on
  boxes (`StackedBoxes`, `BoxTiles`); the rule also on the surface that gives the distance (the two
  `KeepsItsOwnTouchingRule` tests); depth `tol / 4` and depth 0 (`WindowAndSink`, `StackedDrums` from `tol / 32`
  inside); the window's top 0 instead of `tol / 16` (`WindowAndSink`'s quick case); sink `tol / 16` and sink "gap rate
  >= 0" (`WindowAndSink`); the wall's rate without the division by `rho0` (add a case at radius 4 to `WindowAndSink`
  if it survives); the entry beside a rim taken as the hit (every rim test); hit gap share `2^-4`
  (`CylinderRimNearRestGap`: `{4/17, 0, -1/17}`); half the advance (`AdvanceCounts`); twice the advance
  (`GeneratedPathsPastACylinderRim`, late hits); no "gap rate >= 0 is free" (`CylinderRimPassedBeyondOneTolerance`);
  no "would not close within the sweep" (add a start `tol / 2048` outside the round rim closing at 1e-3 tol/s for 1 s:
  free); the advance through `ComputeSecondOrderAdvance` (`GapRateTooSmallToSquare`); `kMaxRimAdvanceCount = 1`
  (distance bounds of the rim tests); the rim move left in `ApplyHit` (`ExpectRimHitAtTrueDistance`'s position on the
  path). Expected survivors, to argue: the search run for an entry on the wall or a cap proper too (the same hit up to
  a few rounding units; `AdvanceCounts`' last case kills it if the bits differ); free instead of a hit at the limit
  (the limit is not reached).
- [ ] **Step 8: Run** build, tests, gate (`uv run python tools/check_commit.py`).
- [ ] **Step 9: Commit.** Subject: `Hit a cylinder's rim at its true distance`.

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
- **Amended 2026-11-15 (P19, Tasks 11b and 12):** a turning solid takes part in the position move and the velocity
  change like any other: its distance at one pose is convex in its own axes (P18), so one solve is enough, and its
  velocity at the particle is `ComputeSolidVelocity` with the full `omega`. Order at a hit on a turning solid: the
  joint response over the active surfaces (amended 2026-11-16, P20: one response that includes the turning solid's
  surface with the hit's normal, where this said the response, then Task 12's change), then P7's near-other-solid
  test and the carry. A turning solid still has no touching rule: after the first response of a step every other
  solid has the touched rule (P20 item 9), the turning solid none, so this task's carry is what keeps it from being
  hit again at 0. A start
  within tol of a turning solid and another solid gets Task 11b's one move at pose 0 (Task 11's
  `TurningBoxStartWithinTolerance` stays pinned and flips here, as Task 11's review asked). The settle's one move
  (Task 11b) reads the turning solid at the end pose; its `d >= -2 tol` assertion is revisited with the carry, as
  above. A particle in a wedge between a turning solid and another solid is within P7's reach of that solid, so P7's
  `kTurningSolidNearOtherSolid` decides before the change matters for the carry
  (`RestingOnTurningBoxNearFloor_StepLengthDecidesCloseCall` is that case); Task 13 adds no wedge test of its own.
- **Amended 2026-11-16 (P21, Task 12b; Task 12a review, deviation 1 and M6):** this task owes the `MoveParticle` check
  of the scene `TurningCylinderRimHit` that Task 12a could only make on `ApplyHit`: no error, face `kHighRim`,
  `reaction.hit_position` within `tol / 1024` of `start + velocity * elapsed` and its distance at the pose of the hit
  within `tol / 1024` of `FindFirstHit`'s `at_hit.distance` (about 1.11 tol). There is no rim move left, so no flag or
  assertion about which search found a hit is needed (M6 is closed by P21), and the scene's 218 path and 1,000 start
  `kTooManyHits` in the every-scene sets go when the carry is built. A turning solid gets no touching rule of a
  bounding surface (P21 item 10): two flush boxes turning together are hit at their seam, a known limit to pin here
  with one test (`TurningStackedBoxes_SeamIsHit`).
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
- **Amended 2026-11-15 (P19, Tasks 11b and 12):** the wedge, groove, pocket and cusp scenes join
  `BuildMoveParticleCases()`, so `SceneTurnedRigidly_ResultTurnsWithIt` turns them too. Its premise holds where
  the result is well determined: a cusp's rest point is determined along the floor only to about `1e-4` of its
  distance after 8 iterations (P19 item 1), so for the cusp scenes compare the distances to the solids (at the rest
  gap) and the velocities, not the end position to `0x1p-40`; the scenes nearest a limit (the narrow wedge at 2e-6
  rad against `2^-40`, the flat wedges at 6e-7 and 3e-7 rad against the cap) sit a factor 2 or more from it, beyond
  what a rotation's rounding moves; a scene with `kStartsInNarrowGap` compares its discrete fields only. Count the
  generated sets of Tasks 11b and 12 again with the production tolerance (`ComputeContactTolerance({1100, 0.01,
  0.01})`, far from the origin, as the oblique set), the tangent set with radii up to 1000 m (`R / tol` near `2^40`):
  no failure but the ones a shorter step cures; the end distances at least `tol / 2`. The control of the oblique set
  is not repeated.
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
  normal for any pose, `FindFirstHit`'s closed forms, the rim search (amended 2026-11-16, P21: it replaces the rim
  move; a loop of at most `kMaxRimAdvanceCount` samples) and the touching rule use only `+ - * /`,
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
