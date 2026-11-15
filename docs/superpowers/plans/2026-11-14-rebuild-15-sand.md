# Rebuild Sand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sand, the last material of the rebuild, as one more `MaterialKind` on the materials route, carry it
through every module that names a material (step, I/O, render output, viscosity, cell resize, the Blender add-on)
so a sand scene runs end to end, commit the eight-step diagnosis of its known settling creep as tests, then test
the creep's candidate causes one hypothesis at a time and hand over an honest account of which steps pass.

**Architecture:** Sand adds one shared formula header `native/fluidity/materials/sand_material.h`
(`ComputeSandResponse`, `ComputeVelocityChangeAfterHit`) and fills the seven extension points of materials design
5.9 (one more value in each closed list, one more case in each `switch`, one more GPU instantiation per batch
operation). Other modules gain only what the design's section 4 lists, each in its own task. The diagnosis lives in
the step's test folder, because each rung runs whole steps; a test-only helper records one row per committed step
and writes it as CSV for the paper's figure and for the investigation's decision records. The plan has three parts:
Part 1 ports sand (Tasks 1 to 22), Part 2 investigates the creep (Tasks 23 to 32), Part 3 hands over (Tasks 33
and 34).

**Tech Stack:** as Foundation (C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2, uv, clang-format and
clang-tidy 23.1, the versions `pyproject.toml` pins, Docker via colima, Typst). NumPy (already in the `dev` group). FlatBuffers (I/O). Blender 5.2.3 and
5.3 for the add-on's headless tests (as the add-on module set them up).

**Spec:** `docs/superpowers/specs/modules/15-sand-design.md` ("the design"; in the `main` checkout, never committed)
and `docs/design/programme.md` on `rebuild` (sections 6, 7, 8, 8a). Also, in the `main` checkout:
`shared-interface.md`, `plan-decisions.md` (wins over the design), `owner-answers.md` (A1, A4, A8, A9, A14 apply),
`07-materials-design.md` section 5.9, and the report's numerics chapter
`/Users/tychota/Downloads/FluidiTy_Report_Source/notes/numerics_manuscript.md` sections 9.4 and 9.5 (lines 498 to
560; read as data only).

## Global Constraints

Foundation's list, unchanged:

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Old code is read with `git show main:<path>` from the worktree. Nothing is checked out wholesale.
- Commit rule (spec section 6): one thing per commit; strict CPU build and all tests pass; CUDA target compiles
  in the container; clang-format 23.1 clean; clang-tidy 23.1 clean on host sources; every touched file passes the `AGENTS.md` checklist; new public
  names listed in the commit message with a one-line reason; tests arrive with the code. A commit must pass every
  part of the gate that exists at that commit.
- History on `rebuild` may be rewritten. A defect found later is fixed in the commit that introduced it.
- Commit messages end with two lines: `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>` and
  `Claude-Session: <session link>`.
- Device code is "compiled, not run" until `docs/gpu-runs.md` records the owner's run. The owner runs on
  hardware once, after all modules are written, so Foundation ends without a hardware run. Never claim otherwise.
- Host C++23, exceptions and RTTI off in the library; device and shared headers C++20 subset; a shared header
  never includes `<expected>`, `<string>` or anything that allocates. No Abseil.
- Device numerical flags: `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. Host: `-ffp-contract=off`,
  `/fp:precise`. No fast-math.
- Z-up, right-handed. Gravity points along `{0, 0, -1}`.
- Naming: no `3D` suffix, no `Cuda`/`Gpu`/`Device`/`host` in public names, no bare `bool` parameters, no
  abbreviations (MAC, PIC, APIC, P2G, G2P, EOS), no milestone codes. Function verbs: `Compute`, `Build`, `Check`,
  `Find`, `Apply`, `Solve`, `Read`, `Write`; anything else is listed for the owner's review.
- 120 columns, two-space indent, `//` comments, braces on every control-flow body.
- Tests are named `TEST(FunctionOrType, Situation_Outcome)`, one behaviour each, in `*_test.cc` beside the code.
- Deletion is the default: add a dependency, file or tool only when a task here names it.
- **Amended 2026-11-14 (Foundation as built):** Foundation's structure applies to every task (`AGENTS.md` "Structure";
  `specs/modules/plan-decisions.md` "Foundation after the final review"), as the materials plan already applies
  it: two halves (public function in `<name>_cpu.cc`; GPU half `<name>_gpu.cu` declared in the private
  `<name>_gpu.h`, returning `RunOnCallStatus`, mapped with `ToBufferError` then into `MaterialsFailure`; launches,
  waits and temporary memory only through `base/gpu_launch.cuh`); every GPU function a thin loop over
  `FLUIDITY_CPU_AND_GPU` per-item functions in a shared header (materials' `*_steps.h`), which `cpu` tests run over
  every thread block and thread of `ComputeLaunchSize(count)`, in launch order and in reverse, against the CPU
  loop; the scratch pattern, with per-step memory owned by the module's buffers; a failed check writes nothing,
  after `kRunOnCallFailed` the outputs are unspecified and the run stops, `kOutOfMemory` comes only from building or
  allocating forms; discrete outputs equal on CPU and GPU, floating-point within the tolerances below unless a
  header promises the same bits unless NaN; tests link `fluidity_base_test_support` and the modules' helper
  libraries (made with `fluidity_add_test_library`, named in `LINKS`); `gpu` tests call `WriteResultDigest` on
  their main outputs; labels `cpu` everywhere (the contact plan's `fluidity_add_shared_test` is retired), `no_gpu`
  only for behaviour on a machine without a GPU, `gpu` only on hardware; every target through the helpers of
  `cmake/fluidity_targets.cmake`, every source file listed in a CMake file.
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the CPU build in the debug and the release preset, and
  CUDA compiled in the container (its `cuda` step fails for a tracked `.cc` or `.cu` file that no build compiles).

Sand's own:

- **Design section 3 is the contract.** Spell every name exactly as there, with the glossary renames of
  `plan-decisions.md` already applied on the branch: `ParticleContact` is `contact::ParticleContact` (and the field
  `MaterialResponseInputs::contact_record` carries whatever spelling the branch gave it), `InternalForceTotals` is
  `InternalForceTotals`, `step::Support` is `step::CapabilityLevel`. **A known limit is
  `CapabilityLevel::kExperimental`** (the design's `Support::kHasKnownLimit`); an exclusion is
  `CapabilityLevel::kUnavailable` (the design's `kNotSupported`). Read `native/fluidity/step/capability.h` once
  and use the values it has.
- **Closed lists.** `MaterialKind::kSand == 4`, `LimitCase::kShapeReduced == 2`, `LimitCase::kPulledApart == 3`,
  `kLimitCaseCount == 4`, each `static_assert`ed. No `default:` in any `switch` on `MaterialKind` or `LimitCase`
  (materials plan rule): the compiler (`-Wswitch`, warnings are errors) finds every place to extend. No thread
  branches on the kind on the GPU: one instantiation per kind.
- **Formulas ported unchanged** (design 5.2, 5.4, 5.5) from `main`: sand return
  `native/include/fluidity/base/m5_constitutive_3d.h:14-33, 84-157`; sound bound
  `native/include/fluidity/base/solid_acoustic_3d.h:36-92`; hit derivative `native/src/dry_solid_step_3d.cc:134-156,
  513-527`; fitted change cleared after a hit `:570`; sand faces at solids `:367-378`. `main` is Y-up; comparisons
  rotate with `R` rows `{1,0,0},{0,0,-1},{0,1,0}` (`F' = R F R^T`, `x' = R x`).
- **Material numbers are fixed.** The pile material (below) is `main`'s and the design's decision 4. No task changes
  a material number, a friction, the support share, the sound share or a travel share to make a test pass, and no
  task adds damping, a velocity decay, a blend, a force floor or a clamp (design 10 lists them as not ported).
- **Tolerances (design 6, 7).** CPU tests as each task states. GPU against CPU: as the materials plan (state arrays
  `1e-13 + 1e-12 |value|`; force per area `1e-12 (mu + lambda) + 1e-11 |value|`; sound bound `1e-11` relative);
  `LimitCase` exact except within `1e-12` of the cone or of the tip, where either case is accepted;
  `hit_corrected_count` exact (integer sum). Rungs 2, 3 and 5 on the GPU: `1e-9` relative of the CPU at 0.75 s.
- **Known-open tests.** A rung expected to fail asserts the physically right bound and is its own CTest test
  (`add_test` with `--gtest_filter=<Suite>.<Name>`, so each one flips alone), with `WILL_FAIL TRUE` and
  `LABELS "known_open;sand_diagnosis"`, and a row in `docs/known-open-physics.md` holding the test name, what it
  measures, its bound and its last measured numbers. Every rung test (plain or known-open) carries the label
  `sand_diagnosis`. The day a known-open rung passes, CTest turns red; the row and the two properties are removed
  in the commit that made it pass. **Amended 2026-11-14 (Foundation as built):** Foundation's `fluidity_add_test` takes one label among `cpu`,
  `gpu`, `no_gpu` and registers every test of its program under it (`gtest_discover_tests`), and the gate runs
  `cpu` in debug, release and the container. So a known-open or long rung test must not also be discovered under
  `cpu`: decided for every plan (plan-decisions "Plans adopt Foundation as built"; the owner may overturn it): rung tests live in their own test programs whose test names carry
  GoogleTest's `DISABLED_` prefix, so discovery registers them as disabled (skipped, never failing), and each is
  registered a second time with `add_test(NAME <Suite>.<Name> COMMAND <program> --gtest_filter=<Suite>.DISABLED_<Name>
  --gtest_also_run_disabled_tests)` and the properties above (`add_test` creates no C++ target, so the helper rule
  holds). Plain (passing) rung tests are registered the same way without `WILL_FAIL`, so no rung program is
  discovered under `cpu`.
- **The pile material** (plan decision from design 2 and decision 4): `kSand`; `stiffness_against_shear` 450 Pa;
  `stiffness_against_squeeze` 700 Pa; `friction_angle` 0.5235987755982989 rad (30 degrees); `cohesion` 0; one
  ingredient, rest density 300 kg/m3, friction 0.7, viscosity 0, surface tension 0, `kCannotBeSqueezed`;
  `own_motion` `{kept 0, in_travel 0}` (`main`'s grid-velocity advection); `at_solid_faces` `kParticlesOnly`.
- **Diagnosis scenes** (plan decisions; `main`'s `TestScene`, `native/tests/test_m5_dry_scenes_3d.cc:95-160`,
  rotated to Z-up). Gravity `{0, 0, -9.81}` unless stated. One size level of width `w`; the grid frame origin is the
  domain box's low corner; every domain side closed.
  - *Coarse* `w = 0.075` m with step 0.001875 s; *fine* `w = 0.0625` m with step 0.0015625 s. The step is imposed
    through `StepSettings::max_step_duration`; `MaterialSettings::sound_step_share` 0.25.
  - *Floor:* a fixed flat solid through `{0,0,0}` with normal `kUp` and `friction_scale` 1. Rung 8 uses a driven
    flat solid with the same plane and zero velocity for the whole run, because `contact::Solid::reports_reaction`
    is set by solids only for driven and free solids (contact design 3) and rung 8 reads the reactions.
  - *Pile* (rungs 5 to 8): 63 particles at `((i - 1) 0.05, (j - 1) 0.05, 0.10 + (k + 0.5) 0.05)` for `i, j` in
    `0..2`, `k` in `0..6` (ids ascending with `i` fastest, then `j`, then `k`); mass `300 * 0.05^3`, rest volume
    `0.05^3`, springy deformation `I`, permanent volume change 0, at rest. Domain `[-0.6, 0.6]^2 x [-2 w, 0.6]`.
  - *Slab* (rung 4): lattice centres at spacing 0.05 with `x, y` in `{-0.425, -0.375, ..., 0.425}` and `z` in
    `{0.025, 0.075}`, kept when `z <= tan(20 degrees) (0.45 - max(|x|, |y|))`: 356 particles, sides below 30
    degrees. Same mass and volume as the pile. Domain `[-0.6, 0.6]^2 x [-2 w, 0.3]`.
  - *Free block* (rung 2): lattice centres at spacing 0.025 inside `[-0.1, 0.1]^3` (512 particles), positions
    mapped `x -> 1.05 x`, springy deformation `diag(1.05, 1, 1)`, mass `300 * 0.025^3`, rest volume `0.025^3`; the
    pile material with `friction_angle` 1.5533430342749535 rad (89 degrees) and `cohesion` 10 ("limits off");
    no gravity, no solid; domain `[-0.4, 0.4]^3`, `w = 0.05`, step 0.00125 s, run 0.5 s.
  - *Falling column* (rung 3): the pile's 63 particles, no floor, domain `[-0.6, 0.6]^2 x [-1.5, 0.6]`, coarse
    cells, run 0.5 s.
  - *Rest angle* (design 5.7): degrees of `atan2(largest z of any particle, largest hypot(x, y) of any particle)`;
    the floor is `z = 0` and the release centre is `(0, 0)` whatever the grid shift.
  - *Samples:* times 0.75, 1.5, 3 and 6 s; the *window* of a sample is the 0.5 s before it. Every rung records
    kinetic energy and the angle's rate (least-squares slope, degrees per second) over the window, not one angle.
- **Decision records** (Part 2): `docs/decisions/sand-NN-<words>.md`, one per task, with a first line
  `Date, commit measured, rungs before -> after` and exactly five sections `## Problem`, `## Hypothesis`,
  `## Experiment`, `## Result`, `## Decision`. Run data, thinned to one row per 0.01 s, goes beside it in
  `docs/decisions/data/sand-NN/`.

## Before you start

On `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, already committed: Foundation and every
module of the build order before sand (contact, grid, particles, transfer, pressure, viscosity, materials, solids,
gas, step, cell resize and batches, I/O and CLI, surface foam and spray, the Blender add-on), each with its comment
pass done. In particular: materials with `MaterialKind { kLiquid, kGas, kSnow, kSpringySolid }` and the
`static_assert`s on its values; `docs/known-open-physics.md`; `docs/gpu-runs.md`; `blender/fluidity/core/roles.py`
with `ROLE_TABLE`.

Foundation as built (amended 2026-11-14) also provides `run_on_call.h`, `gpu_launch.cuh`, `ToBufferError`, the
scratch forms of the sums and of the stable order (`SumScratch`, `OrderScratch`), `test_support.h` in the library
`fluidity_base_test_support`, and `fluidity_add_test_library`; read `AGENTS.md` "Structure" before Task 1.

Prove it: `uv run python tools/check_commit.py` prints seven `PASS` lines (format, rules, cpu debug, cpu release,
tidy, paper, cuda), and
`git grep -n "kSand\|friction_angle\|cohesion" -- native blender schemas` prints nothing. If either fails, stop and
ask the owner: sand does not start on a red branch, and a second sand is not added beside a first.

## Review Focus

1. **A known-open rung that "passes" because it crashed or failed to set up.** `WILL_FAIL` inverts any failure.
   Every scene a known-open rung runs is also run by a plain test that asserts the run itself (every try commits,
   the asked step was used). Pinned in Task 14: `ComputeSandRecord, Pile_EveryTryCommitsToSixSeconds`.
2. **Zero friction with a tiny shear stiffness.** `1.5 lambda / mu` overflows to infinity and `0 * inf` is NaN;
   the design's "0 when `slope` is 0" must hold. Pinned in Task 3:
   `ComputeSandResponse, ZeroFrictionTinyShear_IsPurelyVolumetric` (`main` case 17).
3. **The solid's velocity at the hit point, not at the start.** On a turning solid the two differ. Pinned in
   Task 6: `ApplyMaterialResponse, SandHitOnTurningSolid_UsesSolidVelocityAtHitPoint`.
4. **A halved step that was not halved.** `ComputeStepDuration` may shorten steps; rung 6 means nothing unless
   every committed step has the asked duration. Pinned in Task 14:
   `ComputeSandRecord, EveryCommittedStepHasTheAskedDuration`.
5. **A record that is not a hit, and a particle with no record.** Only `kHit` splits the step; a null record array
   needs none of the three new inputs. Pinned in Task 6: `ApplyMaterialResponse, SandRecordWithoutHit_NotSplit` and
   `NullParticleContacts_NeedNoHitInputs`.

---

# Part 1: Port sand to every module it touches

### Task 1: Method paper, the sand section, equations and pseudocode

**Kind:** mechanical

Load the `typesetting-with-typst` and `typesetting-documents` skills first.

**Files:**
- Modify: the method paper file holding `<sec-snow>` (find it with `git grep -n "<sec-snow>" docs/method-paper`),
  `docs/glossary.md`

**Interfaces:**
- Produces (design 8): `== Sand <sec-sand>` after `<sec-snow>` inside `<sec-materials>`; `== Known limits of sand
  <sec-sand-limits>` after it (a stub paragraph naming the eight rungs; Task 17 fills the numbers); equations
  `<eq:sand-limit>`, `<eq:hit-derivative>`; pseudocode `<proc-sand-response>`, `<proc-hit-split>`.

- [ ] **Step 1: Write `<sec-sand>`** from design 5.1 to 5.6 in the decided vocabulary (no `trial`, `return
  mapping`, `yield`, `plastic`, `deviator`, `apex`, `granular` except once in parentheses where the plain word is
  introduced): the table row of 5.1; the region allowed to sand (a cone of the shape part against the volume part,
  with its tip at the cohesion) and `<eq:sand-limit>` with `slope = sqrt(2/3) 2 sin(friction_angle) / (3 -
  sin(friction_angle))` and `allowed = -over slope (1 + 1.5 lambda / mu)`; why the pulled-apart test runs first (no
  division by a zero shape part) and why `allowed` is 0 when `slope` is 0 (an infinite stiffness ratio); the route
  for sand (5.3); the hit split (5.4) with `<eq:hit-derivative>` and the three-line derivation of it from contact's
  slip response `vt - friction (-vn) t`; the sound bound (5.5); faces at solids, support rule, travel share (5.6).
  Citations [7] Klar 2016, [18] Stomakhin 2014, [1] Jiang 2015 from `references.typ`; "citation needed" for the
  Hencky row-sum bound.
- [ ] **Step 2: Pseudocode** with `procedure(...)`; step numbers are the contract for `// Step N:` comments:
  `<proc-sand-response>`: 1 logs `e` and their `sum`; 2 `over >= 0`: every `e = cohesion / 3`, permanent grows by
  `over`, case pulled apart; 3 shape part and allowed size; 4 shape larger than allowed: shrink it, case shape
  reduced; 5 force per area `2 mu e + lambda sum`. `<proc-hit-split>`: 1 leaving (`vn >= 0`): unchanged; 2 remove
  the change of the speed into the solid; 3 no friction, or the particle sticks; 4 slipping: the formula of 5.4;
  5 update the deformation in two parts and clear the fitted change.
- [ ] **Step 3: Glossary rows** (`| Say | Plain meaning | Never say |`): sand; friction angle; cohesion; shape part;
  pulled apart; shape reduced; hit split; rest angle; settling creep. "Never say" from design 9: granular, Drucker-
  Prager (except once as a literature word), apex, deviator, cone projection, envelope angle.
- [ ] **Step 4: Run** `typst compile` of the paper. Expected: exit 0, no warning. Read the new pages. Run the gate.
- [ ] **Step 5: Commit.** Subject: `Describe sand, its limits and the hit split`.

### Task 2: Method paper figures for sand

**Kind:** mechanical

Load `drawing-with-cetz`; use `figure_style.typ`.

**Files:**
- Create: `docs/method-paper/figures/sand_limit.typ`, `docs/method-paper/figures/hit_split.typ`
- Modify: the paper file of Task 1

**Interfaces:**
- Produces `<fig-sand-limit>`, `<fig-hit-split>` (design 8). `<fig-sand-creep>` needs measured data and is
  Task 17's.

- [ ] **Step 1: Draw** `<fig-sand-limit>`: the plane (volume part on the horizontal axis, size of the shape part
  on the vertical), the cone of 30 degrees with its tip at the cohesion, and three asked states with arrows: one
  inside (unchanged), one beyond the side (straight down to the cone at fixed volume: shape reduced), one beyond
  the tip (to the tip: pulled apart, the excess becomes permanent). `<fig-hit-split>`: a particle's straight path
  through one step crossing a tilted solid at `hit_fraction` 0.25, the first part labelled with `L` and `duration`,
  the second with `L_after_hit` and `duration_after_hit`, the normal and the sideways direction `t` drawn at the hit.
- [ ] **Step 2: Compile and look.** Fix overlaps, clipped labels, inconsistent sizes. Each caption states the one
  thing to see; each figure is referenced from its subsection. Run the gate.
- [ ] **Step 3: Commit.** Subject: `Draw the sand limit and the hit split`.

### Task 3: Sand limits and `ComputeSandResponse`

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the shared-header test is `fluidity_add_test(... LABEL cpu)`; its rung-1 label follows the known-open proposal of Global Constraints.

**Files:**
- Modify: `native/fluidity/materials/material.h` (`MaterialResponse::friction_angle`, `cohesion`),
  `stretch_response.h` (`LimitCase`, `kLimitCaseCount`), `material_test.cc`, and every file `-Wswitch` names for
  a `switch` on `LimitCase` (find them with `git grep -n "LimitCase::kClamped" native`)
- Create: `native/fluidity/materials/sand_material.h`, `sand_material_test.cc` (`fluidity_add_test(... LABEL cpu
  LINKS fluidity_materials_test_scenes)`; these short tests are rung 1 and also carry `sand_diagnosis` once the
  owner rules on the label question of Global Constraints "Known-open tests")

**Interfaces:**
- Consumes: `StretchResponse`, `MaterialResponse`, `MaterialsError`, `math::Log`, `math::Exp`, `math::Sqrt`,
  `math::Sin`, `math::IsFinite`.
- Produces design 3: `MaterialResponse::friction_angle`, `MaterialResponse::cohesion` (appended after the
  existing fields), `LimitCase::kShapeReduced`, `LimitCase::kPulledApart`, `kLimitCaseCount = 4`,
  `ComputeSandResponse`. Private names (design 9): `shape_log_stretch`, its size, `allowed_shape_size`. Ported from
  `main` `m5_constitutive_3d.h:84-157`; the "representable" checks become one finiteness check. A `LimitCase`
  switch elsewhere that cannot meet the new values (snow code) lists them with the same branch as `kClamped` and a
  comment "sand only".

- [ ] **Step 1: Failing tests** (`mu = 2`, `lambda = 3`, `friction_angle` 0.5235987755982989, `cohesion` 0.03,
  permanent 0 unless stated):

  ```cpp
  TEST(LimitCase, CountMatchesLastValue)          // extend: kShapeReduced 2, kPulledApart 3, kLimitCaseCount 4
  TEST(ComputeSandResponse, InsideLimits_IsUnchanged)   // {0.98,0.98,0.98}, cohesion 0: same stretches,
      // kInsideLimits, permanent 0, every force 13 ln 0.98 (= (2 mu + 3 lambda) ln 0.98); 1e-13
  TEST(ComputeSandResponse, MainCase_IsUnchanged)       // {0.9,0.9,0.9}, permanent 0.2: stretch 0.9, permanent 0.2,
      // force 13 ln 0.9 (main m5_constitutive_3d_cases.h:26-35); 1e-11
  TEST(ComputeSandResponse, PulledApart_GoesToCohesionPoint)   // {1.2,1.1,1.0}: every stretch exp(0.01),
      // permanent ln(1.32) - 0.03, kPulledApart, force 0.13 each; then cohesion 0: stretches 1, permanent ln 1.32,
      // force exactly 0; 1e-13
  TEST(ComputeSandResponse, ThirtyDegrees_GivesThreeToOne)   // forces asked (-5,-1,-1) Pa: logs
      // e_i = (f_i + 21/13) / 4 (sum -7/13), stretches exp(e_i), permanent 0.2: forces (-4.2,-1.4,-1.4),
      // kShapeReduced, sum of logs and permanent unchanged; 1e-11 (main's triaxial check)
  TEST(ComputeSandResponse, ShapeReduced_MatchesMainCaseFour)   // {exp(0.4),exp(-0.1),exp(-0.6)}, permanent 0.2:
      // logs -0.1 + d, -0.1, -0.1 - d with d = (13/4) 0.33 sqrt(2/3) 0.4 / sqrt(2); permanent 0.2; 1e-11
  TEST(ComputeSandResponse, ZeroFrictionTinyShear_IsPurelyVolumetric)   // Review Focus 2, main case 17: mu 1e-310,
      // angle 0, cohesion 0, same stretches: every stretch exp(-0.1), permanent 0.2, all finite; 1e-11
  TEST(ComputeSandResponse, NonPositiveStretch_IsBadInputUntouched)    // {0,1,1}; then permanent 1000 (exp not
      // finite): kBadInput; output bytes equal the sentinel written before the call
  TEST(ComputeSandResponse, BadLimits_AreBadMaterialUntouched)   // angle pi/2; angle -0.1; cohesion -1e-3; mu 0
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement** with `// Step N:` of `<proc-sand-response>`; `Read:` `<sec-sand>`, `<eq:sand-limit>`.
  The pulled-apart test runs first; `allowed` is set to 0 without forming `1.5 lambda / mu` when `slope == 0`.
  Field comments of the two new `MaterialResponse` fields as design 3 (units first).
- [ ] **Step 4: Run** build, tests, format, rules, gate.
- [ ] **Step 5: Commit.** Subject: `Add the sand response inside the stretch directions`. New names: the list above.

### Task 4: `ComputeVelocityChangeAfterHit`

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/materials/sand_material.h`, `sand_material_test.cc`

**Interfaces:**
- Consumes: `Mat3`, `Vec3`, `math::Dot`, `math::Length`; in the test only, `contact::ComputeContactResponse`.
- Produces design 3: `ComputeVelocityChangeAfterHit`, design 5.4 steps 1 to 4 exactly, ported from `main`
  `dry_solid_step_3d.cc:134-156` with the normal as an input instead of `{0,1,0}`.

- [ ] **Step 1: Failing tests** (`L` rows `{0.3,-1.2,0.5}, {0.8,0.1,-0.4}, {-0.6,0.9,0.2}` 1/s; normal
  `{0.36, 0.48, 0.8}`; friction 0.7):

  ```cpp
  TEST(ComputeVelocityChangeAfterHit, MatchesContactResponse)   // design 7: velocity at the hit point v0 with
      // vn = -1 m/s and a sideways part of speed 2 (slip), 0.5 (stick), and vn = +0.3 (leaving); column k of the
      // expected matrix = (R(v0 + h L e_k) - R(v0 - h L e_k)) / (2 h), R = contact::ComputeContactResponse(v,
      // {0,0,0}, n, 0.7), h = 1e-6: equal within 1e-8. Stick gives the zero matrix; leaving gives L exactly
  TEST(ComputeVelocityChangeAfterHit, NoFriction_RemovesOnlyTheNormalChange)   // friction 0: L - n (n^T L), 1e-15
  TEST(ComputeVelocityChangeAfterHit, MovingSolid_SameAsMovingFrame)   // the finite difference of R(v + u, u, n,
      // 0.7) with u = {0.7,-0.2,0.4} against the function at velocity v relative to the solid: 1e-14 of the at-rest
      // result
  ```
- [ ] **Step 2: Run** (compile error), **implement** (`// Step N:` of `<proc-hit-split>`, steps 1 to 4; `Read:`
  `<eq:hit-derivative>`), **run**.
- [ ] **Step 3: Gate; commit.** Subject: `Add the velocity change of sand after a hit`.

### Task 5: `kSand` through the table and every switch

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the sand case of each GPU batch operation goes into its shared `*_steps.h` per-particle function (materials plan), so the `cpu` steps tests of materials cover it; shared tests are label `cpu`.

**Files:**
- Modify in `native/fluidity/materials/`: `material.h` (`kSand`, `HasInternalForce`), `material_table.cc`,
  `material_response.h` (`MotionInStep` fields and every per-kind function), `own_motion_rule.h`, every batch
  operation's `_cpu.cc` and `_gpu.cu` that `-Wswitch` names, and their tests
- Modify outside materials: every file `-Wswitch` names for a `switch` on `MaterialKind`
  (`git grep -n "MaterialKind::kSpringySolid" -- native schemas` lists the candidates); `schemas/simulate/v2/common.fbs`
  (`MaterialKind` gains `sand`, appended, value 4) if such a switch maps to it

**Interfaces:**
- Produces design 3 and 3's "Defaults and table rules": `MaterialKind::kSand`, `HasInternalForce(kSand) == true`,
  `MotionInStep::exact_velocity_change_after_hit`, `MotionInStep::duration_after_hit`; table rules
  `0 <= friction_angle < pi/2` and `cohesion >= 0` (checked in field order, `kBadMaterial`), one ingredient,
  `kCannotBeSqueezed`; `BuildParticlesSettings` gives sand `springy_deformation`, `permanent_volume_change`,
  `kKeepsItsCount`; `ComputeOwnMotionInTravel` gives sand `{kept, 0}`. The sand cases of
  `ComputeInternalForcePerArea` (start of step: directions, then design 5.2 step 3 with no limit, world form as
  snow), `ComputeMaterialResponse` (design 5.3 end of step: `F <- (I + duration L) F`, then, when
  `duration_after_hit > 0`, `F <- (I + duration_after_hit L_after_hit) F`; directions; `ComputeSandResponse`;
  rebuild; `volume_ratio = product * exp(permanent)`), `ComputeSoundSpeedBound` (design 5.5, ported from
  `solid_acoustic_3d.h:36-92`; `math::Expm1` for the pair terms), `CheckMaterialState` (zero motion re-run, as
  snow). Each GPU batch operation gains one `case MaterialKind::kSand:` instantiation of its per-particle GPU
  function, whose body is the sand case of the operation's shared `*_steps.h` function (materials plan Tasks 15 to
  26), so the materials `cpu` steps tests, extended with one sand case each, cover it.
- **Plan decision (interim, replaced in Task 6):** `ApplyMaterialResponse` returns `kBadInput`, nothing written,
  for a sand material whose contact record array is not null (its hit inputs do not exist yet).
- **Plan decision (other modules):** each switch outside materials gets the sand case with the snow case's
  behaviour, except where the later task of this plan names a sand-specific one: I/O kind name `"sand"`
  (Task 18); render output "points, no surface" (Task 19); step material sets "its own set" (A8). The later task
  adds the test.

- [ ] **Step 1: Failing tests** (label `cpu`, shared ones included; the pile material unless stated; each
  materials `*_steps_test.cc` gains a sand particle in its every-thread-block-and-thread run):

  ```cpp
  TEST(MaterialKind, StoredValues_AreFixed)            // extend: kSand == 4
  TEST(HasInternalForce, SnowSpringySolidAndSand)      // renamed from OnlySnowAndSpringySolid; liquid, gas false
  TEST(BuildMaterialTable, SandRules)                  // design 7, each alone on a one-sand table: angle
      // 1.5707963267948966; cohesion -1e-3; two ingredients: kBadMaterial with material_index 0; angle and
      // cohesion both bad: the angle's failure; angle 0 and cohesion 0 accepted
  TEST(BuildParticlesSettings, Sand_SpringyAndPermanentAndKeepsItsCount)
  TEST(ComputeOwnMotionInTravel, Sand_KeptOnly)        // all three rule inputs true -> {kept, 0}
  TEST(ComputeSoundSpeedBound, SandAtRest_IsRowSum)    // design 7: mu 2, lambda 3, density 2: sqrt(6.5); 1e-13
  TEST(ComputeInternalForcePerArea, SandRotated_IsWorldFormOfStretchForce)   // F = R diag(0.99,1,1.005) R^T,
      // R = math::Rotation({0.3,-0.2,0.5}): R diag(2 mu e + lambda sum) R^T within 1e-13 (mu + lambda)
  TEST(ComputeMaterialResponse, SandTwoParts_EqualsTwoProducts)   // F as above, L of Task 4, L_after = 0.5 L,
      // dt 0.002, duration 0.0005 and 0.0015: deformation before limits == (I + 0.0015 L_after)(I + 0.0005 L) F,
      // then the response equals ComputeSandResponse of its stretches; 1e-14 of the largest entry
  TEST(ComputeMaterialResponse, SandSqueezedSideways_IsShapeReduced)   // stretches of Task 3's ThirtyDegrees
      // case as F: kShapeReduced, volume_ratio == product * exp(permanent) within 1e-14
  TEST(CheckMaterialState, SandOutsideCone_IsOutsideAllowedRegion)   // stored stretches {exp(0.3), exp(-0.1),
      // exp(-0.5)} (shape size 0.566 > allowed 0.327)
  TEST(ApplyMaterialResponse, SandHit_MissingInputs_IsBadInput)   // interim: sand with a kHit record: kBadInput,
      // nothing written (Task 6 extends it)
  ```
- [ ] **Step 2: Run.** Expected: compile errors at every switch; that list is the work list.
- [ ] **Step 3: Implement** each case; `Read:` `<sec-sand>`. Memory estimate entries need no change (sand counts as a material
  with internal force through `HasInternalForce`); `BuildMaterialsMemoryEstimate, AllKinds_BytesFromSizeof` must still
  pass.
- [ ] **Step 4: Run** build, tests, gate. Expected: `PASS cuda` (the sand instantiations compile).
- [ ] **Step 5: Commit.** Subject: `Take sand through the material table and every material switch`. Body:
  `CUDA: compiled, not run`; the interim `kBadInput` for sand hits. New names: the list above.

### Task 6: The hit split in `ApplyMaterialResponse` on the CPU

**Kind:** integration

**Files:**
- Modify: `native/fluidity/materials/apply_material_response.h`, `apply_material_response_cpu.cc`,
  `apply_material_response_test.cc`, `materials_test_scenes.h/.cc`
- Create: `native/fluidity/materials/sand_repeat_run_test.cc`

**Interfaces:**
- Consumes: `ComputeVelocityChangeAfterHit`, `contact::ParticleContact`, `contact::ContactKind::kHit`,
  `contact::Solid`, `contact::ComputeSolidPose`, `contact::ComputeSolidVelocity(solid, pose, point)`,
  `MaterialBuffers::friction`, `kCarriesFittedChange`.
- Produces design 3: `MaterialResponseInputs::start_position`, `velocity_for_this_step`, `solids`;
  `MaterialResponseResult::hit_corrected_count`; design 5.4 per sand particle with a `kHit` record: `duration =
  hit_fraction * dt`, hit point `start_position + velocity_for_this_step * duration`, solid pose at the step time of
  the hit, `v = velocity_for_this_step - solid velocity at the hit point`, `friction = MaterialBuffers::friction
  [i] * solids[solid_index].friction_scale`; the particle loses `kCarriesFittedChange` and is counted. The interim
  rule of Task 5 is replaced by design 3's: with a non-null record array, any of the three inputs null or empty
  is `kBadInput`; for every other kind they are not read. Test helper (plan decision, `materials::test`):
  `BuildHitRecord(double hit_fraction, Vec3 normal_out_of_solid, std::uint32_t solid_index)`.

- [ ] **Step 1: Failing tests** (label `cpu`; one sand particle of the pile material unless stated; `L` of Task 4;
  dt 0.002; stored `F = R diag(0.99, 1, 1.005) R^T` with Task 5's `R`):

  ```cpp
  TEST(ApplyMaterialResponse, SandHit_SplitsStep)   // design 7: fixed flat solid normal {0.36,0.48,0.8}, record
      // hit_fraction 0.25, particle moving into it at 1 m/s with a sideways part 2 m/s: deformation equals
      // ComputeMaterialResponse with MotionInStep {L, 0.0005, ComputeVelocityChangeAfterHit(...), 0.0015};
      // hit_corrected_count 1; kCarriesFittedChange cleared; 1e-14
  TEST(ApplyMaterialResponse, SandHit_MissingInputs_IsBadInput)   // extend: null start_position, null
      // velocity_for_this_step, empty solids, each in turn: kBadInput, nothing written
  TEST(ApplyMaterialResponse, SandHitOnTurningSolid_UsesSolidVelocityAtHitPoint)   // Review Focus 3: a driven flat
      // solid through the origin turning about z at 2 rad/s; particle start (0.3, 0, 0.01), velocity (0, 0, -10),
      // hit_fraction 0.5: equals the result built from the solid velocity at the hit point, and differs by more
      // than 1e-6 from the one built at the start point
  TEST(ApplyMaterialResponse, SandRecordWithoutHit_NotSplit)       // Review Focus 5: a record of a kind other than
      // kHit: one-part update, count 0, flag kept
  TEST(ApplyMaterialResponse, NullParticleContacts_NeedNoHitInputs) // sand, null records, null inputs: succeeds
  TEST(ApplyMaterialResponse, SnowIgnoresHitInputs)                // snow with a hit and null inputs: succeeds
  TEST(ApplyMaterialResponse, SandHitCountsSumOverRanges)          // two ranges: counts add to the one-call count
  // sand_repeat_run_test.cc
  TEST(Materials, EveryBatchOperationTwiceOnSand_SameBytes)        // design 7 "Repeat run": a sand block x
      // [0.4,0.8)^3 with F = diag(0.98,1,1) through the seven batch operations twice on fresh copies
  ```
- [ ] **Step 2: Run** (fail), **implement** (`// Step N:` of `<proc-hit-split>` step 5; `main` `:513-527`, `:570`),
  **run**.
- [ ] **Step 3: Gate; commit.** Subject: `Split a sand particle's step where it hits a solid`. New names: the three
  input fields, `hit_corrected_count`.

### Task 7: Sand and the hit split on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** the split lives in the shared `apply_material_response_steps.h` (CPU loop and GPU function both call it), tested over every thread block and thread on the CPU; the per-call solids copy, if needed, is a `GpuArray` and makes the sand GPU path an allocating form; the shared-formula runner is materials' `shared_formulas_test_gpu.h/.cu`; result digests.

**Files:**
- Modify: `native/fluidity/materials/apply_material_response_gpu.cu`, `apply_material_response_gpu.h`,
  `apply_material_response_steps.h` (the sand case of `ApplyParticleMaterialResponse` with the hit split, which
  `apply_material_response_cpu.cc` of Task 6 then calls), `apply_material_response_steps_test.cc`,
  `apply_material_response_gpu_test.cc`, `shared_formulas_test_gpu.cu`, `shared_formulas_gpu_test.cc`,
  `materials_no_gpu_test.cc`

**Interfaces:**
- Produces: the `kSand` instantiation of `ApplyMaterialResponseOneThreadPerParticle` with the split in registers in
  the same pass (design 6); `hit_corrected_count` by integer `atomicAdd`. **Plan decision for the solids on the
  processor:** read how `native/fluidity/contact/move_particles_gpu.cu` gets `MoveParticlesInputs::solids` to the
  device and do the same; if contact's device copy is private, copy `solids` per call into a
  `GpuArray<contact::Solid>` held by the GPU half (one small allocation per call, so the sand GPU path may return
  `kOutOfMemory`; record it in the hand-over as a departure from "allocates nothing").

- [ ] **Step 1: Failing tests.** `cpu` (`apply_material_response_steps_test.cc`, the main evidence until the
  hardware run): the sand case of `ApplyParticleMaterialResponse` with `kHit` records for every thread block and
  thread of `ComputeLaunchSize(particle count)`, launch order and reverse, on the Task 6 scenes: states bit for bit
  and `hit_corrected_count` equal to the CPU loop's. `gpu`:
  `ApplyMaterialResponse, OnGpuSandHit_MatchesCpuOnEveryTaskSixScene`
  (Global Constraints tolerances; `LimitCase` rule near the cone); `ApplyMaterialResponse, OnGpuSandRunTwice_SameBytes`;
  `SharedFormulas, OnGpu_MatchCpu` extended with every case of Tasks 3, 4 and 5. `no_gpu`:
  `ApplyMaterialResponse, OnGpuSandWithoutAGpu_IsRunOnCallFailed`. Each `gpu` test passes the written states and
  `hit_corrected_count` to `WriteResultDigest`.
- [ ] **Step 2: Implement; run** the gate. Expected: `PASS cuda`; `no_gpu` tests pass in the container.
- [ ] **Step 3: Commit.** Subject: `Split sand steps at a hit on the GPU`. Body: `CUDA: compiled, not run`.

### Task 8: Python reference for sand and the comparison with `main`

**Kind:** numerics

**Files:**
- Modify: `reference/materials/response.py`, `reference/materials/response_test.py`,
  `native/fluidity/materials/material_reference_test.cc`
- Create: `native/fluidity/materials/testdata/sand_response_cases.txt`, `sand_sound_speed_cases.txt`,
  `main_sand_cases.txt`

**Interfaces:**
- Produces `sand_response(deformation, permanent, response)` beside `snow_response` (through `numpy.linalg.svd`,
  signs fixed so both factors are rotations; the cone and tip of design 5.2; world force and rebuilt deformation)
  and the sand energy `mu sum(e^2) + lambda / 2 sum(e)^2` for `fastest_wave`; `write_cases` adds, with the same
  seed `numpy.random.default_rng(20261114)`: 1,000 sand cases `F = R1 diag(s) R2^T` with `s` in `[0.85, 1.05]`,
  permanent in `[-0.2, 0.2]`, the pile material's stiffnesses and angle, cohesion 0.03 (so all three cases occur;
  the generator asserts each case appears at least 50 times); 100 sand states (`s` in `[0.9, 1.0]`, inside the cone)
  with 10,000 directions on the sphere. Numbers as `float.hex`.
- The `main` numbers (`plan-decisions.md`, first entry): export `main` to the scratchpad, build there, run a scratch
  driver (never committed) calling `ReturnGranularPrincipal3D` on the sand inputs of
  `native/tests/m5_constitutive_3d_cases.h:26-63` (cases 1 to 4 and 17) and `GranularAcousticBoundSquared3D` on
  ten stretch states of the 100 above, printing inputs and outputs as hex floats. These functions work in the
  frame of the stretches, so no rotation applies. Commit only `main_sand_cases.txt` with a header saying how it was
  produced and `main`'s commit hash.

- [ ] **Step 1: Failing tests.** Python: `test_cases_files_are_current` (extended), `test_sand_at_rest_is_zero_force`,
  `test_sand_cases_cover_three_limit_cases`. C++ (label `cpu`):

  ```cpp
  TEST(ComputeMaterialResponse, SandMatchesSvdReference)     // zero motion on each stored F: deformation, permanent,
      // world force within 1e-10 relative to the largest entry; LimitCase equal except within 1e-12 of a limit
  TEST(ComputeSoundSpeedBound, SandAtLeastSampledFastestWave)  // native >= sampled * (1 - 1e-8) for all 100 states;
      // if one fails, stop and report the state: the bound is not relaxed
  TEST(ComputeSandResponse, MatchesMainCases)                // 1e-12 relative
  TEST(ComputeSoundSpeedBound, SandMatchesMainCases)         // squared bound, 1e-12 relative
  ```
- [ ] **Step 2: Run** (fail), **write** the reference and the driver, generate, **run**.
- [ ] **Step 3: Gate; commit.** Subject: `Check sand against a NumPy reference and main`. Body: the uncertainty
  removed (design 7: both branches on general rotated states; the bound for a general state) and how
  `main_sand_cases.txt` was produced.

### Task 9: Solids: sand faces at a solid, and sand blocks liquid

**Kind:** integration

**Files:**
- Modify: `native/fluidity/solids/collide_solids_test.cc`, `native/fluidity/solids/solid_coverage_test.cc`

**Interfaces:**
- Consumes: `solids::CollideSolids`, `ComputeSolidCoverage`, `MaterialSetFaces::at_solid_faces`,
  `MaterialTheLiquidCannotEnter`, the test library `fluidity_solids_test_scenes` (`SceneSolids`), the materials test
  tables. Tests only (design 4, solids).

- [ ] **Step 1: Tests** (label `cpu`):

  ```cpp
  TEST(CollideSolids, Sand_FacesUntouchedAtSolid)   // design 7: a sand set (kParticlesOnly) and a snow set
      // (kNoMotionIntoSolid) side by side moving at (0.3, 0, -1) m/s toward a fixed flat solid at z = 0.2: every
      // sand face velocity byte-equal to its input; snow faces lose exactly the inward part (z) and keep 0.3
  TEST(ComputeSolidCoverage, Sand_BlocksLiquid)      // design 7: a sand layer (blocking material, particle_radius
      // half the particle spacing) under liquid: at each corner inside a sand ball, distance_to_solid equals
      // min over sand particles of (|corner - particle| - radius), negative; 1e-12
  ```
- [ ] **Step 2: Run.** Expected: both pass (sand reaches solids through existing rules). If either fails, the defect
  is in the owning solids commit: fix it there (`git commit --fixup`) and say so in this commit's body.
- [ ] **Step 3: Gate; commit.** Subject: `Show that solids leave sand faces alone and block liquid with sand`.

### Task 10: Cell resize refuses to split a sand cell

**Kind:** mechanical

**Files:**
- Modify: the test file of cell resize's "Refused split" test
  (`git grep -ln "RefusedSplit" native/fluidity/cell_resize`)

- [ ] **Step 1: Test** (label `cpu`): `ResizeCells, RefusedSplit_Sand` (design 7): the scene of the existing
  `RefusedSplit` test with its snow material replaced by the pile material: the split request is dropped,
  `refused_split_count` grows by the number of refused cells, the grid's cell list is byte-equal before and after.
- [ ] **Step 2: Run.** Expected: pass (`CountRule::kKeepsItsCount` from Task 5). **Step 3:** gate. **Commit.**
  Subject: `Show that cells holding sand keep their size`.

### Task 11: Viscosity excludes sand beside viscous liquid

**Kind:** mechanical

**Files:**
- Modify: `native/fluidity/viscosity/viscosity_error.h`, its message and `ToName` file, `viscosity_error_test.cc`,
  `solve_viscosity_and_pressure.h` (`ViscositySceneSummary`), `viscosity_capability.cc`,
  `viscosity_capability_test.cc`, `viscosity_known_open_test.cc`, its CTest lines, `docs/known-open-physics.md`

**Interfaces:**
- Produces design 4 (viscosity), decision 5: `ViscositySceneSummary::has_sand` (appended last) and
  `ViscosityError::kSandNotSupported`. **Plan decision:** the value is appended at the end of the enum so no stored
  value shifts; `CheckViscosityCanRun` checks it after the existing exclusions, with the comment naming the test
  that lifts it.

- [ ] **Step 1: Failing tests:** `ToName, EveryValue` (count + 1, `"sand_not_supported"`);
  `CheckViscosityCanRun, EachExclusion_ReturnsItsError` (the sand summary gives `kSandNotSupported`);
  `CheckViscosityCanRun, SandWithoutViscousIngredient_CanRun`; known-open (its own CTest test, `WILL_FAIL`):
  `ViscosityKnownOpen, SyrupOnSandBed_StaysAtRest` (lifts `kSandNotSupported`; asserts `CheckViscosityCanRun(
  {true, ..., has_sand})` has a value, then the syrup column with its lower half given as a resting solid coverage:
  speeds `<= 1e-9` m/s, as the snow-bed test of the viscosity plan).
- [ ] **Step 2: Run** (fail), **implement**, **run**; add the known-open row. **Step 3:** gate. **Commit.**
  Subject: `Exclude sand beside thick liquid until it is shown to rest`. New names: `has_sand`, `kSandNotSupported`.

### Task 12: The step's capability check and the stable names

**Amended 2026-11-14 (A27):** the main sequence removed `SuggestedChange::kSetViscosityToZero` (no reason there uses
it once the free-solid exclusion is lifted; `plan-decisions.md`, "A viscous liquid sticks, and free solids in the
viscous solve (A26, A27)", "Exclusions"), so this task adds it back for `kViscousLiquidWithSand`: appended after
the existing `SuggestedChange` values, `kSuggestedChangeCount` + 1, stable name `"set_viscosity_to_zero"`, its
`ToMessage`.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/capability.h` (`CapabilityReason`, `SuggestedChange`), `capability.cc`,
  `capability_test.cc`, `step_message.cc`, `stage_outcome_from_modules.cc`, `stage_outcome_from_modules_test.cc`,
  `step_test_scenes.h/.cc`, `native/fluidity/io/stable_names.cc`, `stable_names_test.cc`

**Interfaces:**
- Produces design 4 (step): `CapabilityReason::kSandSettlingCreep`, `CapabilityReason::kViscousLiquidWithSand`,
  appended after the existing values; `kCapabilityReasonCount` + 2 (both tables are `static_assert`ed, so step
  and I/O change in this one commit). Stable names `"sand_settling_creep"`, `"viscous_liquid_with_sand"`. (A27)
  `SuggestedChange::kSetViscosityToZero`, appended after the existing values (the main sequence has none);
  `kSuggestedChangeCount` + 1 (its name table is `static_assert`ed too); stable name `"set_viscosity_to_zero"`;
  `ToMessage` (plan decision) "Set the liquid's viscosity to 0: a viscous liquid beside sand cannot run yet".
  Level and suggestion (plan decision): `kSandSettlingCreep` `kExperimental`, `kNone`, one finding per sand
  material with its `material_id`, message "Sand: long settling is not qualified. A heap's slope keeps creeping
  after about one second; see docs/known-open-physics.md."; `kViscousLiquidWithSand` `kUnavailable`,
  `kSetViscosityToZero`. The viscosity summary sets `has_sand` when any material is `kSand`. Mapping:
  viscosity `kSandNotSupported` -> `kNotSupportedNow` with capability `kViscousLiquidWithSand`.
- Produces (test scenes, plan decision, `step::testing`): `enum class SandFloor : std::uint8_t { kFixed,
  kDrivenAtRest };` `struct SandPileSettings { double cell_width = 0.075; double step_duration = 0.001875; Vec3
  grid_shift{}; SandFloor floor = SandFloor::kFixed; };` `TestScene BuildSandPileScene(const SandPileSettings&)`
  (the pile of Global Constraints; `grid_shift` moves the domain box and the grid frame, never the particles or
  the floor); `TestScene BuildSandSlabScene(double cell_width, double step_duration)`; `TestScene
  BuildFreeSandBlockScene()`; `TestScene BuildFallingSandColumnScene()`; `TestScene BuildSyrupOnSandScene()` (the
  syrup of `BuildSyrupColumnScene` over a sand layer `[-0.1,0.1]^2 x [0, 0.05]`).

- [ ] **Step 1: Failing tests** (label `cpu`):

  ```cpp
  TEST(CheckSceneCanRun, Sand_IsExperimentalWithSettlingCreep)   // BuildSandPileScene({}): one finding, reason
      // kSandSettlingCreep, kExperimental, kNone, material_id of the sand; can_run true
  TEST(CheckSceneCanRun, ViscousWithSand_IsExcluded)    // BuildSyrupOnSandScene: kViscousLiquidWithSand, kUnavailable,
      // module kViscosity, module_code "sand_not_supported", kSetViscosityToZero; can_run false
  TEST(CheckSceneCanRun, SandAcrossSizeChange_IsKnownLimit)      // the pile on two size levels:
      // kInternalForceAcrossSizeChange is among the findings
  TEST(ToStageOutcome, ViscositySandNotSupported_IsNotSupportedNow)   // capability kViscousLiquidWithSand
  TEST(ToName, EveryCapabilityReason_NamedAndUnique)    // extended count; the two new names
  TEST(ToName, EverySuggestedChange_NamedAndUnique)     // A27: count + 1 (13 with kNone); "set_viscosity_to_zero"
  TEST(BuildSandPileScene, SixtyThreeParticlesAtMainsPlaces)     // ids, positions exact, mass 0.0375 kg
  TEST(BuildSandSlabScene, ThreeHundredFiftySixParticles)
  ```
- [ ] **Step 2: Run** (fail), **implement**, **run**. **Step 3:** gate. **Commit.** Subject:
  `Report sand as experimental and exclude it beside thick liquid`. New names: the list above.

### Task 13: `MoveParticles` gives sand its hit inputs; the step report and the reactions

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the `gpu` test compares floating-point results within the Global Constraints tolerances (discrete results exactly) and writes result digests.

**Files:**
- Modify: `native/fluidity/step/move_particles_cpu.cc`, `move_particles_test.cc`, the header declaring
  `StepReport`, the header declaring `StepTry` and its source, `step_stages_test.cc` (or the file holding
  `TryStep, SnowStageTrace`), `move_particles_gpu.cu`

**Interfaces:**
- Produces design 4 (step): in `MoveParticles` (step 5.5, step 5) the three new `MaterialResponseInputs` fields
  from the same `contact::MoveParticlesInputs` given to contact (design decision 3 (a)); `StepReport::
  hit_corrected_count` (integer sum over ranges and materials).
- **Plan decisions** (the diagnosis needs them; nothing in production reads them):
  `StepReport::particle_count_by_case` (`std::array<std::uint64_t, materials::kLimitCaseCount>`, integer sum of
  every `MaterialResponseResult::particle_count_by_case` of the try); and on `StepTry` the read-only accessor
  `std::span<const contact::SolidReaction> solid_reactions(std::size_t material_index) const`: the reactions
  `contact::MoveParticles` wrote for that material's particles in this try (indexed as the material's particles of
  `particles_try()`), valid after stage `MoveParticles`; on a GPU run, an empty span.

- [ ] **Step 1: Failing tests** (label `cpu`):

  ```cpp
  TEST(TryStep, SandColumnHitsFloor_HitCorrectedCountIsBottomLayer)   // BuildSandPileScene({}) with every particle
      // lowered by 0.099 m (bottom layer at z = 0.001) and velocity (0, 0, -1): one step: hit_corrected_count 9;
      // kCarriesFittedChange cleared on exactly the nine bottom particles
  TEST(TryStep, SandStageTrace)                     // the stage list equals SnowStageTrace's
  TEST(TryStep, SandCaseCounts_CoverEveryParticle)  // sum of particle_count_by_case == 63 every step
  TEST(StepTry, SolidReactionsAfterMoveParticles_HoldTheNormalChange)   // floor kDrivenAtRest, one pile particle
      // lowered to z = 0.001 moving (0.2, 0, -1): its reaction's velocity_for_next_step_change has a z part between
      // 1 and 1 + 9.81 dt m/s (the approach speed) and, since 0.2 < 0.7 times that, the particle sticks: the x
      // part is -0.2 within 1e-12; every other particle's reaction is zero
  TEST(TryStepWithStages, RejectAfterMoveParticlesWithSand_ChangesNothing)
  ```
  `gpu` (in the step's GPU scene test file): `MoveParticles, OnGpuSand_EqualsCpu` (amended: `hit_corrected_count`,
  `particle_count_by_case` and flags exactly; positions, velocities and states within the Global Constraints
  tolerances, since no header promises the same bits; those arrays through `WriteResultDigest`).
- [ ] **Step 2: Run** (fail), **implement**, **run**. **Step 3:** gate. **Commit.** Subject:
  `Give sand its hit inputs when particles move`. Body: the two plan-decision additions and why (Task 14's
  diagnosis); `CUDA: compiled, not run`. New names: `hit_corrected_count` (step), `particle_count_by_case`,
  `StepTry::solid_reactions`.

### Task 14: The diagnosis record, rungs 2 and 3

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** helpers join the step's `fluidity_add_test_library(fluidity_step_test_helpers ...)`; rung programs are `fluidity_add_test` programs linking it through `LINKS`, registered as the known-open proposal of Global Constraints says.

**Files:**
- Create: `native/fluidity/step/sand_diagnosis_test_helpers.h`, `sand_diagnosis_test_helpers.cc` (in the test
  library `fluidity_step_test_helpers`), `sand_diagnosis_test.cc`, `sand_record_test.cc`
- Modify: the step's test CMake lines (`sand_diagnosis_test_helpers.cc` added to the SOURCES of
  `fluidity_step_test_helpers`; the rung program through `fluidity_add_test(... LINKS fluidity_step_test_helpers
  fluidity_base_test_support)`; CTest tests registered one per rung, label `sand_diagnosis`, `TIMEOUT 3600`, never
  discovered under `cpu`: Global Constraints "Known-open tests")

**Interfaces:**
- Produces (test only, namespace `fluidity::step::testing`, plan decision):

  ```cpp
  struct SandSample {                  // One committed step.
    double time = 0, step_duration = 0;                       // s.
    double angle_degrees = 0, largest_height = 0, largest_radius = 0;   // Global Constraints "Rest angle"; m.
    double kinetic_energy = 0, springy_energy = 0, height_energy = 0;   // J. springy: rest_volume (mu sum e^2 +
                                                                        //   lambda/2 (sum e)^2); height: m g z.
    Vec3 momentum{};  double omitted_impulse = 0;             // N s. omitted: StepReport::internal_force.
    double sideways_contact_impulse = 0, normal_contact_impulse = 0;   // N s, from StepTry::solid_reactions.
    std::uint64_t hit_corrected_count = 0;
    std::array<std::uint64_t, materials::kLimitCaseCount> particle_count_by_case{};
  };
  struct SandRecord { std::vector<SandSample> samples; std::uint64_t rejected_try_count = 0; double asked_step = 0; };
  // Advances with TryStepWithStages and a stage table equal to kStages except Stage::kMoveParticles, which calls
  // the real stage and then sums, per particle, m |dv - (dv.n) n| and m (dv.n) of its reaction (n = kUp).
  SandRecord ComputeSandRecord(TestScene& scene, double end_time);
  double ComputeRestAngle(std::span<const Vec3> positions);                 // degrees
  const SandSample& FindSampleAt(const SandRecord& record, double time);    // |sample.time - time| <= 1e-9 s
  double ComputeAngleRate(const SandRecord& record, double window_start, double window_end);  // degrees/s
  double ComputeContactImpulseRatio(const SandRecord& record, double window_start, double window_end);
      // sum of sideways over sum of normal contact impulse in the window
  void WriteSandRecord(const SandRecord& record, std::string_view name);
      // When FLUIDITY_SAND_RECORD_DIR is set: <dir>/<name>.csv, one row per sample, columns in field order, and
      // <dir>/<name>_positions_<t>.csv at 0.75, 1.5, 3, 6 s; otherwise nothing.
  ```

- [ ] **Step 1: Failing tests.** Plain (`sand_record_test.cc`):

  ```cpp
  TEST(ComputeRestAngle, TwoPoints_IsFortyFiveDegrees)   // (0.1, 0, 0) and (0, 0, 0.1): 45 within 1e-13
  TEST(ComputeAngleRate, LinearRecord_IsItsSlope)        // angles 30 - 2 t: -2 within 1e-12
  TEST(ComputeContactImpulseRatio, HandBuiltRecord)      // sums 3 and 12 in the window, others outside: 0.25
  TEST(ComputeSandRecord, Pile_EveryTryCommitsToSixSeconds)   // Review Focus 1: coarse and fine piles: no reject,
      // every particle z >= 0, all values finite, samples at 0.75, 1.5, 3, 6 s exist
  TEST(ComputeSandRecord, EveryCommittedStepHasTheAskedDuration)   // Review Focus 4: pile coarse to 1.5 s and with
      // half the step: every step_duration within 1e-9 relative of the asked one
  ```
  Rungs (`sand_diagnosis_test.cc`, plain, design 7):

  ```cpp
  TEST(SandDiagnosis, Rung2_FreeStretchedBlock_EnergyNeverGrows)   // free block: every sample's kinetic + springy
      // <= (1 + 1e-9) * start; precondition: every particle_count_by_case entry other than kInsideLimits is 0
  TEST(SandDiagnosis, Rung2_FreeStretchedBlock_MomentumWithinOmitted)   // |momentum - start| <= running sum of
      // omitted_impulse + 1e-12 * total mass * largest speed
  TEST(SandDiagnosis, Rung3_ColumnInFreeFall_StateUnchanged)   // falling column: every springy_deformation equals I
      // and every permanent_volume_change equals 0 within 1e-12 at every step; kShapeReduced count 0
  ```
  **Plan decision (design contradiction):** with cohesion 0 a state at rest has `over = 0`, which design 5.2 step 1
  calls `kPulledApart`; rung 3 therefore asserts the state and no `kShapeReduced`, and allows `kPulledApart` with a
  zero excess. Say so in the test comment and the hand-over.
- [ ] **Step 2: Run** (fail), **implement**, **run.** If rung 2 or rung 3 fails, it is a port defect, not the creep:
  use `superpowers:systematic-debugging`; if not found, stop and report the numbers to the owner. Never relax a
  bound or commit either as known-open.
- [ ] **Step 3: Gate; commit.** Subject: `Record sand scenes step by step and check rungs 2 and 3`.

### Task 15: Rungs 4 and 7, first measurement decides

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/step/sand_diagnosis_test.cc`, its CTest lines, `docs/known-open-physics.md` (only if a
  rung fails)

- [ ] **Step 1: Tests** (design 7, "unknown" rungs; each for coarse and fine cells, suffix `_CoarseCells` /
  `_FineCells`):

  ```cpp
  TEST(SandDiagnosis, Rung4_LowSlab_StaysPut_CoarseCells)   // slab, 3 s: the largest z of particles with
      // max(|x|,|y|) <= 0.1 changes by less than 1 % of its 0.5 s value between 0.5 s and 3 s; kinetic energy
      // below 1e-4 J at every sample from 0.5 s on
  TEST(SandDiagnosis, Rung7_GridShiftedSideways_SameAngle_CoarseCells)   // grid_shift (w/2, w/2, 0): angle at
      // 1.5 s within 2 degrees of the unshifted pile's
  TEST(SandDiagnosis, Rung7_GridShiftedUpward_SameAngle_CoarseCells)     // grid_shift (0, 0, w/2)
  ```
  **Plan decision:** rung 4's "moves less than 1 % in 3 s" is measured from 0.5 s, after the slab has settled
  onto the floor under its own weight (about 18 % springy shortening under 300 kg/m3 at these stiffnesses, which
  the design's wording would otherwise count as creep). Say so in the hand-over.
- [ ] **Step 2: Run** each with `FLUIDITY_SAND_RECORD_DIR` set; keep the CSVs in the scratchpad.
- [ ] **Step 3: Register.** A rung that passes stays plain. A rung that fails becomes known-open (Global
  Constraints) with its measured numbers in its row. Never change its bound.
- [ ] **Step 4: Gate; commit.** Subject: `Measure the sand slab and the grid shift` (body: each rung's result and
  whether it is plain or known-open, with numbers).

### Task 16: Rungs 5, 6 and 8, committed as known-open

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/step/sand_diagnosis_test.cc`, its CTest lines, `docs/known-open-physics.md`

- [ ] **Step 1: Tests** (design 7; coarse and fine each):

  ```cpp
  TEST(SandDiagnosis, Rung5_Pile_StopsCreeping_CoarseCells)   // pile to 6 s: |angle(6) - angle(1.5)| < 1 degree
      // and kinetic energy at 6 s below 1e-3 J. RecordProperty: angles at 0.75, 1.5, 3, 6 s, the four window rates,
      // kinetic energies
  TEST(SandDiagnosis, Rung6_HalfStep_SameAngle_CoarseCells)   // pile with half the step to 1.5 s: angle at 1.5 s
      // within 2 degrees of the full-step pile's
  TEST(SandDiagnosis, Rung8_HalfStep_SameImpulseRatio_CoarseCells)   // floor kDrivenAtRest, to 1.5 s at the step
      // and at half the step: ComputeContactImpulseRatio over [1.0, 1.5] s differ by less than 10 % of the
      // full-step value. RecordProperty: both ratios, and the full-step angle at 1.5 s next to rung 5's
  ```
- [ ] **Step 2: Run** with `FLUIDITY_SAND_RECORD_DIR` set. Expected (design 7): rung 5, 6 and 8 fail. `main`
  measured 32.83 degrees at 0.75 s and 29.11 at 1.5 s, +7.09 and +4.90 degrees for half the step, impulse ratios
  0.148, 0.243, 0.382, 0.538 over three halvings; this branch's numbers differ (weights, contact, support rule
  changed) and are the ones recorded.
- [ ] **Step 3: Register** each failing test as known-open with its numbers. **A rung that passes is not forced to
  fail:** it stays plain, and the commit body says which rung passed on first measurement.
- [ ] **Step 4: Gate; commit.** Subject: `Commit the sand pile rungs that are still open`. Body: the measured table.

### Task 17: The record summary, the known limits of sand and the creep figure

**Kind:** mechanical

Load `drawing-with-cetz` and `visualizing-data`.

**Files:**
- Create: `tools/summarise_sand_records.py`, `tools/summarise_sand_records_test.py`,
  `docs/method-paper/data/sand_creep_rung5_coarse.csv`, `sand_creep_rung6_coarse.csv`,
  `docs/method-paper/figures/sand_creep.typ`
- Modify: the paper file of Task 1 (`<sec-sand-limits>`)

**Interfaces:**
- Produces `uv run python tools/summarise_sand_records.py <dir>`: reads the CSVs of Task 14's format and prints the
  rung table (rung, width, angle at 0.75/1.5/3/6 s, window rates, kinetic energy at 6 s, half-step gap, impulse
  ratios, pass or fail against the design's bounds) as a Markdown table. Part 2 pastes it into its records.
  `<fig-sand-creep>` (design 8): angle and kinetic energy against time for rungs 5 and 6, coarse cells, from the two
  CSVs (thinned to one row per 0.01 s).

- [ ] **Step 1: Failing test:** `test_summary_of_two_hand_written_records` (a 4-row CSV per rung; exact table text).
- [ ] **Step 2: Implement; run.** Write `<sec-sand-limits>`: the eight rungs as a table with their bounds and the
  numbers of Tasks 15 and 16, what the ladder rules out so far, and the statement that sand is reported as
  experimental while any rung is open (design decision 1). Draw the figure; compile; look.
- [ ] **Step 3: Gate; commit.** Subject: `Summarise sand records and state the known limits of sand`.

### Task 18: I/O: the sand preset, the scene file and the reference scene

**Kind:** integration

**Files:**
- Modify: `native/fluidity/io/material_presets.h/.cc`, `schemas/simulate/v2/scene_file.fbs` (`SceneMaterial`
  fields `friction_angle`, `cohesion`; kind `sand`), `scene_materials.cc`, `scene_materials_test.cc`,
  `start_particles.cc`, `start_particles_test.cc`, `continue_run_test.cc`
- Create: `native/fluidity/io/testdata/scenes/sand_pile.json`

**Interfaces:**
- Produces design 4 (I/O): `MaterialPreset::kSand` (appended after `kAir`, so stored values never shift;
  `kMaterialPresets` grows to 7; name `"sand"`): `friction_angle` 0.5235987755982989, `cohesion` 0,
  `stiffness_against_squeeze` 700 Pa, `stiffness_against_shear` 450 Pa, rest density 300 kg/m3, friction 0.7. The
  derivation gives sand `at_solid_faces = kParticlesOnly` and its own material set. Start regions: "each fill object
  of snow, sand or a springy solid gets its own material" (I/O 5.1 step 4).
- `sand_pile.json` (design decision 4): domain `[-0.6, 0.6]^2 x [-0.125, 0.6]`, all closed; smallest cell 0.0625,
  one level; a fixed flat solid floor at `z = 0`, normal up, `friction_scale` 1; one material, preset `sand`,
  `own_motion_kept` 0; one box start region centre `(0, 0, 0.275)`, half size `(0.075, 0.075, 0.175)` (the pile's
  column, seeded by I/O's rule, 8 particles per smallest cell); frames 0 to 144 at 24 per second (6 s); points of
  the sand material; a saved state every 24 frames.

- [ ] **Step 1: Failing tests** (label `cpu`): `FindMaterialPreset, EveryName_FoundAndUnknownEmpty` (extended);
  `ReadSceneMaterials, SandPresetOnly_FilledFromPreset` (six `FilledDefault` rows, source `kPreset`);
  `ReadSceneMaterials, SandAngleNinetyDegrees_IsModuleFailedFromMaterials` (path `materials[0]`, module code
  `bad_material`); `ReadSceneMaterials, Sand_AtSolidFacesIsParticlesOnly`; `SampleStartRegion,
  SandRegionsGetTheirOwnMaterial`; `BuildRunScene, SandReferenceScene_Reads` (preset values, floor, a non-zero start
  particle count, every start particle inside the box); `ContinueRun, Sand_FourWaysSameFiles` (as
  `Snow_FourWaysSameFiles`, 20 frames of `sand_pile.json`: both state arrays survive a saved state).
- [ ] **Step 2: Run** (fail), **implement**, **run**. **Step 3:** gate. **Commit.** Subject:
  `Read sand from the scene file and add its reference scene`. New names: `MaterialPreset::kSand`.

### Task 19: Render output: sand is drawn as points

**Kind:** mechanical

**Files:**
- Modify: `native/fluidity/render_output/check_render_settings.cc`, `render_settings_test.cc`, the file deciding the
  default point products (`git grep -n "MaterialKind::kSnow" native/fluidity/render_output native/fluidity/io`),
  its test

**Interfaces:**
- Produces design 4 (render output 5.8): sand's row "points by default, no surface, no foam source, no volume" (the
  row snow has); a `kSand` material is in the default point products. No sand surface (A9).

- [ ] **Step 1: Failing tests:** `CheckRenderSettings, SandGroup_IsBadInput` (a surface group with the sand
  material: `kBadInput`, id the material id); `BuildRunScene, SandMaterial_InDefaultPointOutputs` (`sand_pile.json`
  with no `outputs.points`: the default lists the sand material).
- [ ] **Step 2: Run** (fail), **implement** (the switch case of Task 5 is already there: add the paper row of
  render output 5.8 and these tests), **run**. **Step 3:** gate. **Commit.** Subject: `Draw sand as points only`.

### Task 20: The sand reference scene end to end through the program

**Kind:** integration

**Files:**
- Modify: `tests/cli/test_processes.py` (or the file holding `test_small_reference_scenes`)

- [ ] **Step 1: Test** (pytest, run by the CTest `cli_end_to_end`; `TIMEOUT 3600`):
  `test_sand_reference_scene_runs_six_seconds`: `fluidity check` on `sand_pile.json` exits 0 and its findings hold
  exactly one `sand_settling_creep` (level experimental); `fluidity run` ends `done` with 145 frames listed; the
  last saved state read with `reference/io/read_array_file.py` gives every particle `z >= 0`; the test prints the
  rest angle, the largest radius and the kinetic energy at 6 s (recorded in the commit body, not bounded: the
  bound is rung 5's).
- [ ] **Step 2: Run.** **Step 3:** gate. **Commit.** Subject: `Run the sand reference scene end to end`.

### Task 21: The Blender add-on: sand substance, preset and reference scene

**Amended 2026-11-14 (A27):** the add-on's `REPAIR_TABLE` regains the `set_viscosity_to_zero` entry the main
sequence removed (Task 12 adds the step's `kSetViscosityToZero` back for `kViscousLiquidWithSand`).

**Kind:** integration

**Files:**
- Modify: `blender/fluidity/core/roles.py` (`ROLE_TABLE`), `core/scene_fields.py` (`SCENE_FIELDS`),
  `core/presets.py` (`SUBSTANCE_PRESETS`, `REFERENCE_SCENES` and its builder), `core/scene_file.py`,
  `core/quick_check.py` (only if it lists kinds itself), `logic/findings.py` (`REPAIR_TABLE`, A27),
  `properties.py` (`FluiditySubstance`), the panels that
  show a substance's fields, the add-on's pytest files and its headless test that runs each reference scene preset
  (`git grep -ln "snow example" blender tests tools`)

**Interfaces:**
- Produces design 4 (add-on): `SAND` in `FluiditySubstance.kind`; properties `friction_angle` (rad) and
  `cohesion`, each with a `SceneField` (`materials[].friction_angle` "rad", `materials[].cohesion` "no unit",
  layer `PHYSICAL_DETAIL`, changes `SIMULATION`); `"sand"` in the `substance_kinds` of the `START_MATERIAL` row
  only; `regions_form_one_body` applies to sand as to snow; the sand row of `SUBSTANCE_PRESETS` (the I/O preset's
  values, Task 18); the sand row of `REFERENCE_SCENES` (`scene_id` `"sand_pile"`, the scene of `sand_pile.json`).
  (A27) `REPAIR_TABLE` entry `set_viscosity_to_zero` -> `fluidity.apply_field_value` on the property of the
  finding's `scene_file_path`, value 0 (as add-on plan D9 had before A27).

- [ ] **Step 1: Failing tests.** Plain pytest: `test_sand_preset_equals_io_preset` (as the snow row's test does);
  `test_quick_check_accepts_sand_start_material`; `test_quick_check_refuses_sand_on_a_source` (the "wrong-kind
  substance" finding); `test_scene_fields_cover_the_scene_file` (passes with the two new paths);
  `test_role_table_is_the_only_source` (passes); `test_repair_table_covers_suggested_changes` (A27: 12 step
  names, `set_viscosity_to_zero` among them); `test_set_viscosity_to_zero_repair_sets_zero` (A27: a
  `viscous_liquid_with_sand` finding with `scene_file_path` `materials[0].ingredients[0].viscosity` gives a repair
  that sets that property to 0). Headless, Blender 5.3 and 5.2.3: the sand reference scene preset is
  created, read, quick check clean, `fluidity check` exit 0 with no `unknown_scene_field` and one
  `sand_settling_creep` finding shown as experimental; bake frames 0 to 6 on the CPU; `apply_link_plan` links the
  sand points through the native point reader; point count equals the `frame_done` file entries.
- [ ] **Step 2: Run** (fail), **implement**, **run** both Blender builds. **Step 3:** gate. **Commit.** Subject:
  `Let artists fill a shape with sand in Blender`.

### Task 22: GPU scene runs for sand

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** each `gpu` test writes its final positions, velocities and states with `WriteResultDigest`.

**Files:**
- Modify: `native/fluidity/step/step_scene_gpu_test.cc`, `step_scene_no_gpu_test.cc`

- [ ] **Step 1: Tests.** `gpu` (design 7): `SandDiagnosis, OnGpuRung2_MatchesCpu`, `OnGpuRung3_MatchesCpu`,
  `OnGpuRung5CoarseAtThreeQuarters_MatchesCpu` (positions, velocities and state arrays within `1e-9` relative of
  the CPU at 0.75 s); `SandDiagnosis, OnGpuPileTwice_SameBytes` (100 steps); each passes its final arrays to
  `WriteResultDigest`. `no_gpu`: `CheckSceneCanRun` on the
  pile with `RunOn::kGpu` gives the CPU answer.
- [ ] **Step 2:** gate. **Commit.** Subject: `List the sand GPU scene runs`. Body: `CUDA: compiled, not run`.

At this point sand runs end to end, the ladder is committed with its open rungs, and the capability check reports
sand as experimental with reason `kSandSettlingCreep`.

---

# Part 2: Investigate the settling creep

Each task tests one hypothesis from the design's candidates (5.7) and the report's ladder (9.4, 9.5). Rules for every
task of Part 2:

- **Experiments are never committed.** An experiment's code change is made in the working tree, measured, and
  thrown away (`git restore`, `git clean` of new files); its diff goes into the decision record as a fenced
  `diff` block (or, above 80 lines, its summary and the path of the file it touched). Only the record, its thinned
  data and, when the hypothesis is confirmed, the fix below are committed.
- **A ladder run** is `FLUIDITY_SAND_RECORD_DIR=<scratch>/<name> ctest --test-dir out/build/cpu -L sand_diagnosis`
  then `uv run python tools/summarise_sand_records.py <scratch>/<name>`. Its measured quantities: `A5` = |angle(6)
  - angle(1.5)| per width (rung 5's creep), `K6` = kinetic energy at 6 s, `G6` = rung 6's half-step gap at 1.5 s,
  `R8` = rung 8's relative ratio difference, `O` = sum of `omitted_impulse` over the sum of `m g dt` on
  [0.75, 6] s, `P` = mean `permanent_volume_change` at 6 s.
- **Verdicts.** *Confirmed:* the changed run makes the rungs the hypothesis predicts pass at both widths, and every
  other test of the branch still passes. *Partial:* each of `A5`, `G6`, `R8` it targets falls to at most half its
  baseline, without a rung passing. *Rejected:* otherwise. A partial or rejected hypothesis leaves the code
  unchanged.
- **No tuning.** A material number, a friction, the support share, the sound share, `own_motion` and the scene
  setup may be *varied in an experiment* to measure a dependence; none may be *changed as a fix*, and no fix adds
  damping, a decay, a blend, a clamp or a floor.
- **Who may commit a fix.** A confirmed fix inside a method this module owns (the sand response, the hit split, the
  sand face rule, the sand case of `ApplyMaterialResponse`) is committed by the task, with an analytic test of the
  change, the rung's `WILL_FAIL` and `known_open` removed, its `docs/known-open-physics.md` row removed and
  `<sec-sand-limits>` updated, all in one commit. A confirmed fix in another module's method (transfer, contact, the
  support rule, the step) is written into the record as a proposal and the task **stops for the owner's yes** before
  any code is committed.
- **Order.** Task 23 fixes the order of Tasks 24 to 32 from the first rung that fails. A task whose rungs all pass
  already (an earlier fix closed them) writes a three-line record saying so and runs no experiment.

### Task 23: Read the ladder

**Kind:** numerics

**Files:**
- Create: `docs/decisions/sand-01-reading-the-ladder.md`, `docs/decisions/data/sand-01/`

- [ ] **Step 1: Ladder run** on the tip (the baseline every later record compares with). Keep the full CSVs in the
  scratchpad; commit the thinned ones.
- [ ] **Step 2: Write the record.** *Problem:* design 5.7. *Hypothesis:* "the first failing rung names the
  ingredient that causes the creep". *Experiment:* the ladder run. *Result:* the summary table; the first failing
  rung; `O` and `P`; the share of particles in each `LimitCase` over [1.0, 1.5] s; `hit_corrected_count` per step
  over [1.0, 1.5] s. *Decision:* the order of Tasks 24 to 32. Rule: if rung 4 fails, the floor is the first suspect
  (order 25, 26, 27, 24, 29, 28, 30, 31, 32); if rung 4 passes and rung 5 fails, the pile's geometry with time is
  (order 24, 29, 28, 25, 26, 27, 30, 31, 32); Task 30 runs only if a rung 7 test is known-open.
- [ ] **Step 3: Commit.** Subject: `Record where the sand ladder first fails`.

### Task 24: The support rule omits part of the push at the pile's edge

**Kind:** numerics

**Files:**
- Create: `docs/decisions/sand-02-support-rule.md`, `docs/decisions/data/sand-02/`

- [ ] **Hypothesis** (report 9.4): faces at the pile's surface and next to the floor have small weight sums; the
  support rule (materials 5.8, `min_support_share = 1e-4`) omits part of their push, so the pile holds less than
  its friction allows and slumps at a rate that depends on how many such faces each step meets.
- [ ] **Experiment:** ladder rungs 5, 6 and 8 at both widths with `MaterialSettings::min_support_share` set to 0,
  `1e-5`, `1e-4` (baseline) and `1e-3` (four runs, the setting passed through the test scenes; nothing else
  changes). Measure `A5`, `G6`, `R8`, `O`, and `weak_face_count` summed over [1.0, 1.5] s.
- [ ] **Confirms if** `O >= 1e-3` at the baseline and `A5` changes monotonically with the share by more than 1
  degree between `1e-5` and `1e-3`. **Rejects if** `O < 1e-4` or `A5` moves by less than 0.25 degree across the
  four runs. A run that fails to commit (share 0 can be unstable) is recorded as such, not retried with changes.
- [ ] **If confirmed:** the push lost at weak faces is part of the effective material. The support rule is
  materials' method: write the four options of report 9.4 into the record (better sampling of the support; a
  derived low-mass treatment; a limited force with a stated budget; refusing scenes whose faces are too weakly
  supported) with the measured numbers, and stop for the owner's choice. No code change.
- [ ] **Commit** the record. Subject: `Record whether the support rule causes the sand creep`.

### Task 25: Floor friction acts on particles but not on faces

**Kind:** numerics

**Files:**
- Create: `docs/decisions/sand-03-floor-friction-at-faces.md`, `docs/decisions/data/sand-03/`
- Modify (only if confirmed): `native/fluidity/materials/material_table.cc` or wherever the sand default
  `at_solid_faces` is set (materials and `native/fluidity/io/scene_materials.cc`), `solids/collide_solids_test.cc`,
  `sand_diagnosis_test.cc` CTest lines, `docs/known-open-physics.md`, the paper

- [ ] **Hypothesis** (design 5.7 "floor friction at particles versus faces"; design decision 2): with
  `kParticlesOnly`, the floor's friction changes only the particles' velocity during their move; the next transfer
  rebuilds their velocity from grid faces that never felt the floor, so part of each step's frictional stop is
  undone, and how much depends on the step (rung 8's ratio grows as the step shrinks).
- [ ] **Experiment:** rungs 4, 5, 6 and 8 with the sand set's `at_solid_faces` set to `kNoMotionIntoSolid` (snow's
  rule) instead of `kParticlesOnly`. Measure `A5`, `G6`, `R8`, rung 4, and the ratio itself at both steps.
- [ ] **Confirms if** rungs 5, 6 and 8 pass at both widths with `kNoMotionIntoSolid` and rung 4 still passes.
  **Rejects if** `A5`, `G6` and `R8` are no better than half the baseline, or rung 4 starts failing (a floor that
  removes the approach speed gives contact no normal speed for friction: `main`'s reason for the rule).
- [ ] **If confirmed** (sand-owned, design decision 2: "the change is made in this module's commit that fixes the
  rung"): sand's default becomes `kNoMotionIntoSolid` in materials and in I/O's derivation;
  `CollideSolids, Sand_FacesUntouchedAtSolid` becomes `Sand_InwardFaceSpeedRemovedAtSolid` with snow's expected
  values; the passing rungs lose `WILL_FAIL`; one commit. Subject: `Stop sand faces at solids, which closes rungs
  <list>`.
- [ ] **Commit** (otherwise) the record. Subject: `Record whether the sand face rule causes the creep`.

### Task 26: The hit split freezes the floor layer's deformation

**Kind:** numerics

**Files:**
- Create: `docs/decisions/sand-04-hit-split.md`, `docs/decisions/data/sand-04/`
- Modify (only if confirmed): `native/fluidity/materials/apply_material_response_cpu.cc`, `_steps.h` (amended
  2026-11-14, Foundation as built: the shared per-particle function both halves call), `_gpu.cu`, their tests,
  the paper's `<proc-hit-split>`, CTest lines, `docs/known-open-physics.md`

- [ ] **Hypothesis:** particles resting on the floor hit it every step (gravity gives them `g dt` of approach
  speed), so their deformation follows `L_after_hit` for almost the whole step; a sticking particle gets
  `L_after_hit = 0` while the layer above follows `L`, so the bottom layer's remembered shape drifts away from its
  neighbours' by an amount per step, and the cone then turns that mismatch into permanent shape change.
- [ ] **Experiment:** two working-tree variants of the sand case of `ApplyMaterialResponse`: (a) no split
  (`L_after_hit = L`); (b) split with the normal change removed but no friction derivative (`L_after_hit = L1` of
  design 5.4 step 2). Ladder rungs 4, 5, 6, 8. Measure `A5`, `G6`, `R8`, the mean `kShapeReduced` share of the
  bottom layer (particles with `z < w`) over [1.0, 1.5] s, and `hit_corrected_count` per step.
- [ ] **Confirms if** a variant makes rungs 5 and 6 pass at both widths with rung 4 passing. **Rejects if** neither
  variant reaches "partial".
- [ ] **If confirmed** (sand-owned): the sand case uses the variant that passed (when both pass, (b), which keeps
  the particle out of the solid); `ComputeVelocityChangeAfterHit` stays only if still called, else it and its tests
  are deleted and the commit lists the removed names; design 5.4's change is recorded for the hand-over. Subject:
  `Follow the grid velocity in the sand hit split, which closes rungs <list>`.
- [ ] **Commit** (otherwise) the record. Subject: `Record whether the sand hit split causes the creep`.

### Task 27: Clearing the fitted change at every hit loses the floor layer's shear

**Kind:** numerics

**Files:**
- Create: `docs/decisions/sand-05-fitted-change-at-hit.md`, `docs/decisions/data/sand-05/`
- Modify (only if confirmed): `native/fluidity/materials/apply_material_response_cpu.cc`, `_steps.h` (amended
  2026-11-14, Foundation as built: the shared per-particle function both halves call), `_gpu.cu`, their tests,
  CTest lines, `docs/known-open-physics.md`, the paper

- [ ] **Hypothesis:** a sand particle that hits loses `kCarriesFittedChange` (`main` `:570`, design 5.4). Resting
  particles hit every step, so the bottom layer never carries its local velocity change into the next transfer;
  the loss happens once per step, so a shorter step loses more per second (rung 6) and changes the sideways
  impulse that reaches the floor (rung 8).
- [ ] **Experiment:** working-tree change: the sand case keeps `kCarriesFittedChange` after a hit (snow unchanged).
  Ladder rungs 4, 5, 6, 8. Measure `A5`, `G6`, `R8`, `K6`, and the bottom layer's mean sideways speed over
  [1.0, 1.5] s.
- [ ] **Confirms if** rungs 6 and 8 pass at both widths and rung 4 still passes. **Rejects if** `G6` and `R8` are
  not at most half their baselines.
- [ ] **If confirmed** (sand-owned): sand keeps its fitted change at a hit;
  `ApplyMaterialResponse, SandHit_SplitsStep` asserts the flag kept; `hit_corrected_count` still counts the split.
  Subject: `Keep the fitted change of sand at a hit, which closes rungs <list>`.
- [ ] **Commit** (otherwise) the record. Subject: `Record whether clearing the fitted change causes the creep`.

### Task 28: Replacing particle velocities by grid velocities each step

**Kind:** numerics

**Files:**
- Create: `docs/decisions/sand-06-grid-velocity-replacement.md`, `docs/decisions/data/sand-06/`

- [ ] **Hypothesis** (`main`'s timestep note, `docs/native/m5-qualification.md` "Sand timestep sensitivity"; report
  9.5): with `own_motion.kept = 0` each step replaces a particle's velocity by the grid's, which removes the part
  the grid cannot hold once per step; the removed energy per second grows as the step shrinks, so the pile's
  settled shape depends on the step (rung 6) and keeps moving while elastic energy drains (rung 5).
- [ ] **Experiment:** rungs 5, 6, 8 with the pile material's `own_motion.kept` at 0 (baseline), 0.5 and 0.9
  (`in_travel` stays 0). Measure `A5`, `G6`, `R8`, `K6`, and the springy energy's rate over the last window.
- [ ] **Confirms if** `G6` at 0.9 is below 2 degrees at both widths while it is not at 0, and `A5` falls with the
  kept share. **Rejects if** `G6` changes by less than a quarter of its baseline across the three runs.
- [ ] **If confirmed:** the dependence is the transfer's (transfer and materials own `own_motion`), and the
  share is an artist's setting: changing it would be tuning. Record the proposal (an energy-consistent velocity
  transfer for materials with internal force, transfer design's question) and stop for the owner. No code change.
- [ ] **Commit** the record. Subject: `Record whether grid velocity replacement causes the creep`.

### Task 29: The first-order deformation update drifts with the step

**Kind:** numerics

**Files:**
- Create: `docs/decisions/sand-07-first-order-deformation-update.md`, `docs/decisions/data/sand-07/`
- Modify (only if confirmed): `native/fluidity/materials/material_response.h` (the sand case only),
  `material_response_test.cc`, the paper, CTest lines, `docs/known-open-physics.md`

- [ ] **Hypothesis:** `F <- (I + dt L) F` changes the volume by `det(I + dt L)` instead of `exp(dt tr L)`; the error
  per step is of order `(dt |L|)^2` and its sum over a run is of order `dt`. Through the cone, whose allowed shape
  grows with the squeeze, a volume drift changes the slope the grains hold, so the angle depends on the step and
  drifts with time.
- [ ] **Experiment:** working-tree change: the sand case of `ComputeMaterialResponse` uses `exp(duration L)` (and
  `exp(duration_after_hit L_after_hit)`), computed by scaling and squaring with a degree-12 Taylor sum to `1e-16`;
  snow and springy solids unchanged. Rungs 4, 5, 6, 8. Measure `A5`, `G6`, `P`, and `P`'s change between 1.5 and
  6 s.
- [ ] **Confirms if** rungs 5 and 6 pass at both widths. **Partial if** `P`'s drift halves without a rung passing.
- [ ] **If confirmed** (sand-owned: the sand case only): a private `ComputeMatrixExponential` in
  `sand_material.h` with tests `ComputeMatrixExponential, RotationRate_IsRotation` (`L` antisymmetric for 0.3 rad
  about `{0,0,1}` gives `math::Rotation({0,0,0.3})` within 1e-15) and `TraceGivesVolume` (`det == exp(tr)` within
  1e-15 relative); `SandTwoParts_EqualsTwoProducts` updated to the exponential. Subject: `Update sand deformation
  with the exact exponential, which closes rungs <list>`.
- [ ] **Commit** (otherwise) the record. Subject: `Record whether the first-order deformation update causes the creep`.

### Task 30: The pile depends on where the floor cuts the cells

**Kind:** numerics

Run only if Task 23 recorded a known-open rung 7 test; otherwise write the three-line record.

**Files:**
- Create: `docs/decisions/sand-08-grid-position.md`, `docs/decisions/data/sand-08/`

- [ ] **Hypothesis** (report 9.4, grid translation): the faces next to the floor have weight sums that depend on
  the floor's position inside a cell, so the support rule omits a different share of the push for each grid shift.
- [ ] **Experiment:** the coarse pile to 1.5 s with `grid_shift` `(0, 0, s w)` for `s` in `{0, 0.25, 0.5, 0.75}`
  and `(s w, s w, 0)` for the same `s` (eight runs). Measure the angle at 1.5 s, `O`, and `weak_face_count` in the
  cells within one width of the floor.
- [ ] **Confirms if** the angle's spread over the eight runs exceeds 2 degrees and its correlation with `O` is above
  0.9 in magnitude: then it is Task 24's cause seen through the grid, and the record points there. **Rejects**
  otherwise.
- [ ] **If confirmed:** as Task 24 (another module's method): proposal and stop. No code change.
- [ ] **Commit** the record. Subject: `Record whether the grid position causes the sand creep`.

### Task 31: The rest angle measures rim particles, not the heap

**Kind:** numerics

**Files:**
- Create: `docs/decisions/sand-09-angle-measure.md`, `docs/decisions/data/sand-09/`

- [ ] **Hypothesis** (report 9.5: the friction angle need not equal a geometric estimator in a coarse pile): the
  angle `atan2(largest height, largest radius)` is set by one or two particles; rim particles that slide outward
  make the angle fall while the heap's slope holds.
- [ ] **Experiment** (no solver change; the positions CSVs of Task 23's baseline at 0.75, 1.5, 3 and 6 s, both
  widths, rungs 5 and 6): (a) the leave-one-out angle (smallest and largest angle with one particle removed);
  (b) the fitted slope of `main`'s `FitSandSurface` (`native/tests/test_m5_dry_scenes_3d.cc:40-93`, rotated: radius
  `hypot(x, y)`, height `z`; six bins over 0.15 to 0.85 of the largest radius in four quadrants, the highest
  particle per bin, least squares per quadrant, at least four bins): its smallest and largest quadrant angle.
- [ ] **Confirms if** the fitted slope changes by less than 1 degree between 1.5 and 6 s and its half-step gap is
  below 2 degrees, while the rest angle fails both. **Rejects** if the fitted slope creeps as the rest angle does.
- [ ] **If confirmed:** the solver is unchanged and the rungs' assertions are not changed: the angle measure is the
  owner's criterion (design 5.7). Record both measures side by side and stop for the owner's decision on whether
  rungs 5 to 8 should use the fitted slope.
- [ ] **Commit** the record. Subject: `Record whether the rest angle measure explains the sand creep`.

### Task 32: Too few particles per cell

**Kind:** numerics

**Files:**
- Create: `docs/decisions/sand-10-sampling-density.md`, `docs/decisions/data/sand-10/`

- [ ] **Hypothesis:** the pile has 63 particles at 0.05 m spacing in cells of 0.075 m, about 0.3 particle per cell;
  most faces at the heap's surface are weakly supported and the heap's shape is carried by a few particles, so the
  creep is a sampling effect.
- [ ] **Experiment:** the same column sampled at 0.025 m (504 particles, `main`'s `particles_per_axis_multiplier`
  2: positions `((i - 3.5) 0.025, (j - 3.5) 0.025, 0.10 + (k + 0.5) 0.025)`, `i, j` in `0..7`, `k` in `0..13`,
  mass `300 * 0.025^3`), working-tree change to the test scene only. Rungs 5, 6, 8 at both widths. Measure `A5`,
  `G6`, `R8`, `O`.
- [ ] **Confirms if** the dense pile passes rungs 5 and 6 at both widths while the sparse one fails. **Rejects** if
  the dense pile's `A5` and `G6` are not at most half the sparse ones.
- [ ] **If confirmed:** no solver change, and the rungs' scene is not changed: the pile is the design's (decision
  4). Record the result and stop for the owner's decision on whether the ladder moves to the dense pile or to the
  I/O-sampled reference scene (`sand_pile.json`).
- [ ] **Commit** the record. Subject: `Record whether sampling density explains the sand creep`.

---

# Part 3: Hand-over

### Task 33: Lift the known limit when all eight rungs pass

**Kind:** integration

Run only if the ladder run on the tip shows all eight rungs passing with no `known_open` label left in
`sand_diagnosis_test.cc`'s CTest lines; otherwise skip this task and say so in Task 34.

**Files:**
- Modify: `native/fluidity/step/capability.h`, `capability.cc`, `capability_test.cc`, `step_message.cc`,
  `native/fluidity/io/stable_names.cc`, `stable_names_test.cc`, `tests/cli/` (the sand test), the add-on's headless
  sand test, the paper's `<sec-sand-limits>`

- [ ] **Step 1: Tests:** `CheckSceneCanRun, Sand_CanRunWithNoFinding` replaces `Sand_IsExperimentalWithSettlingCreep`;
  the CLI and Blender sand tests expect no finding.
- [ ] **Step 2: Remove** `CapabilityReason::kSandSettlingCreep`, its message, stable name and finding (a value
  nothing reports is deleted, as the viscosity plan did for its lifted exclusions); counts lower by one.
  `git grep -n "SandSettlingCreep\|sand_settling_creep" -- native tests blender docs` finds nothing.
  `<sec-sand-limits>` states that the eight rungs pass, with the final numbers.
- [ ] **Step 3: Gate; commit.** Subject: `Report sand as usable now that its ladder passes`. Body: removed names.

### Task 34: Comment pass and hand-over

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** the hand-over records the owner's ruling on how rung and known-open tests are kept out of the
`cpu` runs, and the per-call solids copy of Task 7 if it was needed.

**Files:**
- Modify: any file this plan touched, `docs/gpu-runs.md`
- Create: `docs/decisions/sand-11-status.md`

- [ ] **Step 1: Comment pass.** Reread every file this plan created or changed from top to bottom as a C++
  programmer who has never read a fluid paper: every comment true of the final code; why, not what; a worked
  example with numbers for `ComputeSandResponse` (the 30-degree case: forces (-5,-1,-1) become (-4.2,-1.4,-1.4))
  and `ComputeVelocityChangeAfterHit` (the stick case gives zero); words in the glossary; `Uses` and `Used by`
  lists; `// Step N:` matching `<proc-sand-response>` and `<proc-hit-split>`.
- [ ] **Step 2: Fold each change into its owning commit** (`git commit --fixup`, `git rebase --autosquash`); format
  and rules on each rewritten commit; the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3 names, 4 additions to other modules, 5 method, 6 CPU and
  GPU, 7 tests, 9 vocabulary, 10 not ported) and record what was delivered differently: the pile material's
  `own_motion` `{0, 0}`; the driven-at-rest floor of rung 8; `StepReport::particle_count_by_case` and
  `StepTry::solid_reactions`; rung 3's allowance for `kPulledApart` with zero excess (design 5.2 step 1 against
  design 7 rung 3); rung 4 measured from 0.5 s; the per-call solids copy on the GPU if Task 7 needed it;
  `ViscosityError::kSandNotSupported` appended at the end; every fix committed in Part 2 and every proposal waiting
  for the owner.
- [ ] **Step 4: `docs/gpu-runs.md`.** Row for sand: `compiled, not run`, listing every `gpu` test of Tasks 7, 13
  and 22.
- [ ] **Step 5: Status record** `docs/decisions/sand-11-status.md`: a table of the eight rungs (rung, what it adds,
  bound, coarse result, fine result, pass or known-open, the record that last measured it); the list of Part 2
  records with their verdicts; the proposals waiting for the owner. **Commit.** Subject: `State which sand rungs
  pass`.
- [ ] **Step 6: Hand over** in one message: `git log --oneline` of the module's commits; the gate output; the new
  public names with reasons; the rung table, stated plainly (which of the eight pass, which are known-open, with
  numbers); the statement that no device code has run. **If any rung is still known-open, stop here for the owner**:
  the module is done only when all eight pass or the owner decides otherwise (design 5.7, owner A9 "important but
  not mandatory"); until then sand stays reported as experimental with `kSandSettlingCreep`.
