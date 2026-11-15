# Rebuild Materials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the materials module on the `rebuild` branch: the scene's ingredient and material table, the one
route by which liquid, gas, snow and springy solids answer the motion (`ComputeInternalForcePerArea`,
`ComputeMaterialResponse`), the internal force on the grid with the support rule and its budget, the sound
speed step limit, the travel share, per-particle friction, per-cell squeeze values and the work identity as a
test oracle, on the CPU and (compiled, not run) on the GPU, with the extension points the sand module fills.

**Architecture:** Folder `native/fluidity/materials/`, namespace `fluidity::materials`. Eight shared formula
headers (`material.h`, `stretch_directions.h`, `stretch_response.h`, `snow_material.h`,
`springy_solid_material.h`, `material_response.h`, `own_motion_rule.h`, `support_rule.h`) hold every formula;
seven batch operations (`ComputeInternalForces`, `ApplyInternalForce`, `ComputeOwnMotionShares`,
`ComputeParticleFriction`, `ApplyMaterialResponse` with `CheckMaterialState`, `ComputeCellSqueezeValues`,
`ComputeWorkRates`) each have one public header and private `_cpu.cc` / `_gpu.cu` files that loop over the same
formulas. Inputs from later modules (the step, solids, gas, I/O) are built by hand in this module's tests.

**Tech Stack:** as Foundation (C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2, uv, clang-format
and clang-tidy 23.1, the versions `pyproject.toml` pins, Docker via colima, Typst). NumPy is already in the `dev` group (transfer plan).

**Spec:** `docs/superpowers/specs/modules/07-materials-design.md` ("the design"; in the `main` checkout, never
committed) and `docs/design/programme.md` on `rebuild` (sections 6, 7, 8, 8a). Names shared with other modules:
`docs/superpowers/specs/modules/shared-interface.md`; decisions of earlier plans: `plan-decisions.md` (wins over
the design). Owner decisions: `owner-answers.md` (A2, A4, A8, A9, A10, A11, A13 apply here).

## Global Constraints

Foundation's list, unchanged:

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Old code is read with `git show main:<path>` from the worktree. Nothing is checked out wholesale.
- Commit rule (spec section 6): one thing per commit; strict CPU build and all tests pass; CUDA target compiles
  in the container; clang-format 23.1 clean; clang-tidy 23.1 clean on host sources; every touched file passes the `AGENTS.md` checklist; new public
  names listed in the commit message with a one-line reason; tests arrive with the code. A commit must pass every
  part of the gate that exists at that commit.
- History on `rebuild` may be rewritten. A defect found later is fixed in the commit that introduced it.
- Commit messages carry no AI trailer (no `Co-Authored-By`, no `Assisted-by`, no session link); the owner is the
  author. After committing, write the commit's note with `uv run python tools/add_session_note.py --session
  <session link> --models <implementer's model id>,claude-fable-5-2`
  (a git note in `refs/notes/claude-session`; owner answer A55, 2026-11-15).
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
  `specs/modules/plan-decisions.md` "Foundation after the final review"). Two halves: the public function lives in
  `<name>_cpu.cc` (checks, `RunOn` dispatch, the CPU loop, and the GPU half under `#if defined(FLUIDITY_HAS_GPU)`);
  the GPU half `<name>_gpu.cu` is declared in the private `<name>_gpu.h`, returns `RunOnCallStatus`, which the CPU
  half maps with `ToBufferError` and then into `MaterialsFailure` through one mapping function, and launches,
  waits and holds temporary memory only through `base/gpu_launch.cuh` (`ComputeLaunchSize`, `ApplyGpuFunction`,
  `CheckGpuWork`, `GpuArray`). Every GPU function is a thin loop over `FLUIDITY_CPU_AND_GPU` per-item or
  per-thread functions in a shared header (`<name>_steps.h` unless one of the eight formula headers holds them),
  which the CPU loop calls too; `cpu` tests run those functions over every item, or every thread block and thread
  of `ComputeLaunchSize(count)` in launch order and in reverse (phase by phase where the GPU function waits),
  against the CPU loop. Working buffers (`<Name>Buffers`, `Build<Name>Buffers`, `Compute<Name>BuffersSize`): one
  form per function, which takes its buffers (A31); callers pass memory owned by `MaterialBuffers` (this module calls no `base/` sum or grouping on
  buffers, so it holds no `SumBuffers` or `OrderBuffers`). Errors: a failed check writes nothing; after
  `kRunOnCallFailed` the outputs are unspecified and the run stops; `kOutOfMemory` comes only from building
  functions and from the one-call GPU memory of `ComputeWorkRates` and `CheckMaterialState` (design decision 10). Exactness: discrete outputs equal on CPU and GPU; floating-point within the module's tolerance
  unless the function's header promises the same bits unless NaN. Tests link `fluidity_base_test_support`
  (`ToBits`, `BuildBufferHolding`, `ReadAll`, `CheckWithinTolerance`, `CheckSameBitsOrBothNaN`) and the module's
  helper library made with `fluidity_add_test_library`, named in `LINKS`; `gpu` tests call `WriteResultDigest` on
  their main outputs. Labels: `cpu` everywhere (shared-header tests too: the container's cpu step runs them; the
  contact plan's `fluidity_add_shared_test` is retired), `no_gpu` only for behaviour on a machine without a GPU,
  `gpu` only on hardware. Every target is created through the helpers of `cmake/fluidity_targets.cmake` and every
  source file is listed in a CMake file.
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the CPU build in the debug and the release preset, and
  CUDA compiled in the container (its `cuda` step fails for a tracked `.cc` or `.cu` file that no build compiles),
  so a task that adds a source adds it to a CMake file in the same commit.

Materials' own:

- **Design section 3 is the contract.** Spell every name exactly as there; a task says "design 3" and lists only
  names. `HasInternalForce` fails the `verb` rule: add it to `tools/check_rules_allowed_names.txt` with
  `# design name (materials design 3); constexpr predicate on the kind`.
- **No sand, no gas header.** Nothing here declares, tests or mentions `kSand`, `friction_angle`, `cohesion`,
  `kShapeReduced`, `kPulledApart` or `sand_material.h`. The gas ingredient's rest density
  (`gas::ComputeAmbientDensity` of the scene's gas law, gas design) is an input value set by I/O:
  `BuildMaterialTable` checks it only as `> 0`, and no file of this module includes a `gas/` header (the gas
  module comes later in the build order and includes this one). This matches design 5.1; the `Ingredient`
  comment may name the gas function in words only.
- **Extension points (design 5.9).** `MaterialKind` and `LimitCase` have explicit values
  (`kLiquid = 0` .. `kSpringySolid = 3`; `kInsideLimits = 0`, `kClamped = 1`), each `static_assert`ed. Every
  `switch` on `MaterialKind` or `LimitCase` lists every value and has no `default:`, so `-Wswitch` (warnings are
  errors) points the sand module at each place to extend. Arrays by case are sized `kLimitCaseCount`. Each GPU
  batch operation instantiates its per-particle kernel once per `MaterialKind` through a
  `template <MaterialKind kind>` and a host-side switch; no thread branches on the kind.
- **Internal force is already the Kirchhoff form.** The formulas of design 5.3 give the force per area that
  pairs with `rest_volume` (the design's "includes the factor `volume_ratio`"). Nothing multiplies it by
  `volume_ratio` again, anywhere.
- **Constants (design 5.2, 3, 6):** in `stretch_directions.h` (plan decision: public, so tests use them):
  `kMinDeterminantShare = 1.5e-14`, `kMaxRotationSweeps = 16`, `kRotationTolerance = 2e-15`,
  `kDirectionTolerance = 1e-12`. In `support_rule.h`: `kMinSupportShare = 1e-4`. `MaterialSettings` defaults:
  `min_support_share = kMinSupportShare`, `sound_step_share = 0.25`, `state_tolerance = 1e-12`; an operation
  taking settings returns `kBadInput` unless `min_support_share >= 0`, `0 < sound_step_share <= 1`,
  `state_tolerance > 0`, all finite. No `0.99` and no force floor anywhere.
- **Order of every sum (design 5.8, 6).** Budget and work sums: per cell block, `math::TreeSum` over the block's
  240 face places in ascending `FacePlaceIndex` (places of block `b` are `[240 b, 240 b + 240)`; a place that is
  not a piece contributes `0.0`, so the tree shape is fixed), or over the block's particles in grouped order;
  then `math::TreeSum` over block partials in ascending block order. Counts are integer sums. The largest sound
  bound per material is exact in any order (plan decision for the GPU: `atomicMax` on the `uint64` bit pattern
  of the non-negative bound; a non-finite bound is `kNotFinite` before it is compared). **Amended 2026-11-14 (Foundation as built):** two
  header promises of same bits on CPU and GPU unless NaN, with their reasons: the per-cell-block budget of
  `ApplyInternalForce` (Task 18: both processors add the same per-face terms in `math::TreeSum`'s tree) and
  `ComputeParticleFriction` (Task 19: products and sums in a fixed order, no fused multiply-add under the flags).
- **Failure contract.** Inputs are read-only. The failure reported is the one of the lowest particle id
  (particle operations) or lowest `CellPlaceIndex` / `FacePlaceIndex` (cell and face operations), after the
  call-level checks in this order: settings, span sizes (`kBadInput`), grid versions and `step_try`
  (`kWrongGridVersion`; a `step_try` of 0 counts as another try), carried arrays (`kWrongFields`), buffer sizes
  (`kBuffersTooSmall`). A particle view (`ParticleFields`, `ParticleFieldsToChange`, both carry `RunOn run_on`
  per `plan-decisions.md`) or an array whose `run_on` differs from `MaterialBuffers::run_on()` is a broken
  precondition (`Requires:` same processor, asserted in debug builds; amended 2026-11-14, A30), not `kBadInput`.
  On the GPU the lowest id comes from a 64-bit integer `atomicMin`, never from timing.
- **`kStepTooLong` (plan decision).** `particle_id = 0`, `material_index` = the material giving the smallest
  limit, `magnitude` = `step_limit`, the smallest `sound_step_share * smallest_cell_width[m] / largest bound of m`
  over the range. A particle failure (collapsed, unstable, not finite) is reported before `kStepTooLong`.
  `step_duration * bound > sound_step_share * width` is strict: equality is accepted.
- **Buffers without allocation (plan decision, transfer's precedent).** `BuildMaterialBuffers` takes a last
  parameter `std::uint32_t max_cell_blocks` (the per-block budget buffer and the table copy live in
  `MaterialBuffers`). **Amended 2026-11-14 (A31):** `ApplyInternalForce` has one form, with `MaterialBuffers&
  buffers` last, and `ComputeCellSqueezeValues` one, with `const MaterialBuffers& buffers` last; they allocate
  nothing; tests build a `MaterialBuffers` locally. `ComputeWorkRates` (test oracle) and `CheckMaterialState`
  (scene load) keep one form that holds one call's GPU memory (`GpuArray`): neither runs per step.
- **Memory estimate entries (plan decision, `plan-decisions.md`: `sizeof`).** All `where = kRunOn`, names start
  `materials: `. Per-particle rows are `kGivenCount` from `max_particle_counts` (the counts the buffers are
  built at): internal force `sizeof(Mat3)` (72) over materials with internal force; friction 8, own motion
  shares `sizeof(OwnMotionShares)` (16), sound bound 8 and error code 1 over all materials; all `kOneStepTry`,
  `copies = 1`. Per cell block `sizeof(InternalForceTotals)` (24), `kWholeRun`. Table copy:
  `materials x sizeof(Material) + ingredients x sizeof(Ingredient)`, `kWholeRun`. No per-face-place row: the GPU
  reduces each block's 240 faces in shared memory (the design's 24 B per face place is dropped).
- **Tolerances (design 6, 7).** CPU tests as each task states. GPU against CPU: state arrays
  `1e-13 + 1e-12 |value|`; `internal_force_per_area` `1e-12 (mu + lambda) + 1e-11 |value|`; sound bound `1e-11`
  relative; work rates and budgets `1e-12` relative (per-block budgets: same bits, by the promise above);
  friction same bits (promise above); `LimitCase` exact except within `1e-12` of a limit; counts, kinds, flags and
  the reported failure exact. Repeat-run tests: identical bytes, and every `gpu` test writes its main outputs with
  `WriteResultDigest`.
- **GPU dispatch shape.** **Amended 2026-11-14 (Foundation as built):** Foundation's two halves (contact's `move_particles.h`,
  `move_particles_cpu.cc`, `move_particles_gpu.h` follow it too): the public function lives in `_cpu.cc`;
  `RunOn::kGpu` calls a private function declared in `<operation>_gpu.h`, defined in `_gpu.cu`, returning
  `RunOnCallStatus`, named for what it does (a `_gpu` file may say `Gpu`); without `FLUIDITY_HAS_GPU` the call site
  returns `kRunOnCallFailed`. Launch sizes from `ComputeLaunchSize` (`kThreadsPerBlock` = 256 threads per thread
  block), except the one-thread-block-per-cell-block reductions of Tasks 18 and 26. The lowest failing id, the
  largest bounds and the case counts are written by integer atomics into the result record of `MaterialBuffers`
  and read back by the CPU half with `ReadBuffer`.
- **Ranges.** An operation touches only the particles, faces and cells of its `CellBlockRange`; particle arrays
  of this module are indexed by the whole-material particle index (particles' views point at particle 0 of the
  material), as transfer reads them.

## Before you start

On `rebuild` already: Foundation (all of `shared-interface.md` section (a), including `math::Expm1`,
`math::TreeSum`, `math::Rotation`, `Buffer`, `memory_count.h` with `MemoryWhere`; `RunOn` alone in the shared
header `base/run_on.h`, so shared headers here may name it; `CheckCanRunOn` in the CPU-only
`base/check_can_run_on.h`; **Amended 2026-11-14 (Foundation as built):** also `run_on_call.h`, `gpu_launch.cuh`, `ToBufferError`, the one-form
functions of `ordered_sums.h` and `stable_order.h` (each takes its buffers), `test_support.h` in the library `fluidity_base_test_support`, and
`fluidity_add_test_library`; read `AGENTS.md` "Structure" before Task 1), the gate, the method paper with
`figure_style.typ` and `pseudocode.typ`; contact (`ParticleContact`, `ContactKind`, `MoveParticlesInputs::friction`), grid (`StaggeredGrid`, `GridTables`, `CellBlockRange` in `grid_ids.h`,
`CellValues`, `FaceValues`, `CornerValues`, `ApplyGridVersion`, `FindCornerPlaces`, `ComputeCornerPosition`,
`ComputeCellWidth`, `ComputeCellCenter`, `kFacePlacesPerBlock`), particles (`ParticleFields`,
`ParticleFieldsToChange`, `BuildFieldsToRead`, `BuildFieldsToChange`, `Field`, `ToFieldList`,
`MaterialParticlesSettings`, `ExtraFields`, `CountRule`, `kMaxIngredientCount`, `kCarriesFittedChange`,
`AddParticles`, `GroupParticlesByBlock`, `CommitTry`), transfer (`GridFromParticles`, `FaceValueKind`,
`ParticlesToGrid` with `internal_force_per_area`, `SaveGridBeforeForces`, `SaveGridAfterPressure`,
`GridToParticles`, `GridAtParticle`, `MixParticleVelocity`, `CellsFromParticles`, `BuildCellsFromParticles`, and
the test library `fluidity_transfer_test_scenes` in `namespace fluidity::transfer::test`), pressure
(`PressureLaw`, `PressureLawKind`, `PressureLawState`, `ComputePressureLawState`), viscosity.

Proof: from the worktree `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, run
`uv run python tools/check_commit.py`. Expected: seven `PASS` lines (format, rules, cpu debug, cpu release, tidy,
paper, cuda). If any fails, stop: materials does not
start on a red branch.

## Review Focus

1. **The force per area multiplied by `volume_ratio` a second time.** The snow and springy formulas already pair
   with rest volume. Pinned in Task 15: `ComputeInternalForces, SpringyStretched_ForceIsNotTimesVolumeRatio`.
2. **An exactly flat deformation with no zero row.** Rotations can turn it into a tiny positive stretch; the
   determinant test must run first. Pinned in Task 7: `ComputeStretchDirections, DependentRows_IsRejected`
   (`main`'s rows `{-2,1,-5},{2,2,-3},{2,5,-11}`).
3. **A step exactly at the sound limit.** Equality must be accepted and the next double rejected with the
   limit in `magnitude`. Pinned in Task 15: `ComputeInternalForces, StepExactlyAtLimit_IsAccepted`.
4. **A face with force and no particle mass, or a support share of 0.** No `0 / 0`; the whole impulse is
   omitted, never turned into velocity. Pinned in Task 12: `ComputeFacePush, ZeroWeightSum_PushesNothing` and
   in Task 17: `ApplyInternalForce, ForceOnFaceWithoutValue_IsAllOmitted`.
5. **The travel rule at a size step.** The particle's own (small) cell width and its own cell's corners must be
   used. Pinned in Task 20: `ComputeOwnMotionShares, ParticleInSmallCellAtSizeStep_UsesItsOwnCell`.

---

### Task 1: Method paper section, text, equations and pseudocode

**Kind:** mechanical

Load the `typesetting-with-typst` and `typesetting-documents` skills first.

**Files:**
- Modify: `docs/method-paper/fluidity-method.typ`, `docs/glossary.md`

**Interfaces:**
- Produces (design 8): `<sec-materials>`, `<sec-material-table>`, `<sec-stretch-directions>`, `<sec-snow>`,
  `<sec-springy-solid>`, `<sec-support-rule>`, `<sec-work-identity>`, `<sec-sound-limit>`; equations
  `<eq:stretch-update>`, `<eq:snow-limit>`, `<eq:springy-force>`, `<eq:face-push>`, `<eq:work-identity>`,
  `<eq:sound-bound>`; pseudocode `<proc-stretch-directions>`, `<proc-snow-response>`,
  `<proc-apply-material-response>`, `<proc-apply-internal-force>`. Every `Read:` line of later tasks cites one.

- [ ] **Step 1: Write `= Materials <sec-materials>`** after the viscosity section, one subsection per label, from
  design 5.1 to 5.8 in the decided vocabulary (design 9: no `trial`, `return mapping`, `yield`, `plastic`,
  `elastic`, `stress`, `Kirchhoff` except once in parentheses where the plain word is introduced). Content:
  - `<sec-material-table>`: the table of 5.1 (four kinds); what an ingredient is; the gas rules; export names.
  - `<sec-stretch-directions>`: the four steps of 5.2 with the four constants and why the determinant test runs
    before any rotation.
  - `<sec-snow>`: `<eq:stretch-update>` (`F <- (I + dt L) F`, `L` on the left) and `<eq:snow-limit>` (clamp,
    permanent volume, `exp(-hardening Expm1(permanent))`, force per area); the world form in `main`'s
    `Recompose` order.
  - `<sec-springy-solid>`: `<eq:springy-force>` `mu (F F^T - I) + lambda ln(J) I`; a paragraph on why the force
    per area pairs with rest volume (the `volume_ratio` factor is inside it).
  - `<sec-support-rule>`: why a rule is needed (weight like distance squared, change like distance), the rule
    stated in bold as in 5.8, `<eq:face-push>`, the equal-mass reduction to `main`'s floor, the budget and the
    order of its sums.
  - `<sec-work-identity>`: `<eq:work-identity>` with the three-line proof that it holds where force and exact
    change use the same faces, and why it does not next to a solid.
  - `<sec-sound-limit>`: `<eq:sound-bound>` with the full stiffness blocks of snow (stretch rows and pair rows)
    and springy solids, the pair test, and the stated limit (principal planes only).
  - Notation rows: `F` springy_deformation, `L` exact_velocity_change, `s` springy_stretch, `mu`
    stiffness_against_shear, `lambda` stiffness_against_squeeze, `V0` rest_volume, `J` volume_ratio.
  - Citations [19] Stomakhin 2013, [3] Hu 2018, [18] Stomakhin 2014, [4] Fei 2021, [1] Jiang 2015 from
    `references.typ`; "citation needed" for the neo-Hookean law, the one-sided rotation search and the row-sum
    bound (design 8: not written from memory).
- [ ] **Step 2: Pseudocode** with `procedure(...)`. Step numbers are the contract for `// Step N:` comments:
  - `<proc-stretch-directions>`: 1 reject non-finite, scale by largest entry; 2 determinant against
    `kMinDeterminantShare`; 3 rotation sweeps; 4 stretches, directions, perpendicular and right-handed check.
  - `<proc-snow-response>`: 1 clamp; 2 permanent volume; 3 stiffen; 4 force per area.
  - `<proc-apply-material-response>`: 1 liquid and gas volume; 2 `F <- (I + dt L) F`; 3 snow (directions,
    response, rebuild, volume ratio); 4 springy (volume ratio = determinant); 5 snow particle that hit loses
    `kCarriesFittedChange`; 6 write every array of the particle.
  - `<proc-apply-internal-force>`: 1 check inputs; 2 per face of the block, `ComputeFacePush` or all omitted;
    3 tree sums of the block's three budget terms; 4 write `totals_of_block`.
- [ ] **Step 3: Glossary.** Rows (`| Say | Plain meaning | Never say |`): ingredient, material kind, export name,
  stretch directions, springy stretch, permanent volume change, internal force per area, allowed region, limit
  case, support rule, omitted impulse, sound speed bound, travel share. "Never say" from design 9 (trial
  stretches, return mapping, yield surface, plastic, elastic (for springy), Kirchhoff stress, Lame parameter,
  force floor, principal frame, constitutive model, substance, constituent, phase id).
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`.
  Expected: exit 0, no warning. Read the new pages. Run the gate.
- [ ] **Step 5: Commit.** Subject: `Describe materials, their response and the support rule`.

### Task 2: Method paper figures for materials

**Kind:** mechanical

Load `drawing-with-cetz`; use `figure_style.typ`.

**Files:**
- Create: `docs/method-paper/figures/stretch_directions.typ`, `snow_limit.typ`, `support_rule.typ`,
  `force_measures.typ`
- Modify: `docs/method-paper/fluidity-method.typ`

**Interfaces:**
- Produces `<fig-stretch-directions>`, `<fig-snow-limit>`, `<fig-support-rule>`, `<fig-force-measures>`.

- [ ] **Step 1: Draw** (design 8): a unit cube and the slanted box it becomes, the three directions before and
  after and the three stretches; the allowed box of snow stretches (two axes shown) with a point outside, its
  clamped image and the volume that became permanent; one particle near the edge of its reach, weight and weight
  change against distance (plotted from the quadratic curve), their ratio growing, and the scaled push of the
  rule with `min_support_share = 1e-4` marked; a block at rest and deformed, rest volume and current volume, the
  force per area paired with rest volume. Each referenced from its subsection; each caption states the one thing
  to see.
- [ ] **Step 2: Compile and look.** Fix overlaps, clipped labels, inconsistent sizes. Run the gate.
- [ ] **Step 3: Commit.** Subject: `Draw the materials figures`.

### Task 3: Materials folder, errors and their names

**Kind:** mechanical

**Files:**
- Create in `native/fluidity/materials/`: `materials_error.h`, `materials_message.h`, `materials_message.cc`,
  `materials_message_test.cc`, `CMakeLists.txt` (or the per-module list the earlier modules use)
- Modify: the CMake file that lists module folders

**Interfaces:**
- Produces design 3: `MaterialsError` (all sixteen values), `MaterialsFailure`, `ToMessage`, `ToName`.

- [ ] **Step 1: Failing tests** (label `cpu`):

  ```cpp
  TEST(ToName, EveryValue)          // design 7: each of the 16 values -> lower snake case without k, all distinct:
                                    // kBadExportName -> "bad_export_name", kStepTooLong -> "step_too_long",
                                    // kRunOnCallFailed -> "run_on_call_failed"
  TEST(ToMessage, EveryValue_IsNonEmptyAndDistinct)
  TEST(MaterialsFailure, Default_IsNoneWithZeros)
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** `materials_error.h` is shared (C++20 subset, no `<string>`); `materials_message.h`
  includes `base/cpu_only.h` first. Messages say what the user can do (`kStepTooLong`: "The step is longer than
  the sound speed of a springy material allows. The step will be shortened to the limit in the report.").
  `Read:` cites `<sec-materials>`.
- [ ] **Step 4: Run** build, tests, format, rules, gate.
- [ ] **Step 5: Commit.** Subject: `Add materials errors and their stable names`. New names: the list above.

### Task 4: Material kinds, materials and the stretch response type

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** the shared-header test is `fluidity_add_test(material_test SOURCES material_test.cc LABEL cpu)`.

**Files:**
- Create: `material.h`, `stretch_response.h`, `material_test.cc` (`fluidity_add_test(... LABEL cpu)`)

**Interfaces:**
- Consumes: `pressure::PressureLaw`.
- Produces design 3: `MaterialKind`, `AtSolidFaces`, `Ingredient`, `MaterialResponse`, `OwnMotionSettings`,
  `Material`, `HasInternalForce`, `LimitCase`, `kLimitCaseCount`, `StretchResponse`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(MaterialKind, StoredValues_AreFixed)        // static_assert and EXPECT: kLiquid 0, kGas 1, kSnow 2,
                                                   // kSpringySolid 3 (sand appends 4, design 5.9)
  TEST(LimitCase, CountMatchesLastValue)           // kLimitCaseCount == int(kClamped) + 1 == 2
  TEST(HasInternalForce, OnlySnowAndSpringySolid)  // constexpr; every kind listed
  TEST(Material, Defaults_AreDesignDefaults)       // ingredient_count 1, kParticlesOnly, kind kLiquid, zeros
  ```
- [ ] **Step 2: Run** (compile error), **implement** (field comments as design 3, units first), **run**.
  `HasInternalForce` is a `switch` with every value and no `default:`. Add it to the allowed-names file (Global
  Constraints). `Read:` `<sec-material-table>`, `<sec-snow>`.
- [ ] **Step 3: Gate; commit.** Subject: `Add the material kinds and what a material is made of`.

### Task 5: The material table: checks and derived spans

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the test library is made with `fluidity_add_test_library` and linked through `LINKS`.

**Files:**
- Create: `material_table.h`, `material_table.cc`, `material_table_test.cc`, `materials_test_tables.h`,
  `materials_test_tables.cc` (test library `fluidity_materials_test_scenes`, never linked by `fluidity`:
  `fluidity_add_test_library(fluidity_materials_test_scenes SOURCES materials_test_tables.cc LINKS
  fluidity_base_test_support)`; every later materials test names it in `LINKS`)
- Modify: the materials `CMakeLists.txt`

**Interfaces:**
- Produces design 3: `MaterialPair`, `MaterialTableInputs`, `PlatesInScene`, `MaterialTable` with every
  accessor, `BuildMaterialTable`, `FindSurfaceTensionBetween`. Plan decisions:
  - `MaterialTable` is move-only; built only by `BuildMaterialTable` (private constructor, friend); it owns
    copies of every input (`std::vector`) and the derived spans, built once.
  - Check order (design 5.1, extended): (0) `ingredient_export_names.size() != ingredients.size()` is
    `kBadInput`; (1) materials in index order, each field in declaration order: `material_id` nonzero and not
    repeated (a repeat is reported at the later index), the response by kind (5.1), `ingredient_count` in
    `1..kMaxIngredientCount` and `> 1` only for `kLiquid`, `first_ingredient` equal to the sum of the earlier
    counts (ingredient ranges tile the ingredients in order), `0 <= in_travel <= kept <= 1`, gas
    `own_motion` 0 and 0, enum values in range; (2) ingredients in index order: id nonzero and not repeated,
    `viscosity`, `surface_tension`, `friction >= 0` and finite, `rest_density > 0` and finite, law kind in
    range; then the rules of the owning material's kind: internal force needs `kCannotBeSqueezed`; gas needs
    viscosity 0, friction 0, surface tension 0, `kCannotBeSqueezed`; snow and springy need viscosity 0; every
    non-liquid needs surface tension 0 (design 3 comments). An ingredient failure sets `ingredient_index` and the
    owning `material_index`. (3) export names (Task 6); (4) pairs: indexes in range, `low < high`, sorted
    strictly by `(low, high)`, tension `>= 0` and finite, else `kBadMaterial` with `material_index = low`;
    (5) liquid ingredients with different `surface_tension` give `kSurfaceTensionsDiffer`, `ingredient_index` =
    the first that differs from the first liquid ingredient, `magnitude` = its value.
  - With no liquid ingredient, `surface_tension_max`, `surface_tension_with_air` and `lightest_density` are 0.
  - `FindSurfaceTensionBetween` accepts its two indexes in either order (binary search on the sorted pairs).
- Test tables (`namespace fluidity::materials::test`, used by every later task):
  `BuildAllKindsInputs()` returns owned storage and a `MaterialTableInputs` view of it: material 0 liquid with two
  ingredients (water: id 11, rest density 1000, viscosity 1e-3, tension 0.072, friction 0.2; syrup: id 12, 1400,
  10, 0.072, 0.4; both `kCannotBeSqueezed`; `own_motion {0.95, 0.5}`), material 1 gas (id 13, rest density 1.2,
  zeros), material 2 snow (id 14, rest density 400, friction 0.3; mu 6e4, lambda 4e4, shortening 0.025,
  lengthening 0.0075, hardening 10; `kNoMotionIntoSolid`), material 3 springy solid (id 15, rest density 1000,
  friction 0.5; mu 1e5, lambda 1e5; `kNoMotionIntoSolid`); material ids 1..4; export names `"Water"`,
  `"syrup"`, `"smoke"`, `"snow"`, `"jelly"`; no pairs. `BuildOneKindInputs(MaterialKind, MaterialResponse)` for
  a one-material table with the same ingredient numbers.

- [ ] **Step 1: Failing tests** (label `cpu`; the design's "Table tests" row):

  ```cpp
  TEST(BuildMaterialTable, AllKinds_SpansAsSpecified)          // first_ingredient_of_material {0,2,3,4};
      // ingredient_viscosity {1e-3,10,0,0,0}; friction {0.2,0.4,0,0.3,0.5}; rest density {1000,1400,1.2,400,1000};
      // surface_tension_max == surface_tension_with_air == 0.072; lightest_density 1000
  TEST(BuildMaterialTable, BadNumbers_AreBadMaterialInCheckOrder)  // each in turn on AllKinds: snow shear 0;
      // shortening 1; in_travel 0.96 > kept; water viscosity -1; snow rest density 0; springy law kSteadyStiffness:
      // kBadMaterial with the stated material_index / ingredient_index; two at once report the first in order
  TEST(BuildMaterialTable, RepeatedIds_AreBadMaterial)          // material id 3 twice -> index 2; ingredient id
                                                                // 12 twice -> ingredient_index 1
  TEST(BuildMaterialTable, NonContiguousIngredients_AreBadMaterial)   // snow first_ingredient 2 -> index 2
  TEST(BuildMaterialTable, TwoTensions_AreSurfaceTensionsDiffer)  // syrup 0.05 -> ingredient_index 1, magnitude 0.05
  TEST(BuildMaterialTable, NamesOfWrongSize_IsBadInput)
  TEST(BuildMaterialTable, UnsortedPairs_IsBadMaterial)         // pairs {(0,3),(0,2)} -> material_index 0
  TEST(FindSurfaceTensionBetween, EitherOrder_SameValue)        // pair (0,2) 0.03: (0,2) and (2,0) give 0.03;
                                                                // (1,3) gives 0
  TEST(BuildMaterialTable, NoLiquid_TensionsAndLightestAreZero)
  ```
- [ ] **Step 2: Run** (compile error), **implement**, **run**. `Read:` `<sec-material-table>`.
- [ ] **Step 3: Gate; commit.** Subject: `Build the ingredient and material table`. New names: the list above.

### Task 6: Export names, the gas rules and the particle settings

**Kind:** mechanical

**Files:**
- Modify: `material_table.h`, `material_table.cc`, `material_table_test.cc`

**Interfaces:**
- Produces design 3: `MaterialTable::ingredient_export_names()`, `gas_ingredient_index()`,
  `BuildParticlesSettings`. Plan decisions: normalising is ASCII only (`A`-`Z` lowered; every byte outside
  `[a-z0-9_]` becomes `_`, so one multi-byte character becomes several `_`); `BuildParticlesSettings` returns an
  empty vector when `max_particle_counts.size()` is not the material count (documented, tested).

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(BuildMaterialTable, ExportNames)   // design 7: "Red Syrup", "water-2" -> "red_syrup", "water_2";
                                          // then "A b" with "a_b" -> kBadExportName, ingredient_index 1;
                                          // then "" first -> kBadExportName, ingredient_index 0
  TEST(BuildMaterialTable, GasRules)      // design 7, AllKinds with the gas changed, each in turn: viscosity 1e-5;
      // friction 0.1; tension 0.01 (each kBadMaterial, ingredient_index 2, material_index 1); ingredient_count 2
      // (material_index 1); kept 0.1 (material_index 1); law kSteadyStiffness (ingredient_index 2). Then the
      // correct gas: gas_ingredient_index() == 2; without the gas material: empty
  TEST(BuildMaterialTable, TwoGasMaterials_IndexIsLowestMaterials)   // second gas appended: still 2
  TEST(BuildParticlesSettings, AllKinds_ExtraFieldsAndCountRuleByKind)   // design 5.1 table: liquid count 2, no
      // extras, kMayBeSplitAndMerged; gas none; snow springy + permanent, kKeepsItsCount; springy springy +
      // plate_side, kKeepsItsCount; material_id and max_particle_count copied
  TEST(BuildParticlesSettings, PlatesInScene_EveryMaterialHasPlateSide)
  TEST(BuildParticlesSettings, CountsOfWrongSize_IsEmpty)
  ```
- [ ] **Step 2: Run, implement, run.** Export-name checks run after every ingredient check (design 5.1 order).
- [ ] **Step 3: Gate; commit.** Subject: `Add export names, the gas rules and the particle settings`.

### Task 7: Stretch directions

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** label `cpu` through `fluidity_add_test` (no `fluidity_add_shared_test`).

**Files:**
- Create: `stretch_directions.h`, `stretch_directions_test.cc` (`fluidity_add_test(... LABEL cpu)`)

**Interfaces:**
- Consumes: `Mat3`, `math::Determinant`, `math::Rotation`, `MaterialsError`.
- Produces design 3: `StretchDirections`, `ComputeStretchDirections`; the four constants (Global Constraints).
  Ported from `main` `native/include/fluidity/base/solid_deformation_3d.h:34-149` (search) and `:151-191`
  (rebuild), unchanged maths: the six-term determinant against the sum of absolute terms (`:62-86`), one-sided
  Jacobi rotations on the column pairs (0,1), (0,2), (1,2), the same rotation applied to `directions_at_rest`.
  `main`'s `Mat3` is three row vectors; Foundation's is nine named doubles: port by name, not by layout.

- [ ] **Step 1: Failing tests** (label `cpu`; the container's cpu step runs them too):

  ```cpp
  TEST(ComputeStretchDirections, RotatedStretch_IsRecovered)  // design 7: F = Rotation({0.3,-0.2,0.5}) *
      // diag(0.5,1,2) * Transpose(Rotation({-0.4,0.1,0.25})); and diag(0.8,0.8,1.3) under the same rotations.
      // directions_now * diag(s) * Transpose(directions_at_rest) == F within 1e-13 of the largest entry;
      // stretches equal {0.5,1,2} as a set within 1e-13; both direction matrices have determinant 1 within 1e-13
  TEST(ComputeStretchDirections, Mirrored_IsRejected)         // diag(-1,1,1): kDeformationCollapsed; output bytes
                                                              // equal the sentinel written before the call
  TEST(ComputeStretchDirections, Flat_IsRejected)             // rows {1,0,0},{0,0,0},{0,0,1}: same
  TEST(ComputeStretchDirections, DependentRows_IsRejected)    // Review Focus 2: rows {-2,1,-5},{2,2,-3},{2,5,-11}
  TEST(ComputeStretchDirections, NotFinite_IsBadInput)        // one NaN entry; output untouched
  TEST(ComputeStretchDirections, Identity_IsUnitStretchesNoRotation)  // exact
  ```
- [ ] **Step 2: Run** (compile error), **implement** with `// Step N:` of `<proc-stretch-directions>`, **run**.
  Plain loops over the three pairs; no pointer-to-member loops.
- [ ] **Step 3: Gate; commit.** Subject: `Find the stretch directions of a deformation`.

### Task 8: Snow response

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** label `cpu` through `fluidity_add_test`.

**Files:**
- Create: `snow_material.h`, `snow_material_test.cc` (`fluidity_add_test(... LABEL cpu)`)

**Interfaces:**
- Consumes: `StretchResponse`, `MaterialResponse`, `math::Expm1`, `math::Log`, `math::Exp`.
- Produces design 3: `ComputeSnowResponse`. Ported from `main`
  `native/include/fluidity/base/m5_constitutive_3d.h:14-33, 159-211` (the snow part only), maths unchanged; the
  "representable" checks of `PublishPrincipalState` become one finiteness check (design 10).

- [ ] **Step 1: Failing tests** (design 7; `mu = 2`, `lambda = 3`, `max_springy_shortening = 0.025`,
  `max_springy_lengthening = 0.01`, `hardening = 5` unless stated):

  ```cpp
  TEST(ComputeSnowResponse, InsideLimits_IsUnchanged)   // {0.99,0.99,0.99}, permanent 0: same stretches,
      // kInsideLimits, permanent 0, force_i = 2*2*(0.99-1)*0.99 + 3*(J-1)*J with J = 0.99^3; within 1e-13
  TEST(ComputeSnowResponse, Clamp_KeepsTotalVolume)     // {0.9,1.0,1.05}: stretches {0.975,1.0,1.01}, kClamped;
      // product(springy) * exp(permanent) == 0.945 within 1e-13
  TEST(ComputeSnowResponse, PackedSnow_IsStiffer)       // {0.99,1,1.005}, permanent -0.4, hardening 4: force ==
      // exp(-4 * Expm1(-0.4)) * force at permanent 0, hardening 4; within 1e-13 relative
  TEST(ComputeSnowResponse, NonPositiveStretch_IsBadInputUntouched)   // {0,1,1}; also permanent 1000
                                                                      // (exp not finite)
  TEST(ComputeSnowResponse, ShorteningOne_IsBadMaterial)
  ```
- [ ] **Step 2: Run, implement** (`// Step N:` of `<proc-snow-response>`; the local log of a stretch is
  `log_stretch`), **run**. `Read:` `<sec-snow>`, `<eq:snow-limit>`.
- [ ] **Step 3: Gate; commit.** Subject: `Add the snow response inside the stretch directions`.

### Task 9: Springy solid force

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** label `cpu` through `fluidity_add_test`.

**Files:**
- Create: `springy_solid_material.h`, `springy_solid_material_test.cc` (`fluidity_add_test(... LABEL cpu)`)

**Interfaces:**
- Produces design 3: `ComputeSpringySolidForcePerArea`. Ported from `main` `native/src/transfers_3d.cc:108-135`;
  the interval determinant bound (`:263-289`) becomes `J > 0` and finite (design 5.3).

- [ ] **Step 1: Failing tests** (`mu = 1e5`, `lambda = 2e5`):

  ```cpp
  TEST(ComputeSpringySolidForcePerArea, Rotation_GivesZero)       // F = Rotation({0.3,-0.7,0.2}): every entry
                                                                   // within 1e-13 * mu
  TEST(ComputeSpringySolidForcePerArea, Stretched_IsClosedForm)   // diag(1.2,1,1): mu*diag(0.44,0,0) +
                                                                   // lambda*ln(1.2)*I within 1e-13 relative; symmetric
  TEST(ComputeSpringySolidForcePerArea, Mirrored_IsDeformationCollapsed)   // output untouched
  TEST(ComputeSpringySolidForcePerArea, ZeroShearStiffness_IsBadMaterial)
  ```
- [ ] **Step 2: Run, implement, run.** `Read:` `<sec-springy-solid>`, `<eq:springy-force>`.
- [ ] **Step 3: Gate; commit.** Subject: `Add the force of a springy solid`.

### Task 10: Deformation update and the sound speed bound

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** label `cpu` through `fluidity_add_test`.

**Files:**
- Create: `material_response.h` (this task: `MaterialState`, `ComputeDeformationAfter`,
  `ComputeVolumeRatioAfter`, `ComputeSoundSpeedBound`), `material_response_test.cc` (`fluidity_add_test(... LABEL
  cpu)`)

**Interfaces:**
- Produces design 3: `MaterialState`, `ComputeDeformationAfter`, `ComputeVolumeRatioAfter`,
  `ComputeSoundSpeedBound`. Ported from `main`: stretch update `native/src/dry_solid_step_3d.cc:24-35`; snow
  bound `native/include/fluidity/base/solid_acoustic_3d.h:97-158` (with `PublishAcousticBound` and
  `PositivePrincipalPlane`, `:8-33`); springy bound `native/src/transfers_3d.cc:291-327`. Liquid and gas give
  bound 0 and `kNone`; snow uses `springy_stretch` and `state.permanent_volume_change`; springy uses
  `state.springy_deformation` (the `springy_stretch` argument is not read).

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(ComputeDeformationAfter, RateIsOnTheLeft)   // design 7: L rows {0,1,0},{0,0,0},{0,0,0}; F rows
      // {1,0,0},{0,1,0},{1,0,1}; dt 0.1: result rows {1,0.1,0},{0,1,0},{1,0,1} within 1e-15, and its zy entry
      // differs from F * (I + dt L) (zy = 0.1) by 0.1
  TEST(ComputeVolumeRatioAfter, TraceTwoForATenth_IsExpPointTwo)   // 0.8 * exp(0.2) within 1e-15 relative
  TEST(ComputeSoundSpeedBound, AtRest_IsRowSum)    // design 7: snow mu 2 lambda 3 density 2 at rest:
      // sqrt(6.5) within 1e-13; springy same numbers at rest: sqrt(3.5); both >= sqrt((lambda + 2 mu) / density)
  TEST(ComputeSoundSpeedBound, PackedSnow_IsStiffenedRowSum)   // permanent ln(0.8), hardening 5: sqrt(6.5 e)
      // within 1e-13 (main m5_deformation_3d_cases.h case 26)
  TEST(ComputeSoundSpeedBound, SpringyHugeExpansion_IsMaterialUnstable)   // F = 2 I, mu 2, lambda 3
  TEST(ComputeSoundSpeedBound, ZeroDensity_IsBadInput)
  TEST(ComputeSoundSpeedBound, Liquid_IsZero)
  ```
- [ ] **Step 2: Run, implement, run.** `Read:` `<sec-sound-limit>`, `<eq:sound-bound>`, `<eq:stretch-update>`.
- [ ] **Step 3: Gate; commit.** Subject: `Add the deformation update and the sound speed bound`.

### Task 11: One route for every material

**Kind:** numerics

**Files:**
- Modify: `material_response.h`, `material_response_test.cc`

**Interfaces:**
- Produces design 3: `InternalForce`, `ComputeInternalForcePerArea`, `MotionInStep`, `MaterialUpdate`,
  `ComputeMaterialResponse`. Each switches once on `response.kind` (no `default:`). Snow world form:
  `directions_now * diag(force) * Transpose(directions_now)` and `directions_now * diag(springy_stretch) *
  Transpose(directions_at_rest)`, each triple product summed as `main`'s `Recompose` (first two terms, then the
  third). Liquid result `0` is `kDeformationCollapsed`, not finite is `kNotFinite` (plan decision).

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(ComputeMaterialResponse, Liquid_VolumeFollowsTrace)   // design 7: ratio 0.8, L = diag(1,0.5,0.5),
                                                             // 0.1 s -> 0.8 * exp(0.2) within 1e-15 relative
  TEST(ComputeMaterialResponse, Gas_StateUnchanged)          // design 7: L with every entry 5: state bytes
      // unchanged, volume_ratio exactly 1; ComputeInternalForcePerArea gives the zero matrix and bound 0 exactly
  TEST(ComputeMaterialResponse, SnowZeroMotion_KeepsAStateInsideLimits)   // F = R diag(0.99,1,1.005) R^T:
      // same F within 1e-13 of the largest entry, kInsideLimits, volume_ratio = product * exp(permanent)
  TEST(ComputeMaterialResponse, SpringyStretch_VolumeRatioIsDeterminant)  // exact against math::Determinant
  TEST(ComputeMaterialResponse, SnowFolded_IsDeformationCollapsedUntouched)   // L = -20 I, 0.1 s
  TEST(ComputeInternalForcePerArea, SnowRotated_IsWorldFormOfStretchForce)    // R diag(f) R^T within
                                                             // 1e-13 (mu + lambda); bound as Task 10
  TEST(ComputeInternalForcePerArea, Springy_EqualsSpringyFormulaExactly)
  TEST(ComputeInternalForcePerArea, ZeroRestDensity_IsBadInput)
  ```
- [ ] **Step 2: Run, implement** (`// Step N:` of `<proc-apply-material-response>`), **run**.
- [ ] **Step 3: Gate; commit.** Subject: `Answer the motion through one route for every material`. New names:
  the list above.

### Task 12: Travel share, mixed friction and the support rule

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** label `cpu` through `fluidity_add_test`.

**Files:**
- Create: `own_motion_rule.h`, `support_rule.h`, `own_motion_rule_test.cc`, `support_rule_test.cc`
  (`fluidity_add_test(... LABEL cpu)`)

**Interfaces:**
- Produces design 3: `OwnMotionShares`, `ComputeOwnMotionInTravel`, `ComputeMixedFriction`,
  `kMinSupportShare`, `FacePush`, `ComputeFacePush`. Travel rule ported from `main`
  `native/src/liquid_substep_3d_cpu.cc:308-313` without `0.99` (owner Q33). Plan decisions:
  `ComputeMixedFriction` with `ingredient_count == 1` accepts null fractions and returns `ingredient_friction[0]`;
  `ComputeFacePush` with `support == 0` returns `{0, 0, step_duration * force}` without dividing (mass may be 0
  there); `min_support_share == 0` gives `support = 1`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(ComputeOwnMotionInTravel, Rule_TruthTable)   // design 7. Liquid kept 0.95, in_travel 0.5, h = 0.1,
      // L = diag(0.1,0,0), d_surface -0.05, d_solid 0.2 -> {0.95, 0.5}. Each off in turn -> {0.95, 0}: trace 0;
      // d_surface 0.01; d_surface -0.11; d_solid 0.1 (not > h). Edges d_surface 0 and -0.1 -> 0.5. Snow, springy
      // with all three true -> {kept, 0}; gas -> {0, 0}. Exact
  TEST(ComputeMixedFriction, Mixture_IsFractionWeightedSum)   // {0.25,0.75}, {0.2,0.6} -> 0.5 within 1e-15
  TEST(ComputeMixedFriction, OneIngredientNullFractions_IsItsFriction)
  TEST(ComputeFacePush, Budget_AddsUp)              // design 7, share 1e-4, force 3, dt 0.01, mass 2 * w:
      // w 1, 1e-4, 1e-6, 0 -> factor (velocity_change / (dt force / mass)) 1, 1 exact, 0.01 within 1e-15, 0 exact;
      // applied + omitted == dt * force within 1e-15 relative; applied == mass * velocity_change
  TEST(ComputeFacePush, ZeroWeightSum_PushesNothing)   // Review Focus 4: mass 0, w 0: {0, 0, 0.03}, finite
  TEST(ComputeFacePush, ZeroShare_IsFullPush)          // share 0, w 1e-9: factor 1
  ```
- [ ] **Step 2: Run, implement, run.** `Read:` `<sec-support-rule>`, `<eq:face-push>`.
- [ ] **Step 3: Gate; commit.** Subject: `Add the travel rule, mixed friction and the support rule`.

### Task 13: Test scenes

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** added to the `fluidity_add_test_library` target with `LINKS fluidity_transfer_test_scenes`.

**Files:**
- Create: `materials_test_scenes.h`, `materials_test_scenes.cc`, `materials_test_scenes_test.cc` (added to
  `fluidity_materials_test_scenes`, which now adds `LINKS fluidity_transfer_test_scenes` in its
  `fluidity_add_test_library` call)

**Interfaces:**
- Consumes: transfer's `test::BuildUniformGrid`, `BuildSizeStepGrid`, `BuildSolidsAtFaces`, `PlateAtX`,
  `ObliqueFloor`, `BuildCloud`, `ReadAllBytes`; particles' `BuildParticles`, `BuildTry`, `AddParticles`,
  `GroupParticlesByBlock`.
- Produces, in `namespace fluidity::materials::test` (later tasks use these exact names):
  - `BuildBlockRowGrid(RunOn)`: one level, width 0.1, cells x `0..47`, y and z `0..3` (12 cell blocks in a row).
  - `MaterialParticleSpec { Vec3 position; Vec3 velocity; double mass; double rest_volume; std::uint32_t
    material_index; double volume_ratio = 1; Mat3 springy_deformation = kIdentity; double
    permanent_volume_change = 0; std::array<double, 2> fractions{1, 0}; }`.
  - `struct TestParticles { particles::Particles particles; particles::ParticlesTry particles_try; }` and
    `std::unique_ptr<TestParticles> BuildTestParticles(RunOn, const grid::StaggeredGrid&, const MaterialTable&,
    std::span<const MaterialParticleSpec>, PlatesInScene)`: settings from `BuildParticlesSettings` (limit 4096
    per material; `ParticlesSettings::max_cell_blocks` = the grid's block count, `plan-decisions.md`), adds per material in input order (ids `1, 2, ...` in that order), groups. Heap-held so the
    try's reference to the particles stays valid.
  - `BuildSpringyBlock(Vec3 low, Vec3 high, Mat3 deformation, std::uint32_t material_index, double rest_density)`:
    particles at the quarter points of every 0.1 m cell of the box (8 per cell), each position mapped to
    `center + deformation * (position - center)`; `springy_deformation = deformation`; `rest_volume = 0.05^3`;
    `mass = rest_density * rest_volume`; `volume_ratio = Determinant(deformation)`.
  - `BuildRandomSymmetric(std::uint64_t seed, double scale)`: entries `scale * (2u - 1)` with
    `u = (draw >> 11) * 0x1.0p-53` from `std::mt19937_64(seed)` (transfer's portable rule), then symmetrised.
  - `ComputeSpringyEnergy(const Mat3& deformation, const MaterialResponse&)`: J/m3,
    `mu/2 (tr(F^T F) - 3) - mu ln J + lambda/2 (ln J)^2`.

- [ ] **Step 1: Failing tests** (label `cpu`): `BlockRowGrid, HasTwelveBlocksInARow`;
  `BuildTestParticles, AllKinds_GroupedWithTheirArrays` (snow has `springy_deformation` and
  `permanent_volume_change`, liquid has `ingredient_fractions`); `BuildSpringyBlock, StretchedBlock_VolumeRatioIsDeterminant`;
  `ComputeSpringyEnergy, AtRestIsZero`.
- [ ] **Step 2: Run, implement, run.** Expected: four pass.
- [ ] **Step 3: Gate; commit.** Subject: `Add the scenes the materials tests share`.

### Task 14: Python reference and the comparison with `main`

**Kind:** numerics

**Files:**
- Create: `reference/materials/response.py`, `reference/materials/response_test.py`,
  `native/fluidity/materials/testdata/snow_response_cases.txt`, `sound_speed_cases.txt`, `main_snow_cases.txt`,
  `native/fluidity/materials/material_reference_test.cc`
- Modify: the materials `CMakeLists.txt` (`FLUIDITY_MATERIALS_TESTDATA_DIR` for this test; a CTest
  `reference_materials_response` running `uv run pytest reference/materials -q`, label `cpu`)

**Interfaces:**
- Produces `reference/materials/response.py` (Z-up, decided words, one function per kind so the sand module can
  add `sand_response` beside them): `snow_response(deformation, permanent, response)` through
  `numpy.linalg.svd` (signs fixed so both factors are rotations), clamp, permanent volume, stiffening, world force
  and rebuilt deformation; `fastest_wave(energy, deformation, rest_density, directions)`: first stiffness
  `A = d2 energy / dF dF` by central differences (step `1e-5` times the largest entry), pushed forward
  `a_ijkl = A_iJkL F_jJ F_lL`, largest eigenvalue of `a_ijkl n_j n_l` over the directions, divided by
  `rest_density`, square root; `write_cases(folder)`: seed `numpy.random.default_rng(20261114)`, 1,000 snow cases
  `F = R1 diag(s) R2^T`, `s` in `[0.9, 1.1]`, permanent in `[-0.3, 0.1]`, the test-table snow numbers; 100 snow
  and 100 springy states (`s` in `[0.7, 1.3]`) with 10,000 directions uniform on the sphere. Numbers as
  `float.hex`.
- The `main` numbers (`plan-decisions.md`, first entry): export `main` to a temporary directory, build there, and run a
  throwaway driver (never committed) that calls `ReturnSnowPrincipal3D` on the snow inputs of
  `native/tests/m5_constitutive_3d_cases.h:64-93` and `ReturnSnowDeformation3D` on the snow inputs of
  `native/tests/m5_deformation_3d_cases.h:9-35`, printing inputs and outputs as hex floats; rotate deformations
  and forces to Z-up with `F' = R F R^T`, `R` rows `{1,0,0},{0,0,-1},{0,1,0}`; commit only
  `main_snow_cases.txt` with a header note saying how it was produced.

- [ ] **Step 1: Failing tests.** Python: `test_cases_files_are_current` (regenerate into `tmp_path`, compare
  text), `test_snow_response_at_rest_is_zero_force`. C++ (label `cpu`):

  ```cpp
  TEST(ComputeMaterialResponse, SnowMatchesSvdReference)      // zero motion on each stored F: deformation,
      // permanent, world force within 1e-10 relative to the largest entry (design 7)
  TEST(ComputeSoundSpeedBound, AtLeastSampledFastestWave)     // native >= sampled * (1 - 1e-8) for all 200
      // states (plan decision: 1e-8 is the finite-difference slack). If one fails, stop and report the state:
      // the bound is not relaxed
  TEST(ComputeSnowResponse, MatchesMainCases)                 // 1e-12 relative
  TEST(ComputeMaterialResponse, SnowMatchesMainDeformationCases)  // zero motion, Z-up, 1e-12 relative
  ```
- [ ] **Step 2: Run** (fail), **write** the reference and the driver, generate the files, **run** both.
- [ ] **Step 3: Gate; commit.** Subject: `Check snow and the sound bound against a NumPy reference and main`.
  Body: the uncertainty removed (design 7) and how `main_snow_cases.txt` was produced.

### Task 15: Material buffers, memory estimate entries and `ComputeInternalForces` on the CPU

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** private `compute_internal_forces_gpu.h` (not `_gpu_calls.h`); the CPU loop calls the per-particle function of the new shared `compute_internal_forces_steps.h`; the table copies are written from `material_buffers_cpu.cc`; builders also map `RunOnCallStatus` through `ToBufferError`.

**Files:**
- Create: `material_buffers.h`, `material_buffers_cpu.cc`, `material_buffers_test.cc`,
  `compute_internal_forces.h`, `compute_internal_forces_cpu.cc`, `compute_internal_forces_gpu.h` (private),
  `compute_internal_forces_steps.h` (shared: `ComputeParticleInternalForce`, one particle's force, bound and error
  code, which the CPU loop here and the GPU function of Task 16 both call), `compute_internal_forces_test.cc`
- Modify: the materials `CMakeLists.txt` (headers, sources, tests with `LINKS fluidity_materials_test_scenes`)

**Interfaces:**
- Produces design 3: `MaterialSettings`, `MaterialBuffers`, `BuildMaterialBuffers` (with the plan's
  `max_cell_blocks` parameter), `BuildMaterialsMemoryEstimate`, `InternalForcesInputs`, `InternalForcesResult`,
  `ComputeInternalForces`. `MaterialBuffers` is move-only, has the public accessor `RunOn run_on() const` (plan decision), and also holds (private accessors for this module's
  files): per material a sound-bound `Buffer<double>` and an error-code `Buffer<std::uint8_t>`; a
  `Buffer<InternalForceTotals>` of `max_cell_blocks`; `Buffer<Material>` and `Buffer<Ingredient>` copies of the
  table; a result record (lowest failing id, largest bound per material, case counts) that the GPU halves write by
  integer atomics. Builders map `BufferError::kOutOfMemory` to `kOutOfMemory` (magnitude = bytes asked)
  and `kRunOnCallFailed` to `kRunOnCallFailed`; a GPU half's `RunOnCallStatus` goes through `ToBufferError` first.
  `BuildMaterialBuffers` writes the `Material` and `Ingredient` copies with `WriteBuffer` from
  `material_buffers_cpu.cc` (no `.cu` file: nothing of it runs on the GPU).
- Per particle of a material with internal force: `rest_density = mass / rest_volume`; `ComputeInternalForcePerArea`
  on `{springy_deformation, permanent_volume_change (0 when not carried), volume_ratio}`; write the matrix and
  the bound. Materials without internal force are skipped.

- [ ] **Step 1: Failing tests** (label `cpu`; AllKinds table, uniform grid, snow and springy particles at
  `{0.55..0.75}` unless stated):

  ```cpp
  TEST(BuildMaterialBuffers, AllKinds_PointersOnlyWithInternalForce)  // internal_force_per_area() size 4,
      // entries 0 and 1 null, 2 and 3 not null; friction and own_motion_shares not null for all four
  TEST(BuildMaterialsMemoryEstimate, AllKinds_BytesFromSizeof)   // counts {100,50,20,10}, MemoryCounts{cell_blocks 10}:
      // 72*30 + (8+16+8+1)*180 + 24*10 + 4*sizeof(Material) + 5*sizeof(Ingredient); static_assert sizes
  TEST(ComputeInternalForces, SpringyStretched_ForceIsNotTimesVolumeRatio)   // Review Focus 1: F = diag(1.2,1,1),
      // volume_ratio 1.2: buffer entry == ComputeSpringySolidForcePerArea(F) bit for bit
  TEST(ComputeInternalForces, NothingWithInternalForce_LimitIsInfinity)      // liquid only
  TEST(ComputeInternalForces, StepLimit_IsShareTimesWidthOverLargestBound)   // springy at rest mu 2, lambda 0,
      // mass/rest_volume 1 (bound exactly 2), width 0.5: step_limit == 0.0625 exactly
  TEST(ComputeInternalForces, StepExactlyAtLimit_IsAccepted)   // Review Focus 3: dt 0.0625 succeeds; dt =
      // nextafter(0.0625, 1) -> kStepTooLong, particle_id 0, material_index 3, magnitude 0.0625
  TEST(ComputeInternalForces, TwoMirrored_IsCollapsedForLowestId)   // ids 7 and 3 mirrored -> particle_id 3
  TEST(ComputeInternalForces, CollapseBeforeStepTooLong)
  TEST(ComputeInternalForces, SnowWithoutItsArrays_IsWrongFields)   // particles built without snow's extras
  TEST(ComputeInternalForces, MoreParticlesThanBuffers_IsBuffersTooSmall)
  TEST(ComputeInternalForces, WidthsOfWrongLength_IsBadInput)
  TEST(ComputeInternalForces, SplitIntoTwoRanges_SameBytesAndSmallestLimit)  // blocks [0,13), [13,27): buffer
      // bytes equal one call; min of the two limits equals the one-call limit
  TEST(ComputeInternalForces, GpuInBuildWithoutGpuCode_IsRunOnCallFailed)   // only without FLUIDITY_HAS_GPU
  ```
- [ ] **Step 2: Run** (compile error), **implement** (per material, per block, per particle in grouped order;
  one thread for now), **run**. `Read:` `<sec-sound-limit>`, `<sec-springy-solid>`.
- [ ] **Step 3: Gate; commit.** Subject: `Compute the internal force of every particle on the CPU`. New names:
  the list above, and the `max_cell_blocks` parameter (plan decision).

### Task 16: Material buffers and `ComputeInternalForces` on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** two halves (`compute_internal_forces_gpu.h` returning `RunOnCallStatus`), launches through
`base/gpu_launch.cuh`, kernel bodies in `compute_internal_forces_steps.h` with a `cpu` test over every thread block
and thread; `material_buffers_gpu.cu` is dropped (Task 15 writes the table copies with `WriteBuffer`); the
shared-formula GPU runner follows Foundation's `shared_math_test_gpu.h/.cu` split; result digests.

**Files:**
- Create: `compute_internal_forces_gpu.cu`, `compute_internal_forces_steps_test.cc`,
  `compute_internal_forces_gpu_test.cc`, `materials_no_gpu_test.cc`, `shared_formulas_gpu_test.cc`,
  `shared_formulas_test_gpu.h`, `shared_formulas_test_gpu.cu` (test-only runner of the shared formulas)
- Modify: `compute_internal_forces_steps.h` (the per-particle reduction step `ComputeFailureAndBoundKeys`), the
  materials `CMakeLists.txt` (`fluidity_add_gpu_sources(compute_internal_forces_gpu.cu)`; the steps test `LABEL
  cpu`; `fluidity_add_test(shared_formulas_gpu_test SOURCES shared_formulas_gpu_test.cc shared_formulas_test_gpu.cu
  LABEL gpu LINKS fluidity_materials_test_scenes)`; the other tests with their labels)

**Interfaces:**
- Produces (private, `compute_internal_forces_gpu.h`) `ComputeInternalForcesOneThreadPerParticle`, returning
  `RunOnCallStatus`, instantiated per `MaterialKind` with internal force; GPU function 1 runs
  `ComputeParticleInternalForce` per particle (grid-stride loop, `ApplyGpuFunction`); GPU function 2 reduces the
  failing id (integer `atomicMin`) and the largest bound per material (`atomicMax` on bits) into the result record;
  the CPU half reads the result record with `ReadBuffer`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  // compute_internal_forces_steps_test.cc, label cpu (the main evidence until the hardware run): both steps
  // functions for every thread block and thread of ComputeLaunchSize(particle count), launch order and reverse,
  // on the Task 15 scenes: forces, bounds and error codes equal the CPU loop's bit for bit (same functions, one
  // processor); the reduced lowest failing id and largest bound per material equal the CPU loop's
  // materials_no_gpu_test.cc, label no_gpu (every later GPU task adds its own line here)
  TEST(BuildMaterialBuffers, GpuWithoutAGpu_IsRunOnCallFailed)
  TEST(ComputeInternalForces, GpuWithoutAGpu_IsRunOnCallFailed)
  // compute_internal_forces_gpu_test.cc, label gpu
  TEST(ComputeInternalForces, OnGpu_MatchesCpuOnEveryTaskFifteenScene)   // Global Constraints tolerances
  TEST(ComputeInternalForces, OnGpuRunTwice_SameBytes)
  TEST(ComputeInternalForces, OnGpuThreePlantedBad_ReportsLowestId)      // ids 900, 41, 517 mirrored in three
                                                                         // different thread blocks: 41 (design 7, GPU)
  // shared_formulas_gpu_test.cc, label gpu
  TEST(SharedFormulas, OnGpu_MatchCpu)   // every shared-header test case of Tasks 7 to 12 run in one GPU function;
      // discrete results (errors, LimitCase away from a limit) exact, floating-point results with
      // CheckWithinTolerance at the Global Constraints tolerances (Exp, Log and Expm1 may differ in the last bits);
      // prints the largest difference per function
  ```
  Each `gpu` test passes its forces, bounds and step limit to `WriteResultDigest`.
- [ ] **Step 2: Implement; run** the gate. Expected: `PASS cuda` (compiled; `no_gpu` tests pass in the
  container).
- [ ] **Step 3: Commit.** Subject: `Compute the internal force on the GPU`. Body: `CUDA: compiled, not run`.

### Task 17: `ApplyInternalForce` on the CPU

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** private `apply_internal_force_gpu.h`; the per-face terms come from a shared `apply_internal_force_steps.h` function the CPU loop calls.

**Amended 2026-11-14 (A31):** one form taking `MaterialBuffers&`; the two-form test `BuffersForm_SameBytesAsOneCallForm` is removed.

**Files:**
- Create: `apply_internal_force.h`, `apply_internal_force_cpu.cc`, `apply_internal_force_gpu.h` (private),
  `apply_internal_force_steps.h` (shared: `ComputeFaceTerms`, one face place's velocity change and budget terms),
  `apply_internal_force_test.cc`

**Interfaces:**
- Consumes: `ComputeFacePush`, `transfer::GridFromParticles`, `ParticlesToGrid`, `math::TreeSum`.
- Produces design 3: `InternalForceTotals`, `ApplyInternalForce` (one form, `MaterialBuffers&` last),
  `ComputeInternalForceTotal`.
  Per face of each block (ascending place): force 0 contributes nothing; `kFromParticles` gets
  `velocity += push.velocity_change`; any other kind with force adds `|dt force|` to proposed and omitted;
  `weak_face_count` counts faces with force and `weight_sum < min_support_share`. Checks: `totals_of_block.size()
  != blocks.block_count` is `kBadInput`; grid version or `step_try == 0` of `grid_from_particles` is
  `kWrongGridVersion`. Ported from `main` `native/src/dry_solid_step_3d.cc:338, 353-360` (the floor is replaced,
  design 5.8).

- [ ] **Step 1: Failing tests** (label `cpu`; force from `ParticlesToGrid` with a hand-filled
  `internal_force_per_area` buffer unless stated):

  ```cpp
  TEST(ApplyInternalForce, LoneParticle_PushIsBounded)   // design 7: uniform grid, one particle, mass 1e-3,
      // S = diag(1e3,0,0), dt 1e-3, at y,z of X face (6,5,5) and x = 0.6 - (1.5 - d) * 0.1 for d in
      // {1e-1,1e-2,1e-3,1e-4,1e-5}. With the rule: |velocity change| <= dt |force| / (1e-3 * 1e-4) at every d
      // (closed form of the rule for one particle: face mass = 1e-3 * weight_sum); 1e-12 relative slack. With
      // min_support_share 0: change(1e-5) >= 500 * change(1e-2) (grows like 1/d)
  TEST(ApplyInternalForce, BudgetTotal_SameForAnyRangeCut)   // design 7: BuildBlockRowGrid; springy block
      // x [0.2,4.6), y,z [0.1,0.3), F = diag(1.05,1,1), forces from ComputeInternalForces; one call over 12 blocks,
      // then ranges of 1, 5 and 6 blocks into one 12-entry span: ComputeInternalForceTotal bytes equal;
      // face velocities equal bytes
  TEST(ApplyInternalForce, ForceOnFaceWithoutValue_IsAllOmitted)   // Review Focus 4: a hand-made
      // GridFromParticles (transfer::test::BuildGridFromParticlesByFunction) with one kTooFewParticles face of
      // force 3: velocity unchanged; proposed == omitted == dt * 3
  TEST(ApplyInternalForce, ZeroSupportShare_NothingOmittedOnValuedFaces)
  TEST(ApplyInternalForce, BudgetSpanOfWrongSize_IsBadInput)
  TEST(ApplyInternalForce, StaleStepTry_IsWrongGridVersion)
  TEST(ComputeInternalForceTotal, IsTreeSumOfBlocks)   // proposed {1e16, 1, -1e16, 1} -> 0.0 exactly;
                                                             // counts {1,2,3,4} -> 10
  ```
- [ ] **Step 2: Run, implement** (`// Step N:` of `<proc-apply-internal-force>`), **run**.
- [ ] **Step 3: Gate; commit.** Subject: `Apply the internal force to grid faces with the support rule`. New
  names: the list above.

### Task 18: `ApplyInternalForce` on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** `apply_internal_force_gpu.h` returning `RunOnCallStatus`; the in-thread-block reduction is built
from per-thread functions of `apply_internal_force_steps.h` with a `cpu` test phase by phase; the per-block budget
promises `math::TreeSum`'s bits unless NaN (stated in `apply_internal_force.h`), so the bit test compares with
`CheckSameBitsOrBothNaN`; result digests.

**Files:**
- Create: `apply_internal_force_gpu.cu`, `apply_internal_force_steps_test.cc`, `apply_internal_force_gpu_test.cc`;
  modify `apply_internal_force_steps.h` (`ApplyBudgetTreeRound`: one round of the in-thread-block tree, pair by
  pair as `math::TreeSum` adds them), `materials_no_gpu_test.cc`, the materials `CMakeLists.txt`

**Interfaces:**
- Produces (private) `ApplyInternalForceOneThreadBlockPerCellBlock`: one thread block of 256 threads per cell
  block, one thread per face place (16 idle), each runs `ComputeFaceTerms`; each term reduced in shared memory over
  the 256 padded slots by `ApplyBudgetTreeRound` in `math::TreeSum`'s tree; one `InternalForceTotals` per block to
  `MaterialBuffers`, then one `ReadBuffer` into `totals_of_block`. Started with `ApplyGpuFunction` and
  `LaunchSize{block_count, 256}`.

- [ ] **Step 1: Failing tests.** `cpu` (`apply_internal_force_steps_test.cc`, the main evidence until the hardware
  run): `ComputeFaceTerms` for every thread of every thread block, in launch order and reverse, then
  `ApplyBudgetTreeRound` round by round (every thread of a round, both orders), on the Task 17 scenes: face
  velocities and each block's budget equal the CPU loop's bit for bit. `no_gpu`:
  `ApplyInternalForce, GpuWithoutAGpu_IsRunOnCallFailed`. `gpu`:
  `ApplyInternalForce, OnGpu_MatchesCpuOnEveryTaskSeventeenScene` (budgets `1e-12` relative, velocities as
  state arrays; the promise does not apply because each processor's forces come from its own
  `ComputeInternalForces`); `OnGpuRangeCut_SameBytes`; `OnGpuRunTwice_SameBytes`; `OnGpuBlockBudget_EqualsCpuTreeSumBitForBit`
  (one block, values from a fixed integer formula: same tree, so `CheckSameBitsOrBothNaN` per term, by the header's
  promise). Each `gpu` test passes face velocities and block budgets to `WriteResultDigest`.
- [ ] **Step 2: Implement; run** the gate.
- [ ] **Step 3: Commit.** Subject: `Apply the internal force on the GPU`. Body: `CUDA: compiled, not run`.

### Task 19: `ComputeParticleFriction` on both processors

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** `compute_particle_friction_gpu.h` returning `RunOnCallStatus`; the GPU function is a grid-stride loop over `ComputeMixedFriction` with a `cpu` test over every thread block and thread; the exact CPU-against-GPU test is kept as a deliberate header promise (same bits unless NaN; reason in Global Constraints) and compares with `CheckSameBitsOrBothNaN`; result digests.

**Files:**
- Create: `compute_particle_friction.h`, `compute_particle_friction_cpu.cc`, `compute_particle_friction_gpu.cu`,
  `compute_particle_friction_gpu.h` (private), `compute_particle_friction_test.cc`,
  `compute_particle_friction_gpu_test.cc`; modify `materials_no_gpu_test.cc`, the materials `CMakeLists.txt`

**Interfaces:**
- Produces design 3: `ComputeParticleFriction` (owner Q35); private `ComputeParticleFrictionOneThreadPerParticle`.
  Fractions are read in scene ingredient order of the material (`first_ingredient` onward).

- [ ] **Step 1: Failing tests.** `cpu`: `MixtureParticle_IsFractionWeightedSum` (fractions `{0.25,0.75}` ->
  `0.25*0.2 + 0.75*0.4`, within 1e-15); `OneIngredient_WritesTheConstant` (snow 0.3, exact); `Gas_IsZero`;
  `SplitIntoTwoRanges_SameBytes`; `ComputeMixedFriction` for every thread block and thread of
  `ComputeLaunchSize(particle count)`, launch order and reverse, equals the CPU loop bit for bit (the main evidence
  until the hardware run). `no_gpu`: `GpuWithoutAGpu_IsRunOnCallFailed`. `gpu`: `OnGpu_MatchesCpu`
  (`CheckSameBitsOrBothNaN`: `compute_particle_friction.h` promises the same bits unless NaN, because each value is
  products and sums in a fixed order with contraction off on both processors), `OnGpuRunTwice_SameBytes`; both
  pass the friction array to `WriteResultDigest`.
- [ ] **Step 2: Implement; run** build, tests, gate.
- [ ] **Step 3: Commit.** Subject: `Give every particle its friction`. Body: `CUDA: compiled, not run`.

### Task 20: `ComputeOwnMotionShares` on the CPU

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** private `compute_own_motion_shares_gpu.h`; the per-particle work is `ComputeParticleOwnMotionShares` in the shared `compute_own_motion_shares_steps.h`, which the CPU loop calls.

**Files:**
- Create: `compute_own_motion_shares.h`, `compute_own_motion_shares_cpu.cc`,
  `compute_own_motion_shares_gpu.h` (private), `compute_own_motion_shares_steps.h` (shared:
  `ComputeParticleOwnMotionShares`, one particle's cell, interpolated distances and shares),
  `compute_own_motion_shares_test.cc`

**Interfaces:**
- Produces design 3: `OwnMotionInputs`, `ComputeOwnMotionShares`. Plan decisions: the particle's cell is found
  from `cell_begin` (no search); `cell_width` is `grid::ComputeCellWidth` of that cell; the two distances are
  trilinear in the cell's eight corners from `grid::FindCornerPlaces`; null `distance_to_solid` means
  `+infinity`; for kinds other than `kLiquid` no distance is read. `grid_at_particles.size()` must be the
  material count (`kBadInput`); corner values of another grid version give `kWrongGridVersion`.

- [ ] **Step 1: Failing tests** (label `cpu`; `GridAtParticle` buffers hand-filled):

  ```cpp
  TEST(ComputeOwnMotionShares, LiquidAtExpandingSurface_GetsInTravel)   // uniform grid; d_surface = z - 0.65 at
      // corners, d_solid = 1.0; particle at z 0.62; L = diag(0.1,0,0) -> {0.95, 0.5}
  TEST(ComputeOwnMotionShares, NullSurfaceDistance_InTravelIsZero)
  TEST(ComputeOwnMotionShares, NullSolidDistance_MeansNoSolid)
  TEST(ComputeOwnMotionShares, ParticleInSmallCellAtSizeStep_UsesItsOwnCell)   // Review Focus 5: size-step
      // grid; d_surface = 0.95 - x; particle in a 0.1 m cell at x 1.1 has d_surface -0.15 -> in_travel 0 (it
      // would be 0.5 with the big width 0.2); at x 1.02 (-0.07) -> 0.5; interpolated values within 1e-15
  TEST(ComputeOwnMotionShares, SnowSpringyGas_KeptOnly)
  TEST(ComputeOwnMotionShares, GridAtParticlesOfWrongCount_IsBadInput)
  ```
- [ ] **Step 2: Run, implement, run.**
- [ ] **Step 3: Gate; commit.** Subject: `Compute the travel shares of every particle on the CPU`.

### Task 21: `ComputeOwnMotionShares` on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** GPU half returns `RunOnCallStatus`; the GPU function is a thin loop over `ComputeParticleOwnMotionShares` with a `cpu` test over every thread block and thread; result digests.

**Files:**
- Create: `compute_own_motion_shares_gpu.cu`, `compute_own_motion_shares_steps_test.cc`,
  `compute_own_motion_shares_gpu_test.cc`; modify `materials_no_gpu_test.cc`, the materials `CMakeLists.txt`

- [ ] **Step 1: Failing tests.** `cpu` (the main evidence until the hardware run):
  `ComputeParticleOwnMotionShares` for every thread block and thread of `ComputeLaunchSize(particle count)`,
  launch order and reverse, on the Task 20 scenes, equals the CPU loop. `no_gpu`:
  `ComputeOwnMotionShares, GpuWithoutAGpu_IsRunOnCallFailed`. `gpu`:
  `OnGpu_MatchesCpuOnEveryTaskTwentyScene` (shares exact except within `1e-12` of a rule edge),
  `OnGpuRunTwice_SameBytes`; both pass the shares to `WriteResultDigest`.
- [ ] **Step 2: Implement** (private `ComputeOwnMotionSharesOneThreadPerParticle`, per kind, started with
  `ApplyGpuFunction`); **run** the gate.
- [ ] **Step 3: Commit.** Subject: `Compute the travel shares on the GPU`. Body: `CUDA: compiled, not run`.

### Task 22: `ApplyMaterialResponse` and `CheckMaterialState` on the CPU

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** private `apply_material_response_gpu.h`; per-particle bodies in the shared `apply_material_response_steps.h`, which the CPU loop calls; `CheckMaterialState`'s per-call GPU memory through `GpuArray` (its one form; `kOutOfMemory`).

**Files:**
- Create: `apply_material_response.h`, `apply_material_response_cpu.cc`, `apply_material_response_gpu.h`
  (private), `apply_material_response_steps.h` (shared: `ApplyParticleMaterialResponse` and
  `CheckParticleMaterialState`, one particle each), `apply_material_response_test.cc`

**Interfaces:**
- Consumes: `ComputeMaterialResponse`, `ComputeStretchDirections`, `ComputeSnowResponse`, `contact::ParticleContact`.
- Produces design 3: `MaterialResponseInputs`, `MaterialResponseResult`, `ApplyMaterialResponse`,
  `CheckMaterialState`. Plan decisions:
  - `to_change` must carry, by kind: liquid and gas `kVolumeRatio`; snow `kVolumeRatio`, `kSpringyDeformation`,
    `kPermanentVolumeChange`, `kFlags`; springy `kVolumeRatio`, `kSpringyDeformation`; else `kWrongFields`.
    `fields.cell_blocks != to_change.cell_blocks` is `kBadInput`. Each particle reads `read[i]` and writes
    `write[i]` only.
  - `CheckMaterialState` (5.4 step 5, `main`'s `MatchesDeformationState`, `dry_solid_step_3d.cc:50-71`): snow and
    springy re-run with zero motion; also (plan decision, invariant of 5.4) a gas `volume_ratio != 1`, or a snow
    or springy `volume_ratio` differing from its formula by more than `state_tolerance` relative, is
    `kOutsideAllowedRegion`. Its GPU path allocates per call through `GpuArray` (scene load only; it may return
    `kOutOfMemory`).

- [ ] **Step 1: Failing tests** (label `cpu`; AllKinds particles, `GridAtParticle` with random `L` of entries in
  `[-1, 1]` from seed 5, dt 0.01):

  ```cpp
  TEST(ApplyMaterialResponse, SnowHit_ClearsFittedChange)   // design 7: three snow particles with
      // kCarriesFittedChange; ParticleContact kind kHit for the second: it loses the bit, the others keep it
  TEST(ApplyMaterialResponse, EveryKind_MatchesComputeMaterialResponse)   // per particle, bit for bit
  TEST(ApplyMaterialResponse, Gas_VolumeRatioStaysOne)                     // exact
  TEST(ApplyMaterialResponse, CountsByCase_CoverEveryParticle)             // sum of counts == particles of range
  TEST(ApplyMaterialResponse, TwoFolded_IsCollapsedForLowestId)            // L = -20 I for ids 8 and 2 -> 2
  TEST(ApplyMaterialResponse, SnowWithoutFlags_IsWrongFields)
  TEST(ApplyMaterialResponse, NullParticleContact_MeansNoHit)
  TEST(ApplyMaterialResponse, SplitIntoTwoRanges_SameBytes)
  TEST(CheckMaterialState, StatesWrittenByApplyMaterialResponse_Pass)
  TEST(CheckMaterialState, SnowBeyondLengthening_IsOutsideAllowedRegion)   // stored diag(1.05,1,1)
  TEST(CheckMaterialState, GasVolumeRatioNotOne_IsOutsideAllowedRegion)
  TEST(CheckMaterialState, MirroredSpringy_IsDeformationCollapsed)
  ```
- [ ] **Step 2: Run, implement** (`// Step N:` of `<proc-apply-material-response>`; snow hit from `main`
  `dry_solid_step_3d.cc:570`), **run**.
- [ ] **Step 3: Gate; commit.** Subject: `Update every particle's remembered state on the CPU`.

### Task 23: `ApplyMaterialResponse` and `CheckMaterialState` on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** GPU half returns `RunOnCallStatus`; the GPU functions are thin loops over the Task 22 steps functions with a `cpu` test over every thread block and thread; result digests.

**Files:**
- Create: `apply_material_response_gpu.cu`, `apply_material_response_steps_test.cc`,
  `apply_material_response_gpu_test.cc`; modify `materials_no_gpu_test.cc`, the materials `CMakeLists.txt`

- [ ] **Step 1: Failing tests.** `cpu` (the main evidence until the hardware run): `ApplyParticleMaterialResponse`
  and `CheckParticleMaterialState` for every thread block and thread of `ComputeLaunchSize(particle count)`, launch
  order and reverse, on the Task 22 scenes: states bit for bit and lowest failing id and case counts equal to the
  CPU loop's. `no_gpu`: `ApplyMaterialResponse, GpuWithoutAGpu_IsRunOnCallFailed`,
  `CheckMaterialState, GpuWithoutAGpu_IsRunOnCallFailed`. `gpu`: `OnGpu_MatchesCpuOnEveryTaskTwentyTwoScene`
  (state tolerances; `LimitCase` rule of the Global Constraints); `OnGpuRunTwice_SameBytes`;
  `OnGpuThreePlantedBad_ReportsLowestId` (ids 900, 41, 517 folded, in three different thread blocks: 41);
  `OnGpuReadAndWriteSameMemory_SameBytes` (both runs on the GPU). Each `gpu` test passes the written states to
  `WriteResultDigest`.
- [ ] **Step 2: Implement** (private `ApplyMaterialResponseOneThreadPerParticle` and
  `CheckMaterialStateOneThreadPerParticle`, per kind; state in registers from read to write; case counts by
  integer `atomicAdd` into the result record; started with `ApplyGpuFunction`); **run** the gate.
- [ ] **Step 3: Commit.** Subject: `Update the remembered state on the GPU`. Body: `CUDA: compiled, not run`.

### Task 24: `ComputeCellSqueezeValues` on both processors

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** private `compute_cell_squeeze_values_gpu.h` returning `RunOnCallStatus`; per-cell body in the shared `compute_cell_squeeze_values_steps.h` with a `cpu` test over every thread block and thread; result digests.

**Amended 2026-11-14 (A31):** one form taking `const MaterialBuffers&`; tests build the buffers locally.

**Files:**
- Create: `compute_cell_squeeze_values.h`, `_cpu.cc`, `_gpu.cu`, `_gpu.h` (private), `_steps.h` (shared:
  `ComputeCellSqueezeValue`, one cell place, which the CPU loop calls), `_steps_test.cc`, `_test.cc`, `_gpu_test.cc`
  (prefix `compute_cell_squeeze_values`); modify `materials_no_gpu_test.cc`, the materials `CMakeLists.txt`

**Interfaces:**
- Consumes: `pressure::ComputePressureLawState`, `transfer::CellsFromParticles`.
- Produces design 3: `CellSqueezeInputs`, `ComputeCellSqueezeValues` (one form, `const MaterialBuffers&` last). Plan decisions: ingredients
  with `v_k == 0` are skipped (a gas ingredient's law is never evaluated in a liquid cell); sums in scene
  ingredient order from `0.0`; `PressureError::kPressureLawOutOfRange` becomes `kPressureLawOutOfRange` with
  `ingredient_index` and `magnitude` = the pressure, lowest `CellPlaceIndex`; outputs bound with
  `grid::ApplyGridVersion` only when their version differs.

- [ ] **Step 1: Failing tests.** `cpu`:

  ```cpp
  TEST(ComputeCellSqueezeValues, TwoIngredients_ClosedForm)   // design 7: volumes 1e-4 and 3e-4 m3, both
      // kSteadyStiffness with rest density 1000 and rest_sound_speed sqrt(2e6), sqrt(6e6); p = 1e6:
      // bulk = 1 / ((v1/K1 + v2/K2)/(v1+v2)), K = rho c*c as computed; target = (v1+v2)/(v1/J1 + v2/J2),
      // J = exp(-p/K); within 1e-13 relative. All kCannotBeSqueezed: +infinity and 1 exactly
  TEST(ComputeCellSqueezeValues, EmptyCell_InfinityAndOne)
  TEST(ComputeCellSqueezeValues, NullCarriedPressure_IsZeroPressure)
  TEST(ComputeCellSqueezeValues, BelowTaitRange_IsPressureLawOutOfRange)   // kStiffensUnderPressure, p = -K0/7
  TEST(ComputeCellSqueezeValues, OtherGridVersion_IsWrongGridVersion)
  ```
  `cpu` (`compute_cell_squeeze_values_steps_test.cc`, the main evidence until the hardware run):
  `ComputeCellSqueezeValue` for every thread block and thread of `ComputeLaunchSize(cell place count)`, launch
  order and reverse, equals the CPU loop and reports the same lowest failing place. `no_gpu`:
  `GpuWithoutAGpu_IsRunOnCallFailed`. `gpu`: `OnGpu_MatchesCpu` (1e-12 relative), `OnGpuRunTwice_SameBytes`; both
  pass the two cell arrays to `WriteResultDigest`.
- [ ] **Step 2: Run, implement** (private `ComputeCellSqueezeValuesOneThreadPerCell` reading the table copy in
  `MaterialBuffers`), **run** the gate.
- [ ] **Step 3: Commit.** Subject: `Compute each cell's squeeze stiffness and target volume`. Body:
  `CUDA: compiled, not run`.

### Task 25: `ComputeWorkRates` on the CPU

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** private `compute_work_rates_gpu.h`; per-particle and per-face terms in the shared `compute_work_rates_steps.h`, which the CPU loop calls.

**Files:**
- Create: `compute_work_rates.h`, `compute_work_rates_cpu.cc`, `compute_work_rates_gpu.h` (private),
  `compute_work_rates_steps.h` (shared: one particle's and one face place's work term, and one round of the
  in-thread-block tree), `compute_work_rates_test.cc`
- Modify: `materials_test_scenes.h`, `materials_test_scenes.cc`

**Interfaces:**
- Produces design 3: `WorkRates`, `ComputeWorkRates`. Test helper (plan decision, in `materials::test`):
  `ComputeExactChangeForFaceVelocity(grid, particles_try, material_is_counted, solids_at_faces,
  grid_from_particles, face_velocity) -> std::vector<Buffer<transfer::GridAtParticle>>`: copies
  `grid_from_particles`, writes `face_velocity` into its `velocity` (kinds kept), runs `SaveGridBeforeForces`,
  `SaveGridAfterPressure(face_velocity, null)` and `GridToParticles`.

- [ ] **Step 1: Failing tests** (label `cpu`):

  ```cpp
  TEST(ComputeWorkRates, CompleteNeighbourhood_Agree)   // design 7, main test_m5_dry_scenes_3d.cc:784-839:
      // one springy particle at {0.62,0.63,0.64}, rest_volume 0.5, hand-filled force per area rows
      // {10,2,3},{2,20,4},{3,4,-30} Pa; u = A x with A rows {3,2,1},{1,4,2},{2,3,-1}: from_grid == from_particles
      // == -87.5 W within 1e-12 relative (the sign follows design 3's definition; see hand-over)
  TEST(ComputeWorkRates, RandomFaceVelocities_Agree)    // 64 springy particles BuildCloud(6, 64, {0.35,0.35,0.35},
      // {0.85,0.85,0.85}), F with BuildRandomSymmetric(7, 0.05) + I, u random in [-1,1] (seed 8); then the size-step
      // grid with x in [0.35,1.25): agree within 1e-12 of the sum of absolute terms
  TEST(ComputeWorkRates, BesideSolid_DifferenceIsReported)   // same beside transfer::test::PlateAtX(0.6): both
      // returned finite; difference written with RecordProperty, not bounded
  TEST(ComputeWorkRates, SplitIntoBlocks_SameBytes)          // partials per block: same for one call or two
  ```
- [ ] **Step 2: Run, implement** (each sum: per block partial, then `math::TreeSum` over blocks), **run**.
  `Read:` `<sec-work-identity>`, `<eq:work-identity>`.
- [ ] **Step 3: Gate; commit.** Subject: `Add the work identity as a test oracle on the CPU`.

### Task 26: `ComputeWorkRates` on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** GPU half returns `RunOnCallStatus`; the per-call GPU memory is a `GpuArray` (test oracle, its one form); a `cpu` test runs the steps functions over every thread block and thread; result digests.

**Files:**
- Create: `compute_work_rates_gpu.cu`, `compute_work_rates_steps_test.cc`, `compute_work_rates_gpu_test.cc`;
  modify `materials_no_gpu_test.cc`, the materials `CMakeLists.txt`

- [ ] **Step 1: Failing tests.** `cpu` (the main evidence until the hardware run): the steps functions for every
  thread of every thread block (one per cell block), launch order and reverse, round by round, on the Task 25
  scenes: block partials equal the CPU loop's bit for bit. `no_gpu`:
  `ComputeWorkRates, GpuWithoutAGpu_IsRunOnCallFailed`. `gpu`:
  `OnGpu_MatchesCpuOnEveryTaskTwentyFiveScene` (1e-12 relative), `OnGpuRunTwice_SameBytes`; both pass the work
  rates to `WriteResultDigest`.
- [ ] **Step 2: Implement** (private `ComputeWorkRatesOneThreadBlockPerCellBlock`; per-call block partials in a
  `GpuArray`, so the call may return `kOutOfMemory`); **run** the gate.
- [ ] **Step 3: Commit.** Subject: `Compute the work rates on the GPU`. Body: `CUDA: compiled, not run`.

### Task 27: Scene-level tests: a free springy block and the repeat run

**Kind:** integration

**Files:**
- Create: `materials_scene_test.cc`
- Modify: `materials_test_scenes.h`, `materials_test_scenes.cc`

**Interfaces:**
- Produces (test only) `RunFreeStep(TestParticles&, const grid::StaggeredGrid&, const MaterialTable&,
  MaterialBuffers&, double step_duration) -> FreeStepReport { InternalForceTotals budget; Vec3 momentum_before;
  Vec3 momentum_after; Vec3 net_applied_impulse; double kinetic_energy; double springy_energy; }`: no gravity, no
  solid, no pressure. In order: `ComputeInternalForces`, `ParticlesToGrid`, `SaveGridBeforeForces`,
  `ApplyInternalForce` (applied impulse summed per axis as `mass * (velocity after - before)` over faces),
  `SaveGridAfterPressure(velocity, null)`, `GridToParticles`, `ComputeOwnMotionShares`, `MixParticleVelocity`,
  `position += step_duration * for_this_step`, `ApplyMaterialResponse`, `CommitTry`, new try, regroup. Energy:
  `sum m |v|^2 / 2` and `sum rest_volume * ComputeSpringyEnergy`.

- [ ] **Step 1: Failing tests** (label `cpu`; springy block `[0.4,0.8)^3` in the uniform grid, mu = lambda = 1e5,
  rest density 1000, F = diag(1.05,1,1), dt 1e-3):

  ```cpp
  TEST(ApplyInternalForce, NetImpulse_IsWithinOmitted)     // design 7: one step: |net_applied_impulse| <=
                                                           // budget.omitted_impulse + 1e-12 * proposed
  TEST(ApplyInternalForce, GridShift_ChangesLittle)        // design 7 (report 9.4): block shifted by 0.025 and
      // 0.05 m on each axis: |dp_shift - dp_0| <= omitted_0 + omitted_shift, dp = momentum change of one step
  TEST(ApplyInternalForce, SpringyBlock_EnergyNeverGrows)  // design 7: 50 steps: energy <= (1 + 1e-9) * start;
      // |momentum - start| <= running sum of omitted_impulse
  TEST(Materials, EveryBatchOperationTwice_SameBytes)      // design 7 "Repeat run": AllKinds particles; each of
      // the seven operations run twice on fresh copies: identical bytes
  ```
- [ ] **Step 2: Run, implement the driver, run.** If `SpringyBlock_EnergyNeverGrows` or
  `GridShift_ChangesLittle` fails, stop and report the numbers to the owner (design decision 2: the face-mask
  option (b) is then the first experiment); do not relax a bound.
- [ ] **Step 3: Gate; commit.** Subject: `Check a free springy block keeps its energy and momentum budget`.

### Task 28: Comment pass and hand-over

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** Step 3 also records the two same-bits promises, the dropped `material_buffers_gpu.cu`, and the retired `fluidity_add_shared_test`.

**Files:**
- Modify: any file of `native/fluidity/materials/`, `reference/materials/`, `docs/` the pass changes;
  `docs/gpu-runs.md`

- [ ] **Step 1: Comment pass.** Reread every file of the module from top to bottom as a C++ programmer who has
  never read a fluid paper: every comment true of the final code; why, not what; a worked example with numbers
  for each non-trivial public function (the 87.5 W case for `ComputeWorkRates`, the 1e-6 weight sum for
  `ComputeFacePush`); words in the glossary; `Uses` and `Used by` lists correct; the Kirchhoff warning in
  `material_response.h`; the extension-point comments of design 5.9 at each closed list.
- [ ] **Step 2: Fold each change into its owning commit** (`git commit --fixup`, `git rebase --autosquash`); run
  format and rules on each rewritten commit and the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3 names, 5 method, 6 CPU and GPU, 7 tests, 10 not ported)
  and record in the hand-over everything delivered differently: the plan decisions of the Global Constraints;
  `BuildMaterialBuffers`'s `max_cell_blocks`; one form per function taking `MaterialBuffers` (A31); memory estimate entries by `sizeof` with no face-place
  row; the `kStepTooLong` context; `atomicMax` on bits; the table check order and the extra ingredient rules; the
  extra `CheckMaterialState` checks; the `-87.5 W` sign; the `1e-8` reference slack; the same-bits promises of the
  per-block budget and of `ComputeParticleFriction`; no `material_buffers_gpu.cu`.
- [ ] **Step 4: `docs/gpu-runs.md`.** Row for materials: `compiled, not run`, listing every `gpu` test of Tasks
  16, 18, 19, 21, 23, 24, 26 to run at the end of the programme (each writes result digests, so the owner's repeat
  run covers it).
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the module's commits, the gate output, the new
  public names with reasons, and the statement that no device code has run.
