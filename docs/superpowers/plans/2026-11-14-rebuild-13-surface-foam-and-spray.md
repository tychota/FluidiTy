# Rebuild Surface, Foam and Spray Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the render output module of the rebuild: the recording of what each committed step offers (motion
samples near the surface, foam source events, the liquid totals at frame times) under try and commit, on the
CPU and (compiled, not run) the GPU; the per-frame surface mesh, foam particles and counted volumes, built on
the CPU from a snapshot; and every open-format writer (Alembic, OpenVDB, OpenUSD, and the solid-motion JSON) in
the build target `fluidity_interchange`, each proven by a test that reads its file back with the same library;
then the format table that chooses one format per kind of geometry from Blender witness records.

**Amended 2026-11-14** for owner answers A19 (each open-format library is pinned to the newest release Conan
Center lists, not to the version Blender bundles) and A16 (Blender 5.3 is the one supported version, so a witness
record holds one result). Each changed task carries an **Amended** line under its title.

**Architecture:** Three folders, one module and one error enum: `native/fluidity/render_output/`
(`fluidity::render_output`: lattice, motion sample, step recorder, volumes, the channel and format tables, the
frame entry point), `native/fluidity/surface/` (`fluidity::surface`), `native/fluidity/foam_and_spray/`
(`fluidity::foam_and_spray`). The writers live in `native/fluidity/render_output/interchange/`, the only sources
of the static library `fluidity_interchange`, which links the native library `fluidity` and the six pinned
third-party libraries and is linked by the CLI and the interchange tests only. Only `RecordStep` runs on the
simulation's processor (`step_recorder_cpu.cc` / `step_recorder_gpu.cu`); everything else is CPU only and runs
on I/O's worker thread. Order of work: paper; the libraries one by one; errors and tables; points first, then
volumes, then the surface, then foam and spray, then solid motion, then the USD candidates, then the format
table; then the frame entry point, the hand-over to I/O, scene tests, comment pass.

**Tech Stack:** as Foundation (C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2, uv, clang-format
and clang-tidy 23.1 as `pyproject.toml` pins them, Docker via colima, Typst with CeTZ), plus, in `fluidity_interchange` only: OpenUSD, Alembic,
OpenVDB, oneTBB, Imath and zlib, each at the newest release on Conan Center when its task runs (A19; Tasks 4 to
9 record the versions). No Python reference, no Lean proof (design section 7).

**Spec:** `docs/superpowers/specs/modules/13-surface-foam-and-spray-design.md` ("the design"; in the `main` checkout,
never committed; its section 3 is the contract) and the programme design `docs/design/programme.md` on `rebuild`
(sections 6, 7, 8, 8a). Also in the `main` checkout: `modules/owner-answers.md` (A1, A2, A8, A9, A10, A14, A16, A19),
`modules/shared-interface.md`, `modules/plan-decisions.md` (it wins over every design), and section 3 of the contact,
grid, particles, transfer, pressure, materials, solids, gas, step, cell resize and batches, and I/O and CLI designs.

## Global Constraints

Foundation's list, unchanged:

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Old code is read with `git show main:<path>` from the worktree. Nothing is checked out wholesale.
- Commit rule (spec section 6): one thing per commit; strict CPU build and all tests pass; CUDA target compiles
  in the container; clang-format 23.1 clean; clang-tidy 23.1 clean on host sources; every touched file passes
  the `AGENTS.md` checklist; new public names listed in the commit message with a one-line reason; tests arrive
  with the code. A commit must pass every part of the gate that exists at that commit.
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
  abbreviations (MAC, PIC, APIC, P2G, G2P, EOS), no milestone codes. Function verbs: `Compute`, `Build`, `Check`,
  `Find`, `Apply`, `Solve`, `Read`, `Write`; anything else is listed for the owner's review.
- 120 columns, two-space indent, `//` comments, braces on every control-flow body.
- Tests are named `TEST(FunctionOrType, Situation_Outcome)`, one behaviour each, in `*_test.cc` beside the code.
- Deletion is the default: add a dependency, file or tool only when a task here names it.
- **Amended 2026-11-14 (Foundation as built):** every task follows `AGENTS.md` "Structure" and `plan-decisions.md`
  "Foundation after the final review". Two halves: `RecordStep`'s public function lives in `step_recorder_cpu.cc`
  (checks, `RunOn` dispatch, the CPU loop, the GPU half under `#if defined(FLUIDITY_HAS_GPU)`); the GPU half
  `step_recorder_gpu.cu` is declared in the private `step_recorder_gpu.h`, returns `RunOnCallStatus`, which the CPU
  half maps with `ToBufferError` and then into `RenderOutputFailure` (`kRunOnCallFailed` or `kOutOfMemory`) through
  one mapping function; it launches, waits and holds per-call GPU memory only through `base/gpu_launch.cuh`. Each
  GPU function is a thin loop over `FLUIDITY_CPU_AND_GPU` per-item functions of the shared `step_recorder_steps.h`,
  which a `cpu` test runs for every thread block and thread, in launch order and in reverse, against the CPU loop:
  the main evidence until the hardware run. Working buffers: the recorder owns, for the whole run, a `SumBuffers`
  and an `OrderBuffers` on its processor, and passes them to `ComputeLiquidTotals`, `BuildStableOrder` and
  `ComputeExclusivePrefixSum` (each has one form, which takes its buffers, A31); `BuildRenderOutputMemoryEstimate`
  counts them with `ComputeSumBuffersSize` / `ComputeOrderBuffersSize`. The frame work on the CPU worker runs once
  per frame and uses one CPU `OrderBuffers` that `RenderOutput` owns for the run (`frame_order_buffers`, sized
  from `RenderLimits::max_field_parts` and `max_foam_particles`; amended 2026-11-14, A31, reconcile item 26).
  Buffers passed together live on the same processor: a `Requires:` precondition with a debug assertion (A30). Errors: a
  failed check writes nothing; after `kRunOnCallFailed` the recorder's pending record is unspecified and the run
  stops; `kOutOfMemory` comes only from building forms. Exactness: discrete outputs equal on CPU and GPU;
  floating-point within the tolerances below unless a header promises the same bits unless NaN
  (`CheckWithinTolerance`, `CheckSameBitsOrBothNaN`). Tests link `fluidity_base_test_support` and the helper
  libraries they use (made with `fluidity_add_test_library`) through `LINKS`; `gpu` tests pass their main outputs
  to `WriteResultDigest`. Labels: `cpu` everywhere, `no_gpu` only for behaviour on a machine without a GPU, `gpu`
  only on hardware. Every target goes through the helpers of `cmake/fluidity_targets.cmake` (the library
  `fluidity_interchange` through `fluidity_apply_compile_flags`, Task 4), and every source file is listed in a
  CMake file.
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the CPU build in the debug and the
  release preset, and CUDA compiled in the container, whose `cuda` step fails for a tracked `.cc` or `.cu` file that
  no build compiles. clang-format and clang-tidy are the 23.1 that `pyproject.toml` pins.

The module's own (values from the design; never change one without the owner):

- **Section 3 is the contract.** Implement its declarations exactly. Additions and corrections are marked
  **Plan decision** in the task that makes them and repeated in the hand-over. The ones that fix a defect of the
  design itself: `lattice.h` is split into a shared `lattice.h` and a CPU-only `lattice_values.h` (a shared
  header cannot hold `std::vector`); `RenderLimits` lives in `render_output/render_limits.h` and `FrameIdentity`
  in `render_output/frame_identity.h` (both are needed by headers that `render_frame.h` includes, so declaring
  them there would make a cycle); `Ramp`, `FoamSourceSettings` and `FoamSourceEvent` live in the shared
  `foam_and_spray/foam_source.h` (shared formulas take them) and `foam_settings.h` is shared (plain values only).
  Byte counts in design comments ("64 bytes") are replaced by `sizeof` of the real type (plan-decisions).
- **Errors:** only `kBadInput`, `kWrongStateVersion`, `kOutOfMemory`, `kRunOnCallFailed` leave `BuildRenderFrame`
  or `RecordStep` (`RecordStep` also `kWrongGridVersion`); only `kBadInput`, `kFormatNotInBuild`,
  `kFileWriteFailed`, ~~`kOutOfMemory`~~ leave a writer. **Amended 2026-11-15 (A58, reconcile item 31):**
  `kOutOfMemory` is the last value (16 values) and comes only from the building functions `BuildStepRecorder`,
  `BuildRenderOutput`, `BuildRenderOutputFromSnapshot` and `BuildRenderFrame` (also for
  `BufferError::kRunMemoryTooSmall`, `magnitude` = bytes asked); `RecordStep` never returns it; a writer's library
  `std::bad_alloc` is `kFileWriteFailed`. Every other value is a reason inside a `SurfaceReport` or
  `FoamReport`. A `CellResizeFailure`, `GridFailure` or `BufferError` from below is `kRunOnCallFailed` with the
  bytes asked in `magnitude` when it was an allocation, except a grid-version mismatch (`kWrongGridVersion`).
- **Memory (A58, amended 2026-11-15; Foundation Tasks 17 to 19).** `BuildStepRecorder` takes `RunMemory&
  run_memory` last (the recorder's run-processor buffers); `BuildRenderOutput` and `BuildRenderOutputFromSnapshot`
  take `RunMemory& run_memory`, the CPU block (the step's `cpu_run_memory`; the one block in a CPU run), for
  `frame_order_buffers` (`kCpu`, `kWholeRun`; `plan-decisions.md` item 3 and its same-day correction). `Requires:`
  the block's `run_on()` matches (debug assertion, A30). Per-frame worker arrays (`kOneSnapshot`, `kOneCall`) keep
  plain allocation. One estimate row per buffer.
- **Wind (A47, A50; amended 2026-11-15).** `RenderSettings::wind` (a `gas::GasWind` from `ReadyScene::wind`) moves
  spray (`v += dt (g + a_wind(t + dt / 2))`) and foam (drift `kFoamWindDriftDuration = 0.1 s` times the horizontal
  wind acceleration); never the liquid.
- **Constants:** `kLatticeBlockWidth = 8` (512 points a block, x fastest); block key 21 bits per axis, z highest,
  offset `2^20`; `kMaxMovesPerPoint = 4096`, each move at most half a spacing; `MotionSettings` defaults
  (`spacing` 0 = smallest cell width, at most twice it; `reach_in_spacings` 4; `samples_per_frame` 1..64);
  `SurfaceSettings` defaults (`spacing` 0 = half the smallest cell width; `smoothing_width` 0 = 2 spacings;
  `max_edge_in_spacings` 2.5; `max_pull_back_in_spacings` 0.25; `volume_tolerance` 0.005;
  `keeps_ids_between_frames` true); Gaussian cut at `round(4 * smoothing_width / spacing)` points; `open >= 0.05`
  else value `+0.5`; field point split `k = min(ceil(width / spacing), max(8, ceil(width / (2 spacing))))` (A65,
  2026-11-15; was "at most 8"); mesh tie rule: `|value| <= 1e-8`
  counts as `+1e-8`; pull back: at most 12 Newton steps, each halved up to 16 times, done at
  `|value| / |value_change| <= min(1e-5 * spacing, 0.01 * smallest triangle height at the vertex)`; volume offset:
  at most 3 passes, limit `0.25 * spacing`, an unsafe pass halved once then dropped; age carried from the nearest
  moved vertex within one spacing (lowest id on a tie); a vertex is "at a solid" at `distance_to_solid <= 0.25 *
  spacing`; `distance_to_solid` capped at `+-4` spacings; foam sources at `-2 spacing <= d <= spacing`, crest when
  `u . n >= 0.6 |u|`, normal not known below length 0.5; `FoamSourceSettings::max_events_per_step = 1 << 20`;
  `FoamSettings` defaults (`particles_per_volume_time` 1000, `count_per_particle` 1, `kind_change_distance`
  0.1 m, held for `0.5 r`, `foam_lifetime` 2 s, `bubble_radius` 0.001 m, `display_radius` 0.005 m);
  `RenderLimits` defaults (`max_lattice_blocks = 1 << 20`, `max_vertices = 1 << 26`, `max_foam_particles =
  1 << 24`); `kOutputContractVersion = 2`.
- **Order of sums** (design 5.3, 5.6, 6): at one lattice point, field parts in (lattice cell, material index,
  grouped order of the particle, part number) order through `BuildStableOrder`, the 8 cells around a point in a
  fixed order (z, y, x ascending); volume counts by lattice point, then particle id (`BuildGroupsByDestination`);
  foam neighbours in x, y, z order; recorded points and events by (block key, point in block); liquid totals by
  `math::TreeSum` over block partials in ascending whole-grid block index. Threads take whole lattice blocks or
  index ranges with no shared sum. No floating-point atomics.
- **The one-way rule** (design 5.1): no function of this module takes a non-const simulation, particles, grid or
  solids; the surface field and mesh are never an input of another module.
- **Tolerances, CPU against GPU, `RecordStep` only** (design 6): point lists equal; `known` equal except where a
  weight sum is within `1e-12` of the grid's floor; velocities `1e-6 * largest speed + 1e-9` m/s; distances
  `1e-6 * spacing`; an event in both lists unless its `volume_time < 1e-9 * spacing^3 * step_duration`;
  `volume_time` `1e-6` relative plus that floor.
- **Libraries and `fluidity_interchange`** (design 5.7, amended 2026-11-14 by A19): OpenUSD, Alembic (Ogawa only,
  no HDF5), OpenVDB (core only, no tools; Blosc and Boost off where the recipe allows, Task 8), oneTBB, Imath and
  zlib, each pinned exactly to the newest release Conan Center lists when its task runs (Tasks 4 to 9 find it
  with `conan list` and record it), through Conan and `conan.lock`, never built inside a module's test target.
  The design's table chose each version to match the one Blender bundles; A19 drops that rule, and nothing in
  this plan relies on it. Files written by a library version other than Blender's (newer, or for OpenVDB
  possibly older: Conan Center's newest OpenVDB was 12.1.1 on 2026-11-14, Blender 5.3 bundles 13.0) are read
  by Blender 5.3's readers; the Blender add-on's witness tests confirm this per format, and a format that fails
  there leaves the table (add-on plan Task 41). Only `native/fluidity/render_output/interchange/*.cc`
  include their headers; no file outside `interchange/` and `cli/` includes an `interchange/` header.
  **Plan decision:** the interchange sources (and only they, besides `cli/read_command_line.cc`) compile with
  exceptions and RTTI on, because the three libraries throw and use `typeid`; every public writer catches every
  exception at its own boundary (~~`std::bad_alloc` gives `kOutOfMemory`, anything else~~ every exception gives
  `kFileWriteFailed`, amended 2026-11-15, reconcile item 31: a writer is not a building function), so no
  exception crosses an `interchange/` header. Library headers are included as system headers through the imported
  Conan targets; our warning flags are never lowered.
- **Files** (design 5.7): one file per frame and product, one time sample at time code `identity.frame`. Alembic
  is the only axis conversion: positions, velocities and transforms written as `(x, z, -y)` and triangle order
  reversed. Only float and int32 per-point and per-face values are written. **Plan decision:** every file stores
  the same six metadata keys, `fluidity_frame`, `fluidity_state_version`, `fluidity_time`,
  `fluidity_frames_per_second`, `fluidity_contract_version`, `fluidity_channels` (Blender names, in
  `ReadOutputChannels()` order, comma-separated): Alembic archive metadata (text, numbers by `std::to_chars`
  shortest round-trip form), USD `customLayerData` (int64, double, string), OpenVDB file metadata (Int64, Double,
  String), JSON top-level fields. The Alembic sample time is `identity.frame / identity.frames_per_second`.
- **Test labels (amended 2026-11-14, Foundation as built):** shared-formula and operation tests `LABEL cpu` (the
  container's `cuda` step runs them too); device tests `gpu`; the `_gpu.cu` operation also has `no_gpu` "on the GPU
  without a GPU gives `kRunOnCallFailed`" (plan-decisions). Interchange tests are
  `fluidity_add_test(<name> SOURCES ... LABEL cpu LINKS fluidity_interchange_test_support fluidity_interchange
  fluidity_base_test_support)`: the earlier helper `fluidity_add_interchange_test` (label `cpu` or `no_gpu` by build,
  plus `interchange`) is not added, since `fluidity_add_test` takes one label and the container runs `cpu` tests.
  The design marks no test known-open.
- **Inputs of later modules are hand-built in tests** (the Blender add-on). Earlier modules are called for real:
  step test scenes (`fluidity_step_test_support`), transfer and pressure test scenes, I/O's `RunFolder` and
  `OutputFile`. Nothing is stubbed.
- **Code comments** number `RecordStep`, `BuildSurfaceFrame`, `BuildRenderFrame` and `MoveFoamParticles`
  `// Step N:` with the numbers of their procedures in the method paper (Tasks 1 to 3), which follow design 5.2,
  5.3 ("Frame strategy"), 4 item 3 and 5.5.

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, already committed: Foundation
with every `base/` addition (`math::Expm1`, `Cbrt`, `TreeSum`, `Buffer` with `CopyBuffer`, `BuildStableOrder`,
`BuildGroupsByDestination`, `MemoryEstimateEntry` with `where`), contact, grid, particles, transfer, pressure, viscosity,
materials, solids, gas (`ToGasSnapshot` with three arguments), step (`FoamAndSprayInputs` with the additions of
design section 4, ~~`SimulationSnapshot` with the region arrays,~~ `Module::kRenderOutput`, `BuildRunPlan` with
`output_estimate`, `step/random_words.h`), cell resize and batches (`cell_resize/regions.h` with `LiquidTotals`,
`ComputeLiquidTotals` over particles only, ~~`GridOnlyAmounts`~~; amended 2026-11-15, A64: no grid-only arrays in
the first version), I/O and CLI (`io/run_folder.h` with `OutputFile`,
`path_while_writing`, `BuildOutputFile`, `WriteListedFile`, `CheckAfterWrite::kSizeOnly`).

**Amended 2026-11-14 (Foundation as built):** Foundation as built also provides `base/run_on_call.h`
(`RunOnCallStatus`, `ComputeLaunchSize`), `base/gpu_launch.cuh`, the sums with their `SumBuffers` and the stable
order, prefix sum and grouping with their `OrderBuffers` (one form each, A31), `base/test_support.h` (library
`fluidity_base_test_support`) and `fluidity_add_test_library`; cell resize provides
`ComputeLiquidTotals`, whose one form takes a `SumBuffers&`. Read `AGENTS.md` "Structure" before the first task. The gate prints seven `PASS` lines:
`format`, `rules`, `cpu` debug, `cpu` release, `tidy`, `paper`, `cuda`.

**Amended 2026-11-15 (A58, A47):** Foundation's Tasks 17 to 19 (`RunMemory`, `BuildBuffer(RunOn, size,
RunMemory&)`, `ComputeRunMemoryBytes` with `MemoryWhere`, `BufferError::kRunMemoryTooSmall`, the `RunMemory&` forms of
`BuildSumBuffers` / `BuildOrderBuffers`) and gas's `GasWind`, `SampleGasWind`, `CheckGasSources`.

Prove it: `uv run python tools/check_commit.py` prints `PASS` on every line, and
`git grep -l "ComputeLiquidTotals\|path_while_writing\|ComputeRandomWords\|ToGasSnapshot\|BuildContactSolids" -- native`
lists files of cell resize, I/O, step, gas and solids. If one is missing, stop and ask the owner. Then run
`git grep -n "render_output" -- native` and keep the output: it tells Task 47 which render-output seams the I/O
module left for this module.

## Review Focus

1. **The Alembic turn and winding.** A writer that turns positions but forgets to reverse triangles, or the
   reverse, passes a read-back test that undoes its own conversion. Pinned:
   `WriteSurfaceFile, AlembicSphere_OutwardAfterBlenderTurn` in Task 28 (the test applies Blender's import turn as
   a literal matrix and checks a positive signed volume equal to the mesh volume).
2. **Ids at or above `2^31`.** `id` must be the low 31 bits as int32, the full id must survive in Alembic
   `.pointIds` and USD `ids`. Pinned: `WritePointsFile, IdAbove31Bits_LowBitsAndFullId` in Task 13 (and its USD
   twin in Task 40).
3. **A try that ends one rounding error away from the frame time.** The frame sample and the liquid totals must
   still be taken. Pinned: `RecordStep, TryEndOneUlpFromFrameTime_KeepsFrameSample` in Task 30.
4. **A library that throws.** An unwritable path or a full disk must give `kFileWriteFailed`, never terminate the
   program. Pinned: `WriteVolumeFile, PathInMissingFolder_IsFileWriteFailed` in Task 16.
5. **OpenUSD without its plugin resources.** A static USD that cannot find `plugInfo.json` must fail with
   `kFileWriteFailed` and `ReadInterchangeLibraries().missing_resource_folder` naming the folder. Pinned:
   `WritePointsFile, UsdWithoutResources_IsFileWriteFailed` in Task 9.

---

### Task 1: Method paper, the one-way rule, the motion sample, volumes and formats

**Amended 2026-11-14 (A16, A19):** the formats section names no library version and states that the format choice
rests on witness tests in Blender 5.3 alone.

**Kind:** mechanical

Load the `typesetting-with-typst`, `typesetting-documents` and `drawing-with-cetz` skills first.

**Files:**
- Create: `docs/method-paper/sections/render_output.typ` (included from `fluidity-method.typ` after the I/O
  section, as earlier modules include theirs), `docs/method-paper/figures/render_output_flow.typ`,
  `figures/known.typ`
- Modify: `docs/method-paper/fluidity-method.typ` (include; notation rows)

**Interfaces:**
- Produces labels `<sec-render-output>`, `<sec-motion-sample>` (**Plan decision**: the lattice and the motion
  sample, outline item 2), `<sec-volume-output>`, `<tab-output-channels>`, `<sec-output-formats>`,
  `<tab-format-choice>`, `<fig-render-output-flow>`, `<fig-known>`, `<proc-record-step>`,
  `<proc-build-render-frame>`.

- [ ] **Step 1: Text**, design section 8 outline items 1, 2, 9, 10, 11, in the vocabulary of design section 9:
  what is looked at is not what is solved (four volumes, three kinds of bubble); the two halves and the one-way
  rule (5.1); the lattice (blocks of 8 x 8 x 8, not the staggered grid) and the motion sample (5.2, steps 1 to 7,
  including why batches give one-piece bytes); counted volumes and zero, unknown, absent (5.6); the channel table
  `<tab-output-channels>` (every row of 5.7's field contract); open formats `<sec-output-formats>`: the rule of
  A14, one file per frame, each format's shape, the Alembic turn, what the Blender manual says
  (quote the 5.7 lines), the libraries (each the newest release on Conan Center when pinned, A19; no version
  number in the text, since `capabilities` reports them), "deliberately not promised"; `<tab-format-choice>` (the
  four kinds, candidates, chosen today; the choice rule: the first candidate whose every witness passed in
  Blender 5.3, else the last).
- [ ] **Step 2: Figures** with `figure_style.typ`: `<fig-render-output-flow>` and `<fig-known>` exactly as design
  section 8 describes them. Each caption states the one thing to see.
- [ ] **Step 3: Pseudocode** with `procedure(...)`: `<proc-record-step>`, numbered as design 5.2 steps 1 to 7
  plus 8 "find foam source events" (5.4) and 9 "keep apart until commit" (5.1 item 2); fails when
  `kRunOnCallFailed`, `kWrongGridVersion`, `kBadInput`. `<proc-build-render-frame>`: 1 check order and
  versions; 2 build the interval (last sample, recorded samples, sample at frame); 3 per group
  `BuildSurfaceFrame`; 4 foam: add, move (with kinds), remove; 5 foam volume; 6 gas volume through
  `gas::ToGasSnapshot`; 7 `liquid_volume_outside_surfaces`; 8 keep what the next frame needs. Fails when
  `kBadInput`, `kWrongStateVersion`, `kOutOfMemory`; then nothing kept changes.
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`.
  Expected: exit 0, no warning. Look at every new page; fix overlaps and clipped labels. Run the gate.
- [ ] **Step 5: Commit.** Subject: `Describe render output, its motion sample, volumes and formats`.

### Task 2: Method paper, the surface

**Kind:** numerics

**Files:**
- Modify: `docs/method-paper/sections/render_output.typ`
- Create: `docs/method-paper/figures/surface_field.typ`, `figures/surface_strategy.typ`, `figures/six_tetrahedra.typ`

**Interfaces:**
- Produces labels `<sec-surface>`, `<eq:surface-field>`, `<fig-surface-field>`, `<fig-surface-strategy>`,
  `<fig-six-tetrahedra>`, `<proc-build-surface-frame>`.

- [ ] **Step 1: Text**, outline items 3 to 6: the field and its three fixes (5.3 "Field", steps 1 to 7) with
  `<eq:surface-field>` `value(x) = 0.5 - (G * V)(x) / (G * open)(x)`; a mesh safe by construction (six tetrahedra,
  the tie rule, no merge into corners); moving the last mesh, the checks, replacing a blob (5.3 "Frame strategy");
  volume as a report; several materials in a group; gas beside liquid. Cite ref 27 (Wojtan 2009), ref 33 (Brochu
  and Bridson 2009), ref 17 (Ferstl 2016), ref 13 (Bhattacharya 2011, the field not used); Zhu and Bridson 2005
  and the six-tetrahedra mesh by author and year (design decision 9).
- [ ] **Step 2: Figures** `<fig-surface-field>`, `<fig-surface-strategy>`, `<fig-six-tetrahedra>` as design
  section 8 describes them.
- [ ] **Step 3: Pseudocode** `<proc-build-surface-frame>`: the seven numbered steps of 5.3 "Frame strategy", with
  2a to 2d; outputs `SurfaceFrame`; "fails when" only `kOutOfMemory` and `kBadInput` (a mesh reason is a report).
- [ ] **Step 4: Run** typst, look at the pages, run the gate. **Commit.** Subject: `Describe the surface mesh and
  how a frame chooses it`.

### Task 3: Method paper, foam and spray

**Kind:** numerics

**Files:**
- Modify: `docs/method-paper/sections/render_output.typ`
- Create: `docs/method-paper/figures/foam_kinds.typ`

**Interfaces:**
- Produces labels `<sec-foam-and-spray>`, `<eq:foam-source>`, `<eq:foam-drag>`, `<fig-foam-kinds>`,
  `<proc-move-foam-particles>`.

- [ ] **Step 1: Text**, outline items 7 and 8: the three measures and `<eq:foam-source>` exactly as the 5.4 block,
  with the normal and curvature from `distance_to_surface` (not from a sphere) and the occupied share; events and
  `max_events_per_step`; the life of a foam particle (5.5 steps 1 to 4) with `<eq:foam-drag>` and its `k = 0`
  form; why particles never read one another. Cite ref 12 (Ihmsen 2012).
- [ ] **Step 2: Figure** `<fig-foam-kinds>`: spray, foam, bubble against distance to the surface, the held range
  `0.5 r` past each limit.
- [ ] **Step 3: Pseudocode** `<proc-move-foam-particles>`: 1 pieces from `max(start_time, last frame time)` to the
  frame time, ending at sample times, at most half a spacing; 2 spray drag; 3 bubble drag (zero liquid velocity
  where not known, counted); 4 foam midpoint (keeps its velocity where not known, counted); 5 solids; 6 kind at
  the frame time.
- [ ] **Step 4: Run** typst, look at the pages, run the gate. **Commit.** Subject: `Describe foam sources and the
  life of a foam particle`.

### Task 4: zlib, newest on Conan Center, and the target `fluidity_interchange`

**Amended 2026-11-14 (A19):** pins the newest zlib release on Conan Center, found in Step 1, instead of the version
Blender bundles.

**Amended 2026-11-14 (Foundation as built):** `fluidity_interchange` is created with `add_library` plus `fluidity_apply_compile_flags(...
EXCEPTIONS ON RTTI ON)` (the `RTTI` argument comes from I/O plan Task 34); no `fluidity_add_interchange_test` helper: interchange tests are `fluidity_add_test(...
LABEL cpu ...)`.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/render_output/interchange/interchange_libraries.h`, `interchange_libraries.cc`,
  `interchange_libraries_test.cc`
- Modify: `conanfile.py`, `conan.lock`, `CMakeLists.txt`, `AGENTS.md` (Language),
  `tools/check_commit.py` and `tools/cuda-container/Dockerfile` only if Step 5 says so

**Interfaces:**
- Produces: static library `fluidity_interchange` (sources listed the way `fluidity`'s are, never globbed; links
  `fluidity` and the Conan targets; exceptions and RTTI on for its sources only). **Plan decision (amended
  2026-11-14, Foundation as built):** it is created with `add_library(fluidity_interchange STATIC ...)` followed by
  `fluidity_apply_compile_flags(fluidity_interchange EXCEPTIONS ON RTTI ON)`, so `fluidity_check_compile_flags`
  accepts it; the optional `RTTI` argument of `fluidity_apply_compile_flags` was added by I/O plan Task 34
  (default `OFF`; `ON` replaces `-fno-rtti`, `/GR-` and `-Xcompiler=/GR-` by `-frtti` / `/GR`). Interchange tests
  are `fluidity_add_test(<name> SOURCES ... LABEL cpu LINKS fluidity_interchange fluidity_base_test_support)`; no
  `fluidity_add_interchange_test`. `struct InterchangeLibraries` and `ReadInterchangeLibraries()` of design 3.6, with one
  **Plan decision** field added: `std::string_view missing_resource_folder;  // Empty when OpenUSD found its
  plugInfo resources (Task 9).` All fields are empty in this task.

- [ ] **Step 1: Find the newest release and pin it.** First `uv run conan remote list` in the worktree's Conan
  home (`CONAN_HOME=$PWD/out/conan-home`): the remote named `conancenter` must point at
  `https://center2.conan.io`. The old `https://center.conan.io` is frozen and lists only old versions (on
  2026-11-14 it had no Imath 3.2, no oneTBB 2022 and no OpenUSD at all); if `conancenter` points there, run
  `uv run conan remote update conancenter --url="https://center2.conan.io"` and say so in the commit body. Then
  `uv run conan list "zlib/*" -r conancenter` and take the newest released version (the highest version number;
  never a pre-release or a `cci.<date>` reference). Below it is written `<zlib version>`. Record the command, its
  output and the chosen version in the commit body. (For information: on 2026-11-14 the newest was `zlib/1.4`.)
  If no zlib is listed: stop, change nothing, and report to the coordinator "zlib is not a Conan Center package";
  the fallback for the coordinator to rule on is a Conan recipe of ours under `conan/recipes/zlib/` that builds
  the newest release from source (`conan export` before `conan install`).
- [ ] **Step 2: Failing test** (`fluidity_add_test(interchange_libraries_test ... LABEL cpu LINKS
  fluidity_interchange)`): `TEST(InterchangeLibraries,
  ZlibIsThePinnedVersion)`: `std::string_view(zlibVersion()) == "<zlib version>"`.
- [ ] **Step 3: Add the package.** `requires` gains `zlib/<zlib version>`, exactly; regenerate `conan.lock` with
  `uv run conan lock create .`. In `CMakeLists.txt`, `find_package(ZLIB)`, the target, and the exclusion of
  `render_output/interchange/` from `fluidity`'s sources. `AGENTS.md` Language: "Exceptions and RTTI are on only
  in `cli/read_command_line.cc` and in `render_output/interchange/*.cc`; each public function there catches
  every exception and returns a status."
- [ ] **Step 4: Build on macOS.** `conan install` and build both macOS presets; run the test. Expected: PASS.
- [ ] **Step 5: Build in the container.** `uv run python tools/check_commit.py --only cuda`. Expected: `PASS cuda`
  with the new test among the `cpu` tests the container runs. Check where the container keeps Conan packages: if not on the
  `fluidity-cuda-out` volume, set `CONAN_HOME=/out/conan-home` in the gate's `docker run` so later libraries
  (OpenUSD above all) build once, and say so in the commit body. Record the first-build time on each platform.
- [ ] **Step 6: Run** the gate. **Commit.** Subject: `Add zlib and the target that will hold the open-format
  writers`. New names: `fluidity_interchange`,
  `InterchangeLibraries`, `ReadInterchangeLibraries`. Body: the pinned zlib version and the `conan list` output; Windows not built here.

### Task 5: Imath, newest on Conan Center

**Amended 2026-11-14 (A19):** pins the newest Imath release on Conan Center, found in Step 1.

**Kind:** mechanical

**Files:**
- Modify: `conanfile.py`, `conan.lock`, `CMakeLists.txt`, `interchange_libraries_test.cc`

**Interfaces:**
- Produces: `fluidity_interchange` links Imath.

- [ ] **Step 1: Find the newest release and pin it.** With `conancenter` pointing at `https://center2.conan.io`
  (check as Task 4 Step 1 says), `uv run conan list "imath/*" -r conancenter`; take the newest released version,
  written `<imath version>` below; record command, output and choice in the commit body (on 2026-11-14:
  `imath/3.2.3`). If none is listed: stop and report as Task 4 Step 1 says (fallback: our recipe
  `conan/recipes/imath/`). If OpenVDB or Alembic later need a different Imath (Tasks 7 and 8), Conan's conflict
  message says so: then those tasks follow Task 8's override rule.
- [ ] **Step 2: Failing test** `TEST(InterchangeLibraries, ImathIsThePinnedVersion)`:
  `std::string_view(IMATH_VERSION_STRING) == "<imath version>"`, and one `Imath::V3f` sum so the library is used.
- [ ] **Step 3: Add** `imath/<imath version>` to `requires`, regenerate the lock, link. **Step 4:** build and test on
  both macOS presets and `--only cuda`. Expected: PASS.
- [ ] **Step 5: Run** the gate. **Commit.** Subject: `Add Imath for the Alembic and OpenVDB writers`. Body: the
  pinned version and the `conan list` output.

### Task 6: oneTBB, newest on Conan Center

**Amended 2026-11-14 (A19):** pins the newest oneTBB release on Conan Center, found in Step 1.

**Kind:** mechanical

**Files:**
- Modify: `conanfile.py`, `conan.lock`, `CMakeLists.txt`, `interchange_libraries_test.cc`

**Interfaces:**
- Produces: `fluidity_interchange` links oneTBB.

- [ ] **Step 1: Find the newest release and pin it.** With `conancenter` pointing at `https://center2.conan.io`
  (Task 4 Step 1), `uv run conan list "onetbb/*" -r conancenter`; take the newest released version, written
  `<onetbb version>` (major `<M>`, minor `<m>`) below; record command, output and choice in the commit body (on
  2026-11-14: `onetbb/2023.1.0`). If none is listed: stop and report (fallback: our recipe
  `conan/recipes/onetbb/`).
- [ ] **Step 2: Failing test** `TEST(InterchangeLibraries, OneTbbIsThePinnedVersion)`: `TBB_VERSION_MAJOR == <M>`,
  `TBB_VERSION_MINOR == <m>`, and `TBB_runtime_interface_version()` equals `TBB_INTERFACE_VERSION` (the header and
  the linked library agree).
- [ ] **Step 3: Add** `onetbb/<onetbb version>` (shared or static as the recipe allows; if only shared, the CMake copies
  the runtime library beside every interchange test executable and the CLI, and the commit body says so).
  Regenerate the lock. **Step 4:** build and test on macOS and in the container. Expected: PASS.
- [ ] **Step 5: Run** the gate. **Commit.** Subject: `Add oneTBB for the OpenUSD and OpenVDB writers`. Body: the
  pinned version and the `conan list` output.

### Task 7: Alembic, newest on Conan Center

**Amended 2026-11-14 (A19):** pins the newest Alembic release on Conan Center, found in Step 1; HDF5 stays off.

**Kind:** mechanical

**Files:**
- Modify: `conanfile.py`, `conan.lock`, `CMakeLists.txt`, `interchange_libraries.cc`, `interchange_libraries_test.cc`

**Interfaces:**
- Produces: `ReadInterchangeLibraries().alembic == "<alembic version>"`.

- [ ] **Step 1: Find the newest release and pin it.** With `conancenter` pointing at `https://center2.conan.io`
  (Task 4 Step 1), `uv run conan list "alembic/*" -r conancenter`; take the newest released version, written
  `<alembic version>` below; record command, output and choice in the commit body (on 2026-11-14:
  `alembic/1.8.8`). If none is listed: stop and report (fallback: our recipe `conan/recipes/alembic/`). Options:
  HDF5 off (Ogawa only). If the recipe cannot turn HDF5 off, stop and report: Blender reads the Ogawa backend,
  and HDF5 would be a large dependency nothing uses.
- [ ] **Step 2: Failing tests:** `TEST(ReadInterchangeLibraries, AlembicVersion_IsThePin)`;
  `TEST(InterchangeLibraries, AlembicOgawaArchive_ReadsBackOnePoint)`: in a temporary folder, an `OArchive`
  (Ogawa) with one `OPoints` of one point `{1, 2, 3}`, closed, reopened with `IArchive`; the point equals.
- [ ] **Step 3: Add** the package, regenerate the lock, link. **Plan decision:** each version string in
  `interchange_libraries.cc` is a literal guarded by a `static_assert` on the library's own version macro
  (`ALEMBIC_LIBRARY_VERSION` equal to the pin's number, `10808` for 1.8.8), so a different library fails to
  compile instead of reporting a wrong pin.
- [ ] **Step 4:** build and test on macOS and in the container. **Step 5:** gate. **Commit.** Subject:
  `Add Alembic, the default format for points and surfaces`. Body: the pinned version, the `conan list` output
  and the HDF5 option used.

### Task 8: OpenVDB, newest on Conan Center

**Amended 2026-11-14 (A19):** pins the newest OpenVDB release on Conan Center, found in Step 1; Boost and Blosc are
turned off where the recipe allows, otherwise accepted and recorded.

**Kind:** mechanical

**Files:**
- Modify: `conanfile.py`, `conan.lock`, `CMakeLists.txt`, `interchange_libraries.cc`, `interchange_libraries_test.cc`

**Interfaces:**
- Produces: `ReadInterchangeLibraries().openvdb == "<openvdb version>"`.

- [ ] **Step 1: Find the newest release and pin it.** With `conancenter` pointing at `https://center2.conan.io`
  (Task 4 Step 1), `uv run conan list "openvdb/*" -r conancenter`; take the newest released version, written
  `<openvdb version>` below; record command, output and choice in the commit body. (On 2026-11-14 the newest was
  `openvdb/12.1.1`, older than the 13.0 Blender 5.3 bundles; that is fine under A19, and the witness tests
  confirm Blender reads the files.) If none is listed: stop and report (fallback: our recipe
  `conan/recipes/openvdb/`). Options: core library only, no tools, zlib on, Blosc off. Then
  `uv run conan graph info . --format=text` and read the dependencies. Boost or Blosc: if the recipe has an
  option that removes it while the zip-compressed `FloatGrid` files of Step 2 still work, set it and record the
  option; if this newest package cannot be built without it, accept it and record in the commit body what it
  adds (packages, build time, library size, licence). A dependency other than those, zlib, Imath and oneTBB:
  stop and report the list (a new dependency needs the coordinator's ruling). A conflict between this
  package's Imath or oneTBB and the pins of Tasks 5 and 6: use Conan's `override` to the newer of the two and
  record it.
- [ ] **Step 2: Failing tests:** `TEST(ReadInterchangeLibraries, OpenVdbVersion_IsThePin)`;
  `TEST(InterchangeLibraries, OpenVdbZipFile_ReadsBackOneVoxel)`: `openvdb::initialize()`, one `FloatGrid` named
  `known` with voxel `(1, 2, 3) = 1`, file compression zip, written, reopened; value and name equal.
- [ ] **Step 3: Add** the package (guarded version literal as Task 7), regenerate the lock, link.
- [ ] **Step 4:** build and test on macOS and in the container. **Step 5:** gate. **Commit.** Subject: `Add
  OpenVDB for volume files`. Body: the pinned version, the `conan list` output, the options set, and every
  dependency accepted with what it adds.

### Task 9: OpenUSD, newest on Conan Center, and its plugin resources

**Amended 2026-11-14 (A19):** pins the newest OpenUSD release on Conan Center, found in Step 1; a static build that
cannot register its schema types still stops the task.

**Kind:** integration

**Files:**
- Create: `native/fluidity/render_output/interchange/usd_resources.h` (private), `usd_resources.cc`
- Modify: `conanfile.py`, `conan.lock`, `CMakeLists.txt`, `cmake/fluidity_targets.cmake`, `interchange_libraries.cc`,
  `interchange_libraries_test.cc`, `.github/workflows/checks.yml` (Conan cache key includes `conan.lock`)

**Interfaces:**
- Produces: `ReadInterchangeLibraries().open_usd == "<openusd version>"`; **Plan decision:** a private
  `std::expected<void, RenderOutputFailure> CheckUsdResources()` (declared once the error type exists in Task 10;
  in this task it returns the missing folder path as a `std::string_view`, empty on success), which registers
  `<folder of the running executable>/usd_resources` with `pxr::PlugRegistry::RegisterPlugins` once
  (`std::call_once`); `missing_resource_folder` carries that folder when it fails. CMake copies the package's
  plugin resource tree to `usd_resources/` beside the CLI and every interchange test executable.

- [ ] **Step 1: Find the newest release and pin it.** With `conancenter` pointing at `https://center2.conan.io`
  (Task 4 Step 1; the frozen old remote has no OpenUSD at all), `uv run conan list "openusd/*" -r conancenter`;
  take the newest released version, written `<openusd version>` below; record command, output and choice in the
  commit body (on 2026-11-14: `openusd/26.08`, the version Blender 5.3 bundles). If none is listed: stop and
  report (fallback: our recipe `conan/recipes/openusd/` building only `tf`, `gf`, `vt`, `ar`, `sdf`, `pcp`,
  `usd`, `usdGeom` statically, no imaging, no Python, no tools). Options as the design's table. Record the build
  time.
- [ ] **Step 2: Failing tests:** `TEST(ReadInterchangeLibraries, OpenUsdVersion_IsThePin)` (`PXR_VERSION` equal to
  the pin's number, `2608` for 26.08);
  `TEST(InterchangeLibraries, UsdcLayer_ReadsBackOnePoint)`: a layer made with
  `SdfLayer::CreateNew(SdfFileFormat::FindById(TfToken("usdc")), path)` on a path that does **not** end in
  `.usdc` (as `path_while_writing()` does not), a stage on it, one `UsdGeomPoints` with one point, saved; renamed
  to `.usdc`, reopened with `UsdStage::Open`; the point equals. `TEST(WritePointsFile,
  UsdWithoutResources_IsFileWriteFailed)` (Review Focus 5) is written now as `TEST(CheckUsdResources,
  MissingFolder_IsNamed)` (the test executable is run with the resource folder renamed through a CTest fixture);
  Task 40 renames it to the writer once the writer exists.
- [ ] **Step 3: Add** the package, regenerate the lock, link. Static libraries register their types through static
  initialisers: link the USD libraries with `$<LINK_LIBRARY:WHOLE_ARCHIVE,...>` if `UsdGeomPoints` is not found
  at run time. If the static build still cannot register `usdGeom`, stop and report (the coordinator rules on
  shared libraries).
- [ ] **Step 4:** build and test on macOS and in the container (expect the container build to be long; it must be
  cached on the volume of Task 4). **Step 5:** gate. **Commit.** Subject: `Add OpenUSD and ship its plugin
  resources beside the programs that write USD`. Body: the pinned version and the `conan list` output, the
  measured build times and library sizes; Windows not built here; CI not run.

### Task 10: Errors, limits and frame identity

**Amended 2026-11-14 (A31, reconcile item 26):** `RenderLimits::max_field_parts` (default `1u << 26`) and `RenderOutputError::kTooManyFieldParts` (value 15, appended) added.

**Amended 2026-11-15 (A58, reconcile item 31):** `kOutOfMemory` moves to the end (value 15); `kRunOnCallFailed`,
`kFormatNotInBuild`, `kFileWriteFailed`, `kTooManyFieldParts` become 11 to 14. Still 16 values.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/render_output/render_output_error.h` (shared), `render_output_message.h`,
  `render_output_message.cc`, `render_output_error_test.cc`, `render_limits.h` (shared), `frame_identity.h`
  (shared)
- Modify: the CMake lines that list library sources and tests; `docs/glossary.md`; `usd_resources.h/.cc`
  (return type of `CheckUsdResources` as Task 9 says)

**Interfaces:**
- Produces: design 3.1 exactly (`RenderOutputError` with 16 values, `RenderOutputFailure`, `ToMessage`, `ToName`);
  `RenderLimits` (design 3.6, moved here, **Plan decision**); `FrameIdentity` (design 3.6, moved here).

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ToName, EveryError_LowerSnakeCase)        // all 16; kFormatNotInBuild -> "format_not_in_build",
                                                 // kTooManyLatticeBlocks -> "too_many_lattice_blocks"; distinct
  TEST(ToMessage, EveryError_IsDistinctAndNotEmpty)
  TEST(RenderOutputError, NumbersNeverChange)    // static_assert kNone == 0 ... kEdgeTooLong == 10,
                                                 // kRunOnCallFailed == 11, kFormatNotInBuild == 12,
                                                 // kFileWriteFailed == 13, kTooManyFieldParts == 14,
                                                 // kOutOfMemory == 15 (last; A58, reconcile item 31)
  TEST(RenderLimits, Defaults)                   // 1 << 20, 1 << 26, 1 << 24, max_field_parts 1 << 26
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** Messages say what the user can do ("The surface needs more
  vertices than max_vertices allows. Raise the limit or coarsen the surface spacing.").
- [ ] **Step 4: Glossary rows** (`| Say | Plain meaning | Never say |`), design decision 1 table, every row:
  lattice, lattice point (never `voxel grid`, `node`); foam particle (never `whitewater point`, `diffuse
  particle`); `FoamSourceEvent` (never `birth`, `emission event`); `strength` (never `potential`); the three
  measures; `owed_particles` (never `remainder`); `count_per_particle` (never `represented count`);
  `OutputInterval`; `MotionSample` (never `carrier`, `stamp`); `StepRecorder`, record (never `capture`,
  `observer`); `SurfaceMesh` (never `isosurface`); piece (never `component`, `shell`); `SurfaceField`,
  `smoothing_width` (never `density field`, `sigma`, `kernel`); move a mesh (never `track`, `tracker`); pull back
  (never `project`); `kBuiltFromField` (never `fallback`, `re-extraction`); safe (never `valid`, `watertight`,
  `manifold`); the six defects; `known` (never `validity`, `mask` alone); measured, reconstructed, display only;
  witness, witness test, format table, default of a kind (never `fallback`).
- [ ] **Step 5: Run** the gate. **Commit.** Subject: `Add render output's errors, limits and frame identity`.
  New names: `RenderOutputError` and its 16 values, `RenderOutputFailure`, `ToMessage`, `ToName`, `RenderLimits`,
  `FrameIdentity`.

### Task 11: The output channel table

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/render_output/output_channels.h`, `output_channels.cc`, `output_channels_test.cc`,
  `format_table.h` (this task: `InterchangeFormat` only)

**Interfaces:**
- Produces: design 3.6 `output_channels.h` exactly (`OutputProduct`, `GeometryKind`, `ToGeometryKind`,
  `ChannelDomain`, `ChannelKind`, `ChannelSpace`, `ChannelSpread`, `ChannelValueType`, `WitnessProperty`,
  `OutputChannel`, `kOutputContractVersion`, `ReadOutputChannels`); `InterchangeFormat` of design 3.6, alone in
  `format_table.h` until Task 43. **Plan decision** for the row "a field of `RequestedOutputs::particle_fields`":
  one row per exportable `particles::Field` other than the three always written: `mass` (float32, 1 component),
  `fitted_velocity_change` (9), `springy_deformation` (9), `permanent_volume_change` (1), `flags` (int32); a
  request for `kPlateSide` or `kVelocityForThisStep` is `kBadInput` in `ToPointObject`. Rows whose Blender name
  holds `<ingredient_export_name>` are patterns, filled per ingredient by the writers.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ReadOutputChannels, RowsInTableOrder)            // the design 5.7 table, row by row (product, blender_name,
                                                        // names per format, unit, storage, kind, witnessed_by)
  TEST(ReadOutputChannels, BlenderNamesUniquePerProduct)
  TEST(ToGeometryKind, EveryProduct)                    // droplets, spray, foam, bubbles, particles: kPoints;
                                                        // surface: kSurfaceMesh; both volumes: kVolume; solid motion
  TEST(ReadOutputChannels, OnlyFloat32AndInt32ForPointsAndSurface)  // kFloat64 only for kObject rows
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** the table as `constexpr` data in `output_channels.cc`, in the
  order of design 5.7. **Step 4:** gate. **Commit.** Subject: `Add the output channel table every writer reads`.
  New names: the enums and their values, `OutputChannel`, `kOutputContractVersion`, `ReadOutputChannels`,
  `InterchangeFormat`.

### Task 12: Interchange test support and points of a particles material

**Amended 2026-11-14 (A31):** `PointScratch` renamed `PointObjectBuffers` (parameter `point_object_buffers`); test `BuffersReused_NoNewAllocationSecondTime`.

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** `fluidity_interchange_test_support` is made with `fluidity_add_test_library` and linked through
`LINKS`; tests `LABEL cpu`.

**Files:**
- Create: `native/fluidity/render_output/interchange/point_objects.h`, `point_objects.cc`, `point_objects_test.cc`,
  `interchange_test_support.h`, `interchange_test_support.cc` (test-only library
  `fluidity_interchange_test_support`, never linked by `fluidity_interchange` or the CLI)
- Modify: `CMakeLists.txt` (`fluidity_add_test_library(fluidity_interchange_test_support SOURCES
  interchange_test_support.cc LINKS fluidity_interchange fluidity_base_test_support)`;
  `point_objects_test` `LABEL cpu LINKS fluidity_interchange_test_support`)

**Interfaces:**
- Consumes: `particles::ParticlesSnapshot::Material`, `particles::FieldList`, `HasField`; I/O's
  `BuildNewRunFolder`, `WriteOutputDeclarations`, `BuildOutputFile`, `WriteListedFile`.
- Produces: `PointChannelValues`, `PointObject`, `PointObjectBuffers`, and the first `ToPointObject` (particles) of
  design 3.6. **Plan decisions:** `PointObjectBuffers` is complete in `point_objects.h` (callers own one): float and
  int32 vectors reused from frame to frame. `PointChannelValues` gains `std::string_view ingredient_export_name;
  // Pattern rows only: the name that replaces <ingredient_export_name>.` Test support, namespace
  `fluidity::render_output::testing`:

  ```cpp
  inline constexpr FrameIdentity kTestIdentity{.frame = 42, .state_version = 7, .time = 1.75,
                                               .frames_per_second = 24};
  class TestRunFolder;   // A new run folder in a fresh temporary directory, removed on destruction. Move-only.
  TestRunFolder BuildTestRunFolder(std::string_view test_name);
  // Declares output_id (if new) with the product and format, then BuildOutputFile for `frame`. Extensions:
  // .abc, .usdc, .vdb, .json.
  io::OutputFile BuildTestOutputFile(TestRunFolder& folder, std::string_view output_id, OutputProduct product,
                                     InterchangeFormat format, std::uint64_t frame);
  // WriteListedFile with kSizeOnly; returns the absolute path of the listed file, which read-back tests open.
  std::filesystem::path WriteTestListedFile(TestRunFolder& folder, io::OutputFile&& file);
  // One material: 1000 particles, two ingredients ("water", "salt"), mass carried, ids from 2^31 - 3 upward.
  particles::ParticlesSnapshot::Material BuildTestParticlesMaterial(std::uint64_t material_id);
  ```

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ToPointObject, ParticlesMaterial_RadiusFromVolume)   // radius == cbrt(3 rest_volume volume_ratio / (4 pi)),
                                                            // 1e-6 relative (float)
  TEST(ToPointObject, TwoIngredients_TwoShareChannels)      // "share_water", "share_salt", point-major
  TEST(ToPointObject, MassRequested_IsAChannel)
  TEST(ToPointObject, FieldNotCarried_IsBadInput)           // kMass requested, mass array empty
  TEST(ToPointObject, BuffersReused_NoNewAllocationSecondTime)   // same sizes twice: allocation count unchanged
  TEST(BuildTestOutputFile, ListedFileExists)               // the helper's own check
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** Channels in `ReadOutputChannels()` order. **Step 4:** gate.
  **Commit.** Subject: `Turn a particles snapshot into points for the writers`. New names: `PointChannelValues`,
  `PointObject`, `PointObjectBuffers`, `ToPointObject`.

### Task 13: Points to Alembic

**Kind:** integration

**Files:**
- Create: `native/fluidity/render_output/interchange/write_points_file.h`, `write_points_alembic.cc` (private
  body), `write_points_file.cc`, `write_points_file_test.cc`, `alembic_frame.h` and `alembic_frame.cc` (private:
  archive with metadata and time sampling, the `(x, z, -y)` turn)

**Interfaces:**
- Consumes: `PointObject`, `FrameIdentity`, `io::OutputFile::path_while_writing()`.
- Produces: `WritePointsFile` of design 3.6. In this task `kUsd` returns `kFormatNotInBuild` (Task 40 adds it);
  any format other than `kUsd` and `kAlembic` always returns `kFormatNotInBuild`. **Plan decisions:** per object
  one `OPoints` named `object_name` with `P`, `.velocities`, `.widths` (diameter, always per point; a constant
  radius is expanded), `.pointIds` (the full 64-bit id, as the schema requires), and the `arbGeomParams` `id`
  (int32 low 31 bits) and every channel (float with `components` per point, or int32), `kVaryingScope`. An empty
  object is an `OPoints` with zero points.

- [ ] **Step 1: Failing tests** (part of design test `WritePointsFile, ReadBack`; its USD part is in Task 40):

  ```cpp
  TEST(WritePointsFile, AlembicParticles_ReadBack)     // BuildTestParticlesMaterial with mass, through
      // ToPointObject: positions turned back by (x, -z, y) equal the input as float; velocities; widths == 2 r;
      // int32 id; mass; two shares; metadata keys; one time sample at 42 / 24
  TEST(WritePointsFile, AlembicDropletsAndFoam_ReadBack)  // hand-made PointObjects "droplets", "spray", "foam",
      // "bubbles" (foam empty): four objects, counts 5, 3, 0, 2, every value equal
  TEST(WritePointsFile, IdAbove31Bits_LowBitsAndFullId) // Review Focus 2: id 2^31 + 5 -> int32 5, .pointIds full
  TEST(WritePointsFile, SpansOfDifferentCounts_IsBadInput)  // velocity count != position count; nothing listed
  TEST(WritePointsFile, OpenVdbFormat_IsFormatNotInBuild)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement**; the public function catches every exception (Global
  Constraints). **Step 4:** gate. **Commit.** Subject: `Write points as one Alembic file per frame`. New names:
  `WritePointsFile`.

### Task 14: The lattice

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the shared-header test is `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Create: `native/fluidity/render_output/lattice.h` (shared), `lattice_test.cc` (`LABEL cpu`),
  `lattice_values.h`, `lattice_values.cc`, `lattice_values_test.cc`

**Interfaces:**
- Produces: design 3.2 `lattice.h`, split (**Plan decision**): `lattice.h` shared holds `LatticeFrame`,
  `LatticeCoord`, `kLatticeBlockWidth`, `ToBlockKey`, `ComputePointPosition`, `FindLowPoint`; `lattice_values.h`
  (CPU only) holds `LatticeBlocks`, `LatticeValues`, `FindBlock`, `LatticeReading`, `ComputeLatticeReading`.
  **Plan decision:** `FindBlock` returns `std::numeric_limits<std::size_t>::max()` when absent, named
  `kNoBlock` in `lattice_values.h`.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ToBlockKey, NegativeAndPositiveBlocks_OrderZThenYThenX)  // key order == lexicographic (z, y, x)
  TEST(ToBlockKey, ExtremeCoords_NoOverlap)                     // -2^20 and 2^20 - 1 per axis
  TEST(FindLowPoint, PointExactlyOnALatticePoint_IsThatPoint)   // and -0.5 spacing -> -1 (math::Floor)
  TEST(ComputeLatticeReading, LinearField_Exact)                // design 7: 1,000 positions in 3 blocks, field
      // 2 + 3x - y + 0.5z: value and value_change, 1e-6 relative (float storage)
  TEST(ComputeLatticeReading, MissingPoint_IsNotKnown)          // one of the 8 not known: is_known false, value 0
  TEST(ComputeLatticeReading, PositionOnBlockBoundary_ReadsNextBlock)  // x on a block's last point + spacing/2
  TEST(FindBlock, AbsentKey_IsNoBlock)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Add the lattice,
  points at equal spacing kept in blocks`. New names: the design 3.2 lattice names, `kNoBlock`.

### Task 15: Volume frames and the gas volume

**Kind:** numerics

**Files:**
- Create: `native/fluidity/render_output/volume_frame.h`, `build_gas_volume.cc`, `build_gas_volume_test.cc`

**Interfaces:**
- Consumes: `gas::GasSnapshot`, `grid::GridFrame`, `grid::ToCellId`, `ComputeCellWidth`, `RenderLimits`.
- Produces: `VolumeChannel`, `VolumeFrame`, `BuildGasVolume` of design 3.6 (`BuildFoamVolume` in Task 37). Method
  design 5.6 item 2: binary search of the point's cell by `CellId` over the size levels present; `known = 1`
  exactly where such a cell exists with `gas_volume_share > 0`; channels `kGasDensity`, `kSmokeDensity`,
  `kTemperature`, `kFuelDensity`, `kFlame`, ascending; blocks only where a point is known; more blocks than
  `max_lattice_blocks`: `kTooManyLatticeBlocks` (the caller reports the product not built).

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(BuildGasVolume, ThreeStates)            // design 7 (first two states; the third, no gas in a frame, is
      // Task 45's BuildRenderFrame, NoGas_NoGasVolume): gas cells with zero smoke -> known 1, smoke 0;
      // space with no gas cell -> known 0, value stored 0
  TEST(BuildGasVolume, MixedSizes_KeepTheSum)  // cells of two sizes, default spacing: sum value * spacing^3 over
      // known == sum value * cell volume, 1e-6 (float)
  TEST(BuildGasVolume, LiquidCell_IsNotKnown)  // a cell with gas_volume_share 0
  TEST(BuildGasVolume, TooManyBlocks_IsTooManyLatticeBlocks)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Count the gas on a
  lattice with known kept apart from zero`. New names: `VolumeChannel` and values, `VolumeFrame`, `BuildGasVolume`.

### Task 16: Volumes to OpenVDB

**Kind:** integration

**Files:**
- Create: `native/fluidity/render_output/interchange/write_volume_file.h`, `write_volume_file.cc`,
  `write_volume_file_test.cc`

**Interfaces:**
- Produces: `WriteVolumeFile` of design 3.6. One `FloatGrid` per listed channel named by its `openvdb_name`, plus
  `known`; only nonzero voxels active; transform `createLinearTransform(spacing)` then translated by the lattice
  origin, so voxel `(i, j, k)` is centred at lattice point `(i, j, k)`; zip compression; per-grid metadata
  `fluidity_unit` from the channel row (**Plan decision**); file metadata as Global Constraints.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(WriteVolumeFile, ReadBack)          // design 7: a hand-made foam VolumeFrame (three channels) and a gas one
      // (five): every grid, known, voxel size, placement (world position of voxel (3, -2, 5)), names, units,
      // contract version; values equal as float; zero values not active
  TEST(WriteVolumeFile, PathInMissingFolder_IsFileWriteFailed)  // Review Focus 4: the temporary file's folder
      // removed before the call; returns kFileWriteFailed, no exception escapes
  TEST(WriteVolumeFile, ChannelCountDiffersFromValues_IsBadInput)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Write volumes as one
  OpenVDB file per frame`. New names: `WriteVolumeFile`.

### Task 17: Surface settings and field points

**Kind:** numerics

**Amended 2026-11-15 (A64):** no grid-only cells in the first version: `BuildFieldPoints` reads particles only; the
grid-only point and its test `BuildFieldPoints, GridOnlyCell_IsOnePointAtItsCentre` move to Task 50 of the later
group; the commit subject drops "and grid-only liquid".

**Amended 2026-11-15 (A65):** particles are sized to their cell (8 per cell, rest volume `(w/2)³`), so a particle `n` levels coarser is `2^n` spacings wide; the split becomes `k = min(ceil(width / spacing), max(8, ceil(width / (2 spacing))))` (parts at most two spacings apart); `VeryWidePoint_SplitsIntoAtMost512` is replaced by `VeryWidePoint_PartsAtMostTwoSpacingsApart`.

**Amended 2026-11-14 (Foundation as built):** `fluidity_render_output_test_support` is made with `fluidity_add_test_library`; the stable order of
the field parts built its own working memory per call (CPU, once per frame); replaced by the A31 amendment below.

**Amended 2026-11-14 (A31, reconcile item 26):** one form: `ComputeAddedVolume` takes the caller's CPU
`OrderBuffers& order_buffers` (`RenderOutput::frame_order_buffers` in a run, built locally in the tests); the field
parts are bounded by the new `RenderLimits::max_field_parts` (default `1u << 26`), above which the group is not
built (`kTooManyFieldParts`); test `ComputeAddedVolume, TooManyParts_IsTooManyFieldParts` added.

**Files:**
- Create: `native/fluidity/surface/surface_settings.h`, `surface_field.h` (types only in this task),
  `surface_field_steps.h` (private), `surface_field_points.cc`, `surface_field_points_test.cc`,
  `native/fluidity/render_output/render_output_test_support.h`, `render_output_test_support.cc` (test-only library
  `fluidity_render_output_test_support`)
- Modify: `native/fluidity/surface/CMakeLists.txt`, `native/fluidity/render_output/CMakeLists.txt`
  (`fluidity_add_test_library(fluidity_render_output_test_support SOURCES render_output_test_support.cc LINKS
  fluidity_step_test_support fluidity_transfer_test_scenes fluidity_pressure_test_scenes fluidity_base_test_support)`;
  tests `LABEL cpu LINKS fluidity_render_output_test_support`)

**Interfaces:**
- Consumes: `step::SimulationSnapshot` (`particles`, `active_cells`; ~~`is_grid_only_region`,
  `grid_only_current_volume`, `grid_only_ingredient_volume`~~, A64: Task 50), grid 3 cell centre and width from `CellId`.
- Produces: `SurfaceGroup`, `SurfaceSettings`, `FieldPoint`, `SurfaceField`, `SurfaceFieldInputs` of design 3.4.
  Private (**Plan decision**): `BuildFieldPoints(const SurfaceFieldInputs&) -> std::vector<FieldPoint>` (5.3 steps
  1 and 2: particles with `volume = rest_volume * volume_ratio`, `width = cbrt(volume)`; ~~grid-only cells as one
  point at the centre;~~ (A64: Task 50) points wider than `spacing` split into `k^3` parts, `k = min(ceil(width / spacing), max(8, ceil(width / (2 *
  spacing))))` (A65));
  `ComputeAddedVolume(points, frame, limits, OrderBuffers& order_buffers) -> std::expected<AddedVolume,
  RenderOutputFailure>` (step 3: blocks and trilinear sums in the order of Global Constraints; amended: the parts
  ordered by `BuildStableOrder` with `order_buffers`, allocating nothing; more parts than `limits.max_field_parts`
  is `kTooManyFieldParts` with `id` the material group, checked before the order; `order_buffers.max_count()`
  below the part count is `kBadInput`; Requires: `order_buffers.run_on() == RunOn::kCpu`, debug assertion). Test support, namespace `fluidity::render_output::testing`:
  `BuildSnapshotWithParticles(std::uint64_t material_id, std::span<const Vec3> positions, double rest_volume)`,
  `BuildSlabPositions(double top, double spacing_between, Vec3 low, Vec3 high)`,
  `BuildSpherePositions(Vec3 centre, double radius, double spacing_between)`,
  `BuildRandomPositions(std::uint64_t seed, int count, Vec3 low, Vec3 high)` (as `transfer::test::BuildCloud`).

- [ ] **Step 1: Failing tests:**

  ```cpp
  ~~TEST(BuildFieldPoints, GridOnlyCell_IsOnePointAtItsCentre)~~   // A64: moved to Task 50 (later)
  TEST(BuildFieldPoints, WidePoint_SplitsIntoKCubedParts)      // width 2.5 spacings: 27 parts, volumes sum exactly
  ~~TEST(BuildFieldPoints, VeryWidePoint_SplitsIntoAtMost512)~~  // A65: replaced by the next test
  TEST(BuildFieldPoints, VeryWidePoint_PartsAtMostTwoSpacingsApart)  // A65: width 8 spacings: k == 8; 8.5: 8; 16: 8;
      // 20: 10; 64: 32; part volumes sum to the point's volume within 1e-12 relative
  TEST(ComputeAddedVolume, SumEqualsPointVolume)               // 10,000 random points: 1e-12 relative
  TEST(ComputeAddedVolume, SameInputTwice_SameBytes)
  TEST(ComputeAddedVolume, TooManyParts_IsTooManyFieldParts)   // max_field_parts 26, a point split into 27 parts
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Spread particles
  onto the surface lattice` (A64; was `... and grid-only liquid ...`). New names: the design 3.4 settings and field types.

### Task 18: The surface field

**Amended 2026-11-14 (A31):** `BuildSurfaceField` takes `OrderBuffers& order_buffers`.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/surface/build_surface_field.cc`, `build_surface_field_test.cc`,
  `native/fluidity/surface/testdata/surface_field_main.json`

**Interfaces:**
- Consumes: Task 17's private functions; `contact::ComputeDistanceToSolid`, `ComputeSolidPose`, `DomainBox`.
- Produces (amended 2026-11-14, A31: `BuildSurfaceField` takes `OrderBuffers& order_buffers` last and passes it to
  `ComputeAddedVolume`): `BuildSurfaceField` of design 3.4, 5.3 "Field" steps 4 to 7 and `<eq:surface-field>`: separable
  Gaussian of `smoothing_width` cut at `round(4 * smoothing_width / spacing)`; `open_share = clamp(0.5 +
  distance_to_solid / spacing, 0, 1)` over all solids and the domain sides; `value = 0.5 - liquid / open` where
  `open >= 0.05`, else `+0.5`; then `value = max(value, -clamp(distance_to_solid / spacing, -0.5, 0.5))`; shares the
  same sum per ingredient over `liquid`. Blocks: within the smoothing reach of any liquid.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(BuildSurfaceField, VolumeIsKept)              // design 7: 10,000 random particles, no solid:
      // sum liquid * spacing^3 == sum volume before and after G, 1e-12 relative
  TEST(BuildSurfaceField, FlatLiquid_ZeroSetAtTheTop)  // slab, 2 particles per cell width, top 0.3 spacing above
      // a lattice plane: zero set height (linear between lattice points) within 0.05 spacing
  TEST(BuildSurfaceField, NoOpenSpace_IsAir)         // a point deep inside a solid: value +0.5
  TEST(BuildSurfaceField, TooManyBlocks_IsTooManyLatticeBlocks)
  TEST(BuildSurfaceField, SameAsMainWithoutSolid)    // main's BuildSurfaceDensity3D on the same particles turned
      // by (x, y, z) -> (x, -z, y): every value within 1e-6 (float against double)
  ```
  The `main` numbers: export `main` as plan-decisions says, write a throwaway driver that calls
  `BuildSurfaceDensity3D` (`native/src/surface_field_3d.cc:20-177`) on 200 particles of a fixed formula, rotate
  the lattice values back to Z-up, commit only the numbers as `testdata/surface_field_main.json` with a note
  saying how they were made. Never the only test of the task.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Build the smoothed
  liquid share that the surface follows`. New names: `BuildSurfaceField`.

### Task 19: Mesh checks

**Kind:** numerics

**Files:**
- Create: `native/fluidity/surface/surface_mesh.h` (types only in this task), `check_surface_mesh.h`,
  `check_surface_mesh.cc`, `check_surface_mesh_test.cc`

**Interfaces:**
- Produces: `SurfaceVertex`, `TriangleKind`, `SurfaceMesh` (design 3.4; `sizeof(SurfaceVertex)` asserted, not 64),
  `SurfaceMeshDefects`, `SurfaceMeshMeasure`, `CheckedMesh`, `CheckSurfaceMesh`. Ported from
  `git show main:native/src/surface_mesh_3d.cc` lines 563-703 (`MeasureSurfaceMesh3D`): signed volume, area,
  pieces, Euler number, open edges, overused edges, pinched vertices, wrongly turned edges, flat triangles,
  crossing pairs by a sweep over triangle boxes ordered by (low x, index), pairs reported in that order.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(CheckSurfaceMesh, EachDefect_IsFound)   // design 7, one test with six cases (SCOPED_TRACE per case):
      // one removed triangle -> open 3; one turned triangle -> wrongly turned > 0; an edge with three
      // triangles -> overused 1; two overlapping spheres -> crossing > 0; one zero-area triangle -> flat 1;
      // two cones tip to tip -> pinched 1; is_safe false each
  TEST(CheckSurfaceMesh, Octahedron_SafeWithExactMeasure)   // volume 4/3, Euler 2, pieces 1
  TEST(CheckSurfaceMesh, SweepEqualsAllPairs)  // 200 random meshes under 2,000 triangles: crossing pairs equal a
                                               // brute-force all-pairs test written in the test file
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Check a surface mesh
  for the six defects`. New names: the mesh types and `CheckSurfaceMesh`.

### Task 20: The mesh from the field

**Kind:** numerics

**Files:**
- Create: `native/fluidity/surface/build_surface_mesh.cc`, `build_surface_mesh_test.cc`,
  `native/fluidity/surface/testdata/surface_mesh_main.json`; test support gains `BuildFieldFromFunction`
  (lattice of `spacing` over a box, value from a function of position)

**Interfaces:**
- Produces: `BuildSurfaceMesh` of design 3.4: six tetrahedra per cube with one diagonal rule; tie rule `+1e-8`; a
  vertex per crossed lattice edge, shared through the edge key; crossings never merged into a corner; triangles
  counter-clockwise from the positive side; vertices ordered by edge key, triangles by (cube, tetrahedron); ids
  `1, 2, ...` in vertex order and `next_id` after them; `triangle_kind` `kLiquidToSolid` where the triangle lies on
  the cut of step 6 of the field (all three vertices with `distance_to_solid <= 0.25 spacing`), else
  `kLiquidToAir`. Ported from `surface_mesh_3d.cc:153-327` without the merge at `1e-7`.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(BuildSurfaceMesh, RandomFields_AlwaysSafe)  // design 7: 1,000 smooth fields and 1,000 noise fields with
      // values at exactly 0 included: is_safe, flat_triangle_count 0
  TEST(BuildSurfaceMesh, Sphere)                   // radius 8 spacings: volume within 1 %, Euler 2, 1 piece,
  TEST(BuildSurfaceMesh, Torus)                    //   normals within 3 degrees; torus Euler 0, 1 piece;
  TEST(BuildSurfaceMesh, TwoSpheres)               //   two spheres Euler 4, 2 pieces
  TEST(BuildSurfaceMesh, TooManyVertices_IsTooManyVertices)
  TEST(BuildSurfaceMesh, SameAsMainAwayFromCorners)  // main's ExtractSurfaceMesh3D on a sphere field whose
      // crossings are away from corners: same triangle count, volume within 1e-9 (testdata as Task 18)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Build a surface mesh
  that is safe by construction`. New names: `BuildSurfaceMesh`.

### Task 21: Several materials in one group

**Kind:** numerics

**Files:**
- Modify: `build_surface_field.cc`, `build_surface_mesh.cc`, `build_surface_mesh_test.cc`

**Interfaces:**
- Produces: design 5.3 "Several materials in a group": `triangle_material_index` (material of largest share at the
  triangle's centre); for exactly two materials the inner sheet, zero set of `share_a - share_b` inside the
  liquid, same tetrahedra, seam vertices shared, `kBetweenTwoMaterials` (ported from
  `surface_mesh_3d.cc:328-562`, `ExtractSurfacePartition3D`); three or more: labels and shares only;
  `ingredient_share` per vertex from the field.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(BuildSurfaceMesh, TwoMaterials_InnerSheetSharesSeamVertices)  // two half-spheres: every inner-sheet vertex
      // on the outer mesh's seam is the same vertex index; mesh with the sheet still has no open edge on the outer
  TEST(BuildSurfaceMesh, ThreeMaterials_LabelsWithoutSheet)          // no kBetweenTwoMaterials triangle
  TEST(BuildSurfaceMesh, OneMaterial_NoLabels)                       // triangle_material_index empty
  TEST(BuildSurfaceMesh, TwoIngredients_SharesSumToOne)              // per vertex, 1e-6
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Label triangles by
  material and build the sheet between two liquids`.

### Task 22: Field and mesh beside solids, big cells and size changes

**Amended 2026-11-15 (A64):** `GridOnlyInside_NoInnerSurface` moves to Task 50 (later); in its place
`BigCellsInside_NoInnerSurface` (big cells holding particles of their own level); the subject drops "grid-only
liquid".

**Amended 2026-11-15 (A65):** new test `ParticlesSixLevelsCoarser_NoInnerSurface` (particles sized to a cell 128 spacings wide).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/surface/surface_field_scene_test.cc`

**Interfaces:**
- Consumes: `contact::FlatSolid`, `contact::Box`, ~~`step::SimulationSnapshot` region arrays~~ (A64: none).

- [ ] **Step 1: Tests** (they should pass on the code of Tasks 18 to 21; any failure is fixed in the commit that
  owns the code):

  ```cpp
  TEST(BuildSurfaceField, BesideASolid_SameHeight)  // design 7: slab against a FlatSolid, and a tilted Box moving at
      // 1 m/s: height at the solid == height far away, kLiquidToSolid triangles within 0.05 spacing of the solid
  ~~TEST(BuildSurfaceField, GridOnlyInside_NoInnerSurface)~~  // A64: moved to Task 50 (later)
  TEST(BuildSurfaceField, BigCellsInside_NoInnerSurface)  // A64, design 7: sphere radius 16 smallest cells; inside
      // the outer 3 cells, particles of 2 levels coarser (rest_volume 64 times, 8 per big cell on
      // ComputeRefillParticle's pattern via BuildSpherePositions at 4 times the spacing): 1 piece, Euler number 2,
      // exact
  TEST(BuildSurfaceField, ParticlesSixLevelsCoarser_NoInnerSurface)  // A65: one cell 128 spacings wide, its 8
      // particles of width 64 spacings at the cell-eighth centres, no solid: k == 32 each; every lattice point at
      // least 8 spacings inside the cell has value < -0.4; mesh 1 piece, Euler number 2
  TEST(BuildSurfaceField, BigAndSmallParticles_NoGap)     // slab across a size change: 1 piece, height step under
                                                          // 0.1 spacing
  ```
- [ ] **Step 2: Run**; **Step 3:** gate. **Commit.** Subject: `Show the surface beside solids and size changes`.

### Task 23: The motion sample

**Kind:** numerics

**Files:**
- Create: `native/fluidity/render_output/motion_sample.h`, `motion_sample.cc`, `motion_sample_test.cc`; test support
  gains `BuildMotionSampleFromFunction(LatticeFrame, Vec3 low, Vec3 high, double time, Vec3 (*velocity)(Vec3,
  double, const void*), double (*distance)(Vec3, const void*), const void* context)`

**Interfaces:**
- Produces: `MotionSample`, `OutputInterval`, `LiquidMotion`, `ComputeLiquidMotion`, `MovedPoint`,
  `ComputeMovedPoint`, `kMaxMovesPerPoint` of design 3.2. Ported from `motion_interval_3d.cc:85-214`
  (`AdvectMotionPoint3D` at `:124`): midpoint rule, moves at most half a spacing, split at sample times, linear in
  time between two samples, never extended outside the interval or the known points.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ComputeMovedPoint, RigidRotation)             // design 7: u = w x r, quarter turn in one interval, 2 samples:
                                                     // end position within 0.01 spacing
  TEST(ComputeLiquidMotion, BetweenTwoSamples_LinearInTime)   // exact for u = (t, 0, 0)
  TEST(ComputeLiquidMotion, OutsideTheInterval_IsNotKnown)
  TEST(ComputeMovedPoint, LeavesKnownPoints_StopsAndSaysSo)   // motion_was_known false, position where it stopped
  TEST(ComputeMovedPoint, TooManyMoves_StopsAt4096)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Follow the liquid
  between two output frames from motion samples`. New names: design 3.2 motion names.

### Task 24: Moving the last mesh

**Kind:** numerics

**Files:**
- Create: `native/fluidity/surface/move_surface_mesh.h`, `move_surface_mesh.cc`, `move_surface_mesh_test.cc`

**Interfaces:**
- Produces: `MoveSurfaceMesh` of design 3.4, 5.3 step 2a to 2d plus the safety check of the whole moved mesh
  (`kMeshNotSafe`): `ComputeMovedPoint` per vertex; pull back with the Global Constraints numbers (ported from
  `surface_tracking_3d.cc:33-196`, not the sweeps and swaps of `:197-406`); edge length; per closed piece, count
  and Euler number against the field's blob (labels as 5.3 step 1, computed here by a private
  `FindBlobs(const SurfaceField&)` in `surface_field_steps.h`, **Plan decision**); ages `+=` interval duration;
  `liquid_velocity` from the last sample. The failure's `id` is the first vertex (lowest id) or block key that
  failed. Changes nothing on failure.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(MoveSurfaceMesh, UniformMotion_KeepsIds)      // design 7: sphere, u = (1, 0, 0), 100 frames, field built
      // from the moved particles each frame: all ids kept, liquid_velocity == u within 1e-6, volume drift < 0.5 %
  TEST(MoveSurfaceMesh, MotionNotKnown_IsMotionNotKnown)
  TEST(MoveSurfaceMesh, FieldMovedTooFar_IsPullBackFailed)    // field shifted by 0.5 spacing more than the motion
  TEST(MoveSurfaceMesh, Stretch_IsEdgeTooLong)
  TEST(MoveSurfaceMesh, FieldSplitInTwo_IsTopologyChanged)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Move last frame's mesh
  and reject it when it no longer fits`. New names: `MoveSurfaceMesh`.

### Task 25: The volume offset

**Kind:** numerics

**Files:**
- Create: `native/fluidity/surface/volume_offset.h`, `volume_offset.cc`, `volume_offset_test.cc`

**Interfaces:**
- Produces: `BlobVolume`, `ApplyVolumeOffset` of design 3.4, 5.3 step 5 (ported idea from
  `surface_mesh_3d.cc:704-815`, never a failure): vertices not at a solid move along their normals, at most 3
  passes of `offset += (material_volume - mesh_volume) / free_area`, total limited to `0.25 * spacing`; a pass that
  makes `CheckSurfaceMesh` unsafe is halved once, then dropped.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ApplyVolumeOffset, ReachesTarget)   // design 7: sphere 2 % short: within 0.5 %, mesh safe, 1e-9
  TEST(ApplyVolumeOffset, StopsAtLimit)    // target 50 % above: offset == 0.25 spacing, offset_is_at_limit, safe
  TEST(ApplyVolumeOffset, VerticesAtASolid_DoNotMove)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Bring each blob's mesh
  volume toward its liquid volume`. New names: `BlobVolume`, `ApplyVolumeOffset`.

### Task 26: The frame strategy

**Amended 2026-11-14 (A31):** `BuildSurfaceFrame` takes `OrderBuffers& order_buffers`.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/surface/surface_frame.h`, `build_surface_frame.cc`, `build_surface_frame_test.cc`

**Interfaces:**
- Produces (amended 2026-11-14, A31: `BuildSurfaceFrame` takes `OrderBuffers& order_buffers` last, for
  `BuildSurfaceField`; `kTooManyFieldParts` is a not-built reason like `kTooManyVertices`): `SurfaceOrigin`,
  `Droplet`, `SurfaceReport`, `SurfaceFrame`, `BuildSurfaceFrame` of design 3.4, steps 1
  to 4 and 7 of 5.3 "Frame strategy" (steps 5 and 6 in Task 27, which completes this function): per blob the moved
  pieces when not rejected, else the built ones; then `CheckSurfaceMesh` on the whole and, if unsafe, `built` for
  every blob; new vertices get new ids in vertex order and the nearest moved old vertex's age within one spacing
  (lowest id on a tie), else 0; `report.origin` the weakest origin of any blob; `moved_mesh_reject_reason` the first
  reject by blob index. `material_group` is the group's lowest material id.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(BuildSurfaceFrame, TwoSpheresMerge_ThatBlobIsBuilt)   // design 7: at the merge frame the merged blob is
      // kBuiltFromField, the far third sphere keeps every id, mesh safe
  TEST(BuildSurfaceFrame, StretchedSheet_IsBuilt)            // u = (x, 0, -z): reason kEdgeTooLong, mesh safe
  TEST(BuildSurfaceFrame, RandomClouds_NeverFailForAMeshReason)  // 1,000 random particle sets and motions: a safe
      // mesh, or kNotBuilt with kTooManyVertices or kTooManyLatticeBlocks; never an error return
  TEST(BuildSurfaceFrame, NoLastMesh_BuiltFromField)
  TEST(BuildSurfaceFrame, KeepsIdsSwitchedOff_BuiltEveryFrame)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Choose each blob's
  mesh: moved when it fits, built otherwise`. New names: the design 3.4 frame names.

### Task 27: Droplets and volumes of a surface frame

**Kind:** numerics

**Files:**
- Modify: `build_surface_frame.cc`, `build_surface_frame_test.cc`, `point_objects.h/.cc`, `point_objects_test.cc`

**Interfaces:**
- Produces: 5.3 steps 5 and 6 inside `BuildSurfaceFrame`: per blob `material_volume` (field points with `value <
  0` in that blob), `ApplyVolumeOffset`, `blobs_outside_volume_tolerance`; droplets for particles at `value >= 0`
  with `radius = cbrt(3 volume / (4 pi))`, `droplet_volume`. `ToPointObject(std::span<const surface::Droplet>,
  PointObjectBuffers&)` of design 3.6 (object `droplets`).

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(BuildSurfaceFrame, LoneParticle_IsADroplet)   // design 7: no triangle, one droplet, radius 1e-6 relative
  TEST(BuildSurfaceFrame, DropletVolume_InNoBlob)    // sum of blob material_volume + droplet_volume == total, 1e-12
  TEST(ToPointObject, Droplets_RadiusVelocityAndIds)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Keep liquid outside
  every mesh visible as droplets`.

### Task 28: The surface to Alembic

**Kind:** integration

**Files:**
- Create: `native/fluidity/render_output/interchange/write_surface_file.h`, `write_surface_alembic.cc`,
  `write_surface_file.cc`, `write_surface_file_test.cc`

**Interfaces:**
- Produces: `WriteSurfaceFile` of design 3.6 (USD in Task 41; other formats `kFormatNotInBuild`): one `OPolyMesh`
  named `surface`, triangles reversed and positions turned, `.velocities` from `liquid_velocity`, per-vertex
  channels (`id` int32 low 31 bits, `age`, `distance_to_solid`, `blob_index` int32, `share_<name>`) in
  `arbGeomParams` with `kVertexScope`, per-face (`material_index`, `triangle_kind`, int32) with `kUniformScope`.
  A frame with `origin == kNotBuilt` is `kBadInput` (the caller lists it not built).

- [ ] **Step 1: Failing tests** (Alembic part of design test `WriteSurfaceFile, ReadBack`):

  ```cpp
  TEST(WriteSurfaceFile, AlembicTwoMaterials_ReadBack)  // a frame of two materials, two ingredients: every channel
      // of 5.7 equals the frame (float), per-face channels at face scope, metadata, one sample at 42 / 24
  TEST(WriteSurfaceFile, AlembicSphere_OutwardAfterBlenderTurn)  // Review Focus 1: read raw positions, apply the
      // literal matrix rows {1,0,0},{0,0,-1},{0,1,0} (Blender's import turn): signed volume > 0 and equal to
      // CheckSurfaceMesh(frame.mesh).measure.volume within 1e-6 relative
  TEST(WriteSurfaceFile, NotBuiltFrame_IsBadInput)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Write the surface as
  one Alembic file per frame`. New names: `WriteSurfaceFile`.

### Task 29: Foam source formulas

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the shared-header test is `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Create: `native/fluidity/foam_and_spray/foam_source.h` (shared), `foam_source_test.cc`
  (`LABEL cpu`), `native/fluidity/foam_and_spray/testdata/foam_source_main.json`

**Interfaces:**
- Produces: `Ramp`, `FoamSourceSettings`, `FoamSourceEvent` (design 3.3, moved here, **Plan decision**;
  `FoamSourceEvent` keeps `Vec3` members whose values are rounded to float when made, `sizeof` asserted, not 64),
  `SourcePoint`, `ComputeRampShare`, `ComputeOccupiedShare`, `SourceStrength`, `ComputeSourceStrength`, all
  `FLUIDITY_CPU_AND_GPU`, the 5.4 block exactly; `x_cj` the unit vector from `c` to `j` as on `main`; neighbours
  read x fastest, centre entry 13; a normal of length below 0.5 is not known; `curvature_is_known` false then.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ComputeSourceStrength, HandValues)  // design 7: |u| = 2 -> speed_squared_half 2; ramp {1, 2} at 1.5 -> 0.5;
      // sphere R = 8 spacings moving out: wave_crest 2 / R within 5 %; moving in or at rest: 0; separating
      // neighbours: trapped_air 0; all 1e-12 except curvature
  TEST(ComputeOccupiedShare, ColumnSum)    // flat surface at any height: column sum == depth, 1e-12
  TEST(ComputeSourceStrength, NormalNotKnown_NoCrestAndSaysSo)
  TEST(ComputeSourceStrength, SameAsMainOnTheSame27Points)  // ComputeWhitewaterSource3D
      // (whitewater_3d.cc:33-212) on 27 points with main's sphere normals given as normals27, turned to Z-up: 1e-12
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Add the three foam
  source measures for both processors`. New names: the design 3.3 foam source names.

### Task 30: The step recorder on the CPU, motion samples

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** `step_recorder_steps.h` is shared (`FLUIDITY_CPU_AND_GPU`), so Task 32's GPU functions call it; the
recorder owns an `OrderBuffers` and a `SumBuffers` built once by `BuildStepRecorder`; the block-key order passes the
recorder's `order_buffers` to `BuildStableOrder`.

**Amended 2026-11-14 (A31):** `BuildStableOrder` has one form; test `ScratchesSizedFromLimits` renamed
`BuffersSizedFromLimits`.

**Amended 2026-11-15 (A58):** `BuildStepRecorder(..., first_frame_time, RunMemory& run_memory)` (design 3.3): every
run-processor buffer of the recorder is a piece of `run_memory`; test `BuildStepRecorder,
IntoRunMemory_HandsOutExactlyTheEstimate`. **(A64)** `RecordStep` reads no region arrays of `fields`.

**Files:**
- Create: `native/fluidity/render_output/step_recorder.h`, `step_recorder.cc`, `step_recorder_cpu.cc`,
  `step_recorder_steps.h` (shared, `FLUIDITY_CPU_AND_GPU`: the per-cell-place functions of the count, fill and values
  passes, used by the CPU loop and, in Task 32, by the GPU functions), `step_recorder_test.cc`; test support gains
  `RecordStepScene` (owns a grid, a
  liquid-set `transfer::GridSnapshot`, `pressure::SurfaceDistances`, solids; `.inputs(range)` gives
  `step::FoamAndSprayInputs`) and `BuildThreeLevelGrid()` (biggest cell 0.2 m, levels 0 to 2, box 1.6 x 0.8 x 0.8,
  level 0 for `x < 0.6`, level 1 for `0.6 <= x < 1.0`, level 2 above, through `grid::BuildGrid`)

**Interfaces:**
- Consumes: `step::FoamAndSprayInputs`, `grid::FindNearbyFaces`, `ComputeNearbyFaceWeights`, `FindCornerPlaces`,
  `transfer::GridSnapshot`, `pressure::SurfaceDistances::at_corners`, `BuildStableOrder`.
- Produces: `MotionSettings`, `RecordedInterval`, `StepRecorder`, `BuildStepRecorder`, `RecordStep`,
  `CommitRecordedStep`, `RejectRecordedStep`, `ReadRecordedInterval` of design 3.3, the motion part (5.2 steps 1
  to 6; events and totals in Task 31). **Plan decisions:** a try ends at a frame time when `|end -
  (first_frame_time + n * frame_duration)| <= 1e-9 * frame_duration` for `n = round((end - first_frame_time) /
  frame_duration)`; a sample is kept for the first commit in each of `samples_per_frame` equal parts and for the
  commit at a frame time; `RecordStep` reads `particles_try` and `fields` only on a try that ends at a frame time
  (both may be null otherwise); `after_pressure` must hold exactly one entry, else `kBadInput`.
- **Plan decision (amended 2026-11-14, Foundation as built):** `StepRecorder` owns `OrderBuffers order_buffers`
  (`BuildOrderBuffers(run_on, limits.max_lattice_blocks, run_memory)`: the stable order of recorded blocks by block key and the
  prefix sums of per-block counts) and `SumBuffers sum_buffers` (`BuildSumBuffers(run_on, 1, run_memory)`: `ComputeLiquidTotals`
  is called once per cell block, Task 31), built by `BuildStepRecorder` (`kOutOfMemory` when they cannot be built);
  `RecordStep` allocates nothing for them. A recorded range with more blocks than `max_lattice_blocks` is the
  existing `kTooManyLatticeBlocks` count, never a working-buffers overflow.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(RecordStep, RejectedTry_LeavesRecorderUnchanged)  // design 7: record, reject, read -> empty; record,
                                                         // commit, read -> holds the step; exact
  TEST(RecordStep, LinearVelocity_IsReproduced)          // design 7: u = A x + b on the faces of BuildThreeLevelGrid
      // (A rows {0.1,0.2,0},{0,-0.3,0.1},{0.2,0,0.2}, b {0.5,-0.25,0.1}): every known point's velocity within
      // 1e-6 of the largest speed
  TEST(RecordStep, TwoLiquidsOneSet_ReadsTheLiquidSet)   // design 7 (A8): two liquids in the one set beside a snow
      // set and a gas marker set (their snapshots filled with NaN): each velocity equals the interpolation of the
      // liquid set's snapshot, exact; no NaN anywhere
  TEST(RecordStep, TryEndOneUlpFromFrameTime_KeepsFrameSample)  // Review Focus 3: end = nextafter(frame time)
  TEST(RecordStep, PointsFartherThanReach_AreNotListed)
  TEST(RecordStep, WrongGridVersion_IsWrongGridVersion)
  TEST(BuildStepRecorder, BuffersSizedFromLimits)        // amended: order_buffers.max_count() == max_lattice_blocks,
                                                         // sum_buffers.max_count() == 1, both on run_on
  TEST(BuildStepRecorder, IntoRunMemory_HandsOutExactlyTheEstimate)  // A58, design 7: foam on, kCpu: a RunMemory of
      // exactly ComputeRunMemoryBytes(BuildRenderOutputMemoryEstimate(settings, kCpu), counts, kRunOn) (Task 44's
      // rows; until Task 44 the recorder's own rows, the same function): handed_out_bytes() == reserved_bytes();
      // one byte fewer: kOutOfMemory, magnitude == bytes the last piece asked; exact
  ```
  Tests `LABEL cpu`, `LINKS fluidity_render_output_test_support fluidity_base_test_support`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement** the CPU loop as design 6 describes (five passes over cell
  places in index order; this task: count, fill, values). **Step 4:** gate. **Commit.** Subject: `Record the
  liquid's motion near its surface under try and commit`. New names: the design 3.3 recorder names.

### Task 31: The step recorder on the CPU, events, liquid totals and batches

**Amended 2026-11-15 (A64):** `ComputeLiquidTotals` over particles only (no `GridOnlyAmounts`).

**Amended 2026-11-14 (A31):** `cell_resize::ComputeLiquidTotals` in its one form with the recorder's `sum_buffers`.

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the per-block liquid totals call `cell_resize::ComputeLiquidTotals` (its one form, A31) with the
recorder's `sum_buffers`; the normals, strength and event passes join the shared `step_recorder_steps.h`.

**Files:**
- Modify: `step_recorder_cpu.cc`, `step_recorder.cc`, `step_recorder_steps.h`, `step_recorder_test.cc`

**Interfaces:**
- Consumes: Task 29's formulas; `cell_resize::ComputeLiquidTotals` (one form, with `recorder.sum_buffers`;
  cell resize plan Task 23; amended 2026-11-15, A64: over particles only, no `GridOnlyAmounts`), ~~`GridOnlyAmounts`
  (view of the try `GridFields`),~~ `math::TreeSum`.
- Produces: the normals and strength passes and events of 5.4; `CommitRecordedStep` ordering events by (step,
  block key, point), keeping the `max_events_per_step` largest `volume_time` (ties: lower block key, then point),
  counting `dropped_event_count` and `curvature_not_known_count`; liquid totals of 5.2 step 7, one partial per cell
  block of the range kept with its whole-grid block index (`block_in_whole_grid`, or the block itself when empty),
  summed in `CommitRecordedStep` by `math::TreeSum` in ascending whole-grid block index.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(RecordStep, BatchedEqualsOnePiece)     // design 7: BuildWaveScene (step test support) run by TryStep with a
      // make_foam_and_spray callback that calls RecordStep once per try, then on the same tries 2 and 5 sub-ranges
      // per try: identical events, samples, counters and liquid_totals_at_frame bytes
  TEST(CommitRecordedStep, AboveMaxEvents_KeepsLargestAndCounts)  // max 3 of 7: the 3 largest, dropped 4
  TEST(CommitRecordedStep, EqualVolumeTime_LowerKeyFirst)
  TEST(RecordStep, LiquidTotalsAtFrame_EqualComputeLiquidTotalsOfTheWholeGrid)  // exact against one whole call
  TEST(RecordStep, NoFoamSettings_NoEvents)   // sources null
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Record foam source
  events and the liquid totals at frame times`.

### Task 32: The step recorder on the GPU

**Amended 2026-11-15 (A58):** the recorder's device buffers are pieces of the `run_memory` given to
`BuildStepRecorder`; the "without a GPU" test is removed.

**Amended 2026-11-14 (A31):** test `OnGpuRejectedTry_BuffersLeftUnused` (was `..._ScratchLeftUnused`).

**Amended 2026-11-14 (A17, batched-run buffers):** the recorder's read-backs are counted, so the run's measured
copies per step include them (step plan Appendix A row M13; `plan-decisions.md` "Copies counted").

**Amended 2026-11-14 (Foundation as built):** private `step_recorder_gpu.h` returning `RunOnCallStatus`; GPU functions as thin loops over
`step_recorder_steps.h` with a `cpu` test over every thread block and thread; the prefix sum and the block-key order
with the recorder's `order_buffers` (one form each, A31); digests; the CPU-recorder copy-count test is `cpu`.

**Kind:** device

**Files:**
- Create: `native/fluidity/render_output/step_recorder_gpu.cu`, `step_recorder_gpu.h` (private),
  `step_recorder_steps_test.cc`, `step_recorder_gpu_test.cc`, `step_recorder_no_gpu_test.cc`
- Modify: `step_recorder_cpu.cc` (the `kGpu` branch under `#if defined(FLUIDITY_HAS_GPU)`, the mapping of
  `RunOnCallStatus` through `ToBufferError` into `RenderOutputFailure`), `CMakeLists.txt` (`fluidity_add_gpu_sources`;
  the steps test `LABEL cpu`, the GPU test `LABEL gpu`, the no-GPU test `LABEL no_gpu`, all `LINKS
  fluidity_render_output_test_support fluidity_base_test_support`)

**Interfaces:**
- Produces: the device side of `RecordStep` (design 6): device memory owned by the recorder for the whole run,
  sized by `RenderLimits` and `MotionSettings`, with its `order_buffers` and `sum_buffers` (Task 30); one launch per
  range for count, fill, values, normals, strength and events, each through `ApplyGpuFunction` with
  `ComputeLaunchSize`, each GPU function a grid-stride loop over a `step_recorder_steps.h` function; a prefix sum
  between count and fill (`ComputeExclusivePrefixSum(counts, block_count, first, recorder.order_buffers)`),
  `BuildStableOrder(keys, block_count, nullptr, order, recorder.order_buffers)` by block key; no sum across threads,
  no allocation per call. Copies back per call: event count, events, four counters; point arrays for kept samples
  only; one `LiquidTotals` per block on a frame-ending try. Private entry points named for what they do
  (plan-decisions), `_gpu` only in file names.
- **Plan decision (A17, batched-run buffers):** `step::CopyCounts ReadRecorderCopyCounts(const StepRecorder&
  recorder);` (in `step_recorder.h`): every `ReadBuffer` call above, one copy each, bytes into
  `bytes_from_processor`, cumulative since `BuildStepRecorder`; `{}` for a CPU recorder. In a batched run the
  per-block partials keep their whole-grid index from `FoamAndSprayInputs::block_in_whole_grid`, which is the
  batched run's CPU map copy (never empty there).

- [ ] **Step 1: Failing tests:**

  ```cpp
  // step_recorder_steps_test.cc, label cpu: the main evidence until the hardware run
  TEST(StepRecorderSteps, EveryThreadOrder_EqualsCpu)              // Task 30 and 31 cases: each pass's function for
      // every thread block and thread of ComputeLaunchSize, launch order and reverse, phase by phase with the CPU
      // prefix sum and stable order between count and fill: point lists, events and counters equal the CPU
      // recorder's, values' bits equal
  // label gpu (compiled, run by the owner at the end of the programme); each passes its point lists, events and
  // liquid totals to WriteResultDigest
  TEST(RecordStep, OnGpuLinearVelocity_EqualsCpuWithinTolerance)   // Global Constraints tolerances
  TEST(RecordStep, OnGpuDamBreak20Steps_EqualsCpuWithinTolerance)  // BuildFallingBlockScene(1.0)
  TEST(RecordStep, OnGpuBreakingWave_EqualsCpuWithinTolerance)     // BuildWaveScene
  TEST(RecordStep, OnGpuTwice_IdenticalEventsAndSamples)
  TEST(RecordStep, OnGpuRejectedTry_BuffersLeftUnused)            // the next commit's record equals a fresh run
  TEST(ReadRecorderCopyCounts, OnGpuFrameTry_CountsEveryReadBack)   // A17: event count, events, counters, kept
      // points and liquid totals: copy_count and bytes equal the sizes read
  // ~~label no_gpu~~
  ~~TEST(BuildStepRecorder, OnGpuWithoutAGpu_IsRunOnCallFailed)~~  // amended 2026-11-15 (A58): removed; a GPU
      // RunMemory cannot be built without a GPU (Foundation's BuildRunMemory, OnGpuWithoutAGpu_IsRunOnCallFailed)
  // label cpu (step_recorder_test.cc)
  TEST(ReadRecorderCopyCounts, CpuRecorder_IsEmpty)                 // A17
  ```
- [ ] **Step 2: Run** in the container (`--only cuda`): the `cpu` and `no_gpu` tests pass, the `gpu` tests compile.
  **Step 3:** gate. **Commit.** Subject: `Record steps on the GPU`. Body: `GPU: compiled, not run`.

### Task 33: Foam motion formulas

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the shared-header test is `LABEL cpu` (`fluidity_add_shared_test` retired).

**Amended 2026-11-15 (A47):** `foam_settings.h` also declares `inline constexpr double kFoamWindDriftDuration =
0.1;` (s, a starting value: Blender Strength 1 gives 0.1 m/s of foam drift), design 3.5; test
`kFoamWindDriftDuration, IsATenthOfASecond`.

**Files:**
- Create: `native/fluidity/foam_and_spray/foam_settings.h` (shared, **Plan decision**), `foam_motion.h` (shared),
  `foam_motion_test.cc` (`LABEL cpu`), `testdata/foam_motion_main.json`

**Interfaces:**
- Produces: `FoamKind`, `FoamSettings`, `DragMotion`, `ComputeDragMotion`, `ToFoamKind` of design 3.5,
  `<eq:foam-drag>` with `math::Expm1` and the exact `k = 0` form (ported from `whitewater_3d.cc:213-263`,
  `IntegrateWhitewaterDrag3D`); `ToFoamKind` ported from `:264-289` (`ClassifyWhitewater3D`), held `0.5 r` past
  each limit.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ComputeDragMotion, Limits)               // design 7: k = 0 -> parabola z0 + v t - g t^2 / 2; k dt = 1e-9:
      // series equals closed form; k dt = 1e6: v == target; fall from rest; 1e-13 relative
  TEST(ToFoamKind, KindIsHeldNearTheLimit)      // design 7: from spray, 1.4 r, 0.9 r, 0.4 r, -0.4 r, -1.6 r ->
                                                // spray, spray, foam, foam, bubble
  TEST(ComputeDragMotion, SameAsMain)           // IntegrateWhitewaterDrag3D, turned to Z-up: 1e-13
  TEST(ToFoamKind, SameAsMain)                  // ClassifyWhitewater3D on 100 fixed distances: exact
  TEST(kFoamWindDriftDuration, IsATenthOfASecond)  // A47: static_assert == 0.1
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Add foam drag and
  the kind rule for both processors`. New names: `FoamKind`, `FoamSettings`, `DragMotion`, `ComputeDragMotion`,
  `ToFoamKind`, `kFoamWindDriftDuration` (A47).

### Task 34: Adding foam particles

**Kind:** numerics

**Files:**
- Create: `native/fluidity/foam_and_spray/foam_particles.h`, `add_foam_particles.h`, `add_foam_particles.cc`,
  `add_foam_particles_test.cc`

**Interfaces:**
- Consumes: `step::ComputeRandomWords`, `step::ComputeSignedUnitValue` (`step/random_words.h`).
- Produces: `FoamParticles`, `FoamReport`, `AddFoamParticles` of design 3.5, 5.5 step 1 (ported from
  `whitewater_3d.cc:290-461`, `EmitWhitewaterBirthsThroughStep3D` at `:356`): `n = floor(particles_per_volume_time
  * volume_time / count_per_particle + owed)`, rest owed to the point; ids rise in event order. **Plan decision**
  for the random spread: words `ComputeRandomWords(child_number = number in event, source_id = block key, key =
  (step_count << 9) | point in block)`; offset per axis `0.5 * spacing * ComputeSignedUnitValue(word[axis])`
  (the design's `spacing * (u - 0.5)`); velocity as the event; `start_time` the event's time. At
  `max_foam_particles` nothing more is added, `not_added_at_limit` counts the rest.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(AddFoamParticles, OwedCarriesOver)              // design 7: two equal events worth 2.5 each: 2 then 3
                                                       // added; owed 0.5 then 0; exact
  TEST(AddFoamParticles, SameTotalInOneOrFourSteps)    // design 7: counts differ by at most 1 per lattice point
  TEST(AddFoamParticles, AtLimit_CountsTheRest)
  TEST(AddFoamParticles, SpreadStaysInsideTheLatticeCell)
  TEST(AddFoamParticles, SameAsMainCountsAndOwed)      // EmitWhitewaterBirthsThroughStep3D: counts and owed exact
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Add foam particles
  from source events`. New names: `FoamParticles`, `FoamReport`, `AddFoamParticles`.

### Task 35: Moving foam particles

**Amended 2026-11-15 (A47):** `MoveFoamParticles` takes `const gas::GasWind& wind` after `gravity` (design 3.5):
spray accelerates by gravity plus the wind sampled at the middle of each piece, foam drifts at
`kFoamWindDriftDuration` times the horizontal wind; two tests added.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/foam_and_spray/move_foam_particles.h`, `move_foam_particles.cc`,
  `move_foam_particles_test.cc`
- Modify (A47): `native/fluidity/foam_and_spray/CMakeLists.txt` (link `fluidity_gas` for `SampleGasWind`; gas is an
  earlier module)

**Interfaces:**
- Produces: `MoveFoamParticles` of design 3.5, `<proc-move-foam-particles>` steps 1 to 6 (5.5 steps 2 and 3,
  ported from `whitewater_transport_3d.cc:71-280`). **Plan decision:** the kind at the frame time (5.5 step 3) is
  set at the end of `MoveFoamParticles`, from `ComputeLiquidMotion` at the frame time; where the distance is not
  known the kind stays. Solids: a particle inside a solid (`ComputeDistanceToSolid < 0` at the sample's pose) at
  the end of a piece is put on its surface and loses its speed into it relative to `ComputeSolidVelocity`.
  **Plan decisions (A47):** for a piece `[t, t + dt]`, `a_w = gas::SampleGasWind(wind, t + dt / 2)`; spray:
  `ComputeDragMotion(start, air_velocity + (gravity + a_w) / spray_drag_rate, spray_drag_rate, dt)` for
  `spray_drag_rate > 0`, and for `spray_drag_rate == 0` the `k = 0` form with acceleration `gravity + a_w` (`v +=
  dt * (gravity + a_w)`, `x += dt * v_start + dt * dt / 2 * (gravity + a_w)`); foam: the midpoint velocity is the
  liquid velocity plus `kFoamWindDriftDuration * (a_w - math::Dot(a_w, kUp) * kUp)`, and that sum is stored as its
  velocity; bubbles feel no wind (under the surface). Empty `wind.keys`: `a_w = 0`, the bytes of the windless code.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(MoveFoamParticles, StopsAtAnySolid)             // design 7: spray thrown at a turning Box: never inside, no
                                                       // speed into the solid, contact tolerance 0x1p-30
  TEST(MoveFoamParticles, SprayFallsTowardAirVelocity) // matches ComputeDragMotion over the whole interval, 1e-12
  TEST(MoveFoamParticles, BubbleWithoutLiquidVelocity_RisesAndCounts)
  TEST(MoveFoamParticles, FoamWithoutLiquidVelocity_KeepsVelocityAndCounts)
  TEST(MoveFoamParticles, ParticleOrderDoesNotMatter)  // reversed input order: same values per id
  // A47 (design 7; suite named as plan-decisions item 8 names it)
  TEST(MakeFoamAndSpray, SprayInWind_AcceleratesByWind)   // one spray particle far from liquid and solids,
      // spray_drag_rate 0, gravity (0, 0, -9.81), wind keys (1, 0, 0) m/s2 at 0 s and (3, 0, 0) at 1 s, two samples
      // 0.04 s apart, 4 pieces of 0.01 s: after each piece v == v_before + dt * (g + SampleGasWind(wind, t + dt / 2))
      // in the same order of operations, exact; empty keys: the gravity-only bytes
  TEST(MakeFoamAndSpray, FoamInWind_DriftsHorizontally)   // one foam particle on a still flat surface (liquid
      // velocity 0, known), wind (1, 0, 2) m/s2: velocity == (0.1, 0, 0) and x grows by 0.1 * dt per piece, exact;
      // wind (0, 0, 3): velocity exactly 0, position unchanged
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Move foam, spray and
  bubbles with the liquid and the air`. New names: `MoveFoamParticles`.

### Task 36: Removing foam particles

**Kind:** numerics

**Files:**
- Create: `native/fluidity/foam_and_spray/remove_foam_particles.h`, `remove_foam_particles.cc`,
  `remove_foam_particles_test.cc`

**Interfaces:**
- Produces: `RemoveFoamParticles` of design 3.5, 5.5 step 4: foam older than `foam_lifetime` (`ended_by_age`), any
  particle outside the domain box (`left_domain`); ascending id kept; kind counts filled.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(RemoveFoamParticles, AgeAndDomain)     // design 7: foam at age lifetime; spray one step outside the box:
                                              // both removed, each in its own counter
  TEST(RemoveFoamParticles, SprayNeverEndsByAge)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Remove old foam and
  particles that left the domain`. New names: `RemoveFoamParticles`.

### Task 37: The foam volume

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the grouping is `BuildGroupsByDestination` in its form that builds a CPU `OrderBuffers` for the
call (once per frame on the worker), counted by Task 44 as a `kOneCall`, `kCpu` entry at `max_foam_particles`.

**Amended 2026-11-14 (A31):** one form: `BuildFoamVolume` takes the caller's CPU `OrderBuffers& order_buffers`
(`RenderOutput::frame_order_buffers` in a run; built locally in the tests); nothing allocated per frame for the
grouping.

**Files:**
- Create: `native/fluidity/render_output/build_foam_volume.cc`, `build_foam_volume_test.cc`; modify `volume_frame.h`

**Interfaces:**
- Produces: `BuildFoamVolume` of design 3.6, 5.6 item 1: `known` inside the domain box and outside every solid;
  foam and bubble particles add `count_per_particle` with trilinear weights, weights of unknown points given to
  the known ones, none known: `value_outside_known_count`; channels `kFoamCountDensity`, `kBubbleCountDensity`,
  `kBubbleVolumeFraction` (bubble sum times `4/3 pi bubble_radius^3` over `spacing^3`); spray adds nothing; sums
  by lattice point then id through `BuildGroupsByDestination` (amended: with the given CPU `order_buffers`, allocating nothing; a
  `BufferError` maps to `kBadInput`, since nothing here builds memory or runs on a GPU; Requires:
  `order_buffers.run_on() == RunOn::kCpu`, debug assertion).

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(BuildFoamVolume, CountsAreKept)    // design 7: 10,000 particles, some beside a solid:
      // sum density * spacing^3 == count * count_per_particle, 1e-12; nothing at points not known
  TEST(BuildFoamVolume, SprayAddsNothing)
  TEST(BuildFoamVolume, AllEightNotKnown_Counted)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Count foam and
  bubbles on a lattice`. New names: `BuildFoamVolume`.

### Task 38: Foam particles as points

**Kind:** mechanical

**Files:**
- Modify: `point_objects.h`, `point_objects.cc`, `point_objects_test.cc`

**Interfaces:**
- Produces: `ToPointObjects(const FoamParticles&, const FoamSettings&, double frame_time, PointObjectBuffers&)` of design
  3.6: three objects `spray`, `foam`, `bubbles` (empty ones included), `constant_radius = display_radius`, `id`,
  `age = frame_time - start_time`.

- [ ] **Step 1: Failing tests:** `TEST(ToPointObjects, ThreeKinds_SplitByKindInIdOrder)`;
  `TEST(ToPointObjects, AgeAtFrameTime)`; `TEST(ToPointObjects, NoBubbles_EmptyThirdObject)`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Turn foam particles
  into three point objects`. New names: `ToPointObjects`.

### Task 39: Solid motion to JSON and Alembic

**Kind:** integration

**Files:**
- Create: `native/fluidity/render_output/interchange/write_solid_motion_file.h`, `write_solid_motion_json.cc`,
  `write_solid_motion_alembic.cc`, `write_solid_motion_file.cc`, `write_solid_motion_file_test.cc`

**Interfaces:**
- Consumes: `solids::SolidsSnapshot`, `SolidState`, `Orientation`, `ToAxes`.
- Produces: `WriteSolidMotionFile` of design 3.6 (USD in Task 42). JSON (no library, written through
  `OutputFile::Write`): the shape of design 5.7 plus the metadata keys of Global Constraints, numbers by
  `std::to_chars` shortest round-trip form, ascending `solid_id`, `orientation` as `(w, x, y, z)` of
  `Orientation`. Alembic: one `OXform` per solid named `solid_<solid_id>`, a double matrix from `origin` and
  `ToAxes(orientation)`, turned as `C M C^T` with `C` the `(x, z, -y)` turn.

- [ ] **Step 1: Failing tests** (JSON and Alembic parts of design test `WriteSolidMotionFile, ReadBack`):

  ```cpp
  TEST(WriteSolidMotionFile, JsonReadBack)      // a turning Box, a fixed plane, a driven solid that left the scene:
      // read with flatbuffers::Parser::ParseFlexBuffer (already a dependency): every double bit for bit, in_scene
      // false for the one that left, metadata
  TEST(WriteSolidMotionFile, AlembicReadBack)   // the same solids: matrix turned back, origin and ToAxes within 1e-12
  TEST(WriteSolidMotionFile, OpenVdbFormat_IsFormatNotInBuild)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Write solid motion as
  JSON and Alembic`. New names: `WriteSolidMotionFile`.

### Task 40: Points to USD

**Kind:** integration

**Files:**
- Create: `native/fluidity/render_output/interchange/write_points_usd.cc`, `usd_frame.h`, `usd_frame.cc` (private:
  a `.usdc` layer created with the explicit file format on `path_while_writing()`, `upAxis = "Z"`,
  `metersPerUnit = 1`, `timeCodesPerSecond = framesPerSecond = fps`, `startTimeCode = endTimeCode = frame`,
  `customLayerData` metadata; `CheckUsdResources` first)
- Modify: `write_points_file.cc`, `write_points_file_test.cc`, `interchange_libraries_test.cc`

**Interfaces:**
- Produces: `WritePointsFile(InterchangeFormat::kUsd, ...)`: per object one `UsdGeomPoints` at `/<object_name>`
  with `points`, `velocities`, `widths` (diameter), `ids` (int64, full id), primvar `id` (int32 low 31 bits) and
  every channel as a `vertex` primvar (float or int32 only), each one time sample at time code `frame`.

- [ ] **Step 1: Failing tests** (USD part of design test `WritePointsFile, ReadBack`):

  ```cpp
  TEST(WritePointsFile, UsdParticles_ReadBack)          // as Task 13's Alembic test, no turn; stage metadata
  TEST(WritePointsFile, UsdDropletsAndFoam_ReadBack)
  TEST(WritePointsFile, UsdIdAbove31Bits_LowBitsAndFullId)
  TEST(WritePointsFile, UsdWithoutResources_IsFileWriteFailed)  // Review Focus 5; replaces Task 9's
      // CheckUsdResources test; missing_resource_folder names the folder
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Write points as one
  USD file per frame, the candidate to be proven in Blender`.

### Task 41: The surface to USD

**Kind:** integration

**Files:**
- Create: `native/fluidity/render_output/interchange/write_surface_usd.cc`
- Modify: `write_surface_file.cc`, `write_surface_file_test.cc`

**Interfaces:**
- Produces: `WriteSurfaceFile(InterchangeFormat::kUsd, ...)`: `UsdGeomMesh` at `/surface` with `points`,
  `faceVertexCounts` (all 3), `faceVertexIndices` (not reversed), `orientation = rightHanded`, `velocities`,
  per-vertex channels as `vertex` primvars, per-face as `uniform` primvars.

- [ ] **Step 1: Failing tests** (USD part of design test `WriteSurfaceFile, ReadBack`):
  `TEST(WriteSurfaceFile, UsdTwoMaterials_ReadBack)` (every channel, `upAxis`, `metersPerUnit`,
  `timeCodesPerSecond`, per-face channels `uniform`); `TEST(WriteSurfaceFile, UsdSphere_OutwardAsWritten)` (signed
  volume from the read points and indices equals the mesh volume, 1e-6 relative).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Write the surface as
  one USD file per frame, the candidate to be proven in Blender`.

### Task 42: Solid motion to USD

**Kind:** integration

**Files:**
- Create: `native/fluidity/render_output/interchange/write_solid_motion_usd.cc`
- Modify: `write_solid_motion_file.cc`, `write_solid_motion_file_test.cc`

**Interfaces:**
- Produces: `WriteSolidMotionFile(InterchangeFormat::kUsd, ...)`: one `UsdGeomXform` per solid at
  `/solid_<solid_id>` with `xformOp:translate` (double3) and `xformOp:orient` (quatd), one time sample at `frame`.

- [ ] **Step 1: Failing test** (USD part of design test `WriteSolidMotionFile, ReadBack`):
  `TEST(WriteSolidMotionFile, UsdReadBack)`: the three solids of Task 39; translation and the rotation matrix of the
  quaternion against `origin` and `ToAxes` within 1e-12.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Write solid motion as
  USD, the candidate to be proven in Blender`.

### Task 43: The format table

**Amended 2026-11-14 (A16):** a witness record holds one Blender result, for Blender 5.3; the choice rule and the
tests follow.

**Amended 2026-11-14 (Foundation as built):** `writers_test.cc` is `fluidity_add_test(... LABEL cpu LINKS fluidity_interchange_test_support
fluidity_interchange)` (no `interchange` label).

**Kind:** mechanical

**Files:**
- Modify: `native/fluidity/render_output/format_table.h`
- Create: `format_table.cc`, `format_table_test.cc` (in `fluidity`, not in `fluidity_interchange`),
  `native/fluidity/render_output/interchange/writers_test.cc`

**Interfaces:**
- Produces: `WitnessResult`, `WitnessRecord`, `FormatCandidate`, `FormatChoice`, `ChooseFormat`, `ReadFormatTable`,
  `ReadFormatChoice` of design 3.6, with one **Plan decision** (A16: Blender 5.3 is the one supported version):

  ```cpp
  struct WitnessRecord {
    WitnessProperty property;
    WitnessResult blender_5_3 = WitnessResult::kNotRun;  // The result measured in Blender 5.3.
    // Manual page, or "<test>; Blender <version> <build hash>; files <digest>"; empty while kNotRun.
    std::string_view evidence;
  };
  ```

  (the design's `blender_5_2` field is gone). `ChooseFormat` returns the first candidate whose every record is
  `kPassed`, else the last candidate. A table line reads
  `{WitnessProperty::<P>, WitnessResult::<r>, "<evidence>"},` (the shape the add-on's
  `write_witness_records.py` prints). The table of 5.7 "The format table today": candidates in preference order
  (`kPoints`: USD, Alembic; `kSurfaceMesh`: USD, Alembic; `kVolume`: OpenVDB; `kSolidMotion`: USD, Alembic, JSON),
  `chosen` written out as the last candidate everywhere; one `WitnessRecord` per property each kind requires, in
  `WitnessProperty` order, with the initial values of 5.7 "What the Blender manual says" (`kDocumented` with the
  quoted manual line in `evidence`, or `kNotRun` with empty evidence) for Blender 5.3. This table is the input the
  Blender add-on's witness tests write back to, one `WitnessRecord` line at a time.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(FormatTable, ChosenFollowsWitnesses)     // design 7 with A16: every chosen == ChooseFormat; hand-made
      // tables: USD all kPassed -> kUsd; one USD property kFailed -> next candidate; one USD property kNotRun or
      // kDocumented, the rest kPassed -> next candidate; all kDocumented -> last
  TEST(FormatTable, EveryChannelIsWitnessed)    // design 7: each channel's witnessed_by is a property its kind
      // requires; every candidate has one record per required property; a name exists for the chosen format
  TEST(ReadFormatChoice, EveryKind_InKindOrder)
  TEST(FormatTable, NotRunRecords_HaveNoEvidence)  // and every kDocumented record has evidence
  TEST(FormatTable, MeasuredRecords_NameBlender53)  // every kPassed or kFailed record's evidence contains
      // "Blender 5.3." (none today: the test passes on the initial table and guards the add-on's write-back)
  // writers_test.cc, LABEL cpu, LINKS fluidity_interchange_test_support
  TEST(Writers, FormatNotInBuild)               // design 7: OpenVDB for points (and for surface, solid motion);
      // kFormatNotInBuild, nothing listed
  TEST(Writers, EveryCandidateHasAWriter)       // each candidate of each kind, on a tiny input: never
      // kFormatNotInBuild
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** `ChooseFormat` is `constexpr`, and a `static_assert` per kind
  compares `chosen` with it, so a hand edit of one without the other does not compile. **Step 4:** gate.
  **Commit.** Subject: `Choose each kind's format from Blender witness records`. New names: the design 3.6
  format-table names.

### Task 44: Render settings, their check and the memory estimate entries

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** the estimate adds the recorder's two working buffers (`where = kRunOn`, `kWholeRun`) and the foam
volume's per-frame order buffers (`kOneCall`, `kCpu`).

**Amended 2026-11-14 (A31, reconcile item 26):** the foam-volume entry becomes `"render output: frame order
buffers"`, sized at `max(max_field_parts, max_foam_particles)` and kept for the whole run by `RenderOutput`; test
`RecorderScratchEntriesMatchTheRecorder` renamed, `FrameOrderEntriesMatchTheRenderOutput` added.

**Amended 2026-11-15 (A58, A47):** one row per buffer (the recorder's order and sum buffers and the frame order
buffers one row each, given count 1, `bytes_each` the whole `ComputeOrderBuffersSize` / `ComputeSumBuffersSize`);
no `kRunOn` row is `kOneCall`; `RenderSettings` gains `gas::GasWind wind` (design 3.6), which `CheckRenderSettings`
checks with `gas::CheckGasSources({}, wind)`.

**Files:**
- Create: `native/fluidity/render_output/render_frame.h` (settings in this task), `check_render_settings.cc`,
  `render_output_memory.cc`, `render_settings_test.cc`

**Interfaces:**
- Consumes: `materials::MaterialTable`, `MaterialKind`, `gas_ingredient_index()`; `MemoryEstimateEntry`, `CountedPer`,
  `KeptFor`, `MemoryWhere`.
- Produces: `VolumeSettings`, `RenderSettings`, `CheckRenderSettings`, `BuildRenderOutputMemoryEstimate` of design 3.6.
  Rows as design 6 "Memory estimate entries", with `sizeof` of the real types (`static_assert` on each): the recorder's device
  working buffers (`kGivenCount`, `kWholeRun`, `kRunOn`); per queued frame the samples and events (`kOneSnapshot`, `kCpu`); the
  worker's field blocks, vertices, triangles, foam particles (`kOneCall`, `kCpu`); the `PointObjectBuffers` copies of the
  largest material (`kOneCall`, `kCpu`). Amended (Foundation as built): ~~`"render output: recorder order buffers"`
  (`kGivenCount` = `limits.max_lattice_blocks`, `bytes_each = ComputeOrderBuffersSize(run_on).bytes_per_item`) and
  `"render output: recorder order buffers fixed bytes"` (`kGivenCount` = `fixed_bytes`, 1 byte each),
  `"render output: recorder sum buffers"` (`kGivenCount` 1, `ComputeSumBuffersSize(run_on).bytes_per_item`) and its
  fixed bytes, all `kWholeRun`, `kRunOn`; `"render output: frame order buffers"` (`kGivenCount` =
  `max(limits.max_field_parts, limits.max_foam_particles)`, `ComputeOrderBuffersSize(RunOn::kCpu).bytes_per_item`;
  the field parts of `ComputeAddedVolume` and the foam grouping of `BuildFoamVolume`, reconcile item 26) and its
  fixed bytes, `kWholeRun`, `kCpu`.~~ **(A58)** `"render output: recorder order buffers"` (given count 1,
  `bytes_each` = `ComputeOrderBuffersSize(run_on)` at `limits.max_lattice_blocks`, whole), `"render output: recorder
  sum buffers"` (given count 1, `ComputeSumBuffersSize(run_on)` at 1, whole), both `kWholeRun`, `kRunOn`; `"render
  output: frame order buffers"` (given count 1, `ComputeOrderBuffersSize(RunOn::kCpu)` at `max(limits.max_field_parts,
  limits.max_foam_particles)`, whole; reconcile item 26), `kWholeRun`, `kCpu`; each other recorder device buffer one
  row.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(CheckRenderSettings, SnowGroup_IsBadInput)   // design 7 (A9): a group with a snow material; with a gas
                                                    // material: kBadInput each, id == the material_id
  TEST(CheckRenderSettings, EmptyGroup_IsBadInput)
  TEST(CheckRenderSettings, RepeatedMaterial_IsBadInput)
  TEST(CheckRenderSettings, MotionSpacingAboveTwiceSmallestCell_IsBadInput)
  TEST(CheckRenderSettings, SamplesPerFrame65_IsBadInput)
  TEST(BuildRenderOutputMemoryEstimate, OnCpu_NoRunOnRowsBesideCpu)  // every row's where is kCpu with RunOn::kCpu
  TEST(BuildRenderOutputMemoryEstimate, NoSurfaceNoFoam_OnlyMotionRows)
  TEST(BuildRenderOutputMemoryEstimate, RecorderBufferEntriesMatchTheRecorder)  // amended (A58: two rows): the recorder
      // buffers entries equal order_buffers.byte_count() + sum_buffers.byte_count() of a CPU BuildStepRecorder
  TEST(BuildRenderOutputMemoryEstimate, FrameOrderEntriesMatchTheRenderOutput)  // A31, reconcile item 26: the
      // frame order entry (A58: one row) equals frame_order_buffers.byte_count() of BuildRenderOutput at the limits
  TEST(BuildRenderOutputMemoryEstimate, NoRunOnRowIsOneCall)    // A58: kCpu and kGpu, every setting combination
  TEST(CheckRenderSettings, WindKeysUnordered_IsBadInput)       // A47: wind keys at 1 s then 0.5 s: kBadInput
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Check render settings
  and count render output's memory`. New names: `VolumeSettings`, `RenderSettings`, `CheckRenderSettings`,
  `BuildRenderOutputMemoryEstimate`.

### Task 45: One output frame

**Amended 2026-11-14 (A31, reconcile item 26):** `RenderOutput` owns the frame's CPU `frame_order_buffers`; `BuildRenderOutput` can fail with `kOutOfMemory`.

**Amended 2026-11-15 (A58, A47, A64):** `BuildRenderOutput(settings, domain_box, grid_frame, gravity, RunMemory&
run_memory)` with the CPU block (the step's `cpu_run_memory`, the one block in a CPU run) for `frame_order_buffers`;
`BuildRenderFrame` passes `settings.wind` to `MoveFoamParticles`; `VolumeTargetFromLiquidTotals` runs on an
all-particle deep tank.

**Kind:** integration

**Files:**
- Modify: `render_frame.h`
- Create: `build_render_frame.cc`, `build_render_frame_test.cc`

**Interfaces:**
- Consumes: everything above; `gas::ToGasSnapshot`, `solids::BuildContactSolids` (for test inputs).
- Produces: `RenderOutput`, `BuildRenderOutput`, `FrameInputs`, `RenderFrame`, `BuildRenderFrame` of design 3.6,
  `<proc-build-render-frame>` steps 1 to 8: frames in order (`kWrongStateVersion` when `recorded` does not follow
  the kept sample); the interval copied into one vector (kept sample, recorded samples, sample at frame); per
  group `BuildSurfaceFrame`; foam add, move, remove; volumes; `liquid_volume_outside_surfaces =
  recorded.liquid_totals_at_frame.current_volume` minus every group's blob `material_volume` and `droplet_volume`;
  on failure `output` unchanged. Threads as design 6 "The worker". Amended 2026-11-14 (A31, reconcile item 26):
  `BuildRenderOutput` builds `RenderOutput::frame_order_buffers` (`BuildOrderBuffers(RunOn::kCpu,
  max(limits.max_field_parts, limits.max_foam_particles), run_memory)` (A58; `Requires: run_memory.run_on() ==
  RunOn::kCpu`), so it returns `std::expected<RenderOutput,
  RenderOutputFailure>` with `kOutOfMemory`), and `BuildRenderFrame` passes it to `BuildSurfaceFrame` and
  `BuildFoamVolume` in turn; no frame allocates working buffers.

- [ ] **Step 1: Failing tests** (inputs from `step::TestScene`, `Simulation`, `ReadSnapshot` and a recorder on the
  step's callbacks, as I/O will compose them):

  ```cpp
  TEST(BuildRenderFrame, SameInputTwice_SameBytes)      // design 7: BuildWaveScene, 10 frames: identical bytes
  TEST(BuildRenderFrame, OneOrEightThreads_SameBytes)
  TEST(BuildRenderFrame, VolumeTargetFromLiquidTotals)  // design 7 (A64): deep tank, cells two levels coarser
      // below 8 w_min, particles of their own level in every liquid cell, one group of every liquid:
      // liquid_volume_outside_surfaces equals total minus the parts, 1e-12 relative
  TEST(BuildRenderOutput, CpuBlockOneByteShort_IsOutOfMemory)   // A58: a CPU RunMemory one byte short of the
      // frame order buffers row: kOutOfMemory, magnitude == bytes asked; exact
  TEST(BuildRenderFrame, WindReachesFoam)                       // A47: one foam particle on a still surface, settings
      // wind (1, 0, 0): after one frame its velocity == (0.1, 0, 0), exact
  TEST(BuildRenderFrame, NoGas_NoGasVolume)             // third state of design test BuildGasVolume, ThreeStates
  TEST(BuildRenderFrame, OutOfOrder_IsWrongStateVersionAndKeepsOutput)  // the same call then succeeds
  TEST(BuildRenderFrame, TooManyVertices_OtherProductsBuilt)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Build one output
  frame from a snapshot and its recorded interval`. New names: `RenderOutput`, `BuildRenderOutput`, `FrameInputs`,
  `RenderFrame`, `BuildRenderFrame`.

### Task 46: The render output snapshot

**Kind:** integration

**Files:**
- Modify: `render_frame.h`, `build_render_frame.cc`
- Create: `render_output_snapshot_test.cc`

**Interfaces:**
- Produces: `RenderOutputSnapshot`, `ReadSnapshot`, `BuildRenderOutputFromSnapshot` of design 3.6 (amended
  2026-11-15, A58: last parameter `RunMemory& run_memory`, the CPU block, as `BuildRenderOutput`); a snapshot
  whose `group_first_material_id` disagrees with the settings' groups is `kBadInput`.

- [ ] **Step 1: Failing tests:** `TEST(BuildRenderFrame, FromSnapshot_GoesOnEqual)` (design 7: 10 frames; or 5,
  `ReadSnapshot`, rebuild, 5 more: identical frames 6 to 10); `TEST(BuildRenderOutputFromSnapshot,
  GroupsDiffer_IsBadInput)`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Let a saved state
  carry the mesh and foam into a continued run`. New names: `RenderOutputSnapshot`, `ReadSnapshot`,
  `BuildRenderOutputFromSnapshot`.

### Task 47: Hand render output to I/O and the CLI

**Amended 2026-11-14 (A16):** `capabilities` prints each witness record with one result key, `blender_5_3`.

**Amended 2026-11-14 (A17, batched-run buffers):** the recorder's copies join the run's measured copies per step
(I/O plan seam 11).

**Kind:** integration

**Files:**
- Modify (only the seams the I/O module left for this module, found with the `git grep` of "Before you start" and
  the I/O plan's own hand-over list): `native/fluidity/io/run_scene.h/.cc` (`render_output::RenderSettings
  render`, the `CheckRenderSettings` call), the `BuildRunPlan` call (`BuildRenderOutputMemoryEstimate`), the
  `StepCallbacks` composition (`make_foam_and_spray` calls `RecordStep` and returns false only on
  `kRunOnCallFailed`; `on_try_done` commits or rejects), `io/frame_inputs.h`, `io/saved_state.h`
  (`RenderOutputSnapshot`), the CLI's `FrameWriter`, `capabilities`, `CMakeLists.txt` (the CLI links
  `fluidity_interchange`, and nothing else does), and their tests.

**Interfaces:**
- Consumes: I/O design section 4 items 1 to 8 and 3.3 to 3.7 exactly. **Amended 2026-11-15 (A47, A58):** I/O sets
  `RenderSettings::wind = ReadyScene::wind` (scene field `wind`, present whenever a Wind field exists, also without
  gas); the step builds the recorder with `BuildStepRecorder(..., run_memory)` and I/O builds `RenderOutput` with
  the step's CPU block (`cpu_run_memory`, the one block in a CPU run); test `Run, WindWithoutGas_MovesFoamOnly`
  (a liquid-only scene with wind keys: foam drifts, the liquid's committed state equals the windless run's, exact).
- Produces: the I/O seams of render output design section 4 "I/O and CLI" items 1 to 8: the worker builds the frame
  and calls, per frame, `WritePointsFile` for every particle material that is not gas, `WriteSurfaceFile` and
  droplets per group, foam particles, both volumes, `WriteSolidMotionFile`, each into an `OutputFile` opened with
  the extension of `ReadFormatChoice(kind)` and the file names of design 5.7, then `WriteListedFile(kSizeOnly)`; a
  failed frame or writer lists the product not built with `ToName` and never stops the simulation;
  `capabilities` reports `ReadFormatTable()`, `ReadOutputChannels()` and `ReadInterchangeLibraries()` (with
  `missing_resource_folder`); each witness record is printed as `{"property": "<name>", "blender_5_3":
  "<not_run|documented|passed|failed>", "evidence": "..."}`, the keys the Blender add-on's `WitnessRecordFact`
  reads (add-on plan Task 10). Seam 11 (A17): the run's `on_step_done` adds the difference of
  `ReadRecorderCopyCounts` since the last committed step to the copies `io::ApplyStepToRunCost` receives (I/O plan
  Task 29a), and `BuildRenderOutputMemoryEstimate` stays the `output_estimate` of the plan so the budget counts them.

- [ ] **Step 1: Failing tests** in I/O's and the CLI's test targets: `TEST(Run, TwoFrames_EveryProductListed)` (falling
  block with foam and a surface group: per frame one listed file per product with the chosen extension); `TEST(Run,
  WriterFails_ProductListedNotBuiltAndRunGoesOn)` (a `max_vertices` of 10); `TEST(Capabilities,
  ReportsFormatTableAndLibraries)` (also: every witness record has the key `blender_5_3` and no `blender_5_2`);
  `TEST(Run, RecorderCopiesInRunCost)` (A17: with a CPU recorder the run cost's copies equal those of the same run
  without foam, since `ReadRecorderCopyCounts` is `{}`; the GPU sum is the owner's `gpu` run of I/O Task 41). If I/O
  already wired a seam, extend its test instead and say so in the commit.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Write every render
  output product from the CLI's frame worker`.

### Task 48: Scene tests

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** tests `LABEL cpu` with `LINKS fluidity_render_output_test_support fluidity_base_test_support`.

**Files:**
- Create: `native/fluidity/render_output/render_output_scene_test.cc`, `surface_sequence_test.cc`
- Modify: `native/fluidity/render_output/CMakeLists.txt` (both tests `LABEL cpu LINKS
  fluidity_render_output_test_support fluidity_base_test_support`)

- [ ] **Step 1: Tests:**

  ```cpp
  TEST(RenderOutput, SwitchedOff_SameCommittedBytes)  // design 7: BuildFallingBlockScene(1.0) (the dam break),
      // 50 steps with and without the recorder and the worker: ExpectSameBytes on the SimulationSnapshots
  TEST(BuildRenderFrame, TwoFrameRates)               // design 7: the same run written at 24 and 48 fps: at shared
      // times foam count within 2 %, mean position within 0.5 spacing
  TEST(BuildGasVolume, GasBesideLiquid)               // design 7 (A10): BuildGasBesideLiquidScene: same
      // FrameIdentity in surface and gas volume; known 0 at every point of a liquid cell; no surface vertex reads
      // gas; no gas marker in any points object
  TEST(SurfaceSequence, ReportNumbers)                // numerics 12.7 on fixed recorded inputs: static sphere,
      // moving sphere, stretching sheet, merging pair, splash with 2,000 droplets; asserts every mesh safe and
      // prints volume, distance to the analytic shape, defects, kept ids and time for the hand-over
  ```
- [ ] **Step 2: The one-way rule.** Run
  `git grep -nE "(^|[(,] *)(step::Simulation|particles::Particles|grid::StaggeredGrid|solids::Solids)&" -- native/fluidity/render_output native/fluidity/surface native/fluidity/foam_and_spray`.
  Expected: no match (every such parameter is `const`). Run
  `git grep -n "render_output/interchange/" -- native ':!native/fluidity/render_output/interchange' ':!native/fluidity/cli'`.
  Expected: no match.
- [ ] **Step 3: Run**; gate. **Commit.** Subject: `Show render output on whole scenes`.

### Task 49: Comment pass and hand-over

**Amended 2026-11-14 (A31):** hand-over lists the frame order buffers, `max_field_parts` and the one-form `ComputeLiquidTotals`.

**Amended 2026-11-14 (A16, A19):** the hand-over lists the pinned library versions and says the witness tests run in
Blender 5.3 only.

**Amended 2026-11-14 (Foundation as built):** the hand-over records the recorder's working buffers, the shared `step_recorder_steps.h` and its `cpu`
test as the GPU evidence until the hardware run, the use of the `RTTI` argument (from I/O Task 34) and the retired `fluidity_add_interchange_test`.

**Amended 2026-11-15 (A58, A47, A64, reconcile item 31):** the hand-over also lists the `RunMemory` builders,
`kOutOfMemory` moved last, the writers' `std::bad_alloc` as `kFileWriteFailed`, `RenderSettings::wind` with
`kFoamWindDriftDuration`, and the grid-only inputs moved to the later group (Task 50, not in the done-criteria).

**Files:**
- Modify: any file of `render_output/`, `surface/`, `foam_and_spray/`, the paper section, `docs/gpu-runs.md`

- [ ] **Step 1: Comment pass.** Reread every file of the module from top to bottom as a C++ programmer who has
  never read a fluid paper: every comment true of the final code; why, not what; a worked example with numbers
  for each non-trivial public function; every word in the glossary; `Uses` and `Used by` lists correct; `Read:`
  lines resolve; `// Step N:` numbers match the four procedures.
- [ ] **Step 2: Fold each change into the commit that owns the file** (`git commit --fixup`, then
  `git rebase --autosquash`); run format and rules on each rewritten commit and the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3.1 to 3.6, 5.1 to 5.8, 6, 7) against the branch and
  record, for the hand-over, everything delivered differently: the plan decisions of this plan, the split test
  names (`WritePointsFile, ReadBack`, `WriteSurfaceFile, ReadBack`, `WriteSolidMotionFile, ReadBack` per format;
  `BuildGasVolume, ThreeStates` across Tasks 15 and 45), the library versions pinned by Tasks 4 to 9 (each the
  newest on Conan Center when pinned, A19) with every Boost or Blosc option of Task 8, the measured library
  build times and sizes, the `SurfaceSequence` numbers; Foundation as built: `StepRecorder::order_buffers` and
  `sum_buffers`, `RenderOutput::frame_order_buffers` and `RenderLimits::max_field_parts` (A31, reconcile item 26),
  the one form of `ComputeLiquidTotals` used, `fluidity_apply_compile_flags(... RTTI ON)` for
  `fluidity_interchange`, interchange tests through `fluidity_add_test` with `LABEL cpu`; (2026-11-15)
  `BuildStepRecorder(..., run_memory)`, `BuildRenderOutput` / `BuildRenderOutputFromSnapshot` with the CPU block,
  one estimate row per buffer, `RenderOutputError::kOutOfMemory` last (value 15), a writer's `std::bad_alloc` as
  `kFileWriteFailed`, `RenderSettings::wind`, `MoveFoamParticles(..., wind, ...)`, `kFoamWindDriftDuration` (A47),
  no grid-only surface input (A64; Task 50 later).
- [ ] **Step 4: `docs/gpu-runs.md`:** a row for render output, "compiled, not run", listing the `gpu` tests of Task
  32 (and naming `StepRecorderSteps, EveryThreadOrder_EqualsCpu` as the CPU evidence until the run). Run the gate; commit (subject: `Record render output as compiled, not run`).
- [ ] **Step 5: Hand over:** `git log --oneline` of the module, the gate output, every new public name with its
  reason, the list of Step 3, and the statement that no device code has run and that no witness test has run (every
  `WitnessRecord` is `kNotRun` or `kDocumented`; the Blender add-on runs them in Blender 5.3).

---

## Deep liquid without particles (later; needs review)

**Amended 2026-11-15 (A64):** new group. These tasks are excluded from the module's done-criteria: render output is
done at Task 49 without them. Owner: the cell resize and batches plan's group of the same name
(`2026-11-14-rebuild-11-cell-resize-and-batches.md`, "Deep liquid without particles (later; needs review)"); an
independent review of that design comes first, and this group runs only after that group's gate tests
(`GridOnlyRegion, StandingWave_KeepsAmplitude`, `GridOnlyRegion, ParticleBandEdge_StaysStable`) are written. In the
first version every liquid cell holds particles, so the surface reads no grid-only array.

### Task 50: Grid-only liquid in the surface field and the liquid totals (later)

**Amended 2026-11-15 (A64):** moved here from Tasks 17, 22, 31 and 45; design 5.2 step 7 and 5.3 step 1 text marked
"not in the first version (A64)".

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/surface/surface_field.h`, `surface_field_points.cc`, `surface_field_points_test.cc`,
  `surface_field_scene_test.cc`, `native/fluidity/render_output/step_recorder_cpu.cc`, `step_recorder_gpu.cu`,
  `step_recorder_test.cc`, `build_render_frame_test.cc`, `render_output_test_support.h/.cc`

**Interfaces:**
- Consumes: the cell resize group's `step::SimulationSnapshot` region arrays (`is_grid_only_region`,
  `grid_only_current_volume`, `grid_only_ingredient_volume`), `cell_resize::GridOnlyAmounts` and the
  `ComputeLiquidTotals` form that takes it.
- Produces: 5.3 step 1's grid-only field point (one point at the cell centre, `volume = grid_only_current_volume`,
  the cell's width, shares from `grid_only_ingredient_volume`) and 5.2 step 7's `GridOnlyAmounts` view in
  `RecordStep`, as the design's "not in the first version" text states.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(BuildFieldPoints, GridOnlyCell_IsOnePointAtItsCentre)   // volume == grid_only_current_volume, its shares;
                                                               // exact
  TEST(BuildSurfaceField, GridOnlyInside_NoInnerSurface)       // sphere radius 16 cells, particles in the outer 3
      // cells, grid-only inside: 1 piece, Euler number 2, exact
  TEST(BuildRenderFrame, VolumeTargetWithGridOnlyRegion)       // deep tank with a grid-only region, one group of
      // every liquid: liquid_volume_outside_surfaces equals total minus the parts, 1e-12 relative
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4: Run** the gate. **Commit.** Subject: `Show the
  surface of liquid without particles`.
