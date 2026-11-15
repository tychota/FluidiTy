# Rebuild I/O and CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the I/O and CLI module on the `rebuild` branch: the scene file and its schema, the ready scene with
its defaults filled once, the run folder whose files appear whole or not at all, FluidiTy's own array file for
saved states and kept frame inputs, the run fingerprint and the rule for continuing, the output queue and the
frame loop, and the `fluidity` program (`capabilities`, `check`, `run`, `resume`, `report`, `read-particle`) with
its events, stop requests and exit codes. The test that matters most, and comes as early as its dependencies
allow (Task 24): a run stopped at a frame and resumed lists the same bytes as the run that never stopped.
Amended 2026-11-14 (A17, batched-run buffers): before a run, `check` shows the planned cost (the budget: how the
scene runs and why, memory on each processor against what is available, copies per step, the largest modules, and
what would make a scene that does not fit fit); `run` shows it at the start and, at the end, planned against
measured with the time per stage, warning when a measure passes its budget; a projected finish appears after the
first frames. The same fields, with units in their names, are in the events and `run_report.json`.

**Architecture:** Folders `native/fluidity/io/` (namespace `fluidity::io`, part of the library target
`fluidity`) and `native/fluidity/cli/` (namespace `fluidity::cli`: OBJECT library `fluidity_cli_parts` and the
program target `fluidity_cli` with `OUTPUT_NAME fluidity`; amended 2026-11-14, Foundation as built: every one of
these targets is given the numerical flags with `fluidity_apply_compile_flags`, or the configure check fails).
Everything is CPU only; there is no `_gpu.cu` and no
shared header. No `render_output/` header is included anywhere. Schemas live in `schemas/simulate/v2/`;
`flatc` from the Conan build context generates private headers. Every product file is written by a
`FrameWriter`, an interface of function pointers; in this module only a test writer (a small text file per
output and frame) and the executable's writer that marks every product "not built" exist. The
render-output module (design 13) comes next and owns every open-format writer: no Alembic, USD or OpenVDB
library enters the build here.

**Tech Stack:** as Foundation (C++23 host, CMake presets + Ninja, Conan 2, uv, clang-format and clang-tidy 23.1
as `pyproject.toml` pins them (amended 2026-11-14, Foundation as built: replaced "19.1"), Docker via colima, Typst with CeTZ), plus, new here: FlatBuffers 25.12.19 with `flatc`, BLAKE3 1.8.5, CLI11 2.7.2
(versions pinned on `main`), and the Python `flatbuffers` package for the one reference decoder.

**Spec:** `docs/superpowers/specs/modules/12-io-and-cli-design.md` ("the design"; in the `main` checkout, never
committed; its section 3 is the contract) and the programme design `docs/design/programme.md` on `rebuild`
(sections 6, 7, 8, 8a). Also in the `main` checkout: `modules/owner-answers.md`, `modules/shared-interface.md`,
`modules/plan-decisions.md` (it wins over every design), section 3 of the step design and section 3 of the cell
resize and batches design.

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
- **Amended 2026-11-14 (Foundation as built):** Foundation's structure patterns (`AGENTS.md` "Structure";
  `plan-decisions.md` "Foundation after the final review") hold here as far as a CPU-only module meets them. This
  module has no GPU half, no `_gpu.h`, no `_steps.h` and no working buffers of its own on a processor; the two-halves, kernel-wrapper and
  working-buffers rules apply to the modules it drives. (Amended 2026-11-14, A30: a processor the user chose that
  is wrong or unavailable is still reported as an error here, because it is user input; mixing processors between
  a module's arguments is that module's precondition.) Errors: a failed check writes nothing; after a module's
  `kRunOnCallFailed` the run's outputs are unspecified and the run stops (exit code as design 5.8); `kOutOfMemory`
  comes only from building forms. Exactness: discrete outputs equal on CPU and GPU, floating-point within the
  step's design 6.4 tolerances unless a header promises the same bits unless NaN; CPU-against-CPU and
  GPU-against-GPU runs of one mode keep their byte tests. Tests link `fluidity_base_test_support` (`ToBits`,
  `BuildBufferHolding`, `ReadAll`, `CheckWithinTolerance`, `CheckSameBitsOrBothNaN`, digests) and
  `fluidity_io_test_support`, made with `fluidity_add_test_library` and named in `LINKS`; every `gpu` test calls
  `WriteResultDigest` on its main outputs. Labels: `cpu` everywhere, `no_gpu` only for behaviour on a machine
  without a GPU, `gpu` only on hardware. Every C++ target is created through the helpers of
  `cmake/fluidity_targets.cmake` (`fluidity_add_test`, `fluidity_add_test_library`) or, for the library parts and
  the program, `add_library` / `add_executable` followed at once by `fluidity_apply_compile_flags(<target>
  EXCEPTIONS <ON|OFF>)`; every source file is listed in a CMake file.
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the CPU build in the debug and
  release presets, and the CUDA build compiled in the container, whose `cuda` step fails for a tracked `.cc` or
  `.cu` file that no build compiles. clang-format and clang-tidy are the 23.1 that `pyproject.toml` pins.

The module's own:

- **Design section 3 is the contract**, with the plan decisions below and in each task. Every fallible function
  returns `std::expected<T, IoFailure>`; `Path` is `std::filesystem::path`. Every header includes
  `base/cpu_only.h` first.
- **What waits for the render-output module** (plan decision, `plan-decisions.md` "Surface, foam and spray": the
  render-output plan creates every `render_output/` header and completes these seams in its Task 47, with these
  tests extended; the list closes this plan as "Seams the render-output plan completes"). No file of this module
  includes a `render_output/` header. In this module these names and fields do not exist:
  `RequestedOutputs::render`, `RecordsSteps`; `OutputDeclaration::product`, `format`, `contract_version`;
  `SavedState::render_output`; `ProductsOfFrame::render_output`; the `kept` argument of `FrameWriter::start`
  (here `start(void* context)`); `RunParts::step_recorder`; `RunReport::interchange_libraries`; the format table,
  channel table and interchange libraries in the `capabilities` event; the `CheckRenderSettings` finding.
  `FrameOutput` and `KeptFrameInputs` carry `std::int64_t frame` and `double time` where the design has
  `render_output::FrameIdentity identity`, and carry no `recorded` and no `sample_at_frame`. `file_pattern` has
  no extension (`outputs/points_17/points.{frame:06d}`): the writer appends its own. `ComputeOutputFingerprint`
  hashes `output_id` and `material_id` where the design names `product`, `format` and the contract version.
  `make_foam_and_spray` stays null. `FailingModule::kRenderOutput` exists; its name rows come with render output.
  The scene file carries the render settings' place as a schema table this module owns: `outputs.render:
  RenderSection` in `scene_file.fbs` and the matching table in the `presentation` part of `ready_scene.fbs`, both
  empty tables here (`"render": {}` is read and recorded, nothing else).
- **Glossary renames of design names** (`plan-decisions.md`, "Names that broke the glossary"; `stored` and
  `tail` are banned words): the design's `KeptFrameInputs` is `KeptFrameInputs`, `FileList::ignored_end_bytes`
  is `ignored_end_bytes`, "torn tail" is "torn end". Other modules' renamed names are used as renamed
  (`GridBuffers`, `InternalForceTotals`, `ParticleContact`, `SolvedArrays`).
- **`RunParts` holds no cell resize and no batched run** (plan decision: both need the run plan, which `BuildRun`
  computes): `struct RunParts { FrameWriter frame_writer; OutputQueueSettings output; RunCallbacks callbacks; }`.
  `Run` builds and owns `cell_resize::CellResize` (when the scene has cell resize settings) and
  `batches::BatchedRun` (when the plan is `kBatched`) after `step::BuildRunPlan`.
- **`RunOverrides` gains `std::optional<std::uint64_t> assume_memory_bytes`** (plan decision): the processor
  memory `BuildRunPlan` is given instead of `ReadMemoryAvailable`. `check --assume-memory` sets it; tests use it to
  force a batched plan. It is not part of any fingerprint.
- **`ReadRunFolderFacts(const Path& run_folder, FileCheck file_check)`** with `enum class FileCheck : std::uint8_t
  { kListOnly, kEveryListedFile };` replaces the design's bare `bool check_files` (rules check `bare-bool`).
- **`FileKind` gains `kStartParticles`** (appended, plan decision) for the scene's `start_particles_file`.
- **Base failures** use `FailingModule::kBase` with `module_error` = the `BufferError` value, or 64 + the
  `RunOnError` value (plan decision; Foundation gives both enums a `ToName`, and the two share one module code).
- **Constants** (design values): writer lock wait 1 s (tries every 50 ms); heartbeat every `kHeartbeatSeconds =
  0.5`, never more than 1 s apart; standard input polled every 25 ms; `max_waiting_frames` 2; a saved state every
  10 frames and at `first_frame`, newest 2 kept; array-file header at most 16 MiB, arrays start at multiples of 64
  bytes; header short lists at most 65,536 items; one event line at most 16 KiB (only `message` is cut);
  `IoFailure::detail` 120 bytes; write buffer 4 MiB; `capabilities` returns within 200 ms;
  `particle_spacing = smallest_cell_width / 2`; start-particle jitter `spacing / 4 * (2 u - 1)` per axis.
  Schema versions (`io/schema_versions.h`, plan decision): every file kind and the events protocol start at 2.
- **Hash domains** (design 5.2): `"fluidity/scene/v2"`, `"fluidity/run/v2"`, `"fluidity/output/v2"`, each
  followed by one zero byte. Numbers are hashed as little-endian bytes, strings as their UTF-8 bytes followed by
  one zero byte.
- **Same bytes.** No internal file (`ready_scene`, saved states, frame inputs, the file list) holds a clock time,
  a path or a process id. Any type written raw has `std::has_unique_object_representations_v` (static_assert);
  a type with padding is split (`contact::PlateSide` becomes `plate_id` and `side`). Little-endian machines only
  (`static_assert(std::endian::native == std::endian::little)`).
- **Exceptions** are on in exactly one file, `cli/read_command_line.cc` (with RTTI). FlatBuffers generated
  headers stay private to `fluidity` and `fluidity_cli_parts`.
- **Test labels:** `LABEL cpu` for library and process tests; `LABEL gpu` for the owner's run (Task 41);
  `LABEL no_gpu` for the one test that needs a CUDA build without a GPU. Long scene runs get CTest
  `TIMEOUT 900`. Test helpers: library `fluidity_io_test_support` (namespace `fluidity::io::testing`), created with
  `fluidity_add_test_library(fluidity_io_test_support SOURCES io_test_support.cc LINKS fluidity_base_test_support
  fluidity_step_test_support)` and named in each test's `LINKS` (amended 2026-11-14, Foundation as built).
- **Names the rules check reports under `verb`** go into `tools/check_rules_allowed_names.txt` with the reason
  `decided I/O name (I/O design 3)`, in the task that adds them: `AdvanceRunTo`, `AdvanceRun`, `RebuildOutputs`,
  `DiscardOutputFile`, `RemoveOutputFiles`, `AddFrameOutput`, `CompareWithRunFolder`.
- **The planned and the measured cost** (amended 2026-11-14, A17, batched-run buffers; `plan-decisions.md`
  "Planned cost shown to the user, and buffers in batched runs" holds every declaration, template line and
  constant; Tasks 23a and 29a build them). The *budget* (`io/run_budget.h`: `RunBudget`, `BuildRunBudget`,
  `BuildBudgetLines`) is computed from `step::BuildRunPlan` (which now returns a plan that does not fit, with
  `PlanFit::kDoesNotFit`) and, in batched runs, `batches::BuildBatchPlan` or its failure; the *cost*
  (`io/run_cost.h`: `RunCost`, `ApplyTryToRunCost`, `ApplyStepToRunCost`, `ApplyFrameToRunCost`, `BuildCostLines`,
  `ComputeTimeProjection`) is measured from the step's reports. Human lines go to standard error (and `log.txt`
  for `run` and `resume`), never to standard output, which stays one JSON event per line. Schema
  `schemas/simulate/v2/run_budget.fbs`, included by `run_report.fbs` and `events.fbs`. ~~Constants
  `kMemoryWarningShare = 1.10`, `kCopyWarningShare = 1.25` (owner to confirm).~~ **Replaced (A43, A58):** no
  margin constants; a measure above its planned budget warns with the amount (`plan-decisions.md` "Owner answers
  A40-A64 applied", item 3); the one constant is `kMemoryTakenToleranceBytes = 2,097,152` (2 MiB, starting value,
  to confirm on the Windows run). Time: nothing is estimated before the run; a projection needs 3 frames listed in
  this process.
- **Amended 2026-11-15 (A40-A64):** `plan-decisions.md` "Owner answers A40-A64 applied (2026-11-15)" wins over the
  design and over earlier text here. In this plan: particles everywhere, no grid-only region (A64; its I/O pieces
  move to the later group after Task 42); artist cell sizes read from the scene (A63, Task 29b); the memory budget,
  one reserved block per run and warnings without margins (A58; Tasks 23a, 29, 29a); viscosity advice (A57, Task
  23a); guided free solids and slip length (A59, A53, Task 26); the liquid force on every solid in `frame_done` and
  `run_report.json` (A60, Task 29c); wind (A47, Task 27); resume in the recorded run mode (A40, Task 29d);
  `kOutOfMemory` last in `IoError` (item 11, Task 4). This module builds no `kRunOn` buffer: its estimate entries
  are `where = kCpu`, so it takes no `RunMemory`; it hands the run's `RunMemory&` to the builders it calls (Task 29).

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, already committed: Foundation
with every `base/` piece (`ReadMemoryAvailable`, `ReadProcessorDescription` in `base/processor_description.h`,
`MemoryEstimateEntry` with `where`), contact, grid, particles, transfer, pressure, viscosity, materials, solids, gas, step
(with `fluidity_step_test_support`: `TestScene`, `ExpectSameBytes`, the small reference scenes of step Task 36)
and cell resize and batches (with `ResizeCells`, `ApplyStagesInBatches`, `BuildCellResize`, `BuildBatchPlan`,
`BuildBatchedRun` as its design section 3 declares them), each with its comment pass done.

Prove it: `uv run python tools/check_commit.py` prints `PASS` on every line, and
`grep -ln "AdvanceTo\|ResizeCells\|ApplyStagesInBatches" native/fluidity/*/*.h` lists one header in each of
`step/`, `cell_resize/`, `batches/`. If either fails, stop and ask the owner.

**Amended 2026-11-14 (Foundation as built):** Foundation as built also provides `base/run_on_call.h`,
`base/gpu_launch.cuh`, the sums and the order with their `SumBuffers` and `OrderBuffers` (one form each, A31) that the
modules this one drives keep inside their buffers, `base/test_support.h` (library `fluidity_base_test_support`) and
`fluidity_add_test_library` with `LINKS`; read `AGENTS.md` "Structure" and "Language" (the "Targets" bullet)
before the first task. The expected gate output is seven `PASS` lines: format, rules, cpu debug, cpu release,
tidy, paper, cuda.

## Review Focus

1. **A value the next step reads is missing from the saved state** (a `CommittedStepLimits` field, a source
   counter's `owed_particles`, `position_when_grouped`, `blob_history.next_blob_id`): the resumed run drifts by
   one bit some frames later. Pinned: `ContinueRun, StopAfterFrame12_SameFilesAsStraight` and
   `ContinueRun, KilledInFrame13_SameFilesAsStraight` in Task 24, then on every material in Task 28.
2. **On Windows, the writer's lock covers the bytes it wrote into `writer.lock`,** so `report` cannot read the
   holder. The lock is taken on one byte at offset 2^40, beyond the content. Pinned:
   `ReadLockHolder, WhileHeld_ReadsProcessIdAndHost` in Task 8 (runs in the Windows CI lane).
3. **A crash between the rename and the list entry** leaves a final-named file that no list names; the next
   writer of that name must delete it, not fail with "already exists". Pinned:
   `WriteListedFile, EndedAfterRename_NextWriterDeletesLeftover` in Task 12.
4. **Uninitialised padding or a clock value in an internal file** makes two equal runs differ. Pinned:
   `WriteArrayFile, SameContentTwice_SameBytes` (Task 10, with the buffer poisoned before packing) and
   `WriteSavedState, TwoWritesOfOneSnapshot_SameDigest` (Task 19).
5. **The output worker fails while the simulation thread waits on a full queue:** the run must end with
   `kOutputStopped`, not hang. Pinned: `AddFrameOutput, WorkerFailsWhileQueueFull_OutputStoppedWithoutHanging`
   in Task 22 (CTest `TIMEOUT 30`).

---

### Task 1: Method paper, files, listing and the saved state

**Kind:** mechanical

Load the `typesetting-with-typst`, `typesetting-documents` and `drawing-with-cetz` skills first.

**Files:**
- Create: `docs/method-paper/sections/files.typ` (included after the cell resize and batches section),
  `docs/method-paper/figures/file_life.typ`, `docs/method-paper/figures/run_folder.typ`
- Modify: `docs/method-paper/fluidity-method.typ` (include), `docs/glossary.md`

**Interfaces:**
- Produces labels `<sec-files>`, `<fig-file-life>`, `<fig-run-folder>`, `<proc-write-listed-file>`.

- [ ] **Step 1: Text,** design section 8 outline items 1 to 4: one scene, two forms (the scene file the artist's
  tool writes, the ready scene the run keeps) and where defaults are filled (5.1 step 3); the run folder, open
  products and internal records (A14, design 5.3 tree); how a file gets listed and what a crash leaves (5.3); what
  a saved state holds and why each value (the table of 5.4). BLAKE3 and FlatBuffers are named without a reference
  (design decision 13). Glossary rows: `scene_file`, `ready_scene`, `run_folder`, `file_list`, `listed`,
  `saved_state`, `WriterLock`, `OutputFile`, `array file`, `kept frame inputs`, `digest`.
- [ ] **Step 2: Figures.** `<fig-file-life>` (design 8 figure a): the life of a file as a time line (temporary,
  written by bytes or by a library through a path, flushed, checked, renamed, listed) with the four crash points
  and, under each, what a reader sees and what the next writer does. `<fig-run-folder>` (figure b): the folder as a
  tree beside the file list as a strip of entries, arrows from entries to files, a torn last entry greyed out.
- [ ] **Step 3: Pseudocode** `<proc-write-listed-file>`: the seven steps of design 5.3, numbered as Task 9's code.
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`; look
  at every new page; fix overlaps. Run the gate. **Commit.** Subject: `Describe how a run's files are listed`.

### Task 2: Method paper, fingerprints, the output queue and the protocol

**Kind:** mechanical

**Files:**
- Modify: `docs/method-paper/sections/files.typ`, `docs/glossary.md`
- Create: `docs/method-paper/figures/output_queue.typ`, `docs/method-paper/figures/fingerprints.typ`

**Interfaces:**
- Produces labels `<sec-fingerprint>`, `<sec-protocol>`, `<fig-output-queue>`, `<fig-fingerprints>`,
  `<proc-advance-run-to>`.

- [ ] **Step 1: Text,** outline items 5 to 7: the run and output fingerprints, the rule for continuing (only the
  same fingerprint), the one bridge (`--start-from`), the comparison with an old run (5.2); the output queue and
  waiting (5.7); events, stop requests and exit codes (5.8, the exit-code table). Glossary rows:
  `run_fingerprint`, `output_fingerprint`, `OutputDeclaration`, `output_id`, `heartbeat`, `sent_at`,
  `last_advance_sent_at`, `waiting_frames`, `RunReport`.
- [ ] **Step 2: Figures.** `<fig-output-queue>` (figure c): two tracks, simulation and output worker, over three
  frames, a full queue and the waiting interval shaded. `<fig-fingerprints>` (figure d): the two fingerprints as a
  tree of their parts, the parts left out (names, outputs, `last_frame`, limits, cadence) drawn outside.
- [ ] **Step 3: Pseudocode** `<proc-advance-run-to>`: design 5.6 steps 1 to 8 with 7.1 to 7.6, numbered as Tasks
  22 and 23 implement them.
- [ ] **Step 4: Run** typst, look at the pages, run the gate. **Commit.** Subject:
  `Describe fingerprints, the output queue and the run protocol`.

### Task 3: FlatBuffers, BLAKE3 and CLI11 in the build

**Amended 2026-11-14 (Foundation as built):** `fluidity_cli_parts` and `fluidity_cli` get
`fluidity_apply_compile_flags(<target> EXCEPTIONS OFF)` right after `add_library` / `add_executable` (the
configure check fails for any target without it); the new test and `main.cc` are listed in the module CMake files.

**Kind:** mechanical

**Files:**
- Modify: `conanfile.py`, `conan.lock`, `CMakeLists.txt`, `cmake/fluidity_targets.cmake` (only if a helper
  must learn the new packages; no new target is created outside the helpers or without
  `fluidity_apply_compile_flags`)
- Create: `cmake/fluidity_schemas.cmake`, `native/fluidity/io/CMakeLists.txt`, `native/fluidity/cli/CMakeLists.txt`,
  `native/fluidity/io/packages_test.cc`, `native/fluidity/cli/main.cc`

**Interfaces:**
- Produces: Conan `requires = "flatbuffers/25.12.19", "blake3/1.8.5", "cli11/2.7.2"`, `tool_requires =
  "flatbuffers/25.12.19"`, `test_requires = "gtest/1.17.0"`; `generate()` sets the CMake variable
  `FLUIDITY_FLATC_EXECUTABLE` as `main`'s `conanfile.py` does. The library `fluidity` links
  `flatbuffers::flatbuffers`, `BLAKE3::blake3` and `Threads::Threads` (PRIVATE); OBJECT library
  `fluidity_cli_parts` (empty for now) and the program target `fluidity_cli` (`OUTPUT_NAME fluidity`; a
  placeholder `native/fluidity/cli/main.cc` returning 0 until Task 37), each followed by
  `fluidity_apply_compile_flags(<target> EXCEPTIONS OFF)`; `packages_test` is
  `fluidity_add_test(packages_test SOURCES packages_test.cc LABEL cpu)`. CMake function
  `fluidity_add_schema(<target> SCHEMAS <names...> [EMBED_TEXT <names...>] [BINARY_SCHEMA <names...>])` in
  `cmake/fluidity_schemas.cmake`: `flatc --cpp --scoped-enums --reflect-names` into
  `<build>/generated/simulate_v2` (a PRIVATE, SYSTEM include directory of `<target>`); `EMBED_TEXT` writes
  `<name>_schema_text.h` holding `common.fbs` followed by `<name>.fbs` without its `include` line (as
  `git show main:cmake/SimulateCli.cmake` lines 3-10) as an `inline constexpr std::array<char, N>` built from
  `file(READ ... HEX)`, not a string literal (MSVC limits literals to 16 KiB); `BINARY_SCHEMA` runs
  `flatc --binary --schema --bfbs-builtins` and embeds the `.bfbs` bytes the same way.

- [ ] **Step 1: Failing tests** (`packages_test.cc`, `LABEL cpu`):

  ```cpp
  TEST(Packages, PinnedVersions)        // FLATBUFFERS_VERSION_MAJOR/MINOR/REVISION == 25/12/19;
                                        // std::string_view(BLAKE3_VERSION_STRING) == "1.8.5"; CLI11_VERSION == "2.7.2"
  TEST(Blake3, EmptyInput_PublishedDigest)  // af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262
  TEST(Blake3, Abc_PublishedDigest)         // 6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85
  ```
- [ ] **Step 2: Run.** Expected: configure or compile error (packages missing).
- [ ] **Step 3: Implement.** Copy the four `requires` lines and the `flatbuffers` and `cmake` `build_requires`
  lines with their recipe revisions from `git show main:conan.lock`, drop `abseil`, then run
  `uv run conan lock create . --lockfile conan.lock --lockfile-out conan.lock`: the revisions must stay those of
  `main`. Port the `flatc` checks of `git show main:cmake/LiquidCacheSchemas.cmake` lines 1-9 (absolute path from
  Conan, `flatc --version` must print `flatc version 25.12.19`); never search the path. `find_package(flatbuffers
  CONFIG REQUIRED)`, `find_package(BLAKE3 CONFIG REQUIRED)`, `find_package(CLI11 2.7.2 EXACT CONFIG REQUIRED)`,
  `find_package(Threads REQUIRED)`. Imported targets are SYSTEM, so `-Werror` never sees their warnings.
- [ ] **Step 4: Run** the gate (the container build fetches the same packages). Expected: 3 new tests pass.
- [ ] **Step 5: Commit.** Subject: `Add FlatBuffers, BLAKE3 and CLI11 to the build`. Body: the versions are
  `main`'s; `flatc` comes from the Conan build context only.

### Task 4: I/O errors and the messages of every module

**Amended 2026-11-15 (A40, item 11):** `IoError` has **31** values (the old "29" miscounted the design's 30):
`kRecordedModeDoesNotFit` (A40, code `recorded_mode_does_not_fit`) follows `kModuleFailed`, and `kOutOfMemory` moves
from before `kModuleFailed` to the **last** place (reconcile item 31: appended last in every module error enum;
nothing is built yet, so no stored number changes). The base name table names Foundation's
`BufferError::kRunMemoryTooSmall` (`run_memory_too_small`); its `module_error` is the `BufferError` value as before.
`ToMessage(kRecordedModeDoesNotFit)`: "This run was recorded in <mode>, which no longer fits in memory (<needed>
needed, <allowed> allowed). Resume with --choose-mode-again to continue in another mode; the results then differ
slightly from an uninterrupted run (within the solver's tolerance)." (`<mode>` from `detail`, the needed and allowed
bytes from `magnitude` and `limit`, filled as design 5.8 fills every message.) Tests: `ToName,
EveryIoError_IsSnakeCaseWithoutK` counts 31; new `TEST(IoError, OutOfMemory_IsLastValue)`
(`static_cast<int>(IoError::kOutOfMemory) == 30` and `kRecordedModeDoesNotFit == 29`).

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/io/io_error.h`, `io_message.h`, `io_message.cc`, `io_message_test.cc`
- Modify: the CMake lines that list module folders (register `native/fluidity/io/` and `native/fluidity/cli/`
  as `native/fluidity/step/` is), `docs/glossary.md`

**Interfaces:**
- Produces design 3.1 exactly: `IoError` (31 values, `kNone` first, `kOutOfMemory` last; amended 2026-11-15),
  `FailingModule`, `ToFailingModule`,
  `FingerprintPart`, `IoFailure`, `ToMessage(IoError)`, `ToName(IoError)`, `ToMessage(FailingModule,
  std::uint8_t)`, `ToName(FailingModule, std::uint8_t)`, `ToName(FailingModule)`. Plan decisions: the base row of
  Global Constraints (64 + `RunOnError`); `ToName(FailingModule::kRenderOutput, e)` returns `"unknown"` until
  render output adds its rows; a private `void WriteDetail(IoFailure&, std::string_view)` cuts to 119 bytes plus
  the zero, never allocates.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ToName, EveryIoError_IsSnakeCaseWithoutK)   // kRunFolderInUse -> "run_folder_in_use"; all 31 distinct
  TEST(IoError, OutOfMemory_IsLastValue)           // amended 2026-11-15: kOutOfMemory == 30, kRecordedModeDoesNotFit
                                                   // == 29, kModuleFailed == 28
  TEST(ToMessage, EveryIoError_DistinctAndNotEmpty)
  TEST(ToFailingModule, EveryStepModule_SameName)       // for each of the 14 step::Module values:
                                                        // ToName(ToFailingModule(m)) is "step", "contact", ...
  TEST(ToName, EveryModuleError_UniquePairs)            // design test 17: for every enumerator of RunOnError,
      // BufferError, ContactError, GridError, ParticlesError, TransferError, PressureError, ViscosityError,
      // MaterialsError, SolidsError, GasError, StepError, CellResizeError, BatchesError and IoError: ToMessage and
      // ToName non-empty, ToName snake case, every (module, code) pair unique
  TEST(WriteDetail, LongText_CutAndTerminated)          // 300 bytes in: 119 kept, detail[119] == '\0'
  ```
- [ ] **Step 2: Run** (compile error). **Step 3: Implement.** One `switch` per enum with no `default:` (the
  compiler finds a missing enumerator); the module-pair functions call each module's `ToMessage` and `ToName`.
  Messages are one English sentence that says what the user can do.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Name every failure a user can see`. New names: the 3.1
  list above.

### Task 5: Stable names of the step's enums

**Amended 2026-11-14 (reconcile):** `ToName(step::RunMode)` has the third value; `ToName(step::Module)` (used by
Task 23a and the budget schema) is declared here.

**Amended 2026-11-14 (A24):** the snow exclusion is lifted (`CapabilityReason::kViscousLiquidWithSnow` and its name
`viscous_liquid_with_snow` no longer exist): 17 capability reasons at most, 14 if the viscosity plan lifts its three
exclusions; `kViscousLiquidWithFreeSolid` stays. **Replaced (A27): see the next amendment.**

**Amended 2026-11-14 (A27):** the free-solid exclusion is lifted as well
(`CapabilityReason::kViscousLiquidWithFreeSolid` and its name `viscous_liquid_with_free_solid` no longer exist;
`plan-decisions.md`, "A viscous liquid sticks, and free solids in the viscous solve (A26, A27)", "Exclusions"):
**16** capability reasons at most, 13 if the viscosity plan lifts its three exclusions. `SuggestedChange::kSetViscosityToZero` has no user in the
main sequence and is removed: `step::kSuggestedChangeCount` is **12** (`kNone` included) and the name
`set_viscosity_to_zero` is not in the table (the sand module adds it back with `kViscousLiquidWithSand`).

**Amended 2026-11-14 (A25):** `ToName(step::ViscousPath)` (`"pressure_solve"`, `"viscous_solve"`; used by Task 23a
and the budget schema) is declared here.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/io/stable_names.h`, `stable_names.cc`, `stable_names_test.cc`

**Interfaces:**
- Produces design 3.1 `stable_names.h`: `ToName(step::CapabilityReason)`, `ToName(step::SuggestedChange)`,
  `ToName(step::RejectReason)`, `ToName(step::Stage)`, `ToName(step::StepLimit)`, plus (plan decision, the same
  table rule) `ToName(step::RunMode)` (`"one_piece"`, `"batched"`, `"batched_with_solves_in_pieces"`; A15),
  `ToName(step::Module)` (reconcile: lower snake case of the enumerator without `k`, `kCellResize` ->
  `"cell_resize"`, `kRenderOutput` -> `"render_output"`; `step::kModuleCount` = 14 entries) and `ToName(RunOn)`
  (`"cpu"`, `"gpu"`), and (A25) `ToName(step::ViscousPath)` (`kPressureSolve` -> `"pressure_solve"`,
  `kViscousSolve` -> `"viscous_solve"`).
  One `std::array<std::string_view, step::k...Count>` per enum. There is no `kLiquidsThatDoNotMixTouch`
  (`plan-decisions.md`), no `kViscousLiquidWithSnow` (A24) and no `kViscousLiquidWithFreeSolid` (A27):
  `step::kCapabilityReasonCount` capability reasons, 16 at most (13 if the viscosity plan lifts its three
  exclusions). No `kSetViscosityToZero` (A27; the sand module adds it back with `kViscousLiquidWithSand`):
  `step::kSuggestedChangeCount` = 12 names.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ToName, EveryCapabilityReason_NamedAndUnique` (`step::kCapabilityReasonCount`, 16 at most; A24: no
  `"viscous_liquid_with_snow"`; A27: no `"viscous_liquid_with_free_solid"`);
  `ToName, EverySuggestedChange_NamedAndUnique` (12, `kUseSmallerCells` -> `"use_smaller_cells"`; A27: no
  `"set_viscosity_to_zero"`);
  `ToName, EveryRejectReason_NamedAndUnique` (23); `ToName, EveryStage_NamedAndUnique` (22,
  `kSolveViscosityAndPressure` -> `"solve_viscosity_and_pressure"`); `ToName, EveryStepLimit_NamedAndUnique` (11);
  `ToName, RunModeAndRunOn` (three run modes; A25: also the two `ViscousPath` names); `ToName, EveryModule_NamedAndUnique` (14, reconcile). Each table's size is `static_assert`ed against the step's count.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Give the step's enums the stable names events carry`.

### Task 6: Digests, the file call table and the executable's digest

**Amended 2026-11-14 (A31):** temporary file wording (no "scratch").

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/io/digest.h` (`Digest`, `RunFingerprint`, `ComputeFileDigest`,
  `ComputeExecutableDigest`), `digest.cc`, `file_calls.h` (private), `file_calls.cc`, `digest_test.cc`,
  `native/fluidity/io/schema_versions.h`

**Interfaces:**
- Produces design 3.3 `Digest`, `RunFingerprint`; design 3.4 `ComputeExecutableDigest`, `ComputeFileDigest`.
  Plan decision: they live in `io/digest.h`, which `run_folder.h` and `run_fingerprint.h` include.
- Produces (private, design test 4, the pattern of step 5.12): `struct FileCalls` of function pointers, one per
  file operation this module makes: `open_for_writing`, `write`, `flush_file`, `flush_folder`, `rename_without_replace`,
  `remove`, `open_for_reading`, `read`, `lock_without_waiting`, `create_folder`, `list_folder`, `file_size`,
  `read_executable_path`; `extern const FileCalls kFileCalls;` and `const FileCalls& ReadFileCalls();` /
  `void ApplyFileCallsForTesting(const FileCalls*)` (null restores `kFileCalls`). Every file access of `io/` goes
  through `ReadFileCalls()`. A private `class Hasher` wraps `blake3_hasher` (`Write(span)`, `ToDigest()`).
- Produces `schema_versions.h`: `kSceneFileSchema`, `kReadySceneSchema`, `kArrayFileContainerVersion`,
  `kArrayFileSchema`, `kSavedStateSchema`, `kFrameInputsSchema`, `kStartParticlesSchema`, `kFileListSchema`,
  `kRunReportSchema`, `kEventsProtocol`, all `2`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ComputeFileDigest, KnownBytes_EqualsHasherOfSameBytes` (3 MiB of a
  fixed integer pattern, read in pieces); `ComputeFileDigest, Missing_IsFileNotFound`;
  `ComputeExecutableDigest, TwiceInOneProcess_Equal`; `ComputeExecutableDigest, InjectedPath_DigestOfThatFile`
  (a test table whose `read_executable_path` returns a temporary file); `Digest, EqualityComparesAllBytes`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** Port the executable path lookup from
  `git show main:native/cli/fluidity_simulate.cc` lines 746-790 (`GetModuleFileNameW`, `/proc/self/exe`,
  `_NSGetExecutablePath`), with `std::expected` and `IoFailure` in place of Abseil. Files are read in 1 MiB pieces.
- [ ] **Step 4:** gate. **Commit.** Subject: `Hash files and route every file call through one table`.

### Task 7: The file list and its torn end

**Kind:** integration

**Files:**
- Create: `schemas/simulate/v2/common.fbs`, `schemas/simulate/v2/file_list.fbs`, `native/fluidity/io/run_folder.h`
  (the file-list part of design 3.3 only), `file_list.cc`, `file_list_entry.h` (private), `file_list_test.cc`

**Interfaces:**
- Produces design 3.3: `FileKind` (with `kStartParticles` appended), `OutputChannelInFile`, `OutputDeclaration`
  (minus the render-output fields), `ListedFile`, `NotBuiltOutput`, `FrameDone`, `FileList`, `ReadFileList`,
  `FindListedFile`.
- `common.fbs`: namespace `fluidity.simulate.v2`; `struct Vec3 { x, y, z: double; }`, `struct Digest { bytes:
  [ubyte:32]; }`, enums `FileKind`, `MaterialKind` (`liquid, gas, snow, springy_solid`), `RunOnName`
  (`cpu, gpu`). Every field has an explicit `(id: n)` and a unit comment.
- `file_list.fbs`: `union Action { Start, Output, Listed, NotBuilt, FrameDone, Removed }`, `table FileListEntry {
  sequence: ulong; action: Action; }`; `Start` holds the run fingerprint; `Removed` holds the `relative_name`.
- Produces (private, `file_list_entry.h`): `std::expected<void, IoFailure> AppendFileListEntry(const Path& file_list,
  std::span<const std::uint8_t> entry_bytes, std::array<std::uint8_t, 8>& previous_check_bytes)` and the entry
  builders; the 16-byte start is `"FLUIDITY"`, `kFileListSchema` (u32 LE), kind 1 (u32 LE). An entry is its byte
  count (u32 LE), the FlatBuffers bytes, then 8 check bytes = first 8 bytes of BLAKE3(entry bytes + previous
  check bytes); the first entry's previous check bytes are 8 zeros.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, lists built with the private builders in a temporary folder):

  ```cpp
  TEST(ReadFileList, StartThenThreeListed_ReturnsAllInOrder)   // sequence 0..3, ignored_end_bytes 0
  TEST(ReadFileList, TornEndAtEveryByte_IsListWithoutLastEntry)  // design test 5: cut at each byte of the last
      // entry: files and frames_done equal those of the list without it; ignored_end_bytes = the cut part
  TEST(ReadFileList, BrokenCheckBytes_StopsThere)               // one flipped bit in entry 2: entries 0..1 only
  TEST(ReadFileList, SequenceGap_StopsThere)
  TEST(ReadFileList, RemovedEntry_FileAbsent)                   // Listed then Removed of the same name
  TEST(ReadFileList, NewerOutputDeclaration_ReplacesOlder)      // two Output entries for "points_17"
  TEST(ReadFileList, NoList_IsNoRunInFolder)
  TEST(ReadFileList, WrongStart_IsWrongFileKind)                // and schema 3: kWrongSchemaVersion,
                                                                // magnitude 3, limit 2
  TEST(FindListedFile, SavedStateOfFrame_FoundOrNull)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** Run the FlatBuffers verifier on each entry before reading a
  field; an entry that fails verification ends the list like a bad check. `last_frame_done` is the last
  `FrameDone` of an unbroken ascending run from `first_frame`.
- [ ] **Step 4:** gate. **Commit.** Subject: `Read the append-only file list and cut its torn end`.

### Task 8: The writer lock and the run folder

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/writer_lock.h` (private), `writer_lock.cc`, `run_folder.cc`, `run_folder_test.cc`
- Modify: `native/fluidity/io/run_folder.h` (`RunFolder`, `RunFolderInputs`, `BuildNewRunFolder`,
  `BuildRunFolderToContinue`, `WriteOutputDeclarations`)

**Interfaces:**
- Produces those five names of design 3.3. Private: `class WriterLock` (exclusive OS lock on `writer.lock`),
  `struct LockHolder` is declared in Task 30; here `std::expected<LockHolderText, IoFailure> ReadLockHolderText(const
  Path& run_folder)` returns the process id, host name and start time written in the file, and
  `ProbeWriterLock(const Path&) -> std::expected<bool /*is_held*/, IoFailure>` (shared lock without waiting,
  released at once). Plan decisions: the lock is on one byte at offset 2^40 (`LockFileEx` / `flock` on POSIX,
  which locks the whole file) so the text at offset 0 stays readable on Windows; the writer tries every 50 ms for
  1 s, then `kRunFolderInUse` with the holder's process id in `magnitude` and host in `detail`; the text is
  `<process id> <host> <start time in s UTC with 17 digits>` on one line.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; a second `RunFolder` in the same process conflicts, because both
  OS locks are per open file):

  ```cpp
  TEST(BuildNewRunFolder, EmptyFolder_StartEntryAndLock)        // ReadFileList: one Start with the fingerprint
  TEST(BuildNewRunFolder, CallersFileLeftAlone)                 // scene.json put there first: untouched, not listed
  TEST(BuildNewRunFolder, FolderWithList_IsRunFolderNotEmpty)
  TEST(BuildNewRunFolder, SecondWhileFirstLives_IsInUseAfterAboutOneSecond)  // 0.9 s <= wait <= 2 s; folder bytes
                                                                             // unchanged
  TEST(BuildRunFolderToContinue, OtherFingerprint_IsWrongRunFingerprint)    // differing_part left kNone here
  TEST(BuildRunFolderToContinue, TornEnd_CutBack)              // file size afterwards ends at the last good entry
  TEST(BuildRunFolderToContinue, TmpAndUnlistedFilesDeleted)    // tmp/x, saved_states/000099.state not listed:
      // deleted; a listed file and scene.json at the root kept
  TEST(ReadLockHolder, WhileHeld_ReadsProcessIdAndHost)         // Review Focus 2
  TEST(ProbeWriterLock, HeldThenReleased)                       // true while a RunFolder lives, false after
  TEST(WriteOutputDeclarations, RebuiltOutput_ReplacesDeclaration)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `git show main:native/src/liquid_cache_files_3d.cc`
  lines 125-138 (root and lock) and `git show main:native/src/liquid_cache_files_3d_internal.h` lines 116-138
  (`LockFileEx` / `flock`, contention without waiting). Only `tmp/`, `saved_states/`, `frame_inputs/` and `outputs/`
  are ever cleaned.
- [ ] **Step 4:** gate. **Commit.** Subject: `Hold one writer per run folder`. New names: the five above.

### Task 9: `OutputFile` and `WriteListedFile`

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/output_file.cc`, `output_file_test.cc`
- Modify: `native/fluidity/io/run_folder.h` (`FileToList`, `OutputFile`, `BuildOutputFile`, `CheckAfterWrite`,
  `WriteListedFile`, `DiscardOutputFile`, `WriteNotBuilt`, `WriteFrameDone`, `CheckListedFile`), `run_folder.cc`

**Interfaces:**
- Produces those names of design 3.3. `WriteListedFile` follows `<proc-write-listed-file>` with `// Step N:`
  comments 1 to 7. Here `CheckAfterWrite::kSizeOnly` is complete; `kHeaderAndSizes` and `kReadBackDigest` call a
  private hook that Task 10 fills (here they return `kBadInput` for a file that is not an array file).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(OutputFile, WhileWriting_ReaderSeesNoFileAndNoEntry)    // design test 21 (first part): ReadFileList and the
      // final path during Write; after WriteListedFile both are there, digest == ComputeFileDigest
  TEST(OutputFile, ThroughPath_DigestReadOnce)                 // write via path_while_writing() with std::ofstream
  TEST(WriteListedFile, SameNameSameBytes_ReturnsListedEntry)  // listed once; temporary file gone
  TEST(BuildOutputFile, SameNameOtherBytesListed_IsFileAlreadyListed)
  TEST(BuildOutputFile, NameOutsideFolder_IsBadInput)          // "../x", an absolute path
  TEST(BuildOutputFile, UndeclaredOutputId_IsBadInput)
  TEST(WriteListedFile, ZeroBytesSizeOnly_IsFileDamaged)
  TEST(DiscardOutputFile, NothingListedNothingLeft)
  TEST(WriteFrameDone, OutOfOrder_IsBadInput)                  // frame 3 after frame 5
  TEST(WriteNotBuilt, EntryReadBack)
  TEST(CheckListedFile, OneFlippedBit_IsDigestDiffers)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `git show main:native/src/liquid_cache_store_3d.cc`
  lines 703-832 (`PublishFrame`) and `liquid_cache_files_3d.cc` lines 224-244 (`Promote`: `link` then `unlink` on
  POSIX, `MoveFileExW` with `MOVEFILE_WRITE_THROUGH` and no replace on Windows; flush the folder). The temporary
  name is `tmp/<final name with / replaced by _>.<process id>`.
- [ ] **Step 4:** gate. **Commit.** Subject: `List a file only after it is whole on disk`. New names: the nine
  above.

### Task 10: The array file

**Amended 2026-11-14 (A31):** Review Focus 4 comment says packing memory.

**Kind:** integration

**Files:**
- Create: `schemas/simulate/v2/array_file.fbs`, `native/fluidity/io/array_file.cc`, `array_file_test.cc`
- Modify: `native/fluidity/io/run_folder.h` (`ElementType`, `ArrayToWrite`, `ArrayFileContent`, `WriteArrayFile`,
  `ArrayInFile`, `ArrayFile`, `ReadArrayFile`, `ReadArray`), `output_file.cc` (the two array checks)

**Interfaces:**
- Produces those names of design 3.3 and the layout of design 5.3 (`"FLUIDITY"`, container version 2, header
  byte count, the `ArrayFileHeader` FlatBuffers buffer, zeros to a multiple of 64, arrays at multiples of 64).
  `ArrayFileHeader` (`array_file.fbs`): `kind`, `schema_version`, `run_fingerprint`, `frame`, `time`,
  `state_version`, `step_count`, `description: [ubyte]` (the kind's own table, a nested buffer), `arrays:
  [ArrayEntry]`. Plan decision: every array's bytes are hashed as written, and a file whose header passes but
  whose size is not the last array's end is `kFileDamaged`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; design test 1):

  ```cpp
  TEST(WriteArrayFile, EveryElementType_RoundTripsBitForBit)  // float64 with a NaN payload and -0.0, uint64 max,
                                                              // uint32, int8 -128, uint8, float32 denormal
  TEST(WriteArrayFile, EmptyArray_RoundTrips)
  TEST(WriteArrayFile, TwoToTheTwentyPlusOneItems_RoundTrips) // 2^20 + 1 Vec3 items
  TEST(WriteArrayFile, NameOf255Bytes_RoundTrips)
  TEST(WriteArrayFile, SameContentTwice_SameBytes)            // Review Focus 4: packing memory filled with 0xA5
                                                              // before each write; file digests equal
  TEST(ReadArrayFile, CutAtEvery64ByteBoundary_IsFileDamaged)
  TEST(CheckListedFile, OneFlippedBitAnywhere_IsDigestDiffers) // a flip at 50 spread positions
  TEST(ReadArray, WrongSize_IsBadInput)                       // and an unknown name
  TEST(ReadArrayFile, HeaderOver16MiB_IsFileDamaged)
  TEST(WriteListedFile, ReadBackDigestOnArrayFile_Listed)     // kReadBackDigest and kHeaderAndSizes both list it
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** Writes go through a 4 MiB buffer; the reader verifies the
  header buffer with the FlatBuffers verifier and checks every array's offset, extent and alignment against the
  file size before any value is read.
- [ ] **Step 4:** gate. **Commit.** Subject: `Add FluidiTy's array file for what no open format fits`.

### Task 11: Saved-state retention and output removal

**Kind:** mechanical

**Files:**
- Modify: `native/fluidity/io/run_folder.h` (`ApplySavedStateRetention`, `RemoveOutputFiles`), `run_folder.cc`,
  `run_folder_test.cc`

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ApplySavedStateRetention, FiveListedKeepTwo_ThreeRemovedOldestFirst`
  (a `Removed` entry before each delete; `FindSavedState` semantics checked through `FindListedFile`);
  `ApplySavedStateRetention, OneListed_KeptEvenWithZero` (there is always one);
  `RemoveOutputFiles, OnlyNamedOutputsRemoved` (points_17 removed, points_18 and saved states untouched).
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `git show main:native/src/liquid_cache_store_3d.cc`
  line 459 (prune). A delete that fails after its `Removed` entry leaves an unlisted file that the next
  `BuildRunFolderToContinue` deletes.
- [ ] **Step 4:** gate; the rules check may report `RemoveOutputFiles`: add it to the allowed names. **Commit.**
  Subject: `Keep the newest saved states and remove rebuilt outputs`.

### Task 12: The listing transaction under injected faults

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/listing_faults_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`; design test 4). For each call of steps 1 to 7 of `<proc-write-listed-file>`
  (open, write, flush file, read back, rename, append entry, flush list, flush folder) and each of three writers
  (`WriteArrayFile`, a write through `path_while_writing()`, `DiscardOutputFile`):
  - **Fails at that call:** a `FileCalls` table whose call returns an error at that call.
  - **Ends the process at that call:** `EXPECT_EXIT` (death test, `GTEST_FLAG_SET(death_test_style,
    "threadsafe")`) with a table whose call runs `std::_Exit(0)`.

  ```cpp
  TEST(WriteListedFile, FailsAtEveryCall_ListBeforeOrAfter)     // ReadFileList is the list before, or the list
      // with the new file; every listed file passes CheckListedFile
  TEST(WriteListedFile, EndsAtEveryCall_NextWriterCleans)       // after the child ends: same as above, and
      // BuildRunFolderToContinue leaves no unlisted file under tmp/, saved_states/, frame_inputs/, outputs/
  TEST(WriteListedFile, EndedAfterRename_NextWriterDeletesLeftover)  // Review Focus 3: the final-named file is
      // deleted by the next writer, and writing the same name then succeeds
  TEST(WriteListedFile, NoSpaceLeft_IsNoSpaceLeftAndNothingListed)   // ENOSPC / ERROR_DISK_FULL from write
  ```
- [ ] **Step 2: Run.** Fix the owning commits (rebase) if a case fails, not the test. **Step 3:** gate.
  **Commit.** Subject: `Prove a crash never leaves a half-listed file`.

### Task 13: The scene file: first tables, strict reading, the run settings

**Kind:** integration

**Files:**
- Create: `schemas/simulate/v2/scene_file.fbs`, `native/fluidity/io/scene_text.h` (private), `scene_text.cc`,
  `flatbuffers_behaviour_test.cc`, `scene_text_test.cc`, `native/fluidity/io/testdata/scenes/` (folder)
- Modify: the `fluidity` CMake lines of `io/` (`fluidity_add_schema(fluidity SCHEMAS common scene_file EMBED_TEXT
  scene_file)`)

**Interfaces:**
- **Step 1 checks FlatBuffers behaviour the design flagged as unverified ("to verify")** before anything here
  depends on it. Record the answers in the header comment of `scene_text.cc` and keep the test.
- `scene_file.fbs` (plan decision): root `SceneFile`. Every settings table mirrors its C++ struct field by field
  with the same snake_case names; **every scalar is optional** (`x: double = null;`), so an absent value is told
  apart from a given one and the C++ default is filled and recorded (design 5.1 step 3). Tables in this task:
  `schema_version`, `frames` (`first_frame`, `last_frame`, `frames_per_second`, `start_time`), `domain`
  (`low_corner`, `high_corner`, `closed` sides as `contact::ClosedSides`), `gravity`, `random_seed`, `run_on`,
  `grid` (`smallest_cell_width`, `size_level_count`), `step` (`step::StepSettings` without `run_on`), `transfer`,
  `pressure_solve`, `air_pockets`, `volume_correction`, `viscosity` (the five settings structs of `ReadyScene`),
  `outputs` (`points { materials: [ulong]; particle_fields: [string]; }`, `solid_motion: bool = null`,
  `keeps_frame_inputs`, `saved_state_every_frames`, `saved_states_kept`). Later tasks add tables: materials (14),
  start regions (15), sources (25), solids (26), gas and open sea (27), cell resize and batches (29).
- Produces (private): `std::expected<SceneText, IoFailure> ReadSceneText(std::span<const char> json)` (parser
  loaded once from the embedded schema text; `strict_json = true`, `skip_unexpected_fields_in_json = false`; then
  the verifier) and `std::string ToScenePath(...)` helpers that build paths like `solids[2].path.keys[4].time`.

- [ ] **Step 1: To verify against FlatBuffers 25.12.19** (`flatbuffers_behaviour_test.cc`, `LABEL cpu`, a tiny
  schema `table T { a: double = null; b: [ubyte]; }`):

  ```cpp
  TEST(FlatBuffersParser, UnknownField_RefusedWithLineAndName)  // {"a": 1, "c": 2}: Parse fails; message holds
                                                                // "c" and the line number
  TEST(FlatBuffersParser, OptionalScalar_AbsentAndGivenDiffer)  // {} -> a() empty; {"a": 0} -> a() == 0
  TEST(FlatBuffersParser, StrictJson_WhatIsRefused)             // one case each: unquoted key, trailing comma,
      // comment, single quotes, nan, inf, hex 0x10, duplicate key, a number with a leading +; assert what the
      // pinned version does and print the table
  ```
  If any of unquoted key, trailing comma, comment, single quotes or duplicate key is **taken**, add (in this
  task, before Step 2) `scene_text.cc`'s private `CheckStrictJsonSyntax`: an RFC 8259 syntax pass that builds
  nothing, refuses those forms and duplicate keys with line and reason as `kSceneTextNotValid`, and runs before the
  parser. `nan` and `inf` may parse: the value check (`kSceneValueOutOfRange`) refuses them. If the unknown-field
  message does not hold the field's path, `detail` holds the parser's message (name and line) and the hand-over
  records it.
- [ ] **Step 2: Failing tests** (`scene_text_test.cc`, `LABEL cpu`):

  ```cpp
  TEST(ReadSceneText, MinimalScene_Parses)                    // testdata/scenes/minimal_run_settings.json
  TEST(ReadSceneText, NotJson_IsSceneTextNotValidWithLine)    // detail starts with the line number
  TEST(ReadSceneText, UnknownField_IsUnknownSceneField)       // "domain": {"lowcorner": ...}
  TEST(ReadSceneText, EveryRefusedForm_IsSceneTextNotValid)   // the forms Step 1 found taken, through the check
  ```
- [ ] **Step 3: Run** (fail). **Step 4: Implement.** **Step 5:** gate. **Commit.** Subject:
  `Read the scene file strictly against its schema`. Body: the Step 1 findings, one line each.

### Task 14: Materials and the preset table

**Amended 2026-11-14 (A25):** the water preset's viscosity is 0 Pa s (was 1e-3): a water scene uses the pressure
solve (stage 12).

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/material_presets.h`, `material_presets.cc`, `scene_materials.cc` (private),
  `scene_materials_test.cc`
- Modify: `schemas/simulate/v2/scene_file.fbs` (`materials: [SceneMaterial]` with `id`, `name`, `kind`, `preset:
  string`, `ingredients: [SceneIngredient]`, the `MaterialResponse` fields, `stiffness`, `squeeze_ratio`,
  `own_motion_kept`, `same_material_as`; `material_pairs`; `material_settings`)

**Interfaces:**
- Produces `io/material_presets.h` (design 5.1 step 3, decision 12): `enum class MaterialPreset : std::uint8_t {
  kCustom, kWater, kSyrup, kSnow, kSpringySolid, kAir };` and `struct MaterialPresetValues` with
  `inline constexpr std::array<MaterialPresetValues, 6> kMaterialPresets`. Values: water 1000 kg/m3, 0 Pa s
  (A25; was 1e-3), 0.072 N/m; syrup 1400 kg/m3, 10 Pa s, 0.08 N/m (owner Q50); snow 300 kg/m3, `stiffness_against_shear` 1200 Pa,
  `stiffness_against_squeeze` 700 Pa, `max_springy_shortening` 0.025, `max_springy_lengthening` 0.01,
  `hardening` 4; springy solid 1000 kg/m3, 1000 Pa, 1000 Pa; air: the gas law defaults of `gas::GasLaw`
  (snow and springy numbers from `git show main:blender/fluidity/__init__.py` lines 77-82 and 132-140; open to the
  owner's review). `std::optional<MaterialPreset> FindMaterialPreset(std::string_view name)`.
- Produces (private): the material part of the derivation (design 5.1 step 4, "Materials"): one
  `materials::Material` per scene material; stiffness `E` and squeeze ratio `nu` give `E / (2 (1 + nu))` and
  `E nu / ((1 + nu)(1 - 2 nu))`; `own_motion.kept` = `own_motion_kept` (no 0.99); `ingredient_export_names` =
  ingredient names; a gas ingredient's rest density = `gas::ComputeAmbientDensity(law)`; then
  `materials::BuildMaterialTable`. Material ids and ingredient ids nonzero and unique (`kDuplicateId` with path).

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `FindMaterialPreset, EveryName_FoundAndUnknownEmpty`;
  `ReadSceneMaterials, WaterPresetOnly_FilledFromPreset` (three `FilledDefault` rows with source `kPreset`; A25: the
  filled viscosity is 0);
  `ReadSceneMaterials, FileValueOverridesPreset`; `ReadSceneMaterials, StiffnessAndSqueezeRatio_GiveTheTwoStiffnesses`
  (E 3000, nu 0.25: 1200 and 1200, `1e-15` relative); `ReadSceneMaterials, UnknownPreset_IsUnknownPreset`;
  `ReadSceneMaterials, DuplicateIngredientId_IsDuplicateIdWithPath` (`materials[1].ingredients[0].id`);
  `ReadSceneMaterials, SharedExportName_IsModuleFailedFromMaterials`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Turn scene materials into the material table`. New names: `MaterialPreset`, `MaterialPresetValues`,
  `kMaterialPresets`, `FindMaterialPreset`.

### Task 15: Start regions and explicit start particles

**Amended 2026-11-14 (A31):** temporary `RunFolder` wording.

**Amended 2026-11-14 (Foundation as built):** `WriteStartParticlesFile` goes into `fluidity_io_test_support`, which
this task creates with `fluidity_add_test_library` (Task 22 adds to it); the bit tests use `ToBits`.

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/start_particles.cc` (private), `start_particles_test.cc`,
  `native/fluidity/io/io_test_support.h`, `io_test_support.cc` (library `fluidity_io_test_support`:
  `fluidity_add_test_library(fluidity_io_test_support SOURCES io_test_support.cc LINKS fluidity_base_test_support
  fluidity_step_test_support)`)
- Modify: `native/fluidity/io/CMakeLists.txt` (the library, `start_particles.cc`, the test with `LABEL cpu LINKS
  fluidity_io_test_support`), `schemas/simulate/v2/scene_file.fbs` (`start_regions: [StartRegion]` with `id`, `name`, `material_id`,
  `shape` (`box | ball`), `center`, `orientation` (w, x, y, z), `half_size`, `radius`, `velocity`,
  `ingredient_shares`, `same_material_as`; `start_particles_file: string`, `start_particles_file_digest: Digest`)

**Interfaces:**
- Produces (private) design 5.1 step 4 "Start particles": lattice of `particle_spacing = smallest_cell_width / 2`
  over the domain box, point `(i, j, k)` at `low_corner + (index + 0.5) * spacing`, lattice index `i + nx (j + ny
  k)`, visited in ascending index (z outermost, x fastest). A point is kept when the **unjittered** point is inside
  the shape and `contact::ComputeDistanceToSolid` gives at least `spacing / 2` to every solid (plan decision: the
  count does not depend on the seed). Jitter: `words = step::ComputeRandomWords(lattice_index, region_id,
  random_seed)`, axis `a` moves by `spacing / 4 * step::ComputeSignedUnitValue(words.word[a])`. `rest_volume =
  spacing^3`, `mass = rest_density * rest_volume` with the mixture rest density of the region's shares. Each start
  region of snow or a springy solid gets its own material and `material_id` unless `same_material_as` names one.
- `start_particles_file`: an array file of kind `kStartParticles` with `particles.<material_id>.position`,
  `.velocity_for_next_step` and optional `.rest_volume`, `.mass`; its digest must equal the scene's
  (`kDigestDiffers`). Test helper (`fluidity_io_test_support`): `WriteStartParticlesFile(const Path& file,
  std::span<const ExplicitParticle> particles)` (writes through a temporary `RunFolder` and copies the listed file
  out).

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `SampleStartRegion, BoxOfOneCell_EightParticles`;
  `SampleStartRegion, SameSeed_SameBytes` (positions compared with `ToBits`); `SampleStartRegion,
  OtherSeed_SameCountOtherPositions`;
  `SampleStartRegion, JitterWithinQuarterSpacing`; `SampleStartRegion, PointsInsideSolidLeftOut` (a box solid
  covering half the region: half the count); `SampleStartRegion, BallVolume` (count x spacing^3 within 5 % of
  4/3 pi r^3 for r = 10 spacings); `SampleStartRegion, SnowRegionsGetTheirOwnMaterial`;
  `ReadStartParticlesFile, WrongDigest_IsDigestDiffers`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Sample start particles from the scene's regions`.

### Task 16: `CheckSceneFile`, `BuildRunScene` and the filled defaults

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/run_scene.h`, `run_scene.cc`, `run_scene_test.cc`,
  `native/fluidity/io/testdata/scenes/falling_block.json`, `required_only.json`

**Interfaces:**
- Produces design 3.2: `FrameRange`, `ComputeFrameTime`, `RequestedOutputs` (minus `render`), `ObjectRole`,
  `ObjectName`, `DefaultSource`, `FilledDefault`, `RunOverrides` (with `assume_memory_bytes`), `RunScene` (every
  accessor; `ready_scene_bytes()` and `scene_digest()` are filled in Task 17), `CheckSceneFile`, `BuildRunScene`.
  `ReadRunScene` comes in Task 17.
- Derivations here (design 5.1 step 4): grid frame origin = domain low corner, `biggest_cell_width =
  smallest_cell_width * 2^(size_level_count - 1)`, `new_cell_size_level = size_level_count - 1`; `end_time =
  ComputeFrameTime(last_frame)`; `snapshots_kept_at_once = max_waiting_frames + 2`; `run_on` from the file or
  `--run-on` into `StepSettings`, `SolidsSettings` (and later `CellResizeSettings`); material sets: every liquid in
  one set, the gas material alone, each other material its own set, in ascending lowest material index
  (shared-interface (d)). `RunScene` owns every vector and key array `step::ReadyScene` borrows and is move-only
  (its arrays behind one `std::unique_ptr`, so moving keeps the spans valid).
- `falling_block.json`: closed box [0, 0.4]^3, smallest cell 0.05 m, one level, water preset (material 1,
  ingredient 1), start region box [0.1, 0.3]^2 x [0.2, 0.3], frames 0 to 20 at 24 fps, points of material 1.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeFrameTime, FrameTen)                      // first 0, 24 fps, start 0.5: 0.5 + 10 / 24, exact
  TEST(BuildRunScene, FallingBlock_ReadySceneValues)    // 256 start particles; grid frame; one liquid set
  TEST(BuildRunScene, RequiredOnly_EveryOptionalFieldOnceInFilledDefaults)  // design test 7 (first part): the
      // scene_path of every optional field the schema has (reflection-free: the list of paths the reading code
      // visited), each exactly once, with kEngineDefault or kPreset
  TEST(BuildRunScene, RunOnOverride_SetsEveryRunOn)     // --run-on gpu: StepSettings and SolidsSettings both kGpu;
                                                        // a FilledDefault with kCommandLine
  TEST(CheckSceneFile, TwoProblems_BothReportedInFileOrder)  // a negative frames_per_second and a duplicate
                                                             // material id
  TEST(CheckSceneFile, PointsOfGasMaterial_IsValueOutOfRange) // outputs.points.materials names gas
  TEST(BuildRunScene, MovedRunScene_SpansStillValid)
  TEST(BuildRunScene, MissingFile_IsFileNotFound)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Fill the ready scene's defaults once and record them`. New names: the 3.2 list above.

### Task 17: The ready scene record and `ReadRunScene`

**Kind:** integration

**Files:**
- Create: `schemas/simulate/v2/ready_scene.fbs`, `native/fluidity/io/ready_scene_record.cc` (private),
  `ready_scene_record_test.cc`, `native/fluidity/io/testdata/scenes/every_default_written.json`
- Modify: `native/fluidity/io/run_scene.h` (`ReadRunScene`), `run_scene.cc`

**Interfaces:**
- `ready_scene.fbs` (plan decision): `attribute "scene_path";` root `ReadyScene { physical: [ubyte]
  (nested_flatbuffer: "PhysicalScene"); presentation: [ubyte] (nested_flatbuffer: "PresentationScene"); }`, so each
  part is one contiguous buffer and `scene_digest = BLAKE3("fluidity/scene/v2" + 0 + physical bytes)`. Every
  field carries `(scene_path: "...")`; a derived field names its source (`particles_per_second` names
  `sources[].volume_rate`). Built with `ForceDefaults(true)`, in a fixed call order. `presentation` holds names,
  requested outputs, `last_frame`, limits, cadence and `filled_defaults`. The start particles are arrays of the
  listed `ready_scene` array file (kind `kReadyScene`, `description` = these bytes); `ready_scene_bytes()` is the
  description.
- **To verify first** (Step 1): the verifier passes a buffer whose `physical` holds a nested buffer, and
  `flatbuffers::GetRoot<PhysicalScene>` on the vector's data works after `VerifyNestedFlatBuffer`.

- [ ] **Step 1: To verify** (`flatbuffers_behaviour_test.cc`): `TEST(FlatBuffersVerifier, NestedBuffer_Verified)`
  (a corrupted nested buffer fails the verification of the outer one).
- [ ] **Step 2: Failing tests** (`LABEL cpu`; design test 7, second part):

  ```cpp
  TEST(ReadRunScene, WrittenReadWrittenAgain_SameBytes)      // falling_block: ready_scene_bytes equal
  TEST(BuildRunScene, EveryDefaultWritten_SameSceneDigest)   // required_only.json and every_default_written.json
  TEST(BuildRunScene, NamesChanged_SameSceneDigest)          // and outputs changed, last_frame changed
  TEST(ReadRunScene, NoDefaultFilledAgain)                   // filled_defaults read back equal; none added
  TEST(ReadRunScene, StartParticlesRestoredBitForBit)
  ```
- [ ] **Step 3: Run** (fail). **Step 4: Implement.** **Step 5:** gate. **Commit.** Subject:
  `Record the ready scene with every default written out`. New names: `ReadRunScene`.

### Task 18: Fingerprints and output declarations

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/run_fingerprint.h`, `run_fingerprint.cc`, `run_fingerprint_test.cc`
- Modify: the `fluidity` CMake lines of `io/` (`BINARY_SCHEMA ready_scene`)

**Interfaces:**
- Produces design 3.4: `RunFingerprintParts`, `ComputeRunFingerprint`, `FindFirstDifference`,
  `ComputeOutputFingerprint` (plan decision: hashes `output_id` and `material_id` plus the product settings
  bytes), `BuildOutputDeclarations` (here: `points_<material_id>` per points material, object name `"points"`,
  and `solid_motion` with one object name `solid_<solid_id>` per solid when asked; channels empty;
  `needs_simulation_to_rebuild` = `!keeps_frame_inputs`). Product settings bytes for points: `material_id` and the
  `FieldList` bits as 8 bytes LE each; for solid motion: none. `FindFirstDifference` order: scene, executable,
  run on, processor, run mode.
- **To verify first** (Step 1): the `.bfbs` of `ready_scene.fbs` keeps the user attribute `scene_path` on every
  field (`reflection::Field::attributes()`), and `flatbuffers::SetAnyFieldF` / `SetAnyFieldI` change a field
  written with `ForceDefaults(true)`.

- [ ] **Step 1: To verify** (`flatbuffers_behaviour_test.cc`): `TEST(FlatBuffersReflection,
  UserAttributeKept)`, `TEST(FlatBuffersReflection, SetAnyFieldOnForcedDefault)`. If the attribute is dropped,
  pass `--bfbs-comments --bfbs-builtins` and check again; record the flags.
- [ ] **Step 2: Failing tests** (`LABEL cpu`; design test 8 and the fingerprint rule):

  ```cpp
  TEST(SceneDigest, EveryPhysicalFieldChangesIt)       // reflection over PhysicalScene, recursing into tables and
      // the first element of every vector (every_default_written.json has every vector non-empty): each scalar
      // changed by one ulp (floats) or + 1 (integers) changes scene_digest
  TEST(SceneDigest, NoPresentationFieldChangesIt)      // the same walk over PresentationScene
  TEST(SceneDigest, EveryFieldHasAScenePath)           // a field without the attribute fails, naming it
  TEST(ComputeRunFingerprint, EachPartChangesIt)       // the six parts, one at a time
  TEST(FindFirstDifference, ReportsThePart)            // kScene, kExecutable, kRunOn, kProcessor, kRunMode
  TEST(BuildOutputDeclarations, PointsOfTwoMaterialsAndSolidMotion)  // ids, patterns, object names
  TEST(ComputeOutputFingerprint, ParticleFieldsChangeOnlyThatOutput)
  ```
  The "neither list" case of design test 8 holds by construction: every field is in `physical` or
  `presentation`.
- [ ] **Step 3: Run** (fail). **Step 4: Implement.** **Step 5:** gate. **Commit.** Subject:
  `Fingerprint a run and declare its outputs`. New names: the 3.4 list above.

### Task 19: The saved state

**Amended 2026-11-14 (reconcile):** ~~`WriteSavedState, RegionArrays_RoundTrip` marked blocked by the open decision on
a springy solid beside a viscous liquid (expectation unchanged).~~ Option C chosen (A24): test unblocked.
`WriteSavedState, RegionArrays_RoundTrip` runs as written (the joint viscous solve carries the material faces).

**Amended 2026-11-14 (Foundation as built):** `saved_state_test` links `fluidity_io_test_support` (and through it
`fluidity_step_test_support` for `ExpectSameBytes`, a same-processor comparison); `SnapshotPackingBuffers` is stated
to be CPU-only packing memory, not processor working buffers of the `AGENTS.md` pattern.

**Amended 2026-11-14 (A31):** `SnapshotArraysScratch` renamed `SnapshotPackingBuffers` (parameter
`packing_buffers`); `ToSnapshotArrays` has one form, which takes it.

**Amended 2026-11-15 (A64, A63):** no grid-only region: "the seven region arrays" become the three merge-wait
arrays of `step::GridFields` by active cell, in this order: `grid.merge_ready_duration` (double, s),
`grid.held_velocity_gradient_size` (double, 1/s), `grid.merge_ready_step_count` (unsigned 8); `is_grid_only_region`,
`layers_from_grid_only_region`, `grid_only_mass`, `grid_only_rest_volume`, `grid_only_current_volume` and
`grid_only_ingredient_volume` are not written (later group after Task 42). `step_limits` holds every
`CommittedStepLimits` field as before, now with `largest_liquid_speed` and without `grid_only_speed`. With
`kFrameInputs` the packing writes `grid.active_cells` and no grid-only array. `WriteSavedState,
RegionArrays_RoundTrip` is renamed `WriteSavedState, MergeWaitArrays_RoundTrip` (same scene; the three arrays
read back equal with `ToBits`, through `ExpectSameBytes` like every other array).

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/saved_state.h`, `saved_state.cc`, `snapshot_arrays.h` (private), `snapshot_arrays.cc`,
  `saved_state_test.cc`
- Modify: `schemas/simulate/v2/array_file.fbs` (`SavedStateDescription`)

**Interfaces:**
- Produces design 3.5 `SavedState` (without `render_output`), `WriteSavedState` (`kReadBackDigest`, name
  `saved_states/<frame:06d>.state`), `ReadSavedState`, `FindSavedState`.
- Produces (private, plan decision on array names and order; shared with Task 20's digest):
  `std::vector<ArrayToWrite> ToSnapshotArrays(const step::SimulationSnapshot&, SnapshotPacking packing,
  SnapshotPackingBuffers& packing_buffers)` with `enum class SnapshotPacking { kSavedState, kFrameInputs }` (amended
  2026-11-14, Foundation as built: `SnapshotPackingBuffers` is a private, move-only struct of CPU byte vectors that
  the output worker keeps across frames so packing does not allocate per frame once warm; it never holds a
  processor buffer, so `Build<Name>Buffers(run_on, max_count)` and a `RunOn` check do not apply; the hand-over
  reports its largest size beside `BuildIoMemoryEstimate`).
  Order:
  per material in material index order `particles.<material_id>.<member>` for every member of
  `particles::ParticlesSnapshot::Material` present, in declaration order (`id`, `rest_volume`, `size_level`,
  `ingredient_fractions`, `mass`, `position`, `velocity_for_next_step`, `fitted_velocity_change` (9 per item, row
  by row), `volume_ratio`, `flags`, `springy_deformation`, `permanent_volume_change`, `plate_id`, `side`,
  `position_when_grouped`); then `grid.active_cells`, `grid.pressure`, `grid.has_pressure`, `grid.pocket_of_cell`,
  the ~~seven region arrays~~ three merge-wait arrays (amended 2026-11-15, A64) `grid.<member name>`,
  `grid.velocity_after_pressure`, `grid.gas_velocity_after_pressure`;
  then `gas.cells` and the seven `gas.<member name>` of `gas::GasAmountsSnapshot`. Header tables of
  `SavedStateDescription`: time, `state_version`, `step_count`, `grid_version`, `next_particle_id`, `is_full`, air
  pockets, `blob_history`, source counters, `step_limits` (every `CommittedStepLimits` field), `solids` (every
  member of `solids::SolidsSnapshot`), `limits`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; snapshots from `fluidity_step_test_support` scenes after 5 committed
  `step::TryStep` calls, compared with `step::testing::ExpectSameBytes`):

  ```cpp
  TEST(WriteSavedState, FallingBlock_RoundTrip)                  // ReadSavedState gives ExpectSameBytes
  TEST(WriteSavedState, SnowAndJelly_RoundTripWithPlateSide)     // BuildSnowAndJellyScene
  TEST(WriteSavedState, GasBesideLiquid_RoundTripWithGasArrays)  // BuildGasBesideLiquidScene(true)
  TEST(WriteSavedState, MergeWaitArrays_RoundTrip)               // BuildFiveMaterialScene(false): the three
                                                                 // merge-wait arrays (amended 2026-11-15, A64)
  TEST(WriteSavedState, FreeBox_SolidsSnapshotRoundTrip)         // BuildFreeBoxScene(500)
  TEST(WriteSavedState, TwoWritesOfOneSnapshot_SameDigest)       // Review Focus 4
  TEST(ReadSavedState, OtherFingerprint_IsWrongRunFingerprint)
  TEST(ReadSavedState, CutFile_IsFileDamagedBeforeAnyValue)      // digest checked first, then extents
  TEST(BuildSimulationFromSnapshot, FromReadSavedState_TenStepsSameBytes)  // falling block: 5 steps, save, read,
      // rebuild, 10 more, against 15 straight: ExpectSameBytes
  TEST(FindSavedState, NewestAtOrBefore)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Save everything a run needs to continue`. New names: `SavedState`, `WriteSavedState`, `ReadSavedState`,
  `FindSavedState`.

### Task 20: Kept frame inputs, the snapshot digest and `ReadParticle`

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/frame_inputs.h`, `frame_inputs.cc`, `frame_inputs_test.cc`,
  `native/fluidity/io/testdata/scenes/asymmetric_z_up.json`
- Modify: `schemas/simulate/v2/array_file.fbs` (`FrameInputsDescription`), `schemas/simulate/v2/scene_file.fbs`
  (solids table, minimal: `id`, `name`, `shape`, `half_size`, one driven key; Task 26 completes it)

**Interfaces:**
- Produces design 3.5 `KeptFrameInputs` (plan decision: `frame`, `time`, `snapshot`), `WriteFrameInputs`
  (`kHeaderAndSizes`, `frame_inputs/<frame:06d>.inputs`), `ReadFrameInputs`, `ComputeSnapshotDigest` (BLAKE3 over
  `ToSnapshotArrays(snapshot, kFrameInputs)` bytes in order, then the solids header table bytes), `ParticleFacts`,
  `ReadParticle` (kept inputs of that frame, else a saved state of that frame, else `kFrameInputsNotKept`).
- `asymmetric_z_up.json` (design test 3): domain [0, 10]^3, smallest cell 0.5 m; one water particle at (1, 2, 3) m
  with velocity (-4, 5, -6) m/s through `start_particles_file`; a fixed box solid turned 30 degrees about X at
  (7, 8, 9), half size (0.5, 1, 0.25); gravity (0, 0, -9.81).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(WriteFrameInputs, AsymmetricFixture_SameComponentsInSameOrder)  // design test 3: scene file -> RunScene ->
      // BuildSimulation -> ReadSnapshot -> WriteFrameInputs -> ReadFrameInputs: position (1, 2, 3) and velocity
      // (-4, 5, -6) exactly, component by component; the box's orientation read back equals 30 degrees about X
  TEST(ComputeSnapshotDigest, SameAsDigestOfPackedArrays)
  TEST(ComputeSnapshotDigest, OneBitOfOnePositionChangesIt)
  TEST(ReadParticle, FromFrameInputs_AllFields)                 // material id, kind, rest_volume, size_level
  TEST(ReadParticle, FromSavedStateWhenNoInputs)
  TEST(ReadParticle, UnknownId_IsParticleNotFound)
  TEST(ReadParticle, NoInputsNoSavedState_IsFrameInputsNotKept)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Keep each frame's inputs and read one particle back`. New names: `KeptFrameInputs`, `WriteFrameInputs`,
  `ReadFrameInputs`, `ComputeSnapshotDigest`, `ParticleFacts`, `ReadParticle`.

### Task 21: An independent decoder of the array file

**Kind:** mechanical

**Files:**
- Create: `reference/io/read_array_file.py`, `reference/io/read_array_file_test.py`,
  `native/fluidity/io/testdata/asymmetric_frame_inputs.inputs` (written by a C++ test, committed)
- Modify: `pyproject.toml`, `uv.lock` (`flatbuffers==25.12.19` in the `dev` group; **to verify** that PyPI has it,
  else the newest 25.x, recorded in the commit body), the `io/` CMake lines (CTest `reference_io_array_file`
  running `uv run pytest reference/io -q`, `LABEL cpu`), `frame_inputs_test.cc`

**Interfaces:**
- Produces `reference/io/read_array_file.py` (design test 2, the one reference piece): written from design 5.3
  and the field ids of `array_file.fbs` alone, with `struct` and `flatbuffers.table.Table` (no generated code, no
  code shared with the writer). `read_array_file(path) -> dict[str, list]`.

- [ ] **Step 1: Failing tests.** C++: `TEST(WriteFrameInputs, CommittedReferenceFile_IsCurrent)` (writes the
  asymmetric fixture's frame 0 inputs and compares bytes with the committed file; regenerate with
  `--gtest_also_run_disabled_tests` on a `DISABLED_WriteReferenceFile` test). Python:
  `test_asymmetric_fixture_numbers` (particle 1: position `[1.0, 2.0, 3.0]`, velocity `[-4.0, 5.0, -6.0]`, exactly),
  `test_every_array_aligned_to_64`, `test_header_count_matches_file_size`.
- [ ] **Step 2: Run** (fail). **Step 3: Write the reference.** The rules check scans `.py`: keep it clean.
- [ ] **Step 4:** gate. **Commit.** Subject: `Decode the array file with an independent Python reader`. Body: the
  uncertainty it removes (a file that is what the specification says, not what our reader expects).

### Task 22: The output queue and the frame writer interface

**Amended 2026-11-14 (Foundation as built):** the helper library already exists (Task 15, `fluidity_add_test_library`);
this task adds to it, and `output_queue_test` names it in `LINKS`.

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/output_queue.h`, `output_queue.cc`, `output_queue_test.cc`
- Modify: `native/fluidity/io/io_test_support.h`, `io_test_support.cc` (library `fluidity_io_test_support`, Task
  15), `native/fluidity/io/CMakeLists.txt` (`output_queue_test` with `LABEL cpu LINKS fluidity_io_test_support`)

**Interfaces:**
- Produces design 3.6: `FrameOutput` (plan decision fields), `ProductsOfFrame`, `FrameWriter` (with
  `start(void* context)`), `OutputQueueSettings`, `OutputQueue`, `BuildOutputQueue`, `AddOutputResult`,
  `AddFrameOutput`, `WriteWaitingFrames`, `BuildIoMemoryEstimate` (one row "io: write buffer" 4 MiB and one row "io:
  hasher" `sizeof(blake3_hasher)`, `kWholeRun`, `MemoryWhere::kCpu`). The worker runs design 5.6 step 7.6 in frame
  order, numbered `// Step 7.6.N` as `<proc-advance-run-to>`: digest; already done: compare, `write_products` with
  `is_already_done`, `on_frame_checked`, or `kFrameDiffersOnResume`; else `write_products`, `WriteFrameInputs`
  (with `keeps_frame_inputs`), `WriteFrameDone`, on a saved-state frame `WriteSavedState` then
  `ApplySavedStateRetention`, then `WriteRunReport` (Task 23 supplies it; here a callback), then `on_frame_done`.
  One worker thread, one lock, two condition variables.
- Produces (test-only, `fluidity_io_test_support`): `struct TextFrameWriterLog { std::vector<std::pair<std::int64_t,
  bool>> calls; std::int64_t fail_folder_write_at_frame = -1; std::int64_t not_built_at_frame = -1; double
  sleep_seconds = 0; };` and `FrameWriter BuildTextFrameWriter(TextFrameWriterLog* log)`: for each declared output
  (or `only_output_ids`) it writes `<file_pattern with the frame>.txt` holding `output_id`, frame, time as hex
  float, `state_version`, the particle count of every material and the snapshot digest in hex, and lists it with
  `kSizeOnly`; stateless, so a resumed run writes the same bytes. `ExpectSameInternalFiles(const FileList&, const
  FileList&)` compares every internal file's digest and every `FrameDone::snapshot_digest`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; hand-built `FrameOutput`s from falling-block snapshots; design test 12):

  ```cpp
  TEST(AddFrameOutput, SlowWriter_NeverMoreThanTwoWaiting)     // sleep 0.05 s: waiting_frames <= 2 always,
      // waited_seconds > 0 summed, frames listed ascending, all 10 present
  TEST(AddFrameOutput, FolderWriteFailsAtFrame5_FramesZeroToFourListed)  // then kOutputStopped at the next call
  TEST(AddFrameOutput, NotBuiltAtFrame5_EntryAndLaterFramesListed)
  TEST(AddFrameOutput, WorkerFailsWhileQueueFull_OutputStoppedWithoutHanging)  // Review Focus 5; TIMEOUT 30
  TEST(AddFrameOutput, AlreadyDoneEqual_CheckedAndNothingListed)
  TEST(AddFrameOutput, AlreadyDoneDiffers_IsFrameDiffersOnResume)
  TEST(BuildIoMemoryEstimate, TwoRowsOnCpu)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate (add `AddFrameOutput` to the allowed names if
  reported). **Commit.** Subject: `Hand frames to one output worker without dropping any`. New names: the 3.6
  queue list above.

### Task 23: The run report, `BuildRun` and `AdvanceRunTo`

**Kind:** integration

**Files:**
- Create: `schemas/simulate/v2/run_report.fbs`, `native/fluidity/io/run_report.h`, `run_report.cc`,
  `native/fluidity/io/run.h`, `run.cc`, `run_state.h` (private), `run_test.cc`

**Interfaces:**
- Produces design 3.4 `RunStatus`, `LastTry`, `RunReport` (minus `interchange_libraries`; `toolchain` names the
  compiler, C++ standard, CUDA toolkit when built, the numerical flags and the three package versions),
  `WriteRunReport` (`run_report.json` printed by the FlatBuffers text generator, replaced by one rename),
  `ReadRunReport(const Path&)`; design 3.6 `RunProgress`, `RunCallbacks`, `RunParts` (plan decision), `Run`,
  `BuildRun`, `StopRequests`, `RunStop`, `AdvanceRunResult`, `AdvanceRunTo`, `ReadRunReport(const Run&)`.
- `BuildRun` runs design 5.6 steps 1 to 6 with `// Step N:` comments: `step::CheckSceneCanRun`;
  `ReadMemoryAvailable(run_on)` and `(kCpu)` (or `assume_memory_bytes`); `step::BuildRunPlan(scene, settings,
  on_processor, on_cpu, BuildIoMemoryEstimate(output), BuildCellResizeMemoryEstimate(...) when resizing)`; fingerprint
  (`ComputeExecutableDigest`, `ReadProcessorDescription(run_on)`); `BuildNewRunFolder`; list `ready_scene`
  (`kHeaderAndSizes`); `WriteOutputDeclarations(BuildOutputDeclarations(...))`; compose `step::StepCallbacks`
  (`on_try_done`, `on_step_done` with the run's own context; cell resize and batches in Task 29);
  `step::BuildSimulation`; `frame_writer.start`; hand the start state over as `first_frame` with a saved state.
  The run's state lives behind a `std::unique_ptr<RunState>` so callback contexts survive moving `Run`.
- `AdvanceRunTo` is design 5.6 step 7 (7.1 to 7.5 on the simulation thread) and step 8.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `falling_block.json`, `TextFrameWriter`):

  ```cpp
  TEST(BuildRun, FallingBlock_FirstFrameListedWithSavedState)  // ready_scene, Output entries, frame 0 inputs,
                                                               // FrameDone(0), saved_states/000000.state
  TEST(AdvanceRunTo, TwentyFrames_EveryFrameDoneAscending)     // saved states at 0, 10, 20; newest 2 kept
  TEST(AdvanceRunTo, FrameTimesExact)                          // FrameDone(f).time == ComputeFrameTime(f)
  TEST(AdvanceRunTo, TwiceSameFingerprint_SameInternalFiles)   // two folders: ExpectSameInternalFiles
  TEST(AdvanceRunTo, ProductsOneTextFilePerOutputAndFrame)
  TEST(BuildRun, SceneThatCannotRun_NothingAllocatedNoFolder)  // a blocking finding: kModuleFailed (step,
                                                               // kSceneNotSupported); the folder does not exist
  TEST(BuildRun, AssumeMemoryOneByte_IsDoesNotFit)
  TEST(WriteRunReport, ReadBackEqual)
  TEST(AdvanceRunTo, OnProgressAndFrameDoneCalled)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate (add `AdvanceRunTo`). **Commit.** Subject:
  `Run a scene frame by frame into its folder`. New names: the 3.4 report list and the 3.6 run list above.

### Task 23a: The planned cost of a run (the budget)

**Amended 2026-11-14 (A17, batched-run buffers):** new task. Closes "Batched solves (A15)" open point 6 and owner
question round 2 item 11: the plan is shown before anything is allocated.

**Amended 2026-11-14 (A25):** the budget says, per liquid, whether the run uses the pressure solve or the viscous
solve and the viscous effect per step (`plan-decisions.md`, "A25: what counts as viscous, and the viscous effect per
step"): `RunBudget` gains `viscous_path`, `viscous_effect_cell_width_m`, `viscous_effect_step_seconds` and
`liquids`; `BuildBudgetLines` prints the `liquids:` block before the last `fits:` line; `run_budget.fbs` gains the
fields and the table `ViscousEffectOfLiquid`. `check` (Task 36) and the run's start (Task 37) print these lines as
part of the budget lines they already print.

**Amended 2026-11-14 (A27):** the I/O struct and the schema table are named `io::ViscousEffectOfLiquid` (were
`LiquidViscousEffect`, the name of the step's different struct `step::LiquidViscousEffect`, which keeps its name);
`RunBudget::liquids` holds them. The JSON field names are unchanged.

**Amended 2026-11-15 (A58):** the memory budget is `allowed = min(memory_share_of_free x free, free -
memory_left_free_bytes)` on both processors (step's `ComputeAllowedMemoryBytes`; `memory_share_of_total` is
removed); the suggestion becomes `more_memory_bytes = max(ceil(needed / memory_share_of_free), needed +
memory_left_free_bytes) - free`; `MemoryBudget` gains `double memory_share_of_free` and `std::uint64_t
memory_left_free_bytes` (copied from `StepSettings`; JSON `memory_share_of_free`, `memory_left_free_bytes`);
`RunBudget` gains `bool notes_windows_gpu_fallback` (set by `BuildRunBudget` when the executable is built for
Windows and `run_on == kGpu`) and `BuildBudgetLines` then prints, once, right after the `gpu memory:` line: `note: on
Windows a full GPU makes the run slow instead of failing; see "Prefer No Sysmem Fallback" in the NVIDIA Control
Panel`. The exact-line tests below are rewritten with the new allowed bytes (GPU 15.10 GB free: 13.59 GB allowed;
CPU 56.00 GB free: 50.40 GB allowed) and the new suggestions. New test `BuildBudgetLines,
WindowsGpu_FallbackNoteOnce`.

**Amended 2026-11-15 (A57; reconcile item 21 kept):** the `liquids:` lines follow `plan-decisions.md` item 4 (the
smoothing ratio, the change of a feature four cells wide over the run, the spreading, the Reynolds number, the
advice, the note on slipping); `kSmallestVisibleViscousEffectPerStep` and the old note line are removed.
`io::ViscousEffectOfLiquid` (I/O's JSON row) and `step::LiquidViscousEffect` (the step's plan row) keep their two
names; the I/O struct replaces `effect_not_visible` by `viscosity_to_smoothing_ratio` (optional),
`four_cell_feature_change`, `spreading_in_cells`, `reynolds_number`, `advises_zero_viscosity`, each copied from the
step's field of the same name; `RunBudget` gains `viscous_effect_run_seconds` and `reynolds_length_m`.
`BuildBudgetLines, TwoLiquids_ViscousEffectLines` becomes `BuildBudgetLines, WaterAndSyrup_ViscosityAdviceLines`.
(A58) The scene file's `step` table mirrors `StepSettings`, so it gains `memory_share_of_free` and `memory_left_free_bytes`
and loses `memory_share_of_total` (now an unknown field); like every memory setting they are presentation fields,
in no fingerprint (design 5.2).

**Amended 2026-11-15 (A64):** in this version every liquid cell holds particles. When the plan does not fit and
the largest processor entry of `modules` is `particles`, `BuildBudgetLines` adds before the last line: `particles:
<n> needed, <m> of them in cells deeper than <d> m below the surface; in this version every liquid cell holds
particles (deep liquid without particles comes later)`. `FitSuggestion` gains `std::uint64_t deep_particle_count`
(`RunPlan::counts.start_particles_in_deep_cells`, step) and `double deep_cell_depth` (m, `surface_reach_in_cell_widths
x smallest_cell_width` of `ReadyScene::cell_size`); JSON
`deep_particle_count`, `deep_cell_depth_m`. New test `BuildBudgetLines, DeepTankDoesNotFit_NamesDeepParticles`.

**Kind:** integration

**Files:**
- Create: `schemas/simulate/v2/run_budget.fbs`, `native/fluidity/io/run_budget.h`, `run_budget.cc`,
  `run_budget_lines.cc`, `run_budget_test.cc`
- Modify: `schemas/simulate/v2/run_report.fbs` (`budget: RunBudget`), `native/fluidity/io/run_report.h`
  (`RunReport` gains `std::optional<RunBudget> budget;`), `run_report.cc`, `run.cc` (`BuildRun` step 3 builds the
  budget and refuses a plan that does not fit), `run_test.cc`, the `io/` CMake lines (`fluidity_add_schema`)

**Interfaces:**
- Consumes: `step::RunPlan` with `fit`, `estimate.on_processor`, `estimate.on_cpu`, `ModuleBytes::on_cpu`,
  `planned_step_copies` (step plan Task 8, amended); `batches::BatchPlan` (A15 fields) and `batches::BatchesFailure`
  (`needed_bytes`, `allowed_bytes`); `MemoryAvailable`; `ProcessorDescription`; `io::ToName(step::Module)` (Task 5,
  amended by the reconcile pass). (A25) `step::RunPlan::liquid_viscous_effects` (`step::LiquidViscousEffect`:
  `ingredient_index`, `viscosity`, `density`, `viscous_effect_per_step`), `viscous_effect_cell_width`,
  `viscous_effect_step_duration`, `viscous_path` (`step::ViscousPath`); ~~`viscosity::
  kSmallestVisibleViscousEffectPerStep`~~ (removed, A57); `io::ToName(step::ViscousPath)` (Task 5, amended).
  (Amended 2026-11-15, A57) `step::LiquidViscousEffect::{viscosity_to_smoothing_ratio, four_cell_feature_change,
  spreading_in_cells, reynolds_number, advises_zero_viscosity}`, `RunPlan::{viscous_effect_run_duration,
  reynolds_length}`. (A58) `StepSettings::{memory_share_of_free, memory_left_free_bytes}`, `RunPlan::allowed_bytes`
  and `allowed_cpu_bytes` from `step::ComputeAllowedMemoryBytes`. (A64) `step::SceneCounts::
  start_particles_in_deep_cells` (and `particles_at_depth_levels`, not printed);
  `ReadyScene::cell_size.surface_reach_in_cell_widths` (`grid::CellSizeSettings`).
  (A40) `BuildRunPlan(..., cell_resize_estimate, std::optional<RunMode> forced_mode)`.
- Produces exactly the "The budget" block of `plan-decisions.md`: `ModeReason`, `BudgetVerdict`, `MemoryBudget`,
  `ModuleBudget`, `CopyBudget`, `FitSuggestion`, `RunBudget`, `BuildRunBudget`, `BuildBudgetLines`,
  `BuildSceneBudget` (below), and the schema
  tables with the field names given there. **Plan decisions:**
  - Sources of every field: one piece: `processor_memory.budget_bytes = estimate.on_processor`, `cpu_memory.
    budget_bytes = estimate.on_cpu`; batched: `BatchPlan::bytes_on_processor`, `bytes_on_cpu`; `allowed_bytes` =
    `RunPlan::allowed_bytes` and `allowed_cpu_bytes`; totals and free from the two `MemoryAvailable` (or
    `assume_memory_bytes` as both total and free). `modules` from `estimate.by_module` (processor bytes = the four
    sums minus `on_cpu`). `copies`: `step_*` from `planned_step_copies`; `pass_*` from `bytes_copied_per_try`,
    `pass_loads_per_try`; `solve_*` from `solve_bytes_per_iteration`, `solve_loads_per_iteration`, limit
    `pressure_solve.max_iterations`; with cell resize on a GPU grid `resize_bytes = (456 + 8 * 64) *
    limits.max_cell_blocks`, `resize_copies = 2` in one piece, and `8 * 64 * limits.max_cell_blocks`, 1 in a batched
    run (the resize reads the step's grid copy and builds the resized CPU grid into the step's free copy, no refresh
    copy; reconcile, step row B5 withdrawn), else 0;
    `snapshot_bytes_per_frame` = the sum of `one_snapshot` over `by_module`; the two totals add step, pass and
    resize. `particle_count` = `counts.start_particles + counts.particles_from_sources`; `smallest_cell_width` from
    the grid frame.
  - Mode and verdict: `fit == kFits` and `kOnePiece`: `kFitsProcessor`, `kFits`; `kBatched`:
    `kParticlesDoNotFitProcessor`; `kBatchedWithSolvesInPieces`: `kSolverBuffersDoNotFitProcessor`; a GPU run
    with `fit == kDoesNotFit` or a `BatchesFailure` `kStateDoesNotFitCpu`: `kStateDoesNotFitCpu`,
    `kDoesNotFitCpu`; `kBatchDoesNotFit`: `kLayerDoesNotFitProcessor`, `kDoesNotFitProcessor`; a CPU run that does
    not fit: `kOnePieceDoesNotFitCpu`, `kDoesNotFitCpu`. `to_fit` by the suggestion formulas of `plan-decisions.md`
    (the failing memory's needed and allowed bytes; ~~`memory_share_of_total` from `StepSettings`~~; amended
    2026-11-15, A58: `more_memory_bytes = max(ceil(needed / memory_share_of_free), needed + memory_left_free_bytes)
    - free`, 0 when negative, with the failing processor's free bytes and `StepSettings::memory_share_of_free`,
    `memory_left_free_bytes`; `s = allowed / needed`, `max_particle_count` and `min_smallest_cell_width`
    unchanged).
  - (Amended 2026-11-15, A58) `MemoryBudget { std::uint64_t budget_bytes = 0, allowed_bytes = 0, total_bytes = 0,
    free_bytes = 0; double memory_share_of_free = 0; std::uint64_t memory_left_free_bytes = 0; }` (the last two
    copied from `StepSettings`, the same for both processors); `RunBudget::notes_windows_gpu_fallback` (true when
    `BuildRunBudget` runs in a Windows build, `#if defined(_WIN32)`, and `run_on == kGpu`; a field so the line
    tests run on every platform; JSON `notes_windows_gpu_fallback: bool`). The note line comes right after the
    `gpu memory:` line (the fourth line) and is printed once; `check` and the add-on show it as part of `budget.lines`.
  - (Amended 2026-11-15, A64) The deep-particles line: printed when `verdict != kFits` and the first entry of
    `modules` sorted by `processor_bytes` (ties: `Module` order) is `step::Module::kParticles`, right before the
    last `fits: no.` line; `<n>` = `to_fit->particle_count`, `<m>` = `to_fit->deep_particle_count` (integers grouped
    by commas), `<d>` = `to_fit->deep_cell_depth` with `%.4f`. `BuildRunBudget` fills the two fields whenever it
    fills `to_fit`.
  - `std::expected<RunBudget, IoFailure> BuildSceneBudget(const RunScene& scene, const RunOverrides& overrides,
    std::span<const MemoryEstimateEntry> output_estimate);` (plan decision): `ReadMemoryAvailable(run_on)` and `(kCpu)` (or the
    overrides), `step::BuildRunPlan` (with `BuildCellResizeMemoryEstimate` when the scene resizes), and, for a batched
    plan from Task 29 on, `batches::BuildBatchPlan`; then `BuildRunBudget`. Allocates no buffer. `check` (Task 36),
    `run` (Task 37, to print the lines even when the run is refused) and `BuildRun` call it.
  - **(A25) What counts as viscous, in the budget.** In `io/run_budget.h` (NEW):

    ```cpp
    struct ViscousEffectOfLiquid {                // One per liquid ingredient, scene ingredient order.
      std::string ingredient_name;                // The ingredient's export name.
      double viscosity_pa_s = 0;                  // The setting.
      double density_kg_per_m3 = 0;               // Rest density of its material.
      double viscous_effect_per_step = 0;         // No unit; 0 when viscosity is 0. The grid number N = nu dt / h^2.
      bool is_viscous = false;                    // viscosity_pa_s > 0.
      // ~~bool effect_not_visible~~ removed (A57). NEW (A57), copied from step::LiquidViscousEffect:
      std::optional<double> viscosity_to_smoothing_ratio;   // No unit. 4 N / (1 - own_motion_kept); empty at 1.
      double four_cell_feature_change = 0;        // No unit, 0..1. -expm1(-pi^2 D / 4), D = nu T / h^2.
      double spreading_in_cells = 0;              // No unit. sqrt(nu T) / h.
      double reynolds_number = 0;                 // No unit. U L / nu, U = sqrt(2 g L).
      bool advises_zero_viscosity = false;        // ratio < 0.1 or change < 0.01 (viscosity_advice.h constants).
    };
    // RunBudget gains:
    step::ViscousPath viscous_path{};             // Stage 12 (kPressureSolve) or 11 (kViscousSolve), fixed for the run.
    double viscous_effect_cell_width_m = 0;       // The grid frame's smallest cell width.
    double viscous_effect_step_seconds = 0;       // StepSettings::max_step_duration.
    double viscous_effect_run_seconds = 0;        // NEW (A57). RunPlan::viscous_effect_run_duration (ReadyScene end_time).
    double reynolds_length_m = 0;                 // NEW (A57). RunPlan::reynolds_length (smallest domain extent).
    std::vector<ViscousEffectOfLiquid> liquids;   // Empty in a scene without liquid.
    ```
    `BuildRunBudget` copies them from `RunPlan::viscous_path`, `viscous_effect_cell_width`,
    `viscous_effect_step_duration`, (A57) `viscous_effect_run_duration`, `reynolds_length` and
    `liquid_viscous_effects` (name from the ready scene's ingredient at `ingredient_index`). `run_budget.fbs`:
    `RunBudget` gains `viscous_path: string` (`ToName(step::ViscousPath)`: `"pressure_solve"`, `"viscous_solve"`),
    `viscous_effect_cell_width_m: double`, `viscous_effect_step_seconds: double`, (A57) `viscous_effect_run_seconds:
    double`, `reynolds_length_m: double`, `liquids: [ViscousEffectOfLiquid]`, with `table ViscousEffectOfLiquid {
    ingredient_name: string; viscosity_pa_s: double; density_kg_per_m3: double; viscous_effect_per_step: double;
    is_viscous: bool; viscosity_to_smoothing_ratio: double = null; four_cell_feature_change: double;
    spreading_in_cells: double; reynolds_number: double; advises_zero_viscosity: bool; }` (amended 2026-11-15, A57:
    `effect_not_visible` removed; the ratio is an optional scalar, absent when empty).
  - **(A25; lines replaced 2026-11-15, A57) The `liquids:` lines.** `BuildBudgetLines` prints them only when
    `liquids` is not empty, right before the last `fits:` line (and before the A64 deep-particles line): `%g` for
    viscosity, density and the run's duration, `%.3g` for the grid number, the ratio, the spreading and the
    Reynolds number, `%.4f` for width and step, the change as an integer percentage (`change x 100` rounded to
    nearest, halves away from zero):
    - `kPressureSolve` (unchanged): `liquids: pressure solve (every liquid has viscosity 0)`, then per liquid `
      <name>: viscosity 0 Pa s, density <d> kg/m3, inviscid`;
    - `kViscousSolve`: `liquids: viscous solve (a liquid has viscosity above 0); at the smallest cell <w> m, the
      largest step <t> s and the run's <T> s`, then per liquid with viscosity 0 `  <name>: viscosity 0 Pa s,
      density <d> kg/m3, inviscid`, and per viscous liquid, in scene ingredient order: `  <name>: viscosity <v> Pa
      s, density <d> kg/m3, grid number <N>; viscosity / transfer smoothing <r>; a feature 4 cells wide changes by
      <c> % over the run (<s> cells of spreading); Reynolds number <Re>` (with an empty ratio, `<r>` is `none: the
      liquid keeps all its own motion`), then when `advises_zero_viscosity` `  advice: <name>'s viscosity will not be
      visible: set it to 0`, then `  note: at viscosity 0 <name> slips along solids instead of sticking`.
    - ~~`  note: <name>'s viscosity changes velocity differences between neighbouring cells by less than 0.1 % per
      step, which is not visible; set its viscosity to 0`~~ (removed, A57).
  - `BuildRun` (Task 23, step 3): takes the plan and budget from `BuildSceneBudget`, keeps the budget in
    `RunReport::budget`; a verdict other than `kFits` is `kModuleFailed` (step, `kDoesNotFit`) before any folder or
    buffer exists, as `BuildRun, AssumeMemoryOneByte_IsDoesNotFit` already expects.

- [ ] **Step 1: Failing tests** (`LABEL cpu`). The three line tests build a `RunBudget` by hand and compare
  `BuildBudgetLines` with these lines exactly (`processor_model` "NVIDIA GeForce RTX 5060 Ti", GPU totals 16.00 GB
  total, 15.10 GB free, ~~11.20~~ 13.59 GB allowed; CPU 64.00 GB total, 56.00 GB free, ~~44.80~~ 50.40 GB allowed;
  ~~`memory_share_of_total` 0.7~~ amended 2026-11-15 (A58): `memory_share_of_free` 0.9, `memory_left_free_bytes`
  536,870,912, so allowed = min(0.9 x 15.10e9, 15.10e9 - 536,870,912) = 13.59e9 and min(0.9 x 56e9, 56e9 -
  536,870,912) = 50.4e9; `notes_windows_gpu_fallback` false in these three):

  ```cpp
  TEST(BuildBudgetLines, OnePieceOnGpu_ExactLines)
      // budgets 1.82e9 (gpu) and 2.1e8 (cpu); modules particles 9.1e8, transfer 3.8e8, pressure 2.2e8, step 1.8e8
      // (cpu 1.2e7), grid 6e7; step copies 20480 B in 6; snapshot 9.6e7. Expected:
      // plan for small_wave on gpu (NVIDIA GeForce RTX 5060 Ti)
      // mode: one piece (everything fits the gpu)
      // gpu memory: budget 1.82 GB of 13.59 GB allowed (16.00 GB total, 15.10 GB free)
      // cpu memory: budget 210.00 MB of 50.40 GB allowed (64.00 GB total, 56.00 GB free)
      // largest on gpu: particles 910.00 MB, transfer 380.00 MB, pressure 220.00 MB, step 180.00 MB, grid 60.00 MB
      // copies per step: at most 20.48 kB in 6 copies
      // copies per frame: snapshot up to 96.00 MB
      // small records: status records, air pockets, gas blobs and per-solid results are not counted
      // time: not estimated before the run; a projected finish is printed after 3 frames
      // fits: yes
  TEST(BuildBudgetLines, SolvesInPiecesOnGpu_ExactLines)
      // 12 batches, 5 solve pieces; budgets 1.09e10 and 3.84e10; modules (gpu, cpu): pressure (3.1e9, 4.2e9),
      // transfer (2.6e9, 2.1e9), particles (2.2e9, 3.01e10), grid (1.4e9, 0), step (9e8, 1.3e9), solids (3e8, 4e8);
      // pass 6.6e9 B in 24 loads, step 1.25e6 B in 9, solve 1.5e9 B in 15 loads per iteration, limit 200;
      // snapshot 2.4e9. Expected (lines 1, 3, 4 and the last four as above with these numbers):
      // mode: batches with solves in pieces (the particles and the solver buffers do not fit the gpu; particle
      //   stages run in 12 batches, solves in 5 pieces)            <- one line
      // largest on gpu: pressure 3.10 GB, transfer 2.60 GB, particles 2.20 GB, grid 1.40 GB, step 900.00 MB
      // largest in cpu memory: particles 30.10 GB, pressure 4.20 GB, transfer 2.10 GB, step 1.30 GB, solids 400.00 MB
      // copies per step: 6.60 GB in 24 batch loads, at most 1.25 MB in 9 step copies
      // solve copies: 1.50 GB in 15 piece loads per iteration (300.00 GB at the limit of 200 iterations)
      // copies per frame: snapshot up to 2.40 GB
  TEST(BuildBudgetLines, DoesNotFitCpu_ExactLinesWithWhatWouldFit)
      // scene wide_breaking_wave; one piece needs 4.53e10 on the gpu; batches need 6.1e10 of 5.04e10 cpu;
      // 52,000,000 particles, 31,000,000 of them deeper than 0.08 m; smallest cell 0.01 m; snapshot 2.9e9;
      // modules particles 2.1e10, transfer 9.1e9, pressure 5.2e9, step 4.1e9, grid 1.3e9. Expected (amended
      // 2026-11-15, A58 allowed bytes and suggestions, A64 deep-particles line):
      // plan for wide_breaking_wave on gpu (NVIDIA GeForce RTX 5060 Ti)
      // mode: none (in one piece it does not fit the gpu, and in batches the state does not fit cpu memory)
      // gpu memory: one piece needs 45.30 GB of 13.59 GB allowed (16.00 GB total, 15.10 GB free)
      // cpu memory: batches need 61.00 GB of 50.40 GB allowed (64.00 GB total, 56.00 GB free)
      // largest on gpu: particles 21.00 GB, transfer 9.10 GB, pressure 5.20 GB, step 4.10 GB, grid 1.30 GB
      // copies per step: not planned (the scene does not fit)
      // copies per frame: snapshot up to 2.90 GB
      // small records: status records, air pockets, gas blobs and per-solid results are not counted
      // time: not estimated
      // particles: 52,000,000 needed, 31,000,000 of them in cells deeper than 0.0800 m below the surface; in this
      //   version every liquid cell holds particles (deep liquid without particles comes later)   <- one line
      // fits: no. To fit (estimates): at most 42,963,934 particles (now 52,000,000), or a smallest cell of at least
      //   0.0107 m (now 0.0100 m), or 11.78 GB more cpu memory      <- one line
  TEST(BuildRunBudget, DoesNotFitCpu_Suggestions)        // the case above through BuildRunBudget from a
      // BatchesFailure{kStateDoesNotFitCpu, needed 6.1e10, allowed 5.04e10} and cpu free 5.6e10 (amended
      // 2026-11-15, A58): s = 5.04e10 / 6.1e10 = 0.8262295; max_particle_count floor(s x 52e6) = 42963934;
      // min_smallest_cell_width ceil_1e-4(0.01 / cbrt(s) = 0.0106570) = 0.0107; more_memory_bytes
      // max(ceil(6.1e10 / 0.9) = 67777777778, 6.1e10 + 536870912) - 5.6e10 = 11777777778; more_memory_on kCpu;
      // deep_particle_count 31000000 and deep_cell_depth 0.08 copied from the plan's counts and 8 x 0.01 m
  TEST(BuildRunBudget, LayerDoesNotFit_CellAndGpuOnly)   // kBatchDoesNotFit needed 4e9 allowed 1e9: no particle
      // suggestion; min width ceil_1e-4(0.01 / sqrt(0.25)) = 0.02; more_memory_on kGpu
  TEST(BuildRunBudget, FallingBlockOnCpu_FieldsEqualThePlan)  // real plan: budget_bytes == estimate.on_processor +
      // on_cpu in the one cpu line; copies all zero; modules sum to the estimate; lines end with "fits: yes"
  TEST(BuildRunBudget, ForcedBatched_FieldsEqualTheBatchPlan)  // liquid_source_solids.json, assume_memory_bytes just
      // below the one-piece estimate: run_mode kBatched, budgets == BatchPlan bytes_on_processor / bytes_on_cpu,
      // pass_bytes == bytes_copied_per_try, batch_count == BatchPlan::batch_count
  TEST(WriteRunReport, BudgetFieldNamesInJson)           // the JSON text holds "budget", "processor_memory",
      // "budget_bytes", "allowed_bytes", "total_bytes", "free_bytes", "copies", "bytes_per_step", "copies_per_step",
      // "snapshot_bytes_per_frame", "lines"; ReadRunReport gives the budget back equal
  // ~~TEST(BuildBudgetLines, TwoLiquids_ViscousEffectLines)~~ replaced 2026-11-15 (A57) by:
  TEST(BuildBudgetLines, WaterAndSyrup_ViscosityAdviceLines)  // hand-built from the step's numbers: water 1e-3 Pa s,
      // 1000 kg/m3, N 1e-4, ratio 0.008, change 0.2186563, spreading 0.3162278, Re 4.4294e6, advised; syrup 10 Pa s,
      // 1400 kg/m3, N 0.7142857, ratio 57.142857, change 1.0 (to rounding), spreading 26.726124, Re 620.12, not
      // advised (alpha 0.95 both, h 0.01 m, dt 0.01 s, T 10 s, L 1 m, g 9.81); kViscousSolve. The six lines right
      // before the last "fits:" line are exactly:
      // liquids: viscous solve (a liquid has viscosity above 0); at the smallest cell 0.0100 m, the largest step 0.0100 s and the run's 10 s
      //   water: viscosity 0.001 Pa s, density 1000 kg/m3, grid number 0.0001; viscosity / transfer smoothing 0.008; a feature 4 cells wide changes by 22 % over the run (0.316 cells of spreading); Reynolds number 4.43e+06
      //   advice: water's viscosity will not be visible: set it to 0
      //   note: at viscosity 0 water slips along solids instead of sticking
      //   syrup: viscosity 10 Pa s, density 1400 kg/m3, grid number 0.714; viscosity / transfer smoothing 57.1; a feature 4 cells wide changes by 100 % over the run (26.7 cells of spreading); Reynolds number 620
      //   note: at viscosity 0 syrup slips along solids instead of sticking
  TEST(BuildBudgetLines, PureFlipLiquid_SmoothingNone)   // A57: syrup with own_motion_kept 1 (ratio empty): its line
      // holds "viscosity / transfer smoothing none: the liquid keeps all its own motion"; not advised (change 100 %)
  TEST(BuildBudgetLines, WindowsGpu_FallbackNoteOnce)    // A58: the one-piece budget above with
      // notes_windows_gpu_fallback true: line 4 is exactly
      // note: on Windows a full GPU makes the run slow instead of failing; see "Prefer No Sysmem Fallback" in the NVIDIA Control Panel
      // and it appears once; with false it is absent and the other lines are unchanged
  TEST(BuildBudgetLines, DeepTankDoesNotFit_NamesDeepParticles)  // A64: a CPU run, kOnePieceDoesNotFitCpu,
      // particle_count 52,000,000, deep_particle_count 31,000,000, deep_cell_depth 0.4 (w_min 0.05 m), particles the
      // largest module: the line right before "fits: no." is exactly
      // particles: 52,000,000 needed, 31,000,000 of them in cells deeper than 0.4000 m below the surface; in this version every liquid cell holds particles (deep liquid without particles comes later)
      // with transfer made the largest module, the line is absent; with verdict kFits, absent
  TEST(BuildBudgetLines, InviscidScene_PressureSolveLine)  // A25: one liquid "water" from the water preset (0 Pa s,
      // 1000 kg/m3), kPressureSolve. The two lines right before the last "fits:" line are exactly:
      // liquids: pressure solve (every liquid has viscosity 0)
      //   water: viscosity 0 Pa s, density 1000 kg/m3, inviscid
  TEST(RunBudget, LiquidsRoundTrip)                       // A25: WriteRunReport then ReadRunReport gives viscous_path,
      // viscous_effect_cell_width_m, viscous_effect_step_seconds and every liquids field back equal; the JSON text
      // holds "viscous_path": "viscous_solve", "liquids", "viscosity_pa_s", "density_kg_per_m3",
      // "viscous_effect_per_step", "is_viscous", ~~"effect_not_visible"~~ and (amended 2026-11-15, A57)
      // "viscosity_to_smoothing_ratio" (absent for a liquid whose ratio is empty), "four_cell_feature_change",
      // "spreading_in_cells", "reynolds_number", "advises_zero_viscosity", "viscous_effect_run_seconds",
      // "reynolds_length_m"
  TEST(WriteRunReport, MemoryBudgetShareFieldsInJson)    // A58: "memory_share_of_free": 0.9 and
      // "memory_left_free_bytes": 536870912 in both processor_memory and cpu_memory, "notes_windows_gpu_fallback",
      // "deep_particle_count", "deep_cell_depth_m"; ReadRunReport gives them back equal
  ```
  The three line tests above leave `liquids` empty, so their expected lines are unchanged (A25); the A58 allowed
  bytes and suggestions and the A64 line are amended in them as written.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** Sizes: below 1000 `"<n> B"`, else the largest of kB, MB, GB, TB
  (powers of 1000) not above the value, `std::to_chars` fixed with 2 decimals; integers grouped by commas; widths
  `std::to_chars` fixed with 4 decimals and `" m"`. **Step 4:** gate. **Commit.** Subject:
  `Show the planned cost of a run before it starts`. New names: the list above.

### Task 24: Continue equals uninterrupted

**Kind:** integration

The test the rest of the module serves (design test 9, the liquid scene; Review Focus 1). It comes as soon as a
run exists, before sources, solids and gas join the scene file, so a missing saved value is found while the
saved state is small.

**Files:**
- Create: `native/fluidity/io/continue_run_test.cc`
- Modify: `native/fluidity/io/run.h` (`BuildRunToContinue`), `run.cc`

**Interfaces:**
- Produces design 3.6 `BuildRunToContinue`: `ReadFileList`, `ReadRunScene` (never the scene file),
  `BuildRunFolderToContinue` (fingerprint recomputed from the listed ready scene and this executable),
  `FindSavedState(last_frame_done)`, `ReadSavedState`; if it fails its checks, the next older listed one is tried
  (the `started` event says so, through `RunProgress::Activity::kReadingSavedState`);
  `step::BuildSimulationFromSnapshot`; frames after the state's frame that are already done are handed over with
  `is_already_done`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `falling_block.json`, 20 frames, `TIMEOUT 900`):

  ```cpp
  TEST(ContinueRun, StopAfterFrame12_SameFilesAsStraight)   // (b) stop_after_frame_requested set in
      // on_frame_done(12): saved state 12 listed, RunStop::kStoppedAfterFrame; BuildRunToContinue + AdvanceRunTo(20):
      // ExpectSameInternalFiles with the straight run (a), same saved state 20, text product files byte-equal
  TEST(ContinueRun, CancelInFrame13_SameFilesAsStraight)    // (c) cancel set by on_progress at the first try of
      // frame 13: kCancelled, nothing of frame 13 listed; resume from saved state 10: frames 11 and 12 reported
      // through on_frame_checked; TextFrameWriterLog shows (11, true), (12, true); then as (b)
  TEST(ContinueRun, KilledInFrame13_SameFilesAsStraight)    // (d) EXPECT_EXIT: on_progress at frame 13 runs
      // std::_Exit(9); the test resumes and compares as (c)
  TEST(ContinueRun, NewestSavedStateDamaged_OlderOneUsed)   // flip a byte of saved state 20 after a run to 20,
      // extend to 25: resumes from 10, frames 11 to 20 checked, 21 to 25 equal a straight run to 25
  TEST(ContinueRun, ExtendLastFrame_SameAsLongerStraight)   // run to 10, resume to 20 (last_frame not in the
                                                            // fingerprint)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** If (b), (c) or (d) differs, find the missing value with
  `step::testing::ExpectSameBytes` on the snapshots of the first differing frame; fix the owning commit (Task 19's
  packing or the step's snapshot), never the test.
- [ ] **Step 4:** gate. **Commit.** Subject: `Continue a run with the bytes of an uninterrupted one`. New names:
  `BuildRunToContinue`.

### Task 25: Scene file sources

**Kind:** integration

**Files:**
- Modify: `schemas/simulate/v2/scene_file.fbs` (`sources: [SceneSource]` with `id`, `name`, `material_id`,
  `volume_rate`, `start_time`, `end_time`, `position`, `position_spread`, `velocity`, `size_level`,
  `ingredient_shares`), `ready_scene.fbs`, `run_scene.cc`, `run_scene_test.cc`

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `BuildRunScene, SourceVolumeRate_GivesParticleRate` (`rest_volume =
  particle_spacing^3` at the source's level, `particles_per_second = volume_rate / rest_volume`, `mass =
  rest_density * rest_volume`; 1e-15 relative); `BuildRunScene, HalfCellWidth_SameVolumeRate` (particle rate x 8,
  volume flow unchanged); `CheckSceneFile, NegativeVolumeRate_IsValueOutOfRangeWithPath` (`sources[0].volume_rate`);
  `BuildRunScene, SourcesAscendingId`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.1 step 4 "Sources". Test 8 (Task 18) now covers the
  new physical fields; run it. **Step 4:** gate. **Commit.** Subject: `Read sources as a volume rate`.

### Task 26: Scene file solids

**Amended 2026-11-15 (A59):** the free-solid table is named `free_solid` (was `free`; the schema mirrors
`solids::SolidSettings::free_solid` field by field, and `plan-decisions.md` names `free_solid.guide`). It gains
`guide: SceneGuide { kind: string; moves_along_x: bool = null; moves_along_y: bool = null; moves_along_z: bool =
null; turns: bool = null; rail_direction: Vec3; hinge_point: Vec3; hinge_axis: Vec3; }`, read into
`solids::FreeMotionLimits` (`guide_kind`, the four flags, `rail_direction`, `hinge_point`, `hinge_axis`; world frame).
`kind`: `"axis_limits"` (default when `guide` or `kind` is absent: `kAxisLimits`, flags default true), `"rail"`,
`"hinge"`; any other text is `kSceneValueOutOfRange` at `solids[i].free_solid.guide.kind`. A flag given with
`rail` or `hinge`, or a rail or hinge field given with another kind, is `kSceneValueOutOfRange` at that field
(strict: nothing is ignored silently). `rail_direction` and `hinge_axis` must be finite with a length above 1e-12;
they are normalised at load, `v / sqrt(x x + y y + z z)` in that order with no contraction, and the normalised
value is what the ready scene records (`FilledDefault` source `kDerived` is not used: the field is given).
`hinge_point` (m, world, at time 0) must be finite. Defaults of absent rail and hinge fields are
`FreeMotionLimits`'s.

**Amended 2026-11-15 (A53):** each solid gains `slip_length: double = null` (m, finite, >= 0) and `free_slip: bool =
null`; `free_slip: true` gives `SolidSettings::slip_length = +infinity` and then `slip_length` must be absent
(`kSceneValueOutOfRange` at `solids[i].slip_length` otherwise). Absent both: 0 (full stick), recorded as a
`FilledDefault`. Plan decision: `plan-decisions.md` item 7 writes the field as `"free"` or a number; a FlatBuffers
field has one type, so the strict schema carries the two fields the add-on already has (`slip_length`, `free_slip`).
The value is read for every motion kind; on a free solid `check` gives the NOTE finding
`slip_ignored_on_free_solid` (a check note, not a capability reason: `kCapabilityReasonCount` is unchanged). Both are physical fields (Task 18's walk covers them;
the ready scene stores +infinity as a double).

**Kind:** integration

**Files:**
- Modify: `schemas/simulate/v2/scene_file.fbs` (`solids: [SceneSolid]` complete: `id`, `name`, `shape`,
  `half_size`, `half_thickness`, `path: { keys: [PathKey { time, origin, orientation }] }`, `after_path`,
  ~~`free`~~ `free_solid` (`mass`, `inertia`, motion limits, A59 `guide`), A53 `slip_length`, `free_slip`,
  `wave_stroke`, `wave_period`, `cuts`), `ready_scene.fbs`, `run_scene.cc`, `run_scene_test.cc`

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `BuildRunScene, DrivenPathAngularVelocity`
  (`ComputeRotationVector(key, next) / (next.time - key.time)`, zero on the last key);
  `BuildRunScene, WavePaddleKeys` (equal to `solids::BuildWavePaddlePath` up to `end_time`);
  `BuildRunScene, FreeBoxInertiaFromHalfSizes` (`mass / 3 * (b^2 + c^2, a^2 + c^2, a^2 + b^2)`);
  `BuildRunScene, BlockingMaterialsAndRadius` (every material with `HasInternalForce`; `particle_radius =
  spacing / 2`); `BuildRunScene, HasLiquidOrGasAndGravityCopied`;
  `CheckSceneFile, SolidIdZeroOrDuplicate_IsDuplicateIdWithPath`; `CheckSceneFile, SolidIdAbove32Bits_IsOutOfRange`.
  Amended 2026-11-15 (A59, A53): `BuildRunScene, NoGuide_AxisLimitsAllFree` (`guide_kind` `kAxisLimits`, four
  flags true); `BuildRunScene, RailGuide_DirectionNormalised` (`rail_direction` (0, 0, 2) gives (0, 0, 1) exactly;
  (1, 1, 0) gives (0.70710678118654746, 0.70710678118654746, 0) with `ToBits` equal to `1 / std::sqrt(2.0)`);
  `BuildRunScene, HingeGuide_PointAndAxisCopied` (point (0.2, 0.2, 0.6), axis (0, 3, 0) gives (0, 1, 0));
  `CheckSceneFile, ZeroRailDirection_IsValueOutOfRangeWithPath` (`solids[0].free_solid.guide.rail_direction`);
  `CheckSceneFile, UnknownGuideKind_IsValueOutOfRange` (`"slider"`, path `solids[0].free_solid.guide.kind`);
  `CheckSceneFile, FlagWithRail_IsValueOutOfRange` (`moves_along_x` with kind `rail`); `BuildRunScene,
  SlipLength_ZeroByDefaultNumberOrFree` (absent: 0 and one `FilledDefault`; 0.002: 0.002; `free_slip` true:
  `std::isinf` and positive); `CheckSceneFile, SlipLengthAndFreeSlip_IsValueOutOfRange`; `CheckSceneFile,
  NegativeSlipLength_IsValueOutOfRangeWithPath` (`solids[1].slip_length`).
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.1 step 4 "Solids". **Step 4:** gate. **Commit.**
  Subject: `Read solids, their paths and the wave paddle`.

### Task 27: Scene file gas and the open sea; the small reference scenes as files

**Amended 2026-11-15 (A47):** the scene field `gas_wind` is renamed `wind` (same keys `{time, acceleration}`, m/s2,
world). It is read whether or not the scene has gas, into `ReadyScene::wind` (`gas::GasWind`, keys owned by
`RunScene`): with gas, `gas::CheckGasSources` checks the keys as before; without gas, this module checks the same
rule (finite, times strictly ascending) as `kSceneValueOutOfRange` at `wind[i].time`. The keys stay in the
`physical` part (with gas the step reads them; without gas they change only spray and foam, which need the
simulation to rebuild anyway). Render output's `RenderSettings::wind` is filled from `ReadyScene::wind` by its plan
(seam 12). The findings `wind_without_gas` and `wind_flow_not_honoured` are the add-on's export findings, not this
module's. New tests: `BuildRunScene, WindWithoutGas_KeysKept` (two keys, no gas material: `ReadyScene::wind.keys`
equal to the file's, bit for bit); `CheckSceneFile, WindTimesNotAscending_IsValueOutOfRangeWithPath`
(`wind[1].time`); `ReadSceneText, GasWindField_IsUnknownSceneField` (the old name is refused).

**Kind:** integration

**Files:**
- Modify: `schemas/simulate/v2/scene_file.fbs` (`gas` (`gas::GasSettings` field by field), `gas_sources:
  [SceneGasSource { id, name, box, keys: [{ time, fuel_rate, heat_rate }] }]`, ~~`gas_wind`~~ `wind: [{ time,
  acceleration }]` (A47),
  `gas_markers`, `gas_active_cells`, `open_sea` (`pressure::OpenSeaSettings` plus `material_id`,
  `particle_spacing`)), `ready_scene.fbs`, `run_scene.cc`, `run_scene_test.cc`
- Create: `native/fluidity/io/testdata/scenes/small_wave.json`, `small_syrup.json`, `small_fire.json`,
  `small_snow.json`, `liquid_source_solids.json`, `snow.json`, `gas.json`, `gas_beside_liquid.json`

**Interfaces:**
- The four small scenes are the step plan's (`plan-decisions.md`, "Step"): `BuildWaveScene(1)`,
  `BuildSmallSyrupScene`, `BuildFireReferenceScene(0.1)`, `BuildSmallSnowExample`, written as scene files.
  `liquid_source_solids.json` (design test 9): `falling_block.json` plus a water source at (0.2, 0.2, 0.35),
  0.5 L/s from 0 to 0.4 s, a driven box moving +X at 0.1 m/s, a free box of 500 kg/m3, points of material 1 and
  solid motion asked. `snow.json`, `gas.json`, `gas_beside_liquid.json`: the step scenes
  `BuildSnowAndJellyScene`, `BuildSmallFireScene`, `BuildGasBesideLiquidScene(true)` as files.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `BuildRunScene, GasKeysOwnedAndChecked` (`gas::CheckGasSources`
  finding as `kModuleFailed` with `gas_sources[0].keys[1].time`); `BuildRunScene, OpenSeaSettingsCopied`;
  `BuildRunScene, SmallReferenceScenes_MatchStepTestScenes` (for each of the four: domain box, grid frame,
  gravity, rest densities and viscosities, solid count, source count and `particles_per_second` within 1e-12
  relative, start particle count equal to the step builder's).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Read gas and the open sea; add the reference scenes as files`.

### Task 28: Continue equals uninterrupted on every material; cadence; other runs refused

**Kind:** integration

**Files:**
- Modify: `native/fluidity/io/continue_run_test.cc`, `native/fluidity/io/run.h` (`BuildRunFromSavedState`),
  `run.cc`
- Create: `native/fluidity/io/run_rules_test.cc`

**Interfaces:**
- Produces design 3.6 `BuildRunFromSavedState`: a new folder; the state is read from the other folder without
  its lock and without writing there; `RunFingerprintParts::started_from` = the other run's fingerprint;
  `kSavedStateDoesNotMatchScene` when `step::BuildSimulationFromSnapshot` gives `kSnapshotDoesNotMatch`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, 20 frames, `TIMEOUT 900`):

  ```cpp
  TEST(ContinueRun, LiquidSourceSolids_FourWaysSameFiles)  // design test 9 (a) to (d) on liquid_source_solids.json;
      // every declared output (points_1, solid_motion) has one listed file per frame (design test 21, part)
  TEST(ContinueRun, Snow_FourWaysSameFiles)
  TEST(ContinueRun, Gas_FourWaysSameFiles)
  TEST(ContinueRun, GasBesideLiquid_FourWaysSameFiles)
  TEST(AdvanceRunTo, CadenceChangesNothing)                // design test 10: saved state every 1, 5, 10; queue 1
      // and 4; keeps_frame_inputs on and off: equal snapshot digests; equal ready_scene and saved states where
      // both runs list the same frame
  TEST(BuildRunToContinue, ViscosityChanged_IsWrongRunFingerprintScene)  // design test 11: an edited scene
      // through BuildRun into a used folder, and the folder's own ready_scene with an injected executable digest
      // (FileCalls read_executable_path): kExecutable
  TEST(BuildRunFromSavedState, EditedScene_NewFolderOldUntouched)  // old folder bytes unchanged; started_from set
  TEST(BuildRunFromSavedState, OtherMaterialCount_IsSavedStateDoesNotMatchScene)
  ```
- [ ] **Step 2: Run.** Fix owning commits if needed. **Step 3:** gate. **Commit.** Subject:
  `Continue runs of every material and refuse other runs`. New names: `BuildRunFromSavedState`.

### Task 29: Cell resize and batches in a run

**Amended 2026-11-14 (A17, batched-run buffers):** the batch plan's choice of the third mode is copied into the run
plan (A15 left this to I/O), its failure becomes the budget's verdict, and the budget of Task 23a is rebuilt from
the batch plan.

**Amended 2026-11-14 (reconcile):** `AdvanceRunTo, BatchedEqualsOnePiece` is replaced by the A15 tolerance test and
per-mode repeat tests; `run_mode` stays in the fingerprint; `CellResizeSettings::grid_copies` is set from the mode.

**Amended 2026-11-15 (A64):** no grid-only region. The scene's `cell_resize` table has no `regions` table:
`particle_region_depth_in_cell_widths`, `grid_only_depth_in_cell_widths`, `overlap_layer_count` and
`particles_per_refilled_cell` are not scene fields (they move to the later group after Task 42, where
`ReadSceneText` refuses them until then as `kUnknownSceneField`). `Run` sets no `apply_grid_only_region`,
`move_grid_only_liquid` or `read_largest_grid_only_speed` callback, and the run report has no `cells_to_grid_only`
or `cells_to_particles` (`RunReport, ResizeTotalsAndBatchPlan` checks `resize_count` and the batch plan only).

**Amended 2026-11-15 (A41, A61-A63):** `cell_resize.cell_size` is `grid::CellSizeSettings` field by field
(`surface_reach_in_cell_widths`, `solid_reach_in_cell_widths`, `shear_width_factor`, `merge_width_share`,
`merge_wait_duration`, `min_merge_wait_steps`, `max_cell_splits_per_resize`); `smallest_size_level` is not a scene
field: it is derived, `size_level_count - 1` (`FilledDefault` source `kDerived`, path `grid.size_level_count`).
`choose_cell_size` and `ChooseCellSize` no longer exist: `Run` sets only `resize_cells` (and `run_stages` in either
batched mode). The artist inputs come in Task 29b.

**Amended 2026-11-15 (A58):** `step::BuildSimulation` reserves the run's one `RunMemory` block, sized by
`ComputeRunMemoryBytes` of the whole plan (the plan's estimate holds the cell resize, batches and render output
entries `BuildSceneBudget` passed to `BuildRunPlan`), before anything else. `Run` therefore builds
`cell_resize::BuildCellResize(settings, plan.limits, ingredient_count, scene.artist_cell_sizes(), run_memory,
  cpu_run_memory)` and
`batches::BuildBatchedRun(batch_plan, scene, run_memory, cpu_run_memory)` right after `BuildSimulation`, with `run_memory =
step::FindRunMemory(simulation)` and `cpu_run_memory = step::FindCpuRunMemory(simulation)` (the step's accessors;
the same object in a CPU run; the two `RunMemory&` last, A58), and the callbacks reach them through
`RunState` (each callback's context is the `RunState`, whose private wrappers forward to `ResizeCells` and
`ApplyStagesInBatches`; `RunState` outlives moves of `Run`). Their `where = kCpu` parts go into `cpu_run_memory`. A
`kOutOfMemory` of either is `kModuleFailed` (cell resize or batches, `kOutOfMemory`, `magnitude` = bytes asked). New
test `BuildRun, CellResizeAndBatchesInTheOneBlock` (liquid_source_solids.json with cell resize, forced `kBatched`
on the CPU: after `BuildRun`, `MemoryReport::reserved_block_count == 1` and `handed_out_bytes == planned_bytes`).

**Amended 2026-11-15 (reconcile, A58):** `step::StepCallbacks` gains `Callback<bool(void* context, Simulation&
simulation, const SimulationLimits& new_limits)> limits_raised`, which `ApplyLimits` calls after the new `RunMemory`
blocks are reserved and before the old ones are freed. `Run` sets it with the `RunState` as context: it rebuilds
`CellResize`, the `BatchedRun` and (render output's plan, seam 3) the `StepRecorder` into `step::FindRunMemory` /
`FindCpuRunMemory` at `new_limits`, carrying over what they keep (`CellResize`'s last report; the recorder's waiting
records), and returns false on any failure, which the step reports as `kLimitsCannotBeRaised`. Test `BuildRun,
LimitRaise_RebuildsIoHeldBuffersIntoNewBlock` (liquid_source_solids.json with cell resize, forced `kBatched` on the
CPU, start limits from a smaller scene so the first frame raises them: `reserved_block_count == 2`, the new
`CellResize` and `BatchedRun` buffers lie inside the second block (`handed_out_bytes == planned_bytes` of the new
plan), `ReadLastReport(cell_resize)` equal before and after the raise, and frames 1 to 3 equal a run started at the
raised limits by `ExpectSameInternalFiles`; a `limits_raised` forced to fail gives `kModuleFailed` (step,
`kLimitsCannotBeRaised`) with the simulation still usable).

**Kind:** integration

**Files:**
- Modify: `schemas/simulate/v2/scene_file.fbs` (`cell_resize` (`cell_resize::CellResizeSettings` without
  `run_on`), `batches` (`batches::BatchSettings`)), `ready_scene.fbs`, `run_scene.cc`, `run.cc`,
  `continue_run_test.cc`
- Create: `native/fluidity/io/testdata/scenes/cell_resize.json`

**Interfaces:**
- `RunScene::cell_resize_settings()` and `batch_settings()` filled; `ReadyScene::has_cell_resize` set when the
  file has `cell_resize`. `Run` composes (design 5.6 step 5; amended 2026-11-15, A41, A64, A58):
  `cell_resize::BuildCellResize(settings, plan.limits, ingredient_count, scene.artist_cell_sizes(), run_memory,
  cpu_run_memory)`;
  `resize_cells = {ResizeCells wrapper, &run_state}`; ~~`choose_cell_size`, `apply_grid_only_region`,
  `move_grid_only_liquid`, `read_largest_grid_only_speed`~~ (removed: A41, A64); with `kBatched`:
  `batches::BuildBatchPlan(scene, plan, settings, batch_settings)`, its `max_blocks_per_batch`,
  `max_loaded_blocks`, `max_loaded_particles` copied into the `RunPlan`, `BuildBatchedRun(batch_plan, scene,
  run_memory, cpu_run_memory)`, `run_stages = {ApplyStagesInBatches wrapper, &run_state}`. The run report gets `batch_plan` and
  `resize_count` from `ReadLastReport` (~~`cells_to_grid_only`, `cells_to_particles`~~, A64).
- **Plan decisions (A17, batched-run buffers):** also `max_solve_loaded_blocks` is copied, and `mode =
  RunMode::kBatchedWithSolvesInPieces` when `BatchPlan::solve_place == SolvePlace::kInPieces`
  (`plan-decisions.md` "Batched solves (A15)"); `BuildSceneBudget` (Task 23a) now calls `BuildBatchPlan` for a
  batched plan and gives `BuildRunBudget` the batch plan or its `BatchesFailure`; `CellResizeSettings::grid_run_on`
  is the run's `run_on`, and (reconcile) `CellResizeSettings::grid_copies` is `kNone` on the CPU, `kHeldByCellResize`
  in a one-piece GPU run, `kHeldByStep` in both batched GPU modes (batches plan Task 10). `RunOverrides` gains `std::optional<std::uint64_t> assume_cpu_memory_bytes` (the CPU memory
  given to the plan; not in any fingerprint; `check --assume-cpu-memory`, Task 34). `check` and `run` build the
  same budget from the same scene and memory (both call `BuildSceneBudget`).

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `TIMEOUT 900`):

  ```cpp
  TEST(ContinueRun, CellResize_FourWaysSameFiles)        // design test 9 with cell resize
  TEST(ContinueRun, Batched_FourWaysSameFiles)           // design test 9 batched: assume_memory_bytes just below the
                                                         // one-piece estimate, so the plan is kBatched
  TEST(AdvanceRunTo, BatchedWithinStepToleranceOfOnePiece)  // design test 18 on CPU, A15 form (reconcile; was
      // BatchedEqualsOnePiece): liquid_source_solids.json one piece, kBatched and kBatchedWithSolvesInPieces (forced
      // with assume_memory_bytes): the last saved states within the step design 6.4 tolerances of one piece; the
      // test prints whether the bytes happen to be equal. run_mode is part of the fingerprint (modes differ, A15)
  TEST(AdvanceRunTo, EachModeRepeatRun_SameFiles)        // reconcile, A15: each of the three modes run twice: equal
      // snapshot digests and saved states, file for file
  TEST(RunReport, ResizeTotalsAndBatchPlan)
  TEST(BuildRun, SolvesInPiecesChosen_ModeCopied)        // A17: liquid_source_solids.json with assume_memory_bytes
      // between the two batched layouts (pressure::testing::CountBlocksInLargestLayer to size it): RunPlan::mode
      // kBatchedWithSolvesInPieces, max_solve_loaded_blocks == the batch plan's, budget run_mode the same
  TEST(BuildSceneBudget, StateDoesNotFitCpu_VerdictAndSuggestions)  // A17: assume_cpu_memory_bytes below the
      // batched CPU rows: verdict kDoesNotFitCpu, to_fit filled; BuildRun on the same overrides: kModuleFailed (step,
      // kDoesNotFit), no folder
  TEST(BuildRun, CellResizeAndBatchesInTheOneBlock)      // amended 2026-11-15 (A58): see the amendment above
  TEST(BuildRun, LimitRaise_RebuildsIoHeldBuffersIntoNewBlock)  // amended 2026-11-15 (reconcile): see above
  TEST(ReadSceneText, RegionsTable_IsUnknownSceneField)  // amended 2026-11-15 (A64): "cell_resize": {"regions":
      // {"grid_only_depth_in_cell_widths": 7}} refused with its path until the later group
  ```
- [ ] **Step 2: Run.** **Step 3:** gate. **Commit.** Subject: `Compose cell resize and batches into a run`.

### Task 29a: The measured cost of a run, planned against measured, and the time

**Amended 2026-11-14 (A17, batched-run buffers):** new task. What a run actually used and copied, beside its budget,
the time spent per stage, a warning when a measure passes its budget, and a projected finish once frames have been
measured.

**Amended 2026-11-15 (A43, A58):** `kMemoryWarningShare` and `kCopyWarningShare` are removed: a planned budget is a
promise, so any measure above it warns with the amount, without margin. `RunCost` gains, from `step::MemoryReport`
(largest over the run unless named): `std::uint64_t planned_bytes`, `reserved_bytes` (current block),
`std::uint32_t reserved_block_count`, `std::uint64_t handed_out_bytes`, `memory_taken_bytes` (GPU only, 0 on the
CPU), `requested_beyond_plan_bytes`, `reserved_cpu_bytes`, `handed_out_cpu_bytes` (the CPU block `cpu_run_memory` a
GPU run also has), and `std::uint64_t solve_bytes_per_iteration_max` (for the max rule below); JSON fields of the
same names. `MemoryReport::allocated_bytes` is now `handed_out_bytes` and `allocated_cpu_bytes` is
`handed_out_cpu_bytes`. `inline constexpr std::uint64_t kMemoryTakenToleranceBytes = 2'097'152;` (2 MiB; a
starting value, to confirm by the Windows run). Warnings, in this order:
- `warning: <p> memory: <b> requested beyond the planned <b>`: gpu when `requested_beyond_plan_bytes > 0` (amount
  = it) or `handed_out_bytes > planned_bytes` (amount = the difference, the larger of the two amounts); cpu when
  `handed_out_cpu_bytes` or `reserved_cpu_bytes` exceeds the planned cpu budget (amount = the larger excess). In a
  CPU run the one `cpu memory` warning compares the summed measures with the summed budget, as the cost line does.
- `warning: copies per step <b> (max) are above the planned <b>` when `copy_bytes_per_step_max >
  copies.bytes_per_step` (max, not mean: the plan is an upper bound); `warning: solve copies per iteration <b> (max)
  are above the planned <b>` when `solve_bytes_per_iteration_max > copies.solve_bytes_per_iteration`.
- GPU runs: `warning: gpu memory taken <b> is less than the <b> reserved; Windows may have placed part of it in
  shared system memory, which is much slower. To make the run fail instead, set "CUDA - Sysmem Fallback Policy" to
  "Prefer No Sysmem Fallback" for fluidity.exe and blender.exe in the NVIDIA Control Panel.` when
  `memory_taken_bytes + kMemoryTakenToleranceBytes < reserved_bytes`; never when taken is above reserved.
The cost lines gain, in GPU runs right after the `gpu memory:` line: `gpu memory: planned <b>, reserved <b> in <n>
block(s), handed out <b>, taken <b>` (`1 block`, `<n> blocks` for n > 1). `BuildCostLines,
WithinMargins_NoWarning` is replaced by `BuildCostLines, MeasuredEqualsPlanned_NoWarning`; new tests
`RequestedBeyondPlan_Warns`, `CopiesAboveBound_Warns`, `MemoryTakenBelowReserved_WarnsSpill`,
`MemoryTakenAboveReserved_NoWarning`.

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/run_cost.h`, `run_cost.cc`, `run_cost_lines.cc`, `time_projection.cc`,
  `run_cost_test.cc`
- Modify: `schemas/simulate/v2/run_budget.fbs` (`RunCost`, `StageSeconds`), `run_report.fbs` (`cost: RunCost`),
  `native/fluidity/io/run_report.h` (`RunReport` gains `RunCost cost;`), `run_report.cc`, `run.h` (`RunProgress`
  gains `TimeProjection time_projection;`), `run.cc`, `run_test.cc`

**Interfaces:**
- Consumes: `step::StepResult` (`report.copies`, `report.timings`, iteration counts), `step::MemoryReport`
  (~~`allocated_bytes`, `allocated_cpu_bytes`~~ amended 2026-11-15, A58: `handed_out_bytes`, `handed_out_cpu_bytes`,
  `planned_bytes`, `reserved_bytes`, `reserved_cpu_bytes`, `reserved_block_count`, `memory_taken_bytes`,
  `requested_beyond_plan_bytes`), `step::ReadCopyCounts`, `batches::ReadLastReport` (pass and A15 solve
  fields), `cell_resize::ReadLastReport` (`bytes_copied`, `copy_count`), `io::ToName(step::Stage)`, Task 23a's
  `RunBudget`.
- Produces exactly the "Measured cost and time" block of `plan-decisions.md`: `StageSeconds`, `RunCost`,
  `ActivitySeconds`, `ApplyTryToRunCost`, `ApplyStepToRunCost`, `ApplyFrameToRunCost`, `BuildCostLines`,
  ~~`kMemoryWarningShare`, `kCopyWarningShare`~~ `kMemoryTakenToleranceBytes` (A58), `FrameMark`, `TimeProjection`,
  `ComputeTimeProjection`. **Plan
  decisions:**
  - Wiring in `Run`: `on_try_done` calls `ApplyTryToRunCost` with the batched run's `ReadLastReport` (null in one
    piece); `on_step_done` calls `ApplyStepToRunCost` with `ReadMemoryReport` and the copies of the resize
    (`ResizeCellsReport` of this step, when it resized or tried) plus the difference of `step::ReadCopyCounts` not
    yet seen in a try (the grid copy refresh after a resize); the frame worker calls `ApplyFrameToRunCost` with the
    snapshot's array bytes (the sum of `size() * sizeof` over every vector of `SimulationSnapshot`). A committed
    step's copies = its tries' copies (rejected ones included) + those after the commit; the mean and max are over
    committed steps. Render output adds its recorder's copies through seam 11. A try's copy count is
    `report.copies.copy_count + 2 * batch_report.batch_count + batch_report.solve_batch_loads` (a load is one
    copy); its bytes are `report.copies` both directions plus `bytes_to_processor`, `bytes_from_processor`,
    `solve_bytes_to_processor`, `solve_bytes_from_processor`; solve bytes per iteration = the try's solve bytes over
    `max(1, pressure_iteration_count)` (the liquid solve's iterations, which the plan's figure is per; gas and
    viscosity sweeps add to the bytes, so a gas or syrup scene can read above plan; the warning then says so).
  - `BuildCostLines` runs at the end of `AdvanceRunTo` (every `RunStop`) and at each `WriteRunReport`, so the report
    always holds the cost so far; warnings follow `plan-decisions.md` (percent rounded to the nearest integer).
  - `ComputeTimeProjection` uses the last up to 6 `FrameMark`s of frames listed **in this process** (a resumed run
    starts again); fewer than 2: rate 0; fewer than 3: `projected_remaining_seconds = -1`. `RunProgress` carries it
    for the heartbeat and `frame_done` events (Task 35).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeTimeProjection, ThreeFrames_RateAndRemaining)   // marks (0.1, 10), (0.2, 12), (0.3, 14), end 1.0:
      // rate 0.05 and remaining 14 within 1e-12 relative; frames_measured 3
  TEST(ComputeTimeProjection, TwoFrames_RateButNoProjection)  // remaining -1, rate 0.05
  TEST(ComputeTimeProjection, UsesTheLastSixOnly)             // 10 marks, the first four slow: rate from the last six
  TEST(BuildCostLines, OnePieceGpu_ExactLinesAndWarnings)     // hand-built: Task 23a's one-piece budget; measured
      // gpu 2.1e9 (limits raised), cpu 2.1e8, copies per step mean 30000 B, max 41000 B over 412 steps, snapshot
      // mean 8.12e7 over 24 frames, stages choose_active_cells 12.3 s and particles_to_grid 40.1 s over 431 tries,
      // activity 812, 20.1, 0, 4.2, 0 s. Amended 2026-11-15 (A43, A58): planned_bytes 1.82e9, reserved_bytes 2.1e9
      // in 2 blocks, handed_out_bytes 2.1e9, memory_taken_bytes 2.1e9, requested_beyond_plan_bytes 2.8e8,
      // reserved_cpu_bytes = handed_out_cpu_bytes = 2.1e8. Expected:
      // cost of this run, planned / measured:
      // gpu memory: 1.82 GB / 2.10 GB
      // gpu memory: planned 1.82 GB, reserved 2.10 GB in 2 blocks, handed out 2.10 GB, taken 2.10 GB
      // cpu memory: 210.00 MB / 210.00 MB
      // copies per step: at most 20.48 kB / mean 30.00 kB, max 41.00 kB over 412 steps
      // copies per frame: up to 96.00 MB / mean 81.20 MB over 24 frames
      // time by stage over 431 tries: choose_active_cells 12.3 s, particles_to_grid 40.1 s
      // time by activity: simulate 812.0 s, snapshot 20.1 s, build outputs 0.0 s, write 4.2 s, wait 0.0 s
      // warning: gpu memory: 280.00 MB requested beyond the planned 1.82 GB
      // warning: copies per step 41.00 kB (max) are above the planned 20.48 kB
      // (replaced: "warning: gpu memory 2.10 GB is 15 % above ..." and "copies per step 30.00 kB (mean) are 46 % ...")
  // ~~TEST(BuildCostLines, WithinMargins_NoWarning)~~ replaced 2026-11-15 (A43) by:
  TEST(BuildCostLines, MeasuredEqualsPlanned_NoWarning)       // gpu handed out = planned = reserved = taken 1.82e9,
      // requested beyond 0, cpu 2.1e8 both, copies max 20480 B: no warning; copies max 20481 B: exactly one warning
  TEST(BuildCostLines, RequestedBeyondPlan_Warns)             // gpu requested_beyond_plan_bytes 1: "warning: gpu memory:
      // 1 B requested beyond the planned 1.82 GB"; cpu handed_out_cpu_bytes 2.15e8 against 2.1e8: "warning: cpu
      // memory: 5.00 MB requested beyond the planned 210.00 MB"
  TEST(BuildCostLines, CopiesAboveBound_Warns)                // max 20500 B, mean 9000 B against 20480 B: "warning:
      // copies per step 20.50 kB (max) are above the planned 20.48 kB" (the mean being below changes nothing)
  TEST(BuildCostLines, MemoryTakenBelowReserved_WarnsSpill)   // reserved 1.82e9, taken 1.0e9: the spill warning of
      // the amendment, word for word, with "gpu memory taken 1.00 GB is less than the 1.82 GB reserved"
  TEST(BuildCostLines, MemoryTakenAboveReserved_NoWarning)    // taken 1.83e9 (other programs) and taken exactly
      // reserved - 2,097,152: no spill warning; taken reserved - 2,097,153: the warning
  TEST(BuildCostLines, SolvesInPieces_SolveLine)              // Task 23a's second budget, measured 1.49e9 B per
      // iteration, 31.2 iterations per step: "solve copies per iteration: 1.50 GB / mean 1.49 GB, 31.2 iterations
      // per step"
  TEST(AdvanceRunTo, TinyRun_PlannedAgainstMeasured)          // falling_block.json on the CPU, 3 frames:
      // largest_processor_bytes == budget processor budget (CPU run: the one cpu line, equal); copies per step mean
      // and max 0; snapshot_bytes_per_frame_mean == the snapshot bytes of frame 1 computed by the test; stage_seconds
      // lists exactly the stages of StepReport::stages_run, in Stage order, totals > 0; no warning
  TEST(AdvanceRunTo, ForcedBatched_MeasuredCopiesEqualTheReports)  // liquid_source_solids.json forced kBatched on
      // the CPU, 3 frames: sum over committed steps of copies == sum of every try's StepReport::copies and
      // BatchReport bytes and loads, plus the resize copies; copies_per_step_max >= mean > 0
  TEST(WriteRunReport, CostFieldNamesInJson)                  // "cost", "largest_processor_bytes",
      // "largest_cpu_bytes", "copy_bytes_per_step_mean", "copy_bytes_per_step_max", "copies_per_step_mean",
      // "copies_per_step_max", "snapshot_bytes_per_frame_mean", "solve_bytes_per_iteration_mean",
      // "stage_seconds", "total_seconds", "warnings", "lines"; ReadRunReport gives it back equal; amended
      // 2026-11-15 (A58): also "planned_bytes", "reserved_bytes", "reserved_block_count", "handed_out_bytes",
      // "memory_taken_bytes", "requested_beyond_plan_bytes", "reserved_cpu_bytes", "handed_out_cpu_bytes",
      // "solve_bytes_per_iteration_max"
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** Seconds print with one decimal, percents as integers,
  durations as `plan-decisions.md` says. **Step 4:** gate. **Commit.** Subject:
  `Measure what a run costs against its budget`. New names: the list above.

### Task 29b: Artist cell sizes: `cell_size_regions` and `cell_size_map`

**Amended 2026-11-15 (A63, A41):** new task. The artist inputs of the shared cell size rule
(`grid/cell_size_rule.h`, `plan-decisions.md` "Owner answers A40-A64 applied" item 2) are read once here, rounded to
size levels, recorded in the ready scene and handed to cell resize.

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/cell_size_map.cc` (private: `ReadCellSizeMap`), `cell_size_map_test.cc`
- Modify: `conanfile.py` and `conan.lock` (`libspng/0.7.4`, with its `zlib`; **to verify** that Conan Center has
  this version, else the newest 0.7.x, recorded in the commit body; chosen over libpng because it reports errors by
  return code, with no `setjmp`, which suits a library built without exceptions), the `io/` CMake lines
  (`find_package(SPNG CONFIG REQUIRED)` with the target name the recipe gives, SYSTEM, private to `fluidity`),
  `schemas/simulate/v2/scene_file.fbs`, `ready_scene.fbs`, `native/fluidity/io/run_scene.h`, `run_scene.cc`,
  `run_scene_test.cc`, `run.cc`, `io_test_support.h`, `io_test_support.cc` (`WriteGreyPng`)

**Interfaces:**
- Scene file (top level, both optional): `cell_size_regions: [SceneCellSizeRegion { low_corner: Vec3; high_corner:
  Vec3; cell_width: double; }]` (m, world; `cell_width` 0 means the smallest cell width, the add-on's default) and `cell_size_map: SceneCellSizeMap { image: string; }` (a path relative
  to the scene file's folder; a PNG, 8- or 16-bit grey). Either one without a `cell_resize` table is
  `kSceneValueOutOfRange` at `cell_size_regions` / `cell_size_map` (detail "needs cell_resize").
- Checks (`CheckSceneFile`, each with its path): corners finite and `low < high` on each axis (at
  `cell_size_regions[i].high_corner`); `cell_width` finite and either 0 or `>= smallest_cell_width` (a width
  between them cannot be honoured; at `cell_size_regions[i].cell_width`); at most 65,536 regions (the header short-list
  limit); the image: missing `kFileNotFound`, unreadable `kFileNotReadable`, not a PNG or damaged `kFileDamaged`,
  a PNG that is not grey (colour type 0) with bit depth 8 or 16 `kWrongFileKind` (palette, alpha, colour and 1, 2
  or 4 bits are refused, never converted), each with the scene path `cell_size_map.image` in `detail`; width and
  height each 1 to `kMaxCellSizeMapSidePixels = 8192` (a starting value: 64 MiB of levels at most) else
  `kSceneValueOutOfRange`.
- Rounding (once, on the CPU, at `BuildRunScene`): `widths.width[l] = grid::ComputeCellWidth(frame, l)`; a region's
  `size_level` = the smallest `l` with `widths.width[l] <= cell_width` (exact comparison; `grid::
  ComputeWantedSizeLevel(widths, cell_width)`). A map pixel `p` of maximum `p_max` (255 or 65535), with `L =
  size_level_count`: 0 when `p == 0`, else `(2 p (L - 1) + p_max) / (2 p_max)` in unsigned 64-bit integers
  (`round(p / p_max x (L - 1))`, halves up, with no floating point). Gamma and colour profile chunks are ignored:
  the stored sample values are used. Rows are flipped: `map_levels[r x width + c]` is PNG row `height - 1 - r`,
  so row 0 is the domain's low y (PNG row 0 is the top of the image, the high y seen from above, Z-up).
- `RunScene` gains `grid::ArtistCellSizes artist_cell_sizes() const;` (CPU views into its own vectors of
  `grid::CellSizeRegion` and `std::uint8_t` map levels; `domain_low_corner`, `domain_width_x`, `domain_width_y` from
  the domain box; all zero and null without artist inputs). `ready_scene.fbs` `physical` gains `cell_size_regions:
  [CellSizeRegion { low_corner; high_corner; size_level: ubyte; asked_cell_width: double; }]` and
  `cell_size_map_levels: [ubyte]`, `cell_size_map_width: uint`, `cell_size_map_height: uint` (scene paths
  `cell_size_regions[]...` and `cell_size_map.image`), so `ReadRunScene` and a resume never read the image again and
  a changed pixel changes the scene digest. `Run` passes `artist_cell_sizes()` to `cell_resize::BuildCellResize`
  (Task 29, amended), which copies them into its own buffers on the run's processor (entries `"cell resize: cell
  size regions"` and `"cell resize: cell size map"`, `kGivenCount`, are cell resize's).
- Test helper (`fluidity_io_test_support`): `std::expected<void, IoFailure> WriteGreyPng(const Path& file,
  std::uint32_t width, std::uint32_t height, std::uint8_t bit_depth, std::span<const std::uint16_t> samples)`
  (rows top first, as in the file; libspng's encoder).

- [ ] **Step 1: Failing tests** (`LABEL cpu`; base scene `cell_resize.json`: smallest cell 0.01 m, 5 levels, widths
  0.16, 0.08, 0.04, 0.02, 0.01 m at levels 0 to 4):

  ```cpp
  TEST(BuildRunScene, CellSizeRegion_WidthRoundedToLevel)      // asked 0.03 -> level 3; 0.04 -> 2 (exact);
      // 0.16 -> 0; 0.01 -> 4; 0 -> 4 (the smallest); asked_cell_width recorded as given
  TEST(CheckSceneFile, RegionWidthBelowSmallest_IsValueOutOfRangeWithPath)  // 0.005: cell_size_regions[0].cell_width
  TEST(CheckSceneFile, RegionCornersNotOrdered_IsValueOutOfRange)  // high z below low z: cell_size_regions[0].high_corner
  TEST(BuildRunScene, CellSizeMapEightBit_PixelToLevel)        // samples 0, 31, 32, 128, 255 -> 0, 0, 1, 2, 4
  TEST(BuildRunScene, CellSizeMapSixteenBit_PixelToLevel)      // samples 0, 8192, 32768, 65535 -> 0, 1, 2, 4
  TEST(BuildRunScene, CellSizeMapTopRow_IsHighY)               // 2 x 2, top-left 255, others 0: map_levels {0, 0, 4, 0}
  TEST(CheckSceneFile, CellSizeMapColour_IsWrongFileKind)      // an 8-bit RGB PNG; detail holds cell_size_map.image
  TEST(CheckSceneFile, CellSizeMapMissing_IsFileNotFound)
  TEST(CheckSceneFile, CellSizeMapTooWide_IsValueOutOfRange)   // 8193 x 1
  TEST(CheckSceneFile, ArtistSizesWithoutCellResize_IsValueOutOfRange)
  TEST(SceneDigest, CellSizeMapPixelChangesIt)                 // one sample 128 -> 255 changes scene_digest
  TEST(ReadRunScene, ArtistCellSizesRestoredWithoutTheImage)   // the PNG deleted after BuildRun: regions and
      // map_levels read back equal, width and height equal
  TEST(BuildRun, ArtistCellSizesReachCellResize)               // a calm pool 0.4 m deep, one region over the whole
      // domain asking 0.01 m: after 1 s every active liquid cell is at level 4; the same scene without the region
      // has cells at level 3 or coarser below 0.08 m depth
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate (the container fetches libspng and zlib).
  **Commit.** Subject: `Read artist cell sizes from regions and a top-view map`. New names:
  `RunScene::artist_cell_sizes`, `kMaxCellSizeMapSidePixels`, `WriteGreyPng` (test support); body: the libspng
  version and why it was chosen.

### Task 29c: The force of the liquid on every solid in `frame_done` and the run report

**Amended 2026-11-15 (A60):** new task.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/io/output_queue.h` (`FrameOutput`), `run.h` (`RunCallbacks::on_frame_done`), `run.cc`,
  `run_report.h`, `run_report.cc`, `schemas/simulate/v2/run_report.fbs`, `run_test.cc`

**Interfaces:**
- Consumes `solids::LiquidForceOnSolid { solid_id; pressure_force; pressure_torque; viscous_force; viscous_torque; }`
  (N, N m, world, mean over the step, torque about the body's origin at the start of the step) and
  `step::StepReport::liquid_force_on_solids` (every solid, ascending `solid_id`).
- `RunState` keeps the vector of the last committed try (from `on_try_done`, assigned into one vector kept for the
  run, so no allocation once its size is reached). At design 5.6 step 7.4 it is copied into `FrameOutput`, which
  gains `std::vector<solids::LiquidForceOnSolid> liquid_force_on_solids;` (the last committed step at or before the
  frame time; empty for `first_frame`'s start state and for a frame whose time was reached by no step).
  `RunCallbacks::on_frame_done` gains a last parameter `std::span<const solids::LiquidForceOnSolid>
  liquid_force_on_solids`; `RunReport` gains `std::vector<solids::LiquidForceOnSolid> liquid_force_on_solids;` (of
  the last listed frame). Schema: `table LiquidForceOnSolid { solid_id: uint; pressure_force_n: Vec3;
  pressure_torque_n_m: Vec3; viscous_force_n: Vec3; viscous_torque_n_m: Vec3; }` in `run_report.fbs`;
  `run_report.json` and the `frame_done` event (Task 35) carry `liquid_force_on_solids: [LiquidForceOnSolid]`.
  Not in any listed file: the solid motion output is unchanged, and a frame computed again after a resume is
  checked by its snapshot digest only.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `liquid_source_solids.json`):
  `AdvanceRunTo, FrameDoneCarriesLiquidForceOfLastStep` (3 frames; a test `on_try_done` keeps the last committed
  `StepReport::liquid_force_on_solids` before each frame time: the span given to `on_frame_done` for frames 1 and 2
  equals it, `ToBits` for every component; one entry per solid of the scene in ascending `solid_id`; frame 0 has
  none); `WriteRunReport, LiquidForceFieldNamesInJson` (`"liquid_force_on_solids"`, `"solid_id"`,
  `"pressure_force_n"`, `"pressure_torque_n_m"`, `"viscous_force_n"`, `"viscous_torque_n_m"`; `ReadRunReport` gives
  the vector back equal); `ContinueRun, StopAfterFrame12_SameFilesAsStraight` (Task 24) unchanged, which shows the
  forces touch no listed byte.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Report the liquid's force
  on every solid per frame`.

### Task 29d: Resume in the recorded run mode

**Amended 2026-11-15 (A40):** new task. Closes design section 11 "Open" item 4.

**Kind:** integration

**Files:**
- Modify: `schemas/simulate/v2/file_list.fbs` (`Start` gains `run_mode: string`; new action `ModeChange { frame:
  long; from: string; to: string; run_fingerprint: Digest; }`), `native/fluidity/io/run_folder.h` (`FileList` gains
  `step::RunMode run_mode` and `std::vector<RunModeChange> run_mode_changes`), `run_folder.cc`, `run_scene.h`
  (`RunOverrides` gains `std::optional<step::RunMode> recorded_run_mode` and `bool choose_mode_again`),
  `run_budget.cc` (`BuildSceneBudget`), `run.h`, `run.cc`, `run_report.h`, `run_report.cc`, `run_report.fbs`
- Create: `native/fluidity/io/resume_mode_test.cc`

**Interfaces:**
- `struct RunModeChange { std::int64_t frame = -1; step::RunMode from{}, to{}; };` (`frame` = the saved state the
  resume continued from). The recorded mode is `FileList::run_mode`: the newest `ModeChange::to`, else `Start`'s
  `run_mode`; the run fingerprint in effect is the newest `ModeChange::run_fingerprint`, else `Start`'s; an array
  file is checked against the fingerprint in effect when its `Listed` entry was written (the newest `Start` or
  `ModeChange` entry before it in the list).
- `BuildRunToContinue` sets `overrides.recorded_run_mode = file_list.run_mode` unless the command has
  `--choose-mode-again`. `BuildSceneBudget` with a recorded mode plans with the mode forced:
  `step::BuildRunPlan(..., cell_resize_estimate, forced_mode)` with `forced_mode = kOnePiece` for a recorded one-piece
  run and `kBatched` for either recorded batched mode (the step plans that mode or returns `fit == kDoesNotFit`);
  a forced plan that does not fit: refused; a forced `kBatched` plan then goes to `batches::BuildBatchPlan` as any
  batched plan; recorded `kBatchedWithSolvesInPieces` and a batch plan with `kWholeGridOnProcessor`: planned
  again with `BatchSettings::max_whole_grid_share = 0` for this plan only (an in-memory override, not in the ready
  scene or any fingerprint), which puts the solves in pieces; recorded `kBatched` and a batch plan in pieces, or any
  batch failure: refused. A refusal is `IoError::kRecordedModeDoesNotFit` with `magnitude` = needed bytes,
  `limit` = allowed bytes of the failing memory and `detail` = the mode in words (`one piece`, `batches`,
  `batches with solves in pieces`); nothing is written (the lock is released, no event but `failed`; exit code 3,
  `kSceneCannotRun`, as a plan that does not fit).
- With `--choose-mode-again` the mode is planned as `run` plans it. If it differs from the recorded mode,
  `BuildRunToContinue` appends one `ModeChange` entry before the first new frame, `RunReport` gains
  `std::vector<RunModeChange> run_mode_changes` (JSON `run_mode_changes: [{frame, from, to}]`, modes by
  `ToName(step::RunMode)`), and frames already listed after the saved state are handed over with
  `is_already_done` but not compared (they were computed in the other mode; `frame_checked` is not sent for them).
  `kWrongRunFingerprint` with `kRunMode` is never raised by a resume: without the flag the mode is the recorded
  one, with it a change is recorded.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `liquid_source_solids.json`, 20 frames, `TIMEOUT 900`; `assume_memory_bytes`
  "just below the one-piece estimate" forces `kBatched` as in Task 29):

  ```cpp
  TEST(Resume, RecordedModeStillFits_ReusesIt)   // run forced kBatched, stop after 10; resume with ample memory
      // (one piece would fit): started.run_mode "batched", no ModeChange entry, frames 11 to 20 ExpectSameInternalFiles
      // with a straight forced-kBatched run to 20
  TEST(Resume, RecordedModeDoesNotFit_RefusesNamingTheFlag)  // run in one piece, stop after 10; resume with memory
      // just below the one-piece estimate: kRecordedModeDoesNotFit, detail "one piece", magnitude = the one-piece
      // processor estimate, limit = the allowed bytes; the message contains "--choose-mode-again"; every file of
      // the folder byte-equal before and after
  TEST(Resume, ChooseModeAgain_RecordsTheChange)  // the same with choose_mode_again: started.run_mode "batched";
      // file list: one ModeChange {frame 10, one_piece, batched}; run_report.json run_mode_changes the same; frames
      // 11 to 20 listed; a later resume without the flag (after a stop at 15) reuses "batched"
  TEST(Resume, SolvesInPiecesRecorded_ForcedAgain)  // run with memory between the two batched layouts
      // (kBatchedWithSolvesInPieces), stop after 5; resume with memory where kBatched would be chosen:
      // started.run_mode "batched_with_solves_in_pieces", frames 6 to 10 equal a straight run in that mode
  TEST(ReadFileList, ModeChange_FingerprintInEffectPerFrame)  // a list with Start(fp0, one_piece) and
      // ModeChange(10, fp1) after frame 12's entries: run_mode kBatched, run_fingerprint fp1; saved state 10 and
      // frame inputs 12 checked against fp0, saved state 20 against fp1
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Resume a run in the mode
  it was recorded in`. New names: `RunModeChange`, `IoError::kRecordedModeDoesNotFit` (Task 4),
  `RunOverrides::recorded_run_mode`, `RunOverrides::choose_mode_again`, `FileList::run_mode`,
  `FileList::run_mode_changes`, `RunReport::run_mode_changes`.

### Task 30: The read-only query of a run folder

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/run_folder_facts.h`, `run_folder_facts.cc`, `run_folder_facts_test.cc`

**Interfaces:**
- Produces design 3.4 `LockHolder`, `RunFolderFacts`, `FileCheck` (plan decision), `ReadRunFolderFacts`. It takes
  no writer lock and opens every file read-only; `is_held` from `ProbeWriterLock`; `is_crashed` (event field) is
  `report.status == kRunning && !lock_holder->is_held`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ReadRunFolderFacts, FinishedRun_FactsOfListAndReport`
  (`last_listed_frame`, `saved_state_frame`, `last_try`, outputs); `ReadRunFolderFacts, WhileRunLives_IsHeld`
  (facts read from `on_frame_done` of a running `Run`); `ReadRunFolderFacts, AfterChildKilled_NotHeldAndRunning`
  (`EXPECT_EXIT` child runs and `_Exit`s at frame 5); `ReadRunFolderFacts, EveryListedFile_FlipFound`
  (`kEveryListedFile` lists the one damaged file); `ReadRunFolderFacts, FolderBytesUnchanged` (every file's bytes
  and modification time before and after); `ReadRunFolderFacts, NoReport_ReportEmpty`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Answer what a run folder holds without writing to it`. New names: `LockHolder`, `RunFolderFacts`,
  `FileCheck`, `ReadRunFolderFacts`.

### Task 31: Comparing a scene with a run folder

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/compare_with_run_folder.cc`, `compare_with_run_folder_test.cc`
- Modify: `native/fluidity/io/run_folder_facts.h` (`CacheMatch`, `OutputMatch`, `CacheComparison`,
  `CompareWithRunFolder`)

**Interfaces:**
- Produces those names of design 3.4 and design 5.2 "Comparing a scene with a folder": `physical` tables
  compared field by field through reflection, each difference reported by its `scene_path`; outputs by output
  fingerprint; `needs_simulation` when the simulation changed or the folder keeps no frame inputs.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; design test 20, against a finished `liquid_source_solids.json` run of
  5 frames): `CompareWithRunFolder, SameScene_SameEverywhere`;
  `CompareWithRunFolder, ParticleFieldsChanged_OnlyPointsChanged` (simulation `kSame`; `points_1` `kChanged` with
  `outputs.points.particle_fields`, `needs_simulation` false; plan decision: render output's surface spacing case
  joins this test in its plan); `CompareWithRunFolder, ViscosityChanged_SimulationChanged` (`kChanged`, `kScene`,
  `materials[0].ingredients[0].viscosity`, every output `kChanged` with `needs_simulation`);
  `CompareWithRunFolder, NoFileList_Absent`; `CompareWithRunFolder, FolderBytesUnchanged`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate (add `CompareWithRunFolder`). **Commit.**
  Subject: `Compare a scene with the run in a folder`.

### Task 32: Rebuilding outputs from kept frame inputs

**Kind:** integration

**Files:**
- Modify: `native/fluidity/io/run.h` (`RebuildOutputs`), `run.cc`
- Create: `native/fluidity/io/rebuild_outputs_test.cc`

**Interfaces:**
- Produces design 3.6 `RebuildOutputs` (design 5.6 "Rebuild"): same run fingerprint, ids declared now or in the
  folder (`kUnknownOutput`), no named output `needs_simulation_to_rebuild` (`kFrameInputsNotKept`), every frame done
  has kept inputs; then lock, `RemoveOutputFiles`, new declarations, `frame_writer.start`, frames `first_frame`
  to `last_frame_done` from `ReadFrameInputs` with `only_output_ids`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; design test 21, the rebuild part):
  `RebuildOutputs, PointsWithNewFields_OldRemovedNewListed` (new `particle_fields`: old `points_1` files removed,
  new ones listed under a new output fingerprint, `solid_motion` and every internal file untouched, no step run:
  the step count in the run report unchanged); `RebuildOutputs, OtherScene_IsWrongRunFingerprint`;
  `RebuildOutputs, UnknownId_IsUnknownOutput`; `RebuildOutputs, NoFrameInputs_IsFrameInputsNotKept`;
  `RebuildOutputs, StoppedHalfWay_ListedUpToLastFrame`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate (add `RebuildOutputs`). **Commit.** Subject:
  `Rebuild outputs without simulating again`.

### Task 33: Stop requests inside a run

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/stop_requests_test.cc`

Plan decision: the stage-by-stage cancel is the step's own test (`TryStep,
CancelBeforeEachStage_ChangesNothingAndNextCommits`, step Task 31). A one-piece run offers no callback inside a
try in this module (`make_foam_and_spray` stays null), so here cancel is set at every point a run exposes; the
standard-input part of design test 13 is in Task 39.

- [ ] **Step 1: Tests** (`LABEL cpu`, `liquid_source_solids.json`; design test 13, run part):
  `AdvanceRunTo, CancelBeforeFrame7_ListAsBefore` (set in `on_frame_done(6)`); `AdvanceRunTo,
  CancelAfterEachCommitOfFrame7_ListAsBefore` (from `on_progress` after the k-th committed step of frame 7, k = 1
  to 3): each `RunStop::kCancelled`, the file list equals the one after frame 6, and the resume passes
  `ExpectSameInternalFiles` with the straight run; `AdvanceRunTo, StopAfterFrame7_FrameAndSavedStateListed`.
- [ ] **Step 2: Run.** **Step 3:** gate. **Commit.** Subject: `Stop a run at any point and keep every listed file`.

### Task 34: The command line and exit codes

**Amended 2026-11-14 (A17, batched-run buffers):** `check` also takes `--assume-cpu-memory <bytes>`, so a user can
see what more or less CPU memory changes in the budget.

**Amended 2026-11-15 (A40):** `resume` takes `--choose-mode-again` (`bool Command::choose_mode_again = false`,
copied into `overrides.choose_mode_again`); on any other command it is a failure (exit 2). `ToExitCode` maps
`kRecordedModeDoesNotFit` to 3 (`kSceneCannotRun`, the exit of a plan that does not fit). The 3.7 block's resume
line becomes `fluidity resume <run_folder> [--last-frame <n>] [--report-every-try] [--choose-mode-again]`. New
test `ReadCommandLine, ChooseModeAgainOnResumeOnly`; `ToExitCode, EveryRowOfTheTable` gains the row.

**Amended 2026-11-15 (A49):** `fluidity --version` (`CommandKind::kVersion`, appended) prints one line on standard
output, `fluidity <executable_version> cuda-runtime <ReadProcessorDescription(RunOn::kGpu).runtime>` (`none` in a
build without GPU code; reading the runtime version needs no driver), exit 0; the add-on's packaging test compares it
with the shipped runtime file. Test `ReadCommandLine, Version_PrintsCudaRuntime` (kind `kVersion`; in a CPU build
the dispatched line ends with `cuda-runtime none`).

**Amended 2026-11-14 (Foundation as built):** `fluidity_command_line` gets `fluidity_apply_compile_flags(...
EXCEPTIONS ON RTTI ON)`; this task adds the optional `RTTI` argument to that helper (render output Task 4 reuses
it); the test is a plain `fluidity_add_test` (exceptions on, as GoogleTest needs), and an escaping
exception still ends it because each test calls `ReadCommandLine` through a `noexcept` test-side wrapper.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/cli/command.h`, `read_command_line.cc`, `exit_code.cc`, `command_line_test.cc`,
  `exit_code_test.cc`
- Modify: `cmake/fluidity_targets.cmake` and `tools/fluidity_targets_test.py` (the optional `RTTI` argument of
  `fluidity_apply_compile_flags(<target> EXCEPTIONS <ON|OFF> [RTTI <ON|OFF>])`: default `OFF`; `ON` replaces
  `-fno-rtti`, `/GR-` and `-Xcompiler=/GR-` by `-frtti` / `/GR` rather than adding a flag after them, so MSVC sees
  no overridden option; one test: a target with `RTTI ON` compiles a `typeid`),
  `native/fluidity/cli/CMakeLists.txt` (`read_command_line.cc` alone in OBJECT library
  `fluidity_command_line`, created with `add_library(fluidity_command_line OBJECT read_command_line.cc)` then
  `fluidity_apply_compile_flags(fluidity_command_line EXCEPTIONS ON RTTI ON)` (CLI11 needs RTTI), so the
  numerical flags and the configure check hold; the tests `fluidity_add_test(fluidity_command_line_test SOURCES
  command_line_test.cc LABEL cpu LINKS fluidity_command_line)` and `fluidity_add_test(exit_code_test ... LABEL
  cpu)`. Replaced: "the test executable built with exceptions **off** (`-fno-exceptions` / `/EHs-c-`), so an
  escaping exception ends the test": `fluidity_add_test` always turns exceptions on, so the test calls
  `ReadCommandLine` inside a `noexcept` lambda, and an escaping exception calls `std::terminate`)

**Interfaces:**
- Produces design 3.7: `ExitCode`, `ToExitCode`, `CommandKind`, `Command` (its `assume_memory_bytes` is copied into
  `overrides.assume_memory_bytes`; A17: `Command` gains `std::optional<std::uint64_t> assume_cpu_memory_bytes`,
  copied into `overrides.assume_cpu_memory_bytes`, option `--assume-cpu-memory <bytes>` of `check` only),
  `CommandLineFailure`, `ReadCommandLine`. Commands and options exactly as the
  3.7 text block. `ToExitCode` is design 5.8's list; `kOutputStopped` and `kEventsNotWritable` are 1.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; design test 15):

  ```cpp
  TEST(ReadCommandLine, EveryCommand_Parsed)             // one case per line of the 3.7 block
  TEST(ReadCommandLine, UnknownOption_IsFailureNotHelp)
  TEST(ReadCommandLine, MissingValue_IsFailure)          // run scene.json --into
  TEST(ReadCommandLine, PathStartingWithDashes_IsFailure)
  TEST(ReadCommandLine, Help_IsHelpWithText)
  TEST(ReadCommandLine, ParticleIdAbove53Bits_Exact)     // 18446744073709551615
  TEST(ReadCommandLine, StartFrameWithoutStartFrom_IsFailure)
  TEST(ReadCommandLine, AssumeCpuMemoryOnCheckOnly)      // A17: parsed on check; on run: failure
  TEST(ReadCommandLine, Version_PrintsCudaRuntime)       // A49: see the amendment
  TEST(ReadCommandLine, ChooseModeAgainOnResumeOnly)     // A40: resume run/ --choose-mode-again sets
                                                         // choose_mode_again; on run or check: failure, not help
  TEST(ToExitCode, EveryRowOfTheTable)                   // one case per IoError and step error the table names,
      // and kModuleFailed with any module's kRunOnCallFailed -> 6
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** `ReadCommandLine` catches `CLI::Error` and `...`; help gives
  exit 0, every other failure exit 2 (decided by the caller from `is_help`). **Step 4:** gate. **Commit.**
  Subject: `Read the command line in the one file that allows exceptions`. New names: the 3.7 list above and
  the `RTTI` argument of `fluidity_apply_compile_flags`.

### Task 35: Events

**Amended 2026-11-14 (A17, batched-run buffers):** the events carry the budget, the measured cost and the time
projection with the field names of `run_budget.fbs`.

**Amended 2026-11-15 (A60, A58, A57):** `frame_done` gains `liquid_force_on_solids: [LiquidForceOnSolid]` (Task
29c; `events.fbs` includes the table from `run_report.fbs`); the budget and cost tables carry their new fields
(Tasks 23a, 29a) by inclusion, nothing else changes here. New test `EventWriter, FrameDoneCarriesLiquidForce` (a
hand-built `frame_done` with two solids: the JSON holds `"liquid_force_on_solids"` with `solid_id` 3 then 7 and the
six components of `pressure_force_n` and `viscous_torque_n_m` as written).

**Kind:** integration

**Files:**
- Create: `schemas/simulate/v2/events.fbs`, `native/fluidity/cli/event_writer.h`, `event_writer.cc`,
  `findings.h` (private), `findings.cc`, `event_writer_test.cc`
- Modify: the `cli/` CMake lines (`fluidity_add_schema(fluidity_cli_parts SCHEMAS events EMBED_TEXT events)`)

**Interfaces:**
- **Step 1 checks the printed digits** (design 5.8, "to verify") before any event depends on them.
- `events.fbs`: one table per event of the design 3.7 table, a `Finding` table with the fields of the `Finding`
  table, the five common fields (`protocol`, `event`, `sequence`, `sent_at`, `elapsed_seconds`); 64-bit ids are
  `string` (decimal text). **A17 plan decision:** `events.fbs` includes `run_budget.fbs`; `scene_checked` and
  `started` gain `budget: RunBudget`; `heartbeat` and `frame_done` gain `simulated_seconds_per_second: double` (0
  until known) and `projected_remaining_seconds: double = -1` (-1 until 3 frames of this process are listed);
  `done`, `stopped` and `failed` gain `budget: RunBudget` and `cost: RunCost`. Produces `class EventWriter` (one lock
  around one whole line written to standard output and, for `run` and `resume`, appended to `<run_folder>/events.jsonl`, each flushed; a failed write to
  standard output is reported once as `kEventsNotWritable` and the file copy goes on), and (private)
  `BuildFinding(const IoFailure&, const io::RunScene*, ...)` assembling the message of design 5.8 (module message,
  object name from `RunScene::names()`, frame, magnitude and limit, the last saved state).

- [ ] **Step 1: To verify against FlatBuffers 25.12.19** (`event_writer_test.cc`):
  `TEST(FlatBuffersText, OneLineStrictJson)` (`indent_step = -1`, `strict_json = true`: no newline, quoted names);
  `TEST(FlatBuffersText, DigitsOfDoubles)` (print 0.1, 1.0 / 3, 1e-300, 1763120000.123456789 and assert what the
  pinned version prints). Decision rule: other doubles take whatever precision it prints, recorded in the
  `events.fbs` header comment; if `sent_at` loses digits below 1 microsecond, `sent_at` is a `string` field filled
  with `std::to_chars` at 17 significant digits (the field name stays `sent_at`).
- [ ] **Step 2: Failing tests** (`LABEL cpu`): `EventWriter, SequenceHasNoGap`; `EventWriter, SentAtNeverDecreases`;
  `EventWriter, LongMessage_CutToSixteenKiB` (only `message` shortened); `EventWriter, FileCopyEqualsStandardOutput`
  (a pipe in place of standard output); `EventWriter, ClosedOutput_IsEventsNotWritableFileGoesOn`;
  `EventWriter, FrameDoneCarriesLiquidForce` (A60, above); `EventWriter, BudgetAndCostTablesPrinted` (A17: a hand-built `scene_checked` with Task 23a's one-piece budget and a
  `done` with Task 29a's cost: the JSON holds `budget.lines` and `cost.warnings` verbatim; a `heartbeat` built
  before 3 frames has `projected_remaining_seconds` -1); `BuildFinding, PressureNotConvergedNearPaddle` (the design
  5.8 sentence with frame 42, `3.1e-4 1/s`, `2e-5 1/s`, last saved state 40); `BuildFinding, IdsAsDecimalText`.
- [ ] **Step 3: Run** (fail). **Step 4: Implement.** **Step 5:** gate. **Commit.** Subject:
  `Print events as one JSON line from their schema`. New names: `EventWriter`.

### Task 36: `capabilities` and `check`

**Amended 2026-11-14 (A17, batched-run buffers):** `check` prints the budget: as the `budget` table of
`scene_checked` and as its lines on standard error; a scene that does not fit exits 3 with the lines saying what
would make it fit.

**Amended 2026-11-14 (reconcile):** the base text plans through `BuildSceneBudget`, as the amendment says.

**Amended 2026-11-14 (A24):** `capabilities` lists `step::kCapabilityReasonCount` reasons (17 at most, 14 if the
viscosity plan lifts its three exclusions): `viscous_liquid_with_snow` no longer exists. **Replaced (A27): 16 at
most, 13 if all three are lifted.**

**Amended 2026-11-14 (A27):** `capabilities` lists `step::kCapabilityReasonCount` reasons, 16 at most (13 if the
viscosity plan lifts its three exclusions): `viscous_liquid_with_free_solid` no longer exists; its
`suggested_changes` names hold no `set_viscosity_to_zero` (the sand module adds it back with
`kViscousLiquidWithSand`).

**Amended 2026-11-14 (A25):** in a scene with liquid the budget lines `check` prints hold the `liquids:` block
(Task 23a) before the last `fits:` line; nothing else changes here.

**Kind:** integration

**Files:**
- Create: `native/fluidity/cli/capabilities.cc`, `check_scene.cc`, `commands.h` (the five command functions of
  design 3.7), `capabilities_test.cc`, `check_scene_test.cc`, `native/fluidity/io/testdata/scenes/four_problems.json`

**Interfaces:**
- Produces design 3.7 `WriteCapabilities`, `CheckScene`. `capabilities` reads only constants (no file, no
  processor unless `--run-on gpu`, then `CheckCanRunOn`, `ReadProcessorDescription`, `ReadMemoryAvailable`).
  `check`: `CheckSceneFile` (every finding), `BuildRunScene`, `step::CheckSceneCanRun`, `CheckCanRunOn` for the
  requested `RunOn` (never another, A12), `BuildSceneBudget` (Task 23a, which calls `step::BuildRunPlan` with
  `assume_memory_bytes` and, for a batched plan, `batches::BuildBatchPlan`; reconcile: `check` does not call
  `step::BuildRunPlan` itself), fingerprint,
  declarations, with `--cache` `CompareWithRunFolder`; `scene_checked` whenever the file could be read; a failing
  `CheckCanRunOn` or a plan that does not fit is a blocking finding with `run_on_cpu` (GPU asked) and
  `use_bigger_cells`. Writes no file. **A17 plan decision:** `check` calls `BuildSceneBudget` (Task 23a), puts the
  budget in `scene_checked.budget` (also when it does not fit) and writes `budget.lines` to standard error, one per
  line, after `scene_checked` is on standard output; exit status 0 when the budget's verdict is `kFits` and no
  other finding blocks, else 3 (`kSceneCannotRun`; the existing `does_not_fit` finding carries `use_bigger_cells`
  and, for `kDoesNotFitCpu`, the suggestion text of the last line in its message).
- `four_problems.json` (design test 19, plan decision on the fourth problem: render output's "surface group of a
  snow material" joins in its plan): a duplicate solid id, a negative source volume rate, a shared export name,
  points of the gas material.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, events captured from an `EventWriter` on a pipe):

  ```cpp
  TEST(WriteCapabilities, WithinTwoHundredMilliseconds)      // design test 16 (first part)
  TEST(WriteCapabilities, EveryCapabilityReasonAndLimit)     // kCapabilityReasonCount reasons (16 at most, A27; no
      // viscous_liquid_with_snow, no viscous_liquid_with_free_solid), CapabilityLevel value and message; kMaxPlateCount,
      // kMaxFreeSolidCount, kMaxIngredientCount, kMaxSizeLevels, motion samples 1..64; run_on_values ["cpu"] in a
      // CPU build; coordinate text right_handed_z_up_metres
  TEST(CheckScene, AssumeMemoryOneByte_DoesNotFitExitThree)  // design test 16 (second part): estimate printed,
      // blocking does_not_fit with use_bigger_cells; no buffer built (the replaced operator new counts no Buffer);
      // A17: standard error ends with a line starting "fits: no. To fit (estimates):"
  TEST(CheckScene, FallingBlockOnCpu_BudgetLinesOnStandardError)  // A17: standard error == the event's budget.lines
      // joined by newlines; first line "plan for falling_block on cpu (<the CPU's model>)", second "mode: one piece
      // (everything fits the cpu)", last "fits: yes"; exit 0; standard output holds JSON lines only
  TEST(CheckScene, AssumedMemoryForcesBatches_BudgetSaysWhy)  // A17: liquid_source_solids.json with
      // --assume-memory just below the one-piece estimate: run_mode "batched", the mode line from the batched
      // template with <p> = cpu; exit 0
  TEST(CheckScene, AssumeCpuMemoryTooSmall_DoesNotFitCpuExitThree)  // A17: both overrides small: verdict
      // "does_not_fit_cpu", to_fit filled, exit 3
  TEST(CheckScene, FourProblems_FourFindingsExitThree)       // design test 19: each with scene_file_path and ids
  TEST(CheckScene, GpuAskedInCpuBuild_RunOnCpuSuggested)     // gpu_not_built, exit 3
  TEST(CheckScene, CacheAgainstFinishedRun_ComparisonPrinted)
  TEST(CheckScene, WritesNoFile)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Report what this build can do and check a scene`. New names: `WriteCapabilities`, `CheckScene`.

### Task 37: `run`, `resume` and the `fluidity` executable

**Amended 2026-11-14 (A17, batched-run buffers):** `run` and `resume` print the budget at the start and the cost at
the end (standard error and `log.txt`), with a projected finish after the third frame.

**Amended 2026-11-14 (A25):** the budget lines printed at the start hold the `liquids:` block (Task 23a) in a scene
with liquid; nothing else changes here.

**Amended 2026-11-15 (A40):** `resume` passes `--choose-mode-again` to `BuildRunToContinue` through the overrides
(Task 29d); a `kRecordedModeDoesNotFit` refusal prints the `failed` event with that code and its message, exit 3,
and writes nothing to the folder but `log.txt`. New test `AdvanceRun, ResumeRecordedModeDoesNotFit_ExitThree`
(Task 29d's second case through the command: exit 3, the `failed` finding has code `recorded_mode_does_not_fit` and
its message names `--choose-mode-again`; the same command with `--choose-mode-again` exits 0).

**Amended 2026-11-14 (Foundation as built):** the CMake lines keep `fluidity_cli`'s compile flags and list every
new source and test.

**Kind:** integration

**Files:**
- Create: `native/fluidity/cli/advance_run.cc`, `stop_input.h` (private), `stop_input.cc`,
  `frame_writer_without_products.cc` (private), `advance_run_test.cc`
- Modify: `native/fluidity/cli/main.cc`, the `cli/` CMake lines (`fluidity_cli` from `main.cc` and
  `fluidity_cli_parts`, with `fluidity`,
  `fluidity_command_line`, `CLI11::CLI11`, `flatbuffers::flatbuffers`, `Threads::Threads`; `OUTPUT_NAME fluidity`;
  its `fluidity_apply_compile_flags(fluidity_cli EXCEPTIONS OFF)` of Task 3 kept; every new `.cc` of this task
  listed, `advance_run_test` as `LABEL cpu LINKS fluidity_io_test_support`)

**Interfaces:**
- Produces design 3.7 `AdvanceRun` (`run`, `run --start-from`, `run --outputs`, `resume`): builds `RunParts`, then
  `BuildRun` / `BuildRunToContinue` / `BuildRunFromSavedState` / `RebuildOutputs`; `started`, `preparing`,
  `heartbeat`, `try_done` (with `--report-every-try`), `finding`, `frame_done`, `frame_checked`,
  `saved_state_listed`, `stopping`, and exactly one of `done`, `stopped`, `failed` last; the `failed` line also on
  standard error and in `<run_folder>/log.txt`.
- Private: one helper thread that polls standard input every 25 ms without blocking (port
  `git show main:native/cli/simulate_job.cc` lines 25-80 for `PeekNamedPipe` / `ReadFile` on Windows; `poll` with
  timeout 0 on POSIX) and writes the heartbeat every 0.5 s (lines 262-288); `cancel` sets `cancel_requested`;
  `stop_after_frame` and the end of input set `stop_after_frame_requested`; SIGINT and SIGTERM (Windows
  `SetConsoleCtrlHandler`) set `cancel_requested` only. Plan decision, the executable's frame writer in this
  module: `BuildFrameWriterWithoutProducts()` lists no product file and calls `WriteNotBuilt` for each declared
  output of each frame with reason `"no_product_writer"`; render output's plan replaces it.
- `main.cc`: `ReadCommandLine`, dispatch, return the `ExitCode` (help 0, bad command line 2).
- **A17 plan decisions:** before `BuildRun` (and `BuildRunToContinue`), `AdvanceRun` builds the budget with
  `BuildSceneBudget` and writes its lines to standard error and `<run_folder>/log.txt` (to standard error only
  when the folder is not made); `started` carries it. After the third frame listed in this process and every 10th
  after it (13th, 23rd, ...), one line `time: about <duration> left (<rate> simulated s per s over frames <a> to
  <b>)` goes to standard error and `log.txt`. At the end, `cost.lines` then `cost.warnings` go to standard error and
  `log.txt`, after the last event is on standard output; the last event carries `budget` and `cost`; the run report
  holds both. A refused budget gives exit 3 with the lines printed.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the command functions called in-process with a pipe):
  `AdvanceRun, RunThreeFrames_DoneLastExitZero`; `AdvanceRun, ResumeAfterStop_DoneAndSameInternalFiles`;
  `AdvanceRun, NotBuiltFindingPerDeclaredOutput`; `AdvanceRun, FolderInUse_ExitFiveNamesHolder`;
  `AdvanceRun, HeartbeatWhileSolving` (a 2 s `step::AdvanceTo` slowed by a `sleep_seconds` text writer and
  `max_waiting_frames = 1`: heartbeats at most 1 s apart, activity `waiting_for_output`);
  `AdvanceRun, BudgetAtStartCostAtEnd` (A17: falling_block.json, 12 frames on the CPU: standard error starts with
  the lines of `started.budget`, holds exactly one `time: about` line (after the third frame listed in this process;
  the next would follow the 13th), ends with `cost.lines`; `log.txt` holds the same lines; `run_report.json` has `budget` and `cost`); `AdvanceRun,
  DoesNotFit_ExitThreeLinesPrinted` (A17: `assume_memory_bytes` 1 through the overrides: exit 3, no folder, standard
  error ends with the "fits: no." line); `AdvanceRun, ResumeRecordedModeDoesNotFit_ExitThree` (A40, above).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate (add `AdvanceRun`). **Commit.** Subject:
  `Add the fluidity program with run and resume`. New names: `AdvanceRun`, the executable `fluidity`.

### Task 38: `report` and `read-particle`

**Kind:** integration

**Files:**
- Create: `native/fluidity/cli/write_report.cc`, `write_particle.cc`, `report_command_test.cc`

**Interfaces:**
- Produces design 3.7 `WriteReport` (`report` event from `ReadRunFolderFacts`, `is_crashed`, with
  `--check-files` each failing file) and `WriteParticle` (`particle` event from `ReadParticle`, ids as decimal
  text).

- [ ] **Step 1: Failing tests** (`LABEL cpu`; design test 23 through the command): `WriteParticle,
  AsymmetricFrameZeroAndOne_Exact` (position and velocity components printed equal to the written doubles at
  frames 0 and 1); `WriteParticle, UnknownId_ParticleNotFoundExitOne`; `WriteParticle,
  NoInputsNoSavedState_FrameInputsNotKept` (`keeps_frame_inputs` false, frame 3); `WriteReport,
  FinishedRun_EveryField`; `WriteReport, CheckFiles_NamesDamagedFile`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Report a run folder and read one particle from the command line`. New names: `WriteReport`, `WriteParticle`.

### Task 39: Events and processes end to end

**Amended 2026-11-14 (A17, batched-run buffers):** one more process test reads the budget and the cost from the
events and from `run_report.json` by their schema names.

**Amended 2026-11-15 (A58, A60):** `test_budget_and_cost_fields_by_schema_name` also reads
`processor_memory.memory_share_of_free`, `memory_left_free_bytes`, and in `done` `cost.planned_bytes`,
`reserved_bytes`, `reserved_block_count`, `handed_out_bytes`, `memory_taken_bytes`, `requested_beyond_plan_bytes`,
`reserved_cpu_bytes`, `handed_out_cpu_bytes`; on `liquid_source_solids.json` every `frame_done` after the first has
`liquid_force_on_solids` with one row per solid and the fields of Task 29c.

**Kind:** integration

**Files:**
- Create: `tests/cli/conftest.py`, `tests/cli/test_events.py`, `tests/cli/test_processes.py`
- Modify: the `cli/` CMake lines (CTest `cli_end_to_end`: `uv run pytest tests/cli -q --fluidity
  $<TARGET_FILE:fluidity_cli> --flatc ${FLUIDITY_FLATC_EXECUTABLE} --scenes native/fluidity/io/testdata/scenes`,
  `LABEL cpu`, `TIMEOUT 900`)

- [ ] **Step 1: Tests** (pytest; each run in its own `tmp_path` folder, under the home folder in the container):
  - `test_every_line_is_strict_json_of_the_schema` (design test 14): every line of a 5-frame run parses with
    `json.loads(..., parse_constant=reject)` and converts with `flatc -b --strict-json events.fbs <line file>`
    without error (unknown fields refused); `sequence` has no gap; `sent_at` never decreases and is within 1 s of
    `time.time()` when read; exactly one of `done`, `stopped`, `failed`, and it is last; `events.jsonl` equals
    standard output line for line.
  - `test_heartbeat_during_long_frame` (design test 14): `small_syrup.json`, one frame: heartbeats at most 1 s
    apart while `last_advance_sent_at` stays unchanged inside the frame.
  - `test_closing_stdout_stops_after_the_frame` (design test 14) and `test_end_of_input_stops_after_frame`
    (design test 13, the input part): stdin closed during frame 7: frame 7 and its saved state listed, exit 4,
    `stopping` with `end_of_input`.
  - `test_second_writer_refused_then_succeeds` (design test 6): the second `resume` exits 5 after 0.9 to 2 s,
    folder bytes unchanged; after the first is killed it succeeds; `report` during the first: `is_held` true and
    its process id; after the kill: `is_held` false; 50 `report` probes while a `run` starts never refuse it.
  - `test_report_is_read_only` (design test 22): during a run, after a kill and after a clean end the folder's
    bytes and modification times are unchanged by `report`, and its facts equal the file list and run report.
  - `test_budget_and_cost_fields_by_schema_name` (A17): `check falling_block.json`: `scene_checked.budget` has
    `run_mode`, `mode_reason`, `verdict`, `processor_memory.budget_bytes`, `allowed_bytes`, `total_bytes`,
    `free_bytes`, `cpu_memory` (same four), `modules[].module`, `processor_bytes`, `cpu_bytes`,
    `copies.bytes_per_step`, `copies_per_step`, `snapshot_bytes_per_frame`, `lines`; standard error equals
    `"\n".join(lines)`; `check --assume-memory 1 --assume-cpu-memory 1`: exit 3, `verdict` `does_not_fit_cpu`,
    `to_fit.max_particle_count`, `min_smallest_cell_width_m`, `more_memory_bytes` present; `run --last-frame 4`:
    `started.budget` equal to the check's except the free bytes, every `frame_done` has
    `projected_remaining_seconds` (-1 for the first two frames of the process, >= 0 from the third), `done` has
    `cost.largest_processor_bytes`, `copy_bytes_per_step_mean`, `stage_seconds[].stage`, `total_seconds`,
    `warnings`, `lines`; `run_report.json` holds `budget` and `cost` with the same names (the add-on reads these
    names, add-on plan Task 12).
  - `test_small_reference_scenes` (the first end-to-end runs): for each of the four small scene files:
    `check` exits 0 with `can_run` true; `run --last-frame 2` exits 0; `resume --last-frame 3` exits 0 with
    `frame_done` for 3; `report` shows `last_listed_frame` 3; `read-particle` of the first particle id of frame 3
    exits 0.
- [ ] **Step 2: Run.** Fix owning commits if needed. **Step 3:** gate. **Commit.** Subject:
  `Run the fluidity program end to end on the reference scenes`.

### Task 40: Behaviour check against the proof of concept

**Kind:** integration

**Files:**
- Create: `native/fluidity/io/testdata/scenes/proof_of_concept_defaults.json`,
  `native/fluidity/io/testdata/proof_of_concept.json`, `native/fluidity/io/proof_of_concept_test.cc`

- [ ] **Step 1: Produce the old numbers** (supporting only). Export `main` outside the worktree
  (`plan-decisions.md`); build its CPU preset; run its `fluidity` on
  `native/tests/fixtures/liquid_cache/resolved_defaults.json` for 3 frames and record the particle count and the
  three frame times, with `main`'s commit hash. Write the same scene as a v2 scene file, rotated from Y-up with
  rows `{1,0,0},{0,0,-1},{0,1,0}`.
- [ ] **Step 2: Test** (`LABEL cpu`) `ProofOfConcept, DefaultsSceneCountAndFrameTimes`: start particle count
  equal, frame times equal to 1e-15 relative. Frame times are also covered by `ComputeFrameTime, FrameTen`.
- [ ] **Step 3:** gate. **Commit.** Subject: `Compare the scene reading with the proof of concept`.

### Task 41: GPU runs for the owner

**Amended 2026-11-14 (reconcile):** the byte test between modes is replaced by the A15 tolerance test and per-mode
repeat runs.

**Amended 2026-11-14 (Foundation as built):** every `gpu` test writes `WriteResultDigest` of its run's internal-file
digests and `FrameDone::snapshot_digest`s, so the owner's repeat run compares them; the two tests are registered in
the module CMake files with `LINKS fluidity_io_test_support`.

**Kind:** device

**Files:**
- Create: `native/fluidity/io/io_gpu_test.cc`, `native/fluidity/cli/capabilities_no_gpu_test.cc`
- Modify: `native/fluidity/io/CMakeLists.txt` (`LABEL gpu`), `native/fluidity/cli/CMakeLists.txt` (`LABEL no_gpu`)

- [ ] **Step 1: Tests.** `LABEL gpu` (owner's run, design 7 "GPU"; each test calls `WriteResultDigest` with the
  digests of the internal files and of every `FrameDone::snapshot_digest` it produced; GPU against CPU only within
  the step's design 6.4 tolerances, GPU against GPU of one mode by files): `ContinueRun, OnGpuFourWaysSameFiles`
  (design test 9 (a) to (d) with `RunOn::kGpu` on `liquid_source_solids.json`); `AdvanceRunTo,
  OnGpuBatchedWithinStepToleranceOfOnePiece` (test 18 on the GPU, A15 form, both batched modes; reconcile, was
  `OnGpuBatchedEqualsOnePiece`); `AdvanceRunTo, OnGpuEachModeRepeatRun_SameFiles` (reconcile: each of the three
  modes twice, same files); `AdvanceRunTo, OnGpuRepeatRunSameFiles`;
  `BuildRunToContinue, CpuSavedStateOnGpu_IsWrongRunFingerprintRunOn` (`differing_part` `kRunOn`);
  `BuildRunFromSavedState, CpuStateOnGpu_OneFrameWithinStepTolerances` (step design 6.4 tolerances). `LABEL no_gpu`
  (CUDA build in the container): `WriteCapabilities, GpuAskedWithoutGpu_NoGpuFoundExitZero` (design test 16:
  finding `no_gpu_found`, `ToName(RunOnError::kNoGpuFound)`, within 200 ms).
- [ ] **Step 2:** gate. **Commit.** Subject: `List the I/O runs for the owner's GPU run`. Body:
  `CUDA: compiled, not run`; this module has no device code, these tests drive the step's.

### Task 42: Comment pass and hand-over

**Amended 2026-11-14 (Foundation as built):** the hand-over records the target-flag, test-wrapper, helper-library and
packing-memory choices.

**Kind:** mechanical

**Files:**
- Modify: any file of `native/fluidity/io/`, `native/fluidity/cli/`, `schemas/simulate/v2/`, `reference/io/`,
  `tests/cli/`, `docs/method-paper/sections/files.typ`, `docs/glossary.md`, `docs/gpu-runs.md`,
  `tools/check_rules_allowed_names.txt`

- [ ] **Step 1: Comment pass.** Reread every file of the module as a C++ programmer who has never read a fluid
  paper: comments true of the final code; why, not what; a worked example with numbers for `WriteListedFile` (a
  saved state of frame 40: the temporary name, the rename, the entry), `ComputeRunFingerprint` (which edits change
  it, which do not) and `AdvanceRunTo` (frames 10 to 13 with a stop after 12); every word in the glossary; `Uses` /
  `Used by` lists; `// Step N:` numbers match `<proc-write-listed-file>` and `<proc-advance-run-to>`; every
  schema field has an explicit id and a unit comment.
- [ ] **Step 2: Fold each change into its owning commit** (`git commit --fixup`, `git rebase --autosquash`); run
  format and rules on each rewritten commit and the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3 to 10) and record what was delivered differently: the
  render-output fields left out (Global Constraints); `RunParts` without cell resize and batched run;
  `assume_memory_bytes`; `FileCheck`; `FileKind::kStartParticles`; the base `module_error` encoding; the unjittered
  inside test of start sampling; `file_pattern` without extension; the executable's writer without products; the
  stage-by-stage cancel left to the step; the fourth problem of test 19 and the case of test 20 replaced; the
  FlatBuffers findings of Tasks 13, 17, 18 and 35 (and the strict JSON check if it was added); the Python
  `flatbuffers` version; design test 21's parts handed to render output (two liquids, foam, gas volume and their
  channels; `--outputs foam_and_spray` refused with `start_new_run`; the surface spacing rebuild); the A17
  budget and cost (amended 2026-11-14: Tasks 23a and 29a, `BuildSceneBudget`, `assume_cpu_memory_bytes` and
  `--assume-cpu-memory`, the lines on standard error, the event and report fields, the two warning constants for
  the owner); Foundation as built (amended 2026-11-14): `fluidity_apply_compile_flags` on `fluidity_cli_parts`,
  `fluidity_cli` and `fluidity_command_line`, the `noexcept` wrapper in place of a test built without exceptions,
  `fluidity_io_test_support` made in Task 15, `SnapshotPackingBuffers` as CPU packing memory and its largest size;
  (amended 2026-11-15) the A40-A64 pass: no grid-only region (the later group stays open), artist cell sizes and
  the libspng version, the slip length as two fields, `free_solid` naming, the budget rule and the warnings without
  margins with `kMemoryTakenToleranceBytes` for the owner's Windows run, the viscosity advice lines, the liquid
  force rows, resume in the recorded mode and `kOutOfMemory` last.
- [ ] **Step 4: `docs/gpu-runs.md`:** an `io and cli` row, `compiled, not run`, listing the `gpu` tests of Task 41.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the module, the gate's output, new public names
  with their one-line reasons, the list of Step 3, and the statement that no device code has run.

## Deep liquid without particles (later; needs review)

**Amended 2026-11-15 (A64):** new group. These tasks are excluded from this module's done-criteria. An
independent review of this design comes first. The owner of the group is the cell resize and batches plan's group
"Deep liquid without particles (later; needs review)", gated by `GridOnlyRegion, StandingWave_KeepsAmplitude` and
`GridOnlyRegion, ParticleBandEdge_StaysStable`; nothing here starts before both pass.

### Task 42a: Grid-only region in the scene file, the run and the files (later)

**Kind:** integration

**Files:**
- Modify: `schemas/simulate/v2/scene_file.fbs`, `ready_scene.fbs`, `array_file.fbs`, `run_report.fbs`,
  `native/fluidity/io/run_scene.cc`, `snapshot_arrays.cc`, `run.cc`, `run_report.h`, the tests named below

**Interfaces** (what A64 took out of Tasks 19, 20 and 29, as the cell resize group then declares it):
- Scene: `cell_resize.regions` (`particle_region_depth_in_cell_widths` 6, `grid_only_depth_in_cell_widths` 7,
  `overlap_layer_count` 2, `particles_per_refilled_cell` 8, each with its range check and path).
- `Run`: the callbacks `apply_grid_only_region`, `move_grid_only_liquid` and `read_largest_grid_only_speed` (a
  private wrapper of `ReadLargestGridOnlySpeed`), each with the `RunState` context.
- Saved state: the six grid-only arrays by active cell (`is_grid_only_region`, `layers_from_grid_only_region`,
  `grid_only_mass`, `grid_only_rest_volume`, `grid_only_current_volume`, `grid_only_ingredient_volume`) after the
  merge-wait arrays; `step_limits.grid_only_speed`. Frame inputs: `grid.is_grid_only_region`,
  `grid.grid_only_current_volume`, `grid.grid_only_ingredient_volume` (what the surface reads).
- Run report: `cells_to_grid_only`, `cells_to_particles`.
- Budget: the A64 deep-particles line is dropped when the scene has a grid-only region.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `WriteSavedState, GridOnlyArrays_RoundTrip` (a deep tank whose bottom
  half is grid-only after one resize: the six arrays read back equal, `ExpectSameBytes`); `ContinueRun,
  GridOnlyTank_FourWaysSameFiles` (design test 9 on that tank); `RunReport, GridOnlyTotals` (`cells_to_grid_only`
  equals the sum of the resize reports); `CheckSceneFile, GridOnlyDepthNotAboveParticleDepth_IsValueOutOfRange`
  (6 and 6: `cell_resize.regions.grid_only_depth_in_cell_widths`).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Carry the grid-only
  region through the scene, the run and the files`.

## Seams the render-output plan completes

Each item is a place this plan leaves for render output's Task 47 (`plans/2026-11-14-rebuild-13-surface-foam-and-
spray.md`); `git grep -n "render output's plan" native/fluidity/io native/fluidity/cli` finds a comment at each.

1. `schemas/simulate/v2/scene_file.fbs` `RenderSection` and its `ready_scene.fbs` presentation table (empty here):
   the `RenderSettings` fields; `RequestedOutputs::render`; `RecordsSteps`; the `CheckRenderSettings` finding in
   `CheckSceneFile` (Tasks 13, 16, 17).
2. `step::BuildRunPlan`'s `output_estimate`: `BuildRenderOutputMemoryEstimate` beside `BuildIoMemoryEstimate` (Task 23).
3. `step::StepCallbacks` composition in `run.cc`: `make_foam_and_spray` (`RecordStep`, false only on
   `kRunOnCallFailed`) and the commit or reject of the recorded step in this module's `on_try_done`;
   `RunParts::step_recorder` (Tasks 23, 29).
4. `FrameOutput` and `KeptFrameInputs`: `frame` and `time` become `render_output::FrameIdentity identity`;
   `recorded` and `sample_at_frame` added, and their arrays in `FrameInputsDescription` (Tasks 20, 22).
5. `SavedState::render_output`, `ProductsOfFrame::render_output`, the `kept` argument of `FrameWriter::start`,
   and their arrays in `SavedStateDescription` (Tasks 19, 22, 24).
6. `OutputDeclaration::product`, `format`, `contract_version`; the format's extension appended to
   `file_pattern`; `channels` from `ReadOutputChannels()`; `ComputeOutputFingerprint` hashing product, format and
   contract version; surface, droplets, foam and spray, foam volume and gas volume declarations in
   `BuildOutputDeclarations` (Task 18).
7. The CLI's frame writer: `BuildFrameWriterWithoutProducts` replaced by the writer that calls `BuildRenderFrame`
   and the interchange writers; `fluidity_cli` links `fluidity_interchange` (Task 37).
8. `capabilities`: `ReadFormatTable()`, `ReadOutputChannels()`, `ReadInterchangeLibraries()` with
   `missing_resource_folder`; `RunReport::interchange_libraries` (Tasks 23, 36).
9. `ToName(FailingModule::kRenderOutput, e)` and `ToMessage` rows (Task 4).
10. Tests: design test 19's surface-group-of-snow problem (Task 36), test 20's surface spacing case (Task 31),
    test 21's two liquids, foam, gas volume, channels and `--outputs foam_and_spray` refused with
    `start_new_run` (Tasks 28, 32).
11. (Amended 2026-11-14, A17, batched-run buffers.) The recorder's copies in the measured cost: `on_step_done`
    adds the difference of `render_output::ReadRecorderCopyCounts` since the last committed step to the copies given
    to `ApplyStepToRunCost` (Task 29a); a comment "render output's plan: recorder copies" marks the place.
12. (Amended 2026-11-15, A47.) `render_output::RenderSettings::wind` (`gas::GasWind`) is set from
    `ReadyScene::wind` when `RequestedOutputs::render` is filled (Task 27), with the keys `RunScene` owns; the scene
    file has no separate wind for render output. A comment "render output's plan: wind" marks the place.
