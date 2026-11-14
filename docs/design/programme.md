# FluidiTy production rebuild: programme design

Date: 2026-11-13. Status: approved by the owner on 2026-11-13.

This is the design for the whole programme. Each module below gets its own short design before it is built.
This document fixes what those module designs may not reopen: the branch, the order, the rules every commit
obeys, the architecture shared by CPU and GPU, and what is deliberately left behind.

## 1. Intent

FluidiTy today is a proof of concept written with LLM help in about a week: 976 commits, about 141,000 lines of
native code, no users. The owner wants it production ready. Two inputs say what that means:

- The **owner's naming review** of 2026-11-13 (`inputs/owner_naming_comments_architecture.md` in the report
  bundle, also `docs/reviews/naming-comments-architecture-2026-11-13.md`, which is gitignored). Its items marked
  Decided are requirements.
- The **engineering report** of 2026-11-13 audited at commit `cb16ffd`
  (`/Users/tychota/Downloads/FluidiTy_Report_Source`). Its diagnosis and its keep/replace/retire table are
  adopted. Its recommendations are proposals; this document says which ones are taken.

Because there are no users, nothing has to keep working during the change and nothing has to stay readable by
old files. The owner chose to rebuild on a fresh branch with a clean history instead of converting in place.

## 2. Goals, non-goals, anti-goals

**Goals**

1. One coherent solver: one scene model, one step entry point, one owned state, for liquids, snow, sand, cutting,
   moving solids and gas (fire and smoke) alike.
2. CPU and GPU implementations of every operation, designed together, behind one public function per operation.
3. Code a good C++ programmer new to fluids can read: decided vocabulary, teaching headers, concept folders.
   The rules drawn from the naming review are the single authority for how code is written and for how it is
   checked.
4. A solver that sizes itself: it analyses the scene, estimates memory, runs in one piece when the scene fits
   the device and in several batches when it does not (section 8a).
5. A history that reads as a design: small commits, each correct when it lands.
6. Heavy deletion. Code, tests, documents, Python reference, Lean proofs and tooling exist on the new branch
   only if a module design asked for them.

**Non-goals**

- Compatibility with any existing file, cache, schema, CLI flag or add-on setting.
- The report's algorithm experiments E01 to E08 (divergence-free interpolation, multilevel pressure hierarchy as
  a research question, new bubble model, neural meshing and so on). The rebuild ports the current methods with
  their known defects fixed. An experiment is a separate decision after the module it touches is done.
- Performance tuning (block sizes, FMA, reduced precision, graph capture). The report defers these to
  measurement and so does this programme. Structural performance (resident device state, parallel solves) is in
  scope because it is architecture.
- A production 2D path.

**Anti-goals** (the rebuild has failed if these happen, even if it runs)

- A liquid-only core with other materials bolted on afterwards as separate routes.
- A GPU path that copies to the host between dependent stages.
- A scene-specific public step method, a `bounded`-style mode flag, or a milestone code in a name.
- A "fix later" commit, a commented-out block, a skipped test without a stated reason, or a claim that GPU code
  works when it has only been compiled.
- Old code carried over because it existed, not because a module design asked for it.

## 3. Approaches considered

- **A. Orphan branch, port module by module (chosen by the owner).** Start empty. Each module arrives in its
  final form. The old branch is a read-only reference. Deletion is the default; history is clean. Cost: no
  end-to-end step until several modules are in, and "same numbers as before" is not available as a check because
  names, axis convention and arithmetic change together.
- **B. Convert in place on `main` following the report's twelve review units.** Always runs and old golden values
  catch mistakes, but everything survives until someone argues for deleting it and the history stays noisy. This
  is the route for a project with users to protect.
- **C. Orphan branch seeded with one pruned snapshot, then convert in place.** Clean history and a working tree
  from day one, but the first commit is a 100,000-line blob and cleanup is still opt-out.

A was chosen because production quality plus heavy deletion needs code to earn its place. It departs from the
report in one respect: the report's rule "behaviour patch first, then a rename patch that changes no number"
cannot apply when a module is written once. Section 7 replaces it with independent checks per module.

## 4. Decisions

Each line says who decided.

| # | Decision | Source |
|---|---|---|
| D1 | Rebuild on an orphan branch; old history stays reachable; the new branch later becomes `main`. | Owner |
| D2 | Every commit is right first time (section 6). | Owner |
| D3 | Snow, sand, cutting, moving solids and gas are in the main sequence, not follow-on projects. Gas is ported like the rest: its proof of concept is the branch `origin/codex/m7-low-mach-gas`, pushed by the owner on 2026-11-14 and read with `git show`. | Owner |
| D4 | CPU and GPU land together in each module. CUDA is compile-checked in a Linux container here. The owner runs everything on the RTX 5060 Ti once, at the end, after all modules are written (changed 2026-11-13; earlier drafts had one run per module). | Owner |
| D5 | Vocabulary, naming rules, comment templates, concept folders, `RunOn`, one error enum per module: naming review sections 4 to 7. | Owner |
| D6 | Z-up, right-handed, like Blender. Gravity is `{0, 0, -g}`. | Owner |
| D7 | Exactness: same GPU and same input give identical output on every run. CPU and GPU agree within a tolerance. No fixed-point deposits, no rounding certificates. Sums are plain doubles in a fixed order. Added during Foundation, for the owner to confirm: discrete outputs are equal on both processors, and the functions whose header promises the same bits (the sums of `base/tree_sum.h` and `base/ordered_sums.h`, and the marked functions of the shared math) give the same bits on both unless the result is NaN. | Owner, except the addition marked for the owner to confirm |
| D8 | No 2D production code. | Owner |
| D9 | Module order for the liquid modules: contact, grid, transfer, pressure, viscosity, step, I/O. The mixed-size grid is not a late module: grid, particles, transfer, pressure and viscosity are built for cells of mixed sizes from their first commit. | Owner (order: naming review; mixed-size from the start: 2026-11-13) |
| D10 | Materials, moving solids and gas are placed before the step, so the step is designed once for everything. | Owner |
| D11 | Host code is C++23 with `std::expected`; device code and shared headers stay in the C++20 subset. Abseil is not carried over. | Owner |
| D12 | Cell resizing is its own transaction after a committed step, not part of the physical step. | Owner |
| D13 | Schema v2 only. No v1 reader. | Proposal in this document; follows from "no users" |
| D14 | Pressure and viscosity are built parallel and multi-block on GPU from the start, not ported as the one-block and one-thread solves the report found. | Proposal (report, CUDA chapter) |
| D15 | Everything that works on `main` today is in scope and has a home in section 9, including several liquids and mixtures, surface tension, sources, particle split and merge, the wave paddle, the open-sea side and outlet, volume output and motion output. | Owner |
| D16 | A `particles/` module is inserted after grid. | Owner |
| D17 | Work is local for now: commits go on the local `rebuild` branch, nothing is pushed and no pull request is opened until the owner says so. History on `rebuild` may be rewritten, even after a push, until the project is production ready. | Owner |
| D18 | The CPU path gives identical output run to run on the same machine and build. Across machines and compilers it agrees within a tolerance. | Proposal in this document |
| D19 | The naming review itself is not committed: it is an audit of the old code. Its decided rules are rewritten as the new `AGENTS.md`, its vocabulary as the glossary, and its open points are carried in this document (section 9). Those two files are the authority both for writing code and for verifying it. | Owner |
| D20 | The method paper is written again from the start in Z-up and the decided vocabulary, not copied. | Owner |
| D21 | Each module ends with a comment pass (section 7). | Owner |
| D22 | Python reference and Lean proofs are trimmed: a piece exists on the new branch only if a module design lists it, whether ported or newly added. Everything not listed is gone. | Owner |
| D23 | The solver estimates a scene's memory before running. If the scene fits the device it runs in one piece; if not, the state lives in host memory and passes through the device in batches of cell blocks (section 8a). Four reference scenes size the design. | Owner |

## 5. Branch and history

- New orphan branch `rebuild`. It shares no commit with `main`.
- The first commit contains this document, the new `AGENTS.md`, the glossary and nothing else.
- Work is local (D17). Commits go straight onto the local `rebuild` branch. Nothing is pushed and no pull
  request is opened. The owner decides when and how the branch reaches the Windows machine and the remote.
- History may be rewritten (D17). A bug found by the owner's hardware run, or later in a finished module, is
  fixed in the commit that introduced it, together with a test that fails without the fix, and the later
  commits are replayed on top. The branch therefore always reads as if it had been right the first time.
  Rewriting stops at the switch-over; from then on fixes are new commits.
- Switch-over, done by the owner when they choose: rename `main` to `poc`, rename `rebuild` to `main`. Proposed
  moment: when module 8 (step) runs a liquid scene on both CPU and GPU.
- The roughly 170 remote `codex/*` branches, the two root `.bundle` files and the gitignored `docs/reviews/`
  folder are not touched by this programme. Deleting remote branches is the owner's call at switch-over.

## 6. The commit rule

A commit lands only when all of these hold:

1. It does one thing and its message says why.
2. The strict CPU build passes locally (warnings are errors) and every test passes.
3. The CUDA target compiles in the Linux container.
4. `clang-format` 23.1 reports no change and `clang-tidy` 23.1 reports nothing on host sources.
5. Every file it touches passes the checklist in `AGENTS.md`: file header, sectioned docstrings,
   unit comments, no banned word, no abbreviation, paper symbol, milestone code, `3D` suffix, bare `bool`
   parameter or repeated magic number, only the allowed verbs and role suffixes, numbered steps in long bodies,
   a worked example per non-trivial public function, `Read:` lines copied from the method paper.
6. New public names are listed in the commit message with a one-line reason. New domain words are added to the
   glossary in the same commit.
7. The method paper section the code cites exists and matches the code.
8. Tests arrive with the code they test, not after.

GPU honesty: device code is "compiled, not run" until the owner's hardware run, which happens once after all
modules are written (D4). Until then every module's status reads "GPU: compiled, not run". `docs/gpu-runs.md`
records each run: the commit, the device and the result per module. Nothing is described as working on GPU
before its record exists.

The cost of running late is accepted and stated: a mistake in the first device code can repeat in every later
module before hardware shows it. Two things limit that. Each module has tests that run in the container
without a GPU wherever the logic can be reached that way, and the device code of every module follows the one
pattern set in Foundation, so a fault in the pattern is fixed in one place and replayed through history.

Stop-and-ask rule: when a rename meets a word the naming review marks Verify or Open, work on that name stops
and the owner is asked. The known ones are listed per module in section 9.

## 7. How a module is shown to be correct

Old golden values cannot be the check, so each module uses, in this order of trust:

1. **Analytic and conservation tests** written from the physics (hydrostatic column, rigid rotation, linear
   shear, swept-volume piston, static drop). The report's numerics chapter lists a set per module; the module
   design picks from it.
2. **The Python reference**, where the module design lists one (D22). A listed piece is written in the new
   vocabulary and Z-up and kept deliberately different in implementation from the native code. The default for
   a module is none; the design must say what uncertainty a reference removes that an analytic test does not.
3. **Comparison with the proof of concept**, where the method is unchanged: run the old code on `main`, rotate
   its output into Z-up, compare within a stated tolerance. This is supporting evidence only.
4. **Lean proofs**, where the module design lists one (D22). Proofs about fixed-point accumulation are gone.
5. **GPU**: the same test cases as CPU compared within the module's tolerance, plus a repeat-run test that
   demands identical bytes on the same device.
6. **Gas**: its proof of concept is the branch `origin/codex/m7-low-mach-gas`, compared in the same way as
   item 3.

CPU tests compare discrete outputs (ids, counts, which cells are active) exactly and floating-point outputs
within a tolerance, so they pass on all three CI platforms. A repeat-run test demands identical bytes on the
same machine (D18). CPU stages may use threads only where each destination value is summed by one thread in a
fixed order.

Tolerances have no global value. Each module design proposes its own (an absolute floor plus a relative term, as
the report recommends) and the owner's hardware run calibrates the CPU-versus-GPU one.

**Comment pass.** After a module's code is complete and before it is called done, every file of the module is reread
from top to bottom as the stated reader: a good C++ programmer who has never read a fluid paper. The pass applies
the comment rules and the checklist in `AGENTS.md`, checks that each comment is still true of the final code,
removes comments that repeat the code, and adds the missing why, invariant or trap. Its changes are folded into the
commits that own the files and change no behaviour. When the last module is done, one more pass runs across all
modules to make wording and the `Words` lists consistent.

**A module is done when:** its design was approved; all commits obey section 6; the comment pass is complete;
the strict CPU build and tests pass here and the CUDA target compiles in the container; its GPU tests exist
and are listed for the final hardware run; and the module design's list of dropped and deferred items is
recorded. The programme is done when the owner's hardware run has passed for every module and is recorded in
`docs/gpu-runs.md`.

## 8. Shared architecture

**Layout**

```text
native/fluidity/
  base/             vectors, small matrices, error and result types (shared by host and device)
  contact/  grid/  particles/  transfer/  pressure/  viscosity/
  materials/  solids/  gas/  step/  cell_resize/
  io/  surface/  foam_and_spray/  cli/
reference/          Python reference, same folder names
proofs/             Lean proofs that describe current code
blender/            the add-on
docs/               glossary, method paper, GPU run log, one design per module
```

The naming review calls the step folder `liquid/`. With every material in one step, `step/` is proposed here and
goes through the naming review like any new public name. `solids/`, `particles/`, `gas/` and `cell_resize/`
are likewise proposed names. The review's `mixed_size_grid/` folder is not used as a separate late module
because every grid is a mixed-size grid from the start (D9); its types live in `grid/`, and what remains for a
later module is changing cell sizes during a run.

**One operation, two private implementations.** Each operation has one public header, for example
`transfer/particles_to_grid.h`, and two private files beside it, `particles_to_grid_cpu.cc` and
`particles_to_grid_gpu.cu`. Callers never see `Cuda`, `Gpu`, `Device` or `host`. Where it runs is
`enum class RunOn : std::uint8_t { kCpu, kGpu }`, chosen when the simulation is built.

**What is shared between CPU and GPU.** The formulas: weight curves, contact response, material response,
local reconstruction. They live in headers that compile for both, written in the C++20 subset with no host
library types. Loops, storage layout and scheduling are not shared.

**Header kinds.** `.h` shared or host; `.cuh` needs the CUDA compiler; `.cc` host source; `.cu` device source.
A shared header never includes `std::expected`, `std::string` or anything that allocates. If a `.cu` file would
pull such a header in, the header is split, not guarded with `#if`.

**Errors.** Inner code returns a small per-module enum plus a compact context record (which particle, which
face, what magnitude) and never allocates. The host boundary turns that into a message. When several items fail
on the GPU, the one reported is the first in a fixed order (by stable id), so the error is the same on every
run.

**State.** Every stored quantity has one owner, a time label and a version. On a GPU run the device buffers are
the authority; the host receives a compact status and explicit snapshots. A step works on a trial copy of the
fields it changes and commits by swapping, so a rejected step changes nothing: not time, not ids, not source
counters, not solid motion.

**Step.** One `TryStep(StepInputs)` for every scene. Which physics runs comes from the scene, checked once by a
capability check before any allocation. The grid layout is fixed during a physical step. `ResizeCells` runs
afterwards as its own transaction (D12).

**Determinism.** Particle-to-grid sums are grouped by destination face and ordered by stable particle id. No
floating-point atomics. Reductions use a fixed tree. The choice between the candidate GPU schemes the report
lists is made in the transfer module design.

**Numerical flags.** Device code keeps `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`, written down
as numerical policy. For doubles only `--fmad=false` changes results: an NVIDIA GPU always keeps subnormal doubles
and rounds double division and square root correctly, so the other three flags govern single precision only. Host
code keeps `-ffp-contract=off` (GCC, Clang) and `/fp:precise` (MSVC). No fast-math anywhere.

**Dependencies.** FlatBuffers, BLAKE3, CLI11, GoogleTest, managed by Conan as today. Abseil is dropped (D11).

## 8a. Reference scenes and sizing

Four scenes size the design and are the programme's end-to-end demonstrations. Their exact dimensions are set
in the module designs that first need them; what is fixed here is that the architecture must serve all four.

| Scene | What it stresses |
|---|---|
| Breaking wave with foam and spray | Largest liquid volume, mixed-size cells, wave paddle, open-sea side, surface and whitewater output |
| Syrup | Stiff viscosity coupled with pressure, free surface, slow settling |
| Fire | Gas: grid-carried temperature and fuel, the second pressure regime, volume output |
| Sand (last module) | Material response, floor friction, long settling |

Before a run allocates anything, the solver:

1. **Analyses the scene:** counts cells per size level, particles, ingredients, solids and requested outputs.
2. **Estimates memory** from those counts: committed state, trial copies of the fields a step changes, solver
   working buffers, output snapshots. The estimate is generated from the same allocation table the run uses, so it
   cannot drift from the code.
3. **Checks the fit** against the device's free memory with a stated headroom.
4. **Chooses the mode.** If the scene fits, the whole state lives on the device for the whole run. If it does
   not, the state lives in host memory and the solver sends it through the device in batches of cell blocks.

The chosen mode and the estimate are reported to the user before the run, and the run reports how the estimate
compared with what was used.

Consequence for every module from grid onwards: a stage is written to work on a range of cell blocks with its
border cells, never on "the whole grid" implicitly. The one-piece run is then the case where the range is
everything. The batched mode's own design (how the global pressure and viscosity solves run across batches,
and what it costs) belongs to row 9.

## 9. Module sequence

Each row becomes its own design, plan and set of commits. "Open names" are naming-review items that must be
settled with the owner in that module's design.

| # | Module | Ported from the proof of concept | Deliberately not ported | Open names |
|---|---|---|---|---|
| 0 | Foundation | `.clang-format`, Conan pins, CI lane choices, shared vector math | 37 KB `CMakeLists.txt`, eight CMake helper files, Abseil, milestone docs | `step/`, `solids/` folder names |
| 1 | Contact | Swept contact and friction math | Rounding enclosures, `NudgeOutward`, mirrored input types, the axis-rewriting trick | `SolidSide` values and whether a stable side id is separate (O2) |
| 2 | Grid | Staggered grid with cells of mixed sizes: sparse cell blocks, size levels, the size-step rule, face pieces, nearby-face weights | 5 KB per-particle weight records, `3D` types, the 4096-cell and 24576-face limits | `weighted_spread` (Verify), `owner` (O5) |
| 2b | Particles | Particle state with stable ids and per-ingredient amounts (so several liquids and mixtures are carried from the start); grouping by cell block; `AddParticles` from sources, `SplitParticle`, `MergeParticles` | Sort-bin vocabulary; split offsets applied outside complete support | `blob`, `ingredient` uses (Verify) |
| 3 | Transfer | Fitted and exact velocity change kept distinct; pre-force snapshot order; weights valid for incomplete and mixed-size neighbourhoods | Fixed-point deposits, 2D transfer, the `0.99 x blend` coefficient under an "energy" name, the inverse spread that assumes a complete neighbourhood | `PerParticleVelocity::state` (O3), `filled_face` (Verify) |
| 4 | Pressure | Equations over face pieces; single-source open area and solid flux; `FindLiquidSurface` (the physical surface, not the render mesh); surface tension as a pressure jump; pressure law; air pockets and their classes; open-sea side and outlet | One-block solve, silent mean subtraction | each `old_`/`new_` (Verify) |
| 5 | Viscosity | Full symmetric-strain operator coupled with pressure, with viscosity that varies between ingredients | One-thread solve | |
| 6 | Materials | Snow response and return; springy solids; how liquid, mixture and gas are described through the same interface | Force floor tied to minimum particle mass (replaced by a stated support rule); sand (moved to the last module) | settled: owner answer A4 |
| 7 | Solids | Analytic shapes; driven and free motion; the wave paddle and `DriveWaves`; cutting with side identity | Separate piston, box and planar coupling paths (one motion map and its transpose instead) | |
| 7b | Gas | The reduced low-Mach model from `origin/codex/m7-low-mach-gas` (substep, transport, thermodynamics, gas markers, Python reference) | Its uniform-domain-only path and its separate CLI and saved-state route | all gas words (not in the naming review) |
| 8 | Step | Trial and commit discipline; capacity restart distinct from numerical reject | 22 `AttemptWith...` methods, `bounded` flag, test hooks in the public interface | request fields (Verify) |
| 9 | Cell resize and batches | `ResizeCells` as a transaction after a committed step; conservative remap; region transfer between particle and grid-only regions; the batched run of section 8a | The scattered admission routes and their limits | `grid_velocity`, `liquid_amount` (Verify) |
| 10 | I/O and CLI | Saved state, playback cache, motion output, fingerprint and lease checks | Schema v1, raw buffer images as file format, 777-line command function | `run_fingerprint` (Verify), CLI command names (O5) |
| 11 | Surface, foam and spray | Surface reconstruction, tracker's fail-safe rejection, whitewater as a separate layer, volume output for liquid and gas | Open-ended local mesh repair | surface and foam words (O5) |
| 12 | Blender add-on | Native-reader playback, cache lease, relink preservation | `Job`/`FrameJob` split, `boundary_mode`, particle-rate source controls | |
| 13 | Sand | Sand response and return, floor friction, the sand reference scene, with the eight-step diagnosis of the settling creep as tests | | |

Sand is the last module by the owner's decision of 2026-11-14 ("important but not mandatory"): the rest of
the programme does not wait for it, and its known settling creep (32.83 to 29.11 degrees) is investigated
there. Gas and liquid run in one scene from the start (owner, same date).

Known open physics carried forward honestly, not tuned away: piston receiving volume, traction across cells of
different sizes. Each stays a visible failing or excluded capability until its module design decides.

Known defects the report found that the rebuild fixes by construction are listed in the module designs, with
the report section that describes each.

## 10. Foundation, in more detail

Foundation is the first project and has no physics. It delivers:

1. The orphan branch with this document and two files written from the naming review, which is itself not
   committed (D19):
   - `AGENTS.md`: the seven-sentence model, the naming rules, the comment rules with the file-header and
     docstring templates and one worked example in the new vocabulary, the target structure (folders, one
     operation with two private implementations, one error enum per module, the exactness requirement), the
     per-file checklist and the commit rule of section 6. It contains no finding about the old code, no hit
     count, no old name and no roll-out plan.
   - `docs/glossary.md`: the decided vocabulary as word, plain meaning, never say. Rows the review marks Verify
     or Open are left out until the module that settles them.
2. A slim CMake build: one `fluidity` library target with host and CUDA sources, strict warnings scoped to our
   own targets, C++23 host and C++20 device with a configure-time check that `std::expected` works.
3. A container definition that compiles the CUDA target on this ARM Mac. **This must be proven first.** If no
   CUDA toolkit image compiles here natively, the fallback is an emulated x86-64 container; if that is too slow
   to run per commit, the fallback is the CI job, and the commit rule's item 3 becomes "CI CUDA compile is
   green" with the owner told.
4. CI workflow files for the strict CPU build and tests on Linux, macOS and Windows, the CUDA compile on Linux
   and the format check. They are written and kept correct, but run only once the owner pushes (D17). Until
   then the local build, tests and container compile are the gate.
5. `base/`: vectors, small matrices, the error-and-result pattern, `RunOn`, with tests on both backends. One
   trivial device kernel and its test, so the owner's first hardware run exercises the whole chain before any
   physics exists.
6. One command the owner runs on Windows that builds the CUDA preset, runs the GPU-labelled tests and the
   sanitizer check, and prints a short pass or fail summary to paste back. It is written now and used at the
   end of the programme (D4).
7. A check script that applies the mechanical items of the `AGENTS.md` checklist (file header present with
   its labelled sections, banned words from the glossary's "never say" column, `3D` suffix, milestone codes,
   bare `bool` parameters, verbs and role suffixes outside the allowed lists). `AGENTS.md`, not the script, is
   the authority: items a script cannot judge (is the example useful, does the comment say why) are checked by
   reading, in the commit rule and again in the comment pass.
8. The method paper started again from an empty file (D20): title, the Z-up and unit conventions, notation,
   and a bibliography whose entries, DOIs and URLs are copied from the old paper's bibliography, never written
   from memory. Each module then writes its own section, in Z-up and the decided vocabulary, in the same
   commits as the code that cites it. The paper is a teaching document: Foundation sets one drawing style
   (CeTZ) and one pseudocode style, and every module section carries at least one drawn figure of its idea and
   the pseudocode of its main procedure, matching the numbered steps in the code.

## 11. Left to each module design

- The old-to-new vocabulary table for that module, built from reading the code.
- Which old files are read, which functions are ported and which are rewritten.
- The exact tests and tolerances.
- The device data layout and reduction scheme for that module.
- Method-paper labels to cite; missing labels are added to the paper in that module.

## 12. Open questions for the owner

None at programme level. Naming-review items marked Verify or Open are raised in the module design that meets
them (section 9, last column).
