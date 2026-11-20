# Rebuild Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `native/fluidity/grid/`, the mixed-size staggered grid every later module computes on: cell and
face identity, cell blocks and their tables built into buffers that are allocated once, the weight curve and the
nearby-face weights, ranges with their border, grid pieces for the batched run, the per-place value buffers, and
the shared cell size rule the step and cell resize read.

**Architecture:** Space is cut into cubic cells of up to 16 size levels (each half the previous), stored in
blocks of 4 x 4 x 4 cell places with 240 face places and 125 corner places each. `BuildGrid` fills a
`GridBuffers` (allocated once, at a block limit) on the processor the run uses: a run is all CPU or all GPU.
In a GPU run the cell list already lives on the GPU, every table is built there (sort, then mark, prefix sum
and scatter), and only a small status record comes back; no table is copied in either direction. The CPU builder
serves CPU runs and is the oracle: GPU tables are compared with it bit for bit. Both builders call the same
per-cell and per-face formulas in a private shared header. Everything a stage needs on either processor
(addressing, geometry, nearby faces and their weights) lives in shared headers that read the raw `GridTables`.
A test-only geometric checker that knows nothing of blocks backs every "random grids" test.

**Tech Stack:** As Foundation: C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2 (GoogleTest),
uv, clang-format and clang-tidy 23.1 (the versions `pyproject.toml` pins), Docker via colima for the CUDA
compile, Typst + CeTZ for the method paper.

**Spec:** `docs/superpowers/specs/modules/02-grid-design.md` (the grid design; in the planning checkout,
untracked). Its section 3 is the contract. Programme design: `docs/design/programme.md` on `rebuild` (sections
6, 7, 8, 8a). Cross-module names: `docs/superpowers/specs/modules/shared-interface.md`, `plan-decisions.md` and
`owner-answers.md` (planning checkout).

## Revision 2026-11-19 after the pre-flight scan

The scan is `.superpowers/sdd/2026-11-14-rebuild-02-grid/preflight.md` in the worktree. This revision rewrote
the whole plan as clean text: the dated "Amended" paragraphs are gone and each task states its contract once.
One line per finding: what changed, and where.

**Blocking findings**

- **B1** `GridFrame` is created by Task 3 in `grid_tables.h` (that type only). Task 5 consumes it. Task 7b adds
  `GridTables` (second revision, K2: the two enums are Task 3's).
- **B2** Task 8 keeps the checks only. The one-argument `ComputeGridCounts` is new Task 8b: it counts on the
  sorted ids (blocks by unique block id, face pieces by id lookups, 8 B per active or split cell). The buffers
  form moved to Task 9b, after the CPU build. Design 3.3 updated.
- **B3** `ComesBefore` is added to `tools/check_rules_allowed_names.txt` by Task 3. Renamed:
  `ComputeSplitCellCountToAdd` (Task 7a), `ComputeIsInBlockSet`, `ComputeIsBorderBlock` (Task 22a, design 5.5).
- **B4** Renamed: `FactorTwoNearSurface_IsSmallest`, the scene tests (their final names are in the second
  revision, K1), `w_depth`; the text says "the cells within 8 smallest widths of the
  surface" (Tasks 13a, 26a, 27; design 3.8, 5.9, 7).

**Rulings**

- **R1** `GridError` and `GridFailure` in `grid_error.h`; `ToMessage` and `ToName` in
  `grid_error_message.h/.cc`, as contact is built (Task 2; design 3.3). Task 2 creates no `staggered_grid.h`.
- **R2** Both sort keys use the offset `2^(18 + level)` before the shift, in unsigned arithmetic; range proof and
  the test `ToBlockSortKey, LowestBlockAtLevelZero_IsZero` in Task 7a.
- **R3** Task 3 produces `ComesBefore` for blocks and for cells; design 3.1 gained the second.
- **R4** `ComputeCellSizeFactorTable` is shared and `constexpr`: an exact power of two times one of three
  hexadecimal literals; no library call (Task 13b; design 3.8).
- **R5** One launch pattern everywhere. Face piece kinds are an ordinary grid-stride loop over face places; a
  thread adds up what its items return and issues one integer `atomicAdd` (Tasks 11b, 25; design 5.1, 5.8, 6).
- **R6** `grid_buffers_internal.h`, `ToGridBuffersArrays`, parameter `grid_count`, parameter `buffers`, test
  names with `Buffers` (Tasks 6a, 6b, 9a, 11b, 24; design 3.3, 3.4, 3.6).
- **R7** "entry" everywhere for an estimate line; tests `NoEntryIsACpuCopy`, `TwoGrids_DoublesEveryEntry`; the
  third test split in two (Tasks 6a, 6b).
- **R8** No `GTEST_SKIP`: `kRunMemoryTooSmall` is returned in debug and release alike (Tasks 6b, 21).
- **R9** Every code block was rewritten without dates, answer numbers or "amended"; Global Constraints
  "Comments" binds every task.
- **R10** Not taken as recommended; it is unsound against contact as built. Contact's builder calls
  `CheckCanRunOn(run_on)` first and returns `kRunOnCallFailed` before it touches the block, and its `no_gpu`
  test passes a CPU `RunMemory`. The four grid builders do the same, so `kRunOnCallFailed` stays in their
  `Fails:` lists and the `no_gpu` tests now reach the builder (Tasks 6b, 21, 22b).
- **R11** `GridFrame` is passed by value in every shared function (Tasks 7a, 10, 12, 13a; design 3.2).
  `GridTables`, `CellSizeSettings`, `CellWidthsByLevel`, `CellSizeFacts` and the lists stay by reference.
- **R12** No build of `main` and no worktree of it. The numbers come from `main`'s Python reference, read with
  `git show` and run outside both checkouts; they live in `grid_main_comparison_test_cases.h` (Task 26b).
- **R13** The grid's own GPU functions keep under the driver's default stack, say so in their `_gpu.h`, and a
  `gpu` test per GPU half pins it; the grid has no `kDriverStack` entry (Global Constraints "GPU stack"; Tasks
  11b, 23, 25). A module whose GPU function calls the weights declares its own entry.
- **R14** The CPU builder sorts with comparators made of `ToCellSortKey` and `ToBlockSortKey` (Task 9a).
- **R15** Contact's layout binds the grid for the steps files: `grid_build_steps.h`, `border_block_steps.h`,
  new `grid_piece_steps.h`; the replay tests live in `<name>_steps_test.cc`. The public type headers keep the
  design's names (no `_types.h`): other designs include them by name.
- **R16** One class template `GridValues<Value, places_per_block>` with four alias names, and `BufferValue`
  (Task 21; design 3.6).
- **R17** P7 ends with a fill: one thread per place from `n + split_cells` to `sort_count` writes `CellId{0}`.
  Tests: `ApplySortTailFillAt, UsedBuffers_HoldNoStaleCell` and the replay on used buffers (Task 11a).
- **R18** `FindCellAt` with `near` tries two hops and then falls back to the search; a coordinate outside the
  id range is `kNoIndex` before any cast (Task 13; design 3.2).
- **R19** The private header declares `GridBuffersArrays` and `GridBuilderBuffersArrays`, each with a const and
  a writable accessor (Task 6b).
- **R20** Task 13a is rewritten and split in three (13a, 13b, 13c).
- **R21** Three senses of "piece" are kept, never bare where two could be meant: "face piece", "grid piece",
  "memory piece" (Global Constraints "Words"; glossary lines in Tasks 3, 6b, 24). Owner question 1.
- **R22** The read set of each border width is defined in Task 22a. The four-widths oracle restates the block
  rule by coordinate arithmetic and says so; a geometric containment test stands beside it.

**Notes**

- **N1** Task 1 split (1a, 1b), Kind `numerics`, figure files prefixed `grid_`. **N2** Task 2 declares `error`
  and `magnitude`. **N3** `grid_ids_gpu_compile_test.cu` and its CMake line are in Task 3's files. **N4** Task 3
  updates the glossary lines `place` and `cell_eighth`. **N5** test renamed `BuiltForTheGpu_HasAnAddress`. **N6** the
  aliasing of out-of-range coordinates is stated in Task 3. **N7** Global Constraints "Random grids" fixes the
  generator's arguments. **N8** test support is in `namespace fluidity::grid`.
- **N9** Task 6 split (6a, 6b). **N10** stale text removed in the rewrite. **N11** `ComputeBigCellAtLevel`
  replaces `ComputeAncestor`; "child" and "ancestor" are gone; `ComputeFailureCells` has a signature; glossary
  lines assigned. **N12** `kTooManyCellBlocks` listed for `CheckSizeStepRule(inputs, buffers)` (Task 8; design
  3.3). **N13** `TestGrid` deletes move assignment. **N14** the "ask the owner" step is removed. **N15** the
  list of counting programs is in Task 9a and includes Task 11b's.
- **N16, N18** commit texts list the `gpu` tests by name, no counts; `kMaxThreadBlockCount`; Task 11a says how
  the block index is recovered. **N17** the mutated-list generator is in `grid_test_support` (Task 10).
  **N19** `math::Min`, `math::Max`. **N20** `NextToBigCell_IsOneLevelSmaller`; glossary lines for the
  refinement words (Task 13b). **N21** the 10,000-grid tests use the small generator arguments.
- **N22** removed from Task 20. **N23** Task 21 is Kind `mechanical`. **N24** Task 22a lists
  `grid_test_support`. **N25** the CPU form of border search now reads its marks array (Task 22b; second revision, Q3). **N26** Task 27 checks 3.8 and 5.9. **N27** `ReadMemoryAvailable(kGpu)->free_bytes`. **N28** design
  lines 323 and 981 reworded; Global Constraints "Comments" forbids copying design comments. **N29** Global
  Constraints "Build registration". **N30** design 6 and 7 corrected. **N31** the compile test follows
  `base_constants_gpu_compile_test.cu`.
- Left as they are: test names with `EveryThreadBlockAndThread` (contact as built has the same name); the word
  "rows" inside design sections no finding quotes (the plan never copies it).

### Second revision 2026-11-19, after the second scan

The scan is `preflight-2.md` beside the first. One line per finding.

- **K1** `tank` is banned too: `SlopedSurfaceInBox_PassesEveryCheck`, `SlopedSurfaceInBoxOnGpu_EqualsCpu`, the
  scene is "a domain box of liquid with a sloped surface" (Task 26a, 27; design 7). The whole banned list of
  the glossary was run over the plan and the design with the rules tool.
- **K2** `CellState` and `FacePieceKind` are produced by Task 3; Task 7b adds only `GridTables`.
- **K3** `CheckSiblingsAt` also walks the eighth-0 chain, so every split cell has its siblings checked once;
  test `MissingCellBesideSplitSiblings_NamesTheMissing`; a matching mutation (Tasks 7a, 10; design 5.1).
- **K4** The tail-fill test uses 14 lone level-0 cells (tail of 2) and asserts the stale ids were there and are
  gone (Task 11a).
- **Q1** `FindCellAt`'s guard is stated in cells of the frame's smallest level, with its proof; tests at 262144 m
  and 524288.25 m (Task 13; design 3.2, 5.2).
- **Q2** A place beside a cell is looked up by raw coordinate through `FindSortedCellAt`; no id is made outside
  the range. `ComputeIsCoordInRange` (Task 3); fixtures at both limits (Tasks 7a, 7b, 8b).
- **Q3** The design binds: the CPU keeps its `border marks` entry, and the CPU form reads it (Task 22b).
- **Q4** `main`'s Python reference, 100 positions as the design says (Task 26b; design 7 says "reference").
- **Q5** `<proc-count-grid>` is added by Task 1b; Task 8b's step comments cite it.
- **Q6** The six questions of the first report take their recommended answers, except the one Q3 reverses.
- **M1** `BuildGridFromBuffers` (Task 9a) is what Task 24 consumes. **M2** `BuildBorderSearchMemoryEstimate` is
  declared in `grid_memory_estimate.h`. **M3** superseded by the third revision (27 at every position). **M4** `ApplyFixedShuffle` (Task 5). **M5**
  `SplitCellsOverflowThePlaces_IsTooManyCellBlocks` and its list in Tasks 11a, 11b. **M6** `ComputeSortCount` in
  64 bits, with its test. **M7**, **M9** Global Constraints "Replay tests". **M8** `BuildTestGrid` takes a CPU
  span and copies it for a GPU grid. **M10** Task 7 split into 7a and 7b; `ComputeCellStateBeside` gives the far
  side of a face place. **M11** the scene of Task 26a is defined. **M12** the tolerance of Task 26b.
  **M13** the header names tests, not tasks. **M14** `BuildSortedCellList` replaces `CheckCellListOnBuffers`.
- Found while re-reading the glossary's banned list (K1): `velocity_gradient_size` holds the banned "velocity
  gradient"; it is `velocity_change_size` in Task 13a and design 3.8, 5.9, 7.
- **M15**, **M16**, **M18**, **M19** design 7 test names, `storage`, "band", `const GridFrame&`, the border
  wording and the 8 steps are brought in line with the plan. **M17** recorded in Task 27. **M20** GPU-half
  names start with an allowed verb. **M21** `BuildSizeStepClosure`. **M22** plain glossary meanings.

### Third revision 2026-11-19: the owner's rule for faces at exactly 1.5 widths

Owner decision: a position exactly on a cell side or at a cell centre still has 27 nearby faces per axis and
`is_complete` true; the faces of weight 0 are kept.

- **The rule.** A candidate's offset `d = x_f - x_p` satisfies `-1.5 h < d <= 1.5 h` on each axis (open on the
  low end, closed on the high end), in place of the strict `|d| < 1.5 h`. The closed end must be the high one:
  a cell owns its low side, so a position on a low side needs the faces 1.5 h above it, and those 1.5 h below
  belong to a cell that does not touch. Checked by computation on a uniform grid at 512 lattice positions (27
  each) and on the all-split worst case at 4,096 lattice and 3,000 random positions (largest count still 76).
- Changed: Task 1a Step 1 (the paper's sentence), Task 16 (the face search keeps its own strict test, and says
  why), Task 17 (the rule, four count tests; its float predicate is replaced in the fourth revision), Task 18 (weights may be 0; three tests
  of the faces of weight 0; the note about lattice positions removed), Task 19 (the same predicate); design
  3.5, 5.4 and 7. Unchanged: the bounds 76, 56, 112 and 270.

### Fourth revision 2026-11-19: the candidate window is chosen in whole quarter widths

The test of the third revision, written on float offsets, gives 2 or 4 planes on an axis at positions within a
rounding step of a cell side or centre (found by program: about 2 % of such positions, also in frames whose
origin and width are powers of two, because `FindCellAt`'s floor and the offsets can disagree). Ruling: the
window is chosen by integers.

- **The rule.** Per axis, `q = floor(4 * ((x_p - origin) / h - c))` clamped to 0..3, with `c` and `h` the
  coordinate and width of the cell `FindCellAt` returned; a face is a candidate when its centre's coordinate
  `n`, in quarter widths from that cell's low corner (an integer, from the ids), has `q - 5 <= n <= q + 6` on
  each axis. It is the half-open rule `-1.5 h < d <= 1.5 h` on the lattice every face centre lies on. The
  float offset is used only for the weight.
- Changed: Task 5 (`NearbyFaces` takes the cell and the three `q`), Task 17 (rule, two functions, tests in a
  non-dyadic frame and one rounding step from sides and centres, the bound 76 checked over the 64 windows),
  Task 18 (one test), Task 19 (the same `q`); design 3.5, 5.4, 7. The outcome of the third revision stands: 27
  faces and `is_complete` at every position among cells of one size.
- Not proven: that this window gives the same candidates as the exact half-open rule on mixed sizes in
  general. It was checked by program on three neighbourhoods. `FindNearbyFaces,
  RandomGrids_MatchGeometricChecker` (Task 17) is the test that settles it on random four-level grids.

### Fifth revision 2026-11-19: 216 nearby faces per axis

An independent check refuted the bound of 76: a valid grid where two cell sizes meet reaches 124 candidates on
one axis. Owner decision: keep every candidate and size the list for the proven ceiling.

- `kMaxNearbyFaces = 216`. 124 is the largest case known and the exact maximum where two cell sizes meet
  (searched over every arrangement of the 26 neighbours); 212 is proven for any valid grid (design 5.4);
  `kTooManyNearbyFaces` stays as a guard no valid grid reaches. `NearbyFaceMask` is four 64-bit words.
- Sizes: `NearbyFaceList` 888 B, `NearbyFaceMask` 32 B, `NearbyFaceWeights` 1,152 B for one axis; 3,456 B for
  three axes held at once. Every GPU function that calls the weights is above the driver's default stack.
- Stack entry: the grid's own GPU functions (build, border search, grid pieces) do not call the weights and
  stay under 1,024 B, so the grid still has no `kDriverStack` entry. The modules whose GPU functions call the
  weights own theirs (`changes-to-apply.md`). Task 20's test runner measures and reports the bytes.
- Changed: Global Constraints (the constant, "GPU stack"), Task 17 (bound, sizes, the 124 fixture, tests
  renamed), Task 18 (the 124 fixture's weights, size), Task 20 (stack measurement); design 3.3, 3.5, 5.4, 6, 7,
  11. Where the third and fourth entries above say "76", read this entry.

### Sixth revision 2026-11-19: three alignments with the reviewed paper

- **No weight at all.** As `<proc-nearby-face-weights>` now has it, the case is decided at step 5: if `W <= 0`
  the fit is `kNone`, every weight, change, mean and spread is zero, and steps 6 to 9 do not run (Task 18;
  design 5.4 steps 5 and 9, and its properties "for every grid and every mask whose fit is not `kNone`").
- **The guard.** `kTooManyNearbyFaces` gets a test that reaches it (Task 17). Hand-built tables cannot be relied
  on for that: the walk reads only the face pieces of at most 56 touching cells, and no arrangement of table
  entries is sure to push a correct walk past 216. So the walk has a private form that takes the room, and the
  test runs it on the valid 124 case with room for 100.
- **Symbols.** In mathematics the position's quarter is `n_p` and a face centre's quarter is `n_f`, as in the
  paper (Tasks 5, 17, 19; design 5.4). Code identifiers are kept: `ComputeCandidateWindowQuarter`,
  `ComputeIsInCandidateWindow`, `window_quarter` (`n_p`), `face_quarter` (`n_f`). Where the fourth entry above
  writes `q` and `n`, read `n_p` and `n_f`.

### Seventh revision 2026-11-19: three findings of the review of the paper's parallel build

- **One order of input checks on both processors.** Frame, version, list shape and list length (more than `64 *
  max_cell_blocks` cells is `kTooManyCellBlocks`, `cell` zero) are checked on the CPU, in that order, before any
  id is looked at and before any keyed check; they never take part in the key minimum. The CPU path checked
  ids first. Tasks 8, 9a, 10; design 5.1 item 8. New tests with a list that is too long and holds a bad id.
- **A prefix sum reads only values written at this launch.** A thread whose earlier step failed does nothing,
  except that a value a prefix sum will read is written as 0. Otherwise the sum adds stale words, passes `2^32
  - 1` (from about 93,000 cells) and the true error is reported as `kBadInput`. Global Constraints "One launch
  pattern"; Tasks 11a (P7 a, P8 marks), 23 (marks); new replay tests that start from working arrays full of
  `0xFFFFFFFF`. Sorts may read stale keys: `BuildStableOrder` fails on no value. On a valid list the sums cannot
  pass `2^32 - 1`: the split-cell counts add up to at most `n / 7`, and marks are 0 or 1 over fewer than `2^32`
  places.
- **The coordinate limit.** At the low limit nothing lies beyond a cell. At the high limit a bigger cell may,
  found at its own size level, where its coordinate is in range. Tasks 1b, 7a, 8b.
- Small: `n` is `active_count`; only an id with bit 63 clear has the largest sort key; the `CellId{0}` entries
  of the tail write nothing at P10 (b); the status step is part of the check (Tasks 10, 11b).

### Eighth revision 2026-11-20: a dedicated error for an impossible internal state

Owner decision: a module reports an impossible internal state through a dedicated error value, checked in every
build, never as bad input and never by a debug-only assertion.

- `GridError::kInternalFault`, just before `kOutOfMemory`, which stays last: thirteen values, `kOutOfMemory` is
  12, stable name `internal_fault`. Its message says the fault is in FluidiTy, not in the scene, and asks the
  user to report it with the scene (Task 2; design 3.3).
- The one `BufferError` mapping sends `kSizeMismatch` to `kInternalFault` (it was `kBadInput`): the grid checks
  every caller's size itself before it calls `base/`, so a size error from `base/` is the grid's own fault.
- "Cannot happen" paths that now return `kInternalFault`: a builder whose arrays and estimate entries do not
  match (Tasks 6b, 21, 22b); a size error from a sort, a prefix sum or a copy of `base/` inside a grid
  operation (every task, through the mapping); a status record that does not fit its own failure key (Task
  10); a block that a build cannot find among the block ids it just wrote, or a count read back from the GPU
  above the room (Tasks 9a, 11b, 24, 25).
- Left as they are: `kBadInput` for what a caller can pass wrongly (frame, version, list shape, a bad id, a
  room of 0, builder buffers too small, a span of the wrong size, a border list that is not ascending);
  `kTooManyNearbyFaces` (a question for the owner, not changed here); the `Requires:` preconditions on
  processors, which are a caller's contract, not an internal state.

### Ninth revision 2026-11-20: too many nearby faces is the internal fault

Owner decision (A107): `kTooManyNearbyFaces` is folded into `kInternalFault`. `GridError` has twelve values
again; `kInternalFault` takes the place `kTooManyNearbyFaces` had (value 10) and `kOutOfMemory` is 11 and last.
`FindNearbyFaces` reports `kInternalFault` when the candidates do not fit the list. Changed: Task 2 (counts and
tests), Task 17 (the guard and its test, now `MoreFacesThanRoom_IsInternalFault`), Task 27; design 3.3, 3.5 and
5.4. Where the fifth, sixth and eighth entries above name `kTooManyNearbyFaces` or count thirteen values, read
this entry.

**Proposed model per task**

| Task | Model | Reason |
|---|---|---|
| 1a, 1b | opus | The paper fixes the step numbers and rules every later task cites. |
| 2 | sonnet | An enum, two text functions, three tests. |
| 3 | opus | Bit packing, floor on negatives, the fixed order; every table depends on it. |
| 4 | sonnet | One ported formula and four exact tests. |
| 5 | opus | The oracle of every random test; a quiet error here hides real ones. |
| 6a | sonnet | Entries and arithmetic are worked out in the task. |
| 6b | opus | Ownership, lending and exact accounting of memory pieces. |
| 7a | opus | The rules both builders share; the sort keys and the sibling walk must be right at the limits. |
| 7b | opus | The face rule across block sides and sizes. |
| 8 | opus | The first-failure order must match the GPU's key. |
| 8b | opus | A second statement of the face rule, held only by its tests. |
| 9a | opus | The CPU oracle. |
| 9b | sonnet | The build run for its counts; the tests are given. |
| 9c | sonnet | A copy between buffers with three tests. |
| 10 | opus | Device code checked only by replay. |
| 11a | opus | Mark, sum and scatter with stale-data traps. |
| 11b | opus | Device code; the status record and the room failures. |
| 12, 13 | opus | Geometry across size levels; rounding at cell sides. |
| 13a | opus | Numerics with infinities. |
| 13b | sonnet | Integer lookups and a table with exact expected values. |
| 13c | opus | A search over probes with a tie rule, and the GPU runner. |
| 14, 15, 16, 17 | opus | Table walks across sizes with exact bounds (56, 216, 112). |
| 18, 19 | opus | Ported numerics with bit identity between two forms. |
| 20 | sonnet | A test runner copied from `base/shared_math_test_gpu`. |
| 21 | sonnet | One template over `Buffer` with stated numbers. |
| 22a | opus | The border rule is Review Focus 2. |
| 22b | opus | The stacked form and its exact reservation. |
| 23, 25 | opus | Device code. |
| 24 | opus | Grid, border and buffers meet. |
| 26a | opus | Every grid part in one scene. |
| 26b | sonnet | Produce numbers from `main`'s reference and compare. |
| 27 | opus for the comment pass; sonnet for `gpu-runs.md` and the hand-over | The comment pass is a review at the full bar. |
| Plan revisions | fable | Rulings, redesign, rewrites of tasks. |

## Global Constraints

Copied from the Foundation plan:

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Old code is read with `git show main:<path>` from the worktree, and in no other way. Nothing is built, checked
  out or changed in the `main` branch's checkout, and no worktree of `main` is made.
- Commit rule (programme design section 6): one thing per commit; strict CPU build and all tests pass; CUDA
  target compiles in the container; clang-format 23.1 clean; clang-tidy 23.1 clean on host sources; every
  touched file passes the `AGENTS.md` checklist; new public names listed in the commit message with a one-line
  reason; tests arrive with the code. A commit passes every part of the gate that exists at that commit.
- **Commit step.** Every task ends the same way: run the full gate, `uv run python tools/check_commit.py`
  (expected: seven `PASS` lines: format, rules, cpu debug, cpu release, tidy, paper, cuda); commit with the
  subject the task gives; the message carries no trailer of any kind; then write the commit's local session note
  with `uv run python tools/add_session_note.py` (its two arguments come with the task brief). Notes are never
  pushed. A task's last step says only "Commit" and its subject.
- History on `rebuild` may be rewritten. A defect found later is fixed in the commit that introduced it.
- Device code is "compiled, not run" until `docs/gpu-runs.md` records the owner's run. Never claim otherwise.
- Host C++23, exceptions and RTTI off in the library; device and shared headers C++20 subset; a shared header
  never includes `<expected>`, `<string>` or anything that allocates. No Abseil.
- Device numerical flags: `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. Host: `-ffp-contract=off`,
  `/fp:precise`. No fast-math.
- Z-up, right-handed. Gravity points along `{0, 0, -1}`.
- Naming: no `3D` suffix, no `Cuda`/`Gpu`/`Device`/`host` in public names, no bare `bool` parameters, no
  abbreviations (MAC, PIC, APIC, P2G, G2P, EOS), no milestone codes. A free function starts with a verb of
  `AGENTS.md` "Allowed verbs"; any other name must be in `tools/check_rules_allowed_names.txt` with its reason
  (this plan adds one: `ComesBefore`, Task 3). Functions of a private `_gpu.h` start with an allowed verb too and end in `OnGpu`.
- 120 columns, two-space indent, `//` comments, braces on every control-flow body.
- Tests are named `TEST(FunctionOrType, Situation_Outcome)`, one behaviour each, in `*_test.cc` beside the code.
- Deletion is the default: add a dependency, file or tool only when a task here names it.

Structure, as Foundation and contact are built (`AGENTS.md` "Structure"):

- **Two halves.** The public function lives in its `.cc` file: checks, `RunOn` dispatch, the CPU loop, and the
  GPU sequence under `#if defined(FLUIDITY_HAS_GPU)`. Each GPU half `<name>_gpu.cu` is declared in the private
  `<name>_gpu.h` (plain pointers and counts, no CUDA header, no CPU-only header) and returns `RunOnCallStatus`.
  The CPU half maps it with `ToBufferError` and then into `GridFailure` through the module's one mapping (below).
  A `.cu` file launches, waits and holds temporary GPU memory only through `base/gpu_launch.cuh`
  (`ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`, `GpuArray`).
- **One launch pattern.** Every GPU function is a thin grid-stride loop (`ComputeGridStride`) over the items of
  `ComputeLaunchSize(item_count)`: thread blocks of `kThreadBlockSize` (256), at most `kMaxThreadBlockCount`
  (65,535). Its body is one `FLUIDITY_CPU_AND_GPU` per-item function of a shared steps header
  (`grid_build_steps.h`, `border_block_steps.h`, `grid_piece_steps.h`). A thread may issue an integer
  `atomicMin` of the key an item returns and an integer `atomicAdd` of the count an item returns, per item or
  summed over its items first. There is no other launch form. A thread that finds an earlier step failed does
  nothing, with one exception: a value that a prefix sum will read is written as 0, so no prefix sum ever adds
  a value of an earlier launch. The GPU's only atomics are these two, and their results do not
  depend on thread timing.
- **Replay tests.** For each operation a `cpu` test in `<name>_steps_test.cc` runs the per-item functions for
  every thread block and thread of the launch, in launch order and in reverse, atomics replaced by a plain
  minimum and addition, `BuildStableOrder` and `ComputeExclusivePrefixSum` on the CPU between the steps, and
  compares with the CPU builder. CPU buffers hold only what the CPU builder needs, so the replay builds the
  GPU's working arrays (sort keys, the two orders, marks, an `OrderBuffers`, a status record) in the test, on
  the CPU, and takes `cell ids` and the tables from CPU buffers. It is the main evidence until the hardware
  run. A no-allocation test builds its reference grids before the counted part begins.
- **GPU stack.** A GPU function that keeps more local bytes than the driver's default stack (1,024 B per thread)
  makes the driver hold a stack for every resident thread outside the run's memory; contact declares that as a
  `MemoryWhere::kDriverStack` entry (`BuildContactStackEstimate`, `contact/move_particles.h`). The grid's own
  GPU functions (build, border search, grid pieces) keep no local array: each `_gpu.h` file header says "Stack:
  every function here keeps under the driver's default of 1,024 B per thread, so the grid has no entry held by
  the driver", and a `gpu` test per GPU half reads the largest local size (`cudaFuncGetAttributes`,
  `localSizeBytes`) and compares it with 1,024. If that test fails on hardware, the remedy is an entry as
  contact's, not a larger limit set in silence. The shared functions `FindNearbyFaces` and
  `ComputeNearbyFaceWeights` return 888 B and 1,152 B by value (one axis; 3,456 B for three axes held at once),
  above the default: their header says that a module whose GPU function calls them owns the stack entry for
  it, as contact owns its own. The grid's test runner of Task 20 measures the bytes.
- **Working buffers.** A function that needs working memory has one form, which takes its buffers. The grid's
  build working memory is `GridBuilderBuffers` (Task 6b), which holds the `OrderBuffers` that every
  `BuildStableOrder` and `ComputeExclusivePrefixSum` of a build takes; border search has `BorderSearchBuffers`
  (Task 22b). Two buffers on different processors break a documented `Requires:` precondition, checked by
  `assert` in debug builds; it is never an error value.
- **Errors.** A failed check writes nothing. After `kRunOnCallFailed` the outputs are unspecified and the run
  stops. `kOutOfMemory` comes only from the four builders. A memory piece that does not fit is an error in every
  build, debug and release alike (`BufferError::kRunMemoryTooSmall`); no test of it is skipped in a debug build.
- **Exactness.** Discrete outputs (every table, count and failure) are equal on CPU and GPU. Floating-point
  results agree within the CPU-versus-GPU tolerance below unless a header promises the same bits.
- **Tests.** Tests link `fluidity_base_test_support` and `fluidity_grid_test_support` (Task 5) through `LINKS`.
  Every `gpu` test passes its main outputs to `WriteResultDigest`. Labels: `cpu` everywhere, `no_gpu` only for
  behaviour on a machine without a GPU, `gpu` only for tests that need hardware. GPU free memory is read as
  `ReadMemoryAvailable(RunOn::kGpu)->free_bytes`.
- **Build registration.** Every target is made through the helpers of `cmake/fluidity_targets.cmake` and every
  source is listed in `native/fluidity/grid/CMakeLists.txt`, as `native/fluidity/contact/CMakeLists.txt` does:
  every `.h` in `fluidity_add_headers` (the `cuda` step fails a tracked header that is not compiled alone,
  unless `test` is a word of its name), every `.cu` in `fluidity_add_gpu_sources` or a test's `SOURCES`. A
  CPU-only header includes `base/cpu_only.h` first. "Register in the build" in a task means this.
- **Comments.** Code, comments and test names cite the method paper's labels (`Read:` lines, `// Step N:`).
  They never cite this plan, a task number, a plan decision, an owner answer, a review, a date, or the words
  "amended", "changed", "new" or "reconcile". A decision with no paper section gets one sentence of reason in
  place. The design's comments are not copied: its declarations are the contract, and each comment is written
  for the reader of the code.
- **Words.** The GPU sense of "block" is always "thread block"; "cell block" is the grid's own word. Write "GPU
  function", never "kernel" (that word is kept out of the code). "piece" has three senses and is never bare
  where two could be meant: a "face piece", a "grid piece" (`GridPiece`), and a "memory piece" (the bytes of
  one buffer inside a `RunMemory`). An estimate line is an "entry"; "row" is a matrix row only. Banned in every
  name, message and comment, also inflected: "refuse", "band", "tank", "tile", "slab", "family", "margin", "nudge",
  "signed distance", a `View` suffix, "system" for a set of equations.

Grid's own:

- Namespace `fluidity::grid`, folder `native/fluidity/grid/`. Shared (S) headers: `grid_ids.h`, `grid_error.h`,
  `grid_tables.h`, `weight_curve.h`, `nearby_face_weights.h`, `cell_size_rule.h`, and the private shared
  `grid_build_status.h`, `grid_build_steps.h`, `border_block_steps.h`, `grid_piece_steps.h`. Every function in
  them is `constexpr` or `inline`, `FLUIDITY_CPU_AND_GPU`, `noexcept`, allocates nothing, uses `math::` from
  `base/scalar_math.h` (never `<cmath>`), takes `Vec3`, `Mat3` and `GridFrame` by value, and sums in list order
  with plain doubles.
- Constants, exactly: `kMaxSizeLevels = 16`, `kBlockWidthInCells = 4`, `kCellPlacesPerBlock = 64`,
  `kFacePlacesPerBlock = 240`, `kCornerPlacesPerBlock = 125`, `kMaxCoord = (1 << 18) - 1`,
  `kNoIndex = 0xFFFFFFFF`, `kWeightCurveReach = 1.5`, `kMaxNearbyFaces = 216`, `kMinWeightSum = 0.1`,
  `kMinPivotShare = 1e-12`, `kMaxCellsNearFace = 112`, `kMaxFaceSearchRadius = 4.0`,
  `kMaxCellBlocksLimit = (1u << 26) - 1`, `GridSettings::max_cell_blocks = 0`. No compile-time room is ever an
  allocation size: `BuildGridBuffers` and `BuildGridBuilderBuffers` with 0, or above `kMaxCellBlocksLimit`, are
  `kBadInput`. With the limit every cell-place count (`64 * max_cell_blocks`) fits the `std::uint32_t` orders
  and prefix sums of `base/`.
- **Builder buffers are lent.** The `grid: builder buffers: ...` entries live in a move-only
  `GridBuilderBuffers` that the caller owns and lends to every `GridBuffers` built with
  `BuildGridBuffers(settings, builder_buffers, run_memory)`. No operation on a built grid reads them, so two
  buffers that hold two live grids may share one, used by one build or check at a time.
- **One reserved block per run.** `BuildGridBuilderBuffers`, `BuildGridBuffers`, `BuildBorderSearchBuffers` and
  every `Build...Values` form take `RunMemory& run_memory` last and build every array as one memory piece of it
  with `BuildBuffer(run_on, size, run_memory)` (the GPU `OrderBuffers` with `BuildOrderBuffers(RunOn::kGpu,
  max_count, run_memory)`). Each calls `CheckCanRunOn(run_on)` first and returns `kRunOnCallFailed` before it
  takes any piece, as `BuildContactBuffers` does. Requires: `run_on == run_memory.run_on()` when `run_on` can
  run. The caller passes the `RunMemory` of the processor the arrays live on. A memory piece is never freed
  alone: a failed builder leaves the pieces it took in the block. Each array is one piece and one estimate
  entry. An array whose size is a per-item part plus a fixed part is one entry with `counted_per =
  kGivenCount`, `bytes_each = 1` and `given_count` = its exact byte count at `max_cell_blocks`, so the
  reservation equals the bytes handed out for any room. Tests build their own `RunMemory` of exactly
  `ComputeRunMemoryBytes` of the estimate.
- **Span forms.** `ComputeGridCounts(inputs)` and `CheckSizeStepRule(frame, span)` stay beside their buffers
  forms for a stated reason: the first counts a list beyond any room (the planner counts before any buffers
  exist), the second checks CPU lists while cell resize splits, where no grid buffers exist. Both are CPU only
  and allocate one CPU copy of the ids (8 B per active cell; the count form 8 B per active or split cell).
- **No allocation**, on either processor, in `BuildGrid`, the buffers forms of `ComputeGridCounts` and
  `CheckSizeStepRule`, both `BuildGridPiece` forms, `TakeBuffers`, `ReadGrid`, `ApplyGridVersion`, and the span
  and buffer forms of `FindBorderBlocks` that take an output.
- **One processor per run.** Every input list, output list and working buffer of a grid operation lives on the
  grid's (or the buffers') `run_on()`: a `Requires:` precondition, checked by `assert` in debug builds. A grid
  operation on the GPU returns to the CPU only its `GridBuildStatus` record and, for border search, a count. A
  GPU grid keeps no CPU copy of its tables: `tables(RunOn::kCpu)` on it returns version 0 and null arrays, and
  `ReadGrid` (Task 9c) is the one explicit copy, for tests and tools.
- **The CPU is the oracle.** Tables, counts and failures built on the GPU equal the CPU's byte for byte.
- The grid never looks at physics. Which faces a particle may use is the caller's kept mask. The grid owns no
  physical field and has no identifier containing `owner`.
- **Fixed order** everywhere: blocks and cells by low corner in level-15 cell units by `z`, then `y`, then
  `x`, then bigger level first. A reported failure is the first in that order: the smallest `(step, position)`
  failure key of Task 7a, on both processors.
- **Errors of this module.** Every `std::expected` has `GridFailure` as its error. The one `BufferError`
  mapping: `kOutOfMemory` and `kRunMemoryTooSmall` become `GridError::kOutOfMemory` with
  `GridFailure::magnitude` = the bytes asked; `kRunOnCallFailed` stays `kRunOnCallFailed`; `kSizeMismatch`
  becomes `kInternalFault`.
- **Impossible states.** `GridError::kInternalFault` is returned, in debug and release alike, when the grid
  finds a state that no input can produce: a size error from a `base/` call inside a grid operation (the grid
  checks every caller's sizes itself first, and returns `kBadInput` for those), arrays that do not match the
  estimate's entries, a table or status record that contradicts what the same operation just wrote. It is never
  `kBadInput`, and never an `assert` alone. `cell` and `other_cell` are zero. A task that adds such a check says
  so in its header comment under `Fails:`.
- **Tolerances**, exactly (design section 7): discrete results (ids, counts, kinds, indexes, areas that are
  powers of two) are compared exactly. Weight curve: `|sum - 1| <= 4e-16`, slope sum `<= 4e-16`. Complete case:
  `W = 1`, `mean_offset = 0`, `weighted_spread = h^2/4 I` within `1e-14 h^2`. Any kept set: `|sum weight - 1| <=
  1e-14`, `|sum weight_change| <= 1e-13 / h`. Linear field: value within `1e-13 (|a| + |b| h)`, fitted matrix
  row within `1e-11 |b|`. Finite difference: step `1e-6 h`, within `1e-7 / h`. Comparison with `main`: `1e-13`
  relative. CPU versus GPU: `|difference| <= 1e-15 + 1e-13 |value|` (offsets: times `h`). Repeat runs:
  identical bytes.
- **Random grids.** Random tests use `std::mt19937_64` with the seed written in the test and map its raw 64-bit
  output with the helpers of Task 5 (never `std::uniform_*_distribution`, whose output differs between
  standard libraries). "The 200 random grids" are `BuildRandomGridCells(seed, 4, 3)` for seeds 1..200 (4 levels,
  a box of 3 x 3 x 3 level-0 cells: at most 252 blocks, so they fit buffers for 4,096 blocks), each list
  shuffled with `ApplyFixedShuffle(cells, seed + 1000)` (Task 5; never `std::shuffle`, whose order differs
  between standard libraries). "The 100 random grids" are seeds 1..100 of the same. "Small random grids"
  are `BuildRandomGridCells(seed, 3, 2)`; every test of 10,000 grids uses them, so it stays fast in a debug
  build.
- Comparisons with the proof of concept turn Y-up into Z-up with `new = {old.x, -old.z, old.y}`: old cell `z = k`
  becomes `y = -k - 1`; old Z-faces become Y-faces with `cell_low`/`cell_high` exchanged.

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`: Foundation (`base/`, with
`scalar_math.h`, `buffer.h`, `run_memory.h`, `memory_count.h`, `stable_order.h`, `ordered_sums.h`,
`run_on_call.h`, `gpu_launch.cuh`, `test_support.h`), the method paper with `figure_style.typ` and
`pseudocode.typ`, and the contact module. The grid includes only `base/`. Read `AGENTS.md` "Structure" and
`native/fluidity/contact/CMakeLists.txt` before the first task. Prove the start state:

```sh
uv run python tools/check_commit.py
```

Expected: seven `PASS` lines (format, rules, cpu debug, cpu release, tidy, paper, cuda).

## Review Focus

1. **A position one rounding step from a cell side at a size change.** `FindCellAt` must give the same cell from
   the binary search and from `near`, and the containing cell must contain the position by the one floor
   formula. Pinned: `FindCellAt, NearForm_EqualsSearchOnRandomPositions` and `..., OnHighSide_BelongsToNext`
   (Task 13).
2. **A face piece stored one block beyond a touching cell** (plane 4 moved to plane 0 of the high block). A
   border that only lists blocks of touching cells misses it. Pinned:
   `FindBorderBlocks, PlaneFourPieceBeyondTouchingCell_IsInBorder` (Task 22a).
3. **A failed build into used buffers**, on either processor. The buffers must stay usable and the next build
   must not read stale entries (on the GPU: no later step may run on a failed step's data, and the sort must not
   see cells of the build before). Pinned: `BuildGrid, AboveBuffersRoom_IsTooManyCellBlocks` and
   `BuildGrid, IntoBuffers_AllocatesNothing` (Task 9a), `ApplySortTailFillAt, UsedBuffers_HoldNoStaleCell`
   (Task 11a), `BuildGrid, OnGpuFailedBuildThenRebuild_EqualsFreshBuffers` (Task 11b).
4. **A grid piece whose border block's high neighbour exists in the whole grid but was not loaded.** Plane 4
   must stay `kNotAPiece` and the other side become `kOutsidePiece`, or a face piece appears that the whole grid
   lacks. Pinned: `BuildGridPiece, KeepsWholeGridOrder` (Task 24) and `BuildGridPiece, OnGpu_EqualsCpuPiece`
   (Task 25).
5. **The face-seen weight drifting one bit from the particle-seen weight**, which would break transfer's
   gather and scatter symmetry. Pinned: `ComputeFaceWeight, SeenFromFace_IsBitIdentical` (Task 19).
6. **The GPU reporting a different first failure than the CPU** when a list breaks several rules at once. The
   rule is "earliest step, then smallest position" on both. Pinned:
   `CheckSizeStepRule, OnGpuSeveralFailures_SameFirstFailureAsCpu` (Task 10).
7. **A wanted width exactly equal to a level's width, or a depth term computed as infinity times a width.** The
   level must come from comparison with `CellWidthsByLevel` (never `log2`), and infinite facts must give
   `w_max`, never NaN, on both processors. Pinned: `ComputeWantedSizeLevel, WidthEqualToALevel_GivesThatLevel`
   and `ComputeWantedCellWidth, NothingWithinReach_IsBiggestWidth` (Task 13a),
   `ComputeWantedCellWidth, OnGpu_SameBytesAsCpu` (Task 13c).
8. **A block or a cell at the coordinate limit.** A sort key that goes negative there sorts the block last on
   the GPU only. Pinned: `ToBlockSortKey, LowestBlockAtLevelZero_IsZero` (Task 7a).

---

### Task 1a: Method paper: the grid, its faces and its weights

**Kind:** numerics

Load the `typesetting-with-typst`, `drawing-with-cetz` and `typesetting-documents` skills first.

**Files:**
- Modify: `docs/method-paper/fluidity-method.typ`, `docs/method-paper/figure_style.typ`,
  `docs/method-paper/references.typ` (nothing removed; Harlow and Welch is not added)
- Create: `docs/method-paper/figures/grid_staggered.typ`, `figures/grid_face_pieces.typ`,
  `figures/grid_nearby_faces.typ`

**Interfaces:**
- Produces labels: `<sec-grid>`, `<sec-cell-sizes>`, `<sec-face-pieces>`, `<sec-cell-blocks>`,
  `<sec-nearby-faces>`, `<sec-grid-memory>`, `<eq:weight-curve>`, `<eq:scaled-weights>`,
  `<eq:weighted-spread>`, `<fig-staggered-grid>`, `<fig-face-pieces>`, `<fig-nearby-faces>`,
  `<proc-build-grid>`, `<proc-nearby-face-weights>`.
- Produces `cell-grid(origin, cells, ...)` in `figure_style.typ`: draws a 2D slice of square cells of mixed
  sizes from a list of `(x, y, size_level)`, with optional face arrows and centre dots. Every later grid figure
  uses it.

- [ ] **Step 1: Text.** `= The grid <sec-grid>` with the outline of design section 8, parts 1 to 7: 1 where
  values live (cell centres, faces, `+axis` orientation, `cell_low`/`cell_high`); 2 `<sec-cell-sizes>` size
  levels (0 is biggest), sibling cells exist together, the size-step rule and why one level; 3
  `<sec-face-pieces>` one identity, one area, stored with the smaller cell, plane 4 versus plane 0; 4
  `<sec-cell-blocks>` blocks, the fixed order, id versus index, grid versions, buffers built once, ranges and
  border, grid pieces, and where a corner is stored (design 3.7, with the tie rule of Task 15); 5 the weight
  curve with `<eq:weight-curve>` (design 5.3); 6 `<sec-nearby-faces>` candidates (with the half-open rule of Task 17, in the sentence given there), kept mask,
  `<eq:scaled-weights>` (design 5.4 steps 4 to 6), mean, `<eq:weighted-spread>`, the three fit outcomes, the
  complete case as a special case, and the stated properties (constants always reproduced, linear fields when
  `kFull`, weights never negative, and 0 only for a face exactly 1.5 widths away, weights jump when a particle changes cell across a size change); 7
  `<sec-grid-memory>` memory per block: the table of design 3.6 with the entries of Task 6a, and one sentence
  that the grid's GPU functions need no stack held by the driver.
- [ ] **Step 2: Citations.** [2] Ding 2020, [1] Jiang 2015, [3] Hu 2018, [18] Stomakhin 2014, [5] Wang 2020,
  [15] Fei 2021, [6] Wang 2026, by their numbers in `references.typ`; Harlow and Welch 1965 by author and year
  only. Copy each from the file; never from memory.
- [ ] **Step 3: Figures** as design section 8 describes, each drawn with `figure_style.typ` and `cell-grid`,
  each with a caption stating the one thing to see. `<fig-nearby-faces>`: kept faces as filled dots sized by
  weight, left-out faces hollow, the mean offset arrow, the spread ellipse, beside the complete 3 x 3 case.
- [ ] **Step 4: Pseudocode** with `procedure(...)`: `<proc-build-grid>` with the 8 steps of design 5.1 item 8
  (1 check frame, version and ids; 2 sort, find overlaps; 3 mark split cells, check siblings; 4 size-step rule;
  5 blocks and their count; 6 neighbour tables; 7 cell states; 8 face piece kinds) and a closing line "on the
  GPU: section 8" (Task 1b fills the reference); `<proc-nearby-face-weights>` with the 9 steps of design 5.4.
  Code comments `// Step N:` in Tasks 8, 9a, 17 and 18 follow these numbers.
- [ ] **Step 5: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`. Read
  every new page; fix overlaps and clipped labels.
- [ ] **Step 6: Commit.** Subject: `Describe the grid in the method paper`.

### Task 1b: Method paper: building the tables in parallel

**Kind:** numerics

Load the same three skills.

**Files:**
- Modify: `docs/method-paper/fluidity-method.typ`
- Create: `docs/method-paper/figures/grid_parallel_build.typ`

**Interfaces:**
- Consumes: Task 1a (`<sec-grid>`, `<proc-build-grid>`, `cell-grid`); `<sec-stable-order>`, `<sec-run-on>`
  (Foundation's sections).
- Produces labels: `<sec-parallel-build>`, `<fig-parallel-build>`, `<proc-build-grid-in-parallel>`,
  `<proc-count-grid>`.

- [ ] **Step 1: Text.** Part 8 of `<sec-grid>`, `<sec-parallel-build>`: a run is all CPU or all GPU, so in a
  GPU run the cell list never leaves the GPU and only a status record comes back. Each sequential step of
  `<proc-build-grid>` becomes "compute a key, stable sort, mark, prefix sum, scatter". One paragraph with the
  block-index example: blocks `[A, A, B, C, C, C]` in sorted order give marks `[1, 0, 1, 1, 0, 0]` and exclusive
  sums `[0, 1, 1, 2, 3, 3]`; the first entry of each run writes its block at index 0, 1, 2, and any entry's
  block index is `sum + mark - 1`. The fixed order is kept because the sort keys are the fixed order itself;
  state the key formula of Task 7a and why it is never negative. The first failure is the smallest
  `(step, position)` pair, found with an integer minimum, so it does not depend on thread timing. The tail of
  the sorted array is filled with the never-valid id before the sort, so no cell of an earlier build is read.
  Why the tables are then identical to the CPU's: each entry is a function of the sorted list, not of the order
  the work was done in.
- [ ] **Step 2: Figure** `<fig-parallel-build>`: a 2D slice of six cells of two sizes (drawn with `cell-grid`)
  above four lines of boxes (the cells' block keys in input order, after the stable sort, the marks, the
  exclusive sums) with an arrow from each marked entry to its block index. Caption: "the first entry of each
  run writes its block; the sums give every block its index in the fixed order".
- [ ] **Step 3: Pseudocode** `<proc-build-grid-in-parallel>` with twelve steps, each line naming what one thread
  handles: P1 input checks on the CPU; P2 id checks, one thread per input cell; P3 sort into fixed cell order;
  P4 overlaps, P5 siblings, P6 size-step rule, one thread per sorted cell; P7 add split cells and fill the tail
  of the array; P8 sort cells by block and number the blocks; P9 neighbour tables, one thread per block; P10
  cell states, one thread per cell place; P11 face piece kinds, one thread per face place; P12 counts and
  status. Replace the closing line of `<proc-build-grid>` by a reference to this procedure. Code comments
  `// Step PN:` in Tasks 10, 11a and 11b follow these numbers.
- [ ] **Step 4: Counting without tables.** One paragraph in `<sec-cell-blocks>` and the procedure
  `<proc-count-grid>` with four steps: 1 check and sort the list (steps 1 to 4 of `<proc-build-grid>`); 2
  count the active cells per level; 3 count the face pieces, one cell at a time, from the place beside each of
  its six sides (active, inside a bigger cell, split or absent); 4 add the split cells, sort by block and
  count the distinct blocks. Say why it exists (the planner counts before any buffers exist) and that a place
  beyond the coordinate limit of its own size level holds no cell of that level and gets no id: at the low limit
  nothing lies beyond a cell; at the high limit a bigger cell may, found at its own size level. Code comments `// Step N:` in Task 8b follow these
  numbers.
- [ ] **Step 5: Run** the `typst compile` command of Task 1a; read the new pages.
- [ ] **Step 6: Commit.** Subject: `Describe the parallel grid build in the method paper`.

### Task 2: Grid errors

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/grid/CMakeLists.txt`, `native/fluidity/grid/grid_error.h` (S),
  `grid_error_message.h` (CPU only), `grid_error_message.cc`, `grid_error_test.cc`
- Modify: the CMake file that adds `contact/` (add `grid/` the same way)

**Interfaces:**
- Follows `contact/contact_error.h` and `contact/contact_error_message.h/.cc` as built.
- Produces in `grid_error.h`: `enum class GridError : std::uint8_t` with the twelve enumerators of design 3.3 in
  that order (`kInternalFault` eleventh, `kOutOfMemory` last), and `struct GridFailure { GridError error = GridError::kNone; std::uint64_t
  magnitude = 0; }` (Task 3 adds `cell` and `other_cell` between them once `CellId` exists). It is shared
  because `FindNearbyFaces`, in a shared header, reports through a `GridError*`.
- Produces in `grid_error_message.h`: `std::string_view ToMessage(GridError error)`, `std::string_view
  ToName(GridError error)`. An out-of-range value gives the text of an unknown error and the name `"unknown"`.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ToName, EveryError_IsStableSnakeCase)`: for each of the 12 values the name is the enumerator without
    `k` in lower snake case; all 12 distinct; `ToName(GridError::kInternalFault) == "internal_fault"`; `ToName(GridError::kTooManyCellBlocks) == "too_many_cell_blocks"`,
    `ToName(GridError::kNone) == "none"`.
  - `TEST(ToMessage, EveryError_HasADistinctNonEmptyMessage)`.
  - `TEST(ToMessage, InternalFault_SaysItIsNotTheScene)`: the text contains "FluidiTy" and "report".
  - `TEST(GridError, OutOfMemory_IsTheLastValue)`: `static_assert(static_cast<int>(GridError::kOutOfMemory) ==
    11)`, `static_cast<int>(GridError::kInternalFault) == 10`, `ToName(GridError::kOutOfMemory) == "out_of_memory"`, and `GridFailure{}.magnitude == 0`.
- [ ] **Step 2: Run** and see the compile fail. **Step 3: Implement.** Messages say what the user can do
  (`kTooManyCellBlocks`: "The grid needs more cell blocks than the limit. Raise max_cell_blocks or use fewer
  small cells."; `kInternalFault`: "FluidiTy found a state inside the grid that should never occur. This is a fault
  in FluidiTy, not in your scene. Please report it and attach the scene."). `Read:` `<sec-grid>`.
- [ ] **Step 4: Commit.** Subject: `Add the grid's errors`. New names: `GridError` (one enum for the module),
  `GridFailure`, `ToMessage`, `ToName`.

### Task 3: Cell, block and face identity

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/grid_ids.h` (S), `grid_ids_test.cc`, `grid_ids_gpu_compile_test.cu` (label
  `no_gpu`, registered as `base_constants_gpu_compile_test.cu` is in `base/CMakeLists.txt`),
  `grid_tables.h` (S; for now `GridFrame`, `CellState` and `FacePieceKind`)
- Modify: `grid_error.h` (`GridFailure` gains `CellId cell{}; CellId other_cell{};`), `docs/glossary.md`,
  `tools/check_rules_allowed_names.txt` (one line: `ComesBefore  # the fixed order of cells and blocks, a
  predicate read as a sentence: first comes before second`), build registration

**Interfaces:**
- Consumes: `Axis` (`base/axis.h`), `Vec3` (`base/vec3.h`), `GridFailure` (Task 2).
- Produces: every declaration of design 3.1 and the `grid_ids.h` part of 3.7 (`kCornerPlacesPerBlock`,
  `CornerPlaceIndex`, both `ToCellBlockIndex`), with these two additions to what the design lists:
  `bool ComesBefore(CellId first, CellId second)` beside the `CellBlockId` form, and `struct CellBlockRange`
  (declared here so shared headers of other modules take it on either processor).
- Produces in `grid_tables.h` the `GridFrame` struct and the `CellState` and `FacePieceKind` enums of design
  3.2 (the buffers of Task 6b hold these element types), and nothing else there yet.
- Produces also `constexpr bool ComputeIsCoordInRange(CellCoord coord)`: every axis in
  `-kMaxCoord..kMaxCoord`. Code that looks at a place beside a cell calls it before it makes an id.
- **Decided here** (the design says only "packs"):
  - Valid cell coordinates are `-kMaxCoord..kMaxCoord` on each axis. A coordinate is stored as `c + kMaxCoord`
    in 19 bits, so the stored values `0..2^19 - 2` are valid and `2^19 - 1` means `kMaxCoord + 1`.
  - `CellId`: bit 63 set in every valid id; bits 57..60 size level; bits 38..56 z, 19..37 y, 0..18 x.
    `CellId{0}` is never valid. `CellBlockId`: the same layout with the block coordinate `floor(c / 4)`.
  - `FacePieceId`: bits 61..62 hold `axis + 1` (so 0 is never valid), 57..60 size level, coordinates as above.
    `cell_high_coord` may be `kMaxCoord + 1` along the normal axis (the high side of the last cell).
  - What the header must say: `ToCellId` keeps only 19 bits per axis. `kMaxCoord + 1` is the one value outside
    the range that keeps its meaning (the checks of Task 7a report it as `kCoordOutOfRange`); any other value
    outside the range gives the id of another cell, so a caller checks its coordinates first.
    `ComputeSmallCell` of a cell whose doubled coordinate leaves the range does the same.
  - `ComesBefore`: compare `(z, y, x)` of the low corner in level-15 units as `std::int64_t` (cell coordinate
    `<< (15 - level)`; for a block, block coordinate `* 4` first), then the smaller level value first.

- [ ] **Step 1: Failing tests** (label `cpu`); each `static_assert`s one case so `constexpr` is proven:
  - `TEST(ToCellId, RoundTrip_AtCoordLimits)`: for levels 0 and 15 and every combination of `-kMaxCoord`, `0`,
    `kMaxCoord` per axis, `ToSizeLevel` and `ToCellCoord` give back the inputs exactly.
  - `TEST(ToCellId, OnePastTheLimit_KeepsItsCoordinate)`: x `kMaxCoord + 1` round-trips.
  - `TEST(ToFacePieceId, HighSideOfCellAtCoordLimit_RoundTrips)`: axis X, level 3, `cell_high_coord`
    `{kMaxCoord + 1, -kMaxCoord, 0}`; the id is non-zero and differs from the same with axis Y.
  - `TEST(ComputeBigCell, NegativeCoords_RoundTowardMinusInfinity)`: level 2 `{-1, -2, 3}` gives level 1
    `{-1, -1, 1}`.
  - `TEST(ComputeSmallCell, EveryEighth_GivesBackItsBigCell)`: for `cell_eighth` 0..7,
    `ComputeBigCell(ComputeSmallCell(c, e)) == c` and `ComputeCellEighth(...) == e`.
  - `TEST(ToCellBlockId, CellsOfOneBlock_ShareIt)`: `{-4..-1}` on each axis at level 1 give one block; `{0..3}`
    another.
  - `TEST(ComesBefore, BiggerBlockAtSameCorner_ComesFirst)`, `TEST(ComesBefore, LowerZ_ComesFirst)`,
    `TEST(ComesBefore, CellsAtCoordLimits_OrderByLowCorner)` (level 0 `{-kMaxCoord, 0, 0}` before level 15
    `{-kMaxCoord, 0, 0}` before level 0 `{kMaxCoord, 0, 0}`; neither of two equal ids comes before the other).
  - `TEST(ComputeIsCoordInRange, OnePastEitherLimit_IsOutside)`: `{kMaxCoord, -kMaxCoord, 0}` is inside;
    `{kMaxCoord + 1, 0, 0}` and `{0, 0, -kMaxCoord - 1}` are outside.
  - `TEST(ToCellBlockIndex, PlaceIndexes_DivideByPlacesPerBlock)`: cell 130 gives 2; face 479 gives 1.
  - `TEST(CellBlockRange, IsTriviallyCopyable)`: `static_assert`, and `sizeof(CellBlockRange) == 8`.
  Label `no_gpu`, `grid_ids_gpu_compile_test.cu`, in the form of `base/base_constants_gpu_compile_test.cu`: a
  GPU function that takes a `CellBlockRange` by value and a `CellPlaceIndex` and writes
  `ToCellBlockIndex(cell).value - range.first_block` to a pointer; it is never launched.
  `TEST(CellBlockRange, BuiltForTheGpu_HasAnAddress)` checks the function's address as the base test does; the
  container's CUDA build is the proof that the struct and `grid_ids.h` compile in device code, and the test's
  comment says so.
- [ ] **Step 2: Run**, see failures. **Step 3: Implement.** File header: `Read:` `<sec-cell-sizes>`,
  `<sec-cell-blocks>`. Glossary: update the existing lines `place` (add `CornerPlaceIndex` if absent; never say
  `slot`) and `cell_eighth`; add `grid version`, `fixed order` (blocks and cells by low corner, `z`, `y`, `x`,
  bigger first) and `face piece` (the part of a face between two cells or at the grid's edge; not a grid piece,
  not a memory piece).
- [ ] **Step 4: Commit.** Subject: `Add stable ids and per-version indexes for cells, blocks and faces`. List
  every new name of 3.1, the 3.7 additions, `CellBlockRange`, `GridFrame`, `CellState`, `FacePieceKind`, `ComputeIsCoordInRange`, and `ComesBefore` as the
  one name
  outside the allowed verbs.

### Task 4: The weight curve

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/weight_curve.h` (S), `native/fluidity/grid/weight_curve_test.cc`
- Modify: build registration

**Interfaces:**
- Produces: `WeightCurveValue`, `ComputeWeightCurve`, `kWeightCurveReach`, `kMaxNearbyFaces`, `kMinWeightSum`,
  `kMinPivotShare` (design 3.5) and `ComputeCompleteWeightedSpread` (3.7), exactly.
- Port: `git show main:native/src/sparse_mac_grid_3d.cc` lines 285-296 (the three factors and derivatives) and
  `git show main:reference/src/fluidity_reference/basis.py` lines 23-24. Only the formula of design 5.3; none of
  the rounding machinery of lines 83-198 and 312-336.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ComputeWeightCurve, AnyPosition_ThreeNeighboursSumToOne)`: 1,000 offsets `s` in `[-0.5, 0.5)` from
    seed `20261114`; `Q(s - 1) + Q(s) + Q(s + 1)` within `4e-16` of 1; the slopes' sum within `4e-16` of 0.
  - `TEST(ComputeWeightCurve, AtAndBeyondReach_IsZero)`: `t = 1.5, -1.5, 2.0` give weight 0 and slope 0.
  - `TEST(ComputeWeightCurve, AtHalf_BothBranchesAgree)`: `t = 0.5`: weight `0.5`, slope `-1.0`, exact.
  - `TEST(ComputeCompleteWeightedSpread, WidthTwo_IsOne)`: exact; `static_assert`.
- [ ] **Step 2: Run**, see failures. **Step 3: Implement**; the header comment gives `Q`, its slope and a worked
  example (`t = 0.25`: weight 0.6875, slope -0.5). `Read:` `<eq:weight-curve>`.
- [ ] **Step 4: Commit.** Subject: `Add the quadratic weight curve`.

### Task 5: Test-only geometric checker and random grids

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/grid_test_support.h` (CPU only), `grid_test_support.cc`,
  `grid_test_support_test.cc`
- Modify: build registration: `fluidity_add_test_library(fluidity_grid_test_support SOURCES grid_test_support.cc
  LINKS fluidity_base_test_support)`; every grid test from here on names it in `LINKS`; never linked by
  `fluidity`

**Interfaces:**
- Consumes: Task 3 (`CellId` and its functions, `GridFrame`, `ComesBefore`).
- Produces (test-only, `namespace fluidity::grid`, as base's test support is in `namespace fluidity`):
  - `std::uint64_t ToUniformIndex(std::uint64_t raw, std::uint64_t count)` (`raw % count`) and `double
    ToUnitDouble(std::uint64_t raw)` (53 high bits / 2^53): the only mappings random tests use.
  - `std::vector<CellId> BuildRandomGridCells(std::uint64_t seed, int size_level_count, int
    biggest_cells_per_axis)`: a valid list (siblings together, size-step rule). Method, fixed so that every
    machine gives the same list: start from the box of level-0 cells `{0..n-1}^3`; visit the cells in fixed
    order, level by level; split a level-0 cell when `ToUniformIndex(raw, 2) == 0` and a smaller cell when
    `ToUniformIndex(raw, 4) == 0`, down to `size_level_count - 1`, one raw number per cell visited; then
    `BuildSizeStepClosure`.
  - `void ApplyFixedShuffle(std::vector<CellId>& cells, std::uint64_t seed)`: the project's own shuffle, the
    same on every machine: a `std::mt19937_64` of `seed`; for `i` from the last index down to 1, swap entry `i`
    with entry `ToUniformIndex(raw, i + 1)`, one raw number per swap.
  - `std::vector<CellId> BuildSizeStepClosure(std::vector<CellId> cells, int size_level_count)`: split any
    active cell two or more levels bigger than a touching cell, repeat until none; for hand-built scenes too.
  - `class GeometricGrid`: built from a `GridFrame` and an active-cell list, with no blocks and no tables. Works
    in integer smallest-cell units; a width in metres is `biggest_cell_width` halved `level` times (exact).
    Answers: `FacePieces()` (every face piece as axis, level, `cell_high_coord`, low cell id or none, high cell
    id or none, area), `TouchingCells(CellId)`, `Corners()` (each physical corner once), `ContainingCell(Vec3)`,
    `NearbyFaces(CellId cell, std::array<int, 3> window_quarters, Axis)` (the candidate rule of design 5.4 step 2 by
    brute force over every face piece, in integer smallest-cell units: touching, levels and the window `n_p - 5
    <= n_f <= n_p + 6`; the caller passes the cell and the three `n_p`, so the checker shares only that definition),
    `FacesWithin(FacePieceId, double radius_in_cell_widths)`.

- [ ] **Step 1: Failing tests** (label `cpu`): `TEST(GeometricGrid, OneBigCell_HasSixFacePieces)`;
  `TEST(GeometricGrid, SplitCellBesideBigCell_HasFourQuarterFacePieces)` (areas 0.25 m2 each; 41 face pieces in
  all for level 0 `{0,0,0}` with the 8 level-1 cells of `{1,0,0}`);
  `TEST(BuildRandomGridCells, SameSeed_GivesSameList)`;
  `TEST(ApplyFixedShuffle, SeedOne_GivesAPermutationThatDiffersFromSeedTwo)` (100 distinct ids: the same set
  after the shuffle, the same order for the same seed, another order for seed 2, and not the input order);
  `TEST(BuildRandomGridCells, AnySeed_KeepsSiblingsAndSizeStep)` (100 seeds, checked by brute force);
  `TEST(BuildRandomGridCells, TwoHundredGrids_UseEveryLevel)` (over seeds 1..200 with arguments `(4, 3)`, each
  of the 4 levels holds an active cell in at least one grid, and no grid has more than 13,824 cells).
- [ ] **Step 2: Run**, fail; **Step 3: Implement** by brute force (clarity over speed: quadratic loops are fine
  up to a few thousand cells).
- [ ] **Step 4: Commit.** Subject: `Add a test-only geometric grid checker`. Body: test-only, never linked into
  the library.

### Task 6a: The build status record and the grid's memory estimate

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/grid/staggered_grid.h` (CPU only; for now `kMaxCellBlocksLimit`, `GridSettings`,
  `GridCounts`, `BuilderBuffersSharing`), `grid_build_status.h` (S, private), `grid_memory_estimate.h`,
  `grid_memory_estimate.cc`, `grid_memory_estimate_test.cc`
- Modify: `docs/glossary.md` (`failure key`: the number `(step, position, detail)` whose smallest value names
  the first failure of a build), build registration

**Interfaces:**
- Consumes: `MemoryEstimateEntry`, `MemoryWhere`, `MemoryCounts`, `CountedPer`, `KeptFor`, both
  `ComputeMemoryBytes` forms, `ComputeRunMemoryBytes`, `ComputeRunMemoryPieceBytes` (`base/memory_count.h`);
  `ComputeOrderBuffersSize` (`base/stable_order.h`; `{6, 4096}` on the GPU); `GridError`, `CellId`,
  `kMaxSizeLevels`.
- Produces: `kMaxCellBlocksLimit`, `GridSettings`, `GridCounts`, `BuilderBuffersSharing` (design 3.3);
  `ToMemoryCounts`, `BuildGridMemoryEstimate` (design 3.6, its second parameter named `grid_count`); private
  `kNoFailureKey`, `GridBuildStatus` (design 3.3, as written there; `static_assert` trivially copyable).

  ```cpp
  // How many GridBuilderBuffers the "builder buffers" entries of BuildGridMemoryEstimate count.
  enum class BuilderBuffersSharing : std::uint8_t {
    kOnePerGrid,      // One GridBuilderBuffers for each of the grid_count buffers.
    kOneForAllGrids,  // One GridBuilderBuffers lent to all of them.
  };
  // The memory estimate of grid_count GridBuffers of max_cell_blocks blocks on run_on, with their builder
  // buffers: one entry per array, in the order the builders take their memory pieces.
  // MemoryCounts::cell_blocks must equal max_cell_blocks when the entries are summed.
  std::vector<MemoryEstimateEntry> BuildGridMemoryEstimate(
      RunOn run_on, std::uint32_t grid_count, std::uint32_t max_cell_blocks,
      BuilderBuffersSharing builder_buffers_sharing = BuilderBuffersSharing::kOnePerGrid);
  ```
- **Entries**, in this fixed order, names prefixed `grid: `, all `KeptFor::kWholeRun`:
  1. The five tables, `where = kRunOn`: `block ids` (kCellBlock, 8 B), `same-size-level neighbour blocks`
     (kCellBlock, 108 B), `bigger and smaller blocks` (kCellBlock, 36 B: one array, the bigger block of every
     block first, then the 8 smaller blocks of every block), `cell states` (kCellPlace, 1 B), `face piece kinds`
     (kFacePlace, 1 B). 456 B per block.
  2. For `run_on == kCpu`: `builder buffers: cell ids` (kCellPlace, 8 B) and `piece map` (kCellBlock, 4 B), both
     `where = kCpu`. The CPU builder sorts with `std::sort` and needs no `OrderBuffers`.
  3. For `run_on == kGpu`, all `kRunOn`: `builder buffers: cell ids` (kCellPlace, 8 B), `builder buffers: sort
     keys` (kCellPlace, 8 B), `builder buffers: order` and `builder buffers: second order` (each one
     `kGivenCount` entry, `bytes_each = 1`, `given_count = 4 * (64 * max_cell_blocks + 1)`: a prefix sum writes
     its total after the last entry), `builder buffers: order buffers` (one `kGivenCount` entry, `given_count =
     bytes_per_item * (64 * max_cell_blocks + 1) + fixed_bytes` of `ComputeOrderBuffersSize(RunOn::kGpu)`), then
     `piece map` (kCellBlock, 4 B) and `build status` (`kGivenCount` 1, `sizeof(GridBuildStatus)`). No entry of
     a GPU grid is `kCpu`.
  `copies = grid_count`, except that with `kOneForAllGrids` the `builder buffers:` entries have `copies = 1`.
  The grid has no `MemoryWhere::kDriverStack` entry (Global Constraints "GPU stack"); the header says so.
- Numbers to state in the header: per block on the GPU, tables 456 B and piece map 4 B per grid; builder buffers
  `1536 + 64 x 6 = 1920` B once, plus the fixed bytes `4096 + 6 + 8`. On the CPU: 456 B (`kRunOn`) and 516 B
  (`kCpu`) per grid. Two GPU grids that share one builder buffers cost `2 x 460 + 1920 = 2840` B per block.
- `ToMemoryCounts` sets `cell_places = 64`, `face_places = 240`, `corner_places = 125` times `cell_blocks`.

- [ ] **Step 1: Failing tests** (label `cpu`); `counts` = `MemoryCounts` at 1,000 blocks:
  - `TEST(BuildGridMemoryEstimate, CpuGrid_IsNineHundredSeventyTwoBytesPerBlock)`:
    `ComputeMemoryBytes(BuildGridMemoryEstimate(kCpu, 1, 1000), counts) == 1000 * 972` (8 + 108 + 36 + 64 + 240
    + 512 + 4); with `where`, `kRunOn` gives `1000 * 456` and `kCpu` gives `1000 * 516`.
  - `TEST(BuildGridMemoryEstimate, GpuRun_KeepsNothingOnTheCpu)`: `BuildGridMemoryEstimate(kGpu, 1, 1000)` gives
    `kCpu` 0 and `kRunOn` `1000 * (456 + 1536 + 4 + 64 * bytes_per_item) + fixed_bytes + bytes_per_item + 8 +
    sizeof(GridBuildStatus)`, with `ComputeOrderBuffersSize(RunOn::kGpu)` read in the test; `kDriverStack` 0.
  - `TEST(BuildGridMemoryEstimate, NoEntryIsACpuCopy)`: no entry name of either processor ends in `(CPU copy)`.
  - `TEST(BuildGridMemoryEstimate, TwoGrids_DoublesEveryEntry)`: every entry has `copies == 2`; the total of
    each `where` doubles.
  - `TEST(BuildGridMemoryEstimate, SharedBuilderBuffers_CountedOnce)`: both processors, 2 grids,
    `kOneForAllGrids`: every `builder buffers:` entry has `copies == 1`, every other entry `copies == 2`; on
    the GPU at 1,000 blocks the `kRunOn` total is `1000 * (2 * 460 + 1920) + 4096 + 6 + 8 + 2 *
    sizeof(GridBuildStatus)`.
  - `TEST(BuildGridMemoryEstimate, GpuEntries_EachIsOneMemoryPiece)`: for 1, 7 and 1,000 blocks, the sum of
    `ComputeRunMemoryPieceBytes` over each GPU entry's bytes equals `ComputeRunMemoryBytes` of the estimate.
  - `TEST(ToMemoryCounts, NineBlocks_CountsPlacesPerBlock)`: 576, 2,160 and 1,125 places.
  - `TEST(GridBuildStatus, IsTriviallyCopyable)`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement.** `Read:` `<sec-grid-memory>`.
- [ ] **Step 4: Commit.** Subject: `Add the grid's memory estimate and build status record`. New names:
  `GridSettings`, `GridCounts`, `kMaxCellBlocksLimit`, `BuilderBuffersSharing`, `ToMemoryCounts`,
  `BuildGridMemoryEstimate`.

### Task 6b: `GridBuilderBuffers` and `GridBuffers`

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/staggered_grid_cpu.cc`, `grid_buffers_internal.h` (private, CPU only),
  `grid_buffers_test.cc`, `grid_buffers_no_gpu_test.cc`
- Modify: `staggered_grid.h`, `docs/glossary.md` (`builder buffers`; and the line `piece` gains the note "say
  memory piece where a face piece or a grid piece could be meant"), build registration

**Interfaces:**
- Consumes: Task 6a; `CellBlockId`, `CellState`, `FacePieceKind` (Task 3: the element types of the tables);
  `Buffer`, `BuildBuffer(RunOn, std::size_t, RunMemory&)`, `RunMemory`, `BuildRunMemory`,
  `OrderBuffers`, `BuildOrderBuffers`, `CheckCanRunOn`, `ToBufferError`, `BufferError`.
- Produces, in `staggered_grid.h` (design 3.3; comments written here without the design's history):

  ```cpp
  // The temporary arrays of grid builds and checks on one processor, for grids of at most max_cell_blocks
  // blocks: the "grid: builder buffers: ..." entries of BuildGridMemoryEstimate. Holds nothing between calls.
  // Owned by the caller and lent to every GridBuffers built with it: those must not outlive it, it must not be
  // moved while they exist, and one build or check uses it at a time. Move-only.
  class GridBuilderBuffers {
   public:
    std::uint32_t max_cell_blocks() const;
    RunOn run_on() const;
    std::size_t byte_count() const;  // Bytes of its arrays; 0 after a move.
  };
  // Builds them as memory pieces of run_memory, in the order of the estimate's entries.
  // Requires: run_on == run_memory.run_on() when run_on can run, checked by assert in debug builds.
  // Fails: kRunOnCallFailed  CheckCanRunOn(run_on) failed; nothing is taken from run_memory.
  //        kBadInput         max_cell_blocks is 0 or above kMaxCellBlocksLimit; nothing is taken.
  //        kOutOfMemory      an array does not fit in what run_memory has left; magnitude is its bytes.
  std::expected<GridBuilderBuffers, GridFailure> BuildGridBuilderBuffers(RunOn run_on,
                                                                         std::uint32_t max_cell_blocks,
                                                                         RunMemory& run_memory);
  // Exact bytes of a GridBuilderBuffers: bytes_per_cell_block * max_cell_blocks + fixed_bytes.
  struct GridBuilderBuffersSize {
    std::uint32_t bytes_per_cell_block = 0;
    std::uint64_t fixed_bytes = 0;
  };
  // The sum of the "grid: builder buffers: ..." entries of BuildGridMemoryEstimate: one definition for both.
  GridBuilderBuffersSize ComputeGridBuilderBuffersSize(RunOn run_on);
  // Builds a grid's own arrays (five tables, the piece map and, on the GPU, the status record) for
  // settings.max_cell_blocks blocks on settings.run_on, as memory pieces of run_memory. The buffers use
  // builder_buffers for every build and check. Requires: builder_buffers.run_on() == settings.run_on ==
  // run_memory.run_on() when that processor can run, checked by assert in debug builds.
  // Fails: kRunOnCallFailed, kBadInput (room 0 or above the limit, or builder_buffers.max_cell_blocks() below
  //        settings.max_cell_blocks), each before any piece is taken; kOutOfMemory as above.
  std::expected<GridBuffers, GridFailure> BuildGridBuffers(const GridSettings& settings,
                                                           GridBuilderBuffers& builder_buffers,
                                                           RunMemory& run_memory);
  ```
  and `class GridBuffers` (`max_cell_blocks()`, `run_on()`; move-only; a moved-from one has room 0).
- Both builders walk the entries of `BuildGridMemoryEstimate(run_on, 1, max_cell_blocks)` in order, one
  `BuildBuffer` (or `BuildOrderBuffers`) per entry, so the estimate and the arrays cannot drift apart. Each
  goes through a private form in `grid_buffers_internal.h` that takes the entries as a span
  (`BuildGridBuffersFromEntries`, `BuildGridBuilderBuffersFromEntries`); an entry it has no array for, an array
  left without an entry, or an entry whose bytes differ from its array's, is `kInternalFault` (in every build;
  the memory pieces already taken stay in the block). Both `Fails:` lists name it.
- Produces, in `grid_buffers_internal.h` (this module's code and tests only): `struct GridBuffersArrays` (one
  `Buffer` per entry of the grid's own arrays, and a pointer to the lent `GridBuilderBuffers`), `struct
  GridBuilderBuffersArrays` (one `Buffer` per builder entry; on the GPU also the `OrderBuffers`), and
  `ToGridBuffersArrays(GridBuffers&)`, `ToGridBuffersArrays(const GridBuffers&)`,
  `ToGridBuilderBuffersArrays(GridBuilderBuffers&)`, `ToGridBuilderBuffersArrays(const GridBuilderBuffers&)`,
  each returning a reference (const for the const form). Tasks 8 to 11 and 24 reach the arrays only this way.

- [ ] **Step 1: Failing tests.** Label `cpu`; every test builds its `RunMemory` from the estimate first:
  - `TEST(BuildGridBuffers, CpuBuffers_HoldTheEstimatedBytes)`: 1,000 blocks: `byte_count()` of the builder
    buffers plus the sum of `size() * sizeof(element)` over `ToGridBuffersArrays` equals `1000 * 972`.
  - `TEST(BuildGridBuffers, TwoBuffersShareBuilderBuffers_HoldTheSharedEstimate)`: one builder buffers, two
    grid buffers at 1,000 blocks: bytes equal `ComputeMemoryBytes(BuildGridMemoryEstimate(kCpu, 2, 1000,
    kOneForAllGrids), counts)`; both point at the one `GridBuilderBuffers`.
  - `TEST(BuildGridBuffers, IntoRunMemory_HandsOutExactlyTheEstimate)`: at 1,000 blocks, for `kOnePerGrid` with
    1 grid and `kOneForAllGrids` with 2: a `RunMemory` of the `kRunOn` plus the `kCpu`
    `ComputeRunMemoryBytes` of the CPU estimate; after the builders, `handed_out_bytes() == reserved_bytes()`.
  - `TEST(BuildGridBuffers, RunMemoryOnePieceShort_IsOutOfMemory)`: the 1,000-block reservation less 256 B:
    `kOutOfMemory`, `magnitude` = the bytes of the array that did not fit. Runs in debug and release.
  - `TEST(BuildGridBuffers, LentBuilderBuffersTooSmall_IsBadInput)`: builder buffers for 500, grid buffers for
    1,000: `kBadInput`, `handed_out_bytes()` unchanged.
  - `TEST(BuildGridBuilderBuffers, ZeroBlocks_IsBadInput)`; `TEST(BuildGridBuffers, ZeroBlocks_IsBadInput)`
    (`GridSettings{}`); `TEST(BuildGridBuffers, AboveLimit_IsBadInput)` (`kMaxCellBlocksLimit + 1`).
  - `TEST(ComputeGridBuilderBuffersSize, Cpu_EqualsByteCount)`: `BuildGridBuilderBuffers(kCpu, 1000,
    run_memory)->byte_count() == 1000 * bytes_per_cell_block + fixed_bytes` (512,000).
  - `TEST(ComputeGridBuilderBuffersSize, EitherProcessor_EqualsBuilderEntries)`: for `kCpu` and `kGpu`, the
    formula at 1,000 blocks equals the sum of the `builder buffers:` entries (no GPU memory needed).
  - `TEST(BuildGridBuffers, EntriesNotMatchingArrays_IsInternalFault)`: the private form with the CPU
    estimate less its `cell states` entry, then with that entry's `bytes_each` doubled: `kInternalFault` both
    times; the same for `BuildGridBuilderBuffersFromEntries` with no entry at all.
  - `TEST(GridBuffers, MovedFrom_HasNoRoom)`.
  Label `no_gpu` (as `contact/move_particles_no_gpu_test.cc`): `TEST(BuildGridBuilderBuffers,
  OnGpuWithoutAGpu_IsRunOnCallFailed)`: `BuildGridBuilderBuffers(RunOn::kGpu, 1000, cpu_run_memory)` fails
  `kRunOnCallFailed` and `handed_out_bytes() == 0`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement.** `Read:` `<sec-cell-blocks>`, `<sec-grid-memory>`,
  `<sec-run-memory>`.
- [ ] **Step 4: Commit.** Subject: `Add the grid buffers that are built once`. New names: `GridBuffers`,
  `BuildGridBuffers`, `GridBuilderBuffers`, `BuildGridBuilderBuffers` (one builder buffers can serve every grid
  of a run), `GridBuilderBuffersSize`, `ComputeGridBuilderBuffersSize` (Foundation's buffers pattern).

### Task 7a: Shared build formulas: keys, checks and split cells

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/grid_build_steps.h` (S, private; included by no public header),
  `grid_build_steps_test.cc` (label `cpu`)
- Modify: build registration

**Interfaces:**
- Consumes: Tasks 2, 3 (`ComesBefore` for cells and blocks, `GridFrame`, `ComputeIsCoordInRange`),
  `GridBuildStatus` (Task 6a); `ApplyFixedShuffle`, `BuildRandomGridCells` (Task 5, tests).
- Produces, all `FLUIDITY_CPU_AND_GPU`, `noexcept`, on raw pointers and counts, allocating nothing:
  - **Sort keys.** `std::uint64_t ToCellSortKey(CellId cell, int pass)` and `std::uint64_t
    ToBlockSortKey(CellId cell, int pass)`: three passes for `BuildStableOrder`, least significant first. For
    a coordinate `c` of a cell at level `l`, in unsigned 64-bit arithmetic,
    `biased(c, l) = (c + 2^(18 + l)) << (15 - l)`. Cell key: pass 0 `(biased(x, l) << 4) | l`, pass 1
    `biased(y, l)`, pass 2 `biased(z, l)`. Block key: the same with `4 * floor(c / 4)` (the block's low corner
    in cells of level `l`) in place of `c`. An id with bit 63 clear has the key `~0` in every pass, so it sorts
    last. `biased` is the low corner in level-15 units plus `2^33`, so key order is `z`, then `y`, then `x`,
    then bigger level first: the order of `ComesBefore`.
  - **Why the keys are in range, and the same on every machine** (write this in the header). A valid `c` is at
    least `-(2^18 - 1)`, so `c + 2^(18 + l) >= 1`. A block's low corner is at least `-2^18`, so its sum is at
    least 0, and 0 only for the lowest block of level 0. Nothing negative is ever shifted. The largest value is
    `(2^18 - 1 + 2^18) << 15 = 2^34 - 2^15`, so `biased < 2^34` and pass 0 is below `2^38`. Only integer adds
    and shifts are used.
  - `GridError CheckCellId(GridFrame frame, CellId cell)`: step 1 for one id: `kBadInput` (bit 63 clear, or
    level `>= size_level_count`), `kCoordOutOfRange` (a stored coordinate of `2^19 - 1`), else `kNone`.
  - `std::uint32_t FindSortedCell(const CellId* sorted_cells, std::uint32_t count, CellId cell)` and
    `CellBlockIndex FindSortedBlock(const CellBlockId* sorted_blocks, std::uint32_t count, CellBlockId block)`:
    binary search with `ComesBefore`; `kNoIndex` if absent. `FindCellBlock` (Task 12) calls the second.
  - `CellId ComputeBigCellAtLevel(CellId cell, SizeLevel size_level)`: the cell of `size_level` (the cell's own
    level or a bigger one) that contains `cell`.
  - **Cells beside a cell: no id outside the range.** Every function here and in Task 8b that looks at a place
    beside a cell works on the raw `CellCoord` (plain integers), derives the bigger cell's coordinate (`floor(c
    / 2)`) or a small cell's (`2 c`, `2 c + 1`) by arithmetic, and makes an id only for a coordinate that
    `ComputeIsCoordInRange` accepts at its own level. A coordinate outside the range of its own level holds no cell of that level; a bigger cell may still cover the
    place at the high limit, and is found at its own level, where the halved coordinate is in range. An id is
    never made first and tested after: outside the range it would be the id of another cell.
  - `std::uint32_t FindSortedCellAt(const CellId* sorted_cells, std::uint32_t count, SizeLevel size_level,
    CellCoord coord)`: `kNoIndex` when `coord` is outside the range, else `FindSortedCell` of its id. The one
    lookup the checks below and Task 8b use for a place beside a cell.
  - `struct CellCheck { GridError error; CellId cell; CellId other_cell; std::uint8_t detail; };` and the
    checks of one sorted position, each `CellCheck Check...At(const CellId* sorted_cells, std::uint32_t count,
    std::uint32_t position)`:
    - `CheckOverlapAt` (step 2): equal to the entry before; else the nearest active bigger cell that contains
      it, one level up first.
    - `CheckSiblingsAt` (step 3): the siblings of the cell, and of every split cell this cell adds. Let `d` be
      the cell. Repeat while `d` is not of level 0: every one of the 8 small cells of `ComputeBigCell(d)` must
      be *present*; the first that is not (eighth order) is the failure, with `cell` = that small cell,
      `other_cell` = the big cell, and `detail = (walk step << 3) | eighth`. Then, if `d` is eighth 0 of its
      big cell, `d` becomes that big cell (the split cell `ComputeSplitCellCountToAdd` adds) and the walk goes
      on; otherwise it stops. A small cell is *present* when it is active (`FindSortedCell`) or split, and it
      is recognised as split by going down its eighth-0 small cells, at most to level 15, until an active one
      is found. So every split cell has its siblings checked exactly once, at the position of the active cell
      at the end of its eighth-0 chain. Why this finds every gap (write it in the header): take the smallest
      cell that has an active cell inside its big cell but none inside itself; its big cell has a small cell
      that is active, or split with a whole eighth-0 chain (a broken chain would give a smaller gap), and the
      walk from that chain's end checks this big cell.
    - `CheckSizeStepAt` (step 4): the first of the 26 directions in `(dz, dy, dx)` order (0..25, the cell
      itself left out) whose same-level place beside the cell is covered by an active cell two or more levels
      bigger; detail = the direction. The covering coordinate at each bigger level is found by halving the
      raw coordinate, and looked up with `FindSortedCellAt`.
    Each check looks only at the smaller cell of a pair, so every broken pair is reported at exactly one
    position.
  - **Failure key.** `ToFailureKey(int step, std::uint64_t position, std::uint8_t detail)` =
    `(step << 56) | (position << 8) | detail`, with `ToFailureStep`, `ToFailurePosition`. The first failure is
    the smallest key: earliest step, then smallest position (input index for step 1, place in fixed cell order
    for steps 2 to 4). Step 5 (active plus split cells above `64 * max_cell_blocks`) and step 6 (blocks above
    `max_cell_blocks`) use position 0 and give `kTooManyCellBlocks` with `cell` zero. The CPU, which runs each
    step to its end before the next and scans positions in order, gives the same failure as the GPU's minimum.
  - `CellCheck ComputeFailureCells(const CellId* sorted_cells, std::uint32_t count, std::uint64_t failure_key)`:
    runs the check of the key's step (2, 3 or 4) again at the key's position, so one thread can fill the status
    record after the minimum is known. Steps 5 and 6 give `kTooManyCellBlocks` and zero cells.
  - `int ComputeSplitCellCountToAdd(CellId cell)`: how many split cells this active cell adds. It adds its big
    cell when it is eighth 0 of it, then that cell's big cell when that one is eighth 0 too, and so on (a loop
    of at most 15 steps, no search); the cells are `ComputeBigCellAtLevel(cell, level - 1)`, `level - 2`, ...
    Because siblings exist together (step 3 has passed), every split cell is added exactly once.

- [ ] **Step 1: Failing tests** (label `cpu`), frame origin 0, width 1 m, 3 levels unless stated:
  - `TEST(ToCellSortKey, ThreePasses_GiveComesBeforeOrder)`: 10,000 random ids at levels 0..3 from seed
    `20261114` (coordinates by `ToUniformIndex` in `-kMaxCoord..kMaxCoord`, of which 100 at a limit); sorting
    index lists by the three passes with `std::stable_sort` (test only) gives the order of `std::sort` with
    `ComesBefore`.
  - `TEST(ToCellSortKey, CoordLimits_FitInThirtyEightBits)`: pass 0 of level 0 `{kMaxCoord, 0, 0}` equals
    `((2^34 - 2^15) << 4)`; of level 0 `{-kMaxCoord, 0, 0}` equals `(2^15 << 4)`; level 15 at both limits is
    below `2^38`.
  - `TEST(ToBlockSortKey, LowestBlockAtLevelZero_IsZero)`: level 0 `{-kMaxCoord, -kMaxCoord, -kMaxCoord}`: all
    three passes are 0, and every pass is smaller than or equal to that of level 0 `{-kMaxCoord + 4, ...}`.
  - `TEST(ToBlockSortKey, CellsOfOneBlock_HaveEqualKeys)`; `TEST(ToBlockSortKey, InvalidId_SortsLast)`
    (`CellId{0}` gives `~0` in each pass).
  - `TEST(FindSortedCellAt, CoordOutsideRange_IsNoIndex)`: a list holding level 1 `{1, 0, 0}`; the coordinate
    `{-kMaxCoord - 1, 0, 0}` at level 0 gives `kNoIndex`, though packing it would give another cell's id.
  - `TEST(CheckOverlapAt, RepeatedCell_NamesBothCells)`; `TEST(CheckOverlapAt,
    CellInsideActiveBigCell_NamesBothCells)`; `TEST(CheckSiblingsAt, SevenOfEight_NamesTheMissing)`.
  - `TEST(CheckSiblingsAt, MissingCellBesideSplitSiblings_NamesTheMissing)`: big cell level 0 `{0,0,0}`; its
    eighth 0 (level 1 `{0,0,0}`) is missing; each of the other 7 level-1 cells is split into its 8 level-2
    cells (56 active cells). Over all positions the smallest failure key belongs to the first cell in fixed
    order that is eighth 0 of its level-1 cell; `error` is `kSiblingCellMissing`, `cell` is level 1 `{0,0,0}`,
    `other_cell` is level 0 `{0,0,0}`, `detail == (1 << 3) | 0`.
  - `TEST(CheckSiblingsAt, SplitSiblings_Pass)`: the same with the 8 level-2 cells of level 1 `{0,0,0}` added
    (64 cells): no failure at any position.
  - `TEST(CheckSizeStepAt, CornerTouchTwoLevelsApart_NamesDirectionZero)` (level 2 `{4,4,4}` with level 0
    `{0,0,0}`: direction `(dz, dy, dx) = (-1, -1, -1)` is 0).
  - `TEST(CheckSizeStepAt, CellAtLowCoordLimit_SeesNoCellBeyond)`: level 2 `{-kMaxCoord, 0, 0}` with level 0
    `{65536, 0, 0}` (524,287 level-2 widths away; the cell a packed id beyond the limit would point at): no
    failure.
  - `TEST(CheckSizeStepAt, CellAtHighCoordLimit_SeesTheBigCellBeside)`: level 2 `{kMaxCoord, 0, 0}` with level
    0 `{65536, 0, 0}`, which really is beside it: `kSizeStepRuleBroken`, direction `(0, 0, +1)`, detail 13.
  - `TEST(ToFailureKey, StepOutranksPosition)`: `ToFailureKey(2, 0, 0) > ToFailureKey(1, 2^40, 255)`.
  - `TEST(ComputeFailureCells, KeyOfEachStep_GivesTheChecksCells)`: for an overlap, a sibling and a size-step
    fixture above, the cells of the check at that position.
  - `TEST(ComputeSplitCellCountToAdd, RandomGrids_AddEachSplitCellOnce)`: the 200 random grids: the added cells
    are distinct and equal the brute-force set of split cells.
  - `TEST(ComputeBigCellAtLevel, TwoLevelsUp_HalvesTwice)`: level 3 `{-5, 6, 7}` at level 1 is `{-2, 1, 1}`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement.** Header comment: "the CPU and the GPU builders call these
  functions and nothing else for a table entry; that is why their tables are identical". `Read:`
  `<sec-parallel-build>`, `<proc-build-grid-in-parallel>`.
- [ ] **Step 4: Commit.** Subject: `Put the build's keys and cell checks in one shared header`. Private names
  only.

### Task 7b: Shared build formulas: neighbour tables, cell states and face piece kinds

**Kind:** numerics

**Files:**
- Modify: `grid_tables.h` (add the `GridTables` struct of design 3.2; the type only), `grid_build_steps.h`,
  `grid_build_steps_test.cc`, build registration

**Interfaces:**
- Consumes: Task 7a (`FindSortedBlock`), Task 3 (`CellState`, `FacePieceKind`, the index types).
- Produces `GridTables`, and in `grid_build_steps.h`:
  - `void FindBlockNeighbours(const CellBlockId* block_ids, std::uint32_t block_count, CellBlockIndex block,
    std::uint32_t* same_level_blocks, std::uint32_t* bigger_block, std::uint32_t* smaller_blocks)`: the 27, 1
    and 8 table entries of one block, by `FindSortedBlock`; a block coordinate outside the range has no block
    (`kNoIndex`), decided before any id is made. Layouts, the one place they are defined:
    `same_level_blocks[block * 27 + ((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1)]` (entry 13 is the block itself),
    `smaller_blocks[block * 8 + cell_eighth]`; face local place for axis X `(z * 4 + y) * 5 + x`, Y
    `(z * 5 + y) * 4 + x`, Z `(z * 4 + y) * 4 + x`, where the coordinate along the normal is the plane 0..4 and
    plane `p` is the low side of local cell `p`.
  - `CellState ComputeCoveredCellState(const GridTables& tables, CellPlaceIndex cell)`: for a place that is
    neither active nor split, `kInsideBiggerCell` when the covering place of `bigger_block` is `kActive`, else
    `kAbsent`. It reads only `kActive` marks, which no thread of that step changes.
  - `CellState ComputeCellStateBeside(const GridTables& tables, CellBlockIndex block, int local_x, int local_y,
    int local_z)`: the state of a cell place of `block`'s level given by local coordinates of which one may be
    `-1` or `4` (one cell outside the block). Inside the block: the stored state. Outside, when the same-level
    neighbour block on that side exists: the stored state of its place (local 3 or 0). When it does not exist,
    no cell of this level is there; the place is `kInsideBiggerCell` when an active cell one level bigger
    covers it, else `kAbsent`. The covering place is found through `bigger_block`: `block` is the low or the
    high half of its bigger block along the axis; if the place lies on the half's outer side, the cover is in
    the bigger block's same-size-level neighbour on that side (absent block: `kAbsent`), otherwise in the bigger
    block itself. A level-0 block has no bigger block: `kAbsent`.
  - `FacePieceKind ComputeFacePieceKind(const GridTables& tables, FacePlaceIndex face)`: design 5.1 item 5 for
    one face place. Plane `p` lies between local cells `p - 1` (low) and `p` (high) along the axis; both
    states come from `ComputeCellStateBeside`. Plane 4 is `kNotAPiece` when the same-level block on the high
    side exists (the face piece is stored at that block's plane 0). Otherwise: active and active, or active
    and `kInsideBiggerCell`: `kBetweenCells`; active on the high side and `kAbsent` on the low:
    `kGridEdgeLow`; active on the low side and `kAbsent` on the high: `kGridEdgeHigh`; anything else (active
    beside `kSplit`, or no active side): `kNotAPiece`.
- **Worked example for the header** (the two-size fixture: level 0 `{0,0,0}` and the 8 level-1 cells of
  `{1,0,0}`; block 0 of level 0, block 1 of level 1): plane 2 of block 1 along X lies between level-1 local
  cells 1 (inside the big cell: `kInsideBiggerCell`) and 2 (active), so its four places beside the active
  cells are `kBetweenCells`; plane 1 of block 0 along X lies between the active big cell and the split cell:
  `kNotAPiece`.

- [ ] **Step 1: Failing tests** (label `cpu`), tables built by hand:
  - `TEST(FindBlockNeighbours, TwoSizeFixture_LinksBiggerAndSmaller)`: block 1's `bigger_block` is 0; block
    0's `smaller_blocks[0]` is 1 and its other 7 are `kNoIndex`; entry 13 of each is itself.
  - `TEST(FindBlockNeighbours, BlockAtCoordLimit_HasNoBlockBeyond)`: the block of level 0 `{-kMaxCoord, 0, 0}`
    and the block of level 0 `{kMaxCoord, 0, 0}`: neither lists the other.
  - `TEST(ComputeCoveredCellState, PlaceInsideActiveBigCell_IsInsideBiggerCell)`.
  - `TEST(ComputeCellStateBeside, NoNeighbourBlockButBigCellBeside_IsInsideBiggerCell)`: level 0 `{-1,0,0}`
    active and the 8 level-1 cells of level 0 `{0,0,0}`: the place beside level-1 cell `{0,0,0}` on its low X
    side (local `-1` of the level-1 block, which has no level-1 neighbour block there) is
    `kInsideBiggerCell`; on its low Y side it is `kAbsent`.
  - `TEST(ComputeFacePieceKind, TwoSizeFixture_GivesFourQuarterFacePieces)`: the worked example: exactly 4
    `kBetweenCells` places on plane 2 of block 1 along X, and `kNotAPiece` on plane 1 of block 0.
  - `TEST(ComputeFacePieceKind, PlaneFourWithHighBlock_IsNotAPiece)`: level 0 `{3,0,0}` and `{4,0,0}`: plane 4
    of the first block is `kNotAPiece` and plane 0 of the second is `kBetweenCells`; with `{3,0,0}` alone,
    plane 4 is `kGridEdgeHigh`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement.** `Read:` `<sec-face-pieces>`, `<sec-cell-blocks>`,
  `<sec-parallel-build>`.
- [ ] **Step 4: Commit.** Subject: `Put the build's table rules in the shared header`. `GridTables` listed as
  a new public name.

### Task 8: Checking a cell list on the CPU: `CheckSizeStepRule`

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/grid_cell_checks.h` (private, CPU only), `grid_cell_checks.cc`,
  `grid_cell_checks_test.cc`
- Modify: `staggered_grid.h`, build registration

**Interfaces:**
- Consumes: Task 7a (`CheckCellId`, the three checks, the failure key, `ToCellSortKey`), Task 6b
  (`ToGridBuffersArrays`, `ToGridBuilderBuffersArrays`).
- Produces: `GridInputs` and both `CheckSizeStepRule` forms (design 3.3).

  ```cpp
  struct GridInputs {
    GridFrame frame;
    std::span<const CellId> active_cells;                // A list in CPU memory, any order.
    const Buffer<CellId>* active_cell_buffer = nullptr;  // Or a list in a buffer, any order.
    std::uint32_t active_cell_buffer_count = 0;          // Entries of *active_cell_buffer used.
    GridVersion version;                                 // 1 or more.
  };
  // Id checks, overlaps and the size-step rule, not siblings: cell resize checks lists while it splits.
  // Allocates one CPU copy of the ids (8 B per cell).
  std::expected<void, GridFailure> CheckSizeStepRule(GridFrame frame, std::span<const CellId> active_cells);
  // The same on buffers.run_on(), allocating nothing. The buffers hold no grid before or after; their contents
  // become unspecified. Also fails kTooManyCellBlocks (`cell` zero) when the list has more than
  // 64 * buffers.max_cell_blocks() cells.
  std::expected<void, GridFailure> CheckSizeStepRule(const GridInputs& inputs, GridBuffers& buffers);
  ```
  When `active_cell_buffer` is set it is the list and `active_cells` must be empty (`kBadInput`); otherwise
  `active_cells` is the list. `active_cell_buffer_count > size()` is `kBadInput`. Requires: the list is on the
  buffers' processor (for the span forms: in CPU memory), except an empty list; checked by `assert`.
- Produces, private, in `grid_cell_checks.h`:
  `enum class SiblingCheck : std::uint8_t { kSkip, kRequire };`
  `std::expected<void, GridFailure> CheckGridInputs(const GridInputs& inputs)` (frame: `origin` finite,
  `biggest_cell_width` finite and positive, `size_level_count` in 1..16; version not 0; the two-list and count
  rules; all `kBadInput`), `std::expected<void, GridFailure> CheckSortedCells(std::span<const CellId>
  sorted_cells, SiblingCheck sibling_check)` (steps 2 to 4 of `<proc-build-grid>`, each run to its end before
  the next, positions in order, by Task 7a's checks), and
  `std::expected<std::uint32_t, GridFailure> BuildSortedCellList(const GridInputs& inputs, GridBuffers&
  buffers, SiblingCheck sibling_check)`: the input checks, then steps 1 to 4, into the lent builder buffers' `cell ids` array, which
  then holds the list in fixed cell order; returns the cell count. (It is named `Build`, not `Check`, because
  it writes the sorted list.) CPU here; Task 10 adds the GPU path.
  `BuildGrid` (Task 9a) starts from it.
- **Failure context**, as Task 7a's checks give it: `kCellsOverlap` names the smaller (or repeated) cell in
  `cell` and the bigger (or earlier) in `other_cell`; `kSiblingCellMissing` names the first missing small cell
  (eighth order) in `cell` and its big cell in `other_cell`; `kSizeStepRuleBroken` names the smaller cell in
  `cell` and the bigger in `other_cell`; `kCoordOutOfRange` and `kBadInput` from an id name the first such id
  in input order. **Order, the same on both processors:** `CheckGridInputs` (frame, version, list shape), then
  the list length against `64 * buffers.max_cell_blocks()` (`kTooManyCellBlocks`, `cell` zero), and only then
  step 1 on the ids. These input checks come before every keyed check and never take part in the key minimum.
  The forms without buffers have no length check. Step 1 runs over the list in input order before the sort. The CPU sorts with `std::sort` and
  the comparator "pass 2, then pass 1, then pass 0 of `ToCellSortKey`".

- [ ] **Step 1: Failing tests** (label `cpu`), frame origin 0, width 1 m, 3 levels:
  - `TEST(CheckSizeStepRule, CornerTouchTwoLevelsApart_IsSizeStepRuleBroken)`: level 0 `{0,0,0}` and level 2
    `{4,4,4}`: `cell` = the level-2 id, `other_cell` = the level-0 id.
    `TEST(CheckSizeStepRule, EdgeTouchTwoLevelsApart_IsSizeStepRuleBroken)` (`{4,4,0}`) and
    `TEST(CheckSizeStepRule, FaceTouchTwoLevelsApart_IsSizeStepRuleBroken)` (`{4,0,0}`).
  - `TEST(CheckSizeStepRule, OneLevelApart_Passes)`: level 0 `{0,0,0}` and level 1 `{2,0,0}`.
  - `TEST(CheckSizeStepRule, SevenOfEightSmallCells_Passes)` (no sibling check).
  - `TEST(CheckSizeStepRule, BadIdAfterRepeat_IsBadInputOfFirstBadId)`: input `{a, a, bad1, bad2}` (two ids
    without bit 63): `kBadInput` naming `bad1` (step 1 outranks step 2; input order within step 1).
  - `TEST(CheckSizeStepRule, CoordBeyondLimit_IsCoordOutOfRange)`: coordinate `kMaxCoord + 1`.
  - `TEST(GridInputs, TwoLists_IsBadInput)`; `TEST(GridInputs, BufferCountAboveSize_IsBadInput)`;
    `TEST(GridInputs, VersionZero_IsBadInput)` (each through `CheckSizeStepRule(inputs, buffers)`).
  - `TEST(CheckSizeStepRule, IntoBuffers_EqualsSpanForm)`: the three `...TwoLevelsApart` lists and the 200
    random grids, with CPU buffers for 4,096 blocks.
  - `TEST(CheckSizeStepRule, MoreCellsThanPlaces_IsTooManyCellBlocks)`: buffers for 1 block, 65 cells.
  - `TEST(CheckSizeStepRule, TooLongListWithBadId_IsTooManyCellBlocks)`: buffers for 1 block, 65 cells of
    which the first has bit 63 clear: `kTooManyCellBlocks` with `cell` and `other_cell` zero, not `kBadInput`;
    the same list with a version of 0 is `kBadInput` (the frame and version come first).
  - `TEST(BuildSortedCellList, ShuffledList_LeavesFixedCellOrder)`: the 200 random grids: the `cell ids`
    array then equals the list sorted with `ComesBefore`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** with `// Step N:` comments from `<proc-build-grid>`.
- [ ] **Step 4: Commit.** Subject: `Check a list of active cells before building a grid`. New names:
  `GridInputs`, `CheckSizeStepRule` (two forms).

### Task 8b: Counting a list without buffers: `ComputeGridCounts(inputs)`

**Kind:** numerics

**Files:**
- Modify: `staggered_grid.h`, `grid_cell_checks.h`, `grid_cell_checks.cc`, build registration
- Create: `native/fluidity/grid/grid_cell_count_test.cc`

**Interfaces:**
- Consumes: Task 8 (`CheckGridInputs`, `CheckSortedCells`), Task 7a (`FindSortedCellAt`,
  `ComputeSplitCellCountToAdd`, `ComputeBigCellAtLevel`, `ToBlockSortKey`), `GeometricGrid` (Task 5, tests).
- Produces: `std::expected<GridCounts, GridFailure> ComputeGridCounts(const GridInputs& inputs)` (design 3.3):
  counts without building. Same checks as `BuildGrid` (siblings required) except the block limit, so the
  planner can count a scene before any buffers exist. Requires a CPU list.
- **Why it does not build tables.** A build needs buffers sized by the block count, which is the number asked
  for; sized for the worst case (one block per cell) they would cost about 1.1 kB per active cell. So this form
  counts on the sorted ids alone. It is a second statement of the face rule. The header says so and names the
  tests that hold the two statements together (`ComputeGridCounts, RandomGrids_EqualGeometricChecker` and
  `ComputeGridCounts, RandomGrids_EqualBuiltCounts`), by their names only.
- **Method** (one `std::vector<CellId>` reserved once for `n + n / 7` ids: every split cell has 8 small cells,
  so there are at most `n / 7` of them; 8 B per active or split cell, at most 9.2 B per active cell):
  1. Copy the `n` ids; steps 1 to 4 as Task 8, with `SiblingCheck::kRequire`. The copy is now in fixed cell
     order.
  2. `active_cells_per_size_level` and `active_cells`: one pass.
  3. `face_pieces`: one pass over the active cells, summing the private
     `int ComputeFacePieceCountOfCell(std::span<const CellId> sorted_cells, std::uint32_t position)`. For a
     cell `c` of level `p` and each axis, let `N` be the coordinate of the same-level place beside `c`. Every
     lookup is `FindSortedCellAt` (Task 7a), on coordinates found by arithmetic, so a coordinate outside
     `-kMaxCoord..kMaxCoord` at its level holds no cell and no id is made for it. The place is *active* when
     `N` is found at level `p`; *inside a bigger cell* when `floor(N / 2)` is found at level `p - 1`; *split*
     when the small cell of `N` that touches `c` by a face is found at level `p + 1` (after step 4 that small
     cell is active); else *absent*. Low side: 1 face piece unless the place is split. High side: 1 face piece
     when the place is inside a bigger cell or absent, else 0 (an active place counts the shared face piece on
     its own low side; a split place's small cells count theirs). At most 18 binary searches per cell.
  4. `cell_blocks`: append the split cells (`ComputeSplitCellCountToAdd`, `ComputeBigCellAtLevel`), sort all
     ids with the comparator "pass 2, then 1, then 0 of `ToBlockSortKey`", and count the runs of equal
     `ToCellBlockId`. A count above `2^32 - 1` cannot occur (`n` fits `std::uint32_t`).
  Cost: two sorts and four linear passes, `O(n log n)`; one allocation.

- [ ] **Step 1: Failing tests** (label `cpu`), frame origin 0, width 1 m, 3 levels:
  - `TEST(ComputeGridCounts, OneBigCell_HasSixFacePieces)`: 1 block, 6 face pieces, `active_cells == 1`.
  - `TEST(ComputeGridCounts, OneSplitCell_CountsBlocksOfBothLevels)`: level 0 `{0,0,0}` and the 8 level-1 cells
    of `{1,0,0}`: `cell_blocks == 2`, `active_cells_per_size_level = {1, 8}`, `active_cells == 9`,
    `face_pieces == 41` (5 whole faces of the big cell, 24 outer quarter faces of the small cells of which 4
    are shared with the big cell, 12 between the small cells).
  - `TEST(ComputeGridCounts, TwoCellsAcrossABlockSide_ShareOneFacePiece)`: level 0 `{3,0,0}` and `{4,0,0}`: 2
    blocks, 11 face pieces.
  - `TEST(ComputeGridCounts, CellsAtBothCoordLimits_CountTheirEdgeFacePieces)`: 2 levels: level 0 `{-kMaxCoord,
    0, 0}`, level 0 `{kMaxCoord, 0, 0}` and the 8 level-1 cells of level 0 `{0,0,0}`: `face_pieces == 48` (6 + 6
    + 36) and `cell_blocks == 4`. A count through packed ids beyond the limit would find level-1 cell `{1,0,0}`
    beside the first cell, and level-1 cell `{0,0,0}` beside the second, and give 46.
  - `TEST(ComputeGridCounts, RandomGrids_EqualGeometricChecker)`: the 200 random grids: `face_pieces ==
    GeometricGrid::FacePieces().size()`; `cell_blocks` equals the size of a `std::set` of `ToCellBlockId` over
    every active cell and every bigger cell that contains one.
  - `TEST(ComputeGridCounts, CellInsideActiveBigCell_IsCellsOverlap)`; `..., SameCellTwice_IsCellsOverlap`;
    `..., CoordBeyondLimit_IsCoordOutOfRange`; `..., LoneSmallCell_IsSiblingCellMissing` (7 of 8, `cell` = the
    missing one).
  - `TEST(ComputeGridCounts, EmptyList_GivesZeroCounts)`.
  - `TEST(ComputeGridCounts, CpuBufferList_GivesSameCountsAsSpan)`: a CPU `Buffer<CellId>` holding the list of
    `OneSplitCell`.
  - `TEST(ComputeGridCounts, MoreBlocksThanAnyTestRoom_CountsThemAll)`: the 4,097 level-0 cells `{4 i, 0, 0}`,
    `i = 0..4096`: `cell_blocks == 4097`, `face_pieces == 6 * 4097`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** with `// Step N:` comments from `<proc-count-grid>`. `Read:`
  `<proc-count-grid>`, `<sec-face-pieces>`, `<sec-cell-blocks>`.
- [ ] **Step 4: Commit.** Subject: `Count the blocks and face pieces of a cell list without building it`. New
  name: `ComputeGridCounts`.

### Task 9a: `BuildGrid` on the CPU, `StaggeredGrid` and `TakeBuffers`

**Kind:** numerics

**Files:**
- Modify: `staggered_grid.h`, `staggered_grid_cpu.cc`, `grid_buffers_internal.h` (`BuildGridFromBuffers`),
  `grid_test_support.h`, `grid_test_support.cc`, build registration
- Create: `native/fluidity/grid/staggered_grid_test.cc`, `allocation_counter_test_support.h`,
  `allocation_counter_test_support.cc`, `staggered_grid_no_gpu_test.cc`
- `allocation_counter_test_support.cc` is listed in the `SOURCES` of every test program that counts allocations
  (this task's, and those of Tasks 9b, 9c, 11b, 21, 22a, 22b and 24), never in a library: a replaced `operator
  new` must be part of the program itself, and a program that does not count must not get it.

**Interfaces:**
- Consumes: Tasks 7a and 7b (types and formulas), Task 8 (`BuildSortedCellList`), Task 8b (`ComputeGridCounts`).
- Produces: `StaggeredGrid` with every accessor of design 3.3 (`all_blocks()` returns `{0,
  counts().cell_blocks}`), `std::expected<StaggeredGrid, GridFailure> BuildGrid(const GridInputs& inputs,
  GridBuffers& buffers)`, `GridBuffers TakeBuffers(StaggeredGrid&& grid)`. On success the buffers move into
  the grid and `buffers` is left moved-from; on failure `buffers` keeps its arrays and no grid is returned.
  `tables(RunOn::kGpu)` on a CPU grid returns version 0 and null arrays.
- Produces (test-only, `grid_test_support.h`):

  ```cpp
  // A grid with its own memory, for tests. run_memory is declared first, so it is destroyed last, after the
  // buffers that are memory pieces of it. The builder buffers are held by pointer so that moving a TestGrid
  // never moves the lent GridBuilderBuffers. Move construction only: a RunMemory must not be assigned over
  // while its pieces are alive.
  struct TestGrid {
    RunMemory run_memory;
    Buffer<CellId> cell_list;  // The list on the grid's processor; empty for a CPU grid.
    std::unique_ptr<GridBuilderBuffers> builder_buffers;
    StaggeredGrid grid;
  };
  // Reserves exactly the estimate for one grid of the room (for a GPU grid, plus one memory piece for the
  // list), then BuildGridBuilderBuffers, BuildGridBuffers and BuildGrid on settings.run_on. The list is
  // inputs.active_cells, a span in CPU memory; inputs.active_cell_buffer must be null. For a GPU grid the span
  // is counted on the CPU and copied to cell_list, which the build then reads. With settings.max_cell_blocks 0
  // (the default) the room is ComputeGridCounts(inputs).cell_blocks, at least 1. Fails as those functions do.
  std::expected<TestGrid, GridFailure> BuildTestGrid(const GridInputs& inputs, const GridSettings& settings = {});
  ```
  `TestGrid` declares its move constructor and deletes move assignment.
- Produces, private, in `grid_buffers_internal.h`: `StaggeredGrid BuildGridFromBuffers(GridBuffers&& buffers,
  GridFrame frame, GridVersion version, const GridCounts& counts)`: the one place a `StaggeredGrid` is made from
  filled buffers. `BuildGrid`, `ReadGrid` (Task 9c) and `BuildGridPiece` (Task 24) call it.
- **The CPU build, steps 5 to 8 of `<proc-build-grid>`.** After `BuildSortedCellList` the `cell ids` array
  holds the `n` active cells. Step 5: append the split cells at `[n, n + s)`; `n + s` above `64 *
  max_cell_blocks` fails. Sort each of the two parts with `std::sort` and the comparator "pass 2, then 1, then
  0 of `ToBlockSortKey`", then walk both in key order and write each new `CellBlockId` to `block_ids`; more
  than `max_cell_blocks` fails. The keys are the fixed order, so `block_ids` needs no sort of its own. Step 6:
  `FindBlockNeighbours` per block. Step 7: every place `kAbsent`; each active cell `kActive` and each split
  cell `kSplit` at its block (`FindSortedBlock`) and local place; then `ComputeCoveredCellState` for the
  places still `kAbsent`. Step 8: `ComputeFacePieceKind` per face place, counting the face pieces. The working
  order of the CPU is its own; every table entry is written through the functions of Tasks 7a and 7b and in no other way. No
  `std::stable_sort` (it may allocate).
- **A block that cannot be found.** In step 7 `FindSortedBlock` of a cell's own block among the ids step 5 just
  wrote cannot return `kNoIndex`; if it does, the build returns `kInternalFault` (checked in every build).
- **The block limit on both processors:** `kTooManyCellBlocks` with `cell` zero. The first cell beyond the
  limit is not named: the GPU cannot name it without working memory beyond the room.
- `BuildGrid` with `buffers.run_on() == kGpu` returns `kRunOnCallFailed` until Task 11b adds the GPU path.
- `allocation_counter_test_support.cc` replaces every global `operator new`/`delete` form (plain, nothrow,
  `std::align_val_t`) and counts calls. `Buffer` allocates through `::operator new(size, std::align_val_t,
  std::nothrow)` (`base/buffer_cpu.cc`), so the counter sees CPU buffers.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(BuildGrid, LoneSmallCell_IsSiblingCellMissing)`: 7 of the 8 level-1 cells of level-0 `{0,0,0}`:
    `cell` = the missing one.
  - `TEST(BuildGrid, AnyInputOrder_GivesSameTables)`: the 200 random grids, each also built from the list
    sorted with `ComesBefore`: the used part of every table array is byte-identical.
  - `TEST(ComputeGridCounts, RandomGrids_EqualBuiltCounts)`: the 200 grids: `ComputeGridCounts(inputs)` equals
    `grid.counts()` field by field.
  - `TEST(BuildGrid, FacePieceKinds_MatchGeometricChecker)`: the 200 grids: every geometric face piece is
    exactly one place whose kind is `kBetweenCells`, `kGridEdgeLow` or `kGridEdgeHigh` as the checker says; no
    other place is a face piece.
  - `TEST(BuildGrid, CellStates_MatchGeometricChecker)`: `kActive` exactly at the list's cells, `kSplit` exactly
    at the bigger cells that contain one, `kInsideBiggerCell` exactly at places an active bigger cell covers.
  - `TEST(BuildGrid, AddedSplitCells_AreThePlacesMarkedSplit)`: the 200 grids: the cells
    `ComputeSplitCellCountToAdd` and `ComputeBigCellAtLevel` add are exactly the `kSplit` places.
  - `TEST(BuildGrid, AboveBuffersRoom_IsTooManyCellBlocks)`: buffers for 8 blocks; 9 level-0 cells `{4 i, 0,
    0}` fail with `cell` zero and return no grid; the same buffers then build the first 8.
  - `TEST(BuildGrid, MoreCellsThanPlaces_IsTooManyCellBlocks)`: buffers for 8 blocks, `64 * 8 + 1` cells.
  - `TEST(BuildGrid, TooLongListWithBadId_IsTooManyCellBlocks)`: the same list with one id whose bit 63 is
    clear and one coordinate of `kMaxCoord + 1`: `kTooManyCellBlocks`, `cell` zero.
  - `TEST(BuildGrid, SplitCellsOverflowThePlaces_IsTooManyCellBlocks)`: buffers for 1 block; the 64 level-1
    cells of the 8 level-0 cells `{0..1, 0..1, 0..1}`: the 64 cells fit the places, the 8 split cells they
    add do not (72 above 64): `kTooManyCellBlocks`, `cell` zero. This is the room failure found after the
    split cells are counted, not at the first check.
  - `TEST(BuildGrid, IntoBuffers_AllocatesNothing)`: buffers for 4,096 blocks; the 100 random grids built and
    their buffers taken back in turn: the counter stays 0 after `BuildGridBuffers`, the addresses in
    `ToGridBuffersArrays` do not change, and each grid's used tables equal those of a `BuildTestGrid` grid built
    before the counted part.
  - `TEST(BuildGrid, EmptyList_GivesEmptyGrid)`: version 1, zero blocks, `all_blocks()` is `{0, 0}`.
  - `TEST(BuildTestGrid, DefaultSettings_SizedToTheList)`: a 9-block list gives `max_cell_blocks() == 9`.
  - `TEST(StaggeredGrid, CpuGrid_HasNoGpuTables)`: `tables(RunOn::kGpu).version.value == 0`, null arrays.
  Label `no_gpu`: `TEST(BuildTestGrid, OnGpuWithoutAGpu_IsRunOnCallFailed)`: settings `run_on = kGpu`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** with `// Step N:` comments. `Read:` `<proc-build-grid>`,
  `<sec-cell-blocks>`, `<sec-face-pieces>`.
- [ ] **Step 4: Commit.** Subject: `Build a grid version into buffers without allocating`. New names:
  `StaggeredGrid`, `BuildGrid`, `TakeBuffers`.

### Task 9b: `ComputeGridCounts` on buffers, and builder buffers shared by two grids

**Kind:** mechanical

**Files:**
- Modify: `staggered_grid.h`, `staggered_grid_cpu.cc`, build registration
- Create: `native/fluidity/grid/grid_counts_on_buffers_test.cc` (its `SOURCES` list the allocation counter)

**Interfaces:**
- Consumes: Task 9a (the CPU build, the allocation counter, `BuildTestGrid`), Task 8b.
- Produces: `std::expected<GridCounts, GridFailure> ComputeGridCounts(const GridInputs& inputs, GridBuffers&
  buffers)` (design 3.3): the build of Task 9a run into `buffers` for its counts, allocating nothing. The
  buffers hold no grid before or after; their contents become unspecified. Unlike the one-argument form it
  fails `kTooManyCellBlocks` (`cell` zero) when the grid does not fit the buffers. On GPU buffers it returns
  `kRunOnCallFailed` until Task 11b.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ComputeGridCounts, IntoBuffers_EqualsSpanForm)`: the 200 random grids into CPU buffers for 4,096
    blocks; the 4,097-cell list of Task 8b gives `kTooManyCellBlocks` with `cell` zero while the one-argument
    form gives `cell_blocks == 4097`.
  - `TEST(ComputeGridCounts, IntoBuffers_AllocatesNothing)`: the 100 random grids through
    `ComputeGridCounts(inputs, buffers)` and `CheckSizeStepRule(inputs, buffers)`: the counter stays 0.
  - `TEST(ComputeGridCounts, SpanForm_AllocatesOneArray)`: the counter rises by exactly 1 per call.
  - `TEST(BuildGrid, SharedBuilderBuffers_OtherLiveGridUnchanged)`: one `GridBuilderBuffers` for 4,096 blocks
    lent to two buffers; grid A built into the first; then the 100 random grids built into the second and
    taken back in turn, with `ComputeGridCounts` and `CheckSizeStepRule` on it between: A's five tables and
    counts are byte-identical before and after, and each B equals B built by `BuildTestGrid`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement.**
- [ ] **Step 4: Commit.** Subject: `Count a grid on buffers without allocating`. New name: the buffers form of
  `ComputeGridCounts`.

### Task 9c: `ReadGrid`

**Kind:** mechanical

**Files:**
- Modify: `staggered_grid.h`, `staggered_grid_cpu.cc`, build registration
- Create: `native/fluidity/grid/read_grid_test.cc` (its `SOURCES` list the allocation counter)

**Interfaces:**
- Consumes: Task 9a; `CopyBuffer`, slice `ReadBuffer` (`base/buffer.h`), `ToBufferError`.
- Produces (design 3.3):

  ```cpp
  // Copies the used part of grid's five tables into cpu_buffers (456 B per block) and returns a grid on the CPU
  // with the same version and counts. The one explicit copy of a GPU grid: for tests, tools and a caller that
  // accepts the transfer. Allocates nothing.
  // Requires: cpu_buffers.run_on() == RunOn::kCpu, checked by assert in debug builds.
  // Fails: kTooManyCellBlocks  the room is below grid.counts().cell_blocks
  //        kRunOnCallFailed    a copy from the GPU failed
  //        On failure cpu_buffers keeps its arrays.
  std::expected<StaggeredGrid, GridFailure> ReadGrid(const StaggeredGrid& grid, GridBuffers& cpu_buffers);
  ```
  No GPU half: a GPU grid's arrays cross through `ReadBuffer` of `base/`, one call per table.

- [ ] **Step 1: Failing tests** (label `cpu`): `TEST(ReadGrid, CpuGrid_CopiesIntoOtherCpuBuffers)` (the 100
  random grids: same used tables, version and counts); `TEST(ReadGrid, BuffersTooSmall_IsTooManyCellBlocks)`
  (9 blocks into room for 8; the buffers then still build an 8-block grid);
  `TEST(ReadGrid, AnyGrid_AllocatesNothing)`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement.** The GPU-to-CPU path is compiled, not run; Task 11b's `gpu`
  tests are the first to use it.
- [ ] **Step 4: Commit.** Subject: `Copy a grid's tables to the CPU on request`. New name: `ReadGrid`.

### Task 10: Checking a cell list on the GPU

**Kind:** device

**Files:**
- Create: `native/fluidity/grid/staggered_grid_gpu.h` (private: the GPU half's functions on plain pointers,
  counts and a `GridBuildStatus*`), `staggered_grid_gpu.cu` (private), `grid_cell_checks_gpu_test.cc` (label
  `gpu`)
- Modify: `grid_cell_checks.cc` (dispatch on `buffers.run_on()`; the GPU sequence under
  `#if defined(FLUIDITY_HAS_GPU)`), `grid_build_steps.h`, `grid_build_steps_test.cc`, `grid_test_support.h`,
  `grid_test_support.cc`, `grid_test_support_test.cc`, build registration
  (`fluidity_add_gpu_sources(staggered_grid_gpu.cu)`)

**Interfaces:**
- Consumes: Task 7a (`ToCellSortKey`, `CheckCellId`, the three checks, the failure key, `ComputeFailureCells`),
  Task 8 (`CheckGridInputs`, `CheckSortedCells` as oracle, `SiblingCheck`, `BuildSortedCellList`), Task 6b
  (the array accessors), `GridBuildStatus`; `BuildStableOrder` with the `OrderBuffers` of the lent
  `GridBuilderBuffers`, `ReadBuffer`, `ToBufferError`, `RunOnCallStatus`, `ComputeLaunchSize`,
  `ApplyGpuFunction`, `CheckGpuWork`, `ComputeGridStride`; test support `ReadAll`, `WriteResultDigest`.
- Produces: the GPU path of `BuildSortedCellList`, and so of `CheckSizeStepRule(inputs, buffers)`. The CPU
  half runs P1, then calls one GPU-half function per step (`ApplyBuildStatusResetOnGpu`, `CheckCellIdsOnGpu`,
  `WriteCellSortKeysOnGpu`, `CopySortedCellsOnGpu`, `CheckSortedCellsOnGpu`, `WriteFailureCellsOnGpu`), each
  returning `RunOnCallStatus`, with `BuildStableOrder` between them. A status other than `kDone`, and a
  `BufferError` from `BuildStableOrder` or `ReadBuffer`, go through `ToBufferError` and the module's one mapping
  into `GridFailure`. After `kRunOnCallFailed` the buffers' contents are unspecified, they hold no grid, and
  the run stops. A status record that contradicts itself (a failure key with error `kNone`, no key with an
  error, or a key whose step is not 1 to 6) is `kInternalFault`, as is a size error from `BuildStableOrder` or
  `ReadBuffer` (through the mapping).
- Shared per-item functions added to `grid_build_steps.h`:
  `std::uint64_t ComputeCellIdFailureKey(GridFrame frame, const CellId* input, std::uint64_t index)` (P2;
  `kNoFailureKey` when the id is good) and `std::uint64_t ComputeSortedCellFailureKey(const CellId*
  sorted_cells, std::uint32_t count, std::uint32_t position, int step)` (P4 to P6: the Task 7a check of that
  step at that position).
- Test support added (`grid_test_support`): `std::vector<CellId> BuildMutatedGridCells(std::uint64_t seed)`:
  the shuffled list of `BuildRandomGridCells(seed, 4, 3)` with 1 to 4 mutations drawn from seed `seed + 7000`
  (repeat a cell; drop a cell; add a cell's big cell; add a cell two levels smaller beside a big cell, with
  its siblings; drop every active cell inside one small cell of a split cell, so a whole sibling is missing;
  clear bit 63 of one id; set a coordinate to `kMaxCoord + 1`). "The 500 mutated lists" are
  seeds 1..500.
- **The parallel form of steps 1 to 4:**
  - In the steps below `n` is the list's cell count (`active_count` in the per-item functions).
  - P1 (CPU, nothing launched yet), in the order of Task 8: `CheckGridInputs`; then a list of more than `64 * max_cell_blocks` cells is
    `kTooManyCellBlocks`, `cell` zero. Then a one-thread GPU function resets the status (`failure_key =
    kNoFailureKey`, counts zero): nothing is written from the CPU.
  - P2, one thread per input index `i`: `atomicMin` of `ComputeCellIdFailureKey` (step 1, position `i`).
  - P3, sort into fixed cell order: for pass 0, 1, 2 a GPU function writes `ToCellSortKey(input[i], pass)` into
    `sort keys`, then `BuildStableOrder` (pass 0 with no previous order, then the previous pass's order),
    alternating `order` and `second order`; a GPU function then gathers `cell ids[j] = input[order[j]]`. The
    keys of an id that failed P2 are still well-defined numbers; the sort runs and its result is never used.
  - P4, P5 (only with `kRequire`), P6, one thread per sorted position `j`: each first reads `failure_key` and
    returns when an earlier step failed; else `atomicMin` of `ComputeSortedCellFailureKey`. So no step runs on
    data an earlier step found bad.
  - Status (part of the check: `BuildSortedCellList` is not done before it): a one-thread GPU function fills `error`, `cell`, `other_cell` from the smallest key with
    `ComputeFailureCells` (step 1: the input id at that index). The CPU reads `GridBuildStatus` with one
    `ReadBuffer`: the only transfer of the operation.

- [ ] **Step 1: Tests.** Label `cpu`:
  - `TEST(BuildMutatedGridCells, FiveHundredLists_BreakEveryRule)` (`grid_test_support_test.cc`): over the 500
    lists, `CheckSortedCells` or step 1 reports each of `kBadInput`, `kCoordOutOfRange`, `kCellsOverlap`,
    `kSiblingCellMissing` and `kSizeStepRuleBroken` at least once.
  - `TEST(BuildSortedCellList, EveryThreadBlockAndThread_SameFirstFailureAsCpu)`
    (`grid_build_steps_test.cc`): P2 to P6 replayed on CPU buffers as Global Constraints "Replay tests" says;
    on the 200 random grids, the three `...TwoLevelsApart` lists of Task 8 and the 500 mutated lists, for both
    `SiblingCheck` values: the sorted list and the `GridFailure` equal the CPU path's exactly.
  Label `gpu` (lists are put on the GPU with `WriteBuffer` in the test's setup only; each test passes the
  status records it reads to `WriteResultDigest`):
  - `TEST(CheckSizeStepRule, OnGpu_EqualsCpu)`: the three `...TwoLevelsApart` lists, `OneLevelApart`, the empty
    list and the 200 random grids: the result is identical to `CheckSizeStepRule(frame, span)`.
  - `TEST(BuildSortedCellList, StatusThatContradictsItself_IsInternalFault)` (label `cpu`, in the replay file:
    the CPU half's reading of a status record is one private function, `ToGridFailure(const GridBuildStatus&)`;
    a record with a key and `kNone`, one with an error and no key, and one with step 9 each give
    `kInternalFault`).
  - `TEST(CheckSizeStepRule, OnGpuTooLongListWithBadId_SameFailureAsCpu)`: Task 8's list of that test, in GPU
    buffers for 1 block: `kTooManyCellBlocks`, `cell` and `other_cell` zero, as on the CPU; nothing is launched.
  - `TEST(CheckSizeStepRule, OnGpuSeveralFailures_SameFirstFailureAsCpu)`: the 500 mutated lists, both
    `SiblingCheck` values through `BuildSortedCellList`: `GridFailure` (error, cell, other_cell) equals the
    CPU path's exactly.
  - `TEST(CheckSizeStepRule, OnGpuTwice_GivesIdenticalFailure)`: the 500 mutated lists, twice.
  - `TEST(CheckSizeStepRule, OnGpu_AllocatesNothing)`: 100 calls; `ReadMemoryAvailable(RunOn::kGpu)->free_bytes`
    unchanged after the first.
- [ ] **Step 2: Run** the gate. Expected: `PASS cuda` (compiled and linked; the container runs the `cpu` tests,
  the replay among them).
- [ ] **Step 3: Commit.** Subject: `Check a cell list on the GPU in the CPU's order`. Body: `GPU: compiled, not
  run`; the new private names; the four `gpu` tests above by name, for the hardware run.

### Task 11a: Split cells and block ids in parallel (P7, P8)

**Kind:** device

**Files:**
- Modify: `grid_build_steps.h`, `grid_build_steps_test.cc`, `staggered_grid_gpu.h`, `staggered_grid_gpu.cu`

**Interfaces:**
- Consumes: Task 10 (the sorted list in `cell ids`, the GPU-half pattern), Task 7a
  (`ComputeSplitCellCountToAdd`, `ComputeBigCellAtLevel`, `ToBlockSortKey`), Task 9a (the CPU build as oracle);
  `BuildStableOrder`, `ComputeExclusivePrefixSum`.
- Shared per-item functions added to `grid_build_steps.h` (`n` = active cells):
  - `void ApplySplitCellWritesAt(CellId* cell_ids, const std::uint32_t* sums, std::uint32_t active_count,
    std::uint32_t position)`: writes the split cells that sorted cell `position` adds at
    `cell_ids[n + sums[position] + k]`.
  - `std::uint64_t ComputeRoomFailureKey(std::uint64_t needed, std::uint64_t room, int step)`: the key of step
    5 or 6 at position 0 when `needed > room`, else `kNoFailureKey`.
  - `std::uint32_t ComputeSortCount(std::uint32_t active_count, std::uint32_t max_cell_blocks)`: `min(n + n /
    7, 64 * max_cell_blocks)`, computed in `std::uint64_t` (the sum can pass `2^32`); the result fits 32 bits
    because `64 * kMaxCellBlocksLimit = 2^32 - 64`.
  - `void ApplySortTailFillAt(CellId* cell_ids, std::uint32_t used_count, std::uint32_t place)`: writes
    `CellId{0}` when `place >= used_count`.
  - `std::uint32_t ComputeBlockStartMark(const CellId* cell_ids, const std::uint32_t* order, std::uint32_t
    position)`: 1 when `cell_ids[order[position]]` is valid and is the first sorted entry of its block
    (position 0, or another `ToCellBlockId` than the entry before); else 0.
  - `std::uint32_t ComputeBlockIndexOfSortedEntry(const CellId* cell_ids, const std::uint32_t* order, const
    std::uint32_t* sums, std::uint32_t position)`: `sums[position] + ComputeBlockStartMark(...) - 1`. After the
    exclusive prefix sum the first entry of a run holds the block index and the others hold it plus one; the
    mark, a function of the sorted ids alone, tells them apart.
  - `void ApplyBlockIdWriteAt(CellBlockId* block_ids, const CellId* cell_ids, const std::uint32_t* order, const
    std::uint32_t* sums, std::uint32_t position)`: a marked entry writes its `CellBlockId` at `sums[position]`.
- Produces: the GPU-half functions of P7 and P8 (`WriteSplitCellCountsOnGpu`, `CheckRoomOnGpu`,
  `WriteSplitCellsOnGpu`, `WriteSortTailOnGpu`, `WriteBlockSortKeysOnGpu`, `WriteBlockStartMarksOnGpu`,
  `WriteBlockIdsOnGpu`), compiled here and sequenced by Task 11b.
- **The parallel form of step 5.** Every GPU function loops grid-stride over a count it reads from
  `GridBuildStatus`, so no count comes back mid-build. When an earlier step failed it writes nothing, except in the two steps
  whose output a prefix sum reads, P7 (a) and the marks of P8: there every thread writes 0. The prefix sums
  then add zeros, never words of an earlier launch (after P3 `second order` holds a permutation of `0..n-1`,
  whose sum passes `2^32 - 1` from 92,683 cells, and the base call would fail with a size error).
  - P7 add split cells. (a) One thread per sorted active cell writes `ComputeSplitCellCountToAdd` into `second
    order`, or 0 when an earlier step failed; `ComputeExclusivePrefixSum` in place over `n` entries (the total lands at index `n`). (b) One
    thread reads the total into `split_cells` and takes `ComputeRoomFailureKey(n + split_cells, 64 *
    max_cell_blocks, 5)`. (c) One thread per active cell: `ApplySplitCellWritesAt`. (d) **The tail fill:** one
    thread per place below `sort_count = ComputeSortCount(n, max_cell_blocks)`: `ApplySortTailFillAt` with
    `used_count = n + split_cells`. `sort_count` is a CPU number that bounds active plus split cells (every
    split cell has 8 small cells, so `split_cells <= n / 7`); the sort of P8 runs over it because the exact
    count stays on the GPU. Without (d) the places between hold cells of the build before, and they would
    become blocks.
  - P8 sort cells by block and number the blocks. Three `ToBlockSortKey` passes over `sort_count` places with
    `BuildStableOrder` (final order in `order`; `CellId{0}` has the key `~0` and sorts last). No gather:
    sorted entry `j` is `cell_ids[order[j]]`, and it comes from the active list exactly when `order[j] < n`.
    One thread per sorted place writes `ComputeBlockStartMark` into `second order`, or 0 when an earlier step
    failed (step 5 at P7 included);
    `ComputeExclusivePrefixSum` in place; one thread reads the total into `cell_blocks` and takes
    `ComputeRoomFailureKey(cell_blocks, max_cell_blocks, 6)`; one thread per sorted place:
    `ApplyBlockIdWriteAt`. The sort keys are the fixed block order, so `block_ids` comes out in fixed order.

- [ ] **Step 1: Failing tests** (label `cpu`, `grid_build_steps_test.cc`):
  - `TEST(ComputeBlockStartMark, SixSortedCells_MarkEachRunOnce)`: six cells in three blocks in the pattern
    `[A, A, B, C, C, C]`: marks `[1, 0, 1, 1, 0, 0]`, sums `[0, 1, 1, 2, 3, 3]`,
    `ComputeBlockIndexOfSortedEntry` `[0, 0, 1, 2, 2, 2]`; a seventh entry `CellId{0}` has mark 0.
  - `TEST(ApplySplitCellWritesAt, RandomGrids_AppendExactlyTheSplitCells)`: the 200 random grids: places
    `[n, n + split_cells)` hold each split cell once, in any thread order.
  - `TEST(ComputeRoomFailureKey, OneAboveRoom_IsTheStepsKey)`: `(513, 512, 5)` gives `ToFailureKey(5, 0, 0)`;
    `(512, 512, 5)` gives `kNoFailureKey`.
  - `TEST(ComputeSortCount, LargestRoom_FitsThirtyTwoBits)`: `(4294967232, kMaxCellBlocksLimit)` gives
    4,294,967,232 (the sum `n + n / 7` is above `2^32`); `(14, 100)` gives 16; `(9, 100)` gives 10; `(100, 1)`
    gives 64.
  - `TEST(ApplySortTailFillAt, UsedBuffers_HoldNoStaleCell)`: buffers for 100 blocks whose `cell ids` array
    first holds the sorted cells of random grid 1 (a full build). Then P2 to P8 run for the 14 lone level-0
    cells `{4 i, 0, 0}`, `i = 0..13`: `n = 14`, no split cell, `sort_count = 16`, so the tail is places 14 and
    15. The test asserts first that, just before step (d), those two places hold valid ids of the build before
    (bit 63 set), then that after it they hold `CellId{0}`, and at the end that `cell_blocks == 14` and
    `block_ids` equals the CPU's. A copy of the sequence with step (d) left out gives more than 14 blocks, the
    extra ones being the blocks of the stale ids at places 14 and 15 (15 blocks for random grid 1, whose two
    stale cells share a block): the test shows it, so the fill is seen to matter.
  - `TEST(BuildGrid, StaleLargeValuesWithFailingList_ReportsTheTrueError)`: every working array of the replay
    (sort keys, both orders, the status's counts) starts as `0xFFFFFFFF` words. Three lists into room for 100
    blocks: one with a repeated cell (step 2), one with a missing sibling (step 3), and the 64 level-1 cells of
    8 level-0 cells into room for 1 (step 5 at P7). Both prefix sums return without error in each, and the
    failure is `kCellsOverlap`, `kSiblingCellMissing`, `kTooManyCellBlocks`, with the CPU's cells, never
    `kInternalFault` (what a size error of the prefix sum maps to). A copy of P7 (a) that returns early instead of writing 0 makes the first prefix sum fail: the
    test shows it.
  - `TEST(BuildGrid, EveryThreadBlockAndThreadUpToBlockIds_EqualsCpuBlockIds)`: P2 to P8 replayed on CPU
    buffers that held another grid before; on the 200 random grids, a list of 9 level-0 blocks into room for 8
    (step 6), a list of `64 * 8 + 1` cells into the same (the length check of P1) and the 64 level-1 cells of 8 level-0
    cells into room for 1 (step 5 at P7, after the split cells are counted): `block_ids`, `cell_blocks`,
    `split_cells` and every `GridFailure` equal the CPU build's.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** with `// Step P7:` and `// Step P8:` comments. **Step 4:**
  the gate; expected `PASS cuda`.
- [ ] **Step 5: Commit.** Subject: `Add split cells and number the blocks in parallel`. Body: `GPU: compiled,
  not run`; private names only.

### Task 11b: Building the tables on the GPU (P9 to P12)

**Kind:** device

**Files:**
- Modify: `staggered_grid_gpu.cu`, `staggered_grid_gpu.h`, `grid_build_steps.h`, `grid_build_steps_test.cc`,
  `staggered_grid_cpu.cc` (the GPU sequence under `#if defined(FLUIDITY_HAS_GPU)`), build registration
- Create: `native/fluidity/grid/staggered_grid_gpu_test.cc` (label `gpu`; its `SOURCES` list the allocation
  counter)

**Interfaces:**
- Consumes: Tasks 10 and 11a, Task 7b (`FindBlockNeighbours`, `ComputeCoveredCellState`,
  `ComputeFacePieceKind`, `ComputeFailureCells`), Task 9c (`ReadGrid`, in tests), slice `ReadBuffer`.
- Shared per-item functions added to `grid_build_steps.h`: `ApplyBlockNeighboursWriteAt` (P9: one block's 36
  entries through `FindBlockNeighbours`), `ApplyAbsentStateAt` (P10a), `int ApplySortedCellStateAt(...)` (P10b: an entry `CellId{0}` of the tail writes nothing and returns `-1`; else
  writes `kActive` for an entry of the active list or `kSplit` for an added one, at
  `ComputeBlockIndexOfSortedEntry` and the cell's local place; returns the level of an active entry, else
  `-1`), `ApplyCoveredCellStateAt` (P10c: a place still `kAbsent` takes `ComputeCoveredCellState`),
  `std::uint32_t ApplyFacePieceKindWriteAt(...)` (P11: writes `ComputeFacePieceKind`, returns 1 for a face
  piece), `ApplyBuildStatusResult` (P12, one thread: `active_cells` as the sum of the 16 levels, or the
  failure's error and cells).
- Produces: the GPU-half functions `WriteBlockNeighboursOnGpu`, `WriteCellStatesOnGpu`,
  `WriteFacePieceKindsOnGpu`, `WriteBuildStatusResultOnGpu`; the GPU paths of `BuildGrid(inputs, buffers)` and
  `ComputeGridCounts(inputs, buffers)`, and in
  `staggered_grid_gpu.h` `RunOnCallStatus ComputeLargestBuildStackOnGpu(std::uint64_t* bytes_per_thread)` (the
  largest `localSizeBytes` over the GPU functions of this file).
- **The parallel form of steps 6 to 8.** Launches are sized on the CPU from the room (`max_cell_blocks`
  blocks, `64` or `240` times that for places); each thread handles only the items below the count it reads
  from the status.
  - P9 neighbour tables: one thread per block.
  - P10 cell states: (a) one thread per cell place writes `kAbsent`; (b) one thread per sorted entry:
    `ApplySortedCellStateAt`, then an integer `atomicAdd` of 1 into `active_cells_per_size_level[level]` for an
    active entry; (c) one thread per cell place: `ApplyCoveredCellStateAt`. Step (c) reads only `kActive`
    marks, which no thread of (c) writes.
  - P11 face piece kinds: one thread per face place, grid-stride like every other step; a thread adds up the
    values its places return and issues one integer `atomicAdd` into `face_pieces`.
  - P12 status: one thread, `ApplyBuildStatusResult`; then one `ReadBuffer` of `GridBuildStatus`. On success
    the buffers move into the returned grid with the record's counts; on failure no grid is returned and the
    buffers stay usable. A record without failure whose `cell_blocks` is above the room is `kInternalFault`.
  - Corner places need no table (design 3.7): `FindCornerPlaces` reads the tables built here.
- File header of `staggered_grid_gpu.h`: the "Stack:" sentence of Global Constraints "GPU stack".

- [ ] **Step 1: Tests.** Label `cpu` (`grid_build_steps_test.cc`):
  `TEST(BuildGrid, EveryThreadBlockAndThread_EqualsCpuTables)`: P2 to P12 replayed on CPU buffers that held
  another grid before; on the lists of the two `gpu` tests `OnGpu_EqualsCpuTablesBitForBit` and
  `OnGpuFailures_EqualCpuFailures` below: version, counts, the used part of the five tables and every
  `GridFailure` equal the CPU build's byte for byte.
  Label `gpu` (lists put on the GPU with `WriteBuffer` in setup only; tables read with `ReadGrid`; each test
  passes the tables, counts and failures it reads to `WriteResultDigest`):
  - `TEST(BuildGrid, OnGpu_EqualsCpuTablesBitForBit)`: the empty list, one cell, the two-size fixture, and the
    200 random grids, into GPU buffers for 4,096 blocks.
  - `TEST(BuildGrid, OnGpuFailures_EqualCpuFailures)`: the 500 mutated lists, 9 level-0 blocks into buffers for
    8 (step 6), `64 * 8 + 1` cells into the same (the length check of P1) and the 64 level-1 cells of 8 level-0 cells
    into room for 1 (step 5 at P7).
  - `TEST(BuildGrid, OnGpuFailedBuildThenRebuild_EqualsFreshBuffers)`: a failing mutated list, then grid B into
    the same buffers; B equals B built into fresh buffers.
  - `TEST(BuildGrid, RebuildIntoUsedGpuBuffers_EqualsFreshBuffers)`: build random grid 1, take its buffers
    back, build the two-size fixture into them.
  - `TEST(BuildGrid, IntoGpuBuffers_AllocatesNothing)`: the 100 random grids built and taken back in turn; GPU
    free bytes unchanged after the first; the allocation counter stays 0.
  - `TEST(ComputeMemoryBytes, MatchesAllocatedGpuBytes)`: at 1,000 blocks the drop in GPU free bytes across
    `BuildRunMemory(RunOn::kGpu, ComputeRunMemoryBytes(BuildGridMemoryEstimate(kGpu, 1, 1000), counts,
    MemoryWhere::kRunOn))` is at least `reserved_bytes()`; building both buffers into it leaves free bytes
    unchanged and allocates no CPU array.
  - `TEST(BuildGridBuffers, OnGpuIntoRunMemory_HandsOutExactlyTheEstimate)`: 1,000 blocks, both sharings:
    `handed_out_bytes() == reserved_bytes()`.
  - `TEST(ComputeGridCounts, OnGpu_EqualsCpu)`: the 200 grids.
  - `TEST(BuildGrid, OnGpuTwice_GivesIdenticalBytes)`.
  - `TEST(BuildGrid, OnGpuSharedBuilderBuffers_OtherLiveGridUnchanged)`: Task 9b's test on GPU buffers,
    compared through `ReadGrid`.
  - `TEST(BuildGrid, OnGpuFunctions_KeepUnderTheDefaultStack)`: `ComputeLargestBuildStackOnGpu` gives at most 1,024.
- [ ] **Step 2: Run** the gate. Expected: `PASS cuda`.
- [ ] **Step 3: Commit.** Subject: `Build the grid tables on the GPU and compare them with the CPU's`. Body:
  `GPU: compiled, not run`; the `gpu` tests above by name.

### Task 12: Addressing and geometry in `grid_tables.h`

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/grid/grid_tables.h`, build registration
- Create: `native/fluidity/grid/grid_tables_test.cc`

**Interfaces:**
- Consumes: Tasks 7a and 7b (`FindSortedBlock`, the table layouts), Task 9a (`BuildTestGrid`, in tests).
- Produces (design 3.2 and 3.7): `ComputeCellWidth(GridFrame frame, SizeLevel size_level)`, `ComputeCellCenter`,
  `ComputeFacePosition`, `ComputeFaceArea`, `ToAxis`, `ToCellId(tables, cell)`, `ToFacePieceId(tables, face)`,
  `FindCellLow`, `FindCellHigh`, `ComputeCellCenterDistance`, `FindCellBlock`, `FindCellPlace`, `FindFacePlace`,
  `ToSizeLevel(tables, block)`. `FindCellBlock` is `FindSortedBlock` over `tables.block_ids`, so the build and
  every stage search blocks with one function.
- **Decided here:** `ComputeCellWidth` halves `biggest_cell_width` `size_level` times (exact, no `pow`).
  `ComputeCellCenterDistance` is the distance along the face piece's axis (as `main`'s
  `cell_center_distance`). `FindCellLow`/`FindCellHigh` on a face piece beside a bigger cell return the bigger
  active cell's place (one level up, through `bigger_block`).
- **Worked example for the header** (2 levels, width 1 m, level 0 `{0,0,0}` and the 8 level-1 cells of
  `{1,0,0}`): the big cell's high X face is four face pieces of 0.25 m2 stored at level 1, each with
  `FindCellLow` = the big cell and a centre distance of 0.75 m (0.5 + 0.25).

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(BuildGrid, OneBigCell_HasSixEdgeFacePieces)`: 1 level, cell `{0,0,0}`, width 1 m: 6 face pieces, each
    `ComputeFaceArea == 1.0`, 3 `kGridEdgeLow`, 3 `kGridEdgeHigh`; `ComputeCellCenterDistance == 0.5` for each.
  - `TEST(BuildGrid, SplitCellBesideBigCell_HasFourQuarterFacePieces)`: the worked example: exactly 4
    `kBetweenCells` X-pieces between the two sizes, each 0.25 m2, the same `FindCellLow`, areas summing to 1.0,
    `ComputeCellCenterDistance == 0.75`.
  - `TEST(FindCellPlace, EveryActiveCell_RoundTripsThroughItsId)` and
    `TEST(FindFacePlace, EveryFacePiece_RoundTrips)`: the 200 random grids; `FindFacePlace` of a `kNotAPiece`
    place's id is `kNoIndex`.
  - `TEST(ComputeFacePosition, EveryFacePiece_MatchesGeometricChecker)`: exact (positions are dyadic).
  - `TEST(ComputeCellWidth, LevelFifteen_IsExact)`: width 1 m gives `0x1p-15`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**. `Read:` `<sec-face-pieces>`, `<sec-cell-blocks>`.
- [ ] **Step 4: Commit.** Subject: `Add addressing and face geometry readable on either processor`.

### Task 13: `FindCellAt`

**Kind:** numerics

**Files:**
- Modify: `grid_tables.h`, build registration. Create: `native/fluidity/grid/find_cell_at_test.cc`

**Interfaces:**
- Consumes: Task 12; `math::Floor`, `math::IsFinite`.
- Produces: both `FindCellAt` forms (design 3.2), by design 5.2.
- **Decided here**, the same first steps for both forms, so they cannot disagree:
  1. Per axis `f = math::Floor((position - origin) / smallest_cell_width)`, a double, in cells of the frame's
     smallest level `L - 1` (`L = size_level_count`). The guard, in those same units: a non-finite `f`, an `f <
     -kMaxCoord * 2^(L - 1)` or an `f >= (kMaxCoord + 1) * 2^(L - 1)` returns `kNoIndex` (both bounds are exact
     doubles, below `2^34`). Only then is `f` cast, to `std::int64_t` (it can reach `2^33`, which does not fit
     32 bits). Proof that the guard is exact: a level-0 cell has a coordinate in `-kMaxCoord..kMaxCoord`, so it
     covers smallest-level coordinates from `c * 2^(L - 1)` to `(c + 1) * 2^(L - 1) - 1`, and the union over
     valid `c` is exactly the guarded interval; every smaller cell lies inside a level-0 cell. Inside the
     guard the level-0 coordinate `floor(f / 2^(L - 1))` is in range, so its block id is never the id of
     another block. The walk down goes through `smaller_blocks` and local places; it makes no id.
  2. The search form finds the level-0 block of that coordinate with `FindCellBlock` and walks down following
     `kSplit` through `smaller_blocks`.
  3. The `near` form starts at `near`'s level-0 block instead (`bigger_block` upward) and moves toward the
     target through `same_level_blocks`, at most two moves. If the target block is not reached, or `near` is
     `kNoIndex` or not below `block_count`, it continues as the search form. The answer is therefore always
     the search form's; `near` only saves the search when the particle moved less than two blocks.

- [ ] **Step 1: Failing tests** (label `cpu`), 2 levels, width 1 m, level 0 `{0,0,0}` and the 8 level-1 cells of
  `{1,0,0}`:
  - `TEST(FindCellAt, PositionOnLowSide_BelongsToCell)`: `{1.0, 0.25, 0.25}` gives level-1 `{2,0,0}`.
  - `TEST(FindCellAt, OnHighSide_BelongsToNext)`: `{std::nextafter(1.0, 0.0), 0.5, 0.5}` gives level-0
    `{0,0,0}`; `{1.0, 0.5, 0.5}` gives level-1 `{2,1,1}`; `{1.5, 0.25, 0.25}` gives level-1 `{3,0,0}`.
  - `TEST(FindCellAt, OutsideEveryCell_IsNoIndex)`: `{-0.5, 0.5, 0.5}` and `{2.0, 0.5, 0.5}`.
  - `TEST(FindCellAt, HugeOrNotFinitePosition_IsNoIndex)`: x of `1e300`, `-1e300`, infinity and NaN, both
    forms.
  - `TEST(FindCellAt, PositionBeyondTheIdRange_IsNoIndex)`: in this fixture (level-0 width 1 m) both forms give
    `kNoIndex` for x = 262144 m (level-0 coordinate `kMaxCoord + 1`, one past the last level-0 cell), for x =
    -262144 m, and for x = 524288.25 m: its level-0 coordinate `2^19` packed into an id is the id of level-0
    `{0,0,0}`, so without the guard the big cell would be returned. x = 262143.5 m (the last level-0 cell,
    which holds no active cell) is `kNoIndex` through the search.
  - `TEST(FindCellAt, NearForm_EqualsSearchOnRandomPositions)`: the 200 random grids, 1,000 positions each in
    the box widened by one cell; `near` in turn the found cell's block, a block two moves away, the last block
    of the grid (many moves away) and `kNoIndex`: every `near` form equals the search form, which equals
    `GeometricGrid::ContainingCell`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**; the comment says why there is no exact-edge recovery (design
  5.2) and gives the fixture's three positions as the example. `Read:` `<sec-cell-blocks>`.
- [ ] **Step 4: Commit.** Subject: `Find the active cell that contains a position`.

### Task 13a: The cell size rule: wanted width and size level

**Kind:** numerics

The grid owns `cell_size_rule.h`, the per-cell formulas of the cell size rule that the step (its stage that
chooses active cells) and cell resize both read; the passes that apply them belong to those modules. Tasks 13a,
13b and 13c build the header in three parts.

**Files:**
- Create: `native/fluidity/grid/cell_size_rule.h` (S), `cell_size_rule_test.cc`,
  `cell_size_rule_test_cases.h` (test-only, CPU only)
- Modify: `docs/method-paper/fluidity-method.typ`, `docs/glossary.md` (`wanted cell width`, `reference speed`),
  build registration

**Interfaces:**
- Consumes: `SizeLevel`, `kMaxSizeLevels`, `GridFrame`, `ComputeCellWidth` (Tasks 3, 12); `math::Sqrt`,
  `math::Min`, `math::Max` (`base/scalar_math.h`; they pass infinities through exactly and give the same bits
  on both processors).
- The field the design first called `velocity_gradient_size` is `CellSizeFacts::velocity_change_size` (1/s): the
  glossary bans "velocity gradient" and gives "velocity change" for it; the rules step fails the old name.
- Produces, of design 3.8: `CellSizeSettings`, `CellWidthsByLevel`, `CellSizeFacts`, `ComputeReferenceSpeed`,
  `ComputeWantedCellWidth`, `ComputeWantedSizeLevel`, and
  `CellWidthsByLevel ComputeCellWidthsByLevel(GridFrame frame)` (`width[l] = ComputeCellWidth(frame, l)` for
  `l < size_level_count`, 0 beyond: the one place the table is filled).
- **Decided here:**
  - `ComputeWantedCellWidth`, in this order of operations, so both processors round alike: `w_min =
    widths.width[settings.smallest_size_level]`, `w_max = widths.width[0]`; `w_depth = w_min * Max(1, depth /
    (surface_reach_in_cell_widths * w_min))`; `w_solid = w_min * Max(1, distance / (solid_reach_in_cell_widths *
    w_min))`; `w_shear = shear_width_factor * reference_speed / velocity_change_size` when that size is above 0, else
    `+infinity`; `wanted = Max(w_min, Min(Min(Min(w_depth, w_solid), w_shear) * cell_size_factor, w_max))`. The
    result is `w_min` at once where `touches_air_pocket`, `wants_smallest_cells`, or `depth <
    surface_reach_in_cell_widths * w_min` (the cells within 8 smallest widths of the surface: the factor does
    not reach them). An infinite fact gives an infinite term, never NaN: every factor is positive, so no
    `infinity * 0` arises. A negative depth or distance counts as the surface or the solid.
    `holds_particles_that_keep_their_count` and `touches_air_gas_or_solid` are read by cell resize's merge
    test, not here; their comments say so.
  - `ComputeWantedSizeLevel` walks `l` from 0 up and returns the first `l` with `width[l] <= wanted_width`;
    `size_level_count - 1` when none is.
  - `ComputeReferenceSpeed` is `Max(largest_liquid_speed, Sqrt(gravity_size * biggest_cell_width))`.
- Test cases (`cell_size_rule_test_cases.h`): `CellSizeFacts BuildTestCellSizeFacts(int i, double
  cell_size_factor)`: depth `(i % 97) * 0.013` m (`+infinity` when `i % 11 == 0`), distance `(i % 89) * 0.011` m
  (`+infinity` when `i % 7 == 0`), `velocity_change_size` `(i % 83) * 0.37` 1/s, `touches_air_pocket` when `i % 31 == 0`,
  `wants_smallest_cells` when `i % 37 == 0`; and `double ComputeTestReferenceSpeed(int i)` = `0.5 + (i % 13) *
  0.25`. "The 10,000 fact sets" are `i = 0..9999`.

- [ ] **Step 1: Paper.** Subsection `== The cell size rule <sec-cell-size-rule>` in `<sec-grid>`, with
  `<eq:wanted-cell-width>`: the depth, solid and shear terms, the cell size factor, the clamp, `w_min` at
  pockets and gas and within 8 smallest widths of the surface; the shear term's `c_u = 0.25` called a starting
  value to calibrate; the reference speed `U_ref = max(u_max, sqrt(g w_max))`; why a new cell starts one level
  smaller than its nearest committed neighbour (it borders air, so its depth is 0, while the cap keeps the
  size-step rule); the start grid built by the same rule from the start shapes (centre bounds minus half the
  diagonal, no shear at the start, then the size-step closure); the refinement volume of cell size factors,
  why a product with the physical width, the lookup of the smallest factor over the cell's box in layers of
  halved size; why levels come from comparing a table of widths (no `log2`, no `pow`, no fused multiply-add:
  same bytes on both processors and in any batch cut); and why the factor table is the same on every machine
  (Task 13b's formula).
- [ ] **Step 2: Failing tests** (label `cpu`). Unless stated: frame `biggest_cell_width` 1 m, 5 levels (widths
  1, 0.5, 0.25, 0.125, 0.0625, all exact), `smallest_size_level` 4, default settings, facts at their "far"
  values (`depth_below_surface` and `distance_to_solid` `+infinity`, `velocity_change_size` 0, `cell_size_factor` 1, flags
  false), `reference_speed` 1 m/s; comparisons exact.
  - `TEST(ComputeCellWidthsByLevel, EveryLevel_EqualsComputeCellWidth)`: bit for bit; 0 beyond level 4.
  - `TEST(ComputeWantedCellWidth, SurfaceCell_IsSmallest)`: depth 0 gives 0.0625, also with distance 10 and
    with `cell_size_factor` 4.
  - `TEST(ComputeWantedCellWidth, UniformTranslation_NoShearSplit)`: `velocity_change_size` 0, `reference_speed` 5: 1.0, and
    level 0.
  - `TEST(ComputeWantedCellWidth, NothingWithinReach_IsBiggestWidth)`: far facts with `velocity_change_size` `1e-300`, then
    with depth `DBL_MAX`: 1.0, never NaN.
  - `TEST(ComputeWantedCellWidth, DepthBeyondReach_GrowsLinearly)`: depth 1.0 gives 0.125 (`0.0625 * 1.0 /
    0.5`); depth 0.25 gives 0.0625; depth 100 gives 1.0 (clamped).
  - `TEST(ComputeWantedCellWidth, SolidBeyondReach_GrowsLinearly)`: distance 0.5 gives 0.25 (`0.0625 * 0.5 /
    0.125`); with depth 1.0 as well, 0.125.
  - `TEST(ComputeWantedCellWidth, ShearTerm_IsShearFactorTimesSpeedOverVelocityChange)`: frame 0.5 m, 6 levels,
    `smallest_size_level` 5 (`w_min` 0.015625), `velocity_change_size` `sqrt(2) * 10`: 0.0176776695296637 within 1e-15
    relative; `ComputeWantedSizeLevel` gives 5.
  - `TEST(ComputeWantedCellWidth, PocketOrGas_IsSmallest)`: each flag with depth 100: 0.0625.
  - `TEST(ComputeWantedCellWidth, FactorOne_SameBitsAsNoFactor)`: the 10,000 fact sets with factor 1 give the
    bits of a test-local copy of the rule with the product left out.
  - `TEST(ComputeWantedCellWidth, FactorHalf_HalvesDeepWidth)`: depth 1.0: 0.0625; distance 0.5: 0.125.
  - `TEST(ComputeWantedCellWidth, FactorTwo_DoublesDeepWidth)`: depth 1.0: 0.25; depth 100: 1.0 (clamped).
  - `TEST(ComputeWantedCellWidth, FactorTwoNearSurface_IsSmallest)`: depth 0.25 (4 `w_min`, within the 8),
    factor 2: 0.0625; depth 0.5 (exactly 8 `w_min`, not within), factor 2: 0.125.
  - `TEST(ComputeWantedCellWidth, NegativeDepthOrDistance_IsSmallest)`: depth `-0.03` gives 0.0625; distance
    `-0.03` gives 0.0625; depth `-0.03` with factor 4 gives 0.0625; never NaN.
  - `TEST(ComputeWantedSizeLevel, WidthEqualToALevel_GivesThatLevel)`: 0.125 gives 3; `std::nextafter(0.125,
    0)` gives 4; 0.2 gives 3; 5 gives 0.
  - `TEST(ComputeWantedSizeLevel, BelowSmallestWidth_GivesSmallestLevel)`: 0.01 gives 4.
  - `TEST(ComputeReferenceSpeed, SlowLiquid_UsesGravityWaveSpeed)`: `(0.1, 9.81, 1.0)` gives
    `math::Sqrt(9.81)` bit for bit; `(5, 9.81, 1.0)` gives 5.
- [ ] **Step 3: Run**, fail; **Step 4: Implement**; the header carries the formula and the worked example of
  the two `...BeyondReach` tests. `Read:` `<sec-cell-size-rule>`, `<eq:wanted-cell-width>`.
- [ ] **Step 5: Commit.** Subject: `Add the shared cell size rule`. New names: `CellSizeSettings`,
  `CellSizeFacts`, `CellWidthsByLevel`, `ComputeCellWidthsByLevel`, `ComputeReferenceSpeed`,
  `ComputeWantedCellWidth`, `ComputeWantedSizeLevel` (one rule for the step and cell resize).

### Task 13b: Cell size factors: the table and the refinement volume lookup

**Kind:** numerics

**Files:**
- Modify: `cell_size_rule.h`, `cell_size_rule_test_cases.h`, `docs/glossary.md` (`cell size factor`,
  `refinement volume`. Meanings to write: a cell size factor is a number that multiplies a cell's wanted width,
  1 leaving it unchanged; a refinement volume is a 3D array of such factors over the domain, one per key time)
- Create: `native/fluidity/grid/cell_size_factor_test.cc`; build registration

**Interfaces:**
- Consumes: Task 13a; `ToSizeLevel`, `ToCellCoord` (Task 3).
- Produces, of design 3.8: `kCellSizeFactorStepsPerDoubling`, `kMaxCellSizeFactorCode`, `CellSizeFactorTable`,
  `ComputeCellSizeFactorTable`, `RefinementFactors`, `ComputeCellSizeFactorCode`.
- **The table, exactly.** `factor[i]`, `i = code + 24` in `0..48`, is `2^(i / 4 - 6) * root[i % 4]` with integer
  division and `root = {1.0, 0x1.306fe0a31b715p+0, 0x1.6a09e667f3bcdp+0, 0x1.ae89f995ad3adp+0}`: the doubles
  nearest to `2^0`, `2^(1/4)`, `2^(1/2)`, `2^(3/4)` (1.189207115002721, 1.4142135623730951, 1.681792830507429),
  written as hexadecimal literals so no compiler rounds a decimal. The power of two is built by doubling
  `0x1p-6`. A double times a power of two is exact while the result stays in range, and here it stays between
  `2^-6` and `2^6`, so every entry is one fixed bit pattern on every machine and both processors.
  `ComputeCellSizeFactorTable` is `FLUIDITY_CPU_AND_GPU constexpr` and calls no library function. The header
  gives this reason in two sentences.
- **The lookup** is integer only: `s = smallest_size_level - ToSizeLevel(cell)`, `coord = ToCellCoord(cell)`.
  When `s <= sample_spacing_log2`: layer 0, index per axis `(coord << s) >> sample_spacing_log2`. Else layer `L
  = s - sample_spacing_log2` (code 0 when `L >= layer_count`), index `coord`. A negative coordinate, or an
  index at or beyond the layer's count (`ceil(count / 2^L)`), gives 0. No volume (`count_x == 0`) gives 0. The
  step and cell resize pass `table.factor[code + 24]` as `CellSizeFacts::cell_size_factor`.
- Test cases: `BuildTestRefinementFactors(...)`, which builds the layers from layer-0 codes (each layer the
  smallest code over 2 x 2 x 2 of the layer below). The fixture: frame 1 m, 5 levels, `smallest_size_level` 4,
  spacing log2 2 (one sample per 4 smallest cells), 2 x 1 x 1 level-0 cells: layer 0 is 8 x 4 x 4 samples,
  layers 1 and 2 are 4 x 2 x 2 and 2 x 1 x 1; code -8 on the samples `x = 5` of layer 0, 0 elsewhere.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ComputeCellSizeFactorTable, QuarterDoublings_AreExact)`: `static_assert`s: `factor[24] == 1`,
    `factor[28] == 2`, `factor[20] == 0.5`, `factor[0] == 0x1p-6`, `factor[48] == 64`, `factor[25] ==
    0x1.306fe0a31b715p+0`, `factor[26] == 0x1.6a09e667f3bcdp+0`, `factor[27] == 0x1.ae89f995ad3adp+0`,
    `factor[1] == 0x1.306fe0a31b715p-6`, `factor[47] == 0x1.ae89f995ad3adp+5`; and for every `i <= 44`,
    `factor[i + 4] == 2 * factor[i]`; strictly increasing.
  - `TEST(ComputeCellSizeFactorTable, MiddleRoot_IsTheSquareRootOfTwo)`: `factor[26] == math::Sqrt(2.0)`.
  - `TEST(ComputeCellSizeFactorCode, NoVolume_IsZero)`: every cell of `BuildRandomGridCells(7, 5, 4)`.
  - `TEST(ComputeCellSizeFactorCode, SmallCell_ReadsItsSample)`: level-4 cell `{20, 3, 3}` (sample 5): -8;
    `{19, 3, 3}`: 0.
  - `TEST(ComputeCellSizeFactorCode, BigCell_IsSmallestCodeOverItsBox)`: level-0 cell `{1, 0, 0}` (samples
    4..7): -8; `{0, 0, 0}`: 0; level-2 cell `{5, 0, 0}`: -8.
  - `TEST(ComputeCellSizeFactorCode, OutsideTheVolume_IsZero)`: level-4 `{40, 0, 0}` and `{-1, 0, 0}`.
  - `TEST(ComputeCellSizeFactorCode, MoreLevelsThanLayers_IsZero)`: the fixture with layers 0 and 1 only:
    level-0 cell `{1, 0, 0}` gives 0.
- [ ] **Step 2: Run**, fail. **Step 3: Implement.** `Read:` `<sec-cell-size-rule>`.
- [ ] **Step 4: Commit.** Subject: `Add cell size factors and the refinement volume lookup`. New names:
  `kCellSizeFactorStepsPerDoubling`, `kMaxCellSizeFactorCode`, `CellSizeFactorTable`,
  `ComputeCellSizeFactorTable`, `RefinementFactors`, `ComputeCellSizeFactorCode`.

### Task 13c: The size level of a new cell, and the cell size rule on the GPU

**Kind:** device

**Files:**
- Modify: `cell_size_rule.h`, `cell_size_rule_test.cc`, build registration
- Create: `native/fluidity/grid/cell_size_rule_gpu_test.cc` (label `gpu`), `cell_size_rule_test_gpu.h`,
  `cell_size_rule_test_gpu.cu` (test-only runner in the form of `base/shared_math_test_gpu`, launched through
  `base/gpu_launch.cuh`, returning `RunOnCallStatus`)

**Interfaces:**
- Consumes: Tasks 13a and 13b; `GridTables`, `ComputeCellCenter`, `ToCellId`, `ToSizeLevel` (Task 12),
  `FindCellAt(tables, position)` (Task 13), `ComesBefore(CellId, CellId)` (Task 3); `BuildTestGrid` (Task 9a),
  the GPU build (Task 11b).
- Produces: `SizeLevel ComputeNewCellSizeLevel(const GridTables& committed, Vec3 position, SizeLevel
  smallest_size_level)` (design 3.8).
- Produces also `CellPlaceIndex FindNearestCommittedCell(const GridTables& committed, Vec3 position)`: the
  search alone, so its tie rule can be tested where both cells have one size.
- **Decided here:** `FindNearestCommittedCell` probes the 27 positions `position + (i, j, k) *
  frame.biggest_cell_width`, `i, j, k` in `{-1, 0, 1}`, `k` outermost, with the search form of `FindCellAt`.
  Among the cells found, the nearest centre by squared distance wins; a tie goes to the cell whose id comes
  first (`ComesBefore`); `kNoIndex` when no probe finds a cell. `ComputeNewCellSizeLevel` is that cell's level
  plus one, capped at `smallest_size_level`; `smallest_size_level` when there is none.

- [ ] **Step 1: Failing tests.** Label `cpu`:
  - `TEST(ComputeNewCellSizeLevel, NextToBigCell_IsOneLevelSmaller)`: committed grid of 3 levels, width 1 m,
    level 0 `{0,0,0}` and the 8 level-1 cells of `{1,0,0}`; `smallest_size_level` 2. Position `{-0.5, 0.5,
    0.5}` (beside the level-0 cell): 1; `{2.25, 0.25, 0.25}` (beside the level-1 cells): 2.
  - `TEST(ComputeNewCellSizeLevel, AtSmallestLevel_IsCapped)`: the second position with `smallest_size_level`
    1: 1.
  - `TEST(ComputeNewCellSizeLevel, NoCellNearby_IsSmallestLevel)`: `{10, 10, 10}`: 2; an empty committed grid:
    2.
  - `TEST(FindNearestCommittedCell, TwoCellsEquallyNear_FirstInFixedOrderWins)`: 1 level, width 1 m, cells
    `{0,0,0}` and `{2,0,0}`; position `{1.5, 0.5, 0.5}` is exactly 1.0 m from both centres: the result is the
    place of `{0,0,0}`. Position `{1.75, 0.5, 0.5}`: the place of `{2,0,0}`. Position `{10, 10, 10}`: `kNoIndex`.
- [ ] **Step 2: GPU tests** (label `gpu`; outputs to `WriteResultDigest`):
  - `TEST(ComputeWantedCellWidth, OnGpu_SameBytesAsCpu)`: the 10,000 fact sets with `cell_size_factor =
    table.factor[i % 49]` and `ComputeTestReferenceSpeed(i)`, default settings, the 5-level widths: the wanted
    widths (`memcmp`), their `ComputeWantedSizeLevel`, and `ComputeCellSizeFactorCode` on Task 13b's fixture for
    the level-`(i % 5)` cell with level-4 coordinates `{i % 41, i % 23, 0}` shifted right by `4 - i % 5`, are
    identical to the CPU's. The table itself is computed on the GPU too and compared with `memcmp`.
  - `TEST(ComputeNewCellSizeLevel, OnGpu_SameBytesAsCpu)`: `BuildRandomGridCells(7, 3, 4)` built on the GPU and
    on the CPU; 10,000 positions from seed `20261115` in the grid's box widened by one biggest width on each
    side: identical levels.
- [ ] **Step 3: Run**, fail; **Step 4: Implement.** `Read:` `<sec-cell-size-rule>`. **Step 5:** the gate;
  expected `PASS cuda`.
- [ ] **Step 6: Commit.** Subject: `Choose the size level of a new cell from its nearest committed cell`. Body:
  `GPU: compiled, not run`; new names `ComputeNewCellSizeLevel`, `FindNearestCommittedCell`; the two `gpu` tests by name.

### Task 14: Touching cells and a cell's face pieces

**Kind:** numerics

**Files:**
- Modify: `grid_tables.h`, build registration. Create: `native/fluidity/grid/touching_cells_test.cc`

**Interfaces:**
- Consumes: Task 12.
- Produces: `TouchingCells`, `FindTouchingCells`, `CellFacePieces`, `FindFacePieces` (design 3.2).
- **Worked example for the header** (the two-size fixture of Task 12): the big cell touches the four level-1
  cells with `x = 2` and no other cell, so its count is 4; its 9 face pieces are 5 whole faces and the 4
  quarter faces stored at level 1.

- [ ] **Step 1: Failing tests** (label `cpu`), the 200 random grids:
  - `TEST(BuildGrid, EveryCellSide_IsCoveredOnce)`: for every active cell and side, the areas of its face
    pieces on that side (from `FindFacePieces`) sum exactly to the cell's face area.
  - `TEST(BuildGrid, ClosedSurface_AreasCancel)`: per cell, the sum over its face pieces of `+area * axis` (the
    cell is `cell_low`) or `-area * axis` (the cell is `cell_high`) is exactly `{0, 0, 0}`.
  - `TEST(FindTouchingCells, RandomGrids_MatchGeometricChecker)`: the same set, ascending, `count <= 56`.
  - `TEST(FindTouchingCells, AllNeighboursSplit_Has56)`: a level-0 cell whose 26 neighbours are split.
  - `TEST(FindTouchingCells, TwoSizeFixture_BigCellTouchesFour)`: the worked example; `FindFacePieces` gives 9.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**. `Read:` `<sec-face-pieces>`.
- [ ] **Step 4: Commit.** Subject: `Find touching cells and the face pieces of a cell`.

### Task 15: Corners and the cells around an edge

**Kind:** numerics

**Files:**
- Modify: `grid_tables.h`, build registration. Create: `native/fluidity/grid/corners_test.cc`

**Interfaces:**
- Consumes: Task 12 (`FindCellBlock`, `ToSizeLevel`).
- Produces (design 3.7): `CellCorners`, `FaceCorners`, both `FindCornerPlaces`, `ComputeCornerPosition`,
  `CellsAroundEdge`, `FindCellsAroundEdge`.
- **Where a corner is stored** (design 3.7, made exact here; the header and the paper say it in these words). A
  corner is a point of the 5 x 5 x 5 lattice of every block whose box holds it, sides included. Among the
  blocks of the grid whose lattice holds the point, it is stored in a block of the smallest cells (the largest
  level value); and among several blocks of that level, in the one that comes last in the fixed order (the
  highest `z`, then `y`, then `x`: the block on the high side, as for face plane 4). The level counts whether
  or not a cell of that block touches the point. `FindCornerPlaces` looks, from the smallest level up to the
  cell's own, at the up to 8 blocks of that level around the point with `FindCellBlock`, and stops at the
  first level that has one.
- **Three fixtures** (frame origin 0, width 1 m; `CornerPlaceIndex` is `block * 125 + (z * 5 + y) * 5 + x`):
  1. *A corner on the high side of a block.* 1 level, cells `{3,0,0}` (block 0) and `{4,0,0}` (block 1). The
     corner of `{3,0,0}` with bit 0 set (high x) is at `{4,0,0}` m: lattice point `(4,0,0)` of block 0 and
     `(0,0,0)` of block 1. It is stored in block 1: index 125. With `{3,0,0}` alone it is index 4.
  2. *A corner in the middle of a big cell's face.* 2 levels, level 0 `{0,0,0}` and the 8 level-1 cells of
     `{1,0,0}`: block 0 is the level-0 block, block 1 the level-1 block (it covers 0 to 2 m). The point `{1,
     0.5, 0.5}` m is the low corner of level-1 cell `{2,1,1}`: lattice `(2,1,1)` of block 1, index `125 + 32 =
     157`. It is none of the big cell's 8 corners. The big cell's corner at `{1,0,0}` m is lattice `(2,0,0)` of
     block 1: index 127, the same as corner 0 of level-1 cell `{2,0,0}`. The big cell's corner at `{0,0,0}` m
     is lattice `(0,0,0)` of block 1 as well: index 125, though no small cell touches it; and its corner at
     `{1,1,1}` m is index `125 + 62 = 187`.
  3. *Four blocks around an edge.* 1 level, cells `{3,3,0}`, `{4,3,0}`, `{3,4,0}`, `{4,4,0}`: blocks 0, 1, 2, 3
     in fixed order. The point `{4,4,0}` m is stored in block 3 for all four cells: index 375. Without cell
     `{4,4,0}` there are three blocks and the point is stored in the last of them (the block of `{3,4,0}`,
     block 2) at lattice `(4,0,0)`: index 254.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(FindCornerPlaces, CornerOnHighSideOfBlock_IsStoredInHighBlock)`: fixture 1, both cases.
  - `TEST(FindCornerPlaces, CornerInMiddleOfBigFace_BelongsToSmallCellsOnly)`: fixture 2: the four indexes
    above; the four small cells around `{1, 0.5, 0.5}` m all give 157; every corner of the big cell is in
    `125..249`; `ComputeCornerPosition` of 125 is `{0,0,0}` and of 157 is `{1, 0.5, 0.5}`.
  - `TEST(FindCornerPlaces, FourBlocksAroundAnEdge_LastBlockInFixedOrderStoresIt)`: fixture 3, both cases.
  - `TEST(FindCornerPlaces, EveryPhysicalCorner_HasOnePlace)`: the 200 random grids; across all active cells,
    two corners have the same `CornerPlaceIndex` exactly when `ComputeCornerPosition` gives the same position;
    the set of positions equals `GeometricGrid::Corners()`.
  - `TEST(FindCornerPlaces, FaceCorners_AreTheCellCornersOnThatSide)`: for every face piece of a uniform 8^3
    grid.
  - `TEST(FindCellsAroundEdge, UniformGrid_FindsFourSameSize)`;
    `TEST(FindCellsAroundEdge, BesideSplitCell_IsNotSameSize)`;
    `TEST(FindCellsAroundEdge, AtGridEdge_IsNoIndexAndSameSize)` (absent places are not "another size").
- [ ] **Step 2: Run**, fail. **Step 3: Implement**; the header carries fixtures 1 and 2 as worked examples.
  `Read:` `<sec-cell-blocks>`.
- [ ] **Step 4: Commit.** Subject: `Find cell corners and the cells around an edge`. Names for pressure, solids
  and viscosity.

### Task 16: Cells near a face and the face search

**Kind:** numerics

**Files:**
- Modify: `grid_tables.h`, build registration. Create: `native/fluidity/grid/face_search_test.cc`

**Interfaces:**
- Consumes: Tasks 12 and 14.
- Produces (design 3.7): `kMaxCellsNearFace`, `CellsNearFace`, `FindCellsNearFace`, `kMaxFaceSearchRadius`,
  `FaceSearch`, `BuildFaceSearch`, `FindNextFace`.
- **Decided here:** a radius above `kMaxFaceSearchRadius` is treated as `kMaxFaceSearchRadius` (documented,
  tested); "within" is the strict `|x_f - x_face| < radius * width` on each axis. It is not the half-open test of
  `FindNearbyFaces`: this search relates two faces and must give the same answer from either of them. The
  search reads the 27 same-level blocks around the face's block, the bigger block of each and the 8 smaller
  blocks of each: at most 270 block indexes, each visited once.
- **Worked example for the header:** on a uniform grid of width 1 m, radius 1.5 around an X face piece finds
  the 3 x 3 x 3 X face pieces centred on it, itself included: 27.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(FindCellsNearFace, RandomGrids_MatchEveryCellThatListsTheFace)`: the 200 random grids; for every
    face piece, the set equals the cells whose positions can list it (low, high and cells touching either),
    ascending.
  - `TEST(FindCellsNearFace, SmallRandomGrids_NeverExceedTheBound)`: 10,000 small random grids; `count <= 112`;
    the largest count found is reported with `RecordProperty("largest_cells_near_face", ...)`.
  - `TEST(FindNextFace, RandomGrids_MatchGeometricChecker)`: radii 1.5, 2.5, 4.0; the same faces as
    `GeometricGrid::FacesWithin`, in ascending order.
  - `TEST(FindNextFace, UniformGridRadiusOneAndAHalf_Finds27)`: the worked example.
  - `TEST(BuildFaceSearch, RadiusAboveLimit_IsTreatedAsTheLimit)`: radius 6 gives the radius-4 result.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**. `Read:` `<sec-nearby-faces>`.
- [ ] **Step 4: Commit.** Subject: `Find the cells near a face and walk faces within a radius`.

### Task 17: `FindNearbyFaces`

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/nearby_face_weights.h` (S), `native/fluidity/grid/nearby_faces_test.cc`
- Modify: `docs/glossary.md` (`candidate`: a face that may get weight from a position, before any is left
  out), build registration

**Interfaces:**
- Consumes: `FindCellAt(tables, position, near)`, `FindTouchingCells`, `FindFacePieces`, `GridError`.
- Produces: `NearbyFaceList`, `NearbyFaceMask`, `NearbyFaceFit`, `FindNearbyFaces`, `ToMaskOfAll` (design 3.5).
  Candidate rule: design 5.4 steps 1 and 2. On `kPositionOutsideGrid` or `kInternalFault` (more candidates than the list holds) the list has
  `count` 0 and `*error` is set; otherwise `*error = kNone`.
- **The window test, exactly** (no float comparison decides whether a face is a candidate). Two private inline
  functions hold it; `FindNearbyFaces` and the face form of Task 19 call them and nothing else, so CPU, GPU
  and both forms make the same choice:
  - `int ComputeCandidateWindowQuarter(double position, double origin, double cell_width, std::int32_t
    cell_coord)`: `n_p = math::Floor(4 * ((position - origin) / cell_width - cell_coord))`, then clamped to 0..3.
    One per axis, for the cell `FindCellAt` returned. The clamp ties `n_p` to that cell: whatever rounding does,
    the window is one of the four the cell has.
  - `bool ComputeIsInCandidateWindow(std::int64_t face_quarter, int window_quarter)`: `window_quarter - 5 <=
    face_quarter && face_quarter <= window_quarter + 6`. `face_quarter` is the face centre's coordinate on that
    axis in quarter widths of the containing cell, from its low corner, found from the ids alone: for a face
    piece of level `l` with coordinate `k` in cells of its level and the cell of level `p` with coordinate `c`,
    it is `2^(p - l + 2) * k - 4 c` along the face's own axis (`2 k`, `4 k` or `8 k` for `l = p + 1, p, p - 1`)
    and `2^(p - l + 1) * (2 k + 1) - 4 c` across (`2 k + 1`, `4 k + 2`, `8 k + 4`), in `std::int64_t`.
  What the header says: in real numbers this is `-1.5 h < x_f - x_p <= 1.5 h` (twelve quarter steps, the low
  end open, the high end closed, because a cell owns its low side), and every face centre of the three levels
  lies on the quarter lattice, so the integer test loses nothing. Among cells of one size the faces on the own
  axis are at `n_f = 4 k` and the window holds exactly three of them (`-4, 0, 4` for `n_p` 0 or 1; `0, 4, 8` for
  `n_p` 2 or 3); across they are at `n_f = 4 k + 2` and the window always holds `-2, 2, 6`, the three touching
  columns. So every position has 27 faces, by construction. A face at exactly 1.5 h has weight 0, so the kept
  weights still sum to 1; a face the window keeps whose float offset rounds to just beyond 1.5 h also has
  weight 0, because the weight curve is 0 at and beyond its reach.
- **The bound.** `kMaxNearbyFaces = 216` and `struct NearbyFaceMask { std::uint64_t bits[4]; }` (bit `i` of
  word `i / 64` is face `i` of the list). The header says, with the proof of design 5.4 in five lines: 124 is
  the largest count known and the exact maximum where two cell sizes meet; no valid grid passes 212; so
  running out of room is therefore an impossible state, reported as `kInternalFault`; the check is kept because
  a list must never be overrun. The
  candidates of a position depend on it only through its cell and the three `n_p`, so a cell has 64 candidate
  lists per axis and tests go through them.
- **The one writer of the list.** A private form `NearbyFaceList FindNearbyFacesWithRoom(const GridTables&
  tables, Vec3 position, Axis axis, CellBlockIndex near, int max_faces, GridError* error)` does the walk;
  `FindNearbyFaces` is that form with `kMaxNearbyFaces`. It appends through one statement that tests `count <
  max_faces` first; on the first face that does not fit it sets `kInternalFault`, sets `count` to 0 and
  returns. So the guard is reached by a valid grid and a smaller room, and the test needs no invalid table.
- **Sizes**, asserted in the header: `sizeof(NearbyFaceList) == 888` (24 B and 216 indexes of 4 B),
  `sizeof(NearbyFaceMask) == 32`, and in Task 18 `sizeof(NearbyFaceWeights) == 1152`. The file header says what
  a caller on the GPU must know (Global Constraints "GPU stack"): one axis keeps 1,152 B per thread and three
  axes held at once 3,456 B, above the driver's default of 1,024 B, so a module whose GPU function calls them
  declares its own stack entry.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(FindNearbyFaces, AllNeighboursSplit_Has76)`: level-0 cell `{1,1,1}` of a 3^3 level-0 box whose 26
    other cells are split; a position at the centre; each axis gives exactly 76.
  - `TEST(FindNearbyFaces, AllNeighboursSplitEveryWindow_Has76)`: the same grid; one position in each of the
    64 windows of the cell (`low + (n_p + 0.5) h / 4` per axis): every window and axis gives exactly 76.
  - `TEST(FindNearbyFaces, TwoSizesWorstCase_Has124)`: 3 levels, origin 0, biggest width 4 m. The level-0 box
    `{0..2}^3` is split everywhere, giving 6^3 level-1 places of width 2 m. The cell is level 1 `{2,2,2}`. Of
    its 26 neighbours, 14 stay at level 1: the 8 corners, `(-1,-1,0)`, `(1,-1,0)` and `(0,+-1,+-1)` (as `(dx,
    dy, dz)`); the 12 others and every level-1 place farther out are split into level-2 cells (15 level-1 and
    1,608 level-2 cells). Position `{4.75, 4.25, 4.75}` m (window `(1, 0, 1)`): the Y axis gives exactly 124,
    and no axis in any of the 64 windows of that cell gives more.
  - `TEST(FindNearbyFaces, MoreFacesThanRoom_IsInternalFault)`: the grid and position of
    `TwoSizesWorstCase_Has124`, Y axis, through `FindNearbyFacesWithRoom` with `max_faces = 100`: `*error ==
    kInternalFault`, `count == 0`, and no entry of `faces` from index 100 on was written (the returned
    list is value-initialised, so those entries are 0; a second run with `max_faces = 124` returns all 124 and
    `kNone`, and with 123 fails).
  - `TEST(FindNearbyFaces, SmallRandomGrids_NeverExceedTheProvenBound)`: 200 small random grids, every active
    cell, all 64 windows, every axis: never above 212; the largest count is reported with
    `RecordProperty("largest_nearby_faces", ...)`.
  - `TEST(FindNearbyFaces, RandomGrids_MatchGeometricChecker)`: the 200 grids, 50 positions each, of which 10
    on a cell side or centre: the same list as `GeometricGrid::NearbyFaces(cell, {n_p of x, y, z}, axis)`, in
    ascending order. This is the test of the window rule on mixed sizes.
  - `TEST(ComputeCandidateWindowQuarter, AnyRounding_StaysInZeroToThree)`: a position one step below the
    cell's low corner and one at its high corner give 0 and 3.
  - `TEST(FindNearbyFaces, CellCentreInNonDyadicFrame_Has27)`: 1 level, origin `{-17.616723516683763, 0, 0}`,
    width 0.0375, the 5^3 cells around cell `{-28, 2, 2}`; position x `-18.647973516683763` (that cell's
    centre as doubles give it), y and z at `2.3` widths: each axis gives 27, and also at one and at three
    rounding steps either side in x (`std::nextafter`). A float test of the offsets gives 36 here.
  - `TEST(FindNearbyFaces, OneStepBelowCellSide_Has27)`: origin `{-0.5, -0.5, -0.5}`, width 0.0625, the 5^3
    cells around cell `{34, 10, 10}`; position x `1.6249999999999998` (one step below the side at 1.625;
    `FindCellAt` gives cell 34), y and z at 10.3 widths from the origin: each axis gives 27; the same one step
    above the side and at the side. A float test of the offsets gives 18 on the Y and Z axes here.
  - `TEST(FindNearbyFaces, NonDyadicFrameAtSidesAndCentres_AlwaysHas27)`: origin `{-17.616723516683763, 0.3,
    -7.1}`, width 0.0375, uniform 9^3; for 1,000 positions made of a cell's side or centre on each axis, moved
    by -3 to +3 rounding steps: 27 on every axis.
  - Four tests on a uniform 5^3 grid, origin 0, width 1 m, the middle cell `{2,2,2}`; each axis gives exactly
    27 in every one:
    `TEST(FindNearbyFaces, UniformGridOffCentre_Has27)`: position `{2.6, 2.7, 2.8}`; no offset is 1.5 h.
    `TEST(FindNearbyFaces, UniformGridAtCellCentre_Has27)`: `{2.5, 2.5, 2.5}`; for each axis the list holds
    the 9 faces at `+1.5 h` along that axis and none at `-1.5 h`.
    `TEST(FindNearbyFaces, UniformGridOnCellSide_Has27)`: `{2.0, 2.6, 2.7}`, on the low X side, which belongs
    to the cell: the X list has no offset of 1.5 h; the Y and Z lists each hold 9 faces at `+1.5 h` along x.
    `TEST(FindNearbyFaces, UniformGridAtCellCorner_Has27)`: `{2.0, 2.0, 2.0}`: each list holds 15 faces with an
    offset of `+1.5 h` on one of its two cross axes (9 + 9 - 3), and none at `-1.5 h`.
  - `TEST(FindNearbyFaces, Outside_IsPositionOutsideGrid)`.
  - `TEST(ToMaskOfAll, OneHundredTwentyFourFaces_SetsExactlyThoseBits)`: bits 0..123 set and 124..255 clear
    (the mask spans two words and stops inside the second); with a count of 216, bits 0..215.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**; the header gives the bound's proof (design 5.4, "Why at most 212"). `Read:`
  `<sec-nearby-faces>`.
- [ ] **Step 4: Commit.** Subject: `List the faces near a position`.

### Task 18: `ComputeNearbyFaceWeights` and `ComputeFaceWeight`

**Kind:** numerics

**Files:**
- Modify: `nearby_face_weights.h`, `docs/glossary.md` (`kept`, `offset`, `offset_from_mean`, `weight_change`,
  `weighted_spread`, `mean_offset`), build registration
- Create: `native/fluidity/grid/nearby_face_weights_test.cc`

**Interfaces:**
- Consumes: `ComputeWeightCurve`, `ComputeFacePosition`, `ComputeFaceArea`, `kMinWeightSum`, `kMinPivotShare`,
  Task 17.
- Produces: `NearbyFaceWeights`, `FaceWeight`, `ComputeNearbyFaceWeights`, `ComputeFaceWeight(tables, weights,
  position_in_list)` (design 3.5). Steps 4 to 9 of `<proc-nearby-face-weights>`, which design 5.4 follows
  step for step. Port the two-pass sum and the Cholesky with
  pivot floor from `git show main:native/src/graded_mac_transfer_3d.cc` lines 185-309 (`BuildSupport`,
  `SolveAffineSlope`; pivot constant at line 19). Not ported: its radial weight (lines 171-176).
- **Decided here:** a private inline `ComputeRawFaceWeight(tables, face, position, cell_width)` returns `w_f`
  and `g_f`; every weight in this module (here and in Task 19) goes through it and scales with the same
  expressions (`weight = w / W`, `weight_change = (g - weight * G) / W`), so the two forms stay bit-identical.
  Sums run over kept faces in list order. `inverse_weighted_spread` is computed from the Cholesky factor of
  `weighted_spread / h^2`, then scaled by `1 / h^2`; it is zero unless `fit == kFull`.
- **No weight at all (step 5).** After the sums, if `W <= 0` (no kept face has weight): `fit = kNone`,
  `weight_sum`, `mean_offset`, `raw_weight_change_sum`, `weighted_spread` and `inverse_weighted_spread` are all
  zero, `is_complete` is false, and the function returns; steps 6 to 9 do not run, so nothing divides by `W`.
  `ComputeFaceWeight` on such weights returns `weight` 0 and zero vectors for `offset_from_mean` and
  `weight_change` (`offset` is still the face position minus the position). `W < kMinWeightSum` with `W > 0`
  is decided at step 9 (`kMeanOnly`).

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ComputeNearbyFaceWeights, CompleteSameSize_MatchesClosedForm)`: uniform 8^3, `h = 0.25` m, 1,000
    random positions (Task 5's generator) inside the central 4^3 cells, all kept: 27 faces; `W = 1` within `1e-14`, `mean_offset = 0`
    within `1e-14 h`, `weighted_spread = h^2/4 I` within `1e-14 h^2`; `is_complete`; `fit == kFull`.
  - `TEST(ComputeNearbyFaceWeights, AnyKeptSet_WeightsSumToOne)`: the 200 random grids, 50 positions, random
    masks with at least one kept face of positive weight: `|sum weight - 1| <= 1e-14`, `|sum weight_change| <=
    1e-13 / h` (each axis), every kept weight `>= 0`.
  - `TEST(ComputeNearbyFaceWeights, AtExactCellCentre_IsCompleteWithNineZeroWeights)`: the uniform 5^3 grid
    and the position `{2.5, 2.5, 2.5}` of Task 17, all kept: for each axis 27 faces, `is_complete`, `fit ==
    kFull`, exactly 9 weights equal to 0 (the faces at `+1.5 h` along the axis), their `weight_change` exactly
    0, the other 18 weights `> 0`, `W = 1` within `1e-14`.
  - `TEST(ComputeNearbyFaceWeights, OnCellSide_IsCompleteWithNineZeroWeights)`: `{2.0, 2.6, 2.7}`: the X axis
    has no weight of 0; the Y and Z axes have 9 each; `is_complete` on all three; `W = 1` within `1e-14`.
  - `TEST(ComputeNearbyFaceWeights, TwoSizesWorstCase_WeightsSumToOne)`: the grid and position of Task 17's
    `TwoSizesWorstCase_Has124`, all kept, each axis: `|sum weight - 1| <= 1e-14`, `|sum weight_change| <= 1e-13
    / h`, every weight `>= 0`, `is_complete` false; on the Y axis all 124 entries are read back through
    `ComputeFaceWeight`.
  - `TEST(NearbyFaceWeights, OneAxis_Is1152Bytes)`: the `static_assert` of the size.
  - `TEST(ComputeNearbyFaceWeights, NonDyadicFrameAtSidesAndCentres_IsComplete)`: the 1,000 positions of Task
    17's test of that name: `is_complete`, `fit == kFull`, every weight `>= 0`, `W = 1` within `1e-14`.
  - `TEST(ComputeNearbyFaceWeights, AtCellCorner_IsCompleteWithFifteenZeroWeights)`: `{2.0, 2.0, 2.0}`: 15
    weights of 0 and 12 above 0 on each axis; `is_complete`; `W = 1` within `1e-14`.
  - `TEST(ComputeNearbyFaceWeights, LinearField_IsReproduced)`: value `a + b . x_f` (`a = 0.7`,
    `b = {0.3, -1.1, 2.0}`) on the 200 grids with random masks, only where `fit == kFull`: `sum weight * value
    - fitted . mean_offset` within `1e-13 (|a| + |b| h)` of `a + b . x_p`; the fitted matrix row `(sum weight *
    value * offset_from_mean) * inverse_weighted_spread` within `1e-11 |b|` of `b`.
  - `TEST(ComputeNearbyFaceWeights, WeightChange_MatchesFiniteDifference)`: centred difference with step
    `1e-6 h`, positions at least `0.01 h` from every cell side and from `t = 0.5, 1.5` for every face, same
    list and mask: within `1e-7 / h`.
  - `TEST(ComputeNearbyFaceWeights, FacesOnOnePlane_IsMeanOnly)`: uniform grid, the mask keeps the 9 faces of
    one plane: `kMeanOnly`.
  - `TEST(ComputeNearbyFaceWeights, EmptyMask_IsNone)`: uniform grid, off-centre position, no bit set: `fit ==
    kNone`; `weight_sum`, the three components of `mean_offset` and of `raw_weight_change_sum`, the nine entries
    of `weighted_spread` and of `inverse_weighted_spread` are exactly 0; `is_complete` false; no NaN anywhere.
  - `TEST(ComputeNearbyFaceWeights, OnlyZeroWeightFacesKept_IsNone)`: the position `{2.5, 2.5, 2.5}` of Task
    17, X axis, the mask keeps only the 9 faces at `+1.5 h`: `W` is exactly 0, so the same assertions hold, and
    `ComputeFaceWeight` of each kept entry gives weight 0 and a zero `weight_change`.
  - `TEST(ComputeNearbyFaceWeights, CrossingSizeChange_WeightsJump)`: two positions `1e-9` m apart on either
    side of the two-size boundary of Task 12: the weight of a shared face differs by more than `1e-3`. It
    documents the stated property (known open physics "traction across cells of different sizes"); a fix must
    change this test.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** with `// Step N:` comments from
  `<proc-nearby-face-weights>`; the header states the four properties of design 5.4 and gives the complete
  case's numbers as a worked example. `Read:` `<sec-nearby-faces>`, `<eq:scaled-weights>`,
  `<eq:weighted-spread>`.
- [ ] **Step 4: Commit.** Subject: `Compute nearby-face weights from the faces actually kept`.

### Task 19: The weight seen from a face

**Kind:** numerics

**Files:**
- Modify: `nearby_face_weights.h`, build registration. Create: `native/fluidity/grid/face_seen_weight_test.cc`

**Interfaces:**
- Consumes: Task 18 (`ComputeRawFaceWeight`, `NearbyFaceWeights`).
- Produces (design 3.7): `NearbyFaceSums`, `ToNearbyFaceSums`, `ComputeFaceWeight(tables, face, position, cell,
  sums)`. Returns weight 0 (and zero vectors) when `face` fails the candidate rule for `position`; the test is
  `ComputeCandidateWindowQuarter` and `ComputeIsInCandidateWindow` of Task 17, with `n_p` computed from
  `position` and `cell` exactly as the list form computes it.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ComputeFaceWeight, SeenFromFace_IsBitIdentical)`: the 200 random grids, 50 positions, random masks;
    for every kept entry, the face form with `ToNearbyFaceSums(weights)` returns the same bytes (`std::memcmp`)
    for `weight`, `offset`, `offset_from_mean`, `weight_change` as the list form.
  - `TEST(ComputeFaceWeight, FaceNotInList_IsZero)`: a face two cells away.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** through `ComputeRawFaceWeight`. `Read:`
  `<sec-nearby-faces>`.
- [ ] **Step 4: Commit.** Subject: `Compute one face's weight from a particle's sums`. Names for transfer.

### Task 20: The shared grid functions on the GPU

**Kind:** device

**Files:**
- Create: `native/fluidity/grid/grid_gpu_test.cc` (label `gpu`), `grid_functions_test_gpu.h` and
  `grid_functions_test_gpu.cu` (test-only GPU runner in the form of `base/shared_math_test_gpu`: one function
  per shared function, each launching a grid-stride GPU function with `ApplyGpuFunction` and returning
  `RunOnCallStatus`)
- Modify: build registration (`fluidity_add_test(grid_gpu_test SOURCES grid_gpu_test.cc
  grid_functions_test_gpu.cu LABEL gpu LINKS fluidity_base_test_support fluidity_grid_test_support)`)

**Interfaces:**
- Consumes: `GridTables` from `tables(RunOn::kGpu)` of a grid built on the GPU (Task 11b), `FindCellAt`,
  `FindNearbyFaces`, `ComputeNearbyFaceWeights`, `ComputeFaceWeight` (both forms), `FindCornerPlaces`;
  `CheckWithinTolerance`, `WriteResultDigest`.
- Each test builds every grid twice from one list: on the CPU (the reference) and on the GPU from a GPU copy of
  the list made in the test's setup.
- The runner's weights function keeps a list and its weights per thread (1,152 B and more), above the driver's
  default stack of 1,024 B; its file header says so and that the driver grows the stack at the first launch (a
  test runner reserves nothing). The runner also has `RunOnCallStatus ComputeWeightsRunnerStackOnGpu(
  std::uint64_t* bytes_per_thread)` (`cudaFuncGetAttributes`, `localSizeBytes`), so the hardware run gives the
  modules that call the weights a measured number for their stack entry.

- [ ] **Step 1: Tests** (label `gpu`, one thread per position, outputs copied back with `ReadBuffer` and passed
  to `WriteResultDigest`):
  - `TEST(ComputeNearbyFaceWeights, OnGpu_MatchesCpu)`: 10,000 positions on random grids 1..10, all axes,
    random masks: lists, masks, `fit`, `is_complete` exact; weights and sums within `1e-15 + 1e-13 |value|`
    (`CheckWithinTolerance`, the CPU value the reference), offsets and spreads with the floor times `h` (`h^2`).
  - `TEST(ComputeNearbyFaceWeights, OnGpuTwice_GivesIdenticalBytes)`.
  - `TEST(FindCornerPlaces, OnGpu_EqualsCpu)`: every active cell of the 10 grids; the 8 corner places
    identical (corners have no table: this shows the GPU-built tables answer them as the CPU's do).
  - `TEST(FindCellAt, OnGpu_EqualsCpu)`: the 10,000 positions, both forms.
  - `TEST(ComputeNearbyFaceWeights, OnGpuRunner_ReportsItsStack)`: the bytes are at least
    `sizeof(NearbyFaceWeights)` and are written with `RecordProperty("weights_runner_stack_bytes", ...)`.
- [ ] **Step 2: Run** the gate. Expected: `PASS cuda` (compiled and linked; nothing of this task runs in the
  container: the shared functions' `cpu` tests are the evidence until the hardware run).
- [ ] **Step 3: Commit.** Subject: `Compile the shared grid functions into GPU tests`. Body: `GPU: compiled, not
  run`; the five tests by name.

### Task 21: Value buffers bound to a grid version

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/grid/grid_values.h` (CPU only), `grid_values_cpu.cc`, `grid_values_test.cc` (its
  `SOURCES` list the allocation counter), `grid_values_no_gpu_test.cc`, `grid_values_gpu_test.cc`
- Modify: build registration

**Interfaces:**
- Consumes: `StaggeredGrid` (Task 9a), `Buffer`, `BufferValue`, `BuildBuffer(RunOn, std::size_t, RunMemory&)`,
  slice `ReadBuffer`/`WriteBuffer`, `CheckCanRunOn`, `ToBufferError`.
- Produces (design 3.6): one class template and four names for it,

  ```cpp
  // One value per place of every block: places_per_block is 64 (cells), 240 (faces), 125 (corners) or 1 (the
  // block itself). Built once with room for max_cell_blocks blocks, then bound to each grid version.
  template <BufferValue Value, int places_per_block>
  class GridValues;  // Members as design 3.6 lists them for CellValues. Move-only.
  template <BufferValue Value> using CellValues = GridValues<Value, kCellPlacesPerBlock>;
  template <BufferValue Value> using FaceValues = GridValues<Value, kFacePlacesPerBlock>;
  template <BufferValue Value> using CornerValues = GridValues<Value, kCornerPlacesPerBlock>;
  template <BufferValue Value> using CellBlockValues = GridValues<Value, 1>;
  ```
  The four are distinct types, so a face buffer cannot be passed for a cell buffer. `BuildCellValues`,
  `BuildFaceValues`, `BuildCornerValues`, `BuildCellBlockValues` (room form `(run_on, max_cell_blocks,
  run_memory)` and grid-bound form `(grid, run_memory)`) are one-line forms of one private
  `BuildGridValues<Value, places_per_block>`; `ApplyGridVersion`, `ReadValues` and `WriteValues` are each
  written once as a template over `places_per_block`, with the parameters of design 3.6. `CheckGridVersion` is
  in `grid_values_cpu.cc` with the module's one `BufferError` mapping. No GPU half: values cross only through
  `base/` `Buffer` calls. Read and Write use the slice forms over the bound part only.
- The builders follow Global Constraints "One reserved block per run": `CheckCanRunOn` first, one memory
  piece, `kOutOfMemory` with the bytes asked. `ReadValues` and `WriteValues` compare the span's size with the
  bound part themselves (`kBadInput`); a size error from the `base/` copy after that check is `kInternalFault`.
- `ApplyGridVersion` reads only `grid.version()` and `grid.counts().cell_blocks`, which every grid keeps on the
  CPU, and records them; it launches nothing and copies nothing, on either processor. Requires: the values are
  on the grid's processor, checked by `assert`. A grid with more blocks than the room is `kTooManyCellBlocks`.

- [ ] **Step 1: Failing tests.** Label `cpu`:
  - `TEST(ApplyGridVersion, RebindsWithoutAllocating)`: each of the four names with `Value = double`, room for
    4,096 blocks, bound to the 100 random grids in turn: zero counted allocations; `grid_version()` and
    `block_count()` follow each grid.
  - `TEST(ApplyGridVersion, MoreBlocksThanRoom_IsTooManyCellBlocks)`: room for 8, a 9-block grid: version and
    count unchanged.
  - `TEST(ReadValues, NotBound_IsWrongGridVersion)`; `TEST(ReadValues, SpanOneEntryShort_IsBadInput)`.
  - `TEST(WriteValues, ThenReadValues_ReturnsTheSameValues)` for each of the four.
  - `TEST(CheckGridVersion, OtherVersion_IsWrongGridVersion)`; `TEST(BuildCellValues, ZeroBlocks_IsBadInput)`.
  - `TEST(BuildFaceValues, IntoRunMemory_TakesOnePieceOfPlacesTimesSize)`: for each of the four with `Value =
    double` at 1,000 blocks, a fresh `RunMemory` of 2 MiB: `handed_out_bytes() ==
    ComputeRunMemoryPieceBytes(1000 * places * 8)` (1,920,000 B for faces); the grid-bound form at a 9-block
    grid takes `ComputeRunMemoryPieceBytes(9 * places * 8)`.
  - `TEST(BuildCellValues, RunMemoryTooSmall_IsOutOfMemory)`: a 256 B reservation, 1,000 blocks asked:
    `kOutOfMemory`, `magnitude == 512000`. Runs in debug and release.
  Label `no_gpu`: `TEST(BuildFaceValues, OnGpuWithoutAGpu_IsRunOnCallFailed)`: `BuildFaceValues<double>(
  RunOn::kGpu, 1000, cpu_run_memory)` fails and `handed_out_bytes() == 0`.
  Label `gpu`: `TEST(WriteValues, OnGpuThenReadValues_ReturnsTheSameValues)` for each of the four, bound to a
  grid built on the GPU; `TEST(ApplyGridVersion, OnGpuGrid_LaunchesNothing)` (100 GPU grids bound in turn: free
  bytes unchanged, and the values read back are unchanged, since a bind does not touch them). Both write
  digests.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**; the header states the rule of design 3.6 (a consumer checks
  `grid_version()` and `block_count()`; values of a grid piece are bound again at every load). `Read:`
  `<sec-cell-blocks>`, `<sec-run-memory>`.
- [ ] **Step 4: Commit.** Subject: `Add value buffers that are built once and bound per grid version`. Body:
  `GPU copies and binds: compiled, not run`; new names: `GridValues` and its four names, the builders,
  `ApplyGridVersion`, `ReadValues`, `WriteValues`, `CheckGridVersion`.

### Task 22a: Ranges and their border on the CPU

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/cell_block_range.h` (CPU only), `cell_block_range_cpu.cc`,
  `border_block_steps.h` (S, private), `cell_block_range_test.cc` (its `SOURCES` list the allocation counter),
  `border_block_steps_test.cc` (label `cpu`)
- Modify: `grid_test_support.h`, `grid_test_support.cc`, `docs/glossary.md` (`range`: a run of blocks in fixed
  order that one stage writes; `border block`: a block outside a range that the stage may read), build
  registration

**Interfaces:**
- Consumes: `CellBlockRange` (Task 3), `StaggeredGrid`, `GridTables`, `FindCellPlace`, `FindFacePlace`,
  `ToCellBlockIndex` (Tasks 9a, 12), `FindNearbyFaces` (Task 17, in a test).
- Produces (design 3.4, 3.7): `RangeWithBorder`, `BorderWidth`, both `BuildRangeWithBorder`, `BuildRanges`, and
  the two span forms of `FindBorderBlocks`, by the single ascending walk of design 5.5. The span forms and
  `BuildRangeWithBorder` read CPU tables: `Requires: grid.run_on() == RunOn::kCpu`, checked by `assert`.
  `BuildRanges` reads only `counts()` and works on either processor.
- Shared, in `border_block_steps.h`:
  `struct BlockSet { CellBlockRange range; const CellBlockIndex* sorted_blocks; std::uint32_t sorted_count; };`
  `bool ComputeIsInBlockSet(const BlockSet& set, CellBlockIndex block)` (in `range`, or found by binary search)
  and `bool ComputeIsBorderBlock(const GridTables& tables, const BlockSet& set, CellBlockIndex block,
  BorderWidth width)`: the per-block test of design 5.5 through `block`'s own 27 same-size-level neighbours, its
  bigger block and its 8 smaller blocks. The grid's neighbour tables are the only search structure. The CPU
  walk is "for each block in index order, if not in the set and `ComputeIsBorderBlock`, append"; the span
  forms use `BlockSet{{0, 0}, blocks.data(), blocks.size()}`.
- **What a stage reads, per width** (the definition the tests check):
  - `kTouchingCells`, block set `S`: for every active cell `c` stored in `S`: `c`, every active cell that
    touches `c`, and every face piece of `c` or of a cell that touches `c`. The border is every block outside
    `S` that stores one of these.
  - `kFourCellWidths`: the same, and every block `FaceSearch` can read from a face piece stored in `S`: the 27
    same-level blocks around the face piece's block, the bigger block of each and the 8 smaller blocks of
    each. That is a rule about blocks, wider than the faces actually within 4 widths.
- Test support added: `std::vector<CellBlockIndex> FindBorderGeometrically(const GeometricGrid& geometry, const
  GridTables& tables, std::span<const CellBlockIndex> blocks, BorderWidth width)`. For `kTouchingCells` it
  takes the read set from `GeometricGrid` (`TouchingCells`, `FacePieces`) and asks the tables only where each
  is stored (`FindCellPlace`, `FindFacePlace`). For `kFourCellWidths` it adds the block rule above computed by
  coordinate arithmetic on every pair of `CellBlockId`s, with no neighbour table. Its comment says plainly that
  this second part restates the rule with other mechanics and is not geometric evidence; the containment test
  below is.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(FindBorderBlocks, TouchingCells_EqualsGeometricReadSet)`: the 200 random grids, 5 random ascending
    block sets each: equal to `FindBorderGeometrically`; ascending, no repeats, no block of the set.
  - `TEST(FindBorderBlocks, FourCellWidths_EqualsBlockRuleByCoordinates)`: the same grids and sets.
  - `TEST(FindBorderBlocks, FourCellWidths_HoldsEveryFaceWithinFourWidths)`: for every face piece stored in the
    set, every face of `GeometricGrid::FacesWithin(piece, 4.0)` is stored in the set or its border.
  - `TEST(FindBorderBlocks, OutputOneShort_WritesFirstEntriesReturnsTotal)`: the output form with a span of
    exact size gives the list; with a span 1 short it writes the first `n - 1` and returns `n`; the counter
    stays 0.
  - `TEST(FindBorderBlocks, PlaneFourPieceBeyondTouchingCell_IsInBorder)`: one level, width 1 m, active cells
    `{3,3,0}` (block A, block coordinate `{0,0,0}`), `{4,3,0}` (block B, `{1,0,0}`) and `{6,6,0}` (block C,
    `{1,1,0}`); indexes A = 0, B = 1, C = 2. Cell `{4,3,0}` touches the range cell `{3,3,0}`; its high Y face
    is plane 4 of B, so it is stored at plane 0 of C; no cell of C touches `{3,3,0}`. The `kTouchingCells`
    border of `{0}` is exactly `{1, 2}`; a border made of touching cells alone would be `{1}`.
  - `TEST(ComputeIsBorderBlock, PlaneFourFixture_MarksBlockC)` (`border_block_steps_test.cc`; the same fixture
    with tables built by hand); `TEST(ComputeIsInBlockSet, RangeOrSortedList_IsFound)`.
  - `TEST(FindBorderBlocks, StackedBorders_HoldEveryRead)`: the 100 random grids cut by `BuildRanges(grid, 4)`;
    for each range `R`, layers for the widths (touching, touching, four widths, touching) of four stages,
    found from the last stage backwards: `L1 = border(R, touching)`, `L2 = border(R + L1, four widths)`, and
    so on. The stage that runs on `R + L1 + ... + Lk` reads, by the definition above, only blocks of `R + L1 +
    ... + L(k+1)`.
  - `TEST(FindNearbyFaces, RangeWithBorder_HoldsEveryFace)`: random grids cut by `BuildRanges(grid, 3)`; 100
    positions in range cells: every listed face is stored in the range or its `kTouchingCells` border.
  - `TEST(BuildRanges, Cuts_CoverEveryBlockOnceInOrder)`; `TEST(FindBorderBlocks, UnsortedSet_IsBadInput)`;
    `TEST(FindBorderBlocks, IndexBeyondTheGrid_IsBadInput)`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**. `Read:` `<sec-cell-blocks>`, `<sec-face-pieces>`.
- [ ] **Step 4: Commit.** Subject: `Cut the grid into ranges and find their border blocks`. New names:
  `RangeWithBorder`, `BorderWidth`, `BuildRangeWithBorder`, `BuildRanges`, `FindBorderBlocks`.

### Task 22b: Stacked borders on buffers

**Kind:** numerics

**Files:**
- Modify: `cell_block_range.h`, `cell_block_range_cpu.cc`, `cell_block_range_test.cc`,
  `grid_memory_estimate.h`, `grid_memory_estimate.cc`, `grid_memory_estimate_test.cc`, build registration
- Create: `native/fluidity/grid/cell_block_range_no_gpu_test.cc`

**Interfaces:**
- Consumes: Task 22a; `GridBuildStatus` (Task 6a); `BuildBuffer`, `BuildOrderBuffers`,
  `ComputeOrderBuffersSize`, `CheckCanRunOn`.
- Produces (design 3.4), running on `grid.run_on()` (CPU here, GPU in Task 23):

  ```cpp
  // Room to search the border of any block set of a grid of at most max_cell_blocks blocks. Move-only.
  class BorderSearchBuffers {
   public:
    std::uint32_t max_cell_blocks() const;
    RunOn run_on() const;
  };
  // Builds them as memory pieces of run_memory, in the order of BuildBorderSearchMemoryEstimate's entries.
  // Requires: run_on == run_memory.run_on() when run_on can run, checked by assert in debug builds.
  // Fails: kRunOnCallFailed (CheckCanRunOn failed; nothing taken), kBadInput (0 or above kMaxCellBlocksLimit),
  //        kOutOfMemory (magnitude is the bytes of the array that did not fit), kInternalFault (the arrays do
  //        not match the estimate's entries).
  std::expected<BorderSearchBuffers, GridFailure> BuildBorderSearchBuffers(RunOn run_on,
                                                                           std::uint32_t max_cell_blocks,
                                                                           RunMemory& run_memory);
  // Every block outside `range` that is in border_so_far or in the border, at `width`, of range plus
  // border_so_far: one more layer of a stacked border, merged with the layers before. Ascending, each once.
  // Writes the first border_blocks.size() entries and returns how many there are in all. Allocates nothing.
  // Requires: every buffer is on grid.run_on(), checked by assert in debug builds.
  // Fails: kBadInput           border_so_far is not ascending, holds a block of `range` or an index at or above
  //                            the grid's block count; border_blocks is border_so_far; border_so_far_count is
  //                            above its size
  //        kTooManyCellBlocks  search_buffers is too small for the grid
  //        kRunOnCallFailed    a call to the GPU failed
  //        kInternalFault      the prefix sum of the marks failed, or the count read back is above the grid's
  //                            block count: a fault in the grid, not in the inputs
  std::expected<std::uint32_t, GridFailure> FindBorderBlocks(const StaggeredGrid& grid, CellBlockRange range,
                                                             const Buffer<CellBlockIndex>& border_so_far,
                                                             std::uint32_t border_so_far_count, BorderWidth width,
                                                             BorderSearchBuffers& search_buffers,
                                                             Buffer<CellBlockIndex>& border_blocks);
  ```
  A stacked border is this call once per layer with two output buffers used in turn; no merge is needed
  between layers.
- Produces also, declared in `grid_memory_estimate.h` beside the grid's estimate (not in
  `cell_block_range.h`): `std::vector<MemoryEstimateEntry> BuildBorderSearchMemoryEstimate(RunOn run_on,
  std::uint32_t max_cell_blocks)`.
- **Entries** of `BuildBorderSearchMemoryEstimate` (`grid: ` prefix, `KeptFor::kWholeRun`, `copies = 1`,
  `where = kRunOn`), each one `kGivenCount` entry with `bytes_each = 1`. On both processors: `border marks`
  (`4 * (max_cell_blocks + 1)` B). For `kGpu` also: `border order buffers` (`bytes_per_item * (max_cell_blocks +
  1) + fixed_bytes` of `ComputeOrderBuffersSize(RunOn::kGpu)`: an `OrderBuffers` built once and passed to every
  prefix sum of the search) and `border status` (`sizeof(GridBuildStatus)`). CPU total: `4 * max_cell_blocks +
  4` B, as the design states.
- The CPU form uses the marks, in two passes, so the array the estimate counts is the array the code reads and
  both processors mark by one rule: pass 1 writes for each block `b` the mark "outside `range`, and in
  `border_so_far` or `ComputeIsBorderBlock(tables, {range, border_so_far, count}, b, width)`"; pass 2 walks the
  marks in index order and appends.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(FindBorderBlocks, StackedBufferForm_EqualsSpanFormLayerByLayer)`: the 100 random grids, ranges from
    `BuildRanges(grid, 4)`, layers (touching, touching, four widths, touching); at each layer the buffer form's
    list equals `std::merge` of `border_so_far` and the span form on range plus `border_so_far`.
  - `TEST(FindBorderBlocks, StackedBufferFormShortOutput_WritesFirstEntriesReturnsTotal)`;
    `TEST(FindBorderBlocks, StackedBufferFormSameBufferTwice_IsBadInput)`;
    `TEST(FindBorderBlocks, BorderSoFarInsideRange_IsBadInput)`;
    `TEST(FindBorderBlocks, SearchBuffersTooSmall_IsTooManyCellBlocks)`;
    `TEST(FindBorderBlocks, StackedBufferForm_AllocatesNothing)`.
  - `TEST(BuildBorderSearchMemoryEstimate, MatchesAllocatedBytes)`: the CPU estimate at 1,000 blocks is one
    entry of 4,004 B (`4 * max_cell_blocks + 4`), and the marks array of `BuildBorderSearchBuffers(kCpu, 1000,
    run_memory)` holds 1,001 entries of 4 B.
  - `TEST(BuildBorderSearchBuffers, IntoRunMemory_HandsOutExactlyTheEstimate)`: a CPU `RunMemory` of
    `ComputeRunMemoryBytes` of the estimate at 1,000 blocks (4,096 B: one memory piece of 4,004 B rounded):
    `handed_out_bytes() == reserved_bytes()`.
  - `TEST(BuildBorderSearchMemoryEstimate, Gpu_CountsMarksOrderBuffersAndStatus)`: at 1,000 blocks the `kRunOn`
    sum is `4004 + bytes_per_item * 1001 + fixed_bytes + sizeof(GridBuildStatus)`; the sum of
    `ComputeRunMemoryPieceBytes` over the three entries equals `ComputeRunMemoryBytes` of the estimate.
  - `TEST(BuildBorderSearchBuffers, ZeroBlocks_IsBadInput)`.
  Label `no_gpu`: `TEST(BuildBorderSearchBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed)` (a CPU `RunMemory`;
  `handed_out_bytes() == 0`).
- [ ] **Step 2: Run**, fail. **Step 3: Implement**. `Read:` `<sec-cell-blocks>`, `<sec-grid-memory>`.
- [ ] **Step 4: Commit.** Subject: `Find stacked borders on buffers without allocating`. New names:
  `BorderSearchBuffers`, `BuildBorderSearchBuffers`, `BuildBorderSearchMemoryEstimate`, the buffer form of
  `FindBorderBlocks`.

### Task 23: Border blocks on the GPU

**Kind:** device

**Files:**
- Create: `native/fluidity/grid/cell_block_range_gpu.h` (private), `cell_block_range_gpu.cu` (private),
  `cell_block_range_gpu_test.cc` (label `gpu`)
- Modify: `cell_block_range_cpu.cc` (dispatch on `grid.run_on()`, the GPU sequence under
  `#if defined(FLUIDITY_HAS_GPU)`), `border_block_steps.h`, `border_block_steps_test.cc`, build registration

**Interfaces:**
- Consumes: `BlockSet`, `ComputeIsInBlockSet`, `ComputeIsBorderBlock` (Task 22a), Task 22b, `GridBuildStatus`,
  `ToFailureKey` (Task 7a); `ComputeExclusivePrefixSum` (with the `OrderBuffers` of `BorderSearchBuffers`),
  `ReadBuffer`, `ToBufferError`, `ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`.
- Shared per-item functions added to `border_block_steps.h`: `std::uint64_t
  ComputeBorderSoFarFailureKey(...)` (step 2, one entry), `std::uint32_t ComputeBorderMark(...)` (step 3, one
  block), `void ApplyBorderBlockWriteAt(...)` (step 5, one block).
- Produces: the GPU path of the buffer form of `FindBorderBlocks` and of `BuildBorderSearchBuffers`; in
  `cell_block_range_gpu.h`, `RunOnCallStatus ComputeLargestBorderSearchStackOnGpu(std::uint64_t* bytes_per_thread)` and
  the "Stack:" sentence in the file header.
- GPU-half functions, one per step: `ApplyBorderStatusResetOnGpu`, `CheckBorderSoFarOnGpu`,
  `WriteBorderMarksOnGpu`, `WriteBorderBlocksOnGpu`.
- **The parallel form:** (1) CPU: the same-buffer, count and room checks; a one-thread GPU function resets the
  status. (2) One thread per `border_so_far` entry `i`: ascending, outside `range`, below the block count,
  else `atomicMin(ToFailureKey(1, i, 0))` with `kBadInput`. (3) One thread per block `b` of the grid, which
  writes mark 0 when step 2 failed, so the prefix sum of step 4 never adds stale words (the
  count is known on the CPU from `grid.counts()`): mark = `b` outside `range` and (`b` in `border_so_far` or
  `ComputeIsBorderBlock`). (4) `ComputeExclusivePrefixSum` in place over the block count (the total lands one
  past the last block). (5) One thread per block: `b` was marked exactly when `sums[b + 1] != sums[b]`; a
  marked `b` writes itself at `border_blocks[sums[b]]` when `sums[b] < border_blocks.size()`; `result_count`
  is the total. (6) One `ReadBuffer` of the status: the count (or the failure) is all that comes back. Index
  order of `b` is kept by the prefix sum, so the list is the CPU's.

- [ ] **Step 1: Tests.** Label `cpu` (`border_block_steps_test.cc`):
  `TEST(FindBorderBlocks, StaleLargeMarksWithBadBorderSoFar_PrefixSumAddsZeros)`: the marks array starts as
  `0xFFFFFFFF` words; `border_so_far` is not ascending: after step 3 every mark is 0, the prefix sum returns
  without error with a total of 0, and the failure is the CPU form's `kBadInput` of step 2.
  `TEST(FindBorderBlocks, EveryThreadBlockAndThread_EqualsCpuStackedLayers)`: steps 2 to 5 replayed on CPU
  buffers as Global Constraints "Replay tests" says; on Task 22b's grids, ranges and layers and the failing
  inputs of `OnGpuFailures_EqualCpuFailures`: lists, counts and failures equal the CPU form's.
  Label `gpu` (every grid built on both processors from one list; lists and counts read back go to
  `WriteResultDigest`):
  - `TEST(FindBorderBlocks, OnGpu_EqualsCpuStackedLayers)`: Task 22b's grids, ranges and layers.
  - `TEST(FindBorderBlocks, OnGpuShortOutput_SameCountAsCpu)`: an output buffer of 3 entries.
  - `TEST(FindBorderBlocks, OnGpuFailures_EqualCpuFailures)`: unsorted, inside-range, beyond-the-grid and
    same-buffer inputs.
  - `TEST(FindBorderBlocks, OnGpu_AllocatesNothing)`; `TEST(FindBorderBlocks, OnGpuTwice_GivesIdenticalBytes)`.
  - `TEST(BuildBorderSearchMemoryEstimate, MatchesAllocatedGpuBytes)`: the drop in GPU free bytes across
    `BuildRunMemory(RunOn::kGpu, ...)` of the estimate is at least `reserved_bytes()`; building the search
    buffers into it leaves free bytes unchanged.
  - `TEST(BuildBorderSearchBuffers, OnGpuIntoRunMemory_HandsOutExactlyTheEstimate)`: 1,000 blocks.
  - `TEST(FindBorderBlocks, OnGpuFunctions_KeepUnderTheDefaultStack)`: at most 1,024.
- [ ] **Step 2: Run** the gate. Expected: `PASS cuda`.
- [ ] **Step 3: Commit.** Subject: `Find border blocks on the GPU through the neighbour tables`. Body: `GPU:
  compiled, not run`; the `gpu` tests above by name.

### Task 24: Grid pieces for the batched run

**Kind:** integration

**Files:**
- Create: `native/fluidity/grid/grid_piece.h` (CPU only; the design puts `GridPiece` in `cell_block_range.h`,
  it gets its own header because it owns buffers), `grid_piece_cpu.cc`, `grid_piece_steps.h` (S, private),
  `grid_piece_test.cc` (its `SOURCES` list the allocation counter), `grid_piece_steps_test.cc` (label `cpu`)
- Modify: `cell_block_range.h` (includes `grid_piece.h`, so the design's spelling holds), `docs/glossary.md`
  (`grid piece`: one range of a whole grid with its border, loaded for a batch; not a face piece, not a memory
  piece. `piece map`: for each block of a grid piece, its index in the whole grid), build registration

**Interfaces:**
- Consumes: the table layouts (Task 7b), `ToGridBuffersArrays` and `BuildGridFromBuffers`
  (`grid_buffers_internal.h`, Tasks 6b and 9a), `StaggeredGrid`, `TakeBuffers` (Task 9a), `FindBorderBlocks` (Tasks 22a, 22b), `FindSortedBlock`.
- Produces (design 3.4): `GridPiece` with its accessors, both `BuildGridPiece` forms, `GridBuffers
  TakeBuffers(GridPiece&& piece)`, by design 5.8 "Pieces keep the fixed order". The parameter of both forms is
  named `buffers`. Both forms run on `whole.run_on()`; the span form requires a CPU whole grid (`assert`).

  ```cpp
  // Same as the span form, with the border in a buffer on whole.run_on() (a stacked FindBorderBlocks result).
  std::expected<GridPiece, GridFailure> BuildGridPiece(const StaggeredGrid& whole, CellBlockRange range,
                                                       const Buffer<CellBlockIndex>& border_blocks,
                                                       std::uint32_t border_block_count, GridBuffers& buffers);
  // GridPiece member: the piece map on grid().run_on(), [grid().counts().cell_blocks] entries used, ascending.
  const Buffer<CellBlockIndex>& block_in_whole_grid_buffer() const;
  ```
  `block_in_whole_grid()` (the span) is the map for a CPU grid piece and empty for a GPU one.
- Shared, in `grid_piece_steps.h` (the CPU form writes every entry through them; Task 25 runs them per thread):
  - `CellBlockIndex ToPiecePlace(const CellBlockIndex* border_blocks, std::uint32_t border_block_count,
    CellBlockRange range, std::uint32_t loaded_item)`: the index, in the grid piece, of loaded item
    `loaded_item` (items below `border_block_count` are border entries, the rest are the range's blocks in
    order). A border entry `i` below the range is at `i`; one above it at `i + range.block_count`; the range's
    block `k` at (border entries below the range, by binary search) `+ k`. Each item's place is computed
    alone, so the map needs no sort.
  - `std::uint32_t ToPieceNeighbour(const CellBlockIndex* map, std::uint32_t count, std::uint32_t
    whole_neighbour)`: binary search in the map; `kNoIndex` when the block is not loaded.
  - `FacePieceKind ComputePieceFaceKind(const GridTables& whole, const CellBlockIndex* map, std::uint32_t
    count, CellBlockIndex piece_block, int face_place_in_block)`: the whole grid's kind, except
    `kOutsidePiece` for a `kBetweenCells` face piece whose other cell's block is not loaded; a plane-4 place
    stays `kNotAPiece` when its high neighbour exists in the whole grid but is not loaded (design 5.8).
- A loaded block's neighbour that `ToPieceNeighbour` should find and does not (a block of the range missing
  from the map the same call wrote) is `kInternalFault`; `kBadInput` stays for the border list a caller passes.
- **Worked example for the header:** whole grid of blocks 0..9, range `{4, 3}`, border `{1, 2, 8}`: the piece
  map is `[1, 2, 4, 5, 6, 8]`, `range()` is `{2, 3}`, `range_in_whole_grid()` is `{4, 3}`.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ToPiecePlace, RangeBetweenBorderEntries_GivesAscendingMap)` (`grid_piece_steps_test.cc`): the worked
    example, each of the 6 items alone.
  - `TEST(ToPieceNeighbour, BlockNotLoaded_IsNoIndex)`.
  - `TEST(BuildGridPiece, KeepsWholeGridOrder)`: the 100 random grids, ranges from `BuildRanges(grid, 4)`,
    borders stacked twice (touching, four widths): `block_in_whole_grid` ascending; `range()` is a run and maps
    to `range_in_whole_grid()`; per block, cell states and face kinds equal the whole grid's except
    `kOutsidePiece` exactly where the other cell's block was not loaded; a plane-4 place whose high neighbour
    exists in the whole grid but was not loaded is `kNotAPiece`.
  - `TEST(BuildGridPiece, SameWeightsAsWholeGrid)`: 100 positions in range cells, all axes, all kept: the lists
    map to the whole grid's through `block_in_whole_grid` (place within block unchanged) and every weight
    field is byte-identical.
  - `TEST(BuildGridPiece, BlockSumsInPieceOrder_EqualWholeGrid)`: per-block partial = ordered sum over the
    block's active cells of `ComputeCellCenter(...).z * ComputeCellWidth(...)`; whole-grid total =
    `math::TreeSum` over the partials; for 3 cuts (`max_blocks_per_range` 1, 3, 7) partials computed in grid
    pieces are placed at `range_in_whole_grid` and tree-summed: identical bytes.
  - `TEST(BuildGridPiece, AboveBuffersRoom_IsTooManyCellBlocks)`;
    `TEST(BuildGridPiece, BorderBlockInsideRange_IsBadInput)`;
    `TEST(FindCellAt, InPieceOutsideLoadedBlocks_IsNoIndex)`; `TEST(BuildGridPiece, AnyPiece_AllocatesNothing)`.
  - `TEST(BuildGridPiece, BufferForm_EqualsSpanForm)`: the grids of `KeepsWholeGridOrder`, border from the
    buffer form of `FindBorderBlocks` on CPU buffers: the same tables, map and counts as the span form.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** (allocates nothing). `Read:` `<sec-cell-blocks>`.
- [ ] **Step 4: Commit.** Subject: `Build a grid piece that keeps the whole grid's block order`. New names:
  `GridPiece`, `BuildGridPiece` (two forms), `TakeBuffers(GridPiece&&)`.

### Task 25: Grid pieces on the GPU

**Kind:** device

**Files:**
- Create: `native/fluidity/grid/grid_piece_gpu.h` (private), `grid_piece_gpu.cu` (private),
  `grid_piece_gpu_test.cc` (label `gpu`; its `SOURCES` list the allocation counter)
- Modify: `grid_piece_cpu.cc` (dispatch on `whole.run_on()`, the GPU sequence under
  `#if defined(FLUIDITY_HAS_GPU)`), `grid_piece_steps.h`, `grid_piece_steps_test.cc`, build registration

**Interfaces:**
- Consumes: `ToPiecePlace`, `ToPieceNeighbour`, `ComputePieceFaceKind` (Task 24), `GridBuildStatus`,
  `ToFailureKey` (Task 7a), `ReadGrid` (Task 9c, tests), the GPU build (Task 11b); `ComputeLaunchSize`,
  `ApplyGpuFunction`, `CheckGpuWork`, `ToBufferError`.
- Shared per-item functions added to `grid_piece_steps.h`: `ComputePieceBorderFailureKey` (step 2),
  `ApplyPieceMapWriteAt` (step 3), `ApplyPieceBlockCopyAt` (step 4), `int ApplyPieceCellStateCopyAt(...)`
  (step 5; returns the level of an active cell, else `-1`), `std::uint32_t ApplyPieceFaceKindWriteAt(...)`
  (step 6; returns 1 for a face piece).
- Produces: the GPU-half functions `ApplyPieceStatusResetOnGpu`, `CheckPieceBorderOnGpu`, `WritePieceMapOnGpu`,
  `CopyPieceBlocksOnGpu`, `CopyPieceCellStatesOnGpu`, `WritePieceFaceKindsOnGpu`; the GPU path of the buffer
  form of `BuildGridPiece`; in `grid_piece_gpu.h`, `RunOnCallStatus
  ComputeLargestGridPieceStackOnGpu(std::uint64_t* bytes_per_thread)` and the "Stack:" sentence.
- **The parallel form** (no sort and no prefix sum: the border is already ascending): (1) CPU: `range` inside
  the whole grid, `range.block_count + border_block_count <= buffers.max_cell_blocks()` (else
  `kTooManyCellBlocks`, `cell` zero); reset the status. (2) One thread per border entry: ascending, outside
  `range`, below the whole block count, else a step-1 key with `kBadInput`. (3) One thread per loaded item
  (the count is known on the CPU): writes the map at `ToPiecePlace`. (4) One thread per block of the grid
  piece: copies the block id and translates its 36 neighbour entries with `ToPieceNeighbour`. (5) One thread
  per cell place: copies the state; an integer `atomicAdd` per active cell into its level's count. (6) One
  thread per face place, grid-stride like every other step: `ComputePieceFaceKind`; a thread adds up its
  face pieces and issues one integer `atomicAdd`. (7) One `ReadBuffer` of the status. Steps 3 to 6 return at
  once after a failure of step 2.

- [ ] **Step 1: Tests.** Label `cpu` (`grid_piece_steps_test.cc`):
  `TEST(BuildGridPiece, EveryThreadBlockAndThread_EqualsCpuPiece)`: steps 2 to 6 replayed on CPU buffers as
  Global Constraints "Replay tests" says; on Task 24's grids, ranges and stacked borders and the failing
  inputs below: tables, map, counts and failures equal the CPU form's byte for byte.
  Label `gpu` (whole grids built on both processors from one list; tables, maps, counts and failures read back
  go to `WriteResultDigest`):
  - `TEST(BuildGridPiece, OnGpu_EqualsCpuPiece)`: Task 24's grids, ranges and stacked borders (each processor
    finding its own border with the buffer form of `FindBorderBlocks`): `ReadGrid(piece.grid(), ...)` tables,
    the map (read back), version and counts equal the CPU grid piece's byte for byte.
  - `TEST(BuildGridPiece, OnGpuFailures_EqualCpuFailures)`: an unsorted border, a border block inside the
    range, a border index beyond the whole grid, too little room.
  - `TEST(BuildGridPiece, OnGpu_AllocatesNothing)`; `TEST(BuildGridPiece, OnGpuTwice_GivesIdenticalBytes)`.
  - `TEST(BuildGridPiece, OnGpuFunctions_KeepUnderTheDefaultStack)`: at most 1,024.
- [ ] **Step 2: Run** the gate. Expected: `PASS cuda`.
- [ ] **Step 3: Commit.** Subject: `Build grid pieces on the GPU in the whole grid's order`. Body: `GPU:
  compiled, not run`; the `gpu` tests above by name.

### Task 26a: Scene-level tests

**Kind:** integration

**Files:**
- Create: `native/fluidity/grid/grid_scene_test.cc` (label `cpu`), `grid_scene_gpu_test.cc` (label `gpu`;
  `SOURCES` also list `grid_functions_test_gpu.cu`); build registration

**Interfaces:**
- Consumes: every public grid function; `BuildSizeStepClosure` (Task 5); Task 20's runner.
- The scene, a domain box of liquid with a sloped surface: frame origin 0, width 1 m, 3 levels, the box of
  16 x 4 x 8 level-0 cells. A level-2 cell is *marked* when its centre `(x, y, z)` has `|z - (4 + 0.25 x)| < 1`:
  the distance is measured along `z`, not at right angles to the plane, so it is exact in doubles. The start
  list: a level-0 cell with no marked centre inside stays whole; one that has a marked centre is replaced by
  its 8 level-1 cells; each of those with a marked centre inside is replaced by its 8 level-2 cells. Then
  `BuildSizeStepClosure(list, 3)`. "The 1,000 positions" are from seed `20261119`, each inside a marked cell.

- [ ] **Step 1: Scene test** (label `cpu`): `TEST(StaggeredGrid, SlopedSurfaceInBox_PassesEveryCheck)`: build;
  `grid.counts()` equals `ComputeGridCounts(inputs)`; cut into ranges of 8 blocks; for every range build a
  grid piece with a stacked touching border; at the 1,000 positions, the grid piece's weights equal the
  whole grid's byte for byte and sum to 1 within `1e-14`.
- [ ] **Step 2: GPU test** (label `gpu`): `TEST(StaggeredGrid, SlopedSurfaceInBoxOnGpu_EqualsCpu)`: the same
  cell list put on the GPU; the GPU grid equals the CPU grid (`ReadGrid`, byte for byte); for every range of 8
  blocks, the GPU's stacked touching border and GPU grid piece equal the CPU's; `ComputeNearbyFaceWeights` run
  through Task 20's runner on the GPU grid piece at the 1,000 positions matches the CPU's within the
  CPU-versus-GPU tolerance (`CheckWithinTolerance`); everything read back goes to `WriteResultDigest`.
- [ ] **Step 3: Run** the gate. Expected: `PASS cuda`.
- [ ] **Step 4: Commit.** Subject: `Check the grid on a box of liquid with a sloped surface`. Body: `GPU: compiled, not
  run`; the `gpu` test by name.

### Task 26b: Comparison with the proof of concept

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/grid/grid_main_comparison_test_cases.h` (test-only, CPU only; formatted and
  rule-checked like any header), `grid_main_comparison_test.cc` (label `cpu`); build registration

**Interfaces:**
- Follows `contact/contact_main_comparison_test_cases.h`: the old code's outcomes are data in a header whose
  comment says how they were found; the comparison needs no code from `main`.
- **How the numbers are produced.** Nothing of `main` is built or checked out. `main` carries a Python
  reference of its uniform weights, `reference/src/fluidity_reference/basis.py` (`quadratic_stencil(position,
  h, offset)` returns the 27 nodes, weights and gradients), with the same formula as its C++
  (`native/src/sparse_mac_grid_3d.cc` lines 285-296). In a temporary directory outside both checkouts: `git
  show main:reference/src/fluidity_reference/basis.py > basis.py`, and a throwaway driver (not committed) run
  with `uv run --no-project --with numpy python driver.py`. For `h = 0.25` m, the 100 positions `p_i = {1.0 +
  0.0071 i, 1.0 + 0.0053 i, 1.0 + 0.0037 i}` (Y-up), `i = 0..99`, and each face axis (offset 0 along the axis
  and 0.5 along the two others; confirm against the C++ lines 266-310 by reading them), it prints each
  sample's node (three integers) and, with `%.17g`, its weight and gradient. If numpy cannot be fetched, stop
  and report; do not retype the formula.
- The header holds the 8,100 samples (100 x 3 x 27), one line each: position index, old axis, node, weight,
  gradient, as `main` gave them (Y-up); the commit of `main` they came from; and the driver's formula in its
  comment. The test turns them to Z-up with `new = {x, -z, y}` (old axes X, Y, Z become X, Z, Y) and computes
  a face position as `(node + offset) * h`.

- [ ] **Step 1: Produce the numbers** as above and write the header.
- [ ] **Step 2: Failing tests** (label `cpu`, supporting evidence only):
  - `TEST(ComputeNearbyFaceWeights, CompleteCase_MatchesMainWithinTolerance)`: the turned positions on a
    uniform 16^3 grid at width 0.25 m with origin `{0, -4, 0}`; faces matched by position. For every sample of
    `main` whose weight is not 0: the face is in the list, and weight and `weight_change` agree within `1e-15
    + 1e-13 |value|` (a plain relative bound fails where the exact value is 0, as the slope at `i = 0`, where
    the position lies on face planes); `offset` equals the face position minus the position within the same
    bound times `h`. For a sample of weight 0 the face is absent from the list or has weight within `1e-15`.
    All 8,100 samples are visited.
  - `TEST(BuildGrid, MainJunctionFixture_HasSamePiecesAsMain)`: `git show
    main:native/tests/test_adaptive_mac_topology_3d.cc`, `MakeJunction(3, true)` (lines 22-37, smallest width
    0.5 m, all 8 small cells present), turned to Z-up: 24 `kBetweenCells` face pieces, all 0.25 m2; 12 with
    centre distance 0.75, 12 with 0.5 (the old test's own assertions, lines 166-177).
- [ ] **Step 3: Run**, see them pass.
- [ ] **Step 4: Commit.** Subject: `Compare the grid's weights and face pieces with the proof of concept`. The
  body names the `main` commit the numbers came from.

### Task 27: Comment pass and hand-over

**Kind:** mechanical

**Files:**
- Modify: any file under `native/fluidity/grid/`, `docs/glossary.md`, `docs/method-paper/`, `docs/gpu-runs.md`

- [ ] **Step 1: Comment pass.** Reread every grid file top to bottom as a C++ programmer new to fluids: every
  comment is true of the final code, says why, and has a worked example where a contract is hard to picture
  (the two-size fixture, the complete case, the plane-4 rule, the corner fixtures); every word is in the
  glossary; `Uses`, `Used by` and `Read:` are correct; no comment or test name cites a plan, a task, a
  decision, an owner answer, a review or a date; "piece" is never bare where two senses could be meant.
- [ ] **Step 2: Fold** each change into its owning commit (`git commit --fixup`, `git rebase --autosquash`); run
  `--only format` and `--only rules` on each rewritten commit and the full gate on the tip; check that every
  commit still has its session note.
- [ ] **Step 3: Check the design section by section** (3.1 to 3.8, 5.1 to 5.9, 6, 7, 10) and record in the
  hand-over what was delivered differently. Known from this plan: the error text functions are in
  `grid_error_message.h`; `GridPiece` is in `grid_piece.h`; `CellBlockRange` and `GridFrame` are declared
  earlier than their design sections; the four value buffers are one template `GridValues`; the one-argument
  `ComputeGridCounts` counts on the sorted ids and never builds tables; `kTooManyCellBlocks` names no cell;
  `CheckSizeStepRule` does not check siblings; `FindCellAt` with `near` falls back to the search; the corner
  tie rule of Task 15; the CPU border search marks then appends; `ComputeCellSizeFactorTable` is shared
  and calls no library function; `FindNearestCommittedCell` is an added name; the comparison with `main` uses
  `main`'s Python reference, with samples of weight 0 handled apart; `SharedBuilderBuffers_OtherLiveGridUnchanged`
  does not call `FindCellAt` (it is built later) and its GPU form does not compare free bytes with the estimate
  (the reservation tests do); the sibling check walks the eighth-0 chain; `kInternalFault` replaces `kTooManyNearbyFaces` and `kSizeMismatch` maps to it; the grid has no stack entry held by the driver, with the three
  `gpu` tests that pin it.
- [ ] **Step 4: `docs/gpu-runs.md`:** add the grid's entry, result `compiled, not run`, with every `gpu` test of
  Tasks 10, 11b, 13c, 20, 21, 23, 25 and 26a by full name (`Suite.Name`), copied from the test files (`ctest -N
  -L gpu` lists them). Commit with subject `Record the grid's GPU status`.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the grid commits, the gate output, the new
  public names, the deviations of Step 3, and the statement that no grid device code has run.
