# Rebuild Mesh Solids (contact part) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a solid be a closed mesh of triangles in `native/fluidity/contact/`: the mesh type and its tree of
boxes, the builder and its checks, distance, normal and inside test, the first hit of a path on a mesh that does
not turn and on one that turns, the start rules and the slide in `MoveParticle` (concave corners, the hit limit),
`MoveParticles` on both processors with a mesh, and the method-paper section. This is the contact part of module
1b only (owner answer A82); the I/O, solids, step and add-on parts land with those modules.

**Architecture:** Three shared headers beside module 1's, in its layout: `mesh_shape.h` (types and constants),
`mesh_shape_steps.h` (private per-triangle and tree steps), `mesh_contact.h` (public functions). Include order:
`solid_shapes.h`, `mesh_shape.h` <- `mesh_shape_steps.h` <- `mesh_contact.h` <- `solid_contact_steps.h`;
`solid_contact_types.h` includes `mesh_shape.h` for the `meshes` field. No forward declarations. Two CPU-only
pairs: `mesh_shape_builder.h/.cc` and `mesh_shape_buffers.h/.cc`. No new GPU file: the mesh rides module 1's
`ApplyMoveParticleAt`. A mesh is never convex, so nothing of module 1's touching rule is used for it: the sweep
meets "the mesh grown by a radius" as intervals on convex pieces, and a resting particle is swept with half the
radius (design 5.4).

**Tech Stack:** as Foundation and contact. Nothing is added.

**Spec:** `docs/superpowers/specs/modules/01b-mesh-solids-design.md` (the contract; "the design" below),
`docs/superpowers/specs/modules/01-contact-design.md`, the contact plan
`docs/superpowers/plans/2026-11-14-rebuild-01-contact.md` ("module 1" below; its plan decisions P1 to P17 hold
here), and `docs/superpowers/specs/2026-11-13-production-rebuild-design.md`, all in the `main` checkout
`/Users/tychota/Documents/Code/openSource/FluidiTy`, untracked. Owner decisions: `.../modules/owner-answers.md`
(A80 to A94).

All paths below are relative to the worktree `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`
(branch `rebuild`). Nothing is ported: `main` has no mesh solid.

**What this plan assumes of module 1 at its Task 18** (re-check this list when module 1 finishes; a task that
names an item stops and reports if it is not so):

| # | From module 1 task | Assumed |
|---|---|---|
| H1 | 8, 8a (built) | `FindFirstHitOnSolid(inputs, solid_index, start, velocity_for_this_step, start_elapsed, duration, FirstHit*)` returns `ContactError`; one `switch (solid.shape)` with every case written out, then an assertion and `kBadInput`; `Sweep`, `OwnPath`, `ComputeOwnPath`, `ComputeDistanceForParticle(inputs, solid_index, pose, point, plate_side)` in `solid_contact_steps.h`. |
| H2 | 9 | A turning solid's first hit is a stepping search behind `FindFirstHitOnSolid` that reads the gap only through `ComputeDistanceForParticle`, decides "turning" only through `ComputeShapeTurningAngularVelocity`, stops when the gap is at most `tol / 4`, and reports `kTooCloseToCall` after `kMaxAdvanceCount` advances. |
| H3 | 10 | `MoveParticle(inputs, SolidContactResult*)` in `solid_contact.h`; a surface list of `kMaxHitsPerStep` entries, each the solid index and the normal recorded at the hit; no line skips "the solid just hit"; each re-sweep passes `duration = step_duration - elapsed` computed fresh; test-only `MoveParticleCase` and `BuildMoveParticleCases()` in `contact_test_cases.h`. |
| H4 | 11 | One private `MoveToRestGap(inputs, solid_index, pose, point)` in `solid_contact_steps.h` returning the moved point, used by step 2 and step 6. |
| H5 | 12 | The earlier-surface rule reads recorded normals and compares "approaches again" with `ComputeLargestApproachLeftAfterResponse`; `kSqueezedBetweenSolids` compares two solids. |
| H6 | 12a | Step 4 has one place where a hit position may be moved before the response (the rim move). |
| H7 | 13 | The carry of the rest of the step in a turning solid's axes is one private function of `solid_contact_steps.h` taking `inputs` and `solid_index`. |
| H8 | 14 | One helper in `contact_test_cases.h` checks "every end distance is at least `tol / 2`, and the naive control enters" for any list of solids. |
| H9 | 15 | `MoveParticlesInputs`, `ContactBuffers`, `MoveParticles` in `move_particles.h`; `static_assert(sizeof(Solid) == 200)` in `solid_shapes.h`. |
| H10 | 16 | `MoveParticlesView` (one struct with the scalar inputs), `ApplyMoveParticleAt(view, index)` in `move_particles_steps.h`; the tests `ApplyMoveParticleAt, EveryThreadBlockAndThread_SameAsCpuLoop`, `MoveParticles, SweepAndMoveCasesOnGpu_MatchCpu`, `SharedFormulas, OnGpu_MatchCpu` take their cases from `BuildMoveParticleCases()` and a fixed input list. |

## Global Constraints

Copied from Foundation (as module 1 states them; its plan has the full text of each):

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Commit rule: one thing per commit; strict CPU build and all tests pass in the debug and release presets; the
  CUDA target compiles in the container; clang-format and clang-tidy 23.1 clean; every touched file passes the
  `AGENTS.md` checklist; new public names listed in the commit message with a one-line reason; tests arrive with
  the code. The gate (`uv run python tools/check_commit.py`, seven `PASS` lines) runs at every commit.
- Foundation's structure (`AGENTS.md` "Structure"): the public function of an operation lives in `<name>_cpu.cc`;
  every GPU function is a thin loop over `FLUIDITY_CPU_AND_GPU` per-item functions of a shared header, and `cpu`
  tests run those functions for every thread block and thread, in launch order and in reverse, against the CPU
  loop. Buffers are pieces of the run's one `RunMemory`; an estimate entry stands for whole buffers.
- History on `rebuild` may be rewritten. A defect found later is fixed in the commit that introduced it.
- Commit messages carry no AI trailer; the owner is the author. After committing, write the commit's note with
  `uv run python tools/add_session_note.py` as module 1's plan says.
- Device code is "compiled, not run" until `docs/gpu-runs.md` records the owner's run. Never claim otherwise.
- Host C++23, exceptions and RTTI off in the library; device and shared headers C++20 subset; a shared header
  never includes `<expected>`, `<string>`, `<vector>` or anything that allocates.
- Device flags `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`; host `-ffp-contract=off`,
  `/fp:precise`. No fast-math.
- Z-up, right-handed. Naming: no `3D` suffix, no `Cuda`/`Gpu`/`Device`/`host` in public names, no bare `bool`
  parameter, no abbreviation. Verbs: `Compute`, `Build`, `Check`, `Find`, `Apply`, `To<Type>`, the stage verbs.
- 120 columns, two-space indent, `//` comments, braces on every control-flow body.
- Tests are `TEST(FunctionOrType, Situation_Outcome)`, one behaviour each, in `*_test.cc` beside the code.
- Deletion is the default: add a dependency, file or tool only when a task here names it.

Learned while building module 1 (every task respects them; reviewers check them):

- **Numbers.** Every expected number in a test is derived before it is written; each task says how its numbers
  were checked ("by hand" with the derivation in the test's comment, or "script": recomputed on 2026-11-15 with
  a short script when this plan was written). An implementer who finds a number wrong stops and reports.
- **Branches.** A task's tests make every branch of the code it adds win at least once, in more than one
  orientation (a solid turned by `math::Rotation({0.3, -0.5, 0.7})` and moved to `{3, -2, 5}`, "the turned
  pose") and with a moving solid (`linear_velocity {0.5, -0.25, 0.125}`, "the moving solid"). Each start rule
  has a test with `start_elapsed > 0` and a moving solid.
- **Mutants.** Tests are confirmed by mutation: after the tests pass, the implementer tries each mutant the
  task lists (one at a time, never committed), sees at least one named test fail, and writes the failing test of
  each in the commit message.
- **Signatures.** Shared functions take small structs by value, `Solid` and `MeshShape` by `const&`. No two
  adjacent `double` parameters (clang-tidy flags them): group them in a struct. Cognitive complexity at most 25
  per function: one function per piece, per step. Every `Requires:` has its `assert`.
- **Where things go.** Types and constants: `mesh_shape.h`. Private steps: `mesh_shape_steps.h` (not
  `solid_contact_steps.h`, already 875 lines with five tasks to come; plan decision). Public functions:
  `mesh_contact.h`. No forward declaration across headers.
- **Same bits.** Shared mesh code uses only `+ - * /`, `math::Sqrt`, `math::Abs`, `math::Min`, `math::Max` and
  comparisons, the operations `base/scalar_math.h` promises the same bits for. `math::Max` and `math::Min` drop
  a NaN: a test that must fail on NaN is written as a comparison that is false for NaN.
- **MSVC and GitHub's Xcode.** No `std::ranges::iota`; no comparison with a NaN in a constant expression; a
  test closes a file before deleting it. A C array where `std::array`'s members are needed on the device.
- **Signs after a response** are compared with `ComputeLargestApproachLeftAfterResponse`, never with 0.
- **Words.** Fixed by module 1 and used as defined: reach, search distance, swept distance, wall (of a
  cylinder), gap, rest gap, sweep, hit fraction, touching rule, approach left after the response. New words get
  glossary rows with a "never say" list in the task that introduces them (design 9).

Mesh solids' own:

- Design section 3 is the contract: names, fields, defaults and signatures are copied from it exactly.
- Constants (design 3.2): `kMaxMeshTriangles = 1U << 24`, `kMaxTrianglesPerEndBox = 4`, `kMaxMeshTreeDepth = 24`,
  `kMaxMeshHitsPerStep = 16`, `kMaxMoveOutCount = 4`, `kMeshNearStartInTolerances = 1.5`,
  `kMeshRestingSweepShare = 0.5`, `kMeshPieceOverlapShare = 0.125`, `kNoMesh = 0xFFFFFFFFU`. Check thresholds
  (design 5.2): `2^-20` (flat), `2^-6` rad (sharp), `-0.5` (back to back), `1024 tol`.
- Order: nearer box first, ties the first box; triangles in array order, ties the lower index; pieces in the
  order face, edge 0, 1, 2, vertex 0, 1, 2. No floating-point sum is made.
- `tol = kTestContactTolerance = 0x1p-30` in tests unless stated. Position tolerance `2 tol`; times and
  velocities `0x1p-44` relative; a normal at a hit within about `tol` of an edge or a vertex is compared within
  `1e-5` only (a rounding of `1e-15` m at a distance `tol` turns it by about `1e-6`).
- No test in this plan is known-open except the two timings of Task 17.

Plan decisions (the design left these open; the paper states M1 to M4):

- **M1 Pieces.** For a triangle `a, b, c` (counter-clockwise from outside), unit normal `n`, radius `r`,
  `o = kMeshPieceOverlapShare * r`: the face piece is `|Dot(n, p - a)| <= r` and `Dot(m_k, p - a_k) <= o` for
  `k = 0, 1, 2`, `m_k = Normalized(Cross(e_k, n))`, `e_k` the edge from corner `k` to corner `k + 1`; edge piece
  `k` is "within `r` of the line of `e_k`" and `-o <= Dot(Normalized(e_k), p - a_k) <= Length(e_k) + o`; vertex
  piece `k` is "within `r` of corner `k`". Each condition gives a `TimeInterval` (M2); a piece's interval is
  their intersection; no hit point is ever tested against a condition.
- **M2 Intervals.** `struct TimeInterval { double entry; double exit; }`, empty when `entry > exit`; "all time"
  is `{-infinity, +infinity}`. A condition `gap(t) = at_start + rate * t <= 0` (`struct LinearGap`): `rate == 0`
  gives all time when `at_start <= 0`, else empty; `rate < 0` gives `{at_start / -rate, +infinity}`; `rate > 0`
  gives `{-infinity, -at_start / rate}`. A condition `a t^2 + 2 b t + c <= 0` (`struct QuadraticGap`, `a >= 0`):
  `a == 0` gives all time when `c <= 0`, else empty; `b * b - a * c < 0` (or NaN: written so that NaN is empty)
  gives empty; else `s = Sqrt(b * b - a * c)`, `{(-b - s) / a, (-b + s) / a}`. A piece is met when
  `entry <= exit && exit >= 0 && entry <= duration`, at `math::Max(entry, 0)`.
- **M3 Start cases** (design 5.4): `at_start = ComputeDistanceToMesh(mesh, own_start, 2 * tol)`, then
  `enum class MeshStartCase { kFar, kHitAtStart, kResting }`: `distance > 1.5 * tol` is `kFar` (sweep radius
  `tol`); else `distance < 0.5 * tol` is `kHitAtStart`; else `Dot(normal, own_velocity) * duration < -tol` is
  `kHitAtStart`; else `kResting` (sweep radius `0.5 * tol`). With `tol == 0` (the naive control): always `kFar`,
  radius 0, where a piece is its triangle, its edges' lines and its corners and only a crossing is met.
- **M4 The guarantee tested** (design 5.4): with `r` the sweep's radius, a reported hit is no later than the
  first moment the path is within `r - tol / 8` of the mesh; the mesh is within `r + tol / 8` of the reported
  hit position; a path reported free never comes within `r - tol / 8`. Checked against exact arithmetic (Task 3).
- **M5 The test-side exact helper.** `contact/exact_rational_test_support.h` (CPU-only, tests only):
  `class ExactRational` (a signed whole number over a positive whole number, each of any size, built exactly
  from a `double`; `+ - * /` and comparisons; no rounding anywhere) and exact squared distances from a point and
  from a segment to a triangle. It is the reference that cannot miss.
- **M6 Search counts.** The private searches take a nullable `MeshSearchCounts*` (`boxes_visited`,
  `most_boxes_kept`, `triangles_tested`), so tests can check cost and the stack bound without a second walk.
- **M7 Morton code.** 21 bits per axis, `x` lowest: each centre coordinate mapped to `[0, 2^21)` by
  `Floor((c - low) / (high - low) * 2097152)` clamped to `2097151` (an axis with `high == low` maps to 0), ties
  by input triangle index; `std::stable_sort`.
- **M8 Test meshes** (`contact/mesh_test_cases.h`, CPU-only, integer formulas, each returning vertices and
  triangles): `BuildCubeMesh(half_size)` (8 vertices, 12 triangles; every face split along the diagonal from
  its low-low corner to its high-high corner); `BuildIcosphereMesh(radius, halvings)` (`20 * 4^halvings`
  triangles, new vertices put on the sphere); `BuildSlabMesh(half_size)` (a cube mesh with its own half sizes);
  `BuildGrooveMesh(half_angle_tangent)` (the prism of cross-section `(-1,-1), (1,-1), (1,1), (0,0), (-1,1)` in
  `x, z` for tangent 1, `y` from -1 to 1; the notch's upper corners at `x = -+tangent`); `BuildTubeMesh(sides,
  inner_radius, outer_radius, half_length)` (axis `y`; the middle of one inner facet at the lowest point);
  `BuildBowlMesh()` (half sphere shell, radius 0.5, 0.05
  thick, 5120 inner triangles); `BuildTorusMesh(0.5, 0.15, 64, 32)`; `BuildLMesh()` (the cross-section
  `(0,0), (1,0), (1,0.4), (0.4,0.4), (0.4,1), (0,1)` in `x, y`, `z` from -0.5 to 0.5); `BuildFinMesh(thickness)`
  (a slab `1 x 1 x thickness`). `BuildTestMeshShape(mesh)` builds it with `SelfCrossingCheck::kSkip` and
  asserts success.

## Before you start

Module 1 (contact) is complete on `rebuild` through its Task 18, and the table H1 to H10 above holds. Foundation
is complete. No other module exists. Prove it from the worktree:

```bash
uv run python tools/check_commit.py
```

Expected: seven `PASS` lines. Then read `AGENTS.md` "Structure", module 1's hand-over message, and
`native/fluidity/contact/solid_contact_steps.h` from top to bottom. If an H row does not hold, stop and report.

## Review Focus

1. **A path through a shared edge or vertex must be caught by a piece.** Triangles meet at edges and vertices;
   a first hit that finds a candidate on one surface and then tests the hit point against another with a
   separately rounded test lets such paths through (module 1's cylinder did, in about 5 % of paths within a few
   rounding units of the rim). Here every piece is an interval intersection and neighbouring pieces overlap by
   `r / 8` (M1). Pinned in Task 8, `FindMeshEntry, PathsThroughSharedEdgesAndVertices_NeverLeakAgainstExactArithmetic`,
   and by Task 8's mutants that reintroduce candidate-then-validate.
2. **A particle resting on a flat face made of several triangles slides freely.** With one radius it would sit on
   the surface of the grown mesh and every shared edge would be decided by the last bits. Pinned in Task 9,
   `FindFirstHitOnMesh, RestingSlideAcrossSharedEdges_IsFree`, and Task 12,
   `MoveParticle, MeshCubeFrictionOnTopFace_SlidesAcrossTheDiagonalsWithOneHit`.
3. **The response after a mesh hit uses the normal recorded at the hit, not at the moved position.** In a
   right-angled corner the moved position is equally near two faces and the nearest triangle is chosen by
   rounding. Pinned in Task 12, `MoveParticle, RightAngleGroove_HitsBothSidesAndRests` (`hit_count` exactly 2).
4. **The inside test at a vertex and an edge.** A point nearest a vertex gets its sign from the angle-weighted
   vertex normal, not from one triangle's. Pinned in Task 6,
   `ComputeDistanceToMesh, PointsNearestAVertexOfAConcaveMesh_SignMatchesExactArithmetic`.
5. **The search's fixed array.** Twenty-four indexes must hold the walk for the largest mesh. Pinned in Task 2,
   `ComputeMeshTreeDepth, UpToTheLargestMesh_NeverAboveTwentyTwo`, and Task 6,
   `ComputeDistanceToMesh, AnyPoint_KeepsAtMostTreeDepthBoxes`.
6. **A mesh among other solids on the GPU path.** The device code is not run; the per-particle function must be
   shown on the CPU for every thread block and thread with a mesh in the solids. Pinned in Task 16,
   `ApplyMoveParticleAt, MeshCasesEveryThreadBlockAndThread_SameAsCpuLoop`.

---

### Task 1: Method paper section "Mesh solids"

**Kind:** mechanical

Load the `typesetting-with-typst` and `drawing-with-cetz` skills first.

**Files:**
- Modify: `docs/method-paper/fluidity-method.typ`, `docs/method-paper/references.typ`
- Create: `docs/method-paper/figures/contact_mesh_solids.typ`

**Interfaces:**
- Produces labels (design 8): `<sec-mesh-solids>` (after `<sec-round-solids>`, before `<sec-contact-response>`);
  `<eq:mesh-distance>`, `<eq:mesh-piece-interval>`; `<fig-mesh-distance>`, `<fig-mesh-grown-triangle>`,
  `<fig-mesh-resting-sweep>`; `<proc-mesh-distance>`, `<proc-mesh-first-hit>`.

- [ ] **Step 1: Text**, in the outline of design 8, from design 5.1 to 5.9, stating M1 to M4 in plain words.
  `<proc-mesh-distance>`: 1 walk the tree, nearer box first 2 nearest point of each triangle of an end box
  3 outward direction and sign 4 distance and normal. `<proc-mesh-first-hit>`: 1 start case 2 radius 3 candidate
  triangles from the tree 4 interval of each piece 5 earliest entry 6 distance at the hit. Code comments
  `// Step N:` in Tasks 6, 8 and 9 follow these numbers. One worked example each: the cube of half size 0.5,
  the path from `{-1, -1, 0}` at `{1, 1, 0}`: hit at `0.5 - tol / Sqrt(2)` through the edge piece; a particle at
  `{-0.4, 0, 0.5 + tol}` sliding at `{5, 0, 0}`: the resting sweep meets nothing. The thin-part paragraph states
  the limit and the finding of design 5.9, and that nothing is built for thinner walls.
- [ ] **Step 2: Citations.** Add to `references.typ`, as it formats entries, Baerentzen and Aanaes 2005, Ericson
  2005, Hart 1996, Wang et al. 2021, and cite them where design 8 says; replace `<sec-first-hit>`'s "citation
  needed" for the stepping search by Hart 1996. Never write a DOI or URL in the paper's text.
- [ ] **Step 3: Figures** as design 8 describes them, with `figure_style.typ`.
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`:
  exit 0, no warning; read every new page. Run the gate.
- [ ] **Step 5: Commit.** Subject: `Explain mesh solids in the method paper`.

### Task 2: The mesh shape among the solids, its types and its errors

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/contact/mesh_shape.h`, `mesh_shape_test.cc`
- Modify: `solid_shapes.h`, `contact_error.h`, `contact_error_message.h`, `contact_error_message.cc`,
  `solid_shapes_test.cc`, `contact_error_test.cc`, `CMakeLists.txt`, `docs/glossary.md`

**Interfaces:**
- Consumes: H9 (`static_assert(sizeof(Solid) == 200)`; if absent, add it here and say so).
- Produces: design 3.1 (the `solid_shapes.h` and `contact_error.h` parts) and 3.2 exactly. `CheckSolid`'s `kMesh`
  case without the table. "Requires: no solid is a `kMesh`" with its assertion on the three-argument
  `ComputeDistanceToSolid`, `ComputeSolidVelocitySample`, `ComputeContactScale`; their release behaviour (the box
  of `half_size`) is the `kBox` branch reached by a shared helper, not a second copy.

- [ ] **Step 1: Failing tests** (`cpu`):

  ```cpp
  TEST(SolidShape, Mesh_IsValueSixAndItsFaceIsNine)             // static_cast values; sizeof(Solid) == 200;
                                                                // offsetof(Solid, mesh_index) == 196
  TEST(ContactFailure, MeshProblem_SitsInExistingPadding)       // sizeof unchanged; offsetof(mesh_problem) == 2
  TEST(ToName, EveryMeshProblem_IsSnakeCaseOfEnumerator)        // kNotClosed -> "not_closed"; all nine unique
  TEST(ToMessage, EveryMeshProblem_IsOneSentence)               // not empty, ends with '.', no newline
  TEST(CheckSolid, MeshWithoutIndex_IsBadInput)                 // mesh_index kNoMesh; also half_size 0 and NaN
  TEST(CheckSolid, MeshTurningTooFar_IsSolidTurnsTooFar)        // angular {0,0,2}, step 1: close call set
  TEST(MeshTreeBox, Layout_IsThirtyTwoBytes)                    // and sizeof(MeshShape) == 120
  TEST(ComputeMeshTreeDepth, SmallCounts_MatchHalving)          // 0,1,4 -> 0; 5 -> 1; 8 -> 1; 9 -> 2; 12 -> 2;
                                                                // 1280 -> 9; 20480 -> 13; 1310720 -> 19
  TEST(ComputeMeshTreeDepth, UpToTheLargestMesh_NeverAboveTwentyTwo)  // every count to 4096 is non-decreasing;
                                                                // 2^k - 1, 2^k, 2^k + 1 for k to 24 are <= 22;
                                                                // 2^24 is exactly 22
  ```
  Numbers: the depths by hand (halve, keeping the larger half, until at most 4) and by script.
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** Glossary rows (design 9): `mesh solid`, `tree of boxes` and `end box` (never say:
  BVH, node, leaf), `wall (of a mesh)` added to the existing `wall` row, `closed (mesh)`.
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** `kMesh` inserted before `kSphere`; `mesh_index` as
  `std::uint64_t`; `ComputeMeshTreeDepth` halving with the smaller half.
- [ ] **Step 5: Commit.** Subject: `Add the mesh shape to the solids and name what can be wrong with a mesh`.
  New names: the 3.1 and 3.2 lists.

### Task 3: Exact arithmetic for the tests that must not miss

**Kind:** numerics

**Files:**
- Create: `native/fluidity/contact/exact_rational_test_support.h`, `exact_rational_test.cc`
- Modify: `native/fluidity/contact/CMakeLists.txt`

**Interfaces:**
- Produces (M5; test-only, CPU-only, namespace `fluidity::contact`): `class ExactRational` with
  `explicit ExactRational(double)` (asserts finite), `+ - * /` (division by zero asserts), `==`, `<`, `<=`;
  `struct ExactPoint { ExactRational x, y, z; }` and `ToExactPoint(Vec3)`;
  `ExactRational ComputeExactSquaredDistance(ExactPoint point, const std::array<ExactPoint, 3>& triangle)`;
  `ExactRational ComputeExactSquaredDistance(ExactPoint segment_start, ExactPoint segment_end, const
  std::array<ExactPoint, 3>& triangle)` (the smallest over the segment: 0 when it crosses the triangle, else the
  least of the two ends against the triangle and the segment against each edge); `ExactPoint
  ComputeExactPointOnPath(Vec3 start, Vec3 velocity, double elapsed)` (`start + velocity * elapsed`, exact).
  Whole numbers are a sign and a `std::vector<std::uint32_t>` of base `2^32` digits; a `double` is its 53-bit
  whole number times a power of two, exact. No reduction of fractions is needed at these sizes.

- [ ] **Step 1: Failing tests** (`cpu`):

  ```cpp
  TEST(ExactRational, SumOfTenthThreeTimes_IsNotThreeTenthsAsDoubles)  // 0.1 + 0.1 + 0.1 != 0.3 exactly, and
                                                                // equals ExactRational(0.1) * ExactRational(3)
  TEST(ExactRational, ProductOfLargeAndSmall_KeepsEveryBit)     // (2^30 + 1) * (2^-30 + 2^-82) - 1 ==
                                                                // 2^-30 + 2^-52 + 2^-82, exactly
  TEST(ExactRational, DivisionThenMultiplication_ReturnsTheStart)     // (1 / 3) * 3 == 1
  TEST(ComputeExactSquaredDistance, PointAboveFaceEdgeAndVertex_MatchHandValues)
      // triangle (0,0,0), (1,0,0), (0,1,0): point (0.25, 0.25, 2) -> 4; (0.5, -3, 4) -> 25; (-3, -4, 12) -> 169
  TEST(ComputeExactSquaredDistance, SegmentThroughTheTriangle_IsZero) // (0.25,0.25,1) to (0.25,0.25,-1)
  TEST(ComputeExactSquaredDistance, SegmentPassingAnEdge_IsTheEdgeGapSquared)
      // (0.5, -3, -1) to (0.5, -3, 1) -> 9; (2, 2, 3) to (3, 3, 3): nearest the edge from (1,0,0) to (0,1,0) at
      // its middle (0.5, 0.5, 0): squared distance 1.5^2 + 1.5^2 + 9 = 13.5
  TEST(ComputeExactSquaredDistance, OneRoundingUnitApart_OrdersCorrectly)
      // point (0.25, 0.25, h) for h = 1 and h = nextafter(1, 2): the second is strictly larger
  ```
  Numbers: by hand (each is a 3-4-5 or 5-12-13 length or a sum written above).
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** Schoolbook multiplication; comparison by cross-multiplying. The closest-point
  cases follow Ericson's region form with exact comparisons, so candidate-then-validate is safe here.
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** drop the carry of the highest digit in the product; the
  segment distance without the crossing case; without the segment-against-edge case.
- [ ] **Step 5: Commit.** Subject: `Add exact arithmetic for tests that compare with rounding-free answers`.

### Task 4: Build a mesh's arrays: order, tree, neighbours, vertex normals

**Kind:** numerics

**Files:**
- Create: `native/fluidity/contact/mesh_shape_builder.h`, `mesh_shape_builder.cc`, `mesh_shape_builder_test.cc`,
  `mesh_test_cases.h`
- Modify: `native/fluidity/contact/CMakeLists.txt` (`mesh_shape_builder.cc` in `target_sources`), `docs/glossary.md`

**Interfaces:**
- Consumes: Task 2.
- Produces: `MeshShapeArrays`, `ToMeshShape`, `SelfCrossingCheck`, `BuildMeshShape` (design 3.4) with steps 1,
  3 to 6 and 8 of design 5.2 and checks 1 to 3 (`kEmpty`, `kTooManyTriangles`, `kNotClosed`,
  `kFacesTurnedBothWays`, `kBadInput`); the measures and checks 4 to 7 arrive in Task 5 (until then
  `thinnest_wall` is +infinity and `smallest_triangle_height` is computed). The test meshes of M8. M7.

- [ ] **Step 1: Failing tests** (`cpu`):

  ```cpp
  TEST(BuildMeshShape, Cube_HasTwelveTrianglesSevenBoxesAndHalfSize)   // tree_box_count 7 (script), half_size
                                                                // {0.5,0.5,0.5}; the turned-and-scaled cube
                                                                // {1,2,3}: half_size {1,2,3}
  TEST(BuildMeshShape, AnyTestMesh_EveryBoxContainsItsTriangles)       // each mesh of M8: float bounds contain
                                                                // every vertex of every triangle below the box;
                                                                // end boxes hold 1 to 4; inner boxes' children
                                                                // adjacent; deepest end box at
                                                                // ComputeMeshTreeDepth(triangle_count)
  TEST(BuildMeshShape, Neighbours_AreMutual)                    // triangle t across its edge k is u, and u has t
                                                                // across the same two vertices
  TEST(BuildMeshShape, CubeVertexNormal_PointsAlongTheDiagonal) // each within 1e-15 of +-{1,1,1} / Sqrt(3):
                                                                // angle weights, not triangle counts (a corner
                                                                // has 1, 2 or 3 triangles per face)
  TEST(BuildMeshShape, CubeInsideOut_IsTurnedOver)              // same arrays as the cube's, byte for byte
  TEST(BuildMeshShape, SameInputTwice_SameBytes)                // icosphere, 3 halvings
  TEST(BuildMeshShape, NoTriangle_IsEmpty)
  TEST(BuildMeshShape, CubeLessOneTriangle_IsNotClosedWithThreeEdges)  // magnitude 3
  TEST(BuildMeshShape, OneTriangleTurnedOver_IsFacesTurnedBothWays)
  TEST(BuildMeshShape, IndexOutOfRangeOrNaNVertex_IsBadInput)
  TEST(BuildIcosphereMesh, ThreeHalvings_Has1280TrianglesAnd642Vertices)  // script
  ```
  Numbers: 7 boxes for 12 triangles (12 -> 6, 6 -> 3, 3: 1 + 2 + 4), 642 vertices (`n / 2 + 2`): by hand and
  script. The vertex normal: a cube corner's three face angles sum the same on each face (90 degrees each, as
  one triangle of 90 or two of 45), so the weighted sum is along the diagonal.
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** One function per step of design 5.2; boxes rounded outward with
  `std::nextafter` in `float`. Glossary rows: `triangle`, `vertex`, `neighbours`, `vertex normal`.
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** boxes rounded to nearest; vertex normals weighted by
  triangle count; the split by `floor(n / 2)` first; Morton ties by code only (`std::sort`).
- [ ] **Step 5: Commit.** Subject: `Build a mesh's triangles into the arrays and the tree the solver reads`.
  New names: `MeshShapeArrays`, `ToMeshShape`, `SelfCrossingCheck`, `BuildMeshShape`.

### Task 5: The checks of a mesh and its thinnest wall

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/mesh_shape_builder.cc`, `mesh_shape_builder.h`, `mesh_shape_builder_test.cc`,
  `mesh_test_cases.h`
- Create: `native/fluidity/contact/mesh_contact.h` (with `CheckMeshShape` only), `mesh_contact_test.cc`

**Interfaces:**
- Consumes: Task 4.
- Produces: checks 4 to 6 of design 5.2 in `BuildMeshShape` (`kTriangleTooFlat` relative part, `kEdgeTooSharp`,
  `kCrossesItself`), the measure of check 7 (`thinnest_wall`, `thinnest_wall_point`), and
  `CheckMeshShape(mesh, contact_tolerance, failure)` for the two tolerance checks (`kTriangleTooFlat` absolute
  part, `kPartTooThin`). The rays of check 7 and the pairs of check 6 walk the tree on the CPU inside the `.cc`.

- [ ] **Step 1: Failing tests** (`cpu`):

  ```cpp
  TEST(BuildMeshShape, SliverBelowTwoToMinusTwentyOfItsEdge_IsTriangleTooFlat)  // a slab 1 x 1 x 2^-21: its
                                                                // side triangles have height 2^-21 of edge 1
  TEST(BuildMeshShape, SliverAtTwoToMinusNineteen_Passes)       // slab 1 x 1 x 2^-19
  TEST(BuildMeshShape, WedgeOfTwoToMinusSevenRadians_IsEdgeTooSharp)   // a prism with a triangular
                                                                // cross-section (0,0), (1,0), (1, tan(2^-7))
  TEST(BuildMeshShape, WedgeOfTwoToMinusFiveRadians_Passes)
  TEST(BuildMeshShape, TwoCubesThroughEachOther_IsCrossesItself)       // cube half 0.5 at 0 and at {0.5,0.5,0.5}
                                                                // in one mesh; with kSkip it builds
  TEST(BuildMeshShape, TwoCubesApart_Passes)                    // the second at {3,0,0}
  TEST(BuildMeshShape, Slab_ThinnestWallIsItsThickness)         // BuildSlabMesh({1,1,0.01}): 0.02 within
                                                                // 0x1p-48, point on a large face (|z| == 0.01);
                                                                // the same mesh turned: the same value
  TEST(BuildMeshShape, Cube_ThinnestWallIsItsWidth)            // 1.0 within 0x1p-48 (opposite faces, Dot -1)
  TEST(BuildMeshShape, RegularTetrahedron_HasNoThinWall)        // vertices (1,1,1), (1,-1,-1), (-1,1,-1),
                                                                // (-1,-1,1): every pair of faces has Dot -1/3,
                                                                // above -0.5: +infinity, point {0,0,0}
  TEST(BuildMeshShape, FortyFiveDegreeWedge_IsThinTowardItsEdge)  // prism (0,0), (1,0), (0,1), y from -5 to 5:
                                                                // a leg against the slope has Dot -Sqrt(0.5):
                                                                // thinnest_wall <= 1/3 + 1e-12 and > 0
  TEST(BuildMeshShape, Fin_ThinnestWallIsOneTenthOfAMillimetre) // BuildFinMesh(1e-4): 1e-4 within 0x1p-60
  TEST(CheckMeshShape, FinOf512Tolerances_IsPartTooThin)        // BuildFinMesh(512 * tol): kMeshNotUsable,
                                                                // mesh_problem kPartTooThin, magnitude 512 tol
  TEST(CheckMeshShape, FinOf2048Tolerances_Passes)
  TEST(CheckMeshShape, TriangleHeightOf512Tolerances_IsTriangleTooFlat)  // slab 2^-11 x 2^-11 x 512 tol: also a
                                                                // thin wall; the flat check is reported first
  ```
  Numbers: by hand, with the normals written in each test's comment. A wall is counted when the first triangle
  an inward ray meets faces the other way by `Dot <= -0.5`, that is a wedge of 60 degrees or sharper; so a
  45 degree wedge is thin near its edge (the highest sample point of a leg triangle is at two thirds of the
  leg, where the slope is one third away), and a regular tetrahedron (faces at 70.5 degrees) has no wall.
- [ ] **Step 2: Run.** Expected: the new tests fail.
- [ ] **Step 3: Implement.** The first failure in check order, lowest triangle index. Glossary: the `wall (of a
  mesh)` row gains `thinnest wall`.
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** the back-to-back test as `Dot < 0` (the tetrahedron); one ray
  per triangle
  from the centre only (add a test mesh whose thin spot is under a half-way point if no test fails: a slab with a
  thin step under one corner region); `kPartTooThin` compared with `512 tol`; self-crossing pairs that share a
  vertex not skipped (the cube must then fail).
- [ ] **Step 5: Commit.** Subject: `Check that a mesh is usable and measure its thinnest wall`.
  New names: `CheckMeshShape`.

### Task 6: Distance, normal and the inside test of a mesh

**Kind:** numerics

**Files:**
- Create: `native/fluidity/contact/mesh_shape_steps.h`, `mesh_distance_test.cc`
- Modify: `native/fluidity/contact/mesh_contact.h`, `CMakeLists.txt`, `docs/glossary.md`

**Interfaces:**
- Consumes: Tasks 3, 4.
- Produces: `ComputeDistanceToMesh` (design 3.3, 5.3). Private steps in `mesh_shape_steps.h`:
  `struct TriangleCorners { Vec3 a, b, c; }`, `ReadTriangleCorners(mesh, triangle_index)`;
  `enum class TrianglePart : std::uint8_t { kFace, kEdge, kVertex }`;
  `struct NearestOnTriangle { Vec3 point; double squared_distance; TrianglePart part; std::uint8_t part_index; }`,
  `FindNearestPointOnTriangle(TriangleCorners corners, Vec3 point)`;
  `ComputeSquaredDistanceToBox(MeshTreeBox box, Vec3 point)`; `struct MeshSearchCounts` (M6);
  `FindNearestTriangle(mesh, own_point, largest_distance, MeshSearchCounts* counts)`;
  `ComputeOutwardDirection(mesh, triangle_index, NearestOnTriangle nearest)`.

- [ ] **Step 1: Failing tests** (`cpu`; generator P11):

  ```cpp
  TEST(FindNearestPointOnTriangle, EachOfSevenRegions_GivesItsPart)   // triangle (0,0,0), (1,0,0), (0,1,0): the
                                                                // three points of Task 3's test and four more,
                                                                // one per remaining edge and vertex; squared
                                                                // distances 4, 25, 169, and each equal to
                                                                // ComputeExactSquaredDistance within 0x1p-50
                                                                // relative; the same triangle turned and moved
  TEST(ComputeDistanceToMesh, CubeAtTenThousandPoints_MatchesTheBoxOrTheEuclideanDistance)
      // points 1.5 * {NextSigned() x 3}: with q = max(|p| - 0.5, 0) per axis: Length(q) when it is not zero
      // (outside), else the box's (largest face gap, negative); within 0x1p-48; the sign agrees with the box
      // at every point; normals equal the box's in face regions within 1e-12
  TEST(ComputeDistanceToMesh, IcosphereThreeFourFiveHalvings_ConvergesToTheSphere)
      // radius 0.5; along each triangle centre's direction at radii 0.4, 0.5, 0.52, 0.6: |d_mesh - (radius -
      // 0.5)| <= 1.02 * longest_edge^2 / 3 (that is longest_edge^2 / (6 r)); measured 2.264e-3, 5.67e-4, 1.42e-4;
      // each at least 3.9 times smaller than the last
  TEST(ComputeDistanceToMesh, PointOnAFaceAnEdgeAndAVertex_HasTheDegenerateNormal)
      // cube: {0.2, 0.1, 0.5} -> {0,0,1}; {0.5, 0.5, 0.1} -> {1,1,0}/Sqrt(2); {0.5,0.5,0.5} -> {1,1,1}/Sqrt(3),
      // each within 1e-15; distance exactly 0
  TEST(ComputeDistanceToMesh, PointsNearestAVertexOfAConcaveMesh_SignMatchesExactArithmetic)
      // BuildGrooveMesh(1) and BuildLMesh(): 2,000 points within 0.05 of each mesh vertex (P11): the sign equals
      // the parity of exact crossings of the segment from the point to {7, 11, 13} (outside) with the triangles
      // (ComputeExactSquaredDistance == 0 counts a crossing; a case whose segment touches an edge exactly is
      // redrawn), and |distance|^2 equals the least exact squared distance within 0x1p-48 relative
  TEST(ComputeDistanceToMesh, EqualDistanceToTwoTriangles_KeepsTheLowerIndex)   // cube, {0, 0, 0.9} lies over
                                                                // the top face's diagonal: triangle_index is the
                                                                // lower of the two
  TEST(ComputeDistanceToMesh, BeyondLargestDistance_ReturnsItAfterOneBox)  // cube, point {5,0,0}, largest 1:
                                                                // distance 1, zero normal, boxes_visited 1
  TEST(ComputeDistanceToMesh, AnyPoint_KeepsAtMostTreeDepthBoxes)  // icosphere 5 halvings (depth 13) and the
                                                                // torus: 10,000 points: most_boxes_kept <=
                                                                // ComputeMeshTreeDepth(triangle_count)
  ```
  Numbers: the icosphere bounds by script (the same subdivision), the rest by hand. The 1.02 covers the
  difference between the longest edge's triangle and the worst centre.
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement** `<proc-mesh-distance>`. The walk keeps a C array of `kMaxMeshTreeDepth` box indexes
  and asserts before every push; it descends into the nearer child and pushes the other only when its box
  distance is below the best. Squared distances until the end; one `Sqrt`. Glossary: `largest distance`.
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** the sign from the triangle's own normal at a vertex and
  at an edge; ties keep the later triangle; farther child first (the counts test or the stack test must fail,
  results must not); `>=` for `>` in the skip (the tie test).
- [ ] **Step 5: Commit.** Subject: `Compute the distance from a point to a mesh and which way is out`.
  New names: `ComputeDistanceToMesh`.

### Task 7: The point functions with a mesh among the solids

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/mesh_contact.h`, `solid_contact_types.h`, `solid_contact_steps.h`,
  `mesh_contact_test.cc`, `find_first_hit_test.cc`

**Interfaces:**
- Consumes: Task 6; H1.
- Produces: `ToSolid(const MeshShape&, ...)`, the mesh-aware `ComputeDistanceToSolid`,
  `ComputeSolidVelocitySample`, `ComputeContactScale`, `CheckSolid` (design 3.3); `SolidContactInputs::meshes`
  (last field); `ComputeDistanceForParticle` calls the mesh-aware form with `inputs.meshes`. The mesh-aware
  forms call module 1's functions for the six shapes (no second copy of a formula).

- [ ] **Step 1: Failing tests** (`cpu`):

  ```cpp
  TEST(ToSolid, Mesh_CarriesIndexPoseMotionAndHalfSize)         // BuildSlabMesh({1,2,3}), index 4: fields equal
  TEST(ComputeDistanceToSolid, MeshCubeInTurnedPose_EqualsTheBoxInFaceRegions)  // 1,000 points; distance within
                                                                // 0x1p-44, world normal within 1e-12, face
                                                                // kMeshTriangle
  TEST(ComputeDistanceToSolid, SixShapesWithEmptyMeshes_SameBitsAsTheFormWithout)  // one solid per shape
  TEST(ComputeSolidVelocitySample, MeshCubeMovingAndTurning_GivesSurfaceVelocityAtTheNearestPoint)
      // cube at origin, linear {1,0,0}, angular {0,0,2}, elapsed 0, point {0.6, 0.1, 0}, search 0.5:
      // has_solid, distance 0.1, velocity {1,0,0} + Cross({0,0,2}, {0.5,0.1,0}) = {0.8, 1, 0}
  TEST(ComputeSolidVelocitySample, FlatSolidNearerThanMesh_MeshCostsOneBox)   // floor at distance 0.01, mesh
                                                                // 3 away, search 1: the floor wins; (counts
                                                                // read through FindNearestTriangle in the test
                                                                // with the same largest_distance: 1 box)
  TEST(ComputeContactScale, MeshSlab_GivesReachAndThinnestWall) // slab {1,2,0.01} at {10,0,0}: largest_coordinate
                                                                // 10 + Length({1,2,0.01}); thinnest_solid 0.02
  TEST(CheckSolid, MeshIndexBeyondTable_IsBadInput)             // and half_size not the shape's: kBadInput
  TEST(ComputeDistanceForParticle, MeshInInputs_UsesTheMesh)    // inputs.meshes set: equals the mesh-aware form
  ```
  Numbers: by hand (`Cross({0,0,2}, {0.5,0.1,0}) = {-0.2, 1, 0}`).
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** `solid_contact_types.h` includes `mesh_shape.h`; `solid_contact_steps.h` includes
  `mesh_contact.h`. Every header's `Uses:` and `Owns:` lines follow.
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** the sample's `largest_distance` not lowered to the best
  found; the world normal left in own axes; `thinnest_solid` from `half_size` only.
- [ ] **Step 5: Commit.** Subject: `Answer distance and surface velocity for a mesh among the solids`.
  New names: the four mesh-aware forms, `ToSolid` for a mesh, `SolidContactInputs::meshes`.

### Task 8: The first time a path meets a mesh grown by a radius

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/mesh_shape_steps.h`, `CMakeLists.txt`, `docs/glossary.md`
- Create: `native/fluidity/contact/mesh_entry_test.cc`

**Interfaces:**
- Consumes: Tasks 3, 6.
- Produces (private, `mesh_shape_steps.h`; M1, M2): `TimeInterval`, `LinearGap`, `QuadraticGap`,
  `ComputeInterval(LinearGap)`, `ComputeInterval(QuadraticGap)`, `IntersectIntervals(TimeInterval, TimeInterval)`;
  `struct OwnSegment { Vec3 start; Vec3 velocity; double duration; }`;
  `ComputeFacePieceInterval(TriangleCorners, OwnSegment, double radius)`, `ComputeEdgePieceInterval(...)` and
  `ComputeVertexPieceInterval(...)` taking the two or one corner; `struct MeshEntry { bool has_entry; double
  elapsed; std::uint32_t triangle_index; }`; `FindTriangleEntry(TriangleCorners, OwnSegment, double radius)`;
  `FindMeshEntry(mesh, OwnSegment segment, double radius, MeshSearchCounts* counts)` (candidates: end boxes
  overlapping the segment's box grown by `radius * (1 + kMeshPieceOverlapShare)`; walk as Task 6's, same array).

- [ ] **Step 1: Failing tests** (`cpu`; `r = tol` unless stated; triangle `T = (0,0,0), (1,0,0), (0,1,0)`):

  ```cpp
  TEST(ComputeInterval, LinearGapEachSignOfRate_GivesItsSide)   // {2, -4}: {0.5, +inf}; {-2, 4}: {-inf, 0.5};
                                                                // {-1, 0}: all time; {1, 0}: empty
  TEST(ComputeInterval, QuadraticGapEachCase_GivesItsRoots)     // a 1, b -3, c 5: {1, 5}; a 1, b 0, c 1: empty;
                                                                // a 0, c -1: all time; a 0, c 1: empty; c NaN:
                                                                // empty
  TEST(FindTriangleEntry, StraightDownOntoTheFace_EntersAtHeightOverSpeed)   // {0.25,0.25,1}, v {0,0,-2}, 1 s:
                                                                // (1 - tol) / 2; from below {0.25,0.25,-1},
                                                                // v {0,0,2}: the same
  TEST(FindTriangleEntry, OntoAnEdgeFromOutside_EntersTheEdgePiece)  // {0.5,-1,0}, v {0,1,0}, 2 s: 1 - tol
  TEST(FindTriangleEntry, OntoAVertexAlongTheDiagonal_EntersTheVertexPiece)  // {-3,-4,12} toward the origin at
                                                                // speed 13 ({3,4,-12}), 2 s: 1 - tol / 13
  TEST(FindTriangleEntry, PastTheTriangleBeyondAllPieces_IsFree)   // {2,2,1}, v {0,0,-2}; and {0.5,-1,0},
                                                                // v {0,0.4,0}, 2 s (stops 0.2 short)
  TEST(FindTriangleEntry, StartInsideAPiece_EntersAtZero)       // {0.25,0.25,tol/2}, any velocity: elapsed 0
  TEST(FindTriangleEntry, AtRestOutside_IsFree)                 // velocity 0 at {0.25,0.25,1}
  TEST(FindTriangleEntry, EveryCaseTurnedAndMoved_SameTimes)    // the five above under the turned pose applied to
                                                                // triangle and path: within 0x1p-44 relative
  TEST(FindMeshEntry, CubeFaceEdgeAndVertex_MatchClosedForms)   // cube half 0.5: {-1,0.1,0.2}, v {2,0,0}, 1 s:
                                                                // (0.5 - tol) / 2; {-1,-1,0}, v {1,1,0}:
                                                                // 0.5 - tol / Sqrt(2) (0.4999999993414555);
                                                                // {-1,-1,-1}, v {1,1,1}: 0.5 - tol / Sqrt(3)
                                                                // (0.49999999946230067)
  TEST(FindMeshEntry, EqualTimesOnTwoTriangles_KeepsTheLowerIndex)  // cube, straight down onto the top
                                                                // face's diagonal
  TEST(FindMeshEntry, ShortPathFarFromTheMesh_VisitsOneBox)     // counts
  TEST(FindMeshEntry, PathsThroughSharedEdgesAndVertices_NeverLeakAgainstExactArithmetic)   // below
  TEST(FindMeshEntry, PathsGrazingAGrownEdge_AnswerWithinTheBand)   // below
  ```
  The leak test (P11; 40,000 cases: cube and `BuildIcosphereMesh(0.5, 1)`, radii `tol` and `tol / 2`, 10,000
  each). Per case, in this draw order: `is_vertex = Next() < 0.5`; a triangle index `Floor(Next() * count)`; a
  corner `k = Floor(Next() * 3)`; the aim is corner `k` (vertex) or `a_k + Next() * (a_k+1 - a_k)` (edge); each
  aim coordinate is then moved by `Floor(Next() * 9) - 4` rounding units with `std::nextafter`; a direction
  `{NextSigned() x 3}` redrawn while its length is below 0.25 or its `Dot` with the corner's vertex normal is
  below `0.2 * length`, then `Normalized`; `speed = 0.5 + 9.5 * Next()`; `start = aim + 0.3 * direction`;
  `velocity = -speed * direction`; `duration = 0.6 / speed`. For each: `entry = FindMeshEntry(...)`. With exact
  arithmetic over every triangle of the mesh: (a) not free: `entry.has_entry` (the path passes within a few rounding
  units of the mesh at
  the aim); (b) not late: the least exact squared distance from the segment `start` to
  `position(entry.elapsed)` (`start + velocity * elapsed` in doubles, as the code forms it) is at least
  `(r - tol / 8)^2`; (c) not early: the least exact squared distance from `position(entry.elapsed)` is at most
  `(r + tol / 8)^2`. The grazing test (4,000 cases, cube, both radii): an edge `k` of a random triangle, a point
  `q` on it, the unit direction `w` across the edge away from the cube (the edge's summed normals, normalised),
  `t` along the edge; path through `q + (r + j * ulp) * w` along `t`, `j` an integer from -4 to 4, `ulp` one
  rounding unit of 0.5, length 0.2 either side: no assertion fails, and when free the least exact squared
  distance over the whole path is at least `(r - tol / 8)^2`; when hit, (b) and (c) hold. Print how many of
  each. Numbers: the closed forms by hand and script; the `tol / 8` of M4 is the design's stated margin.
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement** steps 3 to 5 of `<proc-mesh-first-hit>`. The quadratic of an edge piece uses the
  components of `p - a` and of the velocity across the edge's unit direction. Comments state M1's overlap
  argument beside the three piece functions and why no piece checks a hit point. Glossary: `mesh grown by a
  radius`, `piece` (never say: offset surface, capsule, slab, Minkowski sum).
- [ ] **Step 4: Run** build, tests, gate. **Mutants** (each must fail the leak test or the grazing test, and the
  commit message records how many cases failed): (1) the face piece as "time of the plane, then test that the
  hit point's projection is inside the triangle"; (2) an edge piece as "root of the cylinder, then test the
  hit point between the ends"; (3) the vertex pieces removed; (4) `kMeshPieceOverlapShare` set to 0;
  (5) the tree's query box grown by `radius` only; (6) `exit >= 0` dropped (a piece behind the start counts);
  (7) the earliest entry replaced by the first found. If mutant (1), (2) or (4) passes all tests, raise the
  count of cases aimed at edges until it fails, and report.
- [ ] **Step 5: Commit.** Subject: `Find when a straight path first comes within a radius of a mesh`.

### Task 9: First hit on a mesh that does not turn

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/mesh_contact.h`, `mesh_shape_steps.h`, `solid_contact_steps.h`,
  `find_first_hit_test.cc`, `mesh_entry_test.cc`, `docs/glossary.md`

**Interfaces:**
- Consumes: Tasks 7, 8; H1.
- Produces: `FindFirstHitOnMesh(mesh, MeshSweep)` (design 3.3) with M3; private `MeshStartCase`,
  `ComputeMeshStartCase(DistanceToMesh at_start, MeshSweep sweep)`; `case SolidShape::kMesh` of
  `FindFirstHitOnSolid`: a turning mesh returns `kTooCloseToCall` exactly as the other shapes do today (Task 10
  replaces it when H2 holds); otherwise `ComputeOwnPath`, `FindFirstHitOnMesh`, and a `FirstHit` whose `at_hit`
  is the mesh's distance, the world normal (`pose.axes * own_normal`) and face `kMeshTriangle`.

- [ ] **Step 1: Failing tests** (`cpu`; cube half 0.5 as solid id 1 unless stated):

  ```cpp
  TEST(ComputeMeshStartCase, EachCase_ByDistanceAndRate)        // distance 1.6 tol: kFar; 0.4 tol: kHitAtStart;
                                                                // tol with rate * duration -2 tol: kHitAtStart;
                                                                // tol with -tol exactly: kResting; 1.4 tol with
                                                                // rate 0: kResting; tol 0: kFar
  TEST(FindFirstHitOnMesh, RestingSlideAcrossSharedEdges_IsFree)      // {-0.4, 0, 0.5 + tol}, v {5,0,0}, 0.1 s:
                                                                // free (crosses both diagonals at x = 0); the
                                                                // same on the faces kLowX and kHighY of the
                                                                // cube (start and velocity permuted)
  TEST(FindFirstHitOnMesh, RestingAndPushingIn_HitsAtZero)      // same start, v {5,0,-1}: elapsed 0, normal
                                                                // {0,0,1}, distance tol
  TEST(FindFirstHitOnMesh, RestingAndSinkingSlowly_HitsAtHalfTheTolerance)   // same start, v {0,0,-0.75 tol},
                                                                // 1 s: elapsed 2/3 within 0x1p-40, distance
                                                                // tol / 2 within 0x1p-40 * tol
  TEST(FindFirstHitOnMesh, StartBelowHalfTheTolerance_HitsAtZero)     // {0.1, 0.2, 0.5 + 0.4 tol}, v {1,0,0}
  TEST(FindFirstHitOnMesh, NearStartNotResting_SlidesFree)      // {0.1, 0.2, 0.5 + 1.2 tol}, v {1,0,0}, 0.1 s
  TEST(FindFirstHit, MeshCubeFace_HitsWhereTheBoxDoes)          // {-1,0.1,0.2}, v {2,0,0}, 1 s: (0.5 - tol) / 2,
                                                                // face kMeshTriangle, normal {-1,0,0} 1e-12;
                                                                // the box in the same call as solid id 2 at
                                                                // {0,5,0} is not hit
  TEST(FindFirstHit, MeshCubeEdgeAndVertex_HitTheRoundedCorner) // Task 8's two times; normals {-1,-1,0}/Sqrt(2)
                                                                // and {-1,-1,-1}/Sqrt(3) within 1e-5
  TEST(FindFirstHit, MovingMeshCube_UsesRelativeVelocity)       // centre {2,0,0} moving {-1,0,0}, particle at
                                                                // rest at the origin, 2 s: 1.5 - tol
  TEST(FindFirstHit, MovingMeshCubeWithStartElapsed_SamplesItLater)   // same, step 2 s, start_elapsed 0.5,
                                                                // duration 1.5: 1 - tol
  TEST(FindFirstHit, RestingOnMovingMeshWithStartElapsed_SlidesFree)  // cube moving {0,0,2}, step 1,
                                                                // start_elapsed 0.25: particle at
                                                                // {-0.4, 0, 1 + tol} (the top face is at z = 1
                                                                // then), v {5,0,2}, 0.1 s: free; v {5,0,1}:
                                                                // hit at 0
  TEST(FindFirstHit, MeshCubeInTurnedPose_SameTimesAsUnturned)  // face, edge and vertex cases with solid and
                                                                // path turned and moved: 0x1p-40 relative
  TEST(FindFirstHit, MeshAndFlatSolidAtTheSameTime_LowerIndexWins)    // floor (flat solid, plane z = 0.5,
                                                                // normal {0,0,1}) as id 1 and the cube as id 2;
                                                                // straight down from {0.1,0.2,2} at {0,0,-1},
                                                                // 2 s. ASSERT the two elapsed found alone are
                                                                // equal (1.5 - tol); then solid_index 0. With
                                                                // the order of the two solids swapped: 0 again
  TEST(FindFirstHit, NaiveControlOnMesh_HitsAtTheSurface)       // contact_tolerance 0: {0.1,0.2,2}, v {0,0,-1}:
                                                                // elapsed 1.5 within 0x1p-44
  ```
  Numbers: by hand. `2/3`: the gap to radius `tol / 2` is `tol / 2`, the rate `0.75 tol`. The moving cube with
  `start_elapsed 0.5` is at `{1.5, 0, 0}`, its face at 1: `1 - tol` at speed 1. If the two times of the lower-index
  test differ
  in the last bit, stop and report (the test's premise).
- [ ] **Step 2: Run.** Expected: compile error, then the new tests fail.
- [ ] **Step 3: Implement** steps 1, 2 and 6 of `<proc-mesh-first-hit>`. A comment at `ComputeMeshStartCase`
  says why the resting sweep halves the radius (design 5.4, "Why two radii") and that every threshold is at
  least `tol / 4` from where `MoveParticle` leaves a particle. Glossary: `far start`, `near start`,
  `resting sweep` (never say: touching rule, for a mesh).
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** the resting sweep with radius `tol` (Review Focus 2 must
  fail on at least one of the three faces; if it passes by luck of rounding, add the start heights
  `0.5 + tol * (1 - 0x1p-20)` and `0.5 + tol * (1 + 0x1p-20)` to the test and keep them); the near threshold at
  `tol`; the rate test with `>`; the hit's normal left in own axes; the pose taken at time 0 instead of
  `start_elapsed`; `ComputeOwnPath`'s relative velocity replaced by the particle's.
- [ ] **Step 5: Commit.** Subject: `Find the first hit of a straight path on a mesh that does not turn`.
  New names: `FindFirstHitOnMesh`.

### Task 10: First hit on a mesh that turns

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact_steps.h`, `find_first_hit_test.cc`

**Interfaces:**
- Consumes: Task 9; **H2** (module 1 Task 9). The `kMesh` case hands a turning mesh to the same stepping search
  the box uses; nothing of the search is copied.
- Produces: `FindFirstHit` for a turning mesh (design 5.7, first item).

- [ ] **Step 1: Failing tests** (`cpu`):

  ```cpp
  TEST(FindFirstHit, TurningMeshCubeReachesRestingParticle_AtTrigonometricAngle)
      // module 1 Task 9's case with the mesh cube: angular {0,0,1}, particle at rest at 0.6 * {cos 0.3, sin 0.3,
      // 0}, 1 s: elapsed == 0.3 + acos((0.5 + tol) / 0.6) within 0x1p-30, face kMeshTriangle
  TEST(FindFirstHit, TurningMeshCubeNotReachedInStep_IsFree)    // angular {0,0,-1}, 0.2 s
  TEST(FindFirstHit, RestingOnTurningMesh_IsHitAtZero)          // start {0.5 + tol, 0, 0}, v {0,0,1}, 0.1 s:
                                                                // elapsed 0; control with angular 0: free
  TEST(FindFirstHit, TurningLMeshReachesParticleInItsNotch_AtTrigonometricAngle)
      // BuildLMesh() turning {0,0,1} about the origin, particle at rest at {0.5, 0.8, 0} (in the notch), 1 s:
      // in the L's axes it turns clockwise at radius Sqrt(0.89) and meets the face y = 0.4 (at x = 0.854, inside
      // that face): elapsed == atan2(0.8, 0.5) - asin((0.4 + tol) / Sqrt(0.89)) = 0.5743405112995239 within 0x1p-30;
      // at_hit.distance in (tol, 1.25 tol]
  TEST(FindFirstHit, TurningMeshWithStartElapsed_UsesThePoseThen)     // the first case with start_elapsed 0.1
                                                                // and the particle turned 0.1 further: the same
                                                                // elapsed within 0x1p-30
  ```
  Numbers: the trigonometric angle is module 1's (the particle meets the face `x = 0.5 + tol` at local
  `y = 0.33`, inside the face region, where the mesh and the box agree); the L case by hand and script (`r = 0.9434`,
  the face `y = 0.4` is met at
  angle `asin(0.4 / r)`).
- [ ] **Step 2: Run.** Expected: the new tests fail (`kTooCloseToCall`).
- [ ] **Step 3: Implement.** The `kMesh` case's turning branch calls the search of H2. If that search is not
  reachable for a new shape without copying it, stop and report (H2 does not hold).
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** a touching or resting rule applied to a turning mesh
  (`RestingOnTurningMesh` must fail); the search's gap from a capped distance (`largest_distance = 2 tol`).
- [ ] **Step 5: Commit.** Subject: `Step toward a turning mesh until the path reaches it`.

### Task 11: `MoveParticle` start rules and the rest gap on a mesh

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact_steps.h`, `solid_contact.h`, `move_particle_test.cc`,
  `contact_test_cases.h`

**Interfaces:**
- Consumes: Task 9; **H3, H4, H6** (module 1 Tasks 10, 11, 12a).
- Produces: the `kMesh` case inside `MoveToRestGap`: move along the mesh normal by `tol - d`, recompute, at most
  `kMaxMoveOutCount` times, stopping when `d >= tol` (design 5.5); in `MoveParticle` step 4, a hit on a mesh is
  passed through `MoveToRestGap` before the response, **the response keeps the normal of `FirstHit::at_hit`**,
  and the reaction's `hit_position` is the moved position. Every case is added to `BuildMoveParticleCases()`.

- [ ] **Step 1: Failing tests** (`cpu`; mesh cube half 0.5 id 1, friction 0, both velocities equal, unless
  stated):

  ```cpp
  TEST(MoveParticle, MeshCubeFace_StopsOneToleranceShort)       // {-1,0.1,0.2}, v {2,0,0}, T 1: kHit,
                                                                // hit_fraction (0.5 - tol) / 2, end x
                                                                // -(0.5 + tol), velocities 0, hit_count 1,
                                                                // face kMeshTriangle
  TEST(MoveParticle, MeshShallowOverlap_StartedInsideAndMovedOut)     // start z 0.5 - 0.0005 over the top face,
                                                                // overlap tolerance 0.001, at rest, T 0.1:
                                                                // kStartedInsideSolid, end z 0.5 + tol
  TEST(MoveParticle, MeshDeepOverlap_IsStartsInsideSolid)       // z 0.49, overlap 0.001: magnitude 0.01, id 1
  TEST(MoveParticle, MeshStartWithinToleranceMovingAway_IsMovedSilently)   // {0.1,0.2,0.5 + tol / 4}, v {1,0,2},
                                                                // T 0.1: kNone, hit_count 0, velocities bit for
                                                                // bit, end {0.2, 0.2, 0.5 + tol + 0.2} within
                                                                // 2 tol
  TEST(MoveParticle, MovingMeshPushesRestingParticle_CarriesIt) // cube moving {0,0,2}, start {0.1,0.2,0.5 + tol}
                                                                // at rest, T 0.1: kHit, hit_fraction 0,
                                                                // velocities {0,0,2}, end z 0.7 + tol
  TEST(MoveParticle, InsideCornerStartNearBothFaces_ReachesTheRestGapOfBoth)
      // BuildGrooveMesh(1) (faces z = x and z = -x): start 0.2 tol from both faces, {0, 0, 0.2 * Sqrt(2) * tol},
      // at rest, T 0.1: no error, kNone, end distance to the mesh within 0x1p-40 * tol of tol, end
      // {0, 0, Sqrt(2) tol} within tol / 4 (two moves: the faces are at right angles)
  TEST(MoveParticle, SixtyDegreeGrooveStartShortOfOneFace_FourMovesReachNineTenths)
      // BuildGrooveMesh(tan(30 degrees)) (faces with normals {-+0.866, 0, 0.5}, inside angle 60 degrees): start
      // {0.5196 tol, 0, 1.1 tol} (0.1 tol from the face on its side, tol from the other), at rest, T 0.1: no
      // error, kNone, end distance >= 0.9 tol (shortfalls 0.9, 0.45, 0.225, 0.1125, then 0.056 tol; one move
      // alone leaves 0.55 tol, two leave 0.775)
  TEST(MoveParticle, RestingSweepHit_IsMovedToTheRestGapBeforeTheResponse)
      // cube reacting; start {0.1, 0.2, 0.5 + tol}, v {0,0,-0.75 tol}, T 1: kHit, hit_fraction 2/3 within
      // 0x1p-40, reaction.hit_position z 0.5 + tol within 0x1p-40 * tol (found at 0.5 + tol / 2), velocities 0
  ```
  Each start rule again with the moving solid and `start_elapsed > 0` is covered through the hit loop:
  `MoveParticle, MovingMeshSecondSweep_StartRulesUseThePoseThen`: the cube moving `{0,0,2}` and a floor (id 2,
  flat solid `z = -3`); particle `{0.1, 0.2, 0.5 + tol}`, both velocities `{1,0,0}`, T 0.1: hit at 0, velocity
  `{1,0,2}`; the re-sweep starts at `elapsed 0` on a moving mesh and slides free: `hit_count` 1, end
  `{0.2, 0.2, 0.7 + tol}` within `2 tol`. Numbers: by hand; `Sqrt(2) tol` is the height at which a point is `tol`
  from both faces `z = +-x`; the 60 degree start is `0.1 tol * n_A + 1.2124 tol * (0.5, 0, 0.866)`, and each
  move along one face's normal takes `cos(60 degrees)` of its length off the other's distance.
- [ ] **Step 2: Run.** Expected: the new tests fail.
- [ ] **Step 3: Implement.** One comment at the move: why it repeats, the `cos(theta)` bound and the angle.
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** one move only (the inside-corner test); the response with
  the normal at the moved position (Task 12's Review Focus 3 test is the stronger check; here
  `RestingSweepHit` must still pass, so record that this mutant is caught in Task 12); the mesh hit not moved
  (`RestingSweepHit`'s `hit_position`); the moves along the first normal only.
- [ ] **Step 5: Commit.** Subject: `Bring a particle to the rest gap of a mesh, also in a concave corner`.

### Task 12: `MoveParticle` slide on a mesh: several hits, concave corners, the hit limit

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact.h`, `solid_contact_steps.h`, `solid_contact_types.h`
  (`hit_count`'s comment), `move_particle_test.cc`, `contact_test_cases.h`, `docs/glossary.md`

**Interfaces:**
- Consumes: Task 11; **H3, H5** (module 1 Tasks 10, 12).
- Produces: design 5.6: a full surface list replaces the oldest entry of the same mesh; the count of hits on
  meshes; at hit `kMaxMeshHitsPerStep + 1` the particle rests for the remainder of the step (velocity for this
  step = the solid's surface velocity at the point; velocity for the next step unchanged; not a close call).

- [ ] **Step 1: Failing tests** (`cpu`; friction 0 unless stated):

  ```cpp
  TEST(MoveParticle, MeshCubeFrictionOnTopFace_SlidesAcrossTheDiagonalsWithOneHit)
      // cube; start {-0.3, 0.1, 0.6}, v {3,0,-1}, friction 0.5, T 0.2: hit at t = 0.1 - tol, velocity
      // {2.5,0,0}, end {0.25 - 0.5 tol, 0.1, 0.5 + tol} within 2 tol, hit_count 1 (the slide crosses the
      // diagonal at x = 0.1)
  TEST(MoveParticle, RightAngleGroove_HitsBothSidesAndRests)
      // BuildGrooveMesh(1); start {0.1, 0, 0.5}, v {0,0,-2}, T 1: hit_fraction (0.4 - Sqrt(2) tol) / 2 within
      // 0x1p-40; first response {-1,0,-1}; second hit on the face z = -x; velocities {0,0,0} within 1e-15;
      // hit_count exactly 2; end {0, 0, Sqrt(2) tol} within 2 tol
  TEST(MoveParticle, RightAngleGrooveTurnedAndMoving_SameResultInItsAxes)
      // the same with the solid in the turned pose and moving: hit_count 2, end and velocities turned and
      // carried accordingly, positions 2 tol, velocities equal to the solid's within 0x1p-40 relative
  TEST(MoveParticle, SixtyDegreeGroove_StopsWithoutError)       // BuildGrooveMesh(tan(30 degrees)); start
                                                                // {0.1, 0, 0.5}, v {0,0,-2}, T 1: no error,
                                                                // velocities {0,0,0}, hit_count >= 2 and <= 4,
                                                                // end distance >= tol / 2
  TEST(MoveParticle, GrooveSlideAlongItsLine_KeepsTheAlongPart) // right-angle groove; start {0.1, -0.5, 0.5},
                                                                // v {0,1,-2}, T 0.5: velocities {0,1,0} within
                                                                // 1e-15, end y -0.5 + 0.5 = 0.0 within 2 tol,
                                                                // hit_count 2
  TEST(MoveParticle, ReactingGroove_TwoHitsAreOneReactingSolid) // the groove reacting: no kSecondReactingSolid;
                                                                // velocity_for_next_step_change == {0,0,2}
                                                                // within 1e-15 (the two changes summed);
                                                                // hit_position the first hit's, 2 tol
  TEST(MoveParticle, TubeEightCreases_SlidesOnWithoutRest)
      // BuildTubeMesh(64, 0.5, 0.6, 1): start at rest gap on the lowest facet's middle, {0, 0, -(0.5 *
      // cos(pi / 64) - tol)}, v {2,0,0}, T 0.2: no error; hit_count from 7 to 9; speed for_this_step between
      // 0.955 and 0.969 of 2 (cos(pi / 32)^9 = 0.9576, ^7 = 0.9668); end distance in [tol / 2, 2 tol]
  TEST(MoveParticle, TubeMoreThanSixteenCreases_RestsForTheRemainder)
      // same, T 0.5: no error (not kTooManyHits), hit_count 17, velocity.for_this_step {0,0,0}, for_next_step
      // speed between 1.82 and 1.86, end distance in [tol / 2, 2 tol]
  TEST(MoveParticle, FullListReplacesOldestEntryOfTheSameMesh)  // the tube with T 0.13 (5 creases, 5 entries
                                                                // wanted): no kTooManyHits, hit_count 5
  TEST(MoveParticle, FlatSolidsStillLimitedToFourHits)          // module 1's octagon unchanged: kTooManyHits
  ```
  The tube: a facet subtends `2 pi / 64`, so its inner face is at `0.5 * cos(pi / 64)` from the axis and is
  `2 * 0.5 * sin(pi / 64) = 0.04907` long; at each crease the next facet's normal is turned by `pi / 32`
  (5.625 degrees) and the frictionless response keeps `cos(pi / 32) = 0.99518` of the speed. `T 0.2`: 0.4 m,
  about 8 creases. `T 0.5`: more than 16 creases, so the 17th hit rests; `for_next_step` after 16 or 17 responses is
  `2 * 0.99518^16 = 1.851` or `2 * 0.99518^17 = 1.842`. `T 0.13`: 0.26 m, creases at 0.025, 0.074, 0.123, 0.172
  and 0.221 m. Numbers: by hand and script.
- [ ] **Step 2: Run.** Expected: the new tests fail (`kTooManyHits`, or a wrong `hit_count`).
- [ ] **Step 3: Implement.** Comments: why replacing an entry cannot let a particle in (the sweep keeps it out;
  the list only shapes the velocity); why the rest is not a close call. `hit_count`'s comment per design 5.6.
  Glossary: the `hit` row gains "a mesh may be hit several times in a step".
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** the response with the normal at the moved position
  (`RightAngleGroove`'s `hit_count` must leave 2: Review Focus 3); "approaches again" compared with 0; the
  replaced entry the newest instead of the oldest; the rest at hit 16; the rest returning `kTooManyHits`; a mesh
  just hit skipped in the re-sweep (`RightAngleGroove` or `SixtyDegreeGroove` must fail).
- [ ] **Step 5: Commit.** Subject: `Slide along a mesh through its concave corners and rest after too many hits`.

### Task 13: `MoveParticle` slide on a turning mesh

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/contact/solid_contact_steps.h`, `move_particle_test.cc`, `contact_test_cases.h`

**Interfaces:**
- Consumes: Tasks 10, 12; **H7** (module 1 Task 13).
- Produces: design 5.7, second item: inside the carry function of H7, a mesh's remainder is swept in own axes
  with `FindFirstHitOnMesh` (a near start), the move of Task 11 in own axes, and at each further hit the removal
  of the approach to the new normal and to the one before it (both gone: the relative velocity is zero); at
  most `kMaxMeshHitsPerStep` hits, then rest. Only the position changes; the velocities stay as the first
  response made them. The end is placed with `ComputeSolidPose(solid, step_duration)`.

- [ ] **Step 1: Failing tests** (`cpu`):

  ```cpp
  TEST(MoveParticle, TurningMeshCubeHitsRestingParticle_EndsOnFaceAtEndPose)
      // module 1 Task 13's first case with the mesh cube: hit_fraction 0.3 + acos((0.5 + tol) / 0.6) within
      // 0x1p-30; distance to the mesh at ComputeSolidPose(cube, 1) in [tol / 2, 2 tol]
  TEST(MoveParticle, TurningMeshCubeSlide_KeepsSidewaysPartInSolidAxes)
      // module 1 Task 13's third case with the mesh cube: end == origin + axes * {0.5 + tol, -0.05, 0.1} at
      // ComputeSolidPose(cube, 0.1), within 2 tol; hit_fraction 0
  TEST(MoveParticle, TurningLMeshCarriesParticleInItsInsideCorner_ThousandSteps)
      // BuildLMesh() turning {0,0,1} about the origin; particle at the rest gap of both inside faces,
      // {0.4 + tol, 0.4 + tol, 0}, at rest; 1,000 steps of 1e-3, each: MoveParticle, the end fed back, the
      // solid's pose advanced by ComputeSolidPose(solid, T), both velocities reset to the solid's surface
      // velocity at the particle: no error; distance in [tol / 2, 2 tol] every step; the particle's own-axes
      // position within 1e-9 m of {0.4 + tol, 0.4 + tol, 0} after the last
  TEST(MoveParticle, TurningMeshSlideIntoItsOtherFace_StopsThere)
      // the L turning {0,0,0.5}; particle on the face y = 0.4 at {0.9, 0.4 + tol, 0} with a relative velocity
      // along -x of 2 in the L's axes (world velocity = surface velocity + axes * {-2,0,0}), T 0.5: it reaches
      // the face x = 0.4 after 0.25 s and stops against it: own-axes end {0.4 + tol, 0.4 + tol, 0} within 2 tol
  TEST(MoveParticle, TurningMeshNearFloor_IsTurningSolidNearOtherSolid)   // module 1 Task 13's second case
  ```
  Numbers: module 1's, and by hand (`(0.9 - 0.4) / 2 = 0.25` s).
- [ ] **Step 2: Run.** Expected: the new tests fail.
- [ ] **Step 3: Implement.** If H7's function cannot host the sweep without changing module 1's box path, stop
  and report.
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** the remainder carried as a straight line (the fourth test
  must fail: the particle would pass through the face `x = 0.4`); the end placed with the start pose; the
  own-axes sweep with a far start's radius.
- [ ] **Step 5: Commit.** Subject: `Carry a slide on a turning mesh in its own axes, stopping at its other faces`.

### Task 14: Generated safety tests on meshes

**Kind:** numerics

**Files:**
- Create: `native/fluidity/contact/move_particle_mesh_tolerance_test.cc`
- Modify: `native/fluidity/contact/contact_test_cases.h`, `mesh_test_cases.h`, `CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 11 to 13; **H8** (module 1 Task 14's helper).
- Produces: test-only `std::vector<MoveParticleCase> BuildMeshCases(int count)` (used again in Task 16).

Mesh set (P11; draw order): `kind = Floor(Next() * 3)` (0 bowl, 1 torus, 2 `BuildFinMesh(1e-4)`); pose origin
`100 * {NextSigned() x 3}`; axes `math::Rotation(3 * {NextSigned() x 3})`; `linear_velocity = 10 * {NextSigned()
x 3}`; a surface target: a triangle `Floor(Next() * count)` and the point `(a + b + c) / 3`; a direction as in
Task 8's leak test (against the triangle's normal); `start = target + (0.02 + 0.3 * Next()) * direction` in own
axes, redrawn while `ComputeDistanceToMesh` there is below `2 * tol`; `speed = 1000 * Next()`;
`hit_share = 0.1 + 0.8 * Next()`; `T = 0x1p-6`; own velocity `-(distance start to target) / (T * hit_share) *
direction` capped at `speed + 1`; world `for_this_step` = solid velocity + axes * own velocity; `for_next_step =
for_this_step + 0.5 * {NextSigned() x 3}`; `friction = Next() < 0.5 ? 0 : 0.5`;
`tol = ComputeContactTolerance({110, 0.01, 0.01})`; `start_overlap_tolerance = 0`.

- [ ] **Step 1: Failing tests** (`cpu`):

  ```cpp
  TEST(MoveParticle, MeshCasesFastAndOblique_NeverEndInside)    // BuildMeshCases(10000) through H8's helper:
                                                                // no error but kTooCloseToCall never (assert
                                                                // none); every end distance >= tol / 2; the
                                                                // naive control enters in at least one case
                                                                // (print the count); at least 3,000 are hits
  TEST(MoveParticle, FastParticleThinFin_IsStoppedOnItsOwnSide) // BuildFinMesh(1e-4) at the origin; start
                                                                // {0.1, 0.2, 5}, v {0,0,-1000}, T 0.01: kHit,
                                                                // hit_fraction (5 - 5e-5 - tol) / 10, end z
                                                                // 5e-5 + tol within 2 tol
  TEST(MoveParticle, ConcaveMeshesHoldARestingParticle_TenThousandSteps)
      // bowl (lowest inner point), right-angle groove ({0,0,Sqrt(2) tol}), 60 degree groove, torus (lowest
      // point of the ring's inside): at rest; each step adds {0,0,-9.81 T} to both velocities, T 1e-3,
      // MoveParticle, end fed back, start_overlap_tolerance 0: no error; distance in [tol / 2, 2 tol] every
      // step; position within 1e-9 m of the start after the last
  TEST(MoveParticle, FrictionlessGrooveTiltedThirtyDegrees_SpeedsUpAsGSinTheta)
      // right-angle groove turned 30 degrees about x (its line dips along -y); from rest on its line, gravity
      // added each step, T 1e-3, 1,000 steps: speed along the line 9.81 * sin(30 degrees) = 4.905 m/s within
      // 1 %; hit_count <= 2 in every step
  TEST(MoveParticle, MeshSceneTurnedRigidly_ResultTurnsWithIt)  // every mesh case of BuildMoveParticleCases()
                                                                // under module 1's two turns: discrete fields
                                                                // equal; floating values within 0x1p-40 scaled
  ```
  Numbers: by hand (`5 - 5e-5 - tol` at 1,000 m/s over a step of 0.01 s is the fraction above).
- [ ] **Step 2: Run.** Expected: pass if Tasks 9 to 13 are right. Then prove the control: temporarily make the
  mesh sweep's radius 0 and the move of Task 11 a no-op; `NeverEndInside` must fail; restore. Record the count.
- [ ] **Step 3: Run** the gate. **Mutants:** Task 8's mutants (1), (2) and (4) again against
  `MeshCasesFastAndOblique` (record whether this coarser test also catches them); the end check at `tol / 4`.
- [ ] **Step 4: Commit.** Subject: `Show particles stay outside meshes at any speed and rest in their corners`.

### Task 15: Mesh buffers, the memory estimate, and `MoveParticles` on the CPU with a mesh

**Kind:** integration

**Files:**
- Create: `native/fluidity/contact/mesh_shape_buffers.h`, `mesh_shape_buffers.cc`, `mesh_shape_buffers_test.cc`
- Modify: `native/fluidity/contact/move_particles.h`, `move_particles_cpu.cc`, `move_particles_test.cc`,
  `CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 4, 7; **H9** (module 1 Task 15); `Buffer`, `BuildBuffer(run_on, size, run_memory)`,
  `WriteBuffer`, `RunMemory`, `ComputeRunMemoryBytes`, `MemoryEstimateEntry`.
- Produces: `MeshShapeBuffers`, `BuildMeshShapeBuffers`, `BuildMeshShapeMemoryEstimate` (design 3.4; six rows in
  the order vertices, triangles, neighbours, vertex normals, tree, table; triangle indexes stay per mesh, the
  table's pointers carry each mesh's offset); `MoveParticlesInputs::meshes`; `MoveParticles` copies it into each
  `SolidContactInputs` and adds two checks, in this order after module 1's: a `kMesh` solid whose `mesh_index`
  is not below `meshes.mesh_count` is `kBadInput`; `meshes.mesh_count > 0` with a null `meshes.meshes` is
  `kBadInput`. A failed check writes nothing. Errors of the build map through module 1's one mapping function.

- [ ] **Step 1: Failing tests** (`cpu`):

  ```cpp
  TEST(BuildMeshShapeMemoryEstimate, CubeAndIcosphere_MatchesArraySizes)   // 8 + 642 vertices, 12 + 1280
      // triangles, 7 + 1023 boxes, 2 meshes: ComputeMemoryBytes == 650 * 24 * 2 + 1292 * 12 * 2 + 1030 * 32 +
      // 2 * 120 = 95,408
  TEST(BuildMeshShapeBuffers, IntoRunMemory_HandsOutExactlyTheEstimate)    // kCpu: handed_out_bytes() ==
                                                                // reserved_bytes()
  TEST(BuildMeshShapeBuffers, RunMemoryOnePieceShort_IsOutOfMemory)       // GTEST_SKIP in debug builds
  TEST(BuildMeshShapeBuffers, TwoMeshes_EachAnswersAsItsArrays) // 1,000 points: ComputeDistanceToMesh through
                                                                // meshes() equals ToMeshShape(arrays), ToBits
  TEST(BuildMeshShapeBuffers, GpuInBuildWithoutGpuCode_IsRunOnCallFailed) // only without FLUIDITY_HAS_GPU
  TEST(MoveParticles, EveryMeshCase_SameAsMoveParticle)         // each mesh case of BuildMoveParticleCases() as
                                                                // a one-particle call: bit for bit
  TEST(MoveParticles, MeshIndexBeyondTable_IsBadInput)          // nothing written
  TEST(MoveParticles, MeshAmongOtherSolids_TwoRangesSameAsOneCall)   // 200 particles of BuildMeshCases(200)
                                                                // kind 0 against that bowl, a floor and a
                                                                // sphere, split 0..77 and 77..200: same bytes
  ```
  Numbers: by hand and script (1023 boxes for 1280 triangles; `95,408 = 31,200 + 31,008 + 32,960 + 240`).
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** `mesh_shape_buffers.cc` is CPU-only code that builds and writes buffers on either
  processor through `base/buffer.h`; it needs no `_gpu.cu`. The table is written last, from the buffers'
  addresses on `run_on`.
- [ ] **Step 4: Run** build, tests, gate. **Mutants:** the second mesh's pointers without its offset; the table
  written from CPU addresses on a GPU build (caught only on hardware: list it in the hand-over, Task 18).
- [ ] **Step 5: Commit.** Subject: `Hold the run's meshes on its processor and move particles against them`.
  New names: `MeshShapeBuffers`, `BuildMeshShapeBuffers`, `BuildMeshShapeMemoryEstimate`,
  `MoveParticlesInputs::meshes`.

### Task 16: `MoveParticles` on the GPU with a mesh

**Kind:** device

**Files:**
- Modify: `native/fluidity/contact/move_particles_steps.h`, `move_particles_cpu.cc`, `move_particles_gpu.cu`,
  `move_particles_steps_test.cc`, `move_particles_gpu_test.cc`, `shared_formulas_gpu_test.cc`,
  `shared_formulas_test_gpu.h`, `shared_formulas_test_gpu.cu`, `mesh_shape.h`, `mesh_contact.h` (the same-bits
  sentence)

**Interfaces:**
- Consumes: Task 15; **H10** (module 1 Task 16).
- Produces: `MoveParticlesView::meshes` (one field, copied by the CPU half; on a GPU run it is
  `MeshShapeBuffers::meshes()` of the GPU, device addresses); `ApplyMoveParticleAt` passes it on. No new kernel
  and no new GPU file. Same-bits promise, stated in the headers: `ComputeDistanceToMesh`, `FindFirstHitOnMesh`
  and the mesh-aware point functions give the same bits on CPU and GPU, unless NaN, for a mesh that does not
  turn (only `+ - * /`, `math::Sqrt`, `math::Abs`, `math::Min`, `math::Max`, comparisons).

- [ ] **Step 1: Failing tests.**

  ```cpp
  // move_particles_steps_test.cc, label cpu: the main evidence until the hardware run
  TEST(ApplyMoveParticleAt, MeshCasesEveryThreadBlockAndThread_SameAsCpuLoop)
      // BuildMeshCases(3000) of kind 0 packed as one call against that bowl plus a floor, and every mesh case
      // of BuildMoveParticleCases() as one-particle calls: every thread block and thread of
      // ComputeLaunchSize(count), in launch order and in reverse: outputs, records, reactions equal the CPU
      // loop's (ToBits)
  // move_particles_gpu_test.cc, label gpu (compiled here, run by the owner)
  TEST(MoveParticles, MeshCasesOnGpu_MatchCpu)                  // every mesh case of BuildMoveParticleCases()
                                                                // and BuildMeshCases(10000), one call per case:
                                                                // discrete fields exact; same bytes when the
                                                                // solid does not turn; module 1's tolerance
                                                                // when it does
  TEST(MoveParticles, MeshSameInputTwice_SameBytes)
  TEST(BuildMeshShapeBuffers, OnGpu_AnswersAsOnCpu)             // 1,000 points through the formula runner
  // shared_formulas_gpu_test.cc: SharedFormulas, OnGpu_MatchCpu gains 16 fixed mesh inputs on the cube: Task 6's three
  // degenerate points, its tie point and four points of its first test; Task 9's face, edge, vertex,
  // resting-slide, pushing, slow-sink, near-start and below-half sweeps: CheckSameBitsOrBothNaN
  ```
  Every `gpu` test passes its outputs to `WriteResultDigest` with a unique name.
- [ ] **Step 2: Run** the gate. Expected before implementing: `FAIL cpu`, then `FAIL cuda`.
- [ ] **Step 3: Implement.** Check that the per-thread stack holds the search's 24 indexes beside module 1's
  surface list (design 6; module 1's plan reserved about 400 B) and say the figure in the commit message.
- [ ] **Step 4: Run** the gate. Expected: seven `PASS`; in the container every `cpu` test, the mesh ones
  included, passes under the CUDA compile.
- [ ] **Step 5: Commit.** Subject: `Move particles against meshes on the GPU`. Body: `CUDA: compiled, not run`,
  and the new name `MoveParticlesView::meshes`.

### Task 17: Size tests and scene-level tests

**Kind:** integration

**Files:**
- Create: `native/fluidity/contact/mesh_scene_test.cc`
- Modify: `native/fluidity/contact/CMakeLists.txt`, `docs/known-open-physics.md`

**Interfaces:**
- Consumes: `BuildMeshShape`, `BuildMeshShapeBuffers`, `MoveParticles`, `ComputeSolidVelocity`.

- [ ] **Step 1: Failing tests** (`cpu`):

  ```cpp
  TEST(BuildMeshShape, IcosphereOf1310720Triangles_HasTheCountedSizes)
      // 8 halvings: 655,362 vertices; tree_box_count 1,048,575; deepest end box 19; BuildMeshShapeBuffers on
      // kCpu hands out exactly 96,469,760 bytes (5 array buffers and the table, each rounded up to 256 B);
      // 10,000 points: most_boxes_kept <= 19
  TEST(MoveParticles, FrictionlessHitsOnReactingIcosphere_ChangeHasSmallMomentAboutCentre)
      // module 1 Task 17's sphere test with BuildIcosphereMesh(0.5, 5) at {1,2,3}: per particle with a hit,
      // |Cross(hit_position - centre, change)| <= 0.03 * |change| (a facet's normal is within asin(longest_edge
      // / (2 * 0.5 * Sqrt(3))) = 0.024 rad of the radial direction; the lever arm is 0.5)
  TEST(MoveParticles, ReactionSumOnMesh_EqualsSumOfVelocityChanges)   // BuildMeshCases(200) kind 0, reacting:
                                                                // per particle change == after - before within
                                                                // 0x1p-44 relative
  // known-open form (CTest WILL_FAIL removed once calibrated; label known_open; A38)
  TEST(BuildMeshShape, IcosphereOf1310720Triangles_BuildsInTenSeconds)      // one thread, release preset
  TEST(MoveParticles, MillionParticlesNearIcosphere_AtMostFourTimesTheSphere)
      // 1,000,000 particles (P11) within 0.005 of the icosphere above, speeds up to 1 m/s, T 1e-3: wall time
      // at most 4 times the same particles against the analytic sphere of radius 0.5
  ```
  Numbers: the sizes by script (`20 * 4^8`, `n / 2 + 2`, the tree's count formula, the 256 B rounding); the
  angle by hand (longest edge 0.0207, a facet's circumradius `edge / Sqrt(3)`).
- [ ] **Step 2: Run.** The two timings are committed in the known-open form with a row each in
  `docs/known-open-physics.md` (what is measured, the bound, that a miss triggers the fallback of design 5.1);
  print the measured values.
- [ ] **Step 3: Run** build, tests, gate.
- [ ] **Step 4: Commit.** Subject: `Check mesh solids at production size and through the reaction`.

### Task 18: Comment pass and hand-over

**Kind:** mechanical

**Files:**
- Modify: any file under `native/fluidity/contact/`, `docs/method-paper/`, `docs/glossary.md`,
  `docs/gpu-runs.md` that the pass changes; in the `main` checkout nothing (the design edits below are reported,
  not made).

- [ ] **Step 1: Comment pass.** Reread every mesh file and `<sec-mesh-solids>` as a C++ programmer who has
  never read a fluid paper: every comment true of the final code and saying why; a worked example with numbers
  on `ComputeDistanceToMesh`, `FindFirstHitOnMesh`, `BuildMeshShape`, `CheckMeshShape`; every word in the
  glossary with its "never say" list; `Words`, `Uses`, `Used by`, `Read:` lines resolve; `// Step N:` matches
  `<proc-mesh-distance>` and `<proc-mesh-first-hit>`; every `Requires:` has its assertion. Module 1's headers
  that gained a mesh case say so in their `Owns:` and `Method:` lines.
- [ ] **Step 2: Fold each change into the commit that owns the file** (`git commit --fixup`, `git rebase
  --autosquash`), then the format and rules checks on every commit and the full gate on the tip, as module 1's
  Task 18.
- [ ] **Step 3: Check the design section by section** (3.1 to 3.4, 5.2 to 5.7, 6, 7, 8) against the branch.
  Record in the hand-over: M1 to M8; every H row and whether it held as written; each mutant and the test that
  caught it; the measured largest gap between the exact first approach and the reported hit in Task 8's leak
  test, against the `tol / 8` of M4; the two timings; the mutant of Task 15 that only hardware can catch; and
  the edits the contact design needs (`changes-to-apply.md` "01b Mesh solids", the 01 Contact entries).
- [ ] **Step 4: `docs/gpu-runs.md`.** Add the row: module `mesh solids`, the tip commit, `compiled, not run`,
  the `gpu` tests: `MoveParticles.MeshCasesOnGpu_MatchCpu`, `MoveParticles.MeshSameInputTwice_SameBytes`,
  `BuildMeshShapeBuffers.OnGpu_AnswersAsOnCpu`, `SharedFormulas.OnGpu_MatchCpu` (mesh inputs). Fold into Task
  16's commit.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the commits; the gate's output; the new
  public names with their reasons; the records of Step 3; the statement that no mesh device code has run.
