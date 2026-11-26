# Cell resize and batches module design

Status: decided. Module 9 of the rebuild (after the step, before I/O and CLI). Written against the designs of
contact, grid, particles, transfer, pressure, viscosity, materials, solids, gas and step. Updated on 2026-11-14
with the owner's answers (`owner-answers.md`) and `changes-to-apply.md`; every question of section 11 has a
decision. Updated again on 2026-11-14 with owner answers A15 (the solves are batched too) and A17 (a run is all CPU
or all GPU; the grid is built on the GPU in a GPU run) and the entries of `plan-decisions.md` that name this
module, as carried by `plans/2026-11-14-rebuild-11-cell-resize-and-batches.md` (decisions 13 to 25), and with
`plan-decisions.md` "Planned cost shown to the user, and buffers in batched runs" (decisions 26 to 32), and with
`plan-decisions.md` "Reconcile pass (2026-11-14)" items 11, 12, 13 and the coordinator's items (1), (2), (5), (6), as
carried by the plan's tasks marked reconcile (decisions 33 to 36). Updated on 2026-11-15 with owner answers A40 to
A64 as `plan-decisions.md` "Owner answers A40–A64 applied" gives them (decisions 40 to 47): particles in every liquid
cell (A64; the grid-only region is kept below, each passage marked "not in the first version (A64)"), the shared
cell size rule owned by grid (A41, A61 to A63), one reserved memory block per processor (A58), `kOutOfMemory` last
(reconcile item 31) and span forms as preconditions (item 32). Updated again on 2026-11-15 with owner answers A65
to A68 as `plan-decisions.md` "Owner answers A65–A68 applied" gives them (decisions 48 to 51): every liquid cell holds
about eight particles of its own size (A65); a run starts on `step::BuildStartCells`' grid (A66); particles whose size
no longer matches their cell are split or merged at every pass (A67); the order of the later groups and the slab
limit named in the budget (A68). Additions are marked NEW (A65) to NEW (A68). Not committed to the `rebuild` branch.

## 1. Purpose

**Part A, cell resize.** Small cells are wanted near the liquid surface, near solids, in shear layers and where an
artist asks; deep calm liquid can use big cells. After each committed step this module decides which cells to split
or merge (grid's shared size rule, A41), rebuilds the grid, moves every value the grid carries onto the new cells
without losing any, and splits or merges the particles with their cells, so every liquid cell holds particles of its
own size level at every depth (A64, first version). NEW (A65): every liquid cell holds about eight particles of its own
size: a particle's `size_level` is its cell's level, its rest volume is one eighth of its cell (`(w/2)³` for a cell
of width `w`), so a cell k levels coarser than the smallest holds 8 particles each of `8^k` times the smallest particle's volume. NEW (A67):
at every pass a particle whose size no longer matches its cell (it moved across a size change) is split, or merged
with neighbours of its cell. NEW (A66): a run starts on the grid `step::BuildStartCells` builds from the scene's shapes,
already following the size rule; resizes then follow the motion.

*Not in the first version (A64):* deep liquid needs no particles at all; this module moves liquid between the two
ways of holding it: particles near the surface (`particle_region`) and one record of amounts per cell deeper down
(`grid_only_region`). Kept as the later group "Deep liquid without particles (later; needs review)" of the plan,
after an independent review and two gating tests.

**Part B, batches.** When a scene does not fit the GPU, the particles live in CPU memory and pass through the
GPU in batches of cell blocks, twice per step try. The solves run on the whole grid on the GPU when their arrays
fit; otherwise (A15) their working buffers live in CPU memory and goes through the GPU one loaded piece at a time, while
the state and the geometry stay whole on the GPU. This module owns that schedule, chooses among the three ways to
run, and says what each costs in memory and copies.

**Owns:** the per-cell size decision and its hysteresis (`ComputeCellSizeChanges`; the wanted width itself is grid's
`cell_size_rule.h`, A41; `ChooseCellSize` is removed, A41); `SplitCell`, `MergeCells`, the "which cells must also
split" search; `ResizeCells` as a transaction after a committed step (D12); the remap of grid-carried values
between two grid versions (the gas arrays included); not in the first version (A64): the two regions,
`is_grid_only_region`, the amounts of grid-only cells, their transport inside a step, `region_transfer` and its
ledger; the call of
`particles::AdjustParticleCount`; the batch plan (including the choice between whole-grid solves and solves in
pieces, and the planned budgets and copies), the batch copies, the placement of every stage in a batched run, the
rule for sums that cross batches.

**Does not own:** the grid tables, ids, ranges, storage and pieces (grid); split and merge of particles, the
refill formula and the loaded particle groups (particles); any solve (pressure, viscosity, gas: their signatures
do not change; in pieces they run batch by batch through pressure's `SolveBatches`, which they own); the
try-and-commit of the step, the `Simulation`, the memory estimate, the loaded try and the `RunMode` field
(step; `BuildRunPlan` chooses one piece or batched, the caller copies this module's choice in); files, the scene
file and messages (I/O and CLI).

## 2. What exists today

| Where on `main` | What it does | What is wrong |
|---|---|---|
| `src/adaptive_width_policy_3d.cc:46-108` `DecideAdaptiveWidth3D` | Split at once when `cell_width > min(demands)`; merge 8 siblings when `2 w <= 0.8 x wanted` on the big cell and its neighbours, continuously for a wait time; six vetoes | The five wanted widths (`physical`, `interface`, `support`, `camera`, `maximum`) are numbers the caller supplies; no code computes them. The rule (old paper `<sec-blocks>`) is kept; the inputs are new here |
| `src/adaptive_mac_topology_3d.cc:345-424` `RefinementClosure` | Cells that must also split so that the size-step rule holds: 26 probes per split cell, repeated | Limit 4,096 cells; indexes into one flat list. The search is ported |
| `src/adaptive_family_change_3d.cc`, `host/adaptive_family_change_3d.h` | One split or one merge: new cell list, old-to-new map, pressure copied to the 8 small cells or averaged | One cell per try; a new list per call |
| `src/liquid_remap_3d.cc:157-299`, `reference/.../adaptive_remap.py` | Amounts of 8 small cells summed; amounts of a big cell shared out with a limited slope from 6 same-size neighbours (`:198-299`) | 64-epsilon and subnormal certificates (`:122-135`), `OutOfRange` when a division by 8 is not exact. Child order is z-fastest, the cell list is x-fastest, joined by the permutation `[0,4,2,6,1,5,3,7]` |
| `src/adaptive_liquid_remap_3d.cc:135-200, 439-514` | Face flows: 4 pieces summed into a big face or a big face quartered; the 12 new inner faces get flows over a spanning tree so each small cell keeps the big cell's flow imbalance per volume | `RepresentsPartitionSum` exactness (`:96-130`); needs the caller's neighbour amounts; refuses grid-only cells ("Family remap on Eulerian-owned layouts remains unavailable", `docs/adaptivity/native-band-runtime.md`) |
| `src/liquid_band_handoff_3d.cc:41-42, 330-395` | Particles down to 6 cell widths below the surface, velocity blend over the last 2; per cell: keep, remove particles into the cell record, or refill | One cell per try, chosen by the caller, in a flat tank with the surface at a given height (`liquid_adaptive_step_3d.cc:1185`); the record must equal the particle sums bit for bit (`liquid_exact_inventory_sum_internal.h`) |
| `src/adaptive_liquid_core_transport_3d.cc:130-310` | Grid-only cells exchange amounts: each outgoing face piece takes the share `flow x duration / current volume` of every amount the giving cell had at step start | Ported. Fails when a cell ends above its geometric volume; refuses any flow to a particle cell and any mixture |
| `src/adaptive_liquid_particle_exchange_3d.cc`, `host/adaptive_liquid_owner_exchange_3d.h` | Flow across a face between the two regions; the last particle of the cell absorbs the mass correction | Works only when no particle leaves its cell (a fixture); a particle that enters a grid-only cell rejects the try (`liquid_adaptive_step_3d.cc:2053`). Not ported (5.6) |
| `src/liquid_adaptive_step_3d.cc:1225-1228` | In grid-only cells the committed face velocity is the start velocity of the next step | The old paper says it is followed back along the flow ("advected semi-Lagrangian", line 253); the code keeps it in place. Fixed (5.5) |
| `src/liquid_adaptive_family_candidate_3d.cc` | A particle split or merge is checked on the old cells for the old size and on the new cells for the new size, inside the same try as the cell change | This is the answer to conflict C2 (5.4); it was tied to one named particle per try |
| `proofs/Fluidity/VolumeTransport.lean` | Trilinear weights of the whitewater volume output sum to one | Not about this module |
| Batched execution | Nothing. `cuda/liquid_arena_3d_host.cc` reserves one arena with page-locked CPU mirrors of every range | Part B is new |

Y-up on `main`: depth below the surface is `anchor.y - surface_height` and the tank's closed upper sides are X
and Z. Here depth is a distance through liquid cells in any direction (5.2), and nothing reads an axis.

## 3. Public interface

Part A: folder `native/fluidity/cell_resize/`, namespace `fluidity::cell_resize`. Part B: folder
`native/fluidity/batches/`, namespace `fluidity::batches` (decision 8). Headers marked (S) are shared
(`FLUIDITY_CPU_AND_GPU`, C++20 subset); operation headers have `_cpu.cc` and `_gpu.cu` beside them. Names are
NEW unless marked D (DECIDED). Changes of the 2026-11-14 update are marked NEW (2026-11-14); those of the A15
and A17 update are marked NEW (A15) or NEW (A17); those of the reconcile pass NEW (reconcile); those of the
2026-11-15 update NEW (A<n>). Declarations marked "not in the first version (A64)" belong to the plan's later group
and are not built in the first version. This module has no
long-running solve, so no function takes
`cancel_requested`; `ApplyStagesInBatches` checks the try's cancel flag between stages and between batches
(5.8), and in solves in pieces the solvers read it before every solve batch.

### 3.1 `cell_resize_error.h` (S), `cell_resize_message.h`

```cpp
// CHANGED (A64; reconcile item 31): 10 values. kWrongMaterialSet, kGridOnlyCellDrained and kGridOnlyCellTooFull
// exist only for the grid-only region: not in the first version (A64); the later group appends them after
// kOutOfMemory (values 10, 11, 12) with their 2026-11-14 comments ("A transfer array was written for another set
// than material_is_counted"; "In a step: more liquid would leave a grid-only cell than it holds. Shorten the step";
// "In a step: a grid-only cell would hold more than kMaxFilledShare of its volume").
enum class CellResizeError : std::uint8_t {
  kNone,
  kBadInput,              // A setting is not finite or out of range, or an array has the wrong size.
  kWrongGridVersion,      // An array is bound to another grid version or block count than the grid named.
  kTooManyCellBlocks,     // The resized grid would pass SimulationLimits::max_cell_blocks. Nothing changed.
  kTooManyParticles,      // A particle split would pass a particle limit. Nothing changed.
  kAmountsDoNotBalance,   // Liquid particles hold another total after the resize than before.
  kNotFinite,             // A remapped value or a velocity the size rule reads is not finite.
  kModuleFailed,          // grid, particles, solids, gas or step returned a failure. See module, module_error.
  kRunOnCallFailed,       // CheckCanRunOn failed for the chosen processor, or a call to it failed.
  kOutOfMemory,           // CHANGED (item 31): last. Only from BuildCellResize: a piece of a RunMemory block (or a
                          //   buffer) could not be built; magnitude = the bytes asked. Nothing changed.
};
struct CellResizeFailure {             // The error value of every std::expected here (A2). Never allocates.
  CellResizeError error = CellResizeError::kNone;
  grid::CellId cell{};                 // First failing cell in the grid's fixed order. Zero when unused.
  std::uint64_t particle_id = 0;
  step::Module module = step::Module::kStep;  // Only with kModuleFailed.
  std::uint8_t module_error = 0;       // That module's error value.
  double magnitude = 0;                // In the unit named beside the error (m3 for the amount errors).
  double limit = 0;                    // Same unit.
};
std::string_view ToMessage(CellResizeError error);
// NEW (2026-11-14). Stable code: the enumerator's name without the k, in lower snake case
// ("too_many_cell_blocks"). The CLI and the add-on use it.
std::string_view ToName(CellResizeError error);
```

### 3.2 Cell size: grid's shared rule, `cell_size_change_math.h` (S), `velocity_gradient_math.h` (S), `cell_size_changes.h`

**Moved to grid (A41).** `cell_resize/cell_size_rule.h` is replaced by grid's `native/fluidity/grid/cell_size_rule.h`
(`fluidity::grid`, shared): `CellSizeSettings` and `CellSizeFacts` move there (this module's 2026-11-14 forms, with
`ComputeWantedCellWidth(settings, frame, facts)` and `ComputeCellSizeChange(settings, frame, facts)`, are **replaced**),
beside `CellWidthsByLevel`, `ComputeReferenceSpeed`, `ComputeWantedCellWidth(settings, widths, reference_speed,
facts)`, `ComputeWantedSizeLevel`, `ComputeNewCellSizeLevel` and (A78, A79, replacing `CellSizeRegion`, `ArtistCellSizes`
and `ComputeArtistSizeLevel`) `RefinementFactors`, `CellSizeFactorTable`, `ComputeCellSizeFactorCode`, exactly as `plan-decisions.md` "Owner answers A40–A64 applied" item 2 declares them (grid
design 3; the formulas are in 5.3 here). This module consumes them (section 4) and keeps the per-cell change, the
velocity gradient and the pass.

```cpp
// cell_size_change_math.h (S). CHANGED (A41, A63): kMergeReady and kKeepsItsSize added.
enum class CellSizeChange : std::uint8_t {
  kKeep,
  kSplit,          // cell_width > wanted width (at once, ported).
  kMayMerge,       // 2 x cell_width <= merge_width_share x wanted width, no veto (5.3 rule 5). The wait has not ended.
  kMergeReady,     // NEW (A63). kMayMerge, the sibling test held for merge_wait_duration and min_merge_wait_steps.
  kKeepsItsSize,   // NEW (A41, plan decision). Rule 2 (snow, springy solid, cut solid): never split or merged, and a
                   //   closure that reaches it refuses the split. In the byte so the GPU run reads back one array.
};
// CHANGED (A41, A63). The per-cell part of 5.3 rules 2, 3 and 5: kKeepsItsSize when
// holds_particles_that_keep_their_count; else kSplit when cell_width > wanted_width; else kMayMerge when
// 2 * cell_width <= merge_width_share * wanted_width and none of touches_air_gas_or_solid, touches_air_pocket,
// wants_smallest_cells, depth_below_surface < surface_reach_in_cell_widths * widths.width[smallest_size_level];
// else kKeep. Never kMergeReady. No FMA contraction.
FLUIDITY_CPU_AND_GPU CellSizeChange ComputeCellSizeChange(const grid::CellSizeSettings& settings,
                                                          const grid::CellWidthsByLevel& widths,
                                                          const grid::CellSizeFacts& facts, double wanted_width);

// velocity_gradient_math.h (S). NEW (A62). 1/s: the Frobenius norm of the full 3x3 gradient G[i][j] = du_i/dx_j at
// the cell centre from `velocity` (the committed velocity_after_pressure of the liquid set, one value per face
// place). Diagonal: (area-weighted mean of the +a face's pieces - that of the -a face) / cell width. Off-diagonal:
// the cell-centre u_i of the face-touching cell on the +j side minus that on the -j side, over the distance between
// the centres along j (a bigger neighbour: its centre value; four smaller ones: their mean in ascending CellId; a
// missing side: the cell itself at half the distance). Rows i = x, y, z summed in order; fixed order, no
// contraction. 0 unless the cell's CellContent is liquid. A uniform field gives 0 and a linear field its exact
// gradient where the neighbours are of the same size or smaller; beside a bigger neighbour the off-diagonal term is
// first order (its centre is w/2 off the axis).
FLUIDITY_CPU_AND_GPU double ComputeVelocityGradientSize(const grid::GridTables& tables, const double* velocity,
                                                        grid::CellPlaceIndex cell);

// cell_size_changes.h. NEW (A41, A62, A63). The pass of 5.3, on the grid's processor, after a committed step.
enum class CellSizeChangesPart : std::uint8_t {
  kWantedWidthsAndChanges,  // Part 1: facts, wanted width, ComputeCellSizeChange, held gradient.
  kMergeTest,               // Part 2: reads part 1's outputs of touching cells; wait counters; kMergeReady.
};
struct CellSizeChangesInputs {
  const grid::StaggeredGrid* grid;                              // Committed grid.
  const grid::CellValues<double>* depth_below_surface;          // m. ComputeDepthBelowSurface.
  pressure::SolidCoverage solids;                               // Last try's coverage (ResizeTry).
  const pressure::GeometrySnapshot* geometry;                   // Last try's geometry (content, air volume).
  const grid::CellValues<std::uint32_t>* pocket_of_cell;        // Committed GridFields.
  const grid::CellValues<double>* gas_volume;                   // Committed gas amounts; null without gas.
  const grid::FaceValues<double>* velocity_after_pressure;      // m/s. Committed, liquid set.
  const particles::ParticlesTry* particles_try;                 // Grouped for the committed grid.
  std::span<const std::uint8_t> material_keeps_its_count;       // One per material (CountRule::kKeepsItsCount).
  const grid::CellValues<double>* held_velocity_gradient_size;  // 1/s. Committed GridFields.
  const grid::CellValues<double>* merge_ready_duration;         // s. Committed GridFields.
  const grid::CellValues<std::uint8_t>* merge_ready_step_count; // Committed GridFields.
  grid::RefinementFactors refinement;                           // CHANGED (A78, A79): the current key's codes on the
                                                                //   grid's processor (CellResize's buffer).
  grid::CellSizeFactorTable factor_table{};                     // NEW (A78). From ComputeCellSizeFactorTable.
  const grid::CellSizeSettings* settings;
  grid::CellWidthsByLevel widths;                               // m. From the committed frame, on the CPU.
  double reference_speed = 0;                                   // m/s. grid::ComputeReferenceSpeed.
  double elapsed_since_last_merge_test = 0;                     // s of committed time since the last committed resize.
  std::uint32_t steps_since_last_merge_test = 0;                // Committed steps since then (saturates at 255).
};
struct CellSizeChangesOutputs {                                 // Bound to the committed grid.
  grid::CellValues<std::uint8_t>* size_change;                  // CellSizeChange per cell place.
  grid::CellValues<double>* wanted_width;                       // m. Part 1, read by part 2.
  grid::CellValues<double>* held_velocity_gradient_size_after;  // 1/s.
  grid::CellValues<double>* merge_ready_duration_after;         // s.
  grid::CellValues<std::uint8_t>* merge_ready_step_count_after;
};
// Part 1 per active cell of `blocks`: fills grid::CellSizeFacts (5.3), held = max(committed held, current gradient),
// wanted_width = grid::ComputeWantedCellWidth, size_change = ComputeCellSizeChange. Part 2: the merge test of 5.3
// rule 5 for the cell's 8 siblings and every cell touching their big cell; holds: duration + elapsed, step count +
// steps (saturating at 255), held kept; breaks: 0, 0, held = current gradient; kMayMerge becomes kMergeReady when
// the 8 siblings' smallest duration >= merge_wait_duration and smallest step count >= min_merge_wait_steps. A caller
// cutting the grid into ranges runs part 1 over every range before part 2 over any; any cut gives the same bytes,
// and CPU and GPU give the same bytes (local stencils, exact min and max, widths compared through `widths`). Fails
// kBadInput (a span of the wrong length), kWrongGridVersion, kNotFinite (a non-finite velocity in a liquid cell;
// first failing cell in fixed order), kRunOnCallFailed. Requires: every array and cell_resize's buffers on the
// grid's processor (A30).
std::expected<void, CellResizeFailure> ComputeCellSizeChanges(const CellSizeChangesInputs& inputs,
                                                              grid::CellBlockRange blocks, CellSizeChangesPart part,
                                                              CellSizeChangesOutputs outputs, CellResize& cell_resize);
```

### 3.3 `split_and_merge_cells.h` (CPU only): the cell list

```cpp
// D names. Both work on the sorted list of active cell ids, the only description of a grid.
// Replaces big_cell by its 8 small cells. Fails kBadInput when big_cell is not in the list or is at the
// smallest size level of the frame.
std::expected<void, CellResizeFailure> SplitCell(const grid::GridFrame& frame, grid::CellId big_cell,
                                                 std::vector<grid::CellId>& active_cells);
// Replaces the 8 sibling cells of big_cell by big_cell. Fails kBadInput unless all 8 are in the list.
std::expected<void, CellResizeFailure> MergeCells(const grid::GridFrame& frame, grid::CellId big_cell,
                                                  std::vector<grid::CellId>& active_cells);
// The cells that must split with `cells_to_split` so that the size-step rule holds afterwards, the given
// cells included, in the grid's fixed order. Fails kTooManyCellBlocks above max_cell_count.
std::expected<std::vector<grid::CellId>, CellResizeFailure> FindCellsThatMustSplit(
    const grid::GridTables& committed, std::span<const grid::CellId> cells_to_split, std::uint64_t max_cell_count);
// NEW (A30, reconcile item 32). Requires (all three functions): CPU tables and a CPU list; tables of a GPU grid are a
// debug assertion, not kBadInput.
struct ResizePlan {
  std::vector<grid::CellId> cells_to_split;      // Closed under the size-step rule. Fixed order.
  std::vector<grid::CellId> big_cells_to_merge;  // Each names 8 active sibling cells. Fixed order.
  std::uint32_t wanted_split_count = 0;
  std::uint32_t refused_split_count = 0;
  std::uint32_t refused_merge_count = 0;
};
```

### 3.4 `resize_cells.h`: the transaction and the step callbacks

```cpp
// Not in the first version (A64): RegionSettings, kMaxFilledShare, RegionTransferLedger, CellResizeSettings::regions,
// ResizeCellsReport::ledger, ReadLargestGridOnlySpeed (declarations below the first-version block, kept for the
// later group).
inline constexpr double kAmountTolerance = 1e-12;  // No unit, relative to the total. About 4,000 rounding steps.

// NEW (reconcile, decision 33). Who holds the two CPU grids (committed grid, resized grid) of a resize on a GPU grid.
// Kept (A41): the resize plan, the list edit and BuildResizeMap still read CPU tables in a GPU run.
enum class GridCopies : std::uint8_t {
  kNone,              // A CPU grid: no copy.
  kHeldByCellResize,  // One-piece GPU run: two CPU GridBuffers of this CellResize sharing one CPU grid::GridBuilderBuffers;
                      //   the committed grid is a grid::ReadGrid copy.
  kHeldByStep,        // Batched GPU run: the committed grid is step::FindCpuGridCopy(simulation); the resized CPU
                      //   grid is built into ResizeTry::free_cpu_grid_copy_buffers(). This module allocates none.
};
struct CellResizeSettings {
  grid::CellSizeSettings cell_size{};  // CHANGED (A41): grid's type.
  RunOn run_on = RunOn::kCpu;          // The caller's choice (A12). Never changed silently.
  // NEW (A17). The processor of the simulation's grid (the caller's choice, A12). With kGpu, ResizeCells reads a CPU
  // copy of the committed grid (as grid_copies says) and uploads the resized cell list (5.1, 6).
  RunOn grid_run_on = RunOn::kCpu;
  // NEW (reconcile). I/O sets it from the run plan's mode. kBadInput in BuildCellResize: kHeldByCellResize or
  // kHeldByStep with grid_run_on == kCpu, kNone with kGpu.
  GridCopies grid_copies = GridCopies::kNone;
};
struct ResizeCellsReport {
  bool resized = false;                // The grid, the wait arrays or the particles changed and was committed. CHANGED
                                       //   (A67): also a pass that changed no cell but split, merged or aged
                                       //   (merge_wait_passes) mismatched particles.
  ResizePlan plan{};
  // CHANGED (A67): the result of the extended particles::AdjustParticleCount of this pass, run at every pass. Its new
  // fields mismatched_particle_count, waiting_particle_count, lone_particle_count, incomplete_split_count and
  // kinetic_energy_change (J, after minus before) are the report's; ResizeCells copies kinetic_energy_change and
  // mismatched_particle_count into step::ResizeResult (below). All zero after a failed pass.
  particles::AdjustParticleCountResult particle_count{};
  LiquidTotals totals_before{};        // NEW (A64, plan decision; replaces the ledger's totals). Liquid particles.
  LiquidTotals totals_after{};
  grid::GridVersion grid_version{};    // After the call.
  CellResizeFailure failure{};         // kNone when nothing failed. A failure leaves the committed step as it was.
  std::uint64_t bytes_copied = 0;      // NEW (A17). Both directions, this call: the grid copy (456 B per committed
                                       //   block), the size_change read back (1 B per committed cell place, A41), the
                                       //   cell list upload (8 B per new cell), the CPU mirrors. In a batched GPU run
                                       //   without the 456 B per block: the resize reads step::FindCpuGridCopy (5.1).
  // NEW (plan decision, 2026-11-14, buffers in batched runs). One per ReadGrid, ReadBuffer or WriteBuffer call
  // between CPU memory and the grid's processor, this call. CHANGED (A41): no ChooseCellSize copy.
  std::uint32_t copy_count = 0;
};
class CellResize;                      // Settings, buffers of this module on one processor, the last report, the
                                       //   committed time and step count of the last committed merge test.
// CHANGED (A58; correction, reconcile): the run's two blocks last. kRunOn rows are pieces of run_memory, the kCpu
// rows kept kWholeRun (CPU grid copies, mirrors) pieces of cpu_run_memory; in a CPU run both are the same object.
// Requires: run_memory.run_on() == settings.run_on, cpu_run_memory.run_on() == kCpu (debug assertions, A30); both
// outlive the CellResize. kOutOfMemory (magnitude = bytes asked) when a piece does not fit.
// CHANGED (A78, A79; replaces the A63 regions and map): `refinement_layout` is the refinement volume's layout (I/O's
// RunScene::refinement_volume().layout, codes null; count 0 when the scene has no volume). BuildCellResize builds
// one buffer "cell resize: refinement factors" (the pyramid's code count, 1 B each, kRunOn) on run_on and keeps a
// RefinementFactors view on it; WriteRefinementFactors fills it per key. kBadInput: counts other than
// ceil(domain smallest cells / 2^sample_spacing_log2), sample_spacing_log2 > size_level_count - 1, a layer table
// that does not follow the halving rule.
std::expected<CellResize, CellResizeFailure> BuildCellResize(const CellResizeSettings& settings,
                                                             const step::SimulationLimits& limits,
                                                             std::uint32_t ingredient_count,
                                                             const grid::RefinementFactors& refinement_layout,
                                                             RunMemory& run_memory, RunMemory& cpu_run_memory);
// NEW (A78, A79). Copies one key's codes (CPU span, the whole pyramid) into the refinement buffer on run_on: one
// planned, counted copy in a GPU run (ResizeCellsReport::refinement_copy_bytes), a plain copy in a CPU run. The step
// calls it before ResizeCells when the key at or before the committed time changed, and once before the first resize.
// kBadInput: size other than the buffer's or a code outside [-24, 24]; kRunOnCallFailed.
std::expected<void, CellResizeFailure> WriteRefinementFactors(std::span<const std::int8_t> key_codes,
                                                              CellResize& cell_resize);
const ResizeCellsReport& ReadLastReport(const CellResize& cell_resize);
// Changed (A17): grid_run_on. With kGpu the rows add "cell resize: cell list upload" (8 B per cell place,
// where = kRunOn). A CellResize with run_on = kCpu in a GPU run also counts its CPU mirrors (where = kCpu, 5.8).
// Changed (reconcile, decision 33): grid_copies. With kHeldByCellResize the rows add "cell resize: committed grid
// copy" and "cell resize: resized grid on the CPU": two GridBuffers sharing one CPU GridBuilderBuffers, entries
// grid::BuildGridMemoryEstimate(kCpu, 2, limits.max_cell_blocks, grid::BuilderBuffersSharing::kOneForAllGrids) at limits.max_cell_blocks with
// where = kCpu; with kHeldByStep neither CPU grid is allocated or counted. Rows are named "cell resize: <row>",
// sizeof of the stored type. CHANGED (A41, A63, A58): "cell resize: size changes" (1 B per cell place), the wait
// counters' "after" rows, CHANGED (A78, A79) "cell resize: refinement factors" (1 B per pyramid code, kGivenCount;
// replaces the regions and map rows); every kRunOn row kWholeRun; one entry per buffer, so ComputeRunMemoryBytes of
// the entries per where equals what BuildCellResize hands out of each block.
std::vector<MemoryEstimateEntry> BuildCellResizeMemoryEstimate(std::uint32_t ingredient_count,
                                                               RunOn grid_run_on = RunOn::kCpu,
                                                               GridCopies grid_copies = GridCopies::kNone,
                                                               std::uint64_t refinement_code_count = 0);  // A78

// NEW (2026-11-14). Each step::StepCallbacks member is a step::Callback<Function>{function, context} pair. The
// I/O and CLI design sets resize_cells with `context` = the run's CellResize* (never null when the function is set;
// with no cell resize the pair stays empty), and run_stages with batches::ApplyStagesInBatches and `context` = the
// run's batches::BatchedRun* (3.7). No other design names a context type of this module.

// D name. Signature of step::StepCallbacks::resize_cells. Runs the transaction of 5.1. On any failure the
// committed step, the previous grid and the particles are exactly as before the call, time stays advanced,
// and the result has resized = false with the error number.
// CHANGED (A67): step::ResizeResult gains (step design 3.9; the step copies them into StepReport's fields of the same
// names) `double size_change_kinetic_energy = 0;` (J, = particle_count.kinetic_energy_change) and `std::uint64_t
// mismatched_particle_count = 0;` (= particle_count.mismatched_particle_count); both 0 when the pass failed or found
// nothing to do.
step::ResizeResult ResizeCells(void* context, step::Simulation& simulation);
// REMOVED (A41): ChooseCellSize and step::StepCallbacks::choose_cell_size. Stage 2 calls grid::ComputeNewCellSizeLevel
// on the grid's processor; no copies (the one-piece copies of decision 35 are gone).
```

**Not in the first version (A64)** (the plan's later group; text as of 2026-11-14):

```cpp
struct RegionSettings {                // D suffix.
  double particle_region_depth_in_cell_widths = 6;  // Ported. Shallower cells are carried by particles.
  double grid_only_depth_in_cell_widths = 7;        // Deeper full cells become grid-only. Between: no change.
  int overlap_layer_count = 2;                      // Ported (2 cell widths). Layers where face velocities blend.
  int particles_per_refilled_cell = 8;              // 1..9 (particles::ComputeRefillParticle).
};
inline constexpr double kMaxFilledShare = 2.0;     // No unit. current_volume / open volume of a grid-only cell.
// CellResizeSettings gains: RegionSettings regions{};  ResizeCellsReport gains: RegionTransferLedger ledger{};
struct RegionTransferLedger {          // One resize. Everything that changed holder, by direction.
  particles::ParticleAmounts into_grid_only_region{};   // Removed particles (RemoveReason::kIntoGridOnlyRegion).
  particles::ParticleAmounts into_particle_region{};    // Added particles (AddReason::kRefillRegion).
  std::uint64_t removed_particle_count = 0;
  std::uint64_t added_particle_count = 0;
  std::uint32_t cells_to_grid_only = 0;
  std::uint32_t cells_to_particles = 0;
  std::uint32_t refilled_cell_eighth_count = 0;
  Vec3 momentum_not_carried{};         // kg m/s. Momentum of removed particles minus what the face velocities
                                       //   already held for them. Reported, not corrected (5.6).
  // Liquid materials only (gas markers carry a weight, not a mass). Block partials, then math::TreeSum.
  double total_mass_before = 0;              // kg, particles + grid-only cells.
  double total_mass_after = 0;
  double total_current_volume_before = 0;    // m3.
  double total_current_volume_after = 0;
};
// m/s per cell width (1/s). Largest grid-only face speed of the latest try's MoveGridOnlyLiquid; the step reads it
// only in CommitStep, so it takes the committed try's value into kParticleTravel. 0 before the first step.
double ReadLargestGridOnlySpeed(const CellResize& cell_resize);    // NEW (2026-11-14).
// NEW (2026-11-14). Signature of step::StepCallbacks::apply_grid_only_region, called at the end of stage
// ParticlesToGrid, after every material set was transferred. Calls ApplyGridOnlyRegionToGrid once, for the
// liquid material set (a scene has exactly one; the other sets hold no grid-only record). A failure becomes kStageFailed (kValueNotFinite
// for kNotFinite) with module and error in the context.
step::StageOutcome ApplyGridOnlyRegionInStep(void* context, step::Simulation& simulation,
                                             step::StepTry& step_try, grid::CellBlockRange blocks);
// NEW (2026-11-14). Signature of step::StepCallbacks::move_grid_only_liquid, called at the end of stage
// MoveParticles on the whole grid. Calls MoveGridOnlyLiquid; kGridOnlyCellDrained and kGridOnlyCellTooFull
// become kParticleTravelledTooFar (a shorter step helps), with magnitude and limit as filled shares.
step::StageOutcome MoveGridOnlyLiquidInStep(void* context, step::Simulation& simulation,
                                            step::StepTry& step_try, grid::CellBlockRange blocks);
```

**Replaced (A41)** (the 2026-11-14 and reconcile text): "`grid::SizeLevel ChooseCellSize(void* context, const
grid::GridTables& committed, Vec3 position);` D name. Signature of `step::StepCallbacks::choose_cell_size`: the size
level of a cell that does not exist yet at `position` (5.3). Changed (reconcile, decision 35): the step passes
`committed_grid.tables(RunOn::kCpu)` ... with no current copy it makes one with `grid::ReadGrid` and counts it in the
next `ResizeCellsReport`."

### 3.5 `depth_below_surface.h`, `liquid_totals.h`; not in the first version: `grid_only_regions.h`

CHANGED (A64, plan decision): `regions.h` is split. The first version has the depth and the totals over particles;
the region types and operations are the later group's, in `grid_only_regions.h`.

```cpp
// depth_below_surface.h
// m, per cell centre, 0 in a liquid cell that touches air or tracked gas, growing through touching liquid
// cells by the distance between their centres, capped at max_depth. +infinity outside liquid.
std::expected<void, CellResizeFailure> ComputeDepthBelowSurface(
    const grid::StaggeredGrid& grid, grid::CellBlockRange blocks, const pressure::GeometrySnapshot& geometry,
    double max_depth, grid::CellValues<double>& depth_below_surface);
// ResizeCells passes max_depth = surface_reach_in_cell_widths * widths.width[0] (NEW, A64; 5.2).

// liquid_totals.h. kg, m3 over the particles of counted materials in the given blocks. For the resize's check, for
// render output's volume targets and for tests. One partial per block (its particles in grouped order), then
// math::TreeSum over blocks (5.10). material_is_counted: one entry per material, each 0 or 1, so gas markers, whose
// mass is a weight, are left out.
struct LiquidTotals {
  double mass = 0;                     // kg.
  double rest_volume = 0;              // m3.
  double current_volume = 0;           // m3.
};
// CHANGED (A64): particles only, no GridOnlyAmounts. CHANGED (A31): one form, taking the caller's SumBuffers
// (ResizeCells passes CellResize's, render output its recorder's); allocates nothing. kBadInput when blocks.count is
// above sum_buffers.max_count() or the mask has the wrong length. Requires: sum_buffers.run_on() is the grid's
// processor (debug assertion, A30).
std::expected<LiquidTotals, CellResizeFailure> ComputeLiquidTotals(
    const particles::ParticlesTry& particles_try, std::span<const std::uint8_t> material_is_counted,
    const grid::StaggeredGrid& grid, grid::CellBlockRange blocks, SumBuffers& sum_buffers);
```

**Not in the first version (A64)** (`grid_only_regions.h`, the plan's later group; text as of 2026-11-14; there
`ComputeLiquidTotals` takes `const GridOnlyAmounts& amounts` after `particles_try` and adds the records):

```cpp
enum class RegionKind : std::uint8_t { kParticleRegion, kGridOnlyRegion };  // D words (Q5).
// Plain views of the seven arrays step::GridFields holds for this module (section 4). A record holds one
// liquid material: the material whose ingredients (materials::MaterialTable::first_ingredient_of_material)
// carry its ingredient_volume (5.6).
struct GridOnlyAmounts {
  const grid::CellValues<std::uint8_t>* is_grid_only_region;  // 1: the cell's liquid is in the arrays below.
  const grid::CellValues<std::uint8_t>* layers_from_grid_only_region;  // 0 inside, 1 for a touching cell, ...
  const grid::CellValues<double>* mass;            // kg. Zero in a particle-region cell.
  const grid::CellValues<double>* rest_volume;     // m3.
  const grid::CellValues<double>* current_volume;  // m3. May differ from the open volume of the cell (5.6).
  const Buffer<double>* ingredient_volume;         // m3. [cell place * ingredient_count + scene ingredient].
  std::uint32_t ingredient_count = 1;              // Ingredients of the whole scene.
};
struct GridOnlyAmountsToChange;        // The same arrays, writable: the try copy.

struct GridOnlyToGridInputs {
  const grid::StaggeredGrid* grid;
  grid::CellBlockRange blocks;
  // NEW (2026-11-14, A8). The material set being transferred (transfer's material_is_counted). Only records
  // of a counted material are added; a set with no liquid material is left untouched.
  std::span<const std::uint8_t> material_is_counted;
  std::span<const std::uint32_t> first_ingredient_of_material;  // NEW (2026-11-14). From the material table.
  GridOnlyAmounts amounts;                                 // Try copy, bound to this grid.
  const grid::FaceValues<double>* committed_velocity;      // m/s. step::GridFields::velocity_after_pressure.
  const transfer::SolidsAtFaces* solids_at_start;
  int overlap_layer_count = 2;
  double step_duration = 0;                                // s.
};
// Inside stage ParticlesToGrid, after transfer wrote this set's arrays (5.5). Adds the amounts of grid-only
// cells to cells_from_particles; gives every face piece of a grid-only cell its carried_velocity and its mass
// with FaceValueKind::kFromGridOnlyRegion; blends the two velocities in the overlap layers. Fails kBadInput,
// kWrongGridVersion (an array whose grid_version() or block_count() is not the grid's), kWrongMaterialSet
// (grid_from_particles or cells_from_particles holds another MaterialSetKey), kNotFinite, kRunOnCallFailed.
std::expected<void, CellResizeFailure> ApplyGridOnlyRegionToGrid(
    const GridOnlyToGridInputs& inputs, transfer::GridFromParticles& grid_from_particles,
    transfer::CellsFromParticles& cells_from_particles, CellResize& cell_resize);
// m/s on one face piece: the committed velocity read where the liquid now at the face centre was one step
// ago (5.5). (S), in carried_velocity_math.h. On failure writes the error and returns 0.
FLUIDITY_CPU_AND_GPU double ComputeCarriedVelocity(const grid::GridTables& tables, const double* committed_velocity,
                                                   grid::FacePlaceIndex face, double step_duration,
                                                   CellResizeError* error);

struct MoveGridOnlyLiquidInputs {
  const grid::StaggeredGrid* grid;
  grid::CellBlockRange blocks;
  pressure::GeometrySnapshot geometry;                       // open_area and open_volume of this try.
  const grid::FaceValues<double>* velocity_after_pressure;   // m/s. Of the set holding the record's material.
  GridOnlyAmounts amounts_at_step_start;
  double step_duration = 0;                                  // s.
};
struct MoveGridOnlyLiquidResult {
  double largest_filled_share = 0;     // No unit. Largest current_volume / open volume after the move.
  double moved_volume = 0;             // m3, sum of absolute volumes through face pieces. Block partials, then
                                       //   math::TreeSum.
  double largest_speed_in_cell_widths = 0;  // 1/s. For the next step duration.
};
// After stage MoveParticles (5.5). Moves amounts between grid-only cells with the solved velocity. Writes the
// try copy of the amounts. Fails kGridOnlyCellDrained, kGridOnlyCellTooFull, kNotFinite, kWrongGridVersion.
std::expected<MoveGridOnlyLiquidResult, CellResizeFailure> MoveGridOnlyLiquid(
    const MoveGridOnlyLiquidInputs& inputs, GridOnlyAmountsToChange amounts, CellResize& cell_resize);

// The region transfer of 5.6 on a resize try whose fields are already bound to the resized grid. D word
// region_transfer.
std::expected<RegionTransferLedger, CellResizeFailure> ApplyRegionTransfer(
    step::ResizeTry& resize_try, const grid::StaggeredGrid& resized_grid, const RegionSettings& settings,
    CellResize& cell_resize);
```

### 3.6 `remap_grid_values.h` and `remap_math.h` (S)

```cpp
enum class CellSource : std::uint8_t {
  kSameCell,           // The cell existed with this id. Its value is copied.
  kSmallCellOfSplit,   // One of the 8 small cells of a cell that was split.
  kMergedCell,         // The big cell of 8 cells that were merged.
  kNewCell,            // No cell covered this place before. Does not occur in ResizeCells; kept for the step.
};
class ResizeMap;                       // For each cell and face place of the new grid: its source kind and
                                       //   the place index(es) in the old grid. Lives on one processor.
// Built on the CPU from the two grids' tables and copied up for a GPU remap. Requires: two CPU grids (A30, item 32). Changed (A17): on a GPU grid the two
// grids are CPU grids: the committed grid's grid::ReadGrid copy (in a batched GPU run the step's CPU copy,
// reconcile) and the resized grid built on the CPU from the new list; the map checks that the resized grid matches the version and block count of the one the step built
// on the GPU (kWrongGridVersion otherwise).
std::expected<ResizeMap, CellResizeFailure> BuildResizeMap(const grid::StaggeredGrid& old_grid,
                                                           const grid::StaggeredGrid& new_grid,
                                                           CellResize& cell_resize);
enum class CellValueKind : std::uint8_t {
  kAmount,           // m3, kg: split shares it out (limited slope), merge sums. Totals are kept.
  kPerUnitOfWeight,  // Pa, K, kg/m3: split copies; merge is the mean weighted by `weight` (equal if null).
  kLabel,            // Integers: split copies; merge takes the smallest value that is not `no_label`.
};
// new_values must be bound to the map's new grid, old_values and old_weight to its old grid
// (kWrongGridVersion otherwise).
template <typename Value> std::expected<void, CellResizeFailure> ApplyResizeToCellValues(
    const ResizeMap& map, CellValueKind kind, const grid::CellValues<Value>& old_values,
    const grid::CellValues<double>* old_weight, grid::CellValues<Value>& new_values);
// NEW (plan decision, decision 23). Not in the first version (A64). The three grid-only amounts (mass, rest volume,
// current volume) remapped together with one limiter per split cell, so a split keeps their ratios (density); merge
// sums.
std::expected<void, CellResizeFailure> ApplyResizeToCellAmounts(
    const ResizeMap& map, std::span<const grid::CellValues<double>* const> old_amounts,
    const grid::CellValues<double>* old_weight, std::span<grid::CellValues<double>* const> new_amounts);
// NEW (2026-11-14). Not in the first version (A64).
// m3 per cell place and scene ingredient (the grid-only record): split gives each small cell
// the big cell's fractions times its own new current_volume; merge sums in index order.
std::expected<void, CellResizeFailure> ApplyResizeToIngredientVolumes(
    const ResizeMap& map, std::uint32_t ingredient_count, const Buffer<double>& old_ingredient_volume,
    const grid::CellValues<double>& old_current_volume, const grid::CellValues<double>& new_current_volume,
    Buffer<double>& new_ingredient_volume);
// m/s on face pieces. Keeps the volume flow through every outer face of a resized cell and gives the 8 small
// cells of a split the flow imbalance per volume of their big cell (5.4).
std::expected<void, CellResizeFailure> ApplyResizeToFaceVelocity(
    const ResizeMap& map, const grid::FaceValues<double>& old_velocity, grid::FaceValues<double>& new_velocity);

// remap_math.h (S)
struct SiblingAmounts { double small_cells[8]; };  // Index = cell_eighth = x + 2 y + 4 z (grid 3.1).
double ComputeMergedAmount(const SiblingAmounts& amounts);                 // Sum in index order.
// neighbours[axis][0 = low, 1 = high]: the amount of the same-size cell on that side; has_neighbour false
// where there is none of that size and region. The 8 results sum to big_cell_amount.
SiblingAmounts ComputeSplitAmounts(double big_cell_amount, const double neighbours[3][2],
                                   const bool has_neighbour[3][2], double limiter);
double ComputeSplitLimiter(double big_cell_amount, const double neighbours[3][2], const bool has_neighbour[3][2]);
struct InnerFaceFlows { double flows[12]; };       // m3/s along +axis. Order: axis, then the two other coords.
// outflow_through_outer_faces[k]: m3/s leaving small cell k through its 3 outer sides after the outer faces
// were remapped. Returns inner flows so that every small cell has net outflow big_cell_outflow / 8.
InnerFaceFlows ComputeInnerFaceFlows(const double outflow_through_outer_faces[8], double big_cell_outflow);
```

### 3.7 `batches/`: `batches_error.h` (S), `batch_plan.h`, `batched_run.h`

```cpp
enum class BatchesError : std::uint8_t {
  kNone,
  kBadInput,
  kBatchDoesNotFit,       // One cell block with its stacked border needs more than the allowed bytes.
  kStateDoesNotFitCpu,    // The state does not fit the CPU memory offered.
  kWrongGridVersion,
  kModuleFailed,          // A stage or a copy failed. See module, module_error.
  kRunOnCallFailed,
  kOutOfMemory,           // CHANGED (reconcile item 31): last. Only from BuildBatchedRun; needed_bytes = bytes asked.
};
// NEW (A68). The slab limit of this version's batches: a batch is a horizontal slab of whole block layers at the
// smallest cell (5.9 item 4), so a scene's plan area is capped. The numbers I/O's budget line names (I/O
// BuildBudgetLines, ModeReason::kLayerDoesNotFitProcessor). Plain data, CPU and GPU.
struct SlabLimit {
  std::uint32_t blocks_per_layer = 0;     // (domain cells along x / 4) * (along y / 4) at the smallest width.
  std::uint32_t loaded_layer_count = 0;   // Layers the thinnest allowed slab loads in pass 1: T_min + 8, with
                                          //   T_min = max(1, ceil(min_blocks_per_batch / blocks_per_layer)); so >= 9.
                                          //   0: no slab limit was computed (the failure was not the pass slab's).
  std::uint64_t bytes_per_full_loaded_block = 0;  // bytes_per_loaded_block + particles_per_block x
                                                  //   bytes_per_loaded_particle (512 particles x headroom, A65).
  std::uint64_t loaded_batch_room_bytes = 0;      // Processor bytes left for the loaded batch: allowed - whole, or
                                                  //   (1 - solve_piece_share) x room with kInPieces (5.9 items 1, 3).
  std::uint64_t needed_bytes = 0;         // loaded_layer_count x blocks_per_layer x bytes_per_full_loaded_block.
  double plan_area = 0;                   // m². Domain extent along x times along y.
  double max_plan_area = 0;               // m². loaded_batch_room_bytes x 16 w_min² / (loaded_layer_count x
                                          //   bytes_per_full_loaded_block): the largest plan area a slab batch holds.
};
struct BatchesFailure {                // The error value of every std::expected here (A2).
  BatchesError error = BatchesError::kNone;
  grid::CellBlockId block{};           // First block of the failing range. Zero when unused.
  step::Module module = step::Module::kStep;  // Only with kModuleFailed.
  std::uint8_t module_error = 0;
  std::uint64_t needed_bytes = 0;
  std::uint64_t allowed_bytes = 0;
  // NEW (A68). Filled by BuildBatchPlan with kBatchDoesNotFit when the pass slab (T < 1 or max_blocks_per_batch <
  // min_blocks_per_batch, 5.9 item 4) failed; then needed_bytes = slab_limit.needed_bytes and allowed_bytes =
  // slab_limit.loaded_batch_room_bytes. loaded_layer_count == 0 for every other failure (the solve piece's three
  // layers included).
  SlabLimit slab_limit{};
};
std::string_view ToMessage(BatchesError error);
std::string_view ToName(BatchesError error);   // NEW (2026-11-14). Stable code ("batch_does_not_fit").

enum class StagePlace : std::uint8_t {
  kNotRun,
  kWholeStateOnCpu,       // The CPU implementation, on the state where it lives. No copy.
  kInBatchesBeforeSolve,  // Pass 1: on the run's processor, one loaded batch after another.
  kWholeGridOnProcessor,  // On the run's processor, on arrays of the whole grid that stay there. No pass batch
                          //   (with SolvePlace::kInPieces the solvers' own sweeps load solve batches).
  kInBatchesAfterSolve,   // Pass 2.
};
// The table of 5.8. Changed (plan decision, 2026-11-14, buffers in batched runs): the placement rule computed from
// it gives every step::StepArray its whole_place (5.7).
extern const std::array<StagePlace, step::kStageCount> kStagePlaces;
// Changed (A15): the design's kWholeGridOnCpu (a CPU solve inside a GPU run, against A17) is not built.
enum class SolvePlace : std::uint8_t { kWholeGridOnProcessor, kInPieces };

struct BatchSettings {
  double max_whole_grid_share = 0.5;   // No unit. Above this share of allowed_bytes the solves run in pieces.
  std::uint32_t min_blocks_per_batch = 64;  // Below this a batch is mostly border: kBatchDoesNotFit.
  // NEW (A15). Share of the GPU room left after the whole rows that goes to the loaded solve piece; the rest to
  // the loaded batch. Owner to confirm (section 11, Open).
  double solve_piece_share = 0.25;
};
struct BatchPlan {
  SolvePlace solve_place = SolvePlace::kWholeGridOnProcessor;
  std::uint64_t whole_grid_bytes = 0;        // Arrays of the whole grid kept on the GPU for the run.
  std::uint64_t bytes_per_loaded_block = 0;  // Grid-shaped rows of one loaded cell block, every material set.
  std::uint64_t bytes_per_loaded_particle = 0;
  std::uint32_t max_loaded_blocks = 0;       // Range plus stacked border.
  std::uint64_t max_loaded_particles = 0;    // NEW (2026-11-14). Per material: the loaded particles' limit.
  std::uint32_t max_blocks_per_batch = 0;    // Largest range of the start grid whose load fits (5.9).
  std::uint32_t batch_count = 0;             // For the start grid.
  double range_share_of_loaded = 0;          // 0..1. Blocks of ranges over blocks loaded, summed over batches.
  std::uint64_t bytes_copied_per_try = 0;    // Estimate, both directions, both passes.
  // NEW (A15). The report shown before the run (A22 "budget"). The solve fields are 0 with kWholeGridOnProcessor.
  std::uint32_t max_solve_loaded_blocks = 0;     // Loaded solve piece (range plus stacked border); to RunPlan.
  std::uint32_t solve_batch_count = 0;           // Start grid, cut as pressure::ApplySolveBatches would cut it.
  std::uint64_t bytes_on_processor = 0;          // Memory budget: the rows with where = kRunOn.
  std::uint64_t bytes_on_cpu = 0;                // Memory budget: the rows with where = kCpu.
  std::uint32_t pass_loads_per_try = 0;          // 2 * batch_count.
  std::uint32_t solve_loads_per_iteration = 0;   // Sweeps per liquid-solve iteration (3, or 4 when the scene can
                                                 //   hold a sealed blob: a closed domain) times solve_batch_count.
  std::uint64_t solve_bytes_per_iteration = 0;   // That many sweeps of the streamed rows of every loaded block.
  SlabLimit slab_limit{};                        // NEW (A68). Also on success, for the report.
};
// Allocates nothing; the start grid is not built (it cuts by layers of blocks, 5.9). Works on a machine with no
// GPU. Chooses kWholeGridOnProcessor or kInPieces (5.9); fails kBatchDoesNotFit (a layer, or the solve piece's
// three layers, does not fit; NEW (A68): with BatchesFailure::slab_limit for the layer), kStateDoesNotFitCpu (the
// kCpu rows exceed RunPlan::allowed_cpu_bytes). CHANGED (A66): "start blocks" are SceneCounts::start_cell_blocks, the
// block count of step::BuildStartCells' grid (the cell list itself is not needed).
// Changed (plan decision, 2026-11-14, buffers in batched runs): every row sum takes the rows of
// step::BuildStepMemoryEstimate(scene, counts, &probe_plan) (5.9 item 7), so bytes_on_processor and bytes_on_cpu
// equal what step::BuildSimulation allocates for the copied plan, by where.
std::expected<BatchPlan, BatchesFailure> BuildBatchPlan(const step::ReadyScene& scene, const step::RunPlan& plan,
                                                        const step::StepSettings& step_settings,
                                                        const BatchSettings& settings);
struct BatchReport {                   // One try.
  std::uint32_t batch_count = 0;
  std::uint64_t bytes_to_processor = 0;
  std::uint64_t bytes_from_processor = 0;     // Includes the piece map read back at each load (A17).
  double copy_elapsed_seconds = 0;     // Clock time spent in WriteBatch and ReadBatch.
  // NEW (A15). Sums of the solvers' pressure::SolveCopyReport of the try (0 with kWholeGridOnProcessor).
  std::uint32_t solve_batch_loads = 0;
  std::uint64_t solve_bytes_to_processor = 0;
  std::uint64_t solve_bytes_from_processor = 0;
  double solve_copy_elapsed_seconds = 0;
};
// The plan, a grid::BorderSearchBuffers and two Buffer<grid::CellBlockIndex> of max_loaded_blocks on the run's
// processor (A17), a CPU copy of the piece map (4 B per loaded block, row "batches: piece map copy",
// where = kCpu), the current grid::GridPiece (built into step::FindLoadedGridBuffers), the per-block partial sums
// and the report. The loaded StepTry and its GridBuffers are built by step::BuildSimulation, not here.
class BatchedRun;
// CHANGED (A58; correction, reconcile): the run's two blocks last; the kRunOn entries of BuildBatchesMemoryEstimate
// are pieces of run_memory, the piece map copy a piece of cpu_run_memory (the same object in a CPU run). Requires:
// run_memory.run_on() is the plan's processor, cpu_run_memory.run_on() == kCpu (A30).
std::expected<BatchedRun, BatchesFailure> BuildBatchedRun(const BatchPlan& plan, const step::ReadyScene& scene,
                                                          RunMemory& run_memory, RunMemory& cpu_run_memory);
const BatchReport& ReadLastReport(const BatchedRun& batched_run);
// CHANGED (A58): exactly what BuildBatchedRun builds, all kWholeRun: "batches: border search"
// (grid::BuildBorderSearchMemoryEstimate(run_on, limits.max_cell_blocks) at max_loaded_blocks), "batches: border layer a", "batches: border
// layer b" (4 B per loaded block each), "batches: sum buffers", "batches: sum buffers fixed bytes" (where = kRunOn),
// "batches: piece map copy" (4 B per loaded block, where = kCpu). The step adds them to its estimate (kBatches), so
// its blocks hold them. **Replaced:** "the loaded rows at max_loaded_blocks ... For reporting only" (those rows are
// the step's batched rows, 5.9 item 7, and are not repeated).
std::vector<MemoryEstimateEntry> BuildBatchesMemoryEstimate(const BatchPlan& plan);

// D signature of step::StepCallbacks::run_stages, the loop the step calls instead of its own when the run is
// batched. `context` is the run's BatchedRun*. Runs stages first..last of
// `stages` at the places of kStagePlaces, the transfer stages once per material set inside each batch.
// Between stages and between batches it reads the try's cancel flag and returns kCancelRequested.
step::StageOutcome ApplyStagesInBatches(void* context, step::Simulation& simulation, step::StepTry& step_try,
                                        const step::StageTable& stages, step::Stage first, step::Stage last);
// CPU memory to the GPU for one range: releases the previous piece, builds the grid::GridPiece of the range and
// its stacked border into the loaded GridBuffers (A17: the buffer form of grid::BuildGridPiece, border from
// FindStackedBorder; then one ReadBuffer of block_in_whole_grid_buffer() into the map copy, counted in
// bytes_from_processor; a CPU run uses block_in_whole_grid()), step::ApplyGridPiece(loaded try, piece, map copy)
// (binds every loaded array to the piece's grid again with grid::ApplyGridVersion: a piece has the whole grid's
// version, so a stale binding is caught only by block_count(); keeps the CPU map for stage 18's
// FoamAndSprayInputs::block_in_whole_grid, plan decision 2026-11-14), particles::WriteParticleGroups, then step::WriteStepArrayBlocks (whole ->
// loaded, one step::BlockCopy per run of loaded blocks, runs computed on the CPU from the map copy) for every array
// a stage of the pass reads.
std::expected<void, BatchesFailure> WriteBatch(BatchedRun& batched_run, step::StepTry& step_try,
                                               grid::CellBlockRange range, StagePlace pass);
// GPU to CPU memory for the blocks of the range only: particles::ReadParticleGroups (pass 1 {kPlateSide}, pass 2
// every field BuildFieldsToChange accepts), step::WriteStepArrayBlocks (loaded -> whole) of every array a stage of
// the pass wrote that a later place reads or the commit keeps (for whole-grid arrays the slice copy on the GPU of
// 5.8). After pass 1 also pressure::WriteSnapshotRange(loaded try's piece buffers, piece, whole buffers) (A15) and
// solids::WriteBlockPartialsRange(loaded try's SolidsBuffers, piece, *step::FindContactReactionBuffers(simulation))
// (plan decision, 2026-11-14, buffers in batched runs), its bytes added to bytes_from_processor (part of the load).
std::expected<void, BatchesFailure> ReadBatch(BatchedRun& batched_run, step::StepTry& step_try,
                                              grid::CellBlockRange range, StagePlace pass);
// A sum over the whole grid from one partial per cell block: math::TreeSum over the blocks by block index.
// ComputeTreeSum over block_count() values with `sum_buffers`, allocating nothing; a block count above
// sum_buffers.max_count() or a failed GPU read returns a quiet NaN (documented). CHANGED (A31): one form, taking
// the caller's SumBuffers (BatchedRun::sum_buffers in a run). Requires: sum_buffers.run_on() is the values'
// processor (debug assertion, A30).
double ComputeSumOverBlocks(const grid::CellBlockValues<double>& block_partial_sums, SumBuffers& sum_buffers);
```

`stacked_border.h` (private, CPU only; stated here because it is the A17 form of the stacked border of 5.8):

```cpp
struct StackedBorder { const Buffer<grid::CellBlockIndex>* blocks; std::uint32_t count; };
// One buffer-form grid::FindBorderBlocks per layer, the two buffers used in turn; only counts come back. Pass 1
// widths {kTouchingCells, kTouchingCells, kFourCellWidths, kTouchingCells}, pass 2 {kFourCellWidths,
// kTouchingCells}, listed from the last stage backwards.
std::expected<StackedBorder, BatchesFailure> FindStackedBorder(const grid::StaggeredGrid& grid,
    grid::CellBlockRange range, std::span<const grid::BorderWidth> widths_from_last_stage,
    grid::BorderSearchBuffers& search_buffers, Buffer<grid::CellBlockIndex>& layer_a,
    Buffer<grid::CellBlockIndex>& layer_b);
```

**The three ways to run (NEW, A15).** Declared by the step (`run_plan.h`); `step::BuildRunPlan` still chooses only
`kOnePiece` or `kBatched`; the caller (I/O `Run`, tests) copies `BuildBatchPlan`'s choice into the plan
(`kInPieces` gives `kBatchedWithSolvesInPieces`):

```cpp
enum class RunMode : std::uint8_t { kOnePiece, kBatched, kBatchedWithSolvesInPieces };
// step::RunPlan gains:
std::uint32_t max_solve_loaded_blocks = 0;  // From BatchPlan.
std::uint64_t allowed_cpu_bytes = 0;        // Step 3 of BuildRunPlan applied to on_cpu.
```

`kBatched`: particle stages in batches, every solve once on the whole grid on the run's processor.
`kBatchedWithSolvesInPieces`: as `kBatched`, and the solvers' working buffers live in CPU memory and go through the GPU
one loaded piece at a time (pressure's `SolveBatches`, built by the step into `PressureBuffers`, `ViscosityBuffers`
and `GasBuffers` with `max_loaded_blocks = RunPlan::max_solve_loaded_blocks`). A CPU run can be given
`max_loaded_blocks > 0` too (tests in the container): the loaded piece is then on the CPU and every copy is CPU to
CPU, counted the same way.

## 4. Consumes from other modules

Each item is marked EXISTS (Foundation, including the `base/` additions) or PROVIDED (by a designed module,
with the name its section 3 has; for solids, gas, step, I/O and CLI, surface with foam and spray, and the
Blender add-on, which are updated at the same time or later, the name `changes-to-apply.md` gives).

**Foundation (EXISTS, `shared-interface.md` (a) and `changes-to-apply.md` "Additions to Foundation
`base/`"):** `Vec3`, `Mat3`, `Axis`, `RunOn`, `CheckCanRunOn`, `FLUIDITY_CPU_AND_GPU`, `Buffer`, `BuildBuffer`,
`WriteBuffer`, `ReadBuffer`, `math::TreeSum`, `math::Min`, `Max`, `Abs`, `IsFinite`, `ComputeTreeSum`,
`ComputeLargestMagnitude`, `BuildStableOrder`, `BuildGroupsByDestination`, `MemoryEstimateEntry`, `CountedPer`,
`KeptFor`, `MemoryCounts`, `ComputeMemoryBytes`, `ReadMemoryAvailable`, and:

```cpp
// base/buffer.h. CopyBuffer on the processor both live on, slice forms taking (first, count), CpuMemory.
template <typename Value> std::expected<void, BufferError> CopyBuffer(const Buffer<Value>& from, Buffer<Value>& to);
template <typename Value> std::expected<void, BufferError> CopyBuffer(const Buffer<Value>& from, std::size_t first,
                                                                      std::size_t count, Buffer<Value>& to,
                                                                      std::size_t to_first);
template <typename Value> std::expected<void, BufferError> WriteBuffer(std::span<const Value> values,
                                                                       Buffer<Value>& buffer, std::size_t first);
template <typename Value> std::expected<void, BufferError> ReadBuffer(const Buffer<Value>& buffer,
                                                                      std::size_t first, std::span<Value> values);
enum class CpuMemory : std::uint8_t { kOrdinary, kLockedForCopies };   // Optional argument of BuildBuffer.
struct MemoryAvailable { std::uint64_t total_bytes; std::uint64_t free_bytes; };   // total 0: unknown.
std::expected<MemoryAvailable, BufferError> ReadMemoryAvailable(RunOn run_on);    // base/processor_description.h
// Added 2026-11-14 (plan-decisions.md, Foundation): MemoryEstimateEntry has a field `MemoryWhere where { kRunOn, kCpu }`;
// this form sums only the rows kept there. Every row of this module sets it; BatchPlan's budgets use it.
std::uint64_t ComputeMemoryBytes(std::span<const MemoryEstimateEntry> rows, const MemoryCounts& counts, MemoryWhere where);
```

NEW (A58; Foundation Tasks 17 to 19, `base/run_memory.h`; `plan-decisions.md` "Owner answers A40–A64 applied" item 3
with its correction): `RunMemory` (move-only, one block on one processor; `run_on()`, `reserved_bytes()`,
`handed_out_bytes()`), `BuildRunMemory`, `BuildBuffer(RunOn, std::size_t, RunMemory&)`, `kRunMemoryPieceAlignmentBytes =
256`, `ComputeRunMemoryPieceBytes`, `ComputeRunMemoryBytes(estimate, counts, MemoryWhere where)`,
`BufferError::kRunMemoryTooSmall`, the `RunMemory&` forms of `BuildSumBuffers` and `BuildOrderBuffers`. A piece is
never freed alone; the block must outlive its pieces.

Buffers (and working buffers) passed together live on the same processor: a documented precondition, checked by
an assertion in debug builds (A30; `BufferError::kRunOnMismatch` is removed, so this module maps nothing for it).
Every function below that takes buffers states it as `Requires:`.

**Grid (PROVIDED, grid 3.1 to 3.7):** `StaggeredGrid` (move-only; `version()`, `counts()`, `all_blocks()`,
`tables()`, `max_cell_blocks()`), `GridTables`, `GridFrame`, `GridInputs`, `GridSettings`, `GridVersion` (a
built grid has 1 or more), `GridFailure`, `CellId`, `FacePieceId`, `CellBlockId`, `SizeLevel`, `ToCellId`,
`ToSizeLevel`, `ToCellCoord`, `ComputeBigCell`, `ComputeSmallCell`, `ComputeCellEighth`, `ComesBefore`,
`GridBuffers`, `BuildGrid(const GridInputs&, GridBuffers&)`, `TakeBuffers(StaggeredGrid&&)`,
`ComputeGridCounts`, `CheckSizeStepRule`, NEW (A41, grid `cell_size_rule.h`, `plan-decisions.md` item 2):
`CellSizeSettings`, `CellSizeFacts`, `CellWidthsByLevel`, (A78, A79) `RefinementFactors`, `CellSizeFactorTable`,
`ComputeCellSizeFactorTable`, `ComputeCellSizeFactorCode`, `ComputeReferenceSpeed`,
`ComputeWantedCellWidth`, `ComputeWantedSizeLevel` (and `ComputeNewCellSizeLevel`, which
the step calls in stage 2), `CellPlaceIndex`, `FindCellPlace`, `FindFacePlace`, `FindCellAt`, `FindTouchingCells`,
`FindFacePieces`, `FindCellLow`, `FindCellHigh`, `ToAxis`, `ComputeFaceArea`, `ComputeCellWidth`,
`ComputeCellCenter`, `ComputeCellCenterDistance`, `ComputeFacePosition`, `FindNearbyFaces`, `ToMaskOfAll`,
`ComputeNearbyFaceWeights`, `ComputeFaceWeight`, `CellValues`, `FaceValues`, `CellBlockValues` (with
`grid_version()`, `block_count()`, `buffer()`), `BuildCellValues(RunOn, std::uint32_t max_cell_blocks)` and the
face and block forms, `ApplyGridVersion(values, grid)`, `ReadValues`, `WriteValues`, `CellBlockRange`,
`BuildRanges`, `BorderWidth`, `FindBorderBlocks` (the vector form, CPU grid only, returning
`std::expected<std::vector<CellBlockIndex>, GridFailure>`; the form writing into a `std::span<CellBlockIndex>` is
retired, grid Ruling 66: a CPU run uses the buffer form with CPU buffers where it must not allocate), `GridPiece` (`grid()`, `range()`, `range_in_whole_grid()`,
`block_in_whole_grid()`; blocks in the whole grid's fixed order), `BuildGridPiece(const StaggeredGrid& whole,
CellBlockRange range, std::span<const CellBlockIndex> border_blocks, GridBuffers& storage)` (CPU whole grid),
`TakeBuffers(GridPiece&&)`, `FacePieceKind::kOutsidePiece`, `BuildGridMemoryEstimate(RunOn run_on,
std::uint32_t storage_count, BuilderBuffersSharing builder_buffers_sharing = BuilderBuffersSharing::kOnePerGrid)` (changed,
reconcile: this module counts the piece storage with `(run_on, 1, kOneForAllGrids)` at `max_loaded_blocks` without
the `builder buffers:` entries, and its two CPU grids with `(kCpu, 2, kOneForAllGrids)`). The
border of `kTouchingCells` includes blocks storing a face piece of a touching cell (grid 5.5); the stacked
border of 5.8 relies on it. Values bound to a piece are bound again at every load, and every check compares
both `grid_version()` and `block_count()`. `CellBlockRange` is in the shared `grid_ids.h`.

Grid, NEW (A17; grid 3.3, 3.4): `tables(run_on)` gives tables only on the grid's own processor; the buffer forms
used in a GPU run (and on CPU buffers in a CPU run where named):

```cpp
// GridInputs gains a list in a buffer on the buffers' processor.
const Buffer<CellId>* active_cell_buffer = nullptr;  std::uint32_t active_cell_buffer_count = 0;
std::expected<StaggeredGrid, GridFailure> ReadGrid(const StaggeredGrid& grid, GridBuffers& cpu_buffers);
class BorderSearchBuffers { public: std::uint32_t max_cell_blocks() const; RunOn run_on() const; };  // Move-only.
std::expected<BorderSearchBuffers, GridFailure> BuildBorderSearchBuffers(RunOn run_on,
                                                                         std::uint32_t max_cell_blocks);
std::vector<MemoryEstimateEntry> BuildBorderSearchMemoryEstimate(RunOn run_on, std::uint32_t max_cell_blocks);
std::expected<std::uint32_t, GridFailure> FindBorderBlocks(const StaggeredGrid& grid, CellBlockRange range,
                                                           const Buffer<CellBlockIndex>& border_so_far,
                                                           std::uint32_t border_so_far_count, BorderWidth width,
                                                           BorderSearchBuffers& search_buffers,
                                                           Buffer<CellBlockIndex>& border_blocks);
std::expected<GridPiece, GridFailure> BuildGridPiece(const StaggeredGrid& whole, CellBlockRange range,
                                                     const Buffer<CellBlockIndex>& border_blocks,
                                                     std::uint32_t border_block_count, GridBuffers& buffers);
const Buffer<CellBlockIndex>& GridPiece::block_in_whole_grid_buffer() const;  // Piece map on grid().run_on().
// NEW (reconcile; grid 3.3): one builder buffers lent to several buffers. GridSettings::max_cell_blocks has no
// default room (0 is kBadInput).
class GridBuilderBuffers { public: std::uint32_t max_cell_blocks() const; RunOn run_on() const; };  // Move-only.
std::expected<GridBuilderBuffers, GridFailure> BuildGridBuilderBuffers(RunOn run_on, std::uint32_t max_cell_blocks);
std::expected<GridBuffers, GridFailure> BuildGridBuffers(const GridSettings& settings, GridBuilderBuffers& builder_buffers);
enum class BuilderBuffersSharing : std::uint8_t { kOnePerGrid, kOneForAllGrids };
```

**From the grid as built (2026-11-26)** (grid design 3.4, 3.8, 5.5, 5.8, 6, 11, 12):

- **New cells and the size-step rule: open design point.** `grid::ComputeNewCellSizeLevel` does not keep the
  size-step rule by itself (5.3 corrected): `FindNearestCommittedCell` looks at 27 positions one biggest cell width
  apart, and the cell it finds may not touch the new cell, so a touching smaller cell can be missed. No grid function
  closes the rule over a list of new cells, and `FindCellsThatMustSplit` splits committed cells on the CPU while the
  step computes new levels in its marking pass on the grid's processor. Owner question, to answer before the step's
  plan: should the grid's search look at nearer distances first so that a touching small cell is found, or does the
  step own a closure over its new list, with a GPU form? This design owns a brute-force test of the nearest committed
  cell on random grids (the grid has 8 and 10 hand cases for `ComputeNewCellSizeLevel` and
  `FindNearestCommittedCell`).
- **Border sizes.** On the sloped-surface scene (9,584 cells, 218 blocks, ranges of 8 blocks), a range of 8 blocks
  needs up to 97 blocks as near border and 174 as wide border: about 12 times the range at the near width. Choose the
  range size with the border in mind, and size piece buffers by the stacked total (range plus every layer). Layers are
  found from the last stage's width to the first; range plus border so far above `kMaxSetBlocks = 14,128,181` blocks
  fails `kTooManyCellBlocks`; an empty range is `kBadInput` in `BuildGridPiece`; a border list smaller than the stages
  need is not detected (readers then answer none). The grid's piece tests run on 5 to 16 random grids of one frame and
  16 framed grids: plan a piece test at the run's real range sizes.
- **Sums across batches.** Each block's partial goes into a whole-grid `CellBlockValues` at its whole-grid index
  (`block_in_whole_grid` gives it), and one `math::TreeSum` runs after the last batch. A tree sum per piece, added
  up, is not bit-identical to the whole grid's.
- **Pieces and versions.** Two pieces of one whole grid with equal block counts pass each other's version check:
  values are bound again at every load and never read through another piece (a load counter in a piece's version is
  an open owner question).
- **Driver-stack entries.** The grid now has one `kDriverStack` entry, `BuildBorderSearchStackEstimate` (3,600 B per
  thread for the border search's marking function, about 199 MB on a 36-multiprocessor card), to add to a GPU plan
  beside `BuildBorderSearchMemoryEstimate`. Whether the step sums the modules' `kDriverStack` entries or takes the
  largest (the driver keeps one stack, sized for the largest need) is an open owner question.
- **Memory plan.** A CPU grid's `kRunOn` and `kCpu` entries come from one CPU `RunMemory` (the builders do not read
  `where`); each `GridBuilderBuffers` keeps an 8-byte lent count outside every `RunMemory`.
- **Cell size rule.** The four merge settings this design reads from `grid::CellSizeSettings`
  (`merge_width_share`, `merge_wait_duration`, `min_merge_wait_steps`, `max_cell_splits_per_resize`) are not declared
  yet: they are added when cell resize is planned, unless the owner says otherwise. The table of widths is
  `grid::CellWidthsBySizeLevel` (this design writes `CellWidthsByLevel`); `ComputeWantedCellWidth` takes
  `(facts, reference_speed, widths, settings)` and `ComputeWantedSizeLevel` `(wanted_width, widths)`. `CellSizeFacts`
  has no `cell_width` field and defaults the depth and the distance to +infinity, "nothing near"; two of its four
  flags are read only by this design's merge test.

**Particles (PROVIDED, particles 3.1 to 3.8):** `ParticlesTry`, `ParticleFields`, `ParticleFieldsToChange`,
`Field`, `FieldList`, `ToFieldList`, `BuildFieldsToRead`, `BuildFieldsToChange`, `ParticleAmounts`,
`ComputeAmounts`, `ComputeAmountsPerBlock`, `AddParticles` with `AddReason::kRefillRegion`, `RemoveParticles`
with `RemoveReason::kIntoGridOnlyRegion`, `NewParticles`, `RefillCell`, `ComputeRefillParticle`,
`GroupParticlesByBlock`, `CheckGrouping`, `CheckParticles(const ParticlesTry&, grid::CellBlockRange)`,
`CountRule`, `NeighbourhoodInputs` (`outside_solid`, `face_side`, `is_grid_only_region`, null in the first version
(A64); the step and this module pass `transfer::SolidsAtFaces::outside_solid` and `side`), `ParticlesSettings::cpu_memory`,
`BuildParticlesMemoryEstimate`, and the two-grid count changes called only inside the resize transaction:

```cpp
std::expected<AdjustParticleCountResult, ParticlesFailure> AdjustParticleCount(
    ParticlesTry& particles_try, const grid::StaggeredGrid& grid_before_resize,
    const NeighbourhoodInputs& neighbourhood_before_resize, const grid::StaggeredGrid& grid,
    const NeighbourhoodInputs& neighbourhood, grid::CellBlockRange cell_blocks);
// SplitParticle and MergeParticles take the same two pairs. AdjustParticleCountResult as particles 3.5.
// CHANGED (A67; signature unchanged): also splits or merges every particle of a kMayBeSplitAndMerged material whose
// size_level differs from its active cell's level on `grid` (too large: split, repeated to the smallest level in a
// smallest-level cell; too small: merged in chunks of eight in cell place order then ascending id, or of two to seven
// after kMaxMergeWaitPasses = 16 passes; per-particle merge_wait_passes); the result gains mismatched_particle_count,
// waiting_particle_count, lone_particle_count, incomplete_split_count, kinetic_energy_change. Called with
// grid_before_resize == grid (the committed grid both times) for a pass that changes no cell (5.1 step 3).
std::expected<void, ParticlesFailure> WriteParticleGroups(ParticlesTry& loaded, const grid::GridPiece& piece,
                                                          const ParticlesTry& whole);
// range: whole-grid block indexes, equal to the range of the piece last written into `loaded`.
std::expected<void, ParticlesFailure> ReadParticleGroups(const ParticlesTry& loaded, grid::CellBlockRange range,
                                                         FieldList fields, ParticlesTry& whole);
```

**Transfer (PROVIDED, transfer 3):** `GridFromParticles`, `CellsFromParticles`, `SolidsAtFaces`
(`outside_solid`, `side`, `solid_velocity`), `FaceValueKind` with `kFromGridOnlyRegion`, `MaterialSetKey`,
`ToMaterialSetKey`, `BuildTransferMemoryEstimate(ingredient_count, material_count, material_set_count,
sets_without_solve)`. Material sets (A8): every transfer stage and `ComputeCellsFromParticles` runs once per
set with `material_is_counted` (one entry per material, each 0 or 1), in a one-piece and in a batched run;
each set has its own `GridFromParticles`, two `GridSnapshot` and one `CellsFromParticles`. Memory per extra set
per cell block: 48,576 + 512 x ingredients bytes (40,416 + 512 x ingredients for a set with no solve).

**Pressure (PROVIDED, pressure 3):** `GeometrySnapshot`, `CellGeometry` (`filled_volume`, `open_volume`,
`air_volume`, `content`), `FacePieceGeometry::open_area`, `FacePieceEnds`, `CellContent` (with `kGas`),
`AirPockets`, `AirPocket` (`first_cell_index`), `AirPocketArrays`, `kNoAirPocket`, `SolidCoverage`,
`SurfaceDistanceInputs::is_grid_only_region`, `FindAirPockets(const AirPocketInputs&, AirPocketArrays,
PressureBuffers&)` (whole grid in a batched run), `FindMaterialBoundaries`, `SolvePressure(const
PressureEquations&, const SolveSettings&, const std::atomic<bool>* cancel_requested, SolvedArrays,
PressureBuffers&)` writing into the step's try arrays. In a batched run `ComputeDistanceToSurface` and
`FindLiquidSurface` run per batch and every other function of its section 3 once with `blocks` = all blocks; no
pressure is given at a batch border (5.9). NEW (A15, `plan-decisions.md` "Batched solves (A15)", pressure
`solve_batches.h`, CPU only): `SolveBatches` (built into `PressureBuffers`, `ViscosityBuffers` and `GasBuffers`
when their settings have `max_loaded_blocks > 0`), `BuildSolveBatchesMemoryEstimate(RunOn, max_cell_blocks,
max_loaded_blocks)`, `ApplySolveBatches` (the cut this module's planner predicts), `struct SolveCopyReport {
batch_loads; sweep_count; bytes_to_processor, bytes_from_processor, bytes_within_processor; copy_elapsed_seconds;
}`, `const SolveCopyReport* ReadSolveCopyReport(const PressureBuffers&)` (also `ViscosityBuffers`, `GasBuffers`;
null in one piece), `BufferSizes::max_loaded_blocks`, `PressureError::kCancelled` (mapped to
`kCancelRequested`), and

```cpp
std::expected<void, PressureFailure> WriteSnapshotRange(const PressureBuffers& piece_buffers,
                                                        const grid::GridPiece& piece,
                                                        PressureBuffers& whole_buffers);
```

which copies the range blocks of every pass-1 row and the snapshot's largest values from one-piece buffers bound
to `piece.grid()` into whole buffers bound to the whole grid (GPU to GPU, or CPU to CPU). The memory estimate entries of
`BuildPressureMemoryEstimate`, `BuildViscosityMemoryEstimate` and `BuildGasMemoryEstimate` built with `max_loaded_blocks > 0`
give the streamed rows `where = kCpu` at the limits, the same rows `where = kRunOn` at `max_loaded_blocks`, and
`BuildSolveBatchesMemoryEstimate`; this module's planner reads them. Test library:
`pressure::testing::CountBlocksInLargestLayer`.

**Viscosity (PROVIDED, viscosity 3):** `SolveViscosityAndPressure(inputs, settings, cancel_requested, buffers,
velocity_after_pressure, cell_pressure)`, called unchanged with `block_range` = every block of the grid (A15: in
the batched layout of `ViscosityBufferSettings::max_loaded_blocks` it runs its passes in sweeps with its own
`SolveBatches`, signature unchanged); `cells_from_particles` is a span with one `CellsFromParticles` per liquid
material set; a scene has exactly one liquid set, so it holds one entry, after this module added the grid-only
records.

**Materials (PROVIDED, materials 3):** `MaterialTable` (`materials()`, `MaterialKind`,
`first_ingredient_of_material()`, `ingredients()`), `BuildParticlesSettings` (which materials keep their
count), `InternalForceTotals`, `ApplyInternalForce(..., std::span<InternalForceTotals> totals_of_block, MaterialBuffers& buffers)` (one form,
buffers last, A31; a batch passes the sub-span of the step's whole-grid budgets at `range_in_whole_grid()` and the
loaded instance's `MaterialBuffers`),
`ComputeInternalForceTotal`.

**Solids (PROVIDED, per `changes-to-apply.md` 08):** `ComputeSolidsAtFaces` (the resized grid at the committed
pose, for `NeighbourhoodInputs`), `SolidCoverageValues`, `ToSolidCoverage`, `BuildMotionMap`,
`ApplyPressureToSolids`. Per-solid sums are block partials then a tree sum, independent of where batches are
cut; no `FreeSolidPush` list is merged across batches. NEW (plan decision, 2026-11-14, buffers in batched runs;
solids plan Tasks 13, 28, 38): `std::expected<void, SolidsFailure> WriteBlockPartialsRange(const SolidsBuffers&
piece_buffers, const grid::GridPiece& piece, SolidsBuffers& whole_buffers);` (range blocks only, either processor
pair), called by `ReadBatch` after pass 1; `ApplyContactReactions` and `ComputeCutWork` run on the CPU in a
batched run (stage 17, `step::FindContactReactionBuffers`); `FindGridEdgesOpenToSky` is stage 3's
`run_whole_grid` part.

**Gas (PROVIDED, per `changes-to-apply.md` 09):** `GasAmounts`, `GasAmountsTry` reached through
`step::ResizeTry::gas_try()`, whose seven cell arrays (`gas_volume`, `gas_density`, `fuel_density`,
`smoke_density`, `background_pressure`, `flame`, `pressure`) are writable on a try bound to the resized grid;
the gas `BuildTry` does not count resized cells as removed cells; `BuildResizeTry`, `GasArraysToRemap` and
`ToArraysToRemap` give the writable arrays, committed with `CommitTry(amounts, try, amounts.time())`; a batched
run is not a gas exclusion (`SolveGas` runs with every block; A15: in the batched layout of
`GasBufferSizes::max_loaded_blocks` its per-cell and per-face steps run as sweeps, gas amounts and geometry stay
whole on the GPU). Gas beside liquid (A10): the gas module provides `holds_tracked_gas` bound to the try grid.

**Step (PROVIDED, per `changes-to-apply.md` 10 "From cell resize and batches"):** `Simulation`, `ResizeTry`,
`BuildResizeTry`, `ApplyResizedGrid`, `BuildResizedGrid`, `ResizedCells`, `ApplyResizedCpuGridCopy` (NEW, reconcile),
`CommitResize`, `RejectResize`, `GridFields`, `ResizeResult`, `StepCallbacks`, `Callback`, `StageTable`,
`StageOutcome`, `StepTry`, `Stage` (stage 11 `kSolveViscosityAndPressure`), `kStageCount`, `RunPlan`,
`RunMode`, `SimulationLimits`, `ReadyScene`, `StepSettings`, `Module`, `TryStepWithStages`. NEW (A66):
`StartRegionShape`, `StartCells`, `BuildStartCells` (`step/start_cells.h`, CPU only; the run's first grid, built from
the scene's shapes with the shared size rule and the one-size-step closure) and `SceneCounts::start_cells_per_level`,
`start_particle_count`, `start_cell_blocks` (the batch plan's start blocks). NEW (A67): `ResizeResult::
size_change_kinetic_energy`, `ResizeResult::mismatched_particle_count` (decided here; the step copies them into
`StepReport::size_change_kinetic_energy` and `StepReport::mismatched_particle_count`). What the step adds for this
module:

1. CHANGED (A41, A63, A64): `GridFields` gains, each with a committed and a try copy, bound to the try grid with
   `ApplyGridVersion` and carried by `CellId` in `ChooseActiveCells`: `merge_ready_duration` (`CellValues<double>`,
   s), `held_velocity_gradient_size` (`CellValues<double>`, 1/s; NEW, A63), `merge_ready_step_count`
   (`CellValues<std::uint8_t>`, saturating at 255; NEW, A63); 17 bytes per cell place, two copies (memory entries
   `"step: held velocity gradient size"`, `"step: merge wait steps"`, `kRunOn`); saved in the state. Not in the first
   version (A64): `is_grid_only_region`, `layers_from_grid_only_region` (`CellValues<std::uint8_t>`),
   `grid_only_mass`, `grid_only_rest_volume`, `grid_only_current_volume` (`CellValues<double>`),
   `grid_only_ingredient_volume` (`Buffer<double>`).
   NEW (A62): `CommittedStepLimits::largest_liquid_speed` (m/s; largest particle speed over liquid materials at the
   last commit, an exact max), read by `ResizeCells` for `grid::ComputeReferenceSpeed`.
2. Not in the first version (A64): callbacks `apply_grid_only_region` (end of stage `ParticlesToGrid`) and `move_grid_only_liquid` (end of
   stage `MoveParticles`), both `StageOutcome (*)(void* context, Simulation&, StepTry&, grid::CellBlockRange)`,
   with read access on `StepTry` to the try grid, each set's `GridFromParticles` and `CellsFromParticles`, the
   geometry snapshot, `velocity_after_pressure`, both `GridFields`, `step_duration` and the cancel flag.
3. `ResizeTry` also gives: the last try's geometry snapshot, `SolidCoverage` and end-pose `SolidsAtFaces`
   (still valid for the committed grid), the committed step's duration, the material table, the
   material sets, writable air pockets (`AirPocketArrays` of the try, their `first_cell_index` belongs to a
   grid version), `SolidsAtFaces` storage for the resized grid, and the free `grid::GridBuffers` that
   `BuildGrid` writes the resized grid into.
4. Not in the first version (A64): `ComputeStepDuration` includes grid-only face speeds (`ReadLargestGridOnlySpeed`) in `kParticleTravel`.
5. Batches: (a) ranges are built from the try grid after stage 2 (`TryStepWithStages(simulation, request,
   stages)`) and a `run_stages` callback with the signature of `ApplyStagesInBatches`; null with
   `RunMode::kBatched` is `kBatchedRunNotBuilt`. (b) With `kBatched`, `BuildSimulation` puts the particle
   arrays (`CpuMemory::kLockedForCopies`) and the per-try arrays in CPU memory, the whole-grid arrays of 5.7
   on the GPU, and one loaded `StepTry` of `BatchPlan::max_loaded_blocks` blocks on the GPU; its stage
   functions run on that loaded try unchanged. (c) A `StepArray` table generated from the memory estimate entries:
   `struct StepArray { std::string_view row_name; CountedPer counted_per; std::uint32_t bytes_each,
   material_index, read_by_stages, written_by_stages; }`, one entry per array instance (a per-set array has
   one entry per material set), so `WriteBatch` and `ReadBatch` copy exactly what a pass reads and writes
   without knowing the modules. (d) `RunPlan::max_blocks_per_batch` comes from `BuildBatchPlan`. (e) Every
   total the step, materials and solids sum over the grid is one partial per cell block and `math::TreeSum`
   over block indexes ("every design", sums over ranges).
6. NEW (A15): `RunMode::kBatchedWithSolvesInPieces`, `RunPlan::max_solve_loaded_blocks` and
   `RunPlan::allowed_cpu_bytes` (3.7); with that mode `BuildSimulation` builds the solvers' buffers in the
   batched layout. The loaded `StepTry` owns one-piece pressure buffers at `max_loaded_blocks` for pass 1's
   geometry (read into the whole buffers by `WriteSnapshotRange`).
7. NEW (plan decisions, `plan-decisions.md` "Cell resize and batches"): `StageTable` gains
   `std::array<StageFunction, kStageCount> run_whole_grid{}`, non-null only for `kFindSolidSurfaces`
   (`FindGridEdgesOpenToSky`, whose flood crosses every block; plan decision, 2026-11-14, buffers in batched runs),
   `kFindLiquidSurface` (`FindMaterialBoundaries`, `FindAirPockets`, `CheckViscosityCanRun`), `kMoveParticles` (the
   grid-only move; not in the first version, A64) and `kCheckStep` (`CheckSolidExchange`, the gas changes, the next limits); a one-piece run calls
   it right after `run`. The loaded try, built by `step::BuildSimulation` (not by `BatchedRun`):

   ```cpp
   StepTry* FindLoadedTry(Simulation& simulation);                    // Null in a one-piece run.
   grid::GridBuffers* FindLoadedGridBuffers(Simulation& simulation);  // Null in a one-piece run.
   // Changed (plan decision, 2026-11-14, buffers in batched runs; replaces the two-argument form): block_in_whole_grid
   // is the piece map in CPU memory (piece.block_in_whole_grid() for a CPU piece, the batched run's copy for a GPU
   // piece), one entry per piece block, else kBadInput; stage 18 passes it as FoamAndSprayInputs::block_in_whole_grid.
   std::expected<void, StepFailure> ApplyGridPiece(StepTry& loaded_try, const grid::GridPiece& piece,
                                                   std::span<const grid::CellBlockIndex> block_in_whole_grid);
   struct BlockCopy { grid::CellBlockIndex from_block; grid::CellBlockIndex to_block; std::uint32_t block_count; };
   std::expected<void, StepFailure> WriteStepArrayBlocks(const StepTry& from, std::size_t array_index,
                                                         std::span<const BlockCopy> copies, StepTry& to);
   ```
8. NEW (A17): in a GPU run the step builds the try's grid on the GPU (`ChooseActiveCells`, and, changed by the
   reconcile pass, `BuildResizedGrid` from the cell list this module uploads, item 10, 5.1 step 4; `ApplyResizedGrid`
   stays for a caller that built the grid itself). In a batched GPU run stage 2 builds the
   needed cell list on the CPU and uploads it into `Buffer<grid::CellId>` (row `"step: active cell list"`, 8 B per
   cell place, `where = kRunOn`, counted) for `grid::BuildGrid` on the GPU (plan decision, 2026-11-14, buffers in
   batched runs).
9. NEW (plan decision, 2026-11-14, buffers in batched runs; step plan Tasks 33a, 33b, Appendix A):

   ```cpp
   enum class ArrayPlace : std::uint8_t { kLoadedOnly, kWholeOnProcessor, kWholeInCpuMemory };
   // step::StepArray gains: ArrayPlace whole_place;
   // The CPU copy of the grid in a batched GPU run, the grid itself in a CPU run, null in a one-piece GPU run.
   const grid::StaggeredGrid* FindCpuGridCopy(const Simulation&);
   // The stage-17 SolidsBuffers in CPU memory (run_on = kCpu, whole particle limit, max_cell_blocks); null in one piece.
   solids::SolidsBuffers* FindContactReactionBuffers(Simulation&);
   // With a batched plan the rows follow the step plan's Appendix A (loaded module instances, mirrors, CPU grid copy).
   std::vector<MemoryEstimateEntry> BuildStepMemoryEstimate(const ReadyScene&, const SceneCounts&,
                                                            const RunPlan* batched_plan = nullptr);
   ```
   The placement rule of 5.7 decides `whole_place`; `ReadStepArrays` reports it. The loaded `StepTry` owns one
   instance of `transfer::TransferBuffers` (built by `BuildTransferBuffers(run_on, max_particle_count,
   material_count)`; every transfer stage has one form, taking it last, A31), `materials::MaterialBuffers`, `contact::ContactBuffers` and
   `solids::SolidsBuffers` (at `max_loaded_particles` per material and `max_loaded_blocks`). The step keeps CPU
   mirrors of committed values (`"step: committed gas volume on the CPU"`, `"... gas velocity ..."`, `"...
   grid-only marks ..."` (later group only, A64), `"step: keep active on the CPU"`) ~~and refreshes its CPU grid copy after a committed
   resize with `grid::ReadGrid` (456 B per block, counted)~~ **replaced** (reconcile, coordinator item 6): after a
   committed resize the CPU copy is the CPU resized grid this module built into `free_cpu_grid_copy_buffers()`,
   handed over with `ApplyResizedCpuGridCopy` (item 10); no `ReadGrid`, the step's Appendix B row B5 is withdrawn.
10. NEW (reconcile, `plan-decisions.md` "Reconcile pass" item 12 and coordinator items (1), (6); step plan Task 32,
    `resize_access.h`):

    ```cpp
    // The cell list of a resized grid, on the simulation's run_on, any order (as grid::GridInputs).
    struct ResizedCells {
      std::span<const grid::CellId> cells;                    // A CPU run: the list in CPU memory.
      const Buffer<grid::CellId>* cell_buffer = nullptr;      // A GPU run: the list uploaded by cell resize.
      std::uint32_t cell_buffer_count = 0;                    // Entries of *cell_buffer used.
      const grid::StaggeredGrid* cpu_resized_grid = nullptr;  // A GPU run: cell resize's CPU build of the same list.
    };
    // Builds the resized grid (committed frame, version committed + 1) into free_grid_storage() on the simulation's
    // run_on with grid::BuildGrid, binds the try as ApplyResizedGrid does, and returns the grid for CommitResize. In
    // a GPU run version and block count must equal *cpu_resized_grid's (else kStageFailed, module kGrid). A list on
    // the wrong processor, or a GPU run without cpu_resized_grid: kBadInput. On failure free_grid_storage() is usable.
    std::expected<grid::StaggeredGrid, StepFailure> BuildResizedGrid(ResizeTry& resize_try,
                                                                     const ResizedCells& resized_cells);
    // ResizeTry member: in a batched GPU run, the free half of "step: grid copy on the CPU", into which cell resize
    // builds its CPU resized grid; null in every other run.
    grid::GridBuffers* free_cpu_grid_copy_buffers();
    // A batched GPU run: keeps the CPU resized grid built into free_cpu_grid_copy_buffers() as the CPU copy that
    // CommitResize swaps in (RejectResize releases it). kBadInput in other runs, for a grid in other buffers, or a
    // version or block count different from the resized grid's.
    std::expected<void, StepFailure> ApplyResizedCpuGridCopy(ResizeTry& resize_try,
                                                             grid::StaggeredGrid&& cpu_resized_grid);
    ```
    `CommitResize` in a batched GPU run requires `ApplyResizedCpuGridCopy` first (else `kBadInput`, nothing
    committed) and swaps the CPU copy pair with no copy. ~~`choose_cell_size` receives
    `committed_grid.tables(RunOn::kCpu)`~~ **replaced** (A41): `choose_cell_size` is removed; stage 2 calls
    `grid::ComputeNewCellSizeLevel`.

**I/O and CLI (PROVIDED, per `changes-to-apply.md` 12):** the scene file carries `CellResizeSettings` and
`BatchSettings` with their defaults; the `run` and `resume` commands build `CellResize` and, for a batched
plan, `BatchedRun` (CHANGED, A58: into the run's `run_memory` and `cpu_run_memory`; CHANGED A78, A79: `BuildCellResize`
takes `RunScene::refinement_volume().layout`; I/O builds the refinement volume at load from the scene's
`refinement_zones`, `refinement_camera` and `refinement_volume_files` and keeps every key on the CPU), and set
`ResizeCells` and `ApplyStagesInBatches` (A41, A64: `ChooseCellSize`, `ApplyGridOnlyRegionInStep` and
`MoveGridOnlyLiquidInStep` are gone from the first version) in `step::StepCallbacks`, each as a
`Callback<Function>{function, context}` pair with the context 3.4 and 3.7 name (`CellResize*`, `BatchedRun*`);
`saved_state` stores the three wait arrays (A63; the seven region arrays only in the later group) with the committed
pressure; `run_mode` stays in the fingerprint
(5.10 item 5); the run report and events carry the `BatchPlan` (solve place, batch count, bytes copied per
try; A15: the memory budgets and planned copies) before the run and `ResizeCellsReport` and `BatchReport` totals
(A15: with the measured solve copies) after it; `CellResizeFailure` and `BatchesFailure` become text with
`ToMessage` and codes with `ToName`. `Run` copies `BuildBatchPlan`'s choice into `RunPlan` (mode,
`max_blocks_per_batch`, `max_solve_loaded_blocks`); `RunOverrides::assume_memory_bytes` forces a batched plan in
tests. Printing (planned cost, 2026-11-14; I/O plan Tasks 23a, 29a): `io::BuildRunBudget` takes the
`BatchPlan` (or the `BatchesFailure`) and shows `batch_count`, `solve_batch_count`, the memory budgets
(`bytes_on_processor`, `bytes_on_cpu`) and the copies (`bytes_copied_per_try`, `pass_loads_per_try`,
`solve_loads_per_iteration`, `solve_bytes_per_iteration`) in `CopyBudget`; `io::ApplyTryToRunCost` reads the try's
last `BatchReport` (pass and solve copies) for the measured cost; `ResizeCellsReport::bytes_copied` and
`copy_count` enter the measured copies per step after the commit.

**Surface, foam and spray (PROVIDED, per `changes-to-apply.md` 13):** (not in the first version, A64: counts a
grid-only cell as liquid, `is_grid_only_region`, also through `SimulationSnapshot::is_grid_only_region`; the old
paper's "no false inner surface") takes volume targets from `ComputeLiquidTotals`; its `make_foam_and_spray` works once per batch
in pass 2 on a loaded piece with no whole-grid sum inside; its `FoamAndSprayInputs::block_in_whole_grid` is the
batched run's CPU copy of the piece map handed through `step::ApplyGridPiece`, never the empty span of a GPU piece
(plan decision, 2026-11-14, buffers in batched runs). `VolumeTransport.lean` is not ported.

**Blender add-on:** nothing is consumed. It shows what I/O's events and run report carry (run mode, batch
plan) and writes the `CellResizeSettings` fields of the scene file; CHANGED (A78, A79): refinement zones (Empties
with Cube or Sphere display, closed meshes, each with a "Cell size factor") and the optional camera term, exported as
the scene's `refinement_zones` and `refinement_camera` (no regions, no map).

**Per material, this module carries:** liquids and mixtures: everything (first version: particles of every level
with their fractions; later group: ingredient volumes in grid-only cells, fractions on refilled particles); two liquids that do not mix are two materials, each record holds one
of them and a cell touching another material's particles stays in the particle region (5.6). Snow, springy
solids, cut solids: their cells keep one size and never become grid-only; their particles are never split,
merged, removed or refilled; a split forced under them is refused. Moving solids: small cells within
`solid_reach_in_cell_widths`; nothing of theirs is remapped. Gas (A10, also beside liquid): its seven cell
arrays are remapped; a gas cell asks for the smallest size through `wants_smallest_cells`; gas markers may be
split and merged like liquid particles but are never taken into a record; a grid-only cell never touches a
cell that holds gas; gas has no grid-only region (it is already on the grid).

**For the sand module:** nothing specific. Sand particles have `CountRule::kKeepsItsCount`, so the snow rules
above (one cell size, never grid-only, never split or merged) apply to them unchanged; their batched cost is
measured in the sand module.

## 5. Method

### 5.1 `ResizeCells`: one transaction after a committed step (D12)

1. `step::BuildResizeTry`. Read the committed grid, `GridFields`, the last try's geometry, solid coverage and
   end-pose `SolidsAtFaces`, the committed step duration and the material table. A17, `grid_run_on == kGpu`: with
   `GridCopies::kHeldByCellResize` (one piece) one `grid::ReadGrid` copy of the committed grid into the CPU
   `GridBuffers` `CellResize` holds (456 B per block); the closure search, the plan and `BuildResizeMap` read it.
   With `GridCopies::kHeldByStep` (batched GPU run) it reads `step::FindCpuGridCopy(simulation)` instead (version and
   block count checked against the committed grid, `kWrongGridVersion` otherwise), and `bytes_copied` and
   `copy_count` leave that copy out. ~~The row `"cell resize: committed grid copy"` is still built (one row list for
   every GPU layout) and stays idle (plan decision, 2026-11-14, buffers in batched runs).~~ **Replaced** (reconcile,
   decision 33): with `kHeldByStep` neither CPU grid of `CellResize` is built or counted.
2. CHANGED (A41, A62, A63): `ComputeDepthBelowSurface` (5.2); `grid::ComputeReferenceSpeed` from the committed
   `CommittedStepLimits::largest_liquid_speed`, `|gravity|` and the biggest width; `ComputeCellSizeChanges` part 1
   then part 2 over every block (5.3, "The pass"); on a GPU grid one counted read back of `size_change`; the plan
   (5.3 rules 3 to 5).
3. ~~If the plan is empty, no particle wants a split or merge and no wait array changes: `RejectResize`, `resized =
   false`, no error.~~ **Replaced** (A67): if the plan is empty and no wait array changes, `particles::AdjustParticleCount`
   runs at once with the committed grid as both grids (and the end-pose inputs as both inputs) on the particle try;
   it counts the mismatched particles (`size_level` other than the cell's, materials `kMayBeSplitAndMerged`). When
   `mismatched_particle_count == 0` it changed nothing: `RejectResize`, `resized = false`, no error (nothing to do).
   Otherwise the pass goes on as a same-cell resize (the list unchanged, `version + 1`, as for a wait-array change),
   and step 7 keeps this result instead of running again (same cells, same particles: the same result). The count
   is AdjustParticleCount's own, so no separate count pass and no extra copy exist. (Not in the first version, A64:
   region wishes per cell (5.6); also no region change, no particle in a grid-only cell and no cell eighth needing a
   refill.)
4. Apply `SplitCell` and `MergeCells` to the cell list; `grid::CheckSizeStepRule`; `grid::ComputeGridCounts`
   against `max_cell_blocks` (`kTooManyCellBlocks`); `grid::BuildGrid(inputs, storage)` with `version + 1`
   into the free storage the `ResizeTry` gives (a CPU grid). Allocates nothing. A17,
   grid on the GPU: the resized grid is built on the CPU for the map (into `CellResize`'s second CPU `GridBuffers`
   with `kHeldByCellResize`; into `resize_try.free_cpu_grid_copy_buffers()` with `kHeldByStep`, reconcile), and the
   same list goes up into `CellResize`'s `Buffer<grid::CellId>` with one slice `WriteBuffer` (8 B per cell,
   counted); then `step::BuildResizedGrid(resize_try, ResizedCells{{}, &upload, cell_count, &cpu_resized_grid})`
   builds the try's grid on the GPU into `free_grid_storage()`, binds the try and checks version and block count
   against the CPU grid (changed, reconcile item 12: replaces "the step's `ApplyResizedGrid` builds the try's grid on
   the GPU"); A17 makes their tables equal, and `BuildResizeMap` checks version and block count of both. With
   `kHeldByStep`, after the map is built and before `CommitResize`, `step::ApplyResizedCpuGridCopy` hands the CPU
   resized grid to the step, which keeps it as its CPU copy (no `ReadGrid`).
   Step 3 "nothing to do" also requires that no wait array (`merge_ready_duration`, `merge_ready_step_count`,
   `held_velocity_gradient_size`, A63) changes: a change of those alone commits a same-cell resize with `version + 1`
   (plan decision, decision 23).
5. `step::ApplyResizedGrid(resize_try, resized_grid)` (on a GPU grid `step::BuildResizedGrid` of step 4, which binds
   the same way) binds the try's `GridFields` and `SolidsAtFaces` storage to the resized grid and builds the gas try (`gas::BuildResizeTry`, reached through `gas_try()` afterwards);
   the try's air pockets come from `air_pockets_to_fill()`. `BuildResizeMap`; remap every `GridFields` array and the gas arrays (5.4); translate each
   air pocket's `first_cell_index` to the smallest cell place of the new grid that carries its label.
6. `solids::ComputeSolidsAtFaces` on the new grid at the committed pose; `GroupParticlesByBlock` for the new
   grid. (Not in the first version, A64: `ApplyRegionTransfer` (5.6); `GroupParticlesByBlock`.)
7. `particles::AdjustParticleCount` with the committed grid and its `NeighbourhoodInputs` (from the end-pose
   `SolidsAtFaces`; `is_grid_only_region` null, A64) and the new grid with its inputs (5.4, 5.3 "Particles in deep
   cells"); CHANGED (A67): at **every** pass, over all blocks, so the particles of changed cells and every particle
   that moved across a size change since the last pass are brought to their cell's level (5.4); not run again when
   step 3 already ran it;
   `GroupParticlesByBlock` if anything was split or merged. The result goes to `ResizeCellsReport::particle_count`
   and its energy change and mismatched count to `step::ResizeResult` (3.4).
8. Check: totals of mass, rest volume, current volume and each ingredient volume over liquid particles (A64; the
   later group adds the grid-only cells) before and after agree within `kAmountTolerance` times the total
   (`kAmountsDoNotBalance`); the gas amount (`gas_volume` times `gas_density`, summed) likewise; every
   remapped value is finite.
9. `step::CommitResize(simulation, resize_try, resized_grid)`. `state_version` grows by one; time and
   `step_count` do not change.

Any failure from step 2 on calls `RejectResize` (and `grid::TakeBuffers` on a resized grid already built):
the committed step, the previous grid, the regions and every particle are untouched to the byte, time stays
advanced, and the step is not replayed (report, architecture "The physical step keeps a fixed topology").
The wait arrays are part of the try, so a failed resize also earns no wait time. The run continues on
the previous grid; if a cell is then wider than its wanted width the next resize tries again.

### 5.2 Depth below the surface (new)

`pressure`'s `distance_to_surface` is the distance to the nearest particle ball, so it never reads deeper
than one particle radius. Depth is therefore measured on the cells: `depth = 0` in a liquid cell with
`air_volume > 0`, a liquid-to-air face piece, or a touching cell that holds tracked gas (A10); every other
liquid cell takes `min over touching liquid cells of (their depth + distance between the two centres)`,
repeated until no value changes, capped at `max_depth`. CHANGED (A64): `max_depth = surface_reach_in_cell_widths x
w_max`, because `w_band = depth / surface_reach_in_cell_widths` reaches the biggest width `w_max` there and the clamp
holds it beyond (was: `grid_only_depth_in_cell_widths` biggest widths, at least `surface_reach_in_cell_widths`
smallest widths times 4). A minimum is exact in any order, so the result does not depend on the schedule. The
geometry is that of the committed step's start positions; a particle moved at most half a cell width since, which
the 8 `w_min` no-merge band covers.

### 5.3 The cell size rule and its closure

CHANGED (A41, A61, A62, A63; `plan-decisions.md` "Owner answers A40–A64 applied" item 2; `literature-cell-size.md`
5). Ported: the decision rule of `DecideAdaptiveWidth3D` and of the old paper's `<sec-blocks>`, and the closure
search. New: one shared rule, written once in grid's `cell_size_rule.h` and run on either processor, with fixed
inputs; the facts are computed here; the camera wish is not ported. The 2026-11-14 rule (depth and solid terms only)
is **replaced** by:

1. **Wanted width** (grid `ComputeWantedCellWidth`, per active cell, from committed fields; `w_min` = the smallest
   width, `w_max` = the biggest, both from `CellWidthsByLevel`):

   ```
   w_band   = w_min * max(1, depth_below_surface / (surface_reach_in_cell_widths * w_min))     (8 by default)
   w_solid  = w_min * max(1, distance_to_solid / (solid_reach_in_cell_widths * w_min))         (2 by default)
   w_shear  = shear_width_factor * U_ref / g        for g > 0, else +infinity                  (c_u = 0.25, A62)
   f        = factor_table.factor[ComputeCellSizeFactorCode(refinement, smallest, cell) + 24]   (A78, A79)
   wanted   = clamp(min(w_band, w_solid, w_shear) * f, w_min, w_max)
   wanted   = w_min  where touches_air_pocket, wants_smallest_cells (gas, A10), or depth < 8 w_min (pinned band)
   U_ref    = max(largest_liquid_speed, sqrt(|gravity| * w_max))                                (ComputeReferenceSpeed)
   g        = held_velocity_gradient_size = max(held over the current wait, ‖∇u‖_F now)        (rule 5)
   ```

   The surface always has the smallest cells (A61): `depth = 0` gives `w_band = w_min` and the clamp gives `wanted =
   w_min`, whatever the other terms. `‖∇u‖_F` is `ComputeVelocityGradientSize` (3.2): the full 3 x 3 gradient, not
   speed alone (a uniform translation is exact on any cell size) and not Ando and Batty 2020's diagonal-only sum
   (which misses shear and rotation). `w_shear` says the velocity change across one cell stays below `c_u` times the
   scene's speed (Gerris' `|ω| Δ / max|u|` in width form); `U_ref` makes the term scale-free, with a floor from
   gravity and the biggest cell so a scene at rest still has a speed. `c_u = 0.25` is a starting value, to calibrate
   (A62). Worked numbers: rigid rotation at `Ω = 10 rad/s` gives `‖∇u‖_F = sqrt(2) Ω = 14.142 1/s`; with `U_ref = 1
   m/s`, `w_shear = 0.25 / 14.142 = 0.017678 m`. CHANGED (A78, A79): the artist's wish is the cell's smallest
   cell size factor over its box, one byte of the refinement volume's min-pyramid; the product scales what the flow
   needs (a factor above 1 never coarsens the pinned surface band in the first version). The wanted level is the biggest
   level whose width `<= wanted` (`ComputeWantedSizeLevel`, by comparison with `CellWidthsByLevel`, never `log2`).
   Curvature (`w_curv`) is the later group "Coarser cells on calm surfaces (later)".
2. A cell whose particles keep their count (snow, springy solid, cut solid) keeps its size: `kKeepsItsSize` always.
3. `kSplit` when `cell_width > wanted` (at once, ported). The first `max_cell_splits_per_resize` in fixed
   order; the rest wait. At most one level per cell per resize.
4. `FindCellsThatMustSplit` (ported search): a split cell of level `l` forces every touching active cell of
   level `l - 1` to split, repeated until nothing is added; 26 touching directions per cell. If a forced
   cell is a rule-2 cell, the split that caused it is taken out of the plan and counted
   (`refused_split_count`); the rule is then re-run without it.
5. **Merge test and hysteresis** (A63; `literature-cell-size.md` 5(d)). A cell asks `kMayMerge` when `2 w <=
   merge_width_share * wanted` (0.8) and it touches no air, air pocket, gas or solid and lies at least
   `surface_reach_in_cell_widths * w_min` (8 `w_min`) below the surface. Eight siblings pass the test when all 8 are
   active and `kMayMerge`, and every cell touching the big cell also satisfies `2 w <= merge_width_share * wanted`
   at the big cell's width. While the test holds, per cell `merge_ready_duration += elapsed committed time` and
   `merge_ready_step_count += committed steps` (saturating at 255) since the last committed resize, and the held
   gradient is `max(held, current)`; any break sets both counters to 0 and the held gradient to the current value.
   The 8 become `kMergeReady` when their smallest duration reaches `merge_wait_duration` (0.25 s) **and** their
   smallest step count reaches `min_merge_wait_steps` (8), and they merge unless the merge breaks the size-step rule
   against the cells of step 4 (a refused merge is never an error). Split at once, merge late, one level per resize:
   the gap between the split and merge thresholds (factor `2 / 0.8 = 2.5`), the time memory (wait plus held
   maximum, which is "the maximum over the current wait") and the grading of one level are the three things every
   solver read uses against flicker. The held value is redundant while `ResizeCells` runs after every committed step;
   it keeps a longer resize interval correct. Remap at a resize: a split copies the held value to its 8 cells and
   sets both counters to 0; a merge sets all three to 0.
6. Vetoes of `main` and where they went: unresolved surface and cavities: the surface and pocket cells are
   smallest by rule 1; thin sheets and gaps: both faces of a sheet are within the surface reach; plates and
   contact: the solid reach; singular spread: `AdjustParticleCount` refuses the particle, not the cell.
7. With the surface, pockets and solids in smallest cells, the capability exclusions
   `kViscousLiquidAtSizeChange` and `kInternalForceAcrossSizeChange` hold by construction: a size change
   only occurs in bulk liquid.

**The pass** (`ComputeCellSizeChanges`, 3.2), on the grid's processor over all blocks inside 5.1 step 2. Part 1, per
active cell: the facts (depth from 5.2; distance to a solid = the smallest corner value of `pressure::SolidCoverage`;
held gradient; artist level; `holds_particles_that_keep_their_count` from its particles in grouped order;
`touches_air_pocket`, `wants_smallest_cells`, `touches_air_gas_or_solid` from the cell and its 26 touching cells),
the wanted width, `ComputeCellSizeChange`. Part 2, per active cell: the sibling and neighbour test, the counters,
`kMergeReady`. One byte per cell place, `size_change`; on a GPU grid that array is read back in one counted copy (1 B
per cell place) for the CPU list edit (rules 3 to 5's list work stays on the CPU); nothing else is read back for
sizing. Every term is a local stencil plus exact minimums and maximums, widths are compared with
`CellWidthsByLevel`, no `log2`, `pow` or FMA: CPU and GPU, and any cut into block ranges, give the same bytes.

**Particles in deep cells (A64).** A cell of level `l` holds particles of `size_level == l` (particles 5.6):
`AdjustParticleCount` merges eight particles of level `l + 1` into one of level `l` when a resize merges their cells
and splits one into eight when a resize splits its cell, so a big cell holds about the seeded count (8 per cell)
with `8^k` times the volume; a cell `k` levels too fine coarsens over `k` resizes at least one wait apart and its
particles follow level by level. ~~Drifted particles keep their level (C2) and are counted.~~ **Replaced** (A67):
particles that moved across a size change are split or merged at the next pass (5.4, "Particles whose size no longer
matches their cell"). One wording (A65): every liquid cell holds about eight particles of its own size: a particle's
`size_level` is its cell's level, its rest volume is one eighth of its cell (`(w/2)³` for a cell of width `w`), so a
cell k levels coarser than the smallest holds 8 particles each of `8^k` times the smallest particle's volume. NEW (A66): a run starts on
`step::BuildStartCells`' grid, which already follows the depth and solid terms (with a Euclidean depth minus half a
cell diagonal, so never coarser than this rule wants; 5.2's through-liquid depth is never smaller), seeded 8 per cell
at each cell's level; the first resizes may coarsen further, never refine for depth, and the `k`-resize cascade
above happens only where the surface moves away or in tests that start all-smallest on purpose. Cost: with `w = depth / 8`
the number of cells below the band in a tank of surface area `A` is `∫ A dz / w(z)^3 = 512 A ∫_{8 w_min}^{D} z^-3 dz
< 4 A / w_min^2`, against `8 A / w_min^2` in the band of 8 smallest cells: particles everywhere cost at most about
1.5 times the band-only count for any depth when merges are allowed (discrete levels and the 0.8 share raise this
toward 2). Scenes whose deep cells cannot coarsen (few levels, strong shear, refinement factors below 1, a solid or gas pocket
deep down) pay `8 x (depth / w) x A / w^2`.

**New cells** (stage `ChooseActiveCells`, grid `ComputeNewCellSizeLevel`, called by the step on the grid's
processor): the level of the nearest committed active cell plus one, capped at `smallest_size_level`
(`grid::FindCellAt` probes at `position + {-1, 0, 1}^3 * biggest width`, nearest centre, ties in fixed cell order;
`smallest_size_level` when none is found). This is rule 1 for a new cell: it borders air, so its depth is 0 and
its wanted width `w_min` (A61), and the next `ResizeCells` finishes the job. Corrected (grid as built, 2026-11-26;
grid Rulings 46, 48): the cap does not keep the size-step rule; the cell found may not touch the new cell, and a
nearer, smaller cell that touches it can be missed, so the step closes the rule over its new list (open point, section
4 "From the grid as built"). **Replaced** (A41): `ChooseCellSize` (a CPU callback with its one-piece GPU copies:
new cells read back, levels sent, a `ReadGrid` when no current copy was held) and the open question of a shared rule.

### 5.4 Remap between two grid versions, and conflict C2

Every array is carried by `CellId` or `FacePieceId` where the cell or piece exists in both versions. Rules
for a split (one big cell `B`, small cells `k = 0..7`, `cell_eighth` order) and a merge:

| Value | Split | Merge |
|---|---|---|
| Gas `gas_volume`; grid-only `mass`, `rest_volume`, `current_volume` (not in the first version, A64) | `ComputeSplitAmounts` | sum of 8 in index order |
| `ingredient_volume` (not in the first version, A64) | `B`'s fractions times the small cell's `current_volume` (`ApplyResizeToIngredientVolumes`) | sum |
| `pressure` (carried), gas `background_pressure`, gas `pressure` | copy | mean of the small cells with `has_pressure` (gas: weighted by `gas_volume`) |
| Gas `gas_density`, `fuel_density`, `smoke_density`, `flame` | copy | mean weighted by `gas_volume` |
| `has_pressure` | copy | 1 if any is 1 |
| `pocket_of_cell` | copy | smallest label that is not `kNoAirPocket` (does not occur: rule 5) |
| `is_grid_only_region`, `layers_from_grid_only_region` (not in the first version, A64) | copy | copy (all 8 agree by rule 5) |
| `merge_ready_duration`, `merge_ready_step_count` (A63) | 0 | 0 |
| `held_velocity_gradient_size` (A63) | copy | 0 |
| `velocity_after_pressure` | below | below |

**Amounts on a split (ported, fixed).** Per axis the slope is `minmod(B - low, high - B)` over the same-size,
same-region neighbours, zero where one is missing. Offsets `+-1/4` per axis; one limiter in `[0, 1]` for all
amounts of the cell keeps every small-cell value inside the range of `B` and its neighbours. Small cells 0
to 6 get `B/8 + limiter/8 * offset`; cell 7 gets `B` minus their sum, so the 8 sum to `B` to one rounding
step; if cell 7 leaves the allowed range the limiter is 0 and every small cell gets `B/8`. FIXED: the
certificates and the `OutOfRange` on inexact division are gone; one child order.

**Face velocity (ported).** With `flow = velocity * face piece area` (the whole piece, no solid: open areas
belong to one step and are found again by the next one). Merge: a big face whose other cell is now also big
gets the sum of its 4 pieces in ascending `FacePieceId`, divided by its area; where the other cell stays
small the 4 pieces live on. The 12 inner faces disappear. Split: each outer piece that was one big piece
becomes 4 pieces with the same velocity; pieces that already existed are kept. The 12 inner faces: with
`target = (net outflow of B) / 8` and each small cell's outflow through its outer faces known, walk the
small cells from 7 down to 1; cell `k` sends what it lacks through the inner face to cell `k` with its
lowest set bit cleared (`ComputeInnerFaceFlows`, the tree of `adaptive_liquid_remap_3d.cc:482-501`); the 5
inner faces outside the tree get 0. Each small cell then has the flow imbalance per volume of `B`.

**Conflict C2, settled as on `main` (`changes-to-apply.md` 03).** The particles design asks: parent complete at
its own size, each child complete at the next size. Neither grid alone can answer both, but both grids are
alive inside this transaction: the committed grid for the old size, the resized grid for the new one
(`liquid_adaptive_family_candidate_3d.cc` checks "the parent on the old coarse bank and each of eight
children on the new fine bank"). So `AdjustParticleCount`, `SplitParticle` and `MergeParticles` take both: a
split checks the parent on `grid_before_resize` and the 8 children on `grid`; a merge checks each particle on
`grid_before_resize` and the merged position on `grid`. ~~Consequence, stated plainly: a particle is split or
merged only in the resize that changes the cells around it, and only when it is at least 1.5 cell widths
inside the changed region on both grids. A particle that drifts across an existing size change is never split
or merged: it keeps its `size_level` for good (it is a valid particle; it is counted in
`skipped_incomplete_count`).~~ **Replaced** (A67): the both-grid check stays only for the cells the pass changes,
and it no longer refuses a split: a split whose parent or children lack a complete neighbourhood on either grid is
made and counted in `incomplete_split_count` (the split keeps mass, momentum and angular momentum whatever the
neighbourhood; only its energy is exact for a complete one, and the energy difference is reported). A merge in a
changed cell that the both-grid check refuses leaves its particles too small for their cell; the next pass merges
them by the rule below (gap test only). The cell size rule moves the small cells with the surface, so the liquid near
the surface is split where it stands instead of crossing a size change. Option A of `shared-interface.md` (a flag
from the last `GridToParticles`) would never be set for a drifting particle either, because a fit within 1.5
widths of a size change is not complete.

**Particles whose size no longer matches their cell (NEW, A67; particles 5.4 to 5.6 own the maths).** One wording
(A65): every liquid cell holds about eight particles of its own size: a particle's `size_level` is its cell's level,
its rest volume is one eighth of its cell (`(w/2)³` for a cell of width `w`), so a cell k levels coarser than the smallest holds 8 particles
each of `8^k` times the smallest particle's volume. A particle moves at most half a smallest cell per step, so it can
cross a size change between two passes; `ResizeCells` runs the extended `AdjustParticleCount` at every pass (5.1
steps 3 and 7) and it brings each such particle back to its cell's level on `grid`:

- *Too large* (`size_level` below the cell's): split once by the 1-to-8 pattern (offset `(√3/4) H`), no
  neighbourhood check refuses it; in a smallest-level cell the split repeats until the children match (at most
  `size_level_count - 1` times in one pass), so no too-large particle reaches the next transfer in the surface band;
  elsewhere one level per pass. A child inside a solid is the next `MoveParticles`' `started_inside_count`.
- *Too small:* merged with particles of the same cell, material, `plate_side`, `flags` and `size_level`, chunks taken
  in cell place order then ascending id (so CPU, GPU and every batch cut take the same chunks; a chunk lies in one
  cell, so in one block); a chunk of eight merges at once, a chunk of two to seven once its first particle has waited
  `kMaxMergeWaitPasses = 16` passes (starting value, about 16 committed steps); a lone particle stays, valid and
  counted (`lone_particle_count`); a chunk the gap test refuses waits for the next pass.
- *Conservation:* mass and rest volume exact up to fixed-order sums, momentum exact; `kinetic_energy_change =
  energy(after) - energy(before)`, `energy(p) = (m_p/2)(|v_p|² + (w_p²/4) |C_p|²_F)`, block partials then
  `math::TreeSum`: 0 up to rounding for splits, `<= 0` for merges with a positive gap.

So the leak of `tradeoff-particles.md` 1.2 (small particles piling up in coarse cells, large particles reaching the
surface) is closed: after every pass each particle matches its cell except counted waiting particles (at most 16
passes) and lone ones (at most one per cell, material and level).

### 5.5 The grid-only region inside a step (not in the first version, A64)

This section and 5.6 are the plan's later group "Deep liquid without particles (later; needs review)": an
independent review of this design comes first, and two tests gate it (`GridOnlyRegion, StandingWave_KeepsAmplitude`,
`GridOnlyRegion, ParticleBandEdge_StaysStable`, section 7). Ferstl 2016 reports a particle/grid velocity blend as
"inherently unstable" and Ando and Batty 2020 report energy loss on coarse semi-Lagrangian cells, the two risks
those tests measure (`literature-cell-size.md` 3, 5(e)).

Invariant: **every amount of liquid is held once**, by a particle or by the record of a grid-only cell. A
cell's total is the sum of both. Momentum is held by particles and, in the grid-only region, by the face
velocities; the records hold no momentum (on `main` they did, beside the face velocity: two ledgers). A
record holds one liquid material, and is counted in the material set of that material (A8); a set with no
liquid material (gas markers, snow, springy solids) never receives anything from a record.

*In stage `ParticlesToGrid`* (`ApplyGridOnlyRegionToGrid`, once, for the one liquid material set), for each
face piece of the range whose grid-only cell's record is of a counted material:

1. Both cells grid-only, or one grid-only and the other a solid or a grid edge: `velocity =
   ComputeCarriedVelocity`, `mass = sum over its two cells of (record mass / cell volume) * piece area * half
   the cell's width`, `weight_sum = 8 * (that volume) / (cell volume)`, `value_kind = kFromGridOnlyRegion`.
2. One cell grid-only, the other in the particle region: the same velocity; the particle part of `mass` and
   `momentum` is added to the mass above.
3. Both cells in the particle region with `layers_from_grid_only_region` 1..`overlap_layer_count`: with
   `layer = mean of the two cells' layers`, `particle_share = clamp((layer - 0.5) / overlap_layer_count, 0,
   1)`, `velocity = particle_share * particle velocity + (1 - particle_share) * carried velocity` (ported
   blend, in layers instead of metres). A face with `kTooFewParticles` or `kNoValue` takes the carried
   velocity alone and becomes `kFromGridOnlyRegion`.
4. `CellsFromParticles` of a grid-only cell: record added to what particles inside it gave (a particle that
   drifted in is still counted once), and the record's ingredient volumes to `ingredient_volume`, before
   pressure and viscosity read them.

`ComputeCarriedVelocity` (FIXED, the old paper's rule; `main` keeps the committed value in place):
`start = face centre - step_duration * (committed velocity at the face centre)`, the velocity vector being
read with the grid's nearby-face weights on each axis; the result is the committed value of the face's own
axis read at `start` with the same weights (`sum weight * value - row . mean_offset`, all faces kept). First
order and it loses a little energy; it is exactly the committed value in liquid at rest.

*After `MoveParticles`* (`MoveGridOnlyLiquid`, ported from `adaptive_liquid_core_transport_3d.cc:130-310`):
for each face piece between two grid-only cells, `volume = velocity_after_pressure * open_area *
step_duration`; the giving cell is the one the flow leaves. Each piece takes the share `volume /
(giving cell's current_volume at step start)` of every amount of the giving cell. One loop iteration per
receiving cell: own amounts, minus its outgoing shares, plus incoming, in ascending face place index. Shares
of one cell adding to more than 1: `kGridOnlyCellDrained` (the step shortens, as on `main`). No amount moves
through a face piece between the two regions (5.6). FIXED: a cell may end above or below its open volume
(`main` failed); above `kMaxFilledShare` it is `kGridOnlyCellTooFull`. Mixtures move like one liquid because
every amount takes the same share. Two touching grid-only cells always hold the same material (5.6), so no
record ever mixes two materials. `moved_volume` is one partial per block, then `math::TreeSum`.

Pressure already treats a grid-only cell as full of liquid (distance `-2` widths, no volume correction).

### 5.6 Region transfer (inside `ResizeCells`), and its ledger (not in the first version, A64)

A cell *wants* the grid-only region when it is a liquid cell with `filled_volume == open_volume ==` the
whole cell volume, `depth >= grid_only_depth_in_cell_widths * width`, every particle in it of one material of
`MaterialKind::kLiquid` (and, when it already has a record, of the record's material), no particle with a
plate side, no touching cell that is air, pocket, gas (`holds_tracked_gas` or `CellContent::kGas`) or solid,
and no touching cell holding particles or a record of another material. It wants the particle region when any
condition fails or `depth < particle_region_depth_in_cell_widths * width`. Between the two depths it stays
what it is (ported: 6 widths, "particles deeper than the region plus one cell are deleted"). Then, in cell
place order of the resized grid, particles in id order:

1. **Cell to grid-only.** Its particles are removed (`kIntoGridOnlyRegion`); their mass, rest volume,
   current volume and ingredient volumes are added to the record.
2. **A particle of the record's material found in a grid-only cell** (it moved in during the step): the
   same. A particle of any other material (another liquid, snow, a springy solid, a gas marker) found there
   turns the cell back to the particle region in this resize (rule 3), so one cell never holds two materials'
   records and no non-liquid particle is taken into one.
3. **Cell to particles.** `particles_per_refilled_cell` particles of the record's material from
   `particles::ComputeRefillParticle` with `current_volume` and `density = mass / current_volume` of the
   record, `volume_ratio = current_volume / rest_volume`, velocity = area-weighted mean of the cell's
   face-piece velocities per axis, `fitted_velocity_change` = difference of opposite face means over the
   width (diagonal only), fractions = record's ingredient volumes over its current volume, `size_level` = the
   cell's. Added with `kRefillRegion`, so `kHasOwnMotion` is off. The record becomes zero.
4. **Refill at the region border.** For a grid-only cell `G` with `excess = current_volume - open volume`
   and each particle-region cell `P` sharing a face piece with `G` (ascending face place): each cell eighth
   of `P` that touches the piece and holds no particle gets one particle at its centre while
   `excess >= width_P^3 / 8`; the particle takes exactly that volume and `G`'s density, volume ratio and
   fractions out of `G`'s record.
5. `layers_from_grid_only_region` by the same repeated minimum as 5.2, in whole layers of face-touching
   cells, up to `overlap_layer_count + 1`.

Why this replaces the flow exchange of `main`: liquid that flows from `P` into `G` arrives as particles and
is taken in by rule 2; liquid that flows from `G` into `P` first shows as `G` above its open volume (its
inflow from below had no outflow through the region face) and as empty cell eighths in `P` (the particles
moved away), and rule 4 turns one into the other. Nothing is counted twice and no particle's mass is edited.
`main`'s rule moved amounts by the face flow and corrected one particle's mass, which is right only while no
particle leaves its cell.

The ledger holds both directions as `ParticleAmounts` and the totals before and after, each a block partial
then `math::TreeSum` (5.10). Mass, rest volume, current volume and ingredient volumes balance within
`kAmountTolerance`. Momentum does not have to: removed particles' momentum beyond what the face velocities
hold is `momentum_not_carried`, reported.

### 5.7 Batched run: what lives where

A run is batched when `step::BuildRunPlan` says so. Then:

- **CPU memory:** every particle array (committed and try, `CpuMemory::kLockedForCopies`), the per-particle
  working buffers, and the per-try face and cell arrays that only particle stages read (the two grid snapshots of
  each material set, `SolidsAtFaces`).
- **GPU, whole grid, for the whole run:** the grid tables of the committed and the try grid, `GridFields`
  (both copies), the geometry snapshot, each solved set's `velocity_before_pressure`, the six cell arrays of
  stage 10, `PressureBuffers` (a second instance for gas), and when the scene has them `ViscosityBuffers`
  with `ingredient_volume`, the gas arrays and buffers, `SolidCoverageValues` with the motion map. These are
  the *whole-grid arrays*.
- **GPU, one loaded batch:** a `StepTry` over a `grid::GridPiece` built into one `GridBuffers` of
  `max_loaded_blocks` blocks, with every material set's transfer arrays, one-piece pressure buffers at
  `max_loaded_blocks` for pass 1's geometry, and a loaded `ParticlesTry` of `max_loaded_particles` per material.
  The `BorderSearchBuffers` and two border buffers of `BatchedRun` are on the run's processor too (A17).

**With solves in pieces (`kBatchedWithSolvesInPieces`, A15).** Rule: state and geometry stay whole on the GPU;
solver buffers go to the CPU.

- *Whole grid, on the GPU, for the run:* both grids' tables; `GridFields` (both copies); gas amounts (both
  copies); every array written in pass 1 by `ComputeDistanceToSurface` and `FindLiquidSurface` and the gas
  geometry snapshot; the step's solve inputs (each solved set's `velocity_before_pressure`, the six cell arrays
  of stage 10, `SolidCoverageValues`, the motion map, the blocking-material faces, the free-solid entries);
  `blob_of_cell` and `pocket_of_cell`; the pressure ladder's rungs 1 and deeper with their direct-solve factors;
  every per-block partial, per-group total and progress record.
- *Whole grid, in CPU memory (`CpuMemory::kLockedForCopies`), loaded batch by batch by the solvers:* every other
  row of the liquid and gas `PressureBuffers`, `ViscosityBuffers` and `GasBuffers` counted per cell place, face
  place, corner place, sample or coefficient.
- *GPU, one loaded solve piece* of `max_solve_loaded_blocks` blocks (pressure's `SolveBatches`), beside the
  loaded batch.
- About 15 kB per full block stays on the GPU for a liquid scene (27 kB with viscosity, about 22 kB with gas),
  against 55 kB (190 kB) when the solves are whole; the run is then limited by CPU memory (about 400 kB per
  block of particles plus the streamed working buffers), as A15 asks.

**Placement rule (plan decision, 2026-11-14, buffers in batched runs; owner principle A17; the full inventory is
the step plan's Appendix A).** In both batched modes, on either processor: a whole try array lives on the run's
processor when a part at `kWholeGridOnProcessor` reads or writes it; else in CPU memory when a
`kWholeStateOnCpu` stage reads or writes it, the commit keeps it, or a later pass reads it (a carrier between
passes); an array read and written only inside one pass has no whole instance, only the loaded one.
`step::ReadStepArrays` reports each array's `ArrayPlace`; `kStagePlaces` is the input of the rule (test in 7).
Besides:

- *Loaded instances:* the loaded `StepTry` owns, on the run's processor, one `transfer::TransferBuffers`,
  `materials::MaterialBuffers`, `contact::ContactBuffers` and `solids::SolidsBuffers` at `max_loaded_particles`
  per material and `max_loaded_blocks`, beside the piece `PressureBuffers`; the whole instances keep their
  per-particle rows at zero particles. So `bytes_per_loaded_block` and `bytes_per_loaded_particle` include them.
- *Stage 17 in CPU memory:* `solids::ApplyContactReactions` and `ComputeCutWork` run on the CPU with a separate
  `SolidsBuffers` (`run_on = kCpu`, whole particle limit, `max_cell_blocks`; rows `"solids: "`, `where = kCpu`);
  the `CollideSolids` block partials of each range reach it through `solids::WriteBlockPartialsRange`, called by
  `ReadBatch` after pass 1 and counted in `BatchReport`.
- *CPU copy of the grid (batched GPU run):* stage 2 builds the needed cell list on the CPU, uploads it (`"step:
  active cell list"`, `kRunOn`) for `grid::BuildGrid` on the GPU, and builds a CPU copy of the same grid with the
  CPU builder (`"step: grid copy on the CPU"`, `kCpu`; the pair shares one CPU `grid::GridBuilderBuffers`, reconcile).
  Grouping, marker upkeep and `ResizeCells` (`choose_cell_size` removed, A41) read it through `step::FindCpuGridCopy`. After a
  committed resize it is the CPU resized grid `ResizeCells` built into `free_cpu_grid_copy_buffers()` and handed over
  with `step::ApplyResizedCpuGridCopy`: no `ReadGrid` refresh (reconcile, decision 33).
- *Mirrors of committed values* (both batched modes, CPU to CPU in a batched CPU run): committed gas volume, gas
  velocity, grid-only marks (later group only, A64), keep active; all `where = kCpu`, counted.
- *Stage 3's whole part:* `solids::FindGridEdgesOpenToSky` runs at `kWholeGridOnProcessor` right after pass 1,
  before stage 9's whole part.
- *Foam and spray in pass 2:* `FoamAndSprayInputs::block_in_whole_grid` is the piece map copy `WriteBatch` already
  reads back, handed through the three-argument `step::ApplyGridPiece`.

Per full smallest-size block (64 cells, 512 liquid particles), from the rows of the finished designs:
particles and their working buffers about 345 kB, transfer and solids arrays about 105 kB for one material set (each
extra set adds about 49 kB), whole-grid arrays about 55 kB (190 kB with viscosity). With one set the
whole-grid arrays are a ninth of the total, which is why they can stay.

### 5.8 Where each stage runs (`kStagePlaces`)

| Stages | Place | Why |
|---|---|---|
| 1 `AddNewParticles`, 2 `ChooseActiveCells`, 17 `MoveSolids`, 20 `CommitStep`, 21 `ResizeCells` | whole state on the CPU | Counters and the cell list are CPU work where the particles are (A17: the grid itself is built on the run's processor from the uploaded list, with a CPU copy built from the same list; 5.7). `MoveSolids` adds per-block partials of the contact reactions; `ApplyContactReactions` and `ComputeCutWork` run on the stage-17 `SolidsBuffers` in CPU memory (5.7). `ResizeCells` reads `GridFields` back once into CPU mirrors (about 11 kB per block both ways) and the grid through `step::FindCpuGridCopy` (one piece: its own `ReadGrid`) |
| 3 `FindSolidSurfaces`, 4 `ParticlesToGrid` with `ComputeCellsFromParticles` (and `ApplyGridOnlyRegionInStep`, not in the first version, A64), 5 `SaveGridBeforeForces`, 6 `ApplyForces`, 7 `DriveWaves`, 8 `CollideSolids`, 9 `ComputeDistanceToSurface` and `FindLiquidSurface` (stage 9's `run`) | pass 1, in batches; transfer stages once per material set | Local work: a value depends on particles and faces a bounded distance away |
| stage 3's `run_whole_grid` (`FindGridEdgesOpenToSky`, plan decision 2026-11-14) right after pass 1, then stage 9's `run_whole_grid` (`FindMaterialBoundaries`, `FindAirPockets`, `CheckViscosityCanRun`) right after pass 1; 10 `BuildPressureEquations` with `solids::BuildMotionMap`, whole (plan decision: every call of stage 10 takes all blocks or reads arrays pass 1 wrote back); 11 `SolveViscosityAndPressure`, 12 `SolvePressure`, 13 `SolveGas`; stage 16's `run_whole_grid` (`MoveGridOnlyLiquidInStep`, not in the first version, A64) right after pass 2; stage 19's `run_whole_grid` after stage 17 | whole grid on the run's processor, `blocks` = all (`kWholeGridOnProcessor`; with `kInPieces` the solvers sweep their solve batches) | Global: labels, blobs, sums over the grid, solves. Same code as a one-piece run; with `kInPieces` within the solver's tolerance of it (5.10) |
| 14 `SaveGridAfterPressure` (per set), 15 `GridToParticles` (per set), 16 `MoveParticles` with the per-block partials of the contact reactions, 18 `MakeFoamAndSpray`, the per-block parts of 19 `CheckStep` | pass 2, in batches | Local |
| the sums and decisions of 19 | whole state on the CPU | From block partial sums |

`kStagePlaces` (plan decision): `kNone` `kNotRun`; 1, 2, 17, 20, 21 `kWholeStateOnCpu`; 3 to 9
`kInBatchesBeforeSolve`; 10 to 13 `kWholeGridOnProcessor`; 14, 15, 16, 18, 19 `kInBatchesAfterSolve`. Every
non-null `run_whole_grid` part runs at `kWholeGridOnProcessor`: 3's right after pass 1, then 9's, 19's after stage 17
(16's right after pass 2 only in the later group, A64).

A pass loads a batch once and runs all its stages on it, every material set's transfer stages included. At
each load, `WriteBatch` builds the piece into the run's storage (`grid::BuildGridPiece`, allocating nothing)
and binds every loaded array to the piece's grid again with `ApplyGridVersion`; a piece has the whole grid's
version, so the binding is checked by `block_count()` as well. A stage of the pass reads values the previous
stage wrote in border blocks, so the load carries a **stacked border**: the border of the last stage, then
around it the border of the stage before, and so on (`FindStackedBorder`: one buffer-form
`grid::FindBorderBlocks` per layer on the run's processor, two buffers of `max_loaded_blocks` used in turn, only
counts come back; A17), and stage `i` runs on the range plus the borders of the stages after it. Pass 1 stacks
touching, touching, four cell widths, touching (about four layers of blocks at the biggest size in the range);
pass 2 stacks four cell widths and touching (about two), because `velocity_after_pressure` is a whole-grid array
and needs no border. The piece is built with the buffer form of `grid::BuildGridPiece`; its map is read back once
per load (4 B per loaded block) to compute the `BlockCopy` runs on the CPU. Values computed in border blocks are
thrown away: the batch that owns those blocks computes the same bytes (5.10). Pass 1 copies its results for
the solve from the loaded arrays into the whole-grid arrays on the GPU (`WriteStepArrayBlocks` slices; a range is
one run of blocks in both, at `range()` in the piece and `range_in_whole_grid()` in the whole grid) and its
geometry with `pressure::WriteSnapshotRange` (A15), and `ReadBatch` returns to the CPU only what pass 2 needs,
with `particles::ReadParticleGroups` for the range only.

`ApplyStagesInBatches` (`<proc-apply-stages-in-batches>`): 1 cut: the largest `n <= plan.max_blocks_per_batch` for
which every range of `grid::BuildRanges(try grid, n)` with its pass-1 stacked border has at most
`max_loaded_blocks` blocks (bisection with `FindStackedBorder`); none: `kStageFailed`, `module = kBatches`,
`module_error = kBatchDoesNotFit`; 2 pass 1 per range: `WriteBatch`, the `run` of stages 3 to 9, `ReadBatch`; 3
stage 3's `run_whole_grid`, stage 9's `run_whole_grid`, then 10 to 13 on the whole try; 4 pass 2 (14, 15, 16, 18, 19 `run`), then (later group
only, A64) 16's `run_whole_grid`, stage 17, 19's `run_whole_grid`. A stage outcome other than `kNone` ends the call. After stage 13
the solvers' `SolveCopyReport`s are summed into `BatchReport`. If the guard test on stage order (batched pass order
against the one-piece order of stages 16 to 18) fails, or a stage rejects on incomplete inputs in the outer border
blocks of a piece, the implementer stops and reports: the fix belongs in the module that owns the stage.

### 5.9 The solves, and what a batched run costs

**The solves, two ways (changed, A15).** Replaced: "the pressure solve, the joint viscosity solve and the gas
solve are not batched" and the CPU fallback `SolvePlace::kWholeGridOnCpu` (a CPU solve inside a GPU run, against
A17). Now:

| Solve place | What happens | When |
|---|---|---|
| `kWholeGridOnProcessor` | Each solve runs once on the whole grid on the run's processor, as in a one-piece run: no copy per iteration, no pressure given at a batch border. 55 kB per block on the GPU for the run (190 kB with viscosity) | The whole rows with every solver row on the processor fit within `max_whole_grid_share * allowed_bytes` |
| `kInPieces` | The same iteration in the out-of-core form (pressure's `SolveBatches`, owned by pressure, viscosity and gas): solver buffers in CPU memory, one loaded solve piece on the GPU, every iteration a few sweeps over the solve batches; state and geometry stay whole on the GPU (5.7). No signature of a solver changes and the step's stage calls 9 to 13 do not change | Otherwise, when the solve piece holds three of the largest layers and the CPU rows fit |

What this module owns of the solve in pieces: the choice and the sizes (`BuildBatchPlan`), the run mode copied into
the step's plan, the report of planned and measured copies, and the cancel and failure seams (a solver's
`kCancelled` is `kCancelRequested`; a solve failure rejects the try like any stage failure). What it does not own:
the sweeps, the stacked widths of each solver (pressure and gas `{kTouchingCells}`, viscosity two), the block
rules for grid-wide and per-group sums, the label join, the ladder layout and the direct-solve room: these are
pressure's, viscosity's and gas's (`plan-decisions.md` "Batched solves (A15)", "Method"). The consequences that
matter here: a batched solve is the one-piece arithmetic with sweep boundaries inserted, so its result does not
depend on the cut; liquid solves cost 3 sweeps per iteration (4 with a sealed gas blob), about 50 to 65 kB per
block per iteration with the border, about 0.25 s per iteration for a 100,000-block scene at 25 GB/s (serial
copies); the iteration count is the one-piece count. A domain split with a coarse correction (additive Schwarz)
was weighed and rejected there: more iterations, growing with the batch count, and a different preconditioner from
the one-piece run.

**Free solids.** The motion map is built once on the whole grid from the whole-grid coverage and geometry,
and `ApplyPressureToSolids` sums over it once. Every per-solid sum (pressure pushes, contact reactions) is one
partial per cell block and a tree sum over blocks (solids, `changes-to-apply.md` 08), so it does not depend on
where batches are cut, and no push list is merged across batches. Contact reactions are summed per block in
pass 2 by the batch that owns the block; `MoveSolids` tree-sums the block partials on the CPU.

**`BuildBatchPlan` (changed, A15; the start grid is not built, so it allocates nothing).** `allowed =
RunPlan::allowed_bytes`.

1. Whole-grid rows: the grid, pressure, viscosity, gas and solids rows not counted per particle and the step's
   `GridFields` rows, at the limits. At most `max_whole_grid_share * allowed`: `kWholeGridOnProcessor`. Above:
   `kInPieces`; the whole rows are recomputed from the solvers' row builders in the batched layout (their `kRunOn`
   rows at the limits only); `room = allowed - whole`; `bytes_per_solve_loaded_block` = the solvers' loaded rows at
   one block (liquid and gas pressure, viscosity, gas: they coexist); `max_solve_loaded_blocks =
   min(floor(solve_piece_share * room / bytes_per_solve_loaded_block), limits.max_cell_blocks)`; fewer than
   `3 * layer`: `kBatchDoesNotFit` with the bytes; the loaded batch then gets `(1 - solve_piece_share) * room`
   instead of `room` below.
2. `bytes_per_loaded_block` = the loaded rows (transfer for every set, the step's per-try cell and face rows,
   `grid::BuildGridMemoryEstimate(run_on, 1, limits.max_cell_blocks, grid::BuilderBuffersSharing::kOneForAllGrids)` without its `builder buffers:`
   entries, since the loaded try's buffers borrow the step's `grid::GridBuilderBuffers`; changed, reconcile item 11) at one cell
   block and no particle; `bytes_per_loaded_particle` the same at
   one particle and no block; `particles_per_block = ceil(512 * start_limit_headroom)` per material (8 per cell).
   CHANGED wording (A65): 512 holds for a full block **at every level**, because every liquid cell holds about eight
   particles of its own size (a particle's `size_level` is its cell's level, rest volume `(w/2)³`), never `8 · 8^k`
   per coarse cell; mismatched particles waiting to merge (A67) are a bounded excess the headroom covers.
3. `max_loaded_blocks = min((allowed - whole) / (bytes_per_loaded_block + particles_per_block *
   bytes_per_loaded_particle), limits.max_cell_blocks)`, with `(1 - solve_piece_share) * room` in place of
   `allowed - whole` for `kInPieces`; `max_loaded_particles = max_loaded_blocks * particles_per_block`.
4. Blocks are ordered `z` first, so a range is a slab: `layer = (domain cells along x / 4) * (along y / 4)` at the
   smallest width; `T = floor(max_loaded_blocks / layer) - 8`; `max_blocks_per_batch = T * layer`;
   `kBatchDoesNotFit` when `T < 1` or `max_blocks_per_batch < min_blocks_per_batch`. `range_share_of_loaded =
   T / (T + 8)`; `batch_count = ceil(start blocks / max_blocks_per_batch)`; `bytes_copied_per_try = (2 /
   range_share_of_loaded + 2) * start blocks * (per block + particles_per_block * per particle)`. CHANGED (A66):
   `start blocks = counts.start_cell_blocks` (the blocks of `step::BuildStartCells`' grid, not of an all-smallest
   cover). NEW (A68): `slab_limit` is filled here, on success and on this item's `kBatchDoesNotFit` (3.7): with `L =
   blocks_per_layer`, `b = per block + particles_per_block * per particle` (`bytes_per_full_loaded_block`), `R` = the
   room of item 3, `T_min = max(1, ceil(min_blocks_per_batch / L))`, `n = T_min + 8`: `needed_bytes = n L b`,
   `plan_area = X Y` (domain extents), `max_plan_area = R x 16 w_min² / (n b)` (a block covers `(4 w_min)²` of plan
   area, so a slab of `n` layers of area `a` needs `n a b / (16 w_min²)` bytes). Worked (`tradeoff-particles.md` 1.6):
   `R` about 9.6 GB, `b` about 540 kB, `n = 9`: about 3.2 m² at 1 cm, 0.8 m² at 0.5 cm; the pool 10 x 5 m at 1 cm
   (`L` = 250 x 125 = 31,250) needs `9 x 31,250 x 540 kB` = 152 GB and cannot run batched in this version.
5. CPU: every `where = kCpu` row (the particles, the per-try arrays only particle stages read, and in `kInPieces`
   the solvers' working buffers) at most `RunPlan::allowed_cpu_bytes`, else `kStateDoesNotFitCpu`.
6. Report: `max_solve_loaded_blocks`, `solve_batch_count` (start grid, bisection as `pressure::ApplySolveBatches`
   would cut it), `bytes_on_processor` and `bytes_on_cpu` (sums of the rows by `where`), `pass_loads_per_try`,
   `solve_loads_per_iteration`, `solve_bytes_per_iteration` (3.7).
7. (Plan decision, 2026-11-14, buffers in batched runs.) Every row sum above takes the rows of
   `step::BuildStepMemoryEstimate(scene, counts, &probe_plan)`: per loaded block with a probe plan of
   `max_loaded_blocks = 1`, `max_loaded_particles = 0`; per loaded particle with `0` and `1`; the whole and CPU rows
   (the mirrors, the CPU grid copy in a GPU run, the stage-17 `SolidsBuffers` in CPU memory, the `MarkerChanges`
   rows) at the limits by their `where`. So `bytes_on_processor` / `bytes_on_cpu` equal the allocation of
   `BuildSimulation` for the copied plan, by `where`, and the budget I/O shows is what is allocated.

The exact cut is found per try inside `ApplyStagesInBatches` (5.8, step 1) by bisection with `FindStackedBorder`
on the try grid after stage 2; results do not depend on `n` (5.10), which matters because `allowed` depends on
the free memory of the day (`ReadMemoryAvailable(run_on)`, its `free_bytes`).

**Copies per try.** Per full block and one material set: pass 1 up, the committed particle arrays, 79 kB;
pass 1 down, the snapshot before forces and what pass 2 reads of transfer's arrays, 25 kB; pass 2 up, 79 + 25
kB; pass 2 down, the changed particle arrays, 66 kB. About 275 kB per block of a range, and the uploads are
multiplied by `1 / range_share_of_loaded`. Blocks are ordered by `z` first, so a range is a horizontal slab and
its border is the layers above and below: a slab of `T` layers loads `T + 8` layers in pass 1. A range thinner
than one layer of the scene has more border than content. With `kInPieces`, add per try the solves' copies:
`solve_bytes_per_iteration` times the iteration count, plus the equation build (about 5 sweeps) and the
correction and check (2 sweeps); `BatchReport` measures them.

**Which scenes this is for, plainly.** Nothing here was measured; the numbers are the rows of the designs.

- The GPU holds about 22,000 full blocks in one piece (1.4 million cells, 11 million particles, 11 GB
  allowed). With whole-grid solves a batched run reaches at most about nine times that before the whole-grid
  arrays fill the GPU; with solves in pieces about 15 kB per block stays on the GPU (27 kB with viscosity), so
  the limit is CPU memory: about five times one piece before 64 GB of CPU memory is full (about 400 kB per block
  plus the streamed working buffers).
- A try copies about 0.3 GB per thousand full blocks. At a few GB per second that is seconds per try for a
  100,000-block scene, on top of the CPU-side grouping of all particles. Expect a batched step to take
  several times as long as the same step would on a GPU that held it.
- **Breaking wave:** the intended use. In the first version deep water keeps particles in coarse cells (A64; at most
  about 1.5 to 2 times the band-only count, 5.3), so batches help when the surface or the deep particles are too
  large; the later grid-only region would remove the deep particles.
- **Fire, also over water (A10):** workable; the gas solve stays whole when it fits (about 67 kB per block) and
  runs in pieces otherwise (about 22 kB per block stays), the markers pass in batches as their own material set,
  and the liquid beside it is batched as above.
- **Syrup:** whole-grid solves need 190 kB per block on the GPU, so batches alone gain less than a factor of
  three; in pieces a syrup step takes hundreds to thousands of sweeps: possible, not practical. The reference
  syrup scene fits in one piece.
- A wide, flat scene whose single layer of blocks does not fit with its border cannot run
  (`kBatchDoesNotFit`). NEW (A68): this version keeps slab batches and the surface pinned to the smallest cells; the
  failure carries `SlabLimit`, so the budget line names the plan-area limit (I/O writes the line, 3.7). Batches cut by
  columns or tiles, which remove the limit, are the later group of 5.11.

### 5.10 Sums that cross batches keep their order

Requirement (changed, A15): **a batched run gives the same bytes for every cut and on every repeat, in each
mode** (`kBatched`, `kBatchedWithSolvesInPieces`); **against the one-piece run of the same processor it agrees
within the solver's tolerance**, not byte for byte (one step: pressure's velocity bound `2 *
max_volume_error_rate * cell width + 1e-9 * max |velocity|` and the pressure bound derived from it; whole runs:
the step design 6.4 tolerances). Replaced: "a batched run gives the bytes of a one-piece run of the same processor
mix, for any cut". The local outputs of pass 1 still equal the whole-grid computation's bytes (the check that
stacked borders are wide enough). A batched solve normally does give the one-piece bytes (the same terms in the
same order, sweep boundaries inserted); the tolerance is the rule because the batched layout's direct-solve room
is smaller.

1. *Sums per destination* (a face from particles, a cell from particles, a fill fit, a particle from faces).
   The destination is stored in one block; every source lies in the loaded blocks (that is what the stacked
   border is for); sources are visited in ascending block index, then in grouped order. Because a
   `GridPiece` keeps the grid's fixed block order, the sources come in the same order as in the whole grid,
   so the sum has the same terms in the same order. Each material set is summed on its own, as in one piece.
2. *Sums over the grid inside a solve* (dot products, norms, blob totals, sums per solid over the motion
   map): in both modes (and in one piece) one partial per cell block, then `math::TreeSum` over every block in
   block order; per-group totals by block partials in ascending block index, then a tree sum per group; labels
   joined across solve batches equal the one-piece labels (pressure's rules, A15). So they do not depend on the
   solve cut.
3. *Sums over the grid outside a solve* (`StepReport` amounts, `InternalForceTotals` through
   `totals_of_block`, per-solid contact reactions, `LiquidTotals`, `moved_volume`, the ledger): one partial
   per cell block, summed inside the block in grouped order by the batch that owns the block, stored per
   block; then `ComputeSumOverBlocks` (or the owner's tree sum over blocks, such as
   `ComputeInternalForceTotal`). The one-piece run uses the same two levels ("every design", sums over
   ranges).
4. *Largest values and first failures:* a maximum and "lowest stable id" are exact in any order.
5. Stages 1, 2, 17, 20 and 21 run with the CPU implementation in a batched run, and the solves in pieces may
   take a different direct-solve room, so a batched run agrees with a one-piece GPU run within the tolerances of
   the modules, not to the byte. It agrees to the byte with itself for any cut and on every repeat. This is why
   `run_mode` stays in the run fingerprint.

### 5.11 Later groups and their order (NEW, A68; none is in the first version)

The first version keeps slab batches and the surface pinned to the smallest cells (A61). After it, in this order:

**(1) Can the fine surface band shrink from 8 cells to 4? (later, test only).** No new code: the same build with
`surface_reach_in_cell_widths` 8, 6 and 4. Runs the analytic scene tests that depend on the surface band (standing
wave amplitude and period, dam-break front position, hydrostatic pressure, the sloshing tank) and one visual scene
(the bathtub splash, rendered); records each error, the planned memory (bathtub 2 x 1 x 0.5 m at 1 cm: about 5.0,
4.3, 3.6 GB, `tradeoff-particles.md` 1.7, the middle value interpolated) and the step time. Why it can matter: the
band holds 86 % to 91 % of the cells (`tradeoff-particles.md` 1.4), and its depth `2 x reach x w_min` must cover the
orbital motion of the waves the scene shows (decaying as `e^{-k d}`; at reach 4 the band is `8 w_min` deep, so waves
longer than about `16 w_min` reach below it). Decision rule: the default becomes the smallest reach whose every
analytic error stays within its test's tolerance and within 1.2 x the reach-8 error, and the owner accepts the visual
comparison; otherwise 8 stays.

**(2) Coarser cells on calm surfaces (later)** (decision 42) **together with batches cut by columns or tiles
(later).** Design sketch, marked later, no task in the first version. *Why:* a slab batch must hold `T + 8 >= 9`
layers of smallest-level blocks, each layer `(X / 4 w_min) x (Y / 4 w_min)` blocks at `b` (about 540 kB) per loaded
block, inside the room `R` the plan leaves (about 9.6 GB of a 13.05 GB budget), so the plan area is capped at
`a_max = R x 16 w_min² / (9 b)`: about 3.2 m² at 1 cm and 0.8 m² at 0.5 cm (`SlabLimit`, 5.9 item 4). *Sketch:* each
batch is an x-y tile of blocks spanning every height; the batch plan chooses a tile side `s` (in blocks) so that
`(s + 2 kBorderLayers)² x (blocks per column) x b <= R`, with `kBorderLayers = 4` (a name for the later group: the
stacked border's depth in blocks on each side, the 4 + 4 of a slab's `T + 8`); borders become four lateral sides of
`kBorderLayers` block columns (none above or below); solve pieces become tiles with the same borders; the fixed-order
rule for sums that cross batches stays (block partials, tree sum in the whole grid's block order, 5.10), and the batch
order keeps ascending block index inside each tile; the stage places (5.8) do not change. Since blocks are ordered `z`
first, a tile is not a block-index range: the loaded set is a list of block runs, so `FindStackedBorder` and
`grid::BuildGridPiece` need a form taking a block list (designed with the group; `step::WriteStepArrayBlocks` already
copies by runs). *Cost:* more border blocks per batch (perimeter instead of two
faces), so more copies for tall narrow scenes; the plan keeps slabs when they fit and the tile cut copies more.
Coarse surface cells shrink the band itself; together they are what lets the pool (10 x 5 x 1.5 m) and wider scenes
run.

**(3) Deep liquid without particles (later; needs review)**, last (5.5, 5.6, decision 40): worth it only once the
band has shrunk, so that the deep cells hold a large share of the particles (`tradeoff-particles.md` 5 item 7).

## 6. CPU and GPU

**Shared formula headers:** `remap_math.h`, `cell_size_change_math.h`, `velocity_gradient_math.h` (A62),
`cell_resize_error.h`, `batches_error.h`, and grid's `cell_size_rule.h` (A41). Not in the first version (A64):
`carried_velocity_math.h`, `grid_only_move_math.h` (one receiving cell's new amounts from its face pieces),
`region_wish_math.h` (one cell's region wish; one face's `particle_share`).

**CPU.** The plan (lists of `CellId`, sorting, `SplitCell`, `MergeCells`, the closure search) and
`BuildResizeMap` are CPU code for both processors. Changed (A17; replaces "as grid Q41 accepts: the changed cell
ids come down and the new tables go up"): on a GPU grid they read one `grid::ReadGrid` copy of the committed grid
per call (456 B per block down; in a batched GPU run the step's CPU copy instead, no copy), the resized grid is built
on the CPU for the map (in a batched GPU run into the step's free CPU copy buffers, then handed over with
`step::ApplyResizedCpuGridCopy`), and the new cell list goes up (8 B per cell) for the step to build the try's grid on
the GPU with `step::BuildResizedGrid` (reconcile); all counted in `ResizeCellsReport::bytes_copied`.
In a GPU run a `CellResize` with `run_on = kCpu` keeps CPU mirrors of the arrays it reads (`GridFields` both
velocities, pressure, `has_pressure`, `pocket_of_cell`, the three wait arrays (A63; the seven region arrays only in
the later group, A64); `CellGeometry` and
`FacePieceGeometry` of the last geometry; the seven gas arrays), reads the committed arrays into them, works on the
CPU and writes the try arrays back with slice `WriteBuffer`; the mirrors are the same in both batched modes (state
and geometry stay on the GPU). Value loops run over blocks of the range, then cell or face places; one iteration
writes one destination.

**Batched runs.** Every buffer of a batched run follows the placement rule of 5.7 (plan decision, 2026-11-14,
buffers in batched runs; A17: a run is all CPU or all GPU, and in a batched GPU run only the particles and, with
solves in pieces, the solver buffers stay in CPU memory, moved in planned copies the planner counts and the
reports measure). This module adds no buffer or copy the step plan's Appendix A does not list. Its own copies:
the piece map read back per load and `WriteBlockPartialsRange` (both in `BatchReport`, part of a load), the
resize's `ReadGrid` copy (one piece), cell list upload and mirrors (`ResizeCellsReport::bytes_copied`,
`copy_count`), the `size_change` read back (1 B per cell place, A41). (`ChooseCellSize`'s copy is gone, A41.)
CHANGED (A58): every `kRunOn` array below is a piece of the run's `run_memory` and kept for the whole run (no
`kRunOn` entry is "one call" or "one step try"); the `kCpu` rows kept for the whole run are pieces of
`cpu_run_memory`.

**GPU.** Arrays (owner in brackets; all stay on the run's processor unless a row says `where = kCpu`):

| Array | Element | Per | Kept for |
|---|---|---|---|
| `merge_ready_duration`, `held_velocity_gradient_size`, `merge_ready_step_count` (`step::GridFields`, A63) | `double` x 2, `uint8` | cell place, 2 copies | whole run |
| `depth_below_surface`, wanted width, merge durations after, held gradient after, merge weight; size changes, merge wait steps after (`CellResize`, A41) | `double` x 5, `uint8` x 2 | cell place | whole run |
| refinement factors (`CellResize`, A78, A79; replaces the A63 regions and map) | `int8` code | pyramid code | whole run |
| `ResizeMap`: source kind and old index (`CellResize`) | `uint8` + `uint32` | cell place and face place of the new grid | one call |
| not in the first version (A64): region byte, layers, three amounts, ingredient volumes (`step::GridFields`); region wish, amounts of removed particles per cell, refill list (`CellResize`) | `double` x (3 + ingredients); `particles::NewParticles` arrays | cell place; `max_added_per_try` | one call |
| block partial sums (`BatchedRun`, `CellResize`) | `double` | cell block x sums in use | whole run (A58) |
| loaded piece storage (built and counted by the step (A58: not by `BuildBatchesMemoryEstimate`); borrows the step's `grid::GridBuilderBuffers`, reconcile) | `grid::BuildGridMemoryEstimate(kGpu, 1, limits.max_cell_blocks, kOneForAllGrids)` without the `builder buffers:` entries | cell block at `max_loaded_blocks` | whole run |
| border search (`BatchedRun`, A17) | `grid::BuildBorderSearchMemoryEstimate(run_on, limits.max_cell_blocks)` + two `CellBlockIndex` buffers | cell block at `max_loaded_blocks` | whole run |
| piece map copy (`BatchedRun`, A17; `where = kCpu`) | `uint32` | loaded cell block | whole run |
| committed grid copy, resized grid on the CPU (`CellResize`, A17, grid on the GPU, `GridCopies::kHeldByCellResize` only; `where = kCpu`; changed, reconcile: not built or counted with `kHeldByStep`, where the step's CPU copy pair serves) | `grid::BuildGridMemoryEstimate(kCpu, 2, limits.max_cell_blocks, kOneForAllGrids)`: two buffers, one CPU `GridBuilderBuffers` | cell block at the limit | whole run |
| cell list upload (`CellResize`, A17; `where = kRunOn`) | `grid::CellId` | cell place | whole run |
| CPU mirrors (`CellResize` with `run_on = kCpu` in a GPU run; `where = kCpu`) | as the mirrored arrays | cell and face place | whole run |

Work: one thread per new cell place gathers from the old grid (a merge reads 8 fixed sources, a split 1 and
6 neighbours); one thread per new face place; the 12 inner faces of a split by the thread of the big cell.
Region transfer: one thread per cell sums its particles in id order (at most the particles of one cell);
places in the refill list come from an integer prefix sum of per-cell counts, so ids do not depend on
timing (later group). `MoveGridOnlyLiquid` and `ApplyGridOnlyRegionToGrid` are gathers per cell and per face (later
group). `ComputeCellSizeChanges`: one thread per cell place per part, gathers only (A41). The repeated
minimum of 5.2 runs until a pass changes nothing. Returned to the CPU: the `size_change` bytes (A41), the counts of
5.1 step 3, the ledger's block partials (later group), one failure record with the first failing cell in fixed
order.

**Ranges.** `ComputeDepthBelowSurface`, `ComputeCellSizeChanges` (part 1 over every range before part 2),
`ComputeLiquidTotals`, the value remap (and in the later group `ApplyGridOnlyRegionToGrid`, `MoveGridOnlyLiquid`)
take a `CellBlockRange`. Border: touching cells
(`ApplyGridOnlyRegionToGrid` also the 1.5 widths `ComputeCarriedVelocity` reads, which touching cells
cover). The plan and the closure search are whole-grid CPU work in every run.

**Tolerance, CPU against GPU.** Plans, `size_change`, wait arrays (A41, A63), regions, layers, ids, counts, the
ledger's counts: equal. Remapped
values, moved amounts, carried velocity: `|a - b| <= 1e-13 * max(|a|, |b|) + floor`, floor `1e-13` times the
largest value of that array in the case. `depth_below_surface`: equal bytes expected (sums of the same two
terms), demanded within `1e-13` of the cell width.

## 7. Tests

**Analytic and conservation (CPU, then the same on GPU within section 6).**

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| Wanted width (moved to grid, A41) | depths 0, 4, 8, 16, 64 smallest widths; solid at 1 and 5 widths | smallest, smallest, smallest, 2x, 8x; solid: smallest, 2.5x capped to 2x (grid's tests) | exact |
| Change per cell (CHANGED, A41, A63) | width 2 w wanted 1.5 w; width w wanted 2.5 w and 2.49 w at depth 20 w; keep-count cell; depth 7.99 w and 8 w; each veto flag | `kSplit`; `kMayMerge`, `kKeep`; `kKeepsItsSize`; `kKeep`, `kMayMerge`; `kKeep` (`ComputeCellSizeChange, ...`, plan Task 7) | exact |
| Velocity gradient (NEW, A62) | uniform field; linear field `A` (same-size and smaller neighbours); rigid rotation Ω = 10 rad/s; beside a bigger neighbour | 0; `‖A‖_F = 2.5406692`; `sqrt(2) Ω = 14.142136`; within the off-axis bound (`ComputeVelocityGradientSize, ...`) | exact; 1e-13 relative; bound |
| Shear term (NEW, A62; `literature-cell-size.md` 5(f)4) | rigid rotation Ω = 10 rad/s, `U_ref` = 1 m/s, deep cells 0.04 m; uniform translation | every cell `kSplit`, wanted 0.017678 m; no shear split (`ComputeCellSizeChanges, RigidRotation_Splits`, `..., UniformTranslation_NoShearSplit`) | 1e-15 relative; exact |
| Wait of time and steps (NEW, A63) | 0.05 s and 1 step per call; 0.02 s per call; one break; zero elapsed; a shear burst that ends | `kMergeReady` at call 8 (steps decide); at call 13 (time decides); 8 more calls after a break; counters unchanged; held gradient kept for the wait (`ComputeCellSizeChanges, MergeAfterTimeAndSteps`, `..., BrokenOnce_WaitStartsAgain`, `..., ZeroElapsed_CountersUnchanged`, `..., ShearBurstEnds_HeldForTheWait`) | exact |
| Surface and refinement volume (NEW, A61; CHANGED A78, A79) | depth 0; a factor-0.5 zone over deep calm cells; a factor-2 zone; a key change; factor 1 everywhere | wanted `w_min` whatever the other terms; split inside the zone only; deep cells one level coarser except in the surface band; the new key's splits at the next resize and merges only after the wait; same bytes as no volume (`ComputeCellSizeChanges, SurfaceCell_IsSmallest`, `..., FactorHalfZone_SplitsInsideOnly`, `..., FactorTwoZone_CoarsensOutsideSurfaceBand`; `ResizeCells, KeyChange_SplitsAtOnceMergesAfterWait`, `ResizeCells, FactorOneVolume_SameBytesAsNoVolume`; grid: `ComputeWantedCellWidth, FactorHalf_HalvesDeepWidth`, `ComputeCellSizeFactorCode, BigCell_IsSmallestCodeOverItsBox`, `ComputeNewCellSizeLevel, NextToBigCell_OneLevelFiner`) | exact |
| Same bytes (NEW, A41; A78) | the shear scene, and the shear scene with a factor-0.5 zone, cut into one range, 1-block and 3-block ranges; CPU against GPU | equal `size_change` and after-row bytes (`ComputeCellSizeChanges, AnyBatchCut_SameBytes`, `..., OnGpu_SameBytesAsCpu`; grid: `ComputeWantedCellWidth, OnGpu_SameBytesAsCpu`) | exact |
| Deep tank coarsens with particles (NEW, A64; CHANGED A65 to A67) | calm tank 2 m deep, `w_min` 0.05 m, 5 levels, 3 s; **started all-smallest on purpose** (it tests the coarsening cascade; a run starts on `BuildStartCells`' grid, A66) | every cell deeper than `8 w_min` at the depth term's level; every liquid cell holds about eight particles of its own size: ~~1..16 per cell except counted drifted ones~~ 1..16 per cell, and after every resize every particle's `size_level` equals its cell's level except those counted in `waiting_particle_count` and `lone_particle_count` (A67); mass and momentum kept per resize (`ResizeCells, CalmDeepTank_CoarsensWithParticlesInEveryCell`); merged cells hold particles of their level (`ResizeCells, MergedCellHoldsParticlesOfItsLevel`) | `kAmountTolerance` |
| Start on the start cells (NEW, A66) | the same tank seeded on `step::BuildStartCells`' grid, 8 per cell at each cell's level, at rest, 3 s | no resize splits a cell for depth (`wanted_split_count` 0 at every resize), `mismatched_particle_count` 0 at every pass, particle count never above the start count; every cell deeper than `8 w_min` at the depth term's level after 3 s (`ResizeCells, CalmDeepTankOnStartCells_NeverSplitsForDepth`); against the same tank started all-smallest (step's `ApplyStartCellsForTest(scene, TestStartCells::kAllSmallest)`) run until a resize changes nothing: coarsened count <= start count <= 1.25 x coarsened count (`BuildStartCells, DeepTank_SameParticlesAsCoarsenedRun`, handed over by the step, A66) | exact |
| Flicker and conservation (NEW, A63; 5(f)1, 2) | calm pool 1 m deep, `w_min` 0.02 m, 5 levels, **started all-smallest on purpose** (the coarsening time is measured from it, A66); a layer split then merged | depth level reached within `0.25 s x 4 + 32` steps, then no split or merge for 10 s (`ResizeCells, CalmPool_ReachesBiggestCellsThenStays`); surface cells always at `w_min` (`..., CalmTank_InsideCoarsensSurfaceStaysFinest`); mass and momentum kept (`..., SplitThenMerge_ConservesMassAndMomentum`) | exact; `kAmountTolerance` |
| Rising particle split on the way (NEW, A67) | calm deep tank after 3 s (all-smallest start, coarsened), one particle of a level-2 cell (0.2 m) moved up 0.5 `w_min` per step by a test-side snapshot edit until it reaches the surface | after every resize every liquid particle in a smallest-level cell has `size_level == 4` (so in every committed step where the moved particle, or a child of it, is in a smallest-level cell, it is of the smallest size); total particle mass and momentum within 1e-15 relative per resize (`ResizeCells, ParticleRisingFromDeepCell_ArrivesSplitToSmallest`) | exact; 1e-15 |
| Sinking particles merged (NEW, A67) | the same tank; eight smallest-level particles moved by snapshot edit into one level-3 cell (0.1 m); then three only | eight: merged at the next pass, `kinetic_energy_change <= 0`; three: merged at pass 16, not 15, `waiting_particle_count` 3 before (`ResizeCells, SmallParticlesInCoarseCell_MergedByThePass`, `..., ThreeSmallParticles_MergedAfterSixteenPasses`) | exact |
| Particles-only pass (NEW, A67) | committed state with the cells at their wanted levels and the wait arrays settled, one mismatched particle | the plan is empty, the pass still runs `AdjustParticleCount`, commits a same-cell resize with `version + 1`, `resized = true`; with no mismatched particle: `resized = false`, same `state_version` (`ResizeCells, OnlyMismatchedParticles_PassStillRuns`) | exact |
| Energy of the pass reported (NEW, A67) | a pass with one split and one merge of eight | `ResizeCellsReport::particle_count.kinetic_energy_change` equals `step::ResizeResult::size_change_kinetic_energy` and the step's `StepReport::size_change_kinetic_energy` of that try, bytes; `mismatched_particle_count` likewise (`ResizeCells, ReportFieldsReachTheStepReport`) | exact |
| Sheets, splashes, jets (NEW, A62, A63; 5(f)3, 7) | falling sheet 0.04 m thick and a stretched bridge; a ball splashing into a coarsened tank; a 0.1 m jet at 2 m/s | no merge within `8 w_min` of either face; no particle slower than 1e-3 of the splash speed for 20 steps above the surface, momentum continuous; cells within 2 jet radii of the jet edge at `w_min` after 0.5 s (`ResizeCells, ThinSheetAndPinchOff_StaySmallest`, `..., SplashIntoCoarseRegion_NoStuckParticles`, `..., Jet_KeepsSmallCellsAlongShearLayer`) | exact; 1e-12 relative |
| Reference speed (NEW, A62) | `largest_liquid_speed` 0.5 m/s, biggest width 0.1 m; 2 m/s | `max(0.5, sqrt(9.81 x 0.1)) = 0.990454` m/s; 2 m/s (`ResizeCells, ReferenceSpeedFromCommittedLimits`) | exact |
| One reserved block (NEW, A58) | each builder into a block of exactly its estimate per `where`; the same less 256 B | `handed_out_bytes() == reserved_bytes()`; `kOutOfMemory` with the bytes asked (`BuildCellResize, IntoRunMemory_HandsOutExactlyTheEstimate`, `..., RunMemoryOnePieceShort_IsOutOfMemory`, `BuildBatchedRun, ...` the same two) | exact |
| Error enums (CHANGED, item 31) | value counts | `CellResizeError` 10 values, `kOutOfMemory` 9; `BatchesError` 8, `kOutOfMemory` 7 (`CellResizeError, OutOfMemoryIsLast`, `BatchesError, OutOfMemoryIsLast`) | exact |
| Gating, grid-only region (NEW, A64; later group) | standing wave, wavelength 1 m, tank 0.5 m deep with the bottom half grid-only, 10 periods (`T` = 0.80180 s); a tank sloshed for 20 s | amplitude within 5 % (a starting value) of the all-particle run's; no step rejected for travel at the band edge, kinetic energy never above its start plus the work of gravity, no particle stuck in a grid-only cell (`GridOnlyRegion, StandingWave_KeepsAmplitude`, `GridOnlyRegion, ParticleBandEdge_StaysStable`) | as stated |
| Surface reach 8, 6, 4 (NEW, A68; later group (1), test only) | standing wave (1 m wavelength, 0.5 m deep, 10 periods), dam break, hydrostatic column, sloshing tank, bathtub splash rendered; each at reach 8, 6, 4 | a table per reach: each analytic error, planned memory, step time; the decision rule of 5.11 (1) applied (`SurfaceReach, AnalyticScenesAtReach8_6_4`, report only, never a gate of the first version) | each test's own |
| Split, merge on the list | split a cell, merge it back | the starting list; `CheckSizeStepRule` passes at each step | exact |
| Closure | 3 nested sizes, split a cell touching a corner of a cell two levels bigger | exactly the cells of the hand-made list; the old paper's corner case | exact |
| Refused split | closure reaches a cell holding snow; the same with a springy solid | request dropped, counted, grid unchanged | exact |
| Merge wait | test holds for 0.2 s, breaks one step, holds again | no merge before 0.25 s of unbroken time; a failed resize earns no time | exact |
| Amounts, split then merge | random positive amounts in a 3 x 3 x 3 patch, split the centre, merge it | small cells sum to the big cell; merge returns the start value | 2 rounding steps |
| Amounts, limiter | linear field; a field with a jump | linear: small cells follow the slope; jump: every small cell within the neighbours' range | 1e-14 relative |
| Ingredient volumes | a record of 3 ingredients split, then merged | each small cell's fractions equal the big cell's; merge returns the start volumes | 2 rounding steps |
| Face flow, split | flow with zero imbalance; flow with imbalance `q` | each small cell has imbalance `q / 8`; flow through each outer big face kept | 1e-13 of the largest flow |
| Face flow, merge | 4 pieces with different velocities | big face flow = their sum | 1 rounding step |
| Pressure at rest across a resize | column at rest, mixed sizes, split and merge one layer, then one step | largest speed stays below the solve's limit | pressure's limit |
| Depth | flat surface over 3 sizes; a closed bubble; a tracked gas region above liquid | depth = distance from the surface cell centres; cells under the bubble and under the gas measure from it | 1e-13 of a cell width |
| Carried velocity (not in the first version, A64) | rigid translation; rest | the same field; exactly the committed value | 1e-13; exact |
| Grid-only move (not in the first version, A64) | uniform flow through a row of grid-only cells; a flow that would empty a cell | totals kept, each cell's volume unchanged; `kGridOnlyCellDrained`, then the step's `kShortenStep` | `kAmountTolerance`; exact |
| Region cycle (not in the first version, A64) | deep tank; the surface is lowered and raised 100 times through the two depths | mass, rest volume, current volume, each ingredient volume of particles plus cells unchanged; no particle left in a grid-only cell after a resize | `kAmountTolerance` per resize |
| Border refill (not in the first version, A64) | steady upward flow through the region face, 200 steps | every border cell eighth holds a particle after each resize; grid-only excess stays below one particle volume per face quarter; totals kept | exact counts; `kAmountTolerance` |
| Absorb (not in the first version, A64) | particle of the record's material sent into a grid-only cell | removed at the next resize; the cell's record grew by its amounts | 1 rounding step |
| Foreign particle (NEW) (not in the first version, A64) | a snow particle, then a particle of a second liquid that does not mix, sent into a grid-only cell | the cell returns to the particle region in the next resize; no record holds two materials; totals of each material kept | `kAmountTolerance` |
| Two liquids, material sets (NEW; A64 form) | two non-mixing liquids, deep, side by side, both in the one liquid set, with a snow set beside them; 50 steps | first version: snow cells keep their size, every liquid cell holds particles of its level, each material's particle totals kept per resize, `MaterialSetKey` checked; later group: `ApplyGridOnlyRegionToGrid` adds every record to the liquid set's arrays and nothing to the snow set's; no record holds two materials; cells along the contact stay in the particle region; each set's `MaterialSetKey` checked (`kWrongMaterialSet` on a swapped array) | exact |
| Gas beside liquid (NEW, A10; A64 form) | smoke over a moving water surface in a deep tank, cells resized every step | the seven gas arrays remapped: gas amount kept; every cell holding or touching gas at the smallest size; (later group: no grid-only cell touches a gas cell); gas markers never removed `kIntoGridOnlyRegion`; `ComputeLiquidTotals` with the liquid set leaves the markers out | `kAmountTolerance`; exact |
| Full-particle comparison (not in the first version, A64) | standing wave in a deep tank, with and without the grid-only region, same cells | surface height within one smallest cell; kinetic energy within 5 % over 2 s | as stated |
| C2 (CHANGED, A67) | region of 5 x 5 x 5 cells split in one resize, particles everywhere | every particle is split with the conservation of particles 5.4; ~~the others keep their size and are counted~~ those less than 1.5 widths inside on either grid are counted in `incomplete_split_count` (count from positions, test side) | particles' tolerances |
| ~~Drift across a size change~~ **replaced** (A67) | ~~a particle carried across an existing size change, then several resizes~~ a smallest-level particle carried down across an existing size change into a coarser cell, then several resizes | ~~never split or merged; counted in `skipped_incomplete_count`~~ counted in `mismatched_particle_count` at the first pass after the crossing; merged within 16 passes or counted as lone (`ResizeCells, DriftAcrossSizeChange_MatchedOrCounted`, replaces `..., DriftAcrossSizeChange_NeverSplit`) | exact |
| Failed resize | make each of steps 4 to 8 of 5.1 fail (limit of blocks, limit of particles, a NaN) | snapshot bytes equal to those taken after the step's commit; `state_version` unchanged; the storage is given back and the next step commits | exact |
| Nothing to do (CHANGED, A67) | resize twice in a row, no mismatched particle | second call: `resized = false`, same `state_version`, `mismatched_particle_count` 0 | exact |
| Slab limit named (NEW, A68) | pool 0.8 x 0.8 m at `w_min` 0.0125 m (`blocks_per_layer` 16 x 16 = 256), allowed bytes chosen so the loaded room holds 8 layers of full blocks plus 100 B; the same with room for 10 layers | first: `kBatchDoesNotFit` with `slab_limit` {256, 9, `b`, room, `9 x 256 x b`, 0.64 m², `room x 16 x 0.0125² / (9 b)`} and `needed_bytes`, `allowed_bytes` equal to its two byte fields; `max_plan_area` in [0.5688, 0.5690] m² (8/9 of 0.64); second: a plan with `T` = 2 and the same `slab_limit` formula (`BuildBatchPlan, LayerDoesNotFit_ReportsSlabLimit`, `..., Fits_ReportsSlabLimit`) | exact; 1e-15 relative for the area |
| Batch plan (changed, A15) | rows chosen so the whole-grid arrays are below and above the share; allowed bytes between the batched layout with solves in pieces and the one with whole solves; a solve piece below three layers; CPU rows above `allowed_cpu_bytes`; a layer wider than a batch; one and three material sets | `kWholeGridOnProcessor`, `kInPieces` (every row sum below `allowed_bytes`), `kBatchDoesNotFit`, `kStateDoesNotFitCpu`, `kBatchDoesNotFit`; `max_loaded_blocks` falls with each extra set; runs with no GPU (`BuildBatchPlan, WholeGridBelowShare_SolveOnProcessor`, `..., WholeGridAboveShare_SolveInPieces`, `..., FitsOnlyWithSolvesInPieces`, `..., SolvePieceBelowThreeLayers_IsBatchDoesNotFit`, `..., CpuRowsAboveAllowedCpu_IsStateDoesNotFitCpu`, `..., LayerWiderThanABatch_IsBatchDoesNotFit`, `..., MoreMaterialSets_FewerLoadedBlocks`, `..., RunOnGpuWithoutAGpu_StillPlans`) | exact |
| Plan report (NEW, A15) | open-top wave; closed box | `bytes_on_processor`, `bytes_on_cpu` equal `ComputeMemoryBytes` of the rows by `where`; `solve_loads_per_iteration == 3 * solve_batch_count` (wave), `4 *` (box) (`BuildBatchPlan, ReportsBudgetsAndCopies`) | exact |
| Stacked border (changed, A15: pass 1 only) | for each stage k of pass 1, the run parts after k null, on a mixed-size grid (deep tank after 5 resizes) | the arrays pass 1 wrote, batched against the same stages on the whole grid: same bytes (`ApplyStagesInBatches, PassOneStagePrefix_SameBytesAsWholeGrid`; replaces `EveryStagePrefix_SameBytesAsOnePiece`) | exact |
| Stacked border, buffer form (NEW, A17) | 100 random grids, CPU buffers | `FindStackedBorder` equals the vector form of `grid::FindBorderBlocks` layer by layer (the span form is retired, grid Ruling 66) (`FindStackedBorder, PassOne_EqualsRepeatedFindBorderBlocks`) | exact |
| Piece map (NEW, A17) | loads of several ranges | `bytes_from_processor` grows by 4 B per loaded block per load (`WriteBatch, PieceMapReadBackOncePerLoad`) | exact |
| Pass-1 geometry (NEW, A15) | after pass 1 | the whole buffers' snapshot rows equal `FindLiquidSurface` on the whole try (`ReadBatch, PassOneGeometryEqualsWholeGrid`) | exact |
| Write-back (NEW) | one pass | border blocks of the whole try keep their bytes (`ReadBatch, WritesBackRangeBlocksOnly`) | exact |
| Cut inside a layer (changed, A15) | `max_blocks_per_batch` 7 and 13 against one layer, both batched modes | same bytes as the layer cut; within step 6.4 of one piece (`ApplyStagesInBatches, CutInsideALayer_SameBytesAsLayerCut`; replaces `CutInsideALayer_SameBytes`) | exact; step 6.4 |
| Cancel and failure (NEW) | flag set between batches; between solve batches (5th load, `kInPieces`); pressure `max_iterations` 0 for one try | `kCancelRequested`, snapshot bytes as before the try, the next try commits; the solve failure rejects the try and the retry commits (`ApplyStagesInBatches, CancelBetweenBatches_IsCancelRequested`, `..., CancelBetweenSolveBatches_IsCancelRequested`, `..., SolveFailsInOneBatch_CommittedBytesKept`) | exact |
| Solve copies (NEW, A15) | `kInPieces`; `kWholeGridOnProcessor` | `solve_batch_loads` equals the solvers' reports summed; 0 (`ApplyStagesInBatches, ReportCountsSolveCopies`); first try: loads and bytes per iteration equal the plan's (`BatchReport, FirstTrySolveCopiesEqualThePlan`) | exact |
| Fits only in pieces (NEW, A15) | deep tank lengthened to 3.2 m (16 block layers), processor bytes between the two layouts | plan `kBatchedWithSolvesInPieces` (`kBatchDoesNotFit` when forced whole); `BuildSimulation` allocates the estimate per `where`; 10 steps commit; liquid volume change within pressure's reported change (`BatchedRun, FitsOnlyWithSolvesInPieces_Commits`) | as stated |
| Load binding (NEW) | load two ranges of different block counts in turn; then pass a loaded array bound at the earlier load to a stage | each load rebinds; the stale array is `kWrongGridVersion` from the stage's own module | exact |
| Sum over blocks | 10,000 random partials, cut into 1, 7 and 64 batches | equal bytes | exact |
| Per-solid sums (NEW) | a free box floating in a batched run, cut into 1, 3 and 9 batches (and solve pieces of 3 and 5 layers) | equal pushes and contact reactions, byte for byte (cuts of one mode; `Batches, FreeBoxCuts_SamePushesAndReactions`) | exact |
| Grid copy path (NEW, A17) | deep tank with `grid_run_on = kGpu` forced onto CPU buffers (the copy path runs CPU to CPU) | same report and snapshot bytes as `grid_run_on = kCpu`; `bytes_copied == 456 * committed blocks + 64 * committed blocks (size_change, A41) + 8 * new cells`, `copy_count == 3` (`ResizeCells, GridCopyPath_SameAsCpuGrid`); memory estimate entries add the grid copies (`BuildCellResizeMemoryEstimate, GridOnGpu_AddsTheGridCopies`, `..., CpuMirrorsCountedOnCpu`) | exact |
| Grid copies by layout (NEW, reconcile) | `GridCopies` with each `grid_run_on`; a `kHeldByStep` `CellResize` | no `"cell resize: committed grid copy"` or `"... resized grid on the CPU"` entry with `kHeldByStep`, and `BuildCellResize` allocates exactly its estimate (`BuildCellResizeMemoryEstimate, HeldByStep_NoGridCopyRows`, `..., BatchedGpuLayout_NoGridCopyAllocated`); `kHeldByCellResize` or `kHeldByStep` with a CPU grid, `kNone` with a GPU grid: `kBadInput` (`BuildCellResize, GridCopiesNotMatchingGridRunOn_IsBadInput`) | exact |
| Held grid copy (NEW, buffers in batched runs; **withdrawn**, A41: `ChooseCellSize` removed) | `ChooseCellSize` with null committed tables | same level as with CPU tables; a stale copy is replaced once, counted: `copy_count` 1, `bytes_copied` 456 x blocks (`ChooseCellSize, NullCommittedTables_UsesTheHeldCopy`) | exact |
| Planned equals allocated (NEW, buffers in batched runs) | the plan copied into the step plan, `step::BuildSimulation` on the CPU with a forced batched layout | `ReadMemoryReport` `allocated_bytes` and `allocated_cpu_bytes` equal `bytes_on_processor` and `bytes_on_cpu - estimate.snapshots` (`BuildBatchPlan, BudgetsEqualTheBatchedAllocation`); `bytes_per_loaded_particle` exceeds the particle rows by the per-particle rows of the four loaded module instances, exactly (`BuildBatchPlan, LoadedRowsIncludeModuleInstances`) | exact |
| Array places (NEW, buffers in batched runs) | `BuildFiveMaterialScene` in the batched layout | for every array of `ReadStepArrays` the placement rule from `kStagePlaces` equals `StepArray::whole_place`; no range part of pass 1 reads `grid_edge_is_open_to_sky` (else stop and ask the owner) (`kStagePlaces, GiveTheStepsArrayPlaces`) | exact |
| Stage 3's whole part (NEW, buffers in batched runs) | step: a lid over two ranges cut through the pocket under it | the whole part's marks equal the one-piece stage's bytes, a range-only call would differ (`FindSolidSurfaces, OpenToSkyOnWholeGridEqualsOnePiece`); `StageTable, WholeGridPartsOnlyForStages3_9_19` (was `..._9_16_19`; A64: no stage-16 part) | exact |
| Contact partials to stage 17 (NEW, buffers in batched runs) | `BuildFreeBoxScene(500)`, 3 batches | the CPU instance's block partials equal a one-piece `CollideSolids`' bytes; the report counts their bytes once (`ReadBatch, CollideSolidsPartialsReachTheStage17Buffers`) | exact |
| Piece map to foam (NEW, buffers in batched runs) | hand-written `make_foam_and_spray` | sees `block_in_whole_grid` equal to the map copy, never empty, in every batch (`WriteBatch, FoamInputsGetThePieceMap`); step: `ApplyGridPiece, MapOfWrongLength_IsBadInput`, `MakeFoamAndSpray, LoadedTryPassesThePieceMap` | exact |

**Reference and proofs: none.** `adaptive_remap.py` repeats the split and merge formulas; "the small cells
sum to the big cell and stay within their neighbours" is checked directly by the tests above, and no
uncertainty remains that a second implementation would remove. `VolumeTransport.lean` concerns the volume
output of render output and is not ported.

**Comparison with the proof of concept** (supporting): the amounts split of one cell with six neighbours
against `ProlongLiquidInventory3D` (cell eighths permuted), `1e-12` relative; the inner face flows against
`RedistributeSiblingFluxes`, `1e-12`; `ComputeCellSizeChange` against `DecideAdaptiveWidth3D` for the same
wanted width. The region transfer and the carried velocity changed and are not compared.

**GPU and batches (changed, A15).** Every row on `RunOn::kGpu`; two repeats give equal bytes. Batched, the
central test, in both batched modes (`kBatched`; `kBatchedWithSolvesInPieces` with solve pieces of 3 and 5
layers): the reference scenes (wave, syrup, fire) and the snow example for 20 committed steps with
`max_blocks_per_batch` forced to 1 layer, 3 layers and everything: within one mode equal snapshot bytes for every
cut, and each batched run within the step's tolerance (6.4 of the step design) of the one-piece run of the same
processor; the test also prints whether the bytes happen to be equal. Without a GPU the container runs all CPU
rows, the batch plan rows, and the batched schedule with both processors set to the CPU (the copies become CPU
copies). Replaced byte identity between modes (`plan-decisions.md` "Batched solves (A15)"):

| Old | New |
|---|---|
| `ApplyStagesInBatches, EveryStagePrefix_SameBytesAsOnePiece` | `ApplyStagesInBatches, PassOneStagePrefix_SameBytesAsWholeGrid` (local pass-1 outputs only: the border check) |
| `ApplyStagesInBatches, CutInsideALayer_SameBytes` | `ApplyStagesInBatches, CutInsideALayer_SameBytesAsLayerCut` |
| `ApplyStagesInBatches, Small*_SameBytesAsOnePieceForEveryCut` (4: wave, syrup, fire, snow) | `..._SameBytesForEveryCut` and `..._WithinStepToleranceOfOnePiece` (4 each, both batched modes) |
| `Batches, FloatingBox_SameBytesAsOnePiece` (handed by solids) | `Batches, FloatingBox_WithinStepToleranceOfOnePiece` |
| `TryStep, BatchedFireBesideWater_SameBytes` (handed by gas) | `TryStep, BatchedFireBesideWater_WithinStepTolerance` |
| `Batches, DeepTankWithCellResize_SameBytesAsOnePiece` | `Batches, DeepTankWithCellResize_WithinStepToleranceOfOnePiece` |

`Batches, FreeBoxCuts_SamePushesAndReactions` stays: it compares cuts of one mode. GPU rows (label `gpu`):
`ApplyStagesInBatches, OnGpuEveryScene_WithinStepTolerance` and `..., OnGpuTwice_GivesIdenticalBytes` (both modes);
`ResizeCells, InBatchedGpuRun_EqualsOnePieceCpuResize` (within section 6); `ResizeCells,
InBatchedGpuRun_ReadsTheStepsGridCopy` (buffers in batched runs: `bytes_copied` has no `456 * blocks` term; the result
equals the run with its own copy; reconcile: after the commit `step::FindCpuGridCopy` is the CPU resized grid and
`step::ReadCopyCounts` did not grow); `ResizeCells,
OnePieceGpuRun_UsesTheGridCopy` (A17: `bytes_copied` as the grid copy path; same report as the CPU run within
section 6). Label `no_gpu`: `BuildBatchedRun, OnGpuWithoutAGpu_IsRunOnCallFailed`; `BuildBatchPlan,
OnGpuInCudaBuild_StillPlans`.

## 8. Method paper section

Labels: `<sec-cell-resize>`, `<sec-cell-size-rule>`, `<sec-remap>` (the label the naming review cites),
`<sec-regions>`, `<sec-region-transfer>`, `<sec-batches>`; figures `<fig-cell-size-rule>`,
`<fig-split-remap>`, `<fig-regions>`, `<fig-batch-passes>`; pseudocode `<proc-apply-stages-in-batches>` (steps 1 to
4 of 5.8). Pressure's `<sec-solve-in-batches>` holds the solve in pieces itself.

Outline (amended 2026-11-15, A41 to A64): 1 why resizing is its own transaction and what a failed one leaves; 2 the
shared wanted width (depth, solids, shear with the reference speed, times the refinement volume's factor (A78, A79); the surface smallest),
the held gradient and the merge wait of time and steps, and particles that follow their cells' levels in deep
liquid with the cost bound (`<proc-compute-cell-size-changes>`); items 6 to 8 below only with the later group (A64); 3 the cells that must also split; 4 remap of amounts, pressures, labels and face flows; 5
splitting particles needs both grids, and (A65 to A67) particle size follows the cell: the start on
`BuildStartCells`' grid, mismatched particles split or merged at every pass, the energy of a merge; 6 two regions and one ledger, one material per record; 7 the grid-only
region inside a step; 8 region transfer by particles; 9 batches: what stays on the GPU in each mode, two
passes, the stacked border, the three ways to run and how the planner chooses and reports the memory budgets and
copies, why the CPU fallback is gone (A17), the order of sums and agreement within the solver's tolerance, the
cost, and which scenes each mode is for (syrup in pieces: possible, slow), and (A68) the slab limit on the plan area
with its formula. `<fig-batch-passes>` draws the solve as
whole or as its sweeps over solve batches in a second lane.

Figures: `<fig-cell-size-rule>`, a vertical cut through a wave over a solid step, cells coloured by size,
the two reaches drawn; `<fig-split-remap>`, one big cell and its 8 small cells with the outer pieces, the 7
tree faces and the 5 zero faces; `<fig-regions>`, particle region, overlap layers and grid-only region with
the two depths, and the four transfers as arrows; `<fig-batch-passes>`, a time line of one try: CPU stages,
pass 1 over three batches with their borders, the whole-grid solve, pass 2.

Pseudocode: `ResizeCells` (the nine steps), `FindCellsThatMustSplit`, `ApplyRegionTransfer`,
`ApplyStagesInBatches`. Cite from the old bibliography: [17] Ferstl 2016 (particles near the surface, grid
below), [1] Jiang 2015 (what a split must keep), [6] Wang, Sun, Zhu 2026 (cells of mixed sizes), [5] Wang
2020 and [15] Fei, Huang, Gao 2021 (cell blocks on the GPU). Ando and Batty 2020 and Losasso, Gibou, Fedkiw
2004 are cited by author and year (decision 11).

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| `DecideAdaptiveWidth3D`, `AdaptiveWidthDemand3D` | `ComputeCellSizeChange`, `ComputeWantedCellWidth`, `CellSizeFacts` | NEW (decision 12) |
| `AdaptiveWidthChange3D { kKeep, kRefine, kCoarsen }` | `CellSizeChange { kKeep, kSplit, kMayMerge }` | DECIDED words |
| `coarsening_eligible_duration_s`, `hysteresis_duration` | `merge_ready_duration`, `merge_wait_duration` | NEW (decision 12) |
| `AdaptiveCoarsenVeto3D` | the conditions of 5.3 step 5; `refused_merge_count` | DECIDED (`veto` folds into reject) |
| `RefinementClosure` | `FindCellsThatMustSplit` | NEW (decision 12; DECIDED `size_step_rule`) |
| `PrepareAdaptiveRefinement3D`, `PrepareAdaptiveCoarsening3D` | `SplitCell`, `MergeCells` | DECIDED |
| `AttemptWithAdaptiveFamily`, `...RefinementClosure`, `...Width` | `ResizeCells` | DECIDED |
| `AdaptiveRetainedLeaf3D`, `retained_leaves` | `ResizeMap`, `CellSource` | NEW (decision 12) |
| `RestrictLiquidInventory3D`, `ProlongLiquidInventory3D` | `ComputeMergedAmount`, `ComputeSplitAmounts` | NEW |
| `RestrictLiquidFaceFlux3D`, `ProlongLiquidFaceFlux3D`, `RedistributeSiblingFluxes` | `ApplyResizeToFaceVelocity`, `ComputeInnerFaceFlows` | NEW |
| `AdaptiveLiquidMaterialOwner3D { kParticles, kEulerian }` | `RegionKind { kParticleRegion, kGridOnlyRegion }`, `is_grid_only_region` | DECIDED (Q5) |
| band, core, `kParticleDepthInLeafWidths`, `kOverlapInLeafWidths` | `particle_region`, `grid_only_region`, `particle_region_depth_in_cell_widths`, `overlap_layer_count` | DECIDED |
| `PlanLiquidBandPlacement3D`, `kRetire`, `kReseed`, `RetireLiquidBandCell3D`, `ReseedLiquidCoreCell3D` | `ApplyRegionTransfer`; particles removed `kIntoGridOnlyRegion`, added `kRefillRegion` | DECIDED |
| `PrepareAdaptiveLiquidCoreTransport3D` | `MoveGridOnlyLiquid`, callback `MoveGridOnlyLiquidInStep` | NEW (decision 12); not in the first version (A64) |
| (none) | `ApplyGridOnlyRegionToGrid`, callback `ApplyGridOnlyRegionInStep` | NEW (decision 12; 2026-11-14); not in the first version (A64) |
| (none) | `ComputeCellSizeChanges`, `CellSizeChangesPart`, `CellSizeChange::kMergeReady`, `kKeepsItsSize`, `ComputeVelocityGradientSize`, `velocity gradient size`, `reference speed`, `held_velocity_gradient_size`, `merge_ready_step_count`, `cell size region`, `cell size map`, `ResizeCellsReport::totals_before` / `totals_after` | NEW (A41, A62, A63, A64; decision 41) |
| (none) | `ChooseCellSize` | removed (A41); grid `ComputeNewCellSizeLevel` |
| `BlendLiquidBandCarrier3D`, `particle_weight` | step 3 of 5.5, `particle_share` | NEW |
| `LiquidRemapInventory3D`, `inventory` | `GridOnlyAmounts` with `mass`, `rest_volume`, `current_volume`, `ingredient_volume` | VERIFIED, below (decision 3) |
| `carrier_velocity`, `eulerian_carrier` | `velocity_after_pressure` (committed), `carried_velocity` | VERIFIED, below (decision 3) |

**Verify: `grid_velocity` (was `carrier`).** Read: 715 uses in `native/`. The reviewer's reading ("the face
velocity field that moves liquid where there are no particles") fits one of eight uses.

| Use on `main` | What it is | Name here |
|---|---|---|
| `LiquidAdaptiveFaceState3D::carrier_velocity` (`liquid_state_3d.h:195`), `CommittedCarrierVelocity` (`liquid_substep_3d_staging.cc:49`) | The committed velocity of every face piece, in both regions | `step::GridFields::velocity_after_pressure` |
| `eulerian_carrier`; "the accepted Eulerian carrier is authoritative inside the core" (`liquid_adaptive_step_3d.cc:1225`) | The velocity a grid-only face starts a step with | `carried_velocity` |
| `particle_carrier` (`liquid_band_handoff_3d.h`) | The face velocity from particles | `transfer::GridFromParticles::velocity` |
| `LiquidHandoffCarrier3D` | Not a face velocity: density, volume ratio, centre velocity and fitted change of one cell, for a refill | `particles::RefillCell` |
| `carrier_mass`, `CheckCarrierInventory` (`liquid_band_handoff_3d.cc:226`) | Amounts that record implies, compared with the cell record | gone with the second ledger |
| `FrozenMotionCarrier3D`, `SampleMotionCarrier3D` | An object that reads a velocity at any position of a snapshot, for motion output | I/O and CLI's |
| `FoamCarrierStatus`, `bubble_missing_carrier` (`whitewater_transport_3d.cc`) | The liquid velocity that moves foam | surface, foam and spray's |
| `AdaptiveCarrierMode`, `CoarseFinePicCarrierSample3D` | A switch between three step variants; an experiment | not ported |

So `grid_velocity` is not used as a name here: every face velocity is a grid velocity, and
`grid_velocity_at_particle` is already decided for another thing (decision 3).

**Verify: `liquid_amount` (was `inventory`).** `LiquidRemapInventory3D` (`liquid_remap_3d.h`) is seven
things: mass, reference volume, current volume, momentum, enthalpy, a phase id and up to 8 constituent
volumes; not "mass and volume". In a particle-owned cell it is a copy of the particle sums (a second
ledger, kept equal by exact sums); in a grid-only cell it is the only holder. `LiquidCavityInventory3D` is
the gas amount of an air pocket (pressure's `rest_volume`, `rest_pressure`); `foam_inventory_count` is a
particle count (surface, foam and spray). Here: the record of a grid-only cell is `GridOnlyAmounts`, its
fields carry the names `transfer::CellsFromParticles` and `particles::ParticleAmounts` already use, momentum is
not in it (the faces hold it), enthalpy is not ported (no rebuilt module gives liquid a temperature), and the
copy in particle cells does not exist. `liquid_amount` alone would hide which quantity is meant (decision 3).

Other NEW words: `depth_below_surface`, `wanted cell width`, `whole-grid arrays`, `loaded batch`, `stacked
border`, `pass`, `block partial sum`, `filled share`, `excess`; (A15) `solve batch`, `loaded solve piece`,
`sweep`, `budget` (a planned cost shown to the user, A22); (A65) `particle size follows the cell` (the one wording of
5.3); (A66) `start cells` (`step::BuildStartCells`' grid, the run's first grid); (A67) `mismatched particle` (a
particle whose `size_level` is not its cell's level), `waiting particle`, `lone particle`, `size change kinetic
energy`; (A68) `SlabLimit`, `plan-area limit` (`max_plan_area`), `slab batch`, `column or tile batch` (later),
`surface reach` (the setting `surface_reach_in_cell_widths`, the fine band being twice it in smallest cells).

| Old name on `main` | New name | Source |
|---|---|---|
| (none) | `SlabLimit`, `BatchPlan::slab_limit`, `BatchesFailure::slab_limit`, `bytes_per_full_loaded_block`, `loaded_batch_room_bytes`, `max_plan_area` | NEW (A68) |
| (none) | `ResizeResult::size_change_kinetic_energy`, `ResizeResult::mismatched_particle_count` (step's, decided here) | NEW (A67) |
| `skipped_incomplete_count` as the drift count of C2 | `skipped_incomplete_count` is removed from `AdjustParticleCountResult` (nothing is skipped any more); drifted particles are counted in `mismatched_particle_count` (counted at the start of the pass), incomplete splits in `incomplete_split_count`; `kinetic_energy_lost` (merges only) stays beside the new `kinetic_energy_change`, which is the one wired into the reports (particles' names) | CHANGED (A67) |

Replaced on 2026-11-14 (A29, A31): `grid::ReleaseBuffers` -> `grid::TakeBuffers`; `SumScratch` / `OrderScratch`
-> `SumBuffers` / `OrderBuffers`; `GridScratch` -> `grid::GridBuilderBuffers`; "scratch" -> "working buffers";
the overload pairs of `ComputeLiquidTotals` and `ComputeSumOverBlocks` -> one form each, taking `SumBuffers&`.

## 10. Not ported

- Flow exchange between the regions with a mass correction on one particle, and its exact-sum machinery:
  `adaptive_liquid_particle_exchange_3d.cc`, `adaptive_liquid_particle_exchange_3d_internal.h`,
  `host/adaptive_liquid_owner_exchange_3d.h`, `liquid_exact_inventory_sum_internal.h`,
  `PrepareAdaptiveLiquidOwnerExchange3D` (`adaptive_liquid_core_transport_3d.cc:533-588`).
- The copy of particle sums in every particle cell and its checks (`AccumulateInventories`,
  `CheckParticleOwnedInventories`, `MatchesGradedParticleInventory`); the momentum and enthalpy lanes of
  `LiquidRemapInventory3D`; `phase_id` (two liquids that do not mix are two materials).
- Certificates: `RepresentsPartitionSum`, `RepresentsSubfaceSum`, `Conserves`, the 64-epsilon and subnormal
  rules, `OutOfRange` on inexact halving, `AddUpperBound`, `kUnitCubeHalfDiagonalUpper`,
  `BoundLiquidBandLeafSignedDistance3D` (`liquid_remap_3d.cc:81-135`, `liquid_band_handoff_3d.cc:36-95`).
- One cell, one particle or one region change per try chosen by the caller: the twelve `AttemptWithAdaptive...`
  methods, `liquid_adaptive_family_candidate_3d.cc` as a per-request object, `AdaptiveCarrierMode`,
  `LiquidHandoffOccupancy3D` and `LiquidHandoffCarrier3D` as caller-certified inputs.
- The flat-tank assumptions: `surface_height_`, "the core may not reach the open tank surface" as a tank
  rule, `solid_upper`, exterior face records with apertures (`adaptive_liquid_remap_3d.cc:205-400`): open
  areas are found each step by pressure.
- `camera_width` and `physical_width` as inputs of the size rule; `support_certified`. The size rule's curvature
  term (`w_curv`) and coarse surface cells: the plan's later group "Coarser cells on calm surfaces (later)" (A61).
- The 4,096-cell and 24,576-face limits; `AdaptiveLiquidCoreTransportWorkspace3D`; the permutation
  `[0,4,2,6,1,5,3,7]`.
- `graded_liquid_wet_transition_3d.cc` (one particle crossing a size change in a fixture),
  `coarse_fine_transfer_math_3d.h` and `docs/adaptivity/README.md` (the cubic-normal weights: an experiment),
  `docs/adaptivity/matched-error-pressure.md`, `reference-wave-preflight.md` and the run receipts.
- `reference/src/fluidity_reference/adaptive_remap.py`; `proofs/Fluidity/VolumeTransport.lean` (not ported,
  surface, foam and spray design).
- A domain split of the pressure solve (a pressure given at batch borders; rejected in A15 as option (ii)), a
  second GPU, batches not made of consecutive blocks: later.
- The CPU solve fallback inside a GPU run (`SolvePlace::kWholeGridOnCpu`): not built (A15, A17).

## 11. Decisions

Each question keeps its text. "Recommended option" means: recommended option (owner's standing instruction,
2026-11-14). The owner question numbers (Qn) are those of `owner-questions.md`.

Already asked and assumed here: Q5 (`RegionKind`, no `owner`), Q10 (`CommitResize`, `RejectResize`), Q16 / C1
(A; nothing here depends on it), Q18 (a), Q21 (a; 5.10 relies on it), Q22, Q24 (a; it sets the stacked
border), Q39, Q40, Q41, Q42 (a). Q17 / C2 is answered differently in question 4.
Decision: Q5 `RegionKind { kParticleRegion, kGridOnlyRegion }`, `owner` banned as an identifier; Q10
`CommitTry`, `RejectTry` (here `CommitResize`, `RejectResize`); Q18 a; Q21 a; Q22 the cell-place order and a;
Q24 a; Q39, Q40, Q41 as proposed; Q42 a; all recommended option (owner's standing instruction, 2026-11-14).
Q41 (`BuildGrid` on the CPU in a GPU run) is replaced by owner answer A17 (decision 18).
Q16 / C1: viscosity runs the joint solve (`viscosity::SolveViscosityAndPressure`), owner answer A5 and A6.
Q17 / C2: both grids, as question 4 below, `changes-to-apply.md` (03 Particles).

1. **How liquid crosses between the two regions.** `main` moves amounts by the face flow and corrects one
   particle's mass, and rejects a particle that enters a grid-only cell.
   ```cpp
   // a) particles carry it (5.6): a particle in a grid-only cell is taken into the record; an empty cell
   //    eighth beside a grid-only cell above its open volume gets a particle out of the record.
   RemoveParticles(particles_try, material, entered, RemoveReason::kIntoGridOnlyRegion);
   // b) main: record -= flow * duration * density;  last_particle_of_cell.mass += the same
   ```
   b double-counts as soon as particles move with the flow. Recommendation: a. It lets a grid-only cell be
   above or below its open volume by about one particle per face quarter.
   Decision: a, liquid crosses the regions as particles. Source: recommended option (owner's standing
   instruction, 2026-11-14), confirmed in `changes-to-apply.md` (11).
2. **Velocity in the grid-only region between steps.** a) follow the committed velocity back one step
   (`ComputeCarriedVelocity`; the old paper's rule, first order); b) keep the committed value in place
   (what the code on `main` does: deep currents do not move). Recommendation: a.
   Decision: a, grid-only velocities are followed back. Source: recommended option (owner's standing
   instruction, 2026-11-14), confirmed in `changes-to-apply.md` (11).
3. **`grid_velocity` and `liquid_amount` (your Verify items).** Findings in section 9. Options for the
   first: a) `carried_velocity` for the start velocity of a grid-only face, no `grid_velocity` at all;
   b) `grid_velocity` for it. For the second: a) `GridOnlyAmounts { mass; rest_volume; current_volume;
   ingredient_volume; }`; b) `LiquidAmount` with the same fields. Recommendation: a and a.
   Decision: a and a: `carried_velocity`, no `grid_velocity`; `GridOnlyAmounts`. Source: recommended option
   (owner's standing instruction, 2026-11-14).
4. **Conflict C2 / Q17: both grids, as on `main`.** `AdjustParticleCount` runs inside `ResizeCells` and
   takes the committed grid for the old size and the resized grid for the new one (section 4). A particle
   that drifts across a size change is then never split or merged. Option a of Q17 (a flag) has the same
   consequence and needs transfer and a flag bit; option c drops the check. Recommendation: both grids.
   Please confirm the consequence is acceptable; the alternative is a new split rule for incomplete
   neighbourhoods, which is an experiment.
   Decision: both grids; `AdjustParticleCount`, `SplitParticle` and `MergeParticles` take two grids and two
   `NeighbourhoodInputs` and are called only inside the resize transaction; ~~a particle that drifts across an
   existing size change is never split or merged (5.4)~~ **replaced** by decision 50 (A67: such a particle is split
   or merged at the next pass; the both-grid check counts an incomplete split instead of refusing it). Source:
   `changes-to-apply.md` (03 Particles, "Conflict C2 is settled as module 11 designed it"); owner answer A67.
5. **The cell size rule is new.** `main` only decides from wanted widths the caller supplies. Proposed:
   smallest cells within 8 smallest widths of the surface and 2 of a solid, then width proportional to the
   distance; merge after 0.25 s; no camera wish; snow, sand and cut solids keep one size. Alternatives: a
   fixed number of size bands by depth; a camera wish from module 10 through `wants_smallest_cells`.
   Recommendation: as proposed, defaults to be set on the wave scene.
   Decision: as proposed, defaults set on the wave scene; snow, springy solids and cut solids keep one size
   (sand by the same rule in the sand module, A9); `wants_smallest_cells` marks gas cells (A10), no camera
   wish. Source: recommended option (owner's standing instruction, 2026-11-14); A9, A10.
   **Amended by decision 41** (A41, A61 to A63): the rule gains the shear and artist terms, is owned by grid and
   shared by both processors; the hysteresis gains the step count and the held gradient.
6. **The solves are not batched.** Whole grid on the GPU (55 kB per block; 190 kB with viscosity), CPU as
   fallback, no domain split. `pressure::EquationInputs::border_cell_pressure` and `kLiquidToBorderCell`
   are then never used: delete them from the pressure design, or keep them for a later domain split?
   Recommendation: delete. Also confirm the plain statement of 5.9: batched syrup is not supported in
   practice.
   Decision: not batched; both deleted from the pressure design; batched syrup is not practical (5.9).
   Source: recommended option (owner's standing instruction, 2026-11-14), confirmed in `changes-to-apply.md`
   (05, 11).
   **Replaced by decision 13** (owner answer A15): the solves are batched too, in pieces; the CPU fallback is not
   built. The deletion of `border_cell_pressure` and `kLiquidToBorderCell` stands (no pressure is given at a batch
   border in either mode).
7. **The step design must change for both parts** (section 4, step items 1 to 5): seven arrays in
   `GridFields`, two callbacks, more in `ResizeTry`, and for batches a `run_stages` callback, ranges built
   after stage 2, a table of the try's arrays, and sums by block. Alternative for the two callbacks: module
   9 replaces entries of `StageTable`, which needs the same access to `StepTry` and a context pointer.
   Recommendation: callbacks. Also confirm the requirement of 5.10: results never depend on the batch cut
   (the cut depends on free GPU memory).
   Decision: callbacks (`apply_grid_only_region`, `move_grid_only_liquid`, `run_stages`), no separate stage
   value for the grid-only move (it runs at the end of `MoveParticles`); results never depend on the batch
   cut. Source: recommended option (owner's standing instruction, 2026-11-14); `changes-to-apply.md` (10,
   "From cell resize and batches"; every design, sums over ranges).
   Amended by decision 14 (A15): results still never depend on the cut; the one-piece byte identity of 5.10 is
   replaced by agreement within the solver's tolerance.
8. **Folder for Part B.** a) `native/fluidity/batches/`, `fluidity::batches`; b) inside `cell_resize/`, as
   the programme's table lists one module. Recommendation: a (folders by concept).
   Decision: a, folders `cell_resize/` and `batches/`. Source: recommended option (owner's standing
   instruction, 2026-11-14), `changes-to-apply.md` (11).
9. **Region numbers.** Particle region to 6 cell widths (ported), grid-only from 7, overlap 2 layers, 8
   particles per refilled cell, `kMaxFilledShare = 2`, `kAmountTolerance = 1e-12`. A cell with any solid
   part never becomes grid-only (stricter than `main`). Confirm.
   Decision: confirmed as listed; also, a grid-only record holds one liquid material and a cell touching gas
   or another material stays in the particle region (A8, A10). Source: recommended option (owner's standing
   instruction, 2026-11-14).
   **Not in the first version (A64; decision 40):** the region numbers belong to the later group.
10. **Inner faces of a split.** Ported: flows on 7 faces of a tree, 0 on the other 5 (not symmetric). The
    alternative, the smallest flows that give the same imbalances, is new. Recommendation: port; the next
    pressure solve evens it out.
    Decision: ported tree. Source: recommended option (owner's standing instruction, 2026-11-14).
11. **Two sources missing from the old bibliography:** Ando and Batty 2020 (octree liquids) and Losasso,
    Gibou, Fedkiw 2004. Supply the entries, or agree to author and year only.
    Decision: cited by author and year only; the full entries are added if the owner supplies them. Source:
    recommended option (owner's standing instruction, 2026-11-14), the option that needs no entry.
12. **New names to confirm.** `MoveGridOnlyLiquid` and `ApplyGridOnlyRegionToGrid` (stage-like verbs),
    `FindCellsThatMustSplit`, `ResizePlan`, `ResizeMap`, `CellSource`, `CellValueKind`,
    `merge_ready_duration`, `depth_below_surface`, `layers_from_grid_only_region`, `kFromGridOnlyRegion`,
    `StagePlace`, `SolvePlace`, `BatchPlan`, `BatchedRun`, `WriteBatch`, `ReadBatch`,
    `ApplyStagesInBatches`, `ComputeSumOverBlocks`, "stacked border", "whole-grid arrays", "loaded batch".
    Decision: the names as listed (`kFromGridOnlyRegion` is now in transfer's section 3), plus the names of
    this update: `ApplyGridOnlyRegionInStep`, `MoveGridOnlyLiquidInStep`,
    `ApplyResizeToIngredientVolumes`, `ReadLargestGridOnlySpeed`, `kWrongMaterialSet`. Source: recommended
    option (owner's standing instruction, 2026-11-14). The shared `CallbackContext` of the first update is
    gone: each step callback is a `Callback<Function>{function, context}` pair (step decision in
    `changes-to-apply.md`), with `CellResize*` or `BatchedRun*` as context (3.4, 3.7).
   **Amended by decision 40** (A64): `MoveGridOnlyLiquid`, `ApplyGridOnlyRegionToGrid`, their callbacks,
   `ReadLargestGridOnlySpeed`, `kWrongMaterialSet` and `layers_from_grid_only_region` are the later group's.
13. **(A15) The solves are batched too.** Three ways to run, decided per run: `RunMode { kOnePiece, kBatched,
    kBatchedWithSolvesInPieces }` (step); `SolvePlace { kWholeGridOnProcessor, kInPieces }`; a scene far larger than
    GPU memory runs, limited by CPU memory. The solve in pieces is pressure's `SolveBatches`, used by pressure,
    viscosity and gas with no signature change; this module sizes it and reports its copies. `kWholeGridOnCpu` is
    not built (a CPU solve inside a GPU run is against A17). Replaces decision 6 and the old 5.9. Source: owner
    answer A15; `plan-decisions.md` "Batched solves (A15)".
14. **(A15) What a batched run must equal.** Identical bytes for every cut and on every repeat, in each mode;
    agreement within the solver's tolerance with the one-piece run of the same processor; pass 1's local outputs
    still equal the whole-grid bytes. The byte-identity tests between modes are renamed (section 7 table).
    Replaces the 5.10 requirement "a batched run gives the bytes of a one-piece run". Source: owner answer A15;
    `plan-decisions.md` "Batched solves (A15)", "Tests replaced".
15. **(A15) The planner.** Order: whole-grid solves when the whole rows fit `max_whole_grid_share * allowed`; else
    solves in pieces with `max_solve_loaded_blocks = min(solve_piece_share * room / bytes_per_solve_loaded_block,
    max_cell_blocks)`, at least three of the largest layers (else `kBatchDoesNotFit`); `kStateDoesNotFitCpu` when the
    CPU rows exceed `RunPlan::allowed_cpu_bytes`. `BatchSettings::solve_piece_share = 0.25`; `BatchPlan` carries the
    memory budgets and planned copies (A22 "budget"), `BatchReport` the measured solve copies. Source:
    `plan-decisions.md` "Batched solves (A15)", "Planner and report"; the constant is an Open point.
16. **(A15) What lives where with solves in pieces.** State and geometry stay whole on the GPU; solver buffers
    goes to the CPU (`CpuMemory::kLockedForCopies`) and through one loaded solve piece (5.7). Pass 1's geometry
    reaches the whole buffers through `pressure::WriteSnapshotRange`, called by `ReadBatch`. Source:
    `plan-decisions.md` "Batched solves (A15)".
17. **(A17) Borders and pieces on the run's processor.** `FindStackedBorder` (private) with the buffer form of
    `grid::FindBorderBlocks`, one call per layer, two buffers in turn; `BatchedRun` holds the `BorderSearchBuffers`
    and the two buffers; `WriteBatch` builds the piece with the buffer form of `grid::BuildGridPiece` and reads the
    piece map back once per load (4 B per loaded block, row "batches: piece map copy", `where = kCpu`, counted in
    `bytes_from_processor`) to compute the `BlockCopy` runs; a CPU run uses the span. Source: `plan-decisions.md`
    "Batched solves (A15)", "A17 follow-ups".
18. **(A17) The resize on a GPU grid.** In every GPU run (one piece or batched) `ResizeCells` reads one
    `grid::ReadGrid` copy of the committed grid; the resized grid is built on the CPU for the map and, from the
    same list uploaded into a `Buffer<grid::CellId>`, by the step on the GPU; `BuildResizeMap` checks version and
    block count of both. `CellResizeSettings::grid_run_on`, `BuildCellResizeMemoryEstimate(ingredient_count,
    grid_run_on)`, `ResizeCellsReport::bytes_copied`. Replaces the 6 "CPU" text based on grid Q41. Source: owner
    answer A17; `plan-decisions.md` "A17 follow-ups"; plan decision, 2026-11-14 (cell resize plan Tasks 10, 24).
    **Amended by decisions 33 and 34** (reconcile): the CPU grids follow `GridCopies`; the step builds the GPU grid
    with `step::BuildResizedGrid`.
    **Amended by decision 41** (A41): the resize also reads back the `size_change` bytes (1 B per cell place, counted).
19. **CPU mirrors.** In a batched (or any) GPU run a `CellResize` with `run_on = kCpu` keeps CPU mirrors of the
    arrays it reads, `where = kCpu`, the same in both batched modes. Source: `plan-decisions.md` "Cell resize and
    batches" and "A17 follow-ups".
20. **Stage places.** Stage 10 runs on the whole grid; the step's `StageTable::run_whole_grid` holds the whole-grid
    parts of stages 9, 16 and 19 (5.8). If the guard test on stage order fails, or a stage rejects on incomplete
    inputs in the outer border blocks of a piece, the implementer stops and reports. Source: `plan-decisions.md`
    "Cell resize and batches"; plan decision, 2026-11-14.
21. **The loaded try belongs to the step.** `FindLoadedTry`, `FindLoadedGridBuffers`, `ApplyGridPiece`,
    `BlockCopy`, `WriteStepArrayBlocks` (section 4, step item 7); the loaded batch is built by
    `step::BuildSimulation`; `BuildBatchesMemoryEstimate` is for reporting only. Source: `plan-decisions.md` "Cell
    resize and batches".
22. **The plan cuts by layers.** `BuildBatchPlan` cuts by layers of blocks without building the start grid; the
    exact cut is found per try inside `ApplyStagesInBatches` by bisection. Replaces the 5.9 text "found by
    bisection on the CPU with the no-allocation `FindBorderBlocks`" on the start grid. Source: `plan-decisions.md`
    "Cell resize and batches".
23. **Resize details.** `ApplyResizeToCellAmounts` remaps the three grid-only amounts with one limiter per split
    cell; `merge_ready_duration` grows by the committed time since the last committed merge test, and a change of
    durations alone commits a same-cell resize with version + 1; `ApplyGridOnlyRegionToGrid` writes `velocity` =
    the carried velocity and `momentum = mass * velocity`; `ReadLargestGridOnlySpeed` returns the latest try's
    value and the step reads it only in `CommitStep`. Source: `plan-decisions.md` "Cell resize and batches".
24. **Test support.** Step test scenes added by this module (`BuildDeepTankScene`, `BuildMixedSizeColumnScene`,
    `BuildRisingFloorScene`, `BuildStandingWaveTankScene`, `BuildTwoLiquidsDeepScene`,
    `BuildSmokeOverDeepWaterScene`, `BuildFloatingCubeScene`); test library `fluidity_cell_resize_test_support`.
    Source: `plan-decisions.md` "Cell resize and batches".
25. **Method paper.** `<proc-apply-stages-in-batches>`; outline item 9 gives the three ways to run (8). Source:
    plan decision, 2026-11-14 (plan Task 34).
26. **Placement of every buffer of a batched run.** The placement rule of 5.7 (processor when a
    `kWholeGridOnProcessor` part touches it; else CPU memory when a `kWholeStateOnCpu` stage touches it, the commit
    keeps it or a later pass reads it; else loaded only); `kStagePlaces` is checked against the step's
    `StepArray::whole_place`. This module adds no buffer or copy the step plan's Appendix A does not list. Source:
    plan decision, 2026-11-14 (buffers in batched runs); owner principle A17.
27. **The planner on the step's batched rows.** `BuildBatchPlan` sums the rows of `step::BuildStepMemoryEstimate(scene,
    counts, &probe_plan)` (loaded module instances, mirrors, CPU grid copy, stage-17 `SolidsBuffers`, `MarkerChanges`
    rows), so the budgets shown before the run equal the allocation. Source: plan decision, 2026-11-14 (buffers in
    batched runs; planned cost); owner principle A17.
28. **Stage 17's contact partials.** `ReadBatch` calls `solids::WriteBlockPartialsRange` after pass 1 into
    `*step::FindContactReactionBuffers(simulation)`; its bytes count in `bytes_from_processor` as part of the load.
    Source: plan decision, 2026-11-14 (buffers in batched runs); owner principle A17.
29. **Stage 3's whole part.** `FindGridEdgesOpenToSky` moves to `StageTable::run_whole_grid[kFindSolidSurfaces]`,
    run at `kWholeGridOnProcessor` right after pass 1, before stage 9's whole part. Amends decision 20 (whole-grid
    parts of stages 3, 9, 16, 19). Source: plan decision, 2026-11-14 (buffers in batched runs); owner principle A17.
30. **The piece map for foam and spray.** `step::ApplyGridPiece(StepTry&, const grid::GridPiece&,
    std::span<const grid::CellBlockIndex> block_in_whole_grid)` replaces the two-argument form of decision 21;
    `WriteBatch` passes its map copy, which stage 18 hands to `make_foam_and_spray`. Source: plan decision,
    2026-11-14 (buffers in batched runs); owner principle A17.
31. **The resize and `ChooseCellSize` on a CPU copy.** In a batched GPU run `ResizeCells` reads
    `step::FindCpuGridCopy` instead of its own `ReadGrid` (its row stays, idle); `ChooseCellSize` with null committed
    tables probes the copy it holds (or the step's), making one counted `ReadGrid` when it holds none.
    `ResizeCellsReport::copy_count` counts every `ReadGrid`, `ReadBuffer` and `WriteBuffer` call between CPU memory
    and the grid's processor. Amends decision 18. Source: plan decision, 2026-11-14 (buffers in batched runs;
    planned cost); owner principle A17.
    **Replaced in part by decisions 33 and 35** (reconcile): "its row stays, idle" is withdrawn (with `kHeldByStep`
    no CPU grid of `CellResize` is built or counted); in a batched GPU run the step passes the CPU copy's real tables,
    so `ChooseCellSize` gets null tables only in a one-piece GPU run.
    **Replaced in part by decision 41** (A41): the `ChooseCellSize` half is withdrawn (no callback).
32. **Copies shown to the user.** I/O's budget reads the `BatchPlan` fields and the `BatchesFailure`; its measured
    cost reads `BatchReport` and the resize's copies (section 4, I/O). Source: plan decision, 2026-11-14 (planned
    cost).
33. **(Reconcile) Who holds the CPU grids of a resize.** `enum class GridCopies { kNone, kHeldByCellResize,
    kHeldByStep }`, `CellResizeSettings::grid_copies` (set by I/O from the run plan's mode; mismatch with
    `grid_run_on` is `kBadInput`), `BuildCellResizeMemoryEstimate(ingredient_count, grid_run_on, grid_copies)`. One
    piece on the GPU: two CPU `GridBuffers` sharing one CPU `grid::GridBuilderBuffers` (`grid::BuildGridMemoryEstimate(kCpu,
    2, kOneForAllGrids)`, `where = kCpu`). Batched GPU run: neither is allocated or counted; the committed grid is
    `step::FindCpuGridCopy`, the CPU resized grid is built into `ResizeTry::free_cpu_grid_copy_buffers()` and handed to
    the step with `step::ApplyResizedCpuGridCopy`, so the step's CPU copy needs no `ReadGrid` refresh after a committed
    resize (step Appendix B row B5 withdrawn). Decision: as listed. Source: `plan-decisions.md`, Reconcile pass,
    coordinator items (5), (6); cell resize plan Tasks 10, 24, 43.
34. **(Reconcile) How the step receives the uploaded resized list.** `step::ResizedCells` and
    `step::BuildResizedGrid(ResizeTry&, const ResizedCells&)` (builds into `free_grid_storage()` on the run's
    processor, binds the try as `ApplyResizedGrid`, checks version and block count against `cpu_resized_grid` in a GPU
    run); `ResizeCells` calls it on a GPU grid; `ApplyResizedGrid(ResizeTry&, const grid::StaggeredGrid&)` stays for a
    caller that built the grid itself (a CPU grid). Closes the former open point 4. Decision: as listed. Source:
    `plan-decisions.md`, Reconcile pass item 12 and coordinator item (1); step plan Task 32; cell resize plan Task 24.
35. **(Reconcile) `choose_cell_size`'s `committed` argument.** The step passes `committed_grid.tables(RunOn::kCpu)`:
    real tables in a CPU run; version 0 and null arrays in a one-piece GPU run (then `ChooseCellSize` reads its own
    CPU copy); `FindCpuGridCopy(...)->tables(kCpu)` in a batched GPU run. The one-piece copies stay an owner question
    (Open 2). Decision: as listed. Source: `plan-decisions.md`, Reconcile pass item 13 and coordinator item (2).
    **Replaced** by decision 41 (A41): `choose_cell_size` is removed; stage 2 calls `grid::ComputeNewCellSizeLevel` on
    the grid's processor, with no copy.
36. **(Reconcile) Grid buffers sized by the planner, one builder buffers.** The grid has no default room
    (`GridSettings::max_cell_blocks` 0 is `kBadInput`); the loaded try's grid buffers borrow the step's
    `grid::GridBuilderBuffers`, so `bytes_per_loaded_block` and `BuildBatchesMemoryEstimate` count
    `grid::BuildGridMemoryEstimate(run_on, 1, limits.max_cell_blocks, kOneForAllGrids)` without its `builder buffers:` entries. Decision: as
    listed. Source: `plan-decisions.md`, Reconcile pass item 11; cell resize plan Task 38.
37. **(A29) `Take`, not `Release`.** `grid::ReleaseBuffers` is renamed `grid::TakeBuffers` (for a grid and a
    piece): it gets the preallocated buffers back out of a grid no longer needed. Decision: as listed. Source:
    `owner-answers.md` A29.
38. **(A30) Buffers on different processors.** A precondition checked by a debug assertion, not an error:
    `BufferError::kRunOnMismatch` is gone, `ToCellResizeFailure` maps nothing for it, and `ComputeLiquidTotals` /
    `ComputeSumOverBlocks` state `Requires:` same processor. Decision: as listed. Source: `owner-answers.md` A30.
39. **(A31) No "scratch"; one form per function.** Reusable working memory is "working buffers" (`SumBuffers`,
    `OrderBuffers`, `grid::GridBuilderBuffers`). `ComputeLiquidTotals` and `ComputeSumOverBlocks` each have one
    form, which takes a `SumBuffers&` (the forms that built their own are removed; tests build their buffers).
    Decision: as listed. Source: `owner-answers.md` A31; cell resize plan Tasks 23, 37.

40. **(A64) Particles everywhere in the liquid.** First version: every liquid cell holds particles at every depth
    and cell size; no grid-only region, record, velocity blend, carried velocity or region transfer. A cell of level
    `l` holds particles of level `l`; `AdjustParticleCount` merges and splits them with their cells, one level per
    resize (5.3 "Particles in deep cells", with the cost bound). The grid-only design is kept, marked "not in the
    first version (A64)", as the plan's later group "Deep liquid without particles (later; needs review)": an
    independent review first, then the gating tests `GridOnlyRegion, StandingWave_KeepsAmplitude` and
    `GridOnlyRegion, ParticleBandEdge_StaysStable`. `regions.h` is split into `depth_below_surface.h` and
    `liquid_totals.h` (plan decision); `ComputeLiquidTotals` sums particles only; `ResizeCellsReport::totals_before`
    and `totals_after` replace the ledger's totals (plan decision); `CellResizeError` drops `kWrongMaterialSet`,
    `kGridOnlyCellDrained`, `kGridOnlyCellTooFull` (the later group appends them); `max_depth =
    surface_reach_in_cell_widths x w_max`. Source: owner answer A64; `plan-decisions.md` "Owner answers A40–A64
    applied" item 1.
41. **(A41, A61, A62, A63; A52 refined) The shared cell size rule.** Grid owns `cell_size_rule.h` (wanted width with
    depth, solid, shear terms and the refinement factor (A78, A79), reference speed, new-cell level); this module owns
    `ComputeVelocityGradientSize`, `ComputeCellSizeChange` (with `kMergeReady` and, plan decision, `kKeepsItsSize`, so
    the GPU run reads back one byte per cell) and the two-part pass `ComputeCellSizeChanges` with the hysteresis:
    split at once; merge after `2 w <= 0.8 x wanted` held for 0.25 s and 8 committed steps; never within 8 `w_min` of
    the surface or touching air, a pocket, gas or a solid; one level per resize; a held gradient. The surface always
    has the smallest cells (A61). `ChooseCellSize` and its copies are removed; `GridCopies` stays for the resize map.
    Remap of the held gradient: a split copies it (plan-decisions' explicit remap rule; the safe side, it only
    delays a merge), a merge gives 0. The off-diagonal gradient beside a bigger neighbour is first order (plan
    decision; the exactness claim of `plan-decisions.md` holds for same-size and smaller neighbours). Source: owner
    answers A41, A61, A62, A63; `plan-decisions.md` "Owner answers A40–A64 applied" item 2.
42. **(A61) Coarser cells on calm surfaces.** The curvature term and coarse surface cells are the plan's later
    group "Coarser cells on calm surfaces (later)", with the T-junction-aware surface pressure condition (Ando and
    Batty 2020, Eq. 25) and bump-free resampling (Ando 2013) first. Source: owner answer A61; `plan-decisions.md`
    item 2, "Later group".
43. **(A58) One reserved block per processor.** `BuildCellResize` and `BuildBatchedRun` take `RunMemory&
    run_memory, RunMemory& cpu_run_memory` last; every `kRunOn` row is `kWholeRun`; one entry per buffer; the
    estimate tests become `..., IntoRunMemory_HandsOutExactlyTheEstimate`. `BuildBatchesMemoryEstimate` lists only
    what `BuildBatchedRun` builds (plan decision: the loaded rows it listed for reporting are the step's batched
    rows and are not repeated); `BatchPlan::bytes_on_processor` is `ComputeRunMemoryBytes(..., kRunOn)`. Source:
    owner answer A58; `plan-decisions.md` item 3 and its correction.
44. **(A63) Artist inputs. REPLACED by decision 52 (A78, A79).** `BuildCellResize` took `const grid::ArtistCellSizes& artist` (CPU views from I/O's
    `RunScene::artist_cell_sizes()`) and copies regions and map once into two rows on its processor. Source: owner
    answer A63; `plan-decisions.md` item 2, "Artist inputs"; I/O plan (signature agreed with the I/O plan).
45. **(Reconcile item 31) `kOutOfMemory` last.** `CellResizeError::kOutOfMemory` (value 9) and
    `BatchesError::kOutOfMemory` (value 7) are the last values, returned only by the builders, and what
    `BufferError::kRunMemoryTooSmall` maps to. Source: `plan-decisions.md` item 11.
46. **(Reconcile item 32) Span forms on a GPU grid.** `FindCellsThatMustSplit`, `SplitCell`, `MergeCells`, the
    resize plan and `BuildResizeMap` called with tables of a GPU grid break a `Requires:` (debug assertion), not
    `kBadInput`. Source: `plan-decisions.md` item 11 (A30).
47. **(A64, A41) Batches.** `StageTable::run_whole_grid[kMoveParticles]` is null in the first version (stage
    places 3, 9, 19 only); the batched mirrors hold the three wait arrays; the deep-tank batched test runs with
    particles everywhere. Source: owner answers A41, A64; `plan-decisions.md` items 1 and 2.
48. **(A65) Particle size follows the cell.** One wording everywhere: every liquid cell holds about eight particles
    of its own size: a particle's `size_level` is its cell's level, its rest volume is one eighth of its cell
    (`(w/2)³` for a cell of width `w`), so a cell k levels coarser than the smallest holds 8 particles each of `8^k` times the smallest
    particle's volume. Not one particle size everywhere. The batch plan's 512 particles per full block holds at every
    level (5.9 item 2); the deep-tank tests and 5.3, 5.4 use this wording. Decision: option B of
    `tradeoff-particles.md`. Source: owner answer A65; `plan-decisions.md` "Owner answers A65–A68 applied", A65.
49. **(A66) A run starts on the adaptive grid.** The run's first grid is `step::BuildStartCells`' (shapes, the
    shared size rule with a Euclidean depth minus half a cell diagonal, the one-size-step closure), seeded 8 per cell
    at each cell's level by `io::BuildStartParticles`; the batch plan's start blocks are `SceneCounts::
    start_cell_blocks`. The first resizes may coarsen further, never refine for depth (test
    `CalmDeepTankOnStartCells_NeverSplitsForDepth`). The calm-tank and calm-pool tests keep an all-smallest start on
    purpose, because they test the coarsening cascade and its time; they say so. Source: owner answer A66;
    `plan-decisions.md` A66.
50. **(A67) Mismatched particles at every pass.** `ResizeCells` calls the extended `particles::AdjustParticleCount` at
    every pass over all blocks; the "nothing to do" check (5.1 step 3) runs it on the committed grid as both grids
    and is "nothing to do" only when it finds no mismatched particle (plan decision: the count is the pass's own, no
    extra pass or copy; requires `AdjustParticleCount` to accept the same grid twice, a read-only aliasing); a
    particles-only change commits a same-cell resize with `version + 1`. The both-grid check stays only for changed
    cells and counts an incomplete split instead of refusing it. The statement "a particle that drifts across an
    existing size change ... keeps its `size_level` for good" is **replaced**. Report: the new
    `AdjustParticleCountResult` fields in `ResizeCellsReport::particle_count`; `kinetic_energy_change` and
    `mismatched_particle_count` copied into the new `step::ResizeResult::size_change_kinetic_energy` and
    `mismatched_particle_count` (plan decision: the step reads only `ResizeResult` from the callback), which the step
    copies into `StepReport`. Source: owner answer A67; `plan-decisions.md` A67; `tradeoff-particles.md` 2.2.
51. **(A68) Order of the later work; slab batches in the first version.** Later groups in this order: (1) "Can the
    fine surface band shrink from 8 cells to 4? (later, test only)", (2) "Coarser cells on calm surfaces (later)"
    with "Batches cut by columns or tiles (later)" (sketch in 5.11, no first-version task), (3) "Deep liquid without
    particles (later; needs review)". The first version keeps slab batches and the pinned surface; the batch plan
    exposes `SlabLimit` (blocks per layer, loaded layers, bytes per full loaded block, room, needed bytes, plan area,
    largest plan area) on success and in the `kBatchDoesNotFit` failure, so I/O's budget line names the plan-area
    limit. Source: owner answer A68; `plan-decisions.md` A68; `tradeoff-particles.md` 1.6, 1.7, 5.

52. **(A78, A79; reshapes A72, A73, A77) The refinement volume.** The artist regions and the map are removed. The
    rule's artist term is a cell size factor read from a keyed volume (`grid::RefinementFactors`, a min-pyramid of
    one-byte codes `f = 2^(c/4)`), multiplied into the physical width; `wanted = w_min` stays in the pinned surface
    band. `BuildCellResize` takes the volume's layout and builds one buffer "cell resize: refinement factors" (1 B per
    pyramid code, `kRunOn`, `kWholeRun`); `WriteRefinementFactors(std::span<const std::int8_t>, CellResize&)` (NEW)
    copies a key into it, called by the step when the key at or before the committed time changes (one planned,
    counted copy per key in a GPU run, `ResizeCellsReport::refinement_copy_bytes`, NEW). A key change splits at once
    and merges through the unchanged hysteresis. Zones, meshes and the camera are built outside the solver by I/O.
    Limit of the first version: the factor does not reach the pinned band, so it cannot make a hero surface finer
    than `w_min` nor coarsen any surface. Product, not harmonic mean (the harmonic mean cannot say "as the physics
    wants" and bounds the result between the two widths). Source: A72, A73, A77, A78, A79; `plan-decisions.md`
    "Refinement volume (A78, A79; reshapes A72, A73, A77)".

**Open**

1. `BatchSettings::solve_piece_share = 0.25` and the batched direct-solve room `4 * max_cells` (pressure's) are new
   constants for the owner to confirm.
2. One-piece GPU `choose_cell_size`: the argument is decided (decision 35), and the one-piece GPU run pays counted
   copies (new cells read back, levels sent, a `ReadGrid` when `CellResize` holds no current copy). Open for the
   owner: a shared CPU-and-GPU size rule instead of the callback would remove these copies (`plan-decisions.md`
   "Planned cost shown to the user", open point 2). **Settled** (A41, decision 41): the shared rule replaces the callback; no copies.
3. In a batched GPU run, stage 2 (`ChooseActiveCells`) runs at `kWholeStateOnCpu` while A17 has the step build
   the grid on the GPU from a `Buffer<CellId>` there; how the CPU-made list reaches the GPU builder (an upload like
   the resize's, counted) is not stated in the plans. **Settled** (plan decision, 2026-11-14, buffers in batched
   runs): uploaded into `"step: active cell list"`, one counted copy; section 4, step item 8.
4. The I/O plan must print the planned budgets and copies (`BatchPlan` fields) and the measured ones
   (`BatchReport`); it is not amended yet. **Settled** (decision 32; I/O plan Tasks 23a, 29a).
5. If a scene becomes GPU-limited with solves in pieces, the geometry is the next row to stream, which needs
   `solids::BuildMotionMap`, `BuildBlockingMaterialFaces` and `CheckSolidExchange` to run batch by batch
   (`plan-decisions.md` open point 3).
6. Stage 3's range part calls `solids::BuildTry` and the contact checks once per try; in a batched run it runs per
   batch. The step builds the solids try only in the first call of a try (step plan Task 33a checks it); the guard
   test on stage order covers it. If the owner prefers, those calls move to stage 2 (`plan-decisions.md` "Planned
   cost shown to the user", open point 3).
7. (A58) After `step::ApplyLimits` builds new blocks, the `CellResize` and `BatchedRun` built into the old ones must
   be built again into the new ones before the next call, and `CellResize` must carry the time and step count of its
   last committed merge test across that rebuild. The carrying function (or a `BuildCellResize` argument) is for
   the step and I/O plans to name; owner to note. **Decided (reconcile, 2026-11-15):** the step's new callback
   `StepCallbacks::limits_raised` (step design 3.9, 5.7 step 3a) is called by `ApplyLimits` after the new blocks are
   reserved and before the old ones are freed; I/O's handler builds a new `CellResize` and `BatchedRun` into
   `step::FindRunMemory` / `FindCpuRunMemory` at the new limits and moves the old `CellResize`'s last report into it.
   The merge wait state needs no carrying here: `merge_ready_duration`, `merge_ready_step_count` and
   `held_velocity_gradient_size` live in the step's committed `GridFields`, which `ApplyLimits` copies.
8. (A62) `shear_width_factor = 0.25` and the jet test's expectation are starting values; the jet and flicker tests
   report calibration findings to the owner instead of changing defaults.
9. (A64) The gating tolerance 5 % of `GridOnlyRegion, StandingWave_KeepsAmplitude` is a starting value.
10. (A67) `kMaxMergeWaitPasses = 16` and a lone particle persisting without a time bound (`plan-decisions.md` owner
    question 1 of A65–A68); splits no longer refused for an incomplete neighbourhood (question 2).
11. (A68) The surface-reach decision rule's 1.2 x factor, and the tile side rule of 5.11 (2), are starting values
    for the later groups.
