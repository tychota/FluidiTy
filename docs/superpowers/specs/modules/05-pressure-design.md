# Pressure module design

Status: decided. Module 4 of the rebuild. Sections 3 and 4 were aligned with the other module designs on
2026-11-14 and then updated with `owner-answers.md` and `changes-to-apply.md` of the same day; names shared
between modules are fixed in `shared-interface.md`. Brought in line on 2026-11-14 with the decisions recorded
in `plan-decisions.md` and the pressure plan (A15 batched solves, A17, A18 coupling inside the solve, and the
plan decisions); section 11 lists each one with its source. Updated again on 2026-11-14 for owner answer A24
(the pieces of viscosity's joint solve carry blocking-material faces and sealed gas blobs; decision 16), and for
owner answers A26 and A27 (in the joint solve material face velocities and free-solid motions are velocity
unknowns, through the `CoupledVelocities` forms; decision 17).

## 1. Purpose

A liquid cannot be squeezed, so in every cell the liquid flowing in must equal the liquid flowing out. After
forces have changed the grid velocity this is no longer true; the module finds one pressure number per cell
whose push on the faces restores the balance, and applies that push to the velocity. To write those balance
equations it first decides where the liquid is: which cells hold liquid, how much of each face is blocked by a
solid, and how far the liquid surface is from each cell.

**Owns:** `FindLiquidSurface` and the `GeometrySnapshot` it returns (one per try, read by pressure, viscosity,
solid coupling, gas and checks), including the cell-eighth volumes and liquid-covered centres viscosity reads;
one distance field per liquid material and the boundary between two liquids that do not mix;
`PressureEquations`; `SolvePressure` on CPU and GPU, for scenes with no viscous ingredient; the pieces the
joint solve calls (face inertia, push, correction, volume error, own weight, ladder, the stop rule
`CheckFlowImbalance`); surface tension as a pressure jump; `PressureLaw`; air pockets; the open-sea side and
outlet; the shared clipping formula of a blocked face piece. NEW (A18): the inertia of blocking materials
(`BlockingMaterialFace`) and the springiness of sealed tracked-gas blobs (`SealedGasBlob`) inside the liquid
pressure solve, with the solved material velocities and blob pressures. NEW (A15): `solve_batches.h`, the shared
mechanism by which the pressure, viscosity and gas solves run batch by batch with their working buffers in CPU memory.

**Does not own:** the grid and face pieces (grid); `velocity_before_pressure`, per-cell mixture density and
sensed volume ratio (transfer); solid shapes, `distance_to_solid`, solid velocity and free-solid motion
(contact, solids); the blocking-material face list and the write-back of the solved material velocities
(solids `BuildBlockingMaterialFaces`, `ApplyPressureToMaterials`); the joint velocity-and-pressure solve of
viscous scenes, `viscosity::SolveViscosityAndPressure` (viscosity, owner answer A5); gas state, the sealed-blob
list (gas `ApplyGasStepStart`), its added volume and its pressure (gas); the render mesh (surface), which this
module never reads; the committed pressure and air-pocket arrays and when a step is retried or committed
(step); how a run is cut and how much memory a solve piece gets (batches `BuildBatchPlan`).

## 2. What exists today

Proof of concept, branch `main`. Y-up. Paths are under `native/` unless stated.

| Piece | Where | What it does |
|---|---|---|
| Public solve inputs | `include/fluidity/host/mac_pressure_3d.h:14-102` | Cells, faces with a variant link (internal with jump, Dirichlet, sealed), limits 1 to 4096 cells and 24576 faces |
| Coefficients | `src/mac_pressure_3d_arithmetic.h:110-146`, `174-245` | `compliance = V / (K dt)`; `conductance = open_area * dt / (rho * distance)`; wall flux `= blocked_area * wall velocity` |
| Right-hand side | `mac_pressure_3d_arithmetic.h:263-313` | `V*s + inward wall flux + C*p_previous - sum(sign * (a*u + wall flux)) - sum(sign * k * J) + sum(k * p_exterior)` |
| Components and pin | `mac_pressure_3d_arithmetic.h:326-399` | Union-find over positive links; a component is anchored by compliance or a Dirichlet face; otherwise its balance must vanish within 32 epsilon of its terms and its lowest cell is pinned to 0 Pa |
| Operator, correction | `mac_pressure_3d_arithmetic.h:404-436`, `440-503` | `A p`; `u = u* + (dt/(rho d)) * (p_tail - p_head + J)`; the defect is summed from corrected fluxes without reading the right-hand side |
| Face-piece assembly | `src/mac_pressure_3d_system.cc:279-397`, `399-473` | "Explicit subface" graph and the mixed-size ("adaptive") wrapper; uniform wrapper at `192-252` |
| Evaluation | `mac_pressure_3d_system.cc:475-540` | True residual, physical defect, identity check at 64 epsilon |
| Multilevel hierarchy | `src/mac_pressure_3d_multigrid.cc:102-165`, `167-214`, `218-301` | Piecewise-constant aggregation by `floor(coord / 2)` inside each connected block, coarse coefficients by summing crossing conductances, dense Cholesky of terminal blocks of at most 32 unknowns. A second, allocation-free copy lives in `mac_pressure_3d_bounded_hierarchy.cc` |
| CPU solve | `src/mac_pressure_workspace_3d.cc:462-659` | Conjugate gradient preconditioned by one V-cycle; FP32 fine vectors, FP64 coarse levels and dot products; restart from the FP64 true residual |
| GPU solve | `cuda/mac_pressure_3d.cu:79-586`, launch `607-624`; host `cuda/liquid_pressure_stage_3d_host.cc:206-357` | Same algorithm in one block of 128 threads (stand-alone: one thread). Host builds and uploads the hierarchy, runs batches of 8 iterations, downloads a state record |
| Stopping rule | `mac_pressure_3d_multigrid.cc:341-357` | `norm2(r) <= 2e-6 m3/s + 2e-6 norm2(b)`, and `max |r|/V` and `max |defect|` `<= 2e-5 1/s`; capped at `1e-3 norm2(b)` when an internal jump exists |
| Liquid geometry | `include/fluidity/host/liquid_boundary_geometry_3d.h:93-144`, `268-294`; `src/liquid_boundary_geometry_3d.cc:765-1072` (`FreezeCore`), `1077-1286` (`LinksCore`) | Liquid cell = centre distance negative and clipped volume positive (`873-893`); faces clipped against one analytic obstacle; air ends become Dirichlet |
| Shared geometry math | `include/fluidity/base/liquid_boundary_geometry_math_3d.h:652-655`, `667-689`, `690-729`, `767-847` | Six tetrahedra sharing the body diagonal; `theta = phi_liquid / (phi_liquid - phi_air)` clamped to at least 0.05; exterior pressure `= gas gauge + sigma * kappa`; curvature by centred differences on 19 centre samples; capillary step bound |
| Mixed-size geometry | `src/graded_liquid_boundary_geometry_3d.cc:138-345`, `485-595`, `645-725` | Leaf and subface measures for a planar wall under a horizontal liquid plane only |
| Air pockets | `src/liquid_boundary_geometry_3d.cc:1385-1703`, `1831-1980`; `src/liquid_air_components.cc:36-122`, `124-201` | Gas volume and gas face areas per cell; breadth-first labelling into enclosed, unclassified, vented; overlap between two steps |
| Pocket pressure | `src/liquid_cavity.cc:14`, `24-41`, `45-80`, `82-102`, `104-151`, `153-304` | `p_abs * V^1.4` constant; additive `V0 * p0^(1/1.4)` at splits and merges; step limit `0.5 / omega`; resolved when equivalent radius is at least 2 cell widths |
| Pressure law | `src/liquid_eos.cc:27-85`; `src/liquid_eos_sensor_3d.cc:165-212`, `437-456`, `458-571` | Hencky and Tait (exponent 7) with tangent bulk modulus; volume-ratio sensor; clamped drift rate; zero-sum balance for a sealed component |
| Two liquids | `src/liquid_eos_sensor_3d.cc:213-436`; `liquid_boundary_geometry_3d.cc:1175-1235` | Density by cell from particle mass and volume; series density and one signed jump on a face crossed by the interface of one configured pair |
| Open sea | `src/ocean_boundary_3d.cc:14-140`, `142-156`, `184-221`; called at `src/liquid_substep_3d.cc:1248-1273` | Second-order Stokes wave target; face velocity relaxed toward it (inlet) or toward rest (outlet) with weight `1 - exp(-ramp * rate * dt)` |
| Coupling with viscosity | `src/mac_viscous_pressure_3d.cc:82-134`, `151-249` | Pressure operator `D H^-1 D^T + C` with a velocity inverse supplied as a callback; preconditioner conductance scaled by `inertia / diag(H)` |
| Reference, proofs | `reference/src/fluidity_reference/pressure.py:53-118` (2D only), `cavity.py`, `eos.py`; `proofs/Fluidity/Pressure.lean`, `PressureCorrection.lean`, `PressureTopology.lean` | Dense 2D solve; positive semidefinite form; defect equals minus residual; kernel is the constants of unanchored components and solvable iff compatible |
| Old paper | `docs/papers/fluidity-method.typ:174-217`, `236-249`, `298-309` | Sections "Pressure projection", "Surface tension", "Weak compressibility" |

What is wrong:

1. **One block.** Every phase runs in one block; reductions, restriction, vector updates and checks run in one
   thread of it (`mac_pressure_3d.cu:184-194`, `291-297`, `395-402`; report CUDA 7.1). More cells cannot use
   more of the device.
2. **Host between dependent stages.** The hierarchy is built on the host from host copies of the equations and
   uploaded for every solve (`liquid_pressure_stage_3d_host.cc:144-205`, `301-306`); air pockets and geometry
   classification run on the host. This is a programme anti-goal.
3. **Hard limits** of 4096 cells and 24576 faces (`mac_pressure_3d_arithmetic.h:19-20`), and a device check
   `compiled_architecture != 1200` (`liquid_pressure_stage_3d_host.cc:335`).
4. **A stopping rule that depends on scene size.** `2e-6 m3/s` is compared with a norm over all cells, the
   local limits are rates in `1/s` (report numerics 3.1 and 4.5, validation "Convert pressure residual").
5. **An arbitrary pin.** A sealed component is pinned at its lowest cell, and the warm start subtracts the
   pin's previous pressure from the whole component (`mac_pressure_3d_multigrid.cc:303-321`): the constant in
   the pressure depends on cell order. Incompatible volume is refused only at round-off level and its size is
   not reported.
6. **Geometry is not one snapshot for all materials.** One analytic obstacle variant per scene
   (`liquid_boundary_geometry_3d.h:42-43`), separate floating-box paths (`liquid_boundary_geometry_3d.cc:305-764`),
   a wetted-piston override (`1158-1173`), mixed-size geometry only for planes, a `bool bounded` flag through
   every function (naming review A4).
7. **Exactness machinery** the owner removed: `IsRepresentablePressureProduct`, compensated sums, the 64 epsilon
   identity (`mac_pressure_3d_arithmetic.h:33-84`; naming review A9).
8. **Two meanings in one word.** `LiquidAirPassage3D::kOpen` means both "leads to a known air cell" and "leads
   to an unknown frontier" (`liquid_air_components.h:13-17`).

**Time marks, verified** (naming review 4.5). No identifier in the pressure files starts with `old_` or `new_`;
the marks live at the call site, `src/liquid_substep_3d.cc`:

- `staging.old_velocities` (`:1227`) is the grid velocity straight after particles-to-grid: **before forces**,
  not before pressure. It is never an input of this module.
- `staging.new_velocities` has two meanings: from `:1245` to `:2340` it is the velocity after forces and
  open-sea relaxation, that is `velocity_before_pressure`; from `:2402` on the same array holds
  `velocity_after_pressure`. `staging.prepressure_velocities` (`:2340`) is an explicit copy of the first meaning.
- `old_time`, `new_time` (`:1134-1135`) are the start and end time of the step. Geometry is evaluated at the
  start (`liquid_boundary_geometry_3d.h:103`).
- `previous_pressure` (`mac_pressure_3d.h:19`) is the last committed pressure, or the advected pressure history
  when a pressure law is active (`liquid_substep_3d.cc:1450`, `1466-1476`). New name: `carried_pressure`.
  Checked on `main` during the reconciliation: the history is moved on the grid, not by particles
  (`AdvectLiquidPressureHistory3D`, `liquid_substep_3d_staging.cc:77`, follows the committed face velocity
  back from the cell and reads the committed cell pressures). It becomes `ComputeCarriedPressure` here.

**Y-up specifics.** The hydrostatic correction is applied to `FaceNormalAxis::kY` faces
(`liquid_substep_3d.cc:2343`); the planar liquid and the wave use `world_surface_height_y`, "World Y"
(`liquid_boundary_geometry_3d.h:114`, `119`); the open-sea zones lie along X with a transverse `inlet_low_z_m`.
In Z-up the vertical is Z, the transverse open-sea axis is Y, the up direction is `kUp`. The `(z,y,x)` sort order
and the corner-bit order are index conventions and do not change meaning. Comparison data from `main` is rotated
by the quarter turn about X of the paper's `<fig-axes>`.

## 3. Public interface

Folder `native/fluidity/pressure/`, namespace `fluidity::pressure`. Each operation header has private
`_cpu.cc` and `_gpu.cu` files beside it. Headers marked (shared) compile for CPU and GPU. All names are NEW
unless marked DECIDED. Arrays are `grid::CellValues<T>`, `grid::FaceValues<T>` (one value per face place;
this design first wrote `FacePieceValues`) and `grid::CornerValues<T>` (grid 3.6): they live on the processor
chosen by `RunOn`. Indexes are `grid::CellPlaceIndex` and `grid::FacePlaceIndex`, valid for one grid version
only; "none" is `grid::kNoIndex`. Every fallible function returns `std::expected<T, PressureFailure>` (owner
answer A2). Air pockets, material boundaries, equations, the ladder, the solves and the check take the whole
grid (`blocks` = all blocks). NEW (A15): buffers built with `BufferSizes::max_loaded_blocks > 0` (the batched
layout) keep their per-place rows in CPU memory, and every function taking `PressureBuffers&` then runs batch by
batch through `SolveBatches` (`solve_batches.h`); no signature changes, and the result agrees with the one-piece
run within the solver's tolerance (section 5 F).

```cpp
// pressure_error.h (shared)
enum class PressureError {
  kNone,
  kBadInput,                 // A value is not finite or out of range, or an array has the wrong size.
  kWrongGridVersion,         // An input was built for another grid version.
  kSurfaceDirectionUnclear,  // The distance field is flat at a surface cell, so curvature has no direction.
  kCellSplitBySolid,         // A thin solid leaves liquid on both of its sides inside one cell. Use smaller cells.
  kStepTooLongForSurfaceTension,  // step_duration exceeds the surface tension limit. Shorten the step.
  kStepTooLongForAirPocket,  // step_duration exceeds an enclosed pocket's limit. Shorten the step.
  kAirPocketFateUnknown,     // Gas of a tracked pocket now touches an unknown frontier.
  kPressureLawOutOfRange,    // The carried pressure is below what the pressure law allows.
  kSealedBlobVolumeMismatch, // A sealed blob is asked to gain or lose volume it cannot. magnitude = m3/s.
  kEquationsNotPositive,     // A search direction gained no energy: the equations are not positive.
  kNotConverged,             // max_iterations reached above the limit. The caller may shorten the step.
  kNotFinite,                // A computed value is not finite.
  kCancelled,                // NEW. cancel_requested was set; the outputs of the call are not valid.
  kRunOnCallFailed,          // CheckCanRunOn failed for the chosen processor, or a call to it failed. Also out of
                             // memory (magnitude = bytes asked) and a solve batch that cannot hold one range
                             // with its border (A15).
};  // 15 values. kMaterialBoundaryNotSupported is removed (plan decision: the sharp boundary is delivered).
// Which function returns which (plan decision): kStepTooLongForSurfaceTension by FindLiquidSurface;
// kStepTooLongForAirPocket by FindAirPockets; kPressureLawOutOfRange by BuildPressureEquations when a kLiquid
// cell's bulk_stiffness is not > 0 (and by ComputePressureLawState).
struct PressureFailure {    // The error value of every std::expected here. Never allocates.
  PressureError error = PressureError::kNone;
  grid::CellPlaceIndex cell_index{grid::kNoIndex};            // First failing cell, or kNoIndex.
  grid::FacePlaceIndex face_piece_index{grid::kNoIndex}; // First failing face piece, or kNoIndex.
  double magnitude = 0;  // What was measured, in the unit named beside the error value.
  double limit = 0;      // What was allowed, same unit.
};
// pressure_error_message.h (CPU only)
std::string_view ToMessage(PressureError error);
// NEW. The stable code: the enumerator's name without the k, in lower snake case ("sealed_blob_volume_mismatch").
std::string_view ToName(PressureError error);
```

```cpp
// viscosity_geometry.h (shared)                                                      NEW (conflict C3, A)
// Filled by FindLiquidSurface only when the scene has a viscous ingredient (section 5 step 3b).
struct CellEighthGeometry {                // 96 bytes per cell place.
  double filled_volume_in_cell_eighth[8];  // m3. Bit 0 = high x, bit 1 = high y, bit 2 = high z. The 8 values
                                           // sum to CellGeometry::filled_volume within round-off.
  Vec3 liquid_covered_solid_centre{};      // m, world. Centre of the solid surface that liquid touches here.
  double liquid_covered_area = 0;          // m2 of that surface. 0: the cell has none.     DECIDED word
};
struct FacePieceCentre {                   // 24 bytes per face place.
  Vec3 liquid_covered_open_centre{};       // m, world. Centre of the open part of the piece that lies in liquid.
};                                         // The piece's centre when that part is empty.
```

```cpp
// geometry_snapshot.h (shared)                         snapshot DECIDED, GeometrySnapshot NEW
enum class CellContent : std::uint8_t {
  kLiquid,  // Centre inside liquid, filled_volume > 0.
  kAir,
  kSolid,
  kGas,     // NEW (gas). Set only in the gas module's snapshot; BuildPressureEquations treats it like kLiquid.
};
struct CellGeometry {
  double distance_to_surface = 0;  // m, at the cell centre, negative inside the liquid. DECIDED
  double filled_volume = 0;        // m3 of the cell that is liquid and outside every solid.
  double air_volume = 0;           // m3 of the cell outside liquid and outside every solid.
  double open_volume = 0;          // m3 of the cell outside every solid. NEW, for viscosity's stretch samples.
  double surface_curvature = 0;    // 1/m, positive where liquid bulges out (a drop). 0 away from air.
  CellContent content = CellContent::kSolid;
};
enum class FacePieceEnds : std::uint8_t { kNoLiquid, kLiquidToLiquid, kLiquidToAir, kLiquidToSolid };
struct FacePieceGeometry {
  double open_area = 0;          // m2 not blocked by solid, 0 to the piece's area. DECIDED
  double air_open_area = 0;      // m2 of open_area that lies in air. Used only for air pocket openings.
  double solid_volume_rate = 0;  // m3/s pushed along +axis by solids through the blocked part.
  double pressure_distance = 0;  // m between the two pressure positions of this piece (step 5 of section 5).
  double liquid_share = 1;       // 0..1, part of the centre-to-centre line inside liquid. 1 unless kLiquidToAir.
  FacePieceEnds ends = FacePieceEnds::kNoLiquid;
};
struct GeometrySnapshot {        // Read-only once returned. Lives in PressureBuffers.
  const grid::CellValues<CellGeometry>* cells;
  const grid::FaceValues<FacePieceGeometry>* face_pieces;
  double step_start_time = 0;    // s. Solids and liquid are evaluated at this time.
  double step_duration = 0;      // s.
  grid::GridVersion grid_version{};
  std::uint32_t lengthened_distance_count = 0;  // Liquid-to-air pieces whose liquid_share was raised to 0.05.
  double surface_tension_step_limit = 0;        // s, +infinity without surface tension.
  // NEW (C3). Both null when no ingredient has viscosity.
  const grid::CellValues<CellEighthGeometry>* cell_eighths = nullptr;
  const grid::FaceValues<FacePieceCentre>* face_piece_centres = nullptr;
};
FLUIDITY_CPU_AND_GPU double ComputeOpenFraction(const FacePieceGeometry& piece, double area);  // 0..1 DECIDED word

// geometry_clip_math.h (shared). NEW (from solids). The blocked part of a face piece, shared out to its four
// corners with the same two triangles that give open_area: corner_area[c] = integral over the blocked part of
// corner c's linear weight. Their sum is area - open_area, exactly as FindLiquidSurface computes it.
// distance_to_solid: m at the piece's 4 corners in grid::FindCornerPlaces order. area: m2. corner_area: m2.
FLUIDITY_CPU_AND_GPU void ComputeBlockedCornerAreas(const double distance_to_solid[4], double area,
                                                    double corner_area[4]);
```

```cpp
// pressure_buffers.h (CPU only)                                          buffers DECIDED
struct BufferSizes {
  // max_cells a multiple of 64; max_face_pieces == max_cells / 64 * 240, max_corners == max_cells / 64 * 125
  // (else kBadInput): every array is built with max_cell_blocks = max_cells / 64 (plan decision).
  std::uint32_t max_cells, max_face_pieces, max_corners, max_air_pockets, max_free_solids;
  std::uint32_t max_boundary_materials = 0;  // NEW (C4). Liquid materials that touch another; 0 or 2 and up.
  bool has_viscous_ingredient = false;       // NEW (C3). Allocates CellEighthGeometry and FacePieceCentre.
  std::uint32_t max_blocking_material_faces = 0, max_sealed_gas_blobs = 0;  // NEW (A18). More entries: kBadInput.
  // NEW (A15). 0: one piece. > 0: the batched layout with a loaded piece of this many blocks (kBadInput above
  // max_cells / 64); holds a SolveBatches built with stacked widths {kTouchingCells}.
  std::uint32_t max_loaded_blocks = 0;
  // NEW (A15). Doubles of direct-solve factor room; 0 means the layout's default: 32 * max_cells in one piece,
  // 4 * max_cells in the batched layout. Terminal blobs take room in ascending first-cell order.
  std::uint32_t max_direct_solve_values = 0;
};
// Owns the working buffers of this module on one processor. No allocation after it is built. The committed
// pressure, pocket_of_cell and the air-pocket list are not in it: the step owns them (try and committed copy).
class PressureBuffers;
// The same table BuildPressureBuffers allocates from on run_on (the sum and order buffers' bytes depend on it).
std::uint64_t ComputeBufferBytes(const BufferSizes& sizes, RunOn run_on);
std::expected<std::unique_ptr<PressureBuffers>, PressureFailure> BuildPressureBuffers(
    const BufferSizes& sizes, RunOn run_on);
// NEW (A15). The buffers' SolveBatches and its copy report; null in one piece.
const SolveBatches* FindSolveBatches(const PressureBuffers& buffers);
const SolveCopyReport* ReadSolveCopyReport(const PressureBuffers& buffers);
// NEW (A15). Pass 1 into whole geometry, both batched modes: copies the range blocks of every pass-1 row
// (ComputeDistanceToSurface and FindLiquidSurface outputs) and the snapshot's largest values (maxima, exact) from
// one-piece buffers bound to piece.grid() into whole buffers bound to the whole grid (GPU to GPU, or CPU to CPU).
// batches::ReadBatch calls it after pass 1. piece_buffers in the one-piece layout, whole_buffers in either, both
// otherwise kBadInput; kWrongGridVersion: a piece of another grid. Requires: both on one processor (debug
// assertion, A30).
std::expected<void, PressureFailure> WriteSnapshotRange(const PressureBuffers& piece_buffers,
                                                        const grid::GridPiece& piece, PressureBuffers& whole_buffers);
```

```cpp
// liquid_surface.h (CPU only)                                    FindLiquidSurface DECIDED
// 18 samples around a cell centre at that cell's own width. Order: z, then y, then x; the 8 corners are left out.
struct CurvatureSamples { grid::CellPlaceIndex cell_index; double around[18]; };
struct SurfaceDistances {                          // Samples of distance_to_surface, m.
  const grid::CornerValues<double>* at_corners;    // Every corner of every cell of the range and its border cells.
  const grid::CellValues<double>* at_centers;
  // One entry per liquid cell touching air, sorted by cell_index. Empty when no ingredient has surface tension.
  std::span<const CurvatureSamples> around_surface_cells;
};
inline constexpr std::uint32_t kAllLiquidMaterials = 0xFFFFFFFFu;  // NEW. See material_index below.
struct SurfaceDistanceInputs {
  const grid::StaggeredGrid* grid; grid::CellBlockRange blocks;
  // The liquid materials (A8: the liquid material set). Gas markers and blocking materials are never read.
  const particles::ParticlesTry* particles_try;
  // NEW. Transfer's material-subset input: one entry per material_index, 1 for each liquid material.
  // NEW (decision in changes-to-apply.md). All zero is accepted here, and only here: it means "no liquid",
  // every open cell is air and every distance is positive (a gas-only scene calls it this way). Transfer's
  // stages still require at least one counted material.
  std::span<const std::uint8_t> material_is_counted;
  // NEW (C4). kAllLiquidMaterials: the field of every liquid, the one FindLiquidSurface reads. A material
  // index: the field of that liquid only, kept in its own buffer for FindMaterialBoundaries.
  std::uint32_t material_index = kAllLiquidMaterials;
  double particle_radius = 0;      // m. A particle counts as a ball of this radius.
  double surface_tension_max = 0;  // N/m, largest over ingredients. 0 skips around_surface_cells.
  // 1 for a cell of the grid-only region (from cell resize). Null: every cell is in the particle region.
  const grid::CellValues<std::uint8_t>* is_grid_only_region = nullptr;
};
// Fails: kBadInput (a material index that is not counted, a one-material field with an all-zero
// material_is_counted, a mask whose size is not the material count, or more one-material fields than
// max_boundary_materials), kWrongGridVersion, kNotFinite, kRunOnCallFailed. Overwrites only the field named
// by material_index. With an all-zero material_is_counted the kAllLiquidMaterials field is "no liquid".
std::expected<SurfaceDistances, PressureFailure> ComputeDistanceToSurface(
    const SurfaceDistanceInputs& inputs, PressureBuffers& buffers);

struct SolidCoverage {             // Filled by the solids module for this try. The only source of solid geometry.
  // Over every active cell, also where there is no liquid (gas reads it).
  const grid::CornerValues<double>* distance_to_solid;         // m, positive outside every solid.
  const grid::FaceValues<double>* solid_normal_velocity;  // m/s along +axis, mean over the blocked part.
};
struct LiquidSurfaceInputs {
  const grid::StaggeredGrid* grid; grid::CellBlockRange blocks;
  SurfaceDistances distance_to_surface; SolidCoverage solids;  // distance_to_surface: kAllLiquidMaterials.
  double step_start_time = 0, step_duration = 0;  // s.
  double surface_tension_max = 0;                 // N/m. 0 skips curvature and the step limit.
  double lightest_density = 0;                    // kg/m3, for the surface tension step limit.
  bool has_viscous_ingredient = false;            // NEW (C3). Fills cell_eighths and face_piece_centres.
};
std::expected<GeometrySnapshot, PressureFailure> FindLiquidSurface(
    const LiquidSurfaceInputs& inputs, PressureBuffers& buffers);

// surface_tension.h (shared)
FLUIDITY_CPU_AND_GPU PressureError ComputeSurfaceCurvature(
    double center, const double around[18], double cell_width, double* surface_curvature);
FLUIDITY_CPU_AND_GPU double ComputeSurfaceTensionStepLimit(
    double density_low, double density_high, double cell_width, double surface_tension);  // s
```

```cpp
// pressure_law.h (shared)                                               PressureLaw DECIDED
enum class PressureLawKind : std::uint8_t {
  kCannotBeSqueezed,       // Incompressible.
  kSteadyStiffness,        // p = -K0 ln(volume_ratio). Textbook: Hencky.
  kStiffensUnderPressure,  // p = (K0/7) (volume_ratio^-7 - 1). Textbook: Tait, exponent 7.
};
struct PressureLaw {
  PressureLawKind kind = PressureLawKind::kCannotBeSqueezed;
  double rest_density = 0;      // kg/m3 at zero pressure above atmosphere.
  double rest_sound_speed = 0;  // m/s. K0 = rest_density * rest_sound_speed^2.
};
struct PressureLawState {
  double volume_ratio = 1;    // Current volume over rest volume. Textbook: J.
  double density = 0;         // kg/m3.
  double bulk_stiffness = 0;  // Pa, pressure per unit of relative volume loss at this pressure. Textbook: K.
};
FLUIDITY_CPU_AND_GPU PressureError ComputePressureLawState(
    const PressureLaw& law, double pressure_above_atmosphere, PressureLawState* state);
```

```cpp
// air_pockets.h (CPU only)                                   blob, blob_id, BlobHistory DECIDED
enum class AirPocketClass : std::uint8_t { kEnclosed, kVented, kUnknownFrontier };
struct AirPocket {
  std::uint64_t blob_id = 0;         // Stable across steps. 0 while the pocket carries no tracked gas.
  grid::CellPlaceIndex first_cell_index{grid::kNoIndex};
  AirPocketClass pocket_class = AirPocketClass::kUnknownFrontier;
  double air_volume = 0;             // m3 now.
  double rest_volume = 0;            // m3 the gas fills at rest_pressure. 0 when not tracked.
  double rest_pressure = 0;          // Pa, absolute.
  double pressure = 0;               // Pa, absolute, now.
  double pressure_above_atmosphere = 0;  // Pa. What the equations use, before surface tension.
  bool is_resolved = false;          // Equivalent radius is at least min_radius_in_cells cell widths.
};
struct AirPocketSettings {
  double atmosphere_pressure = 101325;  // Pa, absolute.
  double heat_ratio = 1.4;              // No unit. Textbook: polytropic exponent, air.
  double min_radius_in_cells = 2;       // Below this a pocket is not given a pressure of its own.
};
struct BlobHistory {            // What became of last step's tracked gas, in m3 * Pa^(1/heat_ratio).
  double vented = 0, too_small = 0, unmatched = 0;
  std::uint64_t next_blob_id = 1;
};
inline constexpr std::uint32_t kNoAirPocket = 0xFFFFFFFFu;  // NEW. pocket_of_cell of a cell in no pocket.
struct AirPockets {             // Views of the step's arrays (try copy, or committed copy after the commit).
  std::span<const AirPocket> pockets;                         // Ordered by first_cell_index.
  const grid::CellValues<std::uint32_t>* pocket_of_cell;      // Index into pockets, or kNoAirPocket.
  BlobHistory history;
  double step_limit = 0;        // s, smallest over enclosed resolved pockets; +infinity if none.
};
struct AirPocketArrays {       // NEW (step). The step's try copies; FindAirPockets writes nothing else.
  grid::CellValues<std::uint32_t>* pocket_of_cell;  // Bound to this try's grid version.
  std::span<AirPocket> pocket_storage;              // CPU, max_air_pockets long. The first pockets.size() are used.
};
struct AirPocketInputs {
  const grid::StaggeredGrid* grid; grid::CellBlockRange blocks; GeometrySnapshot geometry;
  const AirPockets* committed;                  // Last committed pockets, or null on the first step.
  const grid::FaceValues<std::uint8_t>* grid_edge_is_open_to_sky;  // Section 5 step 9. From solids.
  // NEW (A10). 1 for a cell that holds the gas module's tracked gas (committed, remapped to this grid). Such a
  // cell belongs to no air pocket (one cell is never both, step 9). Null: the scene has no gas.
  const grid::CellValues<std::uint8_t>* holds_tracked_gas = nullptr;
  AirPocketSettings settings; double lightest_density = 0;  // kg/m3.
};
// The returned views point into outputs. On failure the content of outputs is not valid: they are try copies
// the step discards, never committed. Fails: kBadInput (more than max_air_pockets), kWrongGridVersion,
// kAirPocketFateUnknown, kNotFinite, kRunOnCallFailed.
std::expected<AirPockets, PressureFailure> FindAirPockets(const AirPocketInputs& inputs, AirPocketArrays outputs,
                                                          PressureBuffers& buffers);
FLUIDITY_CPU_AND_GPU double ComputeAirPocketPressure(  // Pa absolute. In air_pocket_math.h (shared).
    double rest_volume, double rest_pressure, double air_volume, double heat_ratio);
```

```cpp
// open_sea.h (CPU only)
struct OpenSeaWave {
  double still_water_height = 0;  // m, world Z.
  double water_depth = 1, wave_height = 0;  // m.
  double wave_period = 1;         // s.
  Vec3 travel_direction{1, 0, 0}; // Horizontal unit vector.
  double phase_at_origin = 0;     // rad.
};
struct OpenSeaSettings {
  OpenSeaWave wave;
  contact::DomainBox domain_box;  // NEW (plan decision). The zone positions are measured from its sides.
  contact::DomainSide open_sea_side, outlet_side;      // Waves enter at one, are calmed at the other.
  double open_sea_zone_width = 1, outlet_zone_width = 1;  // m, measured into the domain.
  double open_sea_pull_rate = 0, outlet_pull_rate = 0;    // 1/s. 0 turns the zone off.
};
struct OpenSeaWaveSample { double surface_height = 0; Vec3 velocity{}; };  // m world Z; m/s.
class ReadyOpenSea;  // Settings checked, wave number solved once.
std::expected<ReadyOpenSea, PressureFailure> BuildOpenSea(const OpenSeaSettings& settings, double gravity);
OpenSeaWaveSample ComputeOpenSeaWave(const ReadyOpenSea& open_sea, Vec3 position, double time);
// Pulls face velocities inside the two zones toward the wave (open-sea side) or toward rest (outlet).
// Called by solids::DriveWaves, which the step calls in stage DriveWaves (plan decision).
std::expected<void, PressureFailure> ApplyOpenSeaRelaxation(
    const ReadyOpenSea& open_sea, const grid::StaggeredGrid& grid, grid::CellBlockRange blocks,
    double step_end_time, double step_duration, grid::FaceValues<double>& velocity);
```

```cpp
// pressure_equations.h (shared types, CPU-only functions)     PressureEquations, cell_equation, cell_connection,
//                                               added_volume_rate, squeeze_allowance, pressure_offset DECIDED
inline constexpr std::uint32_t kNoEquation = 0xFFFFFFFFu;  // NEW (declared). A connection end with no equation.
inline constexpr std::uint32_t kNoBlob = 0xFFFFFFFFu;      // NEW (gas). blob_of_cell of a cell with no equation.
struct CellEquation {              // One per kLiquid or kGas cell, ordered by cell_index.
  grid::CellPlaceIndex cell_index{grid::kNoIndex};
  std::uint32_t blob_index = 0;    // Which connected body of liquid.
  double filled_volume = 0;        // m3.
  double added_volume_rate = 0;    // m3/s the cell must gain: sources, gas expansion, volume correction.
  double squeeze_allowance = 0;    // m3/(Pa s) = filled_volume / (bulk_stiffness * step_duration). 0: none.
  double carried_pressure = 0;     // Pa. Pressure brought into this step; also the starting guess.
  double flow_imbalance_before_pressure = 0;  // m3/s, inflow minus outflow, liquid and solid parts. DECIDED sign
  double own_weight = 0;           // m3/(Pa s): squeeze_allowance plus flow_per_pressure of every connection.
};
struct CellConnection {            // One per face piece with open_area > 0 that touches an equation cell.
  grid::FacePlaceIndex face_piece_index{grid::kNoIndex};
  std::uint32_t equation_low = kNoEquation, equation_high = kNoEquation;  // DECIDED: cell_low, cell_high
  double open_area = 0;            // m2. Copied from the snapshot; the only area used anywhere.
  double face_density = 0;         // kg/m3.
  double velocity_per_pressure = 0;  // (m/s)/Pa = step_duration / (face_density * pressure_distance).
  double flow_per_pressure = 0;      // m3/(Pa s) = open_area * velocity_per_pressure.
  double pressure_jump = 0;        // Pa, high minus low at rest, between two liquids that do not mix.
  double reference_pressure = 0;   // Pa at an end that is not an equation cell: surface tension plus, in this
                                   // order, the sealed gas blob's gas_pressure_above_background (the blob's
                                   // own p_j is the unknown, A18), the tracked gas's pressure, the air
                                   // pocket's, else 0 (atmosphere).
  double solid_volume_rate = 0;    // m3/s along +axis. Copied from the snapshot.
  double velocity_before_pressure = 0;  // m/s along +axis. DECIDED time mark
};
enum class BlobClass : std::uint8_t { kHasReferencePressure, kCanBeSqueezed, kSealed };
struct PressureBlob {              // One connected body of liquid in this solve.
  grid::CellPlaceIndex first_cell_index{grid::kNoIndex};
  BlobClass blob_class = BlobClass::kSealed;
  double filled_volume = 0;             // m3.
  double incompatible_volume_rate = 0;  // m3/s a kSealed blob cannot absorb. Reported, never subtracted.
  double pressure_offset = 0;           // Pa added to a kSealed blob after the solve (section 5 step 16).
};
struct FreeSolidPush {             // Motion map of one free solid, from the solids module.
  std::uint32_t solid_index, equation_index;  // equation_index as ComputeEquationIndexes gives it.
  Vec3 force_per_pressure, torque_per_pressure;  // N/Pa and N m/Pa on the solid from this cell's pressure.
};
// The boundary between two liquids that do not mix. In the particles design these are two materials (the
// ingredients of one material always mix), so this design's first name, IngredientBoundary, was changed.
// Built by FindMaterialBoundaries (section 5 step 1b).
struct MaterialBoundary {
  grid::FacePlaceIndex face_piece_index; double low_share;  // 0..1 of pressure_distance inside the low cell's liquid.
  double surface_tension; double surface_curvature;  // N/m; 1/m, positive when the low liquid bulges out.
};
enum class SealedBlobCheck : std::uint8_t {  // NEW (gas). When kSealedBlobVolumeMismatch is checked.
  kWhenBuilt,         // BuildPressureEquations fails with kSealedBlobVolumeMismatch (liquid).
  kWhenRatesApplied,  // The build only reports incompatible_volume_rate; ApplyAddedVolumeRate checks (gas).
};
// NEW (A18). Built by solids::BuildBlockingMaterialFaces (one entry per piece touching a kLiquid cell whose set
// has face mass) and by gas::ApplyGasStepStart (one entry per sealed blob touching liquid). Section 5 C2.
struct BlockingMaterialFace {          // One face piece where liquid meets a blocking material set.
  grid::FacePlaceIndex face_piece_index{grid::kNoIndex};
  std::uint32_t material_set = 0;      // Index into the step's list of blocking sets.
  double blocked_area = 0;             // m2, > 0: this set's corner areas (ComputeBlockedCornerAreas) summed.
  double face_mass = 0;                // kg, > 0: the set's GridFromParticles::mass on the piece.
  double velocity_before_pressure = 0; // m/s along +axis, after ApplyForces and CollideSolids.
  double coverage_volume_rate = 0;     // m3/s along +axis: this set's share of the snapshot's solid_volume_rate.
};
struct SealedGasBlob {                 // One sealed tracked-gas blob that touches liquid, not held at atmosphere.
  grid::CellPlaceIndex first_cell_index{grid::kNoIndex};
  double gas_volume = 0;               // m3, > 0, step start.
  double background_pressure = 0;      // Pa absolute, > 0, step start, equalised.
  double heat_ratio = 1.4;
  double pressure_above_atmosphere = 0;  // Pa: the blob's carried value.
  double heat_volume_rate = 0;         // m3/s = ((heat_ratio - 1) / heat_ratio) * sum V q / p0.
  double boundary_volume_rate = 0;     // m3/s of solid moving into the blob through pieces that are not liquid
};                                     //   connections (inflow positive).
struct EquationInputs {
  const grid::StaggeredGrid* grid; grid::CellBlockRange blocks; GeometrySnapshot geometry;
  // m/s. The liquid material set's velocity (A8). Kept as a view by PressureEquations; must not alias any
  // output of SolvePressure.
  const grid::FaceValues<double>* velocity_before_pressure;
  const grid::CellValues<double>* mixture_density;        // kg/m3. 0 in a cell with no particle (step 13).
  const grid::CellValues<double>* bulk_stiffness;         // Pa; +infinity where the mixture cannot be squeezed.
  const grid::CellValues<double>* carried_pressure;       // Pa
  const grid::CellValues<double>* added_volume_rate;      // m3/s; null means zero.
  const AirPockets* air_pockets;                          // null: every air end is at atmosphere.
  // NEW (A10). Pa above atmosphere in each cell that holds tracked gas: the gas module's pressure there at the
  // end of the last committed step (background pressure minus atmosphere, plus the gas solve's pressure). An
  // air end whose air cell has holds_tracked_gas, and no sealed gas blob, uses it (step 13; read one step
  // late). Both null: the scene has no gas.
  const grid::CellValues<double>* gas_pressure_above_atmosphere = nullptr;
  const grid::CellValues<std::uint8_t>* holds_tracked_gas = nullptr;
  double surface_tension_with_air = 0;                    // N/m
  std::span<const MaterialBoundary> material_boundaries;  // Sorted by face_piece_index.
  std::span<const FreeSolidPush> free_solid_pushes;       // Sorted by solid_index, then equation_index.
  std::span<const Mat3> free_solid_inverse_inertia;       // Two per solid: 1/kg, then 1/(kg m2), world frame.
  SealedBlobCheck sealed_blob_check = SealedBlobCheck::kWhenBuilt;  // NEW (gas).
  double lightest_density = 0;        // NEW (plan decision). kg/m3, step 13's fallback density.
  double max_volume_error_rate = 2e-5;  // NEW (plan decision). 1/s, step 16's limit (the build has no SolveSettings).
  // NEW (A18). Sorted by face_piece_index, then material_set. Empty: no blocking material beside the liquid.
  std::span<const BlockingMaterialFace> blocking_material_faces;
  std::span<const SealedGasBlob> sealed_gas_blobs;        // NEW (A18). By first_cell_index.
  // NEW (A18). Index into sealed_gas_blobs, kNoBlob elsewhere (an index past the list: kBadInput).
  const grid::CellValues<std::uint32_t>* sealed_gas_blob_of_cell = nullptr;
  // NEW (A18). Pa, the gas solve's pressure of the last commit (added to a sealed blob connection's reference).
  const grid::CellValues<double>* gas_pressure_above_background = nullptr;
  // NEW (A27, plan decision). 6 per free solid (V m/s, then Omega rad/s, world axes, entries [6 s, 6 s + 6) for
  // solid_index s): the motion before pressure, for the kIncluded free-solid flow of the CoupledVelocities forms.
  // Size 0 or 6 x the free solid count (half of free_solid_inverse_inertia), else kBadInput. SolvePressure ignores it.
  std::span<const double> free_solid_motion_before_pressure;
};
struct EquationArrays;             // NEW (plan decision). Opaque; its content is private to the module.
struct PressureEquations {         // Lives in PressureBuffers. Read-only once returned.
  std::span<const PressureBlob> blobs;      // Ordered by first_cell_index.
  std::uint32_t equation_count = 0, connection_count = 0;
  double step_duration = 0; grid::GridVersion grid_version{};
  const grid::FaceValues<double>* velocity_before_pressure = nullptr;  // The input's view; for unconnected pieces.
  SealedBlobCheck sealed_blob_check = SealedBlobCheck::kWhenBuilt;      // NEW. Copied from EquationInputs.
  double max_volume_error_rate = 0;         // NEW (plan decision). Copied from EquationInputs.
  // NEW (plan decision). Every function taking PressureEquations fails kWrongGridVersion when the buffers have
  // been rebuilt since (build_number differs) or after a failed build.
  const EquationArrays* arrays = nullptr;
  std::uint64_t build_number = 0;
  std::uint32_t blocking_material_face_count = 0, sealed_gas_blob_count = 0;  // NEW (A18).
  // cell_equations, cell_connections, the per-equation connection, material-face and blob-term lists stay on the
  // processor (in CPU memory in the batched layout).
};
// Fails: kBadInput (also, A18: a material face with no solved end, face_mass <= 0 or blocked_area <= 0, unsorted
// entries, more entries than BufferSizes allows; a blob with gas_volume or background_pressure not > 0; a
// sealed_gas_blob_of_cell index past the list), kWrongGridVersion, kCellSplitBySolid, kPressureLawOutOfRange,
// kSealedBlobVolumeMismatch (kWhenBuilt only), kNotFinite, kCancelled (batched layout), kRunOnCallFailed.
std::expected<PressureEquations, PressureFailure> BuildPressureEquations(
    const EquationInputs& inputs, PressureBuffers& buffers);
// Replaces added_volume_rate and refreshes blobs. For a caller (gas) that needs blob totals first. With
// SealedBlobCheck::kWhenRatesApplied it then checks each kSealed blob (kSealedBlobVolumeMismatch).
std::expected<void, PressureFailure> ApplyAddedVolumeRate(
    const grid::CellValues<double>& added_volume_rate, PressureEquations& equations, PressureBuffers& buffers);
// NEW (solids). The equation index BuildPressureEquations will give each kLiquid or kGas cell (ascending
// cell index), or kNoEquation. Reads only geometry. Returns the equation count.
std::expected<std::uint32_t, PressureFailure> ComputeEquationIndexes(
    const GeometrySnapshot& geometry, grid::CellBlockRange blocks,
    grid::CellValues<std::uint32_t>& equation_of_cell);
// NEW (gas). Index into PressureEquations::blobs for each cell that has an equation, kNoBlob elsewhere.
std::expected<void, PressureFailure> ComputeBlobOfCell(const PressureEquations& equations,
                                                       grid::CellValues<std::uint32_t>& blob_of_cell);

// material_boundary.h (CPU only)                                                     NEW (conflict C4, A)
struct MaterialBoundaryInputs {
  const grid::StaggeredGrid* grid; grid::CellBlockRange blocks; GeometrySnapshot geometry;
  // Each from ComputeDistanceToSurface with material_index set to that liquid. low: the lower material_index.
  SurfaceDistances low_material, high_material;
  double surface_tension = 0;      // N/m, materials::MaterialPair::surface_tension of the two.
};
// One entry per face piece between two kLiquid cells that the boundary crosses (section 5 step 1b), sorted by
// face_piece_index; the span lives in buffers until the next call (plan decision: the sharp boundary is
// delivered; there is no exclusion).
std::expected<std::span<const MaterialBoundary>, PressureFailure> FindMaterialBoundaries(
    const MaterialBoundaryInputs& inputs, PressureBuffers& buffers);

// volume_correction.h (CPU only)
struct VolumeCorrectionSettings { double correction_duration = 0; double max_rate = 0; };  // s; 1/s.
std::expected<void, PressureFailure> ComputeVolumeCorrectionRate(   // Adds to added_volume_rate, m3/s.
    const GeometrySnapshot& geometry, const grid::CellValues<double>& sensed_volume_ratio,
    const grid::CellValues<double>& target_volume_ratio,
    const grid::CellValues<std::uint8_t>* is_grid_only_region, const VolumeCorrectionSettings& settings,
    grid::CellValues<double>& added_volume_rate, PressureBuffers& buffers);
```

```cpp
// pressure_solve.h (CPU only)                                         SolvePressure DECIDED
struct SolveSettings {             // The one acceptance limit; viscosity's joint solve and gas use it too.
  int max_iterations = 256;
  double max_volume_error_rate = 2e-5;  // 1/s. Largest |volume error| of a cell per second, over its filled_volume.
  double jump_error_fraction = 1e-3;    // No unit. Extra limit on norm of equation_error when a jump is present.
  int iterations_per_progress_read = 8; // The CPU reads one small progress record this often.
};
struct FlowImbalanceReport {           // All from velocity_after_pressure, never from the solver's own vectors.
  double max_volume_error_rate = 0;    // 1/s
  grid::CellPlaceIndex worst_cell_index{grid::kNoIndex};
  double net_volume_change = 0;        // m3 this step, signed sum over cells. Sealed gas blob rows not counted.
  double total_volume_change = 0;      // m3 this step, sum of absolute values.
  double equation_error_norm = 0;      // m3/s, 2-norm over cells.
  // NEW (A18). 1/s, largest |volume error| of a sealed gas blob's row over its gas_volume (section 5 step 21).
  double max_sealed_gas_volume_error_rate = 0;
};
struct SolvedArrays {              // NEW (step). The step's try copies, bound to the equations' grid version.
  grid::CellValues<double>* pressure;                  // Pa, 0 in cells without an equation.
  grid::FaceValues<double>* velocity_after_pressure;   // m/s, every face piece. DECIDED time mark
  // NEW (A18). m/s along +axis, one per BlockingMaterialFace, input order. Size must equal the count (kBadInput).
  std::span<double> blocking_material_velocity_after_pressure;
  // NEW (A18). Pa above atmosphere at the step end, one per SealedGasBlob. Size must equal the count (kBadInput).
  std::span<double> sealed_gas_pressure;
};
struct SolveResult {
  FlowImbalanceReport report;
  std::span<const PressureBlob> blobs;  // pressure_offset set. In buffers until the next build.
  int iteration_count = 0, ladder_rung_count = 0;
  // NEW (A15). Terminal blobs beyond max_direct_solve_values, given one weighted-Jacobi pass down and up instead.
  std::uint32_t blobs_without_direct_solve = 0;
};
// For scenes with no viscous ingredient (a scene with one runs viscosity::SolveViscosityAndPressure, which
// calls the pieces below); also the gas module's solve on its own equations. Solves equations that hold
// blocking-material faces or sealed gas blobs (A18) by eliminating the blob unknowns; the joint viscous solve
// carries the same terms with the blob pressures (A24), material face velocities and free-solid motions (A26, A27)
// as unknowns. Solves, corrects the velocity (pieces with no
// connection keep velocity_before_pressure), recovers each material velocity (ComputeBlockingMaterialVelocities)
// and blob pressure, and checks the
// result, writing the outputs only after the check passes. On failure the outputs hold no valid value: they are
// try copies the step discards (in the batched layout they keep their bytes). cancel_requested may be null; it is
// read at a progress read before the first iteration, then every iterations_per_progress_read iterations, and in
// the batched layout before every batch load; it gives kCancelled. Fails: kBadInput (a coupling span of the wrong
// size), kWrongGridVersion, kEquationsNotPositive, kNotConverged, kNotFinite, kCancelled, kRunOnCallFailed.
std::expected<SolveResult, PressureFailure> SolvePressure(
    const PressureEquations& equations, const SolveSettings& settings,
    const std::atomic<bool>* cancel_requested, SolvedArrays outputs, PressureBuffers& buffers);

// flow_imbalance.h (CPU only): the pieces of SolvePressure, exposed for the joint solve, checks and tests.
// NEW (A24, replaces the A18 refusal): no piece returns kBadInput because the equations hold a blocking-material
// face or a sealed gas blob. The forms without a blob buffer (both ComputePressurePush forms, ComputeVolumeError,
// ComputeOwnWeight, ComputeStartPressure) accept material faces and, given equations with sealed_gas_blob_count > 0,
// return kBadInput writing no output (the blob's far-end pressure is an input they lack); the forms with a blob
// buffer (coupled_solve_pieces.h, below) carry the blobs. ComputeVolumeError adds each material face's flow;
// ComputeOwnWeight adds g_k = dt A_k^2 / m_k per solved end, not multiplied by connection_scale; ComputeFaceInertia,
// ApplyPressure, ComputeFlowImbalance (flow of the velocity only), ApplyZeroMeanToSealedBlobs, ApplyPressureOffset
// and ComputeWasSolved accept both terms with unchanged arithmetic. CheckFlowImbalance below recovers the blob
// pressures by the elimination (SolvePressure's form). NEW (A15): in the batched layout each piece is one sweep and
// accepts whole arrays in CPU memory or on the processor.
// 0 on a face piece with no connection: its velocity is given, not solved.
std::expected<void, PressureFailure> ComputeFaceInertia(      // kg = open_area * pressure_distance * face_density
    const PressureEquations& equations, grid::FaceValues<double>& face_inertia);
// push, N s = step_duration * open_area * (p_low - p_high + pressure_jump), reference_pressure at a missing end.
// The same as the GivenTerms form below with GivenTerms::kIncluded.
std::expected<void, PressureFailure> ComputePressurePush(
    const PressureEquations& equations, const grid::CellValues<double>& pressure, grid::FaceValues<double>& push);
std::expected<void, PressureFailure> ApplyPressure(           // velocity += push / face_inertia
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    grid::FaceValues<double>& velocity);
std::expected<void, PressureFailure> ComputeFlowImbalance(    // m3/s per cell, inflow minus outflow
    const PressureEquations& equations, const grid::FaceValues<double>& velocity,
    grid::CellValues<double>& flow_imbalance);
// The stop rule of step 21, shared by every solve that ends in a pressure. With the A18 terms it recomputes each
// material velocity and each sealed blob pressure from `pressure` (never from the solver's vectors) and checks
// each blob row; kNotConverged when a cell or blob is above the limit.
std::expected<FlowImbalanceReport, PressureFailure> CheckFlowImbalance(
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    const grid::FaceValues<double>& velocity, const SolveSettings& settings, PressureBuffers& buffers);
// NEW (A24). The stop rule with the blob pressures given, instead of recovered by the elimination (SolvePressure's
// form above); material faces still eliminated. Since A26/A27 the joint solve calls the CoupledVelocities form
// (coupled_solve_pieces.h) instead.
std::expected<FlowImbalanceReport, PressureFailure> CheckFlowImbalance(
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    const Buffer<double>& sealed_gas_pressure, const grid::FaceValues<double>& velocity,
    const SolveSettings& settings, PressureBuffers& buffers);
```

Added by the reconciliation of 2026-11-14 (all NEW). The first header holds what
`viscosity::SolveViscosityAndPressure` calls from its outer loop and what transfer's snapshot reads; the
second exposes the ladder as a preconditioner for that loop (A5); the third replaces "carried pressure from
transfer" (no particle carries a pressure; on `main` the step moves it on the grid).

```cpp
// coupled_solve_pieces.h (CPU only)
enum class GivenTerms : std::uint8_t {
  kIncluded,  // The physical candidate: reference pressures, jumps, solid volume rates, added volume rate and
              // carried pressure all count.
  kLeftOut,   // A search direction: only the part that is linear in the pressure and velocity passed in.
};
// As ComputePressurePush above; with kLeftOut the reference pressures and jumps count as zero.
std::expected<void, PressureFailure> ComputePressurePush(
    const PressureEquations& equations, const grid::CellValues<double>& pressure, GivenTerms given_terms,
    grid::FaceValues<double>& push);
// m3/s per cell (step 21): flow_imbalance + added_volume_rate - squeeze_allowance * (p - carried_pressure).
// With kLeftOut: the flow of `velocity` through open areas only, minus squeeze_allowance * p.
std::expected<void, PressureFailure> ComputeVolumeError(
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    const grid::FaceValues<double>& velocity, GivenTerms given_terms, grid::CellValues<double>& volume_error);
// m3/(Pa s) per cell: squeeze_allowance plus the sum over the cell's connections of flow_per_pressure *
// connection_scale, in ascending face_piece_index. A null scale counts as 1 and gives CellEquation::own_weight.
// connection_scale: no unit, 0..1 per face piece; the joint solve passes face_inertia / velocity diagonal.
std::expected<void, PressureFailure> ComputeOwnWeight(
    const PressureEquations& equations, const grid::FaceValues<double>* connection_scale,
    grid::CellValues<double>& own_weight);
// Pa. carried_pressure in every cell that has an equation, 0 elsewhere: the starting guess.
std::expected<void, PressureFailure> ComputeStartPressure(const PressureEquations& equations,
                                                         grid::CellValues<double>& pressure);
// Subtracts from each kSealed blob its volume-weighted mean. For a search direction.
std::expected<void, PressureFailure> ApplyZeroMeanToSealedBlobs(
    const PressureEquations& equations, grid::CellValues<double>& values, PressureBuffers& buffers);
// Step 16: shifts each kSealed blob so that its volume-weighted mean equals that of carried_pressure.
std::expected<void, PressureFailure> ApplyPressureOffset(
    const PressureEquations& equations, grid::CellValues<double>& pressure, PressureBuffers& buffers);
// 1 for each face piece that has a CellConnection and for each cell that has a CellEquation, else 0.
std::expected<void, PressureFailure> ComputeWasSolved(
    const PressureEquations& equations, grid::FaceValues<std::uint8_t>& was_solved,
    grid::CellValues<std::uint8_t>& cell_was_solved);
// NEW (A24). One value per sealed gas blob of the equations, entries [0, sealed_gas_blob_count), on the
// equations' processor: Pa above atmosphere (pressure), m3/s (volume error), m3/(Pa s) (own weight). A blob buffer
// shorter than the count, or a material span of the wrong size: kBadInput, no output written. Requires: every
// blob buffer on the equations' processor (debug assertion, A30). A blob connection's far-end pressure is sealed_gas_pressure[j] + reference_pressure (kLeftOut: the
// reference counts as zero). Blob volume error (section 5 C2): inflow_j(velocity) + heat_volume_rate +
// boundary_volume_rate - S_j (q_j - c_j) (kIncluded), inflow_j(velocity) - S_j q_j (kLeftOut, open areas only).
// Blob own weight: S_j + sum_c g_c * connection_scale_c. Blob start value: c_j = pressure_above_atmosphere.
std::expected<void, PressureFailure> ComputePressurePush(
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    const Buffer<double>& sealed_gas_pressure, GivenTerms given_terms, grid::FaceValues<double>& push);
std::expected<void, PressureFailure> ComputeVolumeError(
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    const Buffer<double>& sealed_gas_pressure, const grid::FaceValues<double>& velocity, GivenTerms given_terms,
    grid::CellValues<double>& volume_error, Buffer<double>& sealed_gas_volume_error);
std::expected<void, PressureFailure> ComputeOwnWeight(
    const PressureEquations& equations, const grid::FaceValues<double>* connection_scale,
    grid::CellValues<double>& own_weight, Buffer<double>& sealed_gas_own_weight);
std::expected<void, PressureFailure> ComputeStartPressure(const PressureEquations& equations,
    grid::CellValues<double>& pressure, Buffer<double>& sealed_gas_pressure);   // c_j per blob
// NEW (A24). Each BlockingMaterialFace's velocity after `pressure` (ComputeMaterialVelocityAfterPressure, unsolved
// end 0), input order; written as SolvePressure writes SolvedArrays::blocking_material_velocity_after_pressure.
std::expected<void, PressureFailure> ComputeBlockingMaterialVelocities(const PressureEquations& equations,
    const grid::CellValues<double>& pressure, std::span<double> velocity_after_pressure);
// NEW (A26, A27), for the joint solve only (the A24 forms above stay for SolvePressure's arithmetic and are not
// called by viscosity). Material faces and free solids are velocity unknowns of the joint solve: their velocities
// are given, never recovered from `pressure`. F = free solid count (half of free_solid_inverse_inertia); a solid's
// xi = (V, Omega), entries [6 s, 6 s + 6). kBadInput, no output written: a CoupledVelocities pointer null while its
// count (blocking_material_face_count, F) is > 0; a given or output buffer shorter than its count (one per
// BlockingMaterialFace, or 6 F); kIncluded with F > 0 and an empty
// free_solid_motion_before_pressure; the blob buffers as the A24 forms. A null pointer with count 0 is accepted.
// Requires: every buffer on the equations' processor (debug assertion, A30).
struct CoupledVelocities {                              // Given velocity unknowns other than the liquid's.
  const Buffer<double>* blocking_material_velocity = nullptr;   // one per BlockingMaterialFace
  const Buffer<double>* free_solid_motion = nullptr;            // 6 per free solid: V then Omega
};
// Cells as the A24 blob form, except: each solved end e of material face k gets -s_e (A_k w_k -
// coverage_volume_rate) (kIncluded) or -s_e A_k w_k (kLeftOut), whatever `pressure` is; each cell gets, per
// FreeSolidPush entry e (ascending solid_index, after the material faces, before the blob terms), -push_e . (xi -
// xi_before) (kIncluded) or -push_e . xi (kLeftOut), push_e = (force_per_pressure, torque_per_pressure). Blob rows
// as the A24 blob form.
std::expected<void, PressureFailure> ComputeVolumeError(const PressureEquations&, const grid::CellValues<double>&
    pressure, const Buffer<double>& sealed_gas_pressure, const grid::FaceValues<double>& velocity, CoupledVelocities,
    GivenTerms, grid::CellValues<double>& volume_error, Buffer<double>& sealed_gas_volume_error);
// Material face flow A_k w_k - coverage_volume_rate (kIncluded) or A_k w_k (kLeftOut); a free solid's flow
// sum_e push_e . (xi - xi_before) (kIncluded) or . xi (kLeftOut), sign as SolvePressure's free-solid term.
std::expected<void, PressureFailure> ComputeCoupledPush(const PressureEquations&, const grid::CellValues<double>&
    pressure, Buffer<double>& blocking_material_push, Buffer<double>& free_solid_push);
// dt A_k (p_low - p_high) per BlockingMaterialFace; dt sum_e (force, torque)_per_pressure p_e per free solid.
// N s along +axis (unsolved end 0); N s then N m s, components force x, y, z, torque x, y, z; each solid's sums by
// the per-group rule (step 20). Pressure only: no reference pressure, jump or given velocity.
std::expected<void, PressureFailure> ComputeOwnWeight(const PressureEquations&, const grid::FaceValues<double>*
    connection_scale, const Buffer<double>* material_face_scale, grid::CellValues<double>& own_weight,
    Buffer<double>& sealed_gas_own_weight);
// As the A24 blob form, except each solved end of material face k adds g_k * material_face_scale[k] (null: 1; the
// joint solve passes face_mass_k / velocity_diagonal_k). Free solids are left out of the diagonal.
std::expected<FlowImbalanceReport, PressureFailure> CheckFlowImbalance(const PressureEquations&,
    const grid::CellValues<double>& pressure, const Buffer<double>& sealed_gas_pressure,
    const grid::FaceValues<double>& velocity, CoupledVelocities, const SolveSettings&, PressureBuffers&);
// The stop rule of step 21 with blob pressures, material velocities and free-solid motions given; cell rows as the
// kIncluded volume error above, blob rows as the A24 given-blob form; same report fields and limits.

// pressure_ladder.h (CPU only)                                                       NEW (A5)
// Builds the rungs of step 18 for these equations, every connection weighted by flow_per_pressure *
// connection_scale (null: 1, the ladder SolvePressure builds for itself). Free-solid terms are left out;
// material-face weights g_k are added unscaled and sealed blob connections stay reference ends, with own weight
// g_c * connection_scale (A18; A24 for a non-null scale, which no caller passes today). Rung 1 is
// numbered by (cell block, blob, merged CellId); rungs 1 and deeper and the factors stay on the processor (A15).
// Stays in buffers until the next BuildPressureLadder, BuildPressureEquations or SolvePressure on these buffers.
// Returns the rung count.
std::expected<int, PressureFailure> BuildPressureLadder(
    const PressureEquations& equations, const grid::FaceValues<double>* connection_scale, PressureBuffers& buffers);
// One pass down and up the built ladder (step 18): pressure_change, Pa, from volume_error, m3/s. Linear,
// symmetric and positive on each blob; each kSealed blob's input total is removed first (shared out by
// filled_volume) and its volume-weighted mean is removed from the result. kWrongGridVersion when the ladder was
// built for other equations. In the batched layout, used alone, it takes 4 sweeps.
std::expected<void, PressureFailure> ApplyPressureLadder(
    const PressureEquations& equations, const grid::CellValues<double>& volume_error,
    grid::CellValues<double>& pressure_change, PressureBuffers& buffers);

// carried_pressure.h (CPU only). Used when a pressure law is active; otherwise carried = committed.
// Pa. Follows the committed grid velocity back for one step from each cell centre and reads the committed
// pressure there (linear between cell centres). A cell with no committed value at that place keeps its own
// committed pressure, or fallback_pressure if it has none.
std::expected<void, PressureFailure> ComputeCarriedPressure(
    const grid::StaggeredGrid& grid, grid::CellBlockRange blocks,
    const grid::CellValues<double>& committed_pressure, const grid::CellValues<std::uint8_t>& has_committed_pressure,
    const grid::FaceValues<double>& committed_velocity, double step_duration, double fallback_pressure,
    grid::CellValues<double>& carried_pressure, PressureBuffers& buffers);

// pressure_memory_estimate.h (CPU only). ComputeBufferBytes is the sum of these rows (base/memory_count.h).
// NEW (A15). With max_loaded_blocks > 0: every row counted per cell, face or corner place that is not a pass-1
// row is returned with where = MemoryWhere::kCpu at the limits and again with where = kRunOn at
// max_loaded_blocks (the loaded piece); pass-1 rows, rungs 1 and deeper, the direct-solve factors and rows
// counted per cell block or given count stay kRunOn; plus BuildSolveBatchesMemoryEstimate. One allocation table.
// NEW (A18). Rows for material faces and sealed gas blobs from max_blocking_material_faces, max_sealed_gas_blobs.
// The sum and order buffers rows are counted for run_on (reconcile item 24).
std::vector<MemoryEstimateEntry> BuildPressureMemoryEstimate(const BufferSizes& sizes, RunOn run_on);
```

Added by the coupling decision (A18, `plan-decisions.md` "Coupling inside the pressure solve"); all NEW.

```cpp
// cell_equation_math.h (shared). Per-element formulas of section 5 C2.
// flow_per_pressure = dt * A^2 / m (m3/(Pa s)); velocity_per_pressure = dt * A / m ((m/s)/Pa).
FLUIDITY_CPU_AND_GPU ConnectionCoefficients ComputeMaterialFaceCoefficients(double blocked_area, double face_mass,
                                                                           double step_duration);
// m3/(Pa s) = gas_volume / (heat_ratio * background_pressure * step_duration).
FLUIDITY_CPU_AND_GPU double ComputeGasSqueezeAllowance(double gas_volume, double background_pressure,
                                                       double heat_ratio, double step_duration);
// m/s = velocity_before_pressure + velocity_per_pressure * (pressure_low - pressure_high).
FLUIDITY_CPU_AND_GPU double ComputeMaterialVelocityAfterPressure(double velocity_before_pressure,
                                                                 double velocity_per_pressure, double pressure_low,
                                                                 double pressure_high);
```

Added by the batched-solve decision (A15, `plan-decisions.md` "Batched solves (A15)"); all NEW. It lives in
pressure because pressure is the lowest module the three solvers share; batches only sizes it and the step
builds the solvers' buffers with it. Pressure includes no later module.

```cpp
// solve_batches.h (CPU only)
// The cut of one grid into batches for a solve whose working buffers are in CPU memory, the one loaded piece, the
// staging buffer and the copy counts. Move-only. Held inside PressureBuffers, ViscosityBuffers and GasBuffers
// built with max_loaded_blocks > 0; one-piece buffers hold none.
class SolveBatches {
 public:
  RunOn run_on() const;                          // Processor of the loaded piece (the buffers' run_on).
  std::uint32_t max_loaded_blocks() const;       // Range plus stacked border of one batch.
  std::uint32_t batch_count() const;             // For the bound grid; 0 before ApplySolveBatches.
  grid::CellBlockRange range(std::uint32_t batch) const;   // Whole-grid blocks this batch writes.
  std::uint32_t loaded_batch() const;            // grid::kNoIndex when none is loaded.
  const grid::GridPiece& loaded_piece() const;   // Precondition: loaded_batch() != kNoIndex.
};
// One allocation: loaded GridBuffers with the GridBuilderBuffers lent to them, an OrderBuffers, BorderSearchBuffers, two border buffers and the map copy at
// max_loaded_blocks, the per-batch border lists (CPU), a locked staging buffer of max_loaded_blocks * 240 * 64
// bytes. kBadInput: max_loaded_blocks 0 or above max_cell_blocks, no width; kRunOnCallFailed: as plan-decisions.
std::expected<SolveBatches, PressureFailure> BuildSolveBatches(RunOn run_on, std::uint32_t max_cell_blocks,
                                                               std::uint32_t max_loaded_blocks,
                                                               std::span<const grid::BorderWidth> stacked_widths);
std::vector<MemoryEstimateEntry> BuildSolveBatchesMemoryEstimate(RunOn run_on, std::uint32_t max_cell_blocks,
                                                   std::uint32_t max_loaded_blocks);
// Cuts `grid` (on run_on) with the largest n such that every range of grid::BuildRanges(grid, n) plus its stacked
// border holds at most max_loaded_blocks blocks (bisection; the buffer form of grid::FindBorderBlocks, one call
// per layer; each border list is read back once, 4 B per border block, counted). No n: kRunOnCallFailed with
// magnitude = blocks needed by the smallest range, limit = max_loaded_blocks. Binds the batches to grid's version.
std::expected<void, PressureFailure> ApplySolveBatches(const grid::StaggeredGrid& grid, SolveBatches& batches);
// Builds the piece of `batch` into the loaded GridBuffers (buffer form of grid::BuildGridPiece). Reads the
// cancel flag first: set gives kCancelled and loads nothing.
std::expected<void, PressureFailure> LoadSolveBatch(const std::atomic<bool>* cancel_requested, std::uint32_t batch,
                                                    SolveBatches& batches);
// Whole array -> the loaded array of the batch last loaded: every loaded block, in piece order. A whole array in
// CPU memory is gathered into the staging buffer block by block and sent with one slice WriteBuffer; a whole
// array on the run's processor is gathered by one private kernel through the piece map (counted "within").
// values_per_block: 64 (cell places), 240 (face places), 125 (corner places), 1 (cell blocks).
template <typename Value>
std::expected<void, PressureFailure> WriteBatchValues(const Buffer<Value>& whole, std::uint32_t values_per_block,
                                                      SolveBatches& batches, Buffer<Value>& loaded);
// Loaded -> whole for the range blocks only: one contiguous slice (the range is one run in both grids).
template <typename Value>
std::expected<void, PressureFailure> ReadBatchValues(const Buffer<Value>& loaded, std::uint32_t values_per_block,
                                                     SolveBatches& batches, Buffer<Value>& whole);
// Arrays counted per item with a begin offset per cell block (samples, coefficients, term lists): the items of
// every loaded block in piece order, and the loaded begin offsets (block count + 1). Returns the item count.
template <typename Value>
std::expected<std::uint32_t, PressureFailure> WriteBatchGroups(const Buffer<Value>& whole,
                                                               const Buffer<std::uint32_t>& whole_begin_of_block,
                                                               SolveBatches& batches, Buffer<Value>& loaded,
                                                               Buffer<std::uint32_t>& loaded_begin_of_block);
template <typename Value>
std::expected<void, PressureFailure> ReadBatchGroups(const Buffer<Value>& loaded,
                                                     const Buffer<std::uint32_t>& loaded_begin_of_block,
                                                     SolveBatches& batches, Buffer<Value>& whole,
                                                     const Buffer<std::uint32_t>& whole_begin_of_block);
// Connected labels over the whole grid from per-batch labels (smallest whole cell place index reachable inside
// the loaded piece). `joins` lists, for every border cell of every batch, (the label this batch gave it, the label
// its own batch gave it). Joined on the CPU, smallest label wins; final_label[l] is the final label of label l
// (CPU, max cell places). Exact: the labels equal the one-piece labels.
struct LabelJoin { std::uint32_t first = 0; std::uint32_t second = 0; };
std::expected<void, PressureFailure> JoinLabelsAcrossBatches(std::span<const LabelJoin> joins,
                                                             Buffer<std::uint32_t>& final_label);
struct SolveCopyReport {                // Since the last ResetSolveCopyReport (the step resets it per try).
  std::uint32_t batch_loads = 0;
  std::uint32_t sweep_count = 0;        // One sweep = every batch loaded once, in order.
  std::uint64_t bytes_to_processor = 0, bytes_from_processor = 0, bytes_within_processor = 0;
  double copy_elapsed_seconds = 0;
};
const SolveCopyReport& ReadSolveCopyReport(const SolveBatches& batches);
void ResetSolveCopyReport(SolveBatches& batches);
```

`solve_batches.h` details (plan decisions): a whole array is valid when its size is at least
`grid.counts().cell_blocks * values_per_block` (else `kBadInput`); the loaded array has `max_loaded_blocks *
values_per_block` values; `sweep_count` grows when batch 0 is loaded; explicit instantiations for `double`,
`std::uint8_t`, `std::uint32_t` and every streamed struct, each with `static_assert(sizeof(Value) <= 64)` (the
staging bound); nothing allocates after `BuildSolveBatches`. Stacked widths: pressure and gas `{kTouchingCells}`,
viscosity `{kTouchingCells, kTouchingCells}`.

**Exposed for the joint solve (A5).** `viscosity::SolveViscosityAndPressure` runs the loop of viscous scenes
and calls: `GeometrySnapshot` (same areas, volumes and distances, plus `cell_eighths` and `face_piece_centres`),
`PressureEquations`, `ComputeFaceInertia` (the mass that both solves share), `ComputePressurePush` with
`GivenTerms`, `ApplyPressure`, `ComputeFlowImbalance`, `ComputeVolumeError`, `ComputeOwnWeight` with
`connection_scale = face_inertia / (face_inertia + step_duration * viscous own weight)`, `ComputeStartPressure`,
`ApplyZeroMeanToSealedBlobs`, `ApplyPressureOffset`, `BuildPressureLadder` and `ApplyPressureLadder` (the ladder
as a preconditioner of its pressure corrections), and the stop rule `CheckFlowImbalance` with `SolveSettings`.
Pressure has no hook into that loop and includes no viscosity header. The step then calls `ComputeWasSolved`
on the same equations whichever solve ran. NEW (A24, replaces the A18 refusal): the joint solve runs on
equations with blocking-material faces and sealed gas blobs, and keeps each sealed gas blob as one more unknown
through the forms with a blob buffer (`ComputePressurePush`, `ComputeStartPressure`); no viscous scene reads sealed
gas one step late. NEW (A26, A27; replaces A24's elimination of material faces in the joint solve): in the joint
solve the material face velocities `w` and the free-solid motions `xi` are velocity unknowns, not eliminated (they
carry viscous terms: the liquid sticks to the material and to the solid). Viscosity gives them to the pieces as
`CoupledVelocities` and calls the `ComputeVolumeError` and `CheckFlowImbalance` forms with `CoupledVelocities`,
`ComputeCoupledPush` (the pressure impulse on each material face and free solid, for its momentum rows) and the
`ComputeOwnWeight` form with `material_face_scale` (`face_mass_k / velocity_diagonal_k` on `g_k`; free solids left
out). It no longer calls the A24 forms that eliminate material faces (`ComputeVolumeError` and `ComputeOwnWeight`
with blob buffers, the given-blob `CheckFlowImbalance`) nor `ComputeBlockingMaterialVelocities`; those stay for
`SolvePressure`'s arithmetic, and `SolvePressure` is unchanged (material faces eliminated, free solids as the
rank-one term). NEW (A15): in the batched layout each piece is one sweep (blob, material-entry and free-solid
buffers whole on the processor, A24, A26, A27), and viscosity's own passes use their own `SolveBatches` (built by
`BuildSolveBatches`).

**What the gas module uses** (second pressure regime, low Mach; a second `PressureBuffers` instance): equations
that do not depend on a liquid surface (gas fills a `GeometrySnapshot` from the liquid's, with
`content = kGas`, `filled_volume` = its gas volume, and the liquid's corrected pieces closed, then calls
`BuildPressureEquations` with `SealedBlobCheck::kWhenRatesApplied`); `ComputeBlobOfCell`; blob totals before
the rates are final (`PressureBlob::filled_volume`, `incompatible_volume_rate`, then `ApplyAddedVolumeRate`,
which makes the sealed-blob check); `reference_pressure` on open domain sides; `SolvePressure`;
`ComputeWasSolved`; `squeeze_allowance`, already present, for a later compressible extension. Nothing else in
`SolvePressure` knows whether the cells hold liquid or gas. In the batched layout the gas `PressureBuffers`
run `SolvePressure` batch by batch like the liquid's (A15). **Gas beside liquid (A10, A18)**, in the other
direction: each sealed tracked-gas blob that touches liquid and is not held at atmosphere comes in as a
`SealedGasBlob` with `sealed_gas_blob_of_cell` and `gas_pressure_above_background` and is solved with the liquid
(one extra unknown, section 5 C2), by `SolvePressure` or by the joint viscous solve (A24); its end pressure goes
back to gas in `SolvedArrays::sealed_gas_pressure` (viscosity's `SolvedCouplingArrays` in a viscous scene).
Everywhere else (open gas, slivers, held blobs, cells that touch gas) the liquid solve reads the gas's committed
pressure one step late through `gas_pressure_above_atmosphere` and `holds_tracked_gas`, so a liquid surface
beside such gas feels that pressure in place of atmosphere. `FindAirPockets` leaves tracked gas cells out of
every air pocket.

## 4. Consumes from other modules

Each item is marked EXISTS (Foundation, including the `base/` additions) or PROVIDED (by a designed module, with
the name as that module's section 3 has it, after the updates of `changes-to-apply.md`).

**Foundation (EXISTS):** `Vec3`, `Mat3`, `math::Dot`, `Axis`, `kUp`, `RunOn`, `CheckCanRunOn`,
`FLUIDITY_CPU_AND_GPU`, `cpu_only.h`.

**Foundation additions (EXISTS: `shared-interface.md` section (a) and the `base/` additions of
`changes-to-apply.md`):**

```cpp
namespace fluidity::math {  // Shared. Sum over a perfect binary tree on the indices; missing leaves count as 0.
FLUIDITY_CPU_AND_GPU double TreeSum(const double* values, std::size_t count);
double Sqrt(double); double Cbrt(double); double Abs(double); double Exp(double); double Log(double);
double Pow(double base, double exponent); double Min(double, double); double Max(double, double); }
// base/ordered_sums.h: the same tree over a whole buffer, on its processor.
std::expected<double, BufferError> ComputeTreeSum(const Buffer<double>& values, std::size_t count);
std::expected<double, BufferError> ComputeTreeDot(const Buffer<double>& first, const Buffer<double>& second,
                                                  std::size_t count);
// base/stable_order.h: the stable sort of (64-bit key, index) pairs, for building each ladder rung.
std::expected<void, BufferError> BuildStableOrder(const Buffer<std::uint64_t>& keys, std::size_t count,
                                                  const Buffer<std::uint32_t>* previous_order,
                                                  Buffer<std::uint32_t>& order);
template <typename Value> class Buffer; struct MemoryEstimateEntry;
```

**Foundation as built and the A17 additions (EXISTS; `plan-decisions.md` "Foundation `base/`", "Foundation as
built", "Grid built on the GPU (A17)"):** `MemoryEstimateEntry::where` (`MemoryWhere { kRunOn, kCpu }`) and
`ComputeMemoryBytes(rows, counts, where)`; buffers passed together on one processor as a precondition (A30, no
error value);
`BuildBuffer(..., CpuMemory::kLockedForCopies)` and the slice forms of `WriteBuffer`, `ReadBuffer`, `CopyBuffer`
(A15); `OrderBuffers`, `BuildOrderBuffers`, `BuildStableOrder` and
`ComputeExclusivePrefixSum` taking them (one form, A31) (equation indexes and rung-1 numbering from per-block counts, A15); `RunOn` in the
shared `base/run_on.h`.

**Grid (PROVIDED by grid 3.1 to 3.7).** What this design first wrote, and the grid's declaration that serves it:

| First written here | Provided by grid |
|---|---|
| `kNoCell`, `kNoFacePiece` | `grid::kNoIndex` |
| `cell_index`, `face_piece_index` as `std::uint32_t` | `grid::CellPlaceIndex`, `grid::FacePlaceIndex` (A3) |
| `CellBlockRange { first_block, block_count }` | the same (grid 3.4) |
| `CellValues`, `FacePieceValues`, `CornerValues`, `CellBlockValues` | `grid::CellValues`, `FaceValues`, `CornerValues`, `CellBlockValues` (grid 3.6) |
| `StaggeredGrid` with `cells()`, `face_pieces()`, `grid_version()` | `grid::StaggeredGrid`, `version()`, `tables(run_on)` giving `grid::GridTables` |
| `ActiveCell { cell_coord; size_level; region; }` | `ToCellId(tables, cell)`, `ToCellCoord`, `ToSizeLevel`. The region is not the grid's (see cell resize below) |
| `FacePiece { axis; cell_low; cell_high; corner[4]; area; center_distance; center; }` | `ToAxis(face)`, `FindCellLow`, `FindCellHigh`, `FindCornerPlaces(tables, face)`, `ComputeFaceArea`, `ComputeCellCenterDistance`, `ComputeFacePosition` |
| `corner_of_cell(cell, 0..7)`, `corner_position(corner)` | `FindCornerPlaces(tables, cell)`, `ComputeCornerPosition` (grid 3.7) |
| `face_pieces_of_cell(cell)` | `FindFacePieces(tables, cell)`, ascending index order |
| `cell_width(size_level)` | `ComputeCellWidth(frame, size_level)` |
| `DomainSide` | `contact::DomainSide` (contact 3.2) |
| `domain_box()` | not the grid's: the step passes `contact::DomainBox` where it is needed |

Invariants: cells are ordered by `CellPlaceIndex` in one fixed order per grid version (grid 5.1, step 4); a
corner shared by cells of two sizes has one `CornerPlaceIndex` (grid 3.7). `cell_coord` may be negative: the
ladder merges with `grid::ComputeBigCell`, which halves with a floor. Totals are taken per cell block in the
grid's fixed block order, then `math::TreeSum` over the block partials (section 5 step 20).

For `solve_batches.h` (A15, A17; PROVIDED by the grid plan's A17 section of `plan-decisions.md`):
`grid::BuildRanges`, `grid::BorderWidth::kTouchingCells`, `BorderSearchBuffers`, `BuildBorderSearchBuffers`,
`BuildBorderSearchMemoryEstimate`, the buffer form of `FindBorderBlocks`, the buffer form of `BuildGridPiece`,
`GridPiece::block_in_whole_grid_buffer()`, `BuildGridBuffers`, `BuildGridMemoryEstimate`, `grid::GridPiece`
(`grid/grid_piece.h`) and `grid::CellBlockRange` (shared `grid/grid_ids.h`). A GPU grid has no CPU tables
(A17): every pass of this module on a GPU run reads `tables(RunOn::kGpu)`.

**Particles (PROVIDED by particles 3.1, 3.3):** `particles::ParticlesTry` and
`BuildFieldsToRead(particles_try, material_index, cell_blocks)` giving `position` and `group_begin` per cell
block (this design first wrote `ParticleGroups`). "The 27 blocks around a block" are
`GridTables::same_level_blocks`, plus `bigger_block` and `smaller_blocks` where sizes change. `blob` as used
here (an air pocket; a connected body of liquid in one solve) does not collide: particles carry no blob id.

**Transfer (PROVIDED by transfer 3):** the material-subset input `std::span<const std::uint8_t>
material_is_counted` (one entry per material, A8), used here with the same meaning; `velocity_before_pressure`
on face pieces (`GridFromParticles::velocity` of the liquid material set after the force stages have changed
it); `CellsFromParticles` of the liquid material set, gas markers left out (A10), with `mass`, `rest_volume`,
`current_volume` and `ingredient_volume` per cell, and the formulas `ComputeMixtureDensity` and
`ComputeSensedVolumeRatio`. `carried_pressure` is this module's (`ComputeCarriedPressure`); the boundary between
two liquids that do not mix is this module's (`FindMaterialBoundaries`, C4).

**Contact (PROVIDED by contact 3.2):** `DomainSide`, `DomainBox`; `ComputeDistanceToSolid` and
`ComputeSolidVelocity`, which the solids module uses to fill `SolidCoverage`. `SolidSide` is not used here
(decision 7).

**Viscosity (PROVIDED by viscosity 3):** nothing is consumed. `viscosity::SolveViscosityAndPressure` (A6, same
parameters as the former `ApplyViscosity`) calls the pieces listed under "Exposed for the joint solve" and
writes the same two outputs as `SolvePressure`, plus (A24) the material velocities and sealed gas blob pressures
through its `SolvedCouplingArrays`. Viscosity stores no viscous jump at a material boundary
(viscosity 5.6): the joint solve carries the viscous stress in its velocity unknowns, so `MaterialBoundary` has
no viscous term.

**Materials (PROVIDED by materials 3):** `materials::MaterialTable::ingredient_pressure_law()`,
`ingredient_rest_density()`, `surface_tension_max()`, `surface_tension_with_air()` (one value per scene, else
`kSurfaceTensionsDiffer`), `lightest_density()`; `materials::MaterialPair::surface_tension` for
`MaterialBoundaryInputs`; `materials::ComputeCellSqueezeValues(inputs, bulk_stiffness, target_volume_ratio, buffers)`
(one form, its `MaterialBuffers` last, A31; the mixed `bulk_stiffness` of step 14 at `carried_pressure`, and `target_volume_ratio`). Liquid, gas, snow and
springy solid only (A9).

**Solids (PROVIDED by solids 3.5, 3.6):** `solids::ToSolidCoverage(const SolidCoverageValues&)` giving
`SolidCoverage` over every active cell, for every fixed, driven and free solid and every blocking material
(snow and springy solids, A9) seen by the liquid as a solid; `solids::FindGridEdgesOpenToSky` giving
`grid_edge_is_open_to_sky`; `solids::ToFreeSolidPushes(motion_map)` and `ToFreeSolidInverseInertia(motion_map)`
for `EquationInputs`. NEW (A18): `solids::ToBlockingMaterialFaces(const BlockingMaterialFaces&)` from
`solids::BuildBlockingMaterialFaces` gives `EquationInputs::blocking_material_faces`; after the solve
`solids::ApplyPressureToMaterials` reads `SolvedArrays::blocking_material_velocity_after_pressure` and writes each
velocity into its set's `GridFromParticles::velocity`. Solids calls this module's `ComputeBlockedCornerAreas` and
`ComputeEquationIndexes`.

**Gas (PROVIDED by gas 3, after its A10 update):** `gas::BuildGasGeometry` (the gas `GeometrySnapshot` with
`kGas` cells) and `gas::ComputeGasAddedVolumeRate` (`added_volume_rate` from blob totals); for the liquid solve,
two arrays bound to the try grid, filled from the committed gas amounts by `gas::ComputeGasSeenByLiquid`: `gas_pressure_above_atmosphere` (Pa,
the gas's pressure at the end of the last committed step) and `holds_tracked_gas` (1 in each cell that holds
tracked gas), passed in `EquationInputs` and `AirPocketInputs`. Null in a scene with no gas. NEW (A18):
`gas::SealedGasForLiquid` from `gas::ApplyGasStepStart` gives (A24: no coupling choice; read by either liquid solve)
`sealed_gas_blobs`, `sealed_gas_blob_of_cell` and `gas_pressure_above_background`; gas reads
`SolvedArrays::sealed_gas_pressure` back in `SolveGas` (a coupled blob's end background pressure is
`atmosphere_pressure + sealed_gas_pressure[j]`).

**Step (PROVIDED by step 3, stage list of its update; A18 calls from `plan-decisions.md`):** calls
`solids::DriveWaves` in stage `DriveWaves`, which calls `ApplyOpenSeaRelaxation`; in stage
`BuildPressureEquations` calls `solids::BuildBlockingMaterialFaces` and `gas::ApplyGasStepStart` before
`BuildPressureEquations` and passes their outputs in `EquationInputs`; in stage `SolvePressure` passes the
step-owned spans `blocking_material_velocity_after_pressure` and `sealed_gas_pressure` in `SolvedArrays` and calls
`solids::ApplyPressureToMaterials` after `ApplyPressureToSolids`; sets `max_blocking_material_faces` and
`max_sealed_gas_blobs` in the liquid's `BufferSizes`; builds every solver buffer with `max_loaded_blocks` from
its `RunPlan::max_solve_loaded_blocks` in `RunMode::kBatchedWithSolvesInPieces` (A15) and resets each
`SolveCopyReport` per try. It also calls `ComputeDistanceToSurface` (once for every liquid and once per liquid material that touches another; in a
scene with no liquid once, with an all-zero `material_is_counted`), then
`FindLiquidSurface`, `FindMaterialBoundaries` and `FindAirPockets` in stage `FindLiquidSurface`;
`ComputeCarriedPressure`, `materials::ComputeCellSqueezeValues`, `ComputeVolumeCorrectionRate` and
`BuildPressureEquations` in stage `BuildPressureEquations`; `SolvePressure` in stage `SolvePressure` when the
scene has no viscous ingredient (stage `SolveViscosityAndPressure` otherwise), then `ComputeWasSolved` in both
cases. It builds the `mixture_density` and `sensed_volume_ratio` arrays from `transfer::CellsFromParticles`;
owns `GridFields::pressure`, `has_pressure` and `pocket_of_cell` and the air-pocket list (a try copy and a
committed copy) and passes the try copies as `SolvedArrays` and `AirPocketArrays`, so "kept only on commit"
is the step's swap and needs no copy out of `PressureBuffers`; takes the smaller of
`surface_tension_step_limit` and the committed `AirPockets::step_limit` into the step duration; counts a second
`PressureBuffers` for gas with `BuildPressureMemoryEstimate`; passes `cancel_requested` to `SolvePressure`.

**Cell resize and batches (PROVIDED by the cell resize and batches design, 3 and 5.8 to 5.9):** remaps `GridFields::pressure`,
`has_pressure` and `pocket_of_cell` when cells are split or merged; fills `is_grid_only_region` (a grid-only
cell is at least two cell widths from the liquid surface) and the density and velocity of grid-only cells;
runs `ComputeDistanceToSurface` and `FindLiquidSurface` per particle batch into one-piece piece buffers and
copies the range into the whole buffers with `WriteSnapshotRange` (`batches::ReadBatch`); every other function
of this module runs on the whole grid (`blocks` = all) on the run's processor, in one piece (`RunMode::kBatched`,
`SolvePlace::kWholeGridOnProcessor`) or batch by batch through `SolveBatches` (`kBatchedWithSolvesInPieces`,
`SolvePlace::kInPieces`). `batches::BuildBatchPlan` chooses the mode and `max_solve_loaded_blocks` (A15); the CPU
fallback `kWholeGridOnCpu` is removed (A17). No pressure is given at a batch border.

## 5. Method

Symbols are used in formulas only. Ported means the maths is the one on `main`.

**A. Liquid surface (`ComputeDistanceToSurface`, `FindLiquidSurface`)**

1. *Distance (ported idea, new search).* At a point, `distance_to_surface = min over nearby particles of
   (|point - particle position|) - particle_radius`, capped at `+2 cell widths`. A point inside a
   cell marked `is_grid_only_region` gets `-2 cell widths`. The minimum is exact in any order, so no ordering rule is
   needed. Particles are found through the 27 blocks around the point's block, not by scanning all particles
   (report CUDA 10.1). The analytic sphere and flat plane of `main` become test inputs, not solver paths.
   Only particles of materials with `material_is_counted` are read: liquids, never gas markers (A10) or
   blocking materials, which reach this module through `SolidCoverage`.
1b. *Boundary between two liquids that do not mix (NEW, C4 option A; `FindMaterialBoundaries`).* For each
   liquid material that touches another, the same distance with that material's particles only (its own
   field in `PressureBuffers`). Between materials `a < b`, the boundary distance is
   `(distance_a - distance_b) / 2`, negative on `a`'s side, at corners and centres. A face piece between two
   `kLiquid` cells whose two centre values differ in sign is crossed: `low_share = d_low / (d_low - d_high)`
   of the boundary distance at the two centres, raised to at least 0.05 like `liquid_share`;
   `surface_curvature` from the boundary distance as in step 6, at the low cell, signed so that it is
   positive when the low liquid bulges out; `surface_tension` from the material pair. The sharp boundary is
   delivered (plan decision; no exclusion). With a material index and `surface_tension_max > 0`, every cell whose
   centre distance is under two cell widths in magnitude, on both sides of that field's zero, gets curvature
   samples (plan decision).
2. *Liquid cells (ported).* A cell is `kLiquid` when its centre distance is negative and its `filled_volume`
   is positive.
3. *Volumes (ported, generalised).* The cell is cut into six tetrahedra that share the diagonal from corner 0
   to corner 7 (`{0,1,3,7} {0,1,5,7} {0,2,3,7} {0,2,6,7} {0,4,5,7} {0,4,6,7}`, corner bit 0 = x, 1 = y, 2 = z).
   Inside a tetrahedron both `distance_to_surface` and `distance_to_solid` are linear in the corner values;
   `filled_volume` is the part with liquid distance `< 0` and solid distance `> 0`, `air_volume` the part with
   both `> 0`. A tetrahedron whose four liquid distances are all zero holds no liquid. New: the solid is any
   sampled field, not one plane.
3b. *Cell eighths and liquid-covered centres (NEW, C3 option A; only with a viscous ingredient).* The liquid
   part of each tetrahedron (convex: both distances are linear in it) is cut by the cell's three mid-planes;
   each piece's volume goes to the eighth it lies in, so the eight values partition `filled_volume` and add
   to it within round-off. `liquid_covered_solid_centre` and `liquid_covered_area`: centroid and area of the
   zero set of `distance_to_solid` inside the tetrahedra, kept where liquid distance `< 0`.
   `liquid_covered_open_centre`: centroid of the part of the face piece's two triangles with solid distance
   `> 0` and liquid distance `< 0`. The same clipping formulas as steps 3 and 4 (`geometry_clip_math.h`).
4. *Areas, single source (ported, generalised).* A face piece is two triangles split along the diagonal the
   tetrahedra induce. `open_area` = part with solid distance `> 0`; `air_open_area` = part of it with liquid
   distance `> 0`. `solid_volume_rate = (area - open_area) * solid_normal_velocity`. Each face piece is clipped
   from its own four corners, so a big face is the sum of its pieces by construction. No other function in the
   solver computes an open area or a solid flow; equations, correction, check, viscosity and coupling copy them.
   The solids module shares the blocked part out to the four corners with `ComputeBlockedCornerAreas`, from
   the same two triangles, so its corner areas add to `area - open_area`.
5. *Pressure distance (ported).* `kLiquidToLiquid`: `pressure_distance = center_distance` (half of each cell's
   width, also across a size step). `kLiquidToAir`: `liquid_share = d_liquid / (d_liquid - d_air)` from the two
   centre distances, raised to at least 0.05 (counted in `lengthened_distance_count`);
   `pressure_distance = liquid_share * center_distance`.
6. *Curvature (ported).* For each liquid cell touching air, with the 19 samples at spacing `w` (the cell's
   width): gradient `g` and second derivatives `H` by centred differences;
   `surface_curvature = sum over axes of (H_aa - n_a (H n)_a) / |g|`, `n = g / |g|`. `|g| < 1e-8` gives
   `kSurfaceDirectionUnclear`. New: the samples are taken at the cell's own width even beside a size step.
   This is the only curvature the solver uses.
7. *Step limit (ported).* `sqrt((rho_low + rho_high) * w^3 / (4 pi sigma))` over surface cells, smallest wins.
8. *A thin solid* leaving liquid on both sides in one `kLiquid` cell gives `kCellSplitBySolid` (ported limit).

**B. Air pockets (`FindAirPockets`)**

9. *Openings (fixed).* Two air cells are joined when their shared face pieces have `air_open_area > 0`. An
   opening on the domain's top side, or through a grid-edge face piece marked `grid_edge_is_open_to_sky`,
   leads to the atmosphere. An opening into any other cell that is not allocated is an unknown frontier. New:
   the mark comes from a flood, from the open top, through space that has no cell block and touches no solid,
   so that block allocation alone never decides between vent and wall (report numerics 4.1). The mark is
   stored on the grid-edge face piece because the grid can index only blocks that exist (this design first
   wrote a per-block `block_is_open_to_sky`). NEW (A10): a cell with `holds_tracked_gas` is not an air cell
   here: it joins no pocket (`pocket_of_cell = kNoAirPocket`) and an opening into it leads to the gas, whose
   pressure the liquid reads in step 13. One cell is never both an air pocket and tracked gas.
10. *Classes (ported).* Pocket label = smallest `cell_index` it contains. `kVented` if any opening leads to
    the atmosphere; else `kUnknownFrontier` if any leads to an unknown frontier; else `kEnclosed`.
11. *History (ported).* Overlap of an old and a new pocket = sum over shared cells of the smaller `air_volume`.
    A tracked pocket shares out its gas amount `rest_volume * rest_pressure^(1/heat_ratio)` in proportion to
    its overlaps; shares add. `blob_id` goes to the largest-overlap resolved enclosed receiver; ties: larger
    overlap, then lower old `blob_id`. Gas reaching a vented, too small or no pocket is recorded in
    `BlobHistory`; reaching an unknown frontier gives `kAirPocketFateUnknown`. A newly enclosed resolved pocket
    with no gas starts at `atmosphere_pressure` (old paper line 309).
12. *Pressure (ported).* `pressure = rest_pressure * (rest_volume / air_volume)^heat_ratio`;
    `pressure_above_atmosphere = pressure - atmosphere_pressure`. Only enclosed resolved pockets use it; the
    others are at atmosphere. Resolved: `cbrt(3 air_volume / (4 pi)) >= min_radius_in_cells * w`, `w` the widest
    cell in the pocket. Step limit `0.5 * radius * sqrt(rho / (3 heat_ratio pressure))`.

**C. Equations (`BuildPressureEquations`)**

13. *Per connection (ported).* `face_density = (rho_low * d_low + rho_high * d_high) / (d_low + d_high)` with
    `d` the part of `pressure_distance` in each cell (half widths; or `low_share` at a material boundary; the
    liquid cell's density alone at an air end). `velocity_per_pressure = dt / (face_density * pressure_distance)`,
    `flow_per_pressure = open_area * velocity_per_pressure`. Air end:
    `reference_pressure = p_air + sigma * surface_curvature`, `pressure_jump = 0`, so surface tension is
    counted once. `p_air` is, in this order (air-end precedence, A18): the sealed gas blob `j` of the air cell
    (`sealed_gas_blob_of_cell != kNoBlob`): the connection becomes a blob connection, other end `p_j +
    reference_pressure` with `reference_pressure = sigma * curvature + gas_pressure_above_background` (step C2
    (b)); else `gas_pressure_above_atmosphere` of the air cell when it has `holds_tracked_gas` (A10: open gas,
    slivers, held blobs and cells that touch gas, A24; the gas solve runs after the liquid's, so this is the pressure
    committed at the end of the last step, one step late); else the enclosed resolved pocket's
    `pressure_above_atmosphere`; else 0 (atmosphere; also beyond a grid edge, and a grid-edge face piece with
    open area has air at atmosphere on its other end). Material boundary: `pressure_jump = - sigma *
    surface_curvature` (no viscous term: the joint solve carries the viscous stress). A `kLiquid` cell with
    `mixture_density = 0` (no particle in it, Q32) takes the value of the touching `kLiquid` cell with mass and
    the lowest `CellPlaceIndex`, else `lightest_density`.
14. *Per cell.* `squeeze_allowance = filled_volume / (bulk_stiffness * dt)`; for a mixture
    `1 / bulk_stiffness = sum over ingredients of volume share / stiffness of that ingredient` at
    `carried_pressure` (new for mixtures). With `s = +1` for the low cell of a piece and `-1` for the high cell:
    `flow_imbalance_before_pressure = - sum over pieces of s * (open_area * velocity_before_pressure
    + solid_volume_rate)`.
15. *The cell equation* (method paper `<eq:pressure>`), terms added in this order: own term, then connections
    by ascending `face_piece_index`:
    `squeeze_allowance * p + sum of flow_per_pressure * (p - p_other) = added_volume_rate
    + flow_imbalance_before_pressure + squeeze_allowance * carried_pressure
    - sum of s * flow_per_pressure * pressure_jump`,
    with `p_other = reference_pressure` at an air end. A `kGas` cell has the same equation as a `kLiquid`
    cell. Free solids add
    `dt * push^T * inverse_inertia * push` over their `FreeSolidPush` entries (one motion map and its
    transpose; replaces the piston and box paths), applied as `dt * (f_i . invM F + t_i . invI T)` with `F`, `T`
    summed over the solid's pushes in their sorted order (plan decision). Order of one cell row (A18): own term,
    connections by ascending `face_piece_index`, material faces by ascending (`face_piece_index`,
    `material_set`), then blob terms by ascending `face_piece_index`.
16. *Blobs and the free constant (fixed).* Blobs are connected sets of equations (label = smallest
    `cell_index`), joined through connections, through cells pushing the same free solid (plan decision),
    through material faces with two solved ends and through cells joined to one sealed gas blob (A18).
    `kHasReferencePressure` if any connection has a reference end (a material face with one solved end counts
    as one, A18); else `kCanBeSqueezed` if any `squeeze_allowance > 0` or the blob is joined to a sealed gas blob
    (`S_j > 0` makes it definite, A18); else `kSealed`. Blob totals by the per-group rule of step 20. For a sealed blob `incompatible_volume_rate` = sum of its right-hand
    sides. If `|incompatible_volume_rate| > max_volume_error_rate * blob filled_volume` the build fails with
    `kSealedBlobVolumeMismatch` and reports the rate; with `SealedBlobCheck::kWhenRatesApplied` the build only
    records the rate and `ApplyAddedVolumeRate` makes the same check after replacing the rates (gas: the
    moving liquid boundary changes a sealed gas blob's volume, and gas gives the matching rates only once it
    knows the blob totals). Below the limit the solve works on the part the blob can
    absorb, and the rest stays visible in `FlowImbalanceReport` because the check never subtracts it. No cell
    is pinned. After the solve `pressure_offset` shifts the blob so that its volume-weighted mean pressure
    equals the volume-weighted mean of `carried_pressure` (decision 1). `ComputeEquationIndexes` gives the
    equation order of this step from the snapshot alone; `ComputeBlobOfCell` maps cells to blobs.
17. *Volume correction (ported).* `rate = clamp((ln target_volume_ratio - ln sensed_volume_ratio) /
    correction_duration, +-max_rate)`, zero in cells marked `is_grid_only_region`. In a sealed blob a shift `lambda` is found by
    64 bisections so that `sum of filled_volume * clamp(raw - lambda, +-max_rate) = 0`.

**C2. Coupling inside the solve (NEW, A18; method decisions (a) to (f) of the pressure plan's Task 44).** A
lagged push is unstable when the liquid's added mass exceeds the body's (Banks, Henshaw and Schwendeman 2014;
`literature-coupling.md`); snow is lighter than the water it displaces, and an enclosed gas pocket is a stiff
spring. Both therefore enter this solve. `s = +1` for the low end of a piece; `flow_imbalance` is inflow minus
outflow.

- (a) *Blocking-material face* `k` on piece `f`, ends `a` (low) and `b` (high): a two-point term of weight
  `g_k = dt A_k^2 / m_k` (m3/(Pa s); `A_k` = `blocked_area`, m2; `m_k` = `face_mass`, kg; velocity per pressure
  `dt A_k / m_k`, (m/s)/Pa; `ComputeMaterialFaceCoefficients`). An end is solved when its cell has an equation;
  an unsolved end's pressure is 0 whatever the cell holds (as in solids' motion map: gas and air-pocket pressure
  push no solid). Each solved end `e` gets own weight `+= g_k` and `flow_imbalance_before_pressure += -s_e * (A_k
  * velocity_before_pressure - coverage_volume_rate)` (the face's flow `A u - coverage_volume_rate` replaces the
  coverage part); two solved ends get the off-diagonal `-g_k`. A face between two equation cells joins their
  blobs; one with an unsolved end is a reference end. This is the split of Fang et al. 2020: the material's
  internal force first (`ApplyForces`), then one solve that holds its inertia; tangential motion stays each set's
  own. No solved end, `face_mass <= 0` or `blocked_area <= 0`: `kBadInput`.
- *Velocity after the solve:* `u_k = ComputeMaterialVelocityAfterPressure(velocity_before_pressure, dt A_k / m_k,
  p_low, p_high) = u + (dt A_k / m_k) (p_low - p_high)`, unsolved end at 0; its flow `A_k u_k` counts in the
  volume error of each solved end. Written to `SolvedArrays::blocking_material_velocity_after_pressure` (input
  order) after the check passes; solids writes it into the set's grid velocity.
- (b) *Sealed gas blob* `j`, one unknown `p_j` (Pa above atmosphere): squeeze allowance `S_j =
  ComputeGasSqueezeAllowance = V / (heat_ratio p0 dt)` (m3/(Pa s); the papers' `V / (gamma p dt^2)` with the row
  divided by `dt`), carried value `c_j = pressure_above_atmosphere`. Its connections are the `CellConnection`s
  whose air cell has `sealed_gas_blob_of_cell == j`, `g_c = flow_per_pressure`, other-end pressure `p_j + ref_c`,
  `ref_c = sigma * curvature + gas_pressure_above_background(air cell)`. Row: `S_j p_j + sum_c g_c (p_j - p_c) =
  r_j`, `r_j = inflow_j + heat_volume_rate + boundary_volume_rate + S_j c_j - sum_c g_c ref_c`, `inflow_j = sum_c
  -s_jc (open_area_c u_c + solid_volume_rate_c)` from `velocity_before_pressure` (`s_jc = +1` when the air cell is
  the piece's low end). The unknown is eliminated exactly: `D_j = S_j + sum_c g_c > 0`, `p_j = (r_j + sum_c g_c
  p_c) / D_j`. Each cell row with blob connections keeps `g_c` in its own weight (as for any air end, counted
  once), its right side gets `g_c ref_c + g_c r_j / D_j`, and the operator subtracts `(g_c / D_j) * sum_c' g_c'
  p_c'` (one rank-one term per blob, like the free-solid term; the blob's sum by the per-group rule of step 20).
  Cells joined to one gas blob are one liquid blob, `kCanBeSqueezed` unless it has a reference end.
- *Recovery and row check:* after the solve `p_j` from the per-group sum of `g_c p_c`; written to
  `SolvedArrays::sealed_gas_pressure`. Each blob row is checked like a cell: `|inflow_after + heat + boundary -
  S_j (p_j - c_j)| <= max_volume_error_rate * gas_volume`, reported as `max_sealed_gas_volume_error_rate`, not
  counted in `net_volume_change`.
- (c) *Air-end precedence* (step 13): sealed gas blob, then `holds_tracked_gas` (one step late), then enclosed
  resolved pocket, then 0.
- (d) *Symmetry and positiveness.* Every material term is `g (e_a - e_b)(e_a - e_b)^T` or `g e e^T` with `g > 0`;
  the blob elimination is the Schur complement of a symmetric positive semidefinite system with a positive
  pivot. Row order: step 15.
- (e) *Ladder:* a material face with two solved ends adds `g_k` to the pair weight of its piece (a pair is created
  if the piece has no connection); one solved end adds `g_k` to own weight; blob connections stay reference ends
  (own weight `g_c`) and the rank-one term is left out, as for free solids.
- (f) *Not in the solve:* the material's own volumetric stiffness (half of the volumetric response in Fang et al.
  2020). `materials::ApplyInternalForce` already applies the full stress; a squeeze allowance for the material
  would count it twice (section 11, Open).
- *The joint viscous solve (A24, decision 16; A26, A27, decision 17).* Equations holding either term are solved by
  `SolvePressure` (as above: material faces eliminated, free solids as the rank-one term) or by
  `viscosity::SolveViscosityAndPressure`. In the joint solve nothing is eliminated except as stated here: each
  material face velocity `w_k` (`BlockingMaterialFace` entry) and each free-solid motion `xi = (V, Omega)` is a
  velocity unknown of the viscous velocity solve (the liquid sticks to the material and to the solid, so they carry
  viscous terms and the elimination `g_k = dt A^2 / m` and the rank-one term do not apply). The pieces see them as
  given velocities (`CoupledVelocities`): a solved end `e` of face `k` gets `-s_e (A_k w_k - coverage_volume_rate)`
  in its volume error (`kLeftOut`: `-s_e A_k w_k`); a cell with `FreeSolidPush` entries gets `-push_e . (xi -
  xi_before)` per entry (`kLeftOut`: `-push_e . xi`), the sign that gives step 15's free-solid term `+dt push^T
  M^-1 push` in `SolvePressure`'s operator; `ComputeCoupledPush` gives viscosity the pressure impulses `dt A_k (p_low - p_high)`
  and `dt sum_e push_e p_e` for the momentum rows; the outer-loop diagonal adds `g_k * face_mass_k /
  velocity_diagonal_k` per solved end and leaves free solids out. `xi_before` comes from
  `EquationInputs::free_solid_motion_before_pressure` (plan decision). The method is viscosity's (viscosity design
  5, "Coupled terms"; `plan-decisions.md` A26/A27). ~~The joint solve eliminates each material face exactly as
  above (no viscous term at the face: the material's velocity is read by no deformation-rate sample), with `g_k`
  added to its outer-loop diagonal per solved end not multiplied by `connection_scale`.~~ (Replaced by A26.)
  It keeps each sealed gas blob as one more unknown `q_j` instead of eliminating it: a liquid
  face's velocity there answers its end pressures through the viscous velocity solve, so no two-point weight exists.
  The blob's row is (b)'s row with the solved velocity, `E_j = inflow_j(u) + heat_volume_rate +
  boundary_volume_rate - S_j (q_j - c_j)`; its diagonal `S_j + sum_c g_c * connection_scale_c`; its start value
  `c_j`. The forms with a blob buffer and `CoupledVelocities` (section 3) serve it (`ComputeBlockingMaterialVelocities`
  is `SolvePressure`'s only). Everything not sealed (open gas, gas above a free surface that
  is not sealed, slivers, held blobs, cells that touch gas) keeps reading gas one step late through step 13, which
  is harmless where the liquid dominates.

**D. Solve (`SolvePressure`)**

`SolvePressure` runs in scenes with no viscous ingredient and in the gas solve. In a scene with a viscous
ingredient, `viscosity::SolveViscosityAndPressure` runs its own loop (A5) over the pieces of section 3: the
ladder of step 18 (through `BuildPressureLadder` with its `connection_scale` and `ApplyPressureLadder`), the
push and correction of step 21, the volume error and the stop rule of step 21 (`CheckFlowImbalance`), and
steps 16 and 20 unchanged. With blocking-material faces, free solids or sealed gas blobs it carries them (A24,
A26, A27, C2 last item): the forms with a blob buffer and `CoupledVelocities`, `ComputeCoupledPush` and the
`ComputeOwnWeight` form with `material_face_scale`.

18. *What is ported:* conjugate gradient on the cell equations, each iteration corrected by one pass down and
    up a ladder of coarser equation sets (papers: multigrid V-cycle); merging cells into coarser cells with
    piecewise-constant weights; coarse coefficients as plain sums of the fine ones; one damped local pass
    (weighted Jacobi, weight 2/3) on the way down and one on the way up; a direct solve (Cholesky) of each
    blob once it has at most 32 coarse cells; a restart from the freshly evaluated equation error whenever the
    running error says "done". *What changes:* the merge rule follows size levels (rung `n` merges the sibling
    cells of the smallest size level still present into their big cell, then continues with `cell_coord / 2`
    on level 0, using `grid::ComputeBigCell`, which halves with a floor; a merge never crosses a blob); so
    the pairing fallback of `main` is not needed, also for negative coordinates; all vectors are `double`; sealed blobs keep all their cells and the correction pass
    removes each sealed blob's mean, with one cell grounded only inside the 32-cell direct solve.
    *Ladder layout, both modes (A15).* Rung 0 is the equations (with `connection_scale` when given; free-solid
    terms left out, material faces in, blob connections as reference ends), streamed in the batched layout.
    Rung 1 is built per cell block: each coarse cell's members lie in one block (siblings, or a level-0 2 x 2 x 2
    group, both inside one block because blocks are 4 cells wide and aligned), numbered by (block, blob, merged
    `CellId`) through per-block counts and `ComputeExclusivePrefixSum`, no grid-wide sort. Rungs 2 and deeper
    are built from rung 1 on the run's processor, whole, by the rule above (blob, then merged id; two passes of
    `BuildStableOrder`). Rung 0 is never terminal: a blob of at most 32 equations passes one to one into rung 1,
    is not smoothed at rung 0, and gets its direct solve there (the same mathematics). Direct-solve room:
    `max_direct_solve_values` (one piece `32 * max_cells`, batched layout `4 * max_cells`); terminal blobs take
    room in ascending first-cell order, and a blob beyond the room gets one weighted-Jacobi pass down and one up
    at its terminal rung instead, counted in `SolveResult::blobs_without_direct_solve`. `ApplyPressureLadder`
    first removes each sealed blob's input total, shared by `filled_volume` (the transpose of the mean removal).
19. *Iteration.* `error = right side - equations(p)`; `z = ladder(error)`; `rz = dot(error, z)`;
    direction `d = z + (rz / rz_before) d`; `q = equations(d)`; `dq = dot(d, q)`; `dq <= 0` or `rz <= 0` gives
    `kEquationsNotPositive`; `p += (rz/dq) d`; `error -= (rz/dq) q`. A progress read happens before the first
    iteration and then every `iterations_per_progress_read` iterations; at each the CPU also reads
    `cancel_requested`; when it is set the solve stops with `kCancelled` (plan decision). The conjugate
    gradient's right side for a sealed blob has its total removed, shared by `filled_volume` (the check uses the
    true right side); when the running error passes the stop rule the solve corrects the velocity and runs the
    check, and restarts from the true error if the check fails (plan decision).
    *Arithmetic of one iteration, both modes (A15).* With `A_local` = own term, connections and material faces:
    `q_local = A_local d`; per free solid `P_s = push_s . d` and per sealed gas blob `G_j = sum_c g_c d_c`
    (per-group rule); `k_s = dt * inverse_inertia_s * P_s`, `h_j = G_j / D_j`; `dq = d . q_local + sum_s P_s . k_s
    - sum_j G_j h_j`; wherever `q` is used, `q_c = q_local_c + sum over c's push entries of push . k_s - sum over
    c's blob terms of g_c h_j`. In the ladder's sealed-blob steps `E_b` is the blob's error total and `m_b` its
    volume-weighted mean of the ladder's raw output `z_raw`: `z = z_raw - m_b`, `rz = error . z_raw - sum_b m_b
    E_b`. A one-piece solve is this arithmetic with one batch.
20. *Order of sums (A15, both modes).* Every dot product, norm and report total is one partial per cell block
    (element products in ascending index), then `math::TreeSum` over every block partial in the grid's fixed
    block order; in the batched layout the partials of a batch's range blocks go into a whole per-block array on
    the run's processor and the tree sum runs once after the sweep. *Per-group totals* (a blob's, sealed gas
    blob's, free solid's push sums, an air pocket's volume, gas blob sums): per cell block, the group's cells (or
    connections) of that block in ascending index; then, per group, `math::TreeSum` over its block partials in
    ascending block index. This replaces "one thread per blob over its cells or connections in ascending index".
    Every per-cell sum (equation, merge into a coarse cell) is done by one thread, own term first, then members
    in ascending index. No sum depends on thread, block, device size or batch cut.
21. *Correction and check* (method paper `<eq:correct>`).
    `velocity_after_pressure = velocity_before_pressure - velocity_per_pressure * (p_high - p_low - pressure_jump)`,
    with `reference_pressure` at a missing end, written into the step's `velocity_after_pressure`; a piece
    with no connection takes `velocity_before_pressure`. Then, from that velocity only:
    `volume_error = flow_imbalance + added_volume_rate - squeeze_allowance * (p - carried_pressure)` (m3/s).
    Accepted when every cell has `|volume_error| <= max_volume_error_rate * filled_volume`, and, if any jump or
    surface tension term is non-zero, `norm2(volume_error) <= jump_error_fraction * norm2(right side)`.
    `net_volume_change = dt * sum`, `total_volume_change = dt * sum of |.|`. The limit of `main` in absolute
    `m3/s` is dropped (decision 3); this one rule is also the stop rule of the joint solve. Not accepted after
    `max_iterations`: `kNotConverged`. With the A18 terms the check recomputes each material velocity and each
    `p_j` from the pressure passed, counts each material face's flow in its solved ends, and checks each blob row
    (step C2); `SolvePressure` writes material velocities and `p_j` only after it passes.

**E. Open sea (`ApplyOpenSeaRelaxation`, ported).** Zone position `x` from 0 at the inner edge to 1 at the domain
side; `ramp = x^2 (3 - 2x)`; `velocity = lerp(velocity, target, 1 - exp(-ramp * pull_rate * dt))`; target = the
wave's velocity on the open-sea side (second-order wave in finite depth with a return current) and zero at the
outlet. No liquid leaves through either side; both stay closed in the equations.

**F. The solve in pieces (NEW, A15; `plan-decisions.md` "Batched solves (A15)", out-of-core form, option (i)).**
Used when the buffers are in the batched layout (`RunMode::kBatchedWithSolvesInPieces`). Rule: state and
geometry stay whole on the run's processor; solver buffers go to CPU memory.

- *What stays whole on the processor:* the pass-1 rows (corner and centre distances, `CellGeometry`,
  `FacePieceGeometry`, curvature samples, one-material distances, `CellEighthGeometry`, `FacePieceCentre`), the
  step's solve inputs (velocity before pressure, the stage-10 cell arrays, coverage, motion map, blocking-material
  faces, free-solid entries), `blob_of_cell`, `pocket_of_cell`, rungs 1 and deeper with their factors, the blob,
  pocket, free-solid and gas-blob lists, `D_j` and `r_j`, every per-block partial, per-group total and progress
  record. *What is copied per sweep:* every other per-place row (`CellEquation` and equation of cell,
  connections and their lists, the material-face and blob-term lists, rung-0 weights, solve vectors,
  `volume_error`, air-pocket and label working buffers), streamed through the loaded piece: whole -> loaded with
  `WriteBatchValues` / `WriteBatchGroups`, range blocks back with `ReadBatchValues` / `ReadBatchGroups`.
- *Sweeps.* A sweep loads every batch once, in order (`LoadSolveBatch`, the cancel flag read before every load).
  Per iteration: A (update `p` and `error` with the full `q`; partials of `E_b` and the largest error per block),
  B (remove `E_b`; weighted Jacobi down at rung 0; residual; gather into rung 1), the whole rungs on the
  processor, C (spread from rung 1; Jacobi up gives `z_raw`; partials of `error . z_raw` and of each sealed blob's
  volume-weighted `z_raw`), D (`z`, `d`, `q_local`; partials of `d . q_local`, `P_s`, `G_j`). With no `kSealed`
  blob A and B are one sweep: **3 sweeps per iteration, 4 with a sealed blob.** Before the iterations: the
  equation build (about 5 sweeps: counts, rows, labels, relabel and classes, rung 1) and, when a sealed blob
  exists, 64 sweeps of two cell arrays for the volume correction's bisections; after: correction and check (2
  sweeps); a restart costs 2. The step sizes and the stop test run on the processor between sweeps.
- *Sums:* block partials then one tree sum (step 20), so a sum does not depend on the cut. *Labels* (blobs, air
  pockets, gas blobs): propagation inside each loaded piece (smallest whole cell place index reachable within
  the piece), `JoinLabelsAcrossBatches` over the border cells, then one relabel sweep: equal to the one-piece
  labels. *Equation indexes:* per-block counts, `ComputeExclusivePrefixSum` over blocks, then local. *Ladder
  numbering by block* (step 18), so rung 1 is the same for every cut. *Material boundaries* per batch,
  concatenated in batch order (face-piece order).
- *Free solids, sealed blobs, material faces:* entries, inverse inertias, `D_j`, `r_j` and the lists stay whole;
  each cell's push entries and blob terms are streamed with its row; `P_s`, `G_j` by the per-group rule; `p_j`
  recovered from the per-group sum of `g_c p_c`. A material face is stored with the rows of its two cells like a
  connection; its solved velocity is written by the batch whose range stores its face place (one contiguous run,
  the input being sorted by `face_piece_index`).
- *Failures, cancel, outputs.* A sweep runs every batch and keeps the smallest failure key (step, cell, face
  piece; an integer minimum), so the failure is the one-piece failure. `SolvedArrays` and `AirPocketArrays` are
  written only in the last write-back sweep after acceptance; on any failure the outputs keep their bytes.
- *Iteration count and copies:* the one-piece count (same preconditioner except the direct-solve room); about
  250 B per cell streamed per sweep (16 kB per full block), 50 to 65 kB per block per iteration with the border;
  copies are serial (no overlap with compute). `SolveCopyReport` counts loads, sweeps and bytes per direction.
- *Agreement:* one arithmetic for both modes, so a batched solve normally gives the one-piece bytes and does not
  depend on the cut; the test is agreement within the solver's tolerance (section 6), because the direct-solve
  room of the batched layout is smaller. Batched against batched with another cut, and a repeat: identical bytes.
- *Rejected, option (ii):* overlapping pieces with a coarse correction inside the outer conjugate gradient
  (textbook: additive Schwarz): two to four times the iterations, a different preconditioner, more copies per
  one-piece iteration. *Revisit when* a measured breaking-wave solve needs more than 60 iterations per try, copy
  time exceeds 80 % of solve time, or the whole rungs plus factors exceed a quarter of the GPU budget.

**Per material.** Several liquids and mixtures: `mixture_density`, mixed `bulk_stiffness` (all liquids are in the
one liquid material set, A8); liquids that do not mix: `MaterialBoundary` from step 1b (delivered, plan
decision). Viscous liquids: the snapshot's cell eighths and centres, and the joint solve of viscosity. Snow and
springy solids (blocking materials): `SolidCoverage` for their blocked area, and, on each piece where their set
has face mass beside a `kLiquid` cell, a `BlockingMaterialFace` that puts their inertia in the solve (A18, step
C2; also in the joint viscous solve, A24); a piece whose set has no face mass stays one-way in
`solid_volume_rate`. Cut and driven solids:
`SolidCoverage` only. Free solids: `FreeSolidPush`. Gas: its own equations through the interface of section 3
(`kGas`, `SealedBlobCheck`, `ComputeBlobOfCell`); beside liquid, a sealed blob as one unknown of the liquid
solve, viscous or not (step C2, A24), other gas read one step late at the liquid's air ends (step 13). Known open physics carried as
is: a driven solid pushes through the whole blocked area of a surface cell (the "piston receiving volume" item
of the programme, a known-open test).

## 6. CPU and GPU

**Shared formula headers:** `geometry_clip_math.h` (tetrahedron and triangle clipping), `surface_tension.h`,
`pressure_law.h`, `air_pocket_math.h`, `cell_equation_math.h` (connection coefficients, one cell's equation
sum, velocity correction, volume error; A18: material-face coefficients, gas squeeze allowance, material velocity
after pressure), `open_sea_math.h`, `math::TreeSum`.

**CPU.** Plain loops over cells, then face pieces, of the block range; each destination value is written by
one loop iteration, so the loops over destinations may be split across threads without changing a result.
The CPU solve is the same algorithm with the same sums in the same order.

**GPU arrays** (in `PressureBuffers` unless marked step, all stay on the device across the stages of a try):

| Array | Element | Count |
|---|---|---|
| corner distances (liquid, solid) | `double` | corners |
| one-material liquid distances (C4), corners and centres | `double` | `max_boundary_materials` x (corners + cells) |
| `CellGeometry`, `FacePieceGeometry` | structs above | cells, face pieces |
| `CellEighthGeometry`, `FacePieceCentre` (C3, viscous scenes only) | 96 B, 24 B | cells, face pieces |
| `MaterialBoundary` | struct above | crossed face pieces |
| `CellEquation`, `CellConnection` | structs above | equations, connections |
| connection list per equation | `uint32` offsets and (connection, other equation) pairs | equations + 1, 2 x connections |
| ladder rungs: own weight, connection weight, member list per coarse cell, coarse cell of each cell | `double`, `double`, `uint32`, `uint32` | under 8/7 of rung 0 in total for uniform cells |
| direct-solve blocks | `double` factor, at most 32 x 32 each | terminal blobs |
| solve vectors `p`, error, `z`, `d`, `q`, products | `double` | 6 x equations |
| `volume_error` | `double` | cells |
| `pressure`, `velocity_after_pressure` (step: try copies passed as `SolvedArrays`) | `double` | cells, face pieces |
| blob labels, `PressureBlob` | `uint32`, structs | cells, blobs |
| `BlockingMaterialFace` with `g` and velocity per pressure, per-equation material lists, ladder pairs (A18) | struct + 16 B; 2 x 12 B; 16 B | `max_blocking_material_faces` |
| `SealedGasBlob` with `D`, `r`, blob sum; blob of connection (A18) | struct + 24 B; `uint32` | `max_sealed_gas_blobs`; face places |
| batched layout (A15): per-place rows above in CPU memory, loaded copies at `max_loaded_blocks`, `SolveBatches` (loaded grid, border search, staging `max_loaded_blocks * 240 * 64` B) | as above | section 3 memory estimate entries |
| `pocket_of_cell`, `AirPocket` list (step: try copies passed as `AirPocketArrays`) | `uint32`, structs | cells, pockets |
| progress record | iteration, finished flag, `PressureFailure`, `rz`, `dq` | 1 |

**Division of work.** One GPU function per parallel phase, launched over as many blocks as the element count
needs: clip cells; clip face pieces; cell eighths and centres (viscous scenes); curvature; label propagation
(each cell takes the smallest label among joined neighbours, then pointer jumping, until a pass changes nothing; the result is the smallest index of
each set whatever the schedule); build connections; build each rung (sort by merge key, one thread per coarse
cell sums its members in order); apply equations (one thread per cell reads its connection list); local pass;
merge down as a gather (one thread per coarse cell reads its fixed member list; no scattered additions);
spread up (one thread per fine cell reads its coarse cell); direct solve (one thread per block); `TreeSum`
(one block per 1024 aligned elements, repeated on the partial sums); per-block group partials, then one thread
per group sums its block partials (A15); material lists and each blob's connection list by
`BuildGroupsByDestination`, each cell gathers (A18); vector updates; velocity correction per
face piece; volume error per cell from its face pieces in fixed order. The step sizes `rz/dq` are computed and
kept on the device; after convergence, failure or cancel every function returns at once on the finished flag. The
first failure is the one with the smallest stage number, then the smallest cell or face piece index (an
integer minimum, exact).

**Returned to the CPU:** the progress record every `iterations_per_progress_read` iterations; at the end
`FlowImbalanceReport`, the `PressureBlob` and `AirPocket` lists and the snapshot's two scalars. Fields are read
back only through explicit snapshot calls of the grid arrays. Not ported: the one-block solve, the FP32
vectors, the small-system crossover (a rung that fits one block already launches one block).

**Block ranges.** Every function takes `CellBlockRange`. Border cells are one layer of cells around the range:
they supply corner samples and centre distances for clipping and curvature. Only `ComputeDistanceToSurface`
and `FindLiquidSurface` run on part of the grid as a caller's range (per particle batch, into piece buffers then
`WriteSnapshotRange`); air pockets, material boundaries, equations, the ladder, both solves and the check take the
whole grid (`blocks` = all), so no pressure is ever given at a batch border (the former `border_cell_pressure` and
`kLiquidToBorderCell` are deleted). In the batched layout (A15, section 5 F) those whole-grid functions cut the
grid themselves (`ApplySolveBatches`, stacked width `kTouchingCells`) and sweep batch by batch with the same
arithmetic. In a one-piece run the range is everything and the border is the outside of the domain.

**Copies (A15, A17).** A run is all CPU or all GPU. In the batched layout the only transfers are the planned,
counted slice copies of `SolveBatches` (whole CPU rows to the loaded piece and back, each border list read back
once, 4 B per border block) and the progress record; a CPU run with `max_loaded_blocks > 0` copies CPU to CPU,
counted the same way.

**Tolerance, CPU against GPU.** Counts, classes, labels, `blob_id`: equal. Geometry, equations and one
application of the equations: `|a - b| <= 1e-12 * scale + 1e-300`, `scale` = the largest magnitude of that
array. Velocity after the solve: `<= 2 * max_volume_error_rate * cell width + 1e-9 * max |velocity|`; pressure:
that velocity bound times `face_density * pressure_distance / dt`; material velocity with the velocity bound,
`sealed_gas_pressure` with the pressure bound. Iteration counts are not compared.

**Tolerance, batched against one piece (A15, same processor).** Counts, labels, classes, blob ids, pocket lists:
equal; the equation build (local arithmetic plus the shared sum rules): identical bytes; one solve: velocity and
pressure within the CPU-against-GPU bounds above. Batched against batched with another cut, and a repeat:
identical bytes. Whole runs over several steps: the step design's 6.4 tolerances.

## 7. Tests

Analytic and conservation, CPU (label `cpu`), each on uniform cells and on two size levels unless stated:

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `ComputeFlowImbalance, OneFaceInflow_IsPositive` | One cell, 1 m3/s entering through its low X face | `+1 m3/s` | `1e-15` |
| Clipping, flat surface | Liquid plane at height `H` through cells; no solid, then a slanted flat solid | `filled_volume`, `open_area` equal the exact polyhedron values; a big face equals the sum of its four pieces | `1e-12` relative |
| Hydrostatic column | Box open at the top, depth `H`, `velocity_before_pressure = -g dt` on Z faces | `p = rho g (H - z)` at centres; velocity after `= 0` | velocity `<= max_volume_error_rate * w`; pressure by the matching bound |
| Two cells, analytic | Fixtures of `docs/adaptivity/native-pressure-boundaries.md` and `adaptive-pressure-graph-inputs.md` | `1.75`, `1.375 Pa`; `1.28125`, `1.09375 Pa`; stated face velocities | `1e-9` relative |
| Linear pressure in a sealed cube | `docs/adaptivity/matched-error-pressure.md`: `p = 0.125 + 0.75 x` | velocity after `= 0`; pressure after removing the mean | `1e-4 Pa`; `2e-5 1/s` |
| Driven solid, swept volume | Driven flat solid on `side_left` at speed `U`, open top | Outflow through air ends `=` sum of `solid_volume_rate` | `max_volume_error_rate *` liquid volume |
| Sealed container moved as one | Full box, all six sides driven at the same velocity | `incompatible_volume_rate = 0`; velocity after equals the solid's | `1e-12` of the terms; solve limit |
| Sealed blob, one side pushed in | Same, one side only | `kSealedBlobVolumeMismatch`, `magnitude = U * area` | `1e-12` relative |
| Two blobs, one sealed | One open pool and one full closed chamber | Open blob referenced to air; sealed blob's mean equals its mean `carried_pressure`; result unchanged when cells are renumbered | `1e-9` relative |
| Static drop | Sphere radius `8 w` and `16 w`, `sigma > 0`, no gravity | curvature `2/R`; inside pressure `2 sigma / R`; velocity after stays near zero and shrinks with `w` | 2 % at `8 w`, 0.5 % at `16 w` |
| Step limit | Water, `sigma = 0.0728`, `w = 3.5 cm`; `w = 1 mm` | `0.2165 s`; `1.0455e-3 s` | `1e-4` relative |
| Pressure law | Both kinds, round trip pressure to `volume_ratio`; infinite sound speed | Closed forms; `squeeze_allowance = 0` and bytes equal to the incompressible build | `1e-13` relative; exact |
| Air pocket law | Halve the volume; merge `(V1, p1)` with `(V2, p2)` | `p * 2^1.4`; gas amounts add | `1e-13` relative |
| Air pocket classes | Closed bubble; bubble under the open top; bubble beside a block that is not allocated and touches a solid | `kEnclosed`, `kVented`, `kUnknownFrontier`; `blob_id` kept by the larger part at a split | exact |
| Equations are symmetric and positive | Random pressures on a mixed-size case with solids | `x.A y = y.A x`; `x.A x >= 0`; zero only for blob constants of sealed blobs | `1e-12` relative |
| Ladder | Any rung | coarse equations applied to `v` equal merge(fine equations(spread(v))) | `1e-12` relative |
| Check is independent | A wrong pressure given to `CheckFlowImbalance` | The report equals the hand-computed error | `1e-12` relative |
| Open sea | One face at zone position 1, rate `r`; wave `omega^2 = g k tanh(k depth)` | `lerp(u, target, 1 - exp(-r dt))`; dispersion holds | `1e-13`; `1e-10` |
| Errors | Two bad cells | The lower `cell_index` is reported; no output is readable | exact |
| Repeat run | Any case twice | Identical bytes | exact |
| `SolvePressure, CancelSet_ReturnsCancelled` (NEW) | Hydrostatic column, `cancel_requested` set before the first progress read; then null | `kCancelled`, the step's try arrays never committed; with null the result is unchanged | exact |
| `SolvePressure, WritesStepArrays` (NEW) | Hydrostatic column with `SolvedArrays` and `AirPocketArrays` from the caller | the caller's arrays hold the result; a piece with no connection keeps `velocity_before_pressure` | exact |
| `ComputeBlockedCornerAreas, SumsToBlockedArea` (NEW) | Random corner distances, both triangle splits, all signs | `sum corner_area = area - open_area` of step 4; each corner area `>= 0` | `1e-14 * area` |
| `ComputeEquationIndexes, MatchesBuild` (NEW) | Mixed-size case with solids, a gas snapshot | the index of each cell equals `BuildPressureEquations`' order; count equal | exact |
| `ComputeBlobOfCell` (NEW) | Two pools and a sealed chamber | each cell's blob equals the blob whose `first_cell_index` is its smallest connected cell; `kNoBlob` elsewhere | exact |
| `SealedBlobCheck, WhenRatesApplied` (NEW) | Sealed gas box squeezed by a driven side | `kWhenBuilt` fails at the build; `kWhenRatesApplied` builds, then `ApplyAddedVolumeRate` with matching rates passes and with zero rates fails with the same magnitude | `1e-12` relative |
| `CellContent::kGas, SameAsLiquid` (NEW) | One snapshot with `kLiquid`, then the same with `kGas` | identical equations, solve and report bytes | exact |
| Cell eighths and centres (NEW, C3) | Flat liquid surface at heights through a cell, slanted flat solid, two size levels | eighths sum to `filled_volume`; each equals the exact polyhedron; centres and `liquid_covered_area` exact; arrays null without viscosity | `1e-12` relative |
| Gas pressure at the liquid surface (NEW, A10) | Water pool under a sealed gas box whose committed gas pressure is `+P` | `p = P + rho g (H - z)`; blob is `kHasReferencePressure`; the gas cells belong to no air pocket | solve limit |
| `PressureScene, RisingGasPressureFollowedEachSolve` (A18: the path read one step late) | Hydrostatic column, no `SealedGasBlob` given; three solves with gas pressure 100, 150, 225 Pa passed as the committed array | the surface cell's pressure in solve k follows the k-th array (the one-step lag is the caller's) | solve limit |
| Two liquids at rest (NEW, C4) | Oil (800) over water (1000) in a closed box, two size levels; sigma between them; public path `ComputeDistanceToSurface` -> `FindLiquidSurface` -> `FindMaterialBoundaries` -> `BuildPressureEquations` -> `SolvePressure` | pressure jumps by `rho_oil g h` across layers, `low_share` exact for a flat boundary, velocity after stays 0 (the sharp boundary is delivered; no exclusion test) | solve limit; `1e-12` |
| `ComputeDistanceToSurface, AllZeroMask_IsNoLiquid` (NEW) | Particles of a gas material only, `material_is_counted` all zero, `kAllLiquidMaterials`; then the same mask with a material index | every distance positive (`+2 cell widths` away from grid-only cells), `FindLiquidSurface` gives every open cell `kAir` and no equation; with a material index `kBadInput` | exact |
| `ToName, EveryError` (NEW) | Every `PressureError` value | 15 values; lower snake case of the name without `k`, all distinct | exact |
| Ladder as preconditioner (NEW, A5) | `BuildPressureLadder` with a random `connection_scale` in `[0.1, 1]` | `ApplyPressureLadder` is symmetric (`x.L y = y.L x`) and positive on each blob; with scale 1 it equals the ladder `SolvePressure` builds | `1e-12` relative |
| `BuildPressureLadder, FirstRungMembersShareOneBlock` (NEW, A15) | 100 random three-level grids, two blobs each | every rung-1 coarse cell's members lie in one block; rung-1 order is (block, blob, merged id) | exact |
| `BuildPressureLadder, BlobOfTwentyEquations_PassesOneToOneIntoRungOne` (NEW, A15) | A 20-cell drop beside a big pool | 20 rung-1 cells with one member each; the drop is terminal at rung 1 | exact |
| `BuildPressureLadder, DirectSolveRoomFull_LaterBlobsWithoutDirectSolve` (NEW, A15) | Three 10-cell drops, room 250 values | the first two (ascending first cell) get factors, the third is counted in `blobs_without_direct_solve` | exact |
| `SolvePressure, FreeSolidRankOne_DqEqualsDirectProduct` (NEW, A15) | The symmetric-equations scene with its free solid, 10 random `d` | `dq` from the shared arithmetic equals `d . (A d)` summed directly | `1e-12` relative |

Coupling inside the solve (NEW, A18), CPU:

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `ComputeMaterialFaceCoefficients, MatchDefinition` | A 0.0025 m2, m 0.0375 kg, dt 0.01 s | `1.6666667e-6` and `6.6666667e-4` | `1e-15` relative |
| `ComputeGasSqueezeAllowance, MatchesDefinition` | V 3.75e-4 m3, p0 101325 Pa, 1.4, dt 1e-3 s | `2.6436e-6` (closed form) | `1e-15` relative |
| `ComputeMaterialVelocityAfterPressure, PushesFromHighToLow` | u 0, `2e-4 * (3333.33 - 0)` | `0.666667` | `1e-15` relative |
| `BuildPressureEquations, MaterialFaceBetweenTwoCells_WeighsLikeAConnection` | Two liquid cells of 0.1 m, shared piece fully blocked, entry A 0.01, m 0.5, dt 0.01 | equations equal a hand product with `g = 2e-6` between them; `flow_imbalance_before` of each from `-s (A u* - coverage)` | `1e-15` relative |
| `BuildPressureEquations, MaterialFaceOneSolvedEnd_IsAReferenceEnd` | Closed box of liquid whose lid is 16 material faces | `kHasReferencePressure`; own weight includes `g`; no right-side reference term | exact |
| `BuildPressureEquations, MaterialFaceAtCoverageVelocity_SameFlowAsWithout` | `u*` equal to the coverage velocity, `A` the blocked area | `flow_imbalance_before` equals the build without entries | `1e-14` relative |
| `BuildPressureEquations, SymmetricAndPositiveWithMaterialFaces`, `...WithSealedBlob` | Symmetric scene plus two faces between equation cells, four with one solved end; then a sealed blob, then both | `x.A y = y.A x`, `x.A x >= 0` | `1e-12` relative |
| `BuildPressureEquations, AirEndPrecedence_SealedBlobFirst` | An air cell in a sealed blob that also `holds_tracked_gas` at 300 Pa | its connection goes to the blob, reference `sigma * curvature + 7 Pa` (`gas_pressure_above_background`); the other air ends as the A10 precedence test | exact |
| `BuildPressureEquations, SealedBlobElimination_EqualsSchurProduct` | Column 2 x 2 x 6 of 0.05 m, four gas cells in one blob, 20 random `x` | `equations(x) = A_cells x - (g g^T x) / D`; right side `r_cells + g r / D` | `1e-13` relative |
| `BuildPressureEquations, LiquidUnderSealedBlob_IsCanBeSqueezed`; `TwoPoolsUnderOneBlob_AreOneLiquidBlob` | Closed box, water under gas | with the blob `kCanBeSqueezed`, without it `kHasReferencePressure`; two pools one blob | exact |
| `BuildPressureEquations, MaterialFaceErrors_AreBadInput`, `SealedBlobErrors_AreBadInput` | No solved end; `face_mass` 0; unsorted; too many entries; blob index past the list; `gas_volume` 0; pressure 0; too many blobs | `kBadInput` | exact |
| `BuildPressureLadder, MaterialFacesInRungZero`; `ApplyPressureLadder, SymmetricAndPositiveWithMaterialFaces`, `SealedBlobConnectionsAreReferenceEnds` | As the ladder tests, with a fully blocked piece that has no connection | coarse = merge(fine(spread)); symmetric and positive | `1e-12` relative |
| `CoupledSolvePieces, MaterialFacesAccepted_MatchHandValues` (NEW, A24; replaces `ComputeVolumeError, CoupledEquations_IsBadInput`) | The two-cell material-face fixture above (`g = 2e-6`); `connection_scale` null and 0.5 on every piece; `ComputeVolumeError` `kLeftOut` with p = (1, 0), velocity 0 | `ComputeOwnWeight` adds exactly `2e-6` to each cell's own weight whatever the scale; volume error `-2e-6` m3/s (low cell) and `+2e-6` m3/s (high cell) from the material term; the other pieces succeed | `1e-15` relative |
| `CoupledSolvePieces, BlobFormsWithoutBlobBuffer_IsBadInput` (NEW, A24) | The Schur test's column (one sealed blob) given to both `ComputePressurePush` forms, `ComputeVolumeError`, `ComputeOwnWeight`, `ComputeStartPressure` without a blob buffer; the same column with material faces only | `kBadInput`, no output written; material faces only: success | exact |
| `SolvePressure, LiquidCellUnderMaterialFace_ShareMomentum` | One liquid cell of 0.1 m, rho 1000, low Z piece to air open 0.01 m2, distance 0.1, u 1 m/s up; top fully blocked, A 0.01, m 0.5 kg, u* 0; dt 0.01; liquid face mass 1 kg | `p = 3333.33 Pa`; liquid and material velocity both `M U / (M + m) = 2/3 m/s` | `1e-9` relative |
| `SolvePressure, LiquidCellUnderSealedGas_SpringStep` | One cell of width 1, rho 1, dt 1, both Z pieces area 1, distance 0.5, u 1 up; air below, blob above V 1.4, p0 1, ratio 1.4 (S = 1), carried 0 | `p = 0.25 Pa`, `sealed_gas_pressure = 0.5 Pa`, both velocities `0.5 m/s` | `1e-9` relative |
| `CheckFlowImbalance, MaterialFlowFromPressure`; `SealedBlobRowChecked` | First fixture with p 3000; second with p 0.3 | worst cell error and `max_sealed_gas_volume_error_rate` equal the hand values; `kNotConverged` above the limit | `1e-12` relative |
| `SolvePressure, CouplingAbsent_SameBytesAsBefore`; `CouplingSpansWrongSize_IsBadInput` | Earlier cases with empty spans; wrong span sizes | identical bytes; `kBadInput` | exact |
| `CoupledSolvePieces, SealedGasRowFromGivenPressure` (NEW, A24) | The spring-step fixture (`g = 2` m3/(Pa s) on both Z pieces, S = 1, c = 0); p 0.25, velocity 0.5 on both Z pieces | `kIncluded`, q 0.5: blob and cell errors 0; q 0.6: blob error `-0.1` m3/s; `kLeftOut`, q 0.6: `-0.1` m3/s | `1e-12` m3/s absolute (zeros); `1e-15` relative |
| `ComputePressurePush, SealedGasFarEndFromGivenPressure` (NEW, A24) | Same fixture, p 0.25, q 0.5, both `GivenTerms` | push `-0.25` N s on both Z pieces; `ApplyPressure` from velocity 1 gives 0.5 m/s | `1e-15` relative |
| `ComputeOwnWeight, SealedGasOwnWeight` (NEW, A24) | Same fixture; scale 0.5 on every piece, then null | blob own weight `1 + 2 * 0.5 = 2`; null scale `3` (= `D_j`) | `1e-15` relative |
| `ComputeStartPressure, SealedGasStartIsCarried` (NEW, A24) | Same fixture, blob carried 0.3 Pa | `sealed_gas_pressure = 0.3` | exact |
| `ComputeBlockingMaterialVelocities, MatchesFormula` (NEW, A24) | Material-face fixture, p 3000; then a span of the wrong size | `0 + 2e-4 * 3000 = 0.6` m/s; `kBadInput`, nothing written | `1e-15` relative |
| `CheckFlowImbalance, GivenBlobPressureChecked` (NEW, A24) | Spring-step fixture, p 0.25, velocity 0.5; q `0.5 + 1.4e-5`, then q 0.6; then a blob buffer of size 0 | `max_sealed_gas_volume_error_rate = 1.4e-5 / 1.4 = 1e-5` 1/s; q 0.6 (`0.1 / 1.4` 1/s): `kNotConverged`; `kBadInput` | `1e-9` relative |
| `CoupledSolvePieces, InBatches_SameAsOnePiece` (NEW, A24) | Both fixtures in `BuildTallBoxGrid(1, {4, 4, 32})` of `kSolid` cells, batched | every A24 form gives the one-piece bytes | exact |
| `CoupledSolvePieces, OnGpu_EqualsCpu` (NEW, A24, label `gpu`) | The Schur column with material faces, every A24 form | cell, face and blob values; report | `1e-12 * scale` |
| `ComputeVolumeError, GivenMaterialVelocity` (NEW, A26) | The two-cell material-face fixture, velocity 0, `w = 0.3` m/s; `kLeftOut` with p = (0, 0) and (1, 0); `kIncluded` with p = (0, 0) | difference from `w = 0`: `-0.003` m3/s (low cell), `+0.003` m3/s (high cell) at both p (`A w = 0.003`, no `g` term); `kIncluded`: material flow `A w - coverage_volume_rate` with the sign rule | `1e-15` relative |
| `ComputeVolumeError, GivenFreeSolidMotion` (NEW, A27) | The symmetric scene's free solid (3 entries), velocity 0, p 0, given `xi_before` and `xi`; then `free_solid_motion_before_pressure` empty | per pushed cell `-sum_e push_e . (xi - xi_before)` (`kIncluded`), `-sum_e push_e . xi` (`kLeftOut`), computed in the test; other cells 0; empty `xi_before` with `kIncluded`: `kBadInput` | `1e-14` relative |
| `ComputeCoupledPush, MaterialAndFreeSolid` (NEW, A26, A27) | Two-cell fixture p = (1, 0) and (0, 1); the one-cell material fixture p 3000; the free solid with a given p | `1e-4`, `-1e-4`, `0.3` N s (`dt A (p_low - p_high)`); free solid `dt sum_e (force, torque)_per_pressure p_e` computed in the test | `1e-15` relative (material), `1e-14` (solid) |
| `ComputeOwnWeight, MaterialFaceScale` (NEW, A26) | Two-cell fixture, `material_face_scale` 0.5, then null; `connection_scale` 0.5; the free-solid scene | own weight gains `g * 0.5 = 1e-6` (null: `2e-6`), unchanged by `connection_scale`; free solid left out (bytes equal without it) | `1e-15` relative |
| `CheckFlowImbalance, CoupledVelocitiesChecked` (NEW, A26, A27) | One-cell material fixture, p 3000, liquid 2/3 m/s, `w = 2/3`; p 10000/3; `w = 2/3 + 1e-3`; the free-solid scene after `SolvePressure` with its recovered `xi` | worst cell error = A24 form's minus `0.01 (2/3 - 0.6) = 6.6667e-4` m3/s; at p 10000/3 both forms agree; `w + 1e-3`: `kNotConverged`; free solid: report equals the A24 form's | `1e-12` relative |
| `CoupledVelocityPieces, ShortBuffers_AreBadInput` (NEW, A26, A27; other-processor case dropped, A30) | Null `CoupledVelocities` pointer with count > 0; each buffer one entry short; `free_solid_motion_before_pressure` of size 5 | `kBadInput`, nothing written; null with count 0: success | exact |
| `CoupledVelocityPieces, InBatches_SameAsOnePiece` (NEW, A26, A27) | The two material fixtures in `BuildTallBoxGrid(1, {4, 4, 32})`, the free solid's entries in 3 batches | every `CoupledVelocities` form, `ComputeCoupledPush`, the `material_face_scale` form give the one-piece bytes | exact |
| `CoupledVelocityPieces, OnGpu_EqualsCpu` (NEW, A26, A27, label `gpu`) | The Schur column with material faces; the free-solid scene; given `w`, `xi`, `xi_before`, scale | cell, blob, material-push and free-solid-push values; report | `1e-12 * scale` |

Coupling scene tests (NEW, A18, label `cpu`). The material's elasticity is a test stand-in applied as `ApplyForces`
would. Each lagged reference is test-only (about ten lines: the material as a driven solid whose next velocity
takes `-dt A p / m`; the gas pressure passed as `gas_pressure_above_atmosphere`, updated with `p0 (V_n /
V_n+1)^heat_ratio`). Banks et al.: the lagged scheme is unstable at every step when the added mass `M` exceeds
`m`; the coupled gas scheme is backward Euler on the spring (factor `1 / sqrt(1 + (omega dt)^2)`, phase `atan(omega
dt)`), the lagged one symplectic Euler, unstable for `omega dt > 2`.

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `PressureScene, WaterOnSnowSlab_InterfaceHoldsAndCarriesTheWeight` | `BuildBoxGrid` w 0.05, 4 x 4 x 8 (0.2 x 0.2 x 0.4 m), walls and floor; slab top at z 0.1: 16 fully blocked Z pieces, each an entry (A 0.0025, `m_s = 300 w^3 = 0.0375 kg`); water to H 0.3 (`h_w` 0.2); g 9.81; dt 0.01; snow stand-in per face `u_s_before = u_s_after_prev + dt (-k x - m_s g) / m_s`, k 475 N/m. `M = 0.4375 kg`, `p_1 = 1716.75 Pa`, `x* = -9.8098e-3 m`, `omega = 31.6`. Start at `x*`, hydrostatic, at rest; 500 steps | `|x_n - x*| <= 1.3e-7 m` each step, no trend (means of steps 1-100 and 401-500 within 1.3e-7 m); interface pressure `1.5 p(z 0.125) - 0.5 p(z 0.175) = 1000 g h_w = 1962 Pa`; push per area `m_s (u_after - u_before) / (dt A) = -p_1` | twice the pressure bound; `1e-9` relative |
| `PressureScene, LightSnowSlab_StableWhereLagGrows` | Same, `m_s / M = 0.0857`, start at `x* + 0.1 w`, 500 steps | max `|x - x*|` over steps 401-500 `<= 1.02 * 0.1 w`. Lagged reference at dt / 10: velocity difference times `-M / m_s = -11.67` per step (steps 2 to 5); at dt: grows at least 5 times per step | 1 % on the growth factor |
| `PressureScene, LightSlab_FloatsAtDisplacedDepth` | `BuildBoxGrid` w 0.025, 16 x 2 x 12 (0.4 x 0.05 x 0.3 m); slab density 200, t 0.1 m over x in [0, 0.2]; free surface over [0.2, 0.4]; each step rebuild the snapshot, 16 entries (A w^2, m 0.0125 kg); `z_b + h_r = 0.295`, `z_b* = 0.1375`, `h_r* = 0.1575`. Start at rest at equilibrium, dt 0.01, 300 steps | `|z_b - z_b*| <= 1e-6 w` each step; `h_r - z_c = 0.2 t` within `1e-6 w`; `z_b + h_r` kept within the solve limit | as stated |
| `PressureScene, LightSlab_OscillationDoesNotGrowWhereLagGrows` | Start `z_b* - 0.2 w`, `h_r* + 0.2 w`, dt 0.01, 500 steps | max `|z_b - z_b*|` over the last 100 `<= 1.02 * 0.2 w`; mean `z_b` over the last 300 within `0.03 w` of `z_b*`. Lagged reference at 0.001 s: grows at least 5 times per step, steps 2 to 5 (`M / m >= 6.9`) | as stated |
| `PressureScene, SealedGasAboveColumn_PeriodIsSpringPeriod` | `BuildBoxGrid` w 0.025, 2 x 2 x 16 (A 0.0025 m2, 0.4 m), side walls and closed top, bottom open; no gravity; water z [0.05, 0.25] (`m = 0.5 kg`); 24 cells above one `SealedGasBlob` (V 3.75e-4, p0 101325, 1.4); `k = 2364.3 N/m`, `omega = 68.765 1/s`; u0 1e-4 m/s, dt `0.1 / omega = 1.4542e-3 s`, 50 periods (3142 steps) | period (mean of periods 1 to 5) `= 2 pi dt / atan(omega dt)` within 0.05 % and within 0.5 % of `2 pi / omega = 0.091371 s`; each period's largest `|u|` not above the previous; per-step factor `1 / sqrt(1.01)` | `1e-4` relative on the factor |
| `PressureScene, SealedGasAboveColumn_StableWhereLagGrows` | Same, u0 1e-9 m/s, dt `4 / omega = 0.058169 s` (about 54 times the lagged air-pocket limit 1.084e-3 s), 79 steps | `|u|` never grows, factor `1 / sqrt(17) = 0.24254`; lagged reference: `|u|` times 13.928 per step, steps 2 to 4 | `1e-4`; 1 % |
| `PressureScene, SealedGasHeatedAboveStillWater_RisesInTheSameSolve` | Closed box 0.2 x 0.2 x 0.4, w 0.05, water to 0.2 at rest, g 9.81; gas above one `SealedGasBlob` (V 0.008) with `heat_volume_rate = (0.4 / 1.4) * 35000 * 1.25e-4 / 101325`; one solve at dt 0.03 s (ten times the lagged limit 3.0e-3 s) | `sealed_gas_pressure = 0.4 * 546.875 * 0.03 = 6.5625 Pa`; liquid pressure `6.5625 + 1000 g (0.2 - z)` at centres; water velocity 0 | `1e-9` relative; pressure bound; velocity bound |

Solve in pieces (NEW, A15), CPU (the loaded piece and the copies on the CPU):

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `ApplySolveBatches, RangesCoverEveryBlockOnceAndFit`; `LargestRangeIsChosen` | 100 random grids, max loaded `3 * CountBlocksInLargestLayer`, widths `{kTouchingCells}` and `{kTouchingCells, kTouchingCells}` | ranges in order, each block once; range plus stacked border `<= max_loaded_blocks`; one block more per range would not fit | exact |
| `ApplySolveBatches, OneLayerDoesNotFit_IsRunOnCallFailed` | max loaded = one layer | `kRunOnCallFailed`, magnitude = blocks needed, limit = `max_loaded_blocks` | exact |
| `WriteBatchValues`, `ReadBatchValues`, `WriteBatchGroups` | `BuildTallBoxGrid(0.1, {8, 8, 64})`, distinct values; 0 to 5 items per block | loaded blocks equal whole blocks through the map; range blocks only written back; items in piece order; too short a whole array `kBadInput` | exact |
| `LoadSolveBatch, CancelSet_IsCancelledAndLoadsNothing`; `ReadSolveCopyReport, CountsLoadsSweepsAndBytes` | Flag set; two sweeps of 4 batches | `kCancelled`, `loaded_batch()` unchanged; 8 loads, 2 sweeps, bytes as computed | exact |
| `ComputeBufferBytes, BatchedLayout_RowsSplitByWhere`; `BuildSolveBatchesMemoryEstimate, MatchesAllocatedBytes` | 64 blocks, loaded 16 | `kCpu` rows equal the one-piece bytes of the same rows; `kRunOn` rows = kept + loaded + batches' | exact |
| `WriteSnapshotRange, PiecesGiveTheWholeGridBytes` | `BuildTwoLevelGrid` with 8 block layers, analytic surface and slanted solid, ranges of 2 layers | every pass-1 row equals `FindLiquidSurface` on the whole grid | bytes |
| `JoinLabelsAcrossBatches, ChainThroughFiveBatches_SmallestLabelWins`; `NoJoins_EveryLabelItsOwn` | Hand-made joins | smallest label wins; labels unchanged | exact |
| `FindAirPockets`, `BuildPressureEquations`, `FindMaterialBoundaries`, `ComputeVolumeCorrectionRate`: `InBatches_Same...AsOnePiece` | Grids of at least 8 block layers, loaded 3 and 5 times `CountBlocksInLargestLayer`: two bubbles (one across a boundary) and a vented column; free solid, material faces and a sealed gas blob over three batches; the oil drop across a boundary; a sealed blob over four batches | labels, lists, indexes, rows, connections, classes, sums equal one piece; the sealed blob's correction sums to zero (`<= 1e-12 * sum |rate|`) | bytes |
| `BuildPressureLadder, InBatches_SameRungsAsOnePiece`; `ApplyPressureLadder, InBatches_SameOutputAsOnePiece`, `InBatches_SymmetricAndPositive` | Two blobs, loaded 3 and 5 layers; a sealed blob over three batches; with the batched room and a drop beyond it | same rungs; same output (with the same room `1e-12` relative); symmetric and positive | bytes; `1e-12` |
| `SolvePressureInBatches, TwoCellFixturesInATallGrid_SameAsOnePiece` | The two-cell fixtures in `BuildTallBoxGrid(1, {4, 4, 32})` of `kSolid` cells | `(1.75, 1.375)` and `(1.28125, 1.09375)` Pa | `1e-9` relative |
| `SolvePressureInBatches, CancelBetweenBatches_ReturnsCancelledOutputsUntouched`; `FailureInOneBatch_PublishesNothing`; `BuildPressureEquations, CancelBetweenBatches_IsCancelled` | Hook at the 7th (3rd) load; a NaN in one right side of the last batch | `kCancelled`; `kNotFinite` with that cell, as one piece; `SolvedArrays` keep their bytes | exact |
| `SolvePressureInBatches, SweepsPerIteration`; `CheckFlowImbalance, InBatches_SameReportAsOnePiece` | Copy report; the wrong-pressure case in a tall grid | 3 sweeps per iteration without a sealed blob, 4 with one; `batch_loads = sweeps * batch_count`; same report | exact; bytes |

Batched against one piece, scene tests (NEW, A15): each scene on at least 8 block layers, once in one piece and
in batches with `max_loaded_blocks` = 3, 5 and 7 times `CountBlocksInLargestLayer`: counts, labels, classes and
pocket lists equal; velocity within `2 * max_volume_error_rate * w + 1e-9 * max |u|` and pressure within that bound
times `face_density * pressure_distance / dt` of the one-piece solve; the three cuts identical bytes; the original
analytic expectation where it has one.

| Test | Setup | Expected |
|---|---|---|
| `SolvePressureInBatches, HydrostaticColumn_WithinToleranceOfOnePiece` | Open-top box 0.4 x 0.4 x 1.6 m, w 0.025, H 1.43 (two-level: big below 0.8) | also `p = rho g (H - z)` within the pressure bound |
| `SolvePressureInBatches, SealedCubeCutIntoThree_WithinToleranceAndSameBytesForEveryCut` | Sealed cube at w 0.0625 (32^3, 8 layers), carried 1e5 Pa | mean kept; linear pressure |
| `SolvePressureInBatches, TwoBlobsOneSealed_WithinTolerance`; `FreeSolidAcrossBatches_WithinTolerance` | Two-blob scene lengthened along z; free solid with entries in 3 batches | within tolerance |
| `SolvePressureInBatches, WaterOnSnowSlab_WithinTolerance` | Snow slab at w 0.0125 (16 x 16 x 32), one step | material velocities within the velocity bound |
| `SolvePressureInBatches, SealedGasAcrossBatches_WithinTolerance` | Heated pocket at w 0.0125, the blob over four layers | `sealed_gas_pressure` within the pressure bound; 6.5625 Pa within `1e-9` relative |
| `SolvePressureInBatches, FitsOnlyInBatches` | Hydrostatic scene; budget B = half the one-piece `kRunOn` bytes | the batched layout at 3 layers has its `kRunOn` bytes below B; solved in that layout within tolerance |
| `SolvePressureInBatches, RepeatRun_IdenticalBytes` | Any batched case twice | identical bytes |

**Python reference (one piece, new):** `reference/pressure/dense_cell_equations.py`, about 150 lines. From a
plain list of cells and face pieces it builds the matrix with a sparse library and solves it directly, with
the mean rule of step 16 applied afterwards. It removes one uncertainty no analytic case covers: a case with a
size step, partly filled cells, a jump, squeeze allowance, a driven solid and a sealed blob all at once has no
closed-form answer. Three such cases are stored as JSON and compared at `1e-8` relative. A fourth (A18),
`coupled.json`: a two-level case with two material faces between equation cells, three with one solved end, and
one sealed gas blob touching four cells, which the script solves as a true extra unknown (one more row and
column, no elimination), checking the elimination of step C2 independently; material velocity and gas pressure
are compared too (`DenseReference, FourCases_MatchStoredAnswers`).
`cavity.py` and `eos.py` are not ported: their formulas are closed forms tested directly.

**Lean (one file):** `proofs/Fluidity/PressureEquations.lean`, from `Pressure.lean` and `PressureTopology.lean`:
the equations are positive semidefinite, their null space is exactly the constants of blobs without reference
pressure or squeeze allowance, and they can be solved exactly when each such blob's right-hand side sums to
zero. Step 16 relies on this for every possible grid, which tests can only sample. The pin theorem
(`pressure_gauge_positive_definite`) and `PressureCorrection.lean` are dropped; the latter becomes the
"check is independent" test. The proof's connections are weighted pairs and weighted single ends, so
blocking-material faces are in its model (a face with one solved end counts as a reference end); the sealed gas
elimination is not proved (tested by the Schur test and the dense reference) (A18).

**Against the proof of concept** (method unchanged, rotated to Z-up): cell coefficients and right-hand sides
against `PrepareExplicitMacPressureSystem3D` at `1e-12` relative; solved pressure, after removing each sealed
blob's mean, and velocity against `ProjectMacPressure3D` at twice the old stopping limit; curvature, pocket
pressure and pressure law against `ComputeGeometry`, `EvaluateLiquidCavity3D`, `EvaluateLiquidEosAtPressure3D`
at `1e-12` relative. A18: `main`'s one-face snow trial `SolvePlanarSnowPressureTrial3D` (`mixed_contact_3d.cc:432-534`;
4 x 8 x 4 cells of 0.05 m, snow mass 0.0375 kg, incoming -0.1 m/s) against the same scene with that face as one
`BlockingMaterialFace`: material velocity and `face_mass * (u - u*)` at twice the old stopping limit.

**GPU (label `gpu`, compiled here, run by the owner):** every row above within the tolerance of section 6; the
repeat-run test on identical bytes for pressure, velocity and all records; a case of at least 2 million
equations to show that no phase runs in one thread; the coupling (A18: equations, ladder and solve against the
CPU, twice identical); in batches (A15): the same cut as the CPU, loaded equals whole through the map, the
batched equation build identical to one piece on the GPU, solves within tolerance of one piece on the GPU, twice
and two cuts identical bytes, and the 128^3 box with `max_loaded_blocks` a quarter of its blocks converging in
the one-piece iteration count (prints seconds and bytes copied per iteration). **Without a GPU (labels `cpu`,
`no_gpu`):** all shared formula headers, `TreeSum`, the label propagation and ladder building through their CPU
implementations, and `BuildPressureBuffers(sizes, RunOn::kGpu)` (also in the batched layout),
`BuildSolveBatches(kGpu, ...)` and the coupled solve on the GPU returning `kRunOnCallFailed` in the container.

## 8. Method paper section

Labels: `<sec-liquid-surface>`, `<sec-pressure>`, `<eq:pressure>`, `<eq:correct>`, `<sec-pressure-solve>`,
`<sec-tension>`, `<sec-pressure-law>`, `<sec-air-pockets>`, `<sec-open-sea>`; figures `<fig-partly-filled-cell>`,
`<fig-face-piece-flow>`, `<fig-pressure-ladder>`, `<fig-air-pockets>`; procedures `<proc-find-liquid-surface>`,
`<proc-solve-pressure>`. NEW (A18): `<sec-coupled-solve>`, `<eq:material-face>`, `<eq:sealed-gas-row>`,
`<eq:sealed-gas-elimination>`, `<fig-coupled-faces>` (a liquid column under a snow slab and beside a sealed gas
pocket drawn as one extra node; both move in the same solve). NEW (A15): `<sec-solve-in-batches>`,
`<fig-solve-batches>` (one iteration's sweeps A to D over three batches, the coarse rungs between B and C, copies
with their sizes per block), `<proc-solve-pressure-in-batches>`; cite Smith, Bjorstad and Gropp 1996 once for the
rejected alternative, and for A18 Banks, Henshaw and Schwendeman 2014, Robinson-Mosher et al. 2011, Fang et al.
2020, Aanjaneya 2013, Patkar et al. 2013.

Outline: (1) where the liquid is: distance, six tetrahedra, open area as the single source; (2) the cell
equation and its sign (`flow_imbalance` is inflow minus outflow), the velocity correction with the same area,
density and distance; (3) blobs, the free constant, incompatible volume; (4) surface tension as one jump;
(5) pressure law and squeeze allowance; (6) air pockets and their classes; (7) the solve and its fixed-order
sums, and the pieces the joint viscosity-and-pressure solve reuses; (8) the check and the volume budget, one
stop rule for every solve; (9) the open-sea side and outlet; (10) what gas reuses, and the gas pressure the
liquid feels at its surface; (11) two liquids that do not mix: one distance per material and the boundary;
(12) blocking materials and sealed gas solved with the liquid, and what stays one step late; (13) solving in
batches.

Figures: a partly filled cell with a solid and the liquid surface, its six tetrahedra and one clipped face
piece; a big cell beside four small cells with the flow through each face piece and the pressure distance,
including an air end with `liquid_share`; the ladder for a mixed-size blob, showing members merged by size
level and the gather; a section of a sparse grid with an enclosed, a vented and an unknown-frontier pocket.
Pseudocode: the numbered steps of section 5 A and of 5 C to D, matching the `// Step N:` comments.

Cite from the old paper's list: [21] Batty 2007 and [22] Ng 2009 (solid flow through blocked area), [23] Gibou
2002 (distance to the surface at an air end), [24] Kang 2000 (two liquids, density in series, jump), [25]
McAdams 2010 and [6] Wang 2026 (multilevel solve), [9] Larionov 2017 (coupling with viscosity), [20] Jacobsen
2012 (relaxation zones), [26] Kwatra 2009 (the squeeze term for gas, in the gas section).

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| `ProjectMacPressure3D`, projection | `SolvePressure` | DECIDED |
| `PreparedMacPressureSystem3D` | `PressureEquations` | DECIDED |
| pressure row; link, edge | `cell_equation`; `cell_connection` | DECIDED |
| `target_divergence * fluid_volume`, source | `added_volume_rate` | DECIDED |
| `compliance` | `squeeze_allowance` | DECIDED |
| gauge, pin | `pressure_offset` | DECIDED |
| `cell_constraint_defect`, divergence | `flow_imbalance` (inflow minus outflow); `volume_error` for what is left | DECIDED; NEW |
| `true_residuals` | `equation_error` | NEW (`residual` is banned) |
| `previous_pressure` | `carried_pressure` | NEW, VERIFIED by reading `liquid_substep_3d.cc:1450-1507` |
| `predicted_normal_velocity`, `staging.new_velocities` before `:2402` | `velocity_before_pressure` | DECIDED, VERIFIED |
| `corrected_face_velocities`, `staging.new_velocities` from `:2402` | `velocity_after_pressure` | DECIDED, VERIFIED |
| `staging.old_velocities` | `velocity_before_forces` (transfer's) | VERIFIED by reading `liquid_substep_3d.cc:1227` |
| `old_time`, `new_time` | `step_start_time`, `step_end_time` | VERIFIED by reading `liquid_substep_3d.cc:1134` |
| `fluid_volume` | `filled_volume` | NEW (after `partly_filled_cell`) |
| aperture, `open_area` | `open_area`, `open_fraction` | DECIDED |
| `prescribed_wall_flux`, `integrated_inward_wall_flux` | `solid_volume_rate` | NEW |
| `conductance`; `pressure_to_velocity_scale` | `flow_per_pressure`; `velocity_per_pressure` | NEW |
| `pressure_gradient_distance`; theta | `pressure_distance`; `liquid_share` | NEW |
| `exterior_pressure` (Dirichlet) | `reference_pressure` | NEW |
| `inverse_density`, mobility | `face_density` | NEW |
| `FrozenLiquidBoundaryGeometry3D`, freeze | `GeometrySnapshot` | NEW from DECIDED `snapshot` |
| `signed_distance` | `distance_to_surface`, `distance_to_solid` | DECIDED; NEW |
| cut cell | `partly_filled_cell` | DECIDED |
| component (pressure graph) | `blob`, `PressureBlob`, sealed blob | DECIDED word, NEW use (O4 keeps `region` away) |
| `LiquidAirComponent3D`, cavity | `AirPocket`, `blob_id` | DECIDED, VERIFIED by reading `liquid_air_components.h:58-62` |
| `LiquidAirConnectivity3D::kUnclassified`, `LiquidAirPassage3D::kOpen` to a missing cell | `kUnknownFrontier` | NEW |
| lineage, inventory (cavity) | `BlobHistory`, gas amount | DECIDED; NEW |
| `LiquidEos3D`, EOS | `PressureLaw` | DECIDED |
| Hencky, Tait | `kSteadyStiffness`, `kStiffensUnderPressure` | NEW |
| `bulk_modulus` | `bulk_stiffness` | NEW (`stiffness_against_squeeze` is taken by another quantity) |
| `volume_ratio` | `volume_ratio` | DECIDED (A4) |
| coupled pressure solve with a velocity callback (`VelocityResponse` in this design's first version) | `viscosity::SolveViscosityAndPressure` calling this module's pieces | DECIDED (A5, A6) |
| `liquid_octant_volumes`, `wet_open_centroid_m`, `wet_solid_centroid_m` | `filled_volume_in_cell_eighth`, `liquid_covered_open_centre`, `liquid_covered_solid_centre` | DECIDED word `liquid_covered`; NEW (C3) |
| interface between two liquids (phase pair) | `MaterialBoundary`, `FindMaterialBoundaries` | DECIDED type (Q12); NEW function |
| gas cell beside liquid | `CellContent::kGas`, `holds_tracked_gas`, `gas_pressure_above_atmosphere` | NEW (A10) |
| ocean inlet, outlet, absorption | open-sea side, outlet, `ApplyOpenSeaRelaxation` | programme wording; NEW function name |
| multigrid hierarchy, level | ladder, rung | NEW (`level` means `size_level`) |
| `MacPressureWorkspace3D`, scratch arena | `PressureBuffers` | DECIDED word |
| (replaced 2026-11-14) `SumScratch`, `OrderScratch`, `sum_scratch`, `order_scratch`, "scratch" | `SumBuffers`, `OrderBuffers`, `sum_buffers`, `order_buffers`, "working buffers" | A31 |
| snow pressure trial, plastic solid pressure mobility (`mixed_contact_3d.cc`, `plastic_solid_boundary_3d.cc`) | `BlockingMaterialFace`, face mass, blocking-material face | NEW (A18) |
| bubble or cavity solved with the liquid | `SealedGasBlob`, sealed gas blob, `sealed_gas_pressure` | NEW (A18) |
| out-of-core solve, streamed tiles | `SolveBatches`, solve batch, sweep, batched layout, per-group total | NEW (A15) |
| `structurally_eligible`, ghost, donor | not used in this module | Verify items left to transfer |

## 10. Not ported

- `src/pressure.cc`, `host/pressure.h`, `cuda/pressure.cu`: the 2D dense solve of at most 64 cells.
- The one-block and one-thread launches of `cuda/mac_pressure_3d.cu`, the batch function, the host-built and
  uploaded hierarchy (`liquid_pressure_stage_3d_host.cc`), `mac_pressure_3d_bounded_hierarchy.cc` (second copy).
- FP32 solve vectors, `FineValue`, FP32 rounding of coefficients, the dense path for 16 cells or fewer
  (`mac_pressure_workspace_3d.cc:495-543`), the small-system crossover: tuning, to be measured later.
- `IsRepresentablePressureProduct`, `CompensatedSum`, `rhs_absolute_term_sums`, the 64 epsilon identity error.
- The zero pin, the pin subtraction of the warm start, `UniformDirichletRestPressure` seeding
  (`mac_pressure_workspace_3d.cc:170-210`).
- The 4096 and 24576 limits; `compiled_architecture`; `bool bounded` and empty-message statuses.
- `LiquidObstacle3D` and its per-shape clipping, the floating-box freezes
  (`liquid_boundary_geometry_3d.cc:305-764`), the wetted-piston flux override (`1158-1173`),
  `graded_liquid_boundary_geometry_3d.cc` (planes only): replaced by sampled `SolidCoverage` and `FreeSolidPush`.
- `LiquidAnalyticSphere3D`, `LiquidHorizontalPlaneSurface3D`, `exact_outward_interface_curvature`, the
  isolated-drop fit: test inputs now.
- The cavity `initialize` callback and `InitializeSubmergedLiquidCavity3D` as a solver path (a scene may give a
  starting pocket its `rest_pressure`); the inlet particle lattice of the open sea (particles module, using
  `ComputeOpenSeaWave`).
- `reference/.../cavity.py`, `eos.py`, the 2D `pressure.py`; `PressureCorrection.lean`; the pin theorem.
- The velocity-response hook and `connection_scale` inside `SolvePressure` (this design's first answer to
  C1): the joint solve is viscosity's (A5). `mac_viscous_pressure_3d.cc` is ported by viscosity.
- A pressure given at batch borders (`border_cell_pressure`, `kLiquidToBorderCell`): the solves run on the
  whole grid, in one piece or in sweeps of the same iteration (section 5 F).
- `main`'s second solver loop for snow and plastic solids (`liquid_substep_3d.cc:1766-1801`, `2213-2228`):
  replaced by blocking-material faces inside the one solve (A18).
- The viscous jump at a material boundary (`normal_viscous_jump_tail_to_head` on `main`, never set).

## 11. Decisions

1. **Which constant does a sealed blob's pressure get?** Options: (a) keep the volume-weighted mean of
   `carried_pressure` (smooth in time, independent of cell order); (b) mean zero; (c) lowest cell at 0 Pa as on
   `main`. Preview: `blob.pressure_offset = mean(carried_pressure) - mean(pressure)`. Recommendation: (a).
   Decision: (a), the volume-weighted mean of `carried_pressure` (owner-questions Q25); recommended option
   (owner's standing instruction, 2026-11-14).
2. **Air beside cells that are not allocated.** Proposed: blocks that are not allocated and touch no solid are
   flooded from the open top and count as atmosphere; anything else is `kUnknownFrontier`, is given atmosphere
   pressure, is counted in the result, and rejects the try only when tracked gas reaches it (as on `main`).
   Alternative: every unknown frontier rejects the try. Recommendation: the proposal.
   Decision: the proposal; solids fills `grid_edge_is_open_to_sky` (Q26); recommended option (owner's standing
   instruction, 2026-11-14).
3. **One acceptance limit.** Proposed: `max_volume_error_rate = 2e-5 1/s` per cell (the local limit of `main`)
   and the reported volume budget; the absolute `2e-6 m3/s` norm of `main` is dropped because it depends on
   scene size. Alternative: keep both. Recommendation: one limit; the value is calibrated on the four scenes.
   Decision: one limit, `max_volume_error_rate`, calibrated on the four scenes; it is also the stop rule of
   `viscosity::SolveViscosityAndPressure` and of the gas solve (Q27); recommended option (owner's standing
   instruction, 2026-11-14).
4. **Plain doubles in the solve.** `main` uses FP32 fine vectors with FP64 restarts. The programme defers
   reduced precision to measurement. Proposed: all `double` now. Confirm.
   Decision: all `double` now (Q48); recommended option (owner's standing instruction, 2026-11-14).
5. **Two liquids that do not mix.** `main` handles one configured pair on uniform cells and refuses it on
   mixed sizes. Proposed: port it here for mixed sizes (`MaterialBoundary`), with the boundary field from
   transfer. Alternative: carry mixtures only and defer the sharp boundary. Recommendation: port.
   Decision: port, with the boundary field computed here: one distance field per liquid material and
   `FindMaterialBoundaries` (conflict C4, option A); the first pressure commits ship with the mixture density
   only and the exclusion `kMaterialBoundaryNotSupported` (C4, option C as a first state). Liquids that do not
   mix are two materials (Q31 a), and the type is `MaterialBoundary` (Q12); recommended option (owner's standing
   instruction, 2026-11-14).
   Replaced in part (plan decision, 2026-11-14): the sharp boundary is delivered inside the pressure module and
   `kMaterialBoundaryNotSupported` does not exist at its end (`ToName(PressureError)` covers 15 values); the step
   has no `kLiquidsThatDoNotMixTouch`. The rest of the decision stands.
6. **New names** (table in section 9): `carried_pressure`, `filled_volume`, `solid_volume_rate`,
   `flow_per_pressure`, `velocity_per_pressure`, `pressure_distance`, `liquid_share`, `reference_pressure`,
   `face_density`, `volume_error`, `equation_error`, `push` (N s on a face piece, as in `ComputePressurePush`
   and `FreeSolidPush`), `own_weight`, `connection_scale`, `GeometrySnapshot`, `bulk_stiffness` (not
   `stiffness_against_squeeze`, which the review gives to another quantity), `PressureLawKind` values,
   ladder and rung, `blob` for a connected body of liquid in one solve (O4), and `volume_ratio` (O6).
   Decision: all as listed (Q14), `connection_scale` kept as the input of `ComputeOwnWeight` and
   `BuildPressureLadder`; `volume_ratio` by owner answer A4; `CellPlaceIndex` and `FacePlaceIndex` by owner
   answer A3; the rest is the recommended option (owner's standing instruction, 2026-11-14).
7. **A thin solid through one cell with liquid on both sides.** Proposed: reject with `kCellSplitBySolid` as on
   `main`. Alternative: two cell equations per cell, one per `SolidSide`, which changes grid, contact and
   transfer. Recommendation: reject; revisit with the solids module.
   Decision: reject with `kCellSplitBySolid` (Q38); recommended option (owner's standing instruction,
   2026-11-14).
8. **Open sea.** It is a velocity relaxation run in stage `DriveWaves`, and its "outlet" calms waves without
   letting liquid leave. Keep it in `pressure/` under these names, or move it to `solids/` beside the wave
   paddle? Recommendation: keep the programme's placement.
   Decision: keep it in `pressure/`; the step calls `ApplyOpenSeaRelaxation` in stage `DriveWaves` (Q47);
   recommended option (owner's standing instruction, 2026-11-14).

Decided elsewhere and applied here: who runs the joint velocity-and-pressure solve, viscosity (owner answer A5;
the `VelocityResponse` hook is removed) and the names `SolvePressure` and `viscosity::SolveViscosityAndPressure`
(A6); error with context `PressureFailure` in `std::expected` (A2); gas and liquid in one scene from the start
(A10: `kGas`, `SealedBlobCheck`, `ComputeBlobOfCell`, the gas pressure at the liquid surface); sand left to the
sand module (A9); cell eighths and centres filled here (conflict C3, option A, Q30); a liquid cell with no
particle takes its neighbour's density (Q32); recommended option (owner's standing instruction, 2026-11-14) for
each of these not answered by the owner.
Replaced (A10 part): "the liquid reads the gas pressure one step late" now holds only for gas that is not a sealed
blob solved with the liquid (decision 9). Replaced (cell resize and batches design 5.9, applied here as "the
solves are not batched"): decision 10.

All questions are merged, with the other modules' questions, in `owner-questions.md`.

Decisions taken after this design was first written (recorded in the pressure plan and `plan-decisions.md`):

9. **Coupling inside the pressure solve.** A blocking material's inertia enters as a two-point term of weight
   `dt A^2 / m` per `BlockingMaterialFace`; a sealed tracked-gas blob that touches liquid is one unknown with
   squeeze allowance `V / (heat_ratio p0 dt)`, eliminated exactly (rank-one term per blob) and recovered after the
   solve with its row checked; air-end precedence sealed blob, then tracked gas (one step late), then enclosed
   pocket, then 0; the ladder adds material weights and treats blob connections as reference ends; the pieces of
   the joint viscous solve refuse such equations. Method decisions (a) to (f) of the plan's Task 44 (section 5
   C2). Source: A18 and `literature-coupling.md`. Replaces the one-step-late reading of sealed gas (A10 part of
   this design) and, for the liquid side, solids design decision 3 ("one step late").
   Replaced in part by decision 16 (A24): the pieces of the joint viscous solve no longer refuse such equations.
10. **Batched solves.** With `max_loaded_blocks > 0` the solve keeps its working buffers in CPU memory and runs in sweeps
    of the same iteration through `SolveBatches` (`solve_batches.h`, in pressure, shared with viscosity and gas);
    no solver signature changes; batched agrees with one piece within the solver's tolerance, each mode is
    repeatable. Option (ii) (overlapping pieces with a coarse correction) rejected. Source: A15, with A17's
    principle. Replaces "the solves run on the whole grid, also in a batched run" and "no iteration between
    batches" (section 10 of this design, cell resize and batches design 5.9) and the CPU fallback
    `kWholeGridOnCpu`.
11. **Sums by block in both modes.** Grid-wide sums: block partials then one tree sum; per-group totals (blobs,
    sealed gas blobs, free solids, air pockets, gas blobs): per-block partials then a tree sum over the group's
    blocks in ascending block index. Source: A15. Replaces "every blob total is done by one thread, members in
    ascending index" (former step 20) and the per-blob thread sums of plan Tasks 24, 44 (b) and 48.
12. **Ladder layout and direct-solve room, both modes.** Rung 1 numbered by (block, blob, merged id), rung 0
    never terminal, `max_direct_solve_values` (one piece `32 * max_cells`, batched layout `4 * max_cells`), and
    `SolveResult::blobs_without_direct_solve`. Source: A15. It changes the one-piece ladder's numbering, not its
    mathematics.
13. **Pass 1 into whole geometry: `WriteSnapshotRange`.** Source: A15 (closes the gap of a piece grid with
    whole-grid buffers).
14. **A run is all CPU or all GPU.** No CPU solve inside a GPU run; the batched layout copies only planned,
    counted slices. Source: A17.
15. **Interface details of the plan:** `OpenSeaSettings::domain_box` and the call through `solids::DriveWaves`;
    `EquationInputs::lightest_density`, `max_volume_error_rate`; `PressureEquations::max_volume_error_rate`,
    `arrays`, `build_number`; `BufferSizes::max_cells` a multiple of 64; the functions that return
    `kStepTooLongForSurfaceTension`, `kStepTooLongForAirPocket`, `kPressureLawOutOfRange`; a grid-edge face piece
    with open area has atmosphere on its other end; the domain's top side marked in `grid_edge_is_open_to_sky`;
    a progress read before the first iteration; the one-material curvature samples within two cell widths on both
    sides; the dense reference with four cases. Source: plan decision, 2026-11-14.
16. **The joint viscous solve carries both terms (A24).** Should `viscosity::SolveViscosityAndPressure` carry
    blocking-material faces and sealed gas blobs, or should viscous scenes keep refusing them or reading them one
    step late? Options for a springy solid beside a viscous liquid: (A) exclude the pair before the run, (B) a
    one-step-late push for viscous liquids, (C) material faces carried by the joint solve.
    Decision: carried (option C for material faces, and the sealed gas term). No piece refuses equations because
    they hold a material face or a sealed gas blob; the forms without a blob buffer refuse only equations with
    `sealed_gas_blob_count > 0`. The joint solve eliminates each material face exactly (`g_k` added unscaled per
    solved end) and keeps each sealed gas blob as one more unknown `q_j` (diagonal `S_j + sum_c g_c *
    connection_scale_c`, start `c_j`). New in section 3: the blob forms of `ComputePressurePush`,
    `ComputeVolumeError`, `ComputeOwnWeight`, `ComputeStartPressure`, the `CheckFlowImbalance` form with given
    blob pressures, and `ComputeBlockingMaterialVelocities` (which `SolvePressure` also uses);
    `BuildPressureLadder` with a scale adds `g_k` unscaled. No viscous scene reads sealed gas one step late; open
    gas, slivers, held blobs and cells that touch gas still do. Source: owner answer A24 and the owner's choice
    of option C; `plan-decisions.md` "Viscous liquids with blocking materials and sealed gas (A24), and what
    counts as viscous (A25)". Replaces in part decision 9 ("the pieces of the joint viscous solve refuse such
    equations").
    Replaced in part by decision 17 (A26, A27): the joint solve no longer eliminates material faces; the A24 forms
    stay for `SolvePressure`'s arithmetic and are no longer called by viscosity.
17. **Material faces and free solids are velocity unknowns of the joint solve (A26, A27).** Does a viscous liquid
    stick to a blocking material, and may a free solid sit beside a viscous liquid? Decision: it sticks (A26), and
    free solids are lifted (A27): the material face velocities `w` and the free-solid motions `xi = (V, Omega)` are
    unknowns of viscosity's velocity solve, not eliminated. New in section 3, for the joint solve only:
    `struct CoupledVelocities`, the `ComputeVolumeError` and `CheckFlowImbalance` forms with `CoupledVelocities`
    (material flow `A_k w_k - coverage_volume_rate`, free-solid flow `push_e . (xi - xi_before)`, sign as
    `SolvePressure`'s free-solid term; `kLeftOut` without the given parts), `ComputeCoupledPush` (`dt A_k (p_low -
    p_high)`, `dt sum_e (force, torque)_per_pressure p_e`), the `ComputeOwnWeight` form with `material_face_scale`
    (`g_k * scale`, free solids left out of the diagonal). Plan decision, for the owner:
    `EquationInputs::free_solid_motion_before_pressure` (the `kIncluded` free-solid flow needs `xi_before`, which no
    other input holds). The A24 forms without `CoupledVelocities` stay (`SolvePressure`'s arithmetic) and are no
    longer called by viscosity; `SolvePressure` is unchanged (inviscid liquids slide: material faces eliminated, free
    solids as the rank-one term). Source: owner answers A26 and A27; `plan-decisions.md` "A viscous liquid sticks,
    and free solids in the viscous solve (A26, A27)". Replaces in part decision 16 (the joint solve's elimination of
    material faces).

18. **Working buffers, one form (A31).** Reusable working memory is "buffers": the sums and orders take the module's `SumBuffers` /
    `OrderBuffers` (members `sum_buffers`, `order_buffers` of `PressureBuffers`), and every base function that needs
    working memory has one form, which takes them. `SolveBatches` also owns the `GridBuilderBuffers` lent to its
    loaded `GridBuffers` (`BuildGridBuffers` has one form). Source: owner answer A31 (2026-11-14).
19. **Processors of the arguments (A30).** Buffers passed together to one function (piece and whole buffers of
    `WriteSnapshotRange`, blob and coupled-velocity buffers beside the equations) live on one processor: a
    documented precondition checked by a debug assertion, not `kBadInput`; `BufferError` has no processor-mismatch
    value, so nothing is mapped. Source: owner answer A30 (2026-11-14).
20. **Memory estimate on a processor; compile flags (reconcile item 24).** `ComputeBufferBytes(sizes, run_on)` and
    `BuildPressureMemoryEstimate(sizes, run_on)` take the processor, because the sum and order buffers' bytes
    depend on it (`ComputeSumBuffersSize(run_on)`, `ComputeOrderBuffersSize(run_on)`). Pressure's targets keep the
    default flags; a target that needs exceptions or RTTI calls
    `fluidity_apply_compile_flags(<target> EXCEPTIONS <ON|OFF> [RTTI <ON|OFF>])` right after `add_library`.
    Source: reconcile-todo item 24; `plan-decisions.md` "Plans adopt Foundation as built (2026-11-14)".

**Open** (left undecided by the plans; for the owner):

- The material's own volumetric stiffness inside the solve (half of the volumetric response in Fang et al.
  2020): needs materials to drop the volumetric part of `ApplyInternalForce` and material cells to get rows.
- The sealed gas elimination is tested, not proved in Lean.
- `solve_piece_share = 0.25` (batches) and the batched direct-solve room `4 * max_cells` are new constants for the
  owner to confirm.
- If a scene becomes GPU-limited in the batched layout, the geometry is the next row to stream, which needs the
  solids functions that read it (`BuildMotionMap`, `BuildBlockingMaterialFaces`, `CheckSolidExchange`) to run
  batch by batch.
