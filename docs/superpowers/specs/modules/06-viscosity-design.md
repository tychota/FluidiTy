# Viscosity module design

Status: updated 2026-11-14 with the owner's answers (`owner-answers.md`) and `changes-to-apply.md`. Module 5 of
the rebuild. Names shared between modules are fixed in `shared-interface.md`. Brought in line on 2026-11-14 with
the decisions recorded in `plan-decisions.md` and the viscosity and pressure plans (A15 batched solves, A17, the
A18 restriction on the joint solve, and the plan decisions); section 11 lists each one with its source. Updated
again on 2026-11-14 for owner answers A24 (the joint solve carries blocking-material faces and sealed gas; the A18
restriction is withdrawn; no snow exclusion) and A25 (what counts as viscous; the viscous effect per step), and then
for A26 (a viscous liquid sticks to snow and springy solids, whose face velocities beside it are unknowns) and A27
(free solids in the solve; no exclusion by material).

## 1. Purpose

A thick liquid such as syrup resists being sheared or stretched: neighbouring parts of the liquid drag on each
other until they move together. This module measures how fast the liquid is being deformed at many small sample
points of the grid, and solves for the face velocities at the end of the step in which that drag has acted. It
solves this together with the pressure, because at a liquid surface and between two liquids of different
viscosity the drag and the pressure balance each other and cannot be found one after the other.

Owns: the deformation-rate samples on a mixed-size staggered grid (where they sit, which faces they read, their
sample weights); cell and edge viscosity from the ingredients; the conditions at solids and at the liquid
surface; the velocity solve `(face mass + step_duration x drag) velocity = momentum`; the outer loop that
finds the pressure consistent with that velocity solve; `SolveViscosityAndPressure`, the one solve of velocity
and pressure for every scene with a viscous ingredient (owner A5, A6: in such a scene `pressure::SolvePressure`
is not called); the viscosity capability check and its exclusions.

Limits: melting of snow by liquid water is not modelled, so liquid-on-snow scenes are supported mechanically but are
not a validation target (owner note, 2026-11-14).

Does not own: open areas, liquid volumes, the liquid surface, surface tension, the pressure equations and their
stop rule (pressure); `pressure::SolvePressure`, the solve for scenes with no viscous ingredient (pressure);
grid layout and neighbour lookup (grid); ingredient amounts per cell (particles and transfer); snow response
(materials); solid shapes and motion (contact, solids); the fixed-tree reduction utilities (Foundation
`base/`, see section 4).

## 2. What exists today

All paths are on `main`. Y-up matters in three places only: cells are ordered by `(z, y, x)` keys
(`viscosity.cc:57`), the tank floor is the low Y side (`liquid_adaptive_step_3d.cc:51-53`), and the traction
fixtures shear along Y (`native/tests/mac_viscous_traction_cases.h`). The operator itself has no preferred axis.

| Where | What it does | What is wrong |
|---|---|---|
| `native/src/viscosity.cc:764-935` `AssembleMacStrain` | Same-size grid. Three normal rows per cell (two faces, `-1/h, +1/h`, weight `2 mu V`). One shear row per cell edge with four complete cells (four faces, `0.5/h`, weight `4 mu V`). Edge viscosity is the harmonic mean of four cells (`:92-102`); with partly filled cells it is volume-weighted over the four cell quarters, `4 V^2 / sum(V_q / mu_q)` (`:115-148`). | Exponent-separated arithmetic (`FactorProduct`, `ScaledProduct`, `StrainWeight3D`, `:36-55, 150-158`) exists only for the "representable" checks the owner dropped (naming review 7.4). |
| `viscosity.cc:965-1194` `AssembleCutCellStrainWithBoundary` | Shear rows at edges with missing cells. Checks the liquid quarters around the edge are connected through open faces (`:973-995`), then moves each derivative to the nearest face pair that shares an open connection (`:999-1045`). | Runs only with no solid or the axis-aligned tank (`liquid_substep_3d_cpu.cc:657-662`). Adds no shear row at a solid, so tank sides are free-slip. |
| `viscosity.cc:1290-1327` `MakeMacTangentialWallStrain3D` | Half-cell shear row at a flat grid-aligned solid: coefficient `-/+ 1/h`, prescribed part from the solid's tangential speed. | Called by tests only. The production same-size path never sticks liquid to a solid. |
| `native/src/liquid_substep_3d_cpu.cc:540-700` `ProjectUniformViscosity` | Measures liquid volume per cell eighth, replaces `1/h` by `1/(open extent)` in tank cells (`:690-698`), then calls the coupled solve. | Third place that edits normal rows. |
| `native/src/liquid_adaptive_step_3d.cc:98-173` `FitFaceGradient`, `:178-247`, `:251-437` | Mixed sizes. Normal rows are the finite-volume sum over up to four face pieces per side, coefficient `sign x open_area / wet_volume`, weight `2 mu wet_volume`; the three rows sum to the pressure flow imbalance (`:331-370`). Shear rows sit at cell centres (weight `4 mu wet_volume`) and use an exact fit through the four nearest independent face positions, including wet-solid centroids that carry the solid velocity (`:279-302`). | One constant viscosity only (`:261`). `FitFaceGradient` scans every face of the grid for every cell (`:112-125`) and allocates. Unrelated to the same-size rule, so the two paths give different answers on the same same-size grid. |
| `viscosity.cc:1331-1511` velocity solve | `(M + dt E^T W E) u = rhs` by conjugate gradients with a diagonal scaling. Rows are stored as `sqrt(dt W) x coefficient` so operator and diagonal share coefficients (`:1360-1406`). Stops at `1e-10 x max|rhs|`, at most 256 iterations, and recomputes the true error before it accepts (`:1485-1502`). | Scatter form (`:1424-1433`): one loop adds into many faces, which cannot run in parallel in a fixed order. |
| `native/src/mac_viscous_pressure_3d.cc:151-461` `SolveCoupled` | Outer loop on pressure. Face inertia `dt A / scale` and surface pressure lift (`:189-206`); pressure diagonal uses the viscous diagonal (`:208-230`); prescribed faces eliminated into the right-hand side (`:232-252`); each iteration solves the velocity, measures the physical flow imbalance (`:82-134`), and keeps up to 32 measured responses (`:393-459`). Affine wall and source terms are added only for the physical candidate, never for a search direction. | Six public entry points for one solve (`mac_viscous_pressure_3d.h`). `CompensatedSum`, per-product representable checks, strain rows re-indexed by binary search on every solve (`:60-80`). |
| `native/cuda/viscosity_velocity_block_3d.cu:44-126` | The same velocity solve on the GPU. | Launched as `Solve<<<1, 1>>>`: the whole solve runs on one GPU thread (CUDA report 8.1). `viscosity_velocity_block_3d_host.cc:70-87` allocates and synchronises on every inner solve, so the outer loop crosses to the CPU each time (CUDA report 8.3). |
| `viscosity.cc:197-760, 1651-1719` `ViscousStep`, `AssembleMacStrain2D` | Dense 2D solve, at most 64 unknowns, explicit or implicit chosen by a rounded-up spectral bound. | 2D and certificate machinery; not used by the step. |
| `liquid_substep_3d.cc:1653-1659`, `liquid_eos_sensor_3d.cc:132, 254-349` | Cell viscosity is the volume-weighted mean over ingredients; cells with no sample take the nearest same-liquid value. | Lives in a pressure-law file. |
| `liquid_substep_3d.cc:1762-1771, 2887-2908`, `liquid_adaptive_step_3d.cc:461-472` | Capability exclusions (listed in 5.7). | Returned as four different status codes, some with empty messages. |
| `reference/.../viscosity.py`, `traction.py`; `proofs/Fluidity/Viscosity.lean` | 2D dense reference and the two traction fixtures; energy identities and the trace bound. | 2D only; the proofs mostly concern the explicit branch. |

The old paper (`docs/papers/fluidity-method.typ:219-234`) still describes viscosity then pressure as two steps
with a coupled solve as a fallback. The code moved on: whenever viscosity is positive the step calls the
coupled solve (`liquid_substep_3d.cc:2180`) and the viscosity stage is an empty marker (`:1619`).

## 3. Public interface

Folder `native/fluidity/viscosity/`, namespace `fluidity::viscosity`. Every name below is NEW unless marked;
`SolveViscosityAndPressure` (owner A6, replaces `ApplyViscosity`), `sample_weight`, `open_area`,
`cell_eighth`, `squeeze_allowance`, `flow_imbalance`, `SolidVelocitySample`, `Inputs`/`Settings`/`Result`,
`deformation_rate`, `stretch`, `shear` (owner A4) are DECIDED.

| File | Kind | Content |
|---|---|---|
| `viscosity_error.h` | shared | `ViscosityError`, `ViscosityFailure` |
| `deformation_rate.h` | shared, formulas | sample types, weight and fit formulas (`FLUIDITY_CPU_AND_GPU`) |
| `solve_viscosity_and_pressure.h` | CPU only | buffers, the capability check, build, the joint solve, read, `ToMessage`, `ToName`, memory estimate entries |
| `solve_viscosity_and_pressure_cpu.cc`, `solve_viscosity_and_pressure_gpu.cu` | private | the two implementations |
| `viscosity_samples_cpu.cc`, `viscosity_samples_gpu.cu`, `velocity_solve.h` | private | sample building; the inner solve, declared for tests |

```cpp
// viscosity_error.h
enum class ViscosityError : std::uint8_t {
  kNone,
  kBadInput,                   // A count, version or value is out of range or not finite; a needed input is
                               // null (geometry cell_eighths or face_piece_centres, pressure_buffers); the
                               // solve's block_range is not every block of the grid; an input bound to another
                               // grid version (plan decision: no kWrongGridVersion); NEW (A24) a
                               // SolvedCouplingArrays span whose size is not its count, or more sealed gas blobs
                               // than max_sealed_gas_blobs. (The A18 clause "equations carrying coupling terms" is
                               // withdrawn by A24.)
  kTooFewNearbyFaces,          // A fitted shear sample found fewer than four independent face positions.
  kLiquidAtEdgeNotConnected,   // Liquid quarters around an edge do not touch through an open face.
  kNoFacePairForShear,         // An edge sample with positive weight has no usable pair of faces.
  kNotFinite,                  // A weight, coefficient or solver value became infinite or NaN.
  kVelocitySolveDidNotConverge,   // The inner solve used all its iterations. Shorten the step.
  kPressureSolveDidNotConverge,   // The outer loop used all its iterations or found no usable direction.
  kBuffersTooSmall,            // More samples or coefficients than the buffers were built for.
  kRunOnCallFailed,            // CheckCanRunOn failed for the chosen processor, or a call to it failed.
  kOutOfMemory,                // NEW (reconcile item 25). BuildViscosityBuffers could not allocate; magnitude =
                               // bytes asked. Returned only by the building functions.
  kCancelled,                  // NEW. *cancel_requested was set; neither output was written.
  // Capability exclusions (5.7). Each stays until its surface-force test passes. (A24 removed kSnowNotSupported and
  // A27 kFreeSolidNotSupported, which stood first: snow, springy solids and free solids are unknowns of the solve.)
  kMovingSolidNotSupported,    // A driven solid cuts partly through cells of viscous liquid.
  kInclinedSolidNotSupported,  // A solid surface that is not axis-aligned touches viscous liquid.
  kSizeChangeNotSupported,     // The liquid surface, a solid or a viscosity change touches a size change.
};

struct ViscosityFailure {
  ViscosityError error = ViscosityError::kNone;
  grid::CellId cell{};            // Stable id of the first failing cell in id order. Zero when unused.
  grid::FacePlaceIndex face{grid::kNoIndex};  // Valid for one grid version only, or kNoIndex.
  double magnitude = 0;           // The offending value or last error size; unit depends on `error`.
};
```

```cpp
// deformation_rate.h
inline constexpr int kMaxFacesPerSample = 8;        // Four face pieces on each side of a big cell.
inline constexpr double kFitReach = 1.5;            // Cell widths. Face positions farther away are ignored.
inline constexpr double kFitIndependence = 1.0e-4;  // In cell-width units; from liquid_adaptive_step_3d.cc:132.

enum class SampleKind : std::uint8_t {
  kStretch,      // Rate of stretch along one axis, at a cell. Papers: normal strain rate.
  kEdgeShear,    // Rate of shear between two axes, at a cell edge shared by same-size cells.
  kFittedShear,  // Rate of shear between two axes, at a cell centre, from a local fit.
};

struct FaceCoefficient {
  grid::FacePlaceIndex face{};  // Valid for one grid version only.
  double coefficient = 0;       // 1/m. Multiplies the face velocity.
};

// One entry of the deformation rate measured at one place. Papers: one row of the strain operator.
struct DeformationRateSample {
  SampleKind kind = SampleKind::kStretch;
  Axis first_axis = Axis::kX;
  Axis second_axis = Axis::kX;          // Equal to first_axis for kStretch.
  grid::CellId cell{};                  // Stable id of the active cell the sample belongs to.
  std::int32_t first_coefficient = 0;   // Index into the coefficient array.
  std::int32_t coefficient_count = 0;   // 1..kMaxFacesPerSample.
  double sample_weight = 0;             // kg m2/s. 2 x viscosity x volume, doubled for shear.   DECIDED word
  double rate_from_solids = 0;          // 1/s. The part of the rate fixed by solid velocities.
};

// kg m2/s. Weight of a stretch sample.
FLUIDITY_CPU_AND_GPU double ComputeStretchWeight(double viscosity, double filled_volume);
// kg m2/s. Weight of a shear sample over four volumes with four viscosities; zero if any wet part has none.
FLUIDITY_CPU_AND_GPU double ComputeShearWeight(const double filled_volume[4], const double viscosity[4]);
// Pa s. Volume-weighted mean over the ingredients of one cell.
FLUIDITY_CPU_AND_GPU double ComputeMixtureViscosity(const double* ingredient_volume,
                                                    const double* ingredient_viscosity, int ingredient_count);
struct NearbyFacePosition {
  grid::FacePlaceIndex face{};   // Or grid::kNoIndex for a solid surface point with a known velocity.
  Vec3 offset{};                 // Cell widths, from the cell centre.
};
struct FittedChange {
  FaceCoefficient per_axis[3][4];  // d(face velocity)/d(axis), four coefficients each, 1/m.
};
// Picks the four nearest independent positions (ties: lower face place index) and fits the exact linear change.
FLUIDITY_CPU_AND_GPU ViscosityError ComputeFittedChange(const NearbyFacePosition* positions, int position_count,
                                                        double cell_width, FittedChange* fitted_change);
// NEW (A25). Below this viscous effect per step, one step changes the velocity difference between two neighbouring
// cells by less than 0.1 % (factor 1 / (1 + 4 N) for the mode k h = pi): viscosity then changes nothing visible.
inline constexpr double kSmallestVisibleViscousEffectPerStep = 2.5e-4;
// NEW (A25). No unit: viscosity (Pa s) * step_duration (s) / (density (kg/m3) * cell_width^2 (m2)); papers: the
// diffusion number. Example: 1e-3 Pa s, 0.01 s, 1000 kg/m3, 0.01 m gives 1e-4. Read by the step's planner only.
FLUIDITY_CPU_AND_GPU double ComputeViscousEffectPerStep(double viscosity, double step_duration, double density,
                                                        double cell_width);
```

```cpp
// solve_viscosity_and_pressure.h
struct ViscosityBufferSettings {
  RunOn run_on = RunOn::kCpu;
  std::int64_t max_face_count = 0;         // == max_cell_count / 64 * 240 (plan decision), else kBadInput.
  std::int64_t max_cell_count = 0;         // A positive multiple of 64 (plan decision), else kBadInput.
  std::int64_t max_sample_count = 0;       // At most 15 per cell: 3 stretch + 12 shear.
  std::int64_t max_coefficient_count = 0;  // At most kMaxFacesPerSample per sample.
  // NEW (A15). 0: one piece. > 0: the batched layout; per-place, per-sample and per-coefficient rows live in CPU
  // memory and a pressure::SolveBatches with stacked widths {kTouchingCells, kTouchingCells} is held inside.
  std::uint32_t max_loaded_blocks = 0;
  // NEW (A24). At least the equations' sealed_gas_blob_count (kBadInput otherwise); 0 without gas.
  std::uint32_t max_sealed_gas_blobs = 0;
  // NEW (A26, A27). At least inputs.material_faces.size() and inputs.free_solids.size().
  std::uint32_t max_material_faces = 0;
  std::uint32_t max_free_solids = 0;
};
std::int64_t ComputeViscosityBufferBytes(const ViscosityBufferSettings& settings);  // Same table as the build.

class ViscosityBuffers;  // Preallocated, reused every step; lives on the CPU or the GPU. Move-only. Holds the
                         // SumBuffers and OrderBuffers every sum, order, prefix sum and grouping takes (A31).
// kBadInput for inconsistent counts; kRunOnCallFailed when the processor cannot run; kOutOfMemory when an
// allocation fails (the only function of the module that returns it). Every other function requires the arrays
// it is given on the buffers' processor (debug assertion, A30).
std::expected<ViscosityBuffers, ViscosityFailure> BuildViscosityBuffers(const ViscosityBufferSettings& settings);
// NEW (A15). The buffers' SolveBatches and its copy report; null in one piece.
const pressure::SolveBatches* FindSolveBatches(const ViscosityBuffers& buffers);
const pressure::SolveCopyReport* ReadSolveCopyReport(const ViscosityBuffers& buffers);
// NEW (reconcile). Zeroes the held SolveBatches' copy report; does nothing for one-piece buffers; allocates and
// copies nothing. The step calls it once per try.
void ResetSolveCopyReport(ViscosityBuffers& buffers);

struct ViscositySceneSummary {             // Filled once by the step's capability check.
  bool has_viscous_ingredient = false;
  // (A24: has_snow removed; A27: has_free_solid_touching_liquid removed.)
  bool has_driven_solid_through_cells = false;
  bool has_inclined_solid = false;
  bool has_size_change_at_surface_solid_or_viscosity_change = false;
};
// Returns the first exclusion of 5.7 that the scene meets, in the order of ViscosityError.
std::expected<void, ViscosityFailure> CheckViscosityCanRun(const ViscositySceneSummary& scene);

struct ViscositySettings {               // Hashed with the scene; holds no cancel flag.
  int max_velocity_iterations = 256;        // From viscosity.cc MacViscousVelocityBlockOptions3D.
  double velocity_tolerance = 1.0e-10;      // No unit; times the largest momentum entry, kg m/s.
  int max_pressure_iterations = 256;        // From mac_viscous_pressure_3d.cc:49.
  int kept_response_count = 32;             // From kCoupledPressureRestartLength3D.
  // The acceptance limits of the pressure module; only max_volume_error_rate and jump_error_fraction are read.
  pressure::SolveSettings pressure_acceptance{};
};

struct ViscosityInputs {
  const grid::StaggeredGrid* grid = nullptr;
  // Cell blocks to work on; border cells are read only. BuildViscositySamples accepts any range;
  // SolveViscosityAndPressure needs every block of the grid (in the batched layout it cuts the grid itself
  // through the buffers' SolveBatches, A15).
  grid::CellBlockRange block_range{};
  // Same step, same grid version. cell_eighths and face_piece_centres must not be null.
  const pressure::GeometrySnapshot* geometry = nullptr;
  // May hold blocking-material faces and sealed gas blobs (A24; the A18 "must be 0" is withdrawn).
  const pressure::PressureEquations* pressure_equations = nullptr;
  // NEW (plan decision). The buffers that built pressure_equations (CheckFlowImbalance, ApplyZeroMeanToSealedBlobs
  // and ApplyPressureOffset take PressureBuffers&). Null: kBadInput in the solve; not read by BuildViscositySamples.
  pressure::PressureBuffers* pressure_buffers = nullptr;
  // One entry per liquid material set (owner A8); all liquids are in one material set, so the span has exactly
  // one entry, the liquid set's. Snow, springy-solid and gas marker sets are not passed (A10).
  // Only ingredient_volume is read: m3 per cell and scene ingredient.
  std::span<const transfer::CellsFromParticles* const> cells_from_particles;
  std::span<const double> ingredient_viscosity;             // Pa s, by scene ingredient index. Zero for snow.
  std::span<const contact::Solid> solids;                   // Ascending solid_id. For solid velocities at points.
  // NEW (A26). The band faces of every blocking set (solids::BuildViscousMaterialFaces): velocity unknowns of this
  // solve; the liquid sticks to them. Empty without a blocking set.
  std::span<const solids::ViscousMaterialFace> material_faces;
  // NEW (A27). One per free solid touching liquid, in FreeSolidPush::solid_index order: six velocity unknowns each.
  std::span<const solids::FreeSolidForViscousSolve> free_solids;
  // m/s. The array pressure::EquationInputs::velocity_before_pressure pointed to for these equations.
  const grid::FaceValues<double>* velocity_before_pressure = nullptr;
  double step_duration = 0;                                 // s, positive.
};

struct ViscosityResult {
  pressure::FlowImbalanceReport flow_imbalance;  // From the accepted velocity, as SolvePressure reports it.
  int pressure_iteration_count = 0;
  int velocity_solve_count = 0;
  int velocity_iteration_count = 0;           // Summed over all inner solves.
  double largest_momentum_error = 0;          // kg m/s, true error of the accepted velocity.
  // W. Sum of sample_weight x rate^2 at the accepted velocity: one partial per cell block in the grid's fixed
  // block order, then math::TreeSum over the partials.
  double power_lost_to_viscosity = 0;
};

// NEW (A24). The coupling outputs, as pressure::SolvedArrays has them. Sizes equal the equations'
// blocking_material_face_count and sealed_gas_blob_count (kBadInput otherwise); empty when both are 0.
struct SolvedCouplingArrays {
  std::span<double> blocking_material_velocity_after_pressure;  // m/s along +axis, one per face, input order.
  std::span<double> sealed_gas_pressure;                        // Pa above atmosphere at the step end, one per blob.
  std::span<double> material_face_velocity_after_solve;         // NEW (A26). m/s, one per inputs.material_faces.
  std::span<solids::FreeSolidMotion> free_solid_motion;         // NEW (A27). One per inputs.free_solids.
};

// Builds this step's samples into the buffers (section 5, steps 1 to 5). Changes nothing else.
std::expected<void, ViscosityFailure> BuildViscositySamples(const ViscosityInputs& inputs,
                                                             ViscosityBuffers& buffers);
// Stage 11 `SolveViscosityAndPressure` (owner A5, A6; replaces ApplyViscosity). Finds the face velocity and
// the cell pressure together, in one solve, for a scene with a viscous ingredient: builds the samples, then
// solves (face mass + step_duration x drag) velocity = momentum inside an outer loop on the pressure until
// pressure::CheckFlowImbalance accepts (section 5, steps 1 to 12). For such a scene the step calls this in
// place of pressure::SolvePressure, which is not called at all; with no viscous ingredient the step calls
// pressure::SolvePressure and never this function.
// Inputs: `inputs.block_range` covers every block of the grid. The equations may carry blocking-material faces and
// sealed gas blobs (A24, section 5 "Coupled terms"): each material face's velocity and each blob's pressure are
// solved in the same solve and written to coupling_arrays, as SolvePressure writes SolvedArrays.
// `cancel_requested` may be null; it is read before each inner solve and, in the batched layout, before every
// batch load. Outputs: velocity_after_pressure (m/s, every face place) and cell_pressure (Pa, 0 in cells without
// an equation), both bound to inputs.grid's version and distinct from every input array.
// On success writes both outputs (in the batched layout in one write-back sweep after acceptance); on any
// failure, including kCancelled, writes none of the four. In the batched layout (A15) it runs in sweeps through the
// buffers' SolveBatches and agrees with the one-piece run within the section 6 tolerances.
std::expected<ViscosityResult, ViscosityFailure> SolveViscosityAndPressure(
    const ViscosityInputs& inputs, const ViscositySettings& settings,
    const std::atomic<bool>* cancel_requested, ViscosityBuffers& buffers,
    grid::FaceValues<double>& velocity_after_pressure, grid::CellValues<double>& cell_pressure,
    SolvedCouplingArrays coupling_arrays = {});   // NEW (A24) last parameter.
// W. Power lost for any velocity with the built samples; zero for rigid motion. Summed as in ViscosityResult.
std::expected<double, ViscosityFailure> ComputePowerLostToViscosity(const ViscosityBuffers& buffers,
                                                                    const grid::FaceValues<double>& velocity);
struct ViscositySamplesSnapshot {             // CPU copy for tests and diagnostics.
  std::vector<DeformationRateSample> samples;
  std::vector<FaceCoefficient> coefficients;
  std::vector<double> cell_viscosity;         // Pa s, by cell index.
};
ViscositySamplesSnapshot ReadViscositySamples(const ViscosityBuffers& buffers);
std::string_view ToMessage(ViscosityError error);
// NEW. Stable code: the enumerator's name without the k, lower snake case ("snow_not_supported").
std::string_view ToName(ViscosityError error);
// ComputeViscosityBufferBytes is the sum of these rows (base/memory_count.h). NEW (A15): with max_loaded_blocks > 0
// every row counted per cell place, face place, sample or coefficient (kept directions and responses included) is
// returned with where = MemoryWhere::kCpu at the limits and again with where = kRunOn at max_loaded_blocks
// (samples 15 * 64 and coefficients 8 * 15 * 64 per loaded block); block-counted rows and the progress record
// stay kRunOn; plus pressure::BuildSolveBatchesMemoryEstimate. NEW (A24): "viscosity: sealed gas blob vectors",
// given max_sealed_gas_blobs, 70 x 8 bytes, always kRunOn. NEW (A26, A27): "viscosity: material face vectors" (given
// max_material_faces, 12 x 8 + 16 bytes, streamed like face rows), "viscosity: free solid vectors and blocks" (given
// max_free_solids, (12 x 6 + 2 x 36) x 8 bytes, kRunOn), "viscosity: sample solid coefficients" (given
// max_sample_count, 4 + 6 x 8 bytes).
std::vector<MemoryEstimateEntry> BuildViscosityMemoryEstimate(const ViscosityBufferSettings& settings);
```

## 4. Consumes from other modules

Each item is marked EXISTS (Foundation, including the `base/` additions) or PROVIDED (by a designed module,
with the name as that module's section 3 has it).

Foundation (EXISTS): `Vec3`, `Mat3`, `math::Dot`, `math::Cross`, `Axis`, `kUp`, `RunOn`, `CheckCanRunOn`,
`FLUIDITY_CPU_AND_GPU`, `cpu_only.h`, `std::atomic<bool>` for the cancel input.

Foundation `base/` additions (EXISTS, `shared-interface.md` section (a)). This design first wrote
`ComputeFixedTreeSum`, `ComputeFixedTreeDot`, `ComputeFixedTreeLargest` and `GroupByDestination`:

A15: every dot product and `power_lost_to_viscosity` is block partials then `math::TreeSum` in both modes;
`ComputeTreeDot` over a whole buffer is no longer used for the solve's dots. `ComputeLargestMagnitude` stays
(exact in any order). Foundation as built also gives `MemoryEstimateEntry::where` (`MemoryWhere { kRunOn, kCpu }`),
`BuildBuffer(..., CpuMemory::kLockedForCopies)`, `ComputeExclusivePrefixSum` taking `OrderBuffers` (one form, A31); buffers on one processor are a precondition, so
no processor mismatch is mapped (A30) (EXISTS, `plan-decisions.md`).

```cpp
namespace fluidity {
// base/tree_sum.h (shared). Block partials of power_lost_to_viscosity and of every dot are summed with it.
namespace math { FLUIDITY_CPU_AND_GPU double TreeSum(const double* values, std::size_t count); }
// base/ordered_sums.h. Sums over a fixed binary tree on the indexes; same bytes on every run on one processor.
std::expected<double, BufferError> ComputeTreeSum(const Buffer<double>& values, std::size_t count);
std::expected<double, BufferError> ComputeTreeDot(const Buffer<double>& first, const Buffer<double>& second,
                                                  std::size_t count);
std::expected<double, BufferError> ComputeLargestMagnitude(const Buffer<double>& values, std::size_t count);
// base/stable_order.h. Groups items by destination and keeps item order inside a group.
std::expected<void, BufferError> BuildGroupsByDestination(
    const Buffer<std::uint32_t>& destination, std::size_t item_count, std::uint32_t destination_count,
    Buffer<std::uint32_t>& first_of_group, Buffer<std::uint32_t>& item_order);
template <typename Value> class Buffer; struct MemoryEstimateEntry;
namespace math { double Sqrt(double); double Abs(double); bool IsFinite(double); }
}
```

A `grid::FaceValues<double>` or `CellValues<double>` is passed to the sums through its `buffer()`.

Grid (PROVIDED by grid 3.1 to 3.7). What this design first wrote, and the grid's declaration that serves it:

| First written here | Provided by grid |
|---|---|
| `MixedSizeGrid` | `grid::StaggeredGrid`; `tables(run_on)` gives `grid::GridTables` for the shared formulas |
| `CellBlockRange` with `std::int32_t` fields | `grid::CellBlockRange { std::uint32_t first_block; std::uint32_t block_count; }` |
| `FaceField`, `CellField` | `grid::FaceValues<double>`, `grid::CellValues<double>` |
| `cell_id` (`std::int64_t`), `face_index` (`std::int32_t`), `-1` for none | `grid::CellId`, `grid::FacePlaceIndex`, `grid::kNoIndex`; cells in arrays are `grid::CellPlaceIndex` |
| `FacePiece`, `ReadFacePiece` | `ToAxis(face)`, `FindCellLow`, `FindCellHigh`, `ComputeFacePosition`, `ComputeFaceArea` |
| `FindFacePiecesOfCell(grid, cell, axis, side, out)` | `FindFacePieces(tables, cell)` (up to 24, ascending index); keep those with `ToAxis(face) == axis` whose `FindCellLow` or `FindCellHigh` is the cell |
| `CellsAroundEdge { cell[4]; same_size; }`, `FindCellsAroundEdge` | `grid::CellsAroundEdge { cells[4]; is_same_size; }`, `FindCellsAroundEdge(tables, cell, first_axis, second_axis, quarter)` (grid 3.7) |
| `FindNearbyFaces(grid, cell, axis, reach, out, max)` | `grid::FindNearbyFaces(tables, ComputeCellCenter(tables, cell), axis, ToCellBlockIndex(cell), &error)`. `kFitReach` equals `grid::kWeightCurveReach` (1.5); at most `kMaxNearbyFaces` (76) faces |
| fixed block order for the power-lost partials | the grid's fixed block order of `CellBlockIndex` |

Pressure (PROVIDED by pressure 3; pressure has no hook into this solve, and `connection_scale` is an input of
`ComputeOwnWeight` and `BuildPressureLadder` only, owner A5). What this design first wrote, and pressure's
declaration:

| First written here | Provided by pressure |
|---|---|
| per face piece `open_area` | `FacePieceGeometry::open_area` |
| per face piece `solid_flux` | `FacePieceGeometry::solid_volume_rate` |
| per face piece `solid_normal_velocity` | not needed: `solid_volume_rate` already holds blocked area times that speed |
| per cell `open_volume` | `CellGeometry::open_volume` |
| per cell `liquid_volume` | `CellGeometry::filled_volume` |
| per cell `liquid_volume_in_cell_eighth[8]` | `CellEighthGeometry::filled_volume_in_cell_eighth`, through `GeometrySnapshot::cell_eighths` (C3) |
| per cell `liquid_covered_solid_centre` | `CellEighthGeometry::liquid_covered_solid_centre` and `liquid_covered_area` (C3) |
| per face piece `liquid_covered_open_centre` | `FacePieceCentre::liquid_covered_open_centre`, through `GeometrySnapshot::face_piece_centres` (C3) |
| `face_mass` (0 = given) | `pressure::ComputeFaceInertia(equations, face_inertia)`; 0 on a face piece with no connection |
| `surface_pressure_push` | the given part of `ComputePressurePush(equations, pressure, GivenTerms::kIncluded, push)`: reference pressures, surface tension, jumps |
| `ApplyPressurePush(equations, pressure, momentum)` | `ComputePressurePush(equations, pressure, given_terms, push)`; push is in N s = kg m/s |
| `GivenFlows { kIncluded, kLeftOut }` | `pressure::GivenTerms { kIncluded, kLeftOut }` |
| `ComputeFlowImbalance(equations, pressure, velocity, given_flows, out)` | `pressure::ComputeVolumeError(equations, pressure, velocity, given_terms, volume_error)` |
| `ComputeScaledPressureDiagonal` | `pressure::ComputeOwnWeight(equations, &connection_scale, own_weight)` |
| `ApplyStartPressure` | `pressure::ComputeStartPressure(equations, pressure)` (the carried pressure) |
| `in_equations` per cell | `pressure::ComputeWasSolved` (`cell_was_solved`) |
| `is_pinned` | none: pressure pins no cell. `ApplyZeroMeanToSealedBlobs` for a direction, `ApplyPressureOffset` for the result |
| `StopRule`, `ComputeStopRule` | `pressure::CheckFlowImbalance(equations, pressure, velocity, settings, buffers)` with `pressure::SolveSettings` |
| `PressureResult` | `pressure::FlowImbalanceReport` |
| the stretch-sample invariant | `pressure::ComputeFlowImbalance(equations, velocity, flow_imbalance)` (tests) |
| batch cut, loaded piece, copies (A15) | `pressure::SolveBatches`, `BuildSolveBatches`, `BuildSolveBatchesMemoryEstimate`, `ApplySolveBatches`, `LoadSolveBatch`, `WriteBatchValues`, `ReadBatchValues`, `WriteBatchGroups`, `ReadBatchGroups`, `SolveCopyReport`, `ReadSolveCopyReport`, `ResetSolveCopyReport` (`solve_batches.h`) |
| "accepted" | `CheckFlowImbalance` returns a report; `PressureError::kNotConverged` means "not yet" (plan decision) |

A24 (pressure, `plan-decisions.md` A24 section): no piece refuses coupled equations. This module calls the forms
with a sealed-gas `Buffer<double>`: `ComputePressurePush(equations, pressure, sealed_gas_pressure, given_terms,
push)`, `ComputeVolumeError(..., sealed_gas_pressure, velocity, given_terms, volume_error,
sealed_gas_volume_error)`, `ComputeOwnWeight(..., own_weight, sealed_gas_own_weight)` (material weights unscaled),
`ComputeStartPressure(..., pressure, sealed_gas_pressure)`, `ComputeBlockingMaterialVelocities` and the
`CheckFlowImbalance` form with the blob pressures given (PROVIDED). The A18 refusal (`kBadInput` for equations with
either term) is withdrawn. In the batched layout each piece is one sweep and
accepts whole arrays in CPU memory or on the processor (pressure, A15).

Transfer (PROVIDED by transfer 3): `transfer::CellsFromParticles::ingredient_volume` (m3 of each scene
ingredient in each cell that holds particles; this design first wrote `transfer::IngredientVolumes`), the
`CellsFromParticles` of the one liquid material set (`ComputeCellsFromParticles` with `material_is_counted`,
owner A8; every liquid is in that set).
Transfer reads nothing from this module; it records `kFromSolve` for faces this solve wrote.
Contact (PROVIDED by contact 3.2): `contact::Solid`, `SolidVelocitySample` (DECIDED name) and
`ComputeSolidVelocitySample(solids, solid_count, elapsed, point, reach)`.
Materials (PROVIDED by materials 3): `materials::MaterialTable::ingredient_viscosity()`, Pa s by scene
ingredient index (zero for snow and springy solids; zero for gas).
Solids (PROVIDED by solids 3, NEW A26, A27): `solids::ViscousMaterialFace` (`BuildViscousMaterialFaces`,
`ToViscousMaterialFaces`), `solids::FreeSolidForViscousSolve` (`ToFreeSolidsForViscousSolve`), `solids::FreeSolidMotion`;
the step writes the outputs back with `ApplyViscousSolveToMaterials` and `ApplySolvedMotionToSolids`. Pressure (A26,
A27): `CoupledVelocities`, the `ComputeVolumeError` and `CheckFlowImbalance` forms with it, `ComputeCoupledPush`,
the `ComputeOwnWeight` form with `material_face_scale`, and the existing `FreeSolidPush` entries.
Solids (PROVIDED by solids 3): `solids::SolidsSummary` from `ComputeSolidsSummary` (`has_driven_solid`,
`has_free_solid`, `has_inclined_solid`, `has_blocking_material`), from which the step fills the solid fields
of `ViscositySceneSummary`.
Step (PROVIDED by step 3): fills `ViscositySceneSummary` in its capability check and calls
`CheckViscosityCanRun`; runs stage 11 `SolveViscosityAndPressure` when any ingredient has viscosity, else
stage 12 `SolvePressure` (owner A6; A25: "has viscosity" means a viscosity setting above 0, fixed for the run); passes the cancel flag; passes the outputs to
`transfer::SaveGridAfterPressure`; sets `pressure_buffers`; builds `ViscosityBuffers` with
`max_loaded_blocks = RunPlan::max_solve_loaded_blocks` in `RunMode::kBatchedWithSolvesInPieces` (A15) and
`max_sealed_gas_blobs` = gas's `max_gas_blobs`; passes the step-owned `SolvedCouplingArrays` and then calls
`solids::ApplyPressureToMaterials`, as in stage 12 (A24; no `SealedGasCoupling` exists); decides stage 11 once per run
when a liquid ingredient's viscosity is above 0 (A25) and prints `ComputeViscousEffectPerStep` per liquid from its
planner.
Cell resize and batches (PROVIDED by the cell resize and batches design, 3.5 and 5.9, as amended by
`plan-decisions.md` "Batched solves (A15)"): `cell_resize::ApplyGridOnlyRegionToGrid` adds the ingredient
volumes of grid-only cells to `CellsFromParticles` before this module reads them; `batches::BuildBatchPlan`
chooses `SolvePlace::kWholeGridOnProcessor` (the joint solve once on the whole grid, `block_range` = all blocks)
or `kInPieces` (the same solve in sweeps) and sizes `max_solve_loaded_blocks` from the loaded rows of every solver
buffer of the scene.

## 5. Method

Ported unchanged unless a step says FIXED or NEW. Symbols appear only here and in the paper.

Equations (papers: variational Stokes, backward Euler). With face velocities `u`, face mass `M`, sample
coefficients `E`, sample weights `W`, solid part `g = rate_from_solids`, flow-imbalance operator `D`, squeeze
allowance `C`, pressure `p`, step duration `dt`:

```text
(M + dt E^T W E) u = M u_before + dt A (p_low - p_high) + dt A push - dt E^T W g
D u + C (p - p_previous) = given flows
```

`W = 2 mu V` for a stretch sample and `4 mu V` for a shear sample (the shear entry appears twice in the
tensor). Power lost is `sum W (E u + g)^2`, never negative.

**Build (steps 1 to 5).**

1. Cell viscosity. For each cell in the range with `filled_volume > 0`: `ComputeMixtureViscosity` over its
   ingredient volumes (from the one liquid material set). A cell with liquid but zero
   ingredient volume takes the value of the touching cell with volume and the lowest `CellId`; if none, zero.
   FIXED: moved here from the pressure-law code; the "nearest same-liquid" search becomes this one-ring rule.
2. Stretch samples. One per cell in the equations and per axis. Coefficient of each of its face pieces (up to
   four per side): `sign x open_area / open_volume`, sign `+` on the high side. Blocked area adds
   `sign x solid_volume_rate / open_volume` to `rate_from_solids`. Weight `2 x viscosity x filled_volume`.
   Invariant: `open_volume x (sum of the three samples)` equals `pressure::ComputeFlowImbalance` for that cell
   with no squeeze term. FIXED: the same-size path used `1/h` and a separate tank correction; this one rule
   gives `1/h` on a full cell. NEW: the divisor is the open volume, so a cell cut by the liquid surface keeps
   the exact derivative (decision 5).
3. Shear samples. For each cell, each axis pair (XY, XZ, YZ) and each of the four edges of that pair, take the
   cell's liquid quarter at that edge (two cell eighths) and `FindCellsAroundEdge`:
   - **Edge sample** when the cells around the edge all have one size and no cell at the edge holds a solid
     surface that is not axis-aligned. Built once, by the touching cell with the lowest `CellId` (FIXED: was
     the first present cell in `(z, y, x)` order). Weight `ComputeShearWeight` over the four liquid quarters.
     Zero weight: no sample. Otherwise check the wet quarters connect through open faces, then for each of the
     two derivatives take the nominal face pair with coefficient `-/+ 0.5 / cell_width`. A missing face is
     replaced, in this order: (a) a solid between: the solid velocity at the solid surface, at its true
     distance (half a cell width for a domain side), added to `rate_from_solids`; (b) liquid surface beyond:
     the nearest face pair that shares an open connection with the edge, smallest squared offset first, ties
     by lower `FacePlaceIndex`. FIXED: (a) is the test-only half-cell row made the production rule, so viscous
     liquid always sticks to solids, on every path and with no setting (decision 3).
   - **Fitted sample** otherwise (a size change at the edge, or an inclined solid). The cell's liquid quarter
     joins that cell's fitted sample for the axis pair; weight `4 x viscosity x (sum of joined quarters)`.
     Positions are `liquid_covered_open_centre` of faces from `FindNearbyFaces` and the cell's
     `liquid_covered_solid_centre` (known velocity). `ComputeFittedChange` gives both derivatives; the sample
     is half their sum. Known velocities go to `rate_from_solids`. NEW: the split by edge. On `main` a
     mixed-size grid used fitted samples for every cell and a same-size grid used edge samples for every
     edge; each rule is kept where it was proven and the liquid quarters still tile every cell exactly once.
4. Order (A15, both modes). Samples are ordered by the cell's block index, then ascending `CellId`, then kind,
   then axis pair (XY, XZ, YZ), then quarter (0..3), and kept grouped by block (begin offset per block); built
   as count per cell, exclusive prefix sum, write. Coefficients inside a sample by `FacePlaceIndex`. A second view lists, for each face, its `(sample index, coefficient)` pairs by
   sample index (`BuildGroupsByDestination`). Each stored coefficient is `sqrt(dt x sample_weight) x coefficient`,
   so the operator and its diagonal use the same numbers and the operator is symmetric by construction.
5. Check. One finite check over weights, coefficients and `rate_from_solids`. Faces with `face_inertia == 0`
   are given, not solved: their velocity times their coefficient moves into `rate_from_solids`, and their
   stored coefficient becomes zero.

**Solve (steps 6 to 12).**

6. Diagonals. `velocity_diagonal = face_inertia + sum of squared stored coefficients`. The pressure diagonal is
   `pressure::ComputeOwnWeight` with `connection_scale = face_inertia / velocity_diagonal`: thick liquid moves
   less for the same pressure difference (`mac_viscous_pressure_3d.cc:208-230`).
7. Base momentum per solved face: `face_inertia x velocity_before_pressure`, plus the given part of the
   pressure push (`pressure::ComputePressurePush` at zero pressure with `GivenTerms::kIncluded`: reference
   pressures at air ends, surface tension, jumps), `- sum over its samples of stored coefficient x
   sqrt(dt x W) x rate_from_solids`.
8. Start pressure from `pressure::ComputeStartPressure`. CHANGED: `main` first solved once at zero pressure to
   measure the start imbalance for its start-total limit; with that limit dropped (step 10) the extra solve is
   not made.
9. Velocity solve for a trial pressure (inner loop, conjugate gradients with diagonal scaling):
   momentum = base (left out for a search direction) plus `pressure::ComputePressurePush` with
   `GivenTerms::kLeftOut`. One operator use is two passes:
   sample pass `rate[s] = sum over its coefficients, in face order`; face pass `out[f] = face_inertia x u[f] +
   sum over its samples, in sample order, of coefficient x rate[s]`. Every dot product is one partial per cell
   block then `math::TreeSum` over the blocks in block order (A15, both modes). Stop when the largest error is at most
   `velocity_tolerance x largest momentum entry`; then recompute the true error from the operator and accept
   only if it also passes, else continue with a fresh direction. Fails: non-positive curvature, non-finite
   value, iterations used up.
10. Measure `pressure::ComputeVolumeError(..., GivenTerms::kIncluded)`. Accept by the pressure module's rule
    (`pressure::CheckFlowImbalance` with `ViscositySettings::pressure_acceptance`), the same rule for both
    solves: one limit, `max_volume_error_rate` per cell (`2e-5 1/s` over its `filled_volume`), plus
    `jump_error_fraction` where a jump is present. `main`'s second limit (`2e-6 m3/s + 2e-6 x` the start
    total) is dropped (`owner-questions.md` Q27, recommended option).
11. Otherwise correct the pressure: direction = imbalance / pressure diagonal (then
    `pressure::ApplyZeroMeanToSealedBlobs`; pressure pins no cell); solve
    the velocity for that direction with no base momentum; response = flow imbalance of that velocity with
    `GivenTerms::kLeftOut`; make the response orthogonal to the kept responses, in the order they were kept,
    and normalise; `pressure -= (imbalance . response) x direction`. Keep the pair; when
    `kept_response_count` pairs are held, drop them all. The response is measured, not assumed, because the
    inner solve is only as exact as its tolerance. Return to step 9.
12. Publish: apply `pressure::ApplyPressureOffset`, copy the accepted velocity (given faces take their solid
    velocity) and pressure to the outputs, fill `ViscosityResult` (`power_lost_to_viscosity` as one partial per
    cell block in the grid's fixed block order, then `math::TreeSum` over the partials). Nothing is written
    before this step.

Cancel. `*cancel_requested` is read before each inner solve of step 9 (on the GPU, inside the one small record
the CPU receives per outer iteration). When set, the solve stops and returns `kCancelled`; the outputs are
untouched, as for every other failure.

No time step limit comes from viscosity: the solve is implicit for any `dt`.

**Coupled terms (NEW, A24; replaces "Coupled terms are refused (A18)").** The equations may hold
`BlockingMaterialFace` entries and `SealedGasBlob` entries (`plan-decisions.md` "Viscous liquids with blocking
materials and sealed gas (A24)"). Signs as pressure: `s = +1` for the low end, `flow_imbalance` inflow minus outflow.

- *Replaced (A26): the material face eliminated, and the slide at a liquid-material face.* (A24 text kept here:)
  > - *Material face `k`* (ends `a`, `b`; `A_k` blocked area, `m_k` face mass): `m_k v_k = m_k v_k_before + dt A_k (p_a -
  >   p_b)` holds no viscous term, so `v_k` is eliminated exactly, as in `SolvePressure`: each solved end's volume error
  >   gains `-s_e (A_k v_k - coverage_volume_rate)` (`kIncluded`) or `-s_e g_k (p_a - p_b)`, `g_k = dt A_k^2 / m_k`
  >   (m3/(Pa s); `kLeftOut`); an unsolved end is at pressure 0. Step 6's pressure diagonal gains `g_k` per solved end,
  >   not multiplied by `connection_scale` (the material has no viscous resistance; its face mass is its whole inertia).
  >   The inner velocity solve does not see the term.
  > - *At a liquid-material face* (follows solids 5.5a item 3, "tangential motion stays the set's own", A8): the liquid
  >   slides; a blocking material is not a `contact::Solid`, so an edge sample beside it takes rule (b), as at a free
  >   surface. The stretch samples read the material's blocked part through `solid_volume_rate`, at its velocity before
  >   pressure, as for a driven solid; the samples do not change. The viscous normal stress of the material's own
  >   velocity change in the step, `(dt A_k / m_k)(p_a - p_b)`, is left out and is not given to the material: zero at
  >   rest and as viscosity goes to 0 (owner question, section 11).
- *Velocity unknowns (NEW, A26, A27).* `x = (u, w, xi)`: liquid faces; one `w` per `ViscousMaterialFace` (per
  blocking set, the faces with face mass whose centre lies within `kFitReach` cell widths of the centre of a cell
  with `filled_volume > 0`: every face a liquid sample can read; the rest of the material keeps its velocity after
  `CollideSolids` this step, the drag reaching it next step through its internal force); six per free solid (`V`,
  `Omega` about its start origin). Rows: `M x_after = M x_before - dt E^T W (E x + g) + dt D^T (p, q)`, `M =
  diag(face_inertia, face_mass, [m I3, 0; 0, I_start])`, `E = [E_u E_w E_s R]` (`R`: solid velocity at a point,
  `e_a . V + ((y - origin) x e_a) . Omega`), `D = [D_u D_w D_s]` (open areas, blocked areas `A_k` on the
  `BlockingMaterialFace` subset, the `FreeSolidPush` entries). A material's `w_before` already holds its internal
  force (`ApplyForces` ran first), so nothing is counted twice; the material has no sample weight of its own.
  Nothing is eliminated (`w` and `xi` have viscous terms); cost one unknown per band face and six per solid.
- *Stick (NEW, A26, A27).* Rule (a) generalised: when the surface between a partner face `P` and a missing face
  `F` belongs to a blocking material (coverage `distance_to_solid` and corner owner), the derivative is `(u_P -
  w_F) / d` at the true distance `d` (the material's velocity at its surface is its band face unknown at `F`; the
  set with the larger face mass when two hold `F`); for a free solid it is `(u_P - R xi (y)) / d`. Using the
  material face one width away with `1 / cell_width` would undercount the interface shear by a factor 4 (the edge
  sample weighs only the liquid half). Fitted samples: a liquid-covered solid centre of a material takes the nearest
  band face of that axis (first order); of a free solid, `R xi` at the point. Driven and fixed solids stay in
  `rate_from_solids`. An inviscid liquid (`SolvePressure`) keeps sliding and the A18 elimination.
- *Sealed gas blob `j`*: one more unknown `q_j` (Pa above atmosphere) of the outer loop, kept, not eliminated. In
  `SolvePressure` the elimination uses the two-point weights `g_c` of its connections; here a connection's velocity
  answers its end pressures through the viscous velocity solve, so no such weight exists, and the blob is carried
  like an equation cell whose faces are its connections. Its far-end pressure in every push is `q_j + ref_c`; its
  row `E_j = inflow_j(u) + heat_volume_rate + boundary_volume_rate - S_j (q_j - c_j)` (`kLeftOut`: `inflow_j(u) - S_j
  q_j`, open areas only), `S_j = V / (heat_ratio p0 dt)`; diagonal `S_j + sum_c g_c connection_scale_c`; start
  `c_j`. Blob sums by the per-group rule.
- *Solvable.* (A26, A27: `K` is over `x = (u, w, xi)` and `L` is gone, the material faces being in `K`.) The extended operator on `(p, q)` is `dt D K^-1 D^T + C + L` with `K = M + dt E^T W E` symmetric
  positive definite on solved faces, `D` the flow rows of cells and blobs (a blob's row is the transpose of its push
  column), `C = diag(squeeze_allowance, S_j)`, `S_j > 0`, `L = sum_k g_k (e_a - e_b)(e_a - e_b)^T` (or `g_k e e^T`):
  symmetric positive semidefinite, null space only the constants of `kSealed` liquid blobs, which
  `ApplyZeroMeanToSealedBlobs` handles as before (a liquid blob joined to a gas blob is `kCanBeSqueezed`). The outer
  loop needs only a response linear in the direction; the inner solve is untouched.
- *Steps changed.* 6: the four-argument `ComputeOwnWeight`. 7: base momentum by the blob form of
  `ComputePressurePush` at `p = 0`, `q = 0`, `kIncluded`. 8: start `q = c_j`. 9: push of a trial `(p, q)` by the blob
  form with `kLeftOut`. 10: volume errors of cells and blobs, accepted by `CheckFlowImbalance` with the given `q`.
  11: blob entries of the direction `E_j / diagonal_j`; response by the blob form of `ComputeVolumeError`
  (`kLeftOut`); every dot is the cell part (block partials, `math::TreeSum`) plus the blob part (`math::TreeSum`
  over blobs in index order), added once in that order; kept pairs hold blob entries. (A26, A27: step 6 adds each
  material's `face_mass + sum of squared coefficients` and each solid's 6 x 6 block, factored once; the outer diagonal
  uses `material_face_scale = face_mass / velocity_diagonal` and leaves solids out; step 9 adds `ComputeCoupledPush`;
  steps 10 and 11 use `CoupledVelocities`; dots add the solids' part by `math::TreeSum` over solids; each solid's six
  sums by the per-group rule; step 12 writes `material_face_velocity_after_solve` and `free_solid_motion`.) 12: material velocities by
  `ComputeBlockingMaterialVelocities` from the accepted pressure, `q` copied to `sealed_gas_pressure`, both only on
  success. No path of this module reads sealed gas one step late; open gas, slivers, held blobs and cells touching
  gas still reach the liquid through `gas_pressure_above_atmosphere` and `holds_tracked_gas`, as for `SolvePressure`.

**The solve in pieces (NEW, A15).** With `max_loaded_blocks > 0` the buffers keep samples, coefficients and the
face and cell vectors (the 2 x 32 kept directions and responses included) in CPU memory (A24: the blob vectors
stay whole on the processor; material-face lists stream with pressure's rows), and every pass runs as
sweeps over batches of the buffers' own `pressure::SolveBatches` (stacked widths `{kTouchingCells,
kTouchingCells}`: the face pass reads sample rates of samples whose cells touch the range, and those read faces
one more layer out). Samples are built batch by batch for the range cells (count per block, an exclusive prefix
sum over blocks for the whole sample index, write) and kept grouped by block; the by-face view of a loaded piece
is rebuilt at each load (`BuildGroupsByDestination`, piece sample indexes, which follow the whole order because
pieces keep the block order). One inner velocity iteration is 2 sweeps (sample pass and face pass with the block
partials of `d . Kd`; then the velocity, error and scaled-error updates with the partials of the next dot and the
largest error per block); the true-error recheck is one sweep. The outer step: the push of the direction and the
volume error through pressure's pieces (one sweep each), the inner solve, one sweep for the dots of the response
against the kept responses and one for the updates and scaling, about 8 sweeps plus the inner solve; acceptance
by `pressure::CheckFlowImbalance` (its sweep). Dots and largest values by block partials, then `math::TreeSum`.
Outputs only in step 12's write-back sweep; cancel before every batch load and each inner solve; a sweep runs all
batches and keeps the smallest failure. Power lost: block partials of each range, then `math::TreeSum` over every
block. A syrup step in this mode takes hundreds to thousands of sweeps: possible, not practical.

**5.6 The liquid surface and other materials.** Sample weights use liquid volume, so air carries no drag
and the surface is free of viscous force without any extra equation; the pressure equations hold the air
pressure and the surface tension jump there. Solving both together is what gives the complete condition at
the surface (drag plus pressure equals the outside pressure). Between two viscosities the edge weight is the
volume-weighted harmonic mean and the stretch weights differ per cell; no jump term is stored (the unused
`normal_viscous_jump_tail_to_head` field is not ported; `pressure::MaterialBoundary` has no viscous term).
Several liquids and mixtures: step 1. A liquid ingredient is viscous exactly when its viscosity is above 0 (A25);
the step decides once per run. Snow: zero viscosity, its response comes through `ApplyForces`. Fixed
and driven solids: known velocities in step 3 and step 5. Gas: zero viscosity; gas marker sets are not passed
in `cells_from_particles` (A10), so gas never enters a cell's mixture viscosity, and the gas solve runs after
this one as the gas design orders them; a sealed gas blob beside the liquid is an unknown of this solve (A24,
"Coupled terms"). Snow and springy solids: their inertia enters through material faces (A24).

**5.7 Capability exclusions** (three, all geometric; A24 removed the snow row, A27 the free-solid row). `CheckViscosityCanRun` rejects, each with its own error, and each is lifted
only by deleting its line in the same commit that makes its test pass:

| Error | From `main` | Test that lifts it |
|---|---|---|
| (`kFreeSolidNotSupported`, removed by A27: the solid is in the solve) | `liquid_substep_3d.cc:1762` | `ViscosityScene, BoxInLiquidBlock_MomentumChangedOnlyAtAirEnds` and the rigid-body scenes pass |
| `kMovingSolidNotSupported` | `:2907` | driven plate cutting cells: shear between plate and floor, surface force within 5 % |
| `kInclinedSolidNotSupported` | `liquid_substep_3d_cpu.cc:657` | linear shear between two plates tilted 30 degrees |
| `kSizeChangeNotSupported` | `liquid_adaptive_step_3d.cc:261, 461` | planar extension and linear shear with the viscosity change on a size change |

Bulk viscous liquid across a size change, away from the surface, solids and viscosity changes, is allowed
from the first commit. The samples for all four cases are built by the rules above; only the check keeps them
out of a run (A27: three cases). Snow (`main`'s `liquid_substep_3d.cc:1770`) is not excluded (A24): its test, syrup on a snow bed at
rest, is a passing scene test.

## 6. CPU and GPU

Shared (`deformation_rate.h`): the two weight formulas, mixture viscosity, the fit, the per-sample builder
for one cell (`BuildSamplesOfCell`, private, C++20 subset). Loops and storage are not shared.

Data in `ViscosityBuffers`, same arrays on both processors, all owned by the buffers and reused every step:

| Array | Element | Count |
|---|---|---|
| `cell_viscosity` | `double` | cells |
| `samples` | `DeformationRateSample` (40 bytes) | samples |
| `coefficients` (by sample), `sample_uses` (by face) | `{int32, double}` (16 bytes) | coefficients, each view |
| `first_use_of_face` | `int32` | faces + 1 |
| `sample_rate` | `double` | samples |
| `velocity_diagonal`, `base_momentum`, `momentum`, trial velocity, error, scaled error, direction, operator output | `double` | faces x 8 |
| pressure trial, imbalance, pressure diagonal, kept directions and responses | `double` | cells x (3 + 2 x 32) |
| the same per sealed gas blob (NEW, A24) | `double` | `max_sealed_gas_blobs` x 70 |
| material face vectors (NEW, A26) | `double` | `max_material_faces` x 12, plus a 16-byte entry |
| free solid vectors, 6 x 6 block and factor (NEW, A27) | `double` | `max_free_solids` x (12 x 6 + 2 x 36) |
| sample solid coefficients (NEW, A27) | solid index + 6 `double` | samples |

Sizing for the syrup scene (decision 6): 0.5 million cells, 1.5 million faces, about 3 million samples and 13
million coefficients give roughly 0.9 GB. `ComputeViscosityBufferBytes` returns the exact figure.

CPU: build loops over cells of the block range in (block index, `CellId`) order (A15). The sample pass and the face pass each
write one destination per iteration, so they may be split across threads by destination; every sum has one
thread and a fixed order (D18).

GPU: build is one thread per cell (count, then an integer prefix sum, then write), then `BuildGroupsByDestination`
for the by-face view; integer keys only, so the layout is identical on every run. Operator: one thread per
sample, then one thread per face, both gathering, so there are no floating-point atomics. Vector updates are
one thread per element. Dot products are block partials then `math::TreeSum` (A15); largest values use
`ComputeLargestMagnitude`. The whole of steps 6 to 12 stays
on the GPU; per outer iteration the CPU receives one small record (two imbalance norms, inner iteration
count, error code) to decide whether to continue, and reads `*cancel_requested` before launching the next inner
solve. On failure the reported cell is the lowest `CellId` whose
flag is set. Nothing is allocated after `BuildViscosityBuffers`.

Block ranges: build and both operator passes work on `block_range`. Border cells are one layer of cells at
the biggest cell size present in the range; that covers the cells around every edge and the `kFitReach`
search. Border face velocities are read, never written. The outer and inner loops are global:
`SolveViscosityAndPressure` runs on the whole grid (`block_range` = all blocks) on the run's processor, in one
piece or, in the batched layout, as sweeps of the same arithmetic over the buffers' own batches (A15, section 5).
A run is all CPU or all GPU (A17): the CPU fallback inside a GPU run is removed; the only transfers are the
planned, counted slice copies of `pressure::SolveBatches` and the small per-outer-iteration record.

Rows of the batched layout (A15): see `BuildViscosityMemoryEstimate`. About 27 kB per full block stays on the GPU for
a viscous liquid scene, against 190 kB when the solves are whole.

Tolerance, CPU against GPU: sample coefficients and weights `1e-15 + 1e-13 x |value|`; one operator use the
same; solved velocity `1e-9 m/s + 1e-7 x largest speed`; pressure `1e-6 Pa + 1e-6 x largest pressure`.
The last two are solve tolerances, not rounding, and are calibrated by the owner's run.

Tolerance, batched against one piece (A15, same processor): the sample build, one operator use and power lost
are local arithmetic with the shared sum rules: identical bytes; the solved velocity and pressure within the
CPU-against-GPU bounds above. Batched against batched with another cut, and a repeat: identical bytes.

## 7. Tests

Analytic, on the CPU, each on a same-size grid and on a grid with one size change through the bulk (Z up;
shear cases use velocity along X varying with Z):

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| Rigid translation | uniform velocity | every sample rate 0; power lost 0 | `1e-13 x speed / cell_width` |
| Rigid rotation | about Z and about a tilted axis, rate 1 rad/s | every sample rate 0; power lost 0; `SolveViscosityAndPressure` with the rotation given on the boundary leaves it unchanged | rate `1e-12`; velocity `1e-9` relative |
| Linear shear | `u_x = shear x z`, fixed floor, driven top side | every XZ sample equals `shear / 2`; power lost `= viscosity x shear^2 x volume`; velocity unchanged | `1e-10` relative |
| Linear shear, two viscosities | ratios 1, 10, 1000; 16 and 32 cells; steps 0.005 and 0.0025 s | surface force per area `2 / (1 + 1/ratio)` | 5 % of the scale (as `main`) |
| Planar extension, two viscosities | same grid of cases | force per area normal to the change continuous | 5 % mean, 10 % largest; sideways 1 % (as `main`) |
| Shear wave decay | `u_x = sin(k z)`, one step | amplitude times `1 / (1 + dt nu k_h^2)`, `k_h = (2 / h) sin(k h / 2)` | `1e-9` relative |
| Stretch samples match pressure | random velocity, partly filled cells | `open_volume x sum` equals `ComputeFlowImbalance` | `1e-12` relative |
| Symmetric and never negative | random `x`, `y` | `x.(K y) = y.(K x)`; `x.(K x) >= 0` | `1e-12` relative |
| Energy balance | driven side, one step | change in kinetic energy `= dt x solid work - dt x power lost - half |change|^2 in face mass` | `1e-10` relative |
| Free surface, stretched slab | `u_x = a x`, `u_z = -a z`, surface on top through partly filled cells | pressure in surface cells `= outside - 2 viscosity a` | 5 % |
| Syrup at rest | column under gravity, free surface | speeds stay below `1e-9 m/s` | absolute |
| Nearly inviscid | viscosity `1e-12 Pa s` | same velocity and pressure as `SolvePressure` | the pressure stop rule |
| `ComputeViscousEffectPerStep, MatchesDefinition` (A25) | (1e-3, 0.01, 1000, 0.01); (10, 0.01, 1400, 0.01) | 1e-4; 0.714285714... | `1e-15` relative |
| `kSmallestVisibleViscousEffectPerStep, IsTenthOfAPercentPerStep` (A25) | `N` and `1.001 N` | `4 N / (1 + 4 N)` below and at least `1e-3` | exact |
| Each error | one case per `ViscosityError` value, including null `cell_eighths` and a `block_range` short of the whole grid (`kBadInput`) | that error, first failing id, outputs untouched | exact |
| Each exclusion | one scene summary per line of 5.7 (three) | rejected with its error | exact |
| Cancel | flag set before the first inner solve; flag set during the third outer iteration | `kCancelled`; both outputs byte-identical to before the call | exact |
| Stable codes | every `ViscosityError` value | `ToName` is the enumerator's name without `k` in lower snake case; all names distinct | exact |
| Material sets | syrup and water (two liquid materials) in the one liquid material set; then a gas marker set beside it, not passed | cell viscosity is the volume-weighted mixture of the two in shared cells; adding the gas set changes no byte | `1e-15` relative; exact |
| Sticks to solids | linear shear on a same-size grid between a fixed floor and a driven top solid (not a domain side) | liquid at the solid moves with it; same profile as on a mixed-size grid | `1e-10` relative |
| `ViscosityScene, SamplesPerRange_SameBytesAsWholeGrid` (A15; replaces the retired "One piece and batched, byte-identical" row) | two-viscosity shear, n 32, ratio 10: `BuildViscositySamples` per range of 4 blocks (`grid::BuildRanges`), each into its own buffers | their samples concatenated in range order equal the whole-grid samples | bytes |
| `BuildViscositySamples, SamplesOrderedByBlockThenCellIdThenKindThenAxes` (A15) | any scene | order of step 4 | exact |

Coupled terms (NEW, A24; plan Tasks 15a, 15b, 19a), on the CPU:

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `SolveViscosityAndPressure, CouplingAbsent_SameBytesAsBefore` | earlier cases, empty spans | identical bytes | exact |
| `SolveViscosityAndPressure, NearlyInviscidCoupled_EqualsSolvePressure` | `1e-12 Pa s`; pressure plan's snow slab (one step from `x* + 0.1 w`), gas column (one step), heated pocket | equal to `SolvePressure` with the same coupling | the pressure stop rule; material velocity with the velocity bound, `sealed_gas_pressure` with the pressure bound |
| `SolveViscosityAndPressure, ResponseIsLinearWithCouplingTerms` | slab (1 Pa s) and column (0.1 Pa s), carried 50 Pa, blob carried 3 Pa, heat on | response linear in the direction, blob entries included | `1e-10` relative |
| `SolveViscosityAndPressure, MaterialVelocityAndBlobRowFromAcceptedPressure`; `CouplingSpansWrongSize_IsBadInput`; `CancelWithCoupling_AllOutputsUntouched` | | spans equal `ComputeBlockingMaterialVelocities` of the accepted pressure, blob row accepted; `kBadInput`; four outputs unchanged | bytes; exact |
| `ViscosityScene, ViscousLiquidOnSpringySlab_InterfaceHoldsAndCarriesTheWeight` (A26: the slab declared springy) | pressure plan's slab, liquid 10 Pa s, 1000 kg/m3, 500 steps at rest | `|x_n - x*| <= 1.3e-7 m`; interface 1962 Pa; push per area `-p_1` (the hydrostatic state has no velocity gradient, so it solves the joint system) | as pressure: twice the pressure bound; `1e-9` relative |
| `ViscosityScene, LightSlabInViscousLiquid_FloatsAtDisplacedDepth` | floating slab, liquid 1 Pa s | `z_b* = 0.1375`, `h_r - z_c = 0.2 t` | `1e-6 w` each step |
| `ViscosityScene, LightSlabInViscousLiquid_DecaysAtLeastAsFastAsInviscid` | start `z_b* - 0.2 w`, 500 steps, viscous and inviscid runs | per window of 100 steps, amplitude `<=` the inviscid one `+ 1e-6 w`; last 100 `<= 1.02 x 0.2 w` | as stated |
| `ViscosityScene, SealedGasAboveViscousColumn_PeriodAndDamping` | gas column, water at `mu = 0.1 Pa s`: plug by symmetry, drag `c = 128 mu w = 0.32 N s/m`; `dt = 0.1 / omega`; backward Euler factor `1 / sqrt(1 + a + b)`, `a = c dt / m = 9.3072e-4`, `b = 0.01`, phase `acos((2 + a) / (2 sqrt(1 + a + b)))` | factor 0.994579 (inviscid 0.995037); period 0.091720 s, within 0.1 % of the inviscid 0.091677 s; amplitude never above the previous period's nor the inviscid run's | `1e-4` relative; 0.05 % |
| `ViscosityScene, SealedGasAboveViscousColumn_StableWhereLagGrows` | same, `dt = 4 / omega`, 79 steps | factor `1 / sqrt(17 + a) = 0.242270` (`a = 0.037229`); never grows | `1e-4` |
| `ViscosityScene, SealedGasHeatedAboveSyrup_RisesInTheSameSolve` | heated pocket, syrup 10 Pa s 1400 kg/m3, `dt` 0.03 s | `sealed_gas_pressure` 6.5625 Pa; liquid `6.5625 + 1400 g (0.2 - z)`; velocity 0 | `1e-9` relative; pressure bound; velocity bound |
| `ViscosityScene, SyrupOnSnowBed_StaysAtRest` (was the snow exclusion's lifting test) | syrup column on a snow bed of material faces | every liquid and material speed `<= 1e-9 m/s`; the bed carries the weight | absolute; pressure bound |
| `SolveViscosityAndPressureInBatches, ViscousLiquidOnSpringySlab_WithinToleranceOfOnePiece`, `SealedGasAcrossBatches_WithinToleranceOfOnePiece`, `CoupledThreeCuts_IdenticalBytes` | 8 block layers | batched against one piece | section 6 bounds; bytes |

Sticking and free solids (NEW, A26, A27; plan Tasks 15c, 15d). Snow appears only in `SyrupOnSnowBed_StaysAtRest`
(owner note); the analytic scenes use springy solids and rigid bodies.

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `BuildViscositySamples, RigidTranslationWithSlabAndBox_EveryRateZero`; `RigidRotationWithBox_EveryRateZero` | liquid with a springy slab and a free box moving (or turning) together | every rate 0 | `1e-13 x speed / w`; `1e-12` |
| `BuildViscositySamples, SlabInterface_UsesTrueDistance` | slab top at `z = w` | `+-0.5 / (w/2)` on the liquid face and the slab face | exact |
| `ComputeMomentumFromVelocity, SymmetricWithMaterialAndSolids`; `SolveViscosityAndPressure, ResponseIsLinearWithMaterialAndSolids` | random `x = (u, w, xi)` | symmetric, never negative; linear response | `1e-12`; `1e-10` |
| `SolveViscosityAndPressure, NearlyInviscidWithSlabAndBox_EqualsSolvePressure` | `1e-12 Pa s`, springy slab and free box | equal to `SolvePressure` (elimination and rank-one term) | the pressure stop rule |
| `ViscosityScene, SpringySlabDraggedByShear_ReachesLiquidSpeed` | one-cell springy slab, `sigma` 50 kg/m2, liquid `H = 2 w = 0.005 m`, `mu` 1, plate at `U = 1e-3 m/s`, `dt` 0.01 | `sigma_eff = 51.5625`, `T = sigma_eff H / mu = 0.25781 s`; `V(T) = 0.632 U`; per-step factor of `U - V` 0.96266; a 1D implicit reference per step | 3 %; 0.5 %; 2 % |
| `ViscosityScene, ViscousBlockOnTiltedSlab_DoesNotSlide`; `InviscidBlockOnTiltedSlab_Slides` | gravity at 30 degrees, springy slab held by friction 0.7, block 0.04 x 0.04 x 0.02 m; `mu` 1000, `rho` 1400 / viscosity 0 | speeds `<= 1.511e-3 m/s` (steady film of height 0.02 m, 10 % margin); inviscid centre velocity `g sin 30 t = 0.4905 m/s` at 0.1 s | bound; 1 % |
| `ViscosityScene, CubeSettlingInSyrup_StokesLinearity`; `NearWallCorrectedStokes` | cube 0.02 m, 2800 in 1400, `mu` 10, box 0.12 x 0.12 x 0.2 m, 8 cells across | speed ratio 2 for `mu` 10/20 and for density excess doubled; terminal in `[0.7, 1.2] x 0.02405 m/s` (Stokes with shape factor 1.08 and Faxen's correction 0.5529; a bound) | 2 %, 3 %; bound |
| `ViscosityScene, NeutralCubeInShear_TurnsAtHalfTheShearRate` | cube 0.02 m neutrally buoyant, shear rate 0.2 1/s | turn rate 0.1 rad/s (exact for cubic symmetry in Stokes flow) | 5 % |
| `ViscosityScene, FloatingBoxInSyrup_SameRestAsInviscid` | box 500 kg/m3, 0.04 m, liquid 1000 kg/m3 at 10 Pa s, at its 0.02 m draft | as the inviscid run | `1e-6 w`; velocity bound |
| `ViscosityScene, BoxInLiquidBlock_MomentumChangedOnlyAtAirEnds` | no gravity; box (then springy cube) moving in a liquid block | momentum change of liquid + body = air-end impulse; angular likewise | `1e-10` relative |
| `SolveViscosityAndPressureInBatches, SpringySlabDragged_WithinToleranceOfOnePiece`; `CubeAcrossBatches_WithinToleranceOfOnePiece` | 8 layers; cube across three batches | within the section 6 bounds | as stated |

Batched layout (NEW, A15), on the CPU with the loaded piece 3 and 5 (solve: 3, 5 and 7) times
`CountBlocksInLargestLayer`:

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `ComputeViscosityBufferBytes, BatchedLayout_RowsSplitByWhere` | 64 blocks, loaded 16 | `kCpu` and `kRunOn` rows as `BuildViscosityMemoryEstimate` states | exact |
| `BuildViscositySamples, InBatches_SameSamplesAsOnePiece` | two-viscosity shear n 32 ratio 10 (8 block layers); floating cube on a grid lengthened to 8 layers | samples, coefficients, order | bytes |
| `ComputeMomentumFromVelocity, InBatches_SameAsOnePiece`; `ComputePowerLostToViscosity, InBatches_SameAsOnePiece` | random velocity, same scenes | same values | bytes |
| `BuildViscositySamples, InBatchesTooManySamples_IsBuffersTooSmall` | `max_sample_count` 64 | `kBuffersTooSmall` | exact |
| `SolveViscosityAndPressureInBatches, LinearShearTwoViscosities_WithinToleranceOfOnePiece` | n 32, ratio 10 | velocity and pressure within the section 6 bounds of one piece | `1e-9 m/s + 1e-7 x` speed; `1e-6 Pa + 1e-6 x` pressure |
| `SolveViscosityAndPressureInBatches, SyrupAtRest_WithinTolerance` | syrup column on 8 layers, `max_volume_error_rate 1e-9`, `velocity_tolerance 1e-13` | within tolerance; every speed `<= 1e-9 m/s` | as above; absolute |
| `SolveViscosityAndPressureInBatches, FreeSurfaceStretchedSlab_WithinTolerance` | the slab on 8 layers | within tolerance | as above |
| `SolveViscosityAndPressureInBatches, ThreeCuts_IdenticalBytes` | three loaded sizes | identical bytes | exact |
| `SolveViscosityAndPressureInBatches, CancelBetweenBatches_OutputsUntouched`; `FailureInOneBatch_OutputsUntouched` | flag at the 5th load; `1e308 Pa s` in one cell of the last batch | `kCancelled`; `kNotFinite` with that `CellId`, as one piece; outputs untouched | exact |
| `SolveViscosityAndPressureInBatches, SweepsPerInnerIteration` | copy report | 2 sweeps per inner iteration plus the recheck; prints sweeps and bytes per outer iteration | exact |
| `BuildViscosityBuffers, BatchedLayoutOnGpuWithoutAGpu_IsRunOnCallFailed` (`no_gpu`) | GPU buffers in the container | `kRunOnCallFailed` | exact |

The lifting tests of 5.7 are written with the module (A27: three; the snow and free-solid cases pass and are listed
above), none committed disabled unless a lifting task falls back, each with a comment naming
the exclusion it lifts. They are the "visible excluded capability" the programme asks for.

Reference and proof pieces: none. The Python 2D reference repeats what the analytic rows above check in 3D
with exact expected values. The Lean energy identity is the "Energy balance" row; the other theorems concern
the explicit branch, which is not ported.

Comparison with `main` (supporting evidence only): run `ExpectPlanarExtensionTraction` and
`ExpectMovingWallCouetteTraction` on `main`, rotate with the Foundation quarter turn (old Y becomes new Z),
match faces by position, and compare velocity and pressure within `1e-6` relative. Valid on the same-size
fixtures only, where the method is unchanged apart from the solid rule; the mixed-size path differs by design.

GPU (label `gpu`, listed for the owner's run): every row above within the section 6 tolerance; repeat-run
test demanding identical bytes for samples, velocity and pressure over two runs of the two-viscosity shear
case; a test that the by-face view lists the same pairs as the by-sample view; in batches (A15): the batched build
identical to one piece on the GPU, the batched solve within tolerance of one piece on the GPU, twice identical,
and the syrup sizing with `max_loaded_blocks` a quarter of its blocks converging (prints outer and inner
iterations, sweeps, seconds and bytes per outer iteration). In the container without a GPU
(label `no_gpu`): everything in `deformation_rate.h`, the full CPU path, `ToMessage` and `ToName`.

## 8. Method paper section

Labels (A24 adds `<sec-viscosity-coupled>`): `<sec-viscosity>`, `<eq:deformation-rate>`, `<eq:viscosity-equations>`, `<sec-viscosity-samples>`,
`<sec-viscosity-solids>`, `<sec-viscosity-surface>`, `<sec-viscosity-solve>`, `<sec-viscosity-exclusions>`,
`<fig-viscosity-samples>`, `<fig-viscosity-size-change>`, `<fig-viscosity-two-passes>`,
`<proc-build-viscosity-samples>`, `<proc-solve-viscosity-and-pressure>`.

Outline: 1 why drag needs the whole deformation rate and not three separate smoothings (rotation must lose
nothing); 2 the equations and their units; 3 where the samples sit and what they weigh; 4 solids; 5 the
liquid surface and changes of viscosity; 6 the two loops; 6a (A24) blocking materials and sealed gas in the joint
solve; 7 what is excluded and the test that lifts each.

Figures: `<fig-viscosity-samples>` one cell in the XZ plane with its stretch samples at the centre, edge
samples at the corners, the four faces each reads, and the liquid quarter each weighs.
`<fig-viscosity-size-change>` a big cell beside two small cells: which quarters keep edge samples, which join
a fitted sample, and the four face positions the fit picks. `<fig-viscosity-two-passes>` the sample pass and
the face pass as two gathers, with the fixed order marked.

Pseudocode: `<proc-build-viscosity-samples>` (steps 1 to 5) and `<proc-solve-viscosity-and-pressure>` (steps 6 to 12),
numbered as in section 5 and in the code. After it, one paragraph on the batched form (A15: working buffers in CPU
memory, two sweeps per inner iteration, the outer step's sweeps, possible and slow for a syrup scene), referring
to pressure's `<sec-solve-in-batches>`.

Sources, by number in the old reference list: [8] Batty and Bridson 2008 (volume-weighted samples, free
surface); [9] Larionov, Batty, Bridson 2017 (pressure and viscosity together); [21] Batty, Bertails, Bridson
2007 (open areas shared with pressure).

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| Viscosity stage; `ApplyViscosity` (this design's draft) | `SolveViscosityAndPressure`, function and step stage | owner A6 |
| `MacStrainRow3D`, strain row | `DeformationRateSample` | `deformation_rate`, `stretch`, `shear` owner A4; the type name decision 2 |
| normal strain, shear strain | `kStretch`, `kEdgeShear`, `kFittedShear` | NEW |
| `MacStrainTerm3D`, `face_slot`, `coefficient` | `FaceCoefficient`, `FacePlaceIndex`, `coefficient` | NEW; `index` DECIDED, O8 comment kept |
| `quadrature_weight` | `sample_weight` | DECIDED |
| `prescribed_strain` | `rate_from_solids` | NEW; `prescribed` banned |
| `dynamic_viscosity` | `viscosity` (Pa s; comment: papers, dynamic viscosity) | NEW |
| `HarmonicEdgeViscosity`, `CutShearWeight` | `ComputeShearWeight` | NEW |
| `liquid_octant_volumes` | `filled_volume_in_cell_eighth` | DECIDED word |
| `wet_volume`, `fluid_volume` | `filled_volume`; `open_volume` for the part not blocked by solid | NEW, VERIFIED by reading `liquid_adaptive_step_3d.cc:305-307` |
| `wet_open_centroid_m`, `wet_solid_centroid_m`, `world_anchor` | `liquid_covered_open_centre`, `liquid_covered_solid_centre`, `position` | DECIDED word `liquid_covered` |
| `wall_velocity`, `wall_normal_velocity`, `prescribed_wall_flux` | `SolidVelocitySample`, `solid_normal_velocity`, `solid_volume_rate` | DECIDED |
| `face_inertias` | `face_inertia` | pressure's (`ComputeFaceInertia`); this design first wrote `face_mass` |
| `FitFaceGradient`, `FaceGradient` | `ComputeFittedChange`, `FittedChange` | NEW; echoes `fitted_velocity_change` |
| `max_residual`, `max_momentum_residual` | `largest_momentum_error` | NEW; `residual` banned |
| `cell_constraint_defects`, `constraint_residual_l2` | `flow_imbalance` (in `pressure::FlowImbalanceReport`) | DECIDED |
| `compliance`, gauge cells | `squeeze_allowance`; sealed blobs and `pressure_offset` (pressure pins no cell) | DECIDED |
| `predicted_normal_velocity` | `velocity_before_pressure` | DECIDED; VERIFIED: it is after forces, before the solve (`liquid_substep_3d_cpu.cc:800`) |
| `previous_pressure` | `carried_pressure`, `ComputeStartPressure` (pressure's) | VERIFIED: last committed step's pressure, not "before forces" |
| owner (of an edge row, of a face slot) | not used; "the touching cell with the lowest `CellId`" | `owner` is Open (O5); avoided |
| `MacViscousVelocityBlockOptions3D`, `MacPressureSolverOptions3D` | `ViscositySettings` | DECIDED suffix |
| `MacViscousPressureProjection3D` | `ViscosityResult` | DECIDED suffix |
| `ViscosityWorkspace`, `VelocityBlockScratch3D` | `ViscosityBuffers` | DECIDED word |
| (replaced 2026-11-14) `SumScratch`, `OrderScratch`, `sum_scratch`, `order_scratch`, "scratch" | `SumBuffers`, `OrderBuffers`, `sum_buffers`, `order_buffers`, "working buffers" | A31 |
| traction | surface force per area (in test names and comments) | decision 2 |
| status codes with empty messages | `ViscosityError`, `ViscosityFailure`, `ToMessage`, `ToName` | owner A2, A13; `ToName` stable code |
| snow or sand exclusion (`kSnowOrSandNotSupported` in this design's draft) | `kSnowNotSupported`, then removed (A24) | owner A9, A24 |
| (none) | `SolvedCouplingArrays`, `max_sealed_gas_blobs`, `ComputeViscousEffectPerStep`, `kSmallestVisibleViscousEffectPerStep` ("viscous effect per step"; papers: diffusion number) | NEW (A24, A25) |

## 10. Not ported

- The 2D dense path and its certificate: `ViscousStep`, `MacStrainExtents2D`, `AssembleMacStrain2D`,
  `ViscosityMode`, `ViscosityDiagnostics::spectral_bound`, `ViscosityWorkspace` (`viscosity.cc:197-760,
  1651-1719`). No explicit branch: the solve is always implicit.
- Exponent-separated products and every "representable" check (`viscosity.cc:36-55, 150-158, 180-195`;
  `mac_viscous_pressure_3d.cc:104-131`); `CompensatedSum`. One finite check replaces them.
- The one-thread GPU solve and its per-call allocation (`native/cuda/viscosity_velocity_block_3d*`,
  `cuda_viscosity.h`).
- The six coupled-solve entry points and the velocity-solve callbacks (`mac_viscous_pressure_3d.h`),
  `RemapStrain`, `OwnProjection`, the old working-memory arena plumbing.
- `ProjectUniformViscosity` and `PrepareCutViscosityGeometry` (`liquid_substep_3d_cpu.cc:540-700`): their
  volumes come from the geometry snapshot, their normal-row edit is step 2.
- The whole-grid scan and allocation in `FitFaceGradient`; the single-viscosity limit of the mixed-size path.
- `normal_viscous_jump_tail_to_head` (always zero on `main`).
- `main`'s second acceptance limit (`2e-6 m3/s + 2e-6 x` the start total) and the zero-pressure first solve
  that measured that start total (Q27, recommended option).
- A velocity-response hook for `pressure::SolvePressure` (owner A5: pressure has none; this module runs the
  joint solve).
- Viscosity that depends on the deformation rate (the old paper's Herschel-Bulkley line); never implemented.
- `reference/src/fluidity_reference/viscosity.py`, `traction.py`, `reference/tests/test_m4_traction.py`,
  `proofs/Fluidity/Viscosity.lean` (reasons in section 7).
- Tests about aliasing, tiny masses and subnormal ranges in `native/tests/test_viscosity.cc` (lines 815-1180).

## 11. Decisions

1. **Who runs the solve that finds velocity and pressure together?** Stages 9 and 10 are separate names, but
   with viscosity they are one solve. Pressure is built before viscosity, so it cannot call this module.
   - A (recommended): `ApplyViscosity` does both and returns pressure's `FlowImbalanceReport` inside its result; the
     step skips `SolvePressure` when any ingredient has viscosity.
     `auto result = ApplyViscosity(inputs, settings, buffers, velocity_after_pressure, cell_pressure);`
   - B: `ApplyViscosity` only builds samples; `SolvePressure` takes an optional velocity-solve hook that
     pressure declares and viscosity implements. `SolvePressure(inputs, &viscous_velocity_solve)`.
   - C: a third public function in a new place, `step::SolveViscosityAndPressure`.

   Decision: A, with the function renamed `viscosity::SolveViscosityAndPressure` and the step stage named after
   it (owner answers A5 and A6). Pressure has no `VelocityResponse` hook; `pressure::SolvePressure` is for
   scenes with no viscous ingredient and is not called in a scene with one.
2. **Words.** `strain` is Open (O6). Proposed: `deformation_rate` for the tensor, `stretch` and `shear` for
   its entries, `DeformationRateSample` for one measured entry, `rate_from_solids` for the prescribed part,
   and "surface force per area" for traction (after the decided `internal_force_per_area`). Alternative for
   the type: `StretchAndShearSample`. Recommendation: as proposed.

   Decision: as proposed. `deformation_rate`, `stretch` and `shear` are owner answer A4;
   `DeformationRateSample`, `rate_from_solids` and "surface force per area" are the recommended option (owner's
   standing instruction, 2026-11-14).
3. **Does viscous liquid stick to solids?** `main` sticks on mixed-size grids and slides on same-size ones.
   - A (recommended): always sticks when viscosity is positive; no setting.
   - B: per solid, `enum class LiquidGrip { kSticks, kSlides }` in the solid's settings.

   Decision: A, always sticks, no setting (recommended option (owner's standing instruction, 2026-11-14);
   `owner-questions.md` Q28).
4. **Shear samples near a size change.** Proposed: edge samples wherever the cells around an edge share one
   size, fitted cell-centre samples only for the quarters at a size change or an inclined solid. The
   alternative is `main`'s behaviour of fitted samples for every cell of any mixed-size grid, which is simpler
   but throws away the edge rule that passes the two-viscosity tests. Recommendation: the split. Related:
   confirm that bulk viscous liquid may cross a size change from the first commit while the five cases of
   5.7 stay excluded.

   Decision: the split; bulk viscous liquid crosses a size change from the first commit and the five
   exclusions of 5.7 stay until their tests pass (recommended option (owner's standing instruction,
   2026-11-14); Q29).
5. **Divisor of the stretch samples.** `main` divides by wet volume on mixed sizes and by `h` on same sizes.
   Proposed: `open_volume` (not blocked by solid), with the weight on `filled_volume`. This needs the pressure
   snapshot to store both volumes. If pressure keeps only `filled_volume`, surface cells lose the exact
   derivative. Recommendation: both volumes.

   Decision: both volumes; `CellGeometry::open_volume` and `filled_volume` are in pressure 3, and the cell
   eighths and liquid-covered centres come from pressure's `CellEighthGeometry` and `FacePieceCentre`
   (recommended option (owner's standing instruction, 2026-11-14); Q30, conflict C3 option A).
6. **Syrup reference scene.** Proposed: domain box 0.4 x 0.4 x 0.3 m; syrup of 10 Pa s, 1400 kg/m3, surface
   tension 0.08 N/m; a 2 cm stream at 0.3 m/s from 0.25 m onto the floor, 1 litre in total, then 10 s of
   settling; cells 1.25, 2.5 and 5 mm. It needs only the domain floor, so it runs before any exclusion is
   lifted. Adding a tilted plate would make the scene wait for `kInclinedSolidNotSupported`. Recommendation:
   floor only.

   Decision: floor only, as proposed (recommended option (owner's standing instruction, 2026-11-14); Q50).
7. **Fixed-tree reductions and `BuildGroupsByDestination` in `base/`.** They are needed identically by transfer,
   pressure and viscosity. Recommendation: `base/`, introduced by the first module that needs each.

   Decision: `base/`, added to Foundation before contact starts, as `shared-interface.md` section (a) and
   `changes-to-apply.md` list them (recommended option (owner's standing instruction, 2026-11-14); Q13, Q45).
   Replaced in part (A15): the solve's dots no longer use `ComputeTreeDot` over a whole buffer (decision 9).

Decisions taken after this design was first written (recorded in the viscosity and pressure plans and
`plan-decisions.md`):

8. **The joint solve runs in pieces when its buffers do not fit.** `ViscosityBufferSettings::max_loaded_blocks`;
   the batched layout keeps per-place, per-sample and per-coefficient rows in CPU memory and runs every pass in
   sweeps through `pressure::SolveBatches` (stacked widths `{kTouchingCells, kTouchingCells}`); no signature
   changes; batched agrees with one piece within the section 6 tolerances, each mode repeatable. Source: A15,
   with A17's principle. Replaces "the joint solve is never cut into batches", "not batched, runs once on the
   whole grid ... so a one-piece and a batched run give the same bytes" and the CPU fallback inside a GPU run.
9. **Sample order and dot order by block, both modes.** Samples by block, then `CellId`, kind, axis pair,
   quarter; dots and power lost by block partials then `math::TreeSum`. Source: A15. Replaces "samples ordered by
   `CellId`" (former step 4; also the `plan-decisions.md` "Viscosity" entry "ascending `CellId` with
   `BuildStableOrder`") and `ComputeTreeDot` for the dots.
10. **The joint solve refuses equations carrying the coupling terms.** **Replaced by decision 13 (A24).** Pressure's pieces return `kBadInput` for
    equations with blocking-material faces or sealed gas blobs; in a viscous scene sealed gas stays one step late
    (`SealedGasCoupling::kReadOneStepLate`). Source: A18 (pressure plan, `plan-decisions.md` "Coupling inside
    the pressure solve").
11. **Retired test.** "One piece and batched: byte-identical" is replaced by
    `ViscosityScene, SamplesPerRange_SameBytesAsWholeGrid` (a local build) and the batched tests of section 7.
    Source: A15.
12. **Interface details of the plan:** `ViscosityInputs::pressure_buffers` (null is `kBadInput` in the solve);
    accepted means `CheckFlowImbalance` returns a report, `kNotConverged` means "not yet"; a face with
    `face_inertia == 0` is given (`velocity_before_pressure`); `max_cell_count` a multiple of 64 and
    `max_face_count == max_cell_count / 64 * 240`; a grid-version mismatch is `kBadInput`. Three exclusions
    (inclined solid, driven solid cutting cells, size change) are lifted in the plan's own tasks, each with the
    fallback "if the lifting test fails, commit it as known-open and keep the exclusion"; free solid and snow
    stay excluded as known-open tests. Source: plan decision, 2026-11-14. Replaced in part (A24): snow is not
    excluded; only the free solid stays as a known-open test (decision 14).
13. **The joint solve carries blocking-material faces and sealed gas blobs.** Replaced in part by decision 18 (A26: material faces are velocity unknowns, the liquid sticks). Material faces eliminated exactly
    into the volume balance (unscaled `g_k` on the diagonal), each sealed gas blob one more unknown of the outer loop
    (not eliminated), outputs in `SolvedCouplingArrays`; method in section 5 "Coupled terms". Decision: as stated.
    Source: owner answer A24; `plan-decisions.md` "Viscous liquids with blocking materials and sealed gas (A24)"; the
    choice to keep the blob unknown and the viscous condition at a material face are plan decisions closest to the
    pressure coupling (A18).
14. **Exclusions after A24.** Replaced in part by decision 19 (A27: the free solid is lifted). Decision: `kSnowNotSupported` and `has_snow` are removed (snow's inertia now enters the
    solve); `kFreeSolidNotSupported` stays (the free solid's velocity would be an unknown of the viscous velocity
    solve, not covered by A24); no springy-solid exclusion (owner chose option C of reconcile item 4). Source: owner
    answer A24 and the owner's choice of option C.
15. **What counts as viscous.** Decision: a liquid ingredient is viscous exactly when its viscosity is above 0; the
    step chooses stage 11 or 12 once per run; the water preset has viscosity 0. Source: owner answer A25.
16. **The viscous effect per step.** Decision: `ComputeViscousEffectPerStep` and
    `kSmallestVisibleViscousEffectPerStep = 2.5e-4` here; the step's planner computes it per liquid at the smallest
    cell and the largest step, and I/O prints it with a note below the constant. Source: owner answer A25;
    `plan-decisions.md` A25 section (the constant awaits the owner's confirmation).
17. **Copy report reset through the buffers.** Decision: `void ResetSolveCopyReport(ViscosityBuffers&)`
    (`FindSolveBatches` stays const); test `ResetSolveCopyReport, BatchedViscosityBuffers_ReportZeroAfterReset`.
    Source: `plan-decisions.md`, Reconcile pass item 3.
18. **A viscous liquid sticks to blocking materials (A26).** Decision: each blocking set's band faces (within
    `kFitReach` widths of a liquid cell centre) are velocity unknowns with their face mass, the blocked-area pressure
    term and the viscous traction; stick by rule (a) at the true distance with the material's face unknown; nothing
    eliminated; the rest of the material keeps its `CollideSolids` velocity this step. Source: owner answer A26;
    `plan-decisions.md` "A viscous liquid sticks, and free solids in the viscous solve (A26, A27)"; the band, the
    true-distance rule, the larger face mass at a shared face are plan decisions (closest to A18 and rule (a)).
19. **Free solids in the joint solve (A27).** Decision: six unknowns per free solid, its mass and inertia at the
    start pose, the `FreeSolidPush` pressure part, stick through `R xi`, per-group sums, an exact 6 x 6 block in the
    inner preconditioner; no exclusion by material remains (`kFreeSolidNotSupported` removed). Source: owner answer
    A27; `plan-decisions.md` A26/A27 section.
20. **Snow not a validation target.** Decision: snow has no special case; liquid-on-snow is tested only at rest and
    in the step's five-material scene. Source: owner note, 2026-11-14.
21. **Working buffers, one form (A31).** `ViscosityBuffers` holds `SumBuffers sum_buffers` and `OrderBuffers
    order_buffers`; every base sum, order, prefix sum and grouping has one form, which takes them. "Working buffers"
    replaces the old word in prose. Source: owner answer A31 (2026-11-14).
22. **Processors of the arguments (A30).** Arrays and buffers passed together live on one processor: a documented
    precondition checked by a debug assertion. `ToViscosityFailure` maps no processor mismatch. Source: owner
    answer A30 (2026-11-14).
23. **Out of memory (reconcile item 25).** `ViscosityError::kOutOfMemory` (magnitude = bytes asked) is returned
    only by `BuildViscosityBuffers`; it replaces "a failed allocation is `kRunOnCallFailed`". `ToViscosityFailure`
    maps `BufferError::kOutOfMemory` to it. Source: reconcile-todo item 25 (2026-11-14).

**Open** (for the owner; the two A18 open points, sealed gas and springy solids beside a viscous liquid, are
decided by A24 and option C):

- `kSmallestVisibleViscousEffectPerStep = 2.5e-4` (0.1 % per step) is a new constant.
- (Answered by A26 and A27: the slide at a material face and the free-solid exclusion.)
- The material band and "the rest of the material keeps its velocity this step"; the set with the larger face mass
  at a shared face; free solids left out of the outer diagonal; a second free solid in one sample folded at its
  velocity before pressure (decisions 18, 19).
