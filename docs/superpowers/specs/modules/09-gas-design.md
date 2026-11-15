# Gas module design

Status: updated 2026-11-14 with the owner's answers (`owner-answers.md`) and `changes-to-apply.md`; brought in
line the same day with A15, A18, A20 and the gas entries of `plan-decisions.md` as the gas plan
(`plans/2026-11-14-rebuild-09-gas.md`, its "Amended 2026-11-14" tasks) carries them, and with `plan-decisions.md`
"Planned cost shown to the user, and buffers in batched runs" (decisions 14, 15), and with A24 (decision 16:
sealed gas beside a viscous liquid solved in the same step). Build step
7b of the rebuild (fire and smoke), gas and liquid in one scene from the start (A10). Written against
`shared-interface.md`, the updated designs of contact, grid, particles, transfer, pressure, viscosity and
materials, and the gas branch `origin/codex/m7-low-mach-gas` at `a430dab` (read with `git show`, never checked
out).

## 1. Purpose

Hot air is lighter than cold air, so it rises; burning fuel makes air hot and makes it take more room. This
module keeps, in every cell that holds air, how much gas, fuel and smoke there is, burns the fuel, pushes the
hot gas up, tells the pressure solve how much volume each cell must gain because it was heated, and then
moves the gas, fuel and smoke between cells with the solved face velocities. The speed of sound never appears:
the pressure that sets the gas density is one number per connected body of gas, and the pressure the solve
finds is only the small part that steers the flow (papers: low-Mach model). Beside liquid, the liquid surface
is a moving boundary of the gas, and the liquid feels the gas pressure in place of atmosphere: a sealed gas
blob that touches liquid is solved inside the liquid's pressure solve of the same step (A18); other gas is
read as committed at the end of the previous step.

**Owns:** the gas amounts per cell and their try and commit; the gas law, burn and lift formulas; keyed gas
sources and wind; the gas `GeometrySnapshot` and `added_volume_rate` handed to `pressure`; the background
pressure of each gas blob; the flow of amounts between cells; the upkeep rule for gas marker particles; the two
arrays the liquid solve reads (`gas_pressure_above_atmosphere`, `holds_tracked_gas`) and the list of sealed gas
blobs it solves with (`SealedGasForLiquid`, A18); the gas solve's batched path (A15); the full gas snapshot for
the saved state and the gas snapshot the volume output reads; the capability check for gas scenes.

**Does not own:** the pressure equations and their solve (pressure); marker storage, the velocity exchange
per material set and marker motion (particles, transfer, contact); where solids and liquid are (solids,
pressure: this module clips nothing itself); air pockets, which are the air cells that hold no tracked gas
(pressure); bubbles as foam (surface, foam and spray); the scene file, saved state and volume files (I/O and
CLI; surface, foam and spray); the remap of the gas arrays when cells are split or merged (cell resize and
batches); the supersonic extension and flow-map transport of the old paper (out of scope).

## 2. What exists today

`main` has no gas code. The gas method lives on `origin/codex/m7-low-mach-gas` (38 files, 3,604 lines, 18
commits on top of `db4636d`). It is a second runtime beside the liquid one: its own domain, frame
loop, saved state, scene file and CLI route. Y-up. Paths are under `native/` unless stated.

| Piece | Where on the branch | What it does |
|---|---|---|
| Cell state | `include/fluidity/host/gas_transport.h:12-17`, `gas_substep_3d.h:15-21` | `GasScalars { carrier, fuel, soot }` in kg/m3 and `enthalpy` in J/m3; per cell a fixed `volume`, `absolute_pressure` ("identical throughout a sealed component") and `gauge_pressure` (last solve, warm start) |
| One step | `src/gas_substep_3d.cc:22-200` (`AdvanceGasSubstep3D`) | Checks stationary walls and zero exterior pressure (`:59-71`); builds the pressure topology once with zero right-hand side to get connected components (`:75-100`); burn (`:121-127`); heating (`:132-139`); buoyancy and face density (`:141-159`); pressure solve through a callback (`:160`); donor-cell transport with the solved fluxes (`:168-183`) |
| Burn | `gas_substep_3d.cc:121-126`; ignition in `src/gas_frame_3d.cc:131-134` | `burned = -fuel * expm1(-dt * burn_rate)`; soot `+= soot_yield * burned`; heat `+= heat_per_fuel * burned`; `burn_rate` is zero in a cell with `T < ignition_temperature`, `T = enthalpy / carrier / (3.5 * 287.05)` |
| Heating | `src/gas_thermodynamics.cc:25-83` (`AdvanceGasHeating`), heat ratio 1.4 at `:11` | Sealed component: `dp0/dt = 0.4 * mean heat rate`, `p0 += dt * that` (`:59-67`). Per cell `target_divergence = (0.4 / 1.4) * (heat - mean) / p0_old` (`:73-80`), the mean subtracted first so uniform heating gives exactly zero. Open component: mean 0, `p0` = ambient |
| Buoyancy | `gas_substep_3d.cc:141-159` | `velocity += dt * gravity[axis] * (soot / carrier - (T / T_ambient - 1))`, half-and-half between the two cells; face `inverse_density = 1 / mean carrier`: inertia from the grid, never from marker weights |
| Transport | `src/gas_transport.cc:44-121` (`TransportGasScalars`) | All four scalars: `new = (1 - dt * outflow / V) * old + dt * inflow / V + dt * source`; a missing cell upwind supplies ambient values; `outward_rate` per channel. Fails when `dt * outflow > V` (`:100`) or a value goes negative (`:115`) |
| Markers | `src/gas_apic_3d.cc:144-244` (`AdvanceGasApicSubstep3D`), `:44-97` (`AdvectMarker`), `:99-140` (`ExtendBoundaryVelocities`), `:246-318` (`RemapGasApicMarkers3D`) | `GasApicMarker3D { world_position, weight, velocity, velocity_slopes }`. `DepositMac3D`, then a face with no marker weight takes last step's velocity (`:200`); optional uniform acceleration per face; after the solve `GatherMac3D` with the new grid velocity only (no own motion); markers move one axis at a time, at most half a cell, stop at a closed face, are dropped beyond an open one. Remap every step: keep the first marker of each cell, add a unit-weight marker at the centre of each empty cell |
| Domain | `src/gas_uniform_domain_3d.cc:30-145` | A box of at most 4096 same-size cells, one marker of weight 1 per cell, six sides each open (zero exterior pressure, half-width distance) or closed |
| Frame loop | `src/gas_frame_3d.cc:62-190` | 1 to 4096 equal steps per frame, no retry; box sources (`fuel_rate` kg/(m3 s), `heat_rate` W/m3, tested at cell centres) and a uniform "draft" acceleration, both keyframed, linear, sampled at each step's midpoint |
| Saved state, scene, CLI | `src/gas_checkpoint_3d.cc`, `schemas/simulate/gas_scene.fbs`, `gas_checkpoint.fbs`, `cli/gas_execution.cc` (448 lines) | Formats `FLGS` and `FLGC` v1; `simulate`, `check`, `inspect`, `replay`; `inspect` prints `[carrier, temperature, fuel, flame, soot]` per cell (`gas_execution.cc:102-104`) |
| Blender | `blender/fluidity/gas_authoring.py:38-82`, `gas_volume.py:13` | One domain box, one source box with `Fuel Rate` and `Heat Rate` custom properties, one Wind field read as an acceleration; five volume grids `density` (the carrier), `temperature`, `fuel`, `flame` (= `fuel_burn_rate`, kg/(m3 s)), `soot` |
| Reference | `reference/src/fluidity_reference/gas.py:40-189`, `reference/tests/test_gas.py` | 2D: `project_heating`, `transport_densities`, `react_densities`, `residual_gas_state`; five tests |
| Tests, evidence | `tests/test_gas_substep_3d.cc`, `test_gas_thermodynamics.cc`, `test_gas_transport.cc`, `test_gas_checkpoint_3d.cc`; `docs/native/m7-gas.md` | 11 native tests; open heating 7.0494519e-5 m3 measured and analytic; +400 Pa and 0 Pa in two sealed components; "Fire in a draft": 13 frames, frame 12 spans about 293 to 777 K with 46 flame voxels |
| Old paper | `docs/papers/fluidity-method.typ:278-291` (`<sec-gas>`) | The model the branch implements, plus conduction and a moving-volume term in `dp0/dt` that the branch leaves out |

What is wrong (the first six are programme anti-goals or fixed decisions):

1. **A separate route.** Own domain, loop, saved state, scene and CLI (`gas_frame_3d.cc`, `gas_execution.cc`).
   Gas cannot share a scene with liquid, a moving solid or a sloped solid (`gas_substep_3d.cc:63-67` rejects
   any wall velocity).
2. **Uniform cells only, at most 4096** (`gas_uniform_domain_3d.cc:31`, `gas_checkpoint_3d.cc:14`).
3. **CPU between GPU stages.** Only the pressure callback runs on the GPU; transport and the marker exchange
   copy through CPU vectors every step (`docs/native/m7-gas.md`, report numerics 11.1).
4. **No retry.** Equal steps per frame; a step that breaks the outflow rule fails the frame.
5. **Fixed topology assumed.** A sealed component whose cells disagree on `absolute_pressure` is an input
   error (`gas_substep_3d.cc:110-113`): components can never join, split or open during a run.
6. **Abseil, `3D` suffix, `Apic`, `tail`/`head`, `component`** in public names.
7. **Enthalpy is carried twice.** For this gas law `enthalpy = 3.5 * absolute_pressure` in every cell (the
   branch's own test asserts it, `test_gas_substep_3d.cc:40`), yet it is a fourth transported scalar with a
   pressure-work source. The two can drift apart by the solve's leftover error and nothing reports it.
8. **One marker per cell of weight 1**, kept by "first in the list wins". The fitted velocity change then
   rests on one sample per cell, and unit weights have no meaning across cell sizes.
9. **Markers have their own motion and boundary rule** (`AdvectMarker`), a second contact path.

**Y-up.** The branch's floor is the low-Y side (`gas_authoring.py:75`, `open_boundaries = 59`), gravity is a
world vector and the add-on converts with `_solver_vector`. Here lift acts along `kUp` = +Z on Z faces only.
Branch data is rotated by the quarter turn about X of `<fig-axes>` before any comparison.

Not to be confused with gas, on `main`: the air pocket law (`native/src/liquid_cavity.cc:14`, `45-79`,
`104-150`) and pocket labelling (`liquid_air_components.h:13-102`), ported by `pressure`; `gas_fraction` of
foam bubbles (`whitewater_volume_3d.h:24`), a render quantity of the surface, foam and spray module (report
numerics 8.1).

## 3. Public interface

Folder `native/fluidity/gas/`, namespace `fluidity::gas`. Every name is NEW (the naming review has no gas
word) except the verbs, the suffixes `Inputs`, `Settings`, `Result`, and `snapshot`, `buffers`, `blob`,
`added_volume_rate`, `sample_weight`, which are DECIDED. Each header marked (+) has private `_cpu.cc` and
`_gpu.cu` files; (shared) headers are `FLUIDITY_CPU_AND_GPU`, C++20 subset. Every fallible function returns
`std::expected<T, GasFailure>` (A2). Every scene total is one partial per cell block in the grid's fixed block
order, then `math::TreeSum` over the block partials, so it is the same in one piece or in batches. Every
per-blob total is one partial per cell block over the blob's cells of that block in ascending cell place, then
`math::TreeSum` over the blob's block partials in ascending block index (A15, both modes).

```cpp
// gas_error.h (shared)
enum class GasError : std::uint8_t {
  kNone,
  kBadInput,                      // A value is not finite or out of range, or an array has the wrong size.
  kWrongGridVersion,              // An input was built for another grid version or another step try.
  kStepTooLongForGasFlow,         // More gas would leave a cell than it holds. Shorten the step.
  kTemperatureOutOfRange,         // A gas cell left min_temperature..max_temperature. magnitude = K.
  kBackgroundPressureOutOfRange,  // A sealed gas blob left the allowed pressure range. magnitude = Pa.
  kPressureSolveFailed,           // pressure returned a failure; it is in GasFailure::pressure_failure.
  kNotFinite,                     // A computed value is not finite.
  kCancelled,                     // NEW. cancel_requested was set; nothing in the try is valid to read.
  kRunOnCallFailed,               // CheckCanRunOn failed for the chosen processor, or a call to it failed.
  // Capability exclusions and scene rules (5.6). Found by CheckGasCanRun before anything is allocated.
  kMoreThanOneGas,                // The scene has two gas materials.
  kGasLawDisagrees,               // NEW. The gas ingredient's rest density is not ComputeAmbientDensity(law), or
                                  // pressure::AirPocketSettings differs from GasLaw or GasSettings (5.6).
};
struct GasFailure {               // The error value of every std::expected here. Never allocates.
  GasError error = GasError::kNone;
  grid::CellPlaceIndex cell_index{grid::kNoIndex};  // First failing cell in index order, or kNoIndex.
  double magnitude = 0;           // What was measured, in the unit named beside the error value.
  double limit = 0;               // What was allowed, same unit.
  pressure::PressureFailure pressure_failure{};     // Set with kPressureSolveFailed and with a kCancelled that
                                                    // came from pressure::SolvePressure.
};
// gas_error_message.h (CPU only)
std::string_view ToMessage(GasError error);
// NEW. The stable code: the enumerator's name without the k, in lower snake case ("step_too_long_for_gas_flow").
std::string_view ToName(GasError error);
```

```cpp
// gas_law.h (shared): every formula of the model, one place.
struct GasLaw {                           // Chosen once per scene.
  double heat_ratio = 1.4;                // No unit. Textbook: gamma. Must equal pressure::AirPocketSettings'.
  double gas_constant = 287.05;           // J/(kg K). Textbook: specific gas constant of dry air.
  double atmosphere_pressure = 101325.0;  // Pa, absolute. Must equal pressure::AirPocketSettings'.
  double ambient_temperature = 293.15;    // K. The air outside the grid and in a cell with no gas neighbour.
  double gravity = 9.81;                  // m/s2, positive. Gravity is -gravity * kUp.
};
struct BurnSettings {
  double ignition_temperature = 300.0;    // K. Fuel burns only in gas at least this hot.
  double burn_rate = 0.0;                 // 1/s. Share of the fuel that burns per second, once ignited.
  double heat_per_fuel_mass = 0.0;        // J/kg. Textbook: heat of combustion.
  double smoke_per_fuel_mass = 0.0;       // No unit, 0..1. Textbook: soot yield.
};
// Forms fixed (plan-decisions "Gas"): ComputeTemperature = p0 / (gas_constant * gas_density); ComputeGasDensity =
// p0 / (gas_constant * temperature); ComputeAmbientDensity(law) = ComputeGasDensity(law, atmosphere_pressure,
// ambient_temperature).
FLUIDITY_CPU_AND_GPU double ComputeTemperature(const GasLaw& law, double background_pressure,
                                               double gas_density);                        // K
FLUIDITY_CPU_AND_GPU double ComputeGasDensity(const GasLaw& law, double background_pressure,
                                              double temperature);                         // kg/m3
FLUIDITY_CPU_AND_GPU double ComputeAmbientDensity(const GasLaw& law);                      // kg/m3
// m/s2 along kUp. Positive: lighter than ambient air. Textbook: buoyancy.
FLUIDITY_CPU_AND_GPU double ComputeLift(const GasLaw& law, double temperature, double smoke_density,
                                        double gas_density);
// NEW (plan decision). m/s2 along kUp. The same value as ComputeLift(law, ComputeTemperature(law, p0, rho),
// smoke, rho) in exact arithmetic, computed as g * (ComputeGasDensity(law, p0, ambient_temperature) - rho -
// smoke) / rho, so that ambient air (rho == ComputeGasDensity(law, p0, ambient_temperature)) gives exactly
// 0.0. ApplyGasLift uses it.
FLUIDITY_CPU_AND_GPU double ComputeLiftFromDensities(const GasLaw& law, double background_pressure,
                                                     double gas_density, double smoke_density);
// NEW (plan decision). Pa. The background pressure whose gas amount V p^(1/heat_ratio) is the sum of two
// (hand-over, joining).
FLUIDITY_CPU_AND_GPU double ComputeJoinedPressure(const GasLaw& law, double volume_a, double pressure_a,
                                                  double volume_b, double pressure_b, double joined_volume);
struct BurnResult {
  double fuel_burned = 0;  // kg/m3 during the step. Never more than the fuel present.
  double heat = 0;         // J/m3 released during the step.
  double smoke_made = 0;   // kg/m3 during the step.
};
FLUIDITY_CPU_AND_GPU BurnResult ComputeBurn(const BurnSettings& burn, double temperature, double fuel_density,
                                            double step_duration);
// Pa/s for one sealed blob. mean_heat_rate: W/m3, volume-weighted over the blob. volume_rate: m3/s the blob
// gains because its boundary (solid or liquid) moves (negative: squeezed). volume: m3.
FLUIDITY_CPU_AND_GPU double ComputeBackgroundPressureRate(
    const GasLaw& law, double background_pressure, double mean_heat_rate, double volume_rate, double volume);
// m3/s one cell must gain. heat_rate: W/m3 in the cell. For an open blob pass mean_heat_rate = 0,
// blob_volume_rate = 0.
FLUIDITY_CPU_AND_GPU double ComputeCellAddedVolumeRate(
    const GasLaw& law, double background_pressure, double heat_rate, double mean_heat_rate, double gas_volume,
    double blob_volume_rate, double blob_volume);
// m3 Pa^(1/heat_ratio). Adds up when gas blobs join (the air pocket rule of pressure).
FLUIDITY_CPU_AND_GPU double ComputeGasAmount(const GasLaw& law, double volume, double background_pressure);
```

`ComputeSealedGasStepLimit` (pressure's air pocket limit for a sealed blob read one step late) is removed (A24):
no sealed blob that touches liquid is read one step late.

```cpp
// gas_sources.h (CPU only). NEW (from I/O): sources and wind that change in time, held in the ready scene.
// Values between two keys are linear in time; before the first key and after the last the end value holds
// (the branch's rule, gas_frame_3d.cc:128-153). The step samples nothing: SolveGas samples at the middle time
// of its step, step_start_time + step_duration / 2.
struct GasSourceKey {
  double time = 0;                 // s. Strictly ascending along a source.
  double fuel_rate = 0;            // kg/(m3 s) inside the box.
  double heat_rate = 0;            // W/m3 inside the box.
};
struct GasSource {                 // A box, world axes, fixed in place.
  std::uint64_t source_id = 0;     // Stable id from the ready scene. Nonzero, unique.
  Vec3 low_corner{};               // m
  Vec3 high_corner{};              // m
  std::span<const GasSourceKey> keys;  // At least one. The storage belongs to the ready scene.
};
struct GasWindKey {
  double time = 0;                 // s. Strictly ascending.
  Vec3 acceleration{};             // m/s2, world, the same on every gas face piece.
};
struct GasWind { std::span<const GasWindKey> keys; };  // Empty: no wind.
struct GasSourceRates {            // One source's values for one step.
  Vec3 low_corner{}, high_corner{};  // m
  double fuel_rate = 0;            // kg/(m3 s)
  double heat_rate = 0;            // W/m3
};
// kBadInput: no key, times not strictly ascending, a value not finite, a negative rate, low_corner not below
// high_corner on every axis.
std::expected<void, GasFailure> CheckGasSources(std::span<const GasSource> sources, const GasWind& wind);
GasSourceRates SampleGasSource(const GasSource& source, double time);
Vec3 SampleGasWind(const GasWind& wind, double time);  // Zero when wind.keys is empty.
// Wind keys (A20): the add-on derives them from Blender's Wind force fields and I/O carries them as gas_wind;
// gas reads them as written and converts nothing.

// gas_source_math.h (shared). NEW (plan decision).
// No unit, 0..1: share of the cell's eight cell-eighth centres inside [low_corner, high_corner) on every axis
// (low sides included, high sides not). cell_low_corner: m, world; cell_width: m.
FLUIDITY_CPU_AND_GPU double ComputeSourceCoverage(Vec3 cell_low_corner, double cell_width, Vec3 low_corner,
                                                  Vec3 high_corner);
```

```cpp
// gas_amounts.h (CPU only)
// What every cell holds, at one committed time. Owns its arrays on one processor. Move-only. A cell holds
// gas when gas_volume > 0. Densities are per m3 of gas, not per m3 of cell.
class GasAmounts {
 public:
  double time() const;                      // s
  std::uint64_t state_version() const;      // DECIDED word. Grows by one at each CommitTry.
  grid::GridVersion grid_version() const;
  RunOn run_on() const;
  const grid::CellValues<double>& gas_volume() const;           // m3 of the cell that holds gas.
  const grid::CellValues<double>& gas_density() const;          // kg/m3. Air; the conserved amount.
  const grid::CellValues<double>& fuel_density() const;         // kg/m3
  const grid::CellValues<double>& smoke_density() const;        // kg/m3
  const grid::CellValues<double>& background_pressure() const;  // Pa, absolute. Equal inside one gas blob.
  const grid::CellValues<double>& flame() const;     // kg/(m3 s). Fuel burned per second in the last step.
  // Pa above background, from the gas solve. The starting guess of the next gas solve, and with
  // background_pressure the pressure the liquid reads next step (ComputeGasSeenByLiquid).
  const grid::CellValues<double>& pressure() const;
};
class GasAmountsTry;  // The values of one try, same accessors. Dropping it changes nothing.
std::expected<GasAmounts, GasFailure> BuildGasAmounts(const grid::StaggeredGrid& grid, const GasLaw& law,
                                                      RunOn run_on);  // Every cell: no gas yet.
// Stage ChooseActiveCells, after grid::BuildGrid of the try grid. Copies the committed values to the cells of
// `grid` by stable grid::CellId (the grid version may differ). A cell with no committed value starts with
// gas_volume 0. Amounts of cells that the try grid no longer has are summed into
// GasAmountChanges::gas_mass_left_with_removed_cells of this try. After a resize the committed amounts are
// already on the resized grid (BuildResizeTry below), so a resized cell is never counted here.
std::expected<GasAmountsTry, GasFailure> BuildTry(GasAmounts& amounts, const grid::StaggeredGrid& grid);
std::expected<void, GasFailure> CommitTry(GasAmounts& amounts, GasAmountsTry&& amounts_try,
                                          double committed_time);
void RejectTry(GasAmounts& amounts, GasAmountsTry&& amounts_try);

// NEW (from cell resize and batches). For cell_resize::ResizeCells only, inside its transaction. A try bound to
// resized_grid whose seven arrays are writable and hold nothing yet: the resize remap writes every cell of all
// seven. Nothing is counted as removed. Committed with CommitTry(amounts, try, amounts.time()).
struct GasArraysToRemap {          // Bound to the try's grid version. Units as in GasAmounts.
  grid::CellValues<double>* gas_volume;
  grid::CellValues<double>* gas_density;
  grid::CellValues<double>* fuel_density;
  grid::CellValues<double>* smoke_density;
  grid::CellValues<double>* background_pressure;
  grid::CellValues<double>* flame;
  grid::CellValues<double>* pressure;
};
std::expected<GasAmountsTry, GasFailure> BuildResizeTry(GasAmounts& amounts,
                                                        const grid::StaggeredGrid& resized_grid);
GasArraysToRemap ToArraysToRemap(GasAmountsTry& amounts_try);

// NEW (A10). What the liquid solve reads of the gas: views of two arrays owned by the try, bound to its grid.
// Filled once per try from the committed values BuildTry copied, before SolveGas changes anything, so the
// liquid reads the gas pressure committed at the end of the previous step (one step late, 5.5). A18: the liquid
// uses gas_pressure_above_atmosphere only where no pressure::SealedGasBlob claims the cell (open blobs, also
// above a viscous free surface, held blobs, slivers, cells that touch gas but are not gas cells; A24: every
// sealed blob touching liquid is claimed, viscous or not); holds_tracked_gas keeps every such cell out of air
// pockets as before.
struct GasSeenByLiquid {
  // Pa. background_pressure - atmosphere_pressure + pressure of the committed gas, in each cell with
  // holds_tracked_gas; 0 elsewhere. For pressure::EquationInputs::gas_pressure_above_atmosphere.
  const grid::CellValues<double>* gas_pressure_above_atmosphere;
  // 1 in each cell whose committed gas_volume > 0, and in each cell with no committed gas that touches such a
  // cell across a face piece (it takes the value of the touching one with the lowest CellPlaceIndex); else 0.
  // For pressure::EquationInputs::holds_tracked_gas and pressure::AirPocketInputs::holds_tracked_gas.
  const grid::CellValues<std::uint8_t>* holds_tracked_gas;
};
// Stage ChooseActiveCells, right after BuildTry. Reads the touching cells of the range (one border cell).
std::expected<GasSeenByLiquid, GasFailure> ComputeGasSeenByLiquid(
    const grid::StaggeredGrid& grid, grid::CellBlockRange blocks, const GasLaw& law, GasAmountsTry& amounts_try);
```

```cpp
// solve_gas.h (+)
struct GasSettings {
  GasLaw law{};
  BurnSettings burn{};
  pressure::SolveSettings pressure_solve{};  // The one acceptance limit of pressure (Q27).
  double min_gas_volume_share = 0.1;   // No unit. A cell with less gas than this share of its volume is
                                       // closed to gas (small-cell rule, 5.3 step 1).
  double min_temperature = 100.0;      // K. Outside these two the try is rejected.
  double max_temperature = 5000.0;     // K
  double max_background_pressure = 1.0e7;  // Pa, absolute.
  // NEW (A10). A sealed blob that touches liquid and whose equivalent radius is below this many widths of its
  // widest cell is held at atmosphere (5.3 step 8). Must equal pressure::AirPocketSettings::min_radius_in_cells.
  double min_sealed_radius_in_cells = 2.0;
};
inline constexpr double kGasFlowShareLimit = 0.9;  // No unit. Share of a cell's gas that may leave per step
                                                   // under the suggested step limit.
// max_cells a multiple of 64 and max_face_pieces == max_cells / 64 * 240, else kBadInput (plan decision).
struct GasBufferSizes {
  std::uint32_t max_cells, max_face_pieces, max_gas_blobs, max_sources;
  std::uint32_t max_loaded_blocks = 0;  // NEW (A15). 0: one piece. Else the batched layout (6, "Batched solve").
};
class GasBuffers;  // Owns the scratch arrays of this module on one processor. No allocation after it is built.
                   // Holds carried_pressure (the committed gas pressure copied before the solve, so that
                   // SolvePressure's output never aliases its input) and, batched, a pressure::SolveBatches.
std::expected<std::unique_ptr<GasBuffers>, GasFailure> BuildGasBuffers(const GasBufferSizes& sizes, RunOn run_on);
// NEW (A15). Null in one piece.
const pressure::SolveBatches* FindSolveBatches(const GasBuffers& buffers);
const pressure::SolveCopyReport* ReadSolveCopyReport(const GasBuffers& buffers);
// NEW (plan decision, reconcile 2026-11-14 item 3; gas plan Task 29). Zeroes the held SolveBatches' copy report;
// does nothing in one piece; allocates and copies nothing. The step calls it per try (FindSolveBatches stays
// const). The gas PressureBuffers uses pressure's own overload.
void ResetSolveCopyReport(GasBuffers& buffers);

struct GasBlob {                       // One connected body of gas in this step. DECIDED word blob.
  grid::CellPlaceIndex first_cell_index{grid::kNoIndex};
  pressure::BlobClass blob_class = pressure::BlobClass::kSealed;  // kHasReferencePressure: open to ambient air.
  double gas_volume = 0;               // m3
  double background_pressure = 0;      // Pa, absolute, at the end of the step.
  double background_pressure_rate = 0; // Pa/s during the step. 0 for an open or held blob.
  double mean_heat_rate = 0;           // W/m3
  bool touches_liquid = false;         // NEW. A piece of the blob is closed by the liquid solve (5.3 step 1).
  bool held_at_atmosphere = false;     // NEW. Sealed, touches liquid, below min_sealed_radius_in_cells.
};
struct GasAmountChanges {              // What the step did to the totals. Block partials, then math::TreeSum.
  double gas_mass_before = 0, gas_mass_after = 0;      // kg
  double fuel_mass_before = 0, fuel_mass_after = 0;    // kg
  double smoke_mass_before = 0, smoke_mass_after = 0;  // kg
  double fuel_mass_from_sources = 0;         // kg
  double gas_mass_through_open_sides = 0;    // kg, positive out. Same for the two below.
  double fuel_mass_through_open_sides = 0, smoke_mass_through_open_sides = 0;
  double fuel_mass_burned = 0;               // kg
  double heat_released = 0;                  // J, burn and sources.
  double gas_volume_through_open_sides = 0;  // m3, positive out.
  double gas_mass_from_pressure_equalising = 0;   // kg. Blobs joined, opened or held at atmosphere (5.3 step 8).
  double gas_mass_from_volume_error = 0;          // kg. What the solve's leftover volume error amounts to.
  double gas_mass_from_geometry = 0;              // kg. Gas volume found now minus what the last step left, and
                                                  // the amounts new gas cells start with (5.3 step 2).
  double gas_mass_left_in_closed_cells = 0;       // kg. Cells that stopped holding gas with no gas cell to take it.
  double gas_mass_left_with_removed_cells = 0;    // kg. Cells that are no longer active.
  std::uint32_t closed_small_cell_count = 0;      // Cells under min_gas_volume_share.
  std::uint32_t closed_unknown_edge_count = 0;    // Grid-edge pieces neither open to sky nor on a domain side.
  std::uint32_t face_without_marker_count = 0;    // Gas face pieces no marker reached; their velocity was 0.
  std::uint32_t handed_over_cell_count = 0;       // NEW. Cells that stopped holding gas and gave it to a gas cell.
};
// NEW (A18). Same inputs whichever liquid solve follows (A24).
struct GasStepStartInputs {
  const grid::StaggeredGrid* grid; grid::CellBlockRange blocks;   // blocks: every block of the grid.
  pressure::GeometrySnapshot liquid_geometry;                     // FindLiquidSurface of this try.
  const grid::FaceValues<std::uint8_t>* grid_edge_is_open_to_sky; // From solids. Null: no solid in the scene.
  contact::DomainBox domain;
  std::span<const GasSource> sources;
  double step_start_time = 0, step_duration = 0;                  // s
};
struct SealedGasForLiquid {          // NEW (A18). Views into gas_buffers and amounts_try, valid until SolveGas returns.
  std::span<const pressure::SealedGasBlob> blobs;               // By first_cell_index. Empty with no liquid.
  const grid::CellValues<std::uint32_t>* sealed_blob_of_cell;   // Index into blobs, pressure::kNoBlob elsewhere.
  const grid::CellValues<double>* pressure_above_background;    // amounts_try.pressure(): the gas solve's
};                                                              //   pressure of the last commit, Pa.
// NEW (A18). Stage BuildPressureEquations, before the liquid's pressure::BuildPressureEquations, whenever that
// stage runs, viscous or not (A24: the listed blobs are unknowns of pressure::SolvePressure or of
// viscosity::SolveViscosityAndPressure); in a gas scene where that stage does not run (no liquid), at the start
// of stage SolveGas. Steps 1 to 4 of 5.3 (step 1 without any liquid
// velocity), then the gas blobs from pressure::BuildPressureEquations on that geometry (kWhenRatesApplied, zero
// rates; the second PressureBuffers) and ComputeBlobOfCell. A sealed blob is listed when it touches liquid (one
// of its cells has a face piece to a kLiquid cell with open_area > 0: a future liquid connection) and is not
// held (step 8's radius rule): gas_volume = sum of V, background_pressure = step 8's equalised p0 (computed, not
// yet applied), heat_volume_rate = ((gamma - 1) / gamma) * sum V q / p0, boundary_volume_rate = sum over the
// blob's cells, in ascending index, of -s * solid_volume_rate over their pieces whose other cell is outside the
// blob and is not a kLiquid cell across open area. Writes amounts_try and gas_buffers only; the changes of steps
// 2 to 4 are kept in gas_buffers and returned in SolveGas's GasResult::changes. Per-blob sums by the per-blob rule.
std::expected<SealedGasForLiquid, GasFailure> ApplyGasStepStart(
    const GasStepStartInputs& inputs, const GasSettings& settings, GasAmountsTry& amounts_try,
    GasBuffers& gas_buffers, pressure::PressureBuffers& gas_pressure_buffers);
struct SolveGasInputs {
  const grid::StaggeredGrid* grid; grid::CellBlockRange blocks;   // blocks: every block of the grid.
  pressure::GeometrySnapshot liquid_geometry;   // FindLiquidSurface of this try; also in a scene with no liquid.
  // The gas material set's sums (transfer, mask 1 for the gas material only): `velocity` after ApplyForces,
  // m/s, and `value_kind`. SolveGas reads it and changes nothing in it.
  const transfer::GridFromParticles* gas_grid_from_particles;
  // NEW (A10). m/s. The velocity after the liquid solve (pressure::SolvePressure or
  // viscosity::SolveViscosityAndPressure) of the set holding the liquid, and pressure::ComputeWasSolved on that
  // solve's equations. Both null: the scene has no liquid.
  const grid::FaceValues<double>* liquid_velocity_after_pressure;
  const grid::FaceValues<std::uint8_t>* liquid_was_solved;
  double liquid_lightest_density = 0;  // NEW. kg/m3, materials::MaterialTable::lightest_density(). 0: no liquid.
  const grid::FaceValues<std::uint8_t>* grid_edge_is_open_to_sky;  // From solids. Null: no solid in the scene.
  contact::DomainBox domain;                    // Its closed sides are closed to gas; the others are open air.
  std::span<const GasSource> sources;           // Rates of overlapping sources add, in this order.
  GasWind wind;
  double step_start_time = 0, step_duration = 0;  // s
  // NEW (A18). SealedGasForLiquid::blobs of this try, and the liquid solve's pressure::SolvedArrays::
  // sealed_gas_pressure or viscosity::SolvedCouplingArrays::sealed_gas_pressure (A24; Pa above atmosphere at the
  // step end, one per blob). Sizes that differ: kBadInput. A gas blob is coupled when its first_cell_index equals
  // an entry's (both lists ordered by it).
  std::span<const pressure::SealedGasBlob> sealed_gas_blobs;
  std::span<const double> sealed_gas_pressure;
};
struct SolvedGasArrays {        // NEW. The step's try arrays of the gas material set, bound to the try grid.
  grid::FaceValues<double>* velocity_after_pressure;  // m/s, every face piece. For SaveGridAfterPressure.
  grid::FaceValues<std::uint8_t>* was_solved;         // 1: the gas solve corrected the piece, or the liquid closed
  grid::CellValues<std::uint8_t>* cell_was_solved;    //   it (5.3 step 12). For SaveGridAfterPressure and
};                                                    //   GridToParticles of the gas set.
struct GasResult {
  pressure::FlowImbalanceReport flow_imbalance;       // Of the gas solve.
  std::span<const GasBlob> blobs;                     // Ordered by first_cell_index. In gas_buffers.
  GasAmountChanges changes;
  // s. Suggested longest next step: kGasFlowShareLimit * gas_volume / outflow, smallest over cells. +infinity
  // when no cell has outflow. No sealed blob term (A24).
  double step_limit = 0;
  int iteration_count = 0;
};
// Stage SolveGas, right after the liquid solve (5.3, step 1 with the liquid's velocity, then steps 5 to 12; A18).
// kBadInput without an ApplyGasStepStart on the same try. Writes amounts_try,
// the outputs and the buffers on success; on failure nothing in them is valid to read. pressure_buffers is a
// second instance, used by gas only (the one ApplyGasStepStart used). cancel_requested may be null; it is passed
// to pressure::SolvePressure (kCancelled). With buffers in the batched layout (A15) ApplyGasStepStart and
// SolveGas run batch by batch; no signature changes.
std::expected<GasResult, GasFailure> SolveGas(
    const SolveGasInputs& inputs, const GasSettings& settings, const std::atomic<bool>* cancel_requested,
    SolvedGasArrays outputs, GasAmountsTry& amounts_try, GasBuffers& gas_buffers,
    pressure::PressureBuffers& pressure_buffers);

// The pieces of ApplyGasStepStart (steps 1 to 4) and SolveGas (1, 5 to 12), exposed for tests and checks. Same
// order as the steps.
std::expected<pressure::GeometrySnapshot, GasFailure> BuildGasGeometry(
    const SolveGasInputs& inputs, const GasSettings& settings, GasBuffers& buffers);              // step 1
std::expected<void, GasFailure> ApplyGasSources(const SolveGasInputs& inputs, const GasSettings& settings,
    const pressure::GeometrySnapshot& gas_geometry, GasAmountsTry& amounts_try, GasBuffers& buffers);  // step 3
std::expected<void, GasFailure> ApplyBurn(const GasSettings& settings, double step_duration,
    const pressure::GeometrySnapshot& gas_geometry, GasAmountsTry& amounts_try, GasBuffers& buffers);  // step 4
std::expected<void, GasFailure> ApplyGasLift(const SolveGasInputs& inputs, const GasSettings& settings,
    const pressure::GeometrySnapshot& gas_geometry, const GasAmountsTry& amounts_try,
    grid::FaceValues<double>& velocity_before_pressure, GasBuffers& buffers);                     // steps 5, 6
// Fills the GasBlob list; writes and range-checks the end background_pressure of every non-held blob (a coupled
// blob's from inputs.sealed_gas_pressure, A18); a held blob's scaling to atmosphere is applied by SolveGas after
// step 11.
std::expected<void, GasFailure> ComputeGasAddedVolumeRate(const SolveGasInputs& inputs,
    const GasSettings& settings, const pressure::PressureEquations& equations, GasAmountsTry& amounts_try,
    grid::CellValues<double>& added_volume_rate, GasBuffers& buffers);                            // steps 8, 9
std::expected<void, GasFailure> ApplyGasFlow(const SolveGasInputs& inputs, const GasSettings& settings,
    const pressure::GeometrySnapshot& gas_geometry, const grid::FaceValues<double>& velocity_after_pressure,
    const grid::CellValues<double>& added_volume_rate, GasAmountsTry& amounts_try, GasBuffers& buffers);  // 11
```

Removed (A24): `enum class SealedGasCoupling` (`kInsideLiquidSolve`, `kReadOneStepLate`),
`GasStepStartInputs::sealed_gas_coupling`, `SolveGasInputs::sealed_gas_coupling` with its `kBadInput` rule, and
the sealed blob term of `GasResult::step_limit`. Viscosity's joint solve carries the sealed blob term, so no
coupling choice remains.

```cpp
// gas_markers.h (+)
struct MarkerSettings {
  int markers_per_cell = 8;       // Target in a gas cell: one per cell eighth.
  int min_markers_per_cell = 4;   // Below this, markers are added up to the target.
  int max_markers_per_cell = 16;  // Above this, the newest (highest id) are removed down to the target.
};
struct MarkerChanges {            // Lives in GasBuffers. Ordered by cell place, then cell eighth.
  particles::NewParticles to_add;                    // mass = ambient density * rest_volume (a sample_weight).
  std::span<const std::size_t> to_remove;            // Ascending indexes in the try's current grouping.
  // NEW (plan decision, 2026-11-14, buffers in batched runs). The GPU form builds the lists in the "gas: marker
  // changes on the processor" scratch, reads the two counts in its status record, then one slice ReadBuffer per
  // list array into the CPU rows: copy_count = those calls, bytes_copied their bytes; 0 and 0 on the CPU. The
  // spans above point into CPU memory in every case (the step hands them to particles::AddParticles and
  // RemoveParticles, which take CPU spans).
  std::uint32_t copy_count = 0;
  std::uint64_t bytes_copied = 0;
};
// Stage AddNewParticles. Uses the committed gas_volume (which cells held gas) and the committed velocity after
// pressure of the gas set (the velocity a new marker starts with, taken at its cell centre). Changes nothing
// itself: the step applies to_add with particles::AddParticles(..., AddReason::kRefillRegion) and to_remove with
// particles::RemoveParticles(..., RemoveReason::kMarkerNotNeeded).
std::expected<MarkerChanges, GasFailure> FindMarkerChanges(
    const GasAmounts& amounts, const grid::StaggeredGrid& grid, grid::CellBlockRange blocks,
    const particles::ParticleFields& gas_markers, const grid::FaceValues<double>& committed_velocity,
    const GasLaw& law, const MarkerSettings& settings, GasBuffers& buffers);
// NEW (plan decision, 2026-11-14, buffers in batched runs). The CPU form, for a batched run (markers in CPU memory,
// amounts on the run's processor, the step's CPU mirrors of the committed gas volume and gas velocity and its CPU
// grid copy). Every input in CPU memory (grid tables, both arrays, the markers), else kBadInput; arrays bound to
// inputs.grid (version and block count), else kWrongGridVersion. Same rule and same lists as the amounts form,
// which calls it.
struct MarkerGridInputs {
  const grid::StaggeredGrid* grid;
  const grid::CellValues<double>* gas_volume;
  const grid::FaceValues<double>* committed_velocity;
};
std::expected<MarkerChanges, GasFailure> FindMarkerChanges(const MarkerGridInputs& inputs,
    grid::CellBlockRange blocks, const particles::ParticleFields& gas_markers, const GasLaw& law,
    const MarkerSettings& settings, GasBuffers& buffers);
// 1 for each cell that must stay or become active in the next grid: within margin_in_cells of a cell whose
// temperature differs from ambient by more than temperature_tolerance, or that holds fuel or smoke above
// density_tolerance, or that a source box touches. For the step's ChooseActiveCells. Marks only cells of `grid`
// (plan-decisions "Gas"): every active cell whose box overlaps a marking cell's box grown by margin_in_cells of
// the marking cell's width, and every cell a source box overlaps; growing the active region is the step's job.
struct ActiveGasSettings { int margin_in_cells = 8; double temperature_tolerance = 1.0;  // K
                           double density_tolerance = 1.0e-6; };                         // kg/m3
std::expected<void, GasFailure> FindCellsToKeepActive(
    const GasAmounts& amounts, const grid::StaggeredGrid& grid, grid::CellBlockRange blocks,
    std::span<const GasSource> sources, const GasLaw& law, const ActiveGasSettings& settings,
    grid::CellValues<std::uint8_t>& keep_active);
```

```cpp
// gas_snapshot.h, gas_scene_check.h, gas_memory_estimate.h (CPU only)
struct GasSnapshot {                 // DECIDED word. CPU-owned copy of committed values, for volume output.
  double time = 0; std::uint64_t state_version = 0; grid::GridVersion grid_version{};
  std::vector<grid::CellId> cells;          // Gas cells only, ascending id. One entry per array below.
  std::vector<double> gas_volume_share;     // 0..1, gas volume over cell volume. The "known" mask of output.
  std::vector<double> gas_density;          // kg/m3.     Output channel `density` (as on the branch).
  std::vector<double> temperature;          // K.         Channel `temperature`. From ComputeTemperature.
  std::vector<double> fuel_density;         // kg/m3.     Channel `fuel`.
  std::vector<double> flame;                // kg/(m3 s). Channel `flame`.
  std::vector<double> smoke_density;        // kg/m3.     Channel `smoke` (branch: `soot`).
};
std::expected<GasSnapshot, GasFailure> ReadGasSnapshot(const GasAmounts& amounts, const grid::StaggeredGrid& grid,
                                                       const GasLaw& law);
// NEW (from I/O). The full committed amounts, for saved_state. Every active cell of the committed grid,
// ascending CellId, one entry per array; arrays as in GasAmounts. Building from it on the same grid gives
// GasAmounts whose next steps are the bytes of an uninterrupted run.
struct GasAmountsSnapshot {
  double time = 0; std::uint64_t state_version = 0; grid::GridVersion grid_version{};
  std::vector<grid::CellId> cells;
  std::vector<double> gas_volume, gas_density, fuel_density, smoke_density, background_pressure, flame, pressure;
};
std::expected<GasAmountsSnapshot, GasFailure> ReadGasAmountsSnapshot(const GasAmounts& amounts,
                                                                     const grid::StaggeredGrid& grid);
// kBadInput: a cell of the snapshot is not active in grid, an active cell is missing, an array has the wrong
// length or a value is not finite. Keeps time and state_version.
std::expected<GasAmounts, GasFailure> BuildGasAmountsFromSnapshot(const GasAmountsSnapshot& snapshot,
                                                                  const grid::StaggeredGrid& grid,
                                                                  const GasLaw& law, RunOn run_on);
// NEW (decision in changes-to-apply.md: gas and render output). The output snapshot from the full amounts, on
// the CPU, with no grid (only its frame): render output calls it on the gas part of the step's SimulationSnapshot. Keeps time,
// state_version and grid_version; keeps the cells with gas_volume > 0 in the snapshot's ascending CellId order;
// temperature = ComputeTemperature(law, background_pressure, gas_density) per cell; gas_density,
// fuel_density, flame and smoke_density copied. gas_volume_share is gas_volume over the cell's volume, with the
// cell width taken from `frame` and the cell's size level.
// Gives the same values as ReadGasSnapshot on the committed amounts and grid.
GasSnapshot ToGasSnapshot(const GasAmountsSnapshot& amounts, const GasLaw& law, const grid::GridFrame& frame);
struct GasSceneSummary {             // Filled once by the step's capability check.
  std::uint32_t gas_material_count = 0;
  std::optional<std::uint32_t> gas_ingredient_index;   // NEW. materials::MaterialTable::gas_ingredient_index().
  double gas_ingredient_rest_density = 0;              // NEW. kg/m3, ingredient_rest_density() at that index.
  pressure::AirPocketSettings air_pockets{};           // NEW. The scene's settings for pressure's air pockets.
};
// kMoreThanOneGas, kGasLawDisagrees (5.6). Liquid in the scene and a batched run are not exclusions.
std::expected<void, GasFailure> CheckGasCanRun(const GasSceneSummary& scene, const GasSettings& settings);
// Rows named "gas: ...", sizes sizeof the real types, each with its MemoryWhere. With max_loaded_blocks > 0
// (A15): the rows streamed in the batched layout (6) with where = kCpu at the limits, the same rows with
// where = kRunOn at max_loaded_blocks, and pressure::BuildSolveBatchesMemoryEstimate. Changed (plan decision,
// 2026-11-14, buffers in batched runs): the MarkerChanges rows ("gas: marker changes to add", "gas: marker changes
// to remove") have where = kCpu with either run_on and in every layout (not part of the batched list); with
// RunOn::kGpu the rows add "gas: marker changes on the processor" (the same given count and element sizes,
// where = kRunOn, kOneCall), the GPU scratch the lists are built in before their read-back.
std::vector<MemoryEstimateEntry> BuildGasMemoryEstimate(const GasBufferSizes& sizes);
```

**Provided for `shared-interface.md` section (c):** gas markers need only the common particle arrays
(confirmed, 5.2); the gas `GeometrySnapshot` and `added_volume_rate` (`BuildGasGeometry`,
`ComputeGasAddedVolumeRate`, using `PressureBlob` totals and `ApplyAddedVolumeRate`); gas viscosity is the
number 0 and marker friction is 0 (5.2); no weights for cell-centred values are asked of the grid (5.3 step 5).
**Provided to pressure (A10):** `ComputeGasSeenByLiquid`, whose two arrays are bound to the try grid and hold
the gas pressure committed at the end of the previous step. **(A18):** `ApplyGasStepStart`'s
`SealedGasForLiquid`, which the step passes as `EquationInputs::sealed_gas_blobs`, `sealed_gas_blob_of_cell` and
`gas_pressure_above_background`.

## 4. Consumes from other modules

**Foundation (EXISTS):** `Vec3`, `Axis`, `kUp`, `RunOn`, `CheckCanRunOn`, `FLUIDITY_CPU_AND_GPU`, `cpu_only.h`,
`std::expected`. **Foundation additions (EXISTS, `shared-interface.md` (a) and `changes-to-apply.md`):**
`Buffer<T>`, `BufferError`, `math::Pow`, `math::Min`, `math::Max`, `math::IsFinite`, `math::TreeSum`,
`ComputeTreeSum`, `ComputeLargestMagnitude`, `BuildGroupsByDestination` (cells grouped by blob), `MemoryEstimateEntry`
with `MemoryWhere { kRunOn, kCpu }` (plan-decisions), `math::Cbrt`, `math::Sqrt`,
`math::Expm1(double)` (`exp(value) - 1` without losing the small part; the burn formula and its 1e-14 tests
need it; last-bits group, not bit-identical between CPU and GPU).

**Grid (PROVIDED by grid 3.1 to 3.7):** `StaggeredGrid`, `GridTables`, `GridVersion`, `CellBlockRange`,
`CellValues<T>`, `FaceValues<T>`, `BuildCellValues(run_on, max_cell_blocks)`, `BuildFaceValues`,
`ApplyGridVersion(values, grid)` (no allocation per grid version), `CellPlaceIndex`, `FacePlaceIndex`, `CellId`,
`kNoIndex`, `ToCellId(tables, cell)`, `FindCellPlace`, `FindFacePieces(tables, cell)` (ascending), `FindCellLow`,
`FindCellHigh`, `ToAxis(face)`, `ComputeFaceArea`, `ComputeFacePosition`, `ComputeCellCenter`,
`ComputeCellCenterDistance`, `ComputeCellWidth`, `ToSizeLevel(tables, block)`, `ToCellBlockIndex`,
`FindTouchingCells` (`FindCellsToKeepActive`, `ComputeGasSeenByLiquid`, 5.3 step 2), `FacePieceKind`
(`kGridEdgeLow`, `kGridEdgeHigh`), `BorderWidth::kTouchingCells`.

**Particles (PROVIDED by particles 3.1, 3.5):** `ParticleFields` (`position`, `id`, `group_begin`,
`cell_begin`), `NewParticles`, `AddParticles(..., AddReason::kRefillRegion)`, `RemoveParticles(...,
RemoveReason::kMarkerNotNeeded)`, `ComputeRefillParticle` (positions of the eight markers of a cell),
`CountRule::kMayBeSplitAndMerged` for the gas material.

**Transfer (PROVIDED by transfer 3):** material sets (A8): the gas material is a set of its own, given to every
stage as `material_is_counted` with one entry per material, 1 for the gas material only; the set holding the
liquid has 0 for the gas material, so markers never enter the liquid's `GridFromParticles` or
`CellsFromParticles` (A10). `GridFromParticles::velocity` and `value_kind` of the gas set (`kNoValue` on a gas
face is counted); `SaveGridAfterPressure` and `GridToParticles` of the gas set take `SolvedGasArrays`.
`FaceValueKind::kFromSolve` is recorded by transfer for a gas face that `was_solved` marks.

**Contact (PROVIDED by contact 3.2):** `DomainBox` with `ClosedSides`, `DomainSide`; `MoveParticles` moves the
markers with friction 0 (run by the step).

**Pressure (PROVIDED by pressure 3). Used exactly:** `GeometrySnapshot`, `CellGeometry`, `CellContent` (with
`kGas`), `FacePieceGeometry`, `FacePieceEnds`; the liquid snapshot's `air_volume`, `open_volume`, `open_area`,
`air_open_area`, `solid_volume_rate` (so gas never clips a cell itself); `PressureBuffers`,
`BuildPressureBuffers` (a second instance); `EquationInputs` with `velocity_before_pressure` = the gas copy
after lift, `mixture_density` = the try's `gas_density`, `bulk_stiffness` = +infinity, `carried_pressure` = the
buffers' copy of the committed gas `pressure`, `lightest_density = ComputeAmbientDensity(law)`,
`max_volume_error_rate = settings.pressure_solve.max_volume_error_rate`, `added_volume_rate` = null at build,
no `blocking_material_faces`, no `sealed_gas_blobs`, `air_pockets` = null,
`gas_pressure_above_atmosphere` = null, `holds_tracked_gas` = null, `surface_tension_with_air` = 0, no
`material_boundaries`, no `free_solid_pushes`, `sealed_blob_check = SealedBlobCheck::kWhenRatesApplied`;
`BuildPressureEquations`; `PressureBlob` (`first_cell_index`, `blob_class`, `filled_volume`,
`incompatible_volume_rate`); `kNoBlob`, `ComputeBlobOfCell`; `ApplyAddedVolumeRate` (called right after
`ComputeGasAddedVolumeRate`); `SolvePressure(equations, settings, cancel_requested, SolvedArrays{pressure,
velocity_after_pressure}, buffers)`; `SolveResult`; `SolveSettings`; `FlowImbalanceReport`; `ComputeWasSolved`;
`PressureFailure`, `PressureError::kCancelled`; `BlobClass`; `AirPocketSettings` (in `GasSceneSummary`). For the
liquid side (A10): `EquationInputs::gas_pressure_above_atmosphere` and `holds_tracked_gas`,
`AirPocketInputs::holds_tracked_gas`, which the step fills from `GasSeenByLiquid`. For the coupled solve (A18,
`plan-decisions.md` "Coupling inside the pressure solve"): `SealedGasBlob` (filled by `ApplyGasStepStart`),
`EquationInputs::sealed_gas_blobs`, `sealed_gas_blob_of_cell`, `gas_pressure_above_background` (filled by the step
from `SealedGasForLiquid`), `SolvedArrays::sealed_gas_pressure` (read through `SolveGasInputs`). For batches
(A15, `pressure/solve_batches.h`): `BufferSizes::max_loaded_blocks` of the gas `PressureBuffers`, `SolveBatches`,
`BuildSolveBatches`, `ApplySolveBatches`, `LoadSolveBatch`, `WriteBatchValues`, `ReadBatchValues`,
`JoinLabelsAcrossBatches`, `SolveCopyReport`, `BuildSolveBatchesMemoryEstimate`, `SolvePressure` in the batched
layout. **Not used:** surface
tension, `PressureLaw`, `AirPockets`, `MaterialBoundary`, `FreeSolidPush`, volume correction,
`ComputeCarriedPressure`, the open sea.

**Materials (PROVIDED by materials 3):** the gas material (`MaterialKind::kGas`, one ingredient, viscosity 0,
friction 0, surface tension 0, rest density `ComputeAmbientDensity(law)`, `PressureLawKind::kCannotBeSqueezed`,
own-motion shares 0 and 0, `volume_ratio` never changed); `MaterialTable::gas_ingredient_index()` and
`ingredient_rest_density()` (for `GasSceneSummary`); `MaterialTable::lightest_density()` (for
`SolveGasInputs::liquid_lightest_density`).

**Solids (PROVIDED by solids 3, after its update):** `solids::ToSolidCoverage` giving `pressure::SolidCoverage`
over every active cell, also where there is no liquid; `solids::FindGridEdgesOpenToSky` giving
`grid_edge_is_open_to_sky`. A free solid moves gas (through `solid_volume_rate`) and is not pushed by it.

**Viscosity (PROVIDED by viscosity 3):** in a scene with a viscous ingredient the liquid velocity after
pressure comes from `viscosity::SolveViscosityAndPressure`; gas marker sets are not passed to it. Since A24 its
joint solve carries the sealed blob term as `pressure::SolvePressure` does: it reads the blobs from the
liquid's equations (built with the `SealedGasForLiquid` views) and returns each blob's pressure in
`viscosity::SolvedCouplingArrays::sealed_gas_pressure`. No coupling choice remains.

**Step (PROVIDED by step 3, after its update; `changes-to-apply.md` "10 Step"):** the calls and order of 5.3,
including the new stage `SolveGas` right after the liquid solve; `ReadyScene` carries the keyed
`gas::GasSource` list and `gas::GasWind`; gravity applied to every Z face piece of every set as the one
product `step_duration * gravity`, so that gas can take it back exactly (5.3 step 5); `CollideSolids` not run
on the gas set; a second `PressureBuffers` instance counted with `pressure::BuildPressureMemoryEstimate`;
`GasResult::step_limit` taken into the next step duration; a retry with a shorter step on
`kStepTooLongForGasFlow`; `kCancelled` mapped to a cancelled try; `SolvedGasArrays` as try arrays of the gas
set; `cancel_requested` passed to `SolveGas`. A18: stage `BuildPressureEquations` calls `ApplyGasStepStart`
before the liquid's `pressure::BuildPressureEquations` and passes `SealedGasForLiquid` into the liquid's
`EquationInputs` (a gas scene without that stage calls `ApplyGasStepStart` at the start of stage `SolveGas`);
stage `SolvePressure` or `SolveViscosityAndPressure` gives `sealed_gas_pressure` (a step-owned span), which stage
`SolveGas` passes in `SolveGasInputs` (A24: the same after either stage). A15: the gas buffers and the gas
`PressureBuffers` built with the run plan's `max_solve_loaded_blocks` in `kBatchedWithSolvesInPieces`; the copy
reports summed into the batch report.

**Cell resize and batches (PROVIDED by cell resize and batches 3, 5.4):** the remap of the seven per-cell
arrays through `BuildResizeTry` and `GasArraysToRemap` when cells are split or merged (the three densities and
`flame` weighted by `gas_volume`; `gas_volume` shared out on a split and summed on a merge;
`background_pressure` and `pressure` copied on a split, `gas_volume`-weighted on a merge); in a batched run the
markers pass in batches, and `ApplyGasStepStart` and `SolveGas` run on the whole grid (`kBatched`) or with their
scratch streamed piece by piece (`kBatchedWithSolvesInPieces`, A15; `batches::BuildBatchPlan` sizes it).

**I/O and CLI (PROVIDED by I/O 3, after its update):** `GasLaw`, `BurnSettings`, `GasSettings`,
`MarkerSettings`, `ActiveGasSettings`, the keyed `GasSource` list and `GasWind` in the ready scene (I/O owns the
key storage and calls `CheckGasSources`); `saved_state` stores `GasAmountsSnapshot` and rebuilds with
`BuildGasAmountsFromSnapshot`.

**Surface, foam and spray (PROVIDED by its design, after its update):** turns the gas part of the step's
snapshot (`GasAmountsSnapshot`) into a `GasSnapshot` with `ToGasSnapshot` and writes it as five OpenVDB volume
channels and the known mask, one file per frame, beside the liquid surface in the same frame (A10, A14).

**Blender (PROVIDED by its design, after its update; A20):** each Blender Wind force field gives
`WIND_ACCELERATION_PER_STRENGTH * strength * direction` (1.0 m/s2 per unit of Strength, the branch's factor,
`gas_authoring.py:63-67`; direction the field's local +Z in world axes; several fields add), written as the
scene's `gas_wind: [{time, acceleration}]` keys (m/s2, world, Z-up; none in a domain without gas), which I/O
hands over as `GasWind` unconverted; settings the gas cannot honour give add-on findings. One scene may hold
liquid and gas.

## 5. Method

Symbols appear in formulas only: `p0` background pressure, `rho` gas density, `T` temperature, `V` gas
volume of a cell, `q` heat rate per volume, `dt` step duration, `gamma` heat ratio, `R` gas constant, `g`
gravity. Each item says PORTED (the branch's maths), FIXED or NEW.

### 5.1 The model, and what is stored

1. *Two pressures (PORTED).* `background_pressure` (the branch's `absolute_pressure`; Pa, one value per gas
   blob) sets the density: `rho = p0 / (R T)`. The `pressure` of the solve (the branch's `gauge_pressure`)
   only moves the gas. A blob open to ambient air has `p0 = atmosphere_pressure`.
2. *Three stored scalars, not four (FIXED, defect 7).* The module stores `gas_density` (the branch's
   `carrier`), `fuel_density`, `smoke_density`, and derives `temperature = p0 / (R rho)`. The branch's
   fourth scalar is implied: at fixed `p0` the heat content of a cell, `rho c_p T = gamma p0 / (gamma - 1)`
   per volume, does not depend on `T`; heating a cell does not raise its heat content, it pushes air out.
   Moving the air conservatively with a velocity whose outflow equals the added volume therefore moves
   temperature correctly. Derivation for the method paper: with `div u = s` and `d rho / dt + div(rho u) = 0`,
   `D rho / Dt = -rho s`, which is the temperature equation of the old paper for
   `s = ((gamma - 1) q - dp0/dt) / (gamma p0)`. `ComputeTemperature` equals the branch's
   `enthalpy / carrier / c_p` whenever its `enthalpy = 3.5 p0` holds.
3. *Fuel and smoke (PORTED)* are kg per m3 of gas, moved the same way. They do not enter the gas law. Smoke
   enters the lift.
4. *Not in the model, as on the branch:* oxygen, computed heat conduction, any speed of sound.

### 5.2 Particles and grid (confirms the particles design)

PORTED: velocity is carried by marker particles and everything else by the grid; the gas's inertia in the
pressure equations comes from `gas_density`, never from marker weights; markers keep no own motion
(`own_motion_kept = 0`, `own_motion_in_travel = 0`: the branch gathers `new_pic_velocity` only,
`gas_apic_3d.cc:233`) but carry `fitted_velocity_change` (the branch's `velocity_slopes`). A marker needs only
the common particle arrays. Friction 0. Viscosity 0 (materials requires it; gas marker sets are never passed
to viscosity). `CountRule::kMayBeSplitAndMerged`.

NEW (A8, A10): the gas material is a material set of its own. Its markers have their own face sums, velocity
and snapshots; the set holding the liquid counts no marker, so markers never change the liquid's mixture
density, sensed volume ratio, distance to surface or face velocity, and liquid particles never enter the gas
sums. FIXED (defect 8): `mass` is a `sample_weight` of `ComputeAmbientDensity(law) * rest_volume`, with
`rest_volume` one eighth of the marker's cell volume when added, instead of the branch's weight 1;
`volume_ratio` stays 1 and nobody changes it. Face velocities are then volume-weighted among markers of
different cell sizes. FIXED (defect 9): markers are moved by `contact::MoveParticles` like every particle;
`AdvectMarker` and its half-cell limit are not ported.

*Upkeep (`FindMarkerChanges`; NEW rule, replaces `RemapGasApicMarkers3D`).* Per cell that held gas at the
last commit, in cell place order: fewer than `min_markers_per_cell` markers: add one at each cell-eighth
centre (`ComputeCellEighthCenter`, shared `gas_marker_math.h`) that holds no marker (eighths ascending) until
`markers_per_cell`; more than `max_markers_per_cell`:
remove the highest ids down to `markers_per_cell`. A marker in a cell that held no gas (in liquid, in a solid,
in a cell under the small-cell share) is removed with `RemoveReason::kMarkerNotNeeded`. A new marker's
velocity is the committed gas-set face velocity averaged to its cell centre (per axis: the area-weighted mean
of the low pieces and of the high pieces, then their mean); it carries no fitted change. The branch keeps the
first marker of each cell and one marker per cell; with eight, a cell's fitted change rests on several samples
and "first" needs no meaning.

### 5.3 Where gas sits in the step

Stages by name (the step design's list). Within one step (A18): `ApplyGasStepStart` (steps 1 to 4, and the
list of sealed blobs that touch liquid) before the liquid's equations; the liquid solve, holding those sealed
blobs as unknowns and reading every other gas cell's pressure as committed at the end of the previous step;
then `SolveGas`, reading the liquid's velocity after this step's solve and the sealed blobs' solved pressures.

| Stage | What gas adds |
|---|---|
| `AddNewParticles` | `FindMarkerChanges` (committed amounts and grid), applied by the step with `AddParticles` and `RemoveParticles(..., RemoveReason::kMarkerNotNeeded)`; the lists are in CPU memory in every layout. In a batched run (both modes) the stage runs on the CPU with the `MarkerGridInputs` form on `step::FindCpuGridCopy` and the step's mirrors of the committed `gas_volume` and gas `velocity_after_pressure`, refreshed once per committed `state_version` (plan decision, 2026-11-14, buffers in batched runs) |
| `ChooseActiveCells` | the marks of `FindCellsToKeepActive` (committed amounts, on the amounts' processor) join the step's list (in a batched run through one counted copy into `"step: keep active on the CPU"`); after `grid::BuildGrid`: `BuildTry` on the try grid, then `ComputeGasSeenByLiquid` |
| `FindSolidSurfaces` | solids fill `SolidCoverage` over every active cell and `grid_edge_is_open_to_sky` |
| `ParticlesToGrid`, `SaveGridBeforeForces`, `GridToParticles`, `MoveParticles` | run once for the gas set as for every set; nothing gas-specific |
| `ApplyForces` | gravity on every Z face with a value, gas set included; gas takes it back in step 5 below |
| `CollideSolids` | not run on the gas set: gas sees solids only through `SolidCoverage` and its geometry |
| `FindLiquidSurface` | runs in every gas scene, also with no liquid (every open cell is then air); `FindAirPockets` gets `holds_tracked_gas` |
| `BuildPressureEquations` | first `ApplyGasStepStart` (A18; whichever liquid solve follows, A24); the liquid's `EquationInputs` get `gas_pressure_above_atmosphere`, `holds_tracked_gas` and the `SealedGasForLiquid` views |
| `SolveViscosityAndPressure` or `SolvePressure` | the liquid solve, then `ComputeWasSolved`; either solves each listed sealed blob as one extra unknown and returns its `sealed_gas_pressure` (`SolvePressure` through `SolvedArrays`, `SolveViscosityAndPressure` through `SolvedCouplingArrays`, A24); skipped when the scene has no liquid |
| `SolveGas` (NEW stage) | in a gas scene without stage `BuildPressureEquations`, first `ApplyGasStepStart`; then `SolveGas` with the liquid's velocity after pressure, `was_solved`, `sealed_gas_blobs` and `sealed_gas_pressure`; writes `SolvedGasArrays` |
| `SaveGridAfterPressure` | the gas set with `SolvedGasArrays::velocity_after_pressure` and `was_solved` |
| `CheckStep`, `CommitStep` | `GasResult::changes` (step-start changes included) is checked and reported; `CommitTry`; `step_limit` enters the next duration |
| `ResizeCells` | cell resize and batches remaps the seven arrays through `BuildResizeTry` |

There is no `MoveGas` stage: the flow of amounts is step 11 of `SolveGas`. The steps, numbered as in the code
and in `<proc-solve-gas>`: steps 1 to 4 run in `ApplyGasStepStart` (A18), so the liquid solve has this step's
heat; `SolveGas` runs step 1 again (only liquid-closed pieces change, now with the liquid's velocity) and steps 5
to 12. The gas equations are therefore built twice per step (step start: zero rates, for the blob list; step 7:
the same blobs, asserted in a test). The order burn, heating, lift, solve, flow is the branch's
(`gas_substep_3d.cc:121-183`); sources come before the flow, where the branch adds them after (plan-decisions
"Gas"), so one-step numbers differ from the branch's for that reason.

1. **Gas geometry (`BuildGasGeometry`, NEW).** Built at the step start without any liquid velocity, and again
   in `SolveGas` with it (A18). From the liquid snapshot, with no clipping of its own. A cell
   is a gas cell when its content is `kAir` and `air_volume >= min_gas_volume_share * cell volume`; it gets
   `content = kGas`, `filled_volume = open_volume = air_volume`. Any other cell is `kSolid` for gas (counted
   in `closed_small_cell_count` when only the share rule closed it). Per face piece touching a gas cell:
   - between two gas cells: `kLiquidToLiquid`, `open_area = air_open_area`, `solid_volume_rate` copied,
     `pressure_distance = ComputeCellCenterDistance`;
   - corrected by the liquid solve (`liquid_was_solved`), or leading to a cell that is not a gas cell:
     `kLiquidToSolid`, `open_area = 0`, `solid_volume_rate = liquid snapshot's solid_volume_rate +
     (liquid_was_solved ? liquid open_area * liquid_velocity_after_pressure : 0)`. The liquid surface pushes
     gas like a moving solid, with the velocity the liquid solve left on the piece; each gas cell with such a
     liquid-closed piece is marked `touches_liquid` (in `GasBuffers`);
   - a grid-edge piece: closed (`open_area = 0`) when it lies on a closed side of `domain`, or when
     `grid_edge_is_open_to_sky` is given and is 0 there (counted in `closed_unknown_edge_count`); otherwise
     open air: `kLiquidToAir`, `liquid_share = 1`, `pressure_distance` = half the cell width. With
     `air_pockets = null` and no surface tension, pressure gives it reference pressure 0. This is the
     branch's open side (`gas_uniform_domain_3d.cc:128-133`).
   In a box of same-size cells with no solid and no liquid this is exactly the branch's domain.
2. **Starting amounts (NEW).** In this order, so that no cell reads a value written in the same pass:
   (a) a gas cell that held gas at the last commit keeps its densities and takes `gas_volume = air_volume`;
   `gas_mass_from_geometry += gas_density * (air_volume - committed gas_volume)`. (b) A cell that held gas
   and is no longer a gas cell (liquid rose into it, a solid moved in, it fell under the share) hands its
   amounts to the touching cell of kind (a) with the lowest `CellPlaceIndex` (`handed_over_cell_count`): the
   receiver keeps its volume, its masses add, and its background pressure becomes the one whose gas amount
   `V p^(1/gamma)` is the sum of both (the air pocket rule: gas squeezed in without heat). With no such
   neighbour the mass goes to `gas_mass_left_in_closed_cells`. A sealed gas blob being filled by liquid
   therefore keeps its gas. (c) A gas cell with committed `gas_volume = 0` (new, or liquid or solid before)
   copies the densities and background pressure of the touching cell of kind (a) with the lowest
   `CellPlaceIndex`, after (b); with none it starts as ambient air (`gas_density = ComputeAmbientDensity`, no
   fuel, no smoke, `background_pressure = atmosphere_pressure`). Its mass goes to `gas_mass_from_geometry`.
   Beside a moving liquid surface the gas a receding surface leaves behind is then the gas next to it.
3. **Sources (`ApplyGasSources`; PORTED, coverage FIXED).** Each `GasSource` is sampled with
   `SampleGasSource` at `step_start_time + dt / 2`. A source covers a cell by `coverage` = the share of the
   cell's eight cell-eighth centres inside the box, low sides included, high sides not (the branch tests the
   cell centre only, `gas_frame_3d.cc:55-58`, which is wrong for a big cell half inside).
   `fuel_density += dt * coverage * fuel_rate`; the cell's `heat_rate` (W/m3) `+= coverage * heat_rate`.
4. **Burn (`ApplyBurn`, PORTED from `gas_substep_3d.cc:121-126`, `gas_frame_3d.cc:131-134`).** Per gas cell,
   with `T` at the start of the step: if `T >= ignition_temperature`, `fuel_burned = -fuel_density *
   expm1(-burn_rate * dt)`: the exact solution of `d fuel / dt = -burn_rate * fuel`, so it never exceeds the
   fuel and does not depend on how a time span is cut into steps. `heat = heat_per_fuel_mass * fuel_burned`,
   `smoke_made = smoke_per_fuel_mass * fuel_burned`, `flame = fuel_burned / dt`, cell `heat_rate += heat / dt`.
5. **Gravity, wind and lift (`ApplyGasLift`; formula PORTED from `gas_substep_3d.cc:141-158`, the gravity
   take-back NEW).** On a copy of the gas set's `velocity`, for each face piece with gas `open_area > 0`, in
   this order: on Z pieces `velocity += step_duration * gravity` (the same product `ApplyForces` subtracted:
   in still ambient air the two cancel exactly and the velocity stays 0.0); `velocity += dt *
   wind_acceleration[axis]` with `wind_acceleration = SampleGasWind(wind, step_start_time + dt / 2)` (the
   branch's draft); on Z pieces `velocity += dt * lift`, `lift = g * ((T - T_ambient) / T_ambient -
   smoke_density / gas_density)`, computed from densities with `ComputeLiftFromDensities` (plan-decisions
   "Gas": the temperature round trip is not exact, `ComputeTemperature(law, atm, ComputeAmbientDensity(law))` is
   293.15000000000003, and would push still air). Values at the face are linear between the two cell centres,
   `(low * d_high + high * d_low) / (d_low + d_high)` with `d` half the width of each cell (the branch's
   half-and-half mean when sizes are equal; ambient values beyond an open edge, as on the branch), so no grid
   weights are needed. NEW (A10): on each piece closed by the liquid (`liquid_was_solved`) the copy takes
   `liquid_velocity_after_pressure`; pressure leaves unconnected pieces at `velocity_before_pressure`, so the
   gas set's velocity there after the solve is the liquid surface's.
6. **Cell density for the equations (PORTED).** `mixture_density = gas_density` (`open_volume` is the gas
   volume). FIXED (A8): liquid spray in a gas cell is not counted. It belongs to the set holding the liquid,
   moves with that set's velocity and shares no face sum with markers, so the gas neither carries nor drags
   it; the first version of this design added spray mass here, which was correct only while both shared one
   grid velocity.
7. **Equations (PORTED idea).** `pressure::BuildPressureEquations` on the gas snapshot with
   `SealedBlobCheck::kWhenRatesApplied`; its blobs are the gas blobs. The branch does the same: it builds
   the pressure topology once with a zero right-hand side to learn the components (`:75-100`). Built twice
   per step (A18): once in `ApplyGasStepStart` on the step-start geometry with zero rates, to list the sealed
   blobs for the liquid (each with `gas_volume`, equalised `p0` computed but not applied, `heat_volume_rate =
   ((gamma - 1) / gamma) * sum V q / p0`, `boundary_volume_rate` from solid inflow through pieces that are not
   liquid connections; listed when it touches liquid across open area and is not held), and once here, giving
   the same blobs.
8. **Background pressure per blob (`ComputeGasAddedVolumeRate`).** Cells are grouped by `ComputeBlobOfCell`;
   each blob's sums use the per-blob rule of section 3 (block partials in ascending cell place, then
   `math::TreeSum` over the blob's blocks; A15, both modes; replaces "one thread over its cells in ascending
   index").
   - *Equalise (NEW, fixes defect 5).* Open blob (`kHasReferencePressure`): `p0 = atmosphere_pressure`.
     Sealed blob: `p0 = (sum of V_i * p_i^(1/gamma) / sum of V_i)^gamma` over the cells' stored pressures
     (the additive amount of the air pocket law). Each cell's three densities are multiplied by
     `(p0 / p_i)^(1/gamma)` (air let in or out without heat); the factor is exactly 1 when nothing changed,
     which is always the case on the branch. The mass this adds goes to `gas_mass_from_pressure_equalising`.
   - *Rate (PORTED, the moving-boundary term NEW).* Sealed blob: `mean_heat_rate = sum of V_i q_i / sum of
     V_i`; `volume_rate = -incompatible_volume_rate` (with no added volume the blob's right-hand sides sum
     to the net inflow its moving boundary forces: solids and, beside liquid, the liquid surface; zero on the
     branch, where nothing moves); `p0_rate = (gamma - 1) * mean_heat_rate - gamma * p0 * volume_rate / V_sum`
     (old paper `:285`). Open blob: `mean_heat_rate = 0`, `p0_rate = 0`.
   - *Held at atmosphere (NEW, A10).* A sealed blob with `touches_liquid` whose equivalent radius
     `cbrt(3 V_sum / (4 pi))` is below `min_sealed_radius_in_cells` widths of its widest cell is
     `held_at_atmosphere`: its rates are those of a sealed blob (so its equations stay solvable), but at the
     end of the step (after step 11) each of its cells' three densities is multiplied by
     `(atmosphere_pressure / (p0 + dt * p0_rate))^(1/gamma)` and `background_pressure = atmosphere_pressure`;
     the mass change goes to `gas_mass_from_pressure_equalising`. This is pressure's rule for an unresolved air
     pocket (at atmosphere): a bubble too small to resolve neither stiffens the liquid nor heads out of range.
   - *Coupled blob (NEW, A18).* A gas blob whose `first_cell_index` equals an entry of `sealed_gas_blobs` was
     solved with the liquid: its end `background_pressure = atmosphere_pressure + sealed_gas_pressure[j]` and
     `background_pressure_rate = (that - p0) / dt`, so the two solves agree; it is never held; the range check
     applies. Step 9 keeps its formula with the gas equations' `volume_rate`; the cell rates still sum to it, so
     the gas solve stays solvable. The formula's rate agrees with the liquid's within `heat_ratio * p0 *
     max_volume_error_rate`.
9. **Added volume (PORTED from `gas_thermodynamics.cc:73-80`).** Per cell, `<eq:gas-added-volume>`:
   `added_volume_rate = V * ((gamma - 1) / gamma) * ((q - mean_heat_rate) / p0) + V * volume_rate / V_sum`,
   with `p0` at the start of the step and the mean subtracted before scaling, so a uniformly heated sealed
   blob gets exactly 0.0. For a sealed blob these sum to `volume_rate`, so the blob can absorb them;
   `pressure::ApplyAddedVolumeRate` (called right after) then performs the sealed-blob check. `background_pressure
   = p0 + dt * p0_rate` (a coupled blob: step 8's value) is written to every cell of the blob (not for a held
   blob); outside `(0, max_background_pressure]`: `kBackgroundPressureOutOfRange`.
10. **Solve.** `pressure::SolvePressure(equations, settings.pressure_solve, cancel_requested,
    SolvedArrays{amounts_try pressure, outputs.velocity_after_pressure}, pressure_buffers)` (the branch's
    `project` callback). The result is the final gas-set face velocity of the step: pieces closed by the
    liquid come back with the liquid's velocity (step 5). A `kCancelled` failure is `GasError::kCancelled`;
    any other is `kPressureSolveFailed`. In the batched layout this is `SolvePressure` on the gas
    `PressureBuffers` in that layout (A15).
11. **Flow of amounts (`ApplyGasFlow`; PORTED from `gas_transport.cc:63-119`, the divisor FIXED).** Per face
    piece with gas `open_area > 0`: `volume_flow = open_area * velocity_after_pressure` (m3/s along +axis).
    The giving cell is the one the flow leaves; at an open edge with inflow the giver is ambient air
    (`ComputeGasDensity(p0 of the cell, ambient_temperature)`, no fuel, no smoke). Pass one, per cell:
    `outflow` and `inflow` = sums of leaving and entering flows (ascending face index); `dt * outflow > V`
    is `kStepTooLongForGasFlow`, as on the branch. Pass two, per cell and per amount `a` in (gas, fuel,
    smoke), terms in ascending face index:
    `mass_end = (V - dt * outflow) * a + dt * sum over entering pieces of (flow * a of the giver)`;
    `volume_end = max(V + dt * (inflow - outflow + added_volume_rate), min_gas_volume_share * cell volume)`;
    `a_end = mass_end / volume_end`, and `gas_volume = volume_end`. With an exact solve and nothing moving,
    `volume_end = V` and this is the branch's update; a liquid surface or solid moving in by `s` m3/s gives
    `volume_end = V - dt * s`. Dividing by the volume the flows imply, not by `V`, keeps a uniform state
    exactly uniform whatever the solve left as volume error; that error becomes `gas_mass_from_volume_error`
    in the report. No amount can go negative and nothing is clipped.
12. **Results.** `temperature` of every gas cell is checked against `min_temperature` and `max_temperature`
    (`kTemperatureOutOfRange`, first cell in index order; the branch's "carrier and enthalpy must stay
    positive"). `outputs.was_solved`, `outputs.cell_was_solved` = `pressure::ComputeWasSolved` on the gas
    equations, plus 1 on every piece closed by the liquid (its value is the liquid's, not a marker's).
    `step_limit` = `kGasFlowShareLimit * min over cells of V / outflow`; `+infinity` when no cell has outflow
    (A24: the sealed blob term and `ComputeSealedGasStepLimit` are removed).
    Balance, checked in tests: `gas_mass_after = gas_mass_before - through_open_sides +
    from_pressure_equalising + from_geometry - left_in_closed_cells`. `gas_mass_from_volume_error` is reported
    but is not a term of the balance (plan-decisions "Gas": the mass it names returns as `gas_mass_from_geometry`
    the next step, and would be counted twice). The branch's `outward_rate` is `..._through_open_sides / dt`.

### 5.4 Invariants and order of sums

A translated temperature field creates no added volume (only heat, `p0_rate` and a moving boundary do). Still
ambient air stays still to the last bit. A uniform state stays uniform across a size step. Every per-cell sum
runs over that cell's face pieces in ascending `FacePlaceIndex`, every per-blob sum per cell block over the
blob's cells of that block in ascending index then `math::TreeSum` over the blob's block partials in block
order (A15, both modes), every scene total as one partial per cell block in the grid's fixed block order followed
by `math::TreeSum` over the block partials; sources apply in list order. The first failure is the one of the
lowest step number, then the lowest cell index. (The branch sums face by face into both cells, in face order;
the per-cell gather gives another rounding, hence the 1e-12 of section 7.)

### 5.5 Gas beside liquid (NEW, A10; the branch has none of it)

One grid, two material sets with their own face velocities, two solves per step: the liquid first (holding
the sealed gas blobs that touch it, A18), then the gas. They meet at the liquid surface, both ways:

- *The gas sees the liquid* as a moving boundary (step 1): a face piece the liquid solve corrected is closed to
  gas and pushes it with the liquid's velocity after this step's solve; the gas set's velocity there is the
  liquid's (step 5). A sealed gas blob squeezed or widened by the liquid changes its background pressure as
  `p V^gamma = constant` through `volume_rate` (step 8); heat adds to it.
- *The liquid sees the gas* in place of atmosphere where they touch: `ComputeGasSeenByLiquid` gives, for every
  cell that holds tracked gas, `gas_pressure_above_atmosphere` = background pressure minus atmosphere plus the
  gas solve's pressure, committed at the end of the previous step; pressure uses it as the air end's pressure
  (pressure 5 step 13). The liquid reads it one step late for open blobs, held blobs and cells that touch gas
  but are not gas cells: there the liquid dominates and the lag is harmless (`literature-coupling.md` 2). An
  unresolved sealed blob is held at atmosphere (step 8), like an unresolved air pocket.
- *A sealed blob that touches liquid is solved with the liquid (A18).* Lagging an enclosed bubble's pressure is
  unstable (Aanjaneya 2013; Patkar et al. 2013), so `ApplyGasStepStart` lists each resolved sealed blob that
  touches liquid as a `pressure::SealedGasBlob`, and the liquid's `SolvePressure` makes it one unknown with
  squeeze allowance `V / (heat_ratio p0 dt)`, its heat as `heat_volume_rate` and its solid inflow as
  `boundary_volume_rate` (pressure plan Task 44); its solved `sealed_gas_pressure` is the liquid's air-end
  pressure in the same step and sets the blob's end background pressure in `SolveGas` (step 8, coupled blob).
  Beside a viscous liquid the same holds (A24): `SolveViscosityAndPressure` carries the blob as one more unknown
  of its outer loop and returns its `sealed_gas_pressure`. No sealed step limit exists. *The lag that remains:*
  open gas (also above a viscous free surface), gas read across a sliver cell, cells that touch gas but are not
  gas cells, and held blobs are read one step late through `gas_pressure_above_atmosphere` and
  `holds_tracked_gas` (air-end precedence unchanged); no sealed blob that touches liquid is.
- *Air pockets versus tracked gas: one cell is never both.* A cell that held gas at the last commit, or that
  touches one and holds no committed gas (a cell the surface just uncovered, or a thin sliver of air), has
  `holds_tracked_gas = 1` and belongs to no air pocket (pressure 5 step 9). Pressure's air pockets are left to
  air cells with no gas near them: in a gas scene every air cell with enough volume becomes a gas cell in the
  same step, so a pocket lasts at most one step.
- *Markers are kept out of the liquid's sums* by the material sets (5.2). A marker carried into liquid is
  removed at the next `AddNewParticles` (`kMarkerNotNeeded`).
- *Liquid entering gas cells* keeps the gas (step 2 (b)): a cell the liquid fills hands its gas to the gas
  next to it. *Liquid leaving cells* is followed by the gas beside them (step 2 (c)).

Not coupled, by design and not excluded: gas does not drag liquid spray (5.3 step 6), and gas does not push
a free solid (the solid moves gas through `solid_volume_rate`). No capability exclusion remains for gas beside
liquid: every combination below runs. Foam bubbles of the surface, foam and spray module are unrelated to gas
cells.

### 5.6 Capability exclusions and scene rules (`CheckGasCanRun`)

- `kMoreThanOneGas`: two gas materials. Lifting test, when wanted: "Two gases mix" (two gas materials of
  different smoke in one closed box, total mass of each conserved to `1e-11` relative). Not planned.
- `kGasLawDisagrees` (a scene rule, not an exclusion): the gas ingredient's rest density differs from
  `ComputeAmbientDensity(law)`, or `AirPocketSettings::atmosphere_pressure`, `heat_ratio` or
  `min_radius_in_cells` differ from `law.atmosphere_pressure`, `law.heat_ratio` or
  `settings.min_sealed_radius_in_cells` (exact comparison: all come from one scene). A sealed gas blob and an
  air pocket must follow the same law, because the liquid reads either.

Not excluded: liquid (the one liquid material set, viscous or not, with or without a pressure law), snow, cut,
driven and free solids (they reach gas only through `SolidCoverage`), sloped solids, mixed cell sizes, a batched run
(`ApplyGasStepStart` and `SolveGas` run on the whole grid or with their scratch in pieces, A15; the markers pass
in batches like any particles).

### 5.7 Every material

Liquids and mixtures: the boundary of 5.5, both ways. Snow, springy solids, cut solids, moving solids: solid
to gas. Gas: this module. Nothing is carried per material except the gas material's set and its ingredient
index (checked in 5.6).

## 6. CPU and GPU

**Shared formula headers:** `gas_law.h` (all formulas of steps 4, 5, 8, 9),
`gas_geometry_math.h` (one cell's and one face piece's gas geometry from the liquid's), `gas_flow_math.h` (one
cell's end amounts from its face pieces), `gas_source_math.h` (coverage of a cell by a box),
`gas_marker_math.h`.

**CPU.** Loops over the cells, then the face pieces, of the block range; each destination value is written by
one iteration, so loops may be split across threads without changing a result.

**GPU arrays** (in `GasAmounts`, `GasAmountsTry`, `GasBuffers`; all stay on the device through the step,
which fixes defect 3). Every cell and face array is built for `max_cell_blocks` with `BuildCellValues` /
`BuildFaceValues` and bound to each grid version with `grid::ApplyGridVersion`; nothing is allocated per step.

| Array | Element | Count | Kept for |
|---|---|---|---|
| `gas_volume`, three densities, `background_pressure`, `flame`, `pressure` | `double` | 7 per cell place, committed and try | whole run, 2 copies |
| `gas_pressure_above_atmosphere`, `holds_tracked_gas` (NEW, `GasSeenByLiquid`) | `double`, `uint8` | cell places | one step try |
| gas `CellGeometry`, `FacePieceGeometry` | pressure's structs | cell places, face places | one step try |
| `heat_rate`, `added_volume_rate`, `bulk_stiffness` (+infinity), `carried_pressure`, `outflow`, `inflow` | `double` | 6 per cell place | one step try |
| `touches_liquid` (NEW) | `uint8` | cell places | one step try |
| `sealed_blob_of_cell` (NEW, A18) | `uint32` | cell places | one step try |
| `velocity_before_pressure` (after gravity, wind and lift) | `double` | face places | one step try |
| `blob_of_cell`, cells grouped by blob | `uint32`, `uint32` x 2 | cell places | one step try |
| `GasBlob`, per-blob sums (block partials) | struct, 4 `double` | `max_gas_blobs` | one step try |
| `pressure::SealedGasBlob` list (NEW, A18) | struct | `max_gas_blobs` | one step try |
| step-start changes (NEW, A18) | `GasAmountChanges` | 1 | one step try |
| sampled `GasSourceRates` | struct | `max_sources` | one step try |
| marker changes (`where = kCpu` in every layout) | `particles::NewParticles` arrays, `size_t` | given limit | one call |
| marker changes on the processor (GPU only, `where = kRunOn`; the lists before their read-back) | the same | given limit | one call |

The gas set's `velocity_after_pressure`, `was_solved` and `cell_was_solved` are the step's try arrays
(`SolvedGasArrays`), counted by the step with the per-set rows. `BuildGasMemoryEstimate` returns the rows above;
the second `pressure::PressureBuffers` is counted by the step with `pressure::BuildPressureMemoryEstimate`. About
285 B per cell place of gas arrays, plus pressure's. Sizes are `sizeof` of the real types (plan-decisions).

**Division of work.** One GPU function per step of 5.3: per cell (2, 3, 4, 6, 9, 11 pass one, 11 pass two, 12)
or per face piece (1, 5); `ComputeGasSeenByLiquid` per cell. Every write is a gather: a cell reads its own face
pieces and the cells across them; no thread adds into another cell, so there is no atomic. Step 2 (b) is a
gather too: a receiver reads the closed cells that touch it and take it as their lowest-index receiver.
Per-blob sums: block partials per blob, then one `math::TreeSum` per blob over its block partials (A15).
**Returned to the CPU:** `GasBlob` list, `GasAmountChanges`, `step_limit`, the failure;
fields only through `ReadGasSnapshot` and `ReadGasAmountsSnapshot`.

**Block ranges and border.** Every function takes `CellBlockRange`. Border cells are the cells touching the
range (`BorderWidth::kTouchingCells`): steps 2, 5 and 11 and `ComputeGasSeenByLiquid` read their amounts and
geometry. `ApplyGasStepStart` and `SolveGas` always take every block of the grid, also in a batched run;
`FindMarkerChanges`, `FindCellsToKeepActive` and `ComputeGasSeenByLiquid` may run per range.

**Batched solve (NEW, A15; `plan-decisions.md` "Batched solves (A15)").** With `GasBufferSizes::max_loaded_blocks
> 0` the buffers hold a `pressure::SolveBatches` (`BuildSolveBatches(run_on, max_cells / 64, max_loaded_blocks,
{kTouchingCells})`). Rule: state and geometry stay whole on the run's processor, solver scratch goes to CPU
memory. Kept `where = kRunOn`: the committed and try amounts, the gas `CellGeometry` and `FacePieceGeometry`,
`blob_of_cell`, the blob lists and every block-counted or given-count row. Moved to `kCpu` at the limits (and
`kRunOn` at `max_loaded_blocks` for the loaded piece): every other row counted per cell or face place
(`heat_rate`, `added_volume_rate`, `bulk_stiffness`, `carried_pressure`, `outflow`, `inflow`, `touches_liquid`,
`velocity_before_pressure`, `sealed_blob_of_cell`). The `MarkerChanges` rows are `kCpu` in every layout, one
piece included, and are not in this list (plan decision, 2026-11-14, buffers in batched runs; replaces "the
`MarkerChanges` arrays, which stage `AddNewParticles` uses on the CPU in a batched run" of the A15 update). A step of 5.3 that touches only rows on the processor runs whole, as in one
piece; a step that touches a `kCpu` row runs as one sweep through the loaded pieces (3 and 4 fused, 5 and 6
fused, 9, the two passes of 11 as two sweeps); per-blob totals of steps 7 and 8 and of the step start by the
per-blob rule; labels by label propagation in each piece then `pressure::JoinLabelsAcrossBatches` (equal to one
piece's); step 10 is `SolvePressure` on the gas `PressureBuffers` in the batched layout. Each sweep keeps the
smallest (step, cell) failure key, so the failure equals one piece's; the cancel flag is read before every batch
load. About 22 kB per full block stays on the GPU for a gas scene.

**Buffers in batched runs (plan decision, 2026-11-14; owner principle A17; the full inventory is the step plan's
Appendix A).** A whole try array lives on the run's processor when a part at `kWholeGridOnProcessor` reads or
writes it; else in CPU memory when a `kWholeStateOnCpu` stage reads or writes it, the commit keeps it or a later
pass reads it; else it has only its loaded instance. For gas: the amounts, geometry and every solver row named
above stay as the batched solve rule says; stage 1 (`AddNewParticles`) and stage 2 run on the CPU, so the step
keeps CPU mirrors of the committed `gas_volume` (8 B per cell place) and gas velocity (8 B per face place) and
of `keep_active` (1 B per cell place), each one counted copy, and calls the `MarkerGridInputs` form;
`FindCellsToKeepActive` runs on the amounts' processor. Copies gas makes itself: the GPU form of
`FindMarkerChanges` reads its two lists back (`MarkerChanges::copy_count`, `bytes_copied`); the solve's copies
go through `pressure::SolveBatches` only.

**Tolerance, CPU against GPU.** Counts, classes, blob labels: equal. One application of steps 1 to 9:
`|a - b| <= 1e-12 * scale + 1e-300`, `scale` the largest magnitude of the array. Amounts after `SolveGas`:
`<= (1e-9 + 12 * max_volume_error_rate * step_duration) * scale + 1e-300` (twice pressure's velocity
tolerance through six sides). Velocity: pressure's tolerance. **Batched against one piece** (A15, same
processor): the same bounds (velocity pressure's batched bound); counts, labels, classes and blob lists equal.
Batched against batched with another cut, and a repeat: identical bytes.

## 7. Tests

Analytic and conservation, CPU, each on uniform cells and on two size levels unless stated. Rows marked
(branch) restate a test of the branch with its numbers, so the two codes are pinned to the same values.

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `ComputeTemperature, RoundTrip` | Density from 300 K, 1500 K at 1 and 2 atm | Same temperature back; equals `3.5 p0 / rho / (3.5 * 287.05)` | `1e-14` relative |
| `ComputeLift, SignAndSize` (branch: `test_ideal_gas_density_and_residual_buoyancy`) | Hot clean gas; ambient gas with smoke | `g * (T / T_ambient - 1)`; `-g * smoke / gas_density` | `1e-14` relative |
| `ComputeBurn, StepSplit` (branch: `test_burn_conserves_fuel_and_adds_soot_and_enthalpy`) | Fuel 0.02 kg/m3 above ignition, one step against ten at fixed `T`; yield 0.1 | Fuel left `0.02 exp(-rate t)`; smoke `0.1 * burned`; `flame * dt = burned`; exact zero below ignition | `1e-14` absolute |
| `ComputeGasAddedVolumeRate, TwoSealedBlobs` (branch: `BalancesClosedHeatByVolumeAndKeepsComponentsIndependent`) | The branch's four cells, volumes and heat rates, `dt` as there | Blob pressures 100010 Pa and 101040 Pa; `p0_rate` 100 and 400 Pa/s; fourth cell's rate per volume `400 / (1.4 * 101325)` 1/s; each blob's rates sum to 0 | `1e-9` Pa; `1e-12`; `1e-16` |
| `ComputeGasAddedVolumeRate, CoupledBlob_EndPressureFromTheLiquid` (NEW, A18; plan Task 16) | `TwoSealedBlobs`' 3 m3 blob given as coupled with `sealed_gas_pressure` 50 Pa | End background 101375 Pa; rate `50 / dt`; the formula's rate agrees; the other blobs as before | exact; exact; `heat_ratio * p0 * max_volume_error_rate`; bytes |
| `ComputeLiftFromDensities` (NEW, plan Task 4) | The `ComputeLift` cases and 1000 values; ambient air at `p0` 202650 Pa | Equals `ComputeLift`; ambient gives `0.0` | `1e-14` relative; exact |
| `ApplyGasStepStart` (NEW, A18; plan Task 18) | Closed box 0.2 x 0.2 x 0.4 m, w 0.05, water to 0.2 m, committed gas above at 101325 Pa with gas pressure 3 Pa, 35 kW/m3 in one gas cell; the same with an open top, with one gas cell under water; a driven side of the gas moving in at 0.1 m/s over 0.04 m2 (the `kReadOneStepLate` case is removed, A24) | One blob: `gas_volume` 0.008, background 101325, `heat_volume_rate (0.4 / 1.4) * 35000 * 1.25e-4 / 101325`, boundary 0, `sealed_blob_of_cell` 0 in the 64 gas cells, `pressure_above_background` 3; no blob open or held; boundary +0.004 m3/s; `SolveGas` step 7 gives the same blobs; `SolveGas` without a step start is `kBadInput` | `1e-14` relative; exact |
| Sealed cell heating (branch: `FixedCellHeatingUsesProjectedFluxOrSealedPressureWork`, sealed part) | One closed 1 m3 cell, 35 kW/m3, 0.1 s | `background_pressure` up by `0.4 * 35000 * 0.1` = 1400 Pa; `gas_density` unchanged; `T` up by the same ratio; `step_limit` is `+infinity` (no outflow, no sealed term, A24) | `1e-8` Pa; exact; `1e-12` relative; exact |
| Open cell heating (same branch test, open part) | Same cell with one open side | Gas volume through the side = `0.4 * dt * Q / (1.4 * p0)`; `gas_density = ambient * (1 - expansion)` | solve limit (branch: 25 %); `1e-12` relative |
| Open heating, many cells (branch doc: 7.0494519e-5 m3) | Box with open top, heat in one cell; and the reference's 4 x 3 case made one cell deep | `gas_volume_through_open_sides = (gamma - 1) E / (gamma p0)` | `max_volume_error_rate * V * t + 1e-12` relative |
| Two sealed boxes (branch: `test_sealed_components_heat_independently`) | Two closed pairs of cells sharing a solid, 20 kW/m3 in one cell of the first, 0.1 s | +400 Pa in the heated pair; exactly 0 Pa change and zero velocity in the other | `1e-12` relative (branch: 1 %); exact |
| `ApplyGasFlow, TwoCells` (branch: `BudgetsIncludeSharedInteriorFluxAndSignedExteriorFlux`) | The branch's two cells and flows; added volume -0.1 m3/s per cell so `volume_end = V` | Fuel 1.7 and 1.1 (the branch's 2.2 adds its source after the flow; here sources come in step 3, plan-decisions "Gas"); gas density 1.05 and 1.975; fuel through open sides 0.2 kg/s times `dt` | `1e-14` |
| `ApplyGasFlow, TooMuchOutflow` (branch: `RejectsCombinedOutflowThatExceedsTheDonorVolume`) | Outflow of 1.1 cell volumes in one step | `kStepTooLongForGasFlow`, that cell reported | exact |
| Still air stays still | Ambient box, gravity on, 100 steps | Every face velocity is `0.0`; amounts unchanged | exact bytes |
| Translation adds no volume | Open box, uniform 1 m/s along X, a Gaussian temperature bump, no heat | `added_volume_rate` is `0.0` everywhere; no value leaves the initial min..max; mass balance holds | exact; exact; `1e-12` relative |
| Uniform wind (branch: `FrameRuntimeAppliesAnimatedDraftToOpenFaces`, `ApicPreservesUniformDraftAndAdvectsMarker`) | Open box, wind keys ramping so that the speed reaches 0.1 m/s | Every face and marker at 0.1 m/s; `gas_density` stays ambient; a marker moves by `dt * velocity` | `1e-12` relative |
| Uniform stays uniform | Uniform 800 K moving across a size step, 200 steps | Density equal in all cells | `1e-12` relative |
| Squeezed sealed blob (NEW physics) | Closed box, one side a driven solid moving in 10 % of the length in 100 steps | `p0 * V^gamma` constant; error halves when the step halves | 0.2 %; ratio 2 within 20 % |
| Blobs join (NEW physics) | Two sealed blobs at 1.0 and 1.2 atm, the solid between removed | `p0 = (sum V p^(1/gamma) / sum V)^gamma`; mass change equals `gas_mass_from_pressure_equalising` | `1e-13` relative |
| Mass balance (branch: `test_scalar_budgets_include_sources_and_signed_boundary_flux`) | Open box with a source, burn, wind, 200 steps | The balance of step 12 for gas (volume-error term not counted), and the like for fuel (plus sources, minus burned) and smoke (plus made) | `1e-11` relative |
| Source coverage and keys (branch: `FrameRuntimeSamplesKeyframedHeatingAtSubstepMidpoints`) | Box source over 2.5 cells; the branch's keyed case | Rates scale with coverage; fuel 0.04 kg/m3 and +80 Pa as on the branch | `1e-13` relative; `1e-8` Pa |
| `SampleGasSource, SampleGasWind` (NEW) | Three keys; times before, between, on and after them | Linear between keys, end values held, a key time gives that key's value; `CheckGasSources` rejects unordered keys and a box with `low >= high` | exact; exact |
| Marker upkeep (branch: `FrameRuntimeReplenishesApicMarkersAfterOpenOutflow`) | Cells with 0, 3, 8, 20 markers; steady 0.5 m/s through an open box | 8 in each after the changes; removed ids are the highest; new markers carry 0.5 m/s | exact; `1e-12` |
| Rising plume keeps its symmetry | Heat patch centred on the floor, open top, 100 steps | Fields mirror in X and in Y; plume top rises | `1e-9` relative |
| Errors | Two cells out of temperature range | The lower cell index is reported; nothing is readable | exact |
| `ToName, EveryError_LowerSnakeCase` (NEW) | every `GasError` | the enumerator's name without `k`, lower snake case; all distinct | exact |
| Cancel (NEW) | `cancel_requested` set before `SolveGas` | `kCancelled` with the pressure failure; the committed amounts unchanged after `RejectTry` | exact |
| `CheckGasCanRun` (NEW) | Two gas materials; a gas ingredient at another density; `AirPocketSettings::heat_ratio` 1.3; liquid and gas in a batched run | `kMoreThanOneGas`; `kGasLawDisagrees` twice; success | exact |
| `ToGasSnapshot, EqualsReadGasSnapshot` (NEW) | Fire scene after 20 steps, some cells with `gas_volume = 0` | `ToGasSnapshot(ReadGasAmountsSnapshot(amounts, grid), law, grid.frame())` equals `ReadGasSnapshot(amounts, grid, law)` field by field; cells without gas left out | exact |
| Snapshot round trip (NEW) | Fire scene, 20 steps, `ReadGasAmountsSnapshot` at step 10, `BuildGasAmountsFromSnapshot`, 10 more steps | The seven arrays and the velocity equal the uninterrupted run | exact bytes |
| Resize try (NEW) | Committed amounts, a resize that splits and merges cells, `BuildResizeTry`, remap, commit, then `BuildTry` | `BuildResizeTry` counts no removed cell; the next `BuildTry` counts none either; total gas mass equal before and after | exact; `1e-14` relative |
| **Gas beside liquid (A10)** | | | |
| Still gas changes no liquid byte (NEW) | Water pool in an open box, 8 markers per cell in the air and some markers placed in the water, ambient still gas | Liquid set's `CellsFromParticles`, mixture density, sensed volume ratio, distance to surface and velocity after pressure equal the same scene with no gas material; markers in water removed at the next `AddNewParticles` with `kMarkerNotNeeded` | exact bytes; exact count |
| `ComputeGasSeenByLiquid` (NEW) | Committed gas in some cells with known `background_pressure` and `pressure`; a cell with no gas touching two gas cells | `background_pressure - atmosphere + pressure` in each gas cell; the copy of the lower-index neighbour in the touching cell; 0 and 0 elsewhere | exact |
| One cell is never both (NEW) | Closed box, water in the lower half, gas above | Every cell with `holds_tracked_gas` has `pocket_of_cell = kNoAirPocket`; the gas blob is the only body of air | exact |
| Smoke over a moving water surface (NEW) | Open-top box 1 x 0.5 x 1 m, water 0.4 m deep moved by a driven paddle (waves of 5 cm), a smoke source 0.3 m above the still surface with `fuel_rate` 0 and smoke made by a 20 kW/m3 heat patch with smoke keys, 400 steps | No gas cell inside the liquid, no marker counted by the liquid set; on every liquid-closed piece the gas velocity equals the liquid's; smoke balance holds; liquid volume equals the same scene without gas | exact; exact; exact; `1e-11` relative; pressure's volume limit |
| Heated sealed pocket above liquid (NEW; amended A18, A24, plan Task 25) | Closed box 0.2 x 0.2 x 0.4 m, water to 0.2 m at rest, 35 kW/m3 in one gas cell (mean 546.875 W/m3), steps of 0.03 s, 16 steps (0.48 s), coupled (the lagged part is removed, A24) | `p0` rises by `0.4 * mean heat * t` = 105 Pa (the gas volume stays); each committed background equals atmosphere plus that step's liquid `sealed_gas_pressure`; liquid pressure `p0 - atmosphere + rho g (H - z)` in the same step; water at rest | `1e-8` Pa; exact; solve limit; pressure's velocity tolerance |
| Heated pocket pushes water (NEW; amended A18, plan Task 26) | Same box with a 0.05 m open tube from the water to the top, steps of 0.03 s | Water rises in the tube until `rho g dh` equals the pocket's rise in `p0`; gas `p0 * V^gamma` follows heat | 2 %; 0.5 % |
| `GasBesideLiquid, BreathingBubble_StableWhereLagGrows` (NEW, A18; amended A24; plan Task 26) | A resolved sealed gas sphere of radius 0.05 m (cells 0.0125 m) in an open-top water tank 0.3 x 0.3 x 0.4 m, centre 0.2 m deep, started at 1.02 atmosphere (Minnaert 412.6 1/s); steps of 0.01212 s (`omega dt` about 5), 30 steps; then the test-side lagged reference (A24, as pressure plan Task 50: the liquid is given the blob as tracked gas, no `SealedGasBlob` listed, the carried value as `gas_pressure_above_atmosphere`; the test passes `sealed_gas_pressure` to `SolveGas` from its own update `p0 (V_n / V_n+1)^heat_ratio - atmosphere`) | Coupled: the bubble's `gas_volume` deviation never grows from one step to the next. Lagged: it grows at least 5 times per step over steps 2 to 4, or a step fails | exact comparison |
| Liquid poured into a gas-filled box (NEW) | Closed box of ambient gas; a liquid source inside adds 20 % of the box volume in 2 s; then the same box with an open top | Closed: `p0 * V_gas^gamma` constant, `gas_mass_left_in_closed_cells = 0` (every closed cell handed over), gas mass balance holds. Open: `gas_volume_through_open_sides` equals the liquid volume added | 0.5 %; exact; `1e-11` relative; pressure's volume limit |
| Unresolved bubble (NEW) | One-cell sealed gas bubble under 0.3 m of water | `held_at_atmosphere`, no `kBackgroundPressureOutOfRange`, the mass change in `gas_mass_from_pressure_equalising`; liquid reads 0 Pa above atmosphere plus the gas solve's pressure | exact; exact; `1e-12` relative |
| Liquid lifts a sealed gas blob (NEW; amended A18, plan Task 26) | Closed box, lower half water raised by a driven floor at 0.02 m/s | Gas velocity at the surface equals the liquid's; gas `p0 * V^gamma` constant; the liquid surface pressure in step `n` equals the `p0` committed at the end of step `n` minus atmosphere plus the gas pressure of the last commit (the same step) | solve limit; 0.2 %; exact |
| Batched fire beside water (NEW; amended A15; run by the step plan, `TryStep, BatchedFireBesideWater_WithinStepTolerance`) | The small fire with a 0.2 m pool under the plume, in both batched modes and in one piece | Gas amounts and both sets' velocities within the step design 6.4 tolerances of one piece; identical bytes across cuts of one mode and on repeat | step 6.4; exact |
| **Batched solve (A15; plan Tasks 29, 30)** | Grids of at least 8 block layers, loaded 3, 5 and 7 times `CountBlocksInLargestLayer`; both processors the CPU, then GPU | | |
| `ApplyGasStepStart, InBatches_SameBlobsAsOnePiece` | The step-start sealed pocket at w 0.0125 (16 x 16 x 32) | Blob list and `sealed_blob_of_cell` | bytes |
| `SolveGas, SmallFireInBatches_WithinToleranceOfOnePiece` | `BuildSmallFireScene` at 0.03125 m cells, 20 loop steps | Seven arrays and the gas velocity | section 6 batched bounds |
| `SolveGas, SmallFireInBatchesThreeCuts_IdenticalBytes` | Same, three cuts | Identical | exact bytes |
| `GasBesideLiquid, HeatedSealedPocketInBatches_WithinTolerance` | The heated pocket at w 0.0125, coupled | Background pressure equals atmosphere plus the liquid's `sealed_gas_pressure`, in both modes | exact |
| `SolveGas, FailureInOneBatch_CommittedBytesKept` | A cell of the last batch made 6000 K | `kTemperatureOutOfRange` with that cell, as one piece; `RejectTry` leaves the committed bytes | exact |
| `SolveGas, CancelBetweenBatches_IsCancelled` | Flag set at the 4th load | `kCancelled` | exact |
| `BuildGasMemoryEstimate, BatchedLayout_MarkerChangesNotDuplicated` (plan decision, buffers in batched runs) | Batched layout | One `kCpu` row per list, none at `max_loaded_blocks` | exact |
| `ResetSolveCopyReport, BatchedGasBuffers_ReportZeroAfterReset` (reconcile) | After one batched `SolveGas` | Every field of `*ReadSolveCopyReport(gas_buffers)` is 0 | exact |
| **Marker lists in CPU memory (plan decision, 2026-11-14, buffers in batched runs; plan Tasks 10, 20)** | | | |
| `BuildGasMemoryEstimate, MarkerChangesInCpuMemoryInEveryLayout` | `RunOn::kGpu` (rows only, no GPU needed) and `max_loaded_blocks > 0` | Both list rows `where = kCpu`; the scratch row `kRunOn` only on `kGpu` | exact |
| `FindMarkerChanges, CpuInputsForm_EqualsAmountsForm` | The 0, 3, 8, 20 markers case through `MarkerGridInputs` (the amounts' `gas_volume` and a velocity copy) | Identical lists; copy fields 0 | exact |
| `FindMarkerChanges, CpuInputsFormWithArrayOfOtherGrid_IsWrongGridVersion` | An array bound to another grid | `kWrongGridVersion` | exact |
| `FindMarkerChanges, OnGpu_ListsInCpuMemoryAndCopiesCounted` (label `gpu`) | The same case on the GPU | Spans in CPU memory; `copy_count` == the list arrays read, `bytes_copied` == entries times their `sizeof` | exact |
| GPU forms (Task 30) and `no_gpu` | `ApplyGasStepStart, InBatchesOnGpu_SameBlobsAsOnePieceOnGpu`, `SolveGas, SmallFireInBatchesOnGpu_WithinToleranceOfOnePieceOnGpu`, `SolveGas, InBatchesOnGpuTwice_GivesIdenticalBytes`; `BuildGasBuffers, BatchedLayoutOnGpuWithoutAGpu_IsRunOnCallFailed` | As the CPU rows; `kRunOnCallFailed` | as above |
| Repeat run | Any case twice | Identical bytes | exact |

**Python reference: none.** The branch's `reference/src/fluidity_reference/gas.py` is 2D and holds four
closed forms; each of its five tests is a row above, checked directly in 3D. A reference would repeat those
forms and remove no uncertainty. **Lean: none** beyond pressure's `PressureEquations.lean` (a sealed blob can
be solved exactly when its right-hand sides sum to zero, which step 9 constructs).

**Against `main`:** nothing; `main` has no gas.

**Against the gas branch (programme design section 7, item 6).** The branch is built in its own worktree at
`a430dab` (CPU preset) by the implementation plan; nothing of it is copied. Its `fluidity simulate` then
`inspect --frame N` prints `[carrier, temperature, fuel, flame, soot]` per cell in (z, y, x) order. Every
comparison uses same-size cells, no solid, no liquid, the branch's fixed step count per frame, and rotates
Y-up to Z-up.

| Case | Branch input | Compared | Tolerance, and why |
|---|---|---|---|
| One step, markers at rest | A 4 x 4 x 6 box built with `CreateGasUniformDomain3D`, heat and fuel in two cells, zero initial velocity, one call of `AdvanceGasSubstep3D` (through a small test program in the branch worktree) | `gas_density`, `fuel_density`, `smoke_density`, `flame`, `background_pressure`, `added_volume_rate` per cell; `temperature` against `enthalpy / carrier / c_p` | `1e-9` relative. Markers do not enter (velocity is zero before lift); the difference is the pressure solve's limit and the order of sums |
| CLI case | The scene of `tools/native/tests/run_gas_cli.py:29-40` (2 x 2 x 2 cells, keyed fuel 0.1 kg/(m3 s) and heat 1500 W/m3), frames 0 to 2 | The five fields per cell from `inspect` | 1 % of each field's range. The marker rule differs (8 per cell against 1) |
| Fire in a draft | The packet written by `gas_authoring.scene_packet` for `tools/blender/verify_m7_fire_in_a_draft.py` (domain 1 x 1 x 1.5 m, source 0.25 m cube, keys at frames 0, 3, 8, 12, wind 0 to 0.85 m/s2, ignition 299 K, burn rate 7 1/s, 4 MJ/kg, yield 0.2, 13 frames, 4 steps each) | Per frame: total fuel, smoke and gas mass, heat released, gas volume through open sides, highest temperature (branch record: about 777 K at frame 12), count of cells with `flame > 0` (branch record: 46) | Totals 2 %; highest temperature 10 %; flame cells 25 %. A larger difference is explained by the marker rule, not tuned away |
| Rejected step | The heat pulse of `run_gas_cli.py:68-77` (1e12 W/m3) | Both reject; the committed frames before it are unchanged | exact outcome |

Supporting evidence only (programme design section 7): the analytic rows are the authority.

**GPU (label `gpu`, compiled here, run by the owner):** every analytic row within the tolerance of section 6
(with `ApplyGasStepStart, OnGpu_EqualsCpu`: blob list and `sealed_blob_of_cell` equal); the repeat-run test on
identical bytes for all seven arrays and the velocity; the fire scene for 10 steps; the smoke-over-water scene
for 10 steps; the batched GPU rows above.
**Without a GPU (labels `cpu`, `no_gpu`):** all shared formula headers, the per-blob grouping, and every
operation with a `_gpu.cu` (`BuildGasAmounts`, `BuildGasBuffers` in both layouts, `SolveGas`,
`ApplyGasStepStart`, `FindMarkerChanges`) returning `kRunOnCallFailed` on the GPU in the container.

**Tests the step plan adds** (they need `TryStep`; gas plan "Tests the step plan must add", amended A15 and
A18 and A24): `TryStep, GasHeatPulse_RejectedAndCommittedFramesKept`; `TryStep, BatchedFireBesideWater_WithinStepTolerance`
(was `_SameBytes`, A15; run in the cell resize and batches plan); `TryStep, GasStepLimitTakenIntoNextDuration`
(amended A24: the small fire's flow-share limit only); `TryStep, SealedPocketAboveWater_NoSealedStepLimit` (added
A18: above inviscid water `step_limit` has no sealed term, steps of `max_step_duration` commit, each committed
background pressure equals atmosphere plus that step's `sealed_gas_pressure`); `TryStep,
SealedPocketAboveSyrup_NoSealedStepLimit` (added A24: the same above syrup of 10 Pa s and 1400 kg/m3, and stage
`SolveViscosityAndPressure` runs); `TryStep, SparseFireKeepsItsActiveCells`.

Removed (A24): the row "Sealed step limit" (`GasBesideLiquid, SealedStepLimit`), the test
`ComputeSealedGasStepLimit, MatchesFormula`, the `kReadOneStepLate` cases of the step-start and heated-pocket rows.

**Fire reference scene (decision 8).** The branch's "Fire in a draft" (1 x 1 x 1.5 m, under 4096 cells) is the
comparison scene above. The sizing scene is larger. Z-up. Domain box x, y in [-0.8, 0.8] m, z in [0, 3.2] m;
`side_bottom` closed, the other five sides open air. Source box x, y in [-0.15, 0.15], z in [0, 0.05]:
`fuel_rate` 0.3 kg/(m3 s), `heat_rate` 240 kW/m3 for the first 0.5 s (pilot), then 0. Burn: ignition 400 K,
rate 7 1/s, 4 MJ/kg, smoke 0.2 (the branch's values but for ignition, which there lets fuel burn at room
temperature): about 5 kW, plume speed about 1.5 m/s. Wind acceleration 0.85 m/s2 along +X from t = 5 s. Cells
of 1.25 cm for x in [-0.3, 0.5], |y| <= 0.3; 2.5 cm out to x in [-0.5, 0.7], |y| <= 0.5; 5 cm beyond: about 0.97
million cells, 7.7 million markers (about 3.2 GB of particle arrays). 10 s, 240 frames. No liquid; the
smoke-over-water row above is the scene of gas beside liquid.

## 8. Method paper section

Labels: `<sec-gas>`, `<sec-gas-law>`, `<eq:gas-law>`, `<eq:gas-added-volume>`, `<eq:background-pressure>`,
`<eq:lift>`, `<eq:burn>`, `<sec-gas-flow>`, `<sec-gas-markers>`, `<sec-gas-beside-liquid>`; figures
`<fig-gas-two-pressures>`, `<fig-gas-in-step>`, `<fig-gas-flow>`, `<fig-gas-beside-liquid>`; procedures
`<proc-solve-gas>`, `<proc-apply-gas-flow>`.

Outline: (1) two pressures, and why the stored amount is the air (the derivation of 5.1); (2) heat makes
volume: the added volume rate and the background pressure of a sealed blob, blobs joining and opening;
(3) lift without gravity, and why still air stays still; (4) burn; (5) the flow of amounts through face
pieces and dividing by the implied volume; (6) markers carry velocity only, in their own material set;
(7) gas beside liquid: the liquid surface as a moving boundary of the gas; a sealed blob that touches liquid
solved inside the liquid's pressure solve (refer to pressure's `<sec-coupled-solve>`; cite Aanjaneya 2013 and
Patkar et al. 2013 for why a lagged bubble pressure is unstable), its end background pressure taken from that
solve, and `ApplyGasStepStart` giving the liquid this step's heat (A18); the same beside a viscous liquid, whose
joint solve carries the blob (A24); open gas (also above a viscous free surface), slivers, cells that touch gas
and held blobs read one step late; small bubbles held at atmosphere; why one cell is never both an air pocket and gas; one paragraph on the gas solve in
batches (what stays on the GPU, which steps become sweeps, per-blob totals by block partials; refer to
pressure's `<sec-solve-in-batches>`, A15); (8) what the old paper reserved and this one leaves out.

Figures: `<fig-gas-two-pressures>`: one open and one sealed blob, each with its background pressure as a
flat level and the solve's pressure as a small ripple on it, heat entering one cell and volume leaving
through the open side. `<fig-gas-in-step>`: the step's stages as a strip with the gas additions hung under
`AddNewParticles`, `ChooseActiveCells`, `FindLiquidSurface`, `BuildPressureEquations` (with
`ApplyGasStepStart`) and the new `SolveGas` stage after the liquid solve, and the 12 steps unfolded (1 to 4 at
the step start, 1 and 5 to 12 in `SolveGas`). `<fig-gas-flow>`: a big cell beside four small ones, arrows
through face pieces, the giving cell shaded. `<fig-gas-beside-liquid>`: a partly filled cell, the liquid part
pushing the gas with the liquid's velocity, the gas part open; beside it a sealed gas pocket above water whose
pressure is one unknown of the liquid's solve of the same step and sits on the water surface.

Pseudocode: `<proc-solve-gas>` (the 12 steps of 5.3), `<proc-apply-gas-flow>` (two passes of step 11).
Cite from the old list: [1] Jiang 2015 and [2] Ding 2020 (markers), [26] Kwatra 2009 (named as a different
model: pressure with a speed of sound), [11] Sancho 2024 and [29] Zhou 2024 (transport variants not used),
[21] Batty 2007 (solid areas, through pressure).

## 9. Vocabulary

Old names are the branch's (`origin/codex/m7-low-mach-gas`); VERIFIED means read there.

| Old name on the branch | New name | Source |
|---|---|---|
| `GasScalars::carrier` | `gas_density` | NEW; VERIFIED `gas_transport.h:13` |
| `GasScalars::fuel`, `soot` | `fuel_density`, `smoke_density` | NEW |
| `GasScalars::enthalpy`, `kGasSpecificHeat` | not stored; `ComputeTemperature` | NEW; VERIFIED `test_gas_substep_3d.cc:40` |
| `GasCell3D::absolute_pressure` | `background_pressure` | NEW |
| `GasCell3D::gauge_pressure` | `pressure` (of the gas solve) | DECIDED word |
| `GasCell3D::volume` | `gas_volume` | NEW |
| `GasCell3D`, `GasFrame3D::cells` | `GasAmounts`, `GasAmountsTry` | NEW |
| `GasThermodynamicComponent`, `component`, `open_to_ambient` | `GasBlob`, gas `blob`, `blob_class` | DECIDED word `blob` |
| `AdvanceGasHeating`, `target_divergence`, `enthalpy_pressure_work_rate` | `ComputeGasAddedVolumeRate`, `added_volume_rate`, `background_pressure_rate` | NEW; DECIDED; NEW |
| `kGasGamma` | `heat_ratio` | VERIFIED by reading `05-pressure-design.md` (`AirPocketSettings::heat_ratio`) |
| `kDryAirGasConstant` | `gas_constant` | NEW |
| `GasReaction3D`, `burn_rate`, `heat_per_fuel`, `soot_yield` | `BurnSettings`, `burn_rate`, `heat_per_fuel_mass`, `smoke_per_fuel_mass` | NEW |
| `burned_fuel`, `fuel_burn_rate` | `fuel_burned`, `flame` | NEW |
| `residual_gravity_factor`, buoyancy | `lift`, `ComputeLift`, `ApplyGasLift` | NEW |
| `GasDraftKey3D`, `draft`, `face_accelerations` | `GasWindKey`, `GasWind`, `wind_acceleration` | NEW |
| `GasBoxSource3D`, `GasSourceKey3D`, emitter, `GasCellSource3D::rates` | `GasSource`, `GasSourceKey`, `GasSourceRates`, source, `heat_rate` | NEW |
| keyframe sampling at the substep midpoint | `SampleGasSource`, `SampleGasWind`, `CheckGasSources` | NEW |
| `TransportGasScalars`, `volume_flux`, donor, receiver, `outgoing_volume_rate` | `ApplyGasFlow`, `volume_flow`, giving cell, `outflow`, `inflow` | NEW |
| `tail`, `head` | `cell_low`, `cell_high` | DECIDED |
| `outward_rate` | `..._through_open_sides` (kg per step) | NEW |
| `AdvanceGasSubstep3D`, `project` callback | `SolveGas` (function and stage), `pressure::SolvePressure` | NEW, verb DECIDED (A6: stage named after the function) |
| `GasApicMarker3D`, `weight`, `velocity_slopes` | a particle of the gas material: `mass` (a `sample_weight`), `fitted_velocity_change` | DECIDED |
| `AdvanceGasApicSubstep3D`, `AdvectMarker` | stages `ParticlesToGrid`, `GridToParticles`, `MoveParticles` for the gas material set | DECIDED |
| `RemapGasApicMarkers3D` | `FindMarkerChanges` with `AddParticles`, `RemoveParticles`, `RemoveReason::kMarkerNotNeeded` | NEW; DECIDED verbs |
| `GasUniformDomainOptions3D::open_boundary` | `contact::DomainBox::closed` | contact |
| `ambient`, `ambient_pressure` | ambient (`ambient_temperature`), `atmosphere_pressure` | NEW; pressure's word |
| `GasCheckpoint3D` (`FLGC`) | `GasAmountsSnapshot`, `ReadGasAmountsSnapshot`, `BuildGasAmountsFromSnapshot` | NEW |
| (none) | `FindCellsToKeepActive`, `GasAmountChanges`, `GasSnapshot`, `CheckGasCanRun`, `GasSceneSummary`, `kGasLawDisagrees` | NEW |
| (none, A10) | `GasSeenByLiquid`, `ComputeGasSeenByLiquid`, `SolvedGasArrays`, `touches_liquid`, `held_at_atmosphere`, `min_sealed_radius_in_cells`, `handed_over_cell_count` (`ComputeSealedGasStepLimit` removed, A24) | NEW |
| (none, A10) | `holds_tracked_gas`, `gas_pressure_above_atmosphere` | pressure's names |
| (none, batches) | `BuildResizeTry`, `GasArraysToRemap`, `ToArraysToRemap` | NEW |
| (none, A18) | `GasStepStartInputs`, `SealedGasForLiquid`, `ApplyGasStepStart`, `sealed_blob_of_cell`, `sealed_gas_blobs`, `sealed_gas_pressure` | NEW (plan-decisions) |
| (none, A18) | `SealedGasCoupling` (`kInsideLiquidSolve`, `kReadOneStepLate`), `GasStepStartInputs::sealed_gas_coupling`, `SolveGasInputs::sealed_gas_coupling`, `ComputeSealedGasStepLimit` | removed (A24) |
| (none, A15) | `GasBufferSizes::max_loaded_blocks`, `FindSolveBatches`, `ReadSolveCopyReport` | NEW (plan-decisions) |
| (none) | `gas::ResetSolveCopyReport(GasBuffers&)` | NEW (plan decision, reconcile 2026-11-14, item 3) |
| (none) | `MarkerGridInputs`, `MarkerChanges::copy_count`, `bytes_copied` | NEW (plan decision, 2026-11-14, buffers in batched runs) |
| (none, plan) | `ComputeLiftFromDensities`, `ComputeJoinedPressure`, `ComputeSourceCoverage`, `ComputeCellEighthCenter`, `carried_pressure` | NEW (plan-decisions "Gas") |
| absl status codes | `GasError`, `GasFailure`, `kNone`, `kRunOnCallFailed`, `kCancelled`, `ToMessage`, `ToName` | A2, A13; Q11 |

Words checked and not used: `soot`, `carrier` (naming review: never say, for the grid velocity), `enthalpy`,
`species`, `advect`, `divergence`, `component`, `draft`, `emitter` (naming review: never say `emit`),
`inventory`, `budget`.

## 10. Not ported

- **The separate gas runtime:** `gas_frame_3d.cc` (frame loop with equal steps), `gas_uniform_domain_3d.cc`
  (box domain and its 4096-cell limit), `cli/gas_execution.cc`. Gas runs through the one `TryStep`.
- **`gas_checkpoint_3d.cc`, `gas_checkpoint.fbs`, `gas_scene.fbs` (`FLGC`, `FLGS` v1):** the step and the I/O
  and CLI designs define the one saved state and scene; the schema v1 copies go with the rest of v1 (D13).
- **The enthalpy scalar and its pressure-work source** (`gas_transport.h:16`, `gas_substep_3d.cc:138`): 5.1.
- **`AdvectMarker`, `ExtendBoundaryVelocities`, `RemapGasApicMarkers3D`** (`gas_apic_3d.cc:44-140`,
  `246-318`): replaced by `contact::MoveParticles`, transfer's filled faces and `FindMarkerChanges`. The
  fallback "a face with no marker takes last step's velocity" (`:200`) is dropped: upkeep keeps at least four
  markers in every gas cell, and a face that still has none is counted.
- **`reference/src/fluidity_reference/gas.py`** and its tests: 2D, closed forms only (section 7).
- **`blender/fluidity/gas_authoring.py`, `gas_volume.py`, `tools/blender/verify_m7_*.py`:** read by the
  surface, foam and spray module and by the Blender add-on for the channel names and the scene; not this
  module's.
- **From the old paper only** (never on the branch): heat conduction, the explosion deposit as its own path,
  the clamped burn step, `J` on markers, the vortex-ring gate, the supersonic extension, flow-map and impulse
  transport. No array, enum value or setting is reserved for them.

## 11. Decisions

Earlier questions this design depended on, now decided: Q2 and Q11 (`GasFailure` in `std::expected`, `kNone`,
`kRunOnCallFailed`: owner answers A2 and A13); Q10 (`CommitTry`, `RejectTry`: recommended option (owner's
standing instruction, 2026-11-14)); Q16 (owner answer A5: viscosity runs the joint solve; `SolveGas` follows
whichever liquid solve ran, A6); Q26 (grid edges marked by `grid_edge_is_open_to_sky`: recommended option);
Q27 (one acceptance limit: recommended option); Q33 (shares 0 for gas come from materials: recommended
option); Q42 (gas markers may be split and merged: recommended option); Q52 (memory estimate entries: recommended option).
Nothing in `shared-interface.md` (c) for gas is left unprovided.

**1. Gas vocabulary (all new).** One table; the recommended word is in the second column.

| Concept (branch word) | Recommended | Alternatives | Preview |
|---|---|---|---|
| Air mass per volume (`carrier`) | `gas_density` | `air_density` | `amounts.gas_density()` |
| One pressure per connected gas (`absolute_pressure`) | `background_pressure` | `blob_pressure`, `rest_pressure` | `amounts.background_pressure()` |
| Upward push on light gas (`residual_gravity_factor`) | `lift` | `buoyancy`, `rise` | `ComputeLift(law, temperature, smoke_density, gas_density)` |
| `soot` | `smoke_density` | `soot_density` | `burn.smoke_per_fuel_mass` |
| Reaction | burn | `combustion`, `reaction` | `ApplyBurn`, `burn_rate`, `fuel_burned` |
| `heat_per_fuel`, `soot_yield` | `heat_per_fuel_mass`, `smoke_per_fuel_mass` | `fuel_heat`, `smoke_yield` | `burn.heat_per_fuel_mass` |
| `kGasGamma`, `kDryAirGasConstant` | `heat_ratio`, `gas_constant` | `heat_capacity_ratio` | `law.heat_ratio` |
| Outside air | ambient | `outside`, `surrounding` | `law.ambient_temperature` |
| A gas particle (`GasApicMarker3D`) | marker | `gas_particle`, `tracer` | `FindMarkerChanges` |
| Moving amounts between cells (`Transport`) | flow | `transport`, `carry` | `ApplyGasFlow`, `volume_flow`, `outflow` |
| Stored per-cell values (`GasCell3D`) | `GasAmounts` | `GasFields`, `GasState` (collides with Q1) | `BuildGasAmounts(grid, law, run_on)` |
| Totals report (`outward_rate`) | `GasAmountChanges` | `GasBalance` | `result.changes.fuel_mass_burned` |
| `fuel_burn_rate` shown as fire | `flame` (kg/(m3 s)) | `burning_rate` | `amounts.flame()` |
| The stage (`AdvanceGasSubstep3D`) | `SolveGas` | `ApplyGas` | `SolveGas(inputs, settings, amounts_try, ...)` |
| `draft` | `wind_acceleration` | `draft_acceleration` | `inputs.wind_acceleration` |
| emitter | `GasSource` | `FuelSource` | `source.fuel_rate` |

Decision: the recommended column (recommended option (owner's standing instruction, 2026-11-14)). Since the
I/O entry made sources and wind keyed, `wind_acceleration` is the sampled value of `GasWind`, and the rates of
`GasSource` live in its `GasSourceKey`s.

**2. Drop the enthalpy scalar?** a) store `gas_density`, derive `temperature` from `background_pressure`
(5.1): three scalars, no second copy of the pressure that can drift, same numbers as the branch wherever its
own identity `enthalpy = 3.5 p0` holds; b) port the branch's four scalars and its pressure-work source, and
report the drift. `ComputeTemperature(law, background_pressure, gas_density)` against
`enthalpy / gas_density / heat_capacity`. Rec: a.

Decision: a (recommended option (owner's standing instruction, 2026-11-14)).

**3. Gas beside liquid, and four changes to finished designs.** The branch has no liquid. Options: a) one-way
(5.5): liquid first, then gas with the liquid surface as a moving boundary; a heated pocket sealed by liquid
is counted, not felt; b) two-way through one solve with a `MaterialBoundary` between liquid and gas: a new
method; c) no gas and liquid in one scene. Rec: a. It needs, in designs already written:

```cpp
// transfer: leave gas markers out of the liquid's per-cell sums (else kGasWithLiquidNotSupported stays).
std::span<const std::uint8_t> material_is_counted;   // new field of CellsFromParticlesInputs; empty: all
// particles: a reason for removing a marker that is not needed.
enum class RemoveReason { kLeftDomain, kThroughOutlet, kIntoGridOnlyRegion, kMarkerNotNeeded };
// pressure: the three additions of section 4.     base: math::Expm1.
ComputeBlobOfCell(equations, blob_of_cell);  SealedBlobCheck::kWhenRatesApplied;  CellContent::kGas;
```

Without `SealedBlobCheck`, a sealed gas blob with a moving boundary fails at the build before gas can give
its rates. Without `kGas`, gas cells are called `kLiquid` (works, reads wrongly). Rec: all.

Decision: gas and liquid in one scene from the start (owner answer A10), built on option a and made two-way:
the liquid solve runs first and reads the gas pressure committed at the end of the previous step through
pressure's `EquationInputs::gas_pressure_above_atmosphere` and `holds_tracked_gas` (one step late, with the
sealed-blob step limit and small blobs held at atmosphere), then the gas solve sees the liquid surface as a
moving boundary (5.5). `kGasWithLiquidNotSupported` is removed. The four changes are made as the updated
designs state them: transfer's material sets (`material_is_counted` always one entry per material, the gas
material its own set, A8; no "empty: all"); `RemoveReason::kMarkerNotNeeded`; pressure's `kGas`,
`SealedBlobCheck`, `ComputeBlobOfCell` and `kNoBlob`; `math::Expm1` in Foundation.

Replaced in part by decision 10 (A18): "reads the gas pressure committed at the end of the previous step ...
(one step late, with the sealed-blob step limit ...)" no longer holds for a resolved sealed blob that touches an
inviscid liquid; it holds for open gas, held blobs, cells that touch gas, and sealed blobs beside a viscous
liquid.

Replaced in part again by decision 16 (A24): it no longer holds for sealed blobs beside a viscous liquid, and the
sealed-blob step limit is removed.

**4. Markers.** a) 8 per gas cell, kept between 4 and 16, weight = ambient density times volume (5.2);
b) the branch's rule: one per cell, weight 1, first one kept. b matches the branch's numbers more closely
but has no meaning beside liquid or across cell sizes, and costs 8 times fewer particles (0.97 million
against 7.7 million in the fire scene). Rec: a.

Decision: a (recommended option (owner's standing instruction, 2026-11-14)).

**5. Too much outflow: reject or cut?** a) reject the try and shorten the step (the branch's rule, 5.3 step
11), with `step_limit` suggested beforehand; b) cut each cell's outflow to what it holds and count the cells
(no retry; less exact where it triggers, which is mostly in small partly filled cells beside liquid).
Rec: a now; revisit when gas beside liquid is tested.

Decision: a; revisited with the gas-beside-liquid tests of section 7 (recommended option (owner's standing
instruction, 2026-11-14)).

**6. Sparse gas: unallocated space is ambient air.** A grid-edge face piece that is neither on a closed
domain side nor closed by `grid_edge_is_open_to_sky` lets ambient air in and gas out, and
`FindCellsToKeepActive` keeps 8 cells of margin around anything that differs from ambient. Gas that reaches
the edge is lost and reported. Alternative: gas cells over the whole domain box, as on the branch (about 6
times the cells in the fire scene). Rec: sparse.

Decision: sparse (recommended option (owner's standing instruction, 2026-11-14)).

**7. First-order flow of amounts** (ported). A limited second-order value at the face would be sharper but
needs neighbours of neighbours across size steps. Rec: keep the branch's scheme; treat the sharper one as an
experiment after the module is done. Likewise a cooling term (smoke stays hot until it mixes): not on the
branch, not designed here.

Decision: keep the branch's first-order scheme; the sharper value and a cooling term are experiments after the
module is done (recommended option (owner's standing instruction, 2026-11-14)).

**8. The fire reference scene** of section 7 (1.6 x 1.6 x 3.2 m, about 5 kW, wind from 5 s, cells 1.25, 2.5
and 5 cm, 10 s), with the branch's "Fire in a draft" kept as the comparison scene. Confirm, or name the
scene to size against. Also confirm ignition at 400 K there (the branch's preset uses 299 K, below room
temperature).

Decision: the scene as proposed, with ignition at 400 K, and "Fire in a draft" as the comparison scene
(recommended option (owner's standing instruction, 2026-11-14)). The Blender add-on's fire reference scene is
this one (A9 lists wave, syrup and fire).

**9. Output channels.** Five, as on the branch: `density` (the gas, kg/m3), `temperature` (K), `fuel`
(kg/m3), `flame` (kg/(m3 s)), and smoke under `smoke` or the branch's `soot`; plus the known mask. The brief
for the surface, foam and spray module lists four. Rec: five, the last named `smoke`.

Decision: five channels, the last named `smoke`, plus the known mask, written as OpenVDB grids (recommended
option (owner's standing instruction, 2026-11-14); file format from A14).

**10. Sealed gas beside liquid: solved with the liquid.** Decision: a resolved sealed gas blob that touches
liquid and is not held is one unknown of the liquid's `pressure::SolvePressure` (`SealedGasCoupling::
kInsideLiquidSolve`); steps 1 to 4 move into `ApplyGasStepStart`, run before the liquid's equations, which lists
those blobs as `pressure::SealedGasBlob`; `SolveGas` runs step 1 again and steps 5 to 12 and takes a coupled
blob's end background pressure as `atmosphere_pressure + sealed_gas_pressure[j]`; the gas equations are built
twice per step. Beside a viscous liquid every sealed blob stays read one step late (`kReadOneStepLate`) because
viscosity's joint solve does not carry the blob term, and `ComputeSealedGasStepLimit` enters `step_limit` only
there. `ComputeGasSeenByLiquid` is unchanged and used where no sealed blob claims a cell. Source: owner answer
A18, `literature-coupling.md` section 2, `plan-decisions.md` "Coupling inside the pressure solve (A18)"; gas plan
Tasks 2, 4, 8, 16, 18, 19, 21, 25, 26, 31. Replaces, for those blobs, the one-step-late reading of decision 3
and of 5.5 in the first version of this design.

Replaced in part by decision 16 (A24): `SealedGasCoupling` and both `sealed_gas_coupling` fields, the
`kReadOneStepLate` reading beside a viscous liquid, and `ComputeSealedGasStepLimit` with its term in
`step_limit` are removed; the rest of this decision stands.

**11. The gas solve in batches.** Decision: `ApplyGasStepStart` and `SolveGas` also run with their scratch in
CPU memory, one loaded piece at a time (`GasBufferSizes::max_loaded_blocks > 0`, `pressure::SolveBatches`), for
the step's `RunMode::kBatchedWithSolvesInPieces`; gas amounts and gas geometry stay whole on the run's
processor; no signature changes; per-blob totals by block partials then `math::TreeSum`, in both modes. A
batched run agrees with one piece within the tolerances of section 6 and is repeatable byte for byte on its own.
Source: owner answer A15 (with A17's all-CPU-or-all-GPU principle), `plan-decisions.md` "Batched solves (A15)";
gas plan Tasks 2, 29, 30, Global Constraints. Replaces "`SolveGas` runs on the whole grid, also in a batched
run" (sections 4, 5.6, 6 of the first version), "each blob's sums are taken by one thread over its cells in
ascending index" (5.3 step 8, 5.4, 6) and the byte-identical batched fire test (now
`TryStep, BatchedFireBesideWater_WithinStepTolerance`).

**12. Wind from Blender's force fields.** Decision: the add-on derives the wind keys from Blender's Wind force
fields with 1.0 m/s2 per unit of Strength (the branch's factor); I/O carries them as `gas_wind`; the gas module
reads them as written in `GasWind` and converts nothing. Source: owner answer A20, `plan-decisions.md`
"Amendments of 2026-11-14 to the Blender and render-output plans" (Wind). No gas declaration changes.

**13. Plan decisions on formulas and sums.** Decision: the forms of `ComputeGasDensity`, `ComputeAmbientDensity`
and `ComputeTemperature` fixed as in section 3; lift computed from densities (`ComputeLiftFromDensities`);
`ComputeJoinedPressure`, `ComputeSourceCoverage`, key sampling by linear interpolation in the key interval;
`GasBufferSizes::max_cells` a multiple of 64 and `carried_pressure` held in `GasBuffers`; the gas
`EquationInputs` take `lightest_density = ComputeAmbientDensity(law)` and `max_volume_error_rate` from
`settings.pressure_solve`, and `SolveGas` calls `pressure::ApplyAddedVolumeRate` right after
`ComputeGasAddedVolumeRate`; sources before the flow (one-step numbers differ from the branch: fuel 1.7, not
2.2, in `ApplyGasFlow, TwoCells`); `gas_mass_from_volume_error` reported, not counted in the balance; new markers
at cell-eighth centres; `FindCellsToKeepActive` marks only cells of the existing grid (the step grows the region);
`ToGasSnapshot` with three arguments; test library `fluidity_gas_test_scenes` with `BuildSmallFireScene`
(0.5 x 0.5 x 1.0 m, 0.0625 m cells). Source: `plan-decisions.md` "Gas".

**14. Marker lists in CPU memory, and a CPU form for batched runs.** Decision: the `MarkerChanges` rows are
`where = kCpu` in every layout (the step hands the lists to `particles::AddParticles` and `RemoveParticles`, which
take CPU spans); a GPU run adds the `"gas: marker changes on the processor"` scratch, and the GPU form reads its
two lists back, counted in the new `MarkerChanges::copy_count` and `bytes_copied`. A batched run (both modes)
calls the new CPU form `FindMarkerChanges(const MarkerGridInputs&, ...)` on the step's CPU grid copy and its
mirrors of the committed `gas_volume` and gas velocity. `FindCellsToKeepActive` stays on the amounts' processor;
the step copies `keep_active` to the CPU. Replaces, in decision 11's batched layout, "the `MarkerChanges` arrays
in the batched list". Source: plan decision, 2026-11-14 (buffers in batched runs), `plan-decisions.md` "Buffers in
batched runs" (Gas markers); owner principle A17; gas plan Tasks 10, 20, 29, 31; step plan Appendix A row M3.

**15. A reset of the solve copy report through the buffers.** Decision: `void gas::ResetSolveCopyReport(GasBuffers&
buffers);` (non-const buffers; zeroes the held `SolveBatches`' report, does nothing in one piece, allocates and
copies nothing), called by the step per try, so `FindSolveBatches` stays const. Source: plan decision,
`plan-decisions.md` "Reconcile pass (2026-11-14)" item 3 (gas plan Task 29).

**16. Sealed gas beside a viscous liquid solved in the same step (A24).** Decision: viscosity's joint solve
`SolveViscosityAndPressure` carries each listed sealed gas blob as one more unknown of its outer loop, as
`SolvePressure` does, and returns its pressure in `viscosity::SolvedCouplingArrays::sealed_gas_pressure`;
`ApplyGasStepStart` runs in stage `BuildPressureEquations` whenever that stage runs (viscous or not) and lists
every sealed blob touching liquid, and only a gas scene without that stage (no liquid) calls it at stage
`SolveGas`. Deleted: `SealedGasCoupling` (`kInsideLiquidSolve`, `kReadOneStepLate`),
`GasStepStartInputs::sealed_gas_coupling`, `SolveGasInputs::sealed_gas_coupling` and its `kBadInput` rule,
`ComputeSealedGasStepLimit` and the sealed term of `GasResult::step_limit` (now the flow-share limit only,
`+infinity` when no cell has outflow). Still read one step late through `gas_pressure_above_atmosphere` and
`holds_tracked_gas`: open gas (also above a viscous free surface), slivers, cells that touch gas but are not gas
cells, held blobs; `ComputeGasSeenByLiquid` unchanged. Lagged references in gas tests are test-side (as pressure
plan Task 50). Removed tests: `ComputeSealedGasStepLimit, MatchesFormula`, `GasBesideLiquid, SealedStepLimit`;
step test `TryStep, SealedPocketAboveSyrup_NoSealedStepLimit` added. Replaces the `kReadOneStepLate` part of
decision 10 and the matching part of decision 3. Source: owner answer A24; `plan-decisions.md` "Viscous liquids
with blocking materials and sealed gas (A24), and what counts as viscous (A25)", "A24: step, gas, solids"; gas plan
Tasks 2, 4, 8, 16, 18, 21, 25, 26, 31, Global Constraints.

**Open** (left undecided by the plans):

- Wind on liquids: A20 and the plans apply wind only to gas (no keys in a domain without gas); whether wind
  should act on a liquid surface is not decided.
- Where the gas `MarkerChanges` arrays and the other `kCpu` rows are needed by stage `AddNewParticles` in a
  batched GPU run beyond what Task 29 states (which buffers stage 1 needs on which processor) is left to the step
  and batches plans. **Settled by decision 14** (plan decision, 2026-11-14, buffers in batched runs).
- `kMemoryWarningShare = 1.10` and `kCopyWarningShare = 1.25` (I/O's measured-cost warnings, which compare the
  gas solve's copies among the others) are new constants awaiting the owner (`plan-decisions.md` "Planned cost
  shown to the user", open point 1).
- `BatchSettings::solve_piece_share = 0.25` and the batched direct-solve room `4 * max_cells`, which size the
  gas solve's loaded piece, await the owner (`plan-decisions.md` A15 "Open points" 1).
