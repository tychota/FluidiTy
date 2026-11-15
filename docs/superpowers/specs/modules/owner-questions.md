# Questions for the owner, all six module designs

> **Every question below is decided.** The owner answered A1 to A14 in `owner-answers.md`; every other
> question takes its recommended option ("Rec"), by the owner's standing instruction, and that choice is
> recorded in the "Decisions" section (section 11) of each module design. The text below is kept as asked; the
> sentence "Until you answer, ..." no longer applies.

Status: written on 2026-11-14. Merges section 11 of the contact, grid, particles, transfer, pressure and
viscosity designs, plus the questions the reconciliation raised (marked "Reconciliation"). Each design keeps
its own section 11; the source is given as, for example, "grid 9" (grid design, question 9).

Three groups, most important first inside each: words and names (Q1 to Q15), behaviour (Q16 to Q44), scope and
numbers (Q45 to Q52). "Rec" is the recommendation. Until you answer, the designs are written with the
recommended option unless a question says otherwise.

## Words and names

**Q1. The word `state` in `PerParticleVelocity` (your O3).** Sources: contact Q2, particles 11.1, transfer 2.
`state` would mean three things: `velocity.state`, `saved_state`, `state_version`.

```cpp
PerParticleVelocity<Vec3> { state; travel; }    // a) as decided
PerParticleVelocity<Vec3> { carried; travel; }  // b) "the velocity the particle carries to the next step"
PerParticleVelocity<Vec3> { kept; travel; }     // c) collides with own_motion_kept
PerParticleVelocity<Vec3> { next; travel; }     // d)
```

Rec: b, asked for by all three designs. The designs write `state` until you choose. Affects: base, contact,
particles, transfer, step.

**Q2. The shape and name of an error with context.** Sources: grid 9, contact Q7, Reconciliation.
All six designs now return `std::expected<T, <Module>Failure>`:

```cpp
struct GridFailure { GridError error; CellId cell; CellId other_cell; };
std::expected<StaggeredGrid, GridFailure> BuildGrid(const GridInputs& inputs, const GridSettings& settings);
std::string_view ToMessage(GridError error);
```

The naming review folds `refuse`, `reject` and `fail` into `reject`, so `Failure` needs your yes. Options:
a) `<Module>Failure`; b) `<Module>Problem` (grid's first word); c) `<Module>ErrorContext` (pressure's first
word); d) `std::expected<T, <Module>Error>` with the context in a separate out-parameter. Rec: a. It matches
the `Fails:` label of the docstring template, and `reject` stays the word for a step try. Affects: all.

**Q3. `SolidSide` values and a separate stable side identity (your O2).** Source: contact Q1. The code on
`main` writes only -1, 0, +1 and compares for equality, so the value is an identity. Options:

```cpp
enum class SolidSide : std::int8_t { kNoPlate = 0, kAgainstNormal = -1, kAlongNormal = +1 };  // a) geometry
struct PlateSide { std::uint32_t plate_id; SolidSide side; };                                 //    identity
std::int64_t side_id;                                                                         // b) free values
// c) the enum alone: one plate per scene
```

Rec: a. The reconciliation applied it everywhere: `contact::PlateSide` replaced transfer's `SideId` and
particles' `SolidSideId`, and the particle array is `plate_side`. Affects: contact, particles, transfer, solids.

**Q4. The grid's type name.** Sources: Reconciliation, grid 13. The naming review decides `StaggeredGrid`; the
grid design wrote `class Grid`; particles wrote `GridLayout`, viscosity `MixedSizeGrid`.
Options: a) `grid::StaggeredGrid` for the owning type, `grid::GridTables` for the plain form shared formulas
read, short `Grid...` on the rest (`GridFrame`, `GridVersion`, `BuildGrid`); b) `grid::Grid`. Rec: a (it is
the decided name). Affects: all.

**Q5. `owner` (your O5).** Source: grid 3. Five meanings were found on `main`. Rec: ban it as an identifier in
grid, transfer and pressure; say "stored in" for a block (`ToCellBlockId(cell)`), `cell_low`/`cell_high` with
`kNoIndex` for an edge piece, and `enum class RegionKind { kParticleRegion, kGridOnlyRegion }` in module 9.
"Owns:" in file headers keeps its ordinary meaning. Alternative: allow `owner` for the region meaning only.
Affects: grid, transfer, pressure, viscosity, cell resize.

**Q6. The new word `place`, and index types in other modules.** Sources: grid 5, Reconciliation.
A `cell_place` or `face_place` is a fixed position in a block that may or may not hold an active cell or a
face piece; hence `CellPlaceIndex`, `FacePlaceIndex`, `CornerPlaceIndex`. Alternative: `CellIndex`,
`FaceIndex` (short, but `CellIndex3D` is on the never-say list, and it hides that unused positions exist).
Pressure keeps its field names with the grid's types:

```cpp
grid::CellPlaceIndex cell_index{grid::kNoIndex};
grid::FacePlaceIndex face_piece_index{grid::kNoIndex};
```

Rec: `place`; confirm that field names such as `cell_index` may stay. Affects: grid, transfer, pressure,
viscosity.

**Q7. Material words that touch O6.** Sources: viscosity 2, particles 11.5, pressure 6.

- Viscosity: `deformation_rate` for the tensor (`strain` is open), `stretch` and `shear` for its entries,
  `DeformationRateSample` for one measured entry (alternative `StretchAndShearSample`), `rate_from_solids`,
  and "surface force per area" for traction.
- Particles: `rest_volume` (was `reference_volume`), `volume_ratio` (kept), `springy_deformation` (was
  `elastic_deformation` and, for cut solids, `deformation_gradient`). Alternatives: `volume_change` for the
  ratio (but `permanent_volume_change` is a log and the ratio is not), `deformation` alone for cut solids.
- Pressure: `volume_ratio` (kept), `bulk_stiffness` (not `stiffness_against_squeeze`, which the review gives
  to another quantity).

Rec: as listed; materials may rename `volume_ratio` when it settles O6. Affects: particles, pressure,
viscosity, materials.

**Q8. `can_hold_liquid` on faces, for all materials.** Source: transfer 6. Verified meaning on `main`: the
face centre is not inside a solid. The same flag gates snow and sand. Options: keep `can_hold_liquid`;
`outside_solid`.

```cpp
if (face.outside_solid && face.side == particle.side)
```

Rec (transfer's): `outside_solid`. The designs write `can_hold_liquid` until you choose. Affects: transfer,
particles, solids.

**Q9. Scope of `weighted_spread`.** Source: grid 4. Confirm that it names only the centred sum
`sum weight * offset_from_mean * offset_from_mean^T` in m2, that the first sum is `mean_offset`, and that
transfer names `sum weight * value * offset` itself (`velocity_weighted_offset_sum`, private). A blanket
rename of `moment` would be wrong in two of its four uses on `main`. Affects: grid, transfer, particles.

**Q10. Verbs for the try, and its type name.** Source: particles 11.6. `trial` is a never-say word, so the
copy is `ParticlesTry`. Verbs outside the allowed list: `CommitTry`, `RejectTry`. Alternative: `ApplyTry`,
and dropping a try by destroying it. Rec: `CommitTry`, `RejectTry`. Affects: particles, step.

**Q11. Two enum conventions.** Source: Reconciliation.
a) Every error enum starts with `kNone` (contact, pressure and viscosity had it; grid, particles and transfer
did not, and shared functions that report through a pointer need it).
b) "The chosen processor cannot be used" is `kRunOnCallFailed` in every module. The designs had
`kProcessorUnavailable`, `kCannotRunOnGpu`, `kGpuCallFailed`; the rule "no `Gpu` in public names" excludes two
of them, although Foundation's `RunOnError` has `kGpuCallFailed`. Alternative: allow `Gpu` in enumerators and
use `kGpuCallFailed` everywhere. Rec: `kNone` first; `kRunOnCallFailed`. Affects: all.

**Q12. `MaterialBoundary`, not `IngredientBoundary`.** Source: Reconciliation. In the particles design the
ingredients of one material always mix and two materials never do (Q31). The boundary between two liquids
that do not mix is then a boundary between materials, so pressure's type was renamed. Alternative:
`LiquidBoundary`. Rec: `MaterialBoundary`, if Q31 is answered a. Affects: pressure, particles, materials.

**Q13. Names added to `base/`.** Sources: Reconciliation, contact Q2 and Q7, viscosity 7.

```cpp
template <typename Value> class Buffer;                       // BuildBuffer, WriteBuffer, ReadBuffer
math::TreeSum(values, count);                                 // the fixed summation order, shared formula
ComputeTreeSum(buffer, count); ComputeTreeDot(first, second, count); ComputeLargestMagnitude(buffer, count);
BuildStableOrder(keys, count, previous_order, order);         // was "stable sort of key and index pairs"
BuildGroupsByDestination(destination, item_count, destination_count, first_of_group, item_order);
struct MemoryRow;                                             // with CountedPer, KeptFor, MemoryCounts
math::Floor, Abs, Sqrt, Cbrt, Exp, Log, Pow, Sin, Cos, Min, Max, Length, Normalized, Rotation
math::IsFinite(double), IsFinite(Vec3), IsFinite(const Mat3&) // replaces contact's AllFinite
```

Points to confirm: `Buffer` beside the decided word `buffers`; `Build...` for the two ordering functions
(`Group...` is not an allowed verb outside `GroupParticlesByBlock`); a predicate `IsFinite` inside `math::`,
where the rule says "plain math word" (alternative `math::Finite`); `PerParticleVelocity` living in `base/`.
Rec: as listed. Affects: all.

**Q14. New names to confirm, by module.** Sources: contact Q7, grid 10, particles 11.8, transfer 10,
pressure 6.

- Contact: `half_thickness`, `half_size`, `SolidMotion`, `SolidPose`, `SolidFace` (`kLowX` ... `kHighZ` in the
  solid's own axes), `ClosedSides`, `DistanceToSolid`, `FirstHit`, `ContactRecord`, `SolidReaction`,
  `reports_reaction`, `friction_scale`, `ContactBuffers`, `ContactCloseCall`, `kSolidTooThin`,
  `kOnWrongSideOfPlate`; added by the reconciliation: `DomainSide`, `ComputeSolidVelocitySample`.
- Grid: `weight_change` (was `weight_gradient`), `NearbyFaceFit { kFull, kMeanOnly, kNone }`, `KeptFor`,
  `CountedPer`, `ComesBefore` (a predicate with no listed verb; alternative `CheckComesBefore`),
  `ToMaskOfAll`; added: `FaceSearch`, `BorderWidth`, `NearbyFaceSums`, `CellsNearFace`, `CellsAroundEdge`.
- Particles: `kHasOwnMotion` (alternative `kFollowedGridOnce`), `size_level` on a particle, `CountRule`
  (alternative: a yes/no field `may_be_split_and_merged`), `ParticleAmounts`; added: `cell_begin`,
  `NeighbourhoodInputs`.
- Transfer: `GridFromParticles`, `GridAtParticle`, `GridSnapshot`, `GridTime`, `FaceValueKind` and its four
  values, `FillKind`, `GridOnlyReason` and its values, `SidesAtFace`, `is_source`, `weight_sum`, `was_solved`,
  and "source face" in the plural; added: `SolidsAtFaces`, `CellsFromParticles`.
- Pressure: `carried_pressure`, `filled_volume`, `solid_volume_rate`, `flow_per_pressure`,
  `velocity_per_pressure`, `pressure_distance`, `liquid_share`, `reference_pressure`, `face_density`,
  `volume_error`, `equation_error`, `push` (N s on a face piece), `own_weight`, `connection_scale`,
  `GeometrySnapshot`, `PressureLawKind` values, ladder and rung, `blob` for a connected body of liquid in one
  solve (O4); added: `open_volume`, `GivenTerms`, `ApplyZeroMeanToSealedBlobs`.

**Q15. Harlow and Welch 1965.** Source: grid 11. The naming review requires it in the grid header comment,
but it is not in the old bibliography. Please supply the entry (DOI), or agree to cite it by author and year
only. Affects: grid, method paper.

## Behaviour

**Q16. Who runs the solve that finds velocity and pressure together?** Sources: viscosity 1, conflict C1 of
`shared-interface.md`. The two designs answer differently and both sets of declarations exist.

```cpp
auto result = ApplyViscosity(inputs, settings, buffers, velocity_after_pressure, cell_pressure);  // A
SolvePressure(equations, viscous_velocity_response, settings, buffers);                           // B
step::SolveViscosityAndPressure(...);                                                             // C
```

A: `ApplyViscosity` runs the loop of `main` and calls pieces of pressure; the step skips `SolvePressure` when
an ingredient has viscosity. B: pressure's conjugate-gradient loop asks a hook that viscosity implements
(gains the ladder; assumes an exact inner solve). C: a third function in the step. Rec: A, and remove the
`VelocityResponse` hook from pressure. Affects: pressure, viscosity, step.

**Q17. Split and merge check a neighbourhood at a cell size that no longer exists.** Source: conflict C2.
A particle is split only once its cell is already smaller, so "the parent's neighbourhood is complete at its
own size" cannot be asked of the current grid. Options: a) check the child positions (or the merged
position) on the current grid, and replace the other check by a flag set at the last `GridToParticles`
(`kFitWasComplete`); b) run `AdjustParticleCount` before the cells change; c) drop the old-size check.
Rec: a. Affects: particles, transfer, cell resize.

**Q18. `AdjustParticleCount` has no rule on `main`.** Source: particles 11.2. Options: a) split when the cell
is smaller than the particle's `size_level`, merge same-level particles of one bigger cell eight at a time
(one byte per particle); b) ship `SplitParticle` and `MergeParticles` only and leave the rule to module 9;
c) a rule on particles per cell (`min..max`), which also thins crowded cells. Rec: a. Affects: particles,
cell resize.

**Q19. Raw weight where cell sizes mix.** Source: grid 1. `main` has two curves that disagree.
a) the uniform curve in the particle's cell width, times the face's area share, then scaled:
`w = Q(dx) Q(dy) Q(dz) * area / h^2`. Same-size regions behave exactly as the uniform solver; one code path;
the mixed-size numbers are new. b) the radial curve of `main` everywhere: tested there for mixed sizes, but
it changes uniform results and has no clean derivative. c) particles only where all 27 faces are same-size.
Rec: a. Affects: grid, transfer, particles.

**Q20. Which faces are near a particle, and when a fit is refused.** Sources: grid 2, Reconciliation.
Bounded rule: faces of the containing cell and of the cells touching it, one size level up or down, at most
76 per axis, one layer of border cells. Alternative: every face within 1.5 widths at any level (no bound,
two border layers). Rec: bounded. Also confirm `grid::kMinWeightSum = 0.1`: below it the fit is `kMeanOnly`,
so the particle follows the grid only (`GridOnlyReason::kSpreadNotInvertible`). Transfer's design had no
such floor; it now inherits the grid's. Affects: grid, transfer.

**Q21. Order of the face sums.** Sources: transfer 1, Reconciliation. The programme says "ordered by stable
particle id".

```cpp
for block (by index) { partial = 0; for particle (by cell, then id) partial += c; sum += partial; }  // a
for particle in merge_by_id(all blocks) sum += c;                                                    // b
```

Rec: a (independent of batching; a block-owned GPU scheme can later give the same bytes). The reconciliation
adds: inside a block the materials are fed in `material_index` order, because each material has its own
arrays. Affects: transfer, particles.

**Q22. Order of particles, and whether grouping is redone every try.** Sources: Reconciliation,
particles 11.7. Particles wrote (cell block, id); transfer needs the particles of one cell together and a
first-particle-per-cell table. Chosen: (cell place of the containing active cell, id), with `group_begin` per
block and `cell_begin` per cell place. Cost: 256 bytes per cell block per material in each of two buffers;
ids ascend inside a cell place, no longer inside a block. Alternative: keep (block, id) and let transfer
test every particle of a block for every face. Then: a) regroup only when `CheckGrouping` fails
(`position_when_grouped` costs 48 bytes per particle); b) regroup every try and delete that array. Results
are identical. Rec: the cell-place order; a. Affects: particles, transfer, step.

**Q23. "Solved faces only" after pressure, for every scene.** Source: transfer 3. `main` applies it in one
scene mode, where it cut volume drift from 2.11 % to 0.051 %. Options: a) always, for particles in solved
cells; b) never; c) a setting. Rec: a. Affects: transfer, pressure.

**Q24. Fill radii, and the border they need.** Sources: transfer 4, Reconciliation. `main`: liquids try 2
cell widths then copy the nearest face; elastic particles also try 3 and 4. Options:
a) `kFillRadiiInCells{2, 3, 4}` for every material; b) `{2}` for all; c) per material. Rec: a. Cost found by
the reconciliation: the grid's ordinary border is one layer of touching cells; a) needs a border of four cell
widths for the two snapshot stages (`grid::BorderWidth::kFourCellWidths`) and a walk over up to 270 blocks
per filled face (`grid::FaceSearch`). With b) the ordinary border is enough. Affects: transfer, grid, batches.

**Q25. Which constant does a sealed blob's pressure get?** Source: pressure 1. a) the volume-weighted mean
of `carried_pressure` (smooth in time, independent of cell order); b) mean zero; c) lowest cell at 0 Pa as
on `main`. `blob.pressure_offset = mean(carried_pressure) - mean(pressure)`. Rec: a. Affects: pressure,
viscosity.

**Q26. Air beside cells that are not allocated.** Sources: pressure 2, Reconciliation. Proposed: space with
no cell block that touches no solid is flooded from the open top and counts as atmosphere; anything else is
`kUnknownFrontier`, gets atmosphere pressure, is counted, and rejects the try only when tracked gas reaches
it. Alternative: every unknown frontier rejects. Rec: the proposal. The flood result is stored per grid-edge
face piece (`grid_edge_is_open_to_sky`, filled by solids), because the grid can index only blocks that exist.
Affects: pressure, solids, grid.

**Q27. One acceptance limit for the pressure solve.** Sources: pressure 3, viscosity step 10.
Proposed: `max_volume_error_rate = 2e-5 1/s` per cell and the reported volume budget; the absolute
`2e-6 m3/s` norm of `main` is dropped because it depends on scene size. Alternative: keep both. Rec: one
limit, calibrated on the four scenes. The viscosity design uses whichever you keep. Affects: pressure,
viscosity.

**Q28. Does viscous liquid stick to solids?** Source: viscosity 3. `main` sticks on mixed-size grids and
slides on same-size ones. a) always sticks when viscosity is positive; b) per solid,
`enum class LiquidGrip { kSticks, kSlides }`. Rec: a. Affects: viscosity, solids.

**Q29. Shear samples near a size change, and what is allowed from the first commit.** Source: viscosity 4.
Proposed: edge samples wherever the cells around an edge share one size; fitted cell-centre samples only for
the quarters at a size change or an inclined solid. Alternative: fitted samples for every cell of any
mixed-size grid, as on `main` (simpler, but drops the rule that passes the two-viscosity tests). Rec: the
split. Also confirm: bulk viscous liquid may cross a size change from the first commit, while the five
exclusions of viscosity 5.7 stay until their tests pass. Affects: viscosity, grid.

**Q30. What viscosity reads from the geometry snapshot.** Sources: viscosity 5, conflict C3. The divisor of
the stretch samples is `open_volume` with the weight on `filled_volume`; pressure now stores both (done).
Still open: liquid volume per cell eighth and two liquid-covered centres. Options: a) pressure adds
`CellEighthGeometry` and `FacePieceCentre`, filled only in viscous scenes (96 bytes per cell, 24 per face
place); b) viscosity clips them itself; c) whole-cell volumes only. Rec: a. Affects: pressure, viscosity.

**Q31. Mixtures, and liquids that do not mix.** Sources: particles 11.4, pressure 5, conflict C4.

```cpp
const double* ingredient_fractions;  // a) no unit, rows sum to 1; liquids that do not mix are two materials
const double* ingredient_volumes;    // b) m3, rows sum to volume_ratio * rest_volume
// c) a per-particle mixing group id (phase_id stays)
```

Rec: a ("a material is what mixes"; `phase_id` disappears). Then, for two liquids that do not mix: port the
sharp boundary to mixed-size cells (pressure's recommendation) or carry the mixed density only. Nobody
produces the boundary's distance field today; rec: pressure computes one distance field per liquid material
(C4, option A), and the first pressure commits ship without the sharp boundary. Affects: particles, pressure,
transfer, materials.

**Q32. Ingredient amounts on the grid.** Sources: transfer 9, Reconciliation. Pressure and viscosity both
need mass, volume and ingredient volumes per cell. a) transfer provides them; b) each module sums its own.
Rec: a, and the reconciliation declared it (`transfer::ComputeCellsFromParticles`). Two details were chosen
without a design behind them, please confirm: each particle adds to the one active cell that contains it (no
weight curve); a liquid cell with no particle gets zeros, and each reader applies its own rule (viscosity
takes the touching cell with the lowest id; pressure has no rule yet). Affects: transfer, pressure,
viscosity.

**Q33. Who owns the travel-share rule and the artist control.** Source: transfer 7. Transfer takes effective
shares `own_motion_kept` and `own_motion_in_travel`. a) materials:
`materials::ComputeOwnMotionInTravel(material, exact_velocity_change, distance_to_surface, distance_to_solid)`,
and the scene maps one control to `own_motion_kept`; b) step. Rec: a. Also confirm that the factor 0.99
disappears as a constant: a scene that wants it sets `own_motion_kept = 0.99`. Affects: transfer, materials.

**Q34. The too-few-particles rule.** Source: transfer 8. `minimum_face_mass` in kg becomes
`min_weight_sum_at_face` (no unit: how many particle shares reached the face), default 0 as on `main`.
Confirm the change of unit, or keep kg. Affects: transfer.

**Q35. Friction per hit.** Source: contact Q4. a) `friction` per particle from its material, times
`friction_scale` per solid (default 1; 0 for a smooth solid); b) a table per material and solid; c) per solid
only. Rec: a. Affects: contact, materials, solids.

**Q36. Contact tolerance and the rest gap (part of O1).** Source: contact Q5. Proposed: `0x1p-40` of the
largest coordinate plus `0x1p-30` of the smallest cell width; a particle in contact rests one tolerance
outside the surface (about `1e-11` m in a metre-sized scene); `kSolidTooThin` below 1024 tolerances.
Alternative: rest on the surface and accept a distance down to minus one tolerance in every module that asks
"is this point inside a solid". Rec: the rest gap. Affects: contact, solids, pressure.

**Q37. One motion for every shape, and the plate's extent.** Source: contact Q6. Proposed: any shape may
translate and turn (at most a quarter turn per step); a plate is unbounded sideways as on `main`, and a
finite thin solid is a `Box`, which has no plate sides. Alternative: give `Plate` a half size. Rec: keep the
unbounded plate; decide the finite one in the solids design. Affects: contact, solids.

**Q38. Thin solids: one cell with liquid on both sides, and several plates at one face.** Sources:
pressure 7, transfer 5. Pressure rejects a cell that a thin solid splits (`kCellSplitBySolid`, as on `main`);
the alternative is two cell equations per cell, which changes grid, contact and transfer. Transfer allows two
sides at one face (`kMaxSidesAtFace = 2`, one plate); more costs 17 bytes per face per snapshot. Rec: reject,
and 2; revisit both in the solids design. Note: contact and transfer do carry both sides of a plate, so a
plate thinner than a cell is usable today only where at most one side holds liquid in any cell.
Affects: pressure, transfer, contact, solids.

**Q39. Sibling cells exist together.** Source: grid 6. A split cell always has all 8 small cells. This
removes quarter edge pieces and one cell state, and constrains `ChooseActiveCells` and `SplitCell` (up to 7
empty small cells at a grid edge). Alternative: partial siblings with a fifth state. Rec: together.
Affects: grid, step, cell resize.

**Q40. `active_cell` and `ChooseActiveCells`.** Source: grid 7. Proposed single meaning: the active cells
are exactly the cells that exist and are not split; `ChooseActiveCells` produces that list (new cells get a
size from the `ChooseCellSize` callback) and so makes a new `GridVersion` before `ParticlesToGrid`;
`ResizeCells` changes sizes after commit. Confirm, or name block existence differently. Affects: grid, step.

**Q41. `BuildGrid` runs on the CPU, also for GPU runs.** Source: grid 8. Per step only block ids come down
(8 bytes per block) and 456 bytes per block go up; no field value moves. Rec: accept this as not being "a
copy between dependent stages". Alternative: a GPU-side builder in the grid module. Affects: grid, step.

**Q42. Which materials may be split and merged.** Source: particles 11.3. a) liquid and gas only; b) also
split (never merge) snow and sand, copying their deformation. Rec: a. Affects: particles, materials.

**Q43. Corner places in the grid.** Sources: Reconciliation, grid 14. Pressure and solids sample distances at
cell corners and need one index per physical corner, also where cells of two sizes meet. Added to the grid:
125 corner places per block, `CornerValues<T>`, `FindCornerPlaces`. Cost: 1,000 bytes per block for each
`CornerValues<double>` (pressure keeps two). Alternative: no corner storage; each cell samples its own 8
corners (8 times the samples, and two cells may round a shared corner differently). Rec: corner places.
Affects: grid, pressure, solids.

**Q44. The carried pressure is moved on the grid, by pressure.** Source: Reconciliation. Pressure's design
expected transfer to move it "with the liquid". No particle carries a pressure, and `main` follows the
committed grid velocity back from each cell. Declared as `pressure::ComputeCarriedPressure`, used only when a
pressure law is active. Alternative: a per-particle pressure array (8 bytes per particle in two buffers) and
a cell sum in transfer. Rec: on the grid, as on `main`. Affects: pressure, step.

## Scope and numbers

**Q45. Foundation must grow before contact starts.** Sources: contact Q3, viscosity 7, Reconciliation.
`MoveParticles` cannot keep data on the GPU without `Buffer<T>`. Options: a) add everything in section (a)
of `shared-interface.md` to Foundation now; b) each module adds what it needs first; c) contact ships only
the shared `MoveParticle` and its batch operation waits. Rec: a (one extra Foundation task, then every plan
is written against a fixed `base/`). Also confirm that `contact::MoveParticles` is the body of step stage 13.
Affects: Foundation, all.

**Q46. Contact between two materials is not contact's.** Source: contact Q8. `distinct_body_contact_3d.h` and
`mixed_contact_3d.cc` (liquid against snow on shared faces) go to module 7 as `CollideSolids`. Confirm.
Affects: contact, solids.

**Q47. Where the open sea lives.** Source: pressure 8. It is a velocity relaxation run in stage `DriveWaves`,
and its outlet calms waves without letting liquid leave. Keep it in `pressure/`, or move it to `solids/`
beside the wave paddle? Rec: keep the programme's placement. Affects: pressure, solids, step.

**Q48. Plain doubles in the pressure solve.** Source: pressure 4. `main` uses 32-bit fine vectors with 64-bit
restarts; the programme defers reduced precision to measurement. Proposed: all `double` now. Confirm.

**Q49. Block width 4 and 240 face places per block.** Source: grid 12. Kept as ported constants (25 % of
face places unused); other widths are measured later with the rest of performance tuning. Confirm.

**Q50. The syrup reference scene.** Source: viscosity 6. Proposed: domain box 0.4 x 0.4 x 0.3 m; syrup of
10 Pa s, 1400 kg/m3, surface tension 0.08 N/m; a 2 cm stream at 0.3 m/s from 0.25 m onto the floor, 1 litre
in total, then 10 s of settling; cells of 1.25, 2.5 and 5 mm. It needs only the domain floor, so it runs
before any exclusion is lifted; a tilted plate would make it wait. Rec: floor only. Affects: viscosity, step.

**Q51. Per-face sides limit and other small numbers to confirm.** Sources: transfer 5, contact 3.4, grid 3.5.
`kMaxSidesAtFace = 2`; `kMaxHitsPerStep = 4`; `kMaxAdvanceCount = 128`; quarter turn per step;
`kMaxNearbyFaces = 76`; `kMaxCellsNearFace = 112` (an upper bound from the reconciliation; a test searches
for the true maximum); `GridSettings::max_cell_blocks = 1 << 22` (about 1.9 GB of grid tables). Rec: as
listed.

**Q52. Every module publishes its memory rows.** Source: Reconciliation. The sizing estimate must come from
the rows the build allocates from. Grid and particles had rows; `BuildContactMemoryRows`,
`BuildTransferMemoryRows`, `BuildPressureMemoryRows` and `BuildViscosityMemoryRows` were added as one line
each, and pressure's and viscosity's byte totals become sums of their rows. Confirm that this is wanted in
the first commits of each module. Affects: all, step.
