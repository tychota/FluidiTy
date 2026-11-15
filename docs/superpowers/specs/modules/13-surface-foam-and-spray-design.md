# Surface, foam and spray module design

Status: updated on 2026-11-14 with the owner's answers (A1 to A22), `changes-to-apply.md`, and
`plan-decisions.md` ("Surface, foam and spray (render output)" and "Amendments of 2026-11-14 to the Blender and
render-output plans": A16 one Blender version in a witness record, A19 newest library versions). Module 11 of the
build order ("surface with foam and spray", file 13 of this folder).

## 1. Purpose

After each committed step the simulation holds particles, grid values and solids. People cannot look at
those directly, so this module turns them into three things to look at: a triangle mesh of the liquid
surface, a separate layer of small foam, spray and bubble particles, and volume data (foam counts, and the
gas module's smoke, temperature, fuel and flame). Everything here reads committed state and never writes it:
a run with this module switched off commits the same bytes.

Owns: the recording of what a step offers (foam source events, the liquid's motion near the surface) under
the try-and-commit rule; the surface mesh of each output frame (field, mesh, move, check, the rule that
replaces a rejected mesh by a newly built one, volume report, per-vertex fields, droplets); the foam
particles and their life cycle; the counted volume data and the "known" data that keeps zero, unknown and
absent apart; the output field contract (one table, section 5.7); every open-format writer (owner A14):
Alembic, USD and OpenVDB files for the surface, droplets, foam particles, volumes, the playback of the
simulation particles (taken over from I/O) and solid motion; the format table that says, per kind of
geometry, which format is written and which Blender witness settled it (5.7); the build target
`fluidity_interchange` that holds the Alembic, OpenUSD and OpenVDB libraries and is linked by the CLI only.

Does not own: the liquid surface the solver uses (`pressure::FindLiquidSurface`; it never reads this mesh);
any particle, grid or solid value; when a snapshot is read, the output queue and its worker threads, the run
folder, `OutputFile`, `WriteListedFile`, the `file_list` and the word `listed`, `WriterLock`, FluidiTy's own
array file (saved state and internal run records) and the run report (I/O and CLI); Blender objects, node
groups, materials and the running of the witness tests (Blender add-on); gas physics (gas).

## 2. What exists today

All on `main`, Y-up, CPU only. 7,956 lines in `native/src`, 1,450 in 23 headers under
`native/include/fluidity/host/`, 20 test files. The old paper describes far more than the code does
(`docs/papers/fluidity-method.typ:316-401`); what follows is the code.

| Piece | Where | What it does | What is wrong |
|---|---|---|---|
| Motion across an output interval | `motion_capture_3d.cc:212-340`, `motion_interval_3d.cc:85-214`, `motion_carrier_3d.cc:30-292` | Keeps up to 65 full copies of the committed state per frame (first commit that crosses each of 1 to 64 equal slots), drops them on a rejected try. Velocity at a point: the transfer's own weights at one copy, linear in time between two copies, never extended past the interval. `AdvectMotionPoint3D`: midpoint rule, each move at most half a cell width, split at copy times. | Full-state copies (the header calls it "the initial reference policy"). Needs the whole transfer and scene to answer one velocity. Mixed-size grids only above a flat liquid in a box. |
| Surface field | `surface_field_3d.cc:20-177`, `surface_occupancy_3d.cc:78-244` | Spreads each particle's volume to 8 points of a dense world-fixed array, smooths with a Gaussian cut at `round(4 sigma / spacing)`, value = 0.5 minus volume share. Deep liquid and air are overwritten with -0.5 and +0.5 from the simulation. | Dense array, 2,000,000 points at most. Unknown coverage fails the frame. Near a solid the share drops by half; fixed for one fixed plane only by mirroring particles (`surface_reconstruction_3d.cc:796-806`). |
| Mesh from the field | `surface_mesh_3d.cc:153-327` | Six tetrahedra per cube, one diagonal rule for all cubes, vertices shared through edge ids. A value within `1e-8 * spacing` of zero counts as air. Closed when the array's outer layer is air. | Crossings within `1e-7` of an edge end are merged into the corner, which can pinch the mesh; hence the extra checks. 100,000 vertices at most. |
| Mesh checks | `surface_mesh_3d.cc:563-703` | Signed volume, area, pieces, Euler number, open edges, edges and vertices shared wrongly, orientation, zero-area triangles, triangle pairs that cross (sweep over boxes). | Sound. Kept. |
| Volume correction | `surface_mesh_3d.cc:704-815` | Moves free vertices along normals, at most `0.25` cell widths, until the mesh volume is within 0.5 % of the material volume; backs off when a check fails. | Not reaching 0.5 % fails the frame (`surface_reconstruction_3d.cc:992-1009`, `surface_cache_3d.h:11`). The report says mesh volume is an observation, not a gate (validation, "Four volumes"). |
| Moving the last mesh | `surface_tracking_3d.cc:33-429` | Moves every vertex through the interval, pulls it back to the field's zero set (Newton steps with halving, at most `0.25` cell widths, tolerance 1 % of the smallest triangle height), keeps ids. Then 8 sweeps of sideways moves and up to 64 diagonal swaps on crossing triangles (`:221-383`). Rejects the mesh with counts when it is still unsafe (`:410-425`). | The rejection is right and is kept (numerics 12.2). The sweeps and swaps are local repair with no rule for when liquid merges or splits (12.2, 12.3). A rejected mesh fails the frame: no surface is written and the stage stops (`m2-bounded-wave-evidence.md`: frame two failed). |
| The driver | `surface_reconstruction_3d.cc:425-1270` | One 850-line function. Frame zero builds from the field; later frames must keep piece count and Euler number or fail "requires local repair" (`:737`, `:968`, `:1064`). | Fixed plane or box only; a moving solid is `Unimplemented` (`:446`); 16 pieces at frame zero; 65 samples. |
| Two liquids | `surface_mesh_3d.cc:328-562`, `surface_cache_3d.h:19-30` | One outer mesh for both, plus one inner sheet between them that shares the seam vertices. Per-vertex ingredient shares. | Exactly two. Liquids taken from an initial sphere set only (`surface_reconstruction_3d.cc:581-592`). |
| Blob history | `surface_component_lineage_3d.cc`, `surface_material_components_3d.cc` | Matches blobs between frames by shared particle ids. | 1,024 particles at most, pair tests in `n^2`. |
| Snow (and the granular points of `main`) | `tools/blender/point_cache_writer/blender_bridge.cc:76-203` | Points only, Alembic, written by the bridge tool for the whole range. No surface exists for snow (Blender chapter, channel matrix). | Whole-range export repeated on every attach. The point writer is ported into 5.7's points writer. |
| Foam sources | `whitewater_3d.cc:33-212`, `whitewater_sphere_source_3d.cc:56-215` | Ihmsen's three measures on a world-fixed source array with 26 neighbours: speed squared, approach speed of neighbours, outward-moving crest. Event weight = occupied volume x strength x step duration, one event per source cell and committed step. | Normals and curvature are the exact ones of a sphere (`whitewater_sphere_source_3d.h:17`): the source works only for a sphere of liquid and a fixed plane. 4,096 source cells. Too many events reject the step in the old paper (`E-WW-SOURCE-CAPACITY`). The step's own stage is empty (`liquid_substep_3d.cc:2736`). |
| Adding foam particles | `whitewater_3d.cc:290-461` | `floor(rate * weight / count_per_particle + owed)` particles per event, the rest owed to the same source cell, one rising id. | All particles of an event start at the same point with the same velocity (`:322-337`). |
| Foam motion | `whitewater_3d.cc:213-289`, `whitewater_transport_3d.cc:71-280`, `whitewater_frame_3d.cc:22-225` | Exact drag toward a target velocity (`expm1`, series for small rates); spray toward air velocity plus fall, bubbles toward liquid velocity plus rise, foam with the liquid; kind from distance to the surface, held until the particle is half the radius past the limit; foam ends by age. | Foam with no known liquid velocity fails the frame. Contact with one fixed plane only. 65,536 particles. |
| Counted volumes | `whitewater_volume_3d.cc:13-98`, `whitewater_volume_cache_3d.cc`, `blender/fluidity/volume_playback.py:12-80` | Foam and bubble count per m3 and bubble gas share on a dense box; a "known" array kept apart from the values. | The box is declared by hand and a particle outside it fails the frame (frame 18 of the beach run). The `.vdb` is written voxel by voxel in Python inside Blender. Gas channels do not exist on `main`. |
| Files | `blender_bridge.cc:164-203, 236-580`, `schemas/effects/*.fbs`, `surface_cache_3d.cc`, `*_stage_3d.cc` | Surface as one USD text file for the whole range (shares, no velocity); surface as Alembic (velocity, no shares); foam as Alembic points. | Each attach writes every frame again (`surface_playback.py:34-57`). 64-bit ids do not reach Blender as an attribute. Surface and foam store their source under two different keys (`surface_playback.py:101`, `foam_projection.py:120`). Per-vertex `wetness` and `thickness_m` are stored and never computed. |

Y-up specifics: arrays are ordered x, then y, then z with y up; source cells are ordered by signed z, y, x;
the bridge writes Alembic unrotated because both are Y-up (`blender_bridge.cc:154`); the Python volume
writer rotates each index by `(x, -z, y)` (`volume_playback.py:16-19`). In the rebuild every array and file
of this module is Z-up; the one rotation left is inside the Alembic writer (5.7).

Not present on `main` and therefore new here: a surface near any moving solid, a surface on mixed-size
cells in general, foam sources for a general liquid, the gas volume channels, running beside the simulation,
OpenVDB written by native code, any USD written per frame, solid motion in a file, and a format choice
backed by a test in Blender.

## 3. Public interface

Three folders, one module, one error enum: `native/fluidity/render_output/` (`fluidity::render_output`,
shared pieces and the two entry points; the folder is NEW, decision 4), `surface/` (`fluidity::surface`),
`foam_and_spray/` (`fluidity::foam_and_spray`). The writers live in `render_output/interchange/`
(`fluidity::render_output`), the only sources of the build target `fluidity_interchange` (5.7). Files form no
cycle: `lattice.h`, `motion_sample.h` and the error header include only earlier modules; `surface/` and
`foam_and_spray/` include those; `step_recorder.h` and `render_frame.h` include everything; `interchange/`
includes everything and the three libraries, and no other file includes `interchange/`. Every name is NEW
unless marked D. Headers are CPU only unless marked (shared). Only 3.3 runs on the simulation's processor.

### 3.1 `render_output/render_output_error.h` (shared), `render_output_message.h`

```cpp
enum class RenderOutputError : std::uint8_t {
  kNone,
  kBadInput,            // A setting or an input value is not finite or is out of range.
  kWrongGridVersion,    // An input of RecordStep belongs to another grid version.
  kWrongStateVersion,   // A recorded interval, a snapshot and the kept state do not follow one another.
  kTooManyLatticeBlocks,// RenderLimits::max_lattice_blocks reached. The product is left out of this frame.
  kTooManyVertices,     // RenderLimits::max_vertices reached. The surface is left out of this frame.
  kMeshNotSafe,         // CheckSurfaceMesh found a defect. From MoveSurfaceMesh this is the fail-safe reject.
  kPullBackFailed,      // A moved vertex could not be brought back to the surface within the allowed move.
  kMotionNotKnown,      // A vertex or a foam particle left the lattice points where motion was recorded.
  kTopologyChanged,     // The moved mesh and the field disagree on closed pieces or on their Euler number.
  kEdgeTooLong,         // The moved mesh has an edge above SurfaceSettings::max_edge_in_spacings.
  kOutOfMemory,
  kRunOnCallFailed,     // RecordStep only: the simulation's processor failed.
  kFormatNotInBuild,    // NEW. A writer was asked for a format that is not a candidate of that kind in this build.
  kFileWriteFailed,     // NEW. The library or io::OutputFile failed. The file is not listed.
  kTooManyFieldParts,   // NEW (reconcile item 26). RenderLimits::max_field_parts reached. The surface is left out.
};
struct RenderOutputFailure {
  RenderOutputError error = RenderOutputError::kNone;
  std::uint64_t frame = 0;       // Output frame. 0 inside RecordStep.
  std::uint64_t id = 0;          // Vertex id, foam particle id or lattice block key, as the error names.
  double magnitude = 0, limit = 0;  // What was measured and what was allowed, in the unit the error names.
};
std::string_view ToMessage(RenderOutputError error);
// NEW. The stable code: the enumerator's name without the k, in lower snake case (`format_not_in_build`).
std::string_view ToName(RenderOutputError error);
```

Only `kBadInput`, `kWrongStateVersion`, `kOutOfMemory` and `kRunOnCallFailed` ever leave `BuildRenderFrame` or
`RecordStep` as a failure; only `kBadInput`, `kFormatNotInBuild`, `kFileWriteFailed` and `kOutOfMemory` leave a
writer. The others are reasons inside a `SurfaceReport` or `FoamReport` (5.3, 5.5).

### 3.2 `render_output/lattice.h` (shared), `motion_sample.h`

A lattice is this module's one new geometric word: world-fixed points at equal spacing, kept in blocks of
8 x 8 x 8 where something is known. It is not the staggered grid and has no faces (decision 1).

```cpp
struct LatticeFrame { Vec3 origin{}; double spacing = 0; };   // m. Point (i, j, k) is origin + spacing * (i, j, k).
struct LatticeCoord { std::int32_t x = 0, y = 0, z = 0; };    // Integer place of a point or of a block.
inline constexpr int kLatticeBlockWidth = 8;                  // Points per block along one axis; 512 per block.
FLUIDITY_CPU_AND_GPU std::uint64_t ToBlockKey(LatticeCoord block);   // 21 bits per axis, z highest, offset 2^20.
FLUIDITY_CPU_AND_GPU Vec3 ComputePointPosition(const LatticeFrame& frame, LatticeCoord point);   // m, world
FLUIDITY_CPU_AND_GPU LatticeCoord FindLowPoint(const LatticeFrame& frame, Vec3 position);  // math::Floor per axis.
struct LatticeBlocks {                 // Which blocks exist. Ascending key: the order of every lattice array.
  LatticeFrame frame;
  std::vector<std::uint64_t> block_key;
};
template <typename Value> struct LatticeValues { std::vector<Value> values; };  // 512 per block; x fastest.
std::size_t FindBlock(const LatticeBlocks& blocks, std::uint64_t block_key);     // npos when absent.
struct LatticeReading { double value = 0; Vec3 value_change{}; bool is_known = false; };  // value_change: per m.
// Trilinear between the 8 points around `position`. is_known is false when any of the 8 is missing or not
// known; value is then 0 and must not be used. Exact for a field that is linear in space.
LatticeReading ComputeLatticeReading(const LatticeBlocks& blocks, const LatticeValues<float>& values,
                                     const LatticeValues<std::uint8_t>& known, Vec3 position);

// motion_sample.h
struct MotionSample {                  // The liquid near its surface at one committed time. CPU-owned.
  double time = 0;                     // s.
  std::uint64_t state_version = 0;     // D.
  LatticeBlocks blocks;
  LatticeValues<std::uint8_t> velocity_known;       // 1 where enough face velocities were near the point.
  std::array<LatticeValues<float>, 3> liquid_velocity;  // m/s, world, x y z.
  LatticeValues<float> distance_to_surface;         // m, negative inside the liquid. D. Known at every point.
  std::vector<contact::Solid> solids;               // Pose and motion at `time`, ascending solid_id.
};
struct OutputInterval {                // From the last output frame to this one. At least two samples.
  std::span<const MotionSample> samples;            // Ascending time. First: last frame. Last: this frame.
};
struct LiquidMotion { Vec3 velocity{}; double distance_to_surface = 0; bool velocity_is_known = false,
                      distance_is_known = false; };
// The sample at `time`, or the two samples around it mixed linearly in time at the same position. Never
// extended outside the interval or outside the known points.
LiquidMotion ComputeLiquidMotion(const OutputInterval& interval, Vec3 position, double time);
struct MovedPoint { Vec3 position{}; int move_count = 0; bool motion_was_known = true; };
// Follows the liquid from from_time to to_time: midpoint rule, each move at most half a spacing, moves end
// at sample times, at most kMaxMovesPerPoint moves. Stops and reports where motion is not known.
MovedPoint ComputeMovedPoint(const OutputInterval& interval, Vec3 position, double from_time, double to_time);
inline constexpr int kMaxMovesPerPoint = 4096;      // Ported.
```

### 3.3 `render_output/step_recorder.h` (+ `_cpu.cc`, `_gpu.cu`), `foam_and_spray/foam_source.h` (shared)

```cpp
struct MotionSettings {                // D suffix.
  double spacing = 0;                  // m. 0: the smallest cell width of the scene. At most twice that width.
  double reach_in_spacings = 4;        // Points this near the surface, on both sides, are recorded.
  int samples_per_frame = 1;           // 1..64 equal parts of a frame; the first commit in each part is kept.
};
struct Ramp { double low = 0, high = 0; };   // high > low, in the unit of the measure.
struct FoamSourceSettings {
  Ramp speed_squared_half{};           // m2/s2. Half the squared speed. Gates both measures below.
  Ramp trapped_air{};                  // m/s. Neighbours that approach one another.
  Ramp wave_crest{};                   // 1/m. Curvature where the liquid moves out of itself.
  double trapped_air_share = 1, wave_crest_share = 1;   // No unit, zero or more.
  std::uint32_t max_events_per_step = 1u << 20;
};
struct FoamSourceEvent {               // One lattice point in one committed step. 64 bytes.
  std::uint64_t step_count = 0;        // The committed step that made it.
  LatticeCoord point{};                // Which lattice point.
  double time = 0;                     // s. End of that step.
  Vec3 position{}, velocity{};         // m, m/s, world. Stored as float.
  double volume_time = 0;              // m3 s. Occupied volume x strength x step duration.
};
struct RecordedInterval {              // What one output interval offered. Moved out by ReadRecordedInterval.
  std::uint64_t first_state_version = 0, last_state_version = 0;
  std::vector<MotionSample> samples;   // Without the sample of the last frame, which the caller still holds.
  std::vector<FoamSourceEvent> events; // Ascending (step_count, block key, point in block).
  std::uint64_t dropped_event_count = 0;      // Events above max_events_per_step, smallest volume_time first.
  std::uint64_t curvature_not_known_count = 0;  // Points whose distance changes too slowly to give a normal.
  // NEW. The liquid of the commit at this frame's time, as cell_resize::ComputeLiquidTotals sums it (particles
  // plus grid-only cells): one total per cell block, then math::TreeSum over blocks in the whole grid's
  // order, so one try recorded in batches gives the bytes of one call. The volume target of 5.3 step 5.
  cell_resize::LiquidTotals liquid_totals_at_frame{};
};
class StepRecorder;                    // Move-only. Holds device working buffers sized by BuildRenderOutputMemoryEstimate.
std::expected<StepRecorder, RenderOutputFailure> BuildStepRecorder(
    RunOn run_on, const MotionSettings& motion, const FoamSourceSettings* sources /* null: no foam */,
    const RenderLimits& limits, double frame_duration, double first_frame_time);
// Body of step stage MakeFoamAndSpray (D). Reads the try over inputs.blocks only; changes only the recorder.
// A batched run calls it once per loaded piece of the same try; the calls add to one pending record and no
// call sums over the whole grid. What it finds is kept apart until CommitRecordedStep. Fails only with
// kRunOnCallFailed, kWrongGridVersion, kBadInput; reaching a limit is counted, not a failure.
std::expected<void, RenderOutputFailure> RecordStep(StepRecorder& recorder, const step::FoamAndSprayInputs& inputs);
// Orders the pending record (events and sample points by lattice key, block totals by whole-grid block),
// applies max_events_per_step over the whole step (5.4) and keeps it. Allocates nothing beyond the recorder.
void CommitRecordedStep(StepRecorder& recorder, std::uint64_t state_version, std::uint64_t step_count);
void RejectRecordedStep(StepRecorder& recorder);   // The try did not commit: forget what RecordStep found.
RecordedInterval ReadRecordedInterval(StepRecorder& recorder);   // At an output frame, after its commit.
// NEW (plan decision, 2026-11-14, buffers in batched runs). Every ReadBuffer call of RecordStep on the GPU (event
// count, events, counters, kept sample points, liquid totals), one copy each, bytes into bytes_from_processor,
// cumulative since BuildStepRecorder; {} for a CPU recorder. I/O adds the difference per committed step to the
// run's measured copies (4, I/O item 9).
step::CopyCounts ReadRecorderCopyCounts(const StepRecorder& recorder);

// foam_source.h (shared formulas, FLUIDITY_CPU_AND_GPU, see 5.4)
struct SourcePoint { Vec3 velocity{}; float distance_to_surface = 0; bool velocity_is_known = false; };
double ComputeRampShare(double value, Ramp ramp);                       // 0..1
double ComputeOccupiedShare(double distance_to_surface, double spacing);  // 0..1 of one lattice cell
// The 27 points around one point, x fastest; the centre is entry 13.
struct SourceStrength { double speed_squared_half = 0, trapped_air = 0, wave_crest = 0, strength = 0;
                        bool curvature_is_known = true; };
SourceStrength ComputeSourceStrength(const SourcePoint* points27, const Vec3* normals27, double spacing,
                                     const FoamSourceSettings& settings);
```

### 3.4 `surface/` headers

```cpp
// surface_settings.h
struct SurfaceGroup {                  // Materials meshed together: they share one outer surface.
  std::vector<std::uint64_t> material_ids;  // Ascending. MaterialKind::kLiquid only (decision 7); else kBadInput.
};
struct SurfaceSettings {
  std::vector<SurfaceGroup> groups;    // One mesh object per group.
  double spacing = 0;                  // m. Lattice spacing of the surface field. 0: half the smallest cell width.
  double smoothing_width = 0;          // m. Standard deviation of the Gaussian. 0: 2 * spacing (main's ratio).
  double max_edge_in_spacings = 2.5;   // A moved mesh with a longer edge is rejected.
  double max_pull_back_in_spacings = 0.25;  // Ported. Farthest a moved vertex may be pulled to the surface.
  double volume_tolerance = 0.005;     // No unit. Ported 0.5 %. A report limit, never a failure.
  bool keeps_ids_between_frames = true;  // false: build from the field every frame (preview).
};
// surface_field.h
struct FieldPoint { Vec3 position{}; double volume = 0; double width = 0; std::uint32_t ingredient_row = 0; };
struct SurfaceField {                  // 0.5 - liquid share of the open space, smoothed. Negative in liquid.
  LatticeBlocks blocks;                // Blocks within the smoothing reach of any liquid.
  LatticeValues<float> value;          // No unit, -0.5..0.5. Blocks that are absent are air.
  LatticeValues<float> distance_to_solid;       // m, capped at +-4 spacings.
  std::vector<LatticeValues<float>> ingredient_share;  // 0..1, one per ingredient when there are several.
};
struct SurfaceFieldInputs {            // snapshot: particles (kOutputSnapshotFields) and the grid-only arrays.
  const step::SimulationSnapshot* snapshot;  std::span<const std::uint64_t> material_ids;
  std::span<const contact::Solid> solids;    const contact::DomainBox* domain_box;
  const grid::GridFrame* grid_frame;         double spacing = 0, smoothing_width = 0;
};
// CHANGED (A31): order_buffers (CPU, at least the group's field parts) orders the field parts of 5.3 step 3;
// RenderOutput::frame_order_buffers in a run. kTooManyFieldParts above limits.max_field_parts.
// Requires: order_buffers.run_on() == RunOn::kCpu (debug assertion, A30).
std::expected<SurfaceField, RenderOutputFailure> BuildSurfaceField(const SurfaceFieldInputs& inputs,
                                                                   const RenderLimits& limits,
                                                                   OrderBuffers& order_buffers);
// surface_mesh.h
struct SurfaceVertex {                 // 64 bytes with padding.
  std::uint64_t id = 0;                // Nonzero. Kept while the vertex is moved from frame to frame.
  Vec3 position{};                     // m, world.
  Vec3 liquid_velocity{};              // m/s, world. The liquid's velocity at the vertex at frame time.
  float age = 0;                       // s since the id first appeared.
  float distance_to_solid = 0;         // m. At or below 0.25 spacing the vertex is "at a solid".
  std::uint32_t blob_index = 0;        // Which closed liquid body, this frame only.
};
enum class TriangleKind : std::uint8_t { kLiquidToAir, kLiquidToSolid, kBetweenTwoMaterials };
struct SurfaceMesh {
  std::vector<SurfaceVertex> vertices;
  std::vector<std::array<std::uint32_t, 3>> triangles;   // Counter-clockwise seen from outside the liquid.
  std::vector<TriangleKind> triangle_kind;
  std::vector<std::uint32_t> triangle_material_index;    // Empty for a group of one material.
  std::vector<std::vector<float>> ingredient_share;      // Per ingredient, per vertex. Empty for one ingredient.
  std::uint64_t next_id = 1;
};
// Six tetrahedra per lattice cube; closed, without crossing triangles, each edge shared by two triangles.
// Fails: kTooManyVertices.
std::expected<SurfaceMesh, RenderOutputFailure> BuildSurfaceMesh(const SurfaceField& field,
                                                                const RenderLimits& limits);
// check_surface_mesh.h
struct SurfaceMeshDefects {            // All zero means safe.
  std::uint64_t open_edge_count = 0, overused_edge_count = 0, pinched_vertex_count = 0,
                wrongly_turned_edge_count = 0, flat_triangle_count = 0, crossing_pair_count = 0;
  std::uint64_t first_triangle = 0;    // Lowest triangle index with a defect, for the message.
};
struct SurfaceMeshMeasure { double volume = 0, area = 0;   // m3, m2
                            std::uint32_t piece_count = 0; std::int64_t euler_number = 0; };
struct CheckedMesh { SurfaceMeshDefects defects; SurfaceMeshMeasure measure; bool is_safe = false; };
CheckedMesh CheckSurfaceMesh(const SurfaceMesh& mesh);
// move_surface_mesh.h
// Moves every vertex of last_mesh through the interval, pulls it back to field's zero set, ages it and
// reads its velocity. Changes nothing on failure. Fails: kMotionNotKnown, kPullBackFailed, kEdgeTooLong,
// kTopologyChanged, kMeshNotSafe: all mean "do not use the moved mesh".
std::expected<SurfaceMesh, RenderOutputFailure> MoveSurfaceMesh(
    const SurfaceMesh& last_mesh, const OutputInterval& interval, const SurfaceField& field,
    const SurfaceSettings& settings);
// volume_offset.h
struct BlobVolume { std::uint32_t blob_index = 0; double material_volume = 0, mesh_volume_before = 0,
                    mesh_volume = 0, offset = 0;  bool offset_is_at_limit = false; };   // m3, m3, m3, m
std::vector<BlobVolume> ApplyVolumeOffset(SurfaceMesh& mesh, const SurfaceField& field,
                                          std::span<const double> material_volume_of_blob,
                                          const SurfaceSettings& settings);
// surface_frame.h
enum class SurfaceOrigin : std::uint8_t { kNotBuilt, kBuiltFromField, kMovedFromLastFrame };
struct Droplet { std::uint64_t particle_id = 0; Vec3 position{}, velocity{}; float radius = 0; };  // m, m/s, m
struct SurfaceReport {
  SurfaceOrigin origin = SurfaceOrigin::kNotBuilt;       // Of the frame: the weakest origin of any blob.
  RenderOutputError not_built_reason = RenderOutputError::kNone;     // With kNotBuilt.
  RenderOutputError moved_mesh_reject_reason = RenderOutputError::kNone;   // First reject, by blob index.
  std::uint32_t blob_count = 0, blobs_built_from_field = 0;  std::uint64_t kept_id_count = 0;
  SurfaceMeshMeasure measure{};  std::vector<BlobVolume> volumes;
  std::uint32_t blobs_outside_volume_tolerance = 0;  double droplet_volume = 0;   // m3
};
struct SurfaceFrame { std::uint64_t material_group = 0; SurfaceMesh mesh; std::vector<Droplet> droplets;
                      SurfaceReport report; };
// The whole strategy of 5.3 for one group. Never fails for a mesh reason: the report says what happened.
std::expected<SurfaceFrame, RenderOutputFailure> BuildSurfaceFrame(
    const SurfaceFieldInputs& inputs, const OutputInterval& interval, const SurfaceMesh* last_mesh,
    const SurfaceSettings& settings, const RenderLimits& limits, OrderBuffers& order_buffers);  // CHANGED (A31).
```

### 3.5 `foam_and_spray/` headers

```cpp
// foam_settings.h
enum class FoamKind : std::uint8_t { kSpray, kFoam, kBubble };   // In air, at the surface, under it.
struct FoamSettings {
  FoamSourceSettings sources;
  double particles_per_volume_time = 1000;  // 1/(m3 s). Ported default.
  double count_per_particle = 1;       // How many real bubbles or drops one particle stands for.
  double kind_change_distance = 0.1;   // m. Above it spray, below minus it bubble; held for half of it.
  double spray_drag_rate = 0, bubble_drag_rate = 0;   // 1/s, zero or more.
  double bubble_rise_scale = 1;        // No unit.
  double foam_lifetime = 2;            // s from the particle's start; counted only while it is foam.
  Vec3 air_velocity{};                 // m/s, world. What spray is dragged toward.
  double bubble_radius = 0.001;        // m. Of one real bubble; for bubble_volume_fraction only.
  double display_radius = 0.005;       // m. Display only: the point size written to the file.
};
// foam_particles.h
struct FoamParticles {                 // Ascending id. One entry per array.
  std::vector<std::uint64_t> id;  std::vector<Vec3> position, velocity;   // m, m/s
  std::vector<double> start_time;      // s.
  std::vector<FoamKind> kind;
  std::uint64_t next_id = 1;
  std::vector<std::uint64_t> owed_block_key;  std::vector<std::uint16_t> owed_point;
  std::vector<double> owed_particles;  // 0..1 per source point, ascending (block key, point). Step's word.
};
struct FoamReport { std::uint64_t added = 0, not_added_at_limit = 0, ended_by_age = 0, left_domain = 0,
                    kind_changes = 0, bubbles_without_liquid_velocity = 0, foam_without_liquid_velocity = 0,
                    spray_count = 0, foam_count = 0, bubble_count = 0; };
// foam_motion.h (shared formulas)
struct DragMotion { Vec3 position{}, velocity{}; };
FLUIDITY_CPU_AND_GPU DragMotion ComputeDragMotion(DragMotion start, Vec3 target_velocity, double drag_rate,
                                                  double duration);            // Exact; see <eq:foam-drag>.
FLUIDITY_CPU_AND_GPU FoamKind ToFoamKind(FoamKind last_kind, double distance_to_surface, double kind_change_distance);
// add_foam_particles.h, move_foam_particles.h, remove_foam_particles.h   (particle verbs, D)
void AddFoamParticles(FoamParticles& particles, std::span<const FoamSourceEvent> events, const LatticeFrame& frame,
                      const FoamSettings& settings, const RenderLimits& limits, FoamReport& report);
void MoveFoamParticles(FoamParticles& particles, const OutputInterval& interval, Vec3 gravity,
                       const FoamSettings& settings, FoamReport& report);
void RemoveFoamParticles(FoamParticles& particles, const contact::DomainBox& domain_box, double frame_time,
                         const FoamSettings& settings, FoamReport& report);
```

### 3.6 `render_output/volume_frame.h`, `output_channels.h`, `format_table.h`, `render_frame.h`, `interchange/`

```cpp
// volume_frame.h
enum class VolumeChannel : std::uint8_t { kFoamCountDensity, kBubbleCountDensity, kBubbleVolumeFraction,
                                          kGasDensity /* NEW */, kSmokeDensity, kTemperature, kFuelDensity,
                                          kFlame };   // The five gas channels of gas decision 9.
struct VolumeFrame {                   // One volume object. Every channel shares blocks and `known`.
  LatticeBlocks blocks;                // A block exists where at least one point is known.
  LatticeValues<std::uint8_t> known;   // 1: the simulation has an answer here (it may be zero). 0: it has none.
  std::vector<VolumeChannel> channels; // Ascending. A channel that is not listed is absent, not zero.
  std::vector<LatticeValues<float>> values;   // One per listed channel. Zero where known is 0.
  std::uint64_t value_outside_known_count = 0;   // Foam particles whose 8 points were all not known.
};
std::expected<VolumeFrame, RenderOutputFailure> BuildFoamVolume(
    const foam_and_spray::FoamParticles& particles, const foam_and_spray::FoamSettings& settings,
    const LatticeFrame& frame,
    const contact::DomainBox& domain_box, std::span<const contact::Solid> solids, const RenderLimits& limits,
    OrderBuffers& order_buffers);   // CHANGED (A31): CPU, at least the foam particle count; the grouping of 5.6.
                                    // Requires: order_buffers.run_on() == RunOn::kCpu (debug assertion, A30).
// `gas` is gas::ToGasSnapshot(*snapshot->gas, *gas_law, grid_frame) of the frame's step snapshot (FrameInputs).
std::expected<VolumeFrame, RenderOutputFailure> BuildGasVolume(
    const gas::GasSnapshot& gas, const grid::GridFrame& grid_frame, const LatticeFrame& frame,
    const RenderLimits& limits);

// output_channels.h: the field contract as data. Writers, the file list and the Blender add-on read this table
// only.
enum class OutputProduct : std::uint8_t { kSurface, kDroplets, kSpray, kFoam, kBubbles, kFoamVolume, kGasVolume,
                                          kParticles /* NEW */, kSolidMotion /* NEW */ };
enum class GeometryKind : std::uint8_t { kPoints, kSurfaceMesh, kVolume, kSolidMotion };   // NEW. One format each.
constexpr GeometryKind ToGeometryKind(OutputProduct product);   // NEW. Droplets, spray, foam, bubbles: kPoints.
enum class ChannelDomain : std::uint8_t { kPoint, kFace, kVolume, kObject /* NEW: a solid's transform */ };
enum class ChannelKind : std::uint8_t { kMeasured, kReconstructed, kDisplayOnly };
enum class ChannelSpace : std::uint8_t { kScalar, kWorldPosition, kWorldVector, kLabel,
                                         kWorldRotation /* NEW: own axes to world */ };
enum class ChannelSpread : std::uint8_t { kLinear, kConstantPerElement };   // How to fill in between elements.
enum class ChannelValueType : std::uint8_t { kFloat32, kInt32, kFloat64 /* NEW: transforms */ };
// NEW. What a test in Blender must show before a format is chosen for a kind (5.7). The witness tests are run
// by the Blender add-on in Blender 5.3 (the one supported version, A16), with the add-on disabled, on files
// these writers made.
enum class WitnessProperty : std::uint8_t {
  kFileSequenceReadNatively,   // One file per frame; the native reader follows the scene frame.
  kTopologyMayChangePerFrame,  // Vertex and triangle counts differ between frames and each frame is right.
  kVelocityGivesMotionBlur,    // Cycles motion blur from the file's velocity, length = speed x shutter.
  kVelocityInSeconds,          // The velocity is read as m/s ("Second") and needs no hand-set scale.
  kPointRadiusArrives,         // The per-point width arrives as the point radius.
  kPointAttributesArrive,      // Per-point or per-vertex float and int32 values arrive as named attributes.
  kFaceAttributesArrive,       // Per-face int32 values arrive as named face attributes.
  kZUpWithoutUserRotation,     // Positions land in Blender's world as written, with no rotation by the user.
  kTransformGivesObjectMotion, // Per-frame transforms move the object and give object motion blur.
  kGridsArriveByName,          // Every float grid arrives under its name, with its voxel size and placement.
};
struct OutputChannel {
  OutputProduct product;  std::string_view blender_name;    // NEW: the attribute or grid name after import.
  std::string_view alembic_name, usd_name, openvdb_name, json_name;  // NEW: name in each format; empty: not there.
  std::string_view native_quantity;
  ChannelDomain domain;   std::string_view unit;           ChannelSpace space;
  ChannelSpread spread;   ChannelValueType value_type;       ChannelKind kind;
  std::string_view known_when;         // The rule that says where the value means something.
  WitnessProperty witnessed_by;        // NEW. The property whose witness shows this channel arriving.
};
inline constexpr int kOutputContractVersion = 2;       // 2: open formats and the format table (A14).
std::span<const OutputChannel> ReadOutputChannels();   // The table of 5.7, in its order.

// render_output/format_table.h. NEW. The format choice of A14 as data (5.7). Not in interchange/, so the
// CLI's `capabilities` and the file list can report it without the libraries.
enum class InterchangeFormat : std::uint8_t { kUsd, kAlembic, kOpenVdb, kSolidMotionJson };
enum class WitnessResult : std::uint8_t {
  kNotRun,       // Nothing known yet.
  kDocumented,   // The Blender manual states it (quoted in 5.7). Still to be run; never enough to choose.
  kPassed,       // The witness test passed. Evidence: the test name and Blender version in `evidence`.
  kFailed,
};
struct WitnessRecord {                 // One result, Blender 5.3 (A16; there is no blender_5_2).
  WitnessProperty property;
  WitnessResult blender_5_3 = WitnessResult::kNotRun;  // The result measured in Blender 5.3.
  // Manual page, or "<test>; Blender <version> <build hash>; files <digest>"; empty while kNotRun.
  std::string_view evidence;
};
// A table line in format_table.cc: {WitnessProperty::<P>, WitnessResult::<r>, "<evidence>"},
struct FormatCandidate {
  InterchangeFormat format;
  std::string_view extension;          // ".usdc", ".abc", ".vdb", ".json". I/O names the file with it.
  std::span<const WitnessRecord> witnesses;   // One per property the kind requires, in WitnessProperty order.
};
struct FormatChoice {
  GeometryKind kind;
  std::span<const FormatCandidate> candidates;   // Preference order: USD first (A14), the default last.
  InterchangeFormat chosen;            // = ChooseFormat(*this). Written out so a change is one reviewed line.
};
// The first candidate whose every record is kPassed; otherwise the last candidate. A static_assert per kind
// compares `chosen` with it, so a hand edit of one without the other does not compile.
constexpr InterchangeFormat ChooseFormat(const FormatChoice& choice);
std::span<const FormatChoice> ReadFormatTable();       // One entry per GeometryKind, in GeometryKind order.
const FormatChoice& ReadFormatChoice(GeometryKind kind);

// render_frame.h
struct RenderLimits { std::uint32_t max_lattice_blocks = 1u << 20; std::uint64_t max_vertices = 1u << 26,
                      max_foam_particles = 1u << 24,
                      max_field_parts = 1u << 26; };             // D pattern max_... . max_field_parts: NEW
                                                                 // (reconcile item 26), the parts of 5.3 step 2.
struct VolumeSettings { double spacing = 0; };   // m. 0: the smallest cell width of the scene.
struct RenderSettings { MotionSettings motion; std::optional<SurfaceSettings> surface;
                        std::optional<FoamSettings> foam; std::optional<VolumeSettings> foam_volume, gas_volume;
                        RenderLimits limits; };
// Fails kBadInput: a setting out of range, a group that is empty, repeats a material or names a material that
// is not MaterialKind::kLiquid (A9: snow is written as points only; gas is a volume). `id` is the material_id.
std::expected<void, RenderOutputFailure> CheckRenderSettings(const RenderSettings& settings,
                                                             double smallest_cell_width,
                                                             const materials::MaterialTable& materials);
class RenderOutput;                    // Move-only. What is kept from frame to frame: per group the last mesh,
                                       // the foam particles, the last motion sample, the last frame number,
                                       // and (A31) the CPU frame_order_buffers for max(max_field_parts,
                                       // max_foam_particles) items that every frame lends to the field and
                                       // the foam volume. CHANGED: fails kOutOfMemory when they cannot be built.
std::expected<RenderOutput, RenderOutputFailure> BuildRenderOutput(
    const RenderSettings& settings, const contact::DomainBox& domain_box, const grid::GridFrame& grid_frame,
    Vec3 gravity);
struct FrameIdentity { std::uint64_t frame = 0, state_version = 0; double time = 0;   // Same in every product.
                       double frames_per_second = 0; };   // NEW. 1/s. Time code rate of USD and Alembic files.
struct FrameInputs { FrameIdentity identity; const step::SimulationSnapshot* snapshot;
                     std::span<const contact::Solid> solids;   // solids::BuildContactSolids of the snapshot.
                     // NEW. The scene's gas law (ReadyScene::gas->law); null without gas. BuildRenderFrame turns
                     // snapshot->gas (a gas::GasAmountsSnapshot) into the output channels with gas::ToGasSnapshot.
                     const gas::GasLaw* gas_law = nullptr;
                     RecordedInterval recorded;  MotionSample sample_at_frame; };
struct RenderFrame { FrameIdentity identity; std::vector<surface::SurfaceFrame> surfaces;
                     std::optional<foam_and_spray::FoamParticles> foam; foam_and_spray::FoamReport foam_report;
                     std::optional<VolumeFrame> foam_volume, gas_volume;
                     // NEW. m3. recorded.liquid_totals_at_frame.current_volume minus every group's blob
                     // material_volume and droplet_volume. Near 0 when the groups hold every liquid material.
                     double liquid_volume_outside_surfaces = 0; };
// One output frame. Frames must come in order; the first has an empty `recorded`. Fails only with kBadInput,
// kWrongStateVersion, kOutOfMemory; then `output` is unchanged and the same call may be made again.
std::expected<RenderFrame, RenderOutputFailure> BuildRenderFrame(RenderOutput& output, FrameInputs&& inputs);
struct RenderOutputSnapshot {          // D word. What a saved state needs to go on with equal bytes. NEW members.
  FrameIdentity last_frame;            // The frame these values belong to.
  std::vector<std::uint64_t> group_first_material_id;  // One per SurfaceSettings group, in group order.
  std::vector<surface::SurfaceMesh> last_mesh;         // Same order; an empty mesh where none was built.
  std::optional<foam_and_spray::FoamParticles> foam;   // Owed particles included.
  MotionSample last_sample;            // The sample at last_frame, the first sample of the next interval.
};
RenderOutputSnapshot ReadSnapshot(const RenderOutput& output);
std::expected<RenderOutput, RenderOutputFailure> BuildRenderOutputFromSnapshot(
    const RenderSettings& settings, const contact::DomainBox& domain_box, const grid::GridFrame& grid_frame,
    Vec3 gravity, const RenderOutputSnapshot& snapshot);
std::vector<MemoryEstimateEntry> BuildRenderOutputMemoryEstimate(const RenderSettings& settings, RunOn run_on);

// interchange/point_objects.h. NEW. Points as every points writer takes them. CPU only.
struct PointChannelValues {            // One named per-point channel of 5.7. Exactly one span is non-empty.
  const OutputChannel* channel;        // Row of ReadOutputChannels(); gives the names in each format.
  std::span<const float> float_values; // components x point count, point-major.
  std::span<const std::int32_t> int_values;
  std::uint8_t components = 1;
};
struct PointObject {                   // One object in a points file. Borrowed for the call.
  std::string_view object_name;        // "points", "droplets", "spray", "foam", "bubbles".
  std::span<const Vec3> position;      // m, world, Z-up.
  std::span<const Vec3> velocity;      // m/s, world. Same count as position.
  std::span<const float> radius;       // m. Same count, or empty with constant_radius.
  float constant_radius = 0;           // m.
  std::span<const std::uint64_t> id;   // Written as the low 31 bits (5.7).
  std::span<const PointChannelValues> channels;
};
struct PointObjectBuffers;                   // Float copies made by the ToPointObject builders; reused frame to frame.
// The points of one material of an output snapshot: position, velocity_for_next_step, radius
// cbrt(3 rest_volume volume_ratio / (4 pi)), id, volume_ratio, size_level, share_<ingredient> with two or more
// ingredients, and every field of `extra_fields` (I/O's RequestedOutputs::particle_fields) under its 5.7 name.
// Fails: kBadInput (a requested field the snapshot does not carry).
std::expected<PointObject, RenderOutputFailure> ToPointObject(
    const particles::ParticlesSnapshot::Material& material, particles::FieldList extra_fields,
    std::span<const std::string> ingredient_export_names, PointObjectBuffers& point_object_buffers);
PointObject ToPointObject(std::span<const surface::Droplet> droplets, PointObjectBuffers& point_object_buffers);
// Three objects, spray, foam, bubbles (empty objects included), age at frame_time.
std::array<PointObject, 3> ToPointObjects(const foam_and_spray::FoamParticles& particles,
                                          const foam_and_spray::FoamSettings& settings, double frame_time,
                                          PointObjectBuffers& point_object_buffers);

// interchange/write_points_file.h, write_surface_file.h, write_volume_file.h, write_solid_motion_file.h.
// Target fluidity_interchange only. Each writes one frame of one product into an io::OutputFile that I/O
// opened with the extension of ReadFormatChoice(kind) and lists after the call; the libraries write to
// file.path_while_writing(). The file holds one time sample at time code identity.frame, with
// timeCodesPerSecond (USD) or the time sampling (Alembic) of identity.frames_per_second. Fails:
// kFormatNotInBuild (format is not a candidate of the kind in this build: a failed candidate's writer is
// removed, 5.7), kFileWriteFailed, kBadInput. On failure the file is left for I/O to discard.
std::expected<void, RenderOutputFailure> WritePointsFile(InterchangeFormat format,
    std::span<const PointObject> objects, const FrameIdentity& identity, io::OutputFile& file);  // USD, Alembic
std::expected<void, RenderOutputFailure> WriteSurfaceFile(InterchangeFormat format,
    const surface::SurfaceFrame& surface, const FrameIdentity& identity,
    std::span<const std::string> ingredient_export_names, io::OutputFile& file);                // USD, Alembic
std::expected<void, RenderOutputFailure> WriteVolumeFile(const VolumeFrame& volume, const FrameIdentity& identity,
    io::OutputFile& file);                                                                       // OpenVDB
// Every solid in the scene at the snapshot: solid_id, origin, orientation, linear and angular velocity.
std::expected<void, RenderOutputFailure> WriteSolidMotionFile(InterchangeFormat format,
    const solids::SolidsSnapshot& solids, const FrameIdentity& identity,
    io::OutputFile& file);                                                       // USD, Alembic, JSON
// Library versions linked into fluidity_interchange, for the run report and `capabilities`: each the version
// pinned at build time (A19, 5.7), a literal guarded by a static_assert on the library's own version macro.
struct InterchangeLibraries {
  std::string_view open_usd, alembic, openvdb;
  std::string_view missing_resource_folder;   // NEW. Empty when OpenUSD found its plugin resources.
};
InterchangeLibraries ReadInterchangeLibraries();
```

**For `shared-interface.md` section (c):** nothing was required of this module. Everything offered to it is
used: `pressure::GeometrySnapshot` and `SurfaceDistances` (through the step callback), particle snapshots,
and blob labels on a surface (`SurfaceVertex::blob_index`, per frame). The gas design's need, "writes
`GasSnapshot` as five volume channels and the known mask" (gas decision 9), is `BuildGasVolume` and
`WriteVolumeFile`. Since A14 this module also writes what I/O wrote before: the playback of the simulation
particles (`ToPointObject` of a particles material, `WritePointsFile`).

## 4. Consumes from other modules

Every item is EXISTS (Foundation, including the `base/` additions) or PROVIDED (by a designed module, with
the name its section 3 has; for the designs updated at the same time or after this one, with the interface
`changes-to-apply.md` gives them, as noted).

**Foundation (EXISTS, shared-interface (a) and the `base/` additions of `changes-to-apply.md`):** `Vec3`,
`Mat3`, `math::Floor`, `Sqrt`, `Cbrt`, `Exp`, `Expm1` (drag formula at small rates; `main` uses
`std::expm1`), `Min`, `Max`, `Length`, `Normalized`, `IsFinite`, `math::TreeSum`, `RunOn`, `CheckCanRunOn`,
`FLUIDITY_CPU_AND_GPU`, `cpu_only.h`, `std::expected`, `Buffer`, `BuildBuffer`, `ReadBuffer`, `CopyBuffer`,
`BuildStableOrder`, `BuildGroupsByDestination`, `MemoryEstimateEntry`, `CountedPer::kGivenCount`, `KeptFor::kWholeRun`,
`kOneCall`, `kOneSnapshot`.

**Grid (PROVIDED, grid 3):** `StaggeredGrid`, `GridTables`, `grid.tables(run_on)`, `GridFrame`, `GridVersion`,
`CellBlockRange`, `CellBlockIndex`, `CellPlaceIndex`, `CellId`, `ToCellId`, `FindCellPlace`, `ToSizeLevel`,
`ComputeCellWidth`, `FindCornerPlaces`, `CornerValues`, `CellValues`, `FaceValues`, `FindNearbyFaces`,
`ToMaskOfAll`, `NearbyFaceMask`, `ComputeNearbyFaceWeights` (with `NearbyFaceWeights::weight_sum` and `fit`),
`ComputeFaceWeight`, `NearbyFaceFit::kNone`, `BorderWidth::kFourCellWidths`. The piece map is not read from
`GridPiece::block_in_whole_grid()` (empty for a GPU piece, A17) but from `FoamAndSprayInputs::block_in_whole_grid`
(step, below). In `RecordStep` only, except `CellId`, `ToCellId`, `ComputeCellWidth` (worker, from snapshots).

**Contact (PROVIDED, contact 3):** `Solid`, `SolidPose`, `ComputeSolidPose`, `ComputeDistanceToSolid`,
`DistanceToSolid`, `ComputeSolidVelocity`, `DomainBox`.

**Particles (PROVIDED, particles 3.7):** `ParticlesSnapshot` and its `Material` arrays (`id`, `rest_volume`,
`size_level`, `ingredient_fractions`, `position`, `velocity_for_next_step`, `volume_ratio`, and `mass` or any
other array asked for), `SnapshotFields`, `kOutputSnapshotFields`, `FieldList`, `Field`, `HasField`,
`ParticlesTry` (passed through to `cell_resize::ComputeLiquidTotals`).

**Transfer (PROVIDED, transfer 3):** `GridSnapshot` (`velocity`, `value_kind`, `outside_solid`, `material_set`,
`grid_version`), `FaceValueKind` (`kNoValue` is the only kind without a value; `kFromParticles`,
`kTooFewParticles`, `kFromSolve`, `kFromGridOnlyRegion` all carry one), `MaterialSetKey`.

**Pressure (PROVIDED, pressure 3):** `GeometrySnapshot`, `CellGeometry::distance_to_surface`,
`SurfaceDistances::at_corners` of the `kAllLiquidMaterials` field.

**Materials (PROVIDED, materials 3):** `MaterialTable::materials()` (`material_id`, `response.kind`, to check
that a `SurfaceGroup` holds liquids only), `first_ingredient_of_material()`, `ingredient_export_names()`
(already normalised; the share of ingredient `i` is exported as `share_<ingredient_export_names()[i]>`),
`MaterialKind`, `gas_ingredient_index()` (which material is gas: never meshed, never written as points).

**Solids (PROVIDED, solids 3, updated at the same time; interface as `changes-to-apply.md` 08 states and as
solids 3 now declares it):** `std::vector<contact::Solid> BuildContactSolids(std::span<const SolidSettings>
settings, const SolidsSnapshot& snapshot)` for `FrameInputs::solids` and the samples; `SolidsSnapshot`
(`time`, `solid_id`, `state` with `SolidState::origin`, `orientation`, `linear_velocity`, `angular_velocity`,
`is_in_scene`), `Orientation`, `ToAxes` for `WriteSolidMotionFile`.

**Gas (PROVIDED, gas 3):** the output `GasSnapshot { time; state_version; grid_version; cells;
gas_volume_share; gas_density; temperature; fuel_density; flame; smoke_density; }` (`changes-to-apply.md` 09:
"the output `GasSnapshot` stays for volumes"), five channels and the known mask (gas decision 9); `GasLaw`;
`GasAmountsSnapshot` (the gas part of `step::SimulationSnapshot`, never read field by field here) and
`GasSnapshot ToGasSnapshot(const GasAmountsSnapshot& amounts, const GasLaw& law, const grid::GridFrame& frame)`, which `BuildRenderFrame`
calls on `snapshot->gas` with `FrameInputs::gas_law` before `BuildGasVolume`.

**Cell resize and batches (PROVIDED, cell resize and batches design 3.5):**
`cell_resize::LiquidTotals { mass; rest_volume; current_volume; }`,
`ComputeLiquidTotals(const particles::ParticlesTry&, const GridOnlyAmounts&, std::span<const std::uint8_t>
material_is_counted, const grid::StaggeredGrid&, grid::CellBlockRange, SumBuffers&)` (one form since A31, given
the recorder's `sum_buffers`; called with the liquid set's mask, so
gas markers are left out), `GridOnlyAmounts` (built in `RecordStep` from the try `GridFields`). Its design
asks this module to count grid-only cells as liquid, to take volume targets from `ComputeLiquidTotals`, and
to do no whole-grid sum inside one `make_foam_and_spray` call: 5.2, 5.3 step 1 and 5.

**Step (PROVIDED by the step design after its update; the first four additions are in `changes-to-apply.md`
10, the rest are asked here and listed in the hand-over report):**

```cpp
struct FoamAndSprayInputs {            // step 3.9. Lines marked + are the additions.
  const grid::StaggeredGrid* grid;  const pressure::GeometrySnapshot* geometry;
  const grid::FaceValues<double>* velocity_after_pressure;  const particles::ParticlesTry* particles_try;
  double step_start_time = 0, step_duration = 0;  std::uint64_t try_number = 0;
+ // The liquid material set's snapshot: exactly one entry, since all liquids are in one set (step decision
+ // of 2026-11-14); snow, springy-solid and gas marker sets are not passed.
+ std::span<const transfer::GridSnapshot* const> after_pressure;
+ const pressure::SurfaceDistances* surface_distances; // The kAllLiquidMaterials field at cell corners.
+ std::span<const contact::Solid> solids;              // The try's solids (solids::SolidsTry::solids()).
+ std::uint64_t step_count = 0;                        // The number this step gets if it commits.
+ grid::CellBlockRange blocks;                         // Blocks of `grid` to record (a batch: the piece's range).
+ std::span<const grid::CellBlockIndex> block_in_whole_grid;  // Of `grid`'s blocks. Empty in a one-piece run;
+   // in a batched run the batched run's CPU copy of the piece map (read back per load), never the empty span of
+   // a GPU piece (plan decision, 2026-11-14, buffers in batched runs).
+ const GridFields* fields;                            // The try GridFields with cell resize's region arrays.
};
struct SimulationSnapshot { ...
+ std::vector<std::uint8_t> is_grid_only_region;       // By active cell, in active_cells order.
+ std::vector<double> grid_only_current_volume;        // m3, by active cell. Saved with the region arrays.
+ std::vector<double> grid_only_ingredient_volume;     // m3, [active cell * ingredient_count + ingredient].
};
enum class Module { ..., kRenderOutput };              // + and MemoryEstimate::by_module grows by one entry.
std::expected<RunPlan, StepFailure> BuildRunPlan(const ReadyScene&, const StepSettings&, MemoryAvailable,
    MemoryAvailable, std::span<const MemoryEstimateEntry> output_estimate /* + */,   // BuildRenderOutputMemoryEstimate.
    std::span<const MemoryEstimateEntry> cell_resize_estimate = {});
// NEW use (plan decision, 2026-11-14, buffers in batched runs), step run_plan.h:
struct CopyCounts { std::uint32_t copy_count = 0; std::uint64_t bytes_to_processor = 0; std::uint64_t
  bytes_from_processor = 0; };
// The step keeps the map for stage 18 (replaces the two-argument form; batches plan Task 36): the piece map in
// CPU memory, one entry per piece block, else kBadInput; stage 18 passes it as FoamAndSprayInputs::block_in_whole_grid.
std::expected<void, StepFailure> ApplyGridPiece(StepTry& loaded_try, const grid::GridPiece& piece,
                                                std::span<const grid::CellBlockIndex> block_in_whole_grid);
```

`StepCallbacks` is used as declared: each member is a `Callback<Function>{function, context}` pair, and I/O,
which composes the modules, sets `make_foam_and_spray` and `on_try_done` with a context of its own that holds
the `StepRecorder` (no context is shared with cell resize or batches). Its `make_foam_and_spray` calls
`RecordStep` (once per batch in a batched run) and returns `false` only on `kRunOnCallFailed`; its
`on_try_done` calls `CommitRecordedStep` or `RejectRecordedStep`. Also used: `Simulation::step_count`,
`step::ComputeRandomWords` (`random_words.h`, shared), `step::ReadSnapshot(simulation,
particles::kOutputSnapshotFields)` (plus I/O's requested fields), `GridFields`. Step decision 8: "two copies and
a blocking snapshot".

**I/O and CLI (PROVIDED by the I/O design after its update, as `changes-to-apply.md` 12 states it):**

```cpp
namespace fluidity::io {
class OutputFile {                     // A file that is not seen by any reader until I/O lists it.
 public:
  std::expected<void, IoFailure> Write(std::span<const std::byte> bytes);
  const Path& path_while_writing() const;   // For the USD, Alembic and OpenVDB libraries.
};
}
```

Also `BuildOutputFile`, `DiscardOutputFile`, `WriteListedFile(RunFolder&, OutputFile&&, CheckAfterWrite)` with
`kSizeOnly` (the one transaction that lists a file), `WriteNotBuilt`, the bounded output queue and its worker, the
`file_list`, the word `listed`, `RequestedOutputs::particle_fields`, `IoFailure`.

1. The scene file gives a `render_output::RenderSettings` and the frame times; I/O calls `CheckRenderSettings`
   in its ready-scene check and passes `BuildRenderOutputMemoryEstimate` to `step::BuildRunPlan`.
2. I/O owns the `StepCallbacks` and composes them as said above.
3. At each output frame, after the commit and its resize: I/O calls `step::ReadSnapshot`,
   `ReadRecordedInterval`, builds `FrameInputs` (with `gas_law` from `ReadyScene::gas` in a scene with gas)
   and puts it in its bounded queue. One worker takes frames in
   order and calls `BuildRenderFrame`, then the writers (points for every particle material that is not gas,
   surface and droplets per group, foam particles, the two volumes, solid motion), each into its own
   `OutputFile` opened with the extension of `ReadFormatChoice(kind)` and the file name pattern of 5.7, then
   lists each with `WriteListedFile`. The internal motion record I/O keeps for a continued run stays in its
   own array file.
4. When the queue is full the simulation waits. No frame is dropped and none is built out of order: the
   foam particles and the moved mesh of frame `n` need frame `n - 1`. The wait is reported as its own time.
5. The `file_list` lists, per frame and product, the file, `FrameIdentity`, `run_fingerprint`,
   `kOutputContractVersion`, the format and the channels present, and the `SurfaceReport` or `FoamReport`. A
   product that was asked for and not built is listed as not built with its reason (`ToName`); one that was not
   asked for is not listed.
6. A saved state written at frame `n` stores `ReadSnapshot(render_output)` taken after frame `n` was built,
   so I/O lets the worker reach frame `n` first. Without it a run goes on with empty foam and a newly built
   mesh, and says so.
7. A failed `BuildRenderFrame` or writer never stops the simulation: I/O lists the frame's products as not
   built and goes on.
8. The CLI links the target `fluidity_interchange` (USD, Alembic and OpenVDB libraries, 5.7); nothing else
   does. `capabilities` reports `ReadFormatTable()` and `ReadInterchangeLibraries()` (with
   `missing_resource_folder`).
9. NEW (seam 11, plan decision, 2026-11-14, buffers in batched runs and planned cost): the run's `on_step_done`
   adds the difference of `ReadRecorderCopyCounts` since the last committed step to the copies
   `io::ApplyStepToRunCost(const step::MemoryReport&, const step::CopyCounts& resize_and_recorder_copies, RunCost&)`
   receives (I/O plan Task 29a), and `BuildRenderOutputMemoryEstimate` stays the plan's `output_estimate`, so the
   budget (`io::BuildSceneBudget`) and the measured cost both count the recorder.

**Blender add-on (PROVIDED by the Blender add-on design after its update, `changes-to-apply.md` 14):**
native readers for USD, Alembic and OpenVDB; the witness tests of 5.7, run in Blender 5.3 with the add-on
disabled on files these writers made, each result reported back as one `WitnessRecord` line of the format table
(its `write_witness_records.py` prints the line shape of 3.6 and refuses results from any other Blender version);
`capabilities` prints each record as `{"property", "blender_5_3", "evidence"}`, the keys the add-on's
`WitnessRecordFact` reads; one accessor for a product's `FrameIdentity` and `run_fingerprint` read from the `file_list`;
channel names read from `ReadOutputChannels` through the `file_list`, never typed in; "Velocity Unit: Second"
on an Alembic cache reader; the flame and temperature scales of the volume material.

## 5. Method

### 5.1 Two halves and the one-way rule

1. **Inside the step** (stage `MakeFoamAndSpray`, on the simulation's processor): `RecordStep` evaluates the
   try on a lattice near the surface and finds foam source events. It writes only into the recorder.
2. **After a commit:** `CommitRecordedStep` keeps what the try found; a reject forgets it. So a rejected try
   leaves the recorder as it was, like every other owner of state.
3. **At an output frame:** a snapshot and the recorded interval go to the worker. `BuildRenderFrame` builds
   the surface, moves the foam, counts the volumes; the writers of `interchange/` turn the frame, the
   particles and the solids of the snapshot into open-format files (5.7). CPU only.
4. Invariant, tested: no function of this module takes a non-const simulation, particles, grid or solids.
   The surface field and mesh are never an input of any earlier module.

### 5.2 The motion sample (new; replaces full-state copies)

For the range `FoamAndSprayInputs::blocks` (the whole grid in one piece, or the range of one loaded piece in
a batched run), with a border of two lattice spacings (covered by `BorderWidth::kFourCellWidths` because
`spacing <= 2 * smallest cell width`, and loaded with the piece by the batches module):

1. List the lattice points of every active cell whose centre has `|distance_to_surface| <= reach + half the
   cell's diagonal`. A point belongs to the one cell whose half-open box holds it. Order: cell place, then
   z, y, x inside the cell. Two passes (count, then fill) give the same list on CPU and GPU.
2. `distance_to_surface` at a point: trilinear in the 8 corner values of its cell
   (`SurfaceDistances::at_corners`). Points farther than `reach` are dropped.
3. Velocity at a point, per axis, from the liquid material set's snapshot (the one entry of `after_pressure`:
   all liquids are in one set and share one grid velocity; two liquids that do not mix are kept apart by
   pressure's material boundary, not by sets): `FindNearbyFaces`, keep the faces with
   `value_kind != kNoValue` and `outside_solid`, `ComputeNearbyFaceWeights`, then `sum weight * velocity` in
   ascending list position. This is the interpolation `GridToParticles` uses, so a lattice value is the grid
   velocity at that point. Snow, springy-solid and gas marker sets are never read (A10: gas moves no foam).
   `fit == kNone` on any axis marks the point not known. Plate
   sides are not told apart here (a known limit: two liquids on the two sides of one thin plate share lattice
   velocities within 1.5 cell widths of it).
4. Values are stored as `float`.
5. A sample is read back to CPU memory only for the first commit in each of `samples_per_frame` equal parts
   of a frame, and for the commit at the frame time. Every other step reads back events only.
6. Batches. Points and events are made only for points inside the range's own cells, so no batch repeats one.
   `CommitRecordedStep` orders the points of every call of the try by (lattice block key, point in block)
   with `BuildStableOrder`; keys are unique, so the order does not depend on where batches were cut.
7. Liquid totals (new, for 5.3 step 5). On the try whose end is a frame time, `RecordStep` calls
   `cell_resize::ComputeLiquidTotals` once per cell block of its range (the try's `particles_try` and a
   `GridOnlyAmounts` view of the try `GridFields`) and keeps the block partials with their whole-grid block
   index. `CommitRecordedStep` sums each total with `math::TreeSum` over the partials in ascending whole-grid
   block index: the order of a one-piece run, whatever the batches. `ResizeCells` after the commit moves
   liquid between regions without changing these totals (its own ledger check), so they hold for the
   snapshot the worker receives.

### 5.3 The surface: one strategy

**Field (ported idea, three fixes).** Per group, on a lattice of `spacing`:

1. Field points: every particle of the group, with `volume = rest_volume * volume_ratio` and
   `width = cbrt(volume)`; every active cell marked `is_grid_only_region` in the snapshot counts as liquid,
   as one point at the cell centre with `volume = grid_only_current_volume` and the cell's width, its shares
   from `grid_only_ingredient_volume` (fix 1: the liquid without particles is put in the same sum, so the
   smoothed share is continuous where the two regions meet and no inner surface appears). These are the two
   amounts `cell_resize::ComputeLiquidTotals` sums as `current_volume`.
2. A field point wider than the spacing is split into `k^3` equal parts on a regular pattern in its box,
   `k = ceil(width / spacing)`, at most 8 (fix 2: particles of big cells do not leave gaps).
3. Each part adds its volume to the 8 lattice points around it with trilinear weights. Order of the sum at
   one lattice point: parts ordered by (lattice cell, material index, grouped order of the particle, part
   number), through `BuildStableOrder`; the point reads the 8 cells around it in a fixed order.
4. `liquid = G * added_volume / spacing^3`, with `G` the separable Gaussian of `smoothing_width` cut at
   `round(4 * smoothing_width / spacing)` points (ported).
5. `open = G * open_share`, where `open_share = clamp(0.5 + distance_to_solid / spacing, 0, 1)` from
   `ComputeDistanceToSolid` over all solids and the domain sides (fix 3: any solid, any motion).
6. `value = 0.5 - liquid / open` where `open >= 0.05`, else `+0.5`. Then the liquid is cut at solids:
   `value = max(value, -clamp(distance_to_solid / spacing, -0.5, 0.5))`. Where liquid touches a solid, the
   zero set lies on the solid's surface, and those triangles are `kLiquidToSolid`.
7. Ingredient shares: the same sum per ingredient, divided by `liquid`.

`<eq:surface-field>`: `value(x) = 0.5 - (G * V)(x) / (G * open)(x)`.

**Mesh from the field (ported, one fix).** Six tetrahedra per cube with one diagonal rule; a value within
`1e-8` of zero counts as `+1e-8` (ported tie rule); a vertex per crossed lattice edge, shared through the
edge's key; triangles counter-clockwise seen from the positive side. Fix: crossings are never merged into a
lattice corner, so each tetrahedron's patch stays inside it and the result is closed, has two triangles per
edge and no crossing pair by construction. Vertices are ordered by edge key, triangles by (cube, tetrahedron).
Absent blocks are air; a block exists wherever `liquid > 0` can occur, so no crossing leaves the blocks.

**Frame strategy.** One rule, no local repair:

1. Build the field. Build the mesh from it: `built`. Label blobs: lattice points with `value < 0` joined
   through the six axis neighbours; blob index by lowest (block key, point). Each closed piece of a mesh
   belongs to the blob it bounds.
2. If there is a last mesh and `keeps_ids_between_frames`: `moved = MoveSurfaceMesh(last, interval, field)`.
   a. Each vertex: `ComputeMovedPoint` over the interval. Motion not known: reject the vertex's blob.
   b. Pull back: up to 12 Newton steps `x <- x - value * value_change / |value_change|^2`, each halved up to
      16 times until `|value|` falls; total move at most `max_pull_back_in_spacings * spacing`; done when
      `|value| / |value_change|` is at most `1e-5 * spacing` or 1 % of the smallest height of a triangle at
      the vertex (ported numbers). Not done: reject the blob.
   c. Any edge longer than `max_edge_in_spacings * spacing`: reject the blob (new; without it a stretched
      mesh loses detail and still passes).
   d. Per blob: the moved pieces and the built pieces must agree in count and Euler number, else reject.
3. Take, per blob, the moved pieces when the blob was not rejected, else the built pieces. Run
   `CheckSurfaceMesh` on the whole. If it is not safe (a kept piece can cross a built one), take `built` for
   every blob. `built` is safe by construction, so this is the end: at most two levels, no loop.
4. New vertices get new ids in vertex order and the age of the nearest moved old vertex within one spacing
   (lowest id on a tie), else 0. `liquid_velocity` is read from the sample at the frame time;
   `distance_to_solid` and shares from the field.
5. Volume: per blob, `material_volume` is the sum of the volumes of the field points whose position has
   `value < 0` in that blob. One offset along vertex normals for the vertices that are not at a solid, at
   most 3 passes of `offset += (material_volume - mesh_volume) / free_area`, limited to `0.25 * spacing`;
   a pass that makes the mesh unsafe is halved once, then dropped. Both volumes are reported. Being outside
   `volume_tolerance` is a count in the report (fix: on `main` it is a failed frame). The frame's volume
   target is `RecordedInterval::liquid_totals_at_frame.current_volume` (`cell_resize::ComputeLiquidTotals`,
   5.2 step 7): every group's blob `material_volume` plus `droplet_volume` is compared with it and the
   difference is `RenderFrame::liquid_volume_outside_surfaces` (liquid of materials in no group, and round-off).
6. Droplets: a particle whose position has `value >= 0` lies outside every mesh. It is written as a droplet
   with `radius = cbrt(3 volume / (4 pi))`, so liquid never vanishes from the picture. Its volume is
   reported apart and is in no blob's `material_volume`.
7. Reaching `max_vertices`, `max_lattice_blocks` or `max_field_parts`: the group has `origin = kNotBuilt` with the reason;
   the other products of the frame are built.

**Several materials in a group.** One outer mesh. `triangle_material_index` is the material with the largest
share at the triangle's centre. For exactly two materials that do not mix, the inner sheet between them is
the zero set of `share_a - share_b` inside the liquid, built with the same tetrahedra and sharing the seam
vertices (ported from `ExtractSurfacePartition3D`), `kBetweenTwoMaterials`; it is built from the field every
frame. With three or more, no inner sheet (decision 8). Snow has no surface: it is written as points, as on
`main` (decision 7); `CheckRenderSettings` rejects a group with a material that is not `kLiquid`.

**Gas beside liquid (A10).** A frame of a scene with liquid and gas holds the liquid's surface groups and
the gas volume together, with the same `FrameIdentity`. The gas volume is known only where a gas cell holds
gas (5.6 step 2); cells of liquid are not gas cells, so the volume ends at the liquid, and the mesh never
reads gas. Gas marker particles are not field points and are never written as points.

### 5.4 Foam sources (ported formulas, new inputs)

On the motion lattice of the try, for each point `c` with `-2 spacing <= distance_to_surface <= spacing`
and a known velocity (ported from `ComputeWhitewaterSource3D`, old paper section "Whitewater"):

```text
Phi(I, ramp) = clamp((I - ramp.low) / (ramp.high - ramp.low), 0, 1)
I_speed = |u_c|^2 / 2
I_air   = sum over the 26 neighbours j with a known velocity of
          (|u_c - u_j| - (u_c - u_j) . x_cj) * max(0, 1 - |x_c - x_j| / (2 spacing)),   x_cj the unit vector of main
I_crest = max(0, curvature_c)  when  u_c . n_c >= 0.6 |u_c|  and  |u_c| > 0,  else 0
strength = Phi(I_speed) * (trapped_air_share * Phi(I_air) + wave_crest_share * Phi(I_crest))        <eq:foam-source>
volume_time = ComputeOccupiedShare(d_c, spacing) * spacing^3 * strength * step_duration
```

Changed, because `main` took them from a sphere: `n_c` is the central difference of `distance_to_surface`
over two spacings, made unit length; it is not known when its length before that is below 0.5.
`curvature_c` is the sum over the three axes of the central difference of that axis' part of `n`
(`2 / R` on a sphere, as on `main`). A point without a known normal or neighbour normal has `I_crest = 0` and
is counted. `ComputeOccupiedShare = clamp(0.5 - d / spacing, 0, 1)` replaces the eight probes of `main`.
One event per point with `volume_time > 0`. Above `max_events_per_step` the events with the largest
`volume_time` are kept (ties: lower block key, then point) and the rest counted. The choice is made in
`CommitRecordedStep` over the events of every call of the try, so it does not depend on batches; a single
call that finds more than `max_events_per_step` keeps its own largest ones first, which loses nothing, since
the whole-step choice ranks by the same key. Neighbours are read in x, y, z order, so the sum has one order.

### 5.5 Foam particles (ported life cycle, four fixes)

Per output frame, in this order:

1. `AddFoamParticles`: events in order. `n = floor(particles_per_volume_time * volume_time /
   count_per_particle + owed)`, the rest stays owed to that lattice point. Ids rise in event order. Fix:
   each particle starts at `position + spacing * (u - 0.5)` per axis with `u` from
   `ComputeRandomWords({block key, point, step_count, number in event})`, velocity as the event. At
   `max_foam_particles` nothing more is added and the rest is counted.
2. `MoveFoamParticles`: each particle from `max(start_time, last frame time)` to the frame time, in pieces
   that end at sample times and move at most half a spacing:
   - spray: `ComputeDragMotion` toward `air_velocity + gravity / spray_drag_rate`;
   - bubble: toward `liquid_velocity - bubble_rise_scale * gravity / bubble_drag_rate`; where the liquid
     velocity is not known, toward that formula with zero liquid velocity (ported), counted;
   - foam: midpoint rule with the liquid velocity. Fix: where it is not known the particle keeps its
     velocity for that piece and is counted, instead of failing the frame;
   - any solid: a particle inside a solid at the end of a piece is put on its surface and loses the speed
     into it, relative to `ComputeSolidVelocity` (fix: any solid; `main` had one fixed plane).
   `<eq:foam-drag>`: `v+ = v_t + (v - v_t) e^(-k dt)`, `x+ = x + v_t dt + (v - v_t) (1 - e^(-k dt)) / k`,
   with `math::Expm1` and the exact `k = 0` form (ported).
3. Kind, at the frame time, from `distance_to_surface` `d` and `r = kind_change_distance`: spray above
   `+r`, bubble below `-r`, foam between; a particle changes kind only when it is `0.5 r` past the limit
   (ported). Where the distance is not known the kind stays.
4. `RemoveFoamParticles`: foam older than `foam_lifetime`; any particle outside the domain box (fix: `main`
   kept them and the volume box then failed).

Particles never read one another, so any thread order gives the same bytes. Foam reads the simulation only;
the surface never reads the foam, and the liquid never reads either.

### 5.6 Volumes and "known"

1. Foam volume: lattice of `VolumeSettings::spacing`. `known = 1` at points inside the domain box and
   outside every solid. Each foam or bubble particle adds `count_per_particle` to its 8 points with
   trilinear weights; weights of points that are not known are given to the known ones; if none is known
   the particle is counted in `value_outside_known_count`. `foam_count_density` and `bubble_count_density`
   are the sums over `spacing^3`; `bubble_volume_fraction` is the bubble sum times
   `4/3 pi bubble_radius^3` over `spacing^3`. Spray adds nothing (ported). Sum order: by lattice point, then
   particle id (`BuildGroupsByDestination`).
2. Gas volume: each lattice point takes the values of the gas cell that holds it (`GasSnapshot::cells`,
   binary search by `CellId` over the size levels present). `known = 1` exactly where such a cell exists and
   its `gas_volume_share` is above 0; a liquid cell, a cell of solid or a cell outside the gas region is
   `known = 0` (A10: the gas volume ends at the liquid surface). Five channels: `gas_density`,
   `smoke_density`, `temperature`, `fuel_density`, `flame` (gas decision 9). With the default
   spacing a smallest cell is one point and a cell one size bigger is 8 equal points, so
   `sum value * spacing^3` over known points equals `sum value * cell volume`.
3. Three states, kept apart everywhere: **zero** is `known = 1` and value 0; **unknown** is `known = 0` (the
   value is stored as 0 and means nothing); **absent** is a channel that is not in `channels`, or a product
   that is not in the `file_list`. In the file only nonzero values are stored; `known` is its own data.

### 5.7 Formats, the format table, the field contract, and what is not promised

**Rule (A14).** Everything a person or another program plays back or renders is written in USD, Alembic or
OpenVDB and is readable with the add-on disabled. FluidiTy's array file is never a product. Where two open
formats could carry a kind of geometry, the format is the one Blender's native reader consumes fully, proven
by a witness test in Blender, and USD when both pass. The choice is data: `ReadFormatTable()` holds one
`FormatChoice` per `GeometryKind`; no writer, file list entry or add-on code picks a format except through it.
Changing a format is a change of one `WitnessRecord` line and of `chosen` in `format_table.cc`, reviewed with
the witness evidence; `FormatTable, ChosenFollowsWitnesses` (7) fails if `chosen` disagrees with
`ChooseFormat`.

**Files.** One file per frame and product, written once into an `io::OutputFile` and then listed by I/O, never
rewritten: a running bake can be viewed, and `main`'s repeated whole-range export is gone. File names, below
I/O's run folder; the extension is the chosen candidate's:

| Product | File | Objects in the file |
|---|---|---|
| Simulation particles, one per material that is not gas | `points_<material_id>/points.<frame:06d>` | `points` |
| Surface, one per group | `surface_<lowest material_id of the group>/surface.<frame:06d>` | `surface` |
| Droplets, one per group | `droplets_<lowest material_id>/droplets.<frame:06d>` | `droplets` |
| Foam particles | `foam_and_spray/foam_and_spray.<frame:06d>` | `spray`, `foam`, `bubbles` (each may be empty) |
| Foam volume, gas volume | `foam_volume/foam_volume.<frame:06d>.vdb`, `gas_volume/gas_volume.<frame:06d>.vdb` | one grid per channel and `known` |
| Solid motion | `solid_motion/solid_motion.<frame:06d>` | one transform per solid in the scene, `solid_<solid_id>` |

**One frame in each format.** Every file also stores `FrameIdentity`, `kOutputContractVersion` and the
channel names as file metadata (USD layer `customLayerData`, Alembic archive metadata, OpenVDB file
metadata, JSON fields).

- *USD* (`.usdc`): one layer per frame, `upAxis = "Z"`, `metersPerUnit = 1`, `timeCodesPerSecond` and
  `framesPerSecond` = `identity.frames_per_second`, `startTimeCode = endTimeCode = frame`. Every value is one
  time sample at time code `frame`, so each file is right alone and a reader that follows the sequence sees
  an animated attribute. Mesh: `UsdGeomMesh` with `points`, `faceVertexCounts` (all 3), `faceVertexIndices`,
  `orientation = rightHanded` (counter-clockwise seen from outside), `velocities` (m/s; USD defines them per
  second, to verify against `timeCodesPerSecond`), per-vertex channels as `vertex` primvars, per-face
  channels as `uniform` primvars. Points: `UsdGeomPoints` with `points`, `velocities`, `widths` (diameter),
  `ids` (the schema's int64) and the int32 primvar `id`, because Blender imports no int64 primvar. Only float
  and int32 primvars are written: the Blender manual lists int64, unsigned, double and half primvars as not
  imported. Solids: one `UsdGeomXform` per solid with `xformOp:translate` (double3) and `xformOp:orient`
  (quatd). Written with the OpenUSD release pinned at build time (A19); that Blender 5.3 reads the crate format
  is shown by the witness tests (W1), not by matching Blender's bundled version.
- *Alembic* (`.abc`, Ogawa): one archive per frame with a time sampling of one sample at the frame time.
  Alembic is Y-up and Blender turns it on import, so positions and velocities are written as `(x, z, -y)` and
  triangles in reverse order (`blender_bridge.cc:505` did the latter): the module's only axis conversion.
  Mesh: `OPolyMesh` with `.velocities`, per-vertex channels in `arbGeomParams` with `kVertexScope`, per-face
  ones with `kUniformScope`. Points: `OPoints` with `.velocities`, `.widths` (diameter), ids. Solids:
  `OXform` per solid, a double matrix from `origin` and `ToAxes(orientation)`.
- *OpenVDB* (`.vdb`): one `FloatGrid` per listed channel plus `known` (only nonzero voxels active), voxel
  `(i, j, k)` centred at lattice point `(i, j, k)` (a linear transform of `spacing` moved by the lattice
  origin), world Z-up, no rotation, zip compression (no Blosc dependency).
- *Solid motion JSON* (`.json`, no library; the default of its kind, A14): `{"contract_version", "frame",
  "time", "solids": [{"solid_id", "origin", "orientation" (w, x, y, z, own axes to world), "linear_velocity",
  "angular_velocity", "in_scene"}]}`, ascending `solid_id`, numbers written to round-trip a `double`. The
  Blender add-on turns these into transform keys.

**The format table today.** Candidates in preference order; `chosen` is the last candidate everywhere,
because no witness has run yet ("to be proven by the Blender add-on"):

| Kind | Candidates | Chosen today | Properties every candidate's witness must show |
|---|---|---|---|
| `kPoints` (particles, droplets, spray, foam, bubbles) | USD `UsdGeomPoints`; Alembic `OPoints` | Alembic | file sequence; velocity gives motion blur; velocity in seconds; point radius; point attributes; Z-up |
| `kSurfaceMesh` | USD `UsdGeomMesh`; Alembic `OPolyMesh` | Alembic | file sequence; topology per frame; velocity gives motion blur; velocity in seconds; point attributes; face attributes; Z-up |
| `kVolume` | OpenVDB | OpenVDB | file sequence; grids by name; Z-up |
| `kSolidMotion` | USD `UsdGeomXform`; Alembic `OXform`; JSON | JSON | file sequence; transform gives object motion; Z-up |

Every candidate's writer is in the build until its witness has run, so the witness can run on real files;
`WriteSurfaceFile(format, ...)` and `WritePointsFile(format, ...)` keep two writers only while both are
candidates. A candidate that fails a property is removed with its writer in the commit that records the
failure; when no USD candidate is left, OpenUSD leaves `fluidity_interchange`. If the last candidate itself
fails a property, the format stays, the channels that depend on that property (column "Witness" below)
are withdrawn from the promise and listed as such in the `file_list`, and the owner is asked: never a silent
change of format.

**What the Blender manual says today (Blender 5.3 manual, read 2026-11-14), and what is unverified.**
These are the initial `WitnessRecord` values; `kDocumented` never chooses a format.

| Property | USD | Alembic | OpenVDB / JSON |
|---|---|---|---|
| File sequence | Not verified, and contradicted for import: "Note that USD file sequences (i.e. a unique file per frame) are not supported." (USD import page). The Mesh Sequence Cache modifier "loads data from Alembic and USD files" and "also handles file sequences": whether a USD cache file with Sequence on works is the witness (`kNotRun`) | `kDocumented`: import "Is Sequence: Set to true if the cache is split into separate files." | VDB `kDocumented`: Volume "Sequence: Loads further VDB files, one for each frame in an animation", numeric suffix. JSON: read by the add-on, `kNotRun` |
| Topology per frame | `kDocumented` for the modifier: "meshes and curves with varying topology (like the result of fluid simulations)" | same quote, `kDocumented` | n/a |
| Velocity gives motion blur | `kNotRun`. Velocities arrive as an attribute ("This includes colors, UVs, velocities, and other generic attribute data"), but "The Velocity Attribute option is currently for Alembic files only" | `kDocumented`: Velocity Attribute "`.velocities` which is standard for most Alembic files", used "for generating motion blur data" | VDB: velocity grids exist in Blender but none is written; not promised |
| Velocity in seconds | `kNotRun` (Velocity Unit is "for Alembic files only") | `kDocumented`: Velocity Unit "Second: ... needs to be scaled by the scene FPS"; the reader must be set to Second | n/a |
| Point radius | `kNotRun` (`UsdGeomPoints` import as "Point Cloud Objects" is documented; `widths` to radius is not) | `kNotRun` (Alembic points import into a point cloud is not documented; `main` imports positions and counts only) | n/a |
| Point attributes | `kDocumented` for meshes: "Read USD `Primvars` as mesh attributes", types limited as above; point clouds `kNotRun` | `kDocumented` for the modifier: Read Data "Attributes: Loads custom attributes"; import of `arbGeomParams` `kNotRun` | VDB grids by name `kDocumented`: "The List View shows the grids in the OpenVDB-file, listing their name" |
| Face attributes | `kNotRun` (`uniform` primvars) | `kNotRun` (`kUniformScope`) | n/a |
| Z-up without user rotation | `kNotRun` (`upAxis` handling not stated) | `kNotRun`; established on `main` only by counts after import | VDB `kNotRun`; JSON n/a |
| Transform gives object motion | `kDocumented` in one file: "If a USD primitive has time-varying transform data, a Transform Cache constraint will be added"; over a file sequence `kNotRun` | `kDocumented` in one file: "For time-varying object transforms ... the Transform Cache Constraint is used"; over a sequence `kNotRun` | JSON: the add-on's keys, `kNotRun` |

Established on `main` (Blender chapter, channel matrix): Alembic positions, widths, velocities and ids for
points and positions and velocities for the surface open in Blender; only point, vertex and triangle counts
were checked. Nothing else is shown.

**Witness tests (run by the Blender add-on in Blender 5.3, A16).** One fixed scene: a sphere of a
two-ingredient liquid falling beside a second liquid that does not mix into a box, a turning `Box` solid,
foam, and gas beside the liquid; written once with every candidate writer of every kind. Blender is started
with the add-on disabled; each file is opened only as the manual describes (import operator, cache file with
Sequence on, Volume with Sequence on). A script reads back, at frames 1, 2 and 10, every channel of the table
below and compares it with the `RenderFrame` and snapshot that made it (positions and values within float
round-off, names and counts exact); motion blur: a Cycles render with and without blur differs, and the streak
of a point of known speed is within 10 % of speed times shutter. Each property gives one `WitnessRecord`
value (the Blender 5.3 result) with the evidence `"<test>; Blender <version> <build hash>; files <digest>"`. A USD
candidate that passes every witness of its kind in Blender 5.3 becomes the chosen format; one that fails is
removed with its writer (and the add-on's link route) in the commit that records the failure.

**Libraries of `fluidity_interchange`** (linked by the CLI only; the native library, every other module's
tests and the add-on's Python never link them). Each is pinned exactly to the newest release that
`uv run conan list "<name>/*" -r conancenter` shows when its build task runs, with the remote `conancenter`
pointing at `https://center2.conan.io` (the old `https://center.conan.io` is frozen and lists only old versions):
the newest released version, never a pre-release or a `cci.<date>` reference (A19). No version is chosen to match
the libraries Blender bundles; the witness tests show that Blender 5.3 reads the files. Each version test compares
with the recorded pin, and each version string in `interchange_libraries.cc` is a literal guarded by a
`static_assert` on the library's own version macro, so another library fails to compile instead of reporting a
wrong pin. `capabilities` and the run report give the versions; no other text names them.

| Library | Found on 2026-11-14 (information only) | Why | Licence | Build notes |
|---|---|---|---|---|
| OpenUSD | 26.08 | USD candidates | Tomorrow Open Source Technology License 1.0 (Apache 2.0 with its own trademark section; the exact name to verify in the pinned release's `LICENSE.txt`) | Static, no imaging, no Python, no tools: `tf`, `gf`, `vt`, `ar`, `sdf`, `pcp`, `usd`, `usdGeom` and their `plugInfo.json` resources, which CMake copies to `usd_resources/` beside the CLI and every interchange test (USD finds its file formats through them at run time; to verify on the three platforms). `capabilities` reports a missing resource folder by name (`missing_resource_folder`) |
| Alembic | 1.8.8 | Alembic candidates | BSD-3-Clause (to verify in the release's `LICENSE.txt`) | Ogawa only, no HDF5 |
| OpenVDB | 12.1.1 (older than the 13.0 Blender 5.3 bundles; accepted under A19) | Volumes | Apache-2.0 (relicensed from MPL-2.0 with 12.0) | Core library only, no tools, zlib on, Blosc off; Boost or Blosc turned off by a recipe option where one exists, otherwise accepted and recorded with what they add |
| oneTBB | 2023.1.0 | Needed by OpenUSD and OpenVDB | Apache-2.0 | Shared by both |
| Imath | 3.2.3 | Needed by Alembic and OpenVDB | BSD-3-Clause | |
| zlib | 1.4 | OpenVDB zip compression | zlib | |

A conflict between OpenVDB's Imath or oneTBB and the pins is settled with Conan's `override` to the newer of the
two, recorded. A build task stops and reports when its package is missing from Conan Center (the fallback, for
the coordinator to rule on, is a recipe of ours under `conan/recipes/<name>/` building the newest release), when
Alembic cannot turn HDF5 off, when static OpenUSD cannot register its schema types, or when OpenVDB needs a
dependency other than zlib, Imath, oneTBB, Boost or Blosc.

Cost to the build (estimates; to verify by measuring the first build on the three platforms): OpenUSD
dominates, a source build of tens of minutes on eight cores and static libraries of tens of MB even with
imaging off; Alembic, OpenVDB, oneTBB and Imath take minutes and a few MB each. Built once per platform and
pinned by version and hash with the other third-party libraries, never inside a module's test target. The
CLI grows by the linked part only. Removing OpenUSD after failed USD witnesses removes most of that cost.

**The field contract.** One row per channel, in `ReadOutputChannels()` order. "Blender name" is what the
witness looks for after import; "=" means the file carries it under the Blender name (an Alembic
`arbGeomParam`, a USD primvar, an OpenVDB grid). Every point and surface channel is float32 unless marked.

| Native quantity | Blender name | Alembic / USD / OpenVDB (or JSON) name | Product, domain | Unit | Space | In between | Known when | Kind | Witness |
|---|---|---|---|---|---|---|---|---|---|
| `SurfaceVertex::position` | `position` | `P` / `points` / - | surface, point | m | world position | linear | always | reconstructed | Z-up |
| `SurfaceVertex::liquid_velocity` | `velocity` | `.velocities` / `velocities` / - | surface, point | m/s ("Second") | world vector | linear | always; 0 where motion was not known (counted) | measured | motion blur, seconds |
| `SurfaceVertex::id` | `id` | = (int32) | surface, point | none | label | constant | same vertex only while `age > 0`; low 31 bits | reconstructed | point attributes |
| `SurfaceVertex::age` | `age` | = | surface, point | s | scalar | linear | always | reconstructed | point attributes |
| `SurfaceVertex::distance_to_solid` | `distance_to_solid` | = | surface, point | m | scalar | linear | capped at 4 spacings | measured | point attributes |
| `SurfaceVertex::blob_index` | `blob_index` | = (int32) | surface, point | none | label | constant | this frame only | reconstructed | point attributes |
| `SurfaceMesh::ingredient_share` | `share_<ingredient_export_name>` | = | surface, point | 0..1 | scalar | linear | present with two or more ingredients; sums to 1 | measured | point attributes |
| `triangle_material_index` | `material_index` | = (int32) | surface, face | none | label | constant | present with two or more materials | reconstructed | face attributes |
| `triangle_kind` | `triangle_kind` | = (int32) | surface, face | 0, 1, 2 | label | constant | always | reconstructed | face attributes |
| `Droplet::position`, `velocity` | `position`, `velocity` | `P`, `.velocities` / `points`, `velocities` / - | droplets, point | m, m/s | world | none | always | measured | motion blur |
| `Droplet::radius` | `radius` | `.widths` / `widths` (2 x radius) / - | droplets, point | m | scalar | none | always | measured | point radius |
| `FoamParticles::position`, `velocity` | `position`, `velocity` | as droplets | spray / foam / bubbles, point | m, m/s | world | none | always | display only | motion blur |
| `FoamSettings::display_radius` | `radius` | as droplets | spray / foam / bubbles, point | m | scalar | none | always | display only | point radius |
| `FoamParticles::id` | `id` | = (int32) | spray / foam / bubbles, point | none | label | none | low 31 bits | display only | point attributes |
| frame time minus `start_time` | `age` | = | spray / foam / bubbles, point | s | scalar | none | always | display only | point attributes |
| `ParticlesSnapshot::Material::position`, `velocity_for_next_step` | `position`, `velocity` | as droplets | particles, point | m, m/s | world | none | always | measured | motion blur |
| `cbrt(3 rest_volume volume_ratio / (4 pi))` | `radius` | as droplets | particles, point | m | scalar | none | always | measured | point radius |
| `Material::id` | `id` | = (int32) | particles, point | none | label | none | low 31 bits | measured | point attributes |
| `Material::volume_ratio` | `volume_ratio` | = | particles, point | none | scalar | none | always | measured | point attributes |
| `Material::size_level` | `size_level` | = (int32) | particles, point | none | label | none | always | measured | point attributes |
| `ingredient_fractions` | `share_<ingredient_export_name>` | = | particles, point | 0..1 | scalar | none | present with two or more ingredients | measured | point attributes |
| a field of `RequestedOutputs::particle_fields` | its `Field` name in lower snake case (`mass`, ...) | = (Vec3: 3 floats; Mat3: 9) | particles, point | the field's | the field's | none | when requested and carried | measured | point attributes |
| foam count / volume | `foam_count_density` | - / - / = | foam volume | 1/m3 | scalar | Blender's volume sampling | `known` | display only | grids by name |
| bubble count / volume | `bubble_count_density` | - / - / = | foam volume | 1/m3 | scalar | same | `known` | display only | grids by name |
| bubble gas volume / volume | `bubble_volume_fraction` | - / - / = | foam volume | 0..1 and above | scalar | same | `known` | display only | grids by name |
| `VolumeFrame::known` | `known` | - / - / = | foam volume, gas volume | 0 or 1 | scalar | same | always | measured | grids by name |
| `GasSnapshot::gas_density` | `density` | - / - / = | gas volume | kg/m3 | scalar | same | `known` | measured | grids by name |
| `GasSnapshot::smoke_density` | `smoke` | - / - / = | gas volume | kg/m3 | scalar | same | `known` | measured | grids by name |
| `GasSnapshot::temperature` | `temperature` | - / - / = | gas volume | K | scalar | same | `known` | measured | grids by name |
| `GasSnapshot::fuel_density` | `fuel` | - / - / = | gas volume | kg/m3 | scalar | same | `known` | measured | grids by name |
| `GasSnapshot::flame` | `flame` | - / - / = | gas volume | kg/(m3 s) | scalar | same | `known` | measured | grids by name |
| `SolidState::origin`, `orientation` | object transform | `OXform` / `xformOp:translate`, `xformOp:orient` / JSON `origin`, `orientation` | solid motion, object | m, rotation | world position, world rotation | Blender's interpolation | `in_scene` | measured | object motion |
| `SolidState::linear_velocity`, `angular_velocity` | (JSON only) | - / - / JSON fields | solid motion, object | m/s, rad/s | world vector | none | `in_scene` | measured | file sequence |

"Measured" means a committed value, or one interpolated from committed values at a stated place.
"Reconstructed" means made by this module's surface method. "Display only" means it depends on display
settings (`particles_per_volume_time`, `count_per_particle`, `display_radius`) and has no physical amount.
A channel is advertised in the `file_list` only when its witness property is `kPassed` for the chosen
format in Blender 5.3; until then it is written and listed as "written, not witnessed".

Deliberately not promised:

- 64-bit ids in Blender (Blender imports no int64 attribute; `id` is the low 31 bits; the full id stays in
  the `RenderFrame` and, for USD points, in `ids`). A vertex id says "same vertex as last frame", never "same
  piece of liquid".
- Motion blur from anything but the surface, point and solid motion above, and from a format whose
  velocity witness has not passed; a velocity grid in the volumes.
- Blob labels that last across frames; a count of how the liquid split or merged.
- A surface that overlaps solids to hide the seam (the mesh lies on the solid; the artist offsets it).
- That mesh volume equals liquid volume. Both are reported; only the particles' amounts are conserved.
- Foam that changes the liquid, air drawn into the liquid, or foam counts that mean a physical density.
- Identical foam at two output frame rates: kinds change at frame times (bounded in section 7).
- `flame` as a 0..1 value, or a temperature already scaled for a shader: the shader scales (gas decision 9).
- Identical bytes of `.usdc`, `.abc` and `.vdb` files (OpenVDB writes a file UUID; library versions change
  layouts). `RenderFrame` bytes are identical; the files are compared by what they decode to.
- A file one program writes and Blender versions other than 5.3 read (A16: 5.3 is the one supported version).

### 5.8 Every material

| Material | Surface | Foam sources | Volumes | Points |
|---|---|---|---|---|
| One liquid, a mixture | mesh with `share_<ingredient>` | yes | foam volume | particles with shares |
| Liquids that do not mix | one group, labels, inner sheet for two | yes, from the liquid set's motion (5.2) | foam volume | particles per material |
| Snow | none (decision 7) | no | none | particles, as on `main` |
| Springy solids | none | no | none | particles |
| Cut solids, moving solids | none (the artist's own mesh); they shape the field and stop foam | no | cut out of `known` | solid motion |
| Gas, also beside liquid (A10) | none | no | gas volume, ends at the liquid | none (markers are not written) |
| Air pockets (pressure) | the mesh's inner pieces around them; no channel | no | none | none |

## 6. CPU and GPU

**What runs where, and why.** `RecordStep` runs where the simulation runs, with `step_recorder_cpu.cc` and
`step_recorder_gpu.cu`, because the try's arrays live there and must not be copied out between stages.
Everything of `BuildRenderFrame` is CPU only (`cpu_only.h`), on I/O's worker thread, because (a) the
simulation must not wait for output, and the device is sized for the solver (programme 8a); (b) it works
from a CPU-owned snapshot that already had to be copied for the files; (c) mesh checks and ordering are
branch-heavy and gain little; (d) one implementation gives one set of bytes. This departs from "every
operation on both" (decision 3). The writers of `interchange/` are CPU only too and run on the same worker,
after `BuildRenderFrame`; the libraries are called from that one thread per frame (OpenUSD and OpenVDB may
use oneTBB threads inside a call; the decoded content does not depend on it).

**Shared formula headers:** `lattice.h` (`ToBlockKey`, `ComputePointPosition`, `FindLowPoint`),
`foam_source.h`, `foam_motion.h`.

**`RecordStep` on the GPU.** Device working buffers, owned by the recorder, kept for the whole run and sized by
`RenderLimits` and `MotionSettings`: per listed lattice point `LatticeCoord` (12 bytes), cell place (4),
velocity (12), distance (4), known (1), normal (12), order (4); per event 64 bytes; a count per cell place
(4). Work: one launch per range for the count, one for the fill, one for values, one for normals, one for
strength and events; a prefix sum between count and fill; `BuildStableOrder` by block key. No sum crosses
threads: each lattice point is one thread's and reads its faces and its 26 neighbours in list order.
Returned to the CPU per call: the event count and the events (64 bytes each, ordered within the call), four
counters; for a kept sample only, the point arrays (33 bytes per point); on the try that ends at a frame
time, one `LiquidTotals` per block of the range (24 bytes each). Each of these `ReadBuffer` calls is one counted
copy (`ReadRecorderCopyCounts`, `bytes_from_processor`; step plan Appendix A row M13); the recorder's working buffers are
`where = kRunOn` and sends nothing to the processor. The CPU loop has the same five passes over
cell places in index order; a thread takes whole cell blocks. The velocity pass reads the one liquid set (5.2).

**Border cells and batches.** `RecordStep` reads faces within the touching cells of a point's cell and
neighbour points two spacings away: a border of `kFourCellWidths` per range, loaded with the piece by the
batches module (cell resize and batches design). Events and points are produced only for points inside the range's own cells, so batches do not repeat
one. Every ordering and every sum that spans calls (events, sample points, the liquid totals) is made in
`CommitRecordedStep` by lattice key or whole-grid block index, so one try recorded in batches gives the bytes of
the same try recorded in one call. (A batched run's tries agree with a one-piece run's only within the step
tolerance, A15; the record follows its try.) The whole-grid block index of a batch comes from
`FoamAndSprayInputs::block_in_whole_grid`, the batched run's CPU copy of the piece map that
`step::ApplyGridPiece` keeps (a GPU piece's own span is empty, A17); the recorder never reads the map back itself.

**The worker (CPU).** Loops go over lattice blocks in key order, vertices in index order, particles in id
order. Threads take whole lattice blocks (field, volumes) or index ranges with no shared sum (vertices, foam
particles); every sum at one lattice point is made by one thread in the order of 5.3 step 3 or 5.6 step 1.
`CheckSurfaceMesh` orders triangle boxes by (low x, index) and reports pairs in that order.

**Memory estimate entries** (`BuildRenderOutputMemoryEstimate`): on the simulation's processor, the working buffers above
(`kGivenCount`, `kWholeRun`); in CPU memory for the whole run, `RenderOutput::frame_order_buffers` at
`max(max_field_parts, max_foam_particles)` items of `ComputeOrderBuffersSize(RunOn::kCpu)` (reconcile item 26); in CPU memory per queued frame, `samples_per_frame + 1` samples at 33 bytes per
recorded point and the events (`kOneSnapshot`, multiplied by `ReadyScene::snapshots_kept_at_once`); for the
worker, field blocks at `512 * (4 + 4 + 4 * ingredients)` bytes, vertices at 64 and triangles at 17, foam
particles at 73 (`kOneCall`); for the writers, the `PointObjectBuffers` float copies at `4 * (8 + ingredients +
requested components)` bytes per simulation particle (`kOneCall`, the largest material). Memory the libraries
allocate inside a call is not counted (to verify by measuring the reference scenes; reported in the run
report).

**Determinism.** Same machine, same build, same recorded interval and snapshot: identical `RenderFrame`
bytes, with any number of worker threads, and one try recorded in batches gives the record of one call. Same GPU
and same input: identical events and samples. Files: identical decoded content for one library version.

**Tolerance, CPU against GPU, for `RecordStep` only:** the list of lattice points: equal; `known`: equal
except where a weight sum is within `1e-12` of the grid's floor; velocities:
`1e-6 * (largest speed) + 1e-9 m/s` (they are stored as float); distances: `1e-6 * spacing`; an event is in
both lists unless its `volume_time < 1e-9 * spacing^3 * step_duration`; `volume_time`: `1e-6` relative plus
that floor. Frames built from the two records are compared only by report numbers (vertex count within
0.1 %, volume within 0.1 %, foam count within 0.5 %), because a mesh is not a continuous function of its
input.

## 7. Tests

Analytic and conservation tests (CPU, no GPU needed):

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `RenderOutput, SwitchedOff_SameCommittedBytes` | dam break 50 steps, with and without the recorder and worker | equal `SimulationSnapshot` bytes | exact |
| `RecordStep, RejectedTry_LeavesRecorderUnchanged` | record, reject, read; and record, commit, read | first is empty; second holds the step | exact |
| `RecordStep, LinearVelocity_IsReproduced` | `u = A x + b` on faces of a mixed-size grid (three size levels) | lattice velocity `A x + b` | `1e-6` of the largest speed (float) |
| `ComputeLatticeReading, LinearField_Exact` and `..., MissingPoint_IsNotKnown` | 1,000 positions; one of 8 points removed | value and change exact; `is_known == false`, never 0 as an answer | `1e-6` relative; exact |
| `ComputeMovedPoint, RigidRotation` | `u = w x r`, quarter turn in one interval, 2 samples | end position | `0.01 * spacing` |
| `BuildSurfaceField, VolumeIsKept` | 10,000 random particles, no solid | `sum liquid * spacing^3 == sum volume` before and after `G` | `1e-12` relative |
| `BuildSurfaceField, FlatLiquid_ZeroSetAtTheTop` | slab, particles 2 per cell width, top at 0.3 spacing above a lattice plane | zero set height | `0.05 * spacing` |
| `BuildSurfaceField, BesideASolid_SameHeight` | the slab against a `FlatSolid`, and a tilted `Box` moving at 1 m/s | height at the solid equals the height far away; `kLiquidToSolid` triangles within `0.05 * spacing` of the solid | `0.05 * spacing` |
| `BuildSurfaceField, GridOnlyInside_NoInnerSurface` | sphere of radius 16 cells, particles in the outer 3 cells, grid-only inside | 1 piece, Euler number 2 | exact |
| `BuildSurfaceField, BigAndSmallParticles_NoGap` | slab across a size change | 1 piece; height step under `0.1 * spacing` | as stated |
| `BuildSurfaceMesh, RandomFields_AlwaysSafe` | 1,000 smooth fields plus 1,000 noise fields, values at exactly 0 included | `CheckSurfaceMesh(...).is_safe`; no flat triangle | exact |
| `BuildSurfaceMesh, Sphere` / `Torus` / `TwoSpheres` | radius 8 spacings | volume within 1 %; Euler 2 / 0 / 4; pieces 1 / 1 / 2; normals within 3 degrees | as stated |
| `CheckSurfaceMesh, EachDefect_IsFound` | one removed triangle; one turned triangle; an edge with three triangles; two spheres that overlap; one zero-area triangle; two cones tip to tip | 3 open edges; wrongly turned > 0; overused 1; crossing pairs > 0; flat 1; pinched 1; `is_safe == false` each | exact |
| `MoveSurfaceMesh, UniformMotion_KeepsIds` | sphere, `u = (1, 0, 0)`, 100 frames | all ids kept, no blob built from the field, `liquid_velocity == u`, volume drift under 0.5 % | `1e-6`; as stated |
| `BuildSurfaceFrame, TwoSpheresMerge_ThatBlobIsBuilt` | two spheres approach, a third stays far | at the merge frame: merged blob `kBuiltFromField`, third sphere keeps every id, mesh safe | exact |
| `BuildSurfaceFrame, StretchedSheet_IsBuilt` | the flow `u = (x, 0, -z)`, which stretches a sheet along x | reject reason `kEdgeTooLong`, mesh safe | exact |
| `BuildSurfaceFrame, RandomClouds_NeverFailForAMeshReason` | 1,000 random particle sets with random motion | a safe mesh, or `kNotBuilt` with a limit reason | exact |
| `ApplyVolumeOffset, ReachesTarget` / `StopsAtLimit` | sphere 2 % short; target 50 % above | within 0.5 %; `offset == 0.25 * spacing`, `offset_is_at_limit`, mesh safe | `1e-9` |
| `BuildSurfaceFrame, LoneParticle_IsADroplet` | one particle | no triangle; one droplet, `radius = cbrt(3 V / (4 pi))` | `1e-6` relative |
| `ComputeSourceStrength, HandValues` | `|u| = 2`: `I_speed = 2`; ramp `{1, 2}` at 1.5 gives 0.5; sphere `R = 8` spacings moving out: `I_crest = 2 / R`; moving in or at rest: 0; neighbours that separate: `I_air = 0` | as listed | `1e-12`; curvature 5 % |
| `ComputeOccupiedShare, ColumnSum` | flat surface at any height | sum over a column equals the liquid depth | `1e-12` |
| `AddFoamParticles, OwedCarriesOver` | two equal events worth 2.5 particles each | 2 then 3 added; owed 0.5 then 0 | exact |
| `AddFoamParticles, SameTotalInOneOrFourSteps` | same `volume_time` split over 4 steps | counts differ by at most 1 per lattice point | exact |
| `ComputeDragMotion, Limits` | `k = 0`; `k dt = 1e-9`; `k dt = 1e6`; fall from rest | parabola `z0 + v t - g t^2 / 2`; series equals closed form; `v = target` | `1e-13` relative |
| `ToFoamKind, KindIsHeldNearTheLimit` | distances `1.4 r, 0.9 r, 0.4 r, -0.4 r, -1.6 r` from spray | spray, spray, foam, foam, bubble | exact |
| `MoveFoamParticles, StopsAtAnySolid` | spray thrown at a turning `Box` | never inside; no speed into the solid | contact tolerance |
| `RemoveFoamParticles, AgeAndDomain` | foam at age `lifetime`; spray one step outside the box | both removed, counted in the right counter | exact |
| `BuildFoamVolume, CountsAreKept` | 10,000 particles, some beside a solid | `sum density * spacing^3 == count * count_per_particle`; nothing at points not known | `1e-12` |
| `BuildGasVolume, ThreeStates` | gas cells holding zero smoke; space with no gas cell; a scene with no gas | `known 1, value 0`; `known 0`; no `gas_volume` in the frame | exact |
| `BuildGasVolume, MixedSizes_KeepTheSum` | cells of two sizes | `sum value * spacing^3 == sum value * cell volume` | `1e-6` (float) |
| `BuildRenderFrame, SameInputTwice_SameBytes` and `..., OneOrEightThreads_SameBytes` | breaking-wave piece, 10 frames | identical `RenderFrame` bytes | exact |
| `BuildRenderFrame, FromSnapshot_GoesOnEqual` | 10 frames; or 5, `ReadSnapshot`, rebuild, 5 more | identical frames 6 to 10 | exact |
| `BuildRenderFrame, TwoFrameRates` | the same run written at 24 and 48 frames per second | at shared times: foam count within 2 %, mean position within `0.5 * spacing` | as stated |
| `BuildGasVolume, GasBesideLiquid` (NEW, A10) | smoke over a moving water surface; frame with a surface group and the gas volume, built through `gas::ToGasSnapshot` from the step snapshot's `GasAmountsSnapshot` | same `FrameIdentity`; `known 0` at every point of a liquid cell; no surface vertex reads gas; gas markers in no points file | exact |
| `RecordStep, BatchedEqualsOnePiece` (NEW) | breaking-wave piece; one call per try, then 2 and 5 batches per try | identical events, samples, counters and `liquid_totals_at_frame` bytes | exact |
| `Run, RecorderCopiesInRunCost` (NEW, A17, I/O's test target, plan Task 47) | a run with a CPU recorder, with and without foam | the run cost's copies are equal (`ReadRecorderCopyCounts` is `{}`); the GPU sum is the owner's `gpu` run of I/O plan Task 41 | exact |
| `RecordStep, TwoLiquidsOneSet_ReadsTheLiquidSet` (NEW, A8) | two liquids that do not mix in the one liquid set, beside a snow set and a gas marker set | each point's velocity equals the interpolation of the liquid set's `after_pressure` snapshot; the snow and gas marker sets are never read | exact |
| `BuildRenderFrame, VolumeTargetFromLiquidTotals` (NEW) | deep tank with a grid-only region; one group holding every liquid | `liquid_volume_outside_surfaces` against `liquid_totals_at_frame.current_volume` | `1e-12` relative |
| `CheckRenderSettings, SnowGroup_IsBadInput` (NEW, A9) | a `SurfaceGroup` with a snow material; with a gas material | `kBadInput` each | exact |
| `FormatTable, ChosenFollowsWitnesses` (NEW) | `ReadFormatTable()`; and hand-made tables (USD all `kPassed`; one USD property `kFailed`; one USD property `kNotRun` or `kDocumented`, the rest `kPassed`; all `kDocumented`) | every `chosen == ChooseFormat`; USD; the next candidate; the next candidate; the last candidate | exact |
| `FormatTable, EveryChannelIsWitnessed` (NEW) | `ReadOutputChannels()` against `ReadFormatTable()` | each channel's `witnessed_by` is a property its kind requires; every candidate has one record per required property; a name exists for the chosen format | exact |
| `ReadFormatChoice, EveryKind_InKindOrder` (NEW) | each `GeometryKind` | the entry of that kind; `ReadFormatTable()` in `GeometryKind` order | exact |
| `FormatTable, NotRunRecords_HaveNoEvidence` (NEW) | `ReadFormatTable()` | every `kNotRun` record has empty evidence; every `kDocumented` record has evidence | exact |
| `FormatTable, MeasuredRecords_NameBlender53` (NEW, A16) | `ReadFormatTable()` | every `kPassed` or `kFailed` record's evidence contains "Blender 5.3." (none in the initial table; guards the add-on's write-back) | exact |
| `Capabilities, ReportsFormatTableAndLibraries` (I/O's test target) | the `capabilities` event | the format table, the channels and the library versions with `missing_resource_folder`; every witness record has the key `blender_5_3` and no `blender_5_2` | exact |
| `ReadInterchangeLibraries, <Library>Version_IsThePin`; `InterchangeLibraries, <Library>IsThePinnedVersion` (NEW, A19) | each of the six libraries | the linked version equals the pin recorded when its build task ran | exact |
| `InterchangeLibraries, AlembicOgawaArchive_ReadsBackOnePoint`, `OpenVdbZipFile_ReadsBackOneVoxel`, `UsdcLayer_ReadsBackOnePoint` (NEW) | one point or voxel written and reopened (the USD layer on a path that does not end in `.usdc`, as `path_while_writing()`) | value and name equal | exact |
| `CheckUsdResources, MissingFolder_IsNamed` (NEW), later `WritePointsFile, UsdWithoutResources_IsFileWriteFailed` | the resource folder renamed | `kFileWriteFailed`; `missing_resource_folder` names the folder | exact |
| `WritePointsFile, ReadBack` (NEW) | each candidate format; particles of a two-ingredient liquid with `mass` requested, droplets, the three foam objects (one empty) | positions (Alembic: turned back), velocities, widths, int32 ids, every channel and the metadata equal the inputs; one time sample at `frame` | exact (float) |
| `WriteSurfaceFile, ReadBack` | each candidate format; a surface of two materials and two ingredients | every channel of 5.7 equals the frame; triangle order outward; USD `upAxis`, `metersPerUnit`, `timeCodesPerSecond`; per-face channels at face scope | exact (float) |
| `WriteVolumeFile, ReadBack` | foam and gas volumes | every grid of 5.7, `known`, voxel size and placement; names, units and `kOutputContractVersion` in the metadata | exact (float) |
| `WriteSolidMotionFile, ReadBack` (NEW) | each candidate format; a turning `Box`, a fixed plane, a driven solid that left the scene | origin and orientation as `ToAxes` within `1e-12`; JSON round-trips every `double`; the solid that left has `in_scene` false | as stated |
| `Writers, FormatNotInBuild` (NEW) | a writer called with a format that is not a candidate of its kind (OpenVDB for points) | `kFormatNotInBuild`, nothing written | exact |
| `Writers, EveryCandidateHasAWriter` (NEW) | each candidate of each kind, on a tiny input | never `kFormatNotInBuild` | exact |

Witness tests, run by the Blender add-on in Blender 5.3 (5.7, A16): one per property and candidate, on the
files of the fixed witness scene; each result is recorded as a `WitnessRecord` line with one result. Until they
run, every record is `kNotRun` or `kDocumented` and the table chooses the last candidate.

Surface sequence of the report (numerics 12.7), on fixed recorded inputs: static sphere, moving sphere,
stretching sheet, merging pair, splash with 2,000 droplets; measured each time: volume, distance to the
analytic shape, defects, kept ids, time.

Reference and proofs: none. The field, the mesh and the foam formulas each have a closed-form case above;
the one piece without one is "the built mesh is always safe", and a second implementation in Python would
share the same construction, so the 2,000-field property test with the independent `CheckSurfaceMesh` is the
check. `CheckSurfaceMesh` itself is tested against a brute-force all-pairs test on meshes under 2,000
triangles. `main`'s four Lean lemmas on trilinear weights are covered by `VolumeIsKept`.

Comparison with the proof of concept, where the method is unchanged (`main` output turned by
`(x, y, z) -> (x, -z, y)`): `IntegrateWhitewaterDrag3D` against `ComputeDragMotion` (`1e-13`),
`ClassifyWhitewater3D` (exact), `EmitWhitewaterBirthsThroughStep3D` counts and owed parts (exact),
`ComputeWhitewaterSource3D` on the same 27 inputs (`1e-12`), `BuildSurfaceDensity3D` with no solid (`1e-6`,
float against double), `ExtractSurfaceMesh3D` on a sphere field whose crossings are away from corners
(same triangle count, volume `1e-9`).

GPU tests (listed for the hardware run; "GPU: compiled, not run" until then): `RecordStep` on the
linear-velocity case, a dam break of 20 steps and a breaking-wave piece, against the CPU within 6's
tolerance; the repeat-run test on events and samples (identical bytes); reject leaves the device working buffers
unused; `ReadRecorderCopyCounts, OnGpuFrameTry_CountsEveryReadBack` (NEW, A17: event count, events, counters,
kept points and liquid totals: `copy_count` and bytes equal the sizes read). In a container with no GPU:
`ReadRecorderCopyCounts, CpuRecorder_IsEmpty` (NEW, A17, label `cpu`), every worker test, every `RecordStep` test on the CPU, and the CUDA
compile of `step_recorder_gpu.cu`. The interchange read-back tests link `fluidity_interchange` (the three
libraries), not a GPU; they are the only tests that do.

## 8. Method paper section

Labels: `<sec-render-output>` (the one-way rule, recording under try and commit, the worker),
`<sec-surface>` (kept from the old paper), `<sec-foam-and-spray>`, `<sec-volume-output>`,
`<tab-output-channels>`, `<sec-output-formats>`, `<tab-format-choice>`, `<eq:surface-field>`,
`<eq:foam-source>`, `<eq:foam-drag>`.

Outline: 1 what is looked at is not what is solved: four volumes, three kinds of bubble; 2 the lattice and
the motion sample; 3 the surface field and its three fixes; 4 a mesh that is safe by construction; 5 moving
the last mesh, the checks, and replacing a blob; 6 volume as a report; 7 foam sources; 8 the life of a foam
particle; 9 counted volumes and zero, unknown, absent; 10 the channel table; 11 open formats, the format
table and its witnesses (what the manual says, what was proven, which libraries).

Figures: `<fig-render-output-flow>`: step, recorder, queue, worker, the products and their open formats, with the single
direction of the arrows and the reject path; `<fig-surface-field>`: a 2D cut with particles near the
surface, grid-only cells inside, a tilted solid, lattice points and the zero set before and after dividing
by `open`; `<fig-surface-strategy>`: last mesh moved, pulled back, checked per blob, one blob replaced;
`<fig-six-tetrahedra>`: one cube, its diagonal, one patch; `<fig-foam-kinds>`: spray, foam and bubble
against distance to the surface, with the range where a kind is held; `<fig-known>`: one slice shown three times as value, known
and what a shader sees.

Pseudocode: `RecordStep`, `BuildSurfaceFrame` (the numbered steps of 5.3), `BuildRenderFrame`,
`MoveFoamParticles`.

Cite, from the old bibliography: ref 12 (Ihmsen 2012, the three measures and the three kinds), ref 27
(Wojtan 2009, a moved mesh that is rebuilt from a field where its topology is wrong; ours replaces whole
blobs), ref 33 (Brochu and Bridson 2009, what a safe moved mesh must satisfy), ref 17 (Ferstl 2016, particles
near the surface and grid inside, which the field must join), ref 13 (Bhattacharya 2011, named as the
alternative field that is not used). The smoothed-share field (Zhu and Bridson 2005), the six-tetrahedra
mesh, the Blender 5.3 manual pages of 5.7, OpenUSD, Alembic and OpenVDB have no entry; they are cited by
author (or project) and year until entries exist (decision 9).

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| whitewater, stage `kWhitewater` | foam and spray, `MakeFoamAndSpray`, `foam_and_spray/` | DECIDED |
| Mesh stage, `surface_*_3d` | `surface/`, `BuildSurfaceFrame` | DECIDED folder; NEW function |
| `MotionCapture3D`, "motion capture" | `StepRecorder`, `RecordStep`, `CommitRecordedStep`, `RejectRecordedStep` | NEW (O5; after owner Q10) |
| `MotionInterval3D`, "interval" | `OutputInterval` | NEW (O5) |
| `MotionStamp3D`, `MotionFrame3D::samples` | `MotionSample` | NEW |
| carrier, `FrozenMotionCarrier3D`, `SampleMotionCarrier3D` | `liquid_velocity`, `ComputeLiquidMotion` | NEW; `carrier` VERIFIED here by reading `motion_carrier_3d.h:17-24`: the interpolated grid velocity, nothing else |
| `AdvectMotionPoint3D` | `ComputeMovedPoint` | NEW |
| `accepted_epoch`, accepted step ordinal | `state_version`, `step_count` | DECIDED; step |
| `SurfaceLattice3D`, source grid, voxel grid | `LatticeFrame`, `LatticeBlocks`, `LatticeValues`, lattice point | NEW (decision 1) |
| density field, `BuildSurfaceDensity3D` | `SurfaceField`, `BuildSurfaceField` | NEW |
| `sigma_m` | `smoothing_width` | NEW |
| `ApplySurfaceOccupancy3D`, certified liquid / air, occupancy class | step 1 of 5.3 (grid-only cells in the sum) | not ported as a concept |
| `ExtractSurfaceMesh3D` | `BuildSurfaceMesh` | NEW |
| `MeasureSurfaceMesh3D`, `SurfaceMetrics3D`, "safe" | `CheckSurfaceMesh`, `SurfaceMeshDefects`, `SurfaceMeshMeasure`, `is_safe` | NEW |
| boundary / nonmanifold edges, nonmanifold vertices, orientation defects, degenerate triangles, self intersections | `open_edge_count`, `overused_edge_count`, `pinched_vertex_count`, `wrongly_turned_edge_count`, `flat_triangle_count`, `crossing_pair_count` | NEW (decision 1) |
| `TrackSurfaceMesh3D`, tracker, tracked | `MoveSurfaceMesh`, `SurfaceOrigin::kMovedFromLastFrame` | NEW |
| projection (of a vertex) | pull back, `max_pull_back_in_spacings`, `kPullBackFailed` | NEW; `project` is a never-say word |
| "requires local repair", `W-MESH-REPAIR-FALLBACK`, fallback | `SurfaceOrigin::kBuiltFromField`, `moved_mesh_reject_reason` | NEW (O5: no identifier says `fallback`) |
| `CorrectSurfaceVolume3D`, volume lock | `ApplyVolumeOffset`, `BlobVolume` | NEW |
| component (of a surface), component lineage | `blob`, `blob_index`; a closed part of a mesh is a piece | DECIDED word; lineage not ported |
| cavity (in a mesh) | inner piece around an air pocket | DECIDED word `AirPocket` (pressure) |
| `geometric_velocity`, `fluid_velocity` | `liquid_velocity` only | NEW |
| `contact` (vertex flag) | `distance_to_solid` | NEW (pressure's word) |
| `phase_id`, `SurfacePartitionFace3D`, `kPhaseAToPhaseB` | `triangle_material_index`, `TriangleKind::kBetweenTwoMaterials` | after owner Q12, Q31 a |
| constituent, substance fraction, `fluidity_fraction_<id>` | `ingredient_share`, `share_<ingredient>` | DECIDED word |
| droplet, `SurfaceComponentRepresentation3D` | `Droplet` (a particle outside every mesh) | NEW rule, kept word |
| `WhitewaterKind3D` | `FoamKind { kSpray, kFoam, kBubble }` | NEW |
| whitewater point | foam particle, `FoamParticles` | NEW (decision 1) |
| birth event, `WhitewaterCellBirth3D`, `AcceptedWhitewaterBirth3D` | `FoamSourceEvent` | NEW; birth is a never-say word |
| `EmitWhitewaterBirthsThroughStep3D`, emit | `AddFoamParticles` | DECIDED verb |
| `AdvanceWhitewaterPoint3D`, `IntegrateWhitewaterDrag3D` | `MoveFoamParticles`, `ComputeDragMotion` | DECIDED verb; NEW |
| expiry | `RemoveFoamParticles`, `ended_by_age` | DECIDED verb |
| `ClassifyWhitewater3D`, classification radius | `ToFoamKind`, `kind_change_distance` | NEW |
| potential, `WhitewaterSourceStatistics3D` | `SourceStrength`, `strength` | NEW |
| kinetic, trapped air, crest | `speed_squared_half`, `trapped_air`, `wave_crest` | NEW |
| `WhitewaterRamp3D`, `trapped_air_weight` | `Ramp`, `trapped_air_share` | NEW |
| `volume_time_m3_seconds` | `volume_time` (unit in the comment) | naming rule 7 |
| `represented_count`, `emission_rate_per_m3_second` | `count_per_particle`, `particles_per_volume_time` | NEW |
| `remainder`, `WhitewaterCellRemainder3D` | `owed_particles` | step's word for the same idea |
| `bubble_buoyancy_scale`, `ambient_velocity`, `*_drag_per_second` | `bubble_rise_scale`, `air_velocity`, `*_drag_rate` | NEW; naming rule 7 |
| `foam_count_per_m3`, `bubble_count_per_m3`, `gas_fraction` | `foam_count_density`, `bubble_count_density`, `bubble_volume_fraction` | NEW; `gas` belongs to the gas module |
| `validity`, `fluidity_known` | `known` | NEW |
| `maximum_*`, budget | `RenderLimits::max_*` | DECIDED pattern |
| `EffectsIdentity3D`, source execution keys | `FrameIdentity` with I/O's `run_fingerprint` | DECIDED word |
| `SurfaceStage3D`, `WhitewaterStage3D`, `VolumesStage3D`, `EffectsCache3D` | `BuildRenderFrame`; the run folder and listing are I/O's, the file writers this module's | not ported |
| point cache, `ParticlesFrame` (I/O's first draft), bridge point writer | particle playback: `ToPointObject`, `WritePointsFile`, `OutputProduct::kParticles` | NEW (A14) |
| channel matrix (Blender chapter), "witness" | `WitnessProperty`, `WitnessResult`, `WitnessRecord`, witness test | NEW (A14) |
| bridge format switches | `InterchangeFormat`, `GeometryKind`, `FormatCandidate`, `FormatChoice`, `ChooseFormat`, `ReadFormatTable` | NEW (A14) |
| (none) | `WriteSolidMotionFile`, `OutputProduct::kSolidMotion`, `fluidity_interchange`, `ReadInterchangeLibraries` | NEW (A14) |
| (none) | `liquid_totals_at_frame`, `liquid_volume_outside_surfaces` | NEW (the cell resize module's volume target) |
| state velocity | the velocity for the next step (`velocity_for_next_step`) | A1 |
| `PointScratch`, "scratch" (replaced 2026-11-14) | `PointObjectBuffers`, "working buffers"; `RenderOutput::frame_order_buffers` | A31 |

Verify and Open words met: `carrier` (O7) verified for this module as above; `eligible` (O5) is not used
(a source point is "a point with a known velocity near the surface"); `owner` (O5, owner Q5) is not used;
`fallback`, `motion capture`, `interval`, surface-mesh words and foam words (O5) are decision 1. The format
table's "last candidate" is the default of its kind, never called a fallback in an identifier.

## 10. Not ported

- The sideways sweeps and diagonal swaps of `surface_tracking_3d.cc:197-406`, and every "requires local
  repair" exit of `surface_reconstruction_3d.cc`: replaced by "take the built pieces of that blob".
- `surface_component_lineage_3d.*`, `surface_material_components_3d.*` (blob history by particle ids, 1,024
  particles, the 2 and 3 cell-width droplet switch): labels are per frame; a droplet is decided by the field.
- `surface_occupancy_3d.*` with its five classes and "unknown fails": grid-only cells enter the sum.
- `surface_closest_point_3d.*` (a triangle tree for nearest points): no caller is left.
- Mirroring particles in one plane, the box-shaped liquid volume check and the 16-piece first frame
  (`surface_reconstruction_3d.cc:263-320, 596-703, 796-806`).
- The 0.5 % volume gate as a failure (`surface_cache_3d.cc`); `wetness`, `thickness_m`, `parent_id`,
  `unresolved_sheet`, `geometric_velocity` (stored, never computed, or without meaning on a built mesh).
- `motion_capture_3d.*` and `motion_carrier_3d.*`: full-state copies and a second copy of the transfer.
- `whitewater_sphere_source_3d.*`: sources that only work for a sphere and a fixed plane.
- `surface_cache_3d.*`, `whitewater_cache_3d.*`, `whitewater_volume_cache_3d.*`, `surface_settings_3d.*`,
  the three `*_stage_3d.*`, `schemas/effects/` (v1): raw copies of structs are not a format; every product is
  an open format (A14).
- The USD text writer and the whole-range Alembic writers of `blender_bridge.cc`, and the bridge tool as a
  separate program; the per-voxel Python `.vdb` writer `volume_playback.py:28-80`; `foam_projection.py`'s
  own source key.
- `proofs/Fluidity/VolumeTransport.lean`: not ported (the cell resize and batches design's question,
  recommended option); the volume target is `cell_resize::ComputeLiquidTotals` and its tests.
- From the old paper and never built: mesh edits (split, collapse, swap) after each move, thin-sheet
  handling by rays, the cage for Blender's Multires, the preview mesh as its own path.
- `docs/references/m2-*.md`, `neural-surface-evidence.md`, `tools/native/m2_*.py`: history. Neural meshing
  and the other experiments of numerics 12.5 and 12.6 are out of scope.
- No Python reference and no Lean proof (section 7).

## 11. Decisions

Already asked and applied here: Q2 (`RenderOutputFailure`, A2), Q10 (`CommitRecordedStep`,
`RejectRecordedStep` follow `CommitTry`, `RejectTry`), Q11 (`kNone` first, `kRunOnCallFailed`), Q12 and Q31 a
(liquids that do not mix are two materials; A8 as the step's decision of 2026-11-14 reads it: all liquids
are in one material set, kept apart by pressure's material boundary), Q13 (`BuildStableOrder`,
`BuildGroupsByDestination`); step 8 (a blocking snapshot); gas 9 (five channels, the last named `smoke`);
the cell resize and batches design's question on `VolumeTransport.lean` (not ported). All by the recommended option (owner's standing
instruction, 2026-11-14) except where an owner answer is named.

1. **Surface, foam and motion words (your O5).** Proposed, with the alternative I would take second:

   | Idea | Proposed | Alternative | Never say |
   |---|---|---|---|
   | World-fixed points at equal spacing | `lattice`, lattice point | `output_grid` (but `grid` already means the staggered grid) | voxel grid, node |
   | A foam, spray or bubble particle | `foam particle`, `FoamKind` | `spray_particle`; `whitewater_particle` | whitewater point, diffuse particle |
   | What a step offers to make foam | `FoamSourceEvent` | `FoamSeed` | birth, emission event |
   | How strongly a point makes foam | `strength` | `foam_rate_share` | potential |
   | The three measures | `speed_squared_half`, `trapped_air`, `wave_crest` | `kinetic`, `air_trapping`, `crest` | |
   | Particles not yet added | `owed_particles` | `carried_count` | remainder |
   | Real bubbles one particle stands for | `count_per_particle` | `stands_for_count` | represented count |
   | From one output frame to the next | `OutputInterval` | `FrameInterval` | motion interval |
   | The liquid's motion at one time, for output | `MotionSample` | `LiquidMotionAt` | carrier, stamp |
   | Collecting during steps | `StepRecorder`, `RecordStep` | `OutputCollector`, `CollectStep` | capture, observer |
   | The triangle mesh of the liquid | `SurfaceMesh`, vertex, triangle, edge | none | isosurface |
   | A closed connected part of a mesh | piece | `closed_part` | component, shell |
   | The smoothed liquid share | `SurfaceField`, `smoothing_width` | `LiquidShareField` | density field, sigma, kernel |
   | Moving last frame's mesh | `MoveSurfaceMesh`, `kMovedFromLastFrame` | `CarrySurfaceMesh`, `kCarriedOver` | track, tracker |
   | Bringing a vertex back to the surface | pull back | `snap` | project |
   | A mesh made from the field alone | `kBuiltFromField` | `kRebuilt` | fallback, re-extraction |
   | A mesh that passes every check | safe, `is_safe` | `is_sound` | valid, watertight, manifold |
   | The six defects | `open_edge`, `overused_edge`, `pinched_vertex`, `wrongly_turned_edge`, `flat_triangle`, `crossing_pair` | `boundary_edge`, ..., `self_intersection` (the textbook words, kept in comments) | nonmanifold, degenerate |
   | Where the simulation has an answer | `known` | `has_answer` | validity, active, mask |
   | Channel kinds | measured, reconstructed, display only | measured, derived, display only | |

   ```cpp
   if (report.origin == SurfaceOrigin::kBuiltFromField) { ... }      // proposed
   foam_and_spray::MoveFoamParticles(particles, interval, gravity, settings, report);
   ```

   Decision: the proposed column for every row, and the format-table words of 5.7 (`InterchangeFormat`,
   `FormatChoice`, `WitnessProperty`, `WitnessResult`, `WitnessRecord`; the last candidate is a kind's
   default, never `fallback`). Recommended option (owner's standing instruction, 2026-11-14).

2. **File formats and two new dependencies.** Proposed: Alembic for surface, droplets and foam particles,
   OpenVDB for volumes, one file per frame, no USD; both libraries in a target of their own that only the
   CLI links. Alternatives: a) as `main`: the library writes its own cache, a bridge tool writes Alembic and
   USD, Blender's Python writes `.vdb` (no new dependency in the build, but two more programs, a slow
   writer and the repeated export); b) USD for the surface (attributes are documented, but no documented
   motion blur and no file per frame). Recommendation: the proposal. Also confirm the list "deliberately
   not promised" in 5.7, and that a failed attribute witness in Blender comes back to you.
   Decision: owner answer A14 replaces the proposal: open formats for every product (USD, Alembic, OpenVDB),
   one file per frame; per kind of geometry the format is chosen by the data-driven format table of 5.7 from
   witness tests run by the Blender add-on in Blender 5.2 and 5.3, USD when both pass; today points and
   surface are Alembic, volumes OpenVDB, solid motion JSON, every USD candidate "to be proven by the Blender
   add-on". This module also writes the simulation-particle playback (from I/O) and solid motion. The three
   libraries (and oneTBB, Imath, zlib) live in `fluidity_interchange`, linked by the CLI only. "Deliberately
   not promised" as rewritten in 5.7; a failed witness of a kind's last candidate withdraws the channel and
   comes back to the owner (recommended option, owner's standing instruction, 2026-11-14).
   Replaced in part (A16, 2026-11-14): the witness tests run in Blender 5.3 only; a `WitnessRecord` holds one
   result, `blender_5_3`, and USD is chosen when it passes every witness there. The library versions of the
   first version of 5.7 (matched to Blender's bundled libraries) are replaced by decision 11.
3. **The per-frame work is CPU only.** `RecordStep` has CPU and GPU code; `BuildRenderFrame` has CPU code
   only, for the four reasons of section 6. Alternative: GPU code for the field, the foam motion and the
   counts as well (the three wide loops), with their memory in the estimate; it roughly doubles this
   module's device code and makes output compete with the solver. Recommendation: CPU only now; measure
   on the breaking wave before adding device code.
   Decision: CPU only now, measured on the breaking wave before any device code; the writers are CPU only.
   Recommended option (owner's standing instruction, 2026-11-14).
4. **A third folder, `render_output/`.** The programme lists `surface/` and `foam_and_spray/`. The lattice,
   the motion sample, the recorder, the volumes, the channel table and the frame entry point belong to
   neither. Options: a) `render_output/` (proposed); b) put them in `surface/`, and gas volumes in
   `foam_and_spray/`; c) put the recorder and frame entry point in `io/`. Recommendation: a.
   Decision: a, `render_output/` (with `render_output/interchange/` for the writers). Recommended option
   (owner's standing instruction, 2026-11-14).
5. **Replace per blob, then the whole mesh.** Proposed: a rejected blob takes its built pieces, and if the
   mix is unsafe the whole frame is built from the field. Alternative: any reject builds the whole frame
   from the field (simpler by about 150 lines; in a breaking wave ids would then almost never last, since
   some droplet merges in nearly every frame). Recommendation: per blob.
   Decision: per blob, then the whole mesh. Recommended option (owner's standing instruction, 2026-11-14).
6. **Volume offset kept, as a report and never a failure.** Alternative: drop the offset and report the
   two volumes only (the report treats mesh volume as an observation). Recommendation: keep it; small
   blobs of a smoothed field are visibly too small without it. Confirm the limit of `0.25` spacing and
   that 0.5 % is a count.
   Decision: keep the offset, limit `0.25` spacing, 0.5 % is a count; the frame's target is
   `cell_resize::ComputeLiquidTotals` (5.3 step 5). Recommended option (owner's standing instruction,
   2026-11-14).
7. **Snow and sand surfaces.** `main` has points only. The same field and mesh work for any material at no
   extra code. Proposed: offered as a `SurfaceGroup` when the scene asks, points by default, no promise on
   how good a granular surface looks. Alternative: reject such a group in `CheckRenderSettings`.
   Decision: the alternative. Snow is written as points only, as on `main`, and `CheckRenderSettings`
   rejects a group with a material that is not a liquid (`kBadInput`); no granular surface in the main
   sequence. Owner answer A9 with the option `changes-to-apply.md` records as recommended for this module
   ("snow points only as on `main`"), owner's standing instruction, 2026-11-14.
8. **Three or more liquids that do not mix in one group.** `main` builds an inner sheet for exactly two.
   Proposed: the same for two; for more, face labels and shares only. Alternative: reject. Note that the
   step lists `kLiquidsThatDoNotMixAreBlended` as a known limit until pressure's sharp boundary exists.
   Decision: an inner sheet for two, face labels and shares for more. The sharp boundary is now pressure's
   (conflict C4, one distance field per liquid material), its first commits behind a capability exclusion;
   the surface does not depend on it. Recommended option (owner's standing instruction, 2026-11-14).
   Updated (plan decision, 2026-11-14, "Pressure (continued)"): the pressure plan implements the sharp boundary
   and removes the exclusion; there is no capability reason for touching liquids that do not mix.
   `kLiquidsThatDoNotMixAreBlended` in the question above is historical.
9. **Citations that are not in the old bibliography.** The smoothed-share surface (Zhu and Bridson 2005 is
   what the old paper names, without an entry), the six-tetrahedra mesh, the Blender manual pages used in
   5.7, the Alembic and OpenVDB references. Please supply entries, or agree to author and year only.
   Decision: author (or project) and year only until entries exist; OpenUSD is added to the
   list. Recommended option (owner's standing instruction, 2026-11-14).
10. **Smaller points to confirm.** Motion is kept as float values on a lattice at `samples_per_frame = 1`
    by default (`main`: full copies, off by default). The queue makes the simulation wait and never drops a
    frame. Above `max_events_per_step` the largest events are kept. New foam particles are spread inside
    their lattice cell (`main`: all at one point). Foam with no known liquid velocity keeps its velocity.
    The mesh lies on solids with no overlap. `math::Expm1` joins `base/`. Defaults: surface spacing half
    the smallest cell width, smoothing two spacings, reach 4 spacings, `max_edge_in_spacings` 2.5.
    Decision: all confirmed as listed; `math::Expm1` is now in `base/` (`changes-to-apply.md`, additions to
    Foundation). Recommended option (owner's standing instruction, 2026-11-14).
11. **Library versions.** The first version of 5.7 matched each library to the version Blender bundles.
    Decision: owner answer A19: each of zlib, Imath, oneTBB, Alembic, OpenVDB and OpenUSD is pinned exactly to
    the newest release Conan Center (`https://center2.conan.io`) lists when its build task runs; version tests
    compare with that pin; no text relies on Blender's bundled versions (5.7).
12. **Witness records with one Blender version.**
    Decision: owner answer A16: `WitnessRecord { property; blender_5_3; evidence; }` (3.6); `ChooseFormat` takes
    the first candidate whose every record is `kPassed`, else the last; `capabilities` prints `{"property",
    "blender_5_3", "evidence"}` (5.7, 7).
13. **The recorder's copies in a batched or GPU run.** Owner principle A17: a run is all CPU or all GPU, and the
    copies between them are planned and counted.
    Decision: `ReadRecorderCopyCounts(const StepRecorder&)` returns the GPU recorder's read-backs as
    `step::CopyCounts` (one copy per `ReadBuffer`, cumulative; `{}` on the CPU); I/O adds the difference per
    committed step to the measured cost (seam 11) and the budget counts the recorder through
    `BuildRenderOutputMemoryEstimate` (3.3, 4 I/O item 9, 6, 7). Plan decision, 2026-11-14 (buffers in batched
    runs; planned cost).
14. **Whole-grid block index of a batch on a GPU grid.** The first form read `GridPiece::block_in_whole_grid()`,
    which is empty for a GPU piece (A17).
    Decision: `FoamAndSprayInputs::block_in_whole_grid` is the batched run's CPU copy of the piece map, kept by the
    three-argument `step::ApplyGridPiece` for stage 18; never the empty span of a GPU piece (4, 6). Plan decision,
    2026-11-14 (buffers in batched runs); owner principle A17.
15. **What "batched equals one piece" means here after A15.** `RecordStep, BatchedEqualsOnePiece` cuts the same
    tries into 2 and 5 sub-ranges: a local computation by range, not two run modes, so it keeps exact bytes.
    Decision: kept exact; the text no longer says a batched run records the bytes of a one-piece run (3.3, 6).
    Owner answer A15.
16. **(A31) No "scratch"; one form per function.** `PointScratch` is `PointObjectBuffers`; the recorder's device
    memory is its working buffers. The once-per-frame CPU orders no longer build their own working memory:
    `BuildSurfaceField`, `BuildSurfaceFrame` and `BuildFoamVolume` take an `OrderBuffers&`, which `RenderOutput`
    owns for the run (`frame_order_buffers`), so `BuildRenderOutput` returns `std::expected` (`kOutOfMemory`).
    `cell_resize::ComputeLiquidTotals` is called in its one form with the recorder's `SumBuffers`. Decision: as
    listed. Source: `owner-answers.md` A31; render output plan Tasks 17, 18, 26, 37, 44, 45.
17. **(A30) Buffers on different processors.** A precondition (`Requires:`, debug assertion), not an error value.
    Decision: as listed. Source: `owner-answers.md` A30.
18. **(Reconcile item 26) A bound for the field parts.** The stable order of 5.3 step 3 had no bound, so no memory
    estimate entry. `RenderLimits::max_field_parts = 1u << 26` (as `max_vertices`) bounds it; reaching it leaves the
    surface of that group out of the frame (`kTooManyFieldParts`, appended as value 15); the estimate counts
    `frame_order_buffers` at `max(max_field_parts, max_foam_particles)`, `kWholeRun`, `kCpu`. Decision: as listed
    (the default is new, for the owner to confirm). Source: `reconcile-todo.md` item 26; plan Tasks 10, 17, 44.

**Open** (for the owner):

- OpenVDB 12.1.1, the newest on Conan Center on 2026-11-14, is older than the 13.0 Blender 5.3 bundles; the
  witness tests (W1, W8, W10) are what shows Blender 5.3 reads its files.
- The Conan remote: every pin depends on `conancenter` pointing at `https://center2.conan.io`, since the old
  `center.conan.io` is frozen; the library tasks check and switch it. For the owner to confirm.
- A maximum Blender version: the Blender add-on's manifest has none, and the witnesses are not run again until
  the supported Blender version changes.
