# Blender add-on module design

Status: updated on 2026-11-14 with the owner's answers (A1 to A22), `changes-to-apply.md`, and the Blender
plan's decisions recorded in `plan-decisions.md` (D1 names under A22, D2 test levels, D3 Blender 5.3 API facts,
D8 finding codes, D12 one-version witness records, D13 wind from Blender force fields). Updated again on 2026-11-14 with A24, A25 and
the reconcile pass's coordinator item (7) (section 11, decisions 17 and 18), and with A27 (decision 19). Module 12 of the
build order ("Blender add-on", file 14 of this folder), the last consumer before sand: it adapts to the final
designs of I/O and CLI (commands, events, exit codes, scene file, `file_list`, `report`, `check --cache`,
`run --outputs`), of surface with foam and spray (products, channel table, format table and its witness
records), and of step, solids, gas, materials, cell resize and batches (what the scene file carries, what the
capability check and the run plan report).

Adaptations of the template for a Python add-on: section 3 gives Python modules, classes, operators, panels
and property groups; section 6 is "Main thread and background work"; identifiers follow the decided
vocabulary in Python style (`snake_case` functions and fields, `CamelCase` classes, Blender's mandatory
`FLUIDITY_OT_`, `FLUIDITY_PT_`, `FLUIDITY_UL_` class prefixes). Blender 5.3 is the one supported version (A16).
The API facts already measured are listed in 5.14; each is re-checked in Blender 5.3 by
`tools/blender/probe_blender_api.py` in the first task that uses it. Any other Blender API detail is marked "to
verify against the Blender 5.3 documentation".

## 1. Purpose

The add-on lets an artist describe a scene in Blender (a domain box and the objects that take part in it),
asks the native program whether that scene can run and what it will cost, runs it without freezing Blender,
and shows the result as ordinary Blender objects that still play back and render when the add-on is switched
off. It computes no physics: every number the solver uses is written into one scene file, and everything the
artist sees about a run comes from events and files the native program wrote.

**Owns:** the property groups stored in the `.blend`; the role, shape and motion model of participants; the
substance presets and the reference scene presets (breaking wave, syrup, fire, and a snow example); reading
Blender data into an immutable scene description and writing the scene file; the quick check (pure Python) and
the display of the native check; the run supervisor (process, events, log, pause, stop, resume, rebuild) and
the run state machine; the run records and the output registry; linking listed product files to Blender
objects with Blender's native readers as one transaction; reading Blender's own force field objects into the
scene file's wind keys and force field findings (A20, 5.13); the panels; the extension package and how the
native executable is found; the **witness tests** that decide between USD and Alembic for points, surface and
solid motion (run headless in Blender 5.3 with the add-on disabled), the program that writes their files, and
the `WitnessRecord` lines they produce; the add-on's tests.

**Does not own:** any solver rule (all other modules); the scene file schema, the ready scene, the saved state,
the `run_folder`, its `file_list` and `WriterLock`, the run fingerprint, the output declarations, the CLI
commands, events and exit codes (I/O and CLI); the bytes of product files, the channel table, the format table
and the choice rule `ChooseFormat` (surface, foam and spray, called render output below); the capability
check, the run plan and the memory estimate (step); shading beyond three small starter assets.

## 2. What exists today

All paths are under `blender/fluidity/` on `main` unless another folder is named. Report references are to
`notes/blender_manuscript.md` (B) by line.

| Where | What it does | What is wrong |
|---|---|---|
| `__init__.py:43-266` | `FluidiTyLiquidSource` and one `FluidiTySettings` group on the scene: 60 fields for every scene kind (`snow_*`, `rigid_*`, `elastic_*`, `boundary_mode`, `paddle`, `cutter`). | One enum `boundary_mode` decides geometry, contact, an example setup and a compatibility path (B 93). Sources are `emission_rate` in particles per second and `emitted_volume` (`:50-51`, B 127). `substance_id` and `phase_id` are bare integers. Live run status is stored in RNA (`:257-266`). |
| `__init__.py:298-330` `SceneAction.execute` | Runs one check, shows the first `SceneError(field, message)`. | One error at a time (B 143). `check_device` (`:287-295`) pins one GPU model by name. |
| `__init__.py:358-392`, `jobs.py:41-55` | Start: builds a text packet, runs `blender_bridge resolve` and `simulate check` synchronously (20 s timeout), then starts the job. | Blocking native calls inside an operator on the UI path (B 243). |
| `jobs.py:66-163` `Job` | One long-lived process; a reader thread that never calls Blender; queue of 64; 16 records per poll; `published: true` moves the committed frame; cancel on stdin, kill after 10 s. **The discipline is kept.** | Heartbeat age is measured when the record is dequeued (`:136-140`, B 295). A full queue blocks the solver (`:105-111`, B 245). Jobs are keyed by `scene.as_pointer()` and hold the scene (`:338-384`, B 247). |
| `jobs.py:177-343` `FrameJob`, `CutterJob`, `RigidJob` | One process per frame for cutting and the free box. | A second lifecycle: no heartbeat, stderr discarded, cancel waits for the frame, another close path (B 168). Resume compares packet bytes and scans checkpoint file names. |
| `jobs.py:22-26` `binary` | `bin/fluidity_<name>.exe`. | Windows-only lookup (B 553). Two executables (`simulate`, `blender_bridge`). |
| `scene_inputs.py:21-59` `box_bounds` | Accepts only an 8-vertex world-aligned box with no parent, modifier or constraint. | Shape is inferred from mesh topology. |
| `scene_inputs.py:62-90, 162-181` | Paddle and collider motion from exactly two linear location keys. | No rotation, no general path. |
| `scene_inputs.py:138-140, 240, 271-272, 295-296, 314-315, 385-395, 416, 494-495, 551` | Y-up conversion `(x, y, z) -> (x, z, -y)` of bounds, velocities, the box pose and inertia axes, written by hand at eleven places. | Disappears (section 5.11). |
| `scene_inputs.py:426-559` | Five text packet versions `FLUIDITY-BLENDER-1` to `-5`, chosen by which features are used. | A format per feature. 24 fps, frames 0 to 240, 1024 particles, 100 m limits are hard-wired (`:16-19`). |
| `authoring.py:23-29, 137-165` | `apply_preset` writes density and surface tension; editing sets the preset to `CUSTOM` (`__init__.py:38-40`). **The rule is kept.** | Only two fields; a second, separate snow preset operator. |
| `authoring.py:109-114`, `interchange.py:42-46`, `ui.py:77-84, 172` | Role enums declared three times. | The snow mound row asks both operators for a role neither declares (B 156): an RNA enum error at draw. |
| `authoring.py:184, 188` | Help text. | Says viscosity and volume output do not exist; both do (B 37). |
| `ui.py:295-315` | Build Surface sits under "Import & Export". | The surface is not in the normal journey (B 35). `ui.py:278-292` shows one `playback_object`. |
| `playback.py:99-103` `relative` | Cache path relative to the project. | Raises when the cache is on another drive (B 585). |
| `playback.py:134-262` `attach_channel` | Exports a fresh `.abc` of all frames (60 s timeout), imports it with `bpy.ops.wm.alembic_import`, checks the point count at the last frame, swaps the cache file into the existing object, restores the frame. **The preservation of the artist's object, modifiers and materials is kept.** | Whole-prefix export on every attach (B 525); a synchronous native call and an import operator on the UI path; water and snow are swapped one after the other, not together (B 493). |
| `playback.py:264-331` `attach_rigid` | Free-box poses become location and quaternion keys on a copy of the object. **Kept** for the JSON solid motion format. | One process per frame to read poses; `solver_to_blender` rotation (`:282`). |
| `surface_playback.py:20-125` | Surface as `.usda` through `bpy.ops.wm.usd_import`; keeps materials, frame and selection. | Writes `fluidity_surface_source_execution` (`:101`). |
| `foam_projection.py:118-125` | Foam mask on the surface by a Geometry Nodes group; requires the same source run. | Reads `fluidity_source_execution`, a key the surface attach never writes (B 495). The test sets the key by hand. |
| `volume_playback.py:16-80`; gas branch `gas_volume.py:16-60` | Python writes `.vdb` with Blender's `openvdb` module, a known-region grid beside the density grids. | Conversion in Python; `(x, -z, y)` index and origin conversion (`:16-19, 46`). |
| gas branch `gas_authoring.py:15-17, 27-90, 121-134` | Fire: a source box with `Fuel Rate` and `Heat Rate` custom properties sampled per frame with `scene.frame_set`; a Wind force field as "draft"; `.vdb` per frame attached as a `Volume` sequence. | Own packet, own start, own attach; fixed burn numbers (`:87-88`); frame stepping on the UI path. |
| `blender_manifest.toml`, `tools/blender/package_m1_extension.py:15-55` | Extension manifest; packager pinned to two Blender builds, offline `extension build` and `extension validate`. **Kept as the method.** | `platforms = ["windows-x64"]`; the packager rewrites `blender_version_min` to the builder's version. |
| `tools/blender/verify_*.py` (18 scripts), `verify_m1_extension.py:27-39` `drain` | Headless journeys that call `tick()` by hand; install, bake, cancel, resume, relink, render with the add-on disabled. **The journey is kept.** | One script per milestone; several build their state by hand instead of through the operators (B 626). |
| `tools/blender/point_cache_writer/blender_bridge.cc` (Blender chapter, channel matrix) | Alembic points and surface, USD surface, opened in Blender; point, vertex and triangle counts checked after import. | Nothing else was ever measured in Blender: no velocity, radius, attribute, axis or motion blur check, and no test that decides a format. |

Y-up specifics: the solver on `main` is Y-up, so every position, velocity, pose and grid index crosses a
hand-written rotation in the add-on, in both directions. `tools/blender/point_cache_writer/blender_bridge.cc:154`
relies on "solver and Alembic are both Y-up", so points needed no conversion at the Alembic file.

## 3. Public interface

Folder `blender/fluidity/`, Python package `fluidity` (installed as a Blender extension, so imports are
relative). The package has two layers. `logic/` (named for what it holds: the logic that runs without Blender;
D1) never imports `bpy`, `mathutils` or `bmesh` and is tested with plain `pytest`. The files beside it are the
thin Blender layer. Names are NEW (N) unless marked D (DECIDED word). The witness tools of 3.12 are not
packaged. Under A22, Blender's own words `link`, `install`, `support` (in "Copy Support Report") and `node` (node
groups) are kept in the names below in their everyday sense; the rules check prints a note, not a failure (D1).

```text
blender/fluidity/
  blender_manifest.toml
  __init__.py              register(), unregister(); nothing else
  logic/
    addon_error.py         AddonError, AddonFailure, to_message
    roles.py               the one table of roles, shapes and motions
    scene_fields.py        the one table: property -> scene file field, unit, label, tooltip
    presets.py             substance presets and reference scene presets (data only)
    scene_description.py   immutable dataclasses of one scene
    scene_file.py          write_scene_file, compute_scene_digest
    findings.py            Finding, Severity, Repair, build_findings_order
    quick_check.py         check_scene_quickly
    native_executable.py   find_native_executable
    native_facts.py        Capabilities, NativeCheck, RunFolderFacts, OutputDeclaration (mirrors of I/O's JSON)
    native_calls.py        one-shot calls of the CLI that print one JSON event
    force_fields.py        WIND_ACCELERATION_PER_STRENGTH, FORCE_FIELD_TYPE_TABLE, build_wind_keys,
                           check_force_fields (3.13)
    events.py              read_events (the journal), Event
    run_state.py           RunState, RunStatus, apply_event, apply_request
    supervisor.py          RunSupervisor
    run_folder.py          names and paths of runs under a cache root
    output_plan.py         LinkAction, LinkStep, LinkPlan, build_link_plan, OutputRow, LINK_ROUTE_KEYS
  properties.py            property groups
  scene_reader.py          Blender data -> SceneDescription (main thread)
  operators_authoring.py  operators_run.py  operators_outputs.py  operators_inspect.py
  output_links.py          LINK_ROUTES, apply_link_plan (the transaction), read_output_row
  node_assets.py           three starter assets
  ui_model.py              the live status the panels draw from
  ui_timer.py              tick()
  panels.py                handlers.py  preferences.py
  bin/<platform>/fluidity[.exe]   one native executable per packaged platform
tools/blender/witness/     3.12; not packaged
tools/blender/probe_blender_api.py   the Blender 5.3 API probe (5.14); not packaged
native/fluidity/cli/write_witness_files.cc   3.12; program fluidity_write_witness_files, never packaged
```

### 3.1 `logic/addon_error.py`

```python
class AddonError(enum.Enum):
    NONE = 0
    PROJECT_NOT_SAVED = 1               # A run folder needs a saved .blend to be placed beside.
    NATIVE_EXECUTABLE_NOT_FOUND = 2     # No executable for this platform in the package or at the given path.
    NATIVE_EXECUTABLE_DOES_NOT_MATCH = 3  # Its `protocol` or scene schema version is not the add-on's, or its
                                        #   coordinate text is not "right_handed_z_up_metres".
    NATIVE_CALL_FAILED = 4              # A one-shot call crashed, timed out, exited 1 or 2, or printed no event.
    EVENTS_NOT_READABLE = 5             # A journal line is too long, not JSON, or its sequence has a gap.
    RUN_ALREADY_ACTIVE = 6              # One run at a time per Blender session.
    RUN_FOLDER_NOT_USABLE = 7           # Missing, not writable, or inside another run's folder.
    OUTPUT_FILE_MISSING = 8             # A file the file list names is not on disk.
    OUTPUT_DOES_NOT_MATCH_RUN = 9       # A declaration's run fingerprint is not the registry's.
    FORMAT_NOT_LINKABLE = 10            # NEW. A declaration names an InterchangeFormat with no route in LINK_ROUTES.
    LINK_FAILED = 11                    # Blender could not read a listed file. Nothing was changed.

@dataclass(frozen=True)
class AddonFailure:                     # Returned, never raised across a module boundary.
    error: AddonError
    detail: str = ""                    # A path, an exit code or the native message. Not translated.

def to_message(error: AddonError) -> str: ...
```

A problem with the artist's scene is not an `AddonError`. It is a `Finding` (3.4). Functions in `logic/` return
`Value | AddonFailure`; operators turn a failure into `self.report({'ERROR'}, ...)`.

### 3.2 `logic/roles.py`, `logic/scene_fields.py`, `logic/presets.py`

```python
class Role(enum.Enum):                  # What an object is in the scene. N.
    NONE = "NONE"
    DOMAIN = "DOMAIN"                   # D word: the DomainBox.
    START_MATERIAL = "START_MATERIAL"   # A shape filled with a substance at the first frame (scene file start_regions).
    SOURCE = "SOURCE"                   # Adds a liquid over time, as volume per second.
    GAS_SOURCE = "GAS_SOURCE"           # Adds fuel and heat: a world-aligned box (gas::GasSource).
    SOLID = "SOLID"                     # D word.
class Shape(enum.Enum):                 # D words, from contact::SolidShape; BALL only for START_MATERIAL.
    BOX = "BOX"; FLAT_SOLID = "FLAT_SOLID"; PLATE = "PLATE"; BALL = "BALL"
class Motion(enum.Enum):                # D words, solids::SolidMotionKind.
    FIXED = "FIXED"; DRIVEN = "DRIVEN"; FREE = "FREE"
class DrivenBy(enum.Enum):
    KEYS = "KEYS"                       # The object's own location and rotation keys.
    WAVE_STROKE = "WAVE_STROKE"         # A sine stroke: the wave paddle (solids::WavePaddleSettings).

@dataclass(frozen=True)
class RoleRow:
    role: Role; label: str; tooltip: str; icon: str
    shapes: tuple[Shape, ...]           # Allowed shapes, first is the default.
    motions: tuple[Motion, ...]         # Allowed motions. Empty when the role does not move.
    substance_kinds: tuple[str, ...]    # Allowed substance kinds; empty when the role takes no substance.
ROLE_TABLE: tuple[RoleRow, ...]
# START_MATERIAL: BOX, BALL; kinds liquid, snow, springy_solid (the sand module adds sand: sand design, section 4).
# SOURCE: BOX; kind liquid.
# GAS_SOURCE: BOX; no substance (the domain's one gas substance).  SOLID: BOX, FLAT_SOLID, PLATE; all motions.
def to_enum_items(table) -> tuple[tuple[str, str, str, str, int], ...]: ...   # For every EnumProperty.
```

`ROLE_TABLE` is the only place a role is declared. The `role` property, the `role` argument of every
operator, the rows of the participants list and the scene file writer are generated from it.

```python
@dataclass(frozen=True)
class SceneField:
    property_path: str                  # "substances[].ingredients[].viscosity", relative to the domain.
    scene_file_path: str                # "materials[].ingredients[].viscosity": the path I/O reports back.
    unit: str                           # "Pa s". Shown in the label and written in the tooltip, first.
    label: str; tooltip: str
    layer: Layer                        # EVERYDAY, PHYSICAL_DETAIL, SOLVER_DETAIL, PERFORMANCE.
    changes: Changes                    # SIMULATION or OUTPUT: what an edit makes stale (5.3 step 5).
SCENE_FIELDS: tuple[SceneField, ...]
def find_scene_field(scene_file_path: str) -> SceneField | None: ...   # Indexes like [2] are matched as [].
```

Property definitions take `name` and `description` from this table, so a label, its unit and its tooltip
cannot disagree with what is written to the scene file. No tooltip states what the solver supports: such
sentences come only from the native capability findings (5.3). NEW (A25): the viscosity row's tooltip is exactly
"Pa s. 0: the liquid has no viscosity and the scene uses the faster pressure solve; above 0: the viscous
solve, for the whole run" (a liquid is viscous exactly when its viscosity is above 0, fixed for the run).

```python
@dataclass(frozen=True)
class SubstancePreset:
    preset_id: str; label: str
    kind: str                                     # "liquid", "gas", "snow", "springy_solid" (MaterialKind, A9).
    values: Mapping[str, float]                   # property_path -> value. Concrete numbers only.
SUBSTANCE_PRESETS: tuple[SubstancePreset, ...]    # water, syrup, snow, springy solid, air (5.2)
@dataclass(frozen=True)
class ReferenceScene:
    scene_id: str; label: str
    builder: str                                  # Name of a function of presets.py that returns a SceneDescription.
REFERENCE_SCENES: tuple[ReferenceScene, ...]      # breaking wave, syrup, fire, snow example (5.2). No sand (A9).
def compute_preset_changes(current: Mapping[str, float], preset: SubstancePreset) -> dict[str, tuple[float, float]]: ...
```

### 3.3 `logic/scene_description.py`, `logic/scene_file.py`

Frozen dataclasses, SI units, Z-up, world frame. They mirror the scene file of section 4.1 field for field and
are the "immutable request" of a check or a run: later edits in Blender cannot change them.

```python
Orientation = tuple[float, float, float, float]   # (w, x, y, z), unit, own axes to world (solids::Orientation).
@dataclass(frozen=True)
class PoseKey:                          # One key of a driven path. I/O derives angular_velocity_until_next.
    time: float                         # s from the first frame.
    origin: Vec3                        # m, world.
    orientation: Orientation
@dataclass(frozen=True)
class ParticipantDescription:
    participant_id: int                 # Stable, nonzero. Becomes region_id, source_id or solid_id.
    object_name: str                    # Written as `name` for messages; never part of the fingerprint.
    role: Role; shape: Shape
    origin: Vec3; orientation: Orientation; half_size: Vec3   # m; m.
    substance_id: int = 0               # Becomes material_id.
    motion: Motion = Motion.FIXED
    path: tuple[PoseKey, ...] = ()      # DRIVEN with KEYS.
    after_path: str = "stays_at_last_key"   # or "leaves_scene" (solids::AfterPath).
    wave_stroke: WaveStroke | None = None   # stroke m, wave_period s, ramp_duration s, travel_direction.
    free_solid: FreeSolid | None = None # mass kg, moves_along_x/_y/_z, turns, start_linear_velocity m/s,
                                        #   start_angular_velocity rad/s, steady_force N.
    friction_scale: float = 1.0
    start_velocity: Vec3 = (0, 0, 0)    # m/s. START_MATERIAL.
    same_material_as: int = 0           # START_MATERIAL of snow or a springy solid: region_id it shares a body with.
    source: SourceDescription | None = None        # volume_rate m3/s, velocity m/s, start_time, end_time s.
    gas_source_keys: tuple[GasSourceKey, ...] = () # time s; fuel_rate kg/(m3 s); heat_rate W/m3.
@dataclass(frozen=True)
class SceneDescription:
    domain_id: str; domain_low_corner: Vec3; domain_high_corner: Vec3; closed_sides: ClosedSides
    gravity: Vec3; frames_per_second: float; first_frame: int; last_frame: int
    smallest_cell_width: float; size_level_count: int
    substances: tuple[SubstanceDescription, ...]; pairs: tuple[PairDescription, ...]
    participants: tuple[ParticipantDescription, ...]
    open_sea: OpenSea | None
    gas: GasDescription | None          # law, burn; the gas substance is in `substances` with kind "gas".
    wind_keys: tuple[WindKey, ...]      # time s; acceleration m/s2, world. From build_wind_keys (3.13): the
                                        #   Wind force fields read by the scene reader. Empty: no wind or no gas.
    cell_resize: CellResizeDescription | None      # None: no resize.
    solver: SolverDetail; run: RunDetail; outputs: OutputRequest

def write_scene_file(description: SceneDescription, path: Path) -> None | AddonFailure: ...
def compute_scene_digest(description: SceneDescription) -> str: ...   # Hash of the bytes write_scene_file writes.
```

`write_scene_file` writes UTF-8 JSON with Python's standard library (I/O decision 6), keys in the order of 4.1,
every float with `repr` (round trip), ids as JSON integers below 2^53. `compute_scene_digest` is only the
add-on's "did anything I write change" test. Whether a run may continue is decided by the native
`run_fingerprint` (D) through `check --cache`, never by this digest.

### 3.4 `logic/findings.py`, `logic/quick_check.py`

```python
class Severity(enum.IntEnum):           # Order of display. The first three are I/O's severities.
    BLOCKS_RUN = 0                      # "blocks_run". The run cannot start.
    HAS_KNOWN_LIMIT = 1                 # "has_known_limit". It runs; a stated limit applies.
    COSTS_MORE = 2                      # Quick check only: time, memory or disk larger than the artist may expect.
    DISPLAY = 3                         # Quick check or an output: what is shown is not what was asked
                                        #   (a product not built in a frame, a channel written but not witnessed).
    NOTE = 4                            # "note".
@dataclass(frozen=True)
class Repair:                           # One suggested change, applied by fluidity.apply_repair.
    label: str                          # "Run on CPU", "Select Paddle", "Use 0.02 m".
    operator: str                       # bl_idname.
    arguments: Mapping[str, str | int | float]
@dataclass(frozen=True)
class Finding:                          # Mirrors I/O's Finding table field for field (4.1).
    module: str                         # io::ToName(FailingModule): "solids", "step", "io"; "addon" for the quick check.
    code: str                           # Stable ToName code: "plate_too_thin_for_liquid". Never translated.
    severity: Severity
    message: str                        # Native findings: io::ToMessage with names filled in.
    participant_id: int = 0             # From solid_id, source_id or a region's id; 0: about the whole scene.
    material_id: int = 0; particle_id: int = 0
    object_name: str = ""; scene_file_path: str = ""   # The field it is about, when there is one.
    magnitude: float = 0.0; limit: float = 0.0; unit: str = ""
    suggested_changes: tuple[str, ...] = ()            # ToName(SuggestedChange) values and I/O's own four.
    repairs: tuple[Repair, ...] = ()    # Built from suggested_changes by REPAIR_TABLE.
    found_by: str = "quick_check"       # or "scene_reader", "native_check", "run", "report".
REPAIR_TABLE: Mapping[str, Callable[[Finding], Repair | None]]   # One entry per suggested change name.
def build_findings_order(findings: Iterable[Finding]) -> tuple[Finding, ...]: ...   # severity, participant, module, code

def check_scene_quickly(description: SceneDescription, capabilities: Capabilities) -> tuple[Finding, ...]: ...
```

`REPAIR_TABLE` has one entry for each of the 11 `SuggestedChange` names of the step (`use_one_surface_tension`,
`use_smaller_cells`, `use_one_cell_size_there`, `make_solid_thicker`, `make_solid_fixed`, `align_solid_with_axes`,
`separate_the_objects`, `use_fewer_ingredients`, `make_domain_smaller`, `put_all_liquids_in_one_set`,
`use_fewer_solids`; A27 removed `set_viscosity_to_zero` and its property edit to viscosity 0: the sand module adds
it back with `kViscousLiquidWithSand`, and its entry with it) and for I/O's `run_on_cpu`, `use_bigger_cells`,
`start_new_run`, `free_disk_space`. An entry may return `None` (the change has no one-click form; the message
says what to do). A name the table lacks is shown as text; a test fails when `capabilities` lists one.

The add-on's own finding codes (module `"addon"`; stable, never translated; D8). Found by `"quick_check"`:
`project_not_saved`, `role_without_domain`, `object_in_two_domains`, `shape_not_allowed`, `motion_not_allowed`,
`substance_missing`, `substance_wrong_kind`, `more_than_one_gas`, `shares_do_not_sum_to_one` (tolerance `1e-9`),
`ingredient_name_empty`, `ingredient_names_collide`, `outside_domain`, `gas_source_turned`, `source_turned` (only
where the scene file schema cannot carry a turned source box), `size_not_positive`, `rate_not_positive`,
`source_window_outside_frames`, `matrix_sheared_or_mirrored`, `motion_not_keys`, `too_many_plates`,
`too_many_free_solids`, `too_many_ingredients`, `too_many_size_levels`, `motion_samples_out_of_range`,
`surface_spacing_costly` (`COSTS_MORE`: spacing under a quarter of the smallest cell width). Found by
`"scene_reader"`, since only Blender knows the field objects (5.13): `wind_falloff_not_honoured`,
`wind_noise_not_honoured`, `wind_shape_not_honoured`, `force_field_not_honoured` (each `HAS_KNOWN_LIMIT`),
`wind_flow_not_honoured`, `wind_without_gas`, `force_field_ignored` (each `NOTE`). Ingredient names are normalised
exactly as the materials module's `BuildMaterialTable` does.

### 3.5 `logic/native_executable.py`, `logic/native_facts.py`, `logic/native_calls.py`

```python
PROTOCOL = 2                            # events.fbs protocol this add-on reads (I/O 3.7).
SCENE_SCHEMA = 2                        # scene_file.fbs version it writes.
@dataclass(frozen=True)
class NativeExecutable:
    path: Path; executable_version: str; executable_digest: str; protocol: int; scene_schema: int
def to_platform_name(os_name: str, machine: str) -> str | None: ...   # platform.system(), platform.machine() ->
                                                                       #   "windows-x64", "linux-x64", "macos-arm64"
def find_native_executable(package_folder: Path, user_folder: Path, given_path: str) -> NativeExecutable | AddonFailure: ...
```

```python
# native_facts.py: frozen mirrors of I/O's JSON, built by from_event(dict) functions. Unknown keys are kept in
# `other` and never read; a missing key the add-on needs is NATIVE_CALL_FAILED.
@dataclass(frozen=True)
class WitnessRecordFact:                # render_output::WitnessRecord as `capabilities` prints it (D12, A16).
    property: str                       # "point_radius_arrives"
    blender_5_3: str                    # "not_run", "documented", "passed", "failed". No other Blender version.
    evidence: str
@dataclass(frozen=True)
class FormatCandidateFact: format: str; extension: str; witnesses: tuple[WitnessRecordFact, ...]
@dataclass(frozen=True)
class FormatChoiceFact:                 # render_output::FormatChoice.
    kind: str                           # "points", "surface_mesh", "volume", "solid_motion"
    candidates: tuple[FormatCandidateFact, ...]    # Preference order: USD first, the default last.
    chosen: str                         # "usd", "alembic", "open_vdb", "solid_motion_json"
@dataclass(frozen=True)
class OutputChannelFact:                # One row of ReadOutputChannels().
    product: str; blender_name: str; alembic_name: str; usd_name: str; openvdb_name: str; json_name: str
    domain: str; unit: str; value_type: str; kind: str; witnessed_by: str   # value_type: ChannelValueType
@dataclass(frozen=True)
class StaticLimits:                     # solids::kMaxPlateCount, kMaxFreeSolidCount; particles::kMaxIngredientCount;
                                        # grid::kMaxSizeLevels; render output's 1..64 motion samples.
    max_plate_count: int; max_free_solid_count: int; max_ingredient_count: int; max_size_levels: int
    max_motion_samples_per_frame: int
@dataclass(frozen=True)
class Capabilities:                     # The `capabilities` event.
    executable_version: str; executable_digest: str; protocol: int; scene_schema: int; coordinates: str
    run_on_values: tuple[str, ...]      # "cpu", and "gpu" when the build has GPU code.
    capability_reasons: tuple[tuple[str, str, str], ...]   # (code, level, message); level =
                                                           #   ToName(step::CapabilityLevel)
    limits: StaticLimits
    format_table: tuple[FormatChoiceFact, ...]
    output_channels: tuple[OutputChannelFact, ...]
    interchange_libraries: Mapping[str, str]         # open_usd, alembic, openvdb versions.
    processor: Mapping[str, str] | None              # With --run-on gpu: description, total and free bytes.
@dataclass(frozen=True)
class ChannelInFile: blender_name: str; name_in_file: str; is_witnessed: bool
@dataclass(frozen=True)
class OutputDeclaration:                # io::OutputDeclaration.
    output_id: str                      # "points_17", "surface_3", "droplets_3", "foam_and_spray",
                                        #   "foam_volume", "gas_volume", "solid_motion".
    product: str                        # "particles", "surface", "droplets", "spray", "foam", "bubbles",
                                        #   "foam_volume", "gas_volume", "solid_motion"
    material_id: int
    format: str                         # As FormatChoiceFact.chosen, fixed for the run.
    file_pattern: str                   # "outputs/points_17/points.{frame:06d}.abc". Relative to the run folder.
    object_names: tuple[str, ...]       # "points"; "spray", "foam", "bubbles"; "solid_<solid_id>".
    channels: tuple[ChannelInFile, ...]
    output_fingerprint: str
    needs_simulation_to_rebuild: bool
    contract_version: int
@dataclass(frozen=True)
class OutputMatch: output_id: str; match: str; changed_fields: tuple[str, ...]; needs_simulation: bool
@dataclass(frozen=True)
class CacheComparison:                  # io::CacheComparison: match is "same", "changed" or "absent".
    simulation: str; differing_part: str          # "", "scene", "executable", "run_on", "processor", "run_mode"
    changed_fields: tuple[str, ...]; outputs: tuple[OutputMatch, ...]
@dataclass(frozen=True)
class RunBudgetFacts:                   # NEW (A17, planned cost). scene_checked.budget, I/O schema names.
    run_mode: str; mode_reason: str; verdict: str                  # Strings as the I/O schema sends them.
    processor_budget_bytes: int; processor_allowed_bytes: int      # From processor_memory.
    cpu_budget_bytes: int; cpu_allowed_bytes: int                  # From cpu_memory.
    bytes_per_step: int; copies_per_step: int                      # From copies.
    viscous_path: str                   # NEW (A25). "pressure_solve" | "viscous_solve", fixed for the run.
    liquids: tuple[tuple[str, float, float, float, bool, bool], ...]   # NEW (A25). Per liquid (I/O table
                                        #   ViscousEffectOfLiquid, renamed A27), the schema's order:
                                        #   (ingredient_name, viscosity_pa_s, density_kg_per_m3,
                                        #   viscous_effect_per_step, is_viscous, effect_not_visible).
    lines: tuple[str, ...]              # BuildBudgetLines, verbatim (with the liquids: lines, A25). The add-on
                                        #   computes nothing from the numbers.
@dataclass(frozen=True)
class RunPlanFacts:
    run_mode: str                       # "one_piece" | "batched" | "batched_with_solves_in_pieces" (A15)
    estimate_by_module: Mapping[str, int]; on_processor_bytes: int; allowed_bytes: int
    counts: Mapping[str, int]; batch_plan: Mapping[str, float] | None
    budget: RunBudgetFacts | None       # NEW (A17). None when the event has no `budget`.
@dataclass(frozen=True)
class NativeCheck:                      # The `scene_checked` event.
    can_run: bool; findings: tuple[Finding, ...]; filled_defaults: tuple[tuple[str, str, str], ...]
    run_on_requested: str; run_on_usable: bool
    plan: RunPlanFacts | None           # None when the scene file could not be read.
    run_fingerprint: str; outputs: tuple[OutputDeclaration, ...]
    cache: CacheComparison | None       # With a run folder.
@dataclass(frozen=True)
class LastTry:                          # io::LastTry.
    frame: int; step_count: int; try_number: int; stage: str; committed: bool; reject_reason: str
    sent_at: float                      # s, UTC.
@dataclass(frozen=True)
class RunFolderFacts:                   # The `report` event.
    run_fingerprint: str; last_listed_frame: int; saved_state_frame: int; last_try: LastTry | None
    lock_is_held: bool; lock_process_id: int; lock_host_name: str
    outputs: tuple[OutputDeclaration, ...]
    frames_done: tuple[int, ...]; not_built: tuple[tuple[str, int, str], ...]   # (output_id, frame, reason)
    ignored_end_bytes: int; is_crashed: bool
    files_failing_check: tuple[str, ...]          # With check_files.
    report: Mapping[str, object] | None           # The RunReport as printed.
@dataclass(frozen=True)
class ParticleFacts:                    # The `particle` event. Ids arrive as decimal text and become int.
    frame: int; time: float; particle_id: int; material_id: int; kind: str
    position: Vec3; velocity_for_next_step: Vec3; rest_volume: float; volume_ratio: float; size_level: int
    ingredient_fractions: tuple[float, ...]; other_fields: Mapping[str, tuple[float, ...]]
```

```python
# native_calls.py. Each starts one process with stdout as a pipe, reads it to the end, and returns the last
# event. They block and are called only from the supervisor thread. Exit 0 and, for check, exit 3 are answers;
# exit 1 or 2 is NATIVE_CALL_FAILED with the `failed` event's finding as detail.
def read_capabilities(executable: NativeExecutable, run_on_gpu: bool) -> Capabilities | AddonFailure: ...
def check_scene_natively(executable, scene_file: Path, run_folder: Path | None,
                         stop: threading.Event) -> NativeCheck | AddonFailure: ...   # check [--cache]
def read_run_folder_facts(executable, run_folder: Path, check_files: bool) -> RunFolderFacts | AddonFailure: ...
def read_particle(executable, run_folder: Path, frame: int, particle_id: int) -> ParticleFacts | AddonFailure: ...
def build_run_arguments(executable: NativeExecutable, request: StartRequest) -> list[str]: ...   # Pure.
```

`build_run_arguments` gives exactly one of: `run <scene_file> --into <run_folder> [--last-frame n]`;
`run <scene_file> --into <run_folder> --start-from <other_run_folder> [--start-frame n] [--last-frame n]`;
`run <scene_file> --into <run_folder> --outputs <id>,<id>...`; `resume <run_folder> [--last-frame n]`. It never
passes `--run-on`: `run_on` is in the scene file (A12), so what was checked is what runs.

### 3.6 `logic/run_state.py`, `logic/events.py`

```python
class RunState(enum.Enum):
    NOT_CHECKED = "not_checked"; READY = "ready"; PREPARING = "preparing"; RUNNING = "running"
    STOPPING = "stopping"; STOPPED = "stopped"; DONE = "done"; FAILED = "failed"
    NEEDS_NEW_RUN = "needs_new_run"     # The simulation changed: says what the artist must do (D1).
class RunKind(enum.Enum):
    NEW_RUN = "new_run"; RESUME = "resume"; REBUILD = "rebuild"; START_FROM = "start_from"
@dataclass(frozen=True)
class RunStatus:
    run_id: str; state: RunState; kind: RunKind
    first_frame: int; last_frame: int
    last_listed_frame: int | None       # Fact 1. Every declared product of frames up to here is listed.
    saved_state_frame: int | None       # Fact 2. The latest frame a resume can start from.
    last_try: LastTry | None            # Fact 3. The last work started, committed or not.
    checking_frame: int | None          # While a resume computes again frames after saved_state_frame.
    activity: str                       # RunProgress::Activity: "preparing", "reading_saved_state", "simulating",
                                        #   "checking_frame", "waiting_for_output", "writing_last_frames",
                                        #   "rebuilding_outputs", "stopping".
    preparing_done: int; preparing_total: int; preparing_what: str
    run_on: str; run_mode: str          # "cpu" | "gpu"; "one_piece" | "batched" | "batched_with_solves_in_pieces"
                                        #   (reconcile: the third value, as the started event sends it).
    waiting_frames: int
    last_event_sent_at: float           # s, UTC, written by the native program.
    last_advance_sent_at: float         # s, UTC, of the last commit or listed frame.
    elapsed_by_category: Mapping[str, float]       # s: simulate, snapshot, build_outputs, write, wait.
    memory: MemoryUse | None            # From frame_done: CPU and GPU in use and largest, limit_raise_count.
    outputs: tuple[OutputDeclaration, ...]         # From `started`.
    not_built: tuple[tuple[str, int, str], ...]    # (output_id, frame, reason) from frame_done.
    findings: tuple[Finding, ...]
    exit_code: int | None
    end_event: str                      # "", "done", "stopped", "failed", "process_lost".
def apply_event(status: RunStatus, event: Event) -> RunStatus: ...       # Pure.
def apply_request(status: RunStatus, request: Request) -> RunStatus | AddonFailure: ...   # check, start, ...
def compute_ages(status: RunStatus, now_utc: float) -> tuple[float, float]: ...   # s since last event, last advance

@dataclass(frozen=True)
class Event:                            # One line of events.jsonl.
    protocol: int; event: str; sequence: int; sent_at: float; elapsed_seconds: float
    fields: Mapping[str, object]
def read_events(journal: BinaryIO, max_bytes: int) -> tuple[tuple[Event, ...], AddonFailure | None]: ...
```

`read_events` reads complete lines only (at most 16 KiB each, I/O 5.8), keeps a partial last line for the next
call, and checks `protocol == PROTOCOL` and that `sequence` grows by one.

### 3.7 `logic/supervisor.py`

```python
@dataclass(frozen=True)
class StartRequest:
    run_id: str; kind: RunKind; run_folder: Path
    scene_file: Path | None             # NEW_RUN, REBUILD, START_FROM: the copy inside run_folder.
    last_frame: int | None = None
    output_ids: tuple[str, ...] = ()    # REBUILD.
    start_from_run_folder: Path | None = None; start_frame: int | None = None   # START_FROM.
class RunSupervisor:                    # One per registration. Owns one worker thread and every native process.
    def __init__(self, executable: NativeExecutable, clock, start_process): ...
    # Requests: called on the main thread, return at once, never block.
    def request_check(self, description: SceneDescription, run_folder: Path | None) -> int: ...
    def request_start(self, request: StartRequest) -> None | AddonFailure: ...
    def request_stop_after_frame(self, run_id: str) -> None: ...   # "Pause": writes `stop_after_frame`.
    def request_cancel(self, run_id: str) -> None: ...             # "Stop Now": writes `cancel` (5.5 step 5).
    def request_report(self, run_id: str, run_folder: Path) -> None: ...
    def request_particle(self, run_folder: Path, frame: int, particle_id: int) -> int: ...
    def request_link_plan(self, run_id: str, run_folder: Path, rows: tuple[OutputRow, ...],
                          project_folder: Path | None) -> int: ...   # build_link_plan on the worker thread.
    def follow_run(self, run_id: str, run_folder: Path) -> None: ...   # Reads the journal of a run with no process.
    def cancel_check(self, ticket: int) -> None: ...
    def read_snapshot(self) -> SupervisorSnapshot: ...     # One lock, returns immutable objects.
    def close(self) -> None: ...        # Closes stdin of the live run (= stop_after_frame) and stops the thread.
```

`SupervisorSnapshot` holds a `RunStatus` per known run, the result of each finished check, report, particle
read and link plan (`link_plans`) by ticket, `RunFolderFacts` per run folder, `Capabilities`, and a change counter. `clock` and
`start_process` are parameters so tests replace them with a fake clock and a scripted process.

### 3.8 `logic/output_plan.py`, `output_links.py`

```python
class LinkAction(enum.Enum): CREATE = "create"; REPOINT = "repoint"; LEAVE = "leave"
@dataclass(frozen=True)
class LinkStep:                         # One Blender object for one object name of one declaration.
    output_id: str; object_name: str; action: LinkAction
    geometry_kind: str; format: str     # From the capabilities format table by product; format from the declaration.
    first_file: Path                    # Absolute: file_pattern of the first listed frame under the run folder.
    path_in_blend: str                  # The path as written in the .blend: `//`-relative when it can be (D1).
    first_frame: int; last_listed_frame: int
    solid_keys: tuple[SolidKey, ...] = ()          # solid_motion_json only: frame, origin, orientation, in_scene.
    witnessed: tuple[str, ...] = (); not_witnessed: tuple[str, ...] = ()   # Channel blender_names.
@dataclass(frozen=True)
class LinkPlan:                         # Everything apply_link_plan needs. No file is read to apply it.
    run_id: str; run_fingerprint: str; steps: tuple[LinkStep, ...]
@dataclass(frozen=True)
class OutputRow:                        # Frozen copy of one FluidityOutputRow; read_output_row returns it.
    output_id: str; object_name: str; product: str; format: str; material_id: int; run_id: str; has_object: bool
    file_pattern: str; first_frame: int; last_listed_frame: int; run_fingerprint: str; output_fingerprint: str
    contract_version: int
LINK_ROUTE_KEYS: frozenset[tuple[str, str]]       # (geometry kind, format) pairs with a route; = keys of LINK_ROUTES.
def build_link_plan(facts: RunFolderFacts, capabilities: Capabilities, rows: tuple[OutputRow, ...], run_folder: Path,
                    project_folder: Path | None, read_json) -> LinkPlan | AddonFailure: ...   # Pure but read_json.

# output_links.py (bpy, main thread)
@dataclass(frozen=True)
class LinkRoute:                        # How one (geometry kind, format) becomes Blender data (5.7).
    geometry_kind: str; format: str
    create: Callable; repoint: Callable; undo: Callable
LINK_ROUTES: Mapping[tuple[str, str], LinkRoute]  # Only formats that are candidates in the format table (7.4).
def apply_link_plan(domain, plan: LinkPlan) -> None | AddonFailure: ...   # All steps, or nothing changes.
def read_output_row(output_object) -> OutputRow | None: ...               # The one accessor (5.7).
```

`build_link_plan` takes every name, pattern, format and channel from the `report` facts; nothing is typed in
that the `file_list` provides (`changes-to-apply.md` 14). It runs on the supervisor thread; `read_json` is
called only for `solid_motion_json` files (one per listed frame).

### 3.9 Property groups (`properties.py`)

Every field marked * takes its label, unit and tooltip from `SCENE_FIELDS` and has the update callback
`mark_scene_changed`. Fields of a substance also set its `preset` to `CUSTOM` unless a preset is being applied.
Fields marked (k) may be keyed; the scene reader samples their F-curves (5.1 step 6).

| Group, stored on | Fields (type, unit) |
|---|---|
| `FluidityIngredient` (D word), item of `substance.ingredients` | `ingredient_id` int; `name` str (the export name: `share_<name>`; lower case, `[a-z0-9_]`, unique in the domain); `share`* float 0..1 (rows sum to 1, owner Q31 a); `rest_density`* kg/m3; `viscosity`* Pa s; `surface_tension`* N/m; `friction`* no unit; `pressure_law`* enum `CANNOT_BE_SQUEEZED`, `STEADY_STIFFNESS`, `STIFFENS_UNDER_PRESSURE`; `rest_sound_speed`* m/s |
| `FluiditySubstance` (N; the native `Material`), item of `domain.substances` | `substance_id` int; `name`; `kind`* enum `LIQUID`, `GAS`, `SNOW`, `SPRINGY_SOLID` (A9: no sand); `preset` enum; `ingredients`; `stiffness_against_shear`* Pa (D); `stiffness_against_squeeze`* Pa (D); `max_springy_shortening`*; `max_springy_lengthening`*; `hardening`*; `own_motion_kept`* 0..1 (D); `own_motion_in_travel`* 0..1 (D); `regions_form_one_body`* bool (snow, springy solid: every start region of it writes `same_material_as` the lowest one); `writes_points` bool |
| `FluidityPair`, item of `domain.pairs` | `substance_a`, `substance_b` int; `surface_tension`* N/m |
| `FluidityParticipant`, `Object.fluidity` | `participant_id` int; `domain` pointer; `role`, `shape`, `motion`, `driven_by` enums from `ROLE_TABLE`; `substance_id` int; `friction_scale`*; `after_path`* enum `STAYS_AT_LAST_KEY`, `LEAVES_SCENE`; `wave_stroke`* m; `wave_period`* s; `wave_ramp_duration`* s; `mass`* kg; `moves_along_x`, `_y`, `_z`, `turns`* bool; `start_linear_velocity`*, `start_angular_velocity`* m/s, rad/s; `steady_force`* N; `start_velocity`* m/s; `volume_rate`* m3/s; `source_velocity`* m/s; `source_first_frame`, `source_last_frame`* int; `fuel_rate`* (k) kg/(m3 s); `heat_rate`* (k) W/m3 |
| `FluidityOutputSettings`, `domain.outputs` | `surface`, `foam_and_spray`, `foam_volume`, `gas_volume`, `solid_motion` bool (points: `substance.writes_points`); `surface_spacing`*, `surface_smoothing_width`* m (0: native default); `volume_spacing`* m (0: smallest cell width); `motion_samples_per_frame`* int 1..64; `foam_particles_per_volume_time`* 1/(m3 s); `foam_count_per_particle`*; `foam_lifetime`* s; `foam_display_radius`* m; `point_fields` enum flag (`MASS`, `SPRINGY_DEFORMATION`, `PERMANENT_VOLUME_CHANGE`, `FITTED_VELOCITY_CHANGE`); `keeps_frame_inputs` bool (default on: needed by Rebuild Outputs and Read Particle); `saved_state_every_frames` int; `saved_states_kept` int |
| `FluidityRunRecord`, item of `domain.runs` | `run_id` str; `label` ("Run 003"); `kind` `PREVIEW`, `BAKE`; `folder` path; `run_fingerprint` str (D); `first_frame`, `last_frame`; `last_listed_frame`, `saved_state_frame` int (the last known, a hint until `report` answers); `end_event` str; `exit_code` int; `started_from_run_id` str |
| `FluidityOutputRow`, item of `domain.output_estimate` | `output_id` str; `object_name` str (name inside the file); `product` str; `format` str; `material_id` int; `run_id`; `object` pointer; `file_pattern` path; `first_frame`, `last_listed_frame`; `run_fingerprint`, `output_fingerprint` str; `contract_version` int |
| `FluidityDomain`, `Object.fluidity_domain` | `domain_id` str; `scene_preset` enum; `participants` collection of object pointers, `active_participant`; `substances`; `pairs`; `closed_left`, `closed_right`, `closed_front`, `closed_back`, `closed_bottom`, `closed_top`* bool (D side names); `smallest_cell_width`* m; `size_level_count`* int (D `size_level`); cell resize: `uses_cell_resize`* bool, `surface_reach_in_cell_widths`*, `solid_reach_in_cell_widths`*, `particle_region_depth_in_cell_widths`*, `grid_only_depth_in_cell_widths`*; open sea: `uses_open_sea`* bool, `open_sea_substance_id`* int, `open_sea_side`, `outlet_side`* enums, `wave_height`*, `water_depth`*, `still_water_height`* m, `open_sea_wave_period`* s, `open_sea_zone_width`*, `outlet_zone_width`* m, `open_sea_pull_rate`*, `outlet_pull_rate`* 1/s; gas: `ambient_temperature`*, `ignition_temperature`* K, `burn_rate`* 1/s, `heat_per_fuel_mass`* J/kg, `smoke_per_fuel_mass`*; `force_field_collection`* (a Blender collection, empty by default: every force field object in the scene; the `gas_wind[].acceleration` row of `SCENE_FIELDS`, whose tooltip states 1.0 m/s2 per unit of Strength; D13 replaces the keyable `wind_acceleration`); solver: `max_step_duration`* s, `max_retries_per_step`*, `max_volume_error_rate`* 1/s, `random_seed`*; run: `run_on` enum `CPU`, `GPU` (D `RunOn`), `memory_share_of_total`, `memory_share_of_free`; `cache_root` path; `outputs`; `runs`; `shown_run_id`; `output_estimate` |
| `FluidityPreferences` (`AddonPreferences`) | `native_executable_path` file path (empty: the packaged one); `default_run_on`; `preview_frame_count` int; `link_outputs_when_ready` bool; `stop_offer_seconds` float (default 10) |

No live status is stored in a property group. Gravity is the scene's (`scene.gravity`, or zero when
`scene.use_gravity` is off). The frame range and frame rate are the scene's. A domain may hold liquid and gas
substances together (A10); at most one substance is `GAS`.

### 3.10 Operators

| `bl_idname` | Label | Does |
|---|---|---|
| `fluidity.add_reference_scene` | Add Reference Scene | Creates a domain, participants and substances from one of the four presets. Undoable. |
| `fluidity.add_domain` | Add Domain | A box around the selection, or a default box. |
| `fluidity.apply_role` | Set Role | Gives the selected objects a role (enum from `ROLE_TABLE`) in the active domain; the only writer of `participant.domain` and `domain.participants`. |
| `fluidity.remove_participant`, `fluidity.select_participant` | Remove from Simulation, Select | |
| `fluidity.add_wave_paddle` | Add Wave Paddle | A `SOLID`, `FLAT_SOLID`, `DRIVEN`, `WAVE_STROKE` object at a chosen domain side, with stroke and period. |
| `fluidity.add_substance`, `fluidity.remove_substance`, `fluidity.add_ingredient`, `fluidity.remove_ingredient` | Add Substance, ... | Ids are the next unused; never reused inside a domain. |
| `fluidity.apply_substance_preset` | Apply Preset | Shows the fields that will change with old and new value, then writes them. |
| `fluidity.check_scene` | Check Scene | Quick check at once, then the full check in the background. |
| `fluidity.cancel_check` | Cancel Check | |
| `fluidity.apply_repair` | (the repair's label) | Runs the operator named by one finding's `Repair`. |
| `fluidity.start_run` | Preview, Bake | `kind` `PREVIEW` or `BAKE`. Creates a run record and a new folder; never writes into an existing run. |
| `fluidity.pause_run` | Pause | Writes `stop_after_frame`: the frame in progress is finished and listed with a saved state. |
| `fluidity.stop_run_now` | Stop Now | Writes `cancel`: the frame in progress is dropped. Offered from `stopping` after `stop_offer_seconds`; a second press ends the process group. Listed frames stay. |
| `fluidity.resume_run` | Resume | From `stopped` or `failed` when `check --cache` says the simulation is `same` and `report` names a saved state. |
| `fluidity.rebuild_outputs` | Rebuild Outputs | NEW. From `done` or `stopped` when `check --cache` says the simulation is `same` and some outputs are `changed` without `needs_simulation`: `run --outputs` with those ids. |
| `fluidity.new_run` | New Run | From `needs_new_run`, `done`, `stopped` or `failed`: keeps the old run, starts at `not_checked`. Option `start_from_frame`: the new run starts from a saved state of the old one (`--start-from`, `--start-frame`). |
| `fluidity.link_outputs` | Show Result | Builds and applies the link plan of one run. |
| `fluidity.relink_run` | Find Moved Cache... | Folder picker; relinks all outputs of one run or none. |
| `fluidity.show_run` | (run label) | Chooses which run's outputs the registry rows point to. |
| `fluidity.select_output` | Select | |
| `fluidity.delete_run` | Delete Run Files... | Lists the folder, its size and the outputs that will stop playing; refused while `report` says the lock is held; confirm. |
| `fluidity.open_log`, `fluidity.copy_support_report` | Open Log, Copy Support Report | Local only. The report includes `run_report.json`, the capabilities and the last 200 events. |
| `fluidity.read_particle` | Read Particle | `read-particle` by id at the current frame. |
| `fluidity.check_outputs` | Check Outputs | Evaluates one frame of each linked output and compares counts with the `frame_done` file entries. |

### 3.11 Panels (`bl_space_type = 'PROPERTIES'`, `bl_context = 'physics'`)

Shown when the active object is a domain or a participant (a participant shows its own role first and a
button that selects its domain).

| Class | Label | Layer | Content |
|---|---|---|---|
| `FLUIDITY_PT_everyday` | FluidiTy | everyday | Status line; one primary action chosen from the state; findings; domain size; smallest cell width with the cell and particle counts and the run mode once checked; frames and seconds; output switches; preset. |
| `FLUIDITY_PT_participants` | Participants | everyday | `FLUIDITY_UL_participants`: name, role, shape, motion, finding mark. Below: role, shape, motion and substance of the active row. |
| `FLUIDITY_PT_outputs` | Outputs | everyday | `FLUIDITY_UL_outputs`: product, format, state (ready, stale and rebuildable, stale and needs a new run, missing files, not built in n frames, not requested), frames, Select. Per row: channels, those not witnessed marked "written, not witnessed". Run menu. Rebuild Outputs. Find Moved Cache. |
| `FLUIDITY_PT_physical_detail` | Physical Detail | physical | Substances and ingredients, pairs, friction, source rates, free solid mass, open sea, gas law and burn, wind. Closed by default. |
| `FLUIDITY_PT_solver_detail` | Solver Detail | solver | Size levels, cell resize, step duration, retries, pressure limit, own motion shares, pressure law. Closed. |
| `FLUIDITY_PT_performance` | Performance | performance | Run on; memory estimate against allowed bytes by module; one piece or batched with the batch plan; NEW (A17): under the heading "Planned cost (estimate)" the last full check's `plan.budget.lines` verbatim, one label per line (nothing when `budget` is None), among them the `liquids:` lines (NEW, A25); saved state interval and expected work on resume; time by category; memory used. Closed. |
| `FLUIDITY_PT_inspect` | Inspect | inspect | The three facts with frames; last try and its reject reason and stage; activity; ages of last event and last advance; lock holder; Read Particle; Check Outputs; Open Log; Copy Support Report. Closed. |

### 3.12 Witness tools (`tools/blender/witness/`, `native/fluidity/cli/write_witness_files.cc`)

Not packaged. They produce the files of the fixed witness scene with every candidate writer, open them in
Blender 5.3 with the add-on disabled, and print `WitnessRecord` lines (7.2).

```text
tools/blender/witness/
  witness_scene.json        the fixed scene file (7.2), checked into the repository
  run_witness.py            runs inside Blender: every case of one Blender build, one result line per case
  witness_cases.py          WitnessCase, CASES_BY_KIND, the opening routes; imports bpy only inside functions
  measure.py                the measurements (pure Python where possible)
  write_witness_records.py  plain Python: turns the Blender 5.3 results file into WitnessRecord lines
  results/                  results_<5.3.x>.jsonl, one per witness run, committed with the table change
```

```python
class WitnessProperty(enum.Enum):       # render_output::WitnessProperty, lower snake case without the k.
    FILE_SEQUENCE_READ_NATIVELY = "file_sequence_read_natively"
    TOPOLOGY_MAY_CHANGE_PER_FRAME = "topology_may_change_per_frame"
    VELOCITY_GIVES_MOTION_BLUR = "velocity_gives_motion_blur"
    VELOCITY_IN_SECONDS = "velocity_in_seconds"
    POINT_RADIUS_ARRIVES = "point_radius_arrives"
    POINT_ATTRIBUTES_ARRIVE = "point_attributes_arrive"
    FACE_ATTRIBUTES_ARRIVE = "face_attributes_arrive"
    Z_UP_WITHOUT_USER_ROTATION = "z_up_without_user_rotation"
    TRANSFORM_GIVES_OBJECT_MOTION = "transform_gives_object_motion"
    GRIDS_ARRIVE_BY_NAME = "grids_arrive_by_name"
class WitnessResult(enum.Enum):         # render_output::WitnessResult.
    NOT_RUN = "not_run"; DOCUMENTED = "documented"; PASSED = "passed"; FAILED = "failed"
@dataclass(frozen=True)
class WitnessCase:
    kind: str                           # "points", "surface_mesh", "volume", "solid_motion"
    format: str                         # "usd", "alembic", "open_vdb", "solid_motion_json"
    property: WitnessProperty
    test_name: str                      # "witness_point_radius_arrives[points,usd]"
@dataclass(frozen=True)
class WitnessCaseResult:                   # One line of results_<blender version>.jsonl.
    case: WitnessCase
    blender_version: str; blender_build_hash: str
    result: WitnessResult               # PASSED, FAILED, or NOT_RUN when the test itself could not measure.
    measured: float; limit: float       # The decisive number and its bound, in the unit `detail` names.
    detail: str                         # One line: what differed first, or why nothing was measured.
    files_digest: str                   # BLAKE3 of expected.json, naming the witness files used.
def list_witness_cases(format_table: tuple[FormatChoiceFact, ...]) -> tuple[WitnessCase, ...]: ...
    # Every candidate of every kind times every property its kind requires (render output 5.7 table).
def run_witness_case(case: WitnessCase, witness_folder: Path) -> WitnessCaseResult: ...   # In Blender.
def write_witness_records(results_5_3: Iterable[WitnessCaseResult],
                          format_table) -> list[str]: ...   # C++ lines for format_table.cc, in table order (D12).
```

`write_witness_records.py <results_5.3.x.jsonl> --witness-folder <folder> --capabilities <json>` prints those
lines and the `chosen` value per kind; it refuses a results file whose `files_digest` differs from the folder's
`files_digest.txt`, or whose recorded Blender version is not 5.3.x.

`fluidity_write_witness_files <run_folder> <witness_folder>` (built from `native/fluidity/cli/write_witness_files.cc`
with `fluidity_cli_parts` and `fluidity_interchange`; beside the other CLI sources, never installed or packaged): reads one run
folder of `witness_scene.json` made by `fluidity run` with `keeps_frame_inputs`, builds each frame's
`RenderFrame` from the stored frame inputs, calls every candidate writer of every kind of
`ReadFormatTable()` (`WritePointsFile`, `WriteSurfaceFile`, `WriteVolumeFile`, `WriteSolidMotionFile` with each
`InterchangeFormat` candidate), and writes `<folder>/<kind>/<format>/<file name of render output 5.7>` plus
`expected.json`: per kind, format, object and frame, the values the files were made from (positions,
velocities, radii, ids, every channel by `blender_name`, triangle lists, grid names, lattice origin and
spacing, sample points with their values, solid poses), as doubles, and the frame rate.

### 3.13 `logic/force_fields.py` (NEW, D13, A20)

```python
WIND_ACCELERATION_PER_STRENGTH: float = 1.0       # m/s2 per unit of Blender's Wind Strength: the gas branch's factor
                                                  #   (gas_authoring.py:63-67 on origin/codex/m7-low-mach-gas).
class FieldHandling(enum.Enum): ...     # HONOURED, NOT_HONOURED, IGNORED
FORCE_FIELD_TYPE_TABLE: Mapping[str, FieldHandling]   # One row per Blender 5.3 FieldSettings.type except NONE:
    # WIND HONOURED; FORCE, VORTEX, MAGNET, HARMONIC, CHARGE, LENNARDJ, TEXTURE, TURBULENCE, DRAG, FLUID_FLOW
    # NOT_HONOURED (a Blender fluid would feel it, FluidiTy does not); GUIDE, BOID IGNORED (they act only on
    # Blender particle systems, hair or boids). Each row checked against Blender 5.3 and its manual (5.13).
@dataclass(frozen=True)
class FieldKey:
    time: float                         # s from the first frame.
    strength: float                     # Blender's Strength, no unit.
    direction: Vec3                     # Unit vector, world: the field object's local +Z (from mathutils).
@dataclass(frozen=True)
class ForceFieldReading:                # Blender's identifiers and values, unchanged.
    object_name: str; field_type: str; keys: tuple[FieldKey, ...]
    shape: str; z_direction: str; falloff_power: float; use_min_distance: bool; use_max_distance: bool
    noise: float; flow: float
# Empty without gas or without a Wind field. Otherwise one key per time in any Wind field's keys (sorted, times
# within 1e-12 s merged), each the sum over Wind fields of WIND_ACCELERATION_PER_STRENGTH * strength * direction;
# a field with no key at that time is interpolated linearly between its keys, held before the first and after the last.
def build_wind_keys(readings: Sequence[ForceFieldReading], has_gas: bool) -> tuple[WindKey, ...]: ...
# The findings of 5.13, in the readings' order: module "addon", found by "scene_reader", object_name set, no repair.
def check_force_fields(readings: Sequence[ForceFieldReading], has_gas: bool) -> tuple[Finding, ...]: ...
```

The scene reader (`scene_reader.read_scene(domain) -> SceneReading` with `description` and `findings`) builds one
`ForceFieldReading` per force field object, then sets `description.wind_keys = build_wind_keys(...)` and adds
`check_force_fields(...)` to its findings.

## 4. Consumes from other modules

### 4.1 I/O and CLI: PROVIDED

**Executable `fluidity`** (I/O 3.7), one process, no child process, UTF-8 JSON events, one per line. Commands
used, as I/O declares them:

| Command | Used for |
|---|---|
| `capabilities [--run-on gpu]` | At registration and when the executable changes: `protocol`, schema versions, coordinate text `right_handed_z_up_metres`, `run_on_values`, every `CapabilityReason` with its level (`ToName(step::CapabilityLevel)`) and message, static limits (`kMaxPlateCount`, `kMaxFreeSolidCount`, `kMaxIngredientCount`, `kMaxSizeLevels`, motion samples 1..64), `ReadFormatTable()` with every `WitnessRecord`, `ReadOutputChannels()`, `ReadInterchangeLibraries()`. Within 200 ms, no GPU touched; `--run-on gpu` only from the Performance panel. |
| `check <scene_file> [--cache <run_folder>]` | The full check: `scene_checked` with `can_run`, every finding, `filled_defaults`, `run_on_requested`, `run_on_usable`, the plan (`run_mode`, estimate by module, `allowed_bytes`, counts, batch plan), NEW (A17, planned cost) `budget` (table `RunBudget` of `schemas/simulate/v2/run_budget.fbs`, read as `RunBudgetFacts`), `run_fingerprint`, the output declarations; with `--cache`: `CacheComparison` (`simulation`, `differing_part`, `changed_fields`, `outputs[] {output_id, match, changed_fields, needs_simulation}`). Exit 0 (can run) or 3 (cannot; also when the budget's `verdict` is not `fits`, with the blocking finding `does_not_fit`), both with `scene_checked`. |
| `run <scene_file> --into <run_folder> [--last-frame n]` | Preview and Bake. The folder holds only the add-on's copy of the scene file (left alone, never listed). |
| `run <scene_file> --into <run_folder> --start-from <other_run_folder> [--start-frame n]` | New Run from a saved state of an old run (`BuildRunFromSavedState`). |
| `run <scene_file> --into <run_folder> --outputs <ids>` | Rebuild Outputs (`RebuildOutputs`): refused with `start_new_run` for an output that `needs_simulation`. |
| `resume <run_folder> [--last-frame n]` | Resume. Never reads the scene file again. |
| `report <run_folder> [--check-files]` | `RunFolderFacts`: `run_fingerprint`, `last_listed_frame`, `saved_state_frame`, `last_try`, `lock_holder` with `is_held`, the output declarations, frames done, not built entries, `ignored_end_bytes`, the `RunReport`, `is_crashed`. Read-only, takes no lock. |
| `read-particle <run_folder> --frame n --particle-id id` | `ParticleFacts`, ids as decimal text. |

**Events consumed** (`events.jsonl` of the run folder; standard output of one-shot commands). Every event has
`protocol` (2), `event`, `sequence`, `sent_at` (s UTC, set when made), `elapsed_seconds`.

| `event` | Fields read | Effect (5.4) |
|---|---|---|
| `started` | `run_fingerprint`, `run_on`, processor, first and last frame, `continues_from_frame`, memory estimate, `allowed_bytes`, `run_mode` (`"one_piece"`, `"batched"`, `"batched_with_solves_in_pieces"`; reconcile), batch plan, output declarations, findings with `has_known_limit` | `preparing` to `running`; outputs known |
| `preparing` | `done`, `total`, `what` | progress |
| `heartbeat` | `RunProgress` fields (`activity`, `frame_in_progress`, `last_frame_done`, `last_saved_state_frame`, `last_try`, `waiting_frames`, `preparing_done`, `preparing_total`), `last_advance_sent_at` | ages, activity |
| `try_done` | as I/O lists; only with `--report-every-try`, which the add-on passes only from Inspect | `last_try` |
| `finding` | one `Finding` | appended |
| `frame_done` | `frame`, `time`, files, `not_built[]`, counts, `MemoryReport` numbers, elapsed by category | `last_listed_frame` |
| `frame_checked` | `frame` | `checking_frame` |
| `saved_state_listed` | `frame`, `byte_count` | `saved_state_frame` |
| `stopping` | `request` (`cancel`, `stop_after_frame`, `end_of_input`) | `stopping` |
| `done`, `stopped`, `failed` | `exit_code`, `last_frame_done`, `last_saved_state_frame`, totals, `MemoryReport`; `failed` one `Finding` with `stage`, `frame`, `time` | terminal |

`frame_done`, `finding`, `saved_state_listed`, `failed`, `done` and `stopped` are never skipped (I/O 4).

**Finding table** (I/O 3.7): `module`, `code`, `severity` (`blocks_run`, `has_known_limit`, `note`), `message`,
`material_id`, `solid_id`, `source_id`, `particle_id`, `object_name`, `scene_file_path`, `magnitude`, `limit`,
`unit`, `suggested_changes`. Mapped to `Finding` (3.4) one to one; `solid_id`, `source_id` and region ids
become `participant_id` through the ids the add-on wrote.

**Exit codes** (`io::ExitCode`): 0 done; 1 failed; 2 bad command line (an add-on bug: `NATIVE_CALL_FAILED`);
3 scene cannot run; 4 stopped on request (`cancel`, `stop_after_frame`, end of standard input); 5 run folder
refused (lock held: the event names the holder; not empty; no run; another fingerprint; no saved state);
6 processor failed (continue from the last saved state).

**Standard input**: `cancel`, `stop_after_frame`; end of standard input is `stop_after_frame`.

**Run folder** (I/O 5.3): `writer.lock`, `file_list`, `run_report.json`, `events.jsonl`, `log.txt`,
`ready_scene`, `saved_states/`, `frame_inputs/`, `outputs/<output_id>/...`, `tmp/`. The add-on reads only
`events.jsonl`, `log.txt` and `run_report.json` directly, and product files through Blender's readers. It reads
the `file_list` only through `report`. It never creates, reads or deletes `writer.lock`.

**Scene file this module writes** (`scene_file.fbs` version 2, JSON, I/O decision 6; SI, Z-up, world frame;
times in seconds from the first frame; orientations (w, x, y, z), own axes to world). Every id is chosen by the
add-on, nonzero, unique per kind, stable across edits, and comes back unchanged in findings. The outline below
is the shape; the field names are those of the branch's `scene_file.fbs` (plan decision: for example the wind is
the top-level `gas_wind`), and a test fails on any field path the schema does not define.

```text
frames:          frames_per_second, first_frame, last_frame
domain:          low_corner, high_corner, closed {left, right, front, back, bottom, top}, gravity
grid:            smallest_cell_width, size_level_count
materials[]:     material_id, name, kind (liquid | gas | snow | springy_solid), own_motion_kept,
                 own_motion_in_travel,
                 response {stiffness_against_shear, stiffness_against_squeeze, max_springy_shortening,
                           max_springy_lengthening, hardening},
                 ingredients[] {ingredient_id, name, share, rest_density, viscosity, surface_tension, friction,
                                pressure_law {kind, rest_sound_speed}}
material_pairs[]: material_id_a, material_id_b, surface_tension
start_regions[]: region_id, name, material_id, shape (box | ball), origin, orientation, half_size, velocity,
                 same_material_as
sources[]:       source_id, name, material_id, origin, orientation, half_size, volume_rate, velocity,
                 start_time, end_time
solids[]:        solid_id, name, shape (box | flat_solid | plate), origin, orientation, half_size, friction_scale,
                 motion_kind (fixed | driven | free),
                 path {keys[] {time, origin, orientation}, after_path}
                 | wave_stroke {stroke, wave_period, ramp_duration, travel_direction}
                 | free_solid {mass, moves_along_x, moves_along_y, moves_along_z, turns, start_linear_velocity,
                               start_angular_velocity, steady_force}
open_sea:        material_id, wave {still_water_height, water_depth, wave_height, wave_period, travel_direction},
                 open_sea_side, outlet_side, open_sea_zone_width, outlet_zone_width, open_sea_pull_rate,
                 outlet_pull_rate
gas:             law {ambient_temperature}, burn {ignition_temperature, burn_rate, heat_per_fuel_mass,
                 smoke_per_fuel_mass},
                 sources[] {source_id, name, low_corner, high_corner, keys[] {time, fuel_rate, heat_rate}}
gas_wind[]:      time (s from the first frame), acceleration (m/s2, world, Z-up): from the Wind force fields
                 (5.13); one key per frame when a strength or a turn is keyed, one key at time 0 otherwise, none
                 in a domain without gas. Neither I/O nor the gas module converts anything.
cell_resize:     cell_size {surface_reach_in_cell_widths, solid_reach_in_cell_widths},
                 regions {particle_region_depth_in_cell_widths, grid_only_depth_in_cell_widths}
batches:         {} (BatchSettings defaults)
solver:          max_step_duration, max_retries_per_step, max_volume_error_rate, random_seed
run:             run_on, memory_share_of_total, memory_share_of_free
outputs:         points {material_ids[], particle_fields[]}, solid_motion,
                 surface {groups[] {material_ids[]}, spacing, smoothing_width},
                 foam {particles_per_volume_time, count_per_particle, foam_lifetime, display_radius},
                 foam_volume {spacing}, gas_volume {spacing}, motion {samples_per_frame},
                 keeps_frame_inputs, saved_state_every_frames, saved_states_kept
```

The gas substance is one `materials[]` entry of kind `gas` with one ingredient (materials 5.1 gas rules; I/O
sets its rest density from the law). `open_sea`, `gas`, `cell_resize` are omitted when not used; any other
field the add-on leaves out takes the native default and is reported in `filled_defaults`. One surface group
holds every liquid substance (all liquids are one material set, step decision); `points.material_ids` never
names the gas. What I/O derives and the add-on therefore never writes (I/O 5.1 step 4): particle rate, mass and
rest volume of a source from `volume_rate`; the start particles of each region (8 per smallest cell);
`angular_velocity_until_next` of driven paths from the orientations; box inertia from mass and shape; wave
paddle keys from `wave_stroke`; blocking materials, plates and `has_liquid_or_gas`; material sets.

**Fingerprint and comparison** (I/O 5.2): the run fingerprint covers the physical part, the executable, `RunOn`,
the processor and the run mode; names, requested outputs, render settings, `last_frame`, memory shares and
saved-state cadence are outside it. `check --cache` is the only test of whether a run may continue or an output
may be rebuilt.

### 4.2 Surface, foam and spray: PROVIDED

**Products and files** (render output 5.7, under `outputs/` of the run folder; one file per frame and product,
listed by I/O, never rewritten; the extension is the chosen candidate's):

| Product | `output_id` | File | Objects in the file |
|---|---|---|---|
| Simulation particles, one per non-gas material asked | `points_<material_id>` | `points.<frame:06d>` | `points` |
| Surface, one per group | `surface_<lowest material_id>` | `surface.<frame:06d>` | `surface` |
| Droplets, one per group | `droplets_<lowest material_id>` | `droplets.<frame:06d>` | `droplets` |
| Foam particles | `foam_and_spray` | `foam_and_spray.<frame:06d>` | `spray`, `foam`, `bubbles` (each may be empty) |
| Foam volume, gas volume | `foam_volume`, `gas_volume` | `<id>.<frame:06d>.vdb` | one grid per channel and `known` |
| Solid motion | `solid_motion` | `solid_motion.<frame:06d>` | `solid_<solid_id>` per solid |

**Format table today** (render output 5.7): points Alembic, surface Alembic, volumes OpenVDB, solid motion
JSON, each with a USD candidate "to be proven by the Blender add-on"; solid motion also has an Alembic `OXform`
candidate. The properties each kind's candidates must show are the `WitnessProperty` values of that table; the
initial records are `kNotRun` or `kDocumented`, and `kDocumented` never chooses. `ChooseFormat` takes the first
candidate whose every record is `kPassed` (Blender 5.3, the one result a record holds), otherwise the last
candidate.

**Channel table** (`ReadOutputChannels()`, printed by `capabilities`, copied per declaration into the
`file_list`): the add-on uses `blender_name` and, per format, the name in the file; a channel is advertised as
witnessed only when its `witnessed_by` property passed for the chosen format in Blender 5.3. Gas grids:
`density`, `smoke`, `temperature`, `fuel`, `flame`, `known`; foam grids: `foam_count_density`,
`bubble_count_density`, `bubble_volume_fraction`, `known`; shares are `share_<ingredient_export_name>`; `id` is
int32 (low 31 bits); `material_index` and `triangle_kind` are int32 per face.

**File metadata**: every file stores `FrameIdentity` (frame, `state_version`, time, `frames_per_second`),
`kOutputContractVersion` (2) and the channel names (USD `customLayerData`, Alembic archive metadata, OpenVDB
file metadata, JSON fields). The solid motion JSON is `{"contract_version", "frame", "time", "solids":
[{"solid_id", "origin", "orientation", "linear_velocity", "angular_velocity", "in_scene"}]}`.

**What render output asks of this module** (its 4): native readers for USD, Alembic and OpenVDB; the witness
tests of its 5.7, run in Blender 5.3 with the add-on disabled on files its writers made, each result
returned as a `WitnessRecord` line; one accessor for a product's `FrameIdentity` and `run_fingerprint`
(`read_output_row`, from the `file_list` through `report`); channel names never typed in; "Velocity Unit:
Second" on an Alembic cache reader; the flame and temperature scales of the volume material. All are in 5.7,
5.8 and 7.

### 4.3 Step: PROVIDED, reached only through the CLI

Through I/O's events and its `io/stable_names.h`: `CapabilityReason` names (`kCapabilityReasonCount` 16 at most,
lower by one for each viscosity exclusion the viscosity plan lifts, 13 if all three are lifted; A24 removed
`viscous_liquid_with_snow`, A27 `viscous_liquid_with_free_solid`), `CapabilityLevel` (`kCanRun`, `kExperimental`,
`kUnavailable`; it replaces `Support`: `kUnavailable` blocks, `kExperimental` does not), `SuggestedChange` names
(`kSuggestedChangeCount` 12 values with `kNone`; A27 removed `set_viscosity_to_zero`, which the sand module adds
back with `kViscousLiquidWithSand`), `CapabilityResult` with every finding
and no cap, `RunPlan` (`mode`, `estimate`, `allowed_bytes`, `counts`, batch fields), `MemoryEstimate::by_module`
(14 modules including `render_output`, `cell_resize`, `batches`), `MemoryReport`, `Stage` (22 values, stage 11
`solve_viscosity_and_pressure`), `RejectReason`, `StepLimit`, `RunMode`; `StepSettings` fields `run_on`,
`max_step_duration`, `max_retries_per_step`, `memory_share_of_total`, `memory_share_of_free`.

### 4.4 Materials, solids, gas, pressure, contact, cell resize, batches: PROVIDED

Field names and units only, mirrored by `SCENE_FIELDS` and the property groups, as each design's section 3 has
them: `materials::Ingredient`, `MaterialResponse`, `MaterialKind` (`kLiquid`, `kGas`, `kSnow`, `kSpringySolid`),
`OwnMotionSettings`, `MaterialPair`, `MaterialTableInputs::ingredient_export_names`; `solids::SolidSettings`,
`SolidMotionKind`, `PathKey`, `AfterPath`, `FreeSolidSettings`, `FreeMotionLimits`, `WavePaddleSettings`,
`Orientation`, `kMaxPlateCount`, `kMaxFreeSolidCount`, `CheckSolidsCanRun` entries (each with `solid_id` and a
`ToName` code, all at once); `contact::SolidShape`, `ClosedSides`, `DomainBox`, `SolidPose`;
`pressure::PressureLaw`, `PressureLawKind`, `OpenSeaSettings`, `OpenSeaWave`, `SolveSettings::max_volume_error_rate`;
`gas::GasLaw`, `BurnSettings`, `GasSource` (world-aligned box, keyed), `GasSourceKey` (`time`, `fuel_rate`,
`heat_rate`), `GasWind`, `GasWindKey` (`time`, `acceleration`; written by the add-on from Blender's Wind fields
and read as written); `cell_resize::CellResizeSettings`,
`CellSizeSettings`, `RegionSettings`; `batches::BatchSettings` (defaults only).

### 4.5 What earlier drafts asked of other designs (all applied)

- Step `ReadyScene` carries keyed `gas::GasSource` and `gas::GasWind` (step 3.2). Applied.
- Step `CapabilityResult` reports every finding with no cap (step 3.4). Applied.
- Solids `CheckSolidsCanRun` returns every broken scene rule with its `solid_id`; `kMaxPlateCount` and
  `kMaxFreeSolidCount` are public; one signature (solids 3.4). Applied.
- I/O commands `check --cache`, `report`, `read-particle`, `run --outputs`, `sent_at`, heartbeat with
  `last_advance_sent_at`, findings with `code` from `ToName` (I/O 3.7). Applied.

## 5. Method

### 5.1 The scene model

1. A **domain** is a Blender object whose world-aligned bounding box is the `DomainBox`. It owns the list of
   participants, the substances and all run settings. A scene may hold several domains; each has its own runs.
2. A **participant** is an object with a role, a shape and, for a solid, a motion. The shape is declared, not
   guessed from the mesh: the object's bounding box in its own axes gives `half_size`, its world matrix gives
   `origin` (the box centre) and `orientation`. A sheared or mirrored matrix is a finding. A `FLAT_SOLID` is the
   plane through the origin with normal the object's own +Z; the list row says "unbounded". A `PLATE` takes
   its thickness from its thinnest own axis. A `GAS_SOURCE` must be world-aligned (`gas::GasSource` is a box
   in world axes): a turned one is a finding with the repair "Clear Rotation"; it writes `low_corner` and
   `high_corner`.
3. The wave paddle and the cutting plate are not roles. A wave paddle is a `SOLID`, `FLAT_SOLID`, `DRIVEN`
   participant driven by `WAVE_STROKE` (D: "a variable, not a type"). Cutting is a `SOLID`, `PLATE`, `DRIVEN`
   participant that meets a `START_MATERIAL` of a `SPRINGY_SOLID` substance. Both have a shortcut operator
   that writes these concrete values.
4. A **substance** is what the native side calls a material: a kind, a response and one or more ingredients
   that mix. Two liquid substances are two liquids that do not mix (Q31 a). Appearance is not part of it: a
   Blender material on an output object is the artist's and no edit of it reaches the scene file. Liquid and
   gas may share one domain (A10): smoke over water is an ordinary scene.
5. A **source** is given as volume per second and a velocity. A **gas source** is given as fuel and heat
   rates per cubic metre of its box, each keyable. Wind comes from Blender's own Wind force field objects in
   the domain's `force_field_collection` (or the whole scene when it is empty), turned into `gas_wind` keys as
   5.13 states (A20, D13). The domain has no wind property of its own.
6. Driven motion with `KEYS` reads the object's own location and rotation curves with `FCurve.evaluate` at
   every frame of the range, without changing the current frame (Blender 5.3 actions are layered: the curves
   are those of the action's channel bag for the object's slot, 5.14), and writes one key per frame with its
   origin and orientation; I/O derives the angular velocity between keys with
   `solids::ComputeRotationVector`. An object whose motion comes from a parent, a constraint, a driver or a
   modifier is a finding (`motion_not_keys`) with the repair "Bake to Keys" (`bpy.ops.nla.bake`, 5.14). Keyed
   rates of gas sources, and a Wind field's strength (`field.strength`) and rotation, are read the same way: one
   key per frame where a curve has keys, and the constant value as one key otherwise.

### 5.2 Presets

A preset is a row of concrete values. Applying one writes those fields and nothing else, then sets `preset`.
Any later edit of a preset field sets `preset` to `CUSTOM`; the values stay. Before writing over values that
differ, the operator lists each field with its old and new value. Substance presets, with I/O's
`kMaterialPresets` as the source of the numbers (I/O decision 12): water (1000 kg/m3, 0 Pa s (A25; was 0.001), 0.072 N/m),
syrup (1400 kg/m3, 10 Pa s, 0.08 N/m, owner Q50), snow (300 kg/m3, 1200 Pa, 700 Pa, 0.025, 0.01, 4, as
`authoring.py:157-165`), springy solid (I/O `kSpringySolid`), air (gas, I/O `kAir`; one ingredient, viscosity,
friction and surface tension 0). The four reference scene presets create the scenes the designs fix: breaking
wave (programme design 8a: paddle, open-sea side, mixed-size cells, surface and foam), syrup (viscosity
decision 6: 0.4 x 0.4 x 0.3 m, a 2 cm stream at 0.3 m/s from 0.25 m, 1 litre, 10 s, floor only), fire (gas
decision 8: 1.6 x 1.6 x 3.2 m, about 5 kW, wind from 5 s made as one Wind force field object "Fire Wind" in
a collection set as the domain's `force_field_collection`, strength keyed with constant interpolation and flow
0, so the read keys equal the builder's; ignition 400 K, 10 s), and a snow example (a snow
start region dropped on the floor, as `main`'s snow mound). No sand preset and no sand scene (A9: the sand
module adds both). Each scene's cell width is its design's coarsest listed width, so it runs on the CPU in the
tests.

### 5.3 The check

1. **Quick check** (main thread, target under 100 ms). `scene_reader` builds the `SceneDescription`;
   `check_scene_quickly` returns every finding it can decide alone: a role without a domain, an object in two
   domains, a shape or motion `ROLE_TABLE` does not allow, a missing or wrong-kind substance, more than one gas
   substance, ingredient shares that do not sum to 1, an ingredient name that is empty or collides once
   normalised, a participant outside the domain, a turned gas source, a non-positive size or rate, a source
   window outside the frame range, motion not readable as keys, an unsaved project, a static limit of
   `capabilities` exceeded (plates, free solids, ingredients, size levels, motion samples), a cost warning
   (surface spacing under a quarter of the cell width). The codes are those of 3.4; what only Blender knows
   (unsaved project, role without a domain, object in two domains, sheared or mirrored matrix, motion not from
   keys, the force field findings of 5.13) comes from the scene reader with the description.
2. **Full check** (supervisor thread, cancellable). Writes the scene file to a temporary folder and calls
   `fluidity check`, with `--cache <run_folder>` of the shown run when there is one. Its findings replace
   nothing: the two lists are merged, de-duplicated by (`module`, `code`, `participant_id`) and ordered by
   severity.
3. Every finding is shown at once, each with the object it is about and its repairs. A repair is an explicit
   operator the artist presses. **Nothing changes `run_on`, the cell width or any physical value by itself.**
   The Performance panel and the everyday panel show `run_on_requested`, `run_on_usable`, and "one piece" or
   "batched" with the estimate against the allowed bytes. The Performance panel also draws the native budget's
   lines (`RunBudgetFacts.lines`) verbatim under "Planned cost (estimate)" (A17, planned cost); the add-on
   computes nothing from the budget's numbers. Those lines include, in a scene with liquid, the `liquids:` block
   (NEW, A25): whether the run uses the pressure solve (every liquid has viscosity 0) or the viscous solve, each
   liquid's viscosity, density and viscous effect per step, and a note advising viscosity 0 when a viscous
   liquid's effect is too small to see.
4. The state becomes `ready` when no finding has `BLOCKS_RUN`. A later edit (a property update callback, or a
   `depsgraph_update_post` that names a participant object) sets it back to `not_checked`.
5. With a run folder, the `CacheComparison` decides what an edit made stale: `simulation` `changed` gives
   `needs_new_run` (the changed fields by their panel labels through `find_scene_field`, or the differing part
   "executable", "run on", "processor", "run mode"), with New Run, optionally from a saved state of the old
   run; `simulation` `same` with outputs `changed` gives those output rows "stale" with Rebuild Outputs when
   none `needs_simulation`, and "needs a new run" otherwise. The old run stays viewable.

### 5.4 The run state machine

`apply_event` and `apply_request` are one transition table. Anything not listed leaves the state unchanged
and is logged.

| From | On | To |
|---|---|---|
| `not_checked` | check finished with `can_run` and no blocking quick finding | `ready` |
| `ready` | an edit that changes the scene file | `not_checked` |
| `ready` | start (new run) accepted | `preparing` |
| `stopped`, `failed` | resume accepted (simulation `same`, a saved state listed) | `preparing` |
| `done`, `stopped` | rebuild accepted (5.3 step 5) | `preparing` |
| `needs_new_run`, `done`, `stopped`, `failed` | new run request, optionally from a saved state | `not_checked` (a new run record) |
| `preparing` | `started` | `running` |
| `preparing`, `running` | pause request (writes `stop_after_frame`), stop now (writes `cancel`), or `stopping` | `stopping` |
| `preparing`, `running`, `stopping` | `stopped` | `stopped` |
| `running` | `done` | `done` |
| any active state | `failed`, or the process ended with no terminal event | `failed` (`end_event` `failed` or `process_lost`) |
| `preparing` | the process ended with exit 3 | `not_checked`, the `failed` finding kept |
| any state with a run folder | `check --cache` says the simulation is `changed` | `needs_new_run` |

Invariants: `last_listed_frame` only grows and only on `frame_done` (or a heartbeat's `last_frame_done`,
which equals it); `saved_state_frame` changes only on `saved_state_listed` and the terminal event;
`last_try` changes on `try_done` and `heartbeat` and never moves the other two; `checking_frame` is set by
`frame_checked` and cleared by the first `frame_done`. Pause and Stop Now are requests: the state is `stopping`
until the native program says where it stopped. Exit 4 is `stopped`; 1 and 6 are `failed` (6 with the note
"continue from the last saved state"); 5 is `failed` with I/O's finding naming the lock holder. Stop Now
pressed a second time ends the process group: `failed` with `end_event` `process_lost`, and the next `report`
gives the three facts as the folder has them. After a failure the status line reads, for example, "Stopped at
frame 38. Frames 0 to 37 are ready. Resume would restart from frame 30."

### 5.5 The supervisor

1. One worker thread, started at registration, stopped at unregistration. It never imports `bpy`.
2. Each loop turn (50 ms): take requests from a queue; start, write to or stop processes; for the live run,
   read at most 256 KiB of new `events.jsonl` lines; fold them through `apply_event`; store the new immutable
   snapshot under one lock.
3. Superseded `heartbeat`, `try_done` and `preparing` events are folded without being kept; `frame_done`,
   `saved_state_listed`, `finding`, `stopping`, `done`, `stopped` and `failed` are never skipped. The journal on
   disk keeps everything.
4. A run process is started in its own process group with stdin a pipe and stdout and stderr discarded: the
   events come from `events.jsonl` and the text from `log.txt`, so the native program never waits for Blender
   to read a pipe. One-shot commands have stdout as a pipe read to the end.
5. Pause writes `stop_after_frame`. Stop Now writes `cancel`; if no terminal event arrives within
   `stop_offer_seconds`, the panel offers Stop Now again, which ends the process group. Because the CLI starts
   no child, the group is the whole tree.
6. On file load, on unregister and at exit the supervisor closes stdin of the live run and does not wait: the
   native program reads the end of standard input as `stop_after_frame`, finishes and lists the frame in
   progress with a saved state, and ends with `stopped` (owner's standing instruction, decision 2). Nothing is
   killed on close.
7. One run at a time per Blender session. A second start returns `RUN_ALREADY_ACTIVE`.
8. A run is identified by `run_id`. The supervisor holds no Blender object. The timer finds the domain by
   name and `domain_id` when it has something to write, and drops the write when the domain is gone.
9. After a file load the supervisor runs `report` on every run record; until the answer comes the panel shows
   the saved hint marked "checking the run folder". A run whose `report` says the lock is held and whose
   journal has no terminal event is followed with `follow_run` (journal only, no process): it is shown as
   `stopping` ("finishing frame n from an earlier session") until its `stopped` arrives.
10. Freshness is `now - sent_at` of the event, not the time it was read. Two ages are shown: last event (is
    the process alive) and last advance (is the simulation moving); heartbeats come at least once per second.

### 5.6 Run records and folders

A run is one folder `<cache_root>/<domain name>/run_<nnn>/`: the add-on writes `scene.json` into the new,
empty folder, then starts `run --into` it; everything else in the folder is I/O's. `cache_root` defaults to
`//fluidity_cache`. Preview and Bake differ only in `kind` and frame range (`preview_frame_count`). Starting
never reuses a folder (I/O refuses a folder with a file list); Resume and Rebuild Outputs continue one; nothing
but Delete Run Files removes one. Paths are stored relative to the project when Blender can express them so and
absolute otherwise: a run folder on another drive works and the Outputs panel notes that the project is then
not movable as one folder.

### 5.7 Outputs: the registry and the link transaction

1. The registry is `domain.output_estimate`: one row per object name of each declaration of the shown run, each
   pointing at one Blender object. `read_output_row(object)` is the only way any code learns which run,
   product, format, material or fingerprint an output object belongs to. Objects carry one custom property,
   `fluidity_output_id`.
2. `build_link_plan` (supervisor thread) takes the declarations, `first_frame` and `last_listed_frame` from
   `report`, refuses a declaration whose format has no route in `LINK_ROUTES` (`FORMAT_NOT_LINKABLE`) and
   declarations whose run fingerprints differ among themselves, then compares with the registry: create,
   repoint or leave. For `solid_motion_json` it reads every listed frame's file into `solid_keys`.
3. **Link routes** (Blender data directly, not import operators, except one small operator,
   `bpy.ops.cachefile.open`, since Blender has no `bpy.data.cache_files.load`; it runs on the main thread and
   reads no frame data. Each API fact is re-checked in Blender 5.3 by the probe before its route is written
   (5.14); if a route fails in the witness tests it switches to the importer used there, inside the same
   transaction, with frame, selection and active object restored). Routes exist only for the keys of
   `LINK_ROUTE_KEYS` (the candidates left in the format table):

   | Geometry kind, format | Blender data |
   |---|---|
   | points, Alembic | `bpy.ops.cachefile.open(filepath=first_file)` (the new `CacheFile` found by comparing `bpy.data.cache_files` before and after), `is_sequence = True`, `velocity_name = ".velocities"`, `velocity_unit = 'SECOND'`; a new point cloud object (`bpy.data.pointclouds.new`) with a Mesh Sequence Cache modifier first in its stack, `object_path` from `CacheFile.object_paths` matching the object name |
   | points, USD (if chosen) | the same with the `.usdc` file; no velocity settings |
   | surface, Alembic or USD | a mesh object with a Mesh Sequence Cache modifier, `read_data` including attributes; the cache file as above |
   | volume, OpenVDB | `bpy.data.volumes.new(name)`, then `filepath = first_file`, `is_sequence = True`, `frame_start` and `frame_duration` from the listed frames (Blender has no `bpy.data.volumes.load`) |
   | solid motion, Alembic or USD (if chosen) | a copy of the solid's object named "<name> · FluidiTy motion" with a Transform Cache constraint on the cache file, `object_path` of `solid_<solid_id>` |
   | solid motion, JSON | the same copy with one location and quaternion key per listed frame from `solid_keys`; hidden where `in_scene` is false |

4. `apply_link_plan`: (a) create every new data-block unlinked; (b) for each existing output object, remember
   its cache file or volume file path; (c) repoint all; (d) write all rows and `shown_run_id`. Any exception
   undoes (c), removes what (a) made and leaves the rows untouched. An existing object keeps its name,
   transform, materials, other modifiers, collections and selection.
5. The foam-on-surface asset reads both rows through `read_output_row` and requires equal `run_fingerprint`.
6. Output frame `n` is Blender frame `n`. One file per frame means a running bake is viewable up to
   `last_listed_frame` and nothing is written twice; after a `frame_done` the timer extends `frame_duration`
   of volumes and the key range of JSON motion (cache file sequences follow the files on disk).
7. With the add-on disabled the objects are ordinary: cache files, modifiers, constraints, volumes, keys and
   materials are Blender's own. The registry and the panels are what is lost.
8. Find Moved Cache: `report` on the chosen folder must give each row's `run_fingerprint`; then one link plan
   with the new paths.
9. A channel the declaration lists with `is_witnessed` false is linked like the others and marked "written,
   not witnessed" in the Outputs panel; a frame listed as not built shows its reason (`too_many_vertices`) and
   the object shows the last file Blender's reader holds.

### 5.8 Starter assets

Three small node assets built in code, each with a family id and a version stored on the node group; a new
version is a new group and an edited group is never rewritten: point display (radius from the `radius`
attribute times a scale, material; raw points output kept), fire volume material (`density`, `smoke`,
`temperature`, `flame`; the temperature and flame scales are inputs of the material, since the files carry
physical units, render output 5.7), foam on surface (mask from foam points within a radius).

### 5.9 What the panels say

The everyday status line is built from `RunStatus` only. Time categories are the native ones, shown as they
arrive: simulate, snapshot, build outputs, write, wait. "Show in Blender" is measured by the add-on around
`apply_link_plan`. Estimates are labelled as estimates; with fewer than three listed frames the remaining time
reads "estimating". The remaining time stays the add-on's own (three listed frames, the native program's rule);
the native `time:` line and the planned cost come only through the budget lines and the I/O events. The activity names the wait: "Waiting for output (2 frames)", "Reading saved state",
"Checking again frame 34 of 37" (the frame bar does not move), "Rebuilding outputs, frame 12".

### 5.10 Packaging

- `blender_manifest.toml`: `schema_version = "1.0.0"`, `id = "fluidity"`, `type = "add-on"`,
  `blender_version_min = "5.3.0"`, no maximum, one platform per built package, `[permissions] files` with
  one short reason. No network permission. (Exact wording rules of the permission text: to verify against the
  Blender 5.3 documentation.)
- Supported Blender: 5.3 only (A16). The packager never rewrites the minimum version. A `bpy.app.version`
  below `(5, 3, 0)` or at or above `(5, 4, 0)` gives one note at registration, "FluidiTy is tested on Blender 5.3
  only"; the add-on still loads. Every commit whose tests ran in Blender names the one build that ran them
  (for example `Blender 5.3.0 (02f0899d78e3): 7 passed.`), or says `Blender 5.3: written, not run in Blender
  (<reason>)`; nothing is called passed that did not run, and each run is a row of `docs/blender-runs.md` (D2).
- Platforms: `windows-x64` and `linux-x64` (CPU and GPU), `macos-arm64` (CPU only). One package per platform,
  each holding `bin/<platform>/fluidity[.exe]`, the OpenUSD resource folder render output 5.7 says the CLI ships
  beside itself (while a USD candidate is in the build), and `bin/<platform>/build.json` (`executable_version`,
  SHA-256).
- `find_native_executable`: (1) `native_executable_path` if set; (2) `bin/<platform>/` of the package, where
  `<platform>` is `to_platform_name(platform.system(), platform.machine())`. Never `PATH`, never a name with a fixed
  extension. The packaged folder is copied once to the extension's user folder under its version and made
  executable there (`bpy.utils.extension_path_user`, 5.14), because an installed extension folder may be
  read-only and a zip may lose the executable bit. Then `fluidity capabilities` must report `protocol` 2, scene
  schema 2 and `right_handed_z_up_metres`.
- Offline: the packager runs `blender --offline-mode --factory-startup --command extension build` and
  `extension validate` (`--offline-mode` is a global Blender option, written before `--command`); installation
  in tests is `blender --offline-mode --command extension install-file -r user_default -e <zip>` from disk.

### 5.11 Z-up

The solver is Z-up, right-handed, metres, like Blender. Every conversion listed in section 2 is gone:
positions, velocities, poses, orientations, grid origins and solid motion pass through the add-on unchanged,
and the add-on contains no rotation matrix (the one direction it needs, a Wind field's axis, comes from
Blender's `mathutils`, 5.13). What remains is inside render output's Alembic writer, because an
Alembic archive is Y-up by convention and Blender's Alembic reader turns it back: positions and velocities are
written as `(x, z, -y)` and triangle winding is reversed. USD files state `upAxis = "Z"`, OpenVDB transforms and
the motion JSON are in world metres, so none is converted. The witness property `z_up_without_user_rotation`
proves it per format (7.2).

### 5.12 Defects fixed by construction

| Defect (report) | Why it cannot recur |
|---|---|
| Snow role enum mismatch (B 156) | One `ROLE_TABLE`; every enum, operator argument and row is generated from it. |
| Foam metadata key mismatch (B 495) | One registry and one accessor; no second key exists. The test starts from `fluidity.link_outputs`. |
| Stale help text (B 37) | No hand-written sentence about what is supported; tooltips come from `SCENE_FIELDS`, support from native findings. |
| Surface hidden under import and export (B 35) | The surface is an output switch of the everyday panel and a product of the one run. |
| One error at a time (B 143) | Findings are a list from two checks, ordered by severity. |
| Heartbeat freshness at dequeue (B 295) | Ages use the native `sent_at`. |
| Windows-only executable lookup (B 553) | `to_platform_name` and one folder per platform. |
| Same-drive requirement (B 585) | Relative when possible, absolute otherwise, never an error. |
| Two lifecycles (B 168) | One `fluidity run`, one supervisor, one state machine for every scene. |
| A format nobody measured in Blender (Blender chapter, channel matrix) | The format table changes only through witness records from 7.2. |

### 5.13 Wind from Blender force fields (NEW, A20, D13)

This replaces the domain property `wind_acceleration` of the first version: the add-on reads Blender's own Wind
force field objects, as the gas branch did (`gas_authoring.py:35-37, 63-67` on `origin/codex/m7-low-mach-gas`:
direction = the object's world rotation applied to its local +Z, acceleration = direction times `field.strength`;
that file's `_solver_vector` axis swap belonged to the Y-up solver and is not ported).

1. **Which objects.** The objects of the domain's `force_field_collection`, or of the whole scene when it is
   empty, whose `field` is not `None` and whose `field.type` is not `'NONE'`. This is Blender's own rule for a
   physics domain's effector weights (`EffectorWeights.collection`: empty means every force field in the scene).
2. **The conversion, stated once.** A Wind field gives `WIND_ACCELERATION_PER_STRENGTH * strength * direction`,
   with `WIND_ACCELERATION_PER_STRENGTH = 1.0` m/s2 per unit of Strength (the gas branch used Strength directly as
   m/s2) and `direction` the field object's local +Z axis in world axes at that key, from `mathutils` (the keyed
   rotation as a `Quaternion` times `Vector((0, 0, 1))`); `logic/` receives a unit vector. Several Wind fields
   add. The factor appears in the tooltip of `force_field_collection` and of the `gas_wind[].acceleration` scene
   field, in the method paper (8) and in `test_wind_strength_to_acceleration`.
3. **Honoured** (written as `gas_wind` keys, `{time, acceleration}` in m/s2, world axes): strength, direction
   from the object's orientation, and the animation of both. The rotation curves and the `field.strength` curve
   are evaluated at every frame of the range when any of them has keys (5.1 item 6), one key at time 0
   otherwise; key time `(frame - first_frame) / frames_per_second`. A Wind object turned by an object above it,
   a constraint, a driver or a modifier gets `motion_not_keys` with "Bake to Keys".
4. **Reported, because the solver applies one uniform acceleration** (`HAS_KNOWN_LIMIT`, no repair, the run
   goes ahead with the uniform part; `object_name` the field object): `wind_falloff_not_honoured` when
   `falloff_power != 0`, `use_min_distance` or `use_max_distance` is set; `wind_noise_not_honoured` when
   `noise != 0`; `wind_shape_not_honoured` when `shape` is not `PLANE` or `POINT`, or `z_direction` is not
   `BOTH`. `wind_flow_not_honoured` when `flow != 0` is a `NOTE`, because a new Wind field in Blender 5.3 has flow
   1.0 and the note would otherwise read as a fault on every default field.
5. **Wind without gas.** `gas_wind` acts on the gas only; a Wind field in a domain without a gas substance gives
   `wind_without_gas` (`NOTE`: "Wind moves only gas in FluidiTy; this domain has no gas") and no keys.
6. **Every other field type** has one row in `FORCE_FIELD_TYPE_TABLE` (3.13): `NOT_HONOURED` gives
   `force_field_not_honoured` (`HAS_KNOWN_LIMIT`, the message names the type), `IGNORED` gives
   `force_field_ignored` (`NOTE`). A type the table lacks (a later Blender may add one) is `NOT_HONOURED`.
   Nothing is dropped without a finding. The table is checked against the type list the probe records from
   Blender 5.3 (`tests/addon/testdata/blender/force_field_types.json`) and against the Blender 5.3 manual page on
   force fields; a row changes only where the manual says otherwise, quoted in the commit.

### 5.14 Blender 5.3 API facts (D3)

Measured in Blender 5.2.3 when the plan was written and found true by a throwaway probe in Blender 5.3.0 (build
`02f0899d78e3`); the first task that uses each re-checks it with `tools/blender/probe_blender_api.py` (one
`PROBE <check> TRUE|FALSE` line per check, then the version and build hash; exit 1 on any false) and stops with
the probe output if it is false. Check names in brackets.

- [`cache_file_open`] No `bpy.data.cache_files.load`; a cache file comes from `bpy.ops.cachefile.open(filepath=...)`
  (works in background mode). [`volume_new`] No `bpy.data.volumes.load`; `bpy.data.volumes.new(name)`, then
  `filepath`, `is_sequence`, `frame_start`, `frame_duration`.
- [`cache_file_fields`] `CacheFile`: `filepath`, `is_sequence`, `object_paths`, `velocity_name`, `velocity_unit`
  (`SECOND`, `FRAME`), `forward_axis`, `up_axis`, `scale`. `MeshSequenceCacheModifier`: `cache_file`,
  `object_path`, `read_data` (`VERT`, `POLY`, `UV`, `COLOR`, `ATTRIBUTES`), `velocity_scale`; it can be added to a
  point cloud object. [`transform_cache`] `TransformCacheConstraint`: `cache_file`, `object_path`.
- [`volume_grids`] `VolumeGrid`: `name`, `data_type`, `channels`, `matrix_object`, `load()`; `VolumeGrids`:
  `frame`, `frame_filepath`, `error_message`. [`sample_grid`] `bpy.types.GeometryNodeSampleGrid` exists.
- [`importers`] `bpy.ops.wm.alembic_import`, `bpy.ops.wm.usd_import`, `bpy.ops.object.volume_import`.
  [`extension_path_user`] `bpy.utils.extension_path_user`. [`temp_override`] `bpy.context.temp_override`.
- [`extension_commands`] `blender --command extension build [--source-dir] [--output-dir] [--split-platforms]`,
  `extension validate [SOURCE_PATH]`, `extension install-file -r <repo> [-e] FILE`; `--offline-mode` is global.
- [`bake_to_keys`] "Bake to Keys" is `bpy.ops.nla.bake`.
- [`action_curves`] Actions are layered: `Action.fcurves` does not exist; an object's curves are reached through
  `bpy_extras.anim_utils.action_get_channelbag_for_slot(animation_data.action, animation_data.action_slot)`; a
  keyed field strength is the curve with data path `field.strength`.
- [`force_fields`] `bpy.ops.object.effector_add(type='WIND')` makes an empty with `field.type == 'WIND'`;
  `FieldSettings` has `type`, `shape`, `strength`, `flow`, `noise`, `seed`, `falloff_type`, `falloff_power`,
  `use_min_distance`, `use_max_distance`, `distance_min`, `distance_max`, `z_direction`, `apply_to_location`,
  `use_absorption`, `wind_factor`; a new Wind field has shape `PLANE`, strength 1.0, flow 1.0, noise 0, falloff
  `SPHERE` with power 0, no minimum or maximum distance, `z_direction` `BOTH`; `EffectorWeights.collection`
  exists. Field types in 5.3.0: `NONE`, `BOID`, `CHARGE`, `GUIDE`, `DRAG`, `FLUID_FLOW`, `FORCE`, `HARMONIC`,
  `LENNARDJ`, `MAGNET`, `TEXTURE`, `TURBULENCE`, `VORTEX`, `WIND`.

## 6. Main thread and background work

| Work | Where | Bound |
|---|---|---|
| Panel `draw`, operator `poll` | main | Read `ui_model` and properties only. No file, no process, no hash. |
| Property update callbacks | main | Set a flag in `ui_model`. |
| `scene_reader` | main | A generator that yields after each participant; run in slices of at most 8 ms by the timer while the state is `preparing` ("Reading objects, 4 of 9"), and to the end in one call in tests. |
| Quick check | main | Pure function of the description. Target 100 ms for the reference scenes. |
| Write scene file, `capabilities`, `check`, `report`, `read-particle`, `build_link_plan` | supervisor thread | Each cancellable by ending its process. |
| `fluidity run`, `resume`, journal reading, pause and stop | supervisor thread | 5.5. |
| `ui_timer.tick` (0.1 s while anything is active, 1 s otherwise) | main | Copies the snapshot if its change counter moved, updates `ui_model`, tags the Properties areas for redraw. Target 4 ms. Writes properties only when a run record or the registry must change (run created, run ended, a frame listed for a JSON motion or volume range). |
| `apply_link_plan` | main | Started by Show Result, or by the timer when `link_outputs_when_ready` is on and a checked plan is waiting. It reads no file and starts no process; its time is measured and shown. |
| `load_pre`, unregister, exit | main | `supervisor.close()`. |
| `load_post`, `undo_post`, `redo_post` | main | Clear `ui_model`, ask for `report` of every run record. |

Rules: only the main thread touches `bpy`. A timer or handler never keeps a Blender object between calls.
The supervisor thread is the only code that blocks. Registration is idempotent: `register` after
`unregister` leaves one timer, one set of handlers and one thread. The witness tools (3.12) run outside the
add-on and may block: they are test programs.

## 7. Tests

### 7.1 The add-on

Three test levels (D2): (a) plain pytest in `tests/addon/logic/` (no Blender, no native program; CTest
`addon_logic`) and `tests/addon/with_fluidity/` (the built `fluidity`, no Blender; CTest `addon_with_fluidity`),
both in the gate; (b) background Blender 5.3, `tests/addon/in_blender/test_*.py`, run by
`tools/blender/run_blender_tests.py`, not in the gate (the container has no Blender); (c) a Blender 5.3 window,
recorded by hand in `docs/blender-runs.md`. For (b) and (c) the commit names the build that ran (5.10).

**Plain `pytest`, no Blender** (`logic/`):

| Test | Setup | Expected |
|---|---|---|
| `test_apply_event_follows_the_table` | Every (state, event) pair and every exit code | The table of 5.4, exactly; other pairs unchanged |
| `test_three_facts_move_independently` | Journal: frames listed to 37, saved state 30, `last_try` at 38 rejected, process lost | `last_listed_frame == 37`, `saved_state_frame == 30`, `last_try.frame == 38`, state `failed`, `process_lost` |
| `test_ages_use_sent_time` | Heartbeat sent at t, read at t + 20 s | age of last event is 20 s at read time |
| `test_supervisor_never_blocks_the_solver` | Scripted process writes 10,000 heartbeats and `try_done` to the journal, then `done`, with stdout discarded; `read_snapshot` never called | Process ends; final snapshot is `done`; no `frame_done` or terminal event lost |
| `test_stop_now_and_lost_process` | Scripted process ignores `cancel` | `stopping`; second Stop Now ends the group; `failed` with `process_lost`; no process left |
| `test_close_stdin_stops_after_frame` | `close()` on a live scripted run | The script saw end of stdin; it wrote `stopping` (`end_of_input`) and `stopped`; nothing was killed |
| `test_follow_run_of_an_earlier_session` | Journal with `started` and `frame_done`, lock reported held, then `stopped` appended | `stopping`, then `stopped` with the journal's facts |
| `test_second_start_is_refused` | Two `request_start` | `RUN_ALREADY_ACTIVE` |
| `test_run_arguments` | Each `RunKind` | The exact argument lists of 3.5; never `--run-on` |
| `test_native_facts_parse_io_examples` | The I/O design's event examples and a recorded `capabilities`, `scene_checked`, `report`, `particle` | Every field of 3.5 filled; unknown keys kept in `other` |
| `test_budget_read_from_scene_checked` (NEW, A17, plan Task 12) | The recorded `scene_checked` of I/O plan Task 39's `check falling_block.json`; an event without `budget` | `budget.lines` equal to the event's list and `verdict == "fits"`; A25: `viscous_path == "pressure_solve"`, one liquid with viscosity 0; `None` |
| `test_water_preset_viscosity_zero` (NEW, A25, plan Task 9) | The `water` row of `SUBSTANCE_PRESETS` | `substances[].ingredients[].viscosity` is exactly 0.0, equal to I/O's `kWater` (`test_preset_values_as_designed` also asserts 0, not 0.001) |
| `test_performance_panel_draws_liquids_lines` (NEW, A25, plan Task 32, background Blender) | `FLUIDITY_PT_performance` with a recording layout; budget lines holding the `liquids:` block of I/O's `BuildBudgetLines, TwoLiquids_ViscousEffectLines` | Each `liquids:`, per-liquid and `note:` line drawn verbatim, once, in order, before `fits:` |
| `test_performance_panel_draws_budget_lines` (NEW, A17, plan Task 32, background Blender) | `FLUIDITY_PT_performance` with a recording layout after a full check with a budget | Each line drawn once, in order, under "Planned cost (estimate)"; nothing when `budget` is None |
| `test_role_table_is_the_only_source` | Import every operator and property | Each `role`, `shape`, `motion` enum equals `to_enum_items(ROLE_TABLE)` |
| `test_scene_fields_cover_the_scene_file` | Write a description with every field set | Every leaf path of the file has one `SceneField`; every `SceneField` appears; each has a unit and a tooltip |
| `test_repair_table_covers_suggested_changes` | A recorded `capabilities` | Every `SuggestedChange` name of the step and I/O's four have an entry |
| `test_preset_edit_becomes_custom`, `test_preset_changes_only_its_fields` | Apply syrup, edit viscosity | `CUSTOM`; other fields equal |
| `test_quick_check_reports_all` | A scene with five independent problems (one a turned gas source, one a colliding ingredient name) | Five findings, blocking first |
| `test_liquid_and_gas_in_one_domain` | Water and air substances, a liquid start region, a gas source | No quick finding; the file has two materials, `gas.sources`, `points.material_ids` without the gas |
| `test_find_native_executable_per_platform` | Fake package folders | Right file on three platforms; `PATH` is never read; a wrong protocol, schema or coordinate text is an error |
| `test_link_plan_from_report` | `report` with every product, one format per kind, one declaration with a format not in `LINK_ROUTES`, two fingerprints | Steps name the declared paths, object names and channels; `FORMAT_NOT_LINKABLE`; refused for two fingerprints |
| `test_cache_comparison_states` | `CacheComparison`: same; surface spacing changed; foam changed; viscosity changed; executable changed | `ready` rows; `surface_<id>` rebuildable; foam needs a new run; `needs_new_run` naming "Viscosity"; `needs_new_run` naming "executable" |
| `test_paths_on_another_drive` | Cache root that cannot be relative | Absolute path kept, no failure |
| `test_write_witness_records` | Hand-made Blender 5.3 results: all passed; one failed; one `not_run` | The exact C++ lines of 7.2, one result each, and the `chosen` values that follow |
| `test_write_witness_records_not_run_where_unmeasured` | A case absent from the results file | It keeps `kNotRun`, or `kDocumented` with its quote; never `kPassed` or `kFailed` |
| `test_records_refuse_other_blender_version` | A results file recorded by 5.2.3 or 5.4.0; one with another `files_digest` | Refused, naming the version; refused, naming the digest |
| `test_wind_strength_to_acceleration` | One Wind field, strength 2.5, axis turned 90 degrees about X (`direction == (0, -1, 0)`) | One key `(0, -2.5, 0)` m/s2 exactly |
| `test_two_wind_fields_add`, `test_animated_strength_and_turn_give_keys`, `test_negative_strength_reverses` | `(1, 0, 0)` and `(0, 2, 0)`; keys at 0, 1/24, 2/24 s with strengths 0, 1, 2 along +X, +Y, +Z | `(1, 2, 0)`; three keys equal to the products; the sign follows the strength |
| `test_wind_without_gas_writes_no_keys_and_a_note` | A Wind field, no gas substance | No keys; `wind_without_gas`, `NOTE` |
| `test_each_unhonoured_wind_setting_is_reported`, `test_default_wind_gives_only_the_flow_note` | Falloff power 1; `use_max_distance`; noise 0.5; shape `LINE`; `z_direction` `POSITIVE`; flow 1.0; Blender 5.3's new-field defaults | One finding each with the codes and severities of 5.13, keys still written; defaults: only `wind_flow_not_honoured` |
| `test_every_field_type_has_one_handling`, `test_unknown_field_type_is_not_honoured` | Each type of 5.14 but `NONE`; a type not in the table | The table's finding, message naming the type; `force_field_not_honoured` |
| `test_every_blender_field_type_has_a_row` | The probe's recorded type list | Every identifier but `NONE` is a key of `FORCE_FIELD_TYPE_TABLE`, and every key is in the list |

**Headless in Blender's Python** (`blender --background --factory-startup --python-exit-code 1`, in Blender 5.3;
timers do not fire in background mode, so tests call `ui_timer.tick()`):

- Register and unregister five times: classes, handlers, timer and thread counts return to zero.
- Each reference scene preset (wave, syrup, fire, snow example): created, read into a description, quick check
  clean, `fluidity check` exit 0 with no `unknown_scene_field`; the same with every optional field written.
- `scene_reader` on the asymmetric fixture (one point at (1, 2, 3) moving at (4, -5, 6), a box turned 30
  degrees about X then 20 about Z, keyed over three frames): the description holds the same numbers, the
  path keys' orientations are right to `1e-12`, and the current frame is unchanged.
- Motion from a parent or constraint gives the "Bake to Keys" finding.
- Wind: a field made with `effector_add(type='WIND')`, strength keyed 0 at frame 0 and 3 at frame 2, rotation
  keyed from none to 90 degrees about Y, linear curves: keys at 0, 1/24, 2/24 s equal to strengths 0, 1.5, 3
  times the axis turned 0, 45, 90 degrees about Y within `1e-12` (frame 2 gives `(3, 0, 0)` m/s2), current frame
  unchanged; two Wind fields, one in the chosen collection: only its keys, collection cleared: both summed; a
  Wind field with noise and a Vortex field: `wind_noise_not_honoured` and `force_field_not_honoured` naming
  `VORTEX`; a Wind field turned by a parent: `motion_not_keys`; one empty per field type with its type set: `WIND`
  gives keys, every other type exactly its table's finding.
- `force_field_collection` accepts a collection and `None`; the domain has no `wind_acceleration`; the
  property's description contains "1.0 m/s2 per unit of Strength".
- A `bpy.app.version` outside 5.3 gives one note: `(5, 3, 0)` and `(5, 3, 2)` none, `(5, 4, 0)` one.
- `apply_link_plan` with the witness files of the chosen formats (3.12): objects evaluate at three frames
  forward and backward, including an empty foam object; point count, positions, velocities, radius and one
  surface attribute match `expected.json` within float32 round-off; volume grid names and the transform
  origin match; JSON motion keys equal the poses. A second plan with a truncated file, a wrong
  `run_fingerprint` and a missing channel leaves every row and every pointer as before. An edited material and
  an added modifier survive a relink.
- The same with the real `fluidity` executable on the CPU: check, bake 6 frames of the small syrup scene,
  pause at 3, resume, link, compare point counts with the `frame_done` file entries; change the surface
  spacing, Rebuild Outputs, only the surface files change; smoke over water (liquid and gas in one domain,
  A10) for 4 frames: surface and gas volume linked in one plan.
- File load and scene deletion during a scripted run: no exception, no process killed, the run ends `stopped`.

**Needs a real Blender session** (a window): every panel draws in every state, with each role assigned and
unassigned, at 424 px, with no error in the console; the timer keeps the viewport responsive during a run
(largest tick time recorded); scrubbing the timeline during `running` shows frames up to
`last_listed_frame`; a viewport render and a final render with motion blur.

**Artist's journey** (`tools/blender/run_artist_journey.py`; per Blender 5.3 build and platform, receipt in
`docs/blender-runs.md` with commit, package hash, Blender version and build hash, platform and processor; each
step its own Blender process with `--offline-mode` and a clean `BLENDER_USER_RESOURCES`): (1) clean profile,
install the package from disk offline;
(2) enable, disable, enable; (3) Add Reference Scene, Check, Bake; (4) Pause, see the three facts; (5) Resume
to the end; (6) save, close Blender during a second run, reopen: the run is `stopped` with the frame in
progress listed, resume works; (7) edit a viscosity: `needs_new_run` names the field, New Run keeps the old
outputs; edit the surface spacing: Rebuild Outputs; (8) move the project and run folder together, reopen,
scrub; move only the run folder, Find Moved Cache; (9) start Blender with the add-on disabled and script
auto-execution off, scrub forward and backward, render two frames, compare with the frames rendered in
step 5. Run for the three reference scenes and the snow example at small size on the CPU; on the GPU in the
owner's hardware run.

### 7.2 Witness tests: USD or Alembic for points, surface and solid motion (A14)

A deliverable of this module, run before the link routes are written, because their result fixes the formats
the routes serve. They decide every `WitnessRecord` of render output's format table.

**The fixed scene** (`tools/blender/witness/witness_scene.json`, render output 5.7's witness scene made exact):
domain 0.6 x 0.6 x 0.8 m, closed on every side but the top; cells 0.02 m, one size level; 24 frames per second,
frames 0 to 10. A sphere of radius 0.1 m of a two-ingredient liquid (ingredients `fresh` and `salt`, shares
0.7 and 0.3), centre (0.15, 0.3, 0.6) m, start velocity (0.4, -0.3, 0) m/s, falling into a 0.06 m layer of a
second liquid that does not mix, which fills the floor of the other half (x above 0.32 m) and does not reach
the sphere's landing point before frame 10. A `Box` solid, half size (0.05, 0.03, 0.02) m, driven by keys from
(0.45, 0.15, 0.4) m moving at (0, 0.6, 0) m/s and turning about its own X at 0.5 rad/s. Foam on (default
settings, display radius 0.005 m). A gas source box (0.05 m cube at the floor, x below 0.1 m) with fuel
0.1 kg/(m3 s) and heat 20,000 W/m3 from frame 0, so the gas volume is not empty at frame 1. Outputs: points of
both liquids with `mass` requested, surface (one group), foam and spray, foam volume, gas volume, solid motion,
`keeps_frame_inputs`. `fluidity check` must exit 0 on it; if a capability exclusion blocks it in the build
under test, the witness run stops and reports the finding: it never edits the scene.

**The files.** `fluidity run witness_scene.json --into <folder>`, then `fluidity_write_witness_files` (3.12)
writes every candidate of every kind, `expected.json` and `files_digest.txt`. The witness then needs no simulation: the files are opened
in Blender 5.3, and `files_digest` names them in every record.

**How Blender is started.** `blender --background --factory-startup --python-exit-code 1 --python
tools/blender/witness/run_witness.py -- <witness_folder> tools/blender/witness/results/results_<version>.jsonl`,
once, with Blender 5.3 (`<version>` as `--version` prints it), after the probe has re-checked the API facts the
routes use (5.14). `--factory-startup` loads no user add-on; the script first asserts that no add-on
module named `fluidity` or `bl_ext.*.fluidity` is enabled (`bpy.context.preferences.addons`) and that no
`fluidity` class is registered, and records `bpy.app.version_string` and `bpy.app.build_hash`. Each case
starts from `bpy.ops.wm.read_factory_settings(use_empty=True)` (to verify against the Blender 5.3
documentation) and sets the scene's frame rate to 24.

**Opening, as the Blender manual describes it** (one route per format; each call to verify against the Blender
5.3 documentation; the route itself is what is being witnessed, so a failing call is recorded, not worked
around):

| Format | Route |
|---|---|
| Alembic (points, surface, solid motion) | `bpy.ops.wm.alembic_import(filepath=<frame 0 file>, is_sequence=True)`; then on the created cache file `velocity_name = ".velocities"`, `velocity_unit = 'SECOND'` (the one reader setting the manual names, 4.2) |
| USD (points, surface, solid motion) | `bpy.ops.wm.usd_import(filepath=<frame 0 file>)`; if the importer created no cache file, the new cache file of `bpy.ops.cachefile.open(filepath=<frame 0 file>)` (5.14) and a Mesh Sequence Cache modifier (points, surface) or Transform Cache constraint (solid motion) on the imported object with its prim path; then `is_sequence = True` on the cache file |
| OpenVDB (volumes) | `bpy.ops.object.volume_import(filepath=<frame 0 file>, files=<every frame>)`, then `is_sequence = True`, `frame_start = 0`, `frame_duration = 11` on the volume data |
| Solid motion JSON | No native reader: the add-on links it (`apply_link_plan`, keys on a copy of the solid's object) in one session, the `.blend` is saved, and the measurement runs after reopening it with `--factory-startup` and the add-on disabled |

**Measuring.** Frames are visited in the order 1, 2, 10, 2, 1 with `scene.frame_set(f)`; geometry is read from
`obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data`, attributes with `attributes[name].data.foreach_get`
into a float or int array. "Equal" means: counts, names, domains, data types and int values exact; a float
within `4 * 2^-24` of the expected value's magnitude (float32 round-off) and, for positions, within
`1e-6 m`; a rotation within `1e-6 rad`. Motion blur renders use Cycles on the CPU, 64 x 64 samples at most,
`film_transparent = True`, an orthographic camera looking along +Y (image axes world X and Z), `ortho_scale
= 0.6 m` over 1200 px (0.5 mm per pixel), an emission material on the measured object and nothing else
visible, `motion_blur_shutter = 0.5` frame centred on the frame; the image is read back with
`bpy.data.images.load(...).pixels` and pixels with alpha above 0.5 are the silhouette. The *extent growth* is
the extent of the silhouette along the expected image-plane direction of motion with blur on, minus the same
without blur.

**Cases**, one per (kind, candidate, property) the render output table requires; each writes one result
line. The `WitnessRecord` line it contributes is, with `<P>` the property's enumerator, `<r>` the Blender 5.3
result and `<test>` the case's test name (D12: one result per record):

```text
{WitnessProperty::<P>, WitnessResult::<r>, "<test>; Blender <version> <build hash>; files <files_digest>"},
```

A `kDocumented` record that is measured keeps its manual quote after `" | "` in the evidence.

| # | Property (render output 5.7 name) | Kinds and candidates | Measured from Python | Passes when | Test name (line contributed) |
|---|---|---|---|---|---|
| W1 | `kFileSequenceReadNatively` | points: USD, Alembic; surface: USD, Alembic; volume: OpenVDB; solid motion: USD, Alembic, JSON | At each visited frame: point or vertex count, the first and last position; volume: `grids.frame_filepath` (to verify) names the frame's file and the active voxel count of `known` (7.2, W10 sampling); solid: `matrix_world` of `solid_<id>` | All five visits equal `expected.json` of that frame, including the way back | `witness_file_sequence_read_natively[<kind>,<format>]` |
| W2 | `kTopologyMayChangePerFrame` | surface: USD, Alembic | `len(mesh.vertices)`, `len(mesh.polygons)` at 1, 2, 10; each polygon's three positions in order; `polygon.normal` against the expected outward normal | Counts and positions equal per frame, every normal's dot product with the expected one positive, and (precondition) the expected counts differ between frames 2 and 10 | `witness_topology_may_change_per_frame[surface_mesh,<format>]` |
| W3 | `kVelocityGivesMotionBlur` | points: USD, Alembic (the particles object of the sphere's liquid); surface: USD, Alembic | Extent growth at frame 2, where the sphere is in free flight with one known velocity, along the image-plane velocity `(v_x, v_z)` | Renders with and without blur differ, and the growth is within 10 % of `sqrt(v_x^2 + v_z^2) * shutter / 24` m (render output 5.7) | `witness_velocity_gives_motion_blur[<kind>,<format>]` |
| W4 | `kVelocityInSeconds` | points, surface: USD, Alembic | The evaluated `velocity` attribute (to verify: name and type `FLOAT_VECTOR` on `POINT`) at frames 1, 2, 10 | Equal to the expected m/s values with no scale other than the Alembic reader setting of the route; absent or scaled (a ratio of 24 or 1/24) fails | `witness_velocity_in_seconds[<kind>,<format>]` |
| W5 | `kPointRadiusArrives` | points: USD, Alembic (particles, droplets, spray, foam, bubbles objects) | The evaluated object type and its `radius` attribute | The object evaluates to a point cloud (`'POINTCLOUD'`) and `radius` equals width / 2 of every point | `witness_point_radius_arrives[points,<format>]` |
| W6 | `kPointAttributesArrive` | points: USD, Alembic; surface: USD, Alembic | Every channel of the product whose `witnessed_by` is this property (`id`, `volume_ratio`, `size_level`, `share_fresh`, `share_salt`, `mass`, `age`, `distance_to_solid`, `blob_index`): attribute by `blender_name`, its `domain` and `data_type` | Each present on `POINT` with `FLOAT` (or `INT` for int32 channels) and every value equal | `witness_point_attributes_arrive[<kind>,<format>]` |
| W7 | `kFaceAttributesArrive` | surface: USD, Alembic | `material_index` and `triangle_kind` attributes | Each present on `FACE` with `INT` and equal per triangle | `witness_face_attributes_arrive[surface_mesh,<format>]` |
| W8 | `kZUpWithoutUserRotation` | points, surface: USD, Alembic; volume: OpenVDB; solid motion: USD, Alembic | `matrix_world` of each imported object at frame 1 (for solid motion: of its parent, if the importer made one); world positions `matrix_world @ co`; volume: `grid.matrix_object` (to verify) | No rotation was added (identity, or for solid motion the expected pose only), and world positions equal the expected Z-up positions, so the sphere's start velocity sign along X and -Y shows | `witness_z_up_without_user_rotation[<kind>,<format>]` |
| W9 | `kTransformGivesObjectMotion` | solid motion: USD, Alembic, JSON | `matrix_world` of `solid_<id>` at the visited frames; extent growth of the box at frame 2 | Poses equal origin and `ToAxes(orientation)`; renders with and without blur differ and the growth is within 10 % of the extent change of the box's eight corners moved by the expected poses over the shutter | `witness_transform_gives_object_motion[solid_motion,<format>]` |
| W10 | `kGridsArriveByName` | volume: OpenVDB (foam volume and gas volume) | `[g.name for g in volume.grids]`, each `g.data_type`, `g.matrix_object`; values at 64 sample points of `expected.json` read by a Geometry Nodes tree (points at the samples, Sample Grid of the named grid, Store Named Attribute; node names to verify against the Blender 5.3 documentation) | Names equal the expected set (`density`, `smoke`, `temperature`, `fuel`, `flame`, `known`; `foam_count_density`, `bubble_count_density`, `bubble_volume_fraction`, `known`), every grid `FLOAT`, voxel size equal to the lattice spacing, voxel (i, j, k) centred at lattice point (i, j, k), sampled values equal | `witness_grids_arrive_by_name[volume,open_vdb]` |

A result is `kPassed` or `kFailed` only when Blender's reader produced the data and the script measured it; a
missing attribute, a wrong count or a wrong value is `kFailed`. When the script itself could not measure (a
Python API of the measurement is missing, or a precondition of the fixture is false, such as no foam at frame
10), the result is `kNotRun` with the reason in `detail`, and the witness run fails as a test failure to be
fixed: it never counts as a pass or as a failure of the format. A record whose case was not measured keeps its
earlier value (`kNotRun` with empty evidence, or `kDocumented` with its quote), never `kPassed` or `kFailed`.

**From results to the format table** (`write_witness_records.py`, plain Python):

1. Read `results_5.3.x.jsonl` (one file; a file recorded by another Blender version is refused); its
   `files_digest` must equal the witness folder's `files_digest.txt`.
2. For each kind and candidate, print one `WitnessRecord` line per required property, in `WitnessProperty`
   order (the `FormatCandidate::witnesses` order), as above; print the `chosen` value `ChooseFormat` gives.
3. The lines replace those of `render_output/format_table.cc` in one commit with the results file; render
   output's tests `FormatTable, ChosenFollowsWitnesses` and `FormatTable, EveryChannelIsWitnessed` must pass.

**The rule that follows** (render output 5.7, A14):

- A USD candidate that passes every witness of its kind in Blender 5.3 becomes the chosen format of that kind
  (`ChooseFormat` takes the first candidate whose every record is `kPassed`, and USD is first).
- A USD candidate that fails any witness is removed with its writer in the commit that
  records the failure, and its route leaves `LINK_ROUTES` in the same commit; when no USD candidate is left,
  OpenUSD leaves `fluidity_interchange` and the package (5.10).
- A non-USD candidate that fails a witness is removed in the same way when another candidate of its kind
  passed. If the kind's last candidate fails and no other passed, the format stays, the channels whose
  `witnessed_by` property failed are listed as "written, not witnessed" and withdrawn from the promise, and
  the owner is asked: never a silent change of format.
- The witnesses run again, and the table is updated by the same steps, when the supported Blender version
  changes, a pinned library of `fluidity_interchange` changes, a writer of `render_output/interchange/` changes,
  or `kOutputContractVersion` changes. CI runs them on every change of those files and of `output_links.py`.

## 8. Method paper section

`== From a Blender scene to playback <sec-blender>`, a short closing section: the scene model (role, shape,
motion; substance and ingredient), the wind from Blender's force fields (one paragraph and the equation
`a = 1.0 m/s^2 * strength * axis` summed over Wind fields, why the factor is 1, what is reported instead of
honoured, and the table of every field type with its finding code), the two checks, one run with three separate
facts, outputs as open-format files linked with Blender's own readers, and how a format is chosen by witness
tests in Blender 5.3. Figures: (1) the run state machine (with `needs_new_run`) and the three facts drawn as
three marks on one frame axis; (2) what depends on what: scene file, simulation, each output, Blender objects,
and which edit makes which stale; (3) the witness matrix (kind and candidate against property, one mark per
cell: the Blender 5.3 result). Pseudocode: `apply_event` as the transition table (`<proc-apply-event>`), and the
link transaction (`<proc-apply-link-plan>`). Sources: none in the old
bibliography; Blender manual pages are cited by URL copied at writing time.

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| `boundary_mode`, `TANK`, `PLANE`, `SLAB`, `SNOW_PLATEN`, `LEGACY_PADDLE` | role, shape, motion; `closed_<side>` | DECIDED words; VERIFIED by reading `__init__.py:93-106` |
| `collider`, `paddle`, `cutter`, `rigid_box`, `elastic_solid`, `snow_mound` | `SOLID` with `FLAT_SOLID`, `PLATE`, `BOX` and `FIXED`, `DRIVEN`, `FREE`; `START_MATERIAL` (`start_regions` in the scene file) | DECIDED |
| `substance_id`, `phase_id` | `substance_id`, `ingredient_id`; mixing by membership (Q31 a) | NEW / owner Q31 |
| substance (UI) | substance = native `material` | NEW, decision 1 |
| `emission_rate`, `emitted_volume` | `volume_rate` | NEW |
| `backend`, `cuda`, `RTX` | `run_on`, `CPU`, `GPU` | DECIDED |
| `Job`, `FrameJob`, `CutterJob`, `RigidJob`, `stage` | run, `RunSupervisor`, `activity` | NEW; `activity` is I/O's |
| `committed_frame`, `published` | `last_listed_frame` | I/O decision 2 (`listed`) |
| checkpoint, `.chk`, `state.flst` | `saved_state`, `saved_state_frame` | DECIDED |
| `execution_id`, `source_execution_id`, `fluidity_execution` | `run_fingerprint`, `output_fingerprint` | DECIDED (`run_fingerprint`) / I/O |
| manifest (`.flmf`) | the run's `file_list`; `OutputDeclaration` | DECIDED / I/O |
| cache writer lease | `WriterLock` (I/O's; the add-on reads only `lock_holder.is_held`) | I/O decision 3 |
| `read-cache` (first draft) | `report`, `RunFolderFacts` | I/O decision 1 |
| `run_started`, `frame_published`, `saved_state_written`, `pause_acknowledged`, `run_done` (first draft) | `started`, `frame_done`, `saved_state_listed`, `stopping`, `done` / `stopped` / `failed` | I/O |
| pause (native request) | `stop_after_frame`; "Pause" stays the button label | I/O 5.8 |
| `SceneError(field, message)`, preflight | `Finding` (`module`, `code`), quick check, full check | DECIDED (`check`) / I/O |
| attach, `playback_object`, `snow_playback_object`, `surface_object` | link, `output_estimate` | NEW |
| carrier (comments) | output file, product | NEW / render output |
| channel matrix | witness test, `WitnessRecord` | render output (A14) |
| `energy_blend` | `own_motion_kept` | DECIDED |
| `mesh_cache`, `mesh_spacing`, `mesh_sigma` | `surface_spacing`, `surface_smoothing_width` (`outputs.surface.spacing`, `.smoothing_width`) | render output's `SurfaceSettings` |
| `snow_shear`, `snow_lambda`, `snow_compression`, `snow_stretch` | `stiffness_against_shear`, `stiffness_against_squeeze`, `max_springy_shortening`, `max_springy_lengthening` | DECIDED / materials (A4) |
| `Fuel Rate`, `Heat Rate` custom properties; Wind force field | `fuel_rate` kg/(m3 s), `heat_rate` W/m3; the Wind field kept, read through `force_field_collection` into `gas_wind` keys (`WIND_ACCELERATION_PER_STRENGTH`) | gas; A20, D13 (replaces `wind_acceleration` of the first version) |
| (first version of this design) `core/` | `logic/` | D1: says what the package holds |
| (first version) `LinkStep.stored_path` | `path_in_blend` | D1: says where the path lives |
| (first version) `(code, support, message)` | `(code, level, message)`, `ToName(step::CapabilityLevel)` | D1; the step's jargon rename of `support` stays (A22) |
| (first version) `RunState.INCOMPATIBLE` | `RunState.NEEDS_NEW_RUN` | D1: says what the artist must do |
| (first version) `ignored_tail_bytes`, `OutputChannelFact.storage` | `ignored_end_bytes`, `value_type` | D1: the names I/O and render output print |
| (first version) `to_platform_name(system, machine)` | `to_platform_name(os_name, machine)` | D1: the value is the operating system's name |
| `WitnessOutcome` (first plan) | `WitnessCaseResult` | D1: role suffix rule |
| acceptance journey | artist's journey, `tools/blender/run_artist_journey.py` | D1: whose path the test walks |
| `heartbeat_age` | age of last event, age of last advance | NEW |
| `validate`, `verify_*` (tools) | check | DECIDED |
| cancel | Pause (`stop_after_frame`, keeps the frame), Stop Now (`cancel`) | I/O |

Words checked: `lease` and `publish` are never-say words and do not appear in this design's names (I/O
decisions 2 and 3). Blender's own file name `blender_manifest.toml` is not ours to rename. Under A22 the
ordinary words `link`, `install`, `support`, `node`, `parent`, `lower` and the others of A22 are banned only in
their fluid-jargon sense: the add-on keeps Blender's own words (`fluidity.link_outputs`, `fluidity.relink_run`,
`LinkPlan`, `LINK_ROUTES`, `link_outputs_when_ready`, `node_assets.py`, "install", "Copy Support Report",
`fluidity.copy_support_report`), and the rules check prints a note for each. The names `attach_plan.py`,
`output_attach.py`, `ATTACH_ROUTES`, `fluidity.attach_outputs`, `fluidity.reattach_run`, `copy_problem_report`,
`starter_assets.py` and "add the package" of the first version of the plan are withdrawn.

## 10. Not ported

- `jobs.py` `FrameJob`, `CutterJob`, `RigidJob`; `blender_bridge` as a second executable; the five text
  packets and `prepare`: one scene file and one `run`.
- `boundary_mode`, `planar_fill`, every `snow_*`, `rigid_*`, `elastic_*` scene field; `check_device` and the
  pinned GPU name; the 24 fps, 240 frame, 1024 particle and 100 m limits (limits now come from the native
  check).
- Particle-rate source controls (programme design, row 12).
- `interchange.py` export of Alembic and USD and the generic import operator: outputs already are open-format
  files; Blender's File menu imports anything else.
- `grid_asset.py` (Rasterize Points density): a derived view for 5.3 only; not needed by the reference scenes.
- `volume_playback.py` and `gas_volume.py` writing OpenVDB from Python; `gas_authoring.py` custom
  properties `Fuel Rate`, `Heat Rate`, `scene.frame_set` sampling, the domain pointer to one Wind object and its
  `_solver_vector` Y-up swap (the Wind force field itself is read, 5.13).
- `fluidity.help` popups with hand-written support text; the `show_debug` preference (Inspect is a panel).
- `translations.py`, `locale/`, `tools/blender/compile_translations.py` (decision 8).
- Example `.blend` files in the package: the reference scene presets create them.
- The 18 `tools/blender/verify_*.py` scripts and `qualify_point_cache.py`: replaced by the tests of section 7.
- Sand: the `SAND` kind, `friction_angle`, `cohesion`, the sand preset and the sand reference scene (A9: the
  sand module adds them).
- Report items left for later, each its own decision: field slices and vector views, picking in the
  viewport, tracking, capture and replay from the UI, Collect for Handoff, automatic restart, evaluated
  mesh shapes, a run that goes on past the frame in progress after Blender closed (decision 2).

## 11. Decisions

Already asked and applied: Q31 a (a substance is what mixes), Q50 (syrup values), Q35 (`friction_scale`), A1
to A22 where they touch this module (A9: no sand; A10: liquid and gas in one domain; A12: `run_on` is never
changed silently; A14: open formats and the witness tests of 7.2; A16: Blender 5.3 only; A20: wind from
Blender's force fields; A22: ordinary words outside their fluid-jargon sense), and the Blender plan's decisions
D1, D2, D3, D8, D12, D13 (`plan-decisions.md`, "Amendments of 2026-11-14 to the Blender and render-output plans").
A decision replaced by a later one keeps its text and is marked "Replaced".

1. **Words that meet the never-say list.** The brief for this module uses `publishing`, "published prefix"
   and "cache writer lease".
   ```python
   RunState.PUBLISHING; status.published_through        # a) as briefed
   RunState.WRITING_FRAME; status.frames_ready_through  # b) keeps "publish" banned everywhere
   ```
   And the artist's word for the native `material`: a) `substance` in panels and Python, `materials` in the
   scene file (Blender already calls appearance a Material); b) `material` everywhere. Recommendation: a for
   both, with `publish` allowed for output frames only and `lease` settled in design 12.
   Decision: the I/O design settled the first part (its decisions 2 and 3: `listed` and `WriterLock`), so no
   `publish` and no `lease` appear here: `last_listed_frame`, no publishing state; `substance` in panels and
   Python, `materials` in the scene file (option a for the second part). Recommended option (owner's standing
   instruction, 2026-11-14).
2. **Does a run stop when Blender closes?** a) yes: closing pauses it at a safe point (end of stdin); long
   bakes without Blender use the same `fluidity run` in a terminal; b) the run continues and Blender
   reconnects through the journal. Recommendation: a now; b is possible later without a format change.
   Decision: a (recommended option, owner's standing instruction, 2026-11-14), in I/O's form: end of standard
   input is `stop_after_frame`, so after Blender closes the run finishes and lists the frame in progress with
   a saved state and then stops; the add-on kills nothing on close and, after reopening, follows the journal
   of such a run until its `stopped` (5.5 steps 6 and 9).
3. **One worker thread plus a journal on disk.** The report offers this or keeping a pipe reader. The thread
   never touches Blender and the solver never waits for it. Alternative: no thread, the timer reads a small
   status file itself (file reads on the UI path). Recommendation: the thread.
   Decision: the thread, reading `events.jsonl`; the run's standard output is discarded so no pipe can block
   the native program (recommended option, owner's standing instruction, 2026-11-14).
4. **Settings live on the domain object**, so a scene can hold several domains and deleting the domain
   deletes its settings. Alternative: one group on the scene as on `main`. Recommendation: the object.
   Decision: the object (recommended option, owner's standing instruction, 2026-11-14).
5. **Driven motion comes only from the object's own keys** (read without changing the frame); anything else
   is a finding with "Bake to Keys". Alternative: step the scene frame by frame during preparing, which
   handles parents and constraints but moves the timeline and costs time on the main thread.
   Recommendation: keys only.
   Decision: keys only (recommended option, owner's standing instruction, 2026-11-14).
6. **Platforms and Blender versions.** Proposed: `windows-x64`, `linux-x64`, `macos-arm64` (CPU only);
   Blender 5.2 LTS from 5.2.3 and 5.3, no maximum version. Confirm or cut.
   Decision: as proposed (recommended option, owner's standing instruction, 2026-11-14); the witness tests
   run in both Blender versions.
   Replaced (A16, 2026-11-14) for the Blender part: Blender 5.3 is the one supported version; manifest
   `blender_version_min = "5.3.0"`, no maximum; a version outside 5.3 gets one note; the witness tests run in
   Blender 5.3 only (5.10, 7.2). The platforms stay as proposed.
7. **Output files.** One file per frame for every kind, so a running bake is viewable and nothing is
   written twice; surface as USD with `velocities`. If Blender 5.3 does not use those velocities for motion
   blur through the Mesh Sequence Cache, motion blur of the surface is left out of the product promise
   rather than switching the surface to Alembic and losing named attributes. Confirm that order of
   preference.
   Decision: owner answer A14 replaces the proposal: one file per frame in an open format for every product;
   per kind the format is the first candidate (USD first) that passes every witness of 7.2 in Blender 5.2 and
   5.3, chosen through render output's format table; today points and surface Alembic, volumes OpenVDB, solid
   motion JSON. A failed USD candidate is removed with its writer and its link route.
   Replaced in part (A16, D12, 2026-11-14): "in Blender 5.2 and 5.3" reads "in Blender 5.3"; a `WitnessRecord`
   holds one result, `blender_5_3`.
8. **French labels.** `main` ships a French catalogue. Proposed: not ported now; native messages are
   English. Alternative: keep the catalogue mechanism for labels only.
   Decision: not ported; English labels (recommended option, owner's standing instruction, 2026-11-14).
9. **Preset values** for sand, fuel gas and water's viscosity (0 on `main`; 0.001 Pa s is the physical
   value) are not fixed by any design. Confirm that they are copied from the reference scenes once those
   are final.
   Decision: copied from I/O's `kMaterialPresets` and the reference scenes the designs fixed (5.2): water
   0.001 Pa s; the gas preset is I/O's `kAir`; no sand preset here (A9, the sand module adds it). Recommended
   option (owner's standing instruction, 2026-11-14).
   Replaced (A25): water 0 Pa s, the owner's choice (a liquid is viscous exactly when its viscosity is above 0;
   water scenes use the pressure solve).
10. **Verbs outside the list**: `start_run`, `request_pause`, `resume_run`, `link` (as in `apply_link_plan`,
    "Find Moved Cache"), plus Blender's own `register`, `draw`, `poll`, `execute`. Confirm.
    Decision: confirmed, with `request_pause` renamed `request_stop_after_frame` and `request_cancel` added to
    follow I/O's stop requests; the button labels stay Pause and Stop Now (recommended option, owner's
    standing instruction, 2026-11-14).
11. **The timer may apply a prepared link plan.** "Timers only read prepared status" is kept for everything
    else. Alternative: outputs appear only when the artist presses Show Result. Recommendation: allow it,
    behind `link_outputs_when_ready`.
    Decision: allowed, behind `link_outputs_when_ready` (recommended option, owner's standing instruction,
    2026-11-14).
12. **Inspect in the first version** is facts and text (the three facts, last try, ages, one particle by
    id, log, support report). Slices, vectors and picking wait for diagnostic output that design 13 does not
    yet promise. Confirm.
    Decision: confirmed as designed (recommended option, owner's standing instruction, 2026-11-14). Read
    Particle takes the full 64-bit id; the `id` attribute in Blender is its low 31 bits (render output 5.7),
    so the panel says so when a typed id is below 2^31 and the run has more particles than that.
13. **Names that meet the ordinary-word list** (A22). The first version of the plan replaced the design's `link`
    names with `attach` names, `copy_support_report` with `copy_problem_report`, `node_assets.py` with
    `starter_assets.py` and "install" with "add the package".
    Decision: owner answer A22 with plan decision D1: Blender's own words are kept as this design wrote them
    (`link` names, `node_assets.py`, "install", "Copy Support Report"); the renames that stay, with their reasons,
    are in section 9 (`logic/`, `path_in_blend`, `(code, level, message)`, `NEEDS_NEW_RUN`, `ignored_end_bytes`,
    `value_type`, `to_platform_name(os_name, machine)`, `WitnessCaseResult`, "artist's journey").
14. **Wind.** The first version read wind as a keyable domain property `wind_acceleration`.
    Decision: owner answer A20 with plan decision D13: Blender's Wind force fields, read through
    `force_field_collection`, `WIND_ACCELERATION_PER_STRENGTH = 1.0` m/s2 per unit of Strength, every other field
    type reported by `FORCE_FIELD_TYPE_TABLE` (3.13, 5.13).
15. **Witness records with one Blender version.**
    Decision: owner answer A16 with plan decision D12: each record holds the Blender 5.3 result; a record not
    measured keeps its earlier value; a USD candidate that passes every witness in 5.3 is chosen, one that fails
    is removed with its writer and its link route (7.2).
16. **The planned cost before a run.** Owner principle A17 (a run is all CPU or all GPU; what does not fit moves
    in planned copies) and the owner's "warning the user about the expected cost (memory and time) is nice"
    (A22: `budget` is a planned cost shown to the user).
    Decision: `RunPlanFacts.budget: RunBudgetFacts | None`, read from `scene_checked.budget` with the I/O schema
    names; `run_mode` also accepts `"batched_with_solves_in_pieces"` (A15); `FLUIDITY_PT_performance` draws its
    `lines` verbatim under "Planned cost (estimate)"; the add-on computes nothing from the numbers; the
    remaining-time estimate of the status line stays the add-on's own (three listed frames, the same rule)
    (3.5, 3.11, 4.1, 5.3, 5.9, 7.1). Plan decision, 2026-11-14 (planned cost).
17. **The third run mode in the run status.** `RunStatus.run_mode` (3.6) and the `started` event's `run_mode` (4.1)
    said "one_piece" | "batched".
    Decision: both take `"one_piece"`, `"batched"` or `"batched_with_solves_in_pieces"` (`plan-decisions.md`,
    Reconcile pass, coordinator item (7); add-on plan Task 17).
18. **What makes a liquid viscous, in the add-on.** The setting decides: a liquid is viscous exactly when its
    viscosity is above 0, for the whole run.
    Decision: the water preset's viscosity is 0 (5.2, decision 9); the viscosity property's tooltip is the A25 text
    of 3.2; `RunBudgetFacts` gains `viscous_path` and `liquids` (3.5); the native `liquids:` budget lines are drawn
    verbatim in `FLUIDITY_PT_performance` as part of `RunBudget.lines` (3.11, 5.3); tests
    `test_water_preset_viscosity_zero`, `test_performance_panel_draws_liquids_lines` (7.1). Owner answer A25;
    `plan-decisions.md`, "A25: what counts as viscous, and the viscous effect per step"; add-on plan Tasks 7, 9,
    12, 32.
19. **No viscosity repair in the main sequence.** A free solid beside a viscous liquid now runs (A27), so the
    step's `CapabilityReason::kViscousLiquidWithFreeSolid` and `SuggestedChange::kSetViscosityToZero` are gone
    (`kCapabilityReasonCount` 16 at most, 13 if the viscosity plan lifts its three exclusions;
    `kSuggestedChangeCount` 12); and I/O's budget table per liquid is `ViscousEffectOfLiquid` (was
    `LiquidViscousEffect`, the step's struct's name).
    Decision: `REPAIR_TABLE` has 11 step names, no `set_viscosity_to_zero` entry (3.4); the sand module adds the
    name back with `kViscousLiquidWithSand`, and the property edit to viscosity 0 with it; `RunBudgetFacts.liquids`
    reads the renamed table's unchanged fields (3.5). Owner answer A27; `plan-decisions.md`, "A viscous liquid
    sticks, and free solids in the viscous solve (A26, A27)", "Exclusions"; add-on plan Tasks 11, 12.

**Open** (for the owner; nothing here is built on an assumption):

- The add-on does not
  yet read `started.budget` or the `cost` of `done` / `stopped` / `failed` (I/O carries both); whether it shows the
  measured cost and its warnings is not decided.
- Default Wind fields carry flow 1.0 in Blender 5.3, so every default field gives the `wind_flow_not_honoured`
  note; whether that note should be dropped for the default value, or flow given a meaning, is not decided.
- A maximum Blender version: the manifest has none, so a later Blender loads the add-on with one note; whether
  to set `blender_version_max` once a later version is known to break it is not decided.
- Wind on liquids: `gas_wind` moves gas only, so a Wind field in a domain without gas gives `wind_without_gas`
  and liquids feel no wind; wind acting on liquids is not decided.
- OpenVDB 12.1.1, the newest on Conan Center on 2026-11-14, is older than the 13.0 Blender 5.3 bundles; the
  witness tests (W1, W8, W10) are what shows Blender 5.3 reads its files.
- The Conan remote: render output's library pins (and so the OpenUSD resources this package ships) depend on
  `conancenter` pointing at `https://center2.conan.io`, since the old `center.conan.io` is frozen; render
  output's library tasks check and switch it. For the owner to confirm.
