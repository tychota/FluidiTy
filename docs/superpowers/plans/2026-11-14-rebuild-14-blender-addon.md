# Rebuild Blender Add-on Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Blender add-on of the rebuild: an artist describes a scene on a domain object, the add-on
writes one scene file, asks the native program `fluidity` whether it can run and what it costs, runs it in a
worker thread without freezing Blender, and shows each listed product file as an ordinary Blender object that
still plays back and renders with the add-on disabled. Also the witness tests that decide, in Blender 5.3, between
USD and Alembic for points, surface and solid motion, and the commit that writes their results into render
output's format table.

**Amended 2026-11-14** for owner answers A16 (Blender 5.3 is the one supported version), A20 (wind comes from
Blender's own Wind force field) and A22 (`link`, `install`, `support` and the other ordinary words are allowed
outside their fluid-jargon sense), and later on 2026-11-14 for A17, A25 and A27 (no `set_viscosity_to_zero`
repair; the budget table `ViscousEffectOfLiquid`). Each changed or added task carries an **Amended** line under its
title.

**Amended 2026-11-15** for owner answers A40, A47 to A50, A53, A57 to A60, A63 and A64 (`plan-decisions.md`, "Owner
answers A40-A64 applied", items 1 to 10; the design's decisions 20 to 31): no grid-only depth properties (moved to
the later group "Deep liquid without particles (later; needs review)", Task 53); artist cell size regions and map
(Tasks 6, 7, 8, 8a, 26, 27, 32); the memory budget without `memory_share_of_total` and the measured cost lines
(Tasks 6, 7, 12, 17, 26, 32); viscosity advice fields (Tasks 12, 32); guided free solids (Tasks 6, 7, 8, 13, 26, 27,
32); the liquid force line (Tasks 12, 17, 19, 32); slip length (Tasks 6, 7, 8, 26, 32); wind on spray and foam and
the Flow note (D13, Tasks 2, 7, 13a, 27, 29); Blender 5.3 and later and the CUDA runtime (Tasks 15, 19, 30, 32, 48);
Resume in another mode (Tasks 14, 18, 31, 32).

**Architecture:** Folder `blender/fluidity/` is the Python package `fluidity`, installed as a Blender extension.
Its pure package `logic/` never imports `bpy` and is tested with plain pytest; the files beside it are the thin
Blender layer (property groups, scene reader, operators, panels, the link routes, the timer). One worker
thread (`RunSupervisor`) owns every native process and reads the run's `events.jsonl`; only the main thread
touches `bpy`. Witness tools live in `tools/blender/witness/` (not packaged) and one native test program,
`fluidity_write_witness_files`, writes every candidate format of the fixed witness scene. Order of work: rules
and paper; pure logic (errors, tables, scene file, native facts, journal, run state, supervisor, link plan);
the scene file against the real program; the Blender layer; the witness tests and the format table; the link
routes for the formats that survived; packaging; whole-scene runs; the artist's journey; comment pass.

**Tech Stack:** Python 3.13 (Blender 5.3.0 bundles 3.13.13; the repository's uv environment is 3.13), the Python
standard library only inside the add-on, pytest 8.3.5 (already in the `dev` group), Blender 5.3 (the one supported
version, A16), the `fluidity` executable and the `fluidity_interchange` writers already on the branch, Typst with CeTZ
for the paper section, C++23 for the one native test program.

**Spec:** `docs/superpowers/specs/modules/14-blender-addon-design.md` ("the design"; in the `main` checkout,
never committed; its section 3 is the contract, with the renames below) and the programme design
`docs/design/programme.md` on `rebuild` (sections 6, 7, 8, 8a). Also in the `main` checkout:
`modules/owner-answers.md` (A9, A10, A12, A14, A16, A20, A22; A40, A47 to A50, A53, A57 to A60, A63, A64), `modules/plan-decisions.md` (it wins over every design),
`plans/2026-11-14-rebuild-12-io-and-cli.md` (commands, events, exit codes, run folder files, the small
reference scene files) and `plans/2026-11-14-rebuild-13-surface-foam-and-spray.md` (products, file names,
channel table, format table, `WitnessRecord`).

## Plan decisions

These decide what the design left open or what the glossary forbids. Every task follows them.

**D1. Names that differ from the design** (amended 2026-11-14 for A22). The owner ruled that `link`, `core`,
`support`, `store`, `stored`, `install`, `accept`, `compatible`, `node`, `system`, `parent`, `children`, `head`,
`tail`, `lower`, `upper`, `coarse`, `reserve`, `resolution`, `epsilon` and `storage` are banned only in their
fluid-jargon sense (the sense their glossary row replaces); ordinary English use is allowed, and the rules check
prints a note, not a failure, when it meets one. Blender itself says "link" for joining data to objects and
"install" for adding an extension, so the add-on uses Blender's words: the design's `link` names
(`output_links.py`, `LinkAction`, `LinkStep`, `LinkPlan`, `build_link_plan`, `LinkRoute`, `LINK_ROUTES`,
`apply_link_plan`, `FORMAT_NOT_LINKABLE`, `LINK_FAILED`, `fluidity.link_outputs`, `fluidity.relink_run`,
`link_outputs_when_ready`), `node_assets.py` (it holds the three starter assets, all node groups) and
"install" are kept as the design wrote them. "Copy Support Report" (`fluidity.copy_support_report`) is kept too:
an artist who needs help looks for the report they send to whoever supports them, and Blender's own menus use
"support" in that everyday sense. The renames that remain:

| Design name | Plan name | Reason |
|---|---|---|
| package `blender/fluidity/core/` (so `core/output_plan.py`) | `blender/fluidity/logic/` (so `logic/output_plan.py`) | Says what the package holds: the logic that runs without Blender |
| `LinkStep.stored_path` | `LinkStep.path_in_blend` | Says where the path lives: as written in the `.blend`, `//`-relative when it can be |
| `Capabilities.capability_reasons` `(code, support, message)` | `(code, level, message)`, level = `ToName(step::CapabilityLevel)` | The value is the step module's `CapabilityLevel` (renamed there in the jargon sense; that rename stays, A22) |
| `RunState.INCOMPATIBLE = "incompatible"` | `RunState.NEEDS_NEW_RUN = "needs_new_run"` | Says what the artist must do, which "incompatible" does not |
| `RunFolderFacts.ignored_tail_bytes` | `ignored_end_bytes` | The name I/O's report prints (I/O ruling) |
| `OutputChannelFact.storage` | `value_type` | The JSON key the branch prints (render output's `ChannelValueType`) |
| `to_platform_name(system, machine)` | `to_platform_name(os_name, machine)` | Says what the value is: the operating system's name from `platform.system()` |
| `WitnessOutcome` | `WitnessCaseResult` | Role suffix rule (`Outcome` is not one of `Inputs`, `Settings`, `Result`) |
| "acceptance journey" | "artist's journey", `tools/blender/run_artist_journey.py` | Says whose path through the add-on the test walks |

Blender's and Python's own spellings (`obj.parent`, `collection.objects.link(...)`, `name.lower()`,
`platform.system()`, `blender --command extension install-file`, `bpy.types.GeometryNodeTree`) are not ours:
the rules check already skips attribute names, keyword-argument names and string literals in Python
(`tools/check_rules.py`, `_split_python`). Ordinary words such as "parent", "lower" or "install" may appear in
comments and labels in their everyday sense; the rules check prints a note for each, which the reviewer reads.

**D2. Test levels.** Every task states which it uses:
- (a) **plain pytest**, `tests/addon/logic/` (no Blender, no native program: CTest `addon_logic`, label `cpu`)
  and `tests/addon/with_fluidity/` (the built `fluidity` program, no Blender: CTest `addon_with_fluidity`, label
  `cpu`, `TIMEOUT 900`). Both run in the gate's `cpu` step on this Mac.
- (b) **background Blender**: `tests/addon/in_blender/test_*.py`, each run by
  `uv run python tools/blender/run_blender_tests.py --blender <executable> [--fluidity <executable>]`, which
  starts `blender --background --factory-startup --python-exit-code 1 --python <file> -- <arguments>` per file.
  Timers do not fire in background mode, so tests call `ui_timer.tick()`. Not part of the gate (the container
  has no Blender).
- (c) **interactive session** (a window), recorded by hand in `docs/blender-runs.md`.

For (b) and (c), the commit message carries one line naming the Blender build that ran, for example
`Blender 5.3.0 (02f0899d78e3): 7 passed.` A test group that did not run gets the line
`Blender 5.3: written, not run in Blender (<reason>).` Never write "passed" for anything that did not run, and
never name a build that did not run the tests. Each run is also a row of `docs/blender-runs.md` (date, commit,
Blender version and build hash, platform, what ran, result).

**Supported Blender** (A16, amended 2026-11-14; **amended 2026-11-15 (A48):** later versions allowed): Blender 5.3 is
the one tested version. The manifest says `blender_version_min = "5.3.0"` with no maximum; Blender 5.4.0 or later
loads the add-on and the domain panel shows one note, "FluidiTy was tested with Blender 5.3 only." (Tasks 19, 30,
32). There is no second version in any test matrix, witness record or receipt.

**What this Mac has** (checked 2026-11-14 for this amendment): `/Applications/Blender.app`, Blender 5.3.0, build hash
`02f0899d78e3`, branch `blender-v5.3-release`, built 2026-11-10, Python 3.13.13. Blender 5.2.3 is no longer installed.
`/opt/homebrew/bin/blender` is a Homebrew wrapper that runs `/Applications/Blender.app/Contents/MacOS/Blender` (its
Caskroom folder is still named `5.2.3`, but the program it starts reports 5.3.0); commands in this plan name
`/Applications/Blender.app/Contents/MacOS/Blender`, and every task reads the version from `--version`, never from a
path. Background-Blender tests are runnable here. Plain pytest needs `uv run pytest` (the system Python has no pytest).

**D3. Blender API facts, checked on 5.2.3 and to be re-checked on 5.3 before use** (amended 2026-11-14, A16).
The design marks these "to verify". They were measured in Blender 5.2.3 when this plan was written; a throwaway
probe run in Blender 5.3.0 while amending it found every one of them still true, but that probe is not part
of the branch. So the first task that uses a fact starts with a step that re-checks it in Blender 5.3 with
`tools/blender/probe_blender_api.py` (created in Task 27) and records the result in the commit body and in
`docs/blender-runs.md`. If a fact is false in 5.3, the task stops and reports, with the probe's output, before
writing code that relies on it. Probe check names are in brackets; the tasks that use each are in parentheses.
- [`cache_file_open`] There is **no `bpy.data.cache_files.load`** (`BlendDataCacheFiles` has only `tag`). A cache
  file is created with `bpy.ops.cachefile.open(filepath=...)` (works in background mode; returns `{'FINISHED'}`
  and adds one `CacheFile`); the new one is found by comparing `bpy.data.cache_files` before and after. (Tasks
  36, 42.)
- [`volume_new`] There is **no `bpy.data.volumes.load`**: `bpy.data.volumes.new(name)`, then `filepath`,
  `is_sequence`, `frame_start`, `frame_duration`. (Tasks 36, 42.)
- [`cache_file_fields`] `CacheFile` has `filepath`, `is_sequence`, `object_paths`, `velocity_name`,
  `velocity_unit` (`SECOND`, `FRAME`), `forward_axis`, `up_axis`, `scale`. `MeshSequenceCacheModifier` has
  `cache_file`, `object_path`, `read_data` (`VERT`, `POLY`, `UV`, `COLOR`, `ATTRIBUTES`), `velocity_scale`; it
  can be added to a point cloud object. (Tasks 36, 42.) [`transform_cache`] `TransformCacheConstraint` has
  `cache_file`, `object_path`. (Tasks 39, 43.)
- [`volume_grids`] `VolumeGrid` has `name`, `data_type`, `channels`, `matrix_object`, `load()`; `VolumeGrids` has
  `frame`, `frame_filepath`, `error_message`. [`sample_grid`] `bpy.types.GeometryNodeSampleGrid` exists. (Tasks 36,
  40, 42.)
- [`importers`] `bpy.ops.wm.alembic_import`, `bpy.ops.wm.usd_import`, `bpy.ops.object.volume_import` exist.
  (Task 42, only if a route uses an importer.) [`extension_path_user`] `bpy.utils.extension_path_user` exists.
  (Task 30.) [`temp_override`] `bpy.context.temp_override` exists. (Task 42.)
- [`extension_commands`] `blender --command extension build [--source-dir] [--output-dir] [--split-platforms]`,
  `extension validate [SOURCE_PATH]`, `extension install-file -r <repo> [-e] FILE`. `--offline-mode` is a global
  Blender option (put before `--command`), not an option of `extension build`. (Tasks 48, 51.)
- [`bake_to_keys`] Blender's "Bake to Keys" operator for the `motion_not_keys` repair is `bpy.ops.nla.bake`.
  (Task 27.)
- New in the amendment, seen in the 5.3.0 probe and re-checked by Task 27: [`action_curves`] actions are layered:
  `Action.fcurves` does not exist; an object's curves are read through
  `bpy_extras.anim_utils.action_get_channelbag_for_slot(animation_data.action, animation_data.action_slot)`, and a keyed
  field strength is the curve with data path `field.strength`. [`force_fields`]
  `bpy.ops.object.effector_add(type='WIND')` makes an empty with `field.type == 'WIND'`; `FieldSettings` has `type`,
  `shape`, `strength`, `flow`, `noise`, `seed`, `falloff_type`, `falloff_power`, `use_min_distance`, `use_max_distance`,
  `distance_min`, `distance_max`, `z_direction`, `apply_to_location`, `use_absorption`, `wind_factor`; a new Wind field
  has shape `PLANE`, strength 1.0, flow 1.0, noise 0, falloff `SPHERE` with power 0, no minimum or maximum distance,
  `z_direction` `BOTH`; `EffectorWeights.collection` exists. The 5.3.0 field types are `NONE`, `BOID`, `CHARGE`,
  `GUIDE`, `DRAG`, `FLUID_FLOW`, `FORCE`, `HARMONIC`, `LENNARDJ`, `MAGNET`, `TEXTURE`, `TURBULENCE`, `VORTEX`, `WIND`.
  (Tasks 13a, 27, 27a, 29.)

The link routes (design 5.7 step 3) therefore use one small operator, `bpy.ops.cachefile.open`, inside
`apply_link_plan`; it runs on the main thread and reads no frame data.

**D4. The scene file's field names come from the branch's schema.** The design's 4.1 outline is the shape; the
names are those of `schemas/simulate/v2/scene_file.fbs` (I/O plan Tasks 13 to 29: for example start regions
carry `center`, sources carry `position` and `position_spread`, gas sources are top-level `gas_sources` with a
`box`, wind is `wind` (A47; was `gas_wind`), `random_seed` and `run_on` are top-level, step settings are the `step` table). The
description dataclasses keep the design's Python names; `SCENE_FIELDS.scene_file_path` and `write_scene_file`
map them to the schema. Where the schema cannot carry a description field (a turned source box when
`SceneSource` has no orientation, for example), the quick check reports the case with its own code (here
`source_turned`, repair Clear Rotation) and the hand-over lists it. A test (Task 7) reads the schema and fails
on any `scene_file_path` it does not define.

**D5. The four reference scenes are I/O's small scene files.** `presets.py`'s builders produce the physical
content of `native/fluidity/io/testdata/scenes/small_wave.json`, `small_syrup.json`, `small_fire.json`,
`small_snow.json` (the step plan's small sizes: wave one size level at 2.5 cm, syrup 5 mm, fire 0.1 m, snow
2.5 cm), plus the output switches of design 5.2 (surface and foam for the wave, surface for the syrup, gas
volume for the fire, points for the snow). Proof: `fluidity check` gives the same `run_fingerprint` for the
add-on's file and for I/O's file (Task 25), because names and outputs are outside the fingerprint.

**D6. Types the design left undefined.**
- `RunKind` and `StartRequest` live in `logic/run_state.py` (not `supervisor.py`), so `native_calls.py` can
  import them without a cycle.
- `Request` (in `run_state.py`): `RequestKind` = `CHECK_FINISHED`, `SCENE_EDITED`, `START_TAKEN`,
  `STOP_AFTER_FRAME_SENT`, `CANCEL_SENT`, `PROCESS_ENDED`, `PROCESS_GROUP_ENDED`, `NEW_RUN_TAKEN`,
  `CACHE_COMPARED`; `@dataclass(frozen=True) class Request: kind; can_run: bool = False;
  has_blocking_quick_finding: bool = False; run_kind: RunKind | None = None; exit_code: int | None = None;
  saw_end_event: bool = False; simulation_match: str = ""`.
- `MemoryUse` (in `run_state.py`): `cpu_bytes_in_use`, `cpu_largest_bytes`, `gpu_bytes_in_use`,
  `gpu_largest_bytes`, `limit_raise_count` (ints, from `frame_done`).
- `OutputRow` (in `logic/output_plan.py`): frozen copy of one `FluidityOutputRow` (`output_id`, `object_name`,
  `product`, `format`, `material_id`, `run_id`, `has_object`, `file_pattern`, `first_frame`,
  `last_listed_frame`, `run_fingerprint`, `output_fingerprint`, `contract_version`). `read_output_row`
  returns it; `build_link_plan` takes `rows: tuple[OutputRow, ...]`.
- `LINK_ROUTE_KEYS: frozenset[tuple[str, str]]` (in `logic/output_plan.py`): the (geometry kind, format)
  pairs that have a route. `output_links.LINK_ROUTES` must have exactly these keys (Blender test, Task 44).
- `RunSupervisor.request_link_plan(run_id, run_folder, rows, project_folder) -> int` (ticket):
  `build_link_plan` runs on the worker thread (design 3.8).
- `SupervisorSnapshot` (frozen): `statuses: Mapping[str, RunStatus]`; `checks`, `particles`, `link_plans`:
  `Mapping[int, <result> | AddonFailure]` by ticket; `reports: Mapping[str, RunFolderFacts | AddonFailure]` by
  `run_id`; `capabilities: Capabilities | AddonFailure | None`; `recent_events: Mapping[str, tuple[str, ...]]`
  (the last 200 journal lines of each run, for Copy Support Report); `change_count: int`.
- `scene_reader.read_scene(domain) -> SceneReading` with `description: SceneDescription` and
  `findings: tuple[Finding, ...]` (what only Blender can know: unsaved project, a role without a domain, an
  object in two domains, a sheared or mirrored matrix, motion from a parent, constraint, driver or modifier,
  and the force field findings of D13). `read_scene_in_slices(domain)` is the generator form of design section 6.
- `presets.compute_preset_after_edit(current_preset: str, property_path: str, applying_preset: bool) -> str`.
- `logic/status_text.py`: `build_status_line(status: RunStatus, now_utc: float) -> str`.
- `logic/output_states.py`: `OutputState` (`READY`, `STALE_REBUILDABLE`, `STALE_NEEDS_NEW_RUN`,
  `MISSING_FILES`, `NOT_BUILT_IN_FRAMES`, `NOT_REQUESTED`), `build_output_states(...)`,
  `describe_scene_change(comparison) -> str`.
- `logic/run_folder.py`: `build_run_folder(cache_root: Path, domain_name: str, run_number: int) -> Path`,
  `find_next_run_number(cache_root, domain_name, recorded_run_ids) -> int`,
  `to_path_in_blend(path: Path, project_folder: Path | None, path_module=os.path) -> str`,
  `to_absolute_path(path_in_blend: str, project_folder: Path | None) -> Path`.

**D7. Ids and keys.** `FluidityDomain.next_id` (int, starts at 1; not in design 3.9) gives every participant,
substance and ingredient id; ids are never reused in a domain. `domain_id` is `uuid.uuid4().hex` at Add Domain.
`run_id` is the folder name `run_<nnn>` (three digits, then more as needed), label "Run 003"; the next number is
one above the largest recorded or on disk. For a snow or springy-solid substance with `regions_form_one_body`,
the start region with the lowest id writes `same_material_as = 0` and every other region of that substance
writes that lowest id. Orientations are written with `w >= 0` (a quaternion and its negative are one turn; one
sign keeps the digest stable). A matrix is mirrored when its 3x3 determinant is negative and sheared when two
of its normalised columns have a dot product above `1e-9` in magnitude. Key times are
`(frame - first_frame) / frames_per_second`.

**D8. Addon finding codes** (module `"addon"`, found by `"quick_check"`; stable, never translated): `project_not_saved`,
`role_without_domain`, `object_in_two_domains`, `shape_not_allowed`, `motion_not_allowed`, `substance_missing`,
`substance_wrong_kind`, `more_than_one_gas`, `shares_do_not_sum_to_one` (tolerance `1e-9`), `ingredient_name_empty`,
`ingredient_names_collide`, `outside_domain`, `gas_source_turned`, `source_turned` (only if D4 applies),
`size_not_positive`, `rate_not_positive`, `source_window_outside_frames`, `matrix_sheared_or_mirrored`,
`motion_not_keys`, `too_many_plates`, `too_many_free_solids`, `too_many_ingredients`, `too_many_size_levels`,
`motion_samples_out_of_range`, `surface_spacing_costly` (spacing under a quarter of the smallest cell width, severity
`COSTS_MORE`), **amended 2026-11-15 (A59):** `guide_object_missing` (`BLOCKS_RUN`: a free solid's guide is Rail or
Hinge and it has no guide object), **(A63):** `cell_size_region_rotated` (found by `"scene_reader"`, `NOTE`,
message "<object> is rotated; the axis-aligned box around it is used."), and the force field codes of D13 (`wind_falloff_not_honoured`, `wind_noise_not_honoured`,
`wind_flow_not_honoured`, `wind_shape_not_honoured`, `wind_without_gas`, `force_field_not_honoured`,
`force_field_ignored`; these are found by `"scene_reader"`, since only Blender knows the field objects). Ingredient
names are normalised exactly as the materials module's `BuildMaterialTable` does on the branch (read it; port the rule;
a comment names the C++ function).

**D9. Repairs** (`REPAIR_TABLE`, design 3.4; **amended 2026-11-14 (A27):** `set_viscosity_to_zero` and its
entry are removed, the step has 11 names; the sand module adds it back with `kViscousLiquidWithSand`, and its
repair with it). One generic operator `fluidity.apply_field_value(property_path, participant_id, value, scale)`
changes one property (a repair never changes anything by itself: the artist presses it). Entries:
`use_smaller_cells` -> "Use <limit> m" on `smallest_cell_width` when `unit == "m"` and `limit > 0`, else `None`;
`use_bigger_cells` -> "Double the cell width" (`scale = 2`); `run_on_cpu` -> "Run on CPU" (`run_on = CPU`);
`make_solid_fixed` -> the participant's `motion = FIXED`; `align_solid_with_axes` -> Blender's
`object.rotation_clear` on the participant, label "Clear Rotation"; `make_solid_thicker`,
`separate_the_objects` -> "Select <object name>" (`fluidity.select_participant`); `make_domain_smaller` ->
"Select domain"; `start_new_run` -> `fluidity.new_run`; `use_one_surface_tension`, `use_one_cell_size_there`,
`use_fewer_ingredients`, `put_all_liquids_in_one_set`, `use_fewer_solids`, `free_disk_space` -> `None`. The
step's `kNone` is never a repair (drop `"none"` if the branch prints it). The 11 step names are those of
`ToName(SuggestedChange)` on the branch (`step::kSuggestedChangeCount` 12 with `kNone`, A27); if that list differs
from the 11 above, stop and report.

**D10. Python rules** (written into `AGENTS.md` by Task 1): four-space indent, 120 columns, type hints on every
public function, frozen dataclasses for values that cross a module, functions of `logic/` return
`Value | AddonFailure` and raise nothing across a module boundary, operators turn a failure into
`self.report({'ERROR'}, to_message(...) + detail)`. Each module starts with a docstring holding the file-header
labels `Purpose:`, `Idea:`, `Owns:`, `Uses:`, `Used by:`, `Read:`. Function names start with an allowed verb or
`to_`; also allowed (design decision 10 and Blender's own): `register`, `unregister`, `draw`, `poll`,
`execute`, `invoke`, `tick`, `close`, `request_*`, `follow_run`, `cancel_check`, `mark_scene_changed`,
`list_witness_cases`, `run_witness_case`. `logic/` imports only the standard library and other `logic/`
modules, by relative import. `blender/fluidity/__init__.py` imports Blender-layer modules only inside
`register()` and `unregister()`, so `import fluidity.logic.<module>` works without Blender.

**D11. Process control.** A run starts in its own process group (POSIX `start_new_session=True`; Windows
`creationflags=CREATE_NEW_PROCESS_GROUP`), stdin a pipe, stdout and stderr `DEVNULL`. Ending the group: POSIX
`os.killpg(pgid, signal.SIGKILL)`; Windows `Popen.kill()` (the CLI starts no child, so the process is the whole
tree). One-shot commands: stdout a pipe read to the end, stderr `DEVNULL`, a timeout of 60 s except
`capabilities` (5 s). `compute_scene_digest` is SHA-256 of the written bytes, in hex.

**D12. Witness records, one Blender version** (amended 2026-11-14, A16). Render output's `WitnessRecord` holds
one result, `blender_5_3` (render output plan, Task 43, amended). A record whose case ran in Blender 5.3 gets
that result (`kPassed` or `kFailed`) and evidence `"<test>; Blender <version> <build hash>; files
<files_digest>"`; a `kDocumented` record that is measured keeps its manual quote after `" | "`. A record whose
case was not measured keeps its earlier value (`kNotRun` with empty evidence, or `kDocumented` with its quote),
never `kPassed` or `kFailed`. The rule (design 7.2, with A16): a USD candidate that passes every witness of its
kind in Blender 5.3 becomes the chosen format of that kind; one that fails any witness is removed with its
writer and its link route in the commit that records the failure.

**D13. Wind from Blender's force fields** (amended 2026-11-14, A20; **amended 2026-11-15 (A47, A50):** the scene
field is `wind` (was `gas_wind`), keys are written whenever a Wind field exists, `build_wind_keys` has no
`has_gas`, `wind_without_gas` has two texts, and `wind_flow_not_honoured` fires only for `flow != 1.0`; where the
text below said `gas_wind` it now says `wind`). This replaces design 5.1 item 5 and the
design 3.9 domain property `wind_acceleration` (keyable m/s2): the domain has no wind property. The add-on reads
Blender's own Wind force field objects, as the gas branch did
(`git show origin/codex/m7-low-mach-gas:blender/fluidity/gas_authoring.py`, lines 35-37 and 63-67: a domain
pointer to one object whose `field.type` must be `'WIND'`; per frame, direction = the object's world rotation
applied to its local +Z axis, acceleration = direction times `field.strength`, with no other factor; the
`_solver_vector` axis swap there was the old Y-up solver's and does not apply to the Z-up rebuild).
- **Which objects.** The domain gets one property, `force_field_collection` (pointer to a Blender collection,
  empty by default). The fields read are the objects of that collection, or of the whole scene when it is empty,
  whose `field` is not `None` and whose `field.type` is not `'NONE'`. This is Blender's own rule for a physics
  domain's effector weights (`EffectorWeights.collection`: empty means every force field in the scene), so a
  scene set up for Blender's fluids behaves the same way here.
- **The conversion, stated once.** `WIND_ACCELERATION_PER_STRENGTH = 1.0` (m/s2 per unit of Blender's Wind
  Strength), in `logic/force_fields.py`. A Wind field gives the acceleration
  `WIND_ACCELERATION_PER_STRENGTH * strength * direction`, where `direction` is the field object's local +Z axis
  in world axes at that key: the scene reader gets it from Blender's own `mathutils` (the keyed rotation as a
  `Quaternion`, times `Vector((0, 0, 1))`), so the add-on adds no rotation code of its own, and `logic/` receives
  a unit vector. Several Wind fields add (each is a uniform acceleration). The factor is
  the gas branch's (strength used directly as m/s2). It appears in the tooltip of `force_field_collection` and of
  the `wind[].acceleration` scene field, in the method paper (Task 2) and in the test
  `test_wind_strength_to_acceleration` (Task 13a).
- **Honoured** (written to the scene file's `wind` keys, `{time, acceleration}` in m/s2, world axes):
  strength; direction from the object's orientation; the animation of both. Keys are read without changing the
  frame, the same way as driven motion (design 5.1 item 6): the object's rotation curves and the curve with data
  path `field.strength` are evaluated with `FCurve.evaluate` at every frame of the range when any of them has
  keys, and one key at time 0 otherwise. Key time `(frame - first_frame) / frames_per_second`. A Wind object
  whose rotation comes from an object above it, a constraint, a driver or a modifier gets `motion_not_keys`
  (repair "Bake to Keys") like any participant.
- **Reported, because the solver applies one uniform acceleration** (module `"addon"`, found by
  `"scene_reader"`, severity `HAS_KNOWN_LIMIT`: the run goes ahead with the uniform part only; `object_name` is
  the field object; no repair): `wind_falloff_not_honoured` when `falloff_power != 0`, `use_min_distance` or
  `use_max_distance` is set (the falloff shape `falloff_type` matters only then); `wind_noise_not_honoured` when
  `noise != 0`; `wind_shape_not_honoured` when `shape` is not `PLANE` or `POINT`, or `z_direction` is not
  `BOTH` (Blender then blows only on one side of the field); `wind_flow_not_honoured` (severity `NOTE`) only when
  `flow != 1.0` (A50; Blender's flow turns the force into an air speed that particles are dragged towards; FluidiTy
  applies the acceleration itself; a new Wind field in Blender 5.3 has flow 1.0 and gives no finding).
- **Wind without gas** (amended 2026-11-15, A47). The scene file's `wind` acts on gas in the solver and on spray
  and foam in render output (`render_output::RenderSettings::wind`), never on the liquid body; keys are written
  whenever a Wind field exists. A Wind field in a domain without a gas substance gives `wind_without_gas`
  (severity `NOTE`): "The Wind field moves spray and foam only; the liquid is not pushed by wind." when
  `outputs.foam_and_spray` or `outputs.foam_volume` is on, otherwise "The Wind field has no effect in this
  scene." With a gas substance there is no such note.
- **Every other field type** has one row in `FORCE_FIELD_TYPE_TABLE` (Task 13a, checked against Blender 5.3 in
  Task 27a): `HONOURED` (only `WIND`), `NOT_HONOURED` (a Blender fluid would feel it and FluidiTy does not:
  finding `force_field_not_honoured`, severity `HAS_KNOWN_LIMIT`, the message names the type), or `IGNORED`
  (the type acts only on Blender particle systems, hair or boids: finding `force_field_ignored`, severity
  `NOTE`). Nothing is dropped without a finding.

## Global Constraints

Foundation's list, unchanged:

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Old code is read with `git show main:<path>` from the worktree. Nothing is checked out wholesale.
- Commit rule (spec section 6): one thing per commit; strict CPU build and all tests pass; CUDA target compiles
  in the container; clang-format 23.1 clean; clang-tidy 23.1 clean on host sources; every touched file passes
  the `AGENTS.md` checklist; new public names listed in the commit message with a one-line reason; tests arrive
  with the code. A commit must pass every part of the gate that exists at that commit.
- History on `rebuild` may be rewritten. A defect found later is fixed in the commit that introduced it.
- Commit messages carry no AI trailer (no `Co-Authored-By`, no `Assisted-by`, no session link); the owner is the
  author. After committing, write the commit's note with `uv run python tools/add_session_note.py --session
  <session link> --models <implementer's model id>,claude-fable-5-2`
  (a git note in `refs/notes/claude-session`; owner answer A55, 2026-11-15).
- Device code is "compiled, not run" until `docs/gpu-runs.md` records the owner's run. The owner runs on
  hardware once, after all modules are written. Never claim otherwise.
- Host C++23, exceptions and RTTI off in the library; device and shared headers C++20 subset; a shared header
  never includes `<expected>`, `<string>` or anything that allocates. No Abseil.
- Device numerical flags: `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. Host: `-ffp-contract=off`,
  `/fp:precise`. No fast-math.
- Z-up, right-handed. Gravity points along `{0, 0, -1}`.
- Naming: no `3D` suffix, no `Cuda`/`Gpu`/`Device`/`host` in public names, no bare `bool` parameters, no
  abbreviations (MAC, PIC, APIC, P2G, G2P, EOS), no milestone codes. Function verbs: `Compute`, `Build`, `Check`,
  `Find`, `Apply`, `Solve`, `Read`, `Write`; anything else is listed for the owner's review.
- 120 columns, two-space indent, `//` comments, braces on every control-flow body.
- Tests are named `TEST(FunctionOrType, Situation_Outcome)`, one behaviour each, in `*_test.cc` beside the code.
- Deletion is the default: add a dependency, file or tool only when a task here names it.
- **Amended 2026-11-14 (Foundation as built):** the native pieces of this plan (Task 34's program and its C++
  tests) follow `AGENTS.md` "Structure" and `plan-decisions.md` "Foundation after the final review": every C++
  target goes through the helpers of `cmake/fluidity_targets.cmake` (a program: `add_executable` followed at once by
  `fluidity_apply_compile_flags(<target> EXCEPTIONS OFF)`, as I/O's `fluidity_cli`; a test: `fluidity_add_test(...
  LABEL cpu [LINKS ...])`), every source file is listed in a CMake file, and C++ tests use the helpers of
  `fluidity_base_test_support` instead of their own. The add-on adds no GPU code.
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the CPU build in the debug and the
  release preset, and CUDA compiled in the container, whose `cuda` step fails for a tracked `.cc` or `.cu` file that
  no build compiles. clang-format and clang-tidy are the 23.1 that `pyproject.toml` pins.

The module's own:

- **Design section 3 is the contract**, with the renames of D1 and the types of D6. Python tests are named
  `test_<what>_<outcome>` (the design's names where it gives them), one behaviour each.
- **The add-on computes no physics** and contains no rotation matrix: positions, velocities, poses and grid
  origins pass through unchanged (design 5.11). The one direction it needs, a Wind field's axis (D13), comes
  from Blender's `mathutils`. It never writes `--run-on` (A12): `run_on` is in the scene file.
- **Constants** (design values): `PROTOCOL = 2`, `SCENE_SCHEMA = 2`, coordinate text
  `right_handed_z_up_metres`; journal lines at most 16 KiB, at most 256 KiB read per worker turn, a 50 ms worker
  turn; timer 0.1 s while anything is active, 1 s otherwise, target 4 ms per tick; scene reader slices of at
  most 8 ms; quick check target 100 ms on the reference scenes; `stop_offer_seconds` default 10;
  `capabilities` answers within 200 ms; motion samples per frame 1..64; ids JSON integers below 2^53;
  `cache_root` default `//fluidity_cache`; run folder `<cache_root>/<domain name>/run_<nnn>/` holding the
  add-on's `scene.json`.
- **Only the main thread touches `bpy`.** A timer or handler keeps no Blender object between calls. The worker
  thread is the only code that blocks. Registration is idempotent (one timer, one set of handlers, one
  thread). Nothing is killed on close (design 5.5 step 6).
- **Witness equality** (design 7.2): counts, names, domains, data types and int values exact; a float within
  `4 * 2^-24` of the expected value's magnitude and, for positions, within `1e-6 m`; a rotation within
  `1e-6 rad`; motion blur growth within 10 %. Frames visited 1, 2, 10, 2, 1.
- **Supported Blender:** 5.3 tested (A16), later versions load with one note (A48; manifest
  `blender_version_min = "5.3.0"`, no maximum); platforms
  `windows-x64`, `linux-x64`, `macos-arm64` (CPU only). Blender tests run in Blender 5.3; see D2 for what is
  recorded when a group did not run.
- **Glossary rows** arrive with the code that needs them: `participant`, `role` (Task 5); `substance`, `quick
  check`, `full check` (Tasks 9, 13); `journal` (`events.jsonl` as the add-on reads it, Task 16); `supervisor`
  (Task 21); `run record`, `output registry` (Task 24); `link` in Blender's sense (Task 44); `force field`
  (Task 13a); `starter asset` (Task 47).

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, already committed: every
module except sand, each with its comment pass done, in particular I/O and CLI (the `fluidity` program with
`capabilities`, `check`, `run`, `resume`, `report`, `read-particle`; `tests/cli/`; the small reference scene
files in `native/fluidity/io/testdata/scenes/`) and render output (the four writers, `ReadFormatTable()`,
`ReadOutputChannels()`, Task 47's CLI frame worker).

Prove it: `uv run python tools/check_commit.py` prints `PASS` on every line;
`out/build/macos-release-cpu/native/fluidity/cli/fluidity capabilities` (or wherever `fluidity_cli` lands)
prints one JSON line with `"protocol": 2`, `right_handed_z_up_metres` and a format table; and
`git grep -ln "ReadFormatTable\|WriteParticle\|CompareWithRunFolder" -- native` lists files of `render_output/`,
`cli/` and `io/`. Run `/Applications/Blender.app/Contents/MacOS/Blender --version` and note the version and build
hash: it must be 5.3.x (A16). If the gate or the program fails, or Blender is not 5.3, stop and ask the owner.

**Amended 2026-11-14 (Foundation as built):** the gate prints seven `PASS` lines: `format`, `rules`, `cpu` debug,
`cpu` release, `tidy`, `paper`, `cuda`. Foundation's `base/test_support.h` (library `fluidity_base_test_support`)
and the target helpers of `cmake/fluidity_targets.cmake` are what Task 34's C++ uses; read `AGENTS.md` "Structure"
and "Targets" before it.

## Review Focus

1. **A pipe that fills blocks the solver.** The run's stdout is discarded and events come from the journal, but
   a regression that keeps `stdout=PIPE` passes every short test. Pinned:
   `test_supervisor_never_blocks_the_solver` in Task 22 (10,000 events, `read_snapshot` never called).
2. **A journal line cut at the 256 KiB read limit or mid-write** is parsed as a broken event or skipped. Pinned:
   `test_read_events_partial_line_at_byte_limit` in Task 16.
3. **A half-applied link plan:** one route fails after others repointed their objects. Pinned:
   `test_failed_plan_changes_nothing` in Task 44 (truncated file, wrong `run_fingerprint`, missing channel:
   every row and every cache path as before).
4. **A witness that could not measure is recorded as a format failure (or a pass).** Pinned:
   `test_write_witness_records_not_run_where_unmeasured` in Task 35 and the `kNotRun` path of
   `run_witness_case` in Task 36.
5. **Closing Blender or loading a file kills the run.** Pinned: `test_close_stdin_stops_after_frame` (Task 23)
   and `test_file_load_during_run_ends_stopped` (Task 31).

---

### Task 1: Rules check, gate and AGENTS.md cover the add-on

**Amended 2026-11-14 (A22):** ordinary words such as `parent` give a rules-check note, not a failure, so the word
test uses a word that is still banned outright.

**Kind:** mechanical. **Tests:** (a).

**Files:**
- Modify: `tools/check_rules.py`, `tools/check_rules_test.py`, `tools/check_commit.py`, `tools/check_commit_test.py`,
  `AGENTS.md` (new section "Python in the Blender add-on")

**Interfaces:**
- Produces: the gate's `rules` step scans `native`, `tools`, `blender`, `tests` (skip any already scanned); the
  `file-header` rule also applies to `.py` files under `blender/fluidity/` and `tools/blender/` as a module
  docstring holding the six labels; the `paper` step also checks `Read:` lines in those folders.

- [ ] **Step 1: Failing tests:** `test_python_module_without_header_docstring_fails` (`blender/fluidity/x.py` with no
  docstring: one FAIL `file-header`); `test_python_module_with_header_passes`; `test_python_test_file_needs_no_header`
  (`tests/addon/logic/test_x.py`); `test_blender_attribute_words_are_not_ours` (`obj.parent`,
  `bpy.data.objects.link(o)`, `name.lower()`, `"install-file"`: no finding; `emitter = 1`: FAIL `banned-word`);
  `test_ordinary_word_in_python_is_a_note` (`parent = 1` and the comment `# link the outputs`: one note each, naming the
  glossary row's banned sense, and no FAIL, A22); `test_rules_step_scans_addon_folders` (gate with a fake runner: the
  rules command line names `blender` and `tests`); `test_paper_step_checks_python_read_lines`.
- [ ] **Step 2: Run** `uv run pytest tools -q` (fail). **Step 3: Implement.** **Step 4:** write the `AGENTS.md`
  section with D10's rules. **Step 5:** gate. **Commit.** Subject: `Check the add-on's Python with the same rules`.

### Task 2: Method paper, the scene model and the run's three facts

**Amended 2026-11-14 (A20):** the scene model states how a Blender Wind force field becomes the scene file's wind
keys, with the conversion factor, and what is reported instead of honoured.

**Amended 2026-11-15 (A47, A50, A63):** the wind paragraph says the wind moves gas, spray and foam and never the
liquid, and that flow is reported only when it is not 1; one paragraph on the artist's cell size regions and map.

**Kind:** mechanical. **Tests:** (a) (the gate's `paper` step).

Load the `typesetting-with-typst`, `typesetting-documents` and `drawing-with-cetz` skills first.

**Files:**
- Create: `docs/method-paper/sections/blender.typ`, `docs/method-paper/figures/run_states.typ`
- Modify: `docs/method-paper/fluidity-method.typ` (include the section before References)

**Interfaces:**
- Produces labels `<sec-blender>`, `<fig-run-states>`, `<proc-apply-event>`.

- [ ] **Step 1: Text** (design section 8): the scene model (domain, participant with role, shape and motion;
  substance and ingredient; the wave paddle and cutting plate as variables, not types); the quick check and
  the full check; one run with three separate facts (`last_listed_frame`, `saved_state_frame`, `last_try`).
  Wind (D13), one paragraph and one equation: the fields read (the domain's force field collection, or the whole
  scene); `a = 1.0 m/s^2 * strength * axis`, where `axis` is the field object's local +Z in world axes, summed
  over Wind fields, one key per frame when animated; why the factor is 1 (the gas branch used Blender's Strength
  directly as m/s2); what it moves (gas, spray and foam; never the liquid body, A47); the settings reported instead
  of honoured (falloff, noise, flow other than 1 (A50), shapes and sides that are not one uniform direction) and the
  other field types, each with its finding code. Cell size (A63), one paragraph: a region is an object's world
  axis-aligned box with a widest cell width; the map is a top view whose luminance (Rec. 709 weights) asks a size
  level at every height; why 2D (Blender paints images, not 3D fields; depth-specific requests use regions).
- [ ] **Step 2: `<fig-run-states>`**: the states of design 5.4 (with `needs_new_run`) and their arrows, and
  beside it one frame axis with three marks for the example "frames 0 to 37 listed, saved state 30, last try
  38 rejected". Compile, look at the page, fix overlaps.
- [ ] **Step 3: `<proc-apply-event>`**: the transition table of design 5.4 as numbered steps, with the
  invariants of the paragraph below the table. Task 17's `# Step N:` comments follow it.
- [ ] **Step 4:** `typst compile`; gate. **Commit.** Subject: `Describe the scene model and the run in the method
  paper`.

### Task 3: Method paper, outputs, what goes stale and the witness matrix

**Amended 2026-11-14 (A16, A22):** the witness matrix has one mark per cell (Blender 5.3), and outputs are
"linked" in Blender's word.

**Kind:** mechanical. **Tests:** (a).

**Files:**
- Create: `docs/method-paper/figures/what_goes_stale.typ`, `docs/method-paper/figures/witness_matrix.typ`
- Modify: `docs/method-paper/sections/blender.typ`

**Interfaces:**
- Produces `<fig-what-goes-stale>`, `<fig-witness-matrix>`, `<proc-apply-link-plan>`.

- [ ] **Step 1:** text: outputs as open-format files linked with Blender's own readers; the output registry;
  how a format is chosen by witness tests in Blender 5.3 (D12's rule).
- [ ] **Step 2:** `<fig-what-goes-stale>`: scene file, simulation, each output, Blender objects; arrows labelled
  with the edits that make each stale (a viscosity: simulation; the surface spacing: one output). 
  `<fig-witness-matrix>`: kind and candidate rows against the ten properties, one mark per cell (the Blender 5.3
  result), drawn from today's table (`kDocumented`, `kNotRun`).
- [ ] **Step 3:** `<proc-apply-link-plan>`: design 5.7 step 4 (a) to (d) and the undo. Blender manual pages
  are cited by URL copied at writing time.
- [ ] **Step 4:** compile, look, gate. **Commit.** Subject: `Describe outputs and the witness tests in the method
  paper`.

### Task 4: Package skeleton, the add-on error and test wiring

**Amended 2026-11-14 (A16, A22):** the manifest's minimum is Blender 5.3.0, and the error names use `LINK` as the
design does.

**Kind:** mechanical. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/__init__.py`, `blender/fluidity/blender_manifest.toml`,
  `blender/fluidity/logic/__init__.py`, `blender/fluidity/logic/addon_error.py`, `tests/addon/conftest.py`,
  `tests/addon/logic/test_addon_error.py`, `tests/addon/logic/test_logic_imports.py`, `docs/blender-runs.md`
- Modify: `CMakeLists.txt` (root; beside `cli_end_to_end`: CTest `addon_logic` =
  `uv run pytest tests/addon/logic -q`, working directory the source folder, `LABEL cpu`)

**Interfaces:**
- Produces design 3.1 exactly (with `FORMAT_NOT_LINKABLE` and `LINK_FAILED` as the design names them): `AddonError`
  (values 0 to 11), `AddonFailure`, `to_message`. `register()` and `unregister()` that do nothing yet. The manifest of
  design 5.10 without `platforms` (the packager adds one per package, Task 48) and with `blender_version_min = "5.3.0"`,
  no maximum (A16, plan decision over the design's "5.2.3"). `tests/addon/conftest.py` puts `blender/` on `sys.path`.
  `docs/blender-runs.md`: one sentence of purpose and the table header of D2.

- [ ] **Step 1: Failing tests:** `test_to_message_every_error_distinct_and_not_empty` (12 values);
  `test_failure_is_frozen`; `test_manifest_blender_version_min_is_5_3_0` (reads the TOML with `tomllib`:
  `blender_version_min == "5.3.0"`, no `blender_version_max`); `test_logic_imports_only_standard_library` (walks every
  `logic/*.py` with `ast`: imports are relative or in `sys.stdlib_module_names`; never `bpy`, `mathutils`, `bmesh`,
  `numpy`); `test_package_imports_without_blender` (`import fluidity.logic.addon_error` in a process where `bpy` is not
  importable).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject: `Start the
  Blender add-on with its error type and tests`. New names: `AddonError`, `AddonFailure`, `to_message`.

### Task 5: Roles, shapes and motions

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/roles.py`, `tests/addon/logic/test_roles.py`
- Modify: `docs/glossary.md` (`participant`, `role`)

**Interfaces:**
- Produces design 3.2 `Role`, `Shape`, `Motion`, `DrivenBy`, `RoleRow`, `ROLE_TABLE`, `to_enum_items` exactly.
  Enum item numbers are the position in the enum (stable; never reordered: new values are appended).

- [ ] **Step 1: Failing tests:** `test_role_table_rows_as_designed` (START_MATERIAL: BOX, BALL, kinds liquid,
  snow, springy_solid; SOURCE: BOX, liquid; GAS_SOURCE: BOX, no substance; SOLID: BOX, FLAT_SOLID, PLATE, all
  motions; NONE and DOMAIN: no shape list beyond BOX for DOMAIN); `test_ball_only_for_start_material`;
  `test_to_enum_items_shape` (5-tuples, identifier, label, tooltip, icon, number); `test_every_role_once`.
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Declare roles,
  shapes and motions in one table`.

### Task 6: The scene description

**Amended 2026-11-15 (A64, A63, A58, A59, A53):** `CellResizeDescription` loses the two grid-only depths; the
description gains cell size regions and the map, `RunDetail` loses `memory_share_of_total` and gains
`memory_left_free_bytes`, `FreeSolid` holds a `FreeSolidGuide`, and a participant has `slip_length`.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/scene_description.py`, `tests/addon/logic/test_scene_description.py`

**Interfaces:**
- Produces design 3.3's `Orientation`, `PoseKey`, `ParticipantDescription`, `SceneDescription` exactly, and the
  types it names only in comments, as frozen dataclasses (plan decision): `Vec3 = tuple[float, float, float]`;
  `ClosedSides(left, right, front, back, bottom, top: bool)`; `IngredientDescription(ingredient_id, name, share,
  rest_density, viscosity, surface_tension, friction, pressure_law, rest_sound_speed)`;
  `SubstanceDescription(substance_id, name, kind, preset, ingredients, stiffness_against_shear,
  stiffness_against_squeeze, max_springy_shortening, max_springy_lengthening, hardening, own_motion_kept,
  own_motion_in_travel, writes_points)`; `PairDescription(substance_a, substance_b, surface_tension)`;
  `WaveStroke(stroke, wave_period, ramp_duration, travel_direction)`; `FreeSolid(mass, start_linear_velocity,
  start_angular_velocity, steady_force, guide: FreeSolidGuide)` and `FreeSolidGuide(kind, moves_along_x,
  moves_along_y, moves_along_z, turns, rail_direction, hinge_point, hinge_axis)` (A59, design 3.3);
  `SourceDescription(volume_rate, velocity, start_time, end_time)`; `GasSourceKey(time, fuel_rate, heat_rate)`;
  `OpenSea(substance_id, open_sea_side, outlet_side, wave_height, water_depth, still_water_height, wave_period,
  open_sea_zone_width, outlet_zone_width, open_sea_pull_rate, outlet_pull_rate)`; `GasDescription(
  ambient_temperature, ignition_temperature, burn_rate, heat_per_fuel_mass, smoke_per_fuel_mass)`;
  `WindKey(time, acceleration)`; `CellResizeDescription(surface_reach_in_cell_widths,
  solid_reach_in_cell_widths)` (A64: no grid-only depths); `CellSizeRegionDescription(object_name, low_corner,
  high_corner, cell_width)`, `CellSizeMapDescription(image_name, width, height, grey_levels)` and
  `SceneDescription.cell_size_regions`, `.cell_size_map` (A63, design 3.3);
  `SolverDetail(max_step_duration, max_retries_per_step, max_volume_error_rate, random_seed)`;
  `RunDetail(run_on, memory_share_of_free=0.9, memory_left_free_bytes=536_870_912)` (A58);
  `ParticipantDescription.slip_length: float | None = None` and `.free_slip: bool = False` (A53); `OutputRequest(surface, foam_and_spray,
  foam_volume, gas_volume, solid_motion, surface_spacing, surface_smoothing_width, volume_spacing,
  motion_samples_per_frame, foam_particles_per_volume_time, foam_count_per_particle, foam_lifetime,
  foam_display_radius, point_fields, keeps_frame_inputs, saved_state_every_frames, saved_states_kept)`.
  Units in a comment on every field (design 3.3 and 3.9). Sequences are tuples, so every value hashes.

- [ ] **Step 1: Failing tests:** `test_description_is_immutable` (assigning a field raises
  `FrozenInstanceError`); `test_description_hashes_and_compares_by_value`; `test_defaults_as_designed`
  (`after_path == "stays_at_last_key"`, `friction_scale == 1.0`, `motion == FIXED`; 2026-11-15: `RunDetail` has
  `memory_share_of_free == 0.9`, `memory_left_free_bytes == 536870912` and no `memory_share_of_total`;
  `FreeSolidGuide().kind == "axis_limits"` with all four flags true, `rail_direction == hinge_axis == (0, 0, 1)`;
  `slip_length is None`; `cell_size_regions == ()`, `cell_size_map is None`; `CellResizeDescription` has exactly
  two fields).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Describe one scene
  as immutable values`.

### Task 7: Scene fields, checked against the schema

**Amended 2026-11-14 (A20):** the wind row points at the domain's force field collection instead of a keyable
wind property, and its tooltip states the conversion factor.

**Amended 2026-11-14 (A25):** the viscosity row's tooltip says what viscosity 0 and above 0 choose for the run.

**Amended 2026-11-15 (A47, A53, A58, A59, A63, A64):** the wind row's path is `wind[].acceleration`; new rows for
the cell size regions and map, the slip length, the guide and Memory Left Free; no `memory_share_of_total` row and no
grid-only depth rows.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/scene_fields.py`, `tests/addon/schema_leaf_paths.py` (test helper),
  `tests/addon/logic/test_scene_fields.py`

**Interfaces:**
- Produces design 3.2 `SceneField`, `SCENE_FIELDS`, `find_scene_field`, and `Layer` (`EVERYDAY`,
  `PHYSICAL_DETAIL`, `SOLVER_DETAIL`, `PERFORMANCE`), `Changes` (`SIMULATION`, `OUTPUT`). One row per `*`
  property of design 3.9; `property_path` relative to the domain (`substances[].ingredients[].viscosity`,
  `participants[].mass` for participant fields); `scene_file_path` as the branch schema names it (D4).
  `changes` is `OUTPUT` exactly for the fields outside the run fingerprint (outputs, render settings,
  `last_frame`, `memory_share_of_free`, `memory_left_free_mib`, saved-state cadence: I/O design 5.2). (2026-11-15)
  The rows of design 3.2's "NEW rows" table exactly: `cell_size_region_collection` (scene paths
  `cell_size_regions[].low_corner` and `.high_corner`), `...fluidity_cell_size_region.cell_width`,
  `cell_size_map_image`, `participants[].slip_length`, `participants[].free_slip`, `participants[].guide_kind`, `participants[].guide_object`,
  `memory_share_of_free` (scene path `step.memory_share_of_free`), `memory_left_free_mib` (scene path
  `step.memory_left_free_bytes`, unit MiB: the writer
  multiplies by 1,048,576); the axis flag rows move to `solids[].free_solid.guide.*`. No `memory_share_of_total`,
  `particle_region_depth_in_cell_widths` or `grid_only_depth_in_cell_widths` row (A58, A64). Where the branch schema
  names a path differently (D4), the schema wins and the hand-over lists it. The wind row (D13): `scene_file_path`
  `wind[].acceleration` (A47; was `gas_wind[]`), `property_path` `force_field_collection`, unit `m/s2`, label
  "Force Fields", tooltip as design 3.2's NEW rows table ("... It moves gas, spray and foam, never the liquid.
  Falloff, noise, flow other than 1 and other field types are reported in the check.") The tooltip's number is built from `WIND_ACCELERATION_PER_STRENGTH` (Task
  13a; until then the literal, replaced in Task 13a), never typed twice. (A25) The viscosity row
  (`materials[].ingredients[].viscosity`, unit `Pa s`) has the tooltip exactly "Pa s. 0: the liquid
  has no viscosity and the scene uses the faster pressure solve; above 0: the viscous solve, for the whole run".
- `schema_leaf_paths.read_leaf_paths(schema_folder: Path) -> frozenset[str]`: reads `scene_file.fbs` and
  `common.fbs` (tables, structs, enums, `[T]` vectors written `[]`) and returns every leaf path from the root
  table (`solids[].path.keys[].time`).

- [ ] **Step 1: Failing tests:** `test_every_scene_field_path_is_in_the_schema` (with `schemas/simulate/v2/`);
  `test_every_scene_field_has_unit_label_tooltip` (unit first in the tooltip; no tooltip contains "support",
  "supported" or "not yet"); `test_find_scene_field_matches_indexes` (`materials[2].ingredients[0].viscosity`
  finds the viscosity row; A25: its tooltip equals the text above); `test_find_scene_field_unknown_is_none`; `test_wind_row_reads_force_fields` (the
  `wind[].acceleration` row (A47) has `property_path == "force_field_collection"` and its tooltip contains "1.0
  m/s2 per unit of Strength");  `test_schema_reader_on_a_small_schema`
  (a three-table fbs text in `tmp_path`); 2026-11-15: `test_new_rows_as_designed` (the ten rows of design 3.2's
  "NEW rows" table: property path, scene file path, unit, label, layer, `changes` and tooltip equal to the table; the
  wind row's `scene_file_path == "wind[].acceleration"` and its tooltip contains "never the liquid");
  `test_no_removed_rows` (no row whose `property_path` or `scene_file_path` contains `memory_share_of_total`,
  `particle_region_depth`, `grid_only_depth` or `gas_wind`); `test_axis_flags_under_guide` (the four flag rows'
  scene file paths start with `solids[].free_solid.guide.`).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Give every scene
  field one label, unit and tooltip`.

### Task 8: Writing the scene file

**Amended 2026-11-15 (A47, A53, A58, A59, A63, A64):** writes `wind`, `cell_size_regions`, `free_solid.guide`,
`slip_length` and `free_slip`, `step.memory_left_free_bytes`; no grid-only depths. The map and its PNG come in Task 8a.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/scene_file.py`, `tests/addon/logic/test_scene_file.py`,
  `tests/addon/logic/full_description.py` (test helper: a description with every field set to a distinct value;
  2026-11-15: `cell_size_map` stays `None` until Task 8a, and `test_scene_fields_cover_the_scene_file` skips the
  `cell_size_map.image` row until then)

**Interfaces:**
- Consumes: `SceneDescription` (Task 6), `SCENE_FIELDS` (Task 7).
- Produces design 3.3 `write_scene_file`, `compute_scene_digest` (D11). (2026-11-15) `wind` (not `gas_wind`)
  whenever `wind_keys` is not empty; `cell_size_regions` only when `cell_resize` is present; `free_solid.guide`
  with every field for every kind; `slip_length` (number) and `free_slip` (bool) when `slip_length` is not `None`; `run_on` top-level;
  `step` with `memory_share_of_free`,
  `memory_left_free_bytes`; `cell_resize.cell_size` with the two reaches only. UTF-8 JSON by `json.dumps` with
  `allow_nan=False`, keys in schema order, floats with `repr`, ids as integers below 2^53, the file written to
  `<path>.tmp` then `os.replace`d. `open_sea`, `gas`, `gas_sources`, `wind`, `cell_resize`, `cell_size_regions`
  are left out when unused; one surface group holds every liquid substance; `points.materials` never names the gas; a
  participant's `object_name` is written as `name`. Fails `RUN_FOLDER_NOT_USABLE` with the path when the folder
  cannot be written.

- [ ] **Step 1: Failing tests:** `test_scene_fields_cover_the_scene_file` (design 7.1: every leaf path of the
  file written from `full_description()` has one `SceneField`, and every `SceneField` appears);
  `test_floats_round_trip` (`0.1`, `1/3`, `1e-300` read back equal); `test_ids_are_integers`;
  `test_unused_sections_left_out`; `test_liquid_and_gas_in_one_domain_file` (water and air, a liquid start
  region, a gas source: two materials, `gas_sources` present, `points.materials` without the gas);
  `test_digest_changes_with_bytes_only` (renaming nothing changes nothing; one value changes the digest);
  `test_unwritable_folder_is_run_folder_not_usable`; 2026-11-15: `test_cell_size_regions_written` (cell resize on,
  regions of 0.02 m and 0.05 m: `cell_size_regions` in description order with corners and widths; cell resize off:
  no key); `test_free_solid_guide_written` (design 7.1: Axes with `moves_along_z` only; Rail along (0, -1, 0); Hinge
  at (1, 2, 3) about (0, -1, 0): `kind` `axis_limits`, `rail`, `hinge` with the given values, every field present);
  `test_slip_length_written` (design 7.1: (`slip_length`, `free_slip`) = (0.0, false), (0.01, false), (0.0, true);
  neither key for `None`; never the string `"free"`); `test_wind_written_without_gas`
  (A47: wind keys and no gas: `wind` present, no `gas_wind` key anywhere); `test_run_memory_fields` (A58: `run_on` top-level; `step` has
  `memory_share_of_free` 0.9 and `memory_left_free_bytes` 536870912, no `memory_share_of_total`);
  `test_no_grid_only_depths_written` (A64: `cell_resize` has no `regions`).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Write the scene
  file the native program reads`. New names: `write_scene_file`, `compute_scene_digest`.

### Task 8a: The cell size map as a grey PNG, in plain Python

**Amended 2026-11-15 (A63):** added. The add-on writes the domain's cell size map image as an 8-bit grey PNG with
the standard library only, so `logic/` stays free of `bpy` and of image libraries.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/grey_png.py`, `tests/addon/logic/test_grey_png.py`
- Modify: `blender/fluidity/logic/scene_file.py` (the map), `tests/addon/logic/test_scene_file.py`,
  `tests/addon/logic/full_description.py` (a 2 x 2 map in the full description; the coverage test stops skipping
  `cell_size_map.image`)

**Interfaces:**
- Produces design 3.14 exactly: `LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)` (Rec. 709; the docstring cites
  ITU-R BT.709), `to_grey_levels(rgba, width, height) -> bytes | AddonFailure` (Blender's `Image.pixels` order, row 0
  at the bottom, to grey bytes top row first, `floor(255 * clamp(Y, 0, 1) + 0.5)`, alpha ignored;
  a length other than `4 * width * height` or a size below 1 is `RUN_FOLDER_NOT_USABLE` with the detail "cell size
  map has <n> values, expected <m>"), `write_grey_png(path, width, height, grey_levels) -> None | AddonFailure` (signature, IHDR colour type 0 bit depth 8, one IDAT of `zlib.compress(
  rows, 9)` with filter byte 0 per row, IEND; CRCs by `zlib.crc32`; `<path>.tmp` then `os.replace`; parent folders
  created).
- `scene_file.py` (design 3.3): `CELL_SIZE_MAP_FILE = "inputs/cell_size_map.png"`; `write_scene_file` writes
  `cell_size_map: {image: CELL_SIZE_MAP_FILE}` only when `cell_resize` and `cell_size_map` are present, the PNG
  first (`write_grey_png` to `<scene folder>/inputs/cell_size_map.png`), then the scene file; `compute_scene_digest`
  is SHA-256 of the scene file bytes followed by the PNG bytes when there is a map.

- [ ] **Step 1: Failing tests:** `test_grey_png_round_trip` (a 3 x 2 image of bytes 0, 1, 2, 253, 254, 255: the file
  starts with the PNG signature, IHDR width 3, height 2, depth 8, colour type 0, interlace 0; the inflated IDAT is
  `b"\x00\x00\x01\x02\x00\xfd\xfe\xff"`; every chunk CRC equals `zlib.crc32` of type and data);
  `test_grey_levels_luminance_and_row_order` (design 7.1: RGBA rows in Blender order, row 0 red, green, blue, row 1
  white, black, grey 0.5, alpha 0.3: bytes 255, 0, 128, 54, 182, 18 exactly); `test_grey_levels_clamped` (an HDR
  pixel (2, 2, 2) gives 255, (-1, 0, 0) gives 0); `test_wrong_pixel_count_is_a_failure` (23 values for 3 x 2:
  `RUN_FOLDER_NOT_USABLE` with "expected 24"); `test_unwritable_png_folder_is_run_folder_not_usable`;
  `test_cell_size_map_written` (design 7.1 `test_cell_size_regions_and_map_written`, map part: cell resize on, a
  4 x 2 map: `cell_size_map.image == "inputs/cell_size_map.png"`, the PNG beside the file with those levels; cell
  resize off: no key, no PNG); `test_digest_covers_the_map` (one grey level changed: the digest changes, the scene
  file bytes do not).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Write the cell size map
  as a grey PNG`. New names: `LUMINANCE_WEIGHTS`, `to_grey_levels`, `write_grey_png` (the design's 3.14).

### Task 9: Substance presets and reference scenes

**Amended 2026-11-14 (A25):** the water preset's viscosity is 0 Pa s (was 0.001), as I/O's `kWater`: a water scene
uses the pressure solve.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/presets.py`, `tests/addon/logic/test_presets.py`
- Modify: `docs/glossary.md` (`substance`)

**Interfaces:**
- Produces design 3.2 `SubstancePreset`, `SUBSTANCE_PRESETS`, `ReferenceScene`, `REFERENCE_SCENES`,
  `compute_preset_changes`, and (D6) `compute_preset_after_edit`. Values (design 5.2, I/O's
  `kMaterialPresets`): water 1000 kg/m3, 0 Pa s (A25; was 0.001), 0.072 N/m; syrup 1400, 10, 0.08; snow 300 kg/m3,
  `stiffness_against_shear` 1200 Pa, `stiffness_against_squeeze` 700 Pa, `max_springy_shortening` 0.025,
  `max_springy_lengthening` 0.01, `hardening` 4; springy solid as I/O's `kSpringySolid`; air kind `gas`, one
  ingredient, viscosity, friction and surface tension 0. Builders `build_wave_scene`, `build_syrup_scene`,
  `build_fire_scene`, `build_snow_example` return the descriptions of D5 (dimensions copied from the I/O
  files, not invented). No sand (A9).

- [ ] **Step 1: Failing tests:** `test_preset_changes_only_its_fields` (apply syrup over water: the changes are
  exactly density, viscosity, surface tension with old and new values); `test_preset_edit_becomes_custom`
  (`compute_preset_after_edit("SYRUP", "substances[].ingredients[].viscosity", False) == "CUSTOM"`; with
  `applying_preset` true it stays `SYRUP`; a non-preset field leaves it); `test_preset_values_as_designed` (A25:
  water's viscosity 0, not 0.001); `test_water_preset_viscosity_zero` (A25: the `water` row's
  `substances[].ingredients[].viscosity` value is exactly 0.0, equal to I/O's `kWater`);
  `test_reference_scenes_are_four_without_sand`; `test_reference_builders_match_io_files` (domain corners,
  cell width, size levels, substance values and participant count equal to the I/O JSON files, read with
  `json`).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Add substance
  presets and the four reference scenes`.

### Task 10: Native facts, the capabilities event

**Amended 2026-11-14 (A16):** `WitnessRecordFact` holds one Blender result, `blender_5_3`, as render output's
`WitnessRecord` now does.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/native_facts.py` (this part), `tests/addon/record_event_examples.py`,
  `tests/addon/testdata/events/capabilities.json`, `tests/addon/logic/test_native_facts_capabilities.py`

**Interfaces:**
- Produces design 3.5 `WitnessRecordFact` (plan decision, A16: `property: str`, `blender_5_3: str` (`"not_run"`,
  `"documented"`, `"passed"`, `"failed"`), `evidence: str`; no `blender_5_2`), `FormatCandidateFact`,
  `FormatChoiceFact`, `OutputChannelFact`
  (`value_type`, D1), `StaticLimits`, `Capabilities` (`capability_reasons` as `(code, level, message)`), each
  with `from_event(fields: Mapping) -> <type> | AddonFailure`; unknown keys kept in `other: Mapping[str,
  object]`; a missing needed key is `NATIVE_CALL_FAILED` naming the key.
- `record_event_examples.py --fluidity <exe> --out tests/addon/testdata/events`: runs `capabilities`, and on
  `small_syrup.json` `check`, `run --last-frame 2`, `report`, `read-particle`, and saves one example of each
  event kind (one JSON object per file, named by event). Committed outputs are real, from the branch.

- [ ] **Step 1:** run the recorder once; commit its outputs in this task.
- [ ] **Step 2: Failing tests:** `test_capabilities_from_recorded_event` (every field of design 3.5 filled;
  protocol 2; coordinates text; four format choices in `GeometryKind` order; every witness record has a
  `blender_5_3` value and no `blender_5_2` key; channels non-empty);
  `test_unknown_keys_kept_in_other`; `test_missing_protocol_is_native_call_failed`.
- [ ] **Step 3:** run (fail). **Step 4:** implement. **Step 5:** gate. **Commit.** Subject: `Read the native
  program's capabilities`.

### Task 11: Findings and repairs

**Amended 2026-11-14 (A27):** `REPAIR_TABLE` has 11 step names, not 12: `SuggestedChange::kSetViscosityToZero` is
removed from the main sequence (no reason uses it once the free-solid exclusion is lifted), so the
`set_viscosity_to_zero` entry (the property edit to viscosity 0) goes (the sand module adds it back with
`kViscousLiquidWithSand`, with the same property edit).

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/findings.py`, `tests/addon/logic/test_findings.py`

**Interfaces:**
- Produces design 3.4 `Severity`, `Repair`, `Finding`, `REPAIR_TABLE` (D9), `build_findings_order`, plus
  `Finding.from_event(fields, participant_ids: Mapping[tuple[str, int], int])` (`("solid", id)`, `("source",
  id)`, `("region", id)` to `participant_id`) and `merge_findings(quick, native) -> tuple[Finding, ...]`
  (de-duplicated by `(module, code, participant_id)`, then `build_findings_order`).

- [ ] **Step 1: Failing tests:** `test_repair_table_covers_suggested_changes` (the 11 step names of D9 and
  I/O's four, nothing else; A27: no `set_viscosity_to_zero`); `test_findings_order_severity_participant_module_code`;
  `test_merge_drops_duplicates_keeps_first`; `test_solid_id_becomes_participant_id`;
  `test_use_smaller_cells_repair_uses_limit`; `test_unknown_change_shown_as_text` (no repair, no error).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `List findings with
  their repairs`. New names: the design 3.4 names, `merge_findings`.

### Task 12: Native facts, check, report, particle and run events

**Amended 2026-11-14 (A17, batched-run buffers):** `RunPlanFacts` carries the native budget (I/O plan Task 23a) so
the performance panel can show the planned cost before a run.

**Amended 2026-11-14 (A25):** `RunBudgetFacts` gains `liquids` and `viscous_path` (I/O plan Task 23a, amended).

**Amended 2026-11-14 (A27):** the budget's per-liquid schema table is `ViscousEffectOfLiquid` (was
`LiquidViscousEffect`; I/O plan Task 23a, amended); its field names, which `liquids` reads, are unchanged.

**Amended 2026-11-15 (A57, A58, A60):** `liquids` holds `ViscousEffectOfLiquidFacts` with the table's new fields
and `RunBudgetFacts` gains the run duration and Reynolds length; `RunCostFacts` and `LiquidForceFacts` are read from
the end events and `frame_done`.

**Kind:** logic. **Tests:** (a).

**Files:**
- Modify: `blender/fluidity/logic/native_facts.py`
- Create: `tests/addon/logic/test_native_facts_events.py`

**Interfaces:**
- Consumes: `Finding.from_event` (Task 11).
- Produces design 3.5 `ChannelInFile`, `OutputDeclaration`, `OutputMatch`, `CacheComparison`, `RunPlanFacts`,
  `NativeCheck`, `LastTry`, `RunFolderFacts` (`ignored_end_bytes`, D1), `ParticleFacts`, each with `from_event`
  (the `participant_ids` map is a keyword argument, empty by default). Ids sent as decimal text become `int`.
- **Plan decision (A17, batched-run buffers):** `RunPlanFacts` gains `budget: RunBudgetFacts | None` (None when the
  event has no `budget`), a frozen dataclass read from `scene_checked.budget` with the I/O schema names:
  `run_mode`, `mode_reason`, `verdict` (strings), `processor_budget_bytes`, `processor_allowed_bytes`,
  `cpu_budget_bytes`, `cpu_allowed_bytes` (from `processor_memory` and `cpu_memory`), `bytes_per_step`,
  `copies_per_step` (from `copies`), `lines: tuple[str, ...]`. `run_mode` also accepts
  `"batched_with_solves_in_pieces"`. The add-on computes nothing from these numbers. (A25) Also `viscous_path: str`
  (`"pressure_solve"` or `"viscous_solve"`) and `liquids: tuple[tuple[str, float, float, float, bool, bool], ...]`,
  one per entry of the budget's `liquids` (table `ViscousEffectOfLiquid`, A27) in the schema's field order (`ingredient_name`, `viscosity_pa_s`,
  `density_kg_per_m3`, `viscous_effect_per_step`, `is_viscous`, `effect_not_visible`).
- **Amended 2026-11-15 (A57):** `liquids: tuple[ViscousEffectOfLiquidFacts, ...]` (design 3.5) replaces the 6-tuple:
  `ingredient_name`, `viscosity_pa_s`, `density_kg_per_m3`, `viscous_effect_per_step`, `is_viscous`,
  `viscosity_to_smoothing_ratio` (`None` when the key is absent), `four_cell_feature_change`, `spreading_in_cells`,
  `reynolds_number`, `advises_zero_viscosity`; `RunBudgetFacts` gains `viscous_effect_run_seconds` and
  `reynolds_length_m`. `effect_not_visible` is gone (a key of that name lands in `other`).
- **Amended 2026-11-15 (A58, A60):** `RunCostFacts(lines, warnings)` from the `cost` of `done`, `stopped`, `failed`
  (`from_event`; `None` when the event has no `cost`; the numbers stay in `other`); `LiquidForceFacts(solid_id,
  pressure_force, viscous_force, pressure_torque, viscous_torque)` from each entry of
  `frame_done.liquid_force_on_solids` (`pressure_force_n`, `viscous_force_n`, `pressure_torque_n_m`,
  `viscous_torque_n_m`; `solid_id` as an integer or decimal text).

- [ ] **Step 1: Failing tests:** `test_native_facts_parse_io_examples` (design 7.1: the recorded
  `scene_checked`, `report`, `particle`; every field filled; `particle_id` above 2^53 kept exact);
  `test_cache_comparison_absent_without_cache`; `test_declaration_channels_keep_witnessed_flag`;
  `test_findings_in_scene_checked_are_findings`; `test_budget_read_from_scene_checked` (A17: the recorded
  `scene_checked` of I/O Task 39's `check falling_block.json` gives `budget.lines` equal to the event's list and
  `verdict == "fits"`; an event without `budget` gives `None`; A25: falling block uses the water preset, so
  `viscous_path == "pressure_solve"` and `liquids` holds one entry with viscosity 0 and `is_viscous` false; A57: that
  entry is a `ViscousEffectOfLiquidFacts`); 2026-11-15: `test_viscous_effect_of_liquid_read_from_budget` (design 7.1:
  the water and syrup entries of I/O's `BuildBudgetLines, WaterAndSyrup_ViscosityAdviceLines`, as a hand-written
  `scene_checked` in `tests/addon/testdata/events/`: water ratio 0.008, change 0.2186563, spreading 0.316, Re
  4.43e+06, advised; syrup ratio 57.1, change 1.0, spreading 26.7, Re 620, not advised, each equal to the event's
  value exactly; a third entry without `viscosity_to_smoothing_ratio` gives `None`; `viscous_effect_run_seconds ==
  10.0`, `reynolds_length_m == 1.0`); `test_run_cost_read_from_end_event` (`cost.lines` and `cost.warnings` of a
  hand-written `done`; none: `None`); `test_liquid_force_entries_read` (two entries, `solid_id` 3 as text "3" and 7
  as an integer: vectors exact).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Read checks,
  reports and particles from the native program`.

### Task 13: The quick check

**Amended 2026-11-15 (A59):** a free solid guided by Rail or Hinge without a guide object is `guide_object_missing`.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/quick_check.py`, `tests/addon/logic/test_quick_check.py`
- Modify: `docs/glossary.md` (`quick check`, `full check`)

**Interfaces:**
- Consumes: `SceneDescription`, `ROLE_TABLE`, `Capabilities.limits`.
- Produces design 3.4 `check_scene_quickly`: every finding of design 5.3 step 1 a description can show, codes
  of D8, each with `participant_id`, `object_name` and `scene_file_path` where there is one. Pure; no early
  return on the first problem.
- (2026-11-15, A59) A Rail or Hinge without a guide object reaches the description as a `FreeSolidGuide` whose
  `rail_direction` (rail) or `hinge_axis` (hinge) is `(0, 0, 0)` (the scene reader's marker, Task 27); the quick check
  reports it as `guide_object_missing` with `scene_file_path` `solids[<i>].free_solid.guide`.

- [ ] **Step 1: Failing tests:** `test_quick_check_reports_all` (design 7.1: five independent problems, one a
  turned gas source, one a colliding ingredient name (`"Salt Water"` and `"salt_water"`): five findings,
  blocking first); `test_liquid_and_gas_in_one_domain` (no finding); `test_static_limits_exceeded` (plates,
  free solids, ingredients, size levels, motion samples 0 and 65: one finding each, magnitude and limit set);
  `test_surface_spacing_costly_is_costs_more`; `test_source_window_outside_frames`;
  `test_reference_scenes_have_no_quick_finding` (the four builders); `test_quick_check_under_100_ms` (the wave
  description, median of 20 calls); 2026-11-15: `test_guide_without_object_is_a_finding` (design 7.1: a free solid
  with guide kind `rail` and the scene reader's marker of no guide object: one `guide_object_missing`, `BLOCKS_RUN`,
  the participant's id; the same with `hinge`; `axis_limits` without an object: nothing).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Find every scene
  problem the add-on can decide alone`.

### Task 13a: Wind from force fields, in plain Python

**Amended 2026-11-14 (A20):** added. The wind comes from Blender's force field objects; this task holds the
conversion, the findings and the table of every field type, without Blender.

**Amended 2026-11-15 (A47, A50):** `build_wind_keys(readings)` has no `has_gas` (keys whenever a Wind field
exists); `check_force_fields` takes `has_gas` and `makes_foam_or_spray` as keyword arguments for the two
`wind_without_gas` texts; `wind_flow_not_honoured` only for `flow != 1.0`.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/force_fields.py`, `tests/addon/logic/test_force_fields.py`
- Modify: `blender/fluidity/logic/scene_fields.py` (the wind tooltip reads `WIND_ACCELERATION_PER_STRENGTH`),
  `docs/glossary.md` (`force field`: a Blender object whose physics field pushes simulated things; FluidiTy honours
  only the Wind type, D13)

**Interfaces:**
- Consumes: `Finding`, `Severity` (Task 11), `WindKey` (Task 6).
- Produces (plan decision, D13):
  - `WIND_ACCELERATION_PER_STRENGTH: float = 1.0` (m/s2 per unit of Blender's Wind Strength).
  - `class FieldHandling(enum.Enum): HONOURED, NOT_HONOURED, IGNORED`; `FORCE_FIELD_TYPE_TABLE: Mapping[str,
    FieldHandling]` with one row per Blender 5.3 field type except `NONE`: `WIND` `HONOURED`; `FORCE`,
    `VORTEX`, `MAGNET`, `HARMONIC`, `CHARGE`, `LENNARDJ`, `TEXTURE`, `TURBULENCE`, `DRAG`, `FLUID_FLOW`
    `NOT_HONOURED`; `GUIDE`, `BOID` `IGNORED`. Task 27a checks this list against Blender 5.3 and its manual.
  - `@dataclass(frozen=True) class FieldKey: time: float; strength: float; direction: Vec3` (direction a unit
    vector in world axes, from the scene reader).
  - `@dataclass(frozen=True) class ForceFieldReading: object_name: str; field_type: str; keys:
    tuple[FieldKey, ...]; shape: str; z_direction: str; falloff_power: float; use_min_distance: bool;
    use_max_distance: bool; noise: float; flow: float` (Blender's identifiers and values, unchanged).
  - `build_wind_keys(readings: Sequence[ForceFieldReading]) -> tuple[WindKey, ...]` (A47): empty without a Wind
    field; otherwise one key per time that appears in any Wind field's keys
    (sorted, duplicates within `1e-12 s` merged), each the sum over Wind fields of
    `WIND_ACCELERATION_PER_STRENGTH * strength * direction`, a field without a key at that time taking its
    value by linear interpolation between its keys (held constant before the first and after the last).
  - `check_force_fields(readings, *, has_gas: bool, makes_foam_or_spray: bool) -> tuple[Finding, ...]` (A47): the
    findings of D13 as amended, in the readings' order, module `"addon"`, found by `"scene_reader"`, `object_name`
    set; the scene reader passes `makes_foam_or_spray = outputs.foam_and_spray or outputs.foam_volume`.

- [ ] **Step 1: Failing tests:** `test_wind_strength_to_acceleration` (one Wind field, strength 2.5, axis `(0, 0, 1)`
  turned 90 degrees about X, so `direction == (0, -1, 0)`: one key `(0, -2.5, 0)` m/s2 exactly, and the docstring of
  `WIND_ACCELERATION_PER_STRENGTH` names the gas branch file and lines); `test_two_wind_fields_add` (`(1, 0, 0)` and
  `(0, 2, 0)`: `(1, 2, 0)`); `test_animated_strength_and_turn_give_keys` (keys at 0, 1/24, 2/24 s with strengths 0, 1, 2
  and directions `+X`, `+Y`, `+Z`: three keys equal to the products); `test_negative_strength_reverses`;
  `test_wind_without_gas_writes_keys_and_a_note` (A47, replaces `test_wind_without_gas_writes_no_keys_and_a_note`:
  one Wind field of strength 2 along +X, no gas, `makes_foam_or_spray` true: one key `(2, 0, 0)` and one
  `wind_without_gas`, `NOTE`, message exactly "The Wind field moves spray and foam only; the liquid is not pushed by
  wind."); `test_wind_without_gas_or_foam_has_no_effect_note` (A47: the same with `makes_foam_or_spray` false: keys
  written, message exactly "The Wind field has no effect in this scene."; with `has_gas` true: no
  `wind_without_gas`);
  `test_each_unhonoured_wind_setting_is_reported` (falloff power 1; `use_max_distance`; noise 0.5; shape `LINE`;
  `z_direction` `POSITIVE`; flow 0.5 and flow 0 (A50; was flow 1.0): one finding each, codes and severities of D13,
  and the keys still written from strength and direction); `test_default_wind_gives_no_finding` (A50, replaces
  `test_default_wind_gives_only_the_flow_note`: Blender 5.3's new-field defaults of D3, flow 1.0, with gas: no
  finding); `test_every_field_type_has_one_handling` (the 13 types of D3 without
  `NONE`; a reading of each non-Wind type gives `force_field_not_honoured` with `HAS_KNOWN_LIMIT` or
  `force_field_ignored` with `NOTE` as the table says, and its message names the type);
  `test_unknown_field_type_is_not_honoured` (a type not in the table, as a later Blender might add:
  `force_field_not_honoured`, never silence).
- [ ] **Step 2:** run (fail). **Step 3:** implement; replace the literal in Task 7's tooltip with the constant.
  **Step 4:** gate. **Commit.** Subject: `Turn Blender Wind force fields into wind keys`. New names:
  `WIND_ACCELERATION_PER_STRENGTH`, `FieldHandling`, `FORCE_FIELD_TYPE_TABLE`, `FieldKey`, `ForceFieldReading`,
  `build_wind_keys`, `check_force_fields`.

### Task 14: One-shot calls and run arguments

**Amended 2026-11-15 (A40):** `StartRequest.choose_mode_again`; a resume with it passes `--choose-mode-again`.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/native_calls.py`, `tests/addon/fake_fluidity.py` (test helper: a Python
  program that prints a recorded event and exits with a given code), `tests/addon/logic/test_native_calls.py`
- Modify: `blender/fluidity/logic/run_state.py` (create it with `RunKind` and `StartRequest` only, D6;
  `StartRequest.choose_mode_again: bool = False`, design 3.7, A40)

**Interfaces:**
- Produces design 3.5 `PROTOCOL`, `SCENE_SCHEMA`, `NativeExecutable`, `read_capabilities`,
  `check_scene_natively`, `read_run_folder_facts`, `read_particle`, `build_run_arguments`. Each call takes a
  keyword-only `start_process` (default: `subprocess.Popen` with D11's pipes) so tests run `fake_fluidity.py`.
  Exit 0, and 3 for `check`, are answers; exit 1 or 2, a timeout, a crash or no event is `NATIVE_CALL_FAILED`
  with the `failed` finding's message or the exit code as detail. `check_scene_natively` ends its process when
  `stop` is set (checked every 50 ms).

- [ ] **Step 1: Failing tests:** `test_run_arguments` (design 7.1: one case per `RunKind`, exact lists of design 3.5;
  `--run-on` never appears); `test_check_exit_three_is_an_answer`; `test_exit_one_is_native_call_failed_with_message`;
  `test_no_event_is_native_call_failed`; `test_stop_ends_the_check_process`; `test_particle_id_as_decimal_text`;
  2026-11-15: `test_run_arguments_choose_mode_again` (design 7.1: `RESUME` with `choose_mode_again` and `last_frame`
  40 gives `[exe, "resume", <folder>, "--choose-mode-again", "--last-frame", "40"]`; false: no flag; any other
  `RunKind` with it true: `NATIVE_CALL_FAILED` naming the field, an add-on bug).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Call the native
  program once and read its answer`.

### Task 15: Finding the native executable

**Amended 2026-11-15 (A49):** the whole `bin/<platform>/` folder is copied, so the CUDA runtime stays beside the
executable.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/native_executable.py`, `tests/addon/logic/test_native_executable.py`

**Interfaces:**
- Produces design 3.5 `to_platform_name(os_name, machine)` (`Windows`/`AMD64` -> `windows-x64`,
  `Linux`/`x86_64` -> `linux-x64`, `Darwin`/`arm64` -> `macos-arm64`, else `None`) and
  `find_native_executable(package_folder, user_folder, given_path, *, platform_name=None, start_process=None)`
  (design 5.10): the given path if set; otherwise the whole `bin/<platform>/` folder (`fluidity` or `fluidity.exe`,
  the CUDA runtime file, `build.json`, the USD resources; A49) copied once to `<user_folder>/<executable_version>/` (version from `bin/<platform>/build.json`) and made executable; then
  `read_capabilities` must report `PROTOCOL`, `SCENE_SCHEMA` and `right_handed_z_up_metres`, else
  `NATIVE_EXECUTABLE_DOES_NOT_MATCH`. Never reads `PATH`.

- [ ] **Step 1: Failing tests:** `test_find_native_executable_per_platform` (design 7.1: fake package folders
  for the three platforms; `PATH` set to a folder with a decoy `fluidity` and `shutil.which` replaced by a
  function that fails the test); `test_wrong_protocol_schema_or_coordinates_is_does_not_match` (three cases);
  `test_copied_once_per_version`; `test_missing_is_not_found_with_path`; 2026-11-15: `test_runtime_copied_beside_executable`
  (a fake `bin/linux-x64/` with `fluidity`, `libcudart.so.12` and `build.json`: all three in the user folder, the
  executable bit kept on `fluidity`).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Find the native
  program for this platform`.

### Task 16: Reading the journal

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/events.py`, `tests/addon/logic/test_events.py`
- Modify: `docs/glossary.md` (`journal`)

**Interfaces:**
- Produces design 3.6 `Event`, `read_events(journal, max_bytes)`, plus `class JournalReader` holding the open
  file, the byte offset, the partial last line and the last sequence (`read(max_bytes=256 * 1024)` returns
  `(events, failure)`).

- [ ] **Step 1: Failing tests:** `test_read_events_partial_line_at_byte_limit` (Review Focus 2: a line split by
  the 256 KiB limit and a line half written: neither returned until complete, then both, in order);
  `test_line_over_16_kib_is_events_not_readable`; `test_sequence_gap_is_events_not_readable`;
  `test_wrong_protocol_is_events_not_readable`; `test_crlf_line_endings_read`.
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Read the run's
  journal line by line`.

### Task 17: Run state, events

**Amended 2026-11-14 (reconcile):** `RunStatus.run_mode` and the `started` event's `run_mode` take the third value.

**Amended 2026-11-15 (A58, A60):** `RunStatus` gains `liquid_forces` (from each `frame_done`) and `cost` (from the
end events).

**Kind:** logic. **Tests:** (a).

**Files:**
- Modify: `blender/fluidity/logic/run_state.py`
- Create: `tests/addon/logic/test_run_state_events.py`

**Interfaces:**
- Produces design 3.6 `RunState` (with `NEEDS_NEW_RUN`), `RunStatus`, `MemoryUse` (D6), `apply_event`,
  `compute_ages`, and `build_first_status(run_id, kind, first_frame, last_frame) -> RunStatus`. `# Step N:`
  comments follow `<proc-apply-event>`.
- **Plan decision (reconcile; design 3.6 line "run_mode" lists two values):** `RunStatus.run_mode` and the
  `started` event's `run_mode` are one of `"one_piece"`, `"batched"`, `"batched_with_solves_in_pieces"` (I/O's
  `ToName(step::RunMode)`, A15); any other string is kept as sent and shown as unknown.
- **Amended 2026-11-15 (A60, A58):** `RunStatus.liquid_forces: Mapping[int, LiquidForceFacts]` is replaced whole by
  each `frame_done` (by `solid_id`; empty when the event has none); `RunStatus.cost: RunCostFacts | None` is set by
  `done`, `stopped` and `failed` when they carry `cost` (design 3.6).

- [ ] **Step 1: Failing tests:** `test_apply_event_follows_the_table` (design 7.1: every (state, event) pair of design
  5.4 and every exit code in the end events; pairs not in the table leave the status unchanged);
  `test_three_facts_from_journal` (listed to 37, saved state 30, `last_try` 38 rejected: the three facts as given, state
  still `running`; Task 18 ends it with the lost process); `test_ages_use_sent_time` (heartbeat sent at t, now t + 20 s:
  20 s); `test_checking_frame_cleared_by_first_frame_done`; `test_last_listed_frame_never_decreases`;
  `test_started_run_mode_three_values` (reconcile: each of the three strings in a `started` event lands in
  `RunStatus.run_mode`); 2026-11-15: `test_liquid_forces_from_frame_done` (design 7.1: a `frame_done` with forces for
  solids 3 and 7, then one for 3 only: keys 3 and 7, then 3 only, vectors as sent); `test_cost_read_from_end_events`
  (design 7.1: `done`, `stopped`, `failed` with `cost`: `RunStatus.cost.lines` and `.warnings` equal the event's; an
  end event without `cost`: `None`).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Follow a run's
  state from its events`.

### Task 18: Run state, requests and the cache comparison

**Amended 2026-11-15 (A40):** a resume refused with `recorded_mode_does_not_fit` ends `failed` with the finding kept,
so the panel can offer "Resume in another mode".

**Kind:** logic. **Tests:** (a).

**Files:**
- Modify: `blender/fluidity/logic/run_state.py`
- Create: `tests/addon/logic/test_run_state_requests.py`

**Interfaces:**
- Produces design 3.6 `apply_request` with D6's `Request` and `RequestKind`: the request rows of the design
  5.4 table; `PROCESS_ENDED` without an end event gives `failed` with `end_event = "process_lost"`; exit 3 from
  `preparing` gives `not_checked` keeping the finding; exit 4 `stopped`, 1 and 6 `failed` (6 with the note
  "continue from the last saved state"), 5 `failed`; `CACHE_COMPARED` with `simulation == "changed"` gives
  `needs_new_run`; a refused request returns `AddonFailure` (`RUN_ALREADY_ACTIVE` for a start while active).
  (2026-11-15, A40; design 5.4's new row) a `RESUME` from `preparing` whose process ends after a `failed` event with
  the finding `recorded_mode_does_not_fit` gives `failed` with that finding in `findings`, whatever the exit code
  (I/O decides it); `build_resume_choices(status) -> tuple[str, ...]` (in `run_state.py`, pure) gives `("resume",)`
  or, when `findings` holds `recorded_mode_does_not_fit`, `("resume", "resume_in_another_mode")`.

- [ ] **Step 1: Failing tests:** `test_apply_request_follows_the_table`; `test_three_facts_move_independently`
  (design 7.1: the journal of Task 17's test, then `PROCESS_ENDED` without an end event:
  `last_listed_frame == 37`, `saved_state_frame == 30`, `last_try.frame == 38`, state `failed`,
  `end_event == "process_lost"`); `test_exit_codes_to_states`;
  `test_pause_is_stopping_until_native_answer`; `test_new_run_keeps_old_status`; 2026-11-15:
  `test_recorded_mode_does_not_fit_keeps_resume_possible` (a stopped run, `START_TAKEN` with `RESUME`, then a `failed`
  event with `recorded_mode_does_not_fit` and `PROCESS_ENDED` with exit 5, and again with exit 3: state `failed` both
  times, the finding kept, `build_resume_choices` gives both choices; without the finding: `("resume",)`).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Apply the artist's
  requests to a run's state`.

### Task 19: The status line

**Amended 2026-11-15 (A48, A60):** the module also builds the newer-Blender note and the liquid force line.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/status_text.py`, `tests/addon/logic/test_status_text.py`

**Interfaces:**
- Produces `build_status_line(status, now_utc)` (D6) from `RunStatus` only (design 5.9). (2026-11-15) Design 3.6's
  `build_liquid_force_line(force)` (A60) and `build_blender_version_note(version)` (A48).

- [ ] **Step 1: Failing tests** (exact strings): `"Stopped at frame 38. Frames 0 to 37 are ready. Resume would
  restart from frame 30."`; `"Waiting for output (2 frames)"`; `"Reading saved state"`; `"Checking again frame 34
  of 37"`; `"Rebuilding outputs, frame 12"`; remaining time `"estimating"` with fewer than three listed frames;
  `"finishing frame 41 from an earlier session"` for a followed run; 2026-11-15: `test_liquid_force_line` (design 7.1:
  pressure (0, 0, 9.81), viscous (0.04, 0, 0): `"Liquid force: 9.81 N (pressure 9.81, viscous 0.04)"`; zero forces:
  `"Liquid force: 0 N (pressure 0, viscous 0)"`); `test_newer_blender_shows_one_note` (design 7.1, the logic half:
  (5, 3, 0) and (5, 3, 2) give `None`; (5, 4, 0) and (6, 0, 0) give exactly `"FluidiTy was tested with Blender 5.3
  only."`).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Say what a run is
  doing in one line`.

### Task 20: Run folders and paths

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/run_folder.py`, `tests/addon/logic/test_run_folder.py`

**Interfaces:**
- Produces D6's `build_run_folder`, `find_next_run_number`, `to_path_in_blend`, `to_absolute_path`. The domain
  name becomes a folder name with every character outside `[A-Za-z0-9_.-]` replaced by `_`. `to_path_in_blend`
  gives `//<relative>` when `path_module.relpath` can express it and an absolute path otherwise; never fails.

- [ ] **Step 1: Failing tests:** `test_paths_on_another_drive` (design 7.1, with `ntpath`: project `D:\proj`,
  cache `C:\cache\run_001`: the absolute path kept, no failure); `test_relative_inside_project`;
  `test_next_run_number_counts_folders_on_disk_and_records`; `test_run_folder_names_three_digits_then_more`
  (`run_001`, `run_999`, `run_1000`).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Name run folders
  and keep paths movable`.

### Task 21: Supervisor, the worker thread, checks and reports

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/supervisor.py`, `tests/addon/fake_clock.py`,
  `tests/addon/logic/test_supervisor_checks.py`
- Modify: `docs/glossary.md` (`supervisor`)

**Interfaces:**
- Produces design 3.7 `RunSupervisor(executable, clock, start_process)` with `request_check`, `cancel_check`,
  `request_report`, `request_particle`, D6's `request_link_plan` (returns a ticket; the plan arrives with
  Task 24), `read_snapshot`, `close`, and `SupervisorSnapshot` (D6). One thread named `fluidity-supervisor`,
  a 50 ms turn. `request_check` writes the scene file into a fresh temporary folder (removed afterwards),
  calls `check_scene_natively` with `--cache` of the given run folder, and stores the result by ticket. No
  method called from the main thread blocks (each puts a request on a queue and returns).

- [ ] **Step 1: Failing tests:** `test_check_request_returns_ticket_then_result`; `test_cancel_check_ends_process`
  (the fake program sleeps; after `cancel_check` the result is `NATIVE_CALL_FAILED` "cancelled" within 1 s);
  `test_report_result_by_run_id`; `test_requests_return_within_10_ms` (each request method, timed);
  `test_close_stops_the_thread` (`threading.enumerate()` has no `fluidity-supervisor` afterwards).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Run native calls
  on one worker thread`. New names: `RunSupervisor`, `SupervisorSnapshot`.

### Task 22: Supervisor, runs and the journal

**Kind:** logic. **Tests:** (a).

**Files:**
- Modify: `blender/fluidity/logic/supervisor.py`
- Create: `tests/addon/scripted_run.py` (test helper: a Python program that writes `events.jsonl` lines from a
  script file at given times, reads stdin lines and the end of stdin, exits with the script's code),
  `tests/addon/logic/test_supervisor_runs.py`

**Interfaces:**
- Produces `request_start` (design 5.5 steps 2 to 4, 7): writes `scene.json` into the new empty run folder for
  `NEW_RUN`, `REBUILD` and `START_FROM`, starts `build_run_arguments` with D11's process settings, reads at most
  256 KiB of journal per turn through `JournalReader`, folds events with `apply_event`, keeps the last 200 lines
  per run, and on process end applies `PROCESS_ENDED`. Superseded `heartbeat`, `try_done`, `preparing` are folded
  without being kept; `frame_done`, `saved_state_listed`, `finding`, `stopping`, `done`, `stopped`, `failed`
  never skipped.

- [ ] **Step 1: Failing tests:** `test_supervisor_never_blocks_the_solver` (design 7.1, Review Focus 1: the
  script writes 10,000 heartbeats and `try_done`, then `done`, stdout discarded, `read_snapshot` never called:
  the process ends; the final snapshot is `done`; every `frame_done` folded); `test_second_start_is_refused`
  (`RUN_ALREADY_ACTIVE`); `test_start_never_reuses_a_folder` (a folder with a file: `RUN_FOLDER_NOT_USABLE`);
  `test_process_lost_without_end_event`; `test_recent_events_keeps_200`.
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Start a run and
  follow its journal without blocking it`.

### Task 23: Supervisor, pause, stop now, close and earlier sessions

**Kind:** logic. **Tests:** (a).

**Files:**
- Modify: `blender/fluidity/logic/supervisor.py`
- Create: `tests/addon/logic/test_supervisor_stops.py`

**Interfaces:**
- Produces `request_stop_after_frame` (writes `stop_after_frame\n`), `request_cancel` (first call writes
  `cancel\n`; a second call while still `stopping` ends the process group, D11), `close()` (closes the live
  run's stdin, does not wait, kills nothing; design 5.5 step 6), `follow_run` (journal only, no process; design
  5.5 step 9).

- [ ] **Step 1: Failing tests:** `test_stop_now_and_lost_process` (design 7.1: the script ignores `cancel`;
  `stopping`; the second Stop Now ends the group; `failed` with `process_lost`; no process left, checked with
  `os.kill(pid, 0)` / `Popen.poll()`); `test_close_stdin_stops_after_frame` (design 7.1, Review Focus 5: the
  script saw the end of stdin, wrote `stopping` with `end_of_input` and `stopped`; nothing was killed);
  `test_follow_run_of_an_earlier_session` (design 7.1: journal with `started` and `frame_done`, lock held,
  then `stopped` appended: `stopping`, then `stopped` with the journal's facts).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Pause, stop and
  hand a run over when Blender closes`.

### Task 24: The link plan and the output states

**Amended 2026-11-14 (A22):** the design's `link` names are back (`logic/output_plan.py`, `LinkPlan`,
`build_link_plan`, `LINK_ROUTE_KEYS`, `request_link_plan`).

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `blender/fluidity/logic/output_plan.py`, `blender/fluidity/logic/output_states.py`,
  `tests/addon/logic/test_output_plan.py`, `tests/addon/logic/test_output_states.py`
- Modify: `blender/fluidity/logic/supervisor.py` (`request_link_plan` builds the plan), `docs/glossary.md`
  (`run record`, `output registry`)

**Interfaces:**
- Produces design 3.8 `LinkAction`, `LinkStep` (`path_in_blend`, D1), `LinkPlan`, `build_link_plan`, D6's `OutputRow`,
  `LINK_ROUTE_KEYS` (today: every (kind, format) candidate of the recorded format table: points usd and alembic,
  surface_mesh usd and alembic, volume open_vdb, solid_motion usd, alembic and solid_motion_json; Task 41 removes the
  keys of removed candidates), `SolidKey(frame, origin, orientation, in_scene)`; and `OutputState`,
  `build_output_states`, `describe_scene_change` (D6; design 5.3 step 5). `first_file` is `file_pattern` formatted with
  the first listed frame under the run folder.

- [ ] **Step 1: Failing tests:** `test_link_plan_from_report` (design 7.1: a report with every product, one
  format per kind, one declaration whose format has no key, two fingerprints: steps name the declared paths,
  object names and channels; `FORMAT_NOT_LINKABLE`; refused for two fingerprints);
  `test_create_repoint_leave_against_rows`; `test_solid_json_read_into_keys` (a fake `read_json`);
  `test_cache_comparison_states` (design 7.1: same; surface spacing changed; foam changed with
  `needs_simulation`; viscosity changed; executable changed: `READY` rows; `surface_<id>`
  `STALE_REBUILDABLE`; foam `STALE_NEEDS_NEW_RUN`; `describe_scene_change` names "Viscosity"; names
  "executable").
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Plan which Blender
  objects show which output files`. New names: the design 3.8 names, `OutputRow`, `LINK_ROUTE_KEYS`,
  `OutputState`.

### Task 25: Scene files against the native program

**Kind:** integration. **Tests:** (a), with the built `fluidity`.

**Files:**
- Create: `tests/addon/with_fluidity/conftest.py` (option `--fluidity`), `tests/addon/with_fluidity/test_scene_files.py`
- Modify: `CMakeLists.txt` (CTest `addon_with_fluidity`: `uv run pytest tests/addon/with_fluidity -q --fluidity
  $<TARGET_FILE:fluidity_cli>`, `LABEL cpu`, `TIMEOUT 900`)

- [ ] **Step 1: Tests:** `test_reference_scene_fingerprints_equal_io_files` (D5: for each builder, `check` on the
  add-on's file and on I/O's `small_<x>.json`: both exit 0, equal `run_fingerprint`);
  `test_every_field_written_is_known` (`full_valid_description()`, added to Task 8's helper: every field set to a valid
  value; no `unknown_scene_field` finding, exit 0 or 3 only for value findings, never a parse failure);
  `test_every_printed_suggested_change_has_an_entry` (every scene in `native/fluidity/io/testdata/scenes/`: each
  `suggested_changes` value of `check` is in `REPAIR_TABLE`); `test_native_finding_ids_map_to_participants` (a scene
  with a too-thin plate: the finding's `solid_id` maps to the participant); `test_cache_comparison_parses` (`run
  --last-frame 1`, edit viscosity, `check --cache`: `CacheComparison` with `simulation == "changed"`).
- [ ] **Step 2: Run** `uv run pytest tests/addon/with_fluidity -q --fluidity <exe>`. Fix owning commits if
  needed. **Step 3:** gate. **Commit.** Subject: `Check the add-on's scene files with the native program`.

### Task 26: Property groups

**Amended 2026-11-14 (A16, A20, A22):** the domain has `force_field_collection` instead of a keyable
`wind_acceleration`, the preference is `link_outputs_when_ready`, and the tests run in Blender 5.3.

**Amended 2026-11-15 (A53, A58, A59, A63, A64):** the domain loses `memory_share_of_total` and the two grid-only
depths and gains `memory_left_free_mib`, `cell_size_region_collection`, `cell_size_map_image`; the participant gains
`guide_kind`, `guide_object`, `slip_length`, `free_slip`; a new group `FluidityCellSizeRegion` on objects.

**Kind:** blender. **Tests:** (a) static checks, (b) background Blender.

**Files:**
- Create: `blender/fluidity/properties.py`, `tools/blender/run_blender_tests.py`,
  `tests/addon/in_blender/blender_test_runner.py` (calls every `test_*` function of the file, prints `PASS
  <name>` or `FAIL <name>: <reason>`, exits 1 on any failure), `tests/addon/in_blender/test_properties.py`,
  `tests/addon/logic/test_property_sources.py`

**Interfaces:**
- Produces design 3.9's groups with D6/D7 additions (`FluidityDomain.next_id`), the preference `link_outputs_when_ready`
  as the design names it, and D13's change: `FluidityDomain` has no `wind_acceleration`; it has `force_field_collection`
  (`PointerProperty(type=bpy.types.Collection)`, a `*` field on the `wind[].acceleration` row of Task 7, so its
  tooltip states the factor). (2026-11-15, design 3.9) `FluidityDomain`: no `memory_share_of_total`,
  `particle_region_depth_in_cell_widths`, `grid_only_depth_in_cell_widths`; `memory_share_of_free` default 0.9;
  `memory_left_free_mib` (`IntProperty`, MiB, default 512, min 0; a Blender `IntProperty` is 32-bit, so bytes would
  not fit above 2 GiB); `cell_size_region_collection` (`PointerProperty(type=bpy.types.Collection)`);
  `cell_size_map_image` (`PointerProperty(type=bpy.types.Image)`). `FluidityParticipant`: `guide_kind` (enum `AXES`,
  `RAIL`, `HINGE`), `guide_object` (`PointerProperty(type=bpy.types.Object, poll=<type == 'EMPTY'>)`),
  `slip_length` (m, default 0, min 0), `free_slip`* (bool, its own scene field `solids[].free_slip`). `FluidityCellSizeRegion` registered as `Object.fluidity_cell_size_region` with
  `cell_width`* (m, default 0, min 0). `FluidityIngredient`, `FluiditySubstance`, `FluidityPair`, `FluidityParticipant`
  (`Object.fluidity`), `FluidityOutputSettings`, `FluidityRunRecord`, `FluidityOutputRow`, `FluidityDomain`
  (`Object.fluidity_domain`), `CLASSES` (registration order), `mark_scene_changed` (sets a flag in `ui_model`, Task 30;
  until then a module-level flag). Fields marked `*` use `field_property(scene_file_path, ...)`, which takes `name` and
  `description` from `SCENE_FIELDS`; substance fields also apply `compute_preset_after_edit`. No live status in any
  group.

- [ ] **Step 1: Failing tests:** (a) `test_role_table_is_the_only_source` (design 7.1, by `ast` over
  `blender/fluidity/*.py`: every `EnumProperty` named `role`, `shape`, `motion`, `driven_by` takes
  `items=to_enum_items(...)`); `test_scene_field_properties_use_the_table` (every `*` field is a
  `field_property(...)` call whose path is in `SCENE_FIELDS`). (b) `test_groups_register_and_unregister`;
  `test_editing_a_substance_field_sets_custom`; `test_gravity_comes_from_the_scene`;
  `test_domain_has_force_field_collection_and_no_wind_property` (`force_field_collection` accepts a collection
  and `None`; `hasattr(domain.fluidity_domain, "wind_acceleration")` is false; the property's description
  contains "1.0 m/s2 per unit of Strength"); 2026-11-15: `test_memory_and_grid_only_properties` (no
  `memory_share_of_total`, `particle_region_depth_in_cell_widths`, `grid_only_depth_in_cell_widths` on the domain;
  `memory_share_of_free == 0.9`, `memory_left_free_mib == 512` on a new domain); `test_cell_size_properties`
  (`cell_size_region_collection` accepts a collection and `None`; `cell_size_map_image` accepts an image;
  `Object.fluidity_cell_size_region.cell_width == 0.0` on a new object); `test_guide_and_slip_properties`
  (`guide_kind == "AXES"`, `guide_object` refuses a mesh object and accepts an empty, `slip_length == 0.0`,
  `free_slip` false).
- [ ] **Step 2: Run** (a) with `uv run pytest`, (b) with `uv run python tools/blender/run_blender_tests.py --blender
  /Applications/Blender.app/Contents/MacOS/Blender --only test_properties.py` (fail). **Step 3:** implement. **Step 4:**
  run both; add a row to `docs/blender-runs.md`; gate. **Commit.** Subject: `Keep the scene's settings on the domain
  object`. Body: D2's Blender lines.

### Task 27: The scene reader

**Amended 2026-11-14 (A16, A20):** starts by creating the Blender 5.3 API probe and re-checking the facts it uses;
reads Wind force field objects into wind keys and force field findings.

**Amended 2026-11-15 (A47, A53, A59, A63):** wind keys without the gas condition; cell size region boxes and the
map's grey levels; the guide empty at the first frame; slip length only beside a viscous liquid.

**Kind:** blender. **Tests:** (b).

**Files:**
- Create: `blender/fluidity/scene_reader.py`, `tests/addon/in_blender/test_scene_reader.py`,
  `tools/blender/probe_blender_api.py` (header docstring like any `tools/blender/` file)

**Interfaces:**
- Produces D6's `read_scene` and `read_scene_in_slices` (design 5.1, section 6): shape from the object's own
  bounding box, origin at its world centre, orientation from the normalised world matrix (D7), keys read with
  `FCurve.evaluate` at every frame of the range without `frame_set` (curves reached through the action's
  channel bag, D3 `action_curves`), keyed gas rates as design 5.1 step 6, gravity from `scene.gravity` (zero when
  `use_gravity` is off), frames and rate from the scene. Wind (D13): one `ForceFieldReading` per force field
  object of `force_field_collection` (or of the scene when it is empty), with each key's direction from
  `mathutils` (`Quaternion` of the keyed rotation times `Vector((0, 0, 1))`); then
  `description.wind_keys = build_wind_keys(readings)` and `check_force_fields(readings, has_gas=...,
  makes_foam_or_spray=...)` added to `findings` (Task 13a, A47).
- (2026-11-15, design 5.1 items 7 to 9) Cell size (A63), only when `uses_cell_resize`: one
  `CellSizeRegionDescription` per object of `cell_size_region_collection` in the collection's object order (box =
  min and max per axis of `matrix_world @ Vector(corner)` over the eight `bound_box` corners; `cell_width` the
  object's, or `smallest_cell_width` when 0; `cell_size_region_rotated` when an off-diagonal term of the normalised
  3x3 of `matrix_world` exceeds `1e-9` in magnitude); `cell_size_map_image.pixels.foreach_get` into an `array('f')`
  and `to_grey_levels` (Task 8a) into `CellSizeMapDescription`. Guide (A59): for `RAIL` and `HINGE` the
  `guide_object`'s pose at `first_frame` through the same key reader as driven motion (`motion_not_keys` for a
  parent, constraint, driver or modifier); Rail: `rail_direction` = its local +Z (normalised, `mathutils`); Hinge:
  `hinge_point` = its world location, `hinge_axis` = its local +Z; no guide object: the zero-axis marker of Task 13.
  Slip (A53): `slip_length` for `FIXED` and `DRIVEN` solids only when a liquid substance has an ingredient with
  viscosity above 0, with `free_slip` copied, else `None`.
- `probe_blender_api.py` (run as `<blender> --background --factory-startup --python-exit-code 1 --python
  tools/blender/probe_blender_api.py -- [--only <check> ...]`): one function per bracketed check of D3, each
  printing `PROBE <check> TRUE` or `PROBE <check> FALSE: <what differs>`, then `PROBE blender <version>
  <build hash>`; exit 1 if any check is false. It never writes outside a temporary folder.

- [ ] **Step 0: Re-check in Blender 5.3.** Write the probe, then run it with `--only action_curves bake_to_keys
  force_fields` in `/Applications/Blender.app/Contents/MacOS/Blender`. Expected: three `TRUE` lines. Copy the output
  into the commit body; a `FALSE` stops the task (report it with the output).
- [ ] **Step 1: Failing tests:** `test_asymmetric_fixture` (design 7.1: one point at (1, 2, 3) moving at (4, -5, 6), a
  box turned 30 degrees about X then 20 about Z, keyed over three frames: the same numbers; path orientations equal to
  an independent quaternion product within `1e-12`; `scene.frame_current` unchanged);
  `test_motion_from_parent_or_constraint_gives_bake_to_keys` (two objects: one with an object above it, one with a
  constraint: `motion_not_keys` with the repair "Bake to Keys" naming `bpy.ops.nla.bake`, re-checked in Step 0);
  `test_sheared_matrix_is_a_finding`; `test_unsaved_project_is_a_finding`; `test_slices_yield_per_participant`;
  `test_wind_field_read_into_keys` (a Wind field made with `effector_add(type='WIND')`, strength keyed 0 at frame 0 and
  3 at frame 2, rotation keyed from none to 90 degrees about Y, both curves with linear interpolation: keys at 0, 1/24,
  2/24 s equal to strengths 0, 1.5, 3 times the axis turned 0, 45, 90 degrees about Y, within `1e-12` (the frame-2 key
  is `(3, 0, 0)` m/s2); `scene.frame_current` unchanged); `test_force_field_collection_limits_the_fields` (two Wind
  fields, one in the chosen collection: only its keys; collection cleared: both, summed);
  `test_wind_settings_and_other_fields_are_findings` (a Wind field with noise and a Vortex field:
  `wind_noise_not_honoured` and `force_field_not_honoured` naming `VORTEX`, each with the object's name);
  `test_turned_wind_from_parent_is_motion_not_keys`; 2026-11-15: `test_wind_without_gas_still_gives_keys` (A47: a
  liquid-only domain with one Wind field: keys written, `wind_without_gas` with the text of D13 as amended);
  `test_region_box_is_world_aabb` (design 7.1: half size 0.1 m cube at (1, 2, 3) scaled (2, 1, 1): low (0.8, 1.9,
  2.9), high (1.2, 2.1, 3.1) within `1e-12` m, no finding; turned 45 degrees about Z: half extents 0.1414214 m in x
  and y within `1e-12`, `cell_size_region_rotated` `NOTE`; `cell_width` 0 gives the domain's `smallest_cell_width`;
  cell resize off: no regions read); `test_cell_size_map_read_as_grey_levels` (a 4 x 2 generated image with known
  pixels: `grey_levels` equal to `to_grey_levels` of those pixels); `test_guide_from_empty` (design 7.1: an empty at
  (1, 2, 3) turned 90 degrees about X: Rail `rail_direction` (0, -1, 0), Hinge `hinge_point` (1, 2, 3) and
  `hinge_axis` (0, -1, 0) within `1e-12`; the empty keyed to move after the first frame gives the first-frame values;
  with a parent: `motion_not_keys`); `test_slip_length_only_beside_viscous_liquid` (water: `None` everywhere; syrup:
  0.01 on a fixed box, `free_slip` true with Free Slip, `None` on a free box).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run in Blender 5.3; row in
  `docs/blender-runs.md`; gate. **Commit.** Subject: `Read Blender objects into a scene description`. Body: the
  probe output of Step 0 and D2's Blender line. New names: `probe_blender_api.py`.

### Task 27a: Every Blender force field type has a row

**Amended 2026-11-14 (A20):** added, so that no Blender force field type is dropped without a finding ("compatible
with Blender force fields", owner).

**Kind:** blender. **Tests:** (a) and (b).

**Files:**
- Create: `tests/addon/testdata/blender/force_field_types.json` (the probe's list, committed),
  `tests/addon/in_blender/test_force_field_types.py`
- Modify: `tools/blender/probe_blender_api.py` (`--write-field-types <json>`), `blender/fluidity/logic/force_fields.py`
  (only if Step 2 changes a row), `tests/addon/logic/test_force_fields.py`,
  `docs/method-paper/sections/blender.typ` (the table)

**Interfaces:**
- Consumes: `FORCE_FIELD_TYPE_TABLE`, `check_force_fields` (Task 13a), `read_scene` (Task 27).

- [ ] **Step 1: Record the list.** Run the probe with `--write-field-types
  tests/addon/testdata/blender/force_field_types.json` in Blender 5.3: the identifiers and UI names of
  `bpy.types.FieldSettings.bl_rna.properties["type"].enum_items`, the version and build hash.
- [ ] **Step 2: Decide each row** with the Blender 5.3 manual page on force fields (URL copied at writing time):
  for each type, "honoured" (only `WIND`), "reported as not honoured" (`NOT_HONOURED`: Blender's fluid domains
  feel it, FluidiTy does not) or "ignored with a finding" (`IGNORED`: the manual says it acts only on particle
  systems, hair or boids). Start from Task 13a's rows; change a row only where the manual says otherwise, and
  quote the manual line in the commit body. Write the table (type, Blender name, handling, finding code, one
  reason) into the method paper section next to the wind paragraph of Task 2.
- [ ] **Step 3: Failing tests:** (a) `test_every_blender_field_type_has_a_row` (every identifier of the JSON except
  `NONE` is a key of `FORCE_FIELD_TYPE_TABLE`, and every key is in the JSON); (b)
  `test_each_field_type_gives_its_finding` (in Blender 5.3: per type, one empty made with `effector_add(type='WIND')`
  whose `field.type` is then set to that type, read with `read_scene`: `WIND` gives keys, every other type gives exactly
  its table's finding).
- [ ] **Step 4:** run (fail where a row is missing), fix the table, run (pass); row in `docs/blender-runs.md`;
  gate. **Commit.** Subject: `Say what happens to every Blender force field type`. Body: the list, each decision
  with its manual quote, D2's Blender line.

### Task 28: Authoring operators, domain, roles and repairs

**Kind:** blender. **Tests:** (b).

**Files:**
- Create: `blender/fluidity/operators_authoring.py` (this part), `tests/addon/in_blender/test_authoring_roles.py`

**Interfaces:**
- Produces design 3.10 `fluidity.add_domain`, `fluidity.apply_role` (the only writer of `participant.domain` and
  `domain.participants`), `fluidity.remove_participant`, `fluidity.select_participant`,
  `fluidity.add_wave_paddle` (SOLID, FLAT_SOLID, DRIVEN, WAVE_STROKE at the chosen side), and D9's
  `fluidity.apply_field_value`. Every operator is undoable (`bl_options = {'REGISTER', 'UNDO'}`); `role`, `shape`,
  `motion` arguments use `to_enum_items(ROLE_TABLE ...)`.

- [ ] **Step 1: Failing tests:** `test_add_domain_around_selection`; `test_apply_role_sets_both_sides`;
  `test_role_in_second_domain_moves_it` (never in two); `test_ids_never_reused` (remove then add: a new id);
  `test_add_wave_paddle_values`; `test_apply_field_value_scales_cell_width`.
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject: `Add the
  domain, roles and the wave paddle`.

### Task 29: Authoring operators, substances, presets and reference scenes

**Amended 2026-11-14 (A20):** the fire reference scene's wind is made as a Blender Wind force field object with
keyed strength, so it round-trips through the scene reader.

**Amended 2026-11-15 (A50):** the fire scene's Wind field keeps Blender's default flow 1.0, which gives no finding
(it was set to 0 to avoid the note that `flow != 0` gave).

**Kind:** blender. **Tests:** (b).

**Files:**
- Modify: `blender/fluidity/operators_authoring.py`
- Create: `tests/addon/in_blender/test_authoring_substances.py`

**Interfaces:**
- Produces design 3.10 `fluidity.add_substance`, `remove_substance`, `add_ingredient`, `remove_ingredient`,
  `apply_substance_preset` (invoke lists `compute_preset_changes` with old and new values, execute writes them),
  `add_reference_scene` (creates the domain, participants and substances of a builder of Task 9; for the fire
  scene also one Wind force field object named "Fire Wind" in a collection set as the domain's
  `force_field_collection`, its strength keyed so that `WIND_ACCELERATION_PER_STRENGTH * strength` gives the
  builder's wind keys and its orientation pointing along their direction, with constant interpolation between
  keys so the read keys equal the builder's; `flow` left at 1.0, A50).

- [ ] **Step 1: Failing tests:** `test_reference_scene_round_trip` (each of the four: created, read with
  `read_scene`, the description equals the builder's except object names and ids, including the fire scene's
  `wind_keys`; quick check clean; no scene-reader finding, the fire wind's flow being 1.0, A50);
  `test_apply_preset_lists_then_writes`; `test_preset_edit_becomes_custom_in_blender`;
  `test_at_most_one_gas_substance_is_a_finding`.
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject: `Add
  substances, presets and the reference scenes`.

### Task 30: Registration, preferences, handlers and the timer

**Amended 2026-11-14 (A16, A22):** re-checks `bpy.utils.extension_path_user` in Blender 5.3 first; any Blender
other than 5.3 gives the one note; the preference is `link_outputs_when_ready`.

**Amended 2026-11-15 (A48):** registration gives no version note; `ui_model` keeps `build_blender_version_note(
bpy.app.version)` and the domain panel draws it (Task 32). Only 5.4.0 and later get it.

**Kind:** blender. **Tests:** (b).

**Files:**
- Create: `blender/fluidity/preferences.py`, `handlers.py`, `ui_model.py`, `ui_timer.py`,
  `tests/addon/in_blender/test_registration.py`
- Modify: `blender/fluidity/__init__.py`

**Interfaces:**
- Produces `FluidityPreferences` (design 3.9 row, with `link_outputs_when_ready`); `ui_model.UiModel` (the last
  copied `SupervisorSnapshot`, quick findings by `domain_id`, scene-changed flags); `ui_timer.tick()` (design
  section 6: copies the snapshot when its change counter moved, tags Properties areas, writes a property only
  when a run record or the registry must change, returns 0.1 or 1.0); handlers `load_pre` (`close`),
  `load_post`, `undo_post`, `redo_post` (clear `ui_model`, `request_report` for every run record),
  `depsgraph_update_post` (a participant object changed: scene changed), each `@persistent`. `register()`
  registers classes, handlers, the timer, finds the executable (Task 15; failure kept for the panel, never
  raised) and builds the supervisor; (A48) it stores `ui_model.blender_version_note =
  build_blender_version_note(tuple(bpy.app.version))` (Task 19) and reports nothing itself.

- [ ] **Step 0: Re-check in Blender 5.3:** the probe of Task 27 with `--only extension_path_user` in
  `/Applications/Blender.app/Contents/MacOS/Blender`; expected `TRUE`; output in the commit body; a `FALSE` stops the
  task.
- [ ] **Step 1: Failing tests:** `test_register_unregister_five_times` (design 7.1: classes, handlers, timer and
  `fluidity-supervisor` thread counts return to zero, and are one each while registered);
  `test_missing_executable_registers_with_a_note`; `test_tick_writes_no_property_without_change`;
  `test_version_note_kept_in_ui_model` (A48, replaces `test_version_outside_5_3_gives_one_note`: with the version
  given as `(5, 3, 0)` and `(5, 3, 2)` the stored note is `None`; with `(5, 4, 0)` it is "FluidiTy was tested with
  Blender 5.3 only."; registration writes no report in any case).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject:
  `Register the add-on with one timer and one worker thread`.

### Task 31: Run operators

**Amended 2026-11-15 (A40):** `fluidity.resume_run` takes `choose_mode_again`; "Resume in another mode" passes
`--choose-mode-again` after a `recorded_mode_does_not_fit` refusal.

**Kind:** blender. **Tests:** (b), with `fake_fluidity.py` and `scripted_run.py` as the executable.

**Files:**
- Create: `blender/fluidity/operators_run.py`, `tests/addon/in_blender/test_run_operators.py`

**Interfaces:**
- Produces design 3.10 `fluidity.check_scene` (quick check now, full check requested), `cancel_check`,
  `apply_repair`, `start_run` (`PREVIEW` or `BAKE`; creates the run record and a new folder), `pause_run`,
  `stop_run_now` (offered again after `stop_offer_seconds`), `resume_run` (A40: option `choose_mode_again`, a
  `BoolProperty` default off, copied into `StartRequest.choose_mode_again`; `poll` with it on requires
  `build_resume_choices` to hold `"resume_in_another_mode"`), `rebuild_outputs`, `new_run`
  (option `start_from_frame`), `show_run`, `delete_run` (refused while `report` says the lock is held; lists
  folder, size and outputs that will stop playing). `poll` reads `ui_model` only.

- [ ] **Step 1: Failing tests:** `test_check_then_ready`; `test_start_creates_record_and_folder`;
  `test_pause_then_resume`; `test_rebuild_only_when_rebuildable`; `test_delete_refused_while_locked`;
  `test_file_load_during_run_ends_stopped` (design 7.1, Review Focus 5: `wm.read_homefile` and deleting the
  domain during a scripted run: no exception, nothing killed, the run ends `stopped`);
  `test_resume_offers_other_mode_on_recorded_mode_does_not_fit` (design 7.1, A40: a stopped scripted run; Resume; the
  script writes `failed` with the finding `recorded_mode_does_not_fit` and exits 5: state `failed` with the finding;
  `fluidity.resume_run(choose_mode_again=True)` polls true and starts the script with `resume <folder>
  --choose-mode-again`; before the refusal it polls false).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject: `Check,
  start, pause and resume runs from Blender`.

### Task 32: Panels

**Amended 2026-11-15 (A40, A48, A53, A57, A58, A59, A60, A63, A64):** the version note, Resume in another mode, the
liquid force line, Guide, Slip Length, the Cell size box, the memory fields and the measured cost lines (Interfaces).

**Kind:** blender. **Tests:** (b) draw calls with a recording layout; (c) in Task 50.

**Files:**
- Create: `blender/fluidity/panels.py`, `tests/addon/in_blender/recording_layout.py`,
  `tests/addon/in_blender/test_panels.py`

**Interfaces:**
- Produces design 3.11 `FLUIDITY_PT_everyday`, `FLUIDITY_PT_participants`, `FLUIDITY_PT_outputs`,
  `FLUIDITY_PT_physical_detail`, `FLUIDITY_PT_solver_detail`, `FLUIDITY_PT_performance`, `FLUIDITY_PT_inspect`,
  `FLUIDITY_UL_participants`, `FLUIDITY_UL_outputs`. `draw` reads `ui_model` and properties only. Channels with
  `is_witnessed` false read "written, not witnessed".
- **Amended 2026-11-14 (A17, batched-run buffers):** `FLUIDITY_PT_performance` draws the last full check's
  `plan.budget.lines` verbatim, one label per line, under the heading "Planned cost (estimate)"; nothing when
  `budget` is None. Test: `test_performance_panel_draws_budget_lines` (a recording layout receives each line once,
  in order).
- **Amended 2026-11-15 (A40, A48, A53, A57, A58, A59, A60, A63, A64):** `FLUIDITY_PT_everyday` draws
  `ui_model.blender_version_note` once when not `None`, and "Resume in another mode"
  (`fluidity.resume_run`, `choose_mode_again=True`) beside Resume when `build_resume_choices` offers it;
  `FLUIDITY_PT_participants` draws one `build_liquid_force_line` per selected `SOLID` participant found in the shown
  run's `liquid_forces`; `FLUIDITY_PT_physical_detail` draws the free solid's Guide (Axes: the four ticks; Rail and
  Hinge: Guide Object), Slip Length with Free Slip (`slip_length` greyed when `free_slip`) for `FIXED` and `DRIVEN`
  solids only when a liquid substance has an ingredient with viscosity above 0, and the box "Cell size" (Cell Size
  Regions, one row per object with its Cell Width, Cell Size Map; inactive with the line "Used when cell resize is
  on" while `uses_cell_resize` is off); `FLUIDITY_PT_solver_detail` draws no grid-only depth; `FLUIDITY_PT_performance`
  draws Share of Free Memory and Memory Left Free (MiB) beside Run on (no share of total) and, under "Measured cost",
  `RunStatus.cost.lines` then `cost.warnings` verbatim; the budget lines (with A57's advice and note lines, A64's deep
  particle line and, on Windows, A58's fallback note) stay verbatim. Nothing is computed from any number.
- **Amended 2026-11-14 (A25):** the `liquids:` lines are part of `plan.budget.lines` (I/O `BuildBudgetLines`, before
  the last `fits:` line), so the panel draws them verbatim with the other lines; nothing is computed from
  `RunBudgetFacts.liquids`. Test: `test_performance_panel_draws_liquids_lines` (a budget whose lines hold the
  `liquids: viscous solve ...` line, the per-liquid lines and the `note:` line of I/O's `BuildBudgetLines,
  TwoLiquids_ViscousEffectLines`: the recording layout receives each verbatim, once, in order, before `fits:`).

- [ ] **Step 1: Failing tests:** `test_every_panel_draws_in_every_state` (each `RunState`, each role assigned and
  unassigned: `draw(context)` with a recording layout raises nothing); `test_draw_opens_no_file` (`builtins.open`
  replaced by a function that fails the test during draws); `test_not_witnessed_label`;
  `test_performance_panel_draws_budget_lines` (A17); `test_performance_panel_draws_liquids_lines` (A25; amended
  2026-11-15, A57: the lines of I/O's `BuildBudgetLines, WaterAndSyrup_ViscosityAdviceLines` instead of
  `TwoLiquids_ViscousEffectLines`: each `liquids:`, per-liquid, `advice:` (water) and `note:` (water, syrup) line
  verbatim, once, in order, before `fits:`); 2026-11-15: `test_performance_panel_draws_cost_warnings_and_fallback_note`
  (design 7.1, A58: a status whose `cost` holds the lines and warnings of I/O's `BuildCostLines,
  RequestedBeyondPlan_Warns` and `BuildCostLines, MemoryTakenBelowReserved_WarnsSpill`, and budget lines ending with
  the Windows fallback note: under "Measured cost" each line then each warning once, in order; the note once; no
  widget for `memory_share_of_total`); `test_budget_deep_particle_line_drawn` (A64: the budget lines of I/O's
  `BuildBudgetLines, DeepTankDoesNotFit_NamesDeepParticles`: its `particles: ... (deep liquid without particles comes
  later)` line drawn once before `fits: no`); `test_newer_blender_note_drawn_once` (A48: `ui_model` note set: the
  everyday panel draws it once; `None`: not drawn); `test_solid_panel_slip_and_force` (design 7.1, A53, A60);
  `test_physical_detail_draws_cell_size_box` (A63: the box with the collection, one row per region object and the
  image; with cell resize off the layout is inactive and draws "Used when cell resize is on"); `test_guide_widgets`
  (A59: Axes draws the four ticks and no Guide Object; Rail and Hinge draw Guide Object and no ticks).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject: `Show
  the scene, the run and the outputs in panels`.

### Task 33: Inspect operators

**Amended 2026-11-14 (A22):** the report operator is `fluidity.copy_support_report`, "Copy Support Report", as the
design names it (D1 says why).

**Kind:** blender. **Tests:** (b).

**Files:**
- Create: `blender/fluidity/operators_inspect.py`, `tests/addon/in_blender/test_inspect.py`

**Interfaces:**
- Produces design 3.10 `fluidity.open_log`, `fluidity.copy_support_report` (`run_report.json`, the
  capabilities and the last 200 events, to the clipboard, local only), `fluidity.read_particle` (full 64-bit id
  at the current frame; the panel notes the 31-bit `id` attribute, design decision 12). `check_outputs` comes
  with Task 45.

- [ ] **Step 1: Failing tests:** `test_support_report_holds_three_parts`; `test_read_particle_result_shown`;
  `test_id_above_31_bits_note`.
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject: `Inspect
  a run from Blender`.

### Task 34: Witness files, the scene and the writer program

**Amended 2026-11-14 (A31):** files are written through a temporary `io::RunFolder` (no "scratch").

**Kind:** integration. **Tests:** (a), with the built programs.

**Amended 2026-11-14 (Foundation as built):** the program is created with `add_executable` plus `fluidity_apply_compile_flags`, its test with
`fluidity_add_test(... LABEL cpu LINKS fluidity_base_test_support)`; every new source is listed in `cli/CMakeLists.txt`.

**Files:**
- Create: `tools/blender/witness/witness_scene.json`, `native/fluidity/cli/write_witness_files.cc`,
  `native/fluidity/cli/witness_json.h` (private), `witness_json.cc`, `witness_json_test.cc`,
  `tests/addon/with_fluidity/test_witness_files.py`
- Modify: the `cli/` CMake lines (executable `fluidity_write_witness_files` from `write_witness_files.cc` and
  `witness_json.cc` with `fluidity_cli_parts` and `fluidity_interchange`; never installed or packaged; amended:
  `add_executable(fluidity_write_witness_files ...)` followed at once by
  `fluidity_apply_compile_flags(fluidity_write_witness_files EXCEPTIONS OFF)`, or the configure check fails;
  `fluidity_add_test(witness_json_test SOURCES witness_json_test.cc witness_json.cc LABEL cpu LINKS
  fluidity_base_test_support)`)

**Interfaces:**
- `witness_scene.json`: design 7.2 "The fixed scene" exactly, in the branch schema (D4).
- `fluidity_write_witness_files <run_folder> <witness_folder>` (design 3.12, last paragraph): reads the kept
  frame inputs of frames 0 to 10 in order with I/O's `ReadFrameInputs`, builds each frame with
  `BuildRenderFrame` exactly as the CLI frame worker of render output Task 47 does (if that composition is not
  a callable function in `cli/`, extract it in this task and make the worker call it), and calls every
  candidate writer of every kind of `ReadFormatTable()`. Each file is written through a temporary `io::RunFolder`
  and `OutputFile` and copied out to `<witness_folder>/<kind>/<format>/<file name of render output 5.7>` (the
  pattern of I/O's `WriteStartParticlesFile`). Then `expected.json` (per kind, format, object and frame: the
  values design 3.12 lists, doubles in shortest round-trip form, plus `frames_per_second` and every file's
  BLAKE3 digest) and `files_digest.txt` (the BLAKE3 hex of `expected.json`, with I/O's digest helper).
- Private `WitnessJsonWriter` (objects, arrays, keys, `std::to_chars` numbers), with
  `TEST(WitnessJsonWriter, NestedObjects_ExactText)` and `TEST(WitnessJsonWriter, Doubles_RoundTrip)`.

- [ ] **Step 1: Failing tests:** the two C++ tests (`LABEL cpu`); `test_witness_scene_checks` (`check` exits 0;
  if a capability finding blocks it, the test fails naming the finding: the scene is never edited);
  `test_witness_files_written` (`run --into`, then the program: one file per candidate, kind and frame 0 to
  10; `expected.json` parses; `files_digest.txt` is 64 lowercase hex characters; a second run of the program
  into a fresh folder gives byte-identical `expected.json` and `files_digest.txt`; the sphere's frame-2
  velocity is non-zero; the gas volume at frame 1 has a non-zero value; frame 10 has foam).
- Plan decision: the program lives in `native/fluidity/cli/` beside the other CLI sources (the design wrote
  `cli/tests/`; the CLI keeps tests beside the code), and passes the C++ gate like any other native file.
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `Write the witness
  files of every candidate format`. New names: `fluidity_write_witness_files`.

### Task 35: Witness cases and records in plain Python

**Amended 2026-11-14 (A16):** one results file, from Blender 5.3; each printed record carries one result.

**Kind:** logic. **Tests:** (a).

**Files:**
- Create: `tools/blender/witness/witness_cases.py`, `tools/blender/witness/measure.py`,
  `tools/blender/witness/write_witness_records.py`, `tests/addon/logic/test_witness_cases.py`,
  `tests/addon/logic/test_write_witness_records.py`

**Interfaces:**
- Produces design 3.12 `WitnessProperty`, `WitnessResult`, `WitnessCase`, `WitnessCaseResult` (D1),
  `list_witness_cases`, `write_witness_records` (D12), and in `measure.py` the pure measurements:
  `check_float_equal(measured, expected) -> bool` (Global Constraints), `compute_extent_growth(mask_blur,
  mask_still, direction) -> float` (pixels times 0.5 mm), `compute_expected_blur_growth(velocity, shutter=0.5,
  frames_per_second=24)`. `bpy` is imported only inside functions of `witness_cases.py`.
  `write_witness_records.py <results_5.3.x.jsonl> --witness-folder <folder> --capabilities <json>` prints, in
  table order, one line per record in the one-version shape of D12,
  `{WitnessProperty::<P>, WitnessResult::<r>, "<test>; Blender <version> <build hash>; files <files_digest>"},`,
  and the `chosen` value per kind (`ChooseFormat`'s rule: the first candidate whose every record is `kPassed`,
  else the last). It refuses a results file whose `files_digest` differs from the folder's `files_digest.txt`, or
  whose recorded Blender version is not 5.3.x.

- [ ] **Step 1: Failing tests:** `test_list_witness_cases_from_table_today` (38 cases: points 2 x 6, surface
  2 x 7, volume 1 x 3, solid motion 3 x 3; names like `witness_point_radius_arrives[points,usd]`);
  `test_write_witness_records` (design 7.1 with A16: all passed; one failed; one `not_run`: the exact C++ lines,
  one result each, and the `chosen` values that follow); `test_write_witness_records_not_run_where_unmeasured`
  (Review Focus 4: a case absent from the results file keeps `kNotRun` or `kDocumented` with its quote, never
  `kPassed` or `kFailed`); `test_records_refuse_different_files_digest`; `test_records_refuse_other_blender_version`
  (a results file recorded by 5.2.3 or 5.4.0: refused, naming the version);
  `test_extent_growth_of_a_drawn_streak` (two masks drawn in numpy-free lists: 12 pixels -> 6 mm).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** gate. **Commit.** Subject: `List witness cases
  and turn their results into format table lines`.

### Task 36: Witness in Blender, the harness, file sequences and Z-up

**Amended 2026-11-14 (A16):** runs in Blender 5.3, the one version, after re-checking the API facts it opens files
with.

**Kind:** blender. **Tests:** (b), Blender 5.3.

**Files:**
- Create: `tools/blender/witness/run_witness.py`, `tools/blender/witness/results/` (results files)
- Modify: `tools/blender/witness/witness_cases.py` (opening routes and W1, W8)

**Interfaces:**
- `run_witness.py -- <witness_folder> <results.jsonl> [--only <case name>]` (design 7.2 "How Blender is
  started"): asserts no `fluidity` add-on is enabled and no `FLUIDITY_` class is registered; records
  `bpy.app.version_string` and `build_hash`; each case starts from `read_factory_settings(use_empty=True)` at 24
  fps, opens files by the design's route per format with D3's facts (USD without a cache file: the new cache
  file from `bpy.ops.cachefile.open`), and writes one `WitnessCaseResult` line. A Python error inside a
  measurement, or a false precondition, is `NOT_RUN` with the reason; the script then exits 1.
- `run_witness_case` for W1 (`file_sequence_read_natively`, every kind and format except JSON) and W8
  (`z_up_without_user_rotation`), design 7.2 rows W1 and W8.

- [ ] **Step 0: Re-check in Blender 5.3:** the probe of Task 27 with `--only cache_file_open volume_new
  cache_file_fields volume_grids`; expected four `TRUE` lines; output in the commit body; a `FALSE` stops the
  task.
- [ ] **Step 1:** run on Blender 5.3: `/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup
  --python-exit-code 1 --python tools/blender/witness/run_witness.py -- <witness_folder>
  tools/blender/witness/results/results_<version>.jsonl` (`<version>` as `--version` prints it, `5.3.0` today) with the
  folder of Task 34. Before implementing, the cases print `NOT_RUN` "not written" and exit 1.
- [ ] **Step 2:** implement until every W1 and W8 case gives `PASSED` or `FAILED` (both are results; a `FAILED`
  is not fixed in the add-on, it is evidence). **Step 3:** commit the results file and the code; row in
  `docs/blender-runs.md`. Gate. **Commit.** Subject: `Witness file sequences and Z-up in Blender`. Body: the
  probe output, the per-case results and D2's Blender line.

### Task 37: Witness, surface topology and face attributes

**Amended 2026-11-14 (A16):** runs in Blender 5.3.

**Kind:** blender. **Tests:** (b).

**Files:**
- Modify: `tools/blender/witness/witness_cases.py`, `measure.py`, the results file

- [ ] **Step 1:** implement W2 (`topology_may_change_per_frame`) and W7 (`face_attributes_arrive`) for
  surface USD and Alembic, design 7.2 rows W2 and W7 (precondition of W2: expected counts differ between frames
  2 and 10, else `NOT_RUN`).
- [ ] **Step 2:** run with `--only` on these cases in Blender 5.3; append to the results file (one line per
  case; a later full run replaces the file). Row; gate. **Commit.** Subject: `Witness surface topology and face
  attributes in Blender`.

### Task 38: Witness, velocity, radius and point attributes

**Amended 2026-11-14 (A16):** runs in Blender 5.3.

**Kind:** blender. **Tests:** (b).

**Files:**
- Modify: `tools/blender/witness/witness_cases.py`, `measure.py`, the results file

- [ ] **Step 1:** implement W4 (`velocity_in_seconds`, points and surface), W5 (`point_radius_arrives`: the
  particles, droplets, spray, foam and bubbles objects, an empty object counts as zero points with the right
  type) and W6 (`point_attributes_arrive`: every channel whose `witnessed_by` is this property, names taken from
  the capabilities channel table, never typed in), design 7.2 rows W4 to W6.
- [ ] **Step 2:** run in Blender 5.3; update the results file; row; gate. **Commit.** Subject: `Witness
  velocities, radii and point attributes in Blender`.

### Task 39: Witness, motion blur from velocities and transforms

**Amended 2026-11-14 (A16):** runs in Blender 5.3, after re-checking the Transform Cache constraint fields.

**Kind:** blender. **Tests:** (b).

**Files:**
- Modify: `tools/blender/witness/witness_cases.py`, `measure.py`, the results file

- [ ] **Step 1:** implement the render set-up of design 7.2 "Measuring" (Cycles CPU, at most 64 samples,
  `film_transparent`, orthographic camera along +Y, `ortho_scale` 0.6 m over a 1200 px wide image, emission
  material, `motion_blur_shutter` 0.5) and W3 (`velocity_gives_motion_blur`, points and surface at frame 2) and
  W9 (`transform_gives_object_motion`, solid motion USD and Alembic; JSON in Task 46), design 7.2 rows W3 and
  W9. Images go to a temporary folder, never committed.
- [ ] **Step 0: Re-check in Blender 5.3:** the probe of Task 27 with `--only transform_cache`; expected `TRUE`;
  output in the commit body; a `FALSE` stops the task.
- [ ] **Step 2:** run in Blender 5.3 (record the render time per case in the commit body); update results;
  row; gate. **Commit.** Subject: `Witness motion blur in Blender`.

### Task 40: Witness, grids by name

**Amended 2026-11-14 (A16):** re-checks Sample Grid in Blender 5.3 first; the whole witness runs once, in
Blender 5.3.

**Kind:** blender. **Tests:** (b).

**Files:**
- Modify: `tools/blender/witness/witness_cases.py`, `measure.py`, the results file

- [ ] **Step 1:** implement W10 (`grids_arrive_by_name`, foam volume and gas volume), design 7.2 row W10: grid
  names, `FLOAT` data type, voxel size equal to the lattice spacing, voxel (i, j, k) centred at lattice point
  (i, j, k) from `matrix_object`, and the 64 sample values read through a modifier tree with Blender's Sample
  Grid (`GeometryNodeSampleGrid`, re-checked in Step 0) and Store Named Attribute.
- [ ] **Step 0: Re-check in Blender 5.3:** the probe of Task 27 with `--only sample_grid volume_grids`; expected
  two `TRUE` lines; output in the commit body; a `FALSE` stops the task.
- [ ] **Step 2:** run the **whole** witness in Blender 5.3 (every case but JSON), replacing
  `results_<version>.jsonl`. Row; gate.
  **Commit.** Subject: `Witness volume grids in Blender`.

### Task 41: The format table from the witness results

**Amended 2026-11-14 (A16):** one results file from Blender 5.3; a USD candidate that passes every witness there
is chosen, one that fails is removed.

**Kind:** integration. **Tests:** (a) render output's C++ tests and the add-on's plain tests.

**Files:**
- Modify: `native/fluidity/render_output/format_table.cc`; when a candidate is removed (Step 3) also its writer
  under `native/fluidity/render_output/interchange/`, its tests, `LINK_ROUTE_KEYS`, and, when no USD
  candidate is left, OpenUSD in `conanfile.py`, `conan.lock`, `fluidity_interchange`'s CMake lines and the USD
  resource copy beside the CLI

**Interfaces:**
- Consumes: the Blender 5.3 results file of Tasks 36 to 40, a fresh `capabilities` event.

- [ ] **Step 1:** `uv run python tools/blender/witness/write_witness_records.py
  tools/blender/witness/results/results_<version>.jsonl --witness-folder <folder of Task 34> --capabilities
  <capabilities.json>`. If the results file holds a `NOT_RUN` case, stop: fix the case (its task) and run again.
- [ ] **Step 2:** replace the `WitnessRecord` lines and `chosen` values in `format_table.cc` with the printed
  lines (D12: one result per record).
- [ ] **Step 3: Apply D12's rule, kind by kind, and write down each outcome in the commit body:** (0) a USD
  candidate that passed every witness of its kind in Blender 5.3 is the chosen format of that kind (the printed
  `chosen`); (1) a USD candidate with a `kFailed`: remove its writer, its tests, its key in `LINK_ROUTE_KEYS`, in
  this commit; (2) no USD candidate left: remove OpenUSD from the build and the package (then `capabilities` no
  longer prints `missing_resource_folder` rows for it); (3) a non-USD candidate that failed while another
  candidate of its kind passed every witness: remove it the same way; (4) the last
  candidate of a kind failed and none passed: keep it, change nothing in the writers, list the channels whose
  `witnessed_by` failed in the commit body, and stop to ask the owner (never a silent change of format).
- [ ] **Step 4:** run render output's `FormatTable, ChosenFollowsWitnesses` and `FormatTable,
  EveryChannelIsWitnessed`, the interchange tests, `uv run pytest tests/addon -q`; gate. **Commit** (results files,
  table and removals together). Subject: `Record the Blender witness results in the format table`. Body: one
  line per record changed, the chosen format per kind, the removals, D2's Blender line.

### Task 42: Link routes for points, surface and volume

**Amended 2026-11-14 (A16, A22):** the design's `link` names are back (`output_links.py`, `LINK_ROUTES`), and the
task re-checks in Blender 5.3 every API fact its routes use.

**Kind:** blender. **Tests:** (b) with the witness files.

**Files:**
- Create: `blender/fluidity/output_links.py` (routes), `tests/addon/in_blender/test_link_routes.py`

**Interfaces:**
- Produces design 3.8 `LinkRoute` and `LINK_ROUTES` entries for each key still in `LINK_ROUTE_KEYS` after
  Task 41, among: points Alembic (cache file from `bpy.ops.cachefile.open`, `is_sequence = True`,
  `velocity_name = ".velocities"`, `velocity_unit = 'SECOND'`, a point cloud object from
  `bpy.data.pointclouds.new` with a Mesh Sequence Cache modifier first in its stack, `object_path` from
  `object_paths` matching the object name); points USD (same, no velocity settings); surface Alembic or USD
  (a mesh object, `read_data` including `ATTRIBUTES`); volume OpenVDB (`bpy.data.volumes.new`, `filepath`,
  `is_sequence`, `frame_start`, `frame_duration`). If a witness showed that a route's data path fails while the
  importer works, the route uses the importer used there, restoring frame, selection and active object
  (design 5.7 step 3). Each route has `create`, `repoint`, `undo`.

- [ ] **Step 0: Re-check in Blender 5.3:** the probe of Task 27 with `--only cache_file_open cache_file_fields
  volume_new volume_grids temp_override` (and `importers` if a route uses an importer); expected all `TRUE`;
  output in the commit body; a `FALSE` stops the task.
- [ ] **Step 1: Failing tests** (with the witness folder of Task 34): `test_points_route_evaluates_three_frames` (frames
  1, 2, 10 and back: point count, positions, velocities, radius equal to `expected.json` within the witness equality,
  including the empty foam object); `test_surface_route_attribute`; `test_volume_route_grids_and_origin`;
  `test_routes_match_route_keys` (`set(LINK_ROUTES) == LINK_ROUTE_KEYS`).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject: `Show
  points, surfaces and volumes with Blender's own readers`.

### Task 43: Link routes for solid motion

**Amended 2026-11-14 (A16, A22):** routes live in `output_links.py`; the Transform Cache constraint is re-checked
in Blender 5.3 first.

**Kind:** blender. **Tests:** (b).

**Files:**
- Modify: `blender/fluidity/output_links.py`, `tests/addon/in_blender/test_link_routes.py`

**Interfaces:**
- Produces the solid-motion routes still keyed: Alembic or USD (a copy of the solid's object named `"<name> ·
  FluidiTy motion"` with a Transform Cache constraint on the cache file, `object_path` of `solid_<solid_id>`);
  JSON (the same copy with one location and quaternion key per listed frame from `solid_keys`, hidden in
  viewport and render where `in_scene` is false, constant interpolation for the hide keys).

- [ ] **Step 0: Re-check in Blender 5.3:** the probe of Task 27 with `--only transform_cache action_curves`;
  expected two `TRUE` lines; output in the commit body; a `FALSE` stops the task.
- [ ] **Step 1: Failing tests:** `test_json_motion_keys_equal_poses` (every listed frame);
  `test_transform_cache_route_poses` (only if a cache-file solid route is keyed); `test_copy_keeps_the_artists_object`
  (the original object is unchanged).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject: `Show
  solid motion on a copy of each solid`.

### Task 44: The link transaction and the output registry

**Amended 2026-11-14 (A22):** `apply_link_plan` and `LINK_FAILED` as the design names them; the glossary records
"link" in Blender's sense instead of adding "attach".

**Kind:** blender. **Tests:** (b).

**Files:**
- Modify: `blender/fluidity/output_links.py` (`apply_link_plan`, `read_output_row`)
- Create: `tests/addon/in_blender/test_link_transaction.py`
- Modify: `docs/glossary.md` (`link`: Blender's word for joining a cache file, volume or object to the scene;
  the row qualifies it so the banned sense, a connection between cells, stays banned, A22)

**Interfaces:**
- Produces design 3.8 `apply_link_plan(domain, plan) -> None | AddonFailure` (design 5.7 step 4 (a) to (d),
  `# Step N:` from `<proc-apply-link-plan>`; any exception undoes (c), removes what (a) made, leaves rows
  untouched, returns `LINK_FAILED` with the exception text) and `read_output_row(output_object) -> OutputRow |
  None` (by the custom property `fluidity_output_id`; the only accessor).

- [ ] **Step 1: Failing tests:** `test_failed_plan_changes_nothing` (Review Focus 3: a second plan with a
  truncated file, a wrong `run_fingerprint` and a missing channel: every row, every cache file path, volume path
  and object pointer as before); `test_edited_material_and_added_modifier_survive_relink`;
  `test_existing_object_keeps_name_transform_collections_selection`; `test_read_output_row_is_the_only_way`
  (`ast` over the package: no other code reads `fluidity_output_id`).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject: `Link
  all outputs of a run, or none`. New names: `apply_link_plan`, `read_output_row`.

### Task 45: Output operators and frame ranges that follow the run

**Amended 2026-11-14 (A22):** the operators are `fluidity.link_outputs` and `fluidity.relink_run`, and the
preference `link_outputs_when_ready`, as the design names them.

**Kind:** blender. **Tests:** (b).

**Files:**
- Create: `blender/fluidity/operators_outputs.py`, `tests/addon/in_blender/test_output_operators.py`
- Modify: `blender/fluidity/ui_timer.py`, `blender/fluidity/operators_inspect.py` (`check_outputs`)

**Interfaces:**
- Produces design 3.10 `fluidity.link_outputs` (Show Result: requests the plan, applies it when it arrives,
  time measured and shown), `fluidity.relink_run` (Find Moved Cache...: `report` on the chosen folder must give
  each row's `run_fingerprint`, then one plan with new paths), `fluidity.select_output`,
  `fluidity.check_outputs` (one frame of each linked output evaluated; counts against the `frame_done` file
  entries). The timer applies a waiting plan when `link_outputs_when_ready` is on, and after a `frame_done`
  extends `frame_duration` of volumes and the key range of JSON motion.

- [ ] **Step 1: Failing tests:** `test_show_result_applies_plan`; `test_relink_after_moving_the_folder`;
  `test_relink_refuses_another_run`; `test_volume_range_follows_frame_done`; `test_check_outputs_counts`.
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject: `Show,
  find again and check a run's outputs`.

### Task 46: Witness, solid motion JSON

**Amended 2026-11-14 (A16, A22):** runs in Blender 5.3; the first session links the JSON files with
`apply_link_plan`.

**Kind:** blender. **Tests:** (b).

**Files:**
- Modify: `tools/blender/witness/witness_cases.py`, the Blender 5.3 results file,
  `native/fluidity/render_output/format_table.cc`

- [ ] **Step 1:** implement the JSON route of design 7.2 ("Opening", last row): in a first Blender session the
  add-on (registered from `blender/`) links the JSON files with `apply_link_plan` and saves a `.blend`; a
  second session with `--factory-startup` and the add-on disabled opens it and measures W1 and W9 for
  `solid_motion_json`.
- [ ] **Step 2:** run in Blender 5.3; regenerate the lines with Task 35's script and replace
  the JSON records in `format_table.cc`, applying Task 41's Step 3 rule again. Run the render output format
  tests; row; gate. **Commit.** Subject: `Witness solid motion keys in Blender`.

### Task 47: Starter assets

**Amended 2026-11-14 (A22):** the file is `node_assets.py` again, as the design names it: it holds node groups.

**Kind:** blender. **Tests:** (b).

**Files:**
- Create: `blender/fluidity/node_assets.py`, `tests/addon/in_blender/test_node_assets.py`
- Modify: `docs/glossary.md` (`starter asset`)

**Interfaces:**
- Produces design 5.8's three assets built in code, each with a family id and a version stored on the group:
  point display (radius from `radius` times a scale, a material, raw points kept), fire volume material
  (`density`, `smoke`, `temperature`, `flame`, with temperature and flame scales as inputs), foam on surface
  (mask from foam points within a radius; reads both rows through `read_output_row` and refuses different
  `run_fingerprint`).

- [ ] **Step 1: Failing tests:** `test_asset_built_once_per_version`; `test_edited_group_never_rewritten` (a new
  version is a new group); `test_foam_on_surface_refuses_two_runs` (the test starts from
  `fluidity.link_outputs`, design 5.12); `test_fire_material_reads_the_four_grids`.
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject: `Add three
  starter assets for points, fire and foam`.

### Task 48: Packaging

**Amended 2026-11-14 (A16):** re-checks the extension commands in Blender 5.3 first, builds with 5.3, and the
package's manifest keeps `blender_version_min = "5.3.0"`.

**Amended 2026-11-15 (A48, A49):** no maximum version; the windows-x64 and linux-x64 packages ship the CUDA runtime
the executable was built with in `bin/<platform>/` and `licenses/cuda-runtime.txt`; three packaging tests.

**Kind:** integration. **Tests:** (a) and (b) (`extension validate` in Blender).

**Files:**
- Create: `tools/blender/build_extension_packages.py`, `tests/addon/logic/test_build_extension_packages.py`,
  `tests/addon/packages/test_packages.py` (2026-11-15: the three tests on built packages; run by hand or in CI where a
  package is built, not in the gate)

**Interfaces:**
- `build_extension_packages.py --blender <exe> --platform <name> --fluidity <exe> --out <folder>`: copies
  `blender/fluidity/` to a temporary folder, adds `bin/<platform>/fluidity[.exe]`, `bin/<platform>/build.json`
  (`executable_version` from `capabilities`, SHA-256 of the executable) and the USD resource folder beside it
  while a USD candidate is in the build; writes `platforms = ["<platform>"]` into the copy's manifest (never
  changes `blender_version_min`); runs `blender --offline-mode --factory-startup --command extension build
  --source-dir <copy> --output-dir <out>` and `... extension validate <zip>`. One package per platform; this Mac
  builds `macos-arm64` only (the others in CI or on the owner's machines: "written, not run" for them).
- **Amended 2026-11-15 (A49; design 5.10):** option `--cuda-runtime <file>`, required for `windows-x64` and
  `linux-x64`, refused for `macos-arm64`: the path the build records (Linux: `CUDA::cudart`'s imported location,
  resolved through its links to the real file, copied as `bin/linux-x64/libcudart.so.<major>`, its SONAME; Windows:
  the toolkit's `bin/cudart64_<major>.dll`, copied under that name), plus `--cuda-toolkit <folder>` whose
  `version.json` gives `cuda_cudart.version`. `build.json` gains `cuda_runtime_file` and `cuda_runtime_version`
  (empty on macOS). `licenses/cuda-runtime.txt` at the package root: "Contains the NVIDIA CUDA Runtime <version>,
  redistributed under the NVIDIA CUDA Toolkit End User License Agreement, Attachment A (redistributable software)."
  then the EULA link `https://docs.nvidia.com/cuda/eula/` (the owner confirms the wording, plan-decisions owner
  question 4). No maximum version is ever written (A48). The Linux executable's `RUNPATH $ORIGIN` and the shared
  runtime link (`CMAKE_CUDA_RUNTIME_LIBRARY Shared`) come from the build (Foundation and I/O's CLI target); this
  task only checks them. **Why CPU runs never depend on a GPU library:** the one GPU library the executable links,
  the runtime, ships beside it, so the executable always starts; the driver (`nvcuda.dll`, `libcuda.so.1`), the only
  GPU library not shipped, is loaded by the runtime at the first CUDA call, which only `CheckCanRunOn(RunOn::kGpu)`
  makes; a CPU run never makes it, and without a driver that call returns an error while CPU runs go on.

- [ ] **Step 0: Re-check in Blender 5.3:** the probe of Task 27 with `--only extension_commands`; expected `TRUE`;
  output in the commit body; a `FALSE` stops the task.
- [ ] **Step 1: Failing tests:** `test_manifest_of_one_platform` (design 5.10 fields with
  `blender_version_min == "5.3.0"` and no maximum; no network permission; `[permissions] files` with one short
  reason); `test_build_json_digest`; `test_no_tests_or_tools_in_package`; 2026-11-15:
  `test_cuda_runtime_option_per_platform` (fake files: `linux-x64` copies the link target under
  `libcudart.so.12`, a regular file; `windows-x64` copies `cudart64_12.dll`; `macos-arm64` with `--cuda-runtime`:
  refused; `build.json` and `licenses/cuda-runtime.txt` name version `12.4.127` from a fake `version.json`).
- [ ] **Step 1b: Package tests** (2026-11-15, A49; design 7.1 "Packaging"), in `tests/addon/packages/`:
  `test_package_contains_cuda_runtime` (windows-x64 and linux-x64 zips: the runtime file is a regular zip member in
  `bin/<platform>/`; `build.json`'s `cuda_runtime_version` equals the `runtime` of
  `ReadProcessorDescription(RunOn::kGpu)` printed by that executable on its build machine; the notice names that
  version; the macos-arm64 zip has neither); `test_linux_executable_runpath_is_origin` (`readelf -d` of
  `bin/linux-x64/fluidity`: exactly one `RUNPATH` `[$ORIGIN]`, no `RPATH`);
  `test_packaged_executable_runs_cpu_scene_without_driver` (in the container, which has no GPU driver: the unpacked
  linux-x64 package runs `fluidity check` and `fluidity run --last-frame 1` of I/O's `small_syrup.json` with `run_on`
  cpu, both exit 0; `ldd bin/linux-x64/fluidity` resolves `libcudart.so.<major>` inside the package and lists no
  `libcuda.so.1`). On this Mac they are "written, not run" (no linux-x64 or windows-x64 package is built here); the
  commit says so (D2).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** build the `macos-arm64` package with `--blender
  /Applications/Blender.app/Contents/MacOS/Blender` (5.3) and validate it; record the package SHA-256 in
  `docs/blender-runs.md`; gate. **Commit.** Subject: `Build one extension package per platform`.

### Task 49: Background Blender with the real native program

**Amended 2026-11-14 (A16, A20):** runs in Blender 5.3; the fire scene's file carries the wind keys read from its
Wind force field.

**Amended 2026-11-15 (A47):** the scene field is `wind` (was `gas_wind`).

**Kind:** integration. **Tests:** (b) with the real `fluidity` on the CPU.

**Files:**
- Create: `tests/addon/in_blender/test_with_fluidity.py`

- [ ] **Step 1: Tests** (design 7.1, headless list): `test_reference_scenes_check_clean` (each preset: created,
  read, quick check clean, `fluidity check` exit 0 with no `unknown_scene_field`; the same with every optional
  field written; for the fire scene the written `wind` keys equal those of I/O's `small_fire.json`);
  `test_small_syrup_bake_pause_resume_link` (6 frames, pause at 3, resume, link, point counts
  equal to the `frame_done` file entries); `test_surface_spacing_rebuild_changes_only_surface_files` (Rebuild
  Outputs: file digests of every other output unchanged); `test_smoke_over_water_one_plan` (liquid and gas in
  one domain, A10, 4 frames: surface and gas volume linked in one plan).
- [ ] **Step 2: Run** `uv run python tools/blender/run_blender_tests.py --blender
  /Applications/Blender.app/Contents/MacOS/Blender --fluidity <exe> --only test_with_fluidity.py`. Fix owning commits.
  Row; gate. **Commit.** Subject: `Bake, pause, resume and show the reference scenes in background Blender`.

### Task 50: The interactive session

**Amended 2026-11-14 (A16):** one window session, in Blender 5.3.

**Kind:** integration. **Tests:** (c).

**Files:**
- Modify: `docs/blender-runs.md` (the checklist results)

- [ ] **Step 1:** in a Blender 5.3 window (`/Applications/Blender.app`), with the add-on installed from the
  package of Task 48, record pass or fail with a note for each check of design 7.1 "Needs a real Blender
  session": every panel draws in every state, each role assigned and unassigned, at a 424 px wide Properties
  editor, no error in the console; the largest tick time during a syrup bake (from a debug print enabled for the
  session only, never committed) under 4 ms target, recorded; scrubbing during `running` shows frames up to
  `last_listed_frame`; a viewport render and a final render with motion blur.
- [ ] **Step 2:** if no one can open a window, write "written, not run in Blender (no one at the screen)" for each
  check; never record a check as passed that no one watched. Gate.
  **Commit.** Subject: `Record the interactive Blender session`.

### Task 51: The artist's journey on the four scenes

**Amended 2026-11-14 (A16, A22):** runs in Blender 5.3 after re-checking the extension commands; step 1 says
"install", Blender's word.

**Kind:** integration. **Tests:** (b) for every step a background session can do; (c) where noted.

**Files:**
- Create: `tools/blender/run_artist_journey.py`, `tools/blender/artist_journey_steps.py` (the per-session
  scripts it starts)
- Modify: `docs/blender-runs.md` (one receipt per scene and Blender build: commit, package SHA-256, Blender
  version and build hash, platform, processor, result per step)

**Interfaces:**
- `run_artist_journey.py --blender <exe> --package <zip> --scene wave|syrup|fire|snow --work <folder>`: each step
  is its own Blender process; the clean profile is `BLENDER_USER_RESOURCES` set to an empty folder under
  `<work>` (check in the first step that `bpy.utils.resource_path('USER')` points into it); every process uses
  `--offline-mode`.

- [ ] **Step 0: Re-check in Blender 5.3:** the probe of Task 27 with `--only extension_commands`; expected `TRUE`;
  output in the commit body; a `FALSE` stops the task.
- [ ] **Step 1:** implement the nine steps of design 7.1 "Acceptance journey": (1) install the package from disk
  with `blender --offline-mode --command extension install-file -r user_default -e <zip>`; (2) enable, disable,
  enable (`bpy.ops.preferences.addon_disable`/`addon_enable` of the extension module); (3) Add Reference Scene,
  Check, Bake (CPU, the small sizes of D5, 6 frames); (4) Pause after frame 3: the three facts read from the
  status; (5) Resume to the end and render frames 2 and 5; (6) save, start a second run, end the Blender process
  normally during it, reopen: the run is `stopped` with the frame in progress listed, Resume works; (7) edit a
  viscosity: `needs_new_run` names "Viscosity", New Run keeps the old outputs; edit the surface spacing (wave and
  syrup): Rebuild Outputs; (8) move project and run folder together, reopen, scrub; move only the run folder,
  Find Moved Cache; (9) start Blender with the add-on disabled and `--disable-autoexec`, scrub frames 5 to 0 and
  back, render frames 2 and 5, compare with step 5's images: largest channel difference at most 1/255.
- [ ] **Step 2: Run** for wave, syrup, fire and snow with `--blender /Applications/Blender.app/Contents/MacOS/Blender`
  (5.3); GPU runs belong to the owner's hardware run (write the command in the receipt). Fix owning commits for any
  failure. Receipts; gate. **Commit.** Subject: `Walk the artist's journey on the four reference scenes`.

### Task 52: Comment pass and hand-over

**Amended 2026-11-14 (A16, A20):** the hand-over covers D13 and reports Blender 5.3 runs only.

**Amended 2026-11-15 (A47 to A49, A53, A57 to A60, A63, A64):** the hand-over lists the 2026-11-15 amendments as
delivered, the package tests run or "written, not run", and states that Task 53 is a later group outside the
done-criteria.

**Kind:** mechanical.

**Files:**
- Modify: any file of `blender/`, `tools/blender/`, `tests/addon/`, `native/fluidity/cli/write_witness_files.cc`,
  `witness_json.*`, `docs/method-paper/sections/blender.typ`, `docs/glossary.md`, `docs/blender-runs.md`,
  `docs/gpu-runs.md`

- [ ] **Step 1: Comment pass.** Reread every file as a programmer who knows Python but has never used Blender's
  API or read a fluid paper: each docstring true of the final code; why, not what; a worked example with numbers
  for `apply_event` (the frame 38 example), `build_link_plan` (one declaration to one step),
  `write_witness_records` (one record line) and `build_wind_keys` (one Wind field of strength 2.5 turned 90
  degrees about X giving `(0, -2.5, 0)` m/s2); every word in the glossary; header labels complete; `# Step N:`
  numbers match `<proc-apply-event>` and `<proc-apply-link-plan>`.
- [ ] **Step 2: Fold each change into its owning commit** (`git commit --fixup`, `git rebase --autosquash`); run
  format and rules on each rewritten commit and the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3 to 10) and record what was delivered differently: D1 to
  D13 (D13 replaces the domain's `wind_acceleration`); every route that uses an importer instead of data (and
  why); the D4 cases the schema could not carry; the witness outcomes and removals of Tasks 41 and 46; the
  force field table of Task 27a; every probe check that was false in Blender 5.3 and what changed because of
  it; every "written, not run in Blender" line still open, with its reason; (2026-11-15) every scene file path of
  design 3.2's NEW rows the branch schema names differently (D4), whether the three package tests of Task 48 ran
  and where, and the Open items of design section 11.
- [ ] **Step 4: `docs/gpu-runs.md`:** an `addon` row: no device code of its own; the GPU part of the artist's
  journey (step 3 with `run_on = GPU`) is listed for the owner's hardware run.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the module, the gate output, the Blender
  build each test group ran on (Blender 5.3.x with its hash), new public names with one-line reasons, Step 3's
  list, and the statement that no device code has run and whether every witness case ran in Blender 5.3.

---

## Deep liquid without particles (later; needs review)

**Amended 2026-11-15 (A64):** added. These tasks are excluded from this module's done-criteria: the module is done
without them, and Task 52's hand-over does not wait for them. An independent review of this design comes first.
The owner of the work is the cell resize and batches plan's group "Deep liquid without particles (later; needs
review)"; nothing here starts before that group's two gate tests pass: `GridOnlyRegion,
StandingWave_KeepsAmplitude` and `GridOnlyRegion, ParticleBandEdge_StaysStable`.

### Task 53: The grid-only depths in the add-on (later)

**Amended 2026-11-15 (A64):** added, holding what A64 took out of Tasks 6, 7, 8, 26 and 32.

**Kind:** logic and blender. **Tests:** (a), (b).

**Files:**
- Modify: `blender/fluidity/logic/scene_description.py`, `logic/scene_fields.py`, `logic/scene_file.py`,
  `properties.py`, `panels.py`, `tests/addon/logic/test_scene_description.py`, `test_scene_fields.py`,
  `test_scene_file.py`, `tests/addon/in_blender/test_properties.py`, `test_panels.py`

**Interfaces:**
- Consumes: the scene file fields the cell resize group defines for the grid-only region (today, before review:
  `cell_resize.regions {particle_region_depth_in_cell_widths, grid_only_depth_in_cell_widths}`; the reviewed names
  win).
- Produces: `CellResizeDescription` regains `particle_region_depth_in_cell_widths` and
  `grid_only_depth_in_cell_widths` (no unit, cell widths); `FluidityDomain` regains the two `*` properties in the
  Solver Detail panel's cell resize rows; their `SCENE_FIELDS` rows (SOLVER_DETAIL, SIMULATION; tooltips written from
  the reviewed design); the writer writes `cell_resize.regions` only when both are set. The design text marked "not
  in the first version (A64)" becomes current.

- [ ] **Step 0:** confirm the cell resize group's two gate tests pass on the branch and the review's names; if not,
  stop and report.
- [ ] **Step 1: Failing tests:** `test_grid_only_depths_written` (depths 8 and 24: `cell_resize.regions` with those
  values; unset: no `regions`); `test_grid_only_rows_in_the_schema` (the two rows' paths are in the schema);
  (b) `test_grid_only_depths_drawn` (Solver Detail draws both with cell resize on).
- [ ] **Step 2:** run (fail). **Step 3:** implement. **Step 4:** run; row; gate. **Commit.** Subject: `Let the artist
  set the deep liquid without particles`.

