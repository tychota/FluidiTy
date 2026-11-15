# I/O and CLI module design

Status: updated on 2026-11-14 with the owner's answers (A1 to A14) and `changes-to-apply.md`. Module 10 of the
build order ("I/O and CLI", file 12 of this folder). Written against the updated designs of contact, grid,
particles, transfer, pressure, viscosity, materials, solids, gas, step, cell resize and batches, and surface
with foam and spray. Updated again on 2026-11-14 with A15 (batched solves), A17 and the plan decisions "Planned
cost shown to the user, and buffers in batched runs" and "I/O and CLI" of `plan-decisions.md` (section 11,
decisions 14 to 22). Updated again on 2026-11-14 with A24, A25 and the reconcile pass's coordinator item (3)
(decisions 23 to 27), and with A27 (decisions 28 and 29).

## 1. Purpose

A run reads one scene, advances the simulation frame by frame, and leaves files behind: products to play back
and states to continue from. This module turns the artist's scene into the numbers the step module runs on,
makes every file appear whole or not at all, decides whether an old run may be continued, answers what a run
folder holds, and is the `fluidity` program that the add-on starts and listens to.

Owns: the `scene_file` and its schema; filling defaults once and recording them (`RunScene`, which holds the
`step::ReadyScene`); the `run_folder` with its `file_list` and `WriterLock`; `OutputFile`, the sink every file
is written into before it is listed; `WriteListedFile`, the one way a file becomes visible ("listed");
FluidiTy's own array file, used only for what no open format fits (A14): `saved_state` and the internal run
records (`ready_scene`, the stored frame inputs that render output rebuilds from); the output list (one
`OutputDeclaration` per product); the `run_fingerprint`, the output fingerprints and the rule for continuing;
the output queue and its waiting rule; the run report and the read-only query of a run folder; the frame loop
(`AdvanceRunTo`); the CLI commands, events, exit codes and stop requests; the text and stable code of every
failure shown to a user.

Does not own: any physics or the step loop inside one frame (`step::AdvanceTo`); what a snapshot contains
(each module's snapshot type); any open-format file (A14): Alembic, USD, OpenVDB and the solid motion file,
including the playback of the simulation particles, are written by render output's writers (target
`fluidity_interchange`, linked by the CLI only) into an `OutputFile` this module opened; the format table and
the channel table (render output); Blender panels, the Python that writes the scene file, the witness tests and
the linking of files to Blender objects (Blender add-on).

## 2. What exists today

| Where on `main` | What it does | What is wrong |
|---|---|---|
| `native/cli/simulate_execution.cc:597-1375` `ExecuteCommand` | Run, resume, replay, step and preview in one function of 778 lines | Naming review A2. Five commands share one body through flags |
| `simulate_execution.cc:719-805` | Opens the cache, compares the scene fingerprint, compares the CUDA device and driver record, takes the job lease | Kept as behaviour. Three separate checks with three file formats (`resolved.flrs`, `FLCUDA1` text, `.writer.lock`) |
| `simulate_execution.cc:37, 40, 1196, 1332` | 24 frames per second; limits of 1024 particles, 128 blocks, 4096 cells, 24576 faces; a saved state every tenth frame | Constants in the CLI. The limits are the ones the grid design removes |
| `native/cli/public_cli.cc:14-42, 142-166, 191-225` | Seven commands parsed by CLI11, then rewritten as "legacy" arguments for a second hand-written parser | Two parsers. `validate` and `inspect` map to one legacy `check` |
| `native/cli/fluidity_simulate.cc:869-917` | `Main`: hidden commands `cutter-frame`, `rigid-frame`, `blender-resolve`; exit codes 0, 2 (bad argument), 1 (all else); error code `E-<domain>-<Abseil status name>` (`:808-820`) | Cutter and free box run one process per frame outside the job protocol. The error code is an Abseil name |
| `native/cli/simulate_job.cc:18-20, 111-135, 216-229, 262-311` | Heartbeat every 500 ms, stdin polled every 25 ms for `cancel`, `frame` event with `published:true`, a status marker file (`v1 baking`) | Good rules, ported. The heartbeat has no sequence number and no "last advance" time (report, Blender "Measure the user's wait") |
| `native/cli/simulate_cuda_compatibility.h:17-34` | Device UUID, driver and runtime versions as an "execution identity" | One of seven meanings of `identity` (section 9) |
| `native/cli/blender_resolve.cc:216-223` | Reads five versions of an ASCII text packet (`FLUIDITY-BLENDER-1` to `5`) and writes `scene.flrs` plus a start state | The scene the artist set up has no schema; defaults live in Python and in C++ |
| `native/src/liquid_resolved_scene_3d.cc` (1148 lines), `schemas/simulate/resolved_scene.fbs` | Canonical FlatBuffers bytes of the resolved scene and their BLAKE3 hash | One liquid material table, one obstacle, a `OpenTank` with a piston; snow in a side table; gas absent. Canonical bytes are fragile (`schemas/simulate/README.md`: a six-byte difference between two encoders) |
| `native/src/liquid_cache_3d.cc` (1100 lines), `schemas/simulate/common.fbs:15-29` | Codecs. One FlatBuffers table per particle | 2^20 records at most (README). A FlatBuffers buffer cannot pass 2 GB |
| `native/src/liquid_cache_store_3d.cc:590-697, 703-832, 459, 873` | Create, open, `PublishFrame`, prune, `ReadCheckpoint` | Good transaction rules, ported. Three artefacts per frame (blob named by digest, binary manifest, JSON mirror) and a reference file per saved state |
| `native/src/liquid_cache_files_3d.cc:131-133, 224, 258` | `.writer.lock`, promote by rename without replace, flush | Ported |
| `native/src/solid_interaction_checkpoint_3d.cc` (509 lines) | A second saved state (`FLSI` versions 1 to 3) for free boxes and cut solids | A second route (solids design, section 10) |
| `schemas/effects/motion_frame.fbs`, `host/motion_capture_3d.h` | Up to 65 whole states per frame as nested byte blobs, for foam and motion blur | Replaced by render output's motion samples (its 5.2) |
| `tools/blender/point_cache_writer/blender_bridge.cc` | Alembic and USD written from the cache by a separate tool, the whole range on every attach | A14: every playback file is an open-format file written once per frame by render output's writers |
| `git show origin/codex/m7-low-mach-gas:native/cli/gas_execution.cc` (448 lines) | Own scene (`FLGS`), own saved state (`.flgc`, every frame), own lock call (`:244-246`), volume values printed as JSON (`:83-104`) | A third route. Programme design row 7b: not ported |
| `blender/fluidity/jobs.py:57-63, 66-149, 177-237` | `Job` reads protocol version 1 lines (queue of 64, 16 per poll, kill 10 s after `cancel`); `FrameJob` starts one process per frame | The consumer of the protocol. `FrameJob` has no heartbeat and cannot cancel a running frame |

Y-up on `main`: `HorizontalPlane.surface_height_y`, `OpenTank.floor_y` and gravity along -Y in the schema; the
add-on turns Z-up into Y-up when it writes a packet and back when it reads points. Here every file of this
module is Z-up, right-handed, in metres, seconds and kilograms, and converts no axis. The one axis conversion
left is inside render output's Alembic writer.

## 3. Public interface

Folders `native/fluidity/io/` (namespace `fluidity::io`) and `native/fluidity/cli/` (namespace
`fluidity::cli`); schemas in `schemas/simulate/v2/`. Everything is CPU only. `io/` may include
`render_output/` headers (`format_table.h`, `output_channels.h`, `render_frame.h`) but never
`render_output/interchange/`; only `cli/` links `fluidity_interchange`. Names are NEW unless marked D
(DECIDED). Paths are `std::filesystem::path`, shortened to `Path` below. Every fallible function returns
`std::expected<T, IoFailure>` (A2).

### 3.1 `io/io_error.h`, `io/io_message.h`, `io/stable_names.h`

```cpp
enum class IoError : std::uint8_t {
  kNone,
  kBadInput,              // An argument is out of range or two arguments disagree.
  kFileNotFound,          // A named or listed file is missing.
  kFileNotReadable,       // The system refused to open or read it.
  kFileNotWritable,       // The system refused to create, write, flush or rename.
  kNoSpaceLeft,           // The disk is full. Retrying does not help.
  kWrongFileKind,         // Not a FluidiTy file, or another kind than asked for.
  kWrongSchemaVersion,    // Written by a schema this executable does not read. magnitude = found, limit = read.
  kFileDamaged,           // Too short, a header that fails its check, or an array outside the file.
  kDigestDiffers,         // The bytes are not the ones the file list names.
  kSceneTextNotValid,     // The scene file is not valid text for the schema. detail holds line and reason.
  kUnknownSceneField,     // NEW. A field the schema does not have. detail holds its path.
  kSceneValueOutOfRange,  // A scene value is not finite or outside its range. detail holds the field path.
  kUnknownPreset,         // A preset name this executable does not have.
  kDuplicateId,           // Two scene objects of one kind share an id, or an id is 0. detail holds the path.
  kRunFolderInUse,        // Another process holds the writer lock.
  kRunFolderNotEmpty,     // A new run was asked for in a folder that already has a file list.
  kNoRunInFolder,         // No file list to continue from.
  kWrongRunFingerprint,   // The run in the folder is not this run. differing_part says which input differs.
  kNoSavedState,          // No listed saved state at or before the asked frame passes its checks.
  kSavedStateDoesNotMatchScene,  // step::kSnapshotDoesNotMatch while building from a saved state.
  kFrameDiffersOnResume,  // A frame computed again after a resume has another snapshot digest than the listed one.
  kFileAlreadyListed,     // The name is listed with other bytes. Listed files are never replaced.
  kFrameInputsNotKept,    // NEW. A rebuild or read-particle needs the stored frame inputs of a frame that has none.
  kUnknownOutput,         // NEW. An output id the run's output list does not have.
  kParticleNotFound,      // NEW. No particle with that id in that frame.
  kOutputStopped,         // The output worker failed earlier; its failure is the one reported.
  kEventsNotWritable,     // Standard output is closed: whoever started the run is gone.
  kOutOfMemory,
  kModuleFailed,          // Another module failed. module and module_error identify it.
};
// NEW. Who failed, for ToMessage and ToName at the boundary. Wider than step::Module: it also names modules the
// step does not call (cell resize, batches) and this one.
enum class FailingModule : std::uint8_t { kIo, kBase, kContact, kGrid, kParticles, kTransfer, kPressure,
                                          kViscosity, kMaterials, kSolids, kGas, kStep, kCellResize, kBatches,
                                          kRenderOutput };
constexpr FailingModule ToFailingModule(step::Module module);
enum class FingerprintPart : std::uint8_t { kNone, kScene, kExecutable, kRunOn, kProcessor, kRunMode };
struct IoFailure {                      // Compact. detail is cut to fit, never allocated.
  IoError error = IoError::kNone;
  FailingModule module = FailingModule::kIo;   // With kModuleFailed, the module that failed.
  std::uint8_t module_error = 0;               // That module's error enum value.
  FingerprintPart differing_part = FingerprintPart::kNone;
  std::int64_t frame = -1;              // -1 when not about one frame.
  std::uint64_t material_id = 0, source_id = 0, particle_id = 0;   // Scene ids, 0 when none.
  std::uint32_t solid_id = 0;
  double magnitude = 0, limit = 0;
  std::array<char, 120> detail{};       // A scene file path, a file name or a parser message. Zero-terminated.
};
std::string_view ToMessage(IoError error);
// NEW. The stable code: the enumerator's name without the k, in lower snake case ("run_folder_in_use").
std::string_view ToName(IoError error);
// The one place that turns any module's failure into a sentence and a stable code for a user (shared-interface
// (c)). Each calls that module's ToMessage or ToName; base errors use this module's own name table.
std::string_view ToMessage(FailingModule module, std::uint8_t module_error);
std::string_view ToName(FailingModule module, std::uint8_t module_error);   // "not_converged"
std::string_view ToName(FailingModule module);                              // "pressure", "cell_resize"

// stable_names.h. NEW. Lower snake case names of the step's enums that reach events (CapabilityReason,
// SuggestedChange, RejectReason, Stage, StepLimit, RunMode, Module, ViscousPath, RunOn). One table per enum, sized
// by the enum's count (step::kCapabilityReasonCount, step::kStageCount, ...); a test fails when an enumerator has
// no name.
// step::kCapabilityReasonCount names: 16 at most, 13 if the viscosity plan lifts its three exclusions (A24: no
// kViscousLiquidWithSnow, no "viscous_liquid_with_snow"; A27: no kViscousLiquidWithFreeSolid, no
// "viscous_liquid_with_free_solid"). step::kSuggestedChangeCount names: 12 (A27: no kSetViscosityToZero, no
// "set_viscosity_to_zero"; the sand module adds it back with kViscousLiquidWithSand).
std::string_view ToName(step::CapabilityReason reason);   // "thin_solid_inside_one_cell"
std::string_view ToName(step::SuggestedChange change);    // "use_smaller_cells"
std::string_view ToName(step::RejectReason reason);
std::string_view ToName(step::Stage stage);               // "solve_viscosity_and_pressure"
std::string_view ToName(step::StepLimit limit);
// Plan decision (same table rule): "one_piece", "batched", "batched_with_solves_in_pieces" (A15: three values).
std::string_view ToName(step::RunMode mode);
// NEW (reconcile). Lower snake case of the enumerator without k, step::kModuleCount = 14 names: kCellResize ->
// "cell_resize", kRenderOutput -> "render_output". Used by the budget's module names.
std::string_view ToName(step::Module module);
// NEW (A25). kPressureSolve -> "pressure_solve", kViscousSolve -> "viscous_solve". RunBudget::viscous_path in JSON.
std::string_view ToName(step::ViscousPath path);
std::string_view ToName(RunOn run_on);                    // "cpu", "gpu"
```

`io/material_presets.h` (plan decision, decision 12): `enum class MaterialPreset : std::uint8_t { kCustom, kWater,
kSyrup, kSnow, kSpringySolid, kAir };`, `struct MaterialPresetValues`, `inline constexpr
std::array<MaterialPresetValues, 6> kMaterialPresets`, `std::optional<MaterialPreset> FindMaterialPreset(std::string_view
name)`. The sand module appends `kSand` after `kAir`. `kWater`: 1000 kg/m3, **0 Pa s** (A25; was 1e-3), 0.072 N/m:
a water scene uses the pressure solve.

### 3.2 `io/run_scene.h`: from `scene_file` to `ready_scene`

```cpp
struct FrameRange {
  std::int64_t first_frame = 0, last_frame = 0;  // D word frame. Inclusive. first_frame >= 0.
  double frames_per_second = 24;                 // 1/s, positive.
  double start_time = 0;                         // s. Time of first_frame.
};
double ComputeFrameTime(const FrameRange& frames, std::int64_t frame);  // start_time + (frame - first) / fps.

struct RequestedOutputs {                        // The presentation part of the scene: not in the run fingerprint.
  std::vector<std::uint64_t> points_material_ids;  // NEW. Materials written as points, ascending. Never gas.
  particles::FieldList particle_fields{};        // Beyond particles::kOutputSnapshotFields, for the points files.
  bool writes_solid_motion = false;              // NEW. One solid motion file per frame.
  render_output::RenderSettings render{};        // NEW. Surface, foam and spray, volumes, the motion lattice.
  bool keeps_frame_inputs = true;                // NEW. Store each frame's inputs (5.5): needed by `run --outputs`
                                                 //   and `read-particle`.
  std::uint32_t saved_state_every_frames = 10;   // Ported from main. first_frame is always saved.
  std::uint32_t saved_states_kept = 2;           // Ported from main.
};
bool RecordsSteps(const RequestedOutputs& outputs);  // NEW. Surface, foam or a volume asked: RecordStep runs.

enum class ObjectRole : std::uint8_t { kDomain, kMaterial, kIngredient, kStartRegion, kSource, kSolid, kGasSource };
struct ObjectName { ObjectRole role{}; std::uint64_t id = 0;   // The id of that kind in the scene file.
                    std::string name; };                       // The Blender name, for messages only.
enum class DefaultSource : std::uint8_t { kPreset, kEngineDefault, kDerived, kCommandLine };
struct FilledDefault { std::string scene_path; std::string value; DefaultSource source{}; };

struct RunOverrides {                            // From the command line. Empty: the scene file decides.
  std::optional<RunOn> run_on;
  std::optional<std::int64_t> last_frame;
  // Plan decisions. Memory the plan is given instead of ReadMemoryAvailable (processor; CPU). check
  // --assume-memory and --assume-cpu-memory; tests use them to force a batched plan. In no fingerprint.
  std::optional<std::uint64_t> assume_memory_bytes;
  std::optional<std::uint64_t> assume_cpu_memory_bytes;
};
// Owns every vector and key array that step::ReadyScene borrows. Move-only. The ready scene stays valid while
// this lives.
class RunScene {
 public:
  const step::ReadyScene& ready_scene() const;         // D word.
  const step::StepSettings& step_settings() const;
  const std::optional<cell_resize::CellResizeSettings>& cell_resize_settings() const;   // NEW. Empty: no resize.
  const batches::BatchSettings& batch_settings() const;                       // NEW.
  const materials::MaterialTable& material_table() const;   // NEW. Built once, for checks and export names.
  const FrameRange& frames() const;
  const RequestedOutputs& outputs() const;
  std::span<const ObjectName> names() const;           // By role, then ascending id.
  std::span<const FilledDefault> filled_defaults() const;
  Digest scene_digest() const;                          // 3.4. Of the physical part only.
  Digest scene_file_digest() const;                     // Of the scene file's bytes as written.
  std::span<const std::uint8_t> ready_scene_bytes() const;  // The listed ready_scene file's description table.
};
// NEW. Every value finding of a scene file at once (the add-on shows all problems together): kUnknownSceneField
// (one, FlatBuffers stops at the first), kSceneValueOutOfRange, kUnknownPreset, kDuplicateId, and the failures
// of materials::BuildMaterialTable, gas::CheckGasSources and render_output::CheckRenderSettings as
// kModuleFailed. Empty: BuildRunScene succeeds unless a file or memory fails. Touches no processor.
std::vector<IoFailure> CheckSceneFile(const Path& scene_file, const RunOverrides& overrides);
// Reads the scene file, checks every value, fills defaults once, derives (5.1), samples the start particles.
// Fails with the first finding of CheckSceneFile, or kFileNotFound, kFileNotReadable, kSceneTextNotValid,
// kOutOfMemory. Touches no processor and allocates no simulation buffer.
std::expected<RunScene, IoFailure> BuildRunScene(const Path& scene_file, const RunOverrides& overrides);
// The same object from the ready scene a run folder lists. Fills no default: every value is in the file.
std::expected<RunScene, IoFailure> ReadRunScene(const Path& run_folder, const FileList& file_list);
```

### 3.3 `io/run_folder.h`: file list, writer lock, `OutputFile`, the one way a file gets listed

```cpp
struct Digest {                                  // BLAKE3-256 of bytes.
  std::array<std::uint8_t, 32> bytes{};
  friend constexpr bool operator==(const Digest&, const Digest&) = default;
};
struct RunFingerprint {                          // D word. Its own type so it is never mixed with a file digest.
  Digest value{};
  friend constexpr bool operator==(const RunFingerprint&, const RunFingerprint&) = default;
};
// Array files (FluidiTy's own format, 5.3) and open-format product files (render output's writers).
enum class FileKind : std::uint8_t { kReadyScene, kSavedState, kFrameInputs, kOutputFile,
                                     kStartParticles };   // Plan decision: appended, the scene's start_particles_file.
struct OutputChannelInFile {                     // NEW. One row of render output's channel table, as written.
  std::string blender_name, name_in_file;
  bool is_witnessed = false;                     // Its witness passed for this format in Blender 5.3 (A16).
};
struct OutputDeclaration {                       // NEW. One product of a run: what one Blender object shows.
  std::string output_id;                         // Its folder under outputs/: "points_17", "surface_3",
                                                 //   "droplets_3", "foam_and_spray", "foam_volume", "gas_volume",
                                                 //   "solid_motion" (render output 5.7 file table).
  render_output::OutputProduct product{};
  std::uint64_t material_id = 0;                 // Points: the material. Surface, droplets: the group's lowest.
  render_output::InterchangeFormat format{};     // render_output::ReadFormatChoice(kind).chosen at the start.
  std::string file_pattern;                      // "outputs/points_17/points.{frame:06d}.abc". Relative.
  std::vector<std::string> object_names;         // "points"; "spray", "foam", "bubbles"; "solid_<solid_id>".
  std::vector<OutputChannelInFile> channels;     // Only channels the file carries.
  Digest output_fingerprint{};                   // 5.2. Changes when its settings or the simulation change.
  bool needs_simulation_to_rebuild = false;      // Foam and spray, and any product of a run without frame inputs.
  std::uint32_t contract_version = render_output::kOutputContractVersion;
};
struct ListedFile {
  FileKind kind{};
  std::int64_t frame = -1;                       // -1 for kReadyScene.
  double time = 0;                               // s. Committed time the content describes.
  std::uint64_t state_version = 0;               // D.
  std::string output_id;                         // NEW. kOutputFile only.
  std::uint64_t byte_count = 0;
  Digest digest{};                               // Of the whole file.
  std::string relative_name;                     // "outputs/points_17/points.000042.abc". Never an absolute path.
  std::uint64_t sequence = 0;                    // Position of its entry in the file list.
  // NEW. kOutputFile of a surface or foam product: the numbers of its SurfaceReport or FoamReport by field name
  // (render output 4, item 5). Empty otherwise.
  std::vector<std::pair<std::string, double>> product_report;
};
struct NotBuiltOutput {                          // NEW. A product asked for and not built in one frame (render output 5.7).
  std::string output_id;  std::int64_t frame = -1;
  std::string reason;                            // ToName of the render output error ("too_many_vertices").
};
struct FrameDone {                               // NEW. One per complete frame, ascending.
  std::int64_t frame = -1;  double time = 0;  std::uint64_t state_version = 0;
  Digest snapshot_digest{};                      // Of the frame's snapshot arrays as packed in 5.5. The check of a
                                                 //   frame computed again after a resume (5.6 step 7.6).
};
struct FileList {                                // D word. Which files belong to a run.
  RunFingerprint run_fingerprint{};
  std::vector<OutputDeclaration> outputs;        // NEW. Newest declaration per output_id.
  std::vector<ListedFile> files;                 // In list order. Files a later entry removed are absent.
  std::vector<NotBuiltOutput> not_built;
  std::vector<FrameDone> frames_done;            // first_frame..last_frame_done are complete.
  std::int64_t last_frame_done = -1;
  std::uint64_t ignored_end_bytes = 0;          // Bytes after the last entry that passes its check.
};
// Takes no lock and changes nothing. Returns the entries that pass their checks, in order (5.3).
// Fails: kNoRunInFolder, kFileNotReadable, kWrongFileKind, kWrongSchemaVersion.
std::expected<FileList, IoFailure> ReadFileList(const Path& run_folder);
const ListedFile* FindListedFile(const FileList& file_list, FileKind kind, std::int64_t frame);  // Null: none.

class RunFolder {                                // Holds the WriterLock while it lives. Move-only.
 public:
  const Path& path() const;
  const FileList& file_list() const;             // Kept current by the Write functions below.
};
struct RunFolderInputs { Path path; RunFingerprint run_fingerprint{}; };
// A folder for a new run: creates it, takes the lock, writes the first file list entry. Files the caller put
// there before (the add-on's copy of the scene file) are left alone and never listed.
// Fails: kRunFolderNotEmpty, kRunFolderInUse, kFileNotWritable, kNoSpaceLeft.
std::expected<RunFolder, IoFailure> BuildNewRunFolder(const RunFolderInputs& inputs);
// The folder of an existing run, to add to it: takes the lock, compares fingerprints, cuts a damaged end off
// the file list, deletes tmp/ and every file under this module's own names that is not listed.
// Fails: kNoRunInFolder, kRunFolderInUse, kWrongRunFingerprint, kFileNotWritable.
std::expected<RunFolder, IoFailure> BuildRunFolderToContinue(const RunFolderInputs& inputs);
// Appends one Output entry per declaration (a rebuilt output replaces the older declaration of its id).
std::expected<void, IoFailure> WriteOutputDeclarations(RunFolder& run_folder,
                                                       std::span<const OutputDeclaration> outputs);

struct FileToList {                              // NEW. What the entry of a file will say.
  FileKind kind{};
  std::int64_t frame = -1;  double time = 0;  std::uint64_t state_version = 0;
  std::string output_id;                         // kOutputFile only.
  std::string relative_name;                     // The final name.
  std::vector<std::pair<std::string, double>> product_report;   // Copied into the ListedFile.
};
// NEW. A file that is not seen by any reader until WriteListedFile lists it. Move-only. Lives in tmp/.
class OutputFile {
 public:
  const FileToList& to_list() const;
  // Appends bytes and hashes them. Fails: kFileNotWritable, kNoSpaceLeft.
  std::expected<void, IoFailure> Write(std::span<const std::byte> bytes);
  // tmp/<final name>.<process id>, for the USD, Alembic and OpenVDB libraries, which write to a path. A caller
  // that uses it closes the file before WriteListedFile and does not call Write; the digest is then computed
  // by reading the file once.
  const Path& path_while_writing() const;
};
// Fails: kBadInput (a name outside the folder, or kOutputFile without a declared output_id),
// kFileAlreadyListed (the final name is listed with another digest: checked again at listing), kFileNotWritable.
std::expected<OutputFile, IoFailure> BuildOutputFile(RunFolder& run_folder, const FileToList& file);
enum class CheckAfterWrite : std::uint8_t {
  kSizeOnly,          // Open-format files: the size on disk is the size hashed, and above zero.
  kHeaderAndSizes,    // Array files: header verified, every array extent and the file size compared.
  kReadBackDigest,    // Array files: also every byte read back and the digest compared (saved states).
};
// The one publication transaction (5.3 steps 3 to 7): flush, check, rename without replacing, file list entry.
// Returns after the file is listed. On failure nothing is listed and the temporary file is gone.
// Fails: kFileNotWritable, kNoSpaceLeft, kFileDamaged, kDigestDiffers, kFileAlreadyListed.
std::expected<ListedFile, IoFailure> WriteListedFile(RunFolder& run_folder, OutputFile&& file,
                                                     CheckAfterWrite check);
void DiscardOutputFile(OutputFile&& file);       // Deletes the temporary file. Nothing is listed.
// Adds the entry that says a product was asked for in `frame` and not built.
std::expected<void, IoFailure> WriteNotBuilt(RunFolder& run_folder, const NotBuiltOutput& not_built);
// Adds the entry that says every file of the frame is listed. Frames must be marked in ascending order.
std::expected<void, IoFailure> WriteFrameDone(RunFolder& run_folder, const FrameDone& frame_done);
// Keeps the newest `kept_count` saved states: adds a removal entry for each older one, then deletes it.
std::expected<void, IoFailure> ApplySavedStateRetention(RunFolder& run_folder, std::uint32_t kept_count);
// NEW. Adds a removal entry for every listed file of these outputs, then deletes them (a rebuild, 5.6).
std::expected<void, IoFailure> RemoveOutputFiles(RunFolder& run_folder, std::span<const std::string> output_ids);

// The array file (5.3).
enum class ElementType : std::uint8_t { kFloat64, kUInt64, kUInt32, kInt8, kUInt8, kFloat32 };
struct ArrayToWrite {                            // Borrowed for the call.
  std::string_view name;                         // "particles.17.position". Names are fixed by the schema notes.
  ElementType element{};
  std::uint8_t elements_per_item = 1;            // 3 for a Vec3, 9 for a Mat3 (row by row).
  std::uint64_t item_count = 0;                  // Exactly the items that exist: no limit, no padding.
  const void* values = nullptr;                  // item_count * elements_per_item elements, little-endian.
};
struct ArrayFileContent {                        // Borrowed for the call.
  std::span<const std::uint8_t> description;     // The kind's own table (3.5), already built.
  std::span<const ArrayToWrite> arrays;
};
// Writes start, header and arrays into `file`, hashing every byte. Fails: kFileNotWritable, kNoSpaceLeft.
std::expected<void, IoFailure> WriteArrayFile(const ArrayFileContent& content, OutputFile& file);
struct ArrayInFile { std::string name; ElementType element{}; std::uint8_t elements_per_item = 1;
                     std::uint64_t item_count = 0, byte_offset = 0; };
struct ArrayFile {                               // The header of one listed array file, read and checked.
  ListedFile listed;  std::uint32_t schema_version = 0;  RunFingerprint run_fingerprint{};
  std::vector<std::uint8_t> description;  std::vector<ArrayInFile> arrays;  Path path;
};
// Reads the header only. Fails: kFileNotFound, kWrongFileKind, kWrongSchemaVersion, kFileDamaged.
std::expected<ArrayFile, IoFailure> ReadArrayFile(const Path& run_folder, const ListedFile& listed);
// Reads one array into `values` (exactly its bytes). Fails: kBadInput (no such name or wrong size).
std::expected<void, IoFailure> ReadArray(const ArrayFile& file, std::string_view name, std::span<std::byte> values);
// Reads every byte of any listed file and compares with the listed digest. Fails: kDigestDiffers, kFileDamaged.
std::expected<void, IoFailure> CheckListedFile(const Path& run_folder, const ListedFile& listed);
```

### 3.4 `io/run_fingerprint.h`, `io/run_report.h`, `io/run_folder_facts.h`

```cpp
struct RunFingerprintParts {                     // Everything that may change the next committed step.
  Digest scene_digest{};                         // Physical part of the ready scene (5.2).
  Digest executable_digest{};                    // BLAKE3 of the running executable's bytes.
  RunOn run_on = RunOn::kCpu;
  ProcessorDescription processor{};              // base: model, driver and runtime versions.
  step::RunMode run_mode = step::RunMode::kOnePiece;
  RunFingerprint started_from{};                 // All zero, or the run whose saved state this run started from.
};
RunFingerprint ComputeRunFingerprint(const RunFingerprintParts& parts);
FingerprintPart FindFirstDifference(const RunFingerprintParts& recorded, const RunFingerprintParts& now);
// NEW. 5.2: the run fingerprint, the product's settings, the format chosen and the output contract version.
Digest ComputeOutputFingerprint(RunFingerprint run, const OutputDeclaration& output,
                                std::span<const std::uint8_t> product_settings_bytes);
// NEW. Every output this scene asks for, with names, channels and fingerprints (5.5).
std::vector<OutputDeclaration> BuildOutputDeclarations(const RunScene& scene, RunFingerprint run);
std::expected<Digest, IoFailure> ComputeExecutableDigest();   // Ported (fluidity_simulate.cc ExecutableDigest).
std::expected<Digest, IoFailure> ComputeFileDigest(const Path& file);

enum class RunStatus : std::uint8_t { kRunning, kDone, kStopped, kFailed };  // kRunning without a held
                                                                             //   writer lock reads as crashed.
struct LastTry {                                 // NEW. The last work started, committed or not.
  std::int64_t frame = -1;  std::uint64_t step_count = 0, try_number = 0;
  step::Stage stage = step::Stage::kNone;        // The stage running, or rejected_in.
  bool committed = false;  step::RejectReason reject_reason = step::RejectReason::kNone;
  double sent_at = 0;                            // s since 1970-01-01 UTC.
};
struct RunReport {                               // The one description of a run (report, delivery "Gate 0").
  std::uint32_t report_version = 2;
  std::string executable_version;                // "0.1.0+<commit>".
  std::string toolchain;                         // Compiler and version, C++ standard, GPU toolkit version,
                                                 //   numerical flags, FlatBuffers, BLAKE3 and CLI11 versions.
  render_output::InterchangeLibraries interchange_libraries{};   // NEW. OpenUSD, Alembic, OpenVDB versions.
  Digest scene_file_digest{};                    // Of the scene file as the artist's tool wrote it.
  RunFingerprintParts parts{};  RunFingerprint run_fingerprint{};
  FrameRange frames{};
  double committed_time = 0;  std::int64_t last_frame_done = -1;  std::int64_t last_saved_state_frame = -1;
  std::uint64_t state_version = 0, step_count = 0, rejected_try_count = 0;
  std::optional<LastTry> last_try;               // NEW.
  double last_advance_sent_at = 0;               // NEW. s UTC of the last commit or listed frame.
  std::uint32_t scene_schema = 0, array_file_schema = 0, saved_state_schema = 0, frame_inputs_schema = 0,
                events_protocol = 0;             // io/schema_versions.h
  RunStatus status = RunStatus::kRunning;
  std::string last_failure_module, last_failure_code, last_failure_message;   // Empty unless kFailed.
  step::MemoryReport memory{};
  std::optional<batches::BatchPlan> batch_plan;  // NEW. With either batched RunMode (A15).
  std::optional<RunBudget> budget;               // NEW (planned cost, 3.9). The plan shown before the run.
  RunCost cost;                                  // NEW (planned cost, 3.9). Measured so far; rewritten per report.
  std::uint64_t resize_count = 0, cells_to_grid_only = 0, cells_to_particles = 0;   // NEW. Cell resize totals.
  double simulate_elapsed_seconds = 0, snapshot_elapsed_seconds = 0, write_elapsed_seconds = 0,
         output_wait_elapsed_seconds = 0, build_outputs_elapsed_seconds = 0;  // D word elapsed. Clock time.
  std::vector<step::CapabilityFinding> known_limits;   // Findings with kHasKnownLimit.
};
// Replaces <run_folder>/run_report.json in one rename. It is not a listed file: it changes during a run.
std::expected<void, IoFailure> WriteRunReport(const RunFolder& run_folder, const RunReport& report);
std::expected<RunReport, IoFailure> ReadRunReport(const Path& run_folder);

// run_folder_facts.h. NEW. The read-only query of a run folder (`report`). Takes no writer lock, writes nothing.
struct LockHolder {
  std::int64_t process_id = 0;  std::string host_name;
  double started_at = 0;                         // s UTC, written by the holder after it took the lock.
  bool is_held = false;                          // false: the holder is gone (5.3, the lock probe).
};
struct RunFolderFacts {
  FileList file_list;                            // Outputs, frames done, saved states, ignored end bytes.
  std::optional<RunReport> report;               // Empty when run_report.json is missing or unreadable.
  std::optional<LockHolder> lock_holder;         // Empty when no writer.lock exists.
  std::int64_t last_listed_frame = -1;           // = file_list.last_frame_done.
  std::int64_t saved_state_frame = -1;           // Newest listed saved state that passes its header check.
  std::optional<LastTry> last_try;
  std::vector<ListedFile> files_failing_check;   // With kEveryListedFile: every listed file CheckListedFile refuses.
};
enum class FileCheck : std::uint8_t { kListOnly, kEveryListedFile };   // Plan decision: replaces bool check_files.
// Fails: kNoRunInFolder, kFileNotReadable, kWrongSchemaVersion.
std::expected<RunFolderFacts, IoFailure> ReadRunFolderFacts(const Path& run_folder, FileCheck file_check);

// NEW. `check --cache`: how a scene compares with the run in a folder (5.2).
enum class CacheMatch : std::uint8_t { kSame, kChanged, kAbsent };
struct OutputMatch {
  std::string output_id;  CacheMatch match{};
  std::vector<std::string> changed_fields;       // Scene file paths ("outputs.surface.spacing").
  bool needs_simulation = false;                 // A changed field is read inside the step (motion, foam sources).
};
struct CacheComparison {
  CacheMatch simulation{};                       // kAbsent: no run in the folder.
  FingerprintPart differing_part = FingerprintPart::kNone;
  std::vector<std::string> changed_fields;       // Scene file paths of the physical part that differ.
  std::vector<OutputMatch> outputs;              // Every output of the scene and of the folder, by output_id.
};
std::expected<CacheComparison, IoFailure> CompareWithRunFolder(const RunScene& scene,
                                                               const RunFingerprintParts& parts,
                                                               const Path& run_folder);
```

### 3.5 `io/saved_state.h` and `io/frame_inputs.h`

```cpp
struct SavedState {                              // D word. Everything a run needs to continue (5.4).
  std::int64_t frame = 0;
  step::SimulationSnapshot simulation;           // Read with every particle field; every addition of step 3.6.
  std::optional<render_output::RenderOutputSnapshot> render_output;   // NEW. Taken after the frame was built.
};
std::expected<ListedFile, IoFailure> WriteSavedState(RunFolder& run_folder, const SavedState& state);
// Checks the digest, then every extent, before any value is used. Fails: kDigestDiffers, kFileDamaged,
// kWrongSchemaVersion, kWrongRunFingerprint.
std::expected<SavedState, IoFailure> ReadSavedState(const Path& run_folder, const ListedFile& listed,
                                                    RunFingerprint expected);
// Newest listed saved state with frame <= at_or_before_frame. Null: none.
const ListedFile* FindSavedState(const FileList& file_list, std::int64_t at_or_before_frame);

// NEW. An internal run record, never a product (A14): what render output needs to build a frame again without
// simulating, and what read-particle reads. Doubles, as the simulation knew them.
struct KeptFrameInputs {
  render_output::FrameIdentity identity;
  step::SimulationSnapshot snapshot;             // kOutputSnapshotFields plus RequestedOutputs::particle_fields;
                                                 //   solids, gas amounts and the region arrays as read.
  render_output::RecordedInterval recorded;      // Empty when RecordsSteps is false.
  render_output::MotionSample sample_at_frame;
};
std::expected<ListedFile, IoFailure> WriteFrameInputs(RunFolder& run_folder, const KeptFrameInputs& inputs);
std::expected<KeptFrameInputs, IoFailure> ReadFrameInputs(const Path& run_folder, const ListedFile& listed);
// The digest of FrameDone::snapshot_digest: BLAKE3 over the snapshot arrays of `snapshot` in the order and
// packing of 5.5, whether or not a frame inputs file is written.
Digest ComputeSnapshotDigest(const step::SimulationSnapshot& snapshot);

// NEW. read-particle: one particle of one frame, from that frame's stored inputs, else from a saved state of
// that frame. Fails: kFrameInputsNotKept, kParticleNotFound.
struct ParticleFacts {
  std::int64_t frame = -1;  double time = 0;
  std::uint64_t particle_id = 0, material_id = 0;
  materials::MaterialKind kind{};
  Vec3 position{}, velocity_for_next_step{};     // m, m/s, world, Z-up.
  double rest_volume = 0, volume_ratio = 0;      // m3; no unit.
  std::uint8_t size_level = 0;
  std::vector<double> ingredient_fractions;      // By the material's ingredients. Empty for one ingredient.
  std::vector<std::pair<std::string, std::vector<double>>> other_fields;  // Every other array carried, by name.
};
std::expected<ParticleFacts, IoFailure> ReadParticle(const Path& run_folder, std::int64_t frame,
                                                     std::uint64_t particle_id);
```

### 3.6 `io/output_queue.h` and `io/run.h`

```cpp
struct FrameOutput {                             // What one frame hands to the output worker. Owns its data.
  render_output::FrameIdentity identity;
  step::SimulationSnapshot snapshot;             // Immutable from here on. Full on a saved-state frame.
  render_output::RecordedInterval recorded;      // Empty when RecordsSteps is false.
  render_output::MotionSample sample_at_frame;
  bool with_saved_state = false;
  bool is_already_done = false;                  // After a resume: build, compare, list nothing (5.6 step 7.6).
  std::vector<std::string> only_output_ids;      // A rebuild: these outputs only. Empty: every output.
};
struct ProductsOfFrame {                         // NEW. What the frame writer did for one frame.
  std::vector<ListedFile> listed;
  std::vector<NotBuiltOutput> not_built;         // Already written with WriteNotBuilt.
  std::optional<render_output::RenderOutputSnapshot> render_output;   // With with_saved_state.
};
struct FrameWriter {                             // NEW. The CLI's (it links fluidity_interchange). On the worker.
  void* context = nullptr;
  // Once, before the first frame: builds render output fresh, or from the snapshot of the saved state.
  std::expected<void, IoFailure> (*start)(void* context,
      const std::optional<render_output::RenderOutputSnapshot>& kept) = nullptr;
  // Frames in order. Builds the RenderFrame and writes and lists every declared product of the frame
  // (BuildOutputFile, a render output writer, WriteListedFile with kSizeOnly). With is_already_done it builds
  // and lists nothing but advances its state. A product that fails is WriteNotBuilt and the run goes on;
  // only an I/O failure of the folder is returned.
  std::expected<ProductsOfFrame, IoFailure> (*write_products)(void* context, RunFolder& run_folder,
                                                              const FrameOutput& output) = nullptr;
};
struct OutputQueueSettings {
  std::uint32_t max_waiting_frames = 2;          // D pattern max_... . Frames read but not yet listed.
};
class OutputQueue;                               // Owns its one worker thread. Move-only.
std::expected<OutputQueue, IoFailure> BuildOutputQueue(RunFolder& run_folder, const RunScene& scene,
    const FrameWriter& frame_writer, const OutputQueueSettings& settings);
struct AddOutputResult { double waited_seconds = 0; std::uint32_t waiting_frames = 0; };
// Hands one frame over. Waits while max_waiting_frames are waiting: nothing is ever dropped (5.7).
// `on_waiting` is called about twice a second while it waits. Fails: kOutputStopped.
std::expected<AddOutputResult, IoFailure> AddFrameOutput(OutputQueue& queue, FrameOutput&& output,
    void* context, void (*on_waiting)(void* context, std::uint32_t waiting_frames));
// Returns when every waiting frame is listed, or with the failure of the worker.
std::expected<void, IoFailure> WriteWaitingFrames(OutputQueue& queue);
std::vector<MemoryEstimateEntry> BuildIoMemoryEstimate(const OutputQueueSettings& settings);  // Write buffer, hasher, CPU.

struct RunProgress {                             // Copied under a lock for the heartbeat. No pointer inside.
  double committed_time = 0;  std::int64_t frame_in_progress = 0;
  std::uint64_t step_count = 0, rejected_try_count = 0, state_version = 0;
  std::int64_t last_frame_done = -1, last_saved_state_frame = -1;
  std::optional<LastTry> last_try;               // NEW.
  double last_advance_sent_at = 0;               // NEW. s UTC of the last commit or listed frame.
  std::uint32_t waiting_frames = 0;
  std::uint32_t preparing_done = 0, preparing_total = 0;   // NEW. With kPreparing.
  TimeProjection time_projection;                // NEW (planned cost, 3.9). For heartbeat and frame_done.
  enum class Activity : std::uint8_t { kPreparing, kReadingSavedState, kSimulating, kCheckingFrame,
                                       kWaitingForOutput, kWritingLastFrames, kRebuildingOutputs,
                                       kStopping } activity{};
};
struct RunCallbacks {                            // D rule: named after the event. Null: not called.
  void* context = nullptr;
  void (*on_progress)(void* context, const RunProgress& progress) = nullptr;          // After every try.
  void (*on_try_done)(void* context, const step::StepResult& result) = nullptr;
  void (*on_frame_done)(void* context, const FrameDone& frame, const ProductsOfFrame& products) = nullptr;
  void (*on_frame_checked)(void* context, std::int64_t frame) = nullptr;
  void (*on_saved_state_listed)(void* context, const ListedFile& saved_state) = nullptr;
};
// NEW. What cli/advance_run.cc hands a run. Plan decision: no cell resize and no batched run here, since both
// need the run plan BuildRun computes; Run builds and owns cell_resize::CellResize (scene with cell resize
// settings) and batches::BatchedRun (either batched RunMode) after step::BuildRunPlan. This module owns the
// step::StepCallbacks given to BuildSimulation and sets each pair (5.6 step 5). Render output's Task 47 adds
// render_output::StepRecorder* step_recorder (with RecordsSteps).
struct RunParts {
  FrameWriter frame_writer;
  OutputQueueSettings output;
  RunCallbacks callbacks;
};
class Run;                                       // RunScene, RunFolder, step::Simulation, OutputQueue, RunReport.
// A new run: 5.6 steps 1 to 6, then the files of first_frame. Nothing is allocated before the scene is
// checked. Takes the plan and budget from BuildSceneBudget and keeps the budget in RunReport::budget; a verdict
// other than kFits is kModuleFailed (step, kDoesNotFit) before any folder or buffer exists.
// Fails: every IoError of BuildNewRunFolder, kModuleFailed (step: kSceneNotSupported, kDoesNotFit, ...).
std::expected<Run, IoFailure> BuildRun(RunScene&& scene, const Path& run_folder, const RunParts& parts);
// Continues the run in a folder from its newest good saved state. Fails: kWrongRunFingerprint, kNoSavedState, ...
std::expected<Run, IoFailure> BuildRunToContinue(const Path& run_folder, const RunOverrides& overrides,
                                                 const RunParts& parts);
// A new run whose first state is a saved state of another run (5.2). Its fingerprint names that run.
std::expected<Run, IoFailure> BuildRunFromSavedState(RunScene&& scene, const Path& run_folder,
    const Path& other_run_folder, std::int64_t at_or_before_frame, const RunParts& parts);
struct StopRequests { std::atomic<bool> cancel_requested{false}; std::atomic<bool> stop_after_frame_requested{false}; };
enum class RunStop : std::uint8_t { kReachedLastFrame, kCancelled, kStoppedAfterFrame, kStepStopped, kOutputFailed };
struct AdvanceRunResult { RunStop stop{}; step::AdvanceResult last_advance{}; IoFailure output_failure{};
                          std::int64_t last_frame_done = -1, last_saved_state_frame = -1; };
// The frame loop of 5.6. Reuses the step's verb AdvanceTo (step decision 7).
AdvanceRunResult AdvanceRunTo(Run& run, std::int64_t last_frame, const StopRequests& stop_requests);
const RunReport& ReadRunReport(const Run& run);
// NEW. `run --outputs`: builds the named outputs again from the stored frame inputs, frames first_frame to
// last_frame_done, without simulating (5.6, "Rebuild"). The scene's run fingerprint must equal the folder's.
// Fails: kWrongRunFingerprint, kUnknownOutput, kFrameInputsNotKept, every IoError of the writers.
std::expected<void, IoFailure> RebuildOutputs(RunScene&& scene, const Path& run_folder,
                                              std::span<const std::string> output_ids, const RunParts& parts,
                                              const StopRequests& stop_requests);
```

**Seams render output completes** (plan decision, `plan-decisions.md` "Surface, foam and spray" and "I/O and
CLI"; I/O plan "Seams the render-output plan completes", items 1 to 11). The declarations above are the final
form. Until render output's Task 47, no `io/` or `cli/` file includes a `render_output/` header, and: `FrameWriter`
has `start(void* context)` without `kept`; `FrameOutput` and `KeptFrameInputs` carry `std::int64_t frame` and
`double time` instead of `identity`, and no `recorded` or `sample_at_frame`; `RequestedOutputs::render`,
`RecordsSteps`, `OutputDeclaration::product`, `format`, `contract_version`, `SavedState::render_output`,
`ProductsOfFrame::render_output`, `RunParts::step_recorder` and `RunReport::interchange_libraries` do not exist;
`file_pattern` has no extension (the writer appends its own); `ComputeOutputFingerprint` hashes `output_id` and
`material_id`; the scene table `outputs.render: RenderSection` is empty; the program's writer is
`cli::BuildFrameWriterWithoutProducts()`, which lists each declared output as not built with reason
`no_product_writer`. Seam 11 (planned cost): `on_step_done` adds the difference of
`render_output::ReadRecorderCopyCounts` since the last committed step to the copies given `ApplyStepToRunCost`.

### 3.7 `cli/`: commands, events, exit codes

```text
fluidity capabilities [--run-on gpu]
fluidity check        <scene_file> [--run-on gpu] [--cache <run_folder>] [--assume-memory <bytes>]
                      [--assume-cpu-memory <bytes>]
fluidity run          <scene_file> --into <run_folder> [--run-on gpu] [--last-frame <n>]
                      [--start-from <other_run_folder> [--start-frame <n>]] [--report-every-try]
fluidity run          <scene_file> --into <run_folder> --outputs <output_id>[,<output_id>...]
fluidity resume       <run_folder> [--last-frame <n>] [--report-every-try]
fluidity report       <run_folder> [--check-files]
fluidity read-particle <run_folder> --frame <n> --particle-id <id>
```

```cpp
enum class ExitCode : int {
  kDone = 0,             // The command did what was asked. For check: the scene can run.
  kFailed = 1,           // Any failure not listed below. The last event says which.
  kBadCommandLine = 2,   // Ported value.
  kSceneCannotRun = 3,   // The scene file is not valid, a finding blocks the run, or it does not fit (budget
                         //   verdict other than kFits). check still printed scene_checked with every finding
                         //   and the budget; the budget lines are on standard error.
  kStoppedOnRequest = 4, // cancel, stop_after_frame or end of standard input. Every listed file is whole.
  kRunFolderRefused = 5, // In use (the event names the holder), not empty, no run, another run's
                         //   fingerprint, no saved state.
  kProcessorFailed = 6,  // kRunOnCallFailed or kSimulationUnusable. Continue from the last saved state.
};
ExitCode ToExitCode(const IoFailure& failure);
enum class CommandKind : std::uint8_t { kCapabilities, kCheck, kRun, kResume, kReport, kReadParticle };
struct Command {
  CommandKind kind{};
  Path scene_file;                                       // check, run. Empty when not given.
  Path run_folder;                                       // --into of run; the argument of resume, report and
                                                         //   read-particle; --cache of check.
  Path start_from_run_folder;                            // --start-from of run.
  io::RunOverrides overrides{};
  std::optional<std::int64_t> start_from_frame;
  std::optional<std::uint64_t> assume_memory_bytes;      // check: plan as if the processor had this much.
                                                         //   Copied into overrides.assume_memory_bytes.
  std::optional<std::uint64_t> assume_cpu_memory_bytes;  // NEW (planned cost). check only: as if the CPU had
                                                         //   this much. Copied into overrides.
  std::vector<std::string> output_ids;                   // NEW. --outputs of run: rebuild these outputs only.
  std::int64_t frame = -1;  std::uint64_t particle_id = 0;   // NEW. read-particle. The id as decimal text.
  bool check_files = false;                              // report: read every listed file and compare digests.
  bool reports_every_try = false;                        // run, resume: one try_done event per try.
};
struct CommandLineFailure { bool is_help = false; std::string text; };  // Help text, or what was wrong.
// The only function compiled with exceptions on. No exception leaves it.
std::expected<Command, CommandLineFailure> ReadCommandLine(std::span<const std::string_view> arguments);

class EventWriter;  // One line per event on standard output, under a lock; run and resume also append the same
                    // line to <run_folder>/events.jsonl.
ExitCode WriteCapabilities(const Command& command, EventWriter& events);   // Each returns after its last event.
ExitCode CheckScene(const Command& command, EventWriter& events);
ExitCode AdvanceRun(const Command& command, EventWriter& events, io::StopRequests& stop_requests);  // run, resume
ExitCode WriteReport(const Command& command, EventWriter& events);
ExitCode WriteParticle(const Command& command, EventWriter& events);       // NEW. read-particle.
```

The events are the tables of `schemas/simulate/v2/events.fbs`, printed as one JSON object per line. Every
event has `protocol` (2), `event` (its name), `sequence` (0, 1, 2, ... per process), `sent_at` (NEW: seconds
since 1970-01-01 UTC, taken when the event was made, not when it was read) and `elapsed_seconds` (clock time
since the process started). Every finding, in any event, is one `Finding` table:

| `Finding` field | Content |
|---|---|
| `module`, `code` | `io::ToName(FailingModule)` and the stable code from `ToName` (`"solids"`, `"plate_too_thin_for_liquid"`); for a capability finding `"step"` and `ToName(CapabilityReason)` |
| `severity` | `blocks_run` (a failure, or `CapabilityLevel::kUnavailable`), `has_known_limit`, `note` |
| `message` | `io::ToMessage` with the object name, frame, magnitude and limit filled in (5.8) |
| `material_id`, `solid_id`, `source_id`, `particle_id`, `object_name`, `scene_file_path` | The scene ids as the scene file gives them, 0 or empty when not about one; the path of the field (`solids[2].path.keys[4].time`) |
| `magnitude`, `limit`, `unit` | What was measured and what was allowed |
| `suggested_changes` | `ToName(SuggestedChange)` values, plus this module's `run_on_cpu`, `use_bigger_cells`, `start_new_run`, `free_disk_space` |

| Event | Fields beyond the five | When |
|---|---|---|
| `capabilities` | executable version and digest, `protocol`, the schema versions, coordinate text `right_handed_z_up_metres`, `run_on_values` (`cpu`, and `gpu` when the build has GPU code), every `CapabilityReason` with its `CapabilityLevel` and message, static limits (`kMaxPlateCount`, `kMaxFreeSolidCount`, `kMaxIngredientCount`, `kMaxSizeLevels`, motion samples 1..64), the format table (`ReadFormatTable()` with every `WitnessRecord`), `ReadOutputChannels()`, `ReadInterchangeLibraries()`; with `--run-on gpu`: the processor description, total and free memory, or `kNoGpuFound` as a finding | last event of `capabilities` |
| `scene_checked` | `can_run`, every finding (scene file, capability, fit, `run_on_usable`), `filled_defaults`, `run_on_requested`, `run_on_usable`, the plan (`run_mode`, estimate by module, `allowed_bytes`, counts, the batch plan), `budget: RunBudget` (NEW, also when it does not fit), `run_fingerprint`, the output declarations; with `--cache`: `cache` = `CacheComparison` (`simulation`, `differing_part`, `changed_fields`, `outputs[] {output_id, match, changed_fields, needs_simulation}`) | last event of `check` |
| `started` | `run_fingerprint`, `run_on`, processor, first and last frame, `continues_from_frame` (or null), memory estimate, `allowed_bytes`, `run_mode`, the batch plan, `budget: RunBudget` (NEW), the output declarations, the findings with `has_known_limit` | once, before the first step |
| `preparing` | `done`, `total`, `what` (`sampling_start_particles`, `building_simulation`, `reading_saved_state`) | while preparing, at most twice a second |
| `heartbeat` | the `RunProgress` fields, `last_advance_sent_at`, `simulated_seconds_per_second: double` (0 until known) and `projected_remaining_seconds: double = -1` (-1 until 3 frames of this process are listed) (NEW) | every `kHeartbeatSeconds` = 0.5 (ported); never more than 1 s apart |
| `try_done` | `frame`, `step_count`, `try_number`, `committed`, `reject_reason`, `rejected_in`, `step_duration`, `time`, `active_limit` | only with `--report-every-try` |
| `finding` | one `Finding`: a product not built in a frame, a known limit met during the run | when it happens |
| `frame_done` | `frame`, `time`, files (`output_id` or kind, name, byte count, digest), `not_built[]`, step and try counts, the `MemoryReport` numbers (CPU and GPU in use and largest, `limit_raise_count`), elapsed seconds by category (simulate, snapshot, build outputs, write, wait), `simulated_seconds_per_second`, `projected_remaining_seconds` (NEW, as `heartbeat`) | a `FrameDone` entry was listed |
| `frame_checked` | `frame` | a frame computed again after a resume equals the listed one |
| `saved_state_listed` | `frame`, `byte_count` | a saved state was listed |
| `stopping` | `request` (`cancel`, `stop_after_frame`, `end_of_input`) | the request was read |
| `report` | `RunFolderFacts`: `run_fingerprint`, `last_listed_frame`, `saved_state_frame`, `last_try`, `lock_holder` with `is_held`, the output declarations, frames done, ignored end bytes, the `RunReport`, `is_crashed`, and with `--check-files` each file that failed | last event of `report` |
| `particle` | `ParticleFacts`; ids as decimal text | last event of `read-particle` |
| `done`, `stopped`, `failed` | `exit_code`, `last_frame_done`, `last_saved_state_frame`, total elapsed by category, the `MemoryReport` numbers, `budget: RunBudget` and `cost: RunCost` (NEW); `failed` adds one `Finding` with `stage`, `frame`, `time` | exactly one, always the last line of `run` and `resume` |

Standard input, one line each: `cancel`, `stop_after_frame`. The end of standard input means
`stop_after_frame` (a run never outlives the program that started it). Standard error carries the same `failed`
line and nothing a program must parse; `run` and `resume` also append it to `<run_folder>/log.txt`. NEW (planned
cost): standard error also carries the human lines of the budget and the cost (3.9, 5.10), never standard output,
which stays one JSON event per line: `check` writes `budget.lines` after `scene_checked`; `run` and `resume` write
the budget lines before `BuildRun` (also into `log.txt`, or to standard error only when the folder is not made),
the time line after the 3rd frame listed in this process and every 10th after it (13th, 23rd, ...), and at the
end `cost.lines` then `cost.warnings`, after the last event is on standard output.

### 3.8 Schema files and what shared-interface (c) asked for

`schemas/simulate/v2/`, namespace `fluidity.simulate.v2`, every table field with an explicit number and a
unit comment: `common.fbs` (`Vec3`, `Digest`, enums), `scene_file.fbs`, `ready_scene.fbs`, `array_file.fbs`
(the header and the description tables of saved state and stored frame inputs), `file_list.fbs` (entries,
`OutputDeclaration`, `NotBuiltOutput`, `FrameDone`), `run_report.fbs`, `events.fbs`, and NEW (planned cost)
`run_budget.fbs` (tables `RunBudget`, `MemoryBudget`, `ModuleBudget`, `CopyBudget`, `FitSuggestion`, `RunCost`,
`StageSeconds`), included by `run_report.fbs` and `events.fbs`. `io/schema_versions.h` holds one constant per file
kind; every file kind and the events protocol start at version 2 (plan decision). Render output's settings are tables of `scene_file.fbs` and
`ready_scene.fbs`, written by this module from `render_output::RenderSettings` field by field.

Shared-interface (c), "I/O and CLI", item by item:

- One `particles::MaterialParticlesSettings` per material: through `ReadyScene::materials`
  (`materials::BuildParticlesSettings` makes them from the table; the step calls it). A separate
  `material_id` for every snow, springy solid or cut solid object: 5.1 step 4.
- `GridSettings`, `OpenSeaSettings`, `AirPocketSettings`, `SolveSettings`, `ViscositySettings`,
  `TransferSettings` with their defaults: the fields of `step::ReadyScene`, filled by 5.1 step 3.
  `GridSettings::max_cell_blocks` is `SimulationLimits::max_cell_blocks`, which the run plan sets.
- `saved_state` stores a full `ParticlesSnapshot`, the committed pressure, `AirPockets` and the active cell
  ids, with every other snapshot: 5.4. Playback reads no FluidiTy file (A14): render output writes the
  particles as open-format points from a snapshot with `kOutputSnapshotFields` (5.6 step 7.4).
- Every `<Module>Failure` becomes text with `ToMessage` and a code with `ToName` at the outer boundary:
  `io::ToMessage(module, error)`, `io::ToName(module, error)` and 5.8.

Nothing in (c) is left out.

### 3.9 `io/run_budget.h` and `io/run_cost.h`: the planned and the measured cost (NEW, CPU only)

Copied from `plan-decisions.md` "Planned cost shown to the user, and buffers in batched runs" (A22: `budget` is a
planned cost shown to the user; a hard limit is still `max_...`).

```cpp
// io/run_budget.h
enum class ModeReason : std::uint8_t { kFitsProcessor, kParticlesDoNotFitProcessor, kSolverBuffersDoNotFitProcessor,
                                       kStateDoesNotFitCpu, kLayerDoesNotFitProcessor, kOnePieceDoesNotFitCpu };
enum class BudgetVerdict : std::uint8_t { kFits, kDoesNotFitCpu, kDoesNotFitProcessor };
struct MemoryBudget { std::uint64_t budget_bytes = 0, allowed_bytes = 0, total_bytes = 0, free_bytes = 0; };
struct ModuleBudget { step::Module module = step::Module::kStep; std::uint64_t processor_bytes = 0, cpu_bytes = 0; };
struct CopyBudget {                          // Per committed step unless named. Upper bounds where Appendix B says so.
  std::uint64_t step_bytes = 0;  std::uint32_t step_copies = 0;            // RunPlan::planned_step_copies.
  std::uint64_t pass_bytes = 0;  std::uint32_t pass_loads = 0;             // BatchPlan bytes_copied_per_try, pass_loads_per_try.
  std::uint64_t resize_bytes = 0; std::uint32_t resize_copies = 0;         // GPU grid, cell resize: (456 + 512) B x blocks
                                                                           //   in one piece, 512 B x blocks batched.
  std::uint64_t solve_bytes_per_iteration = 0; std::uint32_t solve_loads_per_iteration = 0;
  std::uint32_t solve_iteration_limit = 0;                                 // pressure_solve.max_iterations.
  std::uint64_t snapshot_bytes_per_frame = 0;                              // one_snapshot rows at the limits.
  std::uint64_t bytes_per_step = 0;  std::uint32_t copies_per_step = 0;    // step + pass + resize.
};
struct FitSuggestion { std::uint64_t particle_count = 0, max_particle_count = 0; double smallest_cell_width = 0,
                       min_smallest_cell_width = 0; std::uint64_t more_memory_bytes = 0; RunOn more_memory_on{}; };
struct ViscousEffectOfLiquid {               // NEW (A25; renamed A27, was LiquidViscousEffect, the name of the
                                             // step's different struct). One per liquid ingredient, scene order.
  std::string ingredient_name;               // The ingredient's export name.
  double viscosity_pa_s = 0;                 // The setting.
  double density_kg_per_m3 = 0;              // Rest density of its material.
  double viscous_effect_per_step = 0;        // No unit; 0 when viscosity is 0.
  bool is_viscous = false;                   // viscosity_pa_s > 0.
  bool effect_not_visible = false;           // 0 < viscous_effect_per_step < kSmallestVisibleViscousEffectPerStep.
};
struct RunBudget {
  std::string scene_name; RunOn run_on{}; std::string processor_model;
  step::RunMode run_mode{}; step::PlanFit fit{}; ModeReason mode_reason{}; BudgetVerdict verdict{};
  std::uint32_t batch_count = 0, solve_batch_count = 0;
  MemoryBudget processor_memory, cpu_memory;
  std::vector<ModuleBudget> modules;         // Every module with bytes, in Module order.
  CopyBudget copies;
  std::optional<FitSuggestion> to_fit;       // With a verdict other than kFits.
  step::ViscousPath viscous_path{};          // NEW (A25). Stage 12 or 11, fixed for the run (RunPlan::viscous_path).
  double viscous_effect_cell_width_m = 0;    // NEW (A25). The grid frame's smallest cell width.
  double viscous_effect_step_seconds = 0;    // NEW (A25). StepSettings::max_step_duration.
  std::vector<ViscousEffectOfLiquid> liquids;  // NEW (A25). From RunPlan::liquid_viscous_effects; empty without liquid.
  std::vector<std::string> lines;            // BuildBudgetLines, verbatim.
};
std::expected<RunBudget, IoFailure> BuildRunBudget(const step::ReadyScene&, const step::StepSettings&,
    const step::RunPlan&, const std::optional<batches::BatchPlan>&, const std::optional<batches::BatchesFailure>&,
    MemoryAvailable on_processor, MemoryAvailable on_cpu, const ProcessorDescription&, std::string_view scene_name);
std::vector<std::string> BuildBudgetLines(const RunBudget&);
// Reads the memory available (or RunOverrides::assume_memory_bytes / assume_cpu_memory_bytes), plans the run (step,
// and batches for a batched plan) and builds the budget. Allocates no buffer. Used by check, run and BuildRun.
std::expected<RunBudget, IoFailure> BuildSceneBudget(const RunScene& scene, const RunOverrides& overrides,
                                                     std::span<const MemoryEstimateEntry> output_estimate);

// io/run_cost.h
struct StageSeconds { step::Stage stage{}; double total_seconds = 0; std::uint64_t try_count = 0; };
struct RunCost {
  std::uint64_t committed_steps = 0, tries = 0, frames = 0;
  std::uint64_t largest_processor_bytes = 0, largest_cpu_bytes = 0;      // MemoryReport, largest over the run.
  double copy_bytes_per_step_mean = 0; std::uint64_t copy_bytes_per_step_max = 0;
  double copies_per_step_mean = 0; std::uint32_t copies_per_step_max = 0;
  double snapshot_bytes_per_frame_mean = 0;
  double solve_iterations_per_step_mean = 0, solve_bytes_per_iteration_mean = 0;
  std::vector<StageSeconds> stage_seconds;   // Every stage that ran, Stage order, all tries.
  std::vector<std::string> warnings, lines;
};
// Per try (on_try_done): result.report.copies, result.report.timings, solver iteration counts, and the batched
// run's last BatchReport (null in one piece). The try's copies wait until the step commits.
void ApplyTryToRunCost(const step::StepResult& result, const batches::BatchReport* batch_report, RunCost& cost);
// Per committed step (on_step_done): the memory report and the resize's and recorder's copies since the last one.
void ApplyStepToRunCost(const step::MemoryReport& memory, const step::CopyCounts& resize_and_recorder_copies,
                        RunCost& cost);
void ApplyFrameToRunCost(std::uint64_t snapshot_bytes, RunCost&);
struct ActivitySeconds { double simulate = 0, snapshot = 0, build_outputs = 0, write = 0, wait = 0; };  // RunReport's.
void BuildCostLines(const RunBudget&, const ActivitySeconds&, RunCost&);   // Fills warnings and lines.
inline constexpr double kMemoryWarningShare = 1.10;   // Owner to confirm.
inline constexpr double kCopyWarningShare = 1.25;     // Owner to confirm.
struct FrameMark { double time = 0; double sent_at = 0; };   // s simulated; s UTC.
struct TimeProjection { double simulated_seconds_per_second = 0; double projected_remaining_seconds = -1;
                        std::uint32_t frames_measured = 0; };
TimeProjection ComputeTimeProjection(std::span<const FrameMark> frames_done_in_this_process, double end_time);
```

Schema `run_budget.fbs`: field names carry their unit (`_bytes`, `_seconds`, `_m` for `smallest_cell_width_m` and
`min_smallest_cell_width_m`); enums are stable lower-snake strings: `run_mode` `one_piece`, `batched`,
`batched_with_solves_in_pieces`; `mode_reason` `fits_processor`, `particles_do_not_fit_processor`,
`solver_buffers_do_not_fit_processor`, `state_does_not_fit_cpu`, `layer_does_not_fit_processor`,
`one_piece_does_not_fit_cpu`; `verdict` `fits`, `does_not_fit_cpu`, `does_not_fit_processor`; `module` and `stage` by
this module's stable names. NEW (A25): `RunBudget` gains `viscous_path: string` (`ToName(step::ViscousPath)`:
`pressure_solve`, `viscous_solve`), `viscous_effect_cell_width_m: double`, `viscous_effect_step_seconds: double`,
`liquids: [ViscousEffectOfLiquid]`, with `table ViscousEffectOfLiquid { ingredient_name: string; viscosity_pa_s: double;
density_kg_per_m3: double; viscous_effect_per_step: double; is_viscous: bool; effect_not_visible: bool; }`. The
add-on reads these names (its `RunPlanFacts.budget`, drawn verbatim in
`FLUIDITY_PT_performance`).

## 4. Consumes from other modules

Every item is EXISTS (Foundation, including the `base/` additions of `changes-to-apply.md`) or PROVIDED (by a
designed module, with the name its section 3 has).

**Foundation (EXISTS):** `Vec3`, `Mat3`, `RunOn`, `CheckCanRunOn`, `ToMessage(RunOnError)`, `MemoryEstimateEntry`,
`CountedPer`, `KeptFor`, `std::expected`; from the `base/` additions: `ProcessorDescription
ReadProcessorDescription(RunOn run_on)` (model, driver and runtime versions as strings; for the fingerprint and
the run report), `struct MemoryAvailable { std::uint64_t total_bytes; std::uint64_t free_bytes; }` and
`std::expected<MemoryAvailable, BufferError> ReadMemoryAvailable(RunOn run_on)` (a total of 0 means unknown;
for `step::BuildRunPlan` and `capabilities`). Base errors have `ToMessage` only; their stable codes are this
module's table in `io_message.cc`. The build gains three packages (5.9); Foundation's Conan file lists
GoogleTest only.

**Step (PROVIDED, step 3.1 to 3.9):** `ReadyScene` (declared in `step/`;
`ReadyScene::solids` is `solids::SolidsSettings`; it carries the keyed `gas::GasSource` list and `gas::GasWind`,
whose key storage `RunScene` owns), `ParticleSource`, `StartParticles`, `StepSettings`, `SimulationLimits`,
`CheckSceneCanRun`, `CapabilityResult` (every finding, a growing list), `CapabilityFinding`, `CapabilityReason`,
`kCapabilityReasonCount` (16 at most, 13 if the viscosity plan lifts its three exclusions; A24 removed
`kViscousLiquidWithSnow`, A27 `kViscousLiquidWithFreeSolid`), `CapabilityLevel` (`kCanRun`, `kExperimental`,
`kUnavailable`), `SuggestedChange` (A27: no `kSetViscosityToZero`; the sand module adds it back with
`kViscousLiquidWithSand`), `kSuggestedChangeCount` (12, A27), `BuildRunPlan(scene, settings,
on_processor, on_cpu, std::span<const MemoryEstimateEntry> output_estimate, std::span<const MemoryEstimateEntry> cell_resize_estimate = {})`,
`RunPlan` (`max_blocks_per_batch`, `max_loaded_blocks`, `max_loaded_particles` and, A15, `max_solve_loaded_blocks`,
copied here from `batches::BatchPlan`; `allowed_bytes`, `allowed_cpu_bytes` (A15); `fit` and `planned_step_copies`
(planned cost); NEW (A25) `liquid_viscous_effects` (`step::LiquidViscousEffect`: `ingredient_index`, `viscosity`,
`density`, `viscous_effect_per_step`), `viscous_effect_cell_width`, `viscous_effect_step_duration`,
`viscous_path`), `ViscousPath` (`kPressureSolve`, `kViscousSolve`; A25), `Module` (14 values; reconcile),
`RunMode` (`kOnePiece`, `kBatched`, `kBatchedWithSolvesInPieces`; `BuildRunPlan` chooses only the
first two, this module copies the batch plan's choice), `PlanFit` (`kFits`, `kDoesNotFit`: `BuildRunPlan` returns a
plan that does not fit, mode `kBatched` with every estimate filled, and `BuildSimulation` fails `kDoesNotFit` on
it), `MemoryEstimate` (with `on_cpu`), `ModuleBytes` (with `on_cpu`), `MemoryReport` (with `allocated_cpu_bytes`),
`CopyCounts`, `StepReport::copies`, `ReadCopyCounts`,
`BuildSimulation`, `BuildSimulationFromSnapshot`, `Simulation`, `SimulationSnapshot` (with what the next step
reads: `has_pressure`, `pocket_of_cell`, the inputs of `ComputeStepDuration` kept at commit, the gas part as
`gas::GasAmountsSnapshot`, the seven region arrays of cell resize including `is_grid_only_region`,
`grid_only_current_volume`, `grid_only_ingredient_volume`), `SourceCounter`, `ReadSnapshot`, `AdvanceTo`,
`AdvanceResult`, `StopReason`, `Callback<Function>`, `StepCallbacks` (every member a
`Callback<Function>{function, context}` pair: `make_foam_and_spray`, `choose_cell_size`, `on_try_done`,
`resize_cells`, `on_step_done`, `apply_grid_only_region`, `move_grid_only_liquid`, `run_stages`,
`read_largest_grid_only_speed`), `FoamAndSprayInputs`, `StepResult`, `RejectReason`, `kRejectReasonCount`,
`RejectContext`, `Stage` (stage 11 `kSolveViscosityAndPressure`), `kStageCount`, `StepLimit`, `kStepLimitCount`,
`Module` (with `kRenderOutput`), `StepFailure`, `ComputeRandomWords`, and every `ToMessage` and `ToName` of
`step_message.h`. `pressure::SolveSettings` holds no cancel pointer (the "every design" cancel rule), so the
ready scene hashes plain values only.

**Particles (PROVIDED, 3.7):** `ParticlesSnapshot` and `ParticlesSnapshot::Material` member by member (`id`,
`rest_volume`, `size_level`, `ingredient_fractions`, `mass`, `position`, `velocity_for_next_step`,
`fitted_velocity_change`, `volume_ratio`, `flags`, `springy_deformation`, `permanent_volume_change`,
`plate_side`, `position_when_grouped`, with `next_particle_id`, `is_full`), `SnapshotFields`,
`kOutputSnapshotFields`, `Field`, `FieldList`, `ToFieldList`, `HasField`, `NewParticles`,
`kMaxIngredientCount`, flag bits.

**Materials (PROVIDED, 3):** `Ingredient`, `Material`, `MaterialResponse`, `MaterialKind` (`kLiquid`, `kGas`,
`kSnow`, `kSpringySolid`), `OwnMotionSettings`, `AtSolidFaces`, `MaterialPair`, `MaterialTableInputs` with
`ingredient_export_names`, `MaterialTable` (`ingredient_export_names()`, `gas_ingredient_index()`),
`BuildMaterialTable`, `MaterialSettings`, `HasInternalForce`, `ToMessage`, `ToName(MaterialsError)`.

**Solids (PROVIDED, 3.2 to 3.4, 3.8):** `SolidSettings`, `SolidsSettings` (`has_liquid_or_gas`, filled here),
`PathKey`, `DrivenPath`, `AfterPath`, `FreeSolidSettings`, `FreeMotionLimits`, `MaterialTheLiquidCannotEnter`,
`SolidState`, `Orientation`, `ToOrientation`, `ComputeRotationVector(Orientation from, Orientation to)`,
`kMaxPlateCount`, `kMaxFreeSolidCount`, `CheckSolidsCanRun` (through the step's check), `SolidsSnapshot`,
`ExchangeTotals`, `BuildContactSolids` (passed through to render output by the frame writer),
`WavePaddleSettings`, `BuildWavePaddlePath`, `ToName(SolidsError)`.

**Gas (PROVIDED, 3):** `GasLaw`, `BurnSettings`, `GasSettings`, `MarkerSettings`, `ActiveGasSettings`,
`GasSourceKey`, `GasSource`, `GasWindKey`, `GasWind`, `CheckGasSources`, `GasAmountsSnapshot` (stored in the
saved state and restored by the step with `BuildGasAmountsFromSnapshot`), `ToName(GasError)`.

**Contact, grid, pressure, viscosity, transfer (PROVIDED):** `contact::DomainBox`, `ClosedSides`,
`SolidShape` (`kPlate` makes `materials::PlatesInScene::kSome`), `SolidPose`, `DomainSide`; `grid::GridFrame`,
`CellId`, `FacePieceId`, `kMaxSizeLevels`; `pressure::OpenSeaSettings`, `OpenSeaWave`, `SolveSettings`,
`AirPocketSettings`, `VolumeCorrectionSettings`, `AirPocket`, `BlobHistory`, `PressureLaw`;
`viscosity::ViscositySettings`; `transfer::TransferSettings`; each module's `ToMessage` and `ToName`. Each
settings struct is read field by field when the ready scene is written; it holds plain values only.

**Cell resize and batches (PROVIDED, 3.4, 3.7):** `cell_resize::CellResizeSettings`, `CellSizeSettings`,
`RegionSettings`, `BuildCellResize`, `CellResize`, `ResizeCells`, `ChooseCellSize`,
`ApplyGridOnlyRegionInStep`, `MoveGridOnlyLiquidInStep`, `ReadLargestGridOnlySpeed` (wrapped as the step's
`read_largest_grid_only_speed`), `ReadLastReport` (resize totals for the run report),
`BuildCellResizeMemoryEstimate`, `ToName(CellResizeError)`; `batches::BatchSettings`, `BuildBatchPlan`,
`BatchPlan`, `BuildBatchedRun`, `BatchedRun`, `ApplyStagesInBatches`, `BuildBatchesMemoryEstimate`,
`ToName(BatchesError)`; A15 and planned cost: `BatchPlan::solve_place` (`SolvePlace::kWholeGridOnProcessor`,
`kInPieces`), `max_solve_loaded_blocks`, `solve_batch_count`, `bytes_on_processor`, `bytes_on_cpu`,
`bytes_copied_per_try`, `pass_loads_per_try`, `solve_loads_per_iteration`, `solve_bytes_per_iteration`;
`BatchesFailure` (`needed_bytes`, `allowed_bytes`; errors `kStateDoesNotFitCpu`, `kBatchDoesNotFit`);
`batches::ReadLastReport` (`BatchReport` with `batch_count`, `bytes_to_processor`, `bytes_from_processor` and the
solve fields `solve_batch_loads`, `solve_bytes_to_processor`, `solve_bytes_from_processor`); cell resize's
`ResizeCellsReport::bytes_copied`, `copy_count`; `CellResizeSettings::grid_run_on` (set to the run's `run_on`).
A15: a batched run agrees with a one-piece run within the solver's tolerance, not byte for byte; each mode is
repeatable on its own. `run_mode` therefore stays in the fingerprint (decision 15).

**Surface, foam and spray (PROVIDED, render output 3, 4 and 5.7):** `RenderSettings`, `MotionSettings`,
`SurfaceSettings`, `FoamSettings`, `VolumeSettings`, `RenderLimits`, `CheckRenderSettings(settings,
smallest_cell_width, materials)`, `BuildRenderOutputMemoryEstimate` (passed to `step::BuildRunPlan`),
`BuildStepRecorder`, `RecordStep`, `CommitRecordedStep`, `RejectRecordedStep`, `ReadRecordedInterval`,
`RecordedInterval`, `MotionSample`, `FrameIdentity` (with `frames_per_second`), `FrameInputs` (I/O sets
`gas_law` to `&ready_scene.gas->law` in a scene with gas, null otherwise),
`BuildRenderOutput`, `BuildRenderOutputFromSnapshot`, `BuildRenderFrame`, `RenderOutputSnapshot`,
`ReadSnapshot(const RenderOutput&)`, `OutputProduct`, `GeometryKind`, `ToGeometryKind`, `OutputChannel`,
`ReadOutputChannels`, `kOutputContractVersion`, `InterchangeFormat`, `FormatChoice`, `ReadFormatTable`,
`ReadFormatChoice`, `WitnessRecord`, `ToName(RenderOutputError)`, `ReadRecorderCopyCounts(const StepRecorder&)`
(planned cost, seam 11); in `cli/` only (target
`fluidity_interchange`): `ToPointObject`, `ToPointObjects`, `WritePointsFile`, `WriteSurfaceFile`,
`WriteVolumeFile`, `WriteSolidMotionFile`, `ReadInterchangeLibraries`. The file names of its 5.7 table are
placed under `outputs/` of the run folder.

**Blender add-on (PROVIDED by its design, after its update per `changes-to-apply.md` 14):** writes the scene
file as JSON that follows `schemas/simulate/v2/scene_file.fbs`, in SI units and Z-up, with stable nonzero ids
per object kind; runs `fluidity capabilities` and compares `protocol` and the scene schema before it writes a
large scene; reads `<run_folder>/events.jsonl` (or standard output), never drops `frame_done`, `finding`,
`saved_state_listed`, `failed`, `done` or `stopped`, and writes `cancel` or `stop_after_frame`; finds the
executable in its own package, not on the path; reads the output declarations from `report` and `frame_done`
and types in no file name or channel; opens product files with Blender's native readers; runs the witness
tests of render output 5.7; maps one artist control to several physical values where it wants to (snow
"compactness"): the scene file takes the physical values.

**For the sand module:** the sand fields of the scene file (`friction_angle`, `cohesion` on a material of kind
`sand`), the `kSand` preset, and `MaterialKind::kSand` in the material kind enum of `scene_file.fbs`, added with
the sand module's `MaterialKind::kSand = 4`.

## 5. Method

Nothing here is numerical. What is ported is a set of rules that `main` gets right; what is new is one route
for every material, one internal file format, and open formats for everything a person looks at.

### 5.1 From scene file to ready scene (replaces `blender_resolve.cc` and the text packets)

1. **Read.** The scene file is UTF-8 JSON. It is parsed by the FlatBuffers text parser against
   `scene_file.fbs` (embedded in the executable), with strict JSON and unknown fields refused, then checked
   by the FlatBuffers verifier. A parse failure is `kSceneTextNotValid` with the parser's line and reason; an
   unknown field is `kUnknownSceneField` with its path (the parser stops at the first one).
2. **Check every value** before using any: finite, in range, ids nonzero and unique per kind (material,
   ingredient, start region, source, solid, gas source), solid ids below 2^32, `first_frame >= 0`, references
   resolve. `CheckSceneFile` collects every finding, in file order, each with its scene file path
   (`solids[2].path.keys[4].time`); `BuildRunScene` stops at the first.
3. **Fill defaults once.** Start from default-constructed `step::ReadyScene`, `StepSettings`,
   `CellResizeSettings`, `BatchSettings` and `RenderSettings`, so each module's default is written in one
   place, its own settings struct. A value the file gives replaces the default. A value it leaves out is
   recorded as a `FilledDefault` with where it came from. Material values come from `kMaterialPresets`
   (`io/material_presets.h`: `kCustom`, `kWater`, `kSyrup`, `kSnow`, `kSpringySolid`, `kAir`; decision 12;
   `kWater` has viscosity 0, A25)
   and each may be overridden by the file. `run_on` of `StepSettings`, `CellResizeSettings` and
   `SolidsSettings` is the one value of the file or `--run-on` (A12).
4. **Derive.** All NEW except where noted.
   - Grid: `grid_frame.origin = domain_box.low_corner`; `biggest_cell_width = smallest_cell_width *
     2^(size_level_count - 1)`; `new_cell_size_level = size_level_count - 1`.
   - Materials: one `materials::Material` per scene material; each start region of snow or a springy solid
     gets its own material with its own `material_id` unless it names `same_material_as` (shared-interface
     (c)). `stiffness_against_shear = E / (2 (1 + nu))` and `stiffness_against_squeeze = E nu / ((1 + nu)(1 -
     2 nu))` when the file gives stiffness `E` and squeeze ratio `nu` instead of the two numbers.
     `own_motion.kept` is the file's `own_motion_kept` (0.99 is no longer multiplied in: owner Q33).
     `ingredient_export_names` are the ingredients' names; `BuildMaterialTable` normalises and checks them. A gas
     material takes the gas rules of materials 5.1 (rest density `gas::ComputeAmbientDensity(law)`).
   - Start particles: each start region (a box or a ball with a pose) is sampled on a lattice of
     `particle_spacing = smallest_cell_width / 2` (8 per smallest cell), kept where the lattice point is
     inside the shape and outside every solid, moved by `spacing / 4 * (2 u - 1)` per axis with `u` from
     `step::ComputeRandomWords(counter = {lattice index, region_id}, key = random_seed)`, in ascending lattice
     index (z, then y, then x). `rest_volume = spacing^3`, `mass = rest_density * rest_volume` (mixture rest
     density with the region's ingredient shares). A scene may instead name an array file of explicit
     particles (`start_particles_file` with its digest).
   - Sources: the file gives `volume_rate` (m3/s). `rest_volume = particle_spacing^3` at the source's size
     level, `mass = rest_density * rest_volume`, `particles_per_second = volume_rate / rest_volume`, so a change
     of cell width keeps the flow. `main` exposed the particle rate; programme design row 12 drops that control.
   - Solids: `solid_id` is the file's. A driven path is given as keys of time, origin and orientation (w, x,
     y, z, own axes to world); `angular_velocity_until_next = ComputeRotationVector(key, next) / (next.time -
     key.time)`, zero on the last key. A wave paddle (`wave_stroke`) becomes `BuildWavePaddlePath` keys up to
     `end_time`. A free box with no inertia given gets `mass / 3 * (b^2 + c^2, a^2 + c^2, a^2 + b^2)` for half
     sizes `a, b, c`. `blocking_materials`: every material with `HasInternalForce`, `particle_radius =
     particle_spacing / 2`. `has_liquid_or_gas`: some material is `kLiquid` or `kGas`. Plates are the solids
     of shape `kPlate`. `SolidsSettings::gravity` is the scene's gravity.
   - Gas: `GasSource` keys `{time, fuel_rate, heat_rate}` and wind keys `{time, acceleration}` are copied into
     arrays `RunScene` owns; `gas::CheckGasSources` runs in step 2.
   - `end_time = ComputeFrameTime(last_frame)`; `snapshots_kept_at_once = max_waiting_frames + 2`.
   - Render output: `RenderSettings` from `outputs`; zero spacings mean render output's own defaults and
     are left as given; `CheckRenderSettings(render, smallest_cell_width, material_table)` runs in step 2.
5. **Write the record.** `ready_scene_bytes()` is a FlatBuffers buffer of `ready_scene.fbs` built from the
   C++ values with every default written out, in two tables: `physical` (everything that can change a
   committed step) and `presentation` (names, requested outputs with the render settings, `last_frame`,
   limits, saved-state cadence, `filled_defaults`). Every field of `ready_scene.fbs` carries the attribute
   `scene_path` naming the scene file field it comes from (a derived value names the field it was derived
   from: `particles_per_second` names `sources[].volume_rate`). It is listed as `ready_scene` in the run
   folder. `resume`, `report` and `run --outputs` read it; `resume` never reads the scene file again.

### 5.2 Fingerprints and the rule for continuing (ported idea, one check instead of three)

`scene_digest = BLAKE3("fluidity/scene/v2" + one zero byte + bytes of the physical table)`.
`run_fingerprint = BLAKE3("fluidity/run/v2" + zero byte + scene_digest + executable_digest + run_on +
processor model, driver and runtime text + run_mode + started_from)`.
`output_fingerprint = BLAKE3("fluidity/output/v2" + zero byte + run_fingerprint + product + material_id +
format + kOutputContractVersion + bytes of the product's settings tables)`. The product's settings tables are
`motion` and, for foam, foam sources (both read inside the step: `needs_simulation`), plus its own table
(surface: the group's `SurfaceSettings`; foam and spray: `FoamSettings`; a volume: its `VolumeSettings`;
points: the material id and `particle_fields`; solid motion: none).

- The bytes of the physical table need to be the same only for the same executable, and the executable's
  digest is in the fingerprint. So no canonical encoding is promised across builds, which is what made the
  encoders of `main` fragile.
- **Rule.** A run folder may be continued only by a run with the same fingerprint. The reason is the
  exactness requirement: the same executable on the same processor gives the same bytes, and nothing weaker
  can promise that "the next committed step is unchanged". On a mismatch `FindFirstDifference` names the
  part: scene, executable, run-on choice, processor, run mode.
- **Not in the run fingerprint**, each with a test that changing it changes no internal listed byte: object
  names, requested outputs and render settings (render output commits nothing: its 1), `last_frame` (a run can
  be extended), limits and memory shares (step 5.7: no result depends on a limit), saved-state cadence, queue
  size, `keeps_frame_inputs`.
- **The one bridge.** `BuildRunFromSavedState` starts a new run, in a new folder, from a saved state of
  another run (another build, another processor, an edited scene). It needs only that the state fits the
  scene (`step::kSnapshotDoesNotMatch` otherwise). Its fingerprint contains `started_from`, so the two runs
  can never be confused and the old folder is never written to.
- **Comparing a scene with a folder** (`CompareWithRunFolder`, `check --cache`). `simulation` is `kAbsent`
  without a file list, `kSame` when the run fingerprint the scene would have equals the folder's, else
  `kChanged` with `differing_part`. For `kScene`, the two `physical` tables (the scene's and the listed
  `ready_scene`'s) are compared field by field through FlatBuffers reflection and each differing field is
  reported by its `scene_path` attribute. Each output is `kAbsent` (asked now, not in the folder, or in the
  folder and no longer asked), `kSame` (equal output fingerprint), or `kChanged` with its differing
  `presentation` fields; `needs_simulation` is set when the simulation changed, when a `motion` or foam source
  field changed, when the output is foam and spray, or when the folder keeps no frame inputs.
- Playback needs no fingerprint: an open-format file carries its `FrameIdentity`, the run fingerprint and the
  contract version as metadata (render output 5.7).
- A change of a field's meaning or unit raises that file kind's schema version in `io/schema_versions.h`,
  even when the bytes would still decode (report, modernization "Separate serialization structure from
  physical meaning").

### 5.3 The array file, the run folder, the file list and the listing transaction

**Array file** (FluidiTy's own format, A14: `ready_scene`, saved state, stored frame inputs only), version 2:

```text
bytes 0..7    "FLUIDITY"
bytes 8..11   container version, 2 (unsigned 32-bit, little-endian)
bytes 12..15  header byte count H, at most 16 MiB
bytes 16..    H bytes: a FlatBuffers buffer, root ArrayFileHeader (array_file.fbs): kind, schema version,
              run fingerprint, frame, time, state_version, step_count, the kind's description table,
              and one entry per array: name, element type, elements per item, item count, byte offset
then zeros up to a multiple of 64, then the arrays, each starting at a multiple of 64 bytes
```

- An array is one physical quantity of one material or of the grid, with exactly the items that exist, in
  little-endian IEEE numbers or unsigned integers. It is packed from the snapshot's vectors, never from a
  buffer of the simulation: no limit-sized end, no padding inside items. A type with padding is split
  (`contact::PlateSide` becomes `plate_id` and `side`). Short lists (air pockets, source counters, solid
  states, limits; at most 65,536 items each) are tables in the header.
- Why the arrays are outside the FlatBuffers buffer: FlatBuffers offsets are 32 bits, so one buffer stops at
  2 GB, and it must be built whole in memory. A saved state of the breaking wave is larger than that. The
  arrays are written straight from the snapshot while BLAKE3 hashes the same bytes.
- No compression (the report asks for a measurement first; an `encoding` field per array is the later change).
  Little-endian machines only, checked at compile time.

**Run folder.**

```text
<run_folder>/writer.lock         held by the one process allowed to write; holds its process id, host, start
             file_list           append-only
             run_report.json     replaced in one rename; not listed
             events.jsonl        append-only, one event per line; not listed
             log.txt             append-only, the standard error text; not listed
             ready_scene         listed once
             saved_states/000040.state
             frame_inputs/000042.inputs
             outputs/points_17/points.000042.abc     outputs/surface_3/surface.000042.abc    (render output 5.7)
             tmp/                never read
```

**File list.** A 16-byte start ("FLUIDITY", version, kind), then entries. An entry is: byte count (32 bits),
a FlatBuffers `FileListEntry` (`file_list.fbs`: sequence, action `Start | Output | Listed | NotBuilt |
FrameDone | Removed`, and the fields of that action; `Start` holds the run fingerprint), then 8 check bytes =
the first 8 bytes of BLAKE3(entry bytes + the previous entry's check bytes). A reader takes entries until one
fails its check or its sequence is not the previous plus one: the valid entries are the run, the rest is
`ignored_end_bytes`. A file exists for a reader exactly when the list names it. A directory scan proves
nothing. An `Output` entry is written for every declaration at the start of a new run and at a rebuild.

**`OutputFile` and `WriteListedFile`, in this order** (ported from `PublishFrame`,
`liquid_cache_store_3d.cc:703-832`):

1. `BuildOutputFile`: refuse if the final name is listed with another digest (`kFileAlreadyListed`); create
   `tmp/<name>.<process id>`.
2. The writer fills it: `WriteArrayFile` (hashing every byte), or a library writing to `path_while_writing()`
   and closing the file.
3. `WriteListedFile`: flush the file to disk; for a file written through the path, read it once and hash it.
   If the final name is listed with the same digest, delete the temporary file and return the listed entry.
4. Check (`CheckAfterWrite`): the size, and for an array file the header and every array extent; for a saved
   state also every byte read back and the digest compared (decision 7).
5. Rename to the final name without replacing. A file already there is not listed (step 1), so it is a
   leftover of a crash between steps 5 and 6: it is deleted first.
6. Append the `Listed` entry and flush the list. This is the instant the file is listed.
7. Flush the folder (POSIX). Return the entry.

**What a crash leaves, and what the next writer does** (`BuildRunFolderToContinue`, holding the lock):
before step 5, a file in `tmp/`: deleted. Between 5 and 6, a final-named file that is not listed: deleted.
Inside 6, a torn entry: the list is cut back to its last good entry. A listed file that is missing or has
other bytes (a folder copied in part): never repaired silently; `report --check-files` names it, a reader
gets `kFileNotFound` or `kDigestDiffers` for that file alone, and a resume takes the next older saved state.
Only `tmp/` and files under `saved_states/`, `frame_inputs/` and `outputs/` are ever deleted.

**What is promised.** After `WriteListedFile` returns, the file survives the end of the process. Survival of
a power cut is as good as the file system's flush; nothing is claimed for network folders.

**`WriterLock`** (ported, `liquid_cache_files_3d.cc:131-133`): an exclusive lock of the operating system on
`writer.lock`, released by the system when the process ends, crashed or not. The writer waits at most one
second for it, then gets `kRunFolderInUse` and changes nothing; once it holds the lock it writes its process
id, host name and start time into the file. Readers take no writer lock: the list only grows and a listed file
never changes. `report` probes the lock: it asks for a shared lock without waiting and releases it at once;
refused means held (`is_held`), granted means the holder is gone. The one-second wait of a writer means a probe
never refuses one. The public type is `RunFolder`; holding one is holding the lock.

### 5.4 What a saved state holds, and why

The contract: a run continued from a saved state lists, from the next frame on, the internal files and
snapshot digests the uninterrupted run lists. Every value is stored for a reason or rebuilt by a stated rule.

| Stored | From | Why |
|---|---|---|
| time, `state_version`, `step_count`, frame | `SimulationSnapshot` | The step duration to the next frame time is computed from them |
| every particle array of every material, in grouped order, `position_when_grouped`; `next_particle_id` | `particles::ParticlesSnapshot` (full) | Sums run in grouped order (owner Q22); ids are never reused; the grouping is restored to the byte |
| active cell ids | `active_cells` | The only description of a mixed-size grid (`grid::GridInputs`) |
| pressure, `has_pressure`; `velocity_after_pressure` when kept | `SimulationSnapshot` | The next step's carried pressure starts from it; grid-only cells and gas markers read the committed velocity |
| air pockets, `pocket_of_cell`, `BlobHistory` with `next_blob_id` | pressure through the step | `FindAirPockets` matches against the committed pockets |
| source counters (`owed_particles`, `next_child_number`) | step 5.4 | With `random_seed` they are the whole random state: the generator is keyed by seed, source and child number |
| the values the next duration is chosen from (largest speed over cell width, the material sound limit, the air pocket limit, the gas step limit, the grid-only speed of the last commit) | `SimulationSnapshot` | `ComputeStepDuration` reads committed values only |
| solid states and exchange totals | `solids::SolidsSnapshot` | Poses and velocities of free solids; totals for the balance check |
| the seven gas arrays by cell id | `gas::GasAmountsSnapshot` | Committed amounts; restored with `BuildGasAmountsFromSnapshot` |
| the seven region arrays (`is_grid_only_region`, `layers_from_grid_only_region`, `merge_ready_duration`, `grid_only_mass`, `grid_only_rest_volume`, `grid_only_current_volume`, `grid_only_ingredient_volume`) by active cell | `SimulationSnapshot` (cell resize) | Liquid without particles, and the merge wait |
| limits | `SimulationLimits` | So a continued run does not start by raising them again; no result depends on them |
| last mesh per surface group, foam particles with owed particles, the last motion sample | `render_output::RenderOutputSnapshot` | A continued run builds the same products: the mesh is moved, not rebuilt; foam goes on |

Rebuilt, not stored: the grid tables (`grid::BuildGrid` from the cell ids), `group_begin` and `cell_begin`
(particles 5.9), every working buffer, the pressure ladder, `SolidsAtFaces` and `SolidCoverage` (computed per
try), memory reports and timings.

When: at `first_frame`, then every `saved_state_every_frames` frames, then at the frame a `stop_after_frame`
ends on. Only at frame times (ported). It is written on the output worker after the frame's products were
built, so the render output snapshot belongs to the same frame (render output 4, item 6). The newest
`saved_states_kept` are kept; an older one is removed after the new one is listed, so there is always one.

### 5.5 Outputs and the stored frame inputs

**Products** (A14). Everything a person or another program plays back is an open-format file written by render
output's writers. This module declares the products and lists their files; it never writes their bytes.
`BuildOutputDeclarations` makes one `OutputDeclaration` per product the scene asks for:

| Asked for | Output ids | Product, geometry kind |
|---|---|---|
| `points.materials[]` (never gas) | `points_<material_id>` | `kParticles`, points |
| each `SurfaceSettings` group | `surface_<lowest material_id>`, `droplets_<lowest material_id>` | `kSurface`, surface mesh; `kDroplets`, points |
| `FoamSettings` | `foam_and_spray` | `kSpray`, `kFoam`, `kBubbles` in one file, points |
| `foam_volume`, `gas_volume` | `foam_volume`, `gas_volume` | volumes |
| `solid_motion` | `solid_motion` | solid motion |

The format of each is `ReadFormatChoice(ToGeometryKind(product)).chosen`, the file pattern is render output's
5.7 table under `outputs/` with that format's extension, the channels are the rows of `ReadOutputChannels()`
for the product that the scene makes present (shares with two or more ingredients, requested particle fields),
each with `is_witnessed` from the format's witness records. The declaration is fixed for the run; a rebuild
writes a new one.

**Stored frame inputs** (`kFrameInputs`, an internal run record). With `keeps_frame_inputs`, each frame lists
one array file holding `KeptFrameInputs`: what `render_output::FrameInputs` is made of, in doubles. It lets
`run --outputs` rebuild a product after its settings changed without simulating, and `read-particle` answer
with the full 64-bit id and double values. Arrays (per material `m` = its `material_id`):

| Array | Type | Unit, frame | Note |
|---|---|---|---|
| `particles.m.id` | unsigned 64 | none | Stable identity; order inside the file is the grouped order |
| `particles.m.position` | 3 doubles | m, world, Z-up | |
| `particles.m.velocity_for_next_step` | 3 doubles | m/s, world | A1 |
| `particles.m.rest_volume`, `.volume_ratio` | double | m3; none | |
| `particles.m.size_level` | unsigned 8 | none | |
| `particles.m.ingredient_fractions` | `ingredient_count` doubles | none | With two or more ingredients |
| `particles.m.<field>` for `particle_fields` (`mass`, `fitted_velocity_change`, `springy_deformation`, `permanent_volume_change`, `plate_id`, `side`, `flags`) | as in particles 3.7 | as there | The same names and units in a saved state, where all are written |
| `grid.active_cells`, `grid.is_grid_only_region`, `grid.grid_only_current_volume`, `grid.grid_only_ingredient_volume` | unsigned 64; unsigned 8; doubles | none; m3 | What the surface field reads (render output 5.3) |
| `gas.cells` and the seven gas arrays | unsigned 64; doubles | as in `GasAmountsSnapshot` | With gas |
| header: solids | table | m, rad, world | `solid_id`, state, totals of every solid |
| `motion.s.*` for each sample `s` of `recorded` and `sample_at_frame` | lattice keys; floats; unsigned 8 | m/s, m | `MotionSample` arrays as render output 3.2 declares them |
| `foam_events.*`, header: interval counts, `liquid_totals_at_frame` | as `FoamSourceEvent` | m, m/s, m3 s | `RecordedInterval` |

The same packing of the snapshot arrays (the `particles.*`, `grid.*`, `gas.*` rows and the solids table) gives
`ComputeSnapshotDigest`, written in every `FrameDone` entry whether or not the file is kept. Cost per frame:
about 64 bytes per particle plus the motion samples; `keeps_frame_inputs = false` saves it and makes every
product `needs_simulation_to_rebuild`.

Per material, this module carries: liquids and mixtures, `ingredient_fractions` when a material has several
ingredients; snow, `springy_deformation` and `permanent_volume_change`; springy and cut solids, also
`plate_id` and `side`; moving solids, their states in every frame and their totals in a saved state; gas, its
markers and the seven arrays (gas beside liquid, A10: both in one file, the markers are never a points output).

### 5.6 The frame loop (`BuildRun`, `AdvanceRunTo`; replaces the 778-line function)

1. Build the `RunScene` (5.1).
2. `step::CheckSceneCanRun`. A blocking finding ends here: nothing was allocated.
3. `BuildSceneBudget` (5.10): `ReadMemoryAvailable` for the processor and for the CPU (or the overrides
   `assume_memory_bytes`, `assume_cpu_memory_bytes`), `step::BuildRunPlan` with the entries of
   `BuildRenderOutputMemoryEstimate` and `BuildIoMemoryEstimate` as `output_estimate` and
   `BuildCellResizeMemoryEstimate` as `cell_resize_estimate`; with a batched plan, `batches::BuildBatchPlan`, whose
   `max_blocks_per_batch`, `max_loaded_blocks`, `max_loaded_particles` and (A15) `max_solve_loaded_blocks` are
   copied into the `RunPlan`, and `mode = RunMode::kBatchedWithSolvesInPieces` when `BatchPlan::solve_place ==
   SolvePlace::kInPieces` (A15 left this choice to I/O); its failure becomes the budget's verdict; then
   `BuildRunBudget`. Nothing is allocated. A verdict other than `kFits` ends here (`kModuleFailed`, step
   `kDoesNotFit`), before any folder exists. The budget is reported before the run (`started`, standard error,
   `log.txt`) and kept in `RunReport::budget`.
4. Compute the fingerprint. New run: `BuildNewRunFolder`, list the ready scene, `WriteOutputDeclarations`.
   Resume: `BuildRunFolderToContinue` (lock, fingerprint, cleaning).
5. Compose the step callbacks, each pair with its own context: `resize_cells` = `ResizeCells`,
   `choose_cell_size` = `ChooseCellSize`, `apply_grid_only_region` = `ApplyGridOnlyRegionInStep`,
   `move_grid_only_liquid` = `MoveGridOnlyLiquidInStep`, `read_largest_grid_only_speed` = a wrapper of
   `ReadLargestGridOnlySpeed`, and in either batched mode `run_stages` = `ApplyStagesInBatches`, each with the
   context its declaration in the cell resize and batches design names (`Run` builds `CellResize` with
   `BuildCellResize(settings, plan.limits, ingredient_count)` and `BatchedRun` with `BuildBatchedRun(batch_plan,
   scene)` after the plan, and owns both); `make_foam_and_spray` = `RecordStep` on the step recorder
   (returns false only on `kRunOnCallFailed`); `on_try_done` and `on_step_done` are this module's: they call
   `CommitRecordedStep` or `RejectRecordedStep`, copy `RunProgress` and `LastTry`, stop at a frame time, and feed
   the measured cost (`ApplyTryToRunCost`, `ApplyStepToRunCost`, 5.10).
   Build the simulation: `step::BuildSimulation`, or `ReadSavedState` then `step::BuildSimulationFromSnapshot`;
   `FrameWriter::start` with the saved state's render output snapshot. If the newest saved state fails its
   checks, the next older one is tried and the `started` event says so.
6. New run: hand the start state over as the frame `first_frame`, with a saved state.
7. For each frame `f` after the state's frame, up to `last_frame`:
   1. If `stop_after_frame_requested`: stop.
   2. `step::AdvanceTo(simulation, ComputeFrameTime(frames, f), &cancel_requested)`.
   3. `kCancelled`: stop; frame `f` gets no file. Any other stop than `kReachedTarget`: stop with that failure.
   4. `step::ReadSnapshot` with every field if `f` is a saved-state frame, else `kOutputSnapshotFields` plus
      `particle_fields`; `ReadRecordedInterval` with `RecordsSteps`. This is the one large copy per frame and
      the simulation waits for it.
   5. `AddFrameOutput` (waits when the queue is full, 5.7).
   6. On the worker, in frame order: `ComputeSnapshotDigest`. If `f` is already done in the list (a resume
      from an older saved state): compare with the listed `snapshot_digest`, call `write_products` with
      `is_already_done` (it advances render output and lists nothing; its `FrameInputs::gas_law` is the
      ready scene's gas law in a scene with gas), report `frame_checked`, or fail with
      `kFrameDiffersOnResume`. Otherwise: `write_products` (every product file of the frame, or `NotBuilt`),
      `WriteFrameInputs` with `keeps_frame_inputs`, `WriteFrameDone(f)`, then on a saved-state frame
      `WriteSavedState` with the frame writer's render output snapshot and `ApplySavedStateRetention`, then
      `WriteRunReport`, then `on_frame_done`.
   The frame worker also calls `ApplyFrameToRunCost` with the snapshot's array bytes and records a `FrameMark`
   for `ComputeTimeProjection`.
8. `WriteWaitingFrames`, `BuildCostLines`, final `WriteRunReport` with the status, the budget and the cost, the
   last event, then the cost lines and warnings on standard error and in `log.txt`.

The saved state is listed after `FrameDone`: a crash between them leaves a complete frame and the older
saved state, never a saved state for a frame that is not complete. Product files of frames already done are
never written again on a resume: open-format bytes are not reproducible (render output 5.7, OpenVDB writes a
UUID), so the snapshot digest is the check.

**Rebuild** (`run <scene_file> --into <run_folder> --outputs <ids>`, `RebuildOutputs`). The scene's run
fingerprint must equal the folder's (`kWrongRunFingerprint` otherwise), each id must be declared now or in the
folder (`kUnknownOutput`), no named output may need the simulation (`needs_simulation`: `kFrameInputsNotKept`
or the finding `start_new_run`), and every frame done must have stored frame inputs. Then: take the lock,
`RemoveOutputFiles`, write the new declarations, `FrameWriter::start` with nothing kept, and for each frame
from `first_frame` to `last_frame_done` read its `KeptFrameInputs` and call `write_products` with
`only_output_ids`; events as for a run (`frame_done` lists only the rebuilt files). A rebuild stopped half way
leaves the outputs listed up to its last frame; the next rebuild starts again from `first_frame`.

### 5.7 The output queue and waiting

One simulation thread, one output worker, at most `max_waiting_frames` (2) frames read from the simulation and
not yet listed. One worker, because a frame's surface is moved from the previous frame's mesh and foam goes on
from the previous frame (render output 4, item 4). When the queue is full the simulation waits. No frame is
dropped, thinned or written late, for any kind of run; a preview is a short run into its own folder, chosen by
the add-on. While it waits the heartbeat says `waiting_for_output` with the number of waiting frames, and the
time goes to `output_wait_elapsed_seconds`, not to the solver's time. Memory: `snapshots_kept_at_once`
snapshots on the CPU, counted by the step's `kOneSnapshot` rows, so the estimate of step 3 already includes
them; render output's own rows are passed to the plan.

This answers step question 8: no ring of committed versions. A blocking `ReadSnapshot` once per frame and a
queue of owned CPU snapshots are enough for one frame per several steps; a ring is a measured change later.

A failure of the folder on the worker stops the run: the worker records it, takes no further frame, and the
simulation thread gets `kOutputStopped` at its next `AddFrameOutput`, at the latest one frame later. Frames
listed before the failure stay listed. A product that cannot be built is not such a failure: it is listed as
not built with its reason, a `finding` event is sent, and the run goes on (render output 4, item 7).

### 5.8 Stop requests, events and exit codes

- One helper thread polls standard input every 25 ms without blocking and writes the heartbeat every 0.5 s
  (ported, `simulate_job.cc:262-288`). `cancel` sets `cancel_requested`; `stop_after_frame` and the end of
  standard input set the other flag. An interrupt or terminate signal sets `cancel_requested` and does nothing
  else. The program starts no child process: ending its process ends the whole run.
- `cancel` is read by the step at its safe points (step 5.10) and inside the long solves through their cancel
  input: the try in progress ends as a reject, nothing of the current frame is listed, frames already handed
  over are written, the run ends with `stopped`. Work since the last saved state is computed again on resume.
- `stop_after_frame` finishes the frame in progress, lists it with a saved state, then ends with `stopped`.
  Nothing is lost. The add-on's Pause is this; its Stop Now ends the process.
- A heartbeat that cannot be written to standard output means the parent is gone: the run stops as for
  `stop_after_frame` (`kEventsNotWritable`, ported; the events go on into `events.jsonl`).
- `sent_at` is the system clock in UTC when the event is built; `last_advance_sent_at` is the `sent_at` of the
  last commit or the last listed frame, whichever is later. The heartbeat is written even while the simulation
  thread is inside a long solve, so "the process is alive" and "the simulation moves" are two ages.
- Events are built as FlatBuffers tables and printed by the FlatBuffers text generator on one line, so the
  schema file is the protocol and no JSON is assembled by hand. Text printed this way keeps 12 significant
  digits of a double (17 for `sent_at`): it is for progress and reports and is never read back as a source of
  numbers (exact numbers are in binary files). A line is at most 16 KiB; only `message` is cut to fit. 64-bit
  ids are printed as decimal text. `events.jsonl` receives the same lines, each flushed.
- `ToExitCode`: `kSceneTextNotValid`, `kUnknownSceneField`, `kSceneValueOutOfRange`, `kUnknownPreset`,
  `kDuplicateId`, step `kSceneNotSupported`, `kDoesNotFit` (also a budget verdict other than `kFits`): 3. `kRunFolderInUse`, `kRunFolderNotEmpty`,
  `kNoRunInFolder`, `kWrongRunFingerprint`, `kNoSavedState`, `kSavedStateDoesNotMatchScene`: 5. Any
  `kRunOnCallFailed`, step `kSimulationUnusable`: 6. A stop on request: 4. Everything else: 1.
- `check` never picks another `RunOn` (A12): when `CheckCanRunOn` fails for the requested one, or the plan does
  not fit, that is a blocking finding with the suggested changes `run_on_cpu` (when GPU was requested) and
  `use_bigger_cells`. `check` prints `scene_checked` whenever the scene file could be read, also with exit 3,
  and stops promptly when killed, leaving nothing behind (it writes no file). It calls `BuildSceneBudget`, puts
  the budget in `scene_checked.budget` (also when it does not fit) and writes `budget.lines` to standard error;
  exit 0 when the verdict is `kFits` and no other finding blocks, else 3 (the `does_not_fit` finding carries
  `use_bigger_cells` and, for `kDoesNotFitCpu`, the suggestion text of the last line in its message). `check`
  and `run` build the same budget from the same scene and memory.
- `capabilities` reads only constants and the executable's own tables: it returns within 200 ms and touches no
  GPU unless `--run-on gpu` is given.
- The message of a finding is built at this boundary only: `ToMessage` of the module's error, then the object
  name from `RunScene::names()`, the frame, the magnitude and limit, and the last saved state, for example
  "Pressure did not converge near Paddle in frame 42 (error 3.1e-4 1/s, allowed 2e-5 1/s). Frames 0 to 41 are
  complete; the last saved state is frame 40."

### 5.9 The three packages

| Package | Used for | Linked into | Why not less |
|---|---|---|---|
| FlatBuffers 25.12.19, with `flatc` | Schemas of the scene file, ready scene, array file header, file list entry, run report, events. The verifier before any length or offset is trusted. The text parser for the scene file and the text generator for events and reports (`idl_parser.cpp`, `idl_gen_text.cpp`). Reflection for `check --cache` and test 8 | `io/`, `cli/`; generated headers stay private to them | It replaces a hand-written scene parser, a JSON writer and six hand-checked binary layouts with one schema each. Not used for bulk arrays (5.3) |
| BLAKE3 1.8.5 | The digest of every listed file, the scene digest, the executable digest, the run and output fingerprints, the snapshot digest, the check bytes of file list entries | `io/` | One small C library, streaming, fast enough to hash a saved state while it is written. A checksum would detect damage but could not serve as the fingerprint |
| CLI11 2.7.2 | Six commands, their options and help text | `cli/read_command_line.cc` only | Decision 5 |

`cli/read_command_line.cc` is the only file built with exceptions (and run-time type information) on; it
catches every `CLI::Error` and anything else, and returns `CommandLineFailure`. The executable target is
`fluidity`; it links the native library, these three packages and `fluidity_interchange` (render output 5.7:
OpenUSD, Alembic, OpenVDB). `flatc` runs at build time from the Conan build context, never from the path
(ported, `cmake/LiquidCacheSchemas.cmake:1-9`). Version pins are the ones on `main` (`conanfile.py`); the
planning engineer confirms `flatbuffers::Parser`, strict JSON, the refusal of unknown fields, reflection and
the one-line text output against 25.12.19 before the first schema commit.

### 5.10 The planned and the measured cost (NEW; planned cost, A17, A22)

Principle (A17): a run is all CPU or all GPU; in a batched GPU run only what does not fit (the particles, and in
`kBatchedWithSolvesInPieces` the solver buffers) stays in CPU memory, moved in planned copies that the planner
counts and the reports measure. The owner: "warning the user about the expected cost (memory and time) is nice".

**Three ways to run (A15).** `kOnePiece`; `kBatched` (particle stages in batches, every solve once on the whole
grid on the run's processor); `kBatchedWithSolvesInPieces` (as `kBatched`, and the solvers' working buffers live in CPU
memory and go through the GPU one loaded piece at a time, at most `max_solve_loaded_blocks` blocks).
`step::BuildRunPlan` chooses only `kOnePiece` or `kBatched`; `BuildSceneBudget` copies `BuildBatchPlan`'s choice.
`RunPlan::allowed_cpu_bytes` bounds the CPU rows (`kStateDoesNotFitCpu` beyond it). Between modes the rule is
tolerance, not bytes: a batched run agrees with a one-piece run within the solver's tolerance (step 6.4 over
several steps); each mode with the same input gives the same bytes, and a batched run does not depend on the cut.

**Budget sources** (`BuildRunBudget`). One piece: `processor_memory.budget_bytes = estimate.on_processor`,
`cpu_memory.budget_bytes = estimate.on_cpu`; batched: `BatchPlan::bytes_on_processor`, `bytes_on_cpu`;
`allowed_bytes` = `RunPlan::allowed_bytes` and `allowed_cpu_bytes`; totals and free from the two `MemoryAvailable`
(or `assume_memory_bytes` as both total and free). `modules` from `estimate.by_module` (processor bytes = the four
sums minus `on_cpu`). `copies`: `step_*` from `planned_step_copies`; `pass_*` from `bytes_copied_per_try`,
`pass_loads_per_try`; `solve_*` from `solve_bytes_per_iteration`, `solve_loads_per_iteration`, limit
`pressure_solve.max_iterations`; with cell resize on a GPU grid `resize_bytes = (456 + 8 * 64) *
limits.max_cell_blocks`, `resize_copies = 2` in one piece, and `8 * 64 * limits.max_cell_blocks`, 1 in a batched run
(the resize reads the step's grid copy), else 0; `snapshot_bytes_per_frame` = the sum of `one_snapshot` over
`by_module`; the two totals add step, pass and resize. `particle_count` = `counts.start_particles +
counts.particles_from_sources`; `smallest_cell_width` from the grid frame. (A25) `viscous_path`,
`viscous_effect_cell_width_m`, `viscous_effect_step_seconds` from `RunPlan::viscous_path`,
`viscous_effect_cell_width`, `viscous_effect_step_duration`; `liquids` from `RunPlan::liquid_viscous_effects`
(name from the ready scene's ingredient at `ingredient_index`, `is_viscous = viscosity > 0`, `effect_not_visible =
0 < viscous_effect_per_step < viscosity::kSmallestVisibleViscousEffectPerStep`).

**Mode and verdict.** `fit == kFits` and `kOnePiece`: `kFitsProcessor`, `kFits`; `kBatched`:
`kParticlesDoNotFitProcessor`; `kBatchedWithSolvesInPieces`: `kSolverBuffersDoNotFitProcessor`; a GPU run with
`fit == kDoesNotFit` or a `BatchesFailure` `kStateDoesNotFitCpu`: `kStateDoesNotFitCpu`, `kDoesNotFitCpu`;
`kBatchDoesNotFit`: `kLayerDoesNotFitProcessor`, `kDoesNotFitProcessor`; a CPU run that does not fit:
`kOnePieceDoesNotFitCpu`, `kDoesNotFitCpu`.

**Suggestions** (printed as "estimates"): `s = allowed / needed` of the failing memory; `max_particle_count =
floor(s * particle_count)`; `min_smallest_cell_width = ceil_1e-4(w / cbrt(s))` for a memory verdict,
`ceil_1e-4(w / sqrt(s))` for a layer verdict (no particle suggestion); `more_memory_bytes =
ceil(needed / memory_share_of_total) - total` (total known) else `needed - allowed`.

**Budget lines** (exact templates; sizes in decimal units: below 1000 `"<n> B"`, else two decimals of the largest of
kB, MB, GB, TB not above the value; integers grouped by commas; widths `%.4f m`; `<p>` is `cpu` or `gpu`):
`plan for <scene_name> on <p> (<processor_model>)`; `mode: <one piece | batches | batches with solves in pieces |
none> (<reason>)`; `<p> memory: budget <a> of <b> allowed (<t> total, <f> free)` per processor (a CPU run prints
one `cpu memory:` line with both budgets summed; total unknown: `(total not known, <f> free)`; a failed plan:
`one piece needs` / `batches need` in place of `budget`); `largest on <p>: <module> <bytes>, ...` (up to 5 by
processor bytes); in batched modes `largest in cpu memory: ...`; `copies per step: at most <b> in <n> copies` (one
piece GPU), `copies per step: <b> in <n> batch loads, at most <b> in <n> step copies` (batched; `, <b> in <n>
resize copies` appended with cell resize on a GPU grid), `copies per step: none` (CPU one piece), `copies per
step: not planned (the scene does not fit)`; `solve copies: <b> in <n> piece loads per iteration (<b> at the
limit of <n> iterations)` (solves in pieces only); `copies per frame: snapshot up to <b>`; `small records: status
records, air pockets, gas blobs and per-solid results are not counted`; `time: not estimated before the run; a
projected finish is printed after 3 frames` (or `time: not estimated` when it does not fit); last line `fits:
yes` or `fits: no. To fit (estimates): at most <n> particles (now <n>), or a smallest cell of at least <w> m (now
<w> m), or <b> more <p> memory`. Reasons: `everything fits the <p>`; `the particles do not fit the <p>; particle
stages run in <n> batches, solves on the whole grid`; `the particles and the solver buffers do not fit the <p>;
particle stages run in <n> batches, solves in <m> pieces`; `in one piece it does not fit the <p>, and in batches
the state does not fit cpu memory`; `one layer of cell blocks with its border does not fit the <p>`; `the scene
does not fit cpu memory`.

**Liquids lines** (NEW, A25; only when `liquids` is not empty, right before the last `fits:` line; `%g` for viscosity
and density, `%.3g` for the number, `%.4f` for width and step; ingredient export names). A liquid is viscous exactly
when its viscosity setting is above 0; the scene is viscous when any liquid is, then stage 11 runs for the whole run,
else stage 12 (the step chooses once; this module only prints it). `kPressureSolve`: `liquids: pressure solve
(every liquid has viscosity 0)`, then per liquid `  <name>: viscosity 0 Pa s, density <d> kg/m3, inviscid`.
`kViscousSolve`: `liquids: viscous solve (a liquid has viscosity above 0); effect per step at the smallest cell <w> m
and the largest step <t> s`, then per liquid `  <name>: viscosity <v> Pa s, density <d> kg/m3, viscous effect per
step <n>` (viscosity 0: `  <name>: viscosity 0 Pa s, density <d> kg/m3, inviscid`), followed, when
`effect_not_visible`, by `  note: <name>'s viscosity changes velocity differences between neighbouring cells by less
than 0.1 % per step, which is not visible; set its viscosity to 0`. The number is the step's
`viscosity * step_duration / (density * cell_width^2)` at the largest step and the smallest cell; below
`kSmallestVisibleViscousEffectPerStep = 2.5e-4` one step changes the difference between neighbouring cells by under
0.1 %. `check` and the run's start print these lines as part of the budget lines.

**Not counted** (each bounded, listed once as "small records"): a stage's status record (at most 64 B), solver
progress records, `AirPocket` records and `BlobHistory`, `GasBlob` records, per-solid and per-set result rows,
`FindGridEdgesOpenToSky`'s box marks, `StepReport` sums and `CommittedStepLimits`.

**Measured cost** (`RunCost`). `on_try_done` calls `ApplyTryToRunCost` with the batched run's `ReadLastReport` (null
in one piece); `on_step_done` calls `ApplyStepToRunCost` with `ReadMemoryReport` and the copies of the resize (this
step's `ResizeCellsReport`) plus the difference of `step::ReadCopyCounts` not yet seen in a try (the grid copy
refresh after a resize) and, from render output, `ReadRecorderCopyCounts` since the last committed step; the
frame worker calls `ApplyFrameToRunCost` with the snapshot's array bytes. A committed step's copies = its tries'
copies (rejected ones included) + those after the commit; mean and max are over committed steps. A try's copy
count is `report.copies.copy_count + 2 * batch_report.batch_count + batch_report.solve_batch_loads` (a load is one
copy); its bytes are `report.copies` both directions plus `bytes_to_processor`, `bytes_from_processor`,
`solve_bytes_to_processor`, `solve_bytes_from_processor`; solve bytes per iteration = the try's solve bytes over
`max(1, pressure_iteration_count)` (gas and viscosity sweeps add to the bytes, so a gas or syrup scene can read
above plan; the warning then says so). `BuildCostLines` runs at the end of `AdvanceRunTo` (every `RunStop`) and at
each `WriteRunReport`. Warnings: memory measured above `kMemoryWarningShare` times its budget (per processor);
copies per step (mean) above `kCopyWarningShare` times the planned; solve bytes per iteration (mean) the same
(percent rounded to the nearest integer). Lines: `cost of this run, planned / measured:`; `<p> memory: <a> / <b>`
(gpu, then cpu; a CPU run prints one `cpu memory:` line with both summed); `copies per step: at most <b> / mean
<b>, max <b> over <n> steps`; `copies per frame: up to <b> / mean <b> over <n> frames`; `solve copies per
iteration: <b> / mean <b>, <x.x> iterations per step` (solves in pieces only); `time by stage over <n> tries:
<stage name> <x.x> s, ...`; `time by activity: simulate <x.x> s, snapshot <x.x> s, build outputs <x.x> s, write
<x.x> s, wait <x.x> s`; `warning: <p> memory <b> is <k> % above the planned <b>`; `warning: copies per step <b>
(mean) are <k> % above the planned <b>`; `warning: solve copies per iteration <b> (mean) are <k> % above the
planned <b>`. Stage names are `ToName(step::Stage)`.

**Time** (honest): nothing before the run. `simulated_seconds_per_second` over the last up to 6 frames listed in
this process (at least 2; fewer: 0); `projected_remaining_seconds = (end_time - last frame time) / rate` from 3
frames on, `-1` before (a resumed run starts counting again). Sent in `heartbeat` and `frame_done`; `run` and
`resume` print `time: about <duration> left (<rate> simulated s per s over frames <a> to <b>)` on standard error
and in `log.txt` after the 3rd frame listed in this process and every 10th after it. Durations: `<n> s` below 60 s,
`<m> min <ss> s` below 1 h, `<h> h <mm> min`; rate 3 significant digits.

**Where it appears.** `check`: `scene_checked.budget`, the lines on standard error, exit 3 when `verdict != kFits`.
`run` and `resume`: `started.budget`, the lines on standard error and in `log.txt`; at the end `done` / `stopped` /
`failed` carry `budget` and `cost`, the cost lines and warnings go to standard error and `log.txt`;
`run_report.json` carries `budget` and `cost` (cost rewritten with each report). A refused budget gives exit 3 with
the lines printed and no folder.

## 6. CPU and GPU

This module has no `_gpu.cu` file and no shared formula header. It never sees a buffer of the simulation: on
either processor it receives a `step::SimulationSnapshot` and a `render_output::RecordedInterval`, which the
step, particles and render output fill with exactly the items that exist (their designs: "the one large copy
to the CPU"). The report's first output step (CUDA 13.1: pack only the fields a product needs) is therefore
the `SnapshotFields` argument, and its second (overlap packing, copy and writing) is the output queue.

- Threads: the simulation thread; one output worker; one helper thread for the heartbeat and standard input.
  Shared between them: the queue (a lock and two conditions), `RunProgress` (a lock), `StopRequests`
  (atomics), `EventWriter` (a lock around one whole line on both destinations).
- Cell blocks and batches: a snapshot covers the whole state and is in grouped order, so it does not matter
  how a stage was cut into ranges. Where each buffer of a batched run lives is the step's placement rule (A17,
  step plan Appendix A: a whole array on the run's processor when a whole-grid part uses it, else in CPU memory
  when a whole-state stage, the commit or a later pass uses it; otherwise only a loaded instance); in a batched
  GPU run the particles (and, with solves in pieces, the solver buffers) are in CPU memory and the grid state
  stays on the GPU. `ReadSnapshot` gives the same snapshot in every layout; nothing here changes. This module
  only shows the copies the step, batches, cell resize and render output count (5.10).
- Memory and copies are reported per processor: `MemoryEstimate::on_cpu` (rows `where = kCpu` plus snapshots)
  beside `on_processor` (only `kRunOn` rows); a CPU run prints one `cpu memory:` line with both summed.
- Memory estimate entries (`BuildIoMemoryEstimate`): one write buffer of 4 MiB and one hasher for the worker (`kWholeRun`,
  CPU). Snapshots are the step's rows; render output's are its own.
- Same bytes: the internal files (`ready_scene`, saved states, stored frame inputs) and every `FrameDone`
  snapshot digest of two runs of one fingerprint are equal byte for byte, because the snapshot is (owner D7,
  D18), the array order is fixed by the schema, and no internal file holds a clock time, a path or a process
  id. Open-format product files are compared by what they decode to (render output 5.7). Clock times are in
  `run_report.json`, `writer.lock` and events only.
- CPU against GPU: this module adds no tolerance. Two runs that differ in `RunOn` have different fingerprints
  and their frames are compared with the step's tolerances (step 6.4). The same holds between run modes (A15):
  one piece, `kBatched` and `kBatchedWithSolvesInPieces` differ in `run_mode` and are compared within the step's
  tolerances; each mode repeated gives the same bytes.

## 7. Tests

Contract tests first; there is no physics to test. All run in the container without a GPU unless marked.
Tests that need render output's writers (21) are in `cli/`'s test target, which links `fluidity_interchange`.

1. **Array file round trip.** Every element type, an empty array, an array of 2^20 + 1 items, a name of 255
   bytes. Read back equal, bit for bit. A file cut at every 64-byte boundary gives `kFileDamaged`; one flipped
   bit anywhere gives `kDigestDiffers` from `CheckListedFile`.
2. **Independent decoder** (the one reference piece, below) reads the stored frame inputs of test 3 and finds
   the same numbers.
3. **Asymmetric Z-up fixture** (report, modernization "Verify equivariance"): one particle at (1, 2, 3) m with
   velocity (-4, 5, -6) m/s, a box turned 30 degrees about X at (7, 8, 9), gravity along -Z. Scene file to
   stored frame inputs and to `read-particle`: the same components in the same order, exactly.
4. **Listing transaction with injected faults.** The file calls go through a private table
   (`io/file_calls.h`, production table `kFileCalls`; the pattern of step 5.12). For each call of steps 1 to 7
   of 5.3, a table that fails at that call, and one that ends the process there (a test-only child process):
   afterwards `ReadFileList` shows either the list before or the list with the new file, every listed file
   passes `CheckListedFile`, and `BuildRunFolderToContinue` leaves no unlisted file under this module's names.
   The same for a file written through `path_while_writing()` and for `DiscardOutputFile`.
5. **Torn end.** A file list cut at every byte of its last entry reads as the list without that entry.
6. **Writer lock.** Two processes: the second gets `kRunFolderInUse` after about one second and the folder's
   bytes do not change. After the first is killed the second succeeds. `report` while a writer runs:
   `is_held` true and the writer's process id; after it is killed: `is_held` false; a writer started during a
   probe is never refused.
7. **Defaults.** A scene file with only required fields: every optional field appears once in
   `filled_defaults`; the ready scene written, read back by `ReadRunScene` and written again gives the same
   bytes; a file with every default written out gives the same `scene_digest`.
8. **Fingerprint covers the physics.** Using the schema's field list (FlatBuffers reflection), for every
   field of the `physical` table: changing it changes `scene_digest`. For every field of `presentation`: it
   does not. A new field in neither list fails the test. Every field has a `scene_path` attribute.
9. **Continue equals uninterrupted.** A liquid scene with a source, a driven solid and a free box, surface and
   foam asked, 20 frames on the CPU: (a) straight; (b) 12 frames, `stop_after_frame`, resume; (c) `cancel`
   inside frame 13, resume; (d) process killed inside frame 13, resume. The file lists of all four name the
   same digests for every internal file, the same `snapshot_digest` in every `FrameDone`, and the same saved
   state of frame 20; each product file decodes (render output's reader in its tests) to the values of (a).
   In (c) and (d) frames 11 and 12 are reported `frame_checked`. The same with snow, with gas, with gas beside
   liquid, with cell resize, and batched.
10. **Cadence changes nothing.** Saved state every 1, 5, 10 frames; queue of 1 and 4; `keeps_frame_inputs`
    on and off: equal snapshot digests and internal files.
11. **Other run refused.** One viscosity changed: `kWrongRunFingerprint` with `kScene`. Another executable
    digest (injected): `kExecutable`. `BuildRunFromSavedState` with the same edit: a new folder, the old one
    untouched, `started_from` set.
12. **Queue and waiting.** A `FrameWriter` that sleeps: never more than `max_waiting_frames` waiting,
    `output_wait_elapsed_seconds` > 0, frames listed in ascending order, every frame present. A writer whose
    folder write fails at frame 5: frames 0 to 4 listed, the run ends with that failure, exit code 1. A writer
    that reports a product not built at frame 5: a `NotBuilt` entry, a `finding` event, frames 6 on listed.
13. **Stop at every safe point.** `cancel` set before each stage of a step (through the step's stage table):
    exit code 4, the list is as before the frame, the resume passes test 9. Standard input closed during
    frame 7: frame 7 and its saved state are listed, exit code 4, `stopping` with `end_of_input`.
14. **Events.** Every line a run prints parses as strict JSON against `events.fbs`; `sequence` has no gap;
    `sent_at` never decreases and is within one second of the test's own UTC clock; exactly one of `done`,
    `stopped`, `failed`, and it is last; a heartbeat at least every second during a pressure solve injected to
    take five seconds, with `last_advance_sent_at` unchanged meanwhile; `events.jsonl` equals standard output
    line for line; closing standard output stops the run after the frame.
15. **Exit codes and command line.** One case per row of `ToExitCode`. Unknown option, missing value, a path
    that starts with `--`, `--help`: exit 2 or 0, and no exception leaves `ReadCommandLine` (the test binary
    is built with exceptions off around it).
16. **`capabilities` touches nothing.** In a build with GPU code on a machine without a GPU it succeeds within
    200 ms, and with `--run-on gpu` it reports `no_gpu_found` (`ToName(RunOnError::kNoGpuFound)`, Foundation) as a finding and exits 0. `check --assume-memory 1`
    reports the estimate and a blocking `does_not_fit` with `use_bigger_cells` and exits 3 on any machine,
    which shows that no buffer was built; standard error ends with a line starting `fits: no. To fit
    (estimates):` (`CheckScene, AssumeMemoryOneByte_DoesNotFitExitThree`).
17. **Messages and codes.** For every enumerator of every module's error enum and of the step enums of
    `stable_names.h`, `io::ToMessage` and `io::ToName` return non-empty text, the names are lower snake case,
    and each (module, code) pair is unique. `ToName, EveryCapabilityReason_NamedAndUnique` counts
    `step::kCapabilityReasonCount` (16 at most; A24: no `viscous_liquid_with_snow`; A27: no
    `viscous_liquid_with_free_solid`); `ToName, EverySuggestedChange_NamedAndUnique` counts 12 (A27: no
    `set_viscosity_to_zero`); `ToName,
    EveryModule_NamedAndUnique` (14, reconcile); `ToName, RunModeAndRunOn` (three run modes, `ToName(step::RunMode)`
    giving `batched_with_solves_in_pieces`; A25: also the two `ViscousPath` names).
18. **Batched within tolerance of one piece** (A15; replaces "Batched equals one piece", which asked for equal
    internal files and snapshot digests). The scene of test 9 in one piece, `kBatched` and
    `kBatchedWithSolvesInPieces` (forced with `assume_memory_bytes`): every frame within the step design 6.4
    tolerances of the one-piece run; each batched mode run twice gives equal internal files and snapshot digests.
    `run_mode` stays in the fingerprint. Tests (reconcile; I/O plan Task 29): `AdvanceRunTo,
    BatchedWithinStepToleranceOfOnePiece` (was `BatchedEqualsOnePiece`: `liquid_source_solids.json` one piece,
    `kBatched` and `kBatchedWithSolvesInPieces`: the last saved states within the step design 6.4 tolerances of one
    piece; the test prints whether the bytes happen to be equal) and `AdvanceRunTo, EachModeRepeatRun_SameFiles`
    (each of the three modes run twice: equal snapshot digests and saved states, file for file). Also `BuildRun, SolvesInPiecesChosen_ModeCopied` (memory between the two
    batched layouts: `RunPlan::mode` `kBatchedWithSolvesInPieces`, `max_solve_loaded_blocks` equal to the batch
    plan's, budget `run_mode` the same) and `BuildSceneBudget, StateDoesNotFitCpu_VerdictAndSuggestions`
    (`assume_cpu_memory_bytes` below the batched CPU rows: `kDoesNotFitCpu`, `to_fit` filled; `BuildRun` on the same
    overrides: `kModuleFailed` (step, `kDoesNotFit`), no folder).
19. **All findings at once** (NEW). A scene file with a duplicate solid id, a negative source rate, a shared
    export name and a surface group naming a snow material: `check` lists four findings, each with its scene
    file path and ids, exit 3.
20. **`check --cache`** (NEW). Against a finished run: the same scene gives `same` everywhere; the surface
    spacing changed gives the simulation `same` and `surface_<id>` `changed` with `outputs.surface.spacing`,
    `needs_simulation` false; a viscosity changed gives the simulation `changed`, `kScene`,
    `materials[0].ingredients[0].viscosity`, every output `changed` with `needs_simulation`; another folder
    with no file list gives `absent`. The folder's bytes do not change.
21. **Outputs** (NEW, `cli/` tests). `OutputFile`: a reader of the folder during writing sees no file and no
    entry; after listing it sees both. Every declared output of a scene with two liquids, foam, gas beside
    liquid and solids has one listed file per frame under its pattern with the chosen extension; the
    declarations name exactly the channels render output's writer wrote. `run --outputs surface_3` with a new
    spacing: the old surface files removed, new ones listed with a new output fingerprint, every other file
    untouched, no step run. `--outputs foam_and_spray`: refused with `start_new_run`.
22. **`report` is read-only** (NEW). During a run, after a crash and after a clean end: the folder's bytes are
    unchanged by `report`; `last_listed_frame`, `saved_state_frame`, `last_try`, the lock holder and the outputs
    are those of the file list and the run report.
23. **`read-particle`** (NEW). The particle of test 3 at frames 0 and 1: position and velocity exact; an id
    that does not exist: `kParticleNotFound`, exit 1; a frame without stored inputs and without a saved state:
    `kFrameInputsNotKept`.
24. **Budget lines, exact** (NEW, planned cost; I/O plan Task 23a). Hand-built `RunBudget`s (RTX 5060 Ti, GPU 16.00
    GB total, 15.10 GB free, 11.20 GB allowed; CPU 64.00 GB, 56.00 GB, 44.80 GB; `memory_share_of_total` 0.7):
    `BuildBudgetLines, OnePieceOnGpu_ExactLines`, `SolvesInPiecesOnGpu_ExactLines` (12 batches, 5 solve pieces),
    `DoesNotFitCpu_ExactLinesWithWhatWouldFit` (52,000,000 particles: at most 38,190,163, cell 0.0111 m, 23.14 GB
    more cpu memory), each compared with the plan's lines character for character. `BuildRunBudget,
    DoesNotFitCpu_Suggestions`, `LayerDoesNotFit_CellAndGpuOnly` (no particle suggestion, width 0.02 m, `kGpu`),
    `FallingBlockOnCpu_FieldsEqualThePlan`, `ForcedBatched_FieldsEqualTheBatchPlan`; `WriteRunReport,
    BudgetFieldNamesInJson`. NEW (A25): `BuildBudgetLines, TwoLiquids_ViscousEffectLines` (water 1e-3 Pa s 1000 kg/m3
    and syrup 10 Pa s 1400 kg/m3, smallest cell 0.01 m, largest step 0.01 s; the four lines before the last `fits:`
    line exactly:
    ```text
    liquids: viscous solve (a liquid has viscosity above 0); effect per step at the smallest cell 0.0100 m and the largest step 0.0100 s
      water: viscosity 0.001 Pa s, density 1000 kg/m3, viscous effect per step 0.0001
      note: water's viscosity changes velocity differences between neighbouring cells by less than 0.1 % per step, which is not visible; set its viscosity to 0
      syrup: viscosity 10 Pa s, density 1400 kg/m3, viscous effect per step 0.714
    ```
    ), `BuildBudgetLines, InviscidScene_PressureSolveLine` (water preset, 1000 kg/m3: `liquids: pressure solve (every
    liquid has viscosity 0)` then `  water: viscosity 0 Pa s, density 1000 kg/m3, inviscid`), `RunBudget,
    LiquidsRoundTrip` (the A25 fields through `WriteRunReport` and `ReadRunReport`, `"viscous_path"` as a string).
    The three exact-line tests above have no liquid in their hand-built budgets, so their lines are unchanged.
25. **Measured cost and time** (NEW, planned cost; Task 29a). `ComputeTimeProjection, ThreeFrames_RateAndRemaining`,
    `TwoFrames_RateButNoProjection`, `UsesTheLastSixOnly`; `BuildCostLines, OnePieceGpu_ExactLinesAndWarnings`
    (warnings 15 % memory, 46 % copies), `WithinMargins_NoWarning`, `SolvesInPieces_SolveLine`; `AdvanceRunTo,
    TinyRun_PlannedAgainstMeasured` (CPU run, measured equals planned memory, no copies, stage seconds exactly the
    stages run), `ForcedBatched_MeasuredCopiesEqualTheReports` (sum over committed steps equals every try's
    `StepReport::copies` and `BatchReport` bytes and loads plus the resize copies); `WriteRunReport,
    CostFieldNamesInJson`.
26. **Budget and cost through the commands** (NEW, planned cost; Tasks 34 to 37, 39). `ReadCommandLine,
    AssumeCpuMemoryOnCheckOnly`; `EventWriter, BudgetAndCostTablesPrinted` (`budget.lines`, `cost.warnings`
    verbatim; a heartbeat before 3 frames has `projected_remaining_seconds` -1); `CheckScene,
    FallingBlockOnCpu_BudgetLinesOnStandardError` (standard error equals `budget.lines`, exit 0, standard output
    JSON only), `AssumedMemoryForcesBatches_BudgetSaysWhy` (`run_mode` `"batched"`), `AssumeCpuMemoryTooSmall_
    DoesNotFitCpuExitThree`; `AdvanceRun, BudgetAtStartCostAtEnd` (12 frames: exactly one `time: about` line,
    `log.txt` the same lines, `run_report.json` with `budget` and `cost`), `DoesNotFit_ExitThreeLinesPrinted`;
    process test `test_budget_and_cost_fields_by_schema_name` (every field by its `run_budget.fbs` name in
    `scene_checked`, `started`, `frame_done`, `done` and `run_report.json`; `check --assume-memory 1
    --assume-cpu-memory 1`: exit 3, `does_not_fit_cpu`, `to_fit` fields).

**Reference pieces.** One: `reference/io/read_array_file.py`, written from section 5.3 alone with `struct`
and the FlatBuffers Python package, no code shared with the writer. It removes the uncertainty a round-trip
test cannot: that the file is what the specification says, not merely what our reader expects (`main` found
a six-byte encoder difference this way). **Lean proofs:** none; nothing here is a theorem.

**Comparison with the proof of concept:** none by bytes (no old file is read). One behaviour check: the
uniform liquid scene of `native/tests/fixtures/liquid_cache/resolved_defaults.json`, written as a v2 scene
file, gives the particle count and the frame times `main` gives.

**GPU** (owner's run): test 9 (a) to (d) with `--run-on gpu`; test 18 on the GPU (within tolerance, A15):
`AdvanceRunTo, OnGpuBatchedWithinStepToleranceOfOnePiece` (both batched modes; was `OnGpuBatchedEqualsOnePiece`)
and `AdvanceRunTo, OnGpuEachModeRepeatRun_SameFiles` (each of the three modes twice, same files; reconcile, I/O plan
Task 41); the repeat-run test (two runs, equal internal files and snapshot digests); a saved state written by a CPU run is refused by a GPU resume
with `kRunOn`; and `BuildRunFromSavedState` from that CPU state runs and stays within the step's tolerances for
one frame.

## 8. Method paper section

Labels: `<sec-files>` (replaces the old `<sec-stages>`), `<sec-fingerprint>`, `<sec-protocol>`.

Outline: 1 one scene, two forms, and where defaults are filled; 2 the run folder, open products and internal
records (A14); 3 how a file gets listed and what a crash leaves; 4 what a saved state holds and why each value;
5 the fingerprints, the rule for continuing and the comparison with an old run; 6 the output queue and
waiting; 7 events, stop requests and exit codes.

Figures: (a) the life of a file as a time line: temporary, written (bytes or a library through a path),
flushed, checked, renamed, listed, with the four crash points and, under each, what a reader sees and what the
next writer does; (b) the run folder as a tree beside the file list as a strip of entries, with arrows from
entries to files and a torn last entry greyed out; (c) two tracks, simulation and output worker, over three
frames, with a full queue and the waiting interval shaded; (d) the run and output fingerprints as a tree of
their parts, with the parts left out drawn outside.

Pseudocode: `WriteListedFile` (the seven steps of 5.3) and `AdvanceRunTo` (the steps of 5.6), numbered as in
the code.

Sources: the old paper's reference list has no entry on files, hashing or protocols; its own sections
"Stages, manifests, and checkpoints" and "Protocol, settings and Blender delivery" are the text this section
rewrites. BLAKE3 and FlatBuffers are cited by name (decision 13). The random generator of the start particles
is the step's (its decision 9).

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| resolved scene, `LiquidResolvedScene3D`, `FLRS` | `ready_scene`, `step::ReadyScene`, `io::RunScene` (the owner of its vectors) | DECIDED; `RunScene` decided (decision 4) |
| text packet `FLUIDITY-BLENDER-n`, `scene.txt` | `scene_file` | DECIDED |
| checkpoint, `FLST`, `FLSI`, `.flgc` | `saved_state`, `SavedState` | DECIDED |
| point frame, point cache, `FLPT`, `blender_bridge` export | an open-format points output written by render output (`points_<material_id>`); no FluidiTy playback file | A14 |
| (motion samples, a frame's whole state for rebuilding) | `KeptFrameInputs`, `kFrameInputs`, `frame_inputs/` | NEW; render output's `FrameInputs` |
| frame manifest `FLMF`, stage manifest, JSON mirror, checkpoint reference `FLCR` | `file_list`, `FileList`, `ListedFile`, `FrameDone` | DECIDED word; types NEW |
| (the add-on's output list) | `OutputDeclaration`, `output_id`, `output_fingerprint` | NEW |
| temporary file, staged file | `OutputFile`, `path_while_writing` | NEW (changes-to-apply 12) |
| cache directory | `run_folder`, `RunFolder` | Decided (decision 4) |
| lease, job lease, `.writer.lock` | `WriterLock` (as a concept), held by `RunFolder` | Decided (decision 3): `lease` is a never-say word (4.8) |
| publish, published, promote | `listed`; `WriteListedFile`, `frame_done` | Decided (decision 2): `publish` is a never-say word under `commit` (4.7) |
| validate (command and checks) | `check`, `CheckScene`, `CheckSceneFile`, `CheckListedFile`, `CheckAfterWrite` | DECIDED |
| `read-cache` (add-on's wish) | `report`, `RunFolderFacts`, `ReadRunFolderFacts` | Decided (decision 1) |
| `--against` (first draft), `--cache` (add-on) | `check --cache <run_folder>`, `CacheComparison`, `CacheMatch` | changes-to-apply 12 |
| `resolved_fingerprint` | `run_fingerprint`, `RunFingerprint` | VERIFIED by reading `liquid_cache_3d.h:52-82, 234-239`: a hash of the scene, the build and the backend |
| `execution_id` | dropped | VERIFIED (`liquid_cache_3d.h:238`): a hash of the fingerprint alone; it adds nothing |
| `CudaExecutionIdentity`, `compatibility_record`, `device_digest` | `ProcessorDescription` (base), a part of the fingerprint | VERIFIED (`simulate_cuda_compatibility.h:17-34`): device, driver, runtime |
| `EffectsIdentity3D`, `StageIdentity` | `output_fingerprint` of an `OutputDeclaration` | VERIFIED (`effects_cache_3d.h:15-26`): stage, upstream run, settings, build |
| `CudaLiquidEpochIdentity3D` | `state_version` and time | VERIFIED (`cuda_liquid_snapshot_3d.h:15, 41`): which committed state a copy holds |
| "stable identity" of a particle or body | `id` | VERIFIED (`contact_face_bodies_3d.h:30-31`); DECIDED word |
| `kIdentityRoundoff`, `integrated_identity_error`, identity matrix | unchanged meaning, not this module's | VERIFIED (`mac_pressure_3d_system.cc:510-518`): an equation that must hold, or the unit matrix |
| content digest, `LiquidDigest3D` | `Digest`, `file digest`, `snapshot_digest` | Decided (decision 4) |
| `Clock { time, frame, substep, accepted_step, epoch }` | `time`, `frame`, `step_count`, `state_version` | DECIDED words |
| run manifest (report), job status marker | `RunReport`, `run_report.json`, `RunStatus` | Decided (decision 4): `Record` is not an allowed suffix (4.8) |
| backend, `cpu\|cuda`, device index | `run_on`, `--run-on cpu\|gpu` | DECIDED (`RunOn`); `processor` as in the step design. No device index: `ReadProcessorDescription` takes `RunOn` only |
| motion capture, motion interval, motion carrier | render output's `MotionSample`, `RecordedInterval` | A14; render output's words |
| heartbeat, protocol version | `heartbeat`, `protocol`, `sent_at`, `last_advance_sent_at` | NEW (kept from `main`, two fields added) |
| snapshot queue, pending, ready | `OutputQueue`, `waiting_frames`, `max_waiting_frames` | NEW; `max_` DECIDED |
| capacities | `step::SimulationLimits` | step design |
| `simulate`, `validate`, `inspect`, `devices`, `schema`, `step`, `replay` | `capabilities`, `check`, `run`, `resume`, `report`, `read-particle` | Decided (decision 1, changes-to-apply 12) |
| emission source rate in particles per second | `volume_rate` in the scene file | NEW |
| (planned memory and copies shown before a run; what a run used) | `RunBudget`, `budget`, `BudgetVerdict`, `ModeReason`; `RunCost`, `cost`; `TimeProjection` | NEW; A22 allows `budget` for a planned cost shown to the user |
| `Diagnostic { domain, field, object, cause }`, `E-CLI-InvalidArgument` | `IoFailure`, `FailingModule`, `Finding` with `module` and `code` (`ToName`), `ToMessage` | A2; stable codes per changes-to-apply |

## 10. Not ported

- `schemas/simulate/v1/`, `schemas/effects/v1/`, `schemas/simulate/incompatible/` and the `flatc --conform`
  tests of `cmake/LiquidCacheSchemas.cmake`: no v1 reader (D13). `v2/` becomes the baseline to conform to at
  the switch-over; until then fields may be renumbered.
- `native/cli/blender_resolve.cc` and the five text packets; `kMaximumSeedCount = 1024`.
- `native/src/solid_interaction_checkpoint_3d.cc` (`FLSI` 1 to 3) and the gas branch's `gas_execution.cc`,
  `gas_scene.fbs`, `gas_checkpoint.fbs`: every material is in the one scene, the one saved state, the one loop.
- One FlatBuffers table per particle, the 2^20 record limit, the fixed `kCapacities`.
- Canonical bytes across builds and the fixture that pins them (`resolved_defaults.hex`).
- Files named by digest (`blobs/<hex>`), the JSON mirror of every frame, the separate checkpoint reference
  files, `execution_id`, the `FLCUDA1` record file, the `.simulate-job-status` marker: the file list and the
  run report hold all of it.
- The commands `step`, `replay`, `schema`, `devices` and the hidden `cutter-frame`, `rigid-frame`,
  `cutter-inspect`, `rigid-inspect`, `blender-resolve`; the second argument parser; `--preview`, `--trace`,
  `--trace-geometry`, `--measure-job` (timings are always in `frame_done` and the run report; `try_done` is
  the one optional event); `--initial-state`.
- The test-only executable built with `FLUIDITY_SIMULATE_FAULT_FIXTURE`: faults go through the private file
  call table.
- Whole-state motion samples (`MotionSnapshot.state`) and this module's own grid-velocity motion file: render
  output records its own motion samples (its 5.2).
- A FluidiTy playback file of any kind (A14), and the Alembic and USD writers of `blender_bridge`: render output
  writes every open-format file.
- Compression, files for reverse-mode replay (report, CUDA 13.3 and 13.4), automatic restart after a failure,
  a run that goes on after the program that started it closed standard input: none is in the current method;
  each is a later decision.
- `LiquidTransaction.lean` and every other proof: none concerns files.

## 11. Decisions

Already decided before this list: Q1 and A1 (a file stores the velocity kept for the next step, under the array
name `velocity_for_next_step`), Q2 and A2 (`IoFailure`, `std::expected`), Q10, Q11 (`kNone` first;
`kRunOnCallFailed`), Q22 (grouped order is what a file stores), Q31 a (`ingredient_fractions`), Q33 (no 0.99);
step questions 7 (`AdvanceTo`, hence `AdvanceRunTo`), 8 (answered in 5.7: no ring), 9 (`ReadyScene` stays in
`step/`).

1. **CLI command names (your O5).**
   ```text
   fluidity capabilities [--run-on gpu]                         a) proposed
   fluidity check  scene.json [--run-on gpu] [--against run/]
   fluidity run    scene.json --into run/ [--run-on gpu] [--last-frame 120] [--start-from other_run/]
   fluidity resume run/ [--last-frame 240]
   fluidity report run/ [--check-files]

   fluidity inspect | check | run | resume | report             b) a verb for the first
   fluidity devices | validate | simulate [--resume] | inspect  c) as on main; validate is a never-say word
   ```
   Recommendation: a. `resume` takes the folder, not the scene file, so it cannot be given another scene;
   the add-on asks `check --against` whether the artist's scene is still the run's.
   Decision: a (recommended option, owner's standing instruction, 2026-11-14), extended as
   `changes-to-apply.md` 12 asks for the add-on: `--against` is `--cache`, `report` is the read-only query of a
   run folder, `read-particle` is added, and `run --outputs` rebuilds outputs (3.7, 5.6).
2. **The word for "a reader can now see this file".** The task and the report say publish; the naming review
   folds `publish` into `commit`, which means a step try replacing the state.
   ```cpp
   io::WriteListedFile(run_folder, file);   {"event":"frame_done"}       // a) listed: from the decided file_list
   io::PublishFile(run_folder, file);       {"event":"frame_published"}  // b) allow publish, for files only
   io::CommitFile(run_folder, file);        {"event":"frame_committed"}  // c) one word for both
   ```
   With c, "committed frame" means a file and "committed step" a state. Recommendation: a. The design is
   written with it; `Publish` and `published` appear nowhere in its names.
   Decision: a, `listed` (recommended option, owner's standing instruction, 2026-11-14).
3. **`WriterLock`.** `lease` is a never-say word (it was given to `BufferLoan`). Options: `WriterLock` (it is
   an operating-system file lock, and a programmer's word); `WriterLoan`; allow `lease` for this. No public
   type carries the name (`RunFolder` holds it); it appears in comments, the file name and `kRunFolderInUse`.
   Recommendation: `WriterLock`.
   Decision: `WriterLock` (recommended option, owner's standing instruction, 2026-11-14).
4. **What the fingerprint covers, and the name of the run report.** Proposed: scene physics, the exact
   executable, `RunOn`, the processor's model, driver and runtime, the run mode. A rebuilt executable
   therefore cannot continue an old run; it can start a new one from the old saved state (`--start-from`).
   Alternative: a hand-kept "numerical version" number instead of the executable digest, which lets a
   harmless rebuild continue and relies on someone raising the number. Recommendation: the digest. Also
   confirm `RunReport` for "the one run record" (`Record` is not an allowed suffix; `StepReport` and
   `MemoryReport` exist), `RunScene`, `RunFolder`, `Digest`.
   Decision: the executable digest; `RunReport`, `RunScene`, `RunFolder`, `Digest` (recommended option,
   owner's standing instruction, 2026-11-14). `run_mode` stays a part until test 18 passes. **Replaced** (the
   "until test 18 passes" part) by decision 15: `run_mode` stays a part.
5. **The three packages.** Proposed: keep all three (5.9). FlatBuffers describes every header, list entry,
   report and event and parses the scene file; bulk arrays are raw because of its 2 GB limit. Option: drop
   CLI11 for a table of options and a 150-line parser: no exception anywhere in the program, no special
   build flags for one file, but help text and error wording become ours to write. `main` already works
   around CLI11 in three places (`public_cli.cc:79-103, 148-166, 205-207`). Recommendation: keep CLI11 as
   the programme design lists it; drop it if the workarounds return.
   Decision: keep FlatBuffers, BLAKE3 and CLI11 (recommended option, owner's standing instruction,
   2026-11-14).
6. **The scene file is JSON text checked against the schema.** Options: a) JSON written by Python's standard
   library, parsed natively by FlatBuffers against `scene_file.fbs` (readable, diffable, no Python
   package); b) binary FlatBuffers written by the add-on (needs the FlatBuffers Python package and generated
   code in the add-on); c) a text packet as on `main`. Recommendation: a.
   Decision: a (recommended option, owner's standing instruction, 2026-11-14).
7. **Checks after writing.** Proposed: header and sizes for frames; a full read-back and digest comparison for
   saved states, since recovery depends on them (it doubles their read volume once every 10 frames).
   Alternative: read-back for everything, or for nothing. Also confirm the defaults: a saved state every 10
   frames, the newest 2 kept.
   Decision: as proposed (recommended option, owner's standing instruction, 2026-11-14): `kHeaderAndSizes`
   for stored frame inputs and the ready scene, `kReadBackDigest` for saved states, `kSizeOnly` for open-format
   files (their digest is still computed from the bytes on disk); every 10 frames, 2 kept.
8. **Two stop requests, and no saved state inside a frame.** `cancel` stops at the next safe point and loses
   the work since the last saved state; `stop_after_frame` finishes the frame and saves. Alternative: one
   request, and a saved state at any committed step (then a motion file of the cut frame cannot be completed
   on resume). Recommendation: two requests. And: a frame computed again after a resume that differs from the
   listed one is a failure (`kFrameDiffersOnResume`), not a warning as the old paper had it, because it
   breaks the exactness requirement; the bridge of question 4 remains.
   Decision: two requests, and a differing frame is a failure (recommended option, owner's standing
   instruction, 2026-11-14). The end of standard input is `stop_after_frame` (`changes-to-apply.md` 12, the
   add-on's decision 2). The comparison is by `snapshot_digest`, since open-format bytes are not reproducible.
9. **Stable error codes.** `code` in the `failed` event:
   ```text
   "pressure.NotConverged"   a) needs one ToName per module beside ToMessage (ten small additions)
   "pressure.7"              b) the enum's number; needs nothing, but numbers must then never be reused
   ```
   Recommendation: a.
   Decision: a (recommended option, owner's standing instruction, 2026-11-14), in the form
   `changes-to-apply.md` sets for every design: `ToName` gives the enumerator in lower snake case; the event
   carries it as `code` beside `module` (`"pressure"`, `"not_converged"`).
10. **Motion output words (your O5: `motion capture`, `interval`).** Proposed: `MotionFrame` (the file of one
    frame), `MotionSample` (the grid velocity at one committed time), `motion_samples_per_frame`; `carrier`
    is gone. Module 11 confirms the content.
    Decision: superseded by A14 and render output's design: this module has no motion file; render output's
    `MotionSample` and `MotionSettings::samples_per_frame` are the words, and its samples are stored in the
    frame inputs file (5.5).
11. **Doubles in playback files, no compression.** 56 bytes per particle and frame. Alternative: 32-bit
    positions and velocities in particle files (half the size; the surface built from them changes in the
    seventh digit). Recommendation: doubles until a measurement asks otherwise.
    Decision: A14 removes FluidiTy playback files; open-format files carry float32 as render output 5.7 sets.
    The recommended option stands for the internal stored frame inputs: doubles, no compression (owner's
    standing instruction, 2026-11-14).
12. **The preset table and the start sampling.** `kWater` (1000 kg/m3, 1e-3 Pa s, 0.072 N/m), `kSyrup` (the
    values of owner Q50), `kSnow`, `kSand`, `kSpringySolid`, `kAir` (the gas law defaults): the snow, sand and
    springy numbers are to be copied from the add-on's defaults on `main` (`blender/fluidity/__init__.py`)
    and reviewed by you. Start particles: 8 per smallest cell with a quarter-spacing jitter. Confirm both, or
    give the table.
    Decision: the table as proposed without `kSand` (A9: the sand module adds it), and 8 particles per
    smallest cell with a quarter-spacing jitter (recommended option, owner's standing instruction,
    2026-11-14). The snow and springy numbers are copied from `main` and stay open to the owner's review.
    Replaced in part (A25): water 0 Pa s (`kWater`: 1000 kg/m3, 0 Pa s, 0.072 N/m), so a water scene uses the
    pressure solve.
13. **Two bibliography entries** for the method paper: BLAKE3 and FlatBuffers are not in the old list. Supply
    them, or agree to name them without a reference.
    Decision: named without a reference until the owner supplies the entries (recommended option, owner's
    standing instruction, 2026-11-14).
14. **Both batched modes in a run** (A15). `RunMode` has three values; `BuildSceneBudget` copies
    `BatchPlan::solve_place == kInPieces` into `RunPlan::mode = kBatchedWithSolvesInPieces` and copies
    `max_solve_loaded_blocks`; `RunPlan::allowed_cpu_bytes` bounds the CPU rows; `ToName(step::RunMode)` has three
    names; `RunReport::batch_plan` and `BatchedRun` exist in either batched mode.
    Decision: as stated (owner answer A15; plan decision, 2026-11-14 (buffers in batched runs); I/O plan Task 29).
15. **Batched against one piece: tolerance, not bytes** (A15). Replaces the batches design's "batched and
    one-piece runs give the same bytes" as used in section 4, and test 18 "Batched equals one piece" (equal
    internal files and snapshot digests). Test 18 now asks agreement within the step 6.4 tolerances and the same
    bytes for each mode repeated. Since two modes need not give the same bytes, `run_mode` stays a part of the
    run fingerprint for good (the earlier "removed after test 18's GPU run" is withdrawn).
    Decision: as stated (owner answer A15).
16. **`RunParts` without cell resize and batched run.** Replaces the design's `RunParts { cell_resizer,
    batched_run, step_recorder, frame_writer, output, callbacks }`: both need the run plan, so `Run` builds and owns
    `CellResize` and `BatchedRun` after `step::BuildRunPlan`; `step_recorder` returns with render output's Task 47.
    Decision: `RunParts { FrameWriter frame_writer; OutputQueueSettings output; RunCallbacks callbacks; }` (plan
    decision, `plan-decisions.md` "I/O and CLI").
17. **Smaller plan decisions of the I/O plan.** `RunOverrides::assume_memory_bytes`; `FileCheck` instead of `bool
    check_files`; `FileKind::kStartParticles`; base failures as `FailingModule::kBase` with `module_error` = the
    `BufferError` value or 64 + the `RunOnError` value; schema versions start at 2; `io/material_presets.h` with
    `MaterialPreset`; `ToName(RunOn)`; the render-output seams of 3.6 (`FrameWriter::start(void* context)`,
    `BuildFrameWriterWithoutProducts`, `outputs.render: RenderSection` empty until render output's Task 47).
    Decision: as stated (plan decisions, `plan-decisions.md` "I/O and CLI" and "Surface, foam and spray").
18. **The budget before a run.** `io/run_budget.h` (3.9): `BuildRunBudget`, `BuildBudgetLines`, `BuildSceneBudget`
    (allocates no buffer; used by `check`, `run` and `BuildRun`); `BuildRunPlan` returns a plan that does not fit
    (`PlanFit::kDoesNotFit`) and `BuildRun` refuses it as `kModuleFailed` (step, `kDoesNotFit`) before any folder;
    the exact line templates, mode reasons, verdicts and suggestions of 5.10.
    Decision: as stated (plan decision, 2026-11-14 (planned cost); owner principle A17; A22 for the word; closes
    "Batched solves (A15)" open point 6 and owner question round 2 item 11).
19. **`--assume-cpu-memory`.** `check` gains `--assume-cpu-memory <bytes>` (`Command::assume_cpu_memory_bytes`,
    `RunOverrides::assume_cpu_memory_bytes`) beside `--assume-memory`; `run` refuses it; neither is in a
    fingerprint.
    Decision: as stated (plan decision, 2026-11-14 (planned cost)).
20. **The measured cost and the time.** `io/run_cost.h` (3.9): `RunCost` fed per try, per committed step and per
    frame; `BuildCostLines` with planned against measured, time by stage and by activity, and warnings above
    `kMemoryWarningShare = 1.10` (memory) and `kCopyWarningShare = 1.25` (copies per step, solve copies per
    iteration); no time estimate before the run; `ComputeTimeProjection` over the last up to 6 frames listed in
    this process, a projection from the 3rd frame, a `time: about ...` line after the 3rd frame and every 10th
    after it.
    Decision: as stated (plan decision, 2026-11-14 (planned cost); owner principle A17).
21. **Where the budget and cost appear.** Human lines on standard error only (and `log.txt` for `run` and
    `resume`); `scene_checked.budget`, `started.budget`, `heartbeat` and `frame_done` with
    `simulated_seconds_per_second` and `projected_remaining_seconds`, `done` / `stopped` / `failed` with `budget`
    and `cost`; `RunReport::budget` and `cost`; schema `run_budget.fbs` with unit-suffixed field names and
    lower-snake enum strings; exit 3 when the verdict is not `kFits`.
    Decision: as stated (plan decision, 2026-11-14 (planned cost)).
22. **Copies counted in batched runs.** This module adds no copy of its own; it sums the step's `StepReport::copies`
    and `ReadCopyCounts`, `BatchReport` (pass and solve), `ResizeCellsReport::copy_count` and `bytes_copied`, and
    render output's `ReadRecorderCopyCounts` (seam 11), and leaves out the bounded "small records".
    Decision: as stated (plan decision, 2026-11-14 (buffers in batched runs)).
23. **Stable names of `Module` and the third `RunMode`.** `ToName(step::Module)` gives the lower snake case of the
    enumerator without `k` (14 names, `cell_resize`, `render_output`), used by the budget's module names;
    `ToName(step::RunMode)` gives `one_piece`, `batched`, `batched_with_solves_in_pieces` (the schema's three
    `run_mode` strings).
    Decision: as stated (`plan-decisions.md`, Reconcile pass, coordinator item (3); I/O plan Task 5).
24. **Test 18 by name.** `AdvanceRunTo, BatchedEqualsOnePiece` becomes `AdvanceRunTo,
    BatchedWithinStepToleranceOfOnePiece` plus `AdvanceRunTo, EachModeRepeatRun_SameFiles`; on the GPU
    `OnGpuBatchedEqualsOnePiece` becomes `OnGpuBatchedWithinStepToleranceOfOnePiece` plus
    `OnGpuEachModeRepeatRun_SameFiles`; `run_mode` stays in the fingerprint (decision 15).
    Decision: as stated (`plan-decisions.md`, Reconcile pass, coordinator item (3); I/O plan Tasks 29 and 41).
25. **The budget is planned through `BuildSceneBudget`.** `check`, `run` and `BuildRun` get the plan and the budget
    from `BuildSceneBudget` (5.6 step 3, 5.8); `check` does not call `step::BuildRunPlan` itself.
    Decision: as stated (`plan-decisions.md`, Reconcile pass, coordinator item (3); I/O plan Task 36).
26. **What counts as viscous, shown in the budget.** A liquid is viscous exactly when its viscosity setting is above
    0, fixed for the whole run (the step chooses once); the water preset has viscosity 0. The budget carries
    `viscous_path`, `viscous_effect_cell_width_m`, `viscous_effect_step_seconds` and per liquid a
    `ViscousEffectOfLiquid` (3.9; named `LiquidViscousEffect` before decision 29), and `BuildBudgetLines` prints
    the `liquids:` block of 5.10 before the last `fits:` line, with a note advising viscosity 0 when a viscous liquid's effect per step is below
    `kSmallestVisibleViscousEffectPerStep` (2.5e-4, owner to confirm, viscosity's constant). `check` and the run's
    start print it as part of the budget lines; `ToName(step::ViscousPath)` gives `pressure_solve` and
    `viscous_solve`.
    Decision: as stated (owner answer A25; `plan-decisions.md`, "A25: what counts as viscous, and the viscous effect
    per step"; I/O plan Tasks 5, 14, 23a, 36, 37).
27. **The snow exclusion is lifted.** `CapabilityReason::kViscousLiquidWithSnow` and its name
    `viscous_liquid_with_snow` no longer exist, so `kCapabilityReasonCount` is 17 at most (14 if the viscosity plan
    lifts its three exclusions); `viscous_liquid_with_free_solid` stays.
    Decision: as stated (owner answer A24; `plan-decisions.md`, "A24: exclusions"; I/O plan Tasks 5, 36).
    Replaced (A27) by decision 28 for the count and the free-solid name.
28. **The free-solid exclusion is lifted, and `set_viscosity_to_zero` goes.** A free solid beside a viscous liquid
    is a velocity unknown of the viscous solve (A27), so `CapabilityReason::kViscousLiquidWithFreeSolid` and its
    name `viscous_liquid_with_free_solid` no longer exist: `kCapabilityReasonCount` is 16 at most (13 if the
    viscosity plan lifts its three exclusions). `SuggestedChange::kSetViscosityToZero` has no user in the main
    sequence: removed, `kSuggestedChangeCount` 12, no `set_viscosity_to_zero` in `stable_names.h` (the sand module
    adds it back with `kViscousLiquidWithSand`).
    Decision: as stated (owner answer A27; `plan-decisions.md`, "A viscous liquid sticks, and free solids in the
    viscous solve (A26, A27)", "Exclusions"; I/O plan Tasks 5, 36).
29. **One name for one struct.** I/O's budget entry per liquid was `io::LiquidViscousEffect`, the name of the step's
    `step::LiquidViscousEffect`, which has other fields (`ingredient_index`, `viscosity`, `density`,
    `viscous_effect_per_step`).
    Decision: I/O's struct and its schema table are `ViscousEffectOfLiquid` (3.9); the step's keeps its name; the
    JSON field names are unchanged (A27 pass; I/O plan Task 23a; add-on decision 19).

**Open**

1. `kMemoryWarningShare = 1.10` and `kCopyWarningShare = 1.25` are new constants (owner to confirm).
2. `BatchSettings::solve_piece_share = 0.25` and the batched direct-solve room `4 * max_cells` decide when
   `kBatchedWithSolvesInPieces` is chosen (owner to confirm; batches and pressure own them).
3. Planned step copies are upper bounds at the limits; the measured mean is expected well below them, so the copy
   warning fires only when a bound is wrong.
4. `run_mode` is computed from the plan, which depends on the memory free when the run starts; a resume on a day
   with less free memory may get another mode and then `kWrongRunFingerprint` with `kRunMode`. The plans do not
   say whether a resume reuses the recorded mode (with `--assume-memory`-like forcing) or refuses; the user can
   start a new run from the saved state (`--start-from`).
5. `kSmallestVisibleViscousEffectPerStep = 2.5e-4` (0.1 % per step), printed in the budget's note, is a new
   constant of the viscosity module (owner to confirm; `plan-decisions.md`, A25 owner question 1).
