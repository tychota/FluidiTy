#pragma once

#include <cstdio>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "fluidity/host/liquid_cache_3d.h"

namespace fluidity::simulate {
enum class CommandKind {
  kRun,
  kCheck,
  kSchema,
  kStep,
  kReplay,
  kDeviceInfo
};
enum class CommandStage {
  kSimulate,
  kMesh,
  kWhitewater,
  kVolumes,
};
// Parsed process inputs own their paths. Scene paths name the shared FLRS binary,
// not a second JSON settings format. Frame/substep numbers are unsigned physical
// ordinals, not counts; frame_start/end are inclusive. A missing backend inherits
// the resolved scene for execution (device-info defaults explicitly to CPU).
struct Command {
  CommandKind kind = CommandKind::kSchema;
  CommandStage stage = CommandStage::kSimulate;
  std::filesystem::path output_cache_directory;  // Derived output; cache_directory names its upstream stage.
  std::filesystem::path volume_settings_path;  // Canonical FLVC physical box for a Volumes run.
  double surface_spacing_m = 0.0125;
  double surface_sigma_m = 0.025;
  std::filesystem::path scene_path;
  std::filesystem::path initial_state_path;  // Optional exact raw FLST upstream, never a checkpoint envelope.
  std::filesystem::path cache_directory;
  std::optional<LiquidBackend3D> backend;
  int device_index = 0;
  std::optional<std::uint64_t> frame;
  std::optional<std::uint64_t> substep;
  std::uint64_t frame_start = 0;
  std::uint64_t frame_end = 1;
  bool frame_start_explicit = false;
  bool resume = false;
  bool trace = false;
  bool trace_geometry = false;  // Run-only; implies accepted and attempt tracing.
  bool measure_job = false;  // Terminal timing and resource receipt, independent of tracing.
  bool preview = false;
};
// Stable protocol fields identify the failing input independently of localized
// OS errors. object is an optional UTF-8 input value, never an unescaped JSON fragment.
struct Diagnostic {
  std::string domain = "CLI";
  std::string field;
  std::string object;
  std::string cause;
};
// Parse UTF-8 arguments excluding argv[0], with no filesystem or device access.
// Reject unknown, duplicate, misplaced, missing and out-of-range inputs with
// InvalidArgument. On failure no Command escapes; diagnostic identifies the input.
// Bounds are decimal uint64 ordinals and nonnegative int device indices.
absl::StatusOr<Command> ParseCommand(std::span<const std::string> arguments, Diagnostic* diagnostic);
// Quote UTF-8 text as one JSON string, escaping every control byte and delimiter.
std::string JsonString(std::string_view text);
// Hash actual running module bytes with BLAKE3, independent of argv and paths.
// Uses Windows module lookup, Linux /proc/self/exe, or macOS dyld lookup. Streams
// bounded chunks through a read-only handle; lookup/I/O failure returns Unavailable.
absl::StatusOr<LiquidDigest3D> ExecutableDigest();
// Borrowed streams must outlive the writer. Emit flushes one versioned JSON line;
// members must be validated JSON object members without enclosing braces. The
// mutex serializes complete records for the independent heartbeat producer.
// Error emits the same structured record to protocol stdout and diagnostic stderr.
// Failure to write returns Unavailable; callers must return a nonzero exit status.
class ProtocolWriter {
 public:
  explicit ProtocolWriter(std::FILE* output = stdout, std::FILE* diagnostics = stderr);
  absl::Status Emit(std::string_view event, std::string_view members);
  absl::Status Error(const absl::Status& status, const Diagnostic& diagnostic);

 private:
  std::FILE* output_;
  std::FILE* diagnostics_;
  std::mutex mutex_;
};
// Internal Blender bridge. Advances a coupled rigid box by one completed
// frame and optionally publishes a scene-bound joint checkpoint.
absl::Status ExecuteRigidBoxFrame(
    const Command& liquid,
    const std::filesystem::path& rigid_packet,
    const std::filesystem::path& previous,
    const std::filesystem::path& output,
    ProtocolWriter& writer,
    bool inspect = false
);
/// Executes a run, advances one accepted substep, or replays a requested clock.
/// @pre command is a run, step, or replay returned by ParseCommand. diagnostic is non-null.
///
/// Execution matches the running module digest to the resolved scene and checks any requested backend against the
/// resolved backend. Model-v1 execution uses a 24 fps schedule. CUDA execution using an existing cache also verifies
/// the selected device and exact driver/runtime against the recorded compatibility data.
///
/// Replay reads the newest eligible checkpoint at or before frame N. A `NotFound` from `LiquidCache3D::ReadCheckpoint`
/// allows reconstruction from an explicitly supplied and verified frame-zero initial state. Other checkpoint errors
/// propagate.
///
/// N/0 selects completed frame N. N/S with S > 0 selects accepted substep S during the advance from N to N+1.
/// Replay verifies intermediate committed point frames when present and requires the committed point frame for the
/// completed target (N for N/0, N+1 for N/S). It returns `NotFound` if substep S is absent after target verification.
///
/// Replay, step, and preview are read-only and operate without the whole-job writer lease. Run provides heartbeat
/// and cooperative cancellation. A publishing run holds the whole-job writer lease. A new publishing run writes frame
/// zero, prerequisite checkpoints, and the inclusive selected range. Resume verifies and reuses existing content.
///
/// Heartbeat, attempt, trace, and frame publication records can precede the terminal outcome. A publishing run can
/// leave committed frames in the cache even when it returns an error.
///
/// @return An execution or job finalization error with diagnostic context, or the status of writing the terminal
/// `result` record after both succeed. A failure to write that record is returned to the caller.
absl::Status Execute(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic);
/// Dispatches a command returned by ParseCommand to the check, schema, device-info, or execution handler.
/// @param diagnostic Receives diagnostic context for errors. A null pointer returns `InvalidArgument`.
/// @return The status of the selected handler, including any failure to write its terminal `result` record.
/// On failure, the entry point uses the returned status and diagnostic context to attempt one structured error report.
absl::Status Dispatch(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic);
}  // namespace fluidity::simulate
