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
// Parsed process inputs own their paths. Scene paths name the shared FLRS binary,
// not a second JSON settings format. Frame/substep numbers are unsigned physical
// ordinals, not counts; frame_start/end are inclusive. A missing backend inherits
// the resolved scene for execution (device-info defaults explicitly to CPU).
struct Command {
  CommandKind kind = CommandKind::kSchema;
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
// Executes run, one accepted-substep step, or read-only replay. Frame ordinals in
// replay name the physical clock: N/0 is completed frame N, N/S is accepted step
// S during N->N+1. Interior replay completes/verifies that final frame's schedule.
// Preview and step never publish. A new run publishes frame zero, prerequisite
// checkpoints and the inclusive selected range; resume verifies/reuses existing
// content without replacement. Run also owns a heartbeat and cooperative
// cancellation and holds a separate whole-job writer lease when publishing.
absl::Status Execute(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic);
// Dispatch checked inspection or execution. Execution validates the actual module
// digest, resolved backend and model-v1 24 fps schedule before cache mutation.
// All result publication occurs after successful verification; errors are returned
// with diagnostic context for the entry point to emit exactly once.
absl::Status Dispatch(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic);
}  // namespace fluidity::simulate
