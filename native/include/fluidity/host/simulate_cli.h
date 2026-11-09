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
  std::filesystem::path cache_directory;
  std::optional<LiquidBackend3D> backend;
  int device_index = 0;
  std::optional<std::uint64_t> frame;
  std::optional<std::uint64_t> substep;
  std::uint64_t frame_start = 0;
  std::uint64_t frame_end = 1;
  bool resume = false;
  bool trace = false;
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
// mutex serializes complete records for the later independent heartbeat producer.
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
// Read-only 9A dispatch. check verifies portable self-consistency only, never
// authorizes resume; optional --scene additionally compares canonical identity.
// run/step/replay return Unimplemented until the execution unit supplies them.
// All result publication occurs after successful verification; errors are returned
// with diagnostic context for the entry point to emit exactly once.
absl::Status Dispatch(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic);
}  // namespace fluidity::simulate
