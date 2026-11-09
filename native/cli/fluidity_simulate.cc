#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

#include "../src/liquid_cache_files_3d_internal.h"
#include "blake3.h"
#include "checkpoint_reference_generated.h"
#include "flatbuffers/minireflect.h"
#include "fluidity/host/cuda_probe.h"
#include "fluidity/host/simulate_cli.h"
#include "frame_manifest_generated.h"
#include "liquid_state_generated.h"
#include "point_frame_generated.h"
#include "resolved_scene_generated.h"
#include "simulate_schema_text.h"
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace fluidity::simulate {
namespace {
namespace files = liquid_cache_files_internal;
namespace wire = cache::v1;

absl::Status Fail(
    absl::StatusCode code,
    Diagnostic* diagnostic,
    std::string_view field,
    std::string_view object,
    std::string_view cause
) {
  diagnostic->field = field;
  diagnostic->object = object;
  diagnostic->cause = cause;
  return absl::Status(code, "");
}
// UTF-8 validity keeps protocol text portable; reject overlong, surrogate and
// out-of-range encodings rather than silently changing filesystem input bytes.
bool ValidUtf8(std::string_view text) {
  for (std::size_t i = 0; i < text.size();) {
    const auto first = static_cast<unsigned char>(text[i++]);
    if (first == 0) {
      return false;
    }
    if (first < 0x80) {
      continue;
    }
    int following = 0;
    std::uint32_t point = 0;
    std::uint32_t minimum = 0;
    if (first >= 0xc2 && first <= 0xdf) {
      following = 1;
      point = first & 31;
      minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
      following = 2;
      point = first & 15;
      minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
      following = 3;
      point = first & 7;
      minimum = 0x10000;
    } else {
      return false;
    }
    for (int byte = 0; byte < following; ++byte) {
      if (i == text.size()) {
        return false;
      }
      const auto next = static_cast<unsigned char>(text[i++]);
      if ((next & 0xc0) != 0x80) {
        return false;
      }
      point = (point << 6) | (next & 63);
    }
    if (point < minimum || point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff)) {
      return false;
    }
  }
  return true;
}
std::filesystem::path PathFromUtf8(std::string_view text) {
  return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(text.data()), text.size()));
}
bool ParseUnsignedDecimal(std::string_view text, std::uint64_t* value) {
  if (text.empty()) {
    return false;
  }
  const auto result = std::from_chars(text.data(), text.data() + text.size(), *value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}
std::string DigestHex(const LiquidDigest3D& digest) {
  return LiquidDigestHex3D(digest).data();
}
std::string Boolean(bool value) {
  return value ? "true" : "false";
}
std::string JsonDouble(double value) {
  std::array<char, 64> buffer{};
  const auto result = std::to_chars(
      buffer.data(),
      buffer.data() + buffer.size(),
      value,
      std::chars_format::general,
      std::numeric_limits<double>::max_digits10
  );
  return {buffer.data(), result.ptr};
}
absl::Status WriteLine(std::FILE* stream, const std::string& line) {
  if (std::fwrite(line.data(), 1, line.size(), stream) != line.size() || std::fflush(stream) != 0) {
    return absl::UnavailableError("");
  }
  return absl::OkStatus();
}
std::string Record(std::string_view event, std::string_view members) {
  return "{\"protocol_version\":1,\"event\":" + JsonString(event) + (members.empty() ? "" : "," + std::string(members))
      + "}\n";
}
// Generated mini-reflection provides the field names and types of the exact
// ForceDefaults runtime bytes. Override the vendor's fixed-decimal formatting:
// it rounds the mass quantum to 1e-12, which changes its numerical meaning.
class SceneJsonVisitor final : public flatbuffers::ToStringVisitor {
 public:
  SceneJsonVisitor() : ToStringVisitor("", true, "", false) {
  }
  void Double(double value) override {
    s += std::isinf(value) ? JsonString(value > 0 ? "Infinity" : "-Infinity") : JsonDouble(value);
  }
};
absl::StatusOr<std::string> SceneJson(LiquidResolvedSceneView3D scene) {
  auto bytes = EncodeLiquidResolvedScene3D(scene);
  if (!bytes.ok()) {
    return bytes.status();
  }
  SceneJsonVisitor visitor;
  flatbuffers::IterateFlatBuffer(bytes->values().data(), wire::ResolvedSceneTypeTable(), &visitor);
  return visitor.s;
}
absl::Status CheckScene(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic) {
  diagnostic->domain = "CACHE";
  diagnostic->field = "cache";
  diagnostic->cause = "portable cache verification failed";
  std::optional<LiquidDigest3D> explicit_fingerprint;
  if (!command.scene_path.empty()) {
    diagnostic->field = "scene";
    diagnostic->cause = "resolved FLRS scene verification failed";
    auto bytes = files::ReadBytes(command.scene_path, {});
    if (!bytes.ok()) {
      return bytes.status();
    }
    auto scene = DecodeLiquidResolvedScene3D(bytes->values());
    if (!scene.ok()) {
      return scene.status();
    }
    auto fingerprint = ResolvedSceneFingerprint(scene->View());
    if (!fingerprint.ok()) {
      return fingerprint.status();
    }
    explicit_fingerprint = *fingerprint;
  }
  std::string members = "\"command\":\"check\",\"inspection_only\":true,\"resume_authorized\":false";
  if (!command.cache_directory.empty()) {
    diagnostic->field = "cache";
    diagnostic->cause = "portable cache verification failed";
    auto cache = LiquidCache3D::OpenForInspection(command.cache_directory);
    if (!cache.ok()) {
      return cache.status();
    }
    auto scene = cache->ReadResolvedScene();
    if (!scene.ok()) {
      return scene.status();
    }
    auto fingerprint = ResolvedSceneFingerprint(scene->View());
    if (!fingerprint.ok()) {
      return fingerprint.status();
    }
    if (explicit_fingerprint && *explicit_fingerprint != *fingerprint) {
      return Fail(absl::StatusCode::kFailedPrecondition, diagnostic, "scene", "", "resolved scene differs from cache");
    }
    members += ",\"resolved_fingerprint\":" + JsonString(DigestHex(*fingerprint));
    if (command.frame) {
      diagnostic->field = "frame";
      diagnostic->cause = "committed frame verification failed";
      auto frame = cache->ReadCommittedFrame(*command.frame);
      if (!frame.ok()) {
        return frame.status();
      }
      auto name = LiquidContentName3D(frame->manifest.points);
      if (!name.ok()) {
        return name.status();
      }
      const auto& clock = frame->manifest.clock;
      members += ",\"frame\":" + std::to_string(clock.frame_ordinal) + ",\"substep\":"
          + std::to_string(clock.substep_ordinal) + ",\"epoch\":" + std::to_string(clock.epoch) + ",\"accepted_step\":"
          + std::to_string(clock.accepted_step_ordinal) + ",\"time_seconds\":" + JsonDouble(clock.time)
          + ",\"execution_id\":" + JsonString(DigestHex(frame->manifest.execution_id)) + ",\"points_relative_name\":"
          + JsonString(name->data()) + ",\"point_count\":" + std::to_string(frame->points.points.size());
    }
  } else {
    members += ",\"resolved_fingerprint\":" + JsonString(DigestHex(*explicit_fingerprint));
  }
  return writer.Emit("result", members);
}
absl::Status DeviceInfo(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic) {
  diagnostic->domain = "DEVICE";
  diagnostic->field = "backend";
  diagnostic->cause = "requested CUDA backend is unavailable";
#if defined(FLUIDITY_SIMULATE_CUDA)
  constexpr bool cuda_compiled = true;
#else
  constexpr bool cuda_compiled = false;
#endif
  const bool cuda_requested = command.backend == LiquidBackend3D::kCuda;
  std::string members = "\"command\":\"device-info\",\"cpu_available\":true,\"cuda_compiled\":" + Boolean(cuda_compiled)
      + ",\"backend\":" + JsonString(cuda_requested ? "cuda" : "cpu");
  if (!cuda_requested) {
    return writer.Emit("result", members + ",\"cuda_available\":null");
  }
#if defined(FLUIDITY_SIMULATE_CUDA)
  ProbeReport report;
  const auto status = RunCudaProbe(command.device_index, &report);
  if (!status.ok()) {
    diagnostic->object = std::to_string(command.device_index);
    diagnostic->cause = std::string(status.message());
    if (status.code() == absl::StatusCode::kInvalidArgument) {
      diagnostic->field = "device";
    }
    return status;
  }
  members += ",\"cuda_available\":true,\"device_index\":" + std::to_string(command.device_index) + ",\"device_name\":"
      + JsonString(report.device_name) + ",\"compute_major\":" + std::to_string(report.compute_major)
      + ",\"compute_minor\":" + std::to_string(report.compute_minor) + ",\"driver_version\":"
      + std::to_string(report.driver_version) + ",\"runtime_version\":" + std::to_string(report.runtime_version)
      + ",\"compiled_architecture\":" + std::to_string(report.compiled_architecture) + ",\"launch_result\":"
      + std::to_string(report.launch_result) + ",\"copy_result\":" + std::to_string(report.copy_result)
      + ",\"synchronize_result\":" + std::to_string(report.synchronize_result)
      + ",\"sample_count\":" + std::to_string(report.sample_count);
  return writer.Emit("result", members);
#else
  return absl::UnavailableError("");
#endif
}
}  // namespace

std::string JsonString(std::string_view text) {
  constexpr char hex[] = "0123456789abcdef";
  std::string result = "\"";
  for (const unsigned char byte : text) {
    if (byte == '"' || byte == '\\') {
      result += '\\';
      result += static_cast<char>(byte);
    } else if (byte < 0x20) {
      result += "\\u00";
      result += hex[byte >> 4];
      result += hex[byte & 15];
    } else {
      result += static_cast<char>(byte);
    }
  }
  return result + '"';
}

absl::StatusOr<Command> ParseCommand(std::span<const std::string> arguments, Diagnostic* diagnostic) {
  if (diagnostic == nullptr) {
    return absl::InvalidArgumentError("");
  }
  *diagnostic = {};
  auto invalid = [&](std::string_view field, std::string_view object, std::string_view cause) {
    return Fail(absl::StatusCode::kInvalidArgument, diagnostic, field, object, cause);
  };
  if (arguments.empty()) {
    return invalid("command", "", "a command is required");
  }
  for (const auto& argument : arguments) {
    if (!ValidUtf8(argument)) {
      return invalid("arguments", "", "arguments must be valid UTF-8 without NUL");
    }
  }
  Command command;
  const auto& name = arguments[0];
  if (name == "run") {
    command.kind = CommandKind::kRun;
  } else if (name == "check") {
    command.kind = CommandKind::kCheck;
  } else if (name == "schema") {
    command.kind = CommandKind::kSchema;
  } else if (name == "step") {
    command.kind = CommandKind::kStep;
  } else if (name == "replay") {
    command.kind = CommandKind::kReplay;
  } else if (name == "device-info") {
    command.kind = CommandKind::kDeviceInfo;
  } else {
    return invalid("command", name, "unknown command");
  }
  const bool run = command.kind == CommandKind::kRun;
  const bool check = command.kind == CommandKind::kCheck;
  const bool step = command.kind == CommandKind::kStep;
  const bool replay = command.kind == CommandKind::kReplay;
  const bool device = command.kind == CommandKind::kDeviceInfo;
  const bool execution = run || step || replay;
  std::set<std::string_view> seen;
  for (std::size_t i = 1; i < arguments.size(); ++i) {
    const std::string_view flag = arguments[i];
    const bool allowed = (flag == "--stage" && run) || (flag == "--scene" && (execution || check))
        || (flag == "--initial-state" && execution) || (flag == "--cache" && (execution || check))
        || (flag == "--backend" && (execution || device)) || (flag == "--device" && (execution || device))
        || (flag == "--frame" && (replay || check)) || (flag == "--substep" && replay)
        || ((flag == "--frame-start" || flag == "--frame-end" || flag == "--resume" || flag == "--preview") && run)
        || (flag == "--trace" && execution);
    if (!allowed) {
      return invalid("arguments", flag, "unknown or misplaced option");
    }
    if (!seen.insert(flag).second) {
      return invalid(flag, "", "duplicate option");
    }
    if (flag == "--resume") {
      command.resume = true;
      continue;
    }
    if (flag == "--preview") {
      command.preview = true;
      continue;
    }
    if (flag == "--trace") {
      command.trace = true;
      continue;
    }
    if (++i == arguments.size() || arguments[i].empty() || arguments[i].starts_with("--")) {
      return invalid(flag, "", "option requires a value");
    }
    const auto& value = arguments[i];
    if (flag == "--stage") {
      if (value != "simulate") {
        return invalid(flag, value, "only the simulate stage is supported");
      }
    } else if (flag == "--scene") {
      command.scene_path = PathFromUtf8(value);
    } else if (flag == "--initial-state") {
      command.initial_state_path = PathFromUtf8(value);
    } else if (flag == "--cache") {
      command.cache_directory = PathFromUtf8(value);
    } else if (flag == "--backend") {
      if (value == "cpu") {
        command.backend = LiquidBackend3D::kCpu;
      } else if (value == "cuda") {
        command.backend = LiquidBackend3D::kCuda;
      } else {
        return invalid(flag, value, "backend must be cpu or cuda");
      }
    } else {
      std::uint64_t ordinal = 0;
      if (!ParseUnsignedDecimal(value, &ordinal)) {
        return invalid(flag, value, "expected an unsigned decimal integer");
      }
      if (flag == "--device") {
        if (ordinal > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
          return invalid(flag, value, "device index exceeds the supported range");
        }
        command.device_index = static_cast<int>(ordinal);
      } else if (flag == "--frame") {
        command.frame = ordinal;
      } else if (flag == "--substep") {
        command.substep = ordinal;
      } else if (flag == "--frame-start") {
        command.frame_start = ordinal;
        command.frame_start_explicit = true;
      } else {
        command.frame_end = ordinal;
      }
    }
  }
  if (run && !seen.contains("--stage")) {
    return invalid("stage", "", "run requires --stage simulate");
  }
  if ((run || step) && command.scene_path.empty()) {
    return invalid("scene", "", "resolved scene is required");
  }
  if (execution && command.cache_directory.empty()) {
    return invalid("cache", "", "cache directory is required");
  }
  if (check && command.cache_directory.empty() && command.scene_path.empty()) {
    return invalid("cache", "", "check requires --cache or --scene");
  }
  if (check && command.frame && command.cache_directory.empty()) {
    return invalid("frame", "", "frame requires a cache");
  }
  if (replay && (!command.frame || !command.substep)) {
    return invalid("frame/substep", "", "replay requires both ordinals");
  }
  if (command.frame_start > command.frame_end) {
    return invalid("frame-end", "", "frame range is reversed");
  }
  if (seen.contains("--device") && command.backend != LiquidBackend3D::kCuda) {
    return invalid("device", "", "device index requires explicit CUDA backend");
  }
  if (device && !command.backend) {
    command.backend = LiquidBackend3D::kCpu;
  }
  return command;
}

absl::StatusOr<LiquidDigest3D> ExecutableDigest() {
  std::filesystem::path executable;
#if defined(_WIN32)
  std::array<wchar_t, 32768> path{};
  const auto module_path_character_count = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (module_path_character_count == 0 || module_path_character_count >= path.size()) {
    return absl::UnavailableError("");
  }
  executable = std::wstring_view(path.data(), module_path_character_count);
#elif defined(__linux__)
  // Opening this kernel alias reads the running inode even after a rename/unlink.
  executable = "/proc/self/exe";
#elif defined(__APPLE__)
  std::uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::vector<char> path(size);
  if (_NSGetExecutablePath(path.data(), &size) != 0) {
    return absl::UnavailableError("");
  }
  std::error_code error;
  executable = std::filesystem::canonical(path.data(), error);
  if (error) {
    return absl::UnavailableError("");
  }
#else
#error Unsupported executable identity platform
#endif
  // NativeFile rejects symlinks by design. Linux /proc/self/exe is the trusted
  // kernel module alias, so use a read-only stream solely for this trusted path.
#if defined(_WIN32)
  std::FILE* input = nullptr;
  if (_wfopen_s(&input, executable.c_str(), L"rb") != 0) {
    return absl::UnavailableError("");
  }
#else
  std::FILE* input = std::fopen(executable.c_str(), "rb");
#endif
  if (input == nullptr) {
    return absl::UnavailableError("");
  }
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  std::array<std::uint8_t, 64 * 1024> bytes{};
  std::size_t executable_bytes_read = 0;
  while ((executable_bytes_read = std::fread(bytes.data(), 1, bytes.size(), input)) != 0) {
    blake3_hasher_update(&hasher, bytes.data(), executable_bytes_read);
  }
  const bool failed = std::ferror(input) != 0;
  const int close_result = std::fclose(input);
  if (failed || close_result != 0) {
    return absl::UnavailableError("");
  }
  LiquidDigest3D digest{};
  blake3_hasher_finalize(&hasher, digest.data(), digest.size());
  return digest;
}
ProtocolWriter::ProtocolWriter(std::FILE* output, std::FILE* diagnostics) : output_(output), diagnostics_(diagnostics) {
}
absl::Status ProtocolWriter::Emit(std::string_view event, std::string_view members) {
  const std::lock_guard lock(mutex_);
  return WriteLine(output_, Record(event, members));
}
absl::Status ProtocolWriter::Error(const absl::Status& status, const Diagnostic& diagnostic) {
  const std::string name = absl::StatusCodeToString(status.code());
  const auto record = Record(
      "error",
      "\"code\":" + JsonString("E-" + diagnostic.domain + "-" + name) + ",\"domain\":" + JsonString(diagnostic.domain)
          + ",\"name\":" + JsonString(name) + ",\"field\":" + JsonString(diagnostic.field)
          + ",\"object\":" + JsonString(diagnostic.object) + ",\"cause\":" + JsonString(diagnostic.cause)
  );
  const std::lock_guard lock(mutex_);
  const auto output_status = WriteLine(output_, record);
  const auto diagnostic_status = WriteLine(diagnostics_, record);
  return output_status.ok() ? diagnostic_status : output_status;
}
absl::Status Dispatch(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic) {
  if (diagnostic == nullptr) {
    return absl::InvalidArgumentError("");
  }
  if (command.kind == CommandKind::kCheck) {
    return CheckScene(command, writer, diagnostic);
  }
  if (command.kind == CommandKind::kDeviceInfo) {
    return DeviceInfo(command, writer, diagnostic);
  }
  if (command.kind != CommandKind::kSchema) {
    return Execute(command, writer, diagnostic);
  }
  diagnostic->field = "schema";
  diagnostic->cause = "shared schema or executable identity could not be read";
  auto digest = ExecutableDigest();
  if (!digest.ok()) {
    return digest.status();
  }
  LiquidResolvedSceneView3D defaults;
  defaults.build_digest = *digest;
  auto json = SceneJson(defaults);
  if (!json.ok()) {
    return json.status();
  }
  return writer.Emit(
      "result",
      "\"command\":\"schema\",\"execution_available\":true,\"build_digest\":" + JsonString(DigestHex(*digest))
          + ",\"identifiers\":{\"state\":" + JsonString(wire::LiquidStateIdentifier()) + ",\"points\":"
          + JsonString(wire::PointFrameIdentifier()) + ",\"manifest\":" + JsonString(wire::FrameManifestIdentifier())
          + ",\"resolved_scene\":" + JsonString(wire::ResolvedSceneIdentifier())
          + ",\"checkpoint_reference\":" + JsonString(wire::CheckpointReferenceIdentifier())
          + "},\"resolved_schema\":" + JsonString(kResolvedSchemaText) + ",\"defaults\":" + *json
  );
}
}  // namespace fluidity::simulate

namespace {
int Main(std::vector<std::string> arguments) {
  fluidity::simulate::ProtocolWriter writer;
  fluidity::simulate::Diagnostic diagnostic;
  auto command = fluidity::simulate::ParseCommand(arguments, &diagnostic);
  const auto status = command.ok() ? fluidity::simulate::Dispatch(*command, writer, &diagnostic) : command.status();
  if (!status.ok()) {
    const auto output_status = writer.Error(status, diagnostic);
    return output_status.ok() && status.code() == absl::StatusCode::kInvalidArgument ? 2 : 1;
  }
  return 0;
}
}  // namespace
#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
  std::vector<std::string> arguments;
  for (int index = 1; index < argc; ++index) {
    const int utf8_byte_count =
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[index], -1, nullptr, 0, nullptr, nullptr);
    if (utf8_byte_count <= 0) {
      return Main({std::string(1, static_cast<char>(0xff))});
    }
    std::string text(static_cast<std::size_t>(utf8_byte_count), '\0');
    if (WideCharToMultiByte(
            CP_UTF8,
            WC_ERR_INVALID_CHARS,
            argv[index],
            -1,
            text.data(),
            utf8_byte_count,
            nullptr,
            nullptr
        )
        != utf8_byte_count) {
      return Main({std::string(1, static_cast<char>(0xff))});
    }
    text.pop_back();
    arguments.push_back(std::move(text));
  }
  return Main(std::move(arguments));
}
#else
int main(int argc, char** argv) {
  std::vector<std::string> arguments;
  for (int index = 1; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }
  return Main(std::move(arguments));
}
#endif
