#include <atomic>
#include <cstdio>
#include <utility>

#include "liquid_cache_files_3d_internal.h"

namespace fluidity::liquid_cache_files_internal {
#define STORE_TRY(expression) \
  do { \
    const auto store_status = (expression); \
    if (!store_status.ok()) { \
      return store_status; \
    } \
  } while (false)

namespace {
thread_local NativeFileControl3D native_file_control;
absl::Status CheckNativeFile(NativeFileStage3D stage, const fs::path& path) {
  return native_file_control.check == nullptr ? absl::OkStatus()
                                              : native_file_control.check(native_file_control.context, stage, path);
}
}  // namespace
NativeFileControl3D SetNativeFileControlForTesting(NativeFileControl3D control) {
  return std::exchange(native_file_control, control);
}
absl::Status Failure(absl::StatusCode code) {
  return absl::Status(code, {});
}
absl::Status IoFailure() {
#if defined(_WIN32)
  const auto error = GetLastError();
  if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) {
    return Failure(absl::StatusCode::kNotFound);
  }
  if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS) {
    return Failure(absl::StatusCode::kAlreadyExists);
  }
#else
  if (errno == ENOENT) {
    return Failure(absl::StatusCode::kNotFound);
  }
  if (errno == EEXIST) {
    return Failure(absl::StatusCode::kAlreadyExists);
  }
#endif
  return Failure(absl::StatusCode::kUnavailable);
}
absl::Status Check(LiquidCachePublishControl3D control, LiquidCachePublishStage3D stage) {
  return control.check == nullptr ? absl::OkStatus() : control.check(control.context, stage);
}

// POSIX needs a directory fsync after link/unlink. Windows promotions explicitly
// use MOVEFILE_WRITE_THROUGH after FlushFileBuffers on the complete temporary.
absl::Status FlushDirectory(const fs::path& directory) {
  STORE_TRY(CheckNativeFile(NativeFileStage3D::kBeforeDirectoryFlush, directory));
#if defined(_WIN32)
  static_cast<void>(directory);
#else
  const int handle = open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (handle < 0) {
    return IoFailure();
  }
  const int result = fsync(handle);
  const int saved_error = errno;
  close(handle);
  if (result != 0) {
    errno = saved_error;
    return IoFailure();
  }
#endif
  return absl::OkStatus();
}
// Creates one directory link at a time and persists its parent before creating
// descendants. Existing links get the same barrier: an earlier interrupted call
// may have created them without completing that barrier. Ancestors are trusted;
// canonicalizing only the parent used for fsync permits stable ancestor symlinks.
absl::Status CreateDirectoryTree(const fs::path& path) {
  std::error_code error;
  const auto absolute_path = fs::absolute(path, error);
  if (error) {
    return Failure(absl::StatusCode::kUnavailable);
  }
  auto directory = absolute_path.root_path();
  for (const auto& component : absolute_path.relative_path()) {
    const auto parent = directory;
    directory /= component;
    fs::create_directory(directory, error);
    if (error) {
      return Failure(absl::StatusCode::kUnavailable);
    }
    const auto status = fs::status(directory, error);
    if (error || !fs::is_directory(status)) {
      return Failure(absl::StatusCode::kInvalidArgument);
    }
    const auto resolved_parent = fs::canonical(parent, error);
    if (error) {
      return Failure(absl::StatusCode::kUnavailable);
    }
    STORE_TRY(FlushDirectory(resolved_parent));
  }
  return absl::OkStatus();
}
absl::Status Directory(const fs::path& path, bool create_if_missing) {
  std::error_code error;
  if (create_if_missing) {
    STORE_TRY(CreateDirectoryTree(path));
  }
  const auto status = fs::symlink_status(path, error);
  if (error) {
    return Failure(
        error == std::errc::no_such_file_or_directory ? absl::StatusCode::kNotFound : absl::StatusCode::kUnavailable
    );
  }
  if (!fs::is_directory(status) || fs::is_symlink(status)) {
    return Failure(absl::StatusCode::kInvalidArgument);
  }
#if defined(_WIN32)
  const auto attributes = GetFileAttributesW(path.c_str());
  if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
    return Failure(absl::StatusCode::kInvalidArgument);
  }
#endif
  return absl::OkStatus();
}
absl::Status Root(const fs::path& root, bool create) {
  if (root.empty() || root.native().find(fs::path::value_type{}) != fs::path::string_type::npos) {
    return Failure(absl::StatusCode::kInvalidArgument);
  }
  return Directory(root, create);
}
absl::Status Lease(const fs::path& root, NativeFile& lease, bool create_directories, bool require_existing_lease) {
  STORE_TRY(Root(root, create_directories));
  STORE_TRY(lease.Open(root / ".writer.lock", false, true, require_existing_lease));
  STORE_TRY(lease.Lock());
  for (const char* name : {"blobs", "frames", "checkpoints"}) {
    STORE_TRY(Directory(root / name, create_directories));
  }
  return absl::OkStatus();
}
absl::StatusOr<LiquidCacheBytes3D>
ReadBytes(const fs::path& path, LiquidCacheLimits3D limits, std::uint64_t expected_byte_length) {
  NativeFile file;
  STORE_TRY(file.Open(path));
  auto initial_byte_length = file.Size();
  if (!initial_byte_length.ok()) {
    return initial_byte_length.status();
  }
  if (expected_byte_length != 0 && *initial_byte_length != expected_byte_length) {
    return Failure(absl::StatusCode::kDataLoss);
  }
  if (*initial_byte_length > limits.maximum_bytes) {
    return Failure(absl::StatusCode::kResourceExhausted);
  }
  auto bytes = LiquidCacheBytes3D::Create(static_cast<std::size_t>(*initial_byte_length));
  if (!bytes.ok()) {
    return bytes.status();
  }
  STORE_TRY(bytes->Resize(static_cast<std::size_t>(*initial_byte_length)));
  STORE_TRY(file.Read(bytes->values()));
  auto final_byte_length = file.Size();
  if (!final_byte_length.ok()) {
    return final_byte_length.status();
  }
  if (*final_byte_length != *initial_byte_length) {
    return Failure(absl::StatusCode::kDataLoss);
  }
  return std::move(*bytes);
}
std::span<const std::uint8_t> Bytes(std::string_view text) {
  return {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()};
}
LiquidContentReference3D Reference(LiquidContentKind3D kind, std::span<const std::uint8_t> bytes) {
  return {kind, bytes.size(), LiquidContentDigest3D(bytes)};
}
absl::StatusOr<fs::path> ContentPath(const fs::path& root, const LiquidContentReference3D& reference) {
  auto name = LiquidContentName3D(reference);
  if (!name.ok()) {
    return name.status();
  }
  return root / name->data();
}
absl::StatusOr<LiquidCacheBytes3D> ReadContent(
    const fs::path& root,
    const LiquidContentReference3D& reference,
    LiquidContentKind3D expected_kind,
    LiquidCacheLimits3D limits
) {
  if (reference.kind != expected_kind || reference.byte_length == 0) {
    return Failure(absl::StatusCode::kDataLoss);
  }
  auto path = ContentPath(root, reference);
  if (!path.ok()) {
    return path.status();
  }
  auto bytes = ReadBytes(*path, limits, reference.byte_length);
  if (!bytes.ok()) {
    return bytes.status();
  }
  STORE_TRY(VerifyLiquidContent3D(bytes->values(), reference, expected_kind, limits));
  return std::move(*bytes);
}
std::string OrdinalName(std::uint64_t ordinal, std::string_view suffix) {
  auto digits = std::to_string(ordinal);
  return std::string(20 - digits.size(), '0') + digits + std::string(suffix);
}
fs::path FramePath(const fs::path& root, std::uint64_t frame_ordinal, std::string_view suffix) {
  return root / "frames" / OrdinalName(frame_ordinal, suffix);
}
absl::Status Matches(
    const fs::path& path,
    std::span<const std::uint8_t> expected_bytes,
    LiquidCacheLimits3D limits,
    absl::StatusCode mismatch_status_code
) {
  auto existing = ReadBytes(path, limits);
  if (!existing.ok()) {
    return existing.status();
  }
  if (!std::ranges::equal(existing->values(), expected_bytes)) {
    return Failure(mismatch_status_code);
  }
  return absl::OkStatus();
}
absl::Status Promote(const fs::path& temporary, const fs::path& destination) {
  STORE_TRY(CheckNativeFile(NativeFileStage3D::kBeforePromotion, destination));
#if defined(_WIN32)
  if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH)) {
    return IoFailure();
  }
  STORE_TRY(CheckNativeFile(NativeFileStage3D::kAfterPromotion, destination));
#else
  // link is an atomic no-replace publication on the same filesystem. rename()
  // would overwrite an existing winner on POSIX.
  if (link(temporary.c_str(), destination.c_str()) != 0) {
    return IoFailure();
  }
  STORE_TRY(CheckNativeFile(NativeFileStage3D::kAfterPromotion, destination));
  STORE_TRY(FlushDirectory(destination.parent_path()));
  if (unlink(temporary.c_str()) != 0) {
    return IoFailure();
  }
#endif
  return FlushDirectory(destination.parent_path());
}
namespace {
absl::Status ReplaceSidecar(const fs::path& temporary, const fs::path& destination) {
#if defined(_WIN32)
  if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return IoFailure();
  }
#else
  if (rename(temporary.c_str(), destination.c_str()) != 0) {
    return IoFailure();
  }
#endif
  return FlushDirectory(destination.parent_path());
}
absl::Status WriteAndPromoteBytes(
    const fs::path& destination,
    std::span<const std::uint8_t> bytes,
    LiquidCacheLimits3D limits,
    LiquidCachePublishControl3D control,
    absl::Status (*promote)(const fs::path&, const fs::path&)
) {
  static std::atomic<std::uint64_t> temporary_file_sequence{0};
#if defined(_WIN32)
  const auto process_id = GetCurrentProcessId();
#else
  const auto process_id = getpid();
#endif
  // Exclusive creation and a per-process sequence tolerate abandoned temporaries
  // and PID reuse. A temporary has no semantic role and is never read on restart.
  for (int attempt = 0; attempt < 128; ++attempt) {
    const auto temporary = destination.parent_path()
        / (".tmp-" + std::to_string(process_id) + "-" + std::to_string(temporary_file_sequence.fetch_add(1)));
    struct RemoveTemporary {
      fs::path path;
      ~RemoveTemporary() {
        std::error_code ignored;
        fs::remove(path, ignored);
      }
    };
    NativeFile file;
    auto open_status = file.Open(temporary, true);
    if (open_status.code() == absl::StatusCode::kAlreadyExists) {
      continue;
    }
    STORE_TRY(open_status);
    RemoveTemporary cleanup{temporary};
    STORE_TRY(file.Write(bytes));
    STORE_TRY(Check(control, LiquidCachePublishStage3D::kTemporaryWritten));
    STORE_TRY(file.Flush());
    STORE_TRY(Check(control, LiquidCachePublishStage3D::kTemporaryFlushed));
    STORE_TRY(Matches(temporary, bytes, limits, absl::StatusCode::kDataLoss));
    return promote(temporary, destination);
  }
  return Failure(absl::StatusCode::kResourceExhausted);
}
}  // namespace
absl::Status PublishBytes(
    const fs::path& destination,
    std::span<const std::uint8_t> bytes,
    LiquidCacheLimits3D limits,
    LiquidCachePublishControl3D control,
    absl::StatusCode mismatch_status_code
) {
  auto existing_match_status = Matches(destination, bytes, limits, mismatch_status_code);
  if (existing_match_status.ok()) {
    // A verified existing link may come from a publish interrupted before fsync.
    return FlushDirectory(destination.parent_path());
  }
  if (existing_match_status.code() != absl::StatusCode::kNotFound) {
    return existing_match_status;
  }
  const auto promotion_status = WriteAndPromoteBytes(destination, bytes, limits, control, Promote);
  if (promotion_status.code() == absl::StatusCode::kAlreadyExists) {
    STORE_TRY(Matches(destination, bytes, limits, mismatch_status_code));
    return FlushDirectory(destination.parent_path());
  }
  return promotion_status;
}
absl::Status ReplaceOperationalBytes(const fs::path& destination, std::span<const std::uint8_t> bytes) {
  return WriteAndPromoteBytes(destination, bytes, {}, {}, ReplaceSidecar);
}
absl::Status PublishContent(
    const fs::path& root,
    const LiquidContentReference3D& reference,
    std::span<const std::uint8_t> bytes,
    LiquidCacheLimits3D limits,
    LiquidCachePublishControl3D control
) {
  STORE_TRY(VerifyLiquidContent3D(bytes, reference, reference.kind, limits));
  auto path = ContentPath(root, reference);
  if (!path.ok()) {
    return path.status();
  }
  STORE_TRY(PublishBytes(*path, bytes, limits, control));
  return Check(control, LiquidCachePublishStage3D::kContentPublished);
}
#undef STORE_TRY
}  // namespace fluidity::liquid_cache_files_internal
