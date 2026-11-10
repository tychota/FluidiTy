#pragma once

#include <algorithm>
#include <filesystem>
#include <string>

#include "fluidity/host/liquid_cache_3d.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace fluidity::liquid_cache_files_internal {
namespace fs = std::filesystem;

enum class NativeFileStage3D {
  kBeforePromotion,
  kAfterPromotion,
  kBeforeDirectoryFlush,
};
// Thread-local fault/ordering seam for native filesystem regression tests. The
// callback runs before the named operation (after link/move for AfterPromotion),
// borrows path/context only for the call, and must not throw. Production is empty.
struct NativeFileControl3D {
  absl::Status (*check)(void*, NativeFileStage3D, const std::filesystem::path&) = nullptr;
  void* context = nullptr;
};
NativeFileControl3D SetNativeFileControlForTesting(NativeFileControl3D control);

absl::Status Failure(absl::StatusCode code);
absl::Status IoFailure();
absl::Status Check(LiquidCachePublishControl3D control, LiquidCachePublishStage3D stage);
enum class NativeFileLockMode {
  kExclusive,
  kShared
};

// Native handles never follow a final-component symlink/reparse point. Cache
// directory components are separately checked while the writer lease is held.
class NativeFile {
 public:
  NativeFile() = default;
  NativeFile(const NativeFile&) = delete;
  NativeFile& operator=(const NativeFile&) = delete;
  ~NativeFile() {
#if defined(_WIN32)
    if (handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(handle_);
    }
#else
    if (handle_ >= 0) {
      close(handle_);
    }
#endif
  }
  absl::Status Open(const fs::path& path, bool write = false, bool lease = false, bool existing_lease = false) {
#if defined(_WIN32)
    handle_ = CreateFileW(
        path.c_str(),
        write || (lease && !existing_lease) ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        lease ? (existing_lease ? OPEN_EXISTING : OPEN_ALWAYS) : (write ? CREATE_NEW : OPEN_EXISTING),
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
        nullptr
    );
    if (handle_ == INVALID_HANDLE_VALUE) {
      return IoFailure();
    }
    BY_HANDLE_FILE_INFORMATION information{};
    if (!GetFileInformationByHandle(handle_, &information)) {
      return IoFailure();
    }
    if ((information.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)) != 0) {
      return Failure(absl::StatusCode::kDataLoss);
    }
#else
    // Nonblocking open lets fstat reject a FIFO/device before any read can
    // wait for a peer. It does not change regular-file I/O semantics.
    int flags = O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK;
    flags |= lease ? (existing_lease ? O_RDONLY : O_RDWR | O_CREAT) : (write ? O_RDWR | O_CREAT | O_EXCL : O_RDONLY);
    handle_ = open(path.c_str(), flags, 0600);
    if (handle_ < 0) {
      if (errno == ELOOP) {
        return Failure(absl::StatusCode::kDataLoss);
      }
      return IoFailure();
    }
    struct stat information{};
    if (fstat(handle_, &information) != 0) {
      return IoFailure();
    }
    if (!S_ISREG(information.st_mode)) {
      return Failure(absl::StatusCode::kDataLoss);
    }
#endif
    return absl::OkStatus();
  }
  // A shared probe can coexist with other inspectors. An exclusive owner
  // conflicts with both modes. Contention returns Aborted without waiting.
  absl::Status Lock(NativeFileLockMode mode = NativeFileLockMode::kExclusive) {
#if defined(_WIN32)
    const DWORD flags =
        LOCKFILE_FAIL_IMMEDIATELY | (mode == NativeFileLockMode::kExclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0);
    OVERLAPPED position{};
    if (!LockFileEx(handle_, flags, 0, 1, 0, &position)) {
      if (GetLastError() == ERROR_LOCK_VIOLATION) {
        return Failure(absl::StatusCode::kAborted);
      }
      return IoFailure();
    }
#else
    const int flags = (mode == NativeFileLockMode::kExclusive ? LOCK_EX : LOCK_SH) | LOCK_NB;
    if (flock(handle_, flags) != 0) {
      if (errno == EWOULDBLOCK || errno == EAGAIN) {
        return Failure(absl::StatusCode::kAborted);
      }
      return IoFailure();
    }
#endif
    return absl::OkStatus();
  }
  absl::StatusOr<std::uint64_t> Size() const {
#if defined(_WIN32)
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(handle_, &size) || size.QuadPart < 0) {
      return IoFailure();
    }
    return static_cast<std::uint64_t>(size.QuadPart);
#else
    struct stat information{};
    if (fstat(handle_, &information) != 0 || information.st_size < 0) {
      return IoFailure();
    }
    return static_cast<std::uint64_t>(information.st_size);
#endif
  }
  absl::Status Read(std::span<std::uint8_t> bytes) {
    while (!bytes.empty()) {
      const auto count = std::min<std::size_t>(bytes.size(), 1 << 20);
#if defined(_WIN32)
      DWORD transferred = 0;
      if (!ReadFile(handle_, bytes.data(), static_cast<DWORD>(count), &transferred, nullptr)) {
        return IoFailure();
      }
#else
      const auto transferred = read(handle_, bytes.data(), count);
      if (transferred < 0) {
        if (errno == EINTR) {
          continue;
        }
        return IoFailure();
      }
#endif
      if (transferred == 0) {
        return Failure(absl::StatusCode::kDataLoss);
      }
      bytes = bytes.subspan(static_cast<std::size_t>(transferred));
    }
    return absl::OkStatus();
  }
  absl::Status Write(std::span<const std::uint8_t> bytes) {
    while (!bytes.empty()) {
      const auto count = std::min<std::size_t>(bytes.size(), 1 << 20);
#if defined(_WIN32)
      DWORD transferred = 0;
      if (!WriteFile(handle_, bytes.data(), static_cast<DWORD>(count), &transferred, nullptr)) {
        return IoFailure();
      }
#else
      const auto transferred = write(handle_, bytes.data(), count);
      if (transferred < 0) {
        if (errno == EINTR) {
          continue;
        }
        return IoFailure();
      }
#endif
      if (transferred == 0) {
        return Failure(absl::StatusCode::kUnavailable);
      }
      bytes = bytes.subspan(static_cast<std::size_t>(transferred));
    }
    return absl::OkStatus();
  }
  absl::Status Flush() {
#if defined(_WIN32)
    if (!FlushFileBuffers(handle_)) {
      return IoFailure();
    }
#else
    if (fsync(handle_) != 0) {
      return IoFailure();
    }
#endif
    return absl::OkStatus();
  }

 private:
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int handle_ = -1;
#endif
};

// Checked directory validation, native lease, and bounded byte I/O. Paths are
// derived by the store or supplied as trusted root_directory. The caller holds
// the lease across these operations; a failed acquisition returns Aborted.
absl::Status FlushDirectory(const std::filesystem::path& directory);
absl::Status
Lease(const std::filesystem::path& root, NativeFile& lease, bool create = false, bool existing_lease = false);
// Reads an owning copy of the file, bounded by limits.maximum_bytes.
// A zero expected_byte_length skips the initial length check. Returns
// DataLoss for a length mismatch or a length change during the read.
absl::StatusOr<LiquidCacheBytes3D>
ReadBytes(const std::filesystem::path& path, LiquidCacheLimits3D limits, std::uint64_t expected_byte_length = 0);
std::span<const std::uint8_t> Bytes(std::string_view text);
LiquidContentReference3D Reference(LiquidContentKind3D kind, std::span<const std::uint8_t> bytes);
absl::StatusOr<std::filesystem::path>
ContentPath(const std::filesystem::path& root, const LiquidContentReference3D& reference);
absl::StatusOr<LiquidCacheBytes3D> ReadContent(
    const std::filesystem::path& root,
    const LiquidContentReference3D& reference,
    LiquidContentKind3D expected_kind,
    LiquidCacheLimits3D limits
);
std::string OrdinalName(std::uint64_t ordinal, std::string_view suffix);
std::filesystem::path
FramePath(const std::filesystem::path& root, std::uint64_t frame_ordinal, std::string_view suffix = ".flmf");
absl::Status Matches(
    const std::filesystem::path& path,
    std::span<const std::uint8_t> expected,
    LiquidCacheLimits3D limits,
    absl::StatusCode conflict
);
// PublishBytes verifies existing bytes or exclusively writes, flushes, verifies
// and atomically promotes a same-directory temporary without replacing anything.
// A promotion may commit before a subsequent directory-flush error is returned.
// Retry verifies committed bytes and flushes their directory before success;
// abandoned temporaries are never read.
absl::Status PublishBytes(
    const std::filesystem::path& destination,
    std::span<const std::uint8_t> bytes,
    LiquidCacheLimits3D limits,
    LiquidCachePublishControl3D control,
    absl::StatusCode conflict = absl::StatusCode::kDataLoss
);
// Flushes and atomically replaces a mutable operational sidecar in the same
// directory. The caller holds its writer lease. Never use for immutable content.
absl::Status ReplaceOperationalBytes(const std::filesystem::path& destination, std::span<const std::uint8_t> bytes);
absl::Status PublishContent(
    const std::filesystem::path& root,
    const LiquidContentReference3D& reference,
    std::span<const std::uint8_t> bytes,
    LiquidCacheLimits3D limits,
    LiquidCachePublishControl3D control
);
}  // namespace fluidity::liquid_cache_files_internal
