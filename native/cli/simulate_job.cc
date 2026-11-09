#include "simulate_job.h"

#include <algorithm>
#include <array>
#include <span>
#include <string>

#if defined(_WIN32)
#include <io.h>
#else
#include <poll.h>
#endif

namespace fluidity::simulate {
namespace {
namespace files = liquid_cache_files_internal;
constexpr std::string_view kJobLeaseName = ".simulate-job.lock";
constexpr std::string_view kJobStatusName = ".simulate-job-status";
constexpr std::chrono::milliseconds kInputPollInterval{25};
constexpr std::chrono::milliseconds kHeartbeatInterval{500};

std::string ClockJson(const LiquidClock3D& clock) {
  return "{\"frame\":" + std::to_string(clock.frame_ordinal) + ",\"substep\":" + std::to_string(clock.substep_ordinal)
      + ",\"epoch\":" + std::to_string(clock.epoch)
      + ",\"accepted_step\":" + std::to_string(clock.accepted_step_ordinal) + '}';
}

// One bounded poll cannot starve heartbeat output or keep the worker alive
// after the solver stops. The CLI owns the only read end consumer. On Windows,
// PeekNamedPipe checks redirected stdin without blocking on a command or EOF.
class CancelInput {
 public:
  bool Poll() {
    std::array<char, 256> bytes{};
    const auto bytes_read = ReadReadyBytes(bytes);
    for (const char byte : std::span(bytes).first(bytes_read)) {
      if (byte == '\n') {
        const std::string_view command(command_.data(), command_length_);
        if (!discard_line_ && (command == "cancel" || command == "cancel\r")) {
          return true;
        }
        command_length_ = 0;
        discard_line_ = false;
      } else if (command_length_ < command_.size()) {
        command_[command_length_++] = byte;
      } else {
        discard_line_ = true;
      }
    }
    return false;
  }

 private:
  std::size_t ReadReadyBytes(std::span<char> bytes) {
    if (closed_) {
      return 0;
    }
#if defined(_WIN32)
    const auto input = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stdin)));
    const DWORD input_type = GetFileType(input);
    DWORD ready = static_cast<DWORD>(bytes.size());
    if (input_type == FILE_TYPE_PIPE) {
      if (!PeekNamedPipe(input, nullptr, 0, nullptr, &ready, nullptr)) {
        closed_ = true;
        return 0;
      }
      if (ready == 0) {
        return 0;
      }
    } else if (input_type != FILE_TYPE_DISK) {
      // Protocol commands arrive over redirected stdin. Console character I/O
      // can wait for line editing even after input events become available.
      closed_ = true;
      return 0;
    }
    DWORD transferred = 0;
    const DWORD requested = std::min(ready, static_cast<DWORD>(bytes.size()));
    if (!ReadFile(input, bytes.data(), requested, &transferred, nullptr) || transferred == 0) {
      closed_ = true;
      return 0;
    }
#else
    pollfd input{.fd = STDIN_FILENO, .events = POLLIN, .revents = 0};
    const int readiness = poll(&input, 1, 0);
    if (readiness == 0 || (readiness < 0 && errno == EINTR)) {
      return 0;
    }
    if (readiness < 0 || (input.revents & (POLLIN | POLLHUP)) == 0) {
      closed_ = true;
      return 0;
    }
    const auto transferred = read(STDIN_FILENO, bytes.data(), bytes.size());
    if (transferred < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      return 0;
    }
    if (transferred <= 0) {
      closed_ = true;
      return 0;
    }
#endif
    return static_cast<std::size_t>(transferred);
  }

  std::array<char, 7> command_{};
  std::size_t command_length_ = 0;
  bool discard_line_ = false;
  bool closed_ = false;
};

// Call only if no competing writer owns the lease. A leftover baking marker
// then describes a process that exited without publishing a terminal status.
absl::StatusOr<std::string_view> ReadJobStatusMarker(const std::filesystem::path& cache_directory) {
  auto bytes = files::ReadBytes(cache_directory / kJobStatusName, {.maximum_bytes = 32});
  if (absl::IsNotFound(bytes.status())) {
    return std::string_view("idle");
  }
  if (absl::IsDataLoss(bytes.status()) || absl::IsResourceExhausted(bytes.status())) {
    return std::string_view("unknown");
  }
  if (!bytes.ok()) {
    return bytes.status();
  }
  if (bytes->size() == 0) {
    return std::string_view("unknown");
  }
  const std::string_view marker(reinterpret_cast<const char*>(bytes->values().data()), bytes->size());
  if (marker == "v1 baking\n") {
    return std::string_view("crashed");
  }
  if (marker == "v1 complete\n") {
    return std::string_view("complete");
  }
  if (marker == "v1 cancelled\n") {
    return std::string_view("cancelled");
  }
  if (marker == "v1 failed\n") {
    return std::string_view("failed");
  }
  return std::string_view("unknown");
}
}  // namespace

absl::StatusOr<std::string_view> InspectJobStatus(const std::filesystem::path& cache_directory) {
  files::NativeFile probe;
  const auto opened = probe.Open(cache_directory / kJobLeaseName, false, true, true);
  if (absl::IsNotFound(opened)) {
    return ReadJobStatusMarker(cache_directory);
  }
  if (!opened.ok()) {
    return opened;
  }
  const auto locked = probe.Lock(files::NativeFileLockMode::kShared);
  if (absl::IsAborted(locked)) {
    return std::string_view("baking");
  }
  if (!locked.ok()) {
    return locked;
  }
  return ReadJobStatusMarker(cache_directory);
}

SimulateJob::SimulateJob(ProtocolWriter& writer, bool run_command) : writer_(writer) {
  if (run_command) {
    heartbeat_ = std::thread(&SimulateJob::Watch, this);
  }
}

SimulateJob::~SimulateJob() {
  StopHeartbeat();
}

absl::Status SimulateJob::AttachCache(const std::filesystem::path& cache_directory, bool new_run) {
  std::error_code path_error;
  cache_directory_ = std::filesystem::absolute(cache_directory, path_error);
  if (path_error) {
    return absl::UnavailableError("");
  }
  const auto opened = job_lease_.Open(cache_directory_ / kJobLeaseName, false, true);
  if (!opened.ok()) {
    return opened;
  }
  const auto acquired = job_lease_.Lock();
  if (!acquired.ok()) {
    return acquired;
  }
  if (new_run) {
    auto previous_status = ReadJobStatusMarker(cache_directory_);
    if (!previous_status.ok()) {
      return previous_status.status();
    }
    if (*previous_status != "idle") {
      return absl::AlreadyExistsError("");
    }
  }
  return absl::OkStatus();
}

absl::Status SimulateJob::MarkBaking() {
  if (cache_directory_.empty()) {
    return absl::OkStatus();
  }
  const auto marked = files::ReplaceOperationalBytes(cache_directory_ / kJobStatusName, files::Bytes("v1 baking\n"));
  baking_marker_written_ = marked.ok();
  return marked;
}

absl::Status SimulateJob::CheckBoundary() const {
  return absl::Status(boundary_status_.load(), "");
}

void SimulateJob::SetSettledClock(LiquidBackend3D backend, LiquidClock3D clock, bool verified_checkpoint) {
  const std::lock_guard lock(progress_mutex_);
  backend_ = backend;
  last_settled_clock_ = clock;
  if (verified_checkpoint) {
    last_published_frame_ = clock.frame_ordinal;
  }
}

absl::Status SimulateJob::ReportPublishedFrame(std::uint64_t frame_ordinal, std::string_view measurement_members) {
  const std::lock_guard lock(progress_mutex_);
  last_published_frame_ = std::max(last_published_frame_.value_or(0), frame_ordinal);
  return writer_.Emit(
      "frame",
      "\"published\":true,\"frame\":" + std::to_string(frame_ordinal) + std::string(measurement_members)
  );
}

absl::Status SimulateJob::Finish(const absl::Status& execution_status, Diagnostic* diagnostic) {
  StopHeartbeat();
  auto final_status = execution_status;
  if (final_status.ok() && boundary_status_.load() == absl::StatusCode::kUnavailable) {
    final_status = absl::UnavailableError("");
  }
  if (absl::IsCancelled(final_status)) {
    diagnostic->field = "stdin";
    diagnostic->cause = "cancellation requested at a safe solver boundary";
  } else if (execution_status.ok() && !final_status.ok()) {
    diagnostic->field = "stdout";
    diagnostic->cause = "heartbeat could not be written";
  }
  if (baking_marker_written_) {
    const std::string_view terminal = final_status.ok() ? "v1 complete\n"
        : absl::IsCancelled(final_status)               ? "v1 cancelled\n"
                                                        : "v1 failed\n";
    const auto marked = files::ReplaceOperationalBytes(cache_directory_ / kJobStatusName, files::Bytes(terminal));
    if (final_status.ok() && !marked.ok()) {
      diagnostic->field = "job-status";
      diagnostic->cause = "terminal job status could not be persisted";
      final_status = marked;
    }
  }
  return final_status;
}

void SimulateJob::Watch() {
  using Clock = std::chrono::steady_clock;
  CancelInput input;
  auto next_heartbeat = started_at_ + kHeartbeatInterval;
  std::unique_lock lock(progress_mutex_);
  while (!stopping_) {
    lock.unlock();
    const bool cancel_requested = boundary_status_.load() == absl::StatusCode::kOk && input.Poll();
    if (cancel_requested) {
      boundary_status_.store(absl::StatusCode::kCancelled);
    }
    lock.lock();
    if (stopping_) {
      break;
    }
    const auto now = Clock::now();
    if (now >= next_heartbeat) {
      if (!EmitHeartbeat(now).ok()) {
        boundary_status_.store(absl::StatusCode::kUnavailable);
        break;
      }
      next_heartbeat = now + kHeartbeatInterval;
    }
    wake_.wait_until(lock, std::min(next_heartbeat, now + kInputPollInterval), [this] {
      return stopping_;
    });
  }
}

void SimulateJob::StopHeartbeat() {
  if (heartbeat_.joinable()) {
    {
      const std::lock_guard lock(progress_mutex_);
      stopping_ = true;
    }
    wake_.notify_one();
    heartbeat_.join();
  }
}

absl::Status SimulateJob::EmitHeartbeat(std::chrono::steady_clock::time_point now) {
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - started_at_).count();
  const std::string backend = !backend_ ? "null" : JsonString(*backend_ == LiquidBackend3D::kCuda ? "cuda" : "cpu");
  return writer_.Emit(
      "heartbeat",
      "\"elapsed_milliseconds\":" + std::to_string(elapsed_ms) + ",\"backend\":" + backend
          + ",\"last_settled_clock\":" + (last_settled_clock_ ? ClockJson(*last_settled_clock_) : "null")
          + ",\"last_published_frame\":" + (last_published_frame_ ? std::to_string(*last_published_frame_) : "null")
  );
}
}  // namespace fluidity::simulate
