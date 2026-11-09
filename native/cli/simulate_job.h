#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

#include "../src/liquid_cache_files_3d_internal.h"
#include "fluidity/host/simulate_cli.h"

namespace fluidity::simulate {

// Checks the operational sidecar in a verified cache directory without creating
// files. A held OS lease is baking; a baking marker without its lease is crashed.
// A missing marker and unheld lease mean idle. Invalid metadata reports unknown
// independently of whether a committed frame can be read.
absl::StatusOr<std::string_view> InspectJobStatus(const std::filesystem::path& cache_directory);

// Owns an optional run heartbeat thread and an independent whole-job writer
// lease. The solver thread supplies value-only progress at safe boundaries; the
// worker never borrows live solver state, holds a CUDA lease, or mutates the cache.
class SimulateJob {
 public:
  SimulateJob(ProtocolWriter& writer, bool run_command);
  ~SimulateJob();

  // Attaches to a verified cache, holding the lease until destruction. If a
  // second writer owns the lease, return Aborted without touching the sidecar.
  // A new run also rejects a cache claimed by a competing run since initial lookup.
  absl::Status AttachCache(const std::filesystem::path& cache_directory, bool new_run);

  // Marks a validated writer job as baking. This operational record has no role
  // in frame identity or checkpoint replay. Preview jobs have no writer lease.
  absl::Status MarkBaking();

  // Returns Cancelled for an exact stdin cancel line and Unavailable for lost
  // heartbeat output. Call outside physical attempts and before publication.
  absl::Status CheckBoundary() const;

  // Replaces the heartbeat's settled clock only after a complete frame schedule.
  // verified_checkpoint indicates that this clock also names a verified manifest.
  void SetSettledClock(LiquidBackend3D backend, LiquidClock3D clock, bool verified_checkpoint = false);

  // Advances last known publication and emits the frame event together, so a
  // subsequent heartbeat cannot show less progress than this notification.
  absl::Status ReportPublishedFrame(std::uint64_t frame_ordinal, std::string_view measurement_members);

  // Stops the heartbeat before final stdout, and records a terminal operational
  // state before releasing the writer lease. Any execution error keeps precedence.
  absl::Status Finish(const absl::Status& execution_status, Diagnostic* diagnostic);

 private:
  void Watch();
  void StopHeartbeat();
  // The caller holds progress_mutex_ across the coherent snapshot and emit.
  absl::Status EmitHeartbeat(std::chrono::steady_clock::time_point now);

  ProtocolWriter& writer_;
  const std::chrono::steady_clock::time_point started_at_ = std::chrono::steady_clock::now();
  std::filesystem::path cache_directory_;
  liquid_cache_files_internal::NativeFile job_lease_;
  bool baking_marker_written_ = false;

  std::mutex progress_mutex_;
  std::condition_variable wake_;
  bool stopping_ = false;
  std::optional<LiquidBackend3D> backend_;
  std::optional<LiquidClock3D> last_settled_clock_;
  std::optional<std::uint64_t> last_published_frame_;
  std::atomic<absl::StatusCode> boundary_status_{absl::StatusCode::kOk};
  std::thread heartbeat_;
};
}  // namespace fluidity::simulate
