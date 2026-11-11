#pragma once

#include <vector>

#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/motion_interval_3d.h"

namespace fluidity {

struct MotionFrame3D {
  LiquidClock3D first_clock;
  LiquidClock3D last_clock;
  std::vector<LiquidState3D> samples{};  // Owning accepted samples in increasing time/epoch order.
};

// Encodes the initial completed frame as one accepted sample. Returns checked
// state-codec errors and InvalidArgument for a nonzero seed clock.
absl::StatusOr<LiquidCacheBytes3D> EncodeMotionSeed3D(const LiquidState3D& state, LiquidCacheLimits3D limits = {});
// Bounds the complete payload before decoding nested FLST samples. Verifies
// sample ordering and endpoint clocks. No input aliases survive. Returns
// DataLoss for malformed data, FailedPrecondition for unsupported versions and
// ResourceExhausted for byte/count limits. At most 65 samples are supported.
absl::StatusOr<MotionFrame3D> DecodeMotionFrame3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits = {});

// Collector called outside physical attempts through the accepted-state observer.
// The initial policy samples at the first accepted time crossing each of the
// requested equal-duration slots. Actual timestamps are persisted. It never
// alters the simulation schedule, and retains at most intervals+1 full FLST
// samples within the explicit byte budget. Full-state storage is the initial
// reference policy whose measured cost can qualify a later compact carrier.
class MotionCapture3D {
 public:
  // Requires a valid completed start frame, a finite later endpoint and 1..64
  // sample intervals. Returns InvalidArgument or a bounded state-codec failure.
  static absl::StatusOr<MotionCapture3D> Create(
      const LiquidState3D& frame_start,
      double frame_end_time,
      std::size_t intervals,
      LiquidCacheLimits3D limits = {}
  );
  // Accepts strictly later coherent state versions within the open output frame.
  // rollback must name the exact starting state and discards all provisional
  // samples. Failed validation leaves the collector unchanged.
  absl::Status Observe(const LiquidState3D& accepted, bool rollback);
  std::size_t sample_count() const;
  // Requires the completed next frame to match the last observed time and epoch.
  // Requires exact state bytes matching the observed completed endpoint and
  // encodes the interval. No record is published by the collector.
  absl::StatusOr<LiquidCacheBytes3D> Finish(const LiquidState3D& completed) const;

 private:
  MotionCapture3D(
      LiquidClock3D start,
      UniformMacGeometry3D geometry,
      double end_time,
      std::size_t intervals,
      LiquidCacheLimits3D limits
  );
  LiquidClock3D start_;
  UniformMacGeometry3D geometry_;
  LiquidClock3D last_observed_;
  double end_time_;
  std::size_t intervals_;
  std::size_t next_slot_ = 1;
  std::size_t payload_bytes_ = 0;
  LiquidCacheLimits3D limits_;
  std::vector<LiquidCacheBytes3D> samples_;
};
}  // namespace fluidity
