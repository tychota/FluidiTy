#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include "absl/status/statusor.h"
#include "fluidity/base/liquid_boundary_transfer_types_3d.h"
#include "fluidity/host/adaptive_mac_topology_3d.h"
#include "fluidity/host/liquid_remap_3d.h"

namespace fluidity {

// Bounded owning host records. Active size is explicit; Resize never allocates.
// Move invalidates views; operations preserve storage addresses until move/destruction.
// Values outside size() are scratch and do not participate in physical equality.
// Failures carry a status code and empty message: constructing/destroying an error
// must not allocate, including when allocation itself failed.
template <typename Record>
  requires(std::is_trivially_copyable_v<Record> && std::is_nothrow_default_constructible_v<Record>)
class LiquidRecords3D {
 public:
  LiquidRecords3D() = default;
  LiquidRecords3D(LiquidRecords3D&& other) noexcept
      : records_(std::move(other.records_)),
        size_(std::exchange(other.size_, 0)),
        capacity_(std::exchange(other.capacity_, 0)) {
  }
  LiquidRecords3D& operator=(LiquidRecords3D&& other) noexcept {
    if (this != &other) {
      records_ = std::move(other.records_);
      size_ = std::exchange(other.size_, 0);
      capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
  }
  LiquidRecords3D(const LiquidRecords3D&) = delete;
  LiquidRecords3D& operator=(const LiquidRecords3D&) = delete;
  // Allocates outside physical attempts. Overflow/allocation failure returns
  // ResourceExhausted without a partial result. Zero capacity is valid.
  static absl::StatusOr<LiquidRecords3D> Create(std::size_t capacity) {
    if (capacity > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(Record)) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    LiquidRecords3D result;
    if (capacity != 0) {
      result.records_.reset(new (std::nothrow) Record[capacity]{});
      if (!result.records_) {
        return absl::Status(absl::StatusCode::kResourceExhausted, {});
      }
    }
    result.capacity_ = capacity;
    return result;
  }
  // Changes the valid extent within capacity, value-initializing newly active records.
  // ResourceExhausted leaves values/extent intact. No allocation or deallocation.
  absl::Status Resize(std::size_t size) {
    if (size > capacity_) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    for (std::size_t index = size_; index < size; ++index) {
      records_[index] = Record{};
    }
    size_ = size;
    return absl::OkStatus();
  }
  std::size_t size() const {
    return size_;
  }
  std::size_t capacity() const {
    return capacity_;
  }
  // Views borrow this owner; caller must respect size() for indexing.
  std::span<Record> values() {
    return {records_.get(), size_};
  }
  std::span<const Record> values() const {
    return {records_.get(), size_};
  }
  Record& operator[](std::size_t index) {
    return records_[index];
  }
  const Record& operator[](std::size_t index) const {
    return records_[index];
  }
  bool operator==(const LiquidRecords3D& other) const {
    return std::ranges::equal(values(), other.values());
  }

 private:
  std::unique_ptr<Record[]> records_;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

struct LiquidParticleState3D {
  std::uint64_t id = 0;  // Nonzero, unique, below next_particle_id; array order is stable sort order, not ID order.
  std::uint32_t tag = 0;  // Packed substance/phase/flags, preserved verbatim.
  Vec3 world_position{};  // Meters.
  Vec3 world_velocity{};  // Meters per second; stored rather than advecting velocity.
  Mat3 velocity_slopes{};  // APIC Cartesian rows, inverse seconds, finite even when disabled.
  bool affine_enabled = true;
  double mass = 1.0;  // Positive kilograms.
  double reference_volume = 1.0;  // Positive cubic meters.
  double volume_ratio = 1.0;  // Positive dimensionless J.
  std::int64_t side_tag = 0;  // Equality identifies a material/body side.
  bool pic_initialized = false;  // False means a newly emitted particle must start with PIC.
  bool operator==(const LiquidParticleState3D&) const;
};
struct LiquidBlockState3D {
  BlockIndex3D block_index{};  // Canonical unique (z,y,x) order, signed 4-cell blocks.
  std::uint64_t generation = 0;
  std::uint32_t activation_age = 0;  // Accepted steps since activation.
  std::uint32_t inactive_age = 0;  // Hysteresis age in accepted steps.
  bool operator==(const LiquidBlockState3D&) const;
};
enum class LiquidCellOccupancy3D : std::uint8_t {
  kAir,
  kLiquid,
  kSolid,
};
struct LiquidCellState3D {
  CellIndex3D cell_index{};  // Selected cells in unique (z,y,x) order; owner must be active.
  LiquidCellOccupancy3D occupancy = LiquidCellOccupancy3D::kLiquid;
  double previous_pressure = 0.0;  // Finite gauge pressure, Pa.
  bool operator==(const LiquidCellState3D&) const;
};
struct LiquidFaceState3D {
  FaceIndex3D face_index{};  // Unique faces in ascending owned-slot order.
  std::size_t face_slot = 0;  // Block ordinal * 192 + axis * 64 + local z*16+y*4+x.
  double carrier_velocity = 0.0;  // Positive Cartesian normal velocity, m/s.
  LiquidFaceValidity3D validity = LiquidFaceValidity3D::kMissing;  // Independent of mass and side.
  std::int64_t side_tag = 0;
  bool structurally_eligible = true;
  bool operator==(const LiquidFaceState3D&) const;
};
struct LiquidSourceState3D {
  std::uint64_t source_ordinal = 0;  // Resolved source identity, unique ascending order.
  std::uint64_t next_child_ordinal = 0;  // Counter-based emission identity; advance only on accepted emission.
  double fractional_remainder = 0.0;  // Particle count in [0,1).
  bool operator==(const LiquidSourceState3D&) const = default;
};
struct LiquidClock3D {
  double time = 0.0;  // Finite nonnegative seconds.
  std::uint64_t frame_ordinal = 0;  // Completed frames.
  std::uint64_t substep_ordinal = 0;  // Accepted steps in the current frame; zero in a completed-frame version.
  std::uint64_t accepted_step_ordinal = 0;  // Philox accepted-step key input.
  std::uint64_t epoch = 0;  // Coherent version identity, advances once per accepted step.
  bool operator==(const LiquidClock3D&) const = default;
};
inline constexpr std::size_t kMaximumAdaptiveLiquidCells3D = 4096;
inline constexpr std::size_t kMaximumAdaptiveLiquidFaces3D = 24576;

// A graded leaf in canonical signed (z,y,x) lower-corner order. Inventory is
// the extensive Eulerian owner in SI units; previous pressure is in Pa.
struct LiquidAdaptiveCellState3D {
  AdaptiveMacCellKey3D key{};
  LiquidCellOccupancy3D occupancy = LiquidCellOccupancy3D::kLiquid;
  LiquidRemapInventory3D inventory{};
  double previous_pressure = 0;
  bool operator==(const LiquidAdaptiveCellState3D& other) const;
};

// The record index is the global canonical subface slot reconstructed from the
// accepted leaf keys, including exterior and sealed faces.
struct LiquidAdaptiveFaceState3D {
  double carrier_velocity = 0;  // Positive Cartesian normal, m/s.
  LiquidFaceValidity3D validity = LiquidFaceValidity3D::kMissing;
  std::int64_t side_tag = 0;
  bool structurally_eligible = true;
  bool operator==(const LiquidAdaptiveFaceState3D&) const = default;
};

struct LiquidState3D {
  UniformMacGeometry3D geometry{{0.0, 0.0, 0.0}, 1.0};
  LiquidClock3D clock;
  std::uint64_t random_seed = 0;  // Philox seed, with accepted step and source/child ordinal as other inputs.
  std::uint64_t next_particle_id = 1;  // Never zero; greater than every live ID.
  // size() is the valid extent; capacity and unused storage do not participate in equality.
  LiquidRecords3D<LiquidParticleState3D> particles;
  LiquidRecords3D<LiquidBlockState3D> blocks;
  LiquidRecords3D<LiquidCellState3D> cells;
  LiquidRecords3D<LiquidFaceState3D> faces;
  LiquidRecords3D<LiquidSourceState3D> sources;
  // Nonempty adaptive_cells selects graded ownership. geometry names the finest
  // lattice, and blocks/cells/faces are empty. The cells/faces capacity fields
  // reserve the corresponding adaptive records in this mode.
  LiquidRecords3D<LiquidAdaptiveCellState3D> adaptive_cells;
  LiquidRecords3D<LiquidAdaptiveFaceState3D> adaptive_faces;
  bool operator==(const LiquidState3D&) const;
};
struct LiquidStateCapacities3D {
  std::size_t particles = 0;
  std::size_t blocks = 0;
  std::size_t cells = 0;
  std::size_t faces = 0;
  std::size_t sources = 0;
  bool operator==(const LiquidStateCapacities3D&) const = default;
};
// Validates source and allocates an independent copy with the requested capacities.
// Inadequate, overflowing or failed allocations return ResourceExhausted; invalid
// source propagates validation status. Source is unchanged on every result.
absl::StatusOr<LiquidState3D> CloneLiquidState3D(const LiquidState3D& source, LiquidStateCapacities3D capacities);

enum class LiquidAttemptDisposition3D {
  kAccepted,
  kRetryDt,
  kCapacity,
  kFatal,
};
struct LiquidAttemptOutcome3D {
  LiquidAttemptDisposition3D disposition;
  std::optional<double> retry_duration;  // Seconds; present only for an allowed dt/2 retry.
};

// Checks finite fields, positive m/V0/J, unique IDs, signed topology ownership,
// canonical keys, face slots, enum ranges and clock relations. ID uniqueness uses
// an allocation-free quadratic scan; physical particle order is never changed.
// Validation allocates/deallocates nothing on either success or failure. Errors
// carry only a code (empty message); format diagnostics outside the physical attempt.
// Invalid data returns InvalidArgument; lattice/slot overflow is OutOfRange. Read-only input may alias any other
// read-only state.
absl::Status ValidateLiquidState3D(const LiquidState3D& state);

// Single-threaded owner of three independent physical versions. No copy is allowed:
// Moving invalidates all prior borrows; a moved-from owner may only be destroyed
// or assigned. Attempts allocate/deallocate nothing, including failed validation
// and returned-status destruction. All errors are code-only (empty message); the
// scheduler must collect/format detailed diagnostics outside physical attempts.
class LiquidFrameTransaction3D {
 public:
  LiquidFrameTransaction3D(LiquidFrameTransaction3D&&) noexcept = default;
  LiquidFrameTransaction3D& operator=(LiquidFrameTransaction3D&&) noexcept = default;
  LiquidFrameTransaction3D(const LiquidFrameTransaction3D&) = delete;
  LiquidFrameTransaction3D& operator=(const LiquidFrameTransaction3D&) = delete;
  // Validates and copies initial_state into independent reserved versions. Invalid
  // state propagates validation status; inadequate/overflowing capacities return
  // ResourceExhausted, including allocation failure. Does not retain input aliases
  // or modify input on failure.
  static absl::StatusOr<LiquidFrameTransaction3D>
  Create(const LiquidState3D& initial_state, LiquidStateCapacities3D capacities);
  // Const borrowed versions. References to these state objects survive transitions;
  // record spans/pointers must be reacquired after any modifying operation. Neither
  // borrow may outlive this owner. frame_start contents are immutable during a frame.
  const LiquidState3D& committed() const {
    return committed_;
  }
  const LiquidState3D& frame_start() const {
    return frame_start_;
  }
  // Freezes current state for a finite strictly later target (seconds). Requires
  // no active frame and a completed-frame clock (FailedPrecondition). InvalidArgument
  // for invalid target; OutOfRange for exhausted frame count. Errors preserve all state.
  absl::Status BeginFrame(double target_frame_time);
  // Copies committed values into reserved trial storage, without allocation. Requires
  // active, nonfatal frame and no outstanding trial/capacity decision. Borrow ends
  // at FinishAttempt; caller may resize only within configured capacities, may not
  // replace record owners or modify geometry, seed, resolved source identities or
  // clock. Source child ordinals and next_particle_id may only advance on acceptance.
  // FailedPrecondition for invalid sequencing/storage; errors expose no trial.
  absl::StatusOr<LiquidState3D*> ResetTrialFromCommitted();
  // Exact prospective accepted clock for the active trial, including final-frame
  // snapping and checked counters. Does not validate or publish trial payload.
  // FailedPrecondition without an active trial; InvalidArgument for malformed or
  // unexpected retry duration; OutOfRange for nonadvancing time/counter overflow.
  // Read-only: device publication can stage this clock before FinishAttempt.
  absl::StatusOr<LiquidClock3D> PreviewAcceptedClock(double attempted_duration) const;
  // Consumes the borrow. Accepted finite positive dt must advance time without
  // overshoot; a final dt equal to remaining may snap by at most one ULP. Validates
  // before publish; clock changes once. The final accept also advances the frame
  // ordinal and resets substep to zero atomically in that same epoch. Retry allows
  // up to two successive halvings from the same committed version. Invalid attempts, fatal decision and exhausted
  // retries stop the frame and preserve the last accepted version. Returns status
  // InvalidArgument for invalid data, FailedPrecondition for order, OutOfRange for
  // time/counter overflow, ResourceExhausted for storage; explicit fatal returns a
  // fatal outcome. The producer must pass kFatal for incompatible physical constraints.
  absl::StatusOr<LiquidAttemptOutcome3D> FinishAttempt(LiquidAttemptDisposition3D decision, double attempted_duration);
  // Only after kCapacity, once per frame. All capacities must be nondecreasing and
  // at least one must grow. Checks and builds all replacement versions before
  // publication, then rolls back to immutable frame_start with the same target.
  // FailedPrecondition for wrong order; InvalidArgument for nongrowth/shrinkage;
  // ResourceExhausted for overflow/allocation failure. Failed growth preserves every
  // previous version and growth authority and may be corrected/retried.
  absl::Status RestartFrameWithCapacity(LiquidStateCapacities3D capacities);
  // Requires exact target time and no pending attempt/failure. Closes scheduling
  // only: committed bytes/epoch already include completion from the final accept.
  // FailedPrecondition on wrong order; every result preserves physical state.
  absl::Status FinishFrame();

 private:
  LiquidFrameTransaction3D() = default;
  LiquidState3D committed_;
  LiquidState3D trial_;
  LiquidState3D frame_start_;
  LiquidStateCapacities3D capacities_;
  double target_frame_time_ = 0.0;
  std::optional<double> retry_duration_;
  unsigned retry_count_ = 0;
  bool frame_active_ = false;
  bool trial_active_ = false;
  bool capacity_pending_ = false;
  bool capacity_restarted_ = false;
  bool fatal_ = false;
};
}  // namespace fluidity
