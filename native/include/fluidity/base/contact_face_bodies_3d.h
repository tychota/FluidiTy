#pragma once

#include <cstdint>

#include "fluidity/base/distinct_body_contact_3d.h"

namespace fluidity {

enum class ContactFaceDepositStatus3D : std::uint8_t {
  kAccepted,
  kInvalid,
  kRange,
  kBodyCapacity,
};

// Holds only the active body records for one MAC face. Velocities are kept
// separate by stable body ID and are reconstructed from mass and momentum.
class ContactFaceBodies3D {
 public:
  static constexpr int kCapacity = 4;

  // Adds a positive mass in kg and world Cartesian momentum in kg m/s.
  // Call in the desired accumulation order. Records are ordered by increasing
  // nonzero body ID. A fifth distinct ID returns kBodyCapacity even when an
  // existing ID could still receive a contribution. All failures preserve
  // every record. The caller supplies synchronization and face ownership.
  FLUIDITY_HD ContactFaceDepositStatus3D
  Deposit(std::uint64_t body_id, double mass_contribution, Vec3 momentum_contribution) noexcept {
    namespace detail = distinct_contact_detail;
    namespace numeric = detail::numeric;
    if (body_id == 0 || !numeric::isfinite(mass_contribution) || mass_contribution <= 0
        || !detail::Finite(momentum_contribution)) {
      return ContactFaceDepositStatus3D::kInvalid;
    }

    int slot = 0;
    while (slot < size_ && records_[slot].body_id < body_id) {
      ++slot;
    }
    const bool existing = slot < size_ && records_[slot].body_id == body_id;
    if (!existing && size_ == kCapacity) {
      return ContactFaceDepositStatus3D::kBodyCapacity;
    }

    BodyMomentum candidate{
        .body_id = body_id,
        .mass = mass_contribution,
        .world_momentum = momentum_contribution,
    };
    if (existing) {
      candidate.mass += records_[slot].mass;
      candidate.world_momentum = detail::AddScaled(records_[slot].world_momentum, momentum_contribution, 1);
    }
    if (!numeric::isfinite(candidate.mass) || !detail::Finite(candidate.world_momentum)
        || !detail::Finite(Velocity(candidate))) {
      return ContactFaceDepositStatus3D::kRange;
    }

    if (!existing) {
      for (int index = size_; index > slot; --index) {
        records_[index] = records_[index - 1];
      }
      ++size_;
    }
    records_[slot] = candidate;
    return ContactFaceDepositStatus3D::kAccepted;
  }

  // Returns the number of distinct bodies deposited on the face.
  FLUIDITY_HD int size() const noexcept {
    return size_;
  }

  // Reads one body sample in increasing ID order with velocity in m/s.
  // An index outside [0, size()) leaves output unchanged and returns false.
  FLUIDITY_HD bool Read(int index, ContactBodySample3D& output) const noexcept {
    if (index < 0 || index >= size_) {
      return false;
    }
    const BodyMomentum& record = records_[index];
    output = {
        .body_id = record.body_id,
        .mass = record.mass,
        .world_velocity = Velocity(record),
    };
    return true;
  }

 private:
  struct BodyMomentum {
    std::uint64_t body_id;
    double mass;
    Vec3 world_momentum;
  };

  FLUIDITY_HD static Vec3 Velocity(const BodyMomentum& record) noexcept {
    return {
        record.world_momentum.x / record.mass,
        record.world_momentum.y / record.mass,
        record.world_momentum.z / record.mass,
    };
  }

  BodyMomentum records_[kCapacity]{};
  int size_ = 0;
};

}  // namespace fluidity
