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

// Scatters one face impulse to a contributing particle in world m/s.
// The weight is in (0, 1] and body_face_mass is the sum of weighted
// particle masses in kg. Invalid or unrepresentable data preserve output.
FLUIDITY_HD inline bool
ScatterContactFaceImpulse3D(Vec3 body_impulse, double particle_weight, double body_face_mass, Vec3& output) noexcept {
  namespace detail = distinct_contact_detail;
  namespace numeric = detail::numeric;
  if (!detail::Finite(body_impulse) || !numeric::isfinite(particle_weight) || particle_weight <= 0
      || particle_weight > 1 || !numeric::isfinite(body_face_mass) || body_face_mass <= 0) {
    return false;
  }
  const Vec3 candidate{
      (body_impulse.x * particle_weight) / body_face_mass,
      (body_impulse.y * particle_weight) / body_face_mass,
      (body_impulse.z * particle_weight) / body_face_mass,
  };
  if (!detail::Finite(candidate)) {
    return false;
  }
  output = candidate;
  return true;
}

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

    const int slot = LowerBound(body_id);
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

  // Applies a detected pair impulse in world SI units. normal_toward_first
  // points from the second body toward the first. The caller supplies pair
  // order and synchronization. Missing or identical IDs, invalid contact
  // parameters or range failure preserve both the table and output.
  FLUIDITY_HD bool ResolvePair(
      std::uint64_t first_body_id,
      std::uint64_t second_body_id,
      Vec3 normal_toward_first,
      double friction_coefficient,
      DistinctBodyContact3D& output
  ) noexcept {
    const int first_slot = LowerBound(first_body_id);
    const int second_slot = LowerBound(second_body_id);
    ContactBodySample3D first{};
    ContactBodySample3D second{};
    if (!Read(first_slot, first) || first.body_id != first_body_id || !Read(second_slot, second)
        || second.body_id != second_body_id) {
      return false;
    }
    DistinctBodyContact3D candidate{};
    if (!ResolveDistinctBodyContact3D(first, second, normal_toward_first, friction_coefficient, candidate)) {
      return false;
    }
    if (candidate.incoming_relative_normal_speed < 0) {
      namespace detail = distinct_contact_detail;
      BodyMomentum first_record = records_[first_slot];
      BodyMomentum second_record = records_[second_slot];
      first_record.world_momentum = detail::AddScaled(first_record.world_momentum, candidate.impulse_on_first, 1);
      second_record.world_momentum = detail::AddScaled(second_record.world_momentum, candidate.impulse_on_first, -1);
      candidate.first_world_velocity = Velocity(first_record);
      candidate.second_world_velocity = Velocity(second_record);
      if (!detail::Finite(first_record.world_momentum) || !detail::Finite(second_record.world_momentum)
          || !detail::Finite(candidate.first_world_velocity) || !detail::Finite(candidate.second_world_velocity)) {
        return false;
      }
      records_[first_slot] = first_record;
      records_[second_slot] = second_record;
    }
    output = candidate;
    return true;
  }

 private:
  FLUIDITY_HD int LowerBound(std::uint64_t body_id) const noexcept {
    int slot = 0;
    while (slot < size_ && records_[slot].body_id < body_id) {
      ++slot;
    }
    return slot;
  }
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
