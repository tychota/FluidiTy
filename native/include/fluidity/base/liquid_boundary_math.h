#pragma once
#include <cstddef>

#include "fluidity/base/host_device.h"
#include "fluidity/base/mac_types_3d.h"
#if defined(__CUDACC__)
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/numeric>
#else
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#endif
namespace fluidity::boundary_internal {
#if defined(__CUDACC__)
namespace numeric = cuda::std;
#else
namespace numeric = std;
#endif
// Arithmetic outcomes cross the device ABI without host Status or containers.
enum class Error {
  kNone,
  kInvalid,
  kPrecondition,
  kRange,
};
template <typename T>
struct Maybe {
  T value{};
  bool present = false;
  Maybe() = default;
  FLUIDITY_HD Maybe(T input) : value(input), present(true) {
  }
  FLUIDITY_HD bool has_value() const {
    return present;
  }
  FLUIDITY_HD const T& operator*() const {
    return value;
  }
};
template <typename T>
struct Value {
  T value{};
  Error error = Error::kNone;
  FLUIDITY_HD Value(T input) : value(input) {
  }
  FLUIDITY_HD Value(Error input) : error(input) {
  }
  FLUIDITY_HD bool ok() const {
    return error == Error::kNone;
  }
  FLUIDITY_HD Error status() const {
    return error;
  }
  FLUIDITY_HD T& operator*() {
    return value;
  }
  FLUIDITY_HD const T& operator*() const {
    return value;
  }
  FLUIDITY_HD T* operator->() {
    return &value;
  }
  FLUIDITY_HD const T* operator->() const {
    return &value;
  }
};
// kind: 0 absent, 1 halfspace, 2 slab. Coordinate is world meters at start.
struct Obstacle {
  int kind = 0;
  Vec3 normal{};
  double coordinate = 0;
  double half_width = 0;
  Vec3 velocity{};
};
struct ContactInput {
  Vec3 start_world_position{};
  Vec3 stored_world_velocity{};
  Vec3 advecting_world_velocity{};
  Obstacle obstacle{};
  double step_duration = 0;
  double friction_coefficient = 0;
  double penetration_tolerance = 0;
};
struct ContactOutput {
  Vec3 world_position{};
  Vec3 stored_world_velocity{};
  Vec3 advecting_world_velocity{};
  Maybe<double> contact_fraction{};
  Vec3 outward_normal{};
  double incoming_stored_relative_normal_speed = 0;
  double incoming_advecting_relative_normal_speed = 0;
  int diagnostic = 0;  // 0 none, 1 swept, 2 initial penetration.
};
}  // namespace fluidity::boundary_internal
