#pragma once

#include "fluidity/base/host_device.h"

namespace fluidity {

// Two components in Cartesian order.
struct Vec2 {
  double x;
  double y;
};

// Row-major matrix with Cartesian rows and columns.
struct Mat2 {
  double xx;
  double xy;
  double yx;
  double yy;
};

// Adds corresponding components.
FLUIDITY_HD constexpr Vec2 operator+(Vec2 lhs, Vec2 rhs) noexcept {
  return {lhs.x + rhs.x, lhs.y + rhs.y};
}

// Subtracts the corresponding components of rhs from lhs.
FLUIDITY_HD constexpr Vec2 operator-(Vec2 lhs, Vec2 rhs) noexcept {
  return {lhs.x - rhs.x, lhs.y - rhs.y};
}

// Scales both components by the same factor.
FLUIDITY_HD constexpr Vec2 operator*(Vec2 vector, double scale) noexcept {
  return {vector.x * scale, vector.y * scale};
}

// Supports the same scaling with the scalar on the left.
FLUIDITY_HD constexpr Vec2 operator*(double scale, Vec2 vector) noexcept {
  return vector * scale;
}

}  // namespace fluidity
