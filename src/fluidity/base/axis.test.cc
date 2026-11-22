#include "fluidity/base/axis.h"

#include <gtest/gtest.h>

#include <array>

#include "fluidity/base/vec3.h"

namespace fluidity {
namespace {

constexpr std::array<double, 3> ToArray(Vec3 vector) {
  return {vector.x, vector.y, vector.z};
}

TEST(Axis, UnitVectorOfZ_IsUp) {
  constexpr Vec3 kUnitVector = math::UnitVector(Axis::kZ);
  static_assert(ToArray(kUnitVector) == ToArray(kUp));
  static_assert(ToArray(kUnitVector) == std::array{0.0, 0.0, 1.0});
  EXPECT_EQ(ToArray(kUnitVector), ToArray(kUp));
  EXPECT_EQ(ToArray(kUnitVector), (std::array{0.0, 0.0, 1.0}));
}

TEST(Axis, UnitVectorOfX_PointsAlongX) {
  constexpr Vec3 kAlongX = math::UnitVector(Axis::kX);
  static_assert(ToArray(kAlongX) == std::array{1.0, 0.0, 0.0});
  EXPECT_EQ(ToArray(kAlongX), (std::array{1.0, 0.0, 0.0}));
}

TEST(Axis, UnitVectorOfY_PointsAlongY) {
  constexpr Vec3 kAlongY = math::UnitVector(Axis::kY);
  static_assert(ToArray(kAlongY) == std::array{0.0, 1.0, 0.0});
  EXPECT_EQ(ToArray(kAlongY), (std::array{0.0, 1.0, 0.0}));
}

TEST(Axis, XCrossY_IsUp) {
  // Right-handed axes: the third axis is up.
  constexpr Vec3 kCross = math::Cross(math::UnitVector(Axis::kX), math::UnitVector(Axis::kY));
  static_assert(ToArray(kCross) == ToArray(kUp));
  EXPECT_EQ(ToArray(kCross), ToArray(kUp));
}

}  // namespace
}  // namespace fluidity
