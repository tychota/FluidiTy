#include "fluidity/base/mat3.h"

#include <gtest/gtest.h>

#include <array>

#include "fluidity/base/vec3.h"

namespace fluidity {
namespace {

// Every value below is a small integer, so each sum and product is exact in doubles and exact equality is right.
constexpr std::array<double, 3> ToArray(Vec3 vector) {
  return {vector.x, vector.y, vector.z};
}

constexpr std::array<double, 9> ToArray(Mat3 matrix) {
  return {matrix.xx, matrix.xy, matrix.xz, matrix.yx, matrix.yy, matrix.yz, matrix.zx, matrix.zy, matrix.zz};
}

// A quarter turn about X: it takes +Y to +Z and +Z to -Y. It is also the rotation from a Y-up description to the
// Z-up convention of base/axis.h.
constexpr Mat3 kQuarterTurnAboutX = {
    .xx = 1,
    .xy = 0,
    .xz = 0,
    .yx = 0,
    .yy = 0,
    .yz = -1,
    .zx = 0,
    .zy = 1,
    .zz = 0,
};

// A quarter turn about Z: it takes +X to +Y and +Y to -X.
constexpr Mat3 kQuarterTurnAboutZ = {
    .xx = 0,
    .xy = -1,
    .xz = 0,
    .yx = 1,
    .yy = 0,
    .yz = 0,
    .zx = 0,
    .zy = 0,
    .zz = 1,
};

// Swaps the y and z axes of a vector. This is a reflection, not a turn.
constexpr Mat3 kSwapYAndZ = {
    .xx = 1,
    .xy = 0,
    .xz = 0,
    .yx = 0,
    .yy = 0,
    .yz = 1,
    .zx = 0,
    .zy = 1,
    .zz = 0,
};

constexpr Mat3 kCounting = {
    .xx = 1,
    .xy = 2,
    .xz = 3,
    .yx = 4,
    .yy = 5,
    .yz = 6,
    .zx = 7,
    .zy = 8,
    .zz = 9,
};

TEST(Mat3, IdentityTimesVector_ReturnsVector) {
  constexpr Vec3 kProduct = kIdentity * Vec3{.x = 3, .y = -4, .z = 5};
  static_assert(ToArray(kProduct) == std::array{3.0, -4.0, 5.0});
  EXPECT_EQ(ToArray(kProduct), (std::array{3.0, -4.0, 5.0}));
}

TEST(Mat3, TimesVector_UsesMatrixRows) {
  // Each output axis is the dot of one matrix row with the vector: 1 - 3 = -2, 4 - 6 = -2, 7 - 9 = -2.
  constexpr Vec3 kProduct = kCounting * Vec3{.x = 1, .y = 0, .z = -1};
  static_assert(ToArray(kProduct) == std::array{-2.0, -2.0, -2.0});
  EXPECT_EQ(ToArray(kProduct), (std::array{-2.0, -2.0, -2.0}));
}

TEST(Mat3, TimesVector_GeneralVector_UsesEveryEntry) {
  // {1 + 4 + 9, 4 + 10 + 18, 7 + 16 + 27} = {14, 32, 50}
  constexpr Vec3 kProduct = kCounting * Vec3{.x = 1, .y = 2, .z = 3};
  static_assert(ToArray(kProduct) == std::array{14.0, 32.0, 50.0});
  EXPECT_EQ(ToArray(kProduct), (std::array{14.0, 32.0, 50.0}));
}

TEST(Mat3, TimesMatrix_GeneralMatrices_GivesEveryEntry) {
  constexpr Mat3 kReversed = {
      .xx = 9,
      .xy = 8,
      .xz = 7,
      .yx = 6,
      .yy = 5,
      .yz = 4,
      .zx = 3,
      .zy = 2,
      .zz = 1,
  };
  constexpr Mat3 kProduct = kCounting * kReversed;
  static_assert(ToArray(kProduct) == std::array{30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0});
  EXPECT_EQ(ToArray(kProduct), (std::array{30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0}));
}

TEST(Mat3, TimesMatrix_IsNotCommutative) {
  constexpr Mat3 kXThenZ = kQuarterTurnAboutZ * kQuarterTurnAboutX;
  constexpr Mat3 kZThenX = kQuarterTurnAboutX * kQuarterTurnAboutZ;
  static_assert(ToArray(kXThenZ) != ToArray(kZThenX));
  EXPECT_NE(ToArray(kXThenZ), ToArray(kZThenX));
}

TEST(Mat3, TimesMatrix_AppliesRightMatrixFirst) {
  // Turn about X first, then about Z: +Y goes to +Z, which the turn about Z leaves alone.
  constexpr Mat3 kProduct = kQuarterTurnAboutZ * kQuarterTurnAboutX;
  constexpr Vec3 kTurned = kProduct * Vec3{.x = 0, .y = 1, .z = 0};
  static_assert(ToArray(kTurned) == std::array{0.0, 0.0, 1.0});
  EXPECT_EQ(ToArray(kTurned), (std::array{0.0, 0.0, 1.0}));
}

TEST(Mat3, Add_AddsEachEntry) {
  // Adding the transpose pairs entry xy with entry yx, so a swapped index changes the answer.
  constexpr Mat3 kSum = kCounting + math::Transpose(kCounting);
  static_assert(ToArray(kSum) == std::array{2.0, 6.0, 10.0, 6.0, 10.0, 14.0, 10.0, 14.0, 18.0});
  EXPECT_EQ(ToArray(kSum), (std::array{2.0, 6.0, 10.0, 6.0, 10.0, 14.0, 10.0, 14.0, 18.0}));
}

TEST(Mat3, Subtract_SubtractsEachEntry) {
  constexpr Mat3 kDifference = kCounting - math::Transpose(kCounting);
  static_assert(ToArray(kDifference) == std::array{0.0, -2.0, -4.0, 2.0, 0.0, -2.0, 4.0, 2.0, 0.0});
  EXPECT_EQ(ToArray(kDifference), (std::array{0.0, -2.0, -4.0, 2.0, 0.0, -2.0, 4.0, 2.0, 0.0}));
}

TEST(Mat3, ScaleByNumber_ScalesEachEntry) {
  constexpr Mat3 kScaled = kCounting * 2.0;
  static_assert(ToArray(kScaled) == std::array{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0});
  EXPECT_EQ(ToArray(kScaled), (std::array{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0}));
}

TEST(Transpose, SwapsMatrixRowsWithColumns) {
  constexpr Mat3 kTransposed = math::Transpose(kCounting);
  static_assert(kTransposed.xy == kCounting.yx);
  static_assert(kTransposed.xz == kCounting.zx);
  static_assert(kTransposed.yx == kCounting.xy);
  static_assert(kTransposed.yz == kCounting.zy);
  static_assert(kTransposed.zx == kCounting.xz);
  static_assert(kTransposed.zy == kCounting.yz);
  EXPECT_EQ(kTransposed.xy, kCounting.yx);
  EXPECT_EQ(kTransposed.xz, kCounting.zx);
  EXPECT_EQ(kTransposed.yx, kCounting.xy);
  EXPECT_EQ(kTransposed.yz, kCounting.zy);
  EXPECT_EQ(kTransposed.zx, kCounting.xz);
  EXPECT_EQ(kTransposed.zy, kCounting.yz);
}

TEST(Transpose, KeepsDiagonal) {
  constexpr Mat3 kTransposed = math::Transpose(kCounting);
  static_assert(kTransposed.xx == 1.0 && kTransposed.yy == 5.0 && kTransposed.zz == 9.0);
  EXPECT_EQ(ToArray(kTransposed)[0], 1.0);
  EXPECT_EQ(ToArray(kTransposed)[4], 5.0);
  EXPECT_EQ(ToArray(kTransposed)[8], 9.0);
}

TEST(Determinant, Identity_IsOne) {
  constexpr double kDeterminant = math::Determinant(kIdentity);
  static_assert(kDeterminant == 1.0);
  EXPECT_EQ(kDeterminant, 1.0);
}

TEST(Determinant, SwapOfYWithZ_IsMinusOne) {
  // A swap of two axes is a reflection, not a rotation.
  constexpr double kDeterminant = math::Determinant(kSwapYAndZ);
  static_assert(kDeterminant == -1.0);
  EXPECT_EQ(kDeterminant, -1.0);
}

TEST(Determinant, YUpToZUpRotation_IsOne) {
  constexpr double kDeterminant = math::Determinant(kQuarterTurnAboutX);
  static_assert(kDeterminant == 1.0);
  EXPECT_EQ(kDeterminant, 1.0);
}

TEST(Determinant, GeneralMatrix_EveryTermCounts) {
  // Matrix rows {1, 2, 3}, {0, 1, 4}, {5, 6, 0}: 1 * (0 - 24) - 2 * (0 - 20) + 3 * (0 - 5) = -24 + 40 - 15 = 1.
  constexpr Mat3 kMatrix = {
      .xx = 1,
      .xy = 2,
      .xz = 3,
      .yx = 0,
      .yy = 1,
      .yz = 4,
      .zx = 5,
      .zy = 6,
      .zz = 0,
  };
  constexpr double kDeterminant = math::Determinant(kMatrix);
  static_assert(kDeterminant == 1.0);
  EXPECT_EQ(kDeterminant, 1.0);
}

TEST(Determinant, CountingMatrix_IsZero) {
  // The matrix rows {1, 2, 3}, {4, 5, 6}, {7, 8, 9} are linearly dependent.
  constexpr double kDeterminant = math::Determinant(kCounting);
  static_assert(kDeterminant == 0.0);
  EXPECT_EQ(kDeterminant, 0.0);
}

TEST(Mat3, YUpToZUpRotation_SendsOldGravityDown) {
  constexpr Vec3 kGravityInYUp = {.x = 0, .y = -9.81, .z = 0};
  constexpr Vec3 kGravityInZUp = kQuarterTurnAboutX * kGravityInYUp;
  static_assert(ToArray(kGravityInZUp) == std::array{0.0, 0.0, -9.81});
  EXPECT_EQ(ToArray(kGravityInZUp), (std::array{0.0, 0.0, -9.81}));
}

}  // namespace
}  // namespace fluidity
