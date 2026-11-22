#include "fluidity/base/mat3.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

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

// Every entry of a Mat3, in matrix-row order, so a test can visit each one.
constexpr std::array<double Mat3::*, 9> kEntries = {
    &Mat3::xx,
    &Mat3::xy,
    &Mat3::xz,
    &Mat3::yx,
    &Mat3::yy,
    &Mat3::yz,
    &Mat3::zx,
    &Mat3::zy,
    &Mat3::zz,
};

void ExpectMatricesNear(const Mat3& actual, const Mat3& expected, double tolerance) {
  for (std::size_t entry = 0; entry < kEntries.size(); ++entry) {
    EXPECT_NEAR(actual.*kEntries[entry], expected.*kEntries[entry], tolerance) << "entry index " << entry;
  }
}

// Tolerance for an entry of a rotation, whose size is at most 1. Rotation calls Sin, so its entries may differ
// from the exact value in the last bits; a few rounding units of 1 (each about 2.2e-16) cover that.
constexpr double kRotationEntryTolerance = 1e-15;

// Tolerance for a value that combines several rotation entries with inputs up to about 4 in size, such as a
// determinant or a rotation times its own axis. The rounding errors of the entries add up and scale with the input,
// so this is ten times kRotationEntryTolerance.
constexpr double kRotationProductTolerance = 1e-14;

// Rotation vectors with no special direction or length, for the properties every rotation has.
constexpr std::array kGeneralRotationVectors = {
    Vec3{.x = 0.3, .y = -1.2, .z = 2.5},
    Vec3{.x = -4.0, .y = 0.5, .z = 0.25},
    Vec3{.x = 1e-3, .y = 2e-3, .z = -5e-4},
    Vec3{.x = 0, .y = 3.1, .z = 0},
};

TEST(Rotation, ZeroVector_IsIdentity) {
  EXPECT_EQ(ToArray(math::Rotation(Vec3{})), ToArray(kIdentity));
}

TEST(Rotation, QuarterTurnAboutZ_SendsXToY) {
  // Right-handed: a positive turn about +Z takes +X toward +Y, not toward -Y.
  const Mat3 rotation = math::Rotation(Vec3{.x = 0, .y = 0, .z = std::numbers::pi / 2});
  const Vec3 turned = rotation * Vec3{.x = 1, .y = 0, .z = 0};
  EXPECT_NEAR(turned.x, 0.0, kRotationEntryTolerance);
  EXPECT_NEAR(turned.y, 1.0, kRotationEntryTolerance);
  EXPECT_NEAR(turned.z, 0.0, kRotationEntryTolerance);
}

TEST(Rotation, QuarterTurnAboutZ_MatchesHandEntries) {
  ExpectMatricesNear(
      math::Rotation(Vec3{.x = 0, .y = 0, .z = std::numbers::pi / 2}),
      kQuarterTurnAboutZ,
      kRotationEntryTolerance
  );
}

TEST(Rotation, QuarterTurnAboutX_MatchesHandEntries) {
  ExpectMatricesNear(
      math::Rotation(Vec3{.x = std::numbers::pi / 2, .y = 0, .z = 0}),
      kQuarterTurnAboutX,
      kRotationEntryTolerance
  );
}

TEST(Rotation, QuarterTurnAboutY_SendsZToX) {
  // Right-handed: a positive turn about +Y takes +Z toward +X, and +X toward -Z.
  constexpr Mat3 kQuarterTurnAboutY = {
      .xx = 0,
      .xy = 0,
      .xz = 1,
      .yx = 0,
      .yy = 1,
      .yz = 0,
      .zx = -1,
      .zy = 0,
      .zz = 0,
  };
  ExpectMatricesNear(
      math::Rotation(Vec3{.x = 0, .y = std::numbers::pi / 2, .z = 0}),
      kQuarterTurnAboutY,
      kRotationEntryTolerance
  );
}

TEST(Rotation, ThirdTurnAboutDiagonal_CyclesTheAxes) {
  // A third of a turn about the diagonal {1, 1, 1} takes +X to +Y, +Y to +Z and +Z to +X. The rotation vector is
  // the unit diagonal {1, 1, 1} / sqrt(3) times the angle 2 pi / 3, and every entry of the answer is 0 or 1.
  constexpr Mat3 kCycleXToYToZ = {
      .xx = 0,
      .xy = 0,
      .xz = 1,
      .yx = 1,
      .yy = 0,
      .yz = 0,
      .zx = 0,
      .zy = 1,
      .zz = 0,
  };
  const double axis_part = (2 * std::numbers::pi / 3) / std::numbers::sqrt3;
  ExpectMatricesNear(
      math::Rotation(Vec3{.x = axis_part, .y = axis_part, .z = axis_part}),
      kCycleXToYToZ,
      kRotationEntryTolerance
  );
}

TEST(Rotation, AnyVector_HasDeterminantOne) {
  for (const Vec3 rotation_vector : kGeneralRotationVectors) {
    EXPECT_NEAR(math::Determinant(math::Rotation(rotation_vector)), 1.0, kRotationProductTolerance);
  }
}

TEST(Rotation, AnyVector_TransposeUndoesIt) {
  for (const Vec3 rotation_vector : kGeneralRotationVectors) {
    const Mat3 rotation = math::Rotation(rotation_vector);
    ExpectMatricesNear(math::Transpose(rotation) * rotation, kIdentity, kRotationEntryTolerance);
  }
}

TEST(Rotation, AnyVector_LeavesItsAxisUnchanged) {
  for (const Vec3 rotation_vector : kGeneralRotationVectors) {
    const Vec3 turned = math::Rotation(rotation_vector) * rotation_vector;
    EXPECT_NEAR(turned.x, rotation_vector.x, kRotationProductTolerance);
    EXPECT_NEAR(turned.y, rotation_vector.y, kRotationProductTolerance);
    EXPECT_NEAR(turned.z, rotation_vector.z, kRotationProductTolerance);
  }
}

TEST(Rotation, TinyVector_IsIdentityPlusFirstOrderTerm) {
  // For a tiny turn by 1e-12 rad about X, the rotation is the identity plus the cross matrix of the vector: entry zy
  // is +1e-12 and entry yz is -1e-12. The result is exact in doubles: the factor 1 - 1e-24 / 6 rounds to 1, and
  // the diagonal 1 - 0.5e-24 rounds to 1.
  constexpr Mat3 kExpected = {
      .xx = 1,
      .xy = 0,
      .xz = 0,
      .yx = 0,
      .yy = 1,
      .yz = -1e-12,
      .zx = 0,
      .zy = 1e-12,
      .zz = 1,
  };
  EXPECT_EQ(ToArray(math::Rotation(Vec3{.x = 1e-12, .y = 0, .z = 0})), ToArray(kExpected));
}

TEST(Rotation, VectorWhoseSquareUnderflows_KeepsFirstOrderTerm) {
  // The square of 1e-200 is below the smallest double, so its length cannot be computed. The rotation must still
  // carry the first-order term instead of dividing by a zero length.
  const Mat3 rotation = math::Rotation(Vec3{.x = 1e-200, .y = 0, .z = 0});
  EXPECT_EQ(rotation.zy, 1e-200);
  EXPECT_EQ(rotation.yz, -1e-200);
  EXPECT_EQ(rotation.yy, 1.0);
  EXPECT_EQ(rotation.zz, 1.0);
}

// 5e-5 rad is small enough for the short series. Here the term a^2 / 2 of entry yy is about 1e-9 and the term a^3 / 6
// of entry zy is about 2e-14, so kSmallAngleTolerance catches a wrong 1/2 or 1/6. The 1/24 term is about 3e-19 here;
// SmallAngleSquaredTerm_UsesTheSeriesCoefficient checks it.
constexpr double kSmallAngle = 5e-5;
constexpr double kSmallAngleTolerance = 1e-18;  // Above the rounding unit of these entries (at most 2.2e-16 for yy).

TEST(Rotation, SmallAngleAboutX_DiagonalMatchesCosine) {
  const Mat3 rotation = math::Rotation(Vec3{.x = kSmallAngle, .y = 0, .z = 0});
  EXPECT_NEAR(rotation.yy, std::cos(kSmallAngle), kSmallAngleTolerance);
}

TEST(Rotation, SmallAngleAboutX_OffDiagonalMatchesSine) {
  const Mat3 rotation = math::Rotation(Vec3{.x = kSmallAngle, .y = 0, .z = 0});
  EXPECT_NEAR(rotation.zy, std::sin(kSmallAngle), kSmallAngleTolerance);
  EXPECT_NEAR(rotation.yz, -std::sin(kSmallAngle), kSmallAngleTolerance);
}

TEST(Rotation, SmallAngleSquaredTerm_UsesTheSeriesCoefficient) {
  // r = {7e-5, 7e-5, 0} has a^2 = 9.8e-9, inside the short series. Entry xy is -z * (factor of K) + x * y * (factor
  // of K^2), and z = 0, so it is (1/2 - a^2 / 24) * 4.9e-9. Worked by hand in 40-digit decimals:
  // 4.9e-9 / 2 - 9.8e-9 * 4.9e-9 / 24 = 2.45e-9 - 2.0008333e-18 = 2.4499999979991667e-9.
  // A wrong 1/24 (say 1/12) moves the result by 2e-18. The rounding unit of the entry is about 4e-25.
  constexpr double kExpectedXy = 2.4499999979991667e-9;
  constexpr double kTolerance = 1e-20;  // Far below the 2e-18 step, far above the 4e-25 rounding unit.
  const Mat3 rotation = math::Rotation(Vec3{.x = 7e-5, .y = 7e-5, .z = 0});
  EXPECT_NEAR(rotation.xy, kExpectedXy, kTolerance);
  EXPECT_NEAR(rotation.yx, kExpectedXy, kTolerance);
}

TEST(IsFinite, FiniteMatrix_IsFinite) {
  EXPECT_TRUE(math::IsFinite(kCounting));
}

TEST(IsFinite, MatrixWithOneNaN_IsNotFinite) {
  for (const auto entry : kEntries) {
    Mat3 matrix = kIdentity;
    matrix.*entry = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(math::IsFinite(matrix));
  }
}

TEST(IsFinite, MatrixWithOneInfinity_IsNotFinite) {
  Mat3 matrix = kIdentity;
  matrix.zz = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE(math::IsFinite(matrix));
}

}  // namespace
}  // namespace fluidity
