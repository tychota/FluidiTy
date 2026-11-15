#include "fluidity/contact/contact_tolerance.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {
namespace {

// A solid that neither moves nor turns.
constexpr SolidMotion kNoMotion = SolidMotion();
constexpr double kStepDuration = 0.01;  // s
constexpr double kCellWidth = 0.01;  // m

Solid BuildFlatSolid(Vec3 point_on_surface, SolidMotion motion, std::uint32_t solid_id) {
  return ToSolid(FlatSolid{.point_on_surface = point_on_surface}, motion, solid_id);
}

// The box of the scene-scale tests: centre {1, 2, 3} moved by offset, half sizes 0.5.
Solid BuildMovedBox(Vec3 offset) {
  const Box box = {
      .center = Vec3{.x = 1, .y = 2, .z = 3} + offset,
      .half_size = {.x = 0.5, .y = 0.5, .z = 0.5},
  };
  return ToSolid(box, kNoMotion, 1);
}

// --- The constants ---

TEST(ContactToleranceConstants, RelativeShare_IsFourThousandNinetySixEpsilons) {
  constexpr double kMachineEpsilon = 0x1p-52;
  EXPECT_EQ(kContactToleranceRelative, 4096 * kMachineEpsilon);
}

// --- ComputeContactTolerance ---

TEST(ComputeContactTolerance, SceneAtOrigin_IsTheCellWidthFloor) {
  const ContactScale scale = {.largest_coordinate = 0, .smallest_cell_width = 0.01, .thinnest_solid = 0.01};
  EXPECT_EQ(ComputeContactTolerance(scale), 0x1p-30 * 0.01);
}

TEST(ComputeContactTolerance, PowersOfTwo_AddBothSharesExactly) {
  // 2^-30 * 2^-7 + 2^-40 * 2^3 = 2^-37 + 2^-37 = 2^-36. A swapped pair of constants gives 2^-27 + 2^-47.
  const ContactScale scale = {.largest_coordinate = 8, .smallest_cell_width = 0x1p-7, .thinnest_solid = 0x1p-7};
  EXPECT_EQ(ComputeContactTolerance(scale), 0x1p-36);
}

TEST(ComputeContactTolerance, PaperExample_IsAboutEighteenPicometres) {
  // Method paper <sec-contact-tolerance>: largest coordinate 10 m, smallest cell 0.01 m:
  // tol = 9.31e-12 + 9.09e-12 = 1.84e-11 m.
  const ContactScale scale = {.largest_coordinate = 10, .smallest_cell_width = 0.01, .thinnest_solid = 0.01};
  EXPECT_NEAR(ComputeContactTolerance(scale), 1.84e-11, 0.005e-11);
}

TEST(ComputeContactTolerance, SceneMovedAMillionMeters_ScalesWithIt) {
  // The box's largest coordinate is its z, 3, so a move along z adds exactly 1e6 to it.
  const Vec3 move = {.x = 0, .y = 0, .z = 1e6};
  const std::array near_origin = {BuildMovedBox(Vec3{})};
  const std::array far_away = {BuildMovedBox(move)};
  const ContactScale near_scale = ComputeContactScale(near_origin.data(), 1, kStepDuration, kCellWidth);
  const ContactScale far_scale = ComputeContactScale(far_away.data(), 1, kStepDuration, kCellWidth);

  const double largest_coordinate_growth = far_scale.largest_coordinate - near_scale.largest_coordinate;
  EXPECT_NEAR(largest_coordinate_growth, 1e6, 1e-12 * 1e6);
  const double tolerance_growth = ComputeContactTolerance(far_scale) - ComputeContactTolerance(near_scale);
  const double expected_growth = 0x1p-40 * 1e6;
  EXPECT_NEAR(tolerance_growth, expected_growth, 1e-12 * expected_growth);
}

// --- ComputeContactScale ---

TEST(ComputeContactScale, BoxAndPlate_TakesThinnestAndLargest) {
  const Plate plate = {.half_thickness = 0.002};
  const Box box = {.center = {.x = 10, .y = 0, .z = 0}, .half_size = {.x = 1, .y = 2, .z = 3}};
  const std::array solids = {ToSolid(plate, kNoMotion, 1), ToSolid(box, kNoMotion, 2)};
  const ContactScale scale = ComputeContactScale(solids.data(), 2, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.thinnest_solid, 0.004);
  EXPECT_EQ(scale.largest_coordinate, 10 + math::Length(Vec3{.x = 1, .y = 2, .z = 3}));
  EXPECT_EQ(scale.smallest_cell_width, kCellWidth);
}

TEST(ComputeContactScale, MovingSolid_UsesEndOfStepOrigin) {
  const SolidMotion falling = {.linear_velocity = {.x = 0, .y = 0, .z = -5}};
  const std::array solids = {BuildFlatSolid(Vec3{}, falling, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, 2, kCellWidth);
  EXPECT_EQ(scale.largest_coordinate, 10);
}

TEST(ComputeContactScale, MovingBox_AddsReachToEndOfStepOrigin) {
  // From {0, 0, 0} to {0, 0, -10}, reach Length({1, 2, 2}) = 3: the reach counts at the far end, 10 + 3.
  const Box box = {.half_size = {.x = 1, .y = 2, .z = 2}};
  const SolidMotion falling = {.linear_velocity = {.x = 0, .y = 0, .z = -5}};
  const std::array solids = {ToSolid(box, falling, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, 2, kCellWidth);
  EXPECT_EQ(scale.largest_coordinate, 13);
}

TEST(ComputeContactScale, MovingSolid_UsesStartOfStepOrigin) {
  // From {0, -10, 0} to {0, -2, 0}: the start is the farther end, on the negative side.
  const SolidMotion rising = {.linear_velocity = {.x = 0, .y = 4, .z = 0}};
  const std::array solids = {BuildFlatSolid(Vec3{.x = 0, .y = -10, .z = 0}, rising, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, 2, kCellWidth);
  EXPECT_EQ(scale.largest_coordinate, 10);
}

TEST(ComputeContactScale, NoThinSolid_ThinnestIsCellWidth) {
  const std::array solids = {
      BuildFlatSolid(Vec3{.x = 0, .y = 0, .z = -1}, kNoMotion, 1),
      BuildFlatSolid(Vec3{.x = -4, .y = 0, .z = 0}, kNoMotion, 2),
  };
  const ContactScale scale = ComputeContactScale(solids.data(), 2, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.thinnest_solid, kCellWidth);
  EXPECT_EQ(scale.largest_coordinate, 4);
}

TEST(ComputeContactScale, NoSolids_IsCellWidthAtOrigin) {
  const ContactScale scale = ComputeContactScale(nullptr, 0, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.largest_coordinate, 0);
  EXPECT_EQ(scale.thinnest_solid, kCellWidth);
  EXPECT_EQ(scale.smallest_cell_width, kCellWidth);
}

TEST(ComputeContactScale, ThickSolid_ThinnestIsNotTheCellWidth) {
  // A box 2 m thick next to 0.01 m cells: the thinnest solid is the box, not the cell width.
  const Box box = {.half_size = {.x = 1, .y = 1, .z = 1}};
  const std::array solids = {ToSolid(box, kNoMotion, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.thinnest_solid, 2);
}

TEST(ComputeContactScale, Box_ThinnestIsTwiceSmallestHalfSize) {
  const Box box = {.half_size = {.x = 3, .y = 0.25, .z = 2}};
  const std::array solids = {ToSolid(box, kNoMotion, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.thinnest_solid, 0.5);
}

TEST(ComputeContactScale, BoxThinnestAlongZ_ThinnestIsTwiceThatHalfSize) {
  const Box box = {.half_size = {.x = 2, .y = 3, .z = 0.125}};
  const std::array solids = {ToSolid(box, kNoMotion, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.thinnest_solid, 0.25);
}

TEST(ComputeContactScale, Plate_AddsNoReach) {
  const Plate plate = {.center = {.x = 0, .y = 0, .z = 7}, .half_thickness = 0.5};
  const std::array solids = {ToSolid(plate, kNoMotion, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.largest_coordinate, 7);
  EXPECT_EQ(scale.thinnest_solid, 1);
}

TEST(ComputeContactScale, DomainBox_ReachesFromItsCentre) {
  // Corners {0, 0, 0} and {2, 4, 6}: centre {1, 2, 3}, half size {1, 2, 3}.
  const DomainBox domain_box = {.low_corner = {}, .high_corner = {.x = 2, .y = 4, .z = 6}};
  const std::array solids = {ToSolid(domain_box, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.largest_coordinate, 3 + math::Length(Vec3{.x = 1, .y = 2, .z = 3}));
}

TEST(ComputeContactScale, DomainBox_SmallestWidthIsThinnest) {
  // Corners {0, 0, 0} and {6, 4, 2}: the smallest width is along z, so a width read from x alone gives 6.
  const DomainBox domain_box = {.low_corner = {}, .high_corner = {.x = 6, .y = 4, .z = 2}};
  const std::array solids = {ToSolid(domain_box, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.thinnest_solid, 2);
}

TEST(ComputeContactScale, SphereAndCylinder_GiveTheirThinnestSize) {
  const Sphere sphere = {.center = {.x = 5, .y = 0, .z = 0}, .radius = 0.3};
  const Cylinder cylinder = {.radius = 1, .half_height = 0.2};
  const std::array solids = {ToSolid(sphere, kNoMotion, 1), ToSolid(cylinder, kNoMotion, 2)};
  const ContactScale scale = ComputeContactScale(solids.data(), 2, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.thinnest_solid, 0.4);
  // Each solid's own coordinate plus its own reach: 5 + 0.52 beats 0 + 1.43.
  EXPECT_EQ(scale.largest_coordinate, 5 + math::Length(Vec3{.x = 0.3, .y = 0.3, .z = 0.3}));
}

TEST(ComputeContactScale, Sphere_ThinnestIsDiameter) {
  const Sphere sphere = {.radius = 0.3};
  const std::array solids = {ToSolid(sphere, kNoMotion, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.thinnest_solid, 0.6);
}

TEST(ComputeContactScale, LongThinCylinder_ThinnestIsDiameter) {
  const Cylinder cylinder = {.radius = 0.1, .half_height = 2};
  const std::array solids = {ToSolid(cylinder, kNoMotion, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.thinnest_solid, 0.2);
  EXPECT_EQ(scale.largest_coordinate, math::Length(Vec3{.x = 0.1, .y = 0.1, .z = 2}));
}

TEST(ComputeContactScale, ThinnestSolidLast_IsKept) {
  // The thinnest solid comes after a thicker one, so a scale that kept only the first solid would say 2.
  const Box thick_box = {.half_size = {.x = 1, .y = 1, .z = 1}};
  const Plate thin_plate = {.half_thickness = 0.125};
  const std::array solids = {ToSolid(thick_box, kNoMotion, 1), ToSolid(thin_plate, kNoMotion, 2)};
  const ContactScale scale = ComputeContactScale(solids.data(), 2, kStepDuration, kCellWidth);
  EXPECT_EQ(scale.thinnest_solid, 0.25);
}

// --- CheckContactTolerance ---

constexpr double kTolerance = 0x1p-30;  // m

TEST(CheckContactTolerance, PlateThinnerThan1024Tolerances_IsSolidTooThin) {
  const ContactScale scale = {.largest_coordinate = 1, .smallest_cell_width = 1, .thinnest_solid = 1023 * kTolerance};
  EXPECT_EQ(CheckContactTolerance(scale, kTolerance), ContactError::kSolidTooThin);
}

TEST(CheckContactTolerance, Exactly1024Tolerances_IsNone) {
  const ContactScale scale = {.largest_coordinate = 1, .smallest_cell_width = 1, .thinnest_solid = 1024 * kTolerance};
  EXPECT_EQ(CheckContactTolerance(scale, kTolerance), ContactError::kNone);
}

TEST(CheckContactTolerance, CellNarrowerThan1024Tolerances_IsSolidTooThin) {
  const ContactScale scale = {.largest_coordinate = 1, .smallest_cell_width = 1023 * kTolerance, .thinnest_solid = 1};
  EXPECT_EQ(CheckContactTolerance(scale, kTolerance), ContactError::kSolidTooThin);
}

TEST(CheckContactTolerance, CellExactly1024Tolerances_IsNone) {
  const ContactScale scale = {.largest_coordinate = 1, .smallest_cell_width = 1024 * kTolerance, .thinnest_solid = 1};
  EXPECT_EQ(CheckContactTolerance(scale, kTolerance), ContactError::kNone);
}

TEST(CheckContactTolerance, PlateJustBelow1024Tolerances_IsSolidTooThin) {
  // One rounding unit below the limit.
  const double limit = 1024 * kTolerance;
  const ContactScale scale = {
      .largest_coordinate = 1,
      .smallest_cell_width = 1,
      .thinnest_solid = std::nextafter(limit, 0.0),
  };
  EXPECT_EQ(CheckContactTolerance(scale, kTolerance), ContactError::kSolidTooThin);
}

TEST(CheckContactTolerance, SceneFromComputeContactScale_IsNone) {
  // The paper's scene: a plate 0.004 m thick and a box at {10, 0, 0}; the tolerance is about 2e-11 m.
  const Plate plate = {.half_thickness = 0.002};
  const Box box = {.center = {.x = 10, .y = 0, .z = 0}, .half_size = {.x = 1, .y = 2, .z = 3}};
  const std::array solids = {ToSolid(plate, kNoMotion, 1), ToSolid(box, kNoMotion, 2)};
  const ContactScale scale = ComputeContactScale(solids.data(), 2, kStepDuration, kCellWidth);
  EXPECT_EQ(CheckContactTolerance(scale, ComputeContactTolerance(scale)), ContactError::kNone);
}

TEST(CheckContactTolerance, PlateTenBillionTimesThinnerThanItsDistance_IsSolidTooThin) {
  // A plate 1e-6 m thick at 1e4 m from the origin: 1024 * 2^-40 * 1e4 = 9.3e-6 m > 1e-6 m.
  const Plate plate = {
      .center = {.x = 1e4, .y = 0, .z = 0},
      .normal = {.x = 1, .y = 0, .z = 0},
      .half_thickness = 5e-7,
  };
  const std::array solids = {ToSolid(plate, kNoMotion, 1)};
  const ContactScale scale = ComputeContactScale(solids.data(), 1, kStepDuration, kCellWidth);
  EXPECT_EQ(CheckContactTolerance(scale, ComputeContactTolerance(scale)), ContactError::kSolidTooThin);
}

TEST(CheckContactTolerance, NaNTolerance_IsSolidTooThin) {
  // Not constexpr: the comparison with a NaN must happen when the test runs.
  const double not_a_number = std::numeric_limits<double>::quiet_NaN();
  const ContactScale scale = {.largest_coordinate = 1, .smallest_cell_width = 1, .thinnest_solid = 1};
  EXPECT_EQ(CheckContactTolerance(scale, not_a_number), ContactError::kSolidTooThin);
}

TEST(CheckContactTolerance, NaNThinnestSolid_IsSolidTooThin) {
  const double not_a_number = std::numeric_limits<double>::quiet_NaN();
  const ContactScale scale = {.largest_coordinate = 1, .smallest_cell_width = 1, .thinnest_solid = not_a_number};
  EXPECT_EQ(CheckContactTolerance(scale, kTolerance), ContactError::kSolidTooThin);
}

}  // namespace
}  // namespace fluidity::contact
