// The cell size rule: how wide one cell should be.
//
// Purpose: give the step (its stage that chooses active cells) and cell resize one set of per-cell formulas for the
//   wanted cell width, the wanted size level and the reference speed, on the CPU or the GPU. The passes that apply
//   them to every cell belong to those modules.
//
// Idea: cells stay smallest near the liquid surface, near a solid and where the velocity changes over a short
//   distance; elsewhere they grow in proportion to the distance from the surface and from the solids. With w_min the
//   width of the smallest size level the settings allow and w_max the width of size level 0, the wanted cell width is
//     w_depth = w_min * Max(1, depth / (surface_reach_in_cell_widths * w_min))
//     w_solid = w_min * Max(1, distance / (solid_reach_in_cell_widths * w_min))
//     w_shear = shear_width_factor * reference_speed / velocity_change_size, or +infinity when that size is 0 or below
//     wanted  = Max(w_min, Min(Min(Min(w_depth, w_solid), w_shear) * cell_size_factor, w_max))
//   It is w_min at once, whatever the factor, for a cell that touches an air pocket, a cell with gas in or beside
//   it, a cell within the surface reach (less than surface_reach_in_cell_widths smallest widths below the surface),
//   a cell within the solid reach (less than solid_reach_in_cell_widths smallest widths from the nearest solid), and
//   a cell with a NaN fact or a NaN reference speed.
//   The wanted size level is the first size level, from 0 up, whose width is at most the wanted cell width.
//   Example: biggest cells 1 m wide, 5 size levels, so w_min = 0.0625 m, the default surface reach of 8 and solid
//   reach of 2 smallest widths. A depth of 1 m gives 0.0625 * 1 / 0.5 = 0.125 m. A distance of 0.5 m to a solid
//   gives 0.0625 * 0.5 / 0.125 = 0.25 m. A cell with both gets the smaller, 0.125 m, which is size level 3. A
//   distance of 0.1 m is within the solid reach of 0.125 m, so it gives 0.0625 m whatever the factor.
//   Precision. The widths come from ComputeCellWidth and are exact. Every other step is one IEEE 754 product,
//   quotient or comparison, Min or Max (which round nothing) or one correctly rounded square root, in the fixed
//   order written in each function. The formulas hold no sum, so no fused multiply-add can arise, whatever the
//   compiler's contraction setting. The CPU and the GPU therefore give the same bits, and a size level comes from
//   comparing widths, never from a logarithm or a power, whose last bits differ between the processors' libraries.
//   NaN and infinity. A NaN fact or a NaN reference speed gives w_min, tested with the early returns, before any
//   term. Inside, a NaN can still arise from inputs outside the precondition in two ways: +infinity / +infinity in the
//   shear term (a velocity change size and a reference speed both +infinity), and 0 * infinity in the product with the
//   factor (a factor of +0 or -0 with every term +infinity, or a factor of +infinity with a shear term of 0). The rule
//   tests the shear term for NaN, and its last clamp keeps w_min unless the scaled width is above it, so either NaN
//   also gives w_min. The wanted width is never NaN.
//
// Words:
//   cell size rule      the formulas of this file, which say how wide a cell should be
//   wanted cell width   m. How wide one cell should be by the cell size rule
//   wanted size level   the first size level, from 0 up, whose width is at most the wanted cell width
//   reference speed     m/s. The speed the shear term measures a velocity change against
//   cell size factor    no unit, positive. The artist's scale on the wanted cell width, from a volume of factors
//   velocity change size  1/s. How fast the velocity changes per metre around a cell
//   surface reach       no unit. surface_reach_in_cell_widths: how many smallest cell widths below the surface keep
//                       the smallest width. Not the reach of a solid
//   solid reach         no unit. solid_reach_in_cell_widths: how many smallest cell widths from the nearest solid keep
//                       the smallest width. Not the reach of a solid
// Owns:    CellSizeSettings, CellWidthsBySizeLevel, CellSizeFacts, ComputeTableSizeLevelCount,
//          ComputeCellWidthsBySizeLevel, ComputeReferenceSpeed, ComputeWantedCellWidth, ComputeWantedSizeLevel
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), math::Sqrt, math::Min, math::Max, math::IsNan,
//          math::kPositiveInfinity (base/scalar_math.h), SizeLevel, kMaxSizeLevels (grid/grid_ids.h), GridFrame,
//          ComputeCellWidth (grid/grid_tables.h)
// Used by: the step's ChooseActiveCells stage (step/) and ResizeCells (cell_resize/)
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-size-rule>, <eq:wanted-cell-width>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// The scene's choices for the cell size rule, chosen once for a run.
struct CellSizeSettings {
  // The size level of the smallest cells the rule asks for, usually the frame's last size level: the width w_min of
  // the cells at air pockets and gas, within the surface reach and within the solid reach. The default is 0, the
  // biggest cells, until a run sets it: left at 0 it makes every cell as wide as the biggest. At or beyond the
  // frame's size_level_count it reads as the frame's last size level (ComputeWantedCellWidth).
  SizeLevel smallest_size_level{};
  // No unit, positive: the surface reach, counted in smallest cell widths (not the reach of a solid). Cells less than
  // this many smallest widths below the surface get the smallest width; beyond, the depth term grows in proportion to
  // the depth. 8 by the method paper.
  double surface_reach_in_cell_widths = 8.0;
  // No unit, positive: the solid reach, counted in smallest cell widths (not the reach of a solid). Cells less than
  // this many smallest widths from the nearest solid get the smallest width whatever the factor; beyond, the solid
  // term grows in proportion to the distance. 2 by the method paper.
  double solid_reach_in_cell_widths = 2.0;
  // No unit, positive. c_u of the shear term, 0.25 by the method paper: a starting value, to be calibrated against
  // scenes.
  double shear_width_factor = 0.25;
};

// The width of every size level of one frame, in m, so that the cell size rule compares widths instead of computing
// a size level from a logarithm. 136 bytes: shared functions take it by const reference, as they take GridTables.
struct CellWidthsBySizeLevel {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  double width[kMaxSizeLevels] = {};  // m. width[l] is h_l for l < size_level_count, and 0 beyond.
  // 1..16, the frame's size_level_count. Readers clamp any other value to 1..16 (ComputeTableSizeLevelCount).
  std::uint8_t size_level_count = 0;
};

// What one cell knows that the cell size rule reads. A default value says nothing is near and nothing is asked for.
struct CellSizeFacts {
  // m, zero or more, or +infinity when no surface is near. How far the cell lies below the liquid surface. Negative
  // counts as the surface.
  double depth_below_surface = math::kPositiveInfinity;
  // m, zero or more, or +infinity when no solid is near. How far the cell lies from the nearest solid. Negative counts
  // as the solid.
  double distance_to_solid = math::kPositiveInfinity;
  // 1/s, zero or more and finite. How fast the velocity changes per metre around the cell. Zero or below gives no
  // shear term.
  double velocity_change_size = 0.0;
  // No unit, positive and finite. The artist's scale on the wanted cell width; 1 leaves the rule as it is.
  double cell_size_factor = 1.0;
  bool touches_air_pocket = false;  // Yes: the cell touches an air pocket, so it gets the smallest width.
  bool wants_smallest_cells = false;  // Yes: gas lies in or beside the cell, so it gets the smallest width.
  // Yes: the cell holds particles whose count must stay as it is. Read by the merge test of ResizeCells, not here.
  bool holds_particles_that_keep_their_count = false;
  // Yes: the cell touches air, gas or a solid. Read by the merge test of ResizeCells, not here.
  bool touches_air_gas_or_solid = false;
};

// Computes how many size levels a table of widths holds for a given count: the count clamped to 1..kMaxSizeLevels.
// The readers of the table index it only below this count, so no given count reads or writes past the array.
//
// Example: 5 gives 5; 0 gives 1; 20 gives 16.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-size-rule>
FLUIDITY_CPU_AND_GPU constexpr int ComputeTableSizeLevelCount(int size_level_count) noexcept {
  if (size_level_count < 1) {
    return 1;
  }
  if (size_level_count > kMaxSizeLevels) {
    return kMaxSizeLevels;
  }
  return size_level_count;
}

// Computes the table of widths of a frame: width[l] = ComputeCellWidth(frame, l) for l below size_level_count, 0
// beyond. This is the one place the table is filled.
//
// Method: ComputeCellWidth for each size level of the frame. A frame that passes the frame check keeps its smallest
//   width a normal double, so every width is an exact halving of biggest_cell_width.
// Units: m.
// Example: biggest cells 1 m wide and 5 size levels give 1, 0.5, 0.25, 0.125 and 0.0625 m, then 0 for size levels 5
//   to 15, and size_level_count 5.
// Requires: a frame that passes the frame check, with 1..kMaxSizeLevels size levels. Any other count is clamped by
//   ComputeTableSizeLevelCount, so the table is never written past its end: a count of 0 gives the one width h_0 and
//   a count of 1, a count above 16 gives 16 widths and a count of 16.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>, <sec-cell-size-rule>
FLUIDITY_CPU_AND_GPU constexpr CellWidthsBySizeLevel ComputeCellWidthsBySizeLevel(GridFrame frame) noexcept {
  const int size_level_count = ComputeTableSizeLevelCount(frame.size_level_count);
  CellWidthsBySizeLevel widths{};
  widths.size_level_count = static_cast<std::uint8_t>(size_level_count);
  for (int size_level = 0; size_level < size_level_count; ++size_level) {
    widths.width[size_level] = ComputeCellWidth(frame, static_cast<SizeLevel>(size_level));
  }
  return widths;
}

// Computes the reference speed U_ref = Max(largest_liquid_speed, Sqrt(gravity_size * biggest_cell_width)): the larger
// of the fastest liquid and a gravity wave in liquid as deep as the biggest cell is wide.
//
// Method: one product, one correctly rounded square root and one Max, in that order.
// Units: largest_liquid_speed in m/s, gravity_size in m/s^2, biggest_cell_width in m; the result in m/s.
// Example: (0.1 m/s, 9.81 m/s^2, 1 m) gives Sqrt(9.81) = 3.132 m/s; (5 m/s, 9.81 m/s^2, 1 m) gives 5 m/s.
// Requires: largest_liquid_speed zero or more, finite or +infinity; gravity_size zero or more and finite;
//   biggest_cell_width positive and finite. Outside, Max drops one NaN: a NaN speed, or a negative product whose square
//   root is NaN, gives the other value, and the result is NaN only when both are.
// No floor: with zero gravity and a liquid at rest the reference speed is 0. The shear term of ComputeWantedCellWidth
//   is then 0 for any velocity change size above 0, so those cells get the smallest width; a velocity change size of
//   0 still gives no shear term.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-size-rule>
//
// clang-tidy: three doubles of three different units; the swappable-parameters check flags them.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU inline double
ComputeReferenceSpeed(double largest_liquid_speed, double gravity_size, double biggest_cell_width) noexcept {
  const double gravity_times_width = gravity_size * biggest_cell_width;
  const double gravity_wave_speed = math::Sqrt(gravity_times_width);
  return math::Max(largest_liquid_speed, gravity_wave_speed);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Computes the wanted cell width of one cell, in m, by the cell size rule.
//
// Method: <eq:wanted-cell-width>, in this order, so both processors round alike.
//   w_min = widths.width[settings.smallest_size_level] and w_max = widths.width[0]. The result is w_min at once,
//   whatever the factor, where a fact or the reference speed is NaN, the cell touches an air pocket, gas lies in or
//   beside it, it lies less than the surface reach (surface_reach_in_cell_widths * w_min) below the surface, or less
//   than the solid reach (solid_reach_in_cell_widths * w_min) from the nearest solid. Otherwise
//     w_depth = w_min * Max(1, depth / (surface_reach_in_cell_widths * w_min))
//     w_solid = w_min * Max(1, distance / (solid_reach_in_cell_widths * w_min))
//     w_shear = (shear_width_factor * reference_speed) / velocity_change_size if that size is above 0, else +infinity
//     wanted  = Max(w_min, Min(Min(Min(w_depth, w_solid), w_shear) * cell_size_factor, w_max))
//   The last line is computed as w_min unless the scaled width is above w_min, else Min(scaled width, w_max): the
//   same bits for every number, and w_min for a NaN.
// Units: m for the depth, the distance, the widths and the result; 1/s for the velocity change size; m/s for the
//   reference speed. The facts are those of one moment of the step, as the caller computed them.
// Example: biggest cells 1 m wide, 5 size levels, smallest size level 4 (w_min = 0.0625 m), default settings and
//   factor 1. Depth 1 m: 0.0625 * Max(1, 1 / 0.5) = 0.125 m. Distance 0.5 m: 0.0625 * Max(1, 0.5 / 0.125) = 0.25 m.
//   Both: 0.125 m. Depth 0.25 m or distance 0.1 m, within the surface reach or the solid reach: 0.0625 m, whatever
//   the factor. Depth 100 m: 12.5 m, clamped to 1 m.
// Requires: widths from ComputeCellWidthsBySizeLevel; settings.smallest_size_level below widths.size_level_count, so
//   w_min is positive and at most w_max. A smallest_size_level at or beyond that count reads as the table's last size
//   level, so the table is never read past its end. The surface reach, the solid reach and shear_width_factor
//   positive and finite (asserted). Each fact as CellSizeFacts says: depth and distance finite or +infinity, a
//   negative one counting as the surface or the solid; velocity_change_size and reference_speed zero or more and
//   finite; cell_size_factor positive and finite.
//   Outside, the result is still between w_min and w_max and never NaN: a NaN input, a NaN that arises inside (see
//   the file header) and a factor of 0 or below all give w_min; a factor of +infinity otherwise gives w_max; a
//   reference speed of +infinity gives no shear term; a velocity change size of +infinity gives a shear term of 0,
//   so w_min.
// Read: docs/method-paper/fluidity-method.typ <eq:wanted-cell-width>, <sec-cell-size-rule>
FLUIDITY_CPU_AND_GPU constexpr double ComputeWantedCellWidth(
    CellSizeFacts facts,
    double reference_speed,
    const CellWidthsBySizeLevel& widths,
    CellSizeSettings settings
) noexcept {
  // Steps: 1 w_min and w_max  2 the cells kept smallest  3 the three terms  4 the factor and the clamp
  // Step 1: a smallest size level past the table reads its last size level, so no index leaves the array.
  const int last_size_level = ComputeTableSizeLevelCount(widths.size_level_count) - 1;
  const int asked_smallest_size_level = static_cast<int>(settings.smallest_size_level);
  const bool is_within_table = asked_smallest_size_level <= last_size_level;
  const int smallest_size_level = is_within_table ? asked_smallest_size_level : last_size_level;
  assert(settings.surface_reach_in_cell_widths > 0.0);
  assert(settings.solid_reach_in_cell_widths > 0.0);
  assert(settings.shear_width_factor > 0.0);
  const double smallest_width = widths.width[smallest_size_level];
  const double biggest_width = widths.width[0];

  // Step 2: the cells kept smallest, whatever the factor. A NaN input errs toward detail. A negative depth or
  // distance counts as the surface or the solid.
  const bool has_nan_input = math::IsNan(facts.depth_below_surface) || math::IsNan(facts.distance_to_solid)
      || math::IsNan(facts.velocity_change_size) || math::IsNan(facts.cell_size_factor) || math::IsNan(reference_speed);
  const bool has_smallest_flag = facts.touches_air_pocket || facts.wants_smallest_cells;
  const double surface_reach = settings.surface_reach_in_cell_widths * smallest_width;
  const bool is_within_surface_reach = facts.depth_below_surface < surface_reach;
  const double solid_reach = settings.solid_reach_in_cell_widths * smallest_width;
  const bool is_within_solid_reach = facts.distance_to_solid < solid_reach;
  if (has_nan_input || has_smallest_flag || is_within_surface_reach || is_within_solid_reach) {
    return smallest_width;
  }

  // Step 3: each term grows in proportion to its distance beyond the surface reach or the solid reach; the quotient
  // is at least 1 here, and an infinite distance gives an infinite term. A velocity change size of 0 means the
  // liquid moves as a whole there: no shear term.
  const double depth_in_surface_reaches = facts.depth_below_surface / surface_reach;
  const double depth_width = smallest_width * math::Max(1.0, depth_in_surface_reaches);
  const double distance_in_solid_reaches = facts.distance_to_solid / solid_reach;
  const double solid_width = smallest_width * math::Max(1.0, distance_in_solid_reaches);
  double shear_width = math::kPositiveInfinity;
  if (facts.velocity_change_size > 0.0) {
    const double scaled_reference_speed = settings.shear_width_factor * reference_speed;
    shear_width = scaled_reference_speed / facts.velocity_change_size;
  }
  // A NaN here comes only from inputs or settings outside the precondition, such as +infinity / +infinity, a
  // shear_width_factor of +infinity times a reference speed of 0, or a NaN shear_width_factor; Min would drop it.
  if (math::IsNan(shear_width)) {
    return smallest_width;
  }

  // Step 4: the factor scales the physical width; the clamp comes last, so its bounds hold whatever the factor. A
  // scaled width that is not above w_min, a NaN product of 0 and infinity included, gives w_min.
  const double depth_or_solid_width = math::Min(depth_width, solid_width);
  const double physical_width = math::Min(depth_or_solid_width, shear_width);
  const double scaled_width = physical_width * facts.cell_size_factor;
  const bool is_above_smallest_width = scaled_width > smallest_width;
  if (!is_above_smallest_width) {
    return smallest_width;
  }
  return math::Min(scaled_width, biggest_width);
}

// Computes the wanted size level of a cell: the first size level, from 0 up, whose width is at most wanted_width, and
// the frame's last size level when none is.
//
// Method: walks the table of widths from size level 0 and compares; no logarithm, no power.
// Ties: a wanted width equal to a size level's width gives that size level.
// Example: in a frame of biggest cells 1 m wide and 5 size levels, 0.125 m gives 3, the double just below 0.125 gives
//   4, 0.2 m gives 3, 5 m gives 0 and 0.01 m gives 4.
// Requires: widths from ComputeCellWidthsBySizeLevel. A size_level_count outside 1..16 is clamped by
//   ComputeTableSizeLevelCount, so the walk never reads past the table. A NaN wanted_width compares false everywhere
//   and gives the last size level; +infinity gives 0.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-size-rule>
FLUIDITY_CPU_AND_GPU constexpr SizeLevel
ComputeWantedSizeLevel(double wanted_width, const CellWidthsBySizeLevel& widths) noexcept {
  const int last_size_level = ComputeTableSizeLevelCount(widths.size_level_count) - 1;
  for (int size_level = 0; size_level < last_size_level; ++size_level) {
    if (widths.width[size_level] <= wanted_width) {
      return static_cast<SizeLevel>(size_level);
    }
  }
  return static_cast<SizeLevel>(last_size_level);
}

}  // namespace fluidity::grid
