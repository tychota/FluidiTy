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
//   A new cell, added where liquid enters space with no cell, starts one size level smaller than the nearest committed
//   cell, and never smaller than the smallest size level. That cell is the nearest among the active cells that hold 27
//   searched positions one biggest cell width apart. It may not touch the new cell, and a nearer, smaller cell can lie
//   between them, so the new cell can break the size step rule with a cell it touches. The step that adds cells
//   therefore closes the size step rule over its new list; the cell list check turns down a list that breaks it.
//   Example: when the searched position one biggest width to the +X finds a size-level-0 cell, the new cell starts at
//   size level 1.
//
// Words:
//   cell size rule      the formulas of this file, which say how wide a cell should be
//   wanted cell width   m. How wide one cell should be by the cell size rule
//   wanted size level   the first size level, from 0 up, whose width is at most the wanted cell width
//   reference speed     m/s. The speed the shear term measures a velocity change against
//   cell size factor    no unit, positive. The artist's scale on the wanted cell width, from a volume of factors
//   cell size factor code  an integer from -24 to 24: the factor is 2^(code / 4), one quarter doubling per step
//   refinement volume   the 3D array of cell size factor codes over the scene, stored as layers of halved size;
//                       one stored code is an entry of the volume
//   velocity change size  1/s. How fast the velocity changes per metre around a cell
//   surface reach       no unit. surface_reach_in_cell_widths: how many smallest cell widths below the surface keep
//                       the smallest width. Not the reach of a solid
//   solid reach         no unit. solid_reach_in_cell_widths: how many smallest cell widths from the nearest solid keep
//                       the smallest width. Not the reach of a solid
//   nearest committed cell  the active cell of the committed grid, among those that hold the 27 searched positions,
//                       whose centre is nearest a position. It may not touch the position
//   searched position   one of the 27 positions position + (i, j, k) * biggest_cell_width, i, j, k in {-1, 0, 1}
// Owns:    CellSizeSettings, CellWidthsBySizeLevel, CellSizeFacts, ComputeTableSizeLevelCount,
//          ComputeCellWidthsBySizeLevel, ComputeReferenceSpeed, ComputeWantedCellWidth, ComputeWantedSizeLevel,
//          kCellSizeFactorStepsPerDoubling, kMaxCellSizeFactorCode, CellSizeFactorTable,
//          ComputeCellSizeFactorTable, RefinementFactors, ComputeCellSizeFactorCode, ComputeRefinementLayerExtent,
//          ComputeRefinementEntryIndex, ComputeRefinementBoxReachesOutside,
//          FindNearestCommittedCell, ComputeNewCellSizeLevel
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), math::Sqrt, math::Min, math::Max, math::IsNan,
//          math::kPositiveInfinity (base/scalar_math.h), Vec3, math::Dot (base/vec3.h), SizeLevel, CellId,
//          CellCoord, CellPlaceIndex, kNoIndex, kMaxSizeLevels, ToSizeLevel, ToCellCoord, ComesBefore,
//          ToCellBlockIndex (grid/grid_ids.h), GridFrame, GridTables, ComputeCellWidth, ToCellId, ToSizeLevel (of a
//          block) (grid/grid_tables.h), FindCellAt (grid/find_cell_at.h), math::IsFinite (base/scalar_math.h)
// Used by: the step's ChooseActiveCells stage (step/) and ResizeCells (cell_resize/)
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-size-rule>, <eq:wanted-cell-width>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/find_cell_at.h"
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
  // No unit, positive. The scale of the shear term (papers: c_u), 0.25 by the method paper: a starting value, to be
  // calibrated against scenes.
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

// How many steps of the cell size factor code make one doubling of the factor: 4, so the factor of code c is
// 2^(c / 4), and a step is a ratio of about 1.19.
inline constexpr int kCellSizeFactorStepsPerDoubling = 4;
// The largest code magnitude: codes run from -kMaxCellSizeFactorCode to kMaxCellSizeFactorCode, 6 doublings each
// way, so factors run from 2^-6 = 1/64 to 2^6 = 64.
inline constexpr int kMaxCellSizeFactorCode = 24;

// The 49 cell size factors, one per code. No unit: pure numbers. Index a code with factor[code +
// kMaxCellSizeFactorCode], for a code from -24 to 24.
struct CellSizeFactorTable {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  double factor[2 * kMaxCellSizeFactorCode + 1] = {};  // factor[i] is 2^((i - 24) / 4), the nearest double.
};

// Computes the table of cell size factors: factor[i] = 2^(i / 4 - 6) * root[i % 4] for the index i = 0..48 (integer
// division), with root = {1, 2^(1/4), 2^(1/2), 2^(3/4)} each written as the hexadecimal literal of the double nearest
// the real root. So factor[24] = 1 for code 0, factor[28] = 2, factor[20] = 0.5 and factor[26] is the double nearest
// the square root of 2, which is math::Sqrt(2.0).
//
// Method: the power of two starts at 2^-6 and is doubled before every fourth factor; a double times a power of two is
//   exact while the result stays in range, here 2^-6 to 2^6. Each root is a literal, so no compiler rounds a decimal.
//   Every factor is therefore one fixed bit pattern on every machine and on both processors, and the function calls no
//   library function, so it is usable in a static_assert.
// Example: factor[25] = 0x1.306fe0a31b715p+0 = 1.189207115002721 (code 1); factor[1] = 0x1.306fe0a31b715p-6;
//   factor[48] = 64.
// Units: none.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-size-rule>
FLUIDITY_CPU_AND_GPU constexpr CellSizeFactorTable ComputeCellSizeFactorTable() noexcept {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): see CellSizeFactorTable.
  constexpr double kQuarterRoots[kCellSizeFactorStepsPerDoubling] = {
      1.0,
      0x1.306fe0a31b715p+0,
      // NOLINTNEXTLINE(modernize-use-std-numbers): the literal is the specification, equal to std::numbers::sqrt2.
      0x1.6a09e667f3bcdp+0,
      0x1.ae89f995ad3adp+0,
  };
  CellSizeFactorTable table{};
  double power_of_two = 0x1p-6;
  for (int factor_index = 0; factor_index <= 2 * kMaxCellSizeFactorCode; ++factor_index) {
    const int root_index = factor_index % kCellSizeFactorStepsPerDoubling;
    if (factor_index > 0 && root_index == 0) {
      power_of_two = power_of_two * 2.0;
    }
    table.factor[factor_index] = power_of_two * kQuarterRoots[root_index];
  }
  return table;
}

// The refinement volume as the lookup reads it: layer 0 holds one code per entry over the scene, and each layer above
// holds the smallest code over 2 x 2 x 2 entries of the layer below. Plain pointers and counts, so device code reads
// it as it reads GridTables; the caller owns the arrays, which must outlive every lookup. Small enough that shared
// functions take it by const reference.
struct RefinementFactors {
  // Entries of layer 0 per axis; layer L has ceil(count / 2^L) per axis. The volume is usable with every count 1 to
  // 2^20; any count outside that, count_x == 0 (no volume) included, makes every lookup return 0.
  std::int32_t count_x = 0;
  std::int32_t count_y = 0;
  std::int32_t count_z = 0;
  // The distance between neighbouring entries of layer 0, as log2 of a count of smallest cells: 2 means one entry per
  // 4 smallest cells on each axis. Valid from 0 to 31; any other value makes every lookup return 0.
  std::int32_t entry_spacing_log2 = 0;
  // The layers that hold data, 1 to kMaxSizeLevels. Zero or less makes every lookup return 0; more than
  // kMaxSizeLevels reads as kMaxSizeLevels.
  std::int32_t layer_count = 0;
  // layer_codes[L] holds the entries of layer L below layer_count, x fastest, then y, then z: the entry (x, y, z) is
  // at (z * layer_extent_y + y) * layer_extent_x + x, where layer_extent_a is ceil(count_a / 2^L). Each array must
  // hold exactly the layer_extent_x * layer_extent_y * layer_extent_z entries its counts say; the lookup cannot check
  // that. A code is any int8_t; the lookup clamps it. A null layer reads as 0.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  const std::int8_t* layer_codes[kMaxSizeLevels] = {};
};

// Computes how many entries a layer of a refinement volume holds along one axis: ceil(count / 2^layer), where count is
// the entries of layer 0 on that axis. Private to ComputeCellSizeFactorCode.
//
// Example: count 5 gives 5, 3, 2 and 1 entries in layers 0, 1, 2 and 3.
// Requires: count 1..2^20 and layer 0..kMaxSizeLevels - 1, so the sum below fits in 64 bits.
FLUIDITY_CPU_AND_GPU constexpr std::uint64_t ComputeRefinementLayerExtent(std::int32_t count, unsigned layer) noexcept {
  const std::uint64_t layer_unit = std::uint64_t{1} << layer;
  return (static_cast<std::uint64_t>(count) + layer_unit - 1U) >> layer;
}

// Computes where a cell lies along one axis of a refinement volume, as an entry index of the layer the cell reads, or
// -1 when it lies outside that layer. Private to ComputeCellSizeFactorCode, whose comment explains cell_width_log2 and
// the layer.
//
// Method: the index is cell_coord when layer > 0, and (cell_coord << cell_width_log2) >> spacing in layer 0, done in
//   64-bit unsigned arithmetic after a negative coordinate has been turned away, so no shift is of a negative value.
//   It lies inside the layer when it is below ComputeRefinementLayerExtent.
// Requires: cell_width_log2 and spacing 0..31 with cell_width_log2 <= spacing in layer 0; layer 0..kMaxSizeLevels - 1;
//   count 1..2^20. With cell_coord below 2^19, cell_coord << cell_width_log2 stays below 2^50, so the shift loses
//   no bit.
// clang-tidy: four counts of different meaning; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU constexpr std::int64_t ComputeRefinementEntryIndex(
    std::int32_t cell_coord,
    unsigned cell_width_log2,
    unsigned spacing,
    unsigned layer,
    std::int32_t count
) noexcept {
  if (cell_coord < 0) {
    return -1;
  }
  const auto unsigned_coord = static_cast<std::uint64_t>(cell_coord);
  std::uint64_t index = unsigned_coord;
  if (layer == 0) {
    index = (unsigned_coord << cell_width_log2) >> spacing;
  }
  if (index >= ComputeRefinementLayerExtent(count, layer)) {
    return -1;
  }
  return static_cast<std::int64_t>(index);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Tells whether the box of the entry at `index` of layer `layer` reaches past the last entry of layer 0 on one axis:
// the entry covers layer-0 entries index * 2^layer up to (index + 1) * 2^layer - 1, so it reaches outside when
// (index + 1) * 2^layer > count. Private to ComputeCellSizeFactorCode. It is false in layer 0, where an entry of the
// volume is its own box.
//
// Example: count 5, layer 1: index 2 covers layer-0 entries 4 and 5, so it reaches outside; index 1 covers 2 and 3.
// Requires: index below ComputeRefinementLayerExtent(count, layer); layer 0..kMaxSizeLevels - 1; count 1..2^20. Then
//   (index + 1) << layer is at most 2^35, so the shift loses no bit.
// clang-tidy: an index, a layer and a count; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeRefinementBoxReachesOutside(std::int64_t index, unsigned layer, std::int32_t count) noexcept {
  const std::uint64_t box_end = static_cast<std::uint64_t>(index + 1) << layer;
  const auto unsigned_count = static_cast<std::uint64_t>(count);
  return box_end > unsigned_count;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Computes the cell size factor code of one cell: the smallest code the refinement volume holds over the cell's box,
// where outside the volume the code is 0 (factor 1), clamped to -kMaxCellSizeFactorCode..kMaxCellSizeFactorCode. A
// cell wholly outside the volume gives 0, and a cell that reaches outside on any axis gives the smaller of the codes
// inside and 0, so a positive code never enlarges a cell that straddles the volume's edge. A caller turns it into a
// factor with ComputeCellSizeFactorTable().factor[code + kMaxCellSizeFactorCode] and passes that as
// CellSizeFacts::cell_size_factor.
//
// Method: integer only, so every machine reads the same entry. cell_width_log2 (s in the method paper) is
//   smallest_size_level minus the cell's size level: the cell is 2^s smallest cells wide. coord is the cell's
//   coordinate in cells of its own size level, and spacing is entry_spacing_log2. An entry of layer L is as wide as
//   2^s smallest cells when L = s - spacing.
//   When s <= spacing, the cell lies inside one entry of layer 0: the one at (coord << s) >> spacing per axis. That
//   entry is its own box, so the cell cannot reach outside the volume. Otherwise it reads layer L = s - spacing at
//   index coord: an entry of that layer is the smallest code over the layer-0 entries coord * 2^L up to
//   (coord + 1) * 2^L - 1 that exist. If (coord + 1) * 2^L is more than count on an axis, the cell reaches outside
//   the volume there, and the code is the smaller of that entry and 0.
//   Every index is checked against the layer's count ceil(count / 2^L) before the read: a negative coordinate or an
//   index at or beyond the count gives 0. A cell whose layer is not stored (L at or beyond layer_count, or a null
//   layer) gives 0, as does no volume. The layers hold the smallest code over the entries that exist, so the lookup,
//   not the layers, accounts for the outside.
//   The shifts are done in unsigned 64 bits, after a negative coordinate has been turned away: a coordinate is 32 bits,
//   and below 2^19 for a cell of a grid, and a shift of layer 0 is by s <= spacing <= 31, so coord << s stays below
//   2^50 and no shift is of a negative value, by a negative amount or by 64 or more. Layer shifts are by at most
//   kMaxSizeLevels - 1.
// Units: none: a code is a count of quarter doublings of the factor.
// Example: in the fixture of the tests (biggest cells 1 m, 5 size levels, smallest_size_level 4, entry_spacing_log2 2,
//   2 x 1 x 1 size-level-0 cells, so layer 0 is 8 x 4 x 4 entries, layers 1 and 2 are 4 x 2 x 2 and 2 x 1 x 1, and
//   code -8 on x = 5 of layer 0, 0 elsewhere): the size-level-4 cell {20, 3, 3} has s = 0 and reads entry 5 of layer
//   0, giving -8; {19, 3, 3} reads entry 4, giving 0. The size-level-0 cell {1, 0, 0} has s = 4, L = 2, and reads
//   layer 2 at index 1: the smallest code over x = 4..7, so -8; {0, 0, 0} gives 0. The size-level-2 cell {5, 0, 0}
//   has s = 2 and reads entry 5 of layer 0, giving -8. The size-level-4 cell {40, 0, 0} lies beyond 8 entries, so 0.
//   With count_x 5 instead of 8, spacing 0 and code +3 on every entry, the size-level-3 cell {2, 0, 0} (s = 1) reads
//   layer 1 at index 2, which covers layer-0 entries 4 and 5; entry 5 does not exist, so it gives 0, not 3, while
//   {1, 0, 0} covers entries 2 and 3 and gives 3.
// Requires: volume as RefinementFactors says; smallest_size_level and the cell's size level from a grid, the cell's
//   size level at most smallest_size_level (a smaller cell, s < 0, is not in the rule and gives 0). Nothing outside is
//   undefined: for any cell, any smallest size level and any counts, spacing and layer count, the call reads only
//   inside a layer, provided each layer array holds the entries its counts say (the call cannot check that).
// Result outside the preconditions: 0, except that a stored code beyond -24..24 gives the nearest limit, so the result
//   always indexes the table of ComputeCellSizeFactorTable; a caller may rely on -24..24 and never needs its own check.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-size-rule>
FLUIDITY_CPU_AND_GPU constexpr int
ComputeCellSizeFactorCode(const RefinementFactors& volume, SizeLevel smallest_size_level, CellId cell) noexcept {
  // Steps: 1 check the volume  2 choose the layer  3 find the entry  4 treat the outside
  // Step 1: a volume that can be read. 2^20 per axis keeps the product of three counts below 2^60, in 64 bits.
  constexpr std::int32_t kMaxCount = 1'048'576;  // 2^20.
  constexpr std::int32_t kMaxSpacingLog2 = 31;
  const bool has_usable_x = volume.count_x >= 1 && volume.count_x <= kMaxCount;
  const bool has_usable_y = volume.count_y >= 1 && volume.count_y <= kMaxCount;
  const bool has_usable_z = volume.count_z >= 1 && volume.count_z <= kMaxCount;
  const bool has_usable_spacing = volume.entry_spacing_log2 >= 0 && volume.entry_spacing_log2 <= kMaxSpacingLog2;
  const bool has_layers = volume.layer_count >= 1;
  if (!has_usable_x || !has_usable_y || !has_usable_z || !has_usable_spacing || !has_layers) {
    return 0;
  }
  const std::int32_t layer_count = volume.layer_count < kMaxSizeLevels ? volume.layer_count : kMaxSizeLevels;

  // Step 2: cell_width_log2 is below 0 only for a cell smaller than the smallest size level, and at most 255
  // otherwise. The layer is 0 when it is at most spacing (then at most 31), else the difference. A layer the volume
  // does not hold reads 0.
  const int cell_width_log2 = static_cast<int>(smallest_size_level) - static_cast<int>(ToSizeLevel(cell));
  if (cell_width_log2 < 0) {
    return 0;
  }
  const int spacing = volume.entry_spacing_log2;
  const int layer = cell_width_log2 <= spacing ? 0 : cell_width_log2 - spacing;
  if (layer >= layer_count || volume.layer_codes[layer] == nullptr) {
    return 0;
  }

  // Step 3: every index inside the layer's extent before the read; a negative coordinate is outside.
  const auto unsigned_width_log2 = static_cast<unsigned>(layer == 0 ? cell_width_log2 : 0);
  const auto unsigned_spacing = static_cast<unsigned>(spacing);
  const auto unsigned_layer = static_cast<unsigned>(layer);
  const CellCoord cell_coord = ToCellCoord(cell);
  const std::int64_t index_x =
      ComputeRefinementEntryIndex(cell_coord.x, unsigned_width_log2, unsigned_spacing, unsigned_layer, volume.count_x);
  const std::int64_t index_y =
      ComputeRefinementEntryIndex(cell_coord.y, unsigned_width_log2, unsigned_spacing, unsigned_layer, volume.count_y);
  const std::int64_t index_z =
      ComputeRefinementEntryIndex(cell_coord.z, unsigned_width_log2, unsigned_spacing, unsigned_layer, volume.count_z);
  if (index_x < 0 || index_y < 0 || index_z < 0) {
    return 0;
  }
  const auto extent_x = static_cast<std::int64_t>(ComputeRefinementLayerExtent(volume.count_x, unsigned_layer));
  const auto extent_y = static_cast<std::int64_t>(ComputeRefinementLayerExtent(volume.count_y, unsigned_layer));
  const std::int64_t entry_index = (index_z * extent_y + index_y) * extent_x + index_x;
  // NOLINTNEXTLINE(bugprone-signed-char-misuse): the byte is a number from -128 to 127, not a character.
  const int stored_code = static_cast<int>(volume.layer_codes[layer][entry_index]);
  int code = stored_code;
  if (stored_code < -kMaxCellSizeFactorCode) {
    code = -kMaxCellSizeFactorCode;
  }
  if (stored_code > kMaxCellSizeFactorCode) {
    code = kMaxCellSizeFactorCode;
  }
  // Step 4: outside the volume the code is 0, so a box that reaches outside takes the smaller of the code and 0.
  const bool reaches_outside = ComputeRefinementBoxReachesOutside(index_x, unsigned_layer, volume.count_x)
      || ComputeRefinementBoxReachesOutside(index_y, unsigned_layer, volume.count_y)
      || ComputeRefinementBoxReachesOutside(index_z, unsigned_layer, volume.count_z);
  if (reaches_outside && code > 0) {
    return 0;
  }
  return code;
}

// Finds the nearest committed cell of a position: among the active cells that hold the 27 searched positions
// position + (i, j, k) * biggest_cell_width, each of i, j and k in {-1, 0, 1}, the one whose centre is nearest the
// position. kNoIndex when no searched position lies in an active cell. Only those 27 positions are looked at, so the
// cell found may not touch the position, and a nearer, smaller cell that holds none of them is missed.
//
// Method: FindCellAt(committed, searched position), the form that searches, for k from -1 to 1, then j, then i. Each
//   distance is measured in biggest cell widths from the frame's origin, so in any frame that passes the frame check
//   no distance overflows, and an underflow never changes which cell is nearer. The position is q = (position / 2 -
//   origin / 2) / (biggest_cell_width / 2) per axis; the halves keep the difference finite for every finite position
//   and origin. A cell of size level l at coordinate c has its centre at (c + 1/2) 2^-l, exact. The squared distance
//   is Dot(centre - q, centre - q). A smaller squared distance wins; on an equal one the cell whose id comes first
//   in the fixed order wins (ComesBefore), so the answer does not depend on the order of the search. The search
//   visits z, then y, then x, which is the fixed order for cells of one size; a big cell may be found after a small
//   cell that it comes before, and the tie rule settles that.
//   Range. When FindCellAt finds a cell at a searched position, its offset from the origin is at most (kMaxCoord +
//   1) biggest widths, so |q| stays below 2^19, every centre below 2^18 + 1, each square below 2^40 and their sum
//   finite. A q that is not finite therefore comes only when no searched position finds a cell: kNoIndex at once.
//   Two distinct active cells have centres that differ by at least 2^-16 on some axis (each is a multiple of 2^-16),
//   so when one squared distance underflows the other is at least about 2^-32 and the order is kept. Same answer on
//   the CPU and the GPU. Each offset i * biggest_cell_width is exact (a product by -1, 0 or 1), so a searched
//   position is one correctly rounded sum, fused or not. The halves, the difference, the quotient, the centre and
//   each step of the squared distance are each one correctly rounded IEEE 754 operation, in x, y, z order. Host code
//   is compiled with contraction off (-ffp-contract=off; /fp:precise for MSVC) and device code with --fmad=false, so
//   no product and sum are fused on either processor, and every double compared is the same on both. The comparisons
//   themselves round nothing.
// Units: position in m, world. The distances compared are in biggest cell widths.
// Ties: an equal squared distance goes to the cell that comes first in the fixed order (ComesBefore of the two ids).
// Example: cells {0,0,0} and {2,0,0} of one size level, 1 m wide, origin 0. From (1.5, 0.5, 0.5) both centres are
//   1.0 m away: the place of {0,0,0}. From (1.75, 0.5, 0.5), 1.25 m and 0.75 m: the place of {2,0,0}. From
//   (10, 10, 10): kNoIndex.
// Requires: the tables of a built grid on the processor that calls (GridTables), as FindCellAt. Any position.
//   Checked in every compile mode: tables with no block, or a frame whose size_level_count is outside
//   1..kMaxSizeLevels (GridTables{} of no grid included), give kNoIndex before any table is read.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-size-rule>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU inline CellPlaceIndex
FindNearestCommittedCell(const GridTables& committed, Vec3 position) noexcept {
  // Steps: 1 the guards  2 the position in biggest widths  3 the search
  // Step 1: no block or no usable frame means no cell, and no table is read.
  const bool has_blocks = committed.block_count > 0;
  const int size_level_count = committed.frame.size_level_count;
  const bool has_usable_frame = size_level_count >= 1 && size_level_count <= kMaxSizeLevels;
  if (!has_blocks || !has_usable_frame) {
    return {.value = kNoIndex};
  }

  // Step 2: halving is exact for a normal double and correctly rounded below, the same on both processors.
  const double search_step = committed.frame.biggest_cell_width;
  const double half_step = 0.5 * search_step;
  const Vec3 half_origin = 0.5 * committed.frame.origin;
  const Vec3 position_in_widths = {
      .x = ((0.5 * position.x) - half_origin.x) / half_step,
      .y = ((0.5 * position.y) - half_origin.y) / half_step,
      .z = ((0.5 * position.z) - half_origin.z) / half_step,
  };
  const bool is_finite_in_widths = math::IsFinite(position_in_widths.x) && math::IsFinite(position_in_widths.y)
      && math::IsFinite(position_in_widths.z);
  if (!is_finite_in_widths) {
    return {.value = kNoIndex};
  }

  // Step 3: the 27 searched positions, the nearest centre, the fixed order on a tie.
  constexpr GridFrame kUnitFrame = {.origin = {}, .biggest_cell_width = 1.0, .size_level_count = kMaxSizeLevels};
  std::uint32_t nearest_place = kNoIndex;
  CellId nearest_cell{};
  double nearest_squared_distance = 0.0;
  for (int offset_z = -1; offset_z <= 1; ++offset_z) {
    for (int offset_y = -1; offset_y <= 1; ++offset_y) {
      for (int offset_x = -1; offset_x <= 1; ++offset_x) {
        const Vec3 searched_position = {
            .x = position.x + (static_cast<double>(offset_x) * search_step),
            .y = position.y + (static_cast<double>(offset_y) * search_step),
            .z = position.z + (static_cast<double>(offset_z) * search_step),
        };
        const CellPlaceIndex found = FindCellAt(committed, searched_position);
        if (found.value == kNoIndex) {
          continue;
        }
        const CellId found_cell = ToCellId(committed, found);
        const CellCoord found_coord = ToCellCoord(found_cell);
        const double width_in_widths = ComputeCellWidth(kUnitFrame, ToSizeLevel(found_cell));
        const Vec3 center_in_widths = {
            .x = (static_cast<double>(found_coord.x) + 0.5) * width_in_widths,
            .y = (static_cast<double>(found_coord.y) + 0.5) * width_in_widths,
            .z = (static_cast<double>(found_coord.z) + 0.5) * width_in_widths,
        };
        const Vec3 center_offset = center_in_widths - position_in_widths;
        const double squared_distance = math::Dot(center_offset, center_offset);
        const bool is_first_found = nearest_place == kNoIndex;
        const bool is_nearer = squared_distance < nearest_squared_distance;
        const bool is_as_near = squared_distance == nearest_squared_distance;
        const bool wins_tie = is_as_near && ComesBefore(found_cell, nearest_cell);
        if (is_first_found || is_nearer || wins_tie) {
          nearest_place = found.value;
          nearest_cell = found_cell;
          nearest_squared_distance = squared_distance;
        }
      }
    }
  }
  return {.value = nearest_place};
}

// Computes the size level of a new cell, one the step adds where liquid enters space with no cell: one size level
// smaller than its nearest committed cell (FindNearestCommittedCell), but never smaller than smallest_size_level, and
// smallest_size_level itself when no searched position lies in an active cell.
//
// Method: FindNearestCommittedCell, then the size level of its block plus 1, and the smaller (the bigger cell) of that
//   and smallest_size_level. The size step rule is kept with the cell found only: that cell may not touch the new
//   cell, a nearer, smaller cell can be missed, and the cap applies even when the cell found is smaller than
//   smallest_size_level (settings changed between steps). The step that adds cells therefore closes the size step
//   rule over its new list, and the cell list check turns down a list that breaks it.
//   smallest_size_level reads as at most the committed frame's last size level, as in ComputeWantedCellWidth: the
//   frame's size_level_count clamped by ComputeTableSizeLevelCount, so tables of no grid, whose frame has no size
//   level, read as one size level and give 0. The result is always a size level of the frame. Integers only, so the
//   CPU and the GPU agree whenever FindNearestCommittedCell does.
// Units: position in m, world.
// Example: the size-level-0 cell {0,0,0} beside the 8 size-level-1 cells of the size-level-0 cell {1,0,0},
//   3 size levels, biggest cells 1 m wide, smallest_size_level 2. From (-0.5, 0.5, 0.5) the searched position
//   (0.5, 0.5, 0.5) finds the big cell: 1. From (2.25, 0.25, 0.25) the searched position (1.25, 0.25, 0.25) finds the
//   small cell {2,0,0}: 2, and 1 with smallest_size_level 1; the small cell {3,0,0}, whose centre is only 0.5 m away,
//   holds no searched position. (10, 10, 10) finds no cell: 2.
// Requires: as FindNearestCommittedCell. Any position and any smallest_size_level.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-size-rule>, <sec-cell-sizes>
FLUIDITY_CPU_AND_GPU inline SizeLevel
ComputeNewCellSizeLevel(const GridTables& committed, Vec3 position, SizeLevel smallest_size_level) noexcept {
  const int last_size_level = ComputeTableSizeLevelCount(committed.frame.size_level_count) - 1;
  const int asked_smallest_size_level = static_cast<int>(smallest_size_level);
  const int capped_size_level =
      asked_smallest_size_level <= last_size_level ? asked_smallest_size_level : last_size_level;
  const CellPlaceIndex nearest = FindNearestCommittedCell(committed, position);
  if (nearest.value == kNoIndex) {
    return static_cast<SizeLevel>(capped_size_level);
  }
  const int nearest_size_level = static_cast<int>(ToSizeLevel(committed, ToCellBlockIndex(nearest)));
  const int one_size_level_smaller = nearest_size_level + 1;
  const int new_size_level = one_size_level_smaller <= capped_size_level ? one_size_level_smaller : capped_size_level;
  return static_cast<SizeLevel>(new_size_level);
}

}  // namespace fluidity::grid
