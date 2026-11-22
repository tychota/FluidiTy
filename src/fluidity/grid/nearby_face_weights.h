// The faces near a position and the weights they get: the list a particle may give weight to, and what each kept face
// gets.
//
// Purpose: give the transfer step, for one position and one axis, the grid faces that can get weight from it, as a
//   fixed list in ascending index order, from the tables of one grid version, on the CPU or the GPU; then, from the
//   faces the caller kept, the weight each gets, how fast that weight changes with the position, and the fit of how
//   a value varies around the position.
//
// Idea: a particle passes its values to the faces of its own cell and of the cells touching it, whose centres lie
//   within one and a half cell widths of it on each axis. The choice is made in whole quarter widths of the cell, never
//   on rounded distances, so the CPU, the GPU and a face that asks the same question all choose alike. Among cells of
//   one size every position has 27 such faces. Where sizes change it has more, at most 212 in any valid grid. The
//   caller then clears the bits of a mask for the faces it may not use.
//   Proof of the bound, in quarter widths from the low corner of the cell, along the axis and across it:
//   1. The smallest face is a square of half a cell width; its places are half-width places, at even planes and odd
//   centres.
//   2. A face lies on a side of the cell or of a touching cell: at most 6 planes inside the window and -4..8.
//   3. Across, it covers a half-width place whose centre lies in the window and in -3..7: at most 6 on each axis.
//   4. Faces of one axis never overlap, so each place holds at most one: 6 x 6 x 6 = 216, less the 4 places of the
//      plane through the cell's own middle, where no face can lie: 212.
//   The largest count known is 124, and it is the exact maximum where only two cell sizes meet. Running out of room is
//   therefore an impossible case, reported as kInternalFault. The check stays so that a list is never overrun.
//   What a caller on the GPU must know: a list is 888 bytes. With the mask, position and sums of one axis that go with
//   it, a thread keeps 1,152 bytes, and 3,456 bytes with three axes held at once. A call to FindNearbyFaces needs more
//   while it runs: the list it builds (888), its touching cells (228) and its face pieces (100), 1,216 bytes on top of
//   what the caller holds. With three axes held, that is 4,672 bytes at the moment of the call, and 888 more if the
//   caller keeps the returned list apart from the weights. Both are above the driver's default stack of 1,024 bytes,
//   so a module whose GPU function calls them declares its own stack entry, sized from these numbers (the
//   static_asserts below keep them true) plus its own locals; the exact frame is read from the compiler once a GPU
//   function calls it.
//
//   The weights. Each kept face gets a raw weight: the weight curve (grid/weight_curve.h) on each of the three axes,
//   multiplied, times the face's area in cell faces (1 for a face of the cell's size, 1/4 for a smaller one, 4 for a
//   bigger one). The raw weights are added in list order and each is divided by the sum, so what is left out never
//   changes the total. From the scaled weights come the mean offset (the weighted mean of the faces' offsets from the
//   position), the weighted spread around it, and the inverse of that spread, which a transfer uses to fit how a
//   value varies around the position. Four properties hold for every grid and every mask whose fit is not kNone:
//   1. Constants are reproduced for any kept set: the weights add up to 1, and their changes add up to 0.
//   2. Linear fields are reproduced whenever the fit is kFull, in exact arithmetic. In floating point the error of a
//      fitted slope depends on the spread's condition number and on the size of the values
//      (ComputeNearbyFaceWeights).
//   3. A weight is never negative, and is 0 only for a face exactly 1.5 cell widths away on some axis.
//   4. Weights jump when the position changes cell across a size change: the width, the candidates and the area
//      factors all follow the cell that holds the position. This is the known open question of the force passed
//      between cells of different sizes.
//   Worked example, one axis of the complete case (27 same-size faces, all kept). At t = 0.4 the three faces of an
//   axis lie at offsets -1.4, -0.4 and 0.6 cell widths and get Q = 0.005, 0.59 and 0.405, which add up to 1. Their
//   mean offset is -1.4 x 0.005 - 0.4 x 0.59 + 0.6 x 0.405 = 0, and their spread is 1.96 x 0.005 + 0.16 x 0.59 +
//   0.36 x 0.405 = 0.25 cell widths squared. In three dimensions the sum of the products is W = 1, the mean offset is
//   0 and the spread is h^2 / 4 times the identity, so the inverse spread is 4 / h^2 times the identity, up to
//   rounding. That case is never a separate path in the code.
//   The weight of one face is computed with the same operations in ComputeNearbyFaceWeights and in
//   ComputeFaceWeight, so a face that asks what a particle gives it gets the particle's own bits.
//
// Words:
//   candidate       a face that may get weight from a position, before any face is left out
//   window quarter  n_p: the quarter of its cell that holds a position, 0..3 on each axis; the candidates of a position
//                   depend on it only through its cell and these three numbers, so a cell has 64 lists for one axis
//   kept mask       one bit per face of a list; a set bit means the caller keeps that face
//   offset          a face's position minus the particle's position, in m
//   mean offset     the weighted mean of the offsets of the kept faces
//   offset from mean  the offset minus the mean offset; the one used in fits and in writes
//   weight change   the derivative of a face's weight with respect to the particle's position, in 1/m
//   weighted spread the symmetric 3 x 3 matrix of the weighted offsets from the mean (grid/weight_curve.h)
// Owns:    NearbyFaceList, NearbyFaceMask, NearbyFaceFit, NearbyFaceWeights, FaceWeight, FindNearbyFaces, ToMaskOfAll,
//          ComputeNearbyFaceWeights, ComputeFaceWeight; and, for this header's own functions and the tests of the
//          grid only, FindNearbyFacesWithRoom, ComputeCandidateWindowQuarter, ComputeIsInCandidateWindow,
//          ApplyNearbyFaceInsert, RawFaceWeight, ComputeRawFaceWeight, ComputeIsKept, ComputeKeptFaceWeight,
//          ComputeIsCompleteNeighbourhood, CholeskyInverse, ComputeCholeskySolution, ComputeCholeskyInverse,
//          kFindNearbyFacesCallBytes
// Uses:    Axis (base/axis.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), Mat3 (base/mat3.h), math::Floor, math::Sqrt,
//          math::IsFinite (base/scalar_math.h), Vec3 (base/vec3.h), FindCellAt (grid/find_cell_at.h), FindCellBlock,
//          GridTables, ToCellId, ToSizeLevel, ToAxis, ComputeCellWidth, ComputeFacePosition, ComputeFaceArea
//          (grid/grid_tables.h), ComputeFaceCenterInQuarterWidths (grid/face_search.h), FindFacePieces,
//          FindTouchingCells (grid/touching_cells.h), GridError (grid/grid_error.h), ComputeWeightCurve,
//          kMaxNearbyFaces, kMinWeightSum, kMinPivotShare (grid/weight_curve.h), the id types (grid/grid_ids.h)
// Used by: the transfer step and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-nearby-faces>, <proc-nearby-face-weights>,
//          <eq:scaled-weights>, <eq:weighted-spread>, <sec-face-pieces>

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/touching_cells.h"
#include "fluidity/grid/weight_curve.h"

namespace fluidity::grid {

// The candidates of one axis for one position, before any is left out.
struct NearbyFaceList {
  CellPlaceIndex cell;  // The active cell that contains the position.
  double cell_width;  // m. The width of `cell`; every distance to a face is measured in this width.
  Axis axis;  // The axis of every face in the list.
  int count;  // 0..kMaxNearbyFaces. 0 after a failure.
  // The places of the faces, faces[0..count - 1], in ascending index order. Entries from count on mean nothing.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  FacePlaceIndex faces[kMaxNearbyFaces];
};
static_assert(sizeof(NearbyFaceList) == 888, "a list is 24 bytes and 216 indexes of 4 bytes");
static_assert(std::is_trivially_copyable_v<NearbyFaceList>);

// Which faces of a list the caller keeps: bit i of word i / 64 set means face i of the list is kept.
struct NearbyFaceMask {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  std::uint64_t bits[4];
};
static_assert(sizeof(NearbyFaceMask) == 32, "four 64-bit words hold the 216 faces of a list");
static_assert(kMaxNearbyFaces <= 4 * 64);

// What the faces a caller kept allow it to compute.
enum class NearbyFaceFit : std::uint8_t {
  kFull,  // Weights, mean and an invertible spread: a fitted velocity change can be computed.
  kMeanOnly,  // The weights are valid, but the kept faces are too few or too flat to fit a change.
  kNone,  // No kept face has weight. The particle cannot read or write the grid on this axis.
};

// Computes the window quarter n_p of a position on one axis: the quarter of its cell that holds it, 0..3. For this
// header's own functions and the tests of the grid only.
//
// Method: with the quotient q = (position - origin) / cell_width, n_p = floor(4 q) - 4 cell_coord, then clamped to
//   0..3. Multiplying by 4 and taking the floor are exact, and so is subtracting 4 cell_coord from the whole number
//   floor(4 q), since both are far below 2^53 (|cell_coord| < 2^18, so |4 q| < 2^20 for a cell FindCellAt found). So
//   n_p is exact for the quotient as computed. Subtracting cell_coord from q first would lose bits in cells of a
//   negative coordinate, so it is not done. The one rounding left is the division of q, the same as in FindCellAt. The
//   clamp keeps n_p in the four windows of the cell if q lies outside it. A NaN gives 0.
// Units: position and origin in m, world; cell_width in m.
// Example: cell {2,...} of width 1 m from the origin 0: position 2.6 gives floor(4 x 2.6) - 4 x 2 = 10 - 8 = 2;
//   1.999999 gives -1, clamped to 0; 3.0 gives 4, clamped to 3.
// Requires: cell_width is finite and above 0; cell_coord is the coordinate that FindCellAt's cell has on this axis.
// Read: docs/method-paper/fluidity-method.typ <sec-nearby-faces>
// clang-tidy: the width and the coordinate are no longer combined in one expression, so the swappable-parameters check
// flags them.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU inline int ComputeCandidateWindowQuarter(
    double position_on_axis,
    double origin_on_axis,
    double cell_width,
    std::int32_t cell_coord
) noexcept {
  const double quotient = (position_on_axis - origin_on_axis) / cell_width;
  const double quarters_into_cell = math::Floor(4.0 * quotient) - (4.0 * static_cast<double>(cell_coord));
  if (!(quarters_into_cell >= 0.0)) {
    return 0;
  }
  if (quarters_into_cell > 3.0) {
    return 3;
  }
  return static_cast<int>(quarters_into_cell);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Says whether a face centre lies in the window of a window quarter on one axis. For this header's own functions and
// the tests of the grid only.
//
// Method: in real numbers the window is -1.5 h < x_f - x_p <= 1.5 h, twelve quarter steps, the low end open and the
//   high end closed because a cell owns its low side. With the quarter of the position n_p and the quarter of the face
//   centre n_f, both counted in quarter widths of the cell from its low corner, the test is n_p - 5 <= n_f <= n_p + 6.
//   Every face centre of the three size levels lies on that quarter lattice, so the integer test loses nothing.
// Example: among cells of one size, n_p = 0 holds the own-axis faces at n_f = -4, 0 and 4, and n_p = 2 those at 0, 4
//   and 8. Across, the faces sit at n_f = 4 k + 2 and every window holds -2, 2 and 6.
// Read: docs/method-paper/fluidity-method.typ <sec-nearby-faces>
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsInCandidateWindow(std::int64_t face_quarter, int window_quarter) noexcept {
  return window_quarter - 5 <= face_quarter && face_quarter <= window_quarter + 6;
}

// Inserts `face` into list.faces in ascending index order and says whether it fits. A face already listed is left as it
// is. When the list holds max_faces faces and `face` is new, nothing is written and the answer is false. For this
// header's own functions and the tests of the grid only.
// Requires: max_faces is 0..kMaxNearbyFaces and list.count is at most max_faces.
FLUIDITY_CPU_AND_GPU constexpr bool
ApplyNearbyFaceInsert(NearbyFaceList& list, FacePlaceIndex face, int max_faces) noexcept {
  assert(max_faces >= 0 && max_faces <= kMaxNearbyFaces);
  int position = list.count;
  while (position > 0 && list.faces[position - 1].value > face.value) {
    --position;
  }
  if (position > 0 && list.faces[position - 1].value == face.value) {
    return true;
  }
  if (list.count >= max_faces) {
    return false;
  }
  for (int moved = list.count; moved > position; --moved) {
    list.faces[moved] = list.faces[moved - 1];
  }
  list.faces[position] = face;
  ++list.count;
  return true;
}

// Finds the candidates of one axis for a position, with room for only max_faces of them: the one writer of a list.
// FindNearbyFaces is this function with kMaxNearbyFaces. For this header's own functions and the tests of the grid
// only.
//
// Method: finds the cell with FindCellAt, then looks at every face piece of the cell and of each cell touching it. A
//   face piece is a candidate when its axis is `axis`, its size level is the cell's minus 1, the same or plus 1, and
//   on each axis its centre, in quarter widths of the cell from the cell's low corner, is in the window of that
//   axis's window quarter (ComputeIsInCandidateWindow). Each candidate goes in once, through ApplyNearbyFaceInsert. On
//   the first candidate that does not fit, the count becomes 0 and the error kInternalFault.
// Units: m, world.
// Example: a grid whose candidates at one position are 124, with max_faces 100, gives kInternalFault and count 0; with
//   max_faces 124 it gives all 124.
// Requires: as FindCellAt; max_faces is 0..kMaxNearbyFaces; `error` is not null.
// Fails: kPositionOutsideGrid  no active cell contains the position
//        kInternalFault        more candidates than max_faces
//        On failure the list has count 0, and no entry from max_faces on was written.
// Read: docs/method-paper/fluidity-method.typ <sec-nearby-faces>
FLUIDITY_CPU_AND_GPU inline NearbyFaceList FindNearbyFacesWithRoom(
    const GridTables& tables,
    Vec3 position,
    Axis axis,
    CellBlockIndex near,
    int max_faces,
    GridError* error
) noexcept {
  // Steps: 1 find the cell  2 find the window  3 list the candidates of the cell and of its touching cells
  assert(error != nullptr);
  assert(max_faces >= 0 && max_faces <= kMaxNearbyFaces);
  *error = GridError::kNone;
  NearbyFaceList list{};
  // Step 1.
  const CellPlaceIndex cell = FindCellAt(tables, position, near);
  if (cell.value == kNoIndex) {
    *error = GridError::kPositionOutsideGrid;
    return list;
  }
  const CellId cell_id = ToCellId(tables, cell);
  const SizeLevel cell_size_level = ToSizeLevel(cell_id);
  const CellCoord cell_coord = ToCellCoord(cell_id);
  const double cell_width = ComputeCellWidth(tables.frame, cell_size_level);
  list.cell = cell;
  list.cell_width = cell_width;
  list.axis = axis;
  // Step 2: the window quarter of each axis, and the corner of the cell in quarter widths from the origin.
  const GridFrame frame = tables.frame;
  const int window_quarter_x = ComputeCandidateWindowQuarter(position.x, frame.origin.x, cell_width, cell_coord.x);
  const int window_quarter_y = ComputeCandidateWindowQuarter(position.y, frame.origin.y, cell_width, cell_coord.y);
  const int window_quarter_z = ComputeCandidateWindowQuarter(position.z, frame.origin.z, cell_width, cell_coord.z);
  const std::int64_t corner_quarter_x = 4 * std::int64_t{cell_coord.x};
  const std::int64_t corner_quarter_y = 4 * std::int64_t{cell_coord.y};
  const std::int64_t corner_quarter_z = 4 * std::int64_t{cell_coord.z};
  const auto cell_size_level_number = static_cast<int>(cell_size_level);
  // Step 3: entry -1 is the cell itself, then its touching cells.
  const TouchingCells touching = FindTouchingCells(tables, cell);
  for (int entry = -1; entry < touching.count; ++entry) {
    const CellPlaceIndex visited_cell = entry < 0 ? cell : touching.cells[entry];
    const CellFacePieces face_pieces = FindFacePieces(tables, visited_cell);
    for (int face_entry = 0; face_entry < face_pieces.count; ++face_entry) {
      const FacePlaceIndex face = face_pieces.faces[face_entry];
      if (ToAxis(face) != axis) {
        continue;
      }
      const auto face_size_level = static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(face)));
      const int size_level_difference = face_size_level - cell_size_level_number;
      if (size_level_difference < -1 || size_level_difference > 1) {
        continue;
      }
      const QuarterWidthPoint face_center = ComputeFaceCenterInQuarterWidths(tables, face, cell_size_level_number);
      const bool is_in_window = ComputeIsInCandidateWindow(face_center.x - corner_quarter_x, window_quarter_x)
          && ComputeIsInCandidateWindow(face_center.y - corner_quarter_y, window_quarter_y)
          && ComputeIsInCandidateWindow(face_center.z - corner_quarter_z, window_quarter_z);
      if (is_in_window && !ApplyNearbyFaceInsert(list, face, max_faces)) {
        list.count = 0;
        *error = GridError::kInternalFault;
        return list;
      }
    }
  }
  return list;
}

// Finds the faces of one axis that can get weight from a position: the candidates, before any is left out.
//
// Method: the face pieces of this axis that belong to the active cell holding the position, or to a cell touching
//   it, whose own size level is the cell's minus 1, the same or plus 1, and whose centre x_f satisfies
//   -1.5 h < x_f - x_p <= 1.5 h on each axis, h the cell's width. The test is made in whole quarter widths
//   (ComputeIsInCandidateWindow), so a position on a cell side or centre, or a rounding step from one, gives the same
//   faces as its neighbours: among cells of one size, always 27. A face exactly 1.5 h away is listed and has weight 0.
//   At most 212 faces in any valid grid (file header); the list holds kMaxNearbyFaces = 216.
// Units: m, world.
// Order: ascending FacePlaceIndex.
// Example: in a grid of one size level, 1 m cells, position {2.6, 2.7, 2.8} in the cell {2,2,2}: on each axis 27 faces.
//   In the cell {1,1,1} of a level-0 box whose 26 other cells are split: 76.
// Requires: as FindCellAt, and every block index entry of the tables is kNoIndex or below block_count, as in every
//   built grid. Any position and any `near`. `error` is not null.
// Fails: kPositionOutsideGrid  no active cell contains the position
//        kInternalFault        more candidates than the list holds: a broken table, never a grid
//        On failure the list has count 0 and `*error` says why; otherwise `*error` is kNone.
// Read: docs/method-paper/fluidity-method.typ <sec-nearby-faces>
FLUIDITY_CPU_AND_GPU inline NearbyFaceList
FindNearbyFaces(const GridTables& tables, Vec3 position, Axis axis, CellBlockIndex near, GridError* error) noexcept {
  return FindNearbyFacesWithRoom(tables, position, axis, near, kMaxNearbyFaces, error);
}

// Makes the mask that keeps every face of a list: bits 0..count - 1 set, every other bit clear.
// Example: a count of 124 sets bits 0..123, which fills word 0 and the low 60 bits of word 1.
// Requires: list.count is 0..kMaxNearbyFaces.
FLUIDITY_CPU_AND_GPU constexpr NearbyFaceMask ToMaskOfAll(const NearbyFaceList& list) noexcept {
  assert(list.count >= 0 && list.count <= kMaxNearbyFaces);
  NearbyFaceMask mask{};
  for (int word = 0; word < 4; ++word) {
    const int faces_before_word = 64 * word;
    const int faces_in_word = list.count - faces_before_word;
    if (faces_in_word >= 64) {
      mask.bits[word] = ~std::uint64_t{0};
    } else if (faces_in_word > 0) {
      mask.bits[word] = (std::uint64_t{1} << static_cast<unsigned>(faces_in_word)) - 1;
    }
  }
  return mask;
}

// The weights of the kept faces of one list for one position, and what a transfer needs to fit a value around it.
// One axis, one position; built when needed and never kept in memory.
struct NearbyFaceWeights {
  NearbyFaceList list;  // The candidates, as FindNearbyFaces made them.
  NearbyFaceMask kept;  // The faces the caller kept.
  Vec3 position;  // m, world. The position the weights are for.
  double weight_sum;  // No unit. Sum of the kept raw weights; 1 when all 27 same-size faces are kept.
  Vec3 mean_offset;  // m. Weighted mean of (face position - position). Zero up to rounding when complete.
  Vec3 raw_weight_change_sum;  // 1/m. Sum of the kept raw weight changes. Zero up to rounding when complete.
  Mat3 weighted_spread;  // m2, symmetric. cell_width^2 / 4 times the identity, up to rounding, when complete.
  Mat3 inverse_weighted_spread;  // 1/m2, symmetric. Zero unless fit is kFull.
  NearbyFaceFit fit;  // What the kept faces allow.
  bool is_complete;  // All 27 faces kept and all of the cell's size. False when the fit is kNone.
};
static_assert(sizeof(NearbyFaceWeights) == 1152, "a list of 888 bytes, a mask of 32, and 232 bytes of sums");
static_assert(std::is_trivially_copyable_v<NearbyFaceWeights>);

// The bytes a call to FindNearbyFaces holds while it runs, on top of its caller's: the list it builds, its touching
// cells and its face pieces (the file header's GPU text states the numbers).
inline constexpr std::size_t kFindNearbyFacesCallBytes =
    sizeof(NearbyFaceList) + sizeof(TouchingCells) + sizeof(CellFacePieces);
static_assert(sizeof(TouchingCells) == 228 && sizeof(CellFacePieces) == 100, "56 x 4 + 4 and 24 x 4 + 4 bytes");
static_assert(kFindNearbyFacesCallBytes == 1216, "888 + 228 + 100 bytes: the file header's GPU text");
static_assert(3 * sizeof(NearbyFaceWeights) + kFindNearbyFacesCallBytes == 4672, "3,456 + 1,216 bytes");

// What one kept face gets from a position.
struct FaceWeight {
  FacePlaceIndex face;  // The face's place.
  double
      weight;  // No unit, never negative; 0 for a face exactly 1.5 widths away. Kept weights of one axis add up to 1.
  Vec3 offset;  // m. The face's position minus the position.
  Vec3 offset_from_mean;  // m. offset - mean_offset. Use this one in fits and in writes.
  Vec3 weight_change;  // 1/m. Derivative of `weight` with respect to the position.
};

// The raw weight of one face and its change with the position, before the sums scale them. For this header's own
// functions only.
struct RawFaceWeight {
  double weight;  // No unit. The weight curve on each axis, multiplied, times the face's area in cell faces.
  Vec3 weight_change;  // 1/m. The derivative of `weight` with respect to the position.
};

// Computes the raw weight of a face for a position and its change with the position. Every weight of this header
// comes from this function, so the particle's list and a face's own question give the same bits. For this header's
// own functions only.
//
// Method: with d = (x_f - x_p) / h on each axis and Q the weight curve, w_f = Q(d_x) Q(d_y) Q(d_z) area_f / h^2 and
//   g_f = -(area_f / h^3) (Q'(d_x) Q(d_y) Q(d_z), Q(d_x) Q'(d_y) Q(d_z), Q(d_x) Q(d_y) Q'(d_z)). The area in cell
//   faces is area_f / h^2 and the change's scale is that divided by h; each product is taken left to right.
// Units: m, world; `cell_width` in m.
// Example: a face of the cell's size at d = (0.25, 0, 0): w_f = 0.6875 x 0.75 x 0.75 = 0.38671875, and g_f has x
//   x entry -(1 / h) x (-0.5) x 0.75 x 0.75 = 0.28125 / h.
// Requires: `face` is a place of block 0..tables.block_count - 1; `cell_width` is finite and above 0; `position` is
//   finite.
// Read: docs/method-paper/fluidity-method.typ <eq:scaled-weights>
FLUIDITY_CPU_AND_GPU inline RawFaceWeight
ComputeRawFaceWeight(const GridTables& tables, FacePlaceIndex face, Vec3 position, double cell_width) noexcept {
  const Vec3 face_position = ComputeFacePosition(tables, face);
  const WeightCurveValue curve_x = ComputeWeightCurve((face_position.x - position.x) / cell_width);
  const WeightCurveValue curve_y = ComputeWeightCurve((face_position.y - position.y) / cell_width);
  const WeightCurveValue curve_z = ComputeWeightCurve((face_position.z - position.z) / cell_width);
  const double cell_width_square = cell_width * cell_width;
  const double area_in_cell_faces = ComputeFaceArea(tables, face) / cell_width_square;
  const double area_per_cell_width = area_in_cell_faces / cell_width;
  const double curve_product = (curve_x.weight * curve_y.weight) * curve_z.weight;
  const double slope_x_product = (curve_x.slope * curve_y.weight) * curve_z.weight;
  const double slope_y_product = (curve_x.weight * curve_y.slope) * curve_z.weight;
  const double slope_z_product = (curve_x.weight * curve_y.weight) * curve_z.slope;
  return {
      .weight = curve_product * area_in_cell_faces,
      .weight_change = {
          .x = -(area_per_cell_width * slope_x_product),
          .y = -(area_per_cell_width * slope_y_product),
          .z = -(area_per_cell_width * slope_z_product),
      },
  };
}

// Says whether face `position_in_list` of a list is kept by a mask. For this header's own functions only.
// Requires: position_in_list is 0..kMaxNearbyFaces - 1.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsKept(const NearbyFaceMask& mask, int position_in_list) noexcept {
  assert(position_in_list >= 0 && position_in_list < kMaxNearbyFaces);
  const std::uint64_t word = mask.bits[position_in_list / 64];
  return ((word >> static_cast<unsigned>(position_in_list % 64)) & 1U) != 0U;
}

// Computes what a kept face gets from the position of `weights`, scaled by the sums of `weights`, and its offset from
// the mean offset of `weights`. The one place the scaling is written, so the particle's own pass and a face's question
// give the same bits. For this header's own functions only.
//
// Method: weight = w_f / W and weight_change = (g_f - weight G) / W, per axis, with W and G the sums of
//   `weights`. offset_from_mean uses weights.mean_offset as it stands, so a caller that has not set it yet gets the
//   offset back.
// Requires: weights.weight_sum is above 0, position_in_list is 0..weights.list.count - 1.
// Read: docs/method-paper/fluidity-method.typ <eq:scaled-weights>
FLUIDITY_CPU_AND_GPU inline FaceWeight
ComputeKeptFaceWeight(const GridTables& tables, const NearbyFaceWeights& weights, int position_in_list) noexcept {
  assert(position_in_list >= 0 && position_in_list < weights.list.count);
  const FacePlaceIndex face = weights.list.faces[position_in_list];
  const RawFaceWeight raw = ComputeRawFaceWeight(tables, face, weights.position, weights.list.cell_width);
  const Vec3 offset = ComputeFacePosition(tables, face) - weights.position;
  const double weight = raw.weight / weights.weight_sum;
  return {
      .face = face,
      .weight = weight,
      .offset = offset,
      .offset_from_mean = offset - weights.mean_offset,
      .weight_change = {
          .x = (raw.weight_change.x - (weight * weights.raw_weight_change_sum.x)) / weights.weight_sum,
          .y = (raw.weight_change.y - (weight * weights.raw_weight_change_sum.y)) / weights.weight_sum,
          .z = (raw.weight_change.z - (weight * weights.raw_weight_change_sum.z)) / weights.weight_sum,
      },
  };
}

// Says whether a mask keeps all 27 faces of a list and every one has the size level of the cell. For this header's own
// functions only.
FLUIDITY_CPU_AND_GPU inline bool
ComputeIsCompleteNeighbourhood(const GridTables& tables, const NearbyFaceList& list, NearbyFaceMask kept) noexcept {
  constexpr int kFacesOfOneSize = 27;
  if (list.count != kFacesOfOneSize) {
    return false;
  }
  const SizeLevel cell_size_level = ToSizeLevel(ToCellId(tables, list.cell));
  for (int position_in_list = 0; position_in_list < list.count; ++position_in_list) {
    const SizeLevel face_size_level = ToSizeLevel(tables, ToCellBlockIndex(list.faces[position_in_list]));
    if (!ComputeIsKept(kept, position_in_list) || face_size_level != cell_size_level) {
      return false;
    }
  }
  return true;
}

// The inverse of a symmetric matrix by Cholesky factorisation, and whether the factorisation found a usable pivot.
// For this header's own functions only.
struct CholeskyInverse {
  bool is_invertible;  // False when a pivot is at most kMinPivotShare times the trace, or is not finite.
  Mat3 inverse;  // Symmetric. Zero when is_invertible is false.
};

// Solves L L^T x = right_side for a factor L that is zero above its diagonal: forward substitution, then backward
// substitution, each subtracting the known terms in column order. For this header's own functions only. Requires: the
// diagonal of `factor` is not 0; entries above the diagonal are ignored.
FLUIDITY_CPU_AND_GPU inline Vec3 ComputeCholeskySolution(Mat3 factor, Vec3 right_side) noexcept {
  const double forward_x = right_side.x / factor.xx;
  const double forward_y = (right_side.y - (factor.yx * forward_x)) / factor.yy;
  const double forward_z = ((right_side.z - (factor.zx * forward_x)) - (factor.zy * forward_y)) / factor.zz;
  const double solution_z = forward_z / factor.zz;
  const double solution_y = (forward_y - (factor.zy * solution_z)) / factor.yy;
  const double solution_x = ((forward_x - (factor.yx * solution_y)) - (factor.zx * solution_z)) / factor.xx;
  return {.x = solution_x, .y = solution_y, .z = solution_z};
}

// Computes the inverse of a symmetric matrix by a Cholesky factorisation that stops at a small pivot. For this header's
// own functions only.
//
// Method: reads the entries on and below the diagonal, A = L L^T with L zero above its diagonal. Taking the matrix
//   rows in turn, each entry of L is its matrix entry minus the products of the entries to its left in that row and
//   in the row of its column, taken in column order; a diagonal entry is the square root of what is left. A pivot (the
//   value before the square root) at most kMinPivotShare times the trace, or not finite, ends the factorisation: the
//   matrix is too flat to invert. Column k of the inverse is the solution of L L^T x = e_k: forward substitution, then
//   backward substitution, each subtracting in column order. The inverse takes its entries on and below the diagonal
//   from those columns and is mirrored, so it is exactly symmetric.
// Requires: the entries are finite.
// Read: docs/method-paper/fluidity-method.typ <proc-nearby-face-weights>
FLUIDITY_CPU_AND_GPU inline CholeskyInverse ComputeCholeskyInverse(Mat3 matrix) noexcept {
  const CholeskyInverse not_invertible = {.is_invertible = false, .inverse = Mat3{}};
  const double smallest_pivot = kMinPivotShare * ((matrix.xx + matrix.yy) + matrix.zz);
  const double first_pivot = matrix.xx;
  if (first_pivot <= smallest_pivot || !math::IsFinite(first_pivot)) {
    return not_invertible;
  }
  const double factor_xx = math::Sqrt(first_pivot);
  const double factor_yx = matrix.yx / factor_xx;
  const double second_pivot = matrix.yy - (factor_yx * factor_yx);
  if (second_pivot <= smallest_pivot || !math::IsFinite(second_pivot)) {
    return not_invertible;
  }
  const double factor_yy = math::Sqrt(second_pivot);
  const double factor_zx = matrix.zx / factor_xx;
  const double factor_zy = (matrix.zy - (factor_zx * factor_yx)) / factor_yy;
  const double third_pivot = (matrix.zz - (factor_zx * factor_zx)) - (factor_zy * factor_zy);
  if (third_pivot <= smallest_pivot || !math::IsFinite(third_pivot)) {
    return not_invertible;
  }
  const double factor_zz = math::Sqrt(third_pivot);
  const Mat3 factor = {
      .xx = factor_xx,
      .xy = 0.0,
      .xz = 0.0,
      .yx = factor_yx,
      .yy = factor_yy,
      .yz = 0.0,
      .zx = factor_zx,
      .zy = factor_zy,
      .zz = factor_zz,
  };
  const Vec3 first_column = ComputeCholeskySolution(factor, {.x = 1.0, .y = 0.0, .z = 0.0});
  const Vec3 second_column = ComputeCholeskySolution(factor, {.x = 0.0, .y = 1.0, .z = 0.0});
  const Vec3 third_column = ComputeCholeskySolution(factor, {.x = 0.0, .y = 0.0, .z = 1.0});
  return {
      .is_invertible = true,
      .inverse = {
          .xx = first_column.x,
          .xy = first_column.y,
          .xz = first_column.z,
          .yx = first_column.y,
          .yy = second_column.y,
          .yz = second_column.z,
          .zx = first_column.z,
          .zy = second_column.z,
          .zz = third_column.z,
      },
  };
}

// Computes the weights of the kept faces of a list for a position, with the sums, the mean offset, the weighted spread
// and its inverse.
//
// Method: steps 4 to 9 of <proc-nearby-face-weights>, after FindNearbyFaces and the caller's mask. Every sum runs over
//   the kept faces in list order, starting from 0.
//   Step 4 and 5: W = sum w_f and G = sum g_f (ComputeRawFaceWeight), added face by face, W first.
//   If W is not above 0, no kept face has weight: the fit is kNone, every sum, the mean offset and both matrices are 0,
//   and nothing divides by W.
//   Step 6: each face's weight and change are scaled by W (ComputeKeptFaceWeight).
//   Step 7: mean_offset = sum weight (x_f - x_p), per axis, face by face.
//   Step 8: weighted_spread = sum ((weight r_a) r_b) with r the offset from the mean, for the entries ab = xx, yx, yy,
//   zx, zy, zz in that order; the entries above the diagonal copy those below, so the matrix is exactly symmetric.
//   Step 9: the Cholesky factor of weighted_spread / h^2 (each entry divided by h x h) with a limit on the pivots
//   (ComputeCholeskyInverse). The fit is kMeanOnly when a pivot is at most kMinPivotShare times the trace or W is
//   below kMinWeightSum, else kFull, and the inverse is that of the scaled matrix times 1 / (h x h).
//   The faces' raw weights are computed again at each step, with the same operations, so no per-face memory is held.
//   What rounding allows for a fit of values. A kFull fit can still be badly conditioned: the pivot floor lets a spread
//   through whose largest and smallest eigenvalues differ by a factor of up to about 1e12 (the condition number). The
//   scaled weights times the offsets from the mean add up to about 1e-16 h, not to 0. A value v on every face adds
//   v times that sum to the moment, and the inverse spread magnifies it, so the fitted matrix row of a field whose
//   values are about v is off by about 1e-16 x condition number x v / h, whatever its slope. A caller that fits values
//   therefore subtracts their weighted mean first, which gives the same matrix row in exact arithmetic and leaves their
//   size out of the moment; the error is then of the size of the field's variation over a cell width times 1e-16 x
//   condition number. Worked example: a unit grid, X axis, position {2.5 - 2e-6, 2.5, 2.5}, only the 9 faces at x = 2
//   and the 9 at x = 1 kept. The faces at x = 1 lie 1.5 - 2e-6 away and weigh about 2e-12 of the others, so S_xx is
//   about 4e-12, S_yy and S_zz are 0.25 and the floor is 5e-13: the fit is kFull with a condition number near 6e10. For
//   the field 1 + b . (x_f - x_p) the matrix row fitted from the values is off by 8e-6 in length (1e-16 x
//   6e10 x 1 / 1 m, measured); fitted from the values minus their weighted mean it is right to 5e-16.
// Units: position in m, world; the sums, the mean offset and the matrices as in NearbyFaceWeights.
// Order: list order; the three passes (sums, mean, spread) each visit the kept faces in ascending list position.
// Example: the complete case of the file header: W = 1, mean_offset = 0 and weighted_spread = h^2 / 4 times the
//   identity, up to rounding, fit kFull, is_complete true. A mask with only the 9 faces at +1.5 h along the axis gives
//   W = 0 and fit kNone.
// Requires: `list` is from FindNearbyFaces with error kNone, for a finite position inside the list's cell (or a
//   rounding step from it, as every caller's position is); `kept` has no bit set from list.count on; every block
//   index entry of the tables is kNoIndex or below block_count. `tables` is the one `list` was made from.
// Read: docs/method-paper/fluidity-method.typ <proc-nearby-face-weights>, <eq:scaled-weights>,
//       <eq:weighted-spread>
FLUIDITY_CPU_AND_GPU inline NearbyFaceWeights ComputeNearbyFaceWeights(
    const GridTables& tables,
    const NearbyFaceList& list,
    Vec3 position,
    NearbyFaceMask kept
) noexcept {
  assert(list.count >= 0 && list.count <= kMaxNearbyFaces);
  NearbyFaceWeights weights{};
  weights.list = list;
  weights.kept = kept;
  weights.position = position;
  weights.fit = NearbyFaceFit::kNone;
  // Steps 4 and 5: the raw weights and their sums.
  double weight_sum = 0.0;
  Vec3 raw_weight_change_sum = {.x = 0.0, .y = 0.0, .z = 0.0};
  for (int position_in_list = 0; position_in_list < list.count; ++position_in_list) {
    if (!ComputeIsKept(kept, position_in_list)) {
      continue;
    }
    const RawFaceWeight raw = ComputeRawFaceWeight(tables, list.faces[position_in_list], position, list.cell_width);
    weight_sum += raw.weight;
    raw_weight_change_sum.x += raw.weight_change.x;
    raw_weight_change_sum.y += raw.weight_change.y;
    raw_weight_change_sum.z += raw.weight_change.z;
  }
  // No kept face has weight (or the sum is not a number): nothing below may divide by it.
  if (!(weight_sum > 0.0)) {
    return weights;
  }
  weights.weight_sum = weight_sum;
  weights.raw_weight_change_sum = raw_weight_change_sum;
  // Steps 6 and 7: the scaled weights and the mean offset. The mean offset is still 0 here, so the offsets from the
  // mean that these calls give are not used.
  Vec3 mean_offset = {.x = 0.0, .y = 0.0, .z = 0.0};
  for (int position_in_list = 0; position_in_list < list.count; ++position_in_list) {
    if (!ComputeIsKept(kept, position_in_list)) {
      continue;
    }
    const FaceWeight face_weight = ComputeKeptFaceWeight(tables, weights, position_in_list);
    mean_offset.x += face_weight.weight * face_weight.offset.x;
    mean_offset.y += face_weight.weight * face_weight.offset.y;
    mean_offset.z += face_weight.weight * face_weight.offset.z;
  }
  weights.mean_offset = mean_offset;
  // Step 8: the weighted spread, entries on and below the diagonal in the order xx, yx, yy, zx, zy, zz.
  Mat3 spread = {};
  for (int position_in_list = 0; position_in_list < list.count; ++position_in_list) {
    if (!ComputeIsKept(kept, position_in_list)) {
      continue;
    }
    const FaceWeight face_weight = ComputeKeptFaceWeight(tables, weights, position_in_list);
    const Vec3 from_mean = face_weight.offset_from_mean;
    const double weighted_x = face_weight.weight * from_mean.x;
    const double weighted_y = face_weight.weight * from_mean.y;
    const double weighted_z = face_weight.weight * from_mean.z;
    spread.xx += weighted_x * from_mean.x;
    spread.yx += weighted_y * from_mean.x;
    spread.yy += weighted_y * from_mean.y;
    spread.zx += weighted_z * from_mean.x;
    spread.zy += weighted_z * from_mean.y;
    spread.zz += weighted_z * from_mean.z;
  }
  spread.xy = spread.yx;
  spread.xz = spread.zx;
  spread.yz = spread.zy;
  weights.weighted_spread = spread;
  // Step 9: the fit, from the spread in units of the cell width squared.
  const double cell_width_square = list.cell_width * list.cell_width;
  const Mat3 scaled_spread = {
      .xx = spread.xx / cell_width_square,
      .xy = spread.xy / cell_width_square,
      .xz = spread.xz / cell_width_square,
      .yx = spread.yx / cell_width_square,
      .yy = spread.yy / cell_width_square,
      .yz = spread.yz / cell_width_square,
      .zx = spread.zx / cell_width_square,
      .zy = spread.zy / cell_width_square,
      .zz = spread.zz / cell_width_square,
  };
  const CholeskyInverse scaled_inverse = ComputeCholeskyInverse(scaled_spread);
  weights.fit = NearbyFaceFit::kMeanOnly;
  if (scaled_inverse.is_invertible && weight_sum >= kMinWeightSum) {
    const double inverse_scale = 1.0 / cell_width_square;
    weights.fit = NearbyFaceFit::kFull;
    weights.inverse_weighted_spread = scaled_inverse.inverse * inverse_scale;
  }
  weights.is_complete = ComputeIsCompleteNeighbourhood(tables, list, kept);
  return weights;
}

// Computes what one kept face gets from the position of `weights`: its scaled weight, offsets and weight change.
//
// Method: the weight is w_f / W and its change (g_f - weight G) / W, the same operations as ComputeNearbyFaceWeights
//   uses, so the bits are the particle's own. When the fit is kNone no face has weight: the weight and the change are 0
//   and so is offset_from_mean, and `offset` is still the face's position minus the position.
// Order: call in ascending position_in_list for a fixed order of the caller's sums.
// Units: as FaceWeight.
// Example: unit grid (h = 1 m), position {2.5, 2.5, 2.5}, X axis, all 27 faces kept, so W = 1. The face at offset
//   {-0.5, 0, 0} gets weight 0.5 x 0.75 x 0.75 = 0.28125; the face at offset {+1.5, 0, 0} gets weight 0 and weight
//   change 0.
// Requires: `weights` is from ComputeNearbyFaceWeights on `tables`; position_in_list is 0..weights.list.count - 1 and
//   the face is kept.
// Read: docs/method-paper/fluidity-method.typ <eq:scaled-weights>
FLUIDITY_CPU_AND_GPU inline FaceWeight
ComputeFaceWeight(const GridTables& tables, const NearbyFaceWeights& weights, int position_in_list) noexcept {
  assert(position_in_list >= 0 && position_in_list < weights.list.count);
  assert(ComputeIsKept(weights.kept, position_in_list));
  if (weights.fit == NearbyFaceFit::kNone) {
    const FacePlaceIndex face = weights.list.faces[position_in_list];
    return {
        .face = face,
        .weight = 0.0,
        .offset = ComputeFacePosition(tables, face) - weights.position,
        .offset_from_mean = {.x = 0.0, .y = 0.0, .z = 0.0},
        .weight_change = {.x = 0.0, .y = 0.0, .z = 0.0},
    };
  }
  return ComputeKeptFaceWeight(tables, weights, position_in_list);
}

}  // namespace fluidity::grid
