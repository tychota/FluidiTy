// The faces near a position: the list a particle may give weight to, before any face is left out.
//
// Purpose: give the transfer step, for one position and one axis, the grid faces that can get weight from it, as a
//   fixed list in ascending index order, from the tables of one grid version, on the CPU or the GPU.
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
//   it, a thread keeps 1,152 bytes, and 3,456 bytes with three axes held at once. That is above the driver's default
//   stack of 1,024 bytes, so a module whose GPU function calls them declares its own stack entry.
//
// Words:
//   candidate       a face that may get weight from a position, before any face is left out
//   window quarter  n_p: the quarter of its cell that holds a position, 0..3 on each axis; the candidates of a position
//                   depend on it only through its cell and these three numbers, so a cell has 64 lists for one axis
//   kept mask       one bit per face of a list; a set bit means the caller keeps that face
// Owns:    NearbyFaceList, NearbyFaceMask, NearbyFaceFit, FindNearbyFaces, ToMaskOfAll; and, for this header's own
//          functions and the tests of the grid only, FindNearbyFacesWithRoom, ComputeCandidateWindowQuarter,
//          ComputeIsInCandidateWindow, ApplyNearbyFaceInsert
// Uses:    Axis (base/axis.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), math::Floor (base/scalar_math.h), Vec3
//          (base/vec3.h), FindCellAt (grid/find_cell_at.h), FindCellBlock, GridTables, ToCellId, ToSizeLevel, ToAxis,
//          ComputeCellWidth (grid/grid_tables.h), ComputeFaceCenterInQuarterWidths (grid/face_search.h),
//          FindFacePieces, FindTouchingCells (grid/touching_cells.h), GridError (grid/grid_error.h), kMaxNearbyFaces
//          (grid/weight_curve.h), the id types (grid/grid_ids.h)
// Used by: the transfer step and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-nearby-faces>, <sec-face-pieces>

#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"
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
// Method: n_p = floor(4 ((position - origin) / cell_width - cell_coord)), then clamped to 0..3. FindCellAt chose the
//   cell from the same quotient, so rounding can put the quotient a step outside the cell. The clamp ties n_p to that
//   cell: whatever rounding does, the window is one of the four the cell has. A NaN gives 0.
// Units: position and origin in m, world; cell_width in m.
// Example: cell {2,...} of width 1 m from the origin 0: position 2.6 gives floor(4 x 0.6) = 2; 1.999999 gives -1,
//   clamped to 0; 3.0 gives 4, clamped to 3.
// Requires: cell_width is finite and above 0; cell_coord is the coordinate that FindCellAt's cell has on this axis.
// Read: docs/method-paper/fluidity-method.typ <sec-nearby-faces>
FLUIDITY_CPU_AND_GPU inline int ComputeCandidateWindowQuarter(
    double position_on_axis,
    double origin_on_axis,
    double cell_width,
    std::int32_t cell_coord
) noexcept {
  const double quarters_into_cell =
      math::Floor(4.0 * (((position_on_axis - origin_on_axis) / cell_width) - static_cast<double>(cell_coord)));
  if (!(quarters_into_cell >= 0.0)) {
    return 0;
  }
  if (quarters_into_cell > 3.0) {
    return 3;
  }
  return static_cast<int>(quarters_into_cell);
}

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

}  // namespace fluidity::grid
