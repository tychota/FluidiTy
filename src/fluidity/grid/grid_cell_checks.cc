#include "fluidity/grid/grid_cell_checks.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <span>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"

#if defined(FLUIDITY_HAS_GPU)
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/grid/staggered_grid_gpu.h"
#endif

namespace fluidity::grid {
namespace {

// The most cells a list may have: positions in a list are 32-bit, as in the GPU's orders.
constexpr std::uint64_t kMaxListCount = std::numeric_limits<std::uint32_t>::max();
// A list within the room's cell places always has 32-bit positions, so the buffers form needs no count check of its
// own beyond the length check.
static_assert(std::uint64_t{kCellPlacesPerBlock} * kMaxCellBlocksLimit <= kMaxListCount);

// The last step of the failure key, a block without its bigger block (<sec-parallel-build>): a status record holds a
// key of step kIdStep to this one.
constexpr int kLastFailureStep = 7;

GridFailure ToBadInput() {
  return {.error = GridError::kBadInput};
}

GridFailure ToInternalFault() {
  return {.error = GridError::kInternalFault};
}

// Checks the frame: the first input check of every build and every check of a list.
std::expected<void, GridFailure> CheckGridFrame(GridFrame frame) {
  const bool is_origin_finite =
      math::IsFinite(frame.origin.x) && math::IsFinite(frame.origin.y) && math::IsFinite(frame.origin.z);
  const bool is_width_positive = math::IsFinite(frame.biggest_cell_width) && frame.biggest_cell_width > 0;
  const bool is_size_level_count_in_range = frame.size_level_count >= 1 && frame.size_level_count <= kMaxSizeLevels;
  // A frame not filled in, or one computed from a bad scene value; the caller fixes the frame.
  if (!is_origin_finite || !is_width_positive || !is_size_level_count_in_range) {
    return std::unexpected(ToBadInput());
  }
  // The smallest width must be a normal double, at least 2^-1022: every halving of the biggest width down to it is
  // then exact, and no position divided by it loses its cell to underflow. Below, halving rounds or gives 0. A frame
  // that small is a bad scene value too (biggest cells 2^-1008 m wide with 16 size levels, say).
  const auto smallest_size_level = static_cast<SizeLevel>(frame.size_level_count - 1);
  if (ComputeCellWidth(frame, smallest_size_level) < std::numeric_limits<double>::min()) {
    return std::unexpected(ToBadInput());
  }
  return {};
}

// Names the failure of a failure key with the cells ComputeFailureCells gives, as the GPU does for its smallest key.
// Steps 2 to 4 read only the sorted list, and step 1 only the frame and the input list.
GridFailure ToGridFailureOfKey(
    GridFrame frame,
    std::span<const CellId> input_cells,
    std::span<const CellId> sorted_cells,
    std::uint64_t failure_key
) {
  const CellCheck cell_check = ComputeFailureCells(
      frame,
      input_cells.data(),
      static_cast<std::uint32_t>(input_cells.size()),
      sorted_cells.data(),
      static_cast<std::uint32_t>(sorted_cells.size()),
      failure_key
  );
  // The key comes from a check that just failed on these lists, and ComputeFailureCells runs that check again: any
  // other answer is a fault in FluidiTy.
  if (cell_check.error == GridError::kNone || cell_check.error == GridError::kInternalFault) {
    return ToInternalFault();
  }
  return {.error = cell_check.error, .cell = cell_check.cell, .other_cell = cell_check.other_cell};
}

// <proc-build-grid> step 1: the key ComputeCellIdFailureKey gives every id in input order, the GPU's per-item rule;
// the first failure has the smallest key of the step. Requires: at most kMaxListCount ids.
std::expected<void, GridFailure> CheckInputIds(GridFrame frame, std::span<const CellId> input_cells) {
  for (std::size_t index = 0; index < input_cells.size(); ++index) {
    const std::uint64_t failure_key = ComputeCellIdFailureKey(frame, input_cells.data(), index);
    if (failure_key != kNoFailureKey) {
      // The list holds an id no grid of this frame can have; the caller made the list wrong.
      return std::unexpected(ToGridFailureOfKey(frame, input_cells, std::span<const CellId>(), failure_key));
    }
  }
  return {};
}

// One pass of a sort key of a cell: ToCellSortKey or ToBlockSortKey.
using SortKeyOfPass = std::uint64_t (*)(CellId cell, int pass) noexcept;

// Says whether `first` comes before `second` by the three passes of a sort key, the last pass compared first: the
// order the stable order of the passes gives, least significant first.
bool ComputeIsEarlierBySortKeys(CellId first, CellId second, SortKeyOfPass sort_key_of_pass) {
  for (int pass = kSortKeyPassCount - 1; pass >= 0; --pass) {
    const std::uint64_t first_key = sort_key_of_pass(first, pass);
    const std::uint64_t second_key = sort_key_of_pass(second, pass);
    if (first_key != second_key) {
      return first_key < second_key;
    }
  }
  return false;
}

// Sorts cells into the fixed order, in place, without allocating. Equal keys mean equal ids once step 1 has passed,
// so the unstable std::ranges::sort gives the bytes of the GPU's stable order.
void ApplyFixedCellOrder(std::span<CellId> cells) {
  std::ranges::sort(cells, [](CellId first, CellId second) {
    return ComputeIsEarlierBySortKeys(first, second, ToCellSortKey);
  });
}

// The failure key of the first failing position of one step over a sorted list, or kNoFailureKey: the key
// ComputeSortedCellFailureKey gives each position in order, the GPU's per-item rule.
std::uint64_t FindFirstFailureKey(std::span<const CellId> sorted_cells, int step) {
  const auto count = static_cast<std::uint32_t>(sorted_cells.size());
  for (std::uint32_t position = 0; position < count; ++position) {
    const std::uint64_t failure_key = ComputeSortedCellFailureKey(sorted_cells.data(), count, position, step);
    if (failure_key != kNoFailureKey) {
      return failure_key;
    }
  }
  return kNoFailureKey;
}

// Says whether a list shares memory with the cell ids that the sorted copy will be written to. Compared with
// std::less, which orders any two pointers, so lists of unrelated arrays compare safely.
bool ComputeIsListInCellIds(std::span<const CellId> list, const Buffer<CellId>& cell_ids) {
  if (list.empty() || cell_ids.size() == 0) {
    return false;
  }
  const std::less<> is_before;
  const CellId* const list_end = list.data() + list.size();
  const CellId* const cell_ids_end = cell_ids.data() + cell_ids.size();
  return is_before(list.data(), cell_ids_end) && is_before(cell_ids.data(), list_end);
}

// The step from a cell to the place beside its high side along x, y and z. The low side's step is the negation.
constexpr std::array<CellCoord, 3> kHighSideSteps = {
    CellCoord{.x = 1, .y = 0, .z = 0},
    CellCoord{.x = 0, .y = 1, .z = 0},
    CellCoord{.x = 0, .y = 0, .z = 1},
};

// Computes the place of a cell's own size level one `step` beside it, on the raw coordinate: -2^18..2^18 on each axis,
// so at most one place beyond a coordinate limit. `step` is one of kHighSideSteps or its negation.
CellCoord ComputeCoordBeside(CellId cell, CellCoord step) {
  const CellCoord cell_coord = ToCellCoord(cell);
  return {
      .x = cell_coord.x + step.x,
      .y = cell_coord.y + step.y,
      .z = cell_coord.z + step.z,
  };
}

// Says whether a cell of the cell's own size level is listed at the place one `step` beside it.
// FindSortedCellAt makes no id for a place beyond a coordinate limit, where no cell of that size level can be.
// Requires: sorted_cells is in the fixed order and passed CheckSortedCells with siblings; `cell` is one of them.
bool ComputeIsPlaceActive(std::span<const CellId> sorted_cells, CellId cell, CellCoord step) {
  const auto count = static_cast<std::uint32_t>(sorted_cells.size());
  return FindSortedCellAt(sorted_cells.data(), count, ToSizeLevel(cell), ComputeCoordBeside(cell, step)) != kNoIndex;
}

// Says whether the place of the cell's own size level one `step` beside it is split: whether the small cell of that
// place on the cell's side is listed. Along the axis that small cell is 2 N + 1 beside the low side and 2 N beside
// the high side; across it is the low half, 2 c. It is one of the four small cells that touch the side, and the size
// step rule makes all four active once one is. The small cells of a place beyond a coordinate limit are beyond it
// too. FindSortedCellAt makes no id for them: packed, such an id would name another cell near the middle of the
// range, such as coordinate 0 or 1 for the small cells beyond kMaxCoord or -kMaxCoord.
// Requires: as ComputeIsPlaceActive.
bool ComputeIsPlaceSplit(std::span<const CellId> sorted_cells, CellId cell, CellCoord step) {
  const int size_level_number = static_cast<int>(ToSizeLevel(cell));
  if (size_level_number + 1 >= kMaxSizeLevels) {
    return false;
  }
  const CellCoord beside_coord = ComputeCoordBeside(cell, step);
  // |N| <= 2^18, so 2 N + 1 fits a 32-bit integer.
  const CellCoord small_coord = {
      .x = (2 * beside_coord.x) + (step.x < 0 ? 1 : 0),
      .y = (2 * beside_coord.y) + (step.y < 0 ? 1 : 0),
      .z = (2 * beside_coord.z) + (step.z < 0 ? 1 : 0),
  };
  const auto count = static_cast<std::uint32_t>(sorted_cells.size());
  const auto small_size_level = static_cast<SizeLevel>(size_level_number + 1);
  return FindSortedCellAt(sorted_cells.data(), count, small_size_level, small_coord) != kNoIndex;
}

// Computes how many face pieces the active cell at `position` of a sorted list keeps: <proc-count-grid> step 3. A face
// piece is kept once, with its smaller cell, and between two cells of one size level by the cell on its high side,
// which owns its low side.
// Low side: one face piece unless the place beside it is split, whose small cells keep their own. An active place,
// a place inside a bigger cell and an absent place each leave one with this cell: shared with a cell of the same
// size, a quarter of the bigger cell's side, or the grid's edge.
// High side: one face piece unless the place is active (that cell keeps it on its low side) or split.
// A place inside a bigger cell and an absent place give the same count on either side, so the count never tells them
// apart; only the kind of the face piece differs, and kinds exist only in a built grid's tables. At most 9 binary
// searches: the high side's split search is skipped when the place is active, and every split search at size level 15.
// Requires: as ComputeIsPlaceActive; position < sorted_cells.size().
int ComputeFacePieceCountOfCell(std::span<const CellId> sorted_cells, std::uint32_t position) {
  const CellId cell = sorted_cells[position];
  int face_piece_count = 0;
  for (const CellCoord high_step : kHighSideSteps) {
    const CellCoord low_step = {.x = -high_step.x, .y = -high_step.y, .z = -high_step.z};
    if (!ComputeIsPlaceSplit(sorted_cells, cell, low_step)) {
      ++face_piece_count;
    }
    const bool is_high_place_kept_elsewhere =
        ComputeIsPlaceActive(sorted_cells, cell, high_step) || ComputeIsPlaceSplit(sorted_cells, cell, high_step);
    if (!is_high_place_kept_elsewhere) {
      ++face_piece_count;
    }
  }
  return face_piece_count;
}

}  // namespace

std::expected<void, GridFailure> CheckGridInputs(const GridInputs& inputs) {
  // Steps: 1 the frame  2 the version  3 the shape of the list
  // Step 1.
  if (const std::expected<void, GridFailure> frame_checked = CheckGridFrame(inputs.frame); !frame_checked.has_value()) {
    return frame_checked;
  }
  // Step 2: version 0 means bound to no grid; the caller passes the version before plus 1.
  if (inputs.version.value == 0) {
    return std::unexpected(ToBadInput());
  }
  // Step 3: one list only, and a count its buffer holds. A count without a buffer counts entries of no buffer.
  const bool has_buffer_list = inputs.active_cell_buffer != nullptr;
  if (has_buffer_list && !inputs.active_cells.empty()) {
    return std::unexpected(ToBadInput());
  }
  const std::size_t buffer_size = has_buffer_list ? inputs.active_cell_buffer->size() : 0;
  if (inputs.active_cell_buffer_count > buffer_size) {
    return std::unexpected(ToBadInput());
  }
  return {};
}

std::expected<void, GridFailure> CheckLentBuilderBuffers(const GridBuffers& buffers) {
  // Each failure is the caller's: buffers moved from, or arrays pointed at builder buffers that are not the ones lent
  // by BuildGridBuffers. Nothing has been written, so the caller can build its buffers again.
  const GridBuilderBuffers* const builder_buffers = ToGridBuffersArrays(buffers).builder_buffers;
  // (1) and (2): ComputeHasLoanOf also fails a loan count that is null.
  if (builder_buffers == nullptr || !ComputeHasLoanOf(buffers, *builder_buffers)) {
    return std::unexpected(ToBadInput());
  }
  // (3) cannot fail once (2) passes: BuildGridBuffers checked the room when it lent them, and builder buffers change
  // their room only by a move or an assignment, which moves the count away or leaves it behind, so (2) fails first.
  // It stays so the room the copy below relies on is read here, not inferred from how the count moves.
  if (builder_buffers->max_cell_blocks() < buffers.max_cell_blocks()) {
    return std::unexpected(ToBadInput());
  }
  return {};
}

std::expected<void, GridFailure> CheckSortedCells(std::span<const CellId> sorted_cells, SiblingCheck sibling_check) {
  // Steps: 2 overlaps  3 siblings  4 the size step rule, as numbered in <proc-build-grid>. Each step runs to its end
  // before the next, and stops at its first failing position, so the failure found has the smallest key.
  // Every caller checked the count against kMaxListCount; a longer list here is a fault in FluidiTy.
  if (sorted_cells.size() > kMaxListCount) {
    return std::unexpected(ToInternalFault());
  }
  // Step 2: a cell listed twice, or inside a bigger listed cell.
  std::uint64_t failure_key = FindFirstFailureKey(sorted_cells, kOverlapStep);
  // Step 3: every small cell of every split cell is present.
  if (failure_key == kNoFailureKey && sibling_check == SiblingCheck::kRequire) {
    failure_key = FindFirstFailureKey(sorted_cells, kSiblingStep);
  }
  // Step 4: no two touching cells two or more size levels apart.
  if (failure_key == kNoFailureKey) {
    failure_key = FindFirstFailureKey(sorted_cells, kSizeStepRuleStep);
  }
  if (failure_key == kNoFailureKey) {
    return {};
  }
  // The caller's list breaks a rule of <sec-cell-sizes>. The frame and the input list are not read for steps 2 to 4.
  return std::unexpected(ToGridFailureOfKey(GridFrame(), std::span<const CellId>(), sorted_cells, failure_key));
}

bool ComputeIsEarlierBlock(CellId first, CellId second) {
  return ComputeIsEarlierBySortKeys(first, second, ToBlockSortKey);
}

void ApplyFixedBlockOrder(std::span<CellId> cells) {
  std::ranges::sort(cells, ComputeIsEarlierBlock);
}

GridFailure ToGridFailure(const GridBuildStatus& status) {
  const bool has_failure_key = status.failure_key != kNoFailureKey;
  const bool has_error = status.error != GridError::kNone;
  if (!has_failure_key && !has_error) {
    return {};
  }
  // The record the GPU just wrote contradicts itself: a key without an error, an error without a key, or a key of no
  // step of <sec-parallel-build>. No input can cause it; it is a fault in FluidiTy.
  // Step 5 has two keys only: detail 0, the room's cell places, and detail 1, the sort count (kSortCountFaultKey).
  const int failure_step = ToFailureStep(status.failure_key);
  const bool is_known_step = failure_step >= kIdStep && failure_step <= kLastFailureStep;
  const bool is_unknown_cell_places_key = failure_step == kCellPlacesStep && ToFailureDetail(status.failure_key) > 1;
  if (!has_failure_key || !has_error || !is_known_step || is_unknown_cell_places_key) {
    return ToInternalFault();
  }
  // An internal fault names no cell, whatever the record holds.
  if (status.error == GridError::kInternalFault) {
    return ToInternalFault();
  }
  return {.error = status.error, .cell = status.cell, .other_cell = status.other_cell};
}

std::span<const CellId> ToInputCells(const GridInputs& inputs) {
  if (inputs.active_cell_buffer != nullptr) {
    return {inputs.active_cell_buffer->data(), inputs.active_cell_buffer_count};
  }
  return inputs.active_cells;
}

#if defined(FLUIDITY_HAS_GPU)

std::expected<void, GridFailure> ApplyListChecksInParallel(
    GridFrame frame,
    std::span<const CellId> input_cells,
    GridBuffers& buffers,
    SiblingCheck sibling_check
) {
  // Steps: 1 the arrays  2 P1, the reset  3 P2, the ids  4 P3, the sort  5 P4 to P6, the sorted list
  // Step 1: BuildGridBuilderBuffers and BuildGridBuffers built every array from the estimate's entries for a room at
  // least the buffers', so an array too small for the list, or a missing one, contradicts them.
  GridBuffersArrays& arrays = ToGridBuffersArrays(buffers);
  GridBuilderBuffersArrays& builder_arrays = ToGridBuilderBuffersArrays(*arrays.builder_buffers);
  const std::size_t count = input_cells.size();
  const bool has_room_for_list = builder_arrays.cell_ids.size() >= count && builder_arrays.sort_keys.size() >= count
      && builder_arrays.order.size() >= count && builder_arrays.second_order.size() >= count;
  if (!has_room_for_list || !builder_arrays.order_buffers.has_value() || arrays.build_status.size() < 1) {
    return std::unexpected(ToInternalFault());
  }
  const CellId* const input = input_cells.data();
  const auto input_count = static_cast<std::uint32_t>(count);
  CellId* const sorted_cells = builder_arrays.cell_ids.data();
  GridBuildStatus* const status = arrays.build_status.data();
  // A failed GPU call leaves the buffers unspecified, and the run stops (kRunOnCallFailed). ToGridFailure maps a size
  // error of base/ to kInternalFault: the sizes were checked above.
  const auto to_call_failure = [](RunOnCallStatus call_status) {
    return std::unexpected(ToGridFailure(ToBufferError(call_status), 0));
  };

  // Step 2: P1, the reset. Nothing is written from the CPU.
  if (const RunOnCallStatus call_status = ApplyBuildStatusResetOnGpu(status); call_status != RunOnCallStatus::kDone) {
    return to_call_failure(call_status);
  }
  // Step 3: P2.
  if (const RunOnCallStatus call_status = CheckCellIdsOnGpu(frame, input, input_count, status);
      call_status != RunOnCallStatus::kDone) {
    return to_call_failure(call_status);
  }
  // Step 4: P3, three passes, the least significant first, alternating the two orders so that no pass reads the order
  // it writes; the last pass leaves the order in `order`. A list that failed P2 is sorted too: its keys are ordinary
  // numbers, and P4 to P6 then read the failure and do nothing.
  const std::array<const Buffer<std::uint32_t>*, kSortKeyPassCount> previous_orders = {
      nullptr,
      &builder_arrays.order,
      &builder_arrays.second_order,
  };
  const std::array<Buffer<std::uint32_t>*, kSortKeyPassCount> pass_orders = {
      &builder_arrays.order,
      &builder_arrays.second_order,
      &builder_arrays.order,
  };
  for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
    const auto pass_index = static_cast<std::size_t>(pass);
    if (const RunOnCallStatus call_status =
            WriteCellSortKeysOnGpu(input, input_count, pass, builder_arrays.sort_keys.data());
        call_status != RunOnCallStatus::kDone) {
      return to_call_failure(call_status);
    }
    const std::expected<void, BufferError> ordered = BuildStableOrder(
        builder_arrays.sort_keys,
        count,
        previous_orders.at(pass_index),
        *pass_orders.at(pass_index),
        *builder_arrays.order_buffers
    );
    if (!ordered.has_value()) {
      return std::unexpected(ToGridFailure(ordered.error(), 0));
    }
  }
  if (const RunOnCallStatus call_status =
          CopySortedCellsOnGpu(input, builder_arrays.order.data(), input_count, sorted_cells);
      call_status != RunOnCallStatus::kDone) {
    return to_call_failure(call_status);
  }
  // Step 5: P4, P5 when siblings are required, P6.
  const std::array<int, 3> sorted_steps = {kOverlapStep, kSiblingStep, kSizeStepRuleStep};
  for (const int step : sorted_steps) {
    if (step == kSiblingStep && sibling_check == SiblingCheck::kSkip) {
      continue;
    }
    if (const RunOnCallStatus call_status = CheckSortedCellsOnGpu(sorted_cells, input_count, step, status);
        call_status != RunOnCallStatus::kDone) {
      return to_call_failure(call_status);
    }
  }
  return {};
}

#endif

namespace {

#if defined(FLUIDITY_HAS_GPU)

// <proc-build-grid-in-parallel> steps P1 (the reset) to P6 and the failure half of P12 on the GPU, for a list whose
// input checks passed: ApplyListChecksInParallel, then one thread names the smallest key, then the one read of the
// status record. On success the first `input_cells.size()` cell ids hold the list in the fixed order.
// Requires: buffers.run_on() is the GPU; input_cells is GPU memory, or empty; CheckCellListInputs passed.
std::expected<std::uint32_t, GridFailure> BuildSortedCellListOnGpu(
    GridFrame frame,
    std::span<const CellId> input_cells,
    GridBuffers& buffers,
    SiblingCheck sibling_check
) {
  // Steps: 1 P1 to P6  2 P12, the failure, and the read of the record
  // Step 1.
  if (const std::expected<void, GridFailure> checked =
          ApplyListChecksInParallel(frame, input_cells, buffers, sibling_check);
      !checked.has_value()) {
    return std::unexpected(checked.error());
  }
  // Step 2: P12, the failure half, then the one transfer of the check. ApplyListChecksInParallel checked the arrays.
  GridBuffersArrays& arrays = ToGridBuffersArrays(buffers);
  const CellId* const input = input_cells.data();
  const auto input_count = static_cast<std::uint32_t>(input_cells.size());
  const CellId* const sorted_cells = ToGridBuilderBuffersArrays(*arrays.builder_buffers).cell_ids.data();
  if (const RunOnCallStatus call_status =
          WriteFailureCellsOnGpu(frame, input, input_count, sorted_cells, input_count, arrays.build_status.data());
      call_status != RunOnCallStatus::kDone) {
    return std::unexpected(ToGridFailure(ToBufferError(call_status), 0));
  }
  GridBuildStatus read_status{};
  const std::expected<void, BufferError> read =
      ReadBuffer(arrays.build_status, 0, std::span<GridBuildStatus>(&read_status, 1));
  if (!read.has_value()) {
    return std::unexpected(ToGridFailure(read.error(), sizeof(GridBuildStatus)));
  }
  const GridFailure failure = ToGridFailure(read_status);
  if (failure.error != GridError::kNone) {
    return std::unexpected(failure);
  }
  return input_count;
}

#endif

}  // namespace

std::expected<void, GridFailure> CheckCellListInputs(const GridInputs& inputs, const GridBuffers& buffers) {
  // Steps: 1 the frame, the version and the list shape  2 the lent builder buffers and the list's memory  3 the length
  // Step 1.
  if (const std::expected<void, GridFailure> inputs_checked = CheckGridInputs(inputs); !inputs_checked.has_value()) {
    return inputs_checked;
  }
  // Step 2.
  if (const std::expected<void, GridFailure> loan_checked = CheckLentBuilderBuffers(buffers);
      !loan_checked.has_value()) {
    return loan_checked;
  }
  const std::span<const CellId> input_cells = ToInputCells(inputs);
  const Buffer<CellId>& cell_ids = ToGridBuilderBuffersArrays(*ToGridBuffersArrays(buffers).builder_buffers).cell_ids;
  // The sorted copy would overwrite the list while it is read; the caller passes a list of its own.
  if (ComputeIsListInCellIds(input_cells, cell_ids)) {
    return std::unexpected(ToBadInput());
  }
  // The list must be on the buffers' processor, which only a caller's bug breaks; an empty list has none.
  const bool has_buffer_list = inputs.active_cell_buffer != nullptr;
  const bool is_list_on_processor = input_cells.empty()
      || (has_buffer_list ? inputs.active_cell_buffer->run_on() == buffers.run_on() : buffers.run_on() == RunOn::kCpu);
  assert(is_list_on_processor);
  static_cast<void>(is_list_on_processor);
  // Step 3: the length, before any id is read: a list longer than the room's cell places cannot fit, whatever its ids.
  // The caller raises the room. It also bounds every list to 32-bit positions: 64 * kMaxCellBlocksLimit < 2^32, so a
  // span of any length is kTooManyCellBlocks here, as the length check of a build promises, never bad input.
  const std::uint64_t cell_place_count = std::uint64_t{kCellPlacesPerBlock} * buffers.max_cell_blocks();
  if (input_cells.size() > cell_place_count) {
    return std::unexpected(GridFailure{.error = GridError::kTooManyCellBlocks});
  }
  return {};
}

std::expected<std::uint32_t, GridFailure>
BuildSortedCellList(const GridInputs& inputs, GridBuffers& buffers, SiblingCheck sibling_check) {
  // Steps: 1 the input checks, then the ids (<proc-build-grid> step 1)  2 the copy into the cell ids and its sort
  // 3 the checks of the sorted list (<proc-build-grid> steps 2 to 4). On the GPU, <proc-build-grid-in-parallel> steps
  // P1 to P6 after the input checks.
  // Step 1: the input checks, in the order of <sec-parallel-build>, before any id is read.
  if (const std::expected<void, GridFailure> inputs_checked = CheckCellListInputs(inputs, buffers);
      !inputs_checked.has_value()) {
    return std::unexpected(inputs_checked.error());
  }
  const std::span<const CellId> input_cells = ToInputCells(inputs);
  if (buffers.run_on() != RunOn::kCpu) {
#if defined(FLUIDITY_HAS_GPU)
    return BuildSortedCellListOnGpu(inputs.frame, input_cells, buffers, sibling_check);
#else
    // Without GPU code BuildGridBuffers never builds GPU buffers, so no caller reaches this.
    return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
#endif
  }
  Buffer<CellId>& cell_ids = ToGridBuilderBuffersArrays(*ToGridBuffersArrays(buffers).builder_buffers).cell_ids;
  // The builder buffers hold 64 cell ids per block of a room at least the buffers': fewer contradicts their builder.
  if (cell_ids.size() < input_cells.size()) {
    return std::unexpected(ToInternalFault());
  }
  // <proc-build-grid> step 1: the ids, in input order.
  if (const std::expected<void, GridFailure> ids_checked = CheckInputIds(inputs.frame, input_cells);
      !ids_checked.has_value()) {
    return std::unexpected(ids_checked.error());
  }

  // Step 2: the copy, then the sort in place. The sizes and the overlap were checked above, so a copy error of base/
  // is a fault in FluidiTy (ToGridFailure maps kSizeMismatch to kInternalFault).
  const std::expected<void, BufferError> copied = inputs.active_cell_buffer != nullptr
      ? CopyBuffer(*inputs.active_cell_buffer, 0, input_cells.size(), cell_ids, 0)
      : WriteBuffer(input_cells, cell_ids, 0);
  if (!copied.has_value()) {
    return std::unexpected(ToGridFailure(copied.error(), input_cells.size_bytes()));
  }
  const std::span<CellId> sorted_cells(cell_ids.data(), input_cells.size());
  ApplyFixedCellOrder(sorted_cells);

  // Step 3.
  if (const std::expected<void, GridFailure> sorted_checked = CheckSortedCells(sorted_cells, sibling_check);
      !sorted_checked.has_value()) {
    return std::unexpected(sorted_checked.error());
  }
  return static_cast<std::uint32_t>(sorted_cells.size());
}

std::expected<void, GridFailure> CheckSizeStepRule(GridFrame frame, std::span<const CellId> active_cells) {
  // Steps: 1 the frame and the count  2 the ids  3 a sorted copy and its checks
  // Step 1: the input checks of a list without buffers: no version and no length.
  if (const std::expected<void, GridFailure> frame_checked = CheckGridFrame(frame); !frame_checked.has_value()) {
    return frame_checked;
  }
  if (active_cells.size() > kMaxListCount) {
    return std::unexpected(ToBadInput());
  }
  // Step 2: <proc-build-grid> step 1.
  if (const std::expected<void, GridFailure> ids_checked = CheckInputIds(frame, active_cells);
      !ids_checked.has_value()) {
    return ids_checked;
  }
  // Step 3: <proc-build-grid> steps 2 and 4 on one CPU copy, the one allocation of this form. It is allocated
  // without throwing, since the library runs without exceptions: a copy that does not fit is kOutOfMemory.
  if (active_cells.empty()) {
    return {};
  }
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a std::vector cannot report a failed allocation without exceptions.
  const std::unique_ptr<CellId[]> copy(new (std::nothrow) CellId[active_cells.size()]);
  if (copy == nullptr) {
    return std::unexpected(GridFailure{.error = GridError::kOutOfMemory, .magnitude = active_cells.size_bytes()});
  }
  const std::span<CellId> sorted_cells(copy.get(), active_cells.size());
  std::ranges::copy(active_cells, sorted_cells.begin());
  ApplyFixedCellOrder(sorted_cells);
  return CheckSortedCells(sorted_cells, SiblingCheck::kSkip);
}

std::expected<void, GridFailure> CheckSizeStepRule(const GridInputs& inputs, GridBuffers& buffers) {
  const std::expected<std::uint32_t, GridFailure> built = BuildSortedCellList(inputs, buffers, SiblingCheck::kSkip);
  if (!built.has_value()) {
    return std::unexpected(built.error());
  }
  return {};
}

std::expected<GridCounts, GridFailure> ComputeGridCounts(const GridInputs& inputs) {
  // Steps of <proc-count-grid>: 1 check and sort the list  2 count the active cells  3 count the face pieces
  // 4 count the blocks
  // Step 1: the input checks of a build, without the length check, since there is no room yet.
  if (const std::expected<void, GridFailure> inputs_checked = CheckGridInputs(inputs); !inputs_checked.has_value()) {
    return std::unexpected(inputs_checked.error());
  }
  const bool has_buffer_list = inputs.active_cell_buffer != nullptr;
  const std::span<const CellId> input_cells = has_buffer_list
      ? std::span<const CellId>(inputs.active_cell_buffer->data(), inputs.active_cell_buffer_count)
      : inputs.active_cells;
  // The list must be in CPU memory, which only a caller's bug breaks; an empty list has no processor.
  const bool is_list_on_cpu =
      input_cells.empty() || !has_buffer_list || inputs.active_cell_buffer->run_on() == RunOn::kCpu;
  assert(is_list_on_cpu);
  static_cast<void>(is_list_on_cpu);
  // No 32-bit position indexes a longer list, and no grid holds one; the scene needs bigger cells.
  if (input_cells.size() > kMaxListCount) {
    return std::unexpected(ToBadInput());
  }
  // <proc-build-grid> step 1, before anything is allocated.
  if (const std::expected<void, GridFailure> ids_checked = CheckInputIds(inputs.frame, input_cells);
      !ids_checked.has_value()) {
    return std::unexpected(ids_checked.error());
  }
  GridCounts counts{};
  if (input_cells.empty()) {
    return counts;
  }
  // The one allocation: the n ids, then the split cells of step 4. A split cell has 8 small cells, each active or
  // split, so n + s cells hold 8 s small cells and 7 s <= n - 1: at most floor(n / 7) split cells. It is allocated
  // without throwing, since the library runs without exceptions.
  const std::size_t active_count = input_cells.size();
  const std::size_t copy_room = active_count + (active_count / 7);
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a std::vector cannot report a failed allocation without exceptions.
  const std::unique_ptr<CellId[]> copy(new (std::nothrow) CellId[copy_room]);
  if (copy == nullptr) {
    return std::unexpected(GridFailure{.error = GridError::kOutOfMemory, .magnitude = copy_room * sizeof(CellId)});
  }
  const std::span<CellId> sorted_cells(copy.get(), active_count);
  std::ranges::copy(input_cells, sorted_cells.begin());
  ApplyFixedCellOrder(sorted_cells);
  if (const std::expected<void, GridFailure> sorted_checked = CheckSortedCells(sorted_cells, SiblingCheck::kRequire);
      !sorted_checked.has_value()) {
    return std::unexpected(sorted_checked.error());
  }

  // Step 2.
  for (const CellId cell : sorted_cells) {
    ++counts.active_cells_per_size_level[static_cast<std::size_t>(ToSizeLevel(cell))];
  }
  counts.active_cells = active_count;

  // Step 3.
  for (std::uint32_t position = 0; position < active_count; ++position) {
    counts.face_pieces += static_cast<std::uint64_t>(ComputeFacePieceCountOfCell(sorted_cells, position));
  }

  // Step 4: every split cell is added once, by the active cell at its low corner (ComputeSplitCellCountToAdd).
  std::size_t used_count = active_count;
  for (std::size_t position = 0; position < active_count; ++position) {
    const CellId cell = sorted_cells[position];
    const int added_count = ComputeSplitCellCountToAdd(cell);
    // Step 1 checked the siblings, so the bound above holds; more split cells is a fault in FluidiTy.
    if (used_count + static_cast<std::size_t>(added_count) > copy_room) {
      return std::unexpected(ToInternalFault());
    }
    const int size_level_number = static_cast<int>(ToSizeLevel(cell));
    for (int added = 1; added <= added_count; ++added) {
      copy[used_count] = ComputeBigCellAtSizeLevel(cell, static_cast<SizeLevel>(size_level_number - added));
      ++used_count;
    }
  }
  const std::span<CellId> cells_by_block(copy.get(), used_count);
  ApplyFixedBlockOrder(cells_by_block);
  std::uint64_t block_count = 0;
  for (std::size_t position = 0; position < used_count; ++position) {
    const bool is_new_block = position == 0
        || ToCellBlockId(cells_by_block[position]).value != ToCellBlockId(cells_by_block[position - 1]).value;
    if (is_new_block) {
      ++block_count;
    }
  }
  // Each block of size level 1 or more holds the small cells of a split cell of the size level above, and no two blocks
  // share one; each block of size level 0 holds a size-level-0 cell. So the blocks are at most the size-level-0 cells
  // plus the split cells, which is at most n. More is a fault in FluidiTy.
  if (block_count > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected(ToInternalFault());
  }
  counts.cell_blocks = static_cast<std::uint32_t>(block_count);
  return counts;
}

}  // namespace fluidity::grid
