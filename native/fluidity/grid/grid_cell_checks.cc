#include "fluidity/grid/grid_cell_checks.h"

#include <algorithm>
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

namespace fluidity::grid {
namespace {

// The most cells a list may have: positions in a list are 32-bit, as in the GPU's orders.
constexpr std::uint64_t kMaxListCount = std::numeric_limits<std::uint32_t>::max();
// A list within the room's cell places always has 32-bit positions, so the buffers form needs no count check of its
// own beyond the length check.
static_assert(std::uint64_t{kCellPlacesPerBlock} * kMaxCellBlocksLimit <= kMaxListCount);

// The failure keys' steps of <proc-build-grid> (<sec-parallel-build>).
constexpr int kIdStep = 1;
constexpr int kOverlapStep = 2;
constexpr int kSiblingStep = 3;
constexpr int kSizeStepRuleStep = 4;

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

// <proc-build-grid> step 1: CheckCellId on every id in input order; the first failure has the smallest key of the
// step. Requires: at most kMaxListCount ids.
std::expected<void, GridFailure> CheckInputIds(GridFrame frame, std::span<const CellId> input_cells) {
  for (std::size_t index = 0; index < input_cells.size(); ++index) {
    if (CheckCellId(frame, input_cells[index]) != GridError::kNone) {
      // The list holds an id no grid of this frame can have; the caller made the list wrong.
      return std::unexpected(
          ToGridFailureOfKey(frame, input_cells, std::span<const CellId>(), ToFailureKey(kIdStep, index, 0))
      );
    }
  }
  return {};
}

// Says whether `first` comes before `second` by the three passes of ToCellSortKey, the last pass compared first: the
// order the stable order of the passes gives, least significant first.
bool ComputeIsEarlierBySortKeys(CellId first, CellId second) {
  for (int pass = kSortKeyPassCount - 1; pass >= 0; --pass) {
    const std::uint64_t first_key = ToCellSortKey(first, pass);
    const std::uint64_t second_key = ToCellSortKey(second, pass);
    if (first_key != second_key) {
      return first_key < second_key;
    }
  }
  return false;
}

// Sorts cells into the fixed order, in place, without allocating. Equal keys mean equal ids once step 1 has passed,
// so the unstable std::ranges::sort gives the bytes of the GPU's stable order.
void ApplyFixedCellOrder(std::span<CellId> cells) {
  std::ranges::sort(cells, ComputeIsEarlierBySortKeys);
}

// The failure key of the first failing position of one check over a sorted list, or kNoFailureKey.
template <typename CheckAt>
std::uint64_t FindFirstFailureKey(std::span<const CellId> sorted_cells, int step, CheckAt check_at) {
  const auto count = static_cast<std::uint32_t>(sorted_cells.size());
  for (std::uint32_t position = 0; position < count; ++position) {
    const CellCheck cell_check = check_at(sorted_cells.data(), count, position);
    if (cell_check.error != GridError::kNone) {
      return ToFailureKey(step, position, cell_check.detail);
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
  std::uint64_t failure_key = FindFirstFailureKey(sorted_cells, kOverlapStep, CheckOverlapAt);
  // Step 3: every small cell of every split cell is present.
  if (failure_key == kNoFailureKey && sibling_check == SiblingCheck::kRequire) {
    failure_key = FindFirstFailureKey(sorted_cells, kSiblingStep, CheckSiblingsAt);
  }
  // Step 4: no two touching cells two or more size levels apart.
  if (failure_key == kNoFailureKey) {
    failure_key = FindFirstFailureKey(sorted_cells, kSizeStepRuleStep, CheckSizeStepAt);
  }
  if (failure_key == kNoFailureKey) {
    return {};
  }
  // The caller's list breaks a rule of <sec-cell-sizes>. The frame and the input list are not read for steps 2 to 4.
  return std::unexpected(ToGridFailureOfKey(GridFrame(), std::span<const CellId>(), sorted_cells, failure_key));
}

std::expected<std::uint32_t, GridFailure>
BuildSortedCellList(const GridInputs& inputs, GridBuffers& buffers, SiblingCheck sibling_check) {
  // Steps: 1 the input checks, then the ids (<proc-build-grid> step 1)  2 the copy into the cell ids and its sort
  // 3 the checks of the sorted list (<proc-build-grid> steps 2 to 4)
  // Step 1: the input checks, in the order of <sec-parallel-build>, before any id is read.
  if (const std::expected<void, GridFailure> inputs_checked = CheckGridInputs(inputs); !inputs_checked.has_value()) {
    return std::unexpected(inputs_checked.error());
  }
  if (const std::expected<void, GridFailure> loan_checked = CheckLentBuilderBuffers(buffers);
      !loan_checked.has_value()) {
    return std::unexpected(loan_checked.error());
  }
  const bool has_buffer_list = inputs.active_cell_buffer != nullptr;
  const std::span<const CellId> input_cells = has_buffer_list
      ? std::span<const CellId>(inputs.active_cell_buffer->data(), inputs.active_cell_buffer_count)
      : inputs.active_cells;
  Buffer<CellId>& cell_ids = ToGridBuilderBuffersArrays(*ToGridBuffersArrays(buffers).builder_buffers).cell_ids;
  // The sorted copy would overwrite the list while it is read; the caller passes a list of its own.
  if (ComputeIsListInCellIds(input_cells, cell_ids)) {
    return std::unexpected(ToBadInput());
  }
  // Requires: one processor. An empty list has none.
  const bool is_list_on_processor = input_cells.empty()
      || (has_buffer_list ? inputs.active_cell_buffer->run_on() == buffers.run_on() : buffers.run_on() == RunOn::kCpu);
  assert(is_list_on_processor);
  static_cast<void>(is_list_on_processor);
  // The length, before any id is read: a list longer than the room's cell places cannot fit, whatever its ids. The
  // caller raises the room. It also bounds every list to 32-bit positions: 64 * kMaxCellBlocksLimit < 2^32, so a span
  // of any length is kTooManyCellBlocks here, as the length check of a build promises, never bad input.
  const std::uint64_t cell_place_count = std::uint64_t{kCellPlacesPerBlock} * buffers.max_cell_blocks();
  if (input_cells.size() > cell_place_count) {
    return std::unexpected(GridFailure{.error = GridError::kTooManyCellBlocks});
  }
  // The library has no GPU form of this check. A list on the GPU is never read back to the CPU for it, so the
  // function stops here without reading an id.
  if (buffers.run_on() != RunOn::kCpu) {
    return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
  }
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
  const std::expected<void, BufferError> copied = has_buffer_list
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

}  // namespace fluidity::grid
