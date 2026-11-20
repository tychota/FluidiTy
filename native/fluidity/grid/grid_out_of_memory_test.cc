// Test-only: reaches the allocations the grid makes outside a RunMemory, by replacing the nothrow forms of the
// global allocation functions in this test program alone. A replacement holds for the whole program, so these tests
// live in a program of their own. The replacements forward to the ordinary forms, so the memory they give is freed by
// the ordinary delete the code calls, and they fail only while a test asks them to. The ordinary forms throw when the
// machine itself runs out of memory, and inside these noexcept replacements that ends the test program instead of
// returning null: acceptable for a test program, whose lists need a few bytes.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <new>
#include <span>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"

namespace {

// What the replaced nothrow forms do: count every call, and give no memory while their failure is asked for.
struct NothrowAllocationControl {
  bool is_array_failure_asked = false;
  bool is_single_failure_asked = false;
  int array_call_count = 0;
  int single_call_count = 0;
};

NothrowAllocationControl& ToAllocationControl() {
  static NothrowAllocationControl control;
  return control;
}

}  // namespace

// The replaceable nothrow forms ([new.delete.single], [new.delete.array]), each with the nothrow delete that frees
// what it gives. The ordinary new and delete stay the library's own.
// NOLINTNEXTLINE(readability-inconsistent-declaration-parameter-name): the library names them differently.
void* operator new(std::size_t byte_count, const std::nothrow_t& /*nothrow*/) noexcept {
  NothrowAllocationControl& control = ToAllocationControl();
  ++control.single_call_count;
  if (control.is_single_failure_asked) {
    return nullptr;
  }
  return ::operator new(byte_count);
}

// NOLINTNEXTLINE(readability-inconsistent-declaration-parameter-name): the library names them differently.
void* operator new[](std::size_t byte_count, const std::nothrow_t& /*nothrow*/) noexcept {
  NothrowAllocationControl& control = ToAllocationControl();
  ++control.array_call_count;
  if (control.is_array_failure_asked) {
    return nullptr;
  }
  return ::operator new[](byte_count);
}

// NOLINTNEXTLINE(readability-inconsistent-declaration-parameter-name): the library names them differently.
void operator delete(void* memory, const std::nothrow_t& /*nothrow*/) noexcept {
  ::operator delete(memory);
}

// NOLINTNEXTLINE(readability-inconsistent-declaration-parameter-name): the library names them differently.
void operator delete[](void* memory, const std::nothrow_t& /*nothrow*/) noexcept {
  ::operator delete[](memory);
}

namespace fluidity::grid {
namespace {

// A frame with its origin at 0, biggest cells 1 m wide and 3 size levels.
GridFrame ToUnitFrame() {
  return {.origin = {.x = 0, .y = 0, .z = 0}, .biggest_cell_width = 1.0, .size_level_count = 3};
}

CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// Size level 0 {0,0,0}, size level 1 {2,0,0} beside it, and size level 0 {5,0,0}: a list the span form passes.
std::vector<CellId> BuildThreeCells() {
  return {ToCell(0, 0, 0, 0), ToCell(1, 2, 0, 0), ToCell(0, 5, 0, 0)};
}

// Whether the replaced nothrow forms fail.
enum class NothrowFailure : std::uint8_t {
  kOff,  // The replaced forms allocate.
  kAsked,  // The replaced forms give no memory.
};

// Starts counting the nothrow calls again and asks for (or stops asking for) their failure.
void ApplyNothrowFailure(NothrowFailure failure) {
  const bool is_failure_asked = failure == NothrowFailure::kAsked;
  ToAllocationControl() = {
      .is_array_failure_asked = is_failure_asked,
      .is_single_failure_asked = is_failure_asked,
      .array_call_count = 0,
      .single_call_count = 0,
  };
}

// 3 cells of 8 bytes: the copy asks for 24 bytes, in one nothrow array call, and names no cell.
TEST(CheckSizeStepRule, CopyNotAllocated_IsOutOfMemoryOfItsBytes) {
  const std::vector<CellId> cells = BuildThreeCells();
  ApplyNothrowFailure(NothrowFailure::kAsked);
  const std::expected<void, GridFailure> result = CheckSizeStepRule(ToUnitFrame(), cells);
  const int array_call_count = ToAllocationControl().array_call_count;
  ApplyNothrowFailure(NothrowFailure::kOff);
  EXPECT_EQ(array_call_count, 1);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, GridError::kOutOfMemory);
  EXPECT_EQ(result.error().magnitude, 24U);
  EXPECT_EQ(result.error().cell.value, 0U);
  EXPECT_EQ(result.error().other_cell.value, 0U);
}

// The failed call leaves nothing behind: the same list then passes, with its one allocation.
TEST(CheckSizeStepRule, CopyAllocatedAfterAFailedOne_Passes) {
  const std::vector<CellId> cells = BuildThreeCells();
  ApplyNothrowFailure(NothrowFailure::kAsked);
  ASSERT_FALSE(CheckSizeStepRule(ToUnitFrame(), cells).has_value());
  ApplyNothrowFailure(NothrowFailure::kOff);
  EXPECT_TRUE(CheckSizeStepRule(ToUnitFrame(), cells).has_value());
  EXPECT_EQ(ToAllocationControl().array_call_count, 1);
}

// Step 1 fails at input index 1 before the copy is made.
TEST(CheckSizeStepRule, BadId_AllocatesNothing) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0), CellId{0}};
  ApplyNothrowFailure(NothrowFailure::kAsked);
  const std::expected<void, GridFailure> result = CheckSizeStepRule(ToUnitFrame(), cells);
  const int array_call_count = ToAllocationControl().array_call_count;
  ApplyNothrowFailure(NothrowFailure::kOff);
  EXPECT_EQ(array_call_count, 0);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, GridError::kBadInput);
}

TEST(CheckSizeStepRule, EmptyList_AllocatesNothing) {
  ApplyNothrowFailure(NothrowFailure::kAsked);
  const std::expected<void, GridFailure> result = CheckSizeStepRule(ToUnitFrame(), std::span<const CellId>());
  const int array_call_count = ToAllocationControl().array_call_count;
  ApplyNothrowFailure(NothrowFailure::kOff);
  EXPECT_EQ(array_call_count, 0);
  EXPECT_TRUE(result.has_value());
}

// Size level 0 {0,0,0} beside the 8 size-level-1 cells of size level 0 {1,0,0}: a list the count passes.
std::vector<CellId> BuildOneSplitCell() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (std::int32_t z = 0; z < 2; ++z) {
    for (std::int32_t y = 0; y < 2; ++y) {
      for (std::int32_t x = 2; x < 4; ++x) {
        cells.push_back(ToCell(1, x, y, z));
      }
    }
  }
  return cells;
}

GridInputs ToInputs(std::span<const CellId> active_cells) {
  return {.frame = ToUnitFrame(), .active_cells = active_cells, .version = {.value = 1}};
}

// 9 cells: the copy has room for 9 + 9 / 7 = 10 ids of 8 bytes, 80 bytes, asked in one nothrow array call.
TEST(ComputeGridCounts, CopyNotAllocated_IsOutOfMemoryOfItsBytes) {
  const std::vector<CellId> cells = BuildOneSplitCell();
  ApplyNothrowFailure(NothrowFailure::kAsked);
  const std::expected<GridCounts, GridFailure> counts = ComputeGridCounts(ToInputs(cells));
  const int array_call_count = ToAllocationControl().array_call_count;
  ApplyNothrowFailure(NothrowFailure::kOff);
  EXPECT_EQ(array_call_count, 1);
  ASSERT_FALSE(counts.has_value());
  EXPECT_EQ(counts.error().error, GridError::kOutOfMemory);
  EXPECT_EQ(counts.error().magnitude, 80U);
  EXPECT_EQ(counts.error().cell.value, 0U);
  EXPECT_EQ(counts.error().other_cell.value, 0U);
}

// The failed call leaves nothing behind: the same list then counts, with its one allocation.
TEST(ComputeGridCounts, CopyAllocatedAfterAFailedOne_Counts) {
  const std::vector<CellId> cells = BuildOneSplitCell();
  ApplyNothrowFailure(NothrowFailure::kAsked);
  ASSERT_FALSE(ComputeGridCounts(ToInputs(cells)).has_value());
  ApplyNothrowFailure(NothrowFailure::kOff);
  const std::expected<GridCounts, GridFailure> counts = ComputeGridCounts(ToInputs(cells));
  EXPECT_EQ(ToAllocationControl().array_call_count, 1);
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(counts->face_pieces, 41U);
}

// The ids are checked before the copy is made.
TEST(ComputeGridCounts, BadId_AllocatesNothing) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0), CellId{0}};
  ApplyNothrowFailure(NothrowFailure::kAsked);
  const std::expected<GridCounts, GridFailure> counts = ComputeGridCounts(ToInputs(cells));
  const int array_call_count = ToAllocationControl().array_call_count;
  ApplyNothrowFailure(NothrowFailure::kOff);
  EXPECT_EQ(array_call_count, 0);
  ASSERT_FALSE(counts.has_value());
  EXPECT_EQ(counts.error().error, GridError::kBadInput);
}

TEST(ComputeGridCounts, EmptyList_AllocatesNothing) {
  ApplyNothrowFailure(NothrowFailure::kAsked);
  const std::expected<GridCounts, GridFailure> counts = ComputeGridCounts(ToInputs({}));
  const int array_call_count = ToAllocationControl().array_call_count;
  ApplyNothrowFailure(NothrowFailure::kOff);
  EXPECT_EQ(array_call_count, 0);
  EXPECT_TRUE(counts.has_value());
}

// The lent count of builder buffers is 8 bytes on the heap, allocated before any memory piece: its failure is
// kOutOfMemory with magnitude 8, and the RunMemory has handed out nothing.
TEST(BuildGridBuilderBuffers, LentCountNotAllocated_IsOutOfMemoryOfEightBytes) {
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kCpu, 1, 1);
  GridCounts room_counts = {};
  room_counts.cell_blocks = 1;
  const MemoryCounts memory_counts = ToMemoryCounts(room_counts, 0);
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(
      RunOn::kCpu,
      ComputeRunMemoryBytes(estimate, memory_counts, MemoryWhere::kRunOn)
          + ComputeRunMemoryBytes(estimate, memory_counts, MemoryWhere::kCpu)
  );
  ASSERT_TRUE(run_memory.has_value());
  ApplyNothrowFailure(NothrowFailure::kAsked);
  const std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1, *run_memory);
  const int single_call_count = ToAllocationControl().single_call_count;
  ApplyNothrowFailure(NothrowFailure::kOff);
  EXPECT_EQ(single_call_count, 1);
  ASSERT_FALSE(builder_buffers.has_value());
  EXPECT_EQ(builder_buffers.error().error, GridError::kOutOfMemory);
  EXPECT_EQ(builder_buffers.error().magnitude, sizeof(std::size_t));
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
}

}  // namespace
}  // namespace fluidity::grid
