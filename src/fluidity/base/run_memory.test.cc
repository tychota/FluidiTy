#include "fluidity/base/run_memory.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"

namespace fluidity {
namespace {

constexpr std::uint64_t kLargestBytes = std::numeric_limits<std::uint64_t>::max();

// The distance in bytes from `first` to `later`, two pointers into one block.
std::ptrdiff_t ComputeDistanceBytes(const void* first, const void* later) {
  return static_cast<const std::byte*>(later) - static_cast<const std::byte*>(first);
}

// A CPU block of `byte_count` bytes. Fails the calling test, and returns nothing, when building fails.
std::optional<RunMemory> BuildCpuRunMemory(std::size_t byte_count) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, byte_count);
  EXPECT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  if (!run_memory.has_value()) {
    return std::nullopt;
  }
  return std::move(*run_memory);
}

// Builds one buffer of bytes per copy of `entry` from `run_memory` and appends them to `pieces`. Knows the counts
// the accounting test uses: particles, cell blocks and given counts. Fails the calling test when a piece fails.
void BuildPiecesOfEntry(
    const MemoryEstimateEntry& entry,
    const MemoryCounts& counts,
    RunMemory& run_memory,
    std::vector<Buffer<std::byte>>& pieces
) {
  std::uint64_t item_count = entry.given_count;
  if (entry.counted_per == CountedPer::kParticle) {
    item_count = counts.particles;
  } else if (entry.counted_per == CountedPer::kCellBlock) {
    item_count = counts.cell_blocks;
  }
  for (std::uint32_t copy = 0; copy < entry.copies; ++copy) {
    std::expected<Buffer<std::byte>, BufferError> piece =
        BuildBuffer<std::byte>(RunOn::kCpu, item_count * entry.bytes_each, run_memory);
    ASSERT_TRUE(piece.has_value()) << entry.name << ": " << ToName(piece.error());
    pieces.push_back(std::move(*piece));
  }
}

TEST(ComputeRunMemoryPieceBytes, RoundsUpTo256) {
  EXPECT_EQ(ComputeRunMemoryPieceBytes(0), 0U);
  EXPECT_EQ(ComputeRunMemoryPieceBytes(1), 256U);
  EXPECT_EQ(ComputeRunMemoryPieceBytes(256), 256U);
  EXPECT_EQ(ComputeRunMemoryPieceBytes(257), 512U);
}

TEST(ComputeRunMemoryPieceBytes, ByteCountNearTheLargest_Saturates) {
  constexpr std::size_t kLargestSize = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(ComputeRunMemoryPieceBytes(kLargestSize - 10), kLargestSize);
  EXPECT_EQ(ComputeRunMemoryPieceBytes(kLargestSize), kLargestSize);
}

TEST(ComputeRunMemoryBytes, CountsOnlyEntriesOfTheWhereRoundedPerCopy) {
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "particles: positions",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 24,
          .copies = 2,
      },
      MemoryEstimateEntry{
          .name = "particles: positions kept on the CPU",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 24,
          .where = MemoryWhere::kCpu,
      },
  };
  const MemoryCounts counts = {.particles = 1000};
  // 1000 x 24 = 24000 bytes per copy, rounded up to 24064; two copies.
  EXPECT_EQ(ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kRunOn), 48128U);
  EXPECT_EQ(ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kCpu), 24064U);
}

TEST(ComputeRunMemoryBytes, ThreeBuffersPerItem_NeedThreeEntries) {
  // A module that keeps 24 bytes per particle as three buffers of 8 bytes builds three pieces, each rounded.
  const std::array one_entry = {
      MemoryEstimateEntry{.name = "positions as one buffer", .counted_per = CountedPer::kParticle, .bytes_each = 24},
  };
  const std::array three_entries = {
      MemoryEstimateEntry{.name = "positions x", .counted_per = CountedPer::kParticle, .bytes_each = 8},
      MemoryEstimateEntry{.name = "positions y", .counted_per = CountedPer::kParticle, .bytes_each = 8},
      MemoryEstimateEntry{.name = "positions z", .counted_per = CountedPer::kParticle, .bytes_each = 8},
  };
  const MemoryCounts counts = {.particles = 10};
  EXPECT_EQ(ComputeRunMemoryBytes(one_entry, counts, MemoryWhere::kRunOn), 256U);
  EXPECT_EQ(ComputeRunMemoryBytes(three_entries, counts, MemoryWhere::kRunOn), 768U);
  // Three buffers of 80 bytes take what the three entries say, not what the one entry says.
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(768);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<Buffer<double>, BufferError> x = BuildBuffer<double>(RunOn::kCpu, 10, *run_memory);
  const std::expected<Buffer<double>, BufferError> y = BuildBuffer<double>(RunOn::kCpu, 10, *run_memory);
  const std::expected<Buffer<double>, BufferError> z = BuildBuffer<double>(RunOn::kCpu, 10, *run_memory);
  ASSERT_TRUE(x.has_value() && y.has_value() && z.has_value());
  EXPECT_EQ(run_memory->handed_out_bytes(), 768U);
}

TEST(ComputeRunMemoryBytes, HugeCounts_Saturate) {
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "huge",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 24,
      },
  };
  // A product that does not fit in 64 bits, and one that fits but whose rounding up would not.
  const MemoryCounts too_many_particles = {.particles = kLargestBytes / 2};
  EXPECT_EQ(ComputeRunMemoryBytes(estimate, too_many_particles, MemoryWhere::kRunOn), kLargestBytes);
  const std::array one_byte_each = {
      MemoryEstimateEntry{
          .name = "almost the largest",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 1,
      },
  };
  const MemoryCounts almost_largest = {.particles = kLargestBytes - 10};
  EXPECT_EQ(ComputeRunMemoryBytes(one_byte_each, almost_largest, MemoryWhere::kRunOn), kLargestBytes);
}

TEST(BuildRunMemory, OnCpu_ReservesTheAskedBytes) {
  const std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 4096);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  EXPECT_EQ(run_memory->run_on(), RunOn::kCpu);
  EXPECT_EQ(run_memory->reserved_bytes(), 4096U);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
  EXPECT_EQ(run_memory->unused_bytes(), 4096U);
}

TEST(BuildRunMemory, ImpossibleByteCount_IsOutOfMemory) {
  const std::expected<RunMemory, BufferError> run_memory =
      BuildRunMemory(RunOn::kCpu, std::numeric_limits<std::size_t>::max());
  ASSERT_FALSE(run_memory.has_value());
  EXPECT_EQ(run_memory.error(), BufferError::kOutOfMemory);
}

TEST(BuildRunMemory, LockedForCopiesOnCpu_HandsOutUsablePieces) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 1024, CpuMemory::kLockedForCopies);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kCpu, 4, *run_memory);
  ASSERT_TRUE(piece.has_value()) << ToName(piece.error());
  const std::array values = {1.0, -2.0, 3.5, 0.25};
  ASSERT_TRUE(WriteBuffer(values, *piece).has_value());
  EXPECT_EQ(ReadAll(*piece), (std::vector<double>(values.begin(), values.end())));
}

#if !defined(FLUIDITY_HAS_GPU)
TEST(BuildRunMemory, OnGpuInBuildWithoutGpuCode_IsRunOnCallFailed) {
  const std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, 4096);
  ASSERT_FALSE(run_memory.has_value());
  EXPECT_EQ(run_memory.error(), BufferError::kRunOnCallFailed);
}
#endif

TEST(BuildBuffer, FromRunMemory_HandsOutAlignedZeroedPieces) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(4096);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<Buffer<double>, BufferError> first = BuildBuffer<double>(RunOn::kCpu, 10, *run_memory);
  const std::expected<Buffer<double>, BufferError> second = BuildBuffer<double>(RunOn::kCpu, 100, *run_memory);
  const std::expected<Buffer<double>, BufferError> third = BuildBuffer<double>(RunOn::kCpu, 3, *run_memory);
  ASSERT_TRUE(first.has_value() && second.has_value() && third.has_value());
  // The first piece starts the block; 80 bytes round up to 256 and 800 bytes to 1024.
  EXPECT_EQ(ComputeDistanceBytes(first->data(), second->data()), 256);
  EXPECT_EQ(ComputeDistanceBytes(first->data(), third->data()), 256 + 1024);
  EXPECT_EQ(std::bit_cast<std::uintptr_t>(first->data()) % kBufferAlignmentBytes, 0U);
  EXPECT_EQ(run_memory->handed_out_bytes(), 256U + 1024U + 256U);
  EXPECT_EQ(run_memory->live_piece_count(), 3U);
  EXPECT_EQ(ReadAll(*second), std::vector<double>(100, 0.0));
}

TEST(BuildBuffer, FromRunMemory_HasTheAskedSizeAndProcessor) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(1024);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<Buffer<std::uint32_t>, BufferError> piece =
      BuildBuffer<std::uint32_t>(RunOn::kCpu, 17, *run_memory);
  ASSERT_TRUE(piece.has_value()) << ToName(piece.error());
  EXPECT_EQ(piece->size(), 17U);
  EXPECT_EQ(piece->run_on(), RunOn::kCpu);
}

TEST(BuildBuffer, FromRunMemoryEveryEntryOfAnEstimate_HandsOutExactlyTheComputedBytes) {
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "positions",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 24,
          .copies = 2,
      },
      MemoryEstimateEntry{
          .name = "velocities",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 48,
      },
      MemoryEstimateEntry{
          .name = "working buffers",
          .counted_per = CountedPer::kGivenCount,
          .given_count = 256,
          .bytes_each = 4,
      },
      MemoryEstimateEntry{
          .name = "block flags",
          .counted_per = CountedPer::kCellBlock,
          .bytes_each = 1,
          .copies = 3,
      },
      MemoryEstimateEntry{
          .name = "kept on the CPU, not in this block",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 8,
          .where = MemoryWhere::kCpu,
      },
  };
  const MemoryCounts counts = {.cell_blocks = 7, .particles = 1000};
  const std::uint64_t computed_bytes = ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kRunOn);
  // 2 x 24064 + 48128 + 1024 + 3 x 256.
  ASSERT_EQ(computed_bytes, 98048U);
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(computed_bytes);
  ASSERT_TRUE(run_memory.has_value());

  // One buffer of bytes per copy of each kRunOn entry, as a module builds its arrays.
  std::vector<Buffer<std::byte>> pieces;
  for (const MemoryEstimateEntry& entry : estimate) {
    if (entry.where == MemoryWhere::kRunOn) {
      BuildPiecesOfEntry(entry, counts, *run_memory, pieces);
    }
  }
  EXPECT_EQ(pieces.size(), 7U);
  EXPECT_EQ(run_memory->handed_out_bytes(), computed_bytes);
  EXPECT_EQ(run_memory->unused_bytes(), 0U);
}

TEST(BuildBuffer, FromRunMemoryPieceDoesNotFit_IsRunMemoryTooSmall) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(512);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kCpu, 65, *run_memory);
  ASSERT_FALSE(piece.has_value());
  EXPECT_EQ(piece.error(), BufferError::kRunMemoryTooSmall);
}

TEST(BuildBuffer, FromRunMemoryPieceDoesNotFit_TakesNothing) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(512);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<Buffer<double>, BufferError> first = BuildBuffer<double>(RunOn::kCpu, 10, *run_memory);
  ASSERT_TRUE(first.has_value());
  // 300 bytes round up to 512, more than the 256 left.
  const std::expected<Buffer<std::byte>, BufferError> too_large = BuildBuffer<std::byte>(RunOn::kCpu, 300, *run_memory);
  ASSERT_FALSE(too_large.has_value());
  EXPECT_EQ(run_memory->handed_out_bytes(), 256U);
  EXPECT_EQ(run_memory->live_piece_count(), 1U);
  // The block is unchanged: a piece that fits still comes right after the first one.
  const std::expected<Buffer<std::byte>, BufferError> fitting = BuildBuffer<std::byte>(RunOn::kCpu, 200, *run_memory);
  ASSERT_TRUE(fitting.has_value());
  EXPECT_EQ(ComputeDistanceBytes(first->data(), fitting->data()), 256);
}

TEST(BuildBuffer, FromRunMemoryImpossibleSize_IsRunMemoryTooSmall) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(512);
  ASSERT_TRUE(run_memory.has_value());
  const std::size_t overflowing_size = (std::numeric_limits<std::size_t>::max() / sizeof(double)) + 1;
  const std::expected<Buffer<double>, BufferError> piece =
      BuildBuffer<double>(RunOn::kCpu, overflowing_size, *run_memory);
  ASSERT_FALSE(piece.has_value());
  EXPECT_EQ(piece.error(), BufferError::kRunMemoryTooSmall);
}

TEST(BuildBuffer, FromRunMemorySizeZero_TakesNothing) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(256);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kCpu, 0, *run_memory);
  ASSERT_TRUE(piece.has_value()) << ToName(piece.error());
  EXPECT_EQ(piece->size(), 0U);
  EXPECT_EQ(piece->data(), nullptr);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
  EXPECT_EQ(run_memory->live_piece_count(), 0U);
}

TEST(BuildBuffer, FromRunMemoryFillingTheBlockExactly_Succeeds) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(512);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kCpu, 64, *run_memory);
  ASSERT_TRUE(piece.has_value()) << ToName(piece.error());
  EXPECT_EQ(run_memory->unused_bytes(), 0U);
}

TEST(Buffer, PieceDestroyed_FreesNothingAndBlockStaysUsable) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(1024);
  ASSERT_TRUE(run_memory.has_value());
  {
    const std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kCpu, 32, *run_memory);
    ASSERT_TRUE(piece.has_value());
    EXPECT_EQ(run_memory->live_piece_count(), 1U);
  }
  // The destroyed piece's bytes stay taken; the next piece comes after them.
  EXPECT_EQ(run_memory->live_piece_count(), 0U);
  EXPECT_EQ(run_memory->handed_out_bytes(), 256U);
  std::expected<Buffer<double>, BufferError> next = BuildBuffer<double>(RunOn::kCpu, 4, *run_memory);
  ASSERT_TRUE(next.has_value()) << ToName(next.error());
  const std::array values = {1.0, 2.0, 3.0, 4.0};
  ASSERT_TRUE(WriteBuffer(values, *next).has_value());
  EXPECT_EQ(ReadAll(*next), (std::vector<double>(values.begin(), values.end())));
  EXPECT_EQ(run_memory->handed_out_bytes(), 512U);
}

TEST(Buffer, PieceMoveAssignedOver_GivesBackItsPiece) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(1024);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<Buffer<double>, BufferError> target = BuildBuffer<double>(RunOn::kCpu, 4, *run_memory);
  std::expected<Buffer<double>, BufferError> moved_from = BuildBuffer<double>(RunOn::kCpu, 8, *run_memory);
  ASSERT_TRUE(target.has_value() && moved_from.has_value());
  *target = std::move(*moved_from);
  EXPECT_EQ(target->size(), 8U);
  EXPECT_EQ(run_memory->live_piece_count(), 1U);
}

TEST(Buffer, OwnedBufferMoveAssignedFromPiece_TakesThePiece) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(1024);
  ASSERT_TRUE(run_memory.has_value());
  Buffer<double> target = BuildBufferHolding(RunOn::kCpu, std::array{9.0, 9.0});
  std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kCpu, 3, *run_memory);
  ASSERT_TRUE(piece.has_value());
  target = std::move(*piece);
  EXPECT_EQ(target.size(), 3U);
  EXPECT_EQ(run_memory->live_piece_count(), 1U);
  target = Buffer<double>();
  EXPECT_EQ(run_memory->live_piece_count(), 0U);
}

TEST(RunMemory, Moved_PiecesStayValid) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(1024);
  ASSERT_TRUE(run_memory.has_value());
  // Declared before the piece, so it is destroyed after it, as the block must be.
  std::optional<RunMemory> moved;
  std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kCpu, 4, *run_memory);
  ASSERT_TRUE(piece.has_value());
  moved.emplace(std::move(*run_memory));
  const std::array values = {5.0, 6.0, 7.0, 8.0};
  ASSERT_TRUE(WriteBuffer(values, *piece).has_value());
  EXPECT_EQ(ReadAll(*piece), (std::vector<double>(values.begin(), values.end())));
  EXPECT_EQ(moved->live_piece_count(), 1U);
  EXPECT_EQ(moved->handed_out_bytes(), 256U);
  EXPECT_EQ(moved->reserved_bytes(), 1024U);
}

TEST(RunMemory, MovedFrom_IsEmpty) {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(1024);
  ASSERT_TRUE(run_memory.has_value());
  const RunMemory moved = std::move(*run_memory);
  // What a moved-from block holds is part of the contract, as in Buffer's MovedFrom_HasSizeZero.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(run_memory->reserved_bytes(), 0U);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
  EXPECT_EQ(run_memory->live_piece_count(), 0U);
  EXPECT_EQ(moved.reserved_bytes(), 1024U);
}

TEST(ToName, RunMemoryTooSmall_IsRunMemoryTooSmall) {
  EXPECT_EQ(ToName(BufferError::kRunMemoryTooSmall), "run_memory_too_small");
}

TEST(ToMessage, RunMemoryTooSmall_SaysItIsAPlanningError) {
  EXPECT_NE(ToMessage(BufferError::kRunMemoryTooSmall).find("planning error"), std::string_view::npos);
}

TEST(Buffer, PieceDestroyedAfterItsBlock_IsSafe) {
#if defined(NDEBUG)
  // The block leaves its live-piece count behind, so the late piece counts down into memory that still exists. This
  // proves the release build goes on and the late destruction completes; that no freed byte is written is shown
  // only by reading the code or by a memory checker, which the gate does not run.
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(1024);
  ASSERT_TRUE(run_memory.has_value());
  std::optional<Buffer<double>> piece;
  std::expected<Buffer<double>, BufferError> built = BuildBuffer<double>(RunOn::kCpu, 4, *run_memory);
  ASSERT_TRUE(built.has_value());
  piece.emplace(std::move(*built));
  run_memory.reset();
  piece.reset();
  EXPECT_FALSE(piece.has_value());
#else
  GTEST_SKIP() << "debug build: destroying a block before its pieces stops at an assert (RunMemoryDeathTest)";
#endif
}

#if !defined(NDEBUG)
// What each death test runs. A braced statement inside EXPECT_DEATH cannot hold a comma outside parentheses, so
// each one is its own function. A failed build returns early, and EXPECT_DEATH then fails for want of a stop.
void ApplyDestructionOfBlockBeforeItsPiece() {
  std::optional<RunMemory> run_memory = BuildCpuRunMemory(1024);
  if (!run_memory.has_value()) {
    return;
  }
  const std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kCpu, 4, *run_memory);
  run_memory.reset();
}

void BuildPieceFromMovedFromBlock() {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 512);
  if (!run_memory.has_value()) {
    return;
  }
  const RunMemory moved = std::move(*run_memory);
  // Building from the moved-from block is the point of the test.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  const std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kCpu, 4, *run_memory);
}

void BuildPieceOnTheOtherProcessor() {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 512);
  if (!run_memory.has_value()) {
    return;
  }
  const std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kGpu, 4, *run_memory);
}

#endif

// In a release build `assert` compiles to nothing, so each death test reports itself skipped there.

TEST(RunMemoryDeathTest, DestroyedBeforeItsPieces_Asserts) {
#if defined(NDEBUG)
  GTEST_SKIP() << "release build: assert compiles to nothing";
#else
  EXPECT_DEATH(ApplyDestructionOfBlockBeforeItsPiece(), "no_piece_is_alive");
#endif
}

TEST(RunMemoryDeathTest, BufferOnTheOtherProcessor_Asserts) {
#if defined(NDEBUG)
  GTEST_SKIP() << "release build: assert compiles to nothing";
#else
  EXPECT_DEATH(BuildPieceOnTheOtherProcessor(), "run_memory.run_on");
#endif
}

TEST(RunMemoryDeathTest, BufferFromMovedFromBlock_Asserts) {
#if defined(NDEBUG)
  GTEST_SKIP() << "release build: assert compiles to nothing";
#else
  EXPECT_DEATH(BuildPieceFromMovedFromBlock(), "run_memory_is_not_moved_from");
#endif
}

}  // namespace
}  // namespace fluidity
