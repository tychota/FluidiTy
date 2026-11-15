// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/processor_description.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/test_support.h"

namespace fluidity {
namespace {

// Values with a negative zero, a subnormal and ordinary numbers, so a round trip is checked on every bit kind.
std::vector<double> BuildTestValues(std::size_t count) {
  std::vector<double> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = (static_cast<double>(index) - 500.0) / 7.0;
  }
  values[0] = -0.0;
  values[1] = 4.9406564584124654e-324;
  return values;
}

TEST(BuildBuffer, FromGpuRunMemory_WriteThenReadReturnsTheSameValues) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, 16384);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  // A first piece, so the tested one starts inside the block, not at its start.
  const std::expected<Buffer<double>, BufferError> first = BuildBuffer<double>(RunOn::kGpu, 3, *run_memory);
  ASSERT_TRUE(first.has_value()) << ToName(first.error());
  const std::vector<double> written = BuildTestValues(1000);
  std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kGpu, written.size(), *run_memory);
  ASSERT_TRUE(piece.has_value()) << ToName(piece.error());
  ASSERT_TRUE(WriteBuffer(written, *piece).has_value());
  const std::vector<double> read = ReadAll(*piece);
  EXPECT_EQ(ToBits(read), ToBits(written));
  WriteResultDigest("WriteBuffer then ReadBuffer on a piece of a GPU RunMemory", std::span<const double>(read));
}

TEST(BuildBuffer, FromGpuRunMemory_HandsOutAlignedZeroedPieces) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, 4096);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  const std::expected<Buffer<double>, BufferError> first = BuildBuffer<double>(RunOn::kGpu, 10, *run_memory);
  const std::expected<Buffer<double>, BufferError> second = BuildBuffer<double>(RunOn::kGpu, 100, *run_memory);
  ASSERT_TRUE(first.has_value() && second.has_value());
  // cudaMalloc starts the block at a multiple of 256 bytes, so every piece does too.
  EXPECT_EQ(std::bit_cast<std::uintptr_t>(first->data()) % kRunMemoryPieceAlignmentBytes, 0U);
  EXPECT_EQ(std::bit_cast<std::uintptr_t>(second->data()) % kRunMemoryPieceAlignmentBytes, 0U);
  EXPECT_EQ(run_memory->handed_out_bytes(), 256U + 1024U);
  const std::vector<double> read = ReadAll(*second);
  EXPECT_EQ(read, std::vector<double>(100, 0.0));
  WriteResultDigest("zeros of a piece of a GPU RunMemory", std::span<const double>(read));
}

TEST(BuildBuffer, FromGpuRunMemory_CopyBufferBetweenPiecesCopiesEveryValue) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, 4096);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  const std::vector<double> written = BuildTestValues(64);
  std::expected<Buffer<double>, BufferError> from = BuildBuffer<double>(RunOn::kGpu, written.size(), *run_memory);
  std::expected<Buffer<double>, BufferError> to = BuildBuffer<double>(RunOn::kGpu, written.size(), *run_memory);
  ASSERT_TRUE(from.has_value() && to.has_value());
  ASSERT_TRUE(WriteBuffer(written, *from).has_value());
  ASSERT_TRUE(CopyBuffer(*from, *to).has_value());
  const std::vector<double> read = ReadAll(*to);
  EXPECT_EQ(ToBits(read), ToBits(written));
  WriteResultDigest("CopyBuffer between pieces of a GPU RunMemory", std::span<const double>(read));
}

TEST(RunMemory, OnGpuMoved_PiecesStayValid) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, 1024);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  // Declared before the piece, so it is destroyed after it, as the block must be.
  std::optional<RunMemory> moved;
  std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kGpu, 4, *run_memory);
  ASSERT_TRUE(piece.has_value()) << ToName(piece.error());
  moved.emplace(std::move(*run_memory));
  const std::vector<double> written = BuildTestValues(4);
  ASSERT_TRUE(WriteBuffer(written, *piece).has_value());
  const std::vector<double> read = ReadAll(*piece);
  EXPECT_EQ(ToBits(read), ToBits(written));
  EXPECT_EQ(moved->live_piece_count(), 1U);
  WriteResultDigest("a piece of a moved GPU RunMemory", std::span<const double>(read));
}

TEST(BuildRunMemory, OnGpu_FreeMemoryDropsByAtLeastTheReservedBytes) {
  constexpr std::size_t kReservedBytes = std::size_t{512} << 20U;  // 512 MiB.
  // Other programs may take or give back a little GPU memory meanwhile, as in Buffer's OnGpuManyTimes_FreesEverything.
  constexpr std::uint64_t kAllowedChangeBytes = std::uint64_t{64} << 20U;
  const std::expected<MemoryAvailable, BufferError> before = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(before.has_value()) << ToName(before.error());
  const std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, kReservedBytes);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  const std::expected<MemoryAvailable, BufferError> after = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(after.has_value()) << ToName(after.error());
  EXPECT_GE(before->free_bytes + kAllowedChangeBytes, after->free_bytes + kReservedBytes);
}

TEST(BuildRunMemory, LockedForCopiesWithAGpu_HandsOutUsablePieces) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 4096, CpuMemory::kLockedForCopies);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  const std::vector<double> written = BuildTestValues(64);
  std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kCpu, written.size(), *run_memory);
  ASSERT_TRUE(piece.has_value()) << ToName(piece.error());
  EXPECT_EQ(std::bit_cast<std::uintptr_t>(piece->data()) % kBufferAlignmentBytes, 0U);
  ASSERT_TRUE(WriteBuffer(written, *piece).has_value());
  EXPECT_EQ(ToBits(ReadAll(*piece)), ToBits(written));
}

}  // namespace
}  // namespace fluidity
