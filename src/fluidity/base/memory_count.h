// The count of bytes a run needs, from the memory estimate each module declares.
//
// Purpose: let each module declare its memory as a short list of entries, so the step can add them up and compare
//   the sum with the memory of the processor before it builds anything.
//
// Idea: most arrays have one item per something: per cell block, per particle, per face place. An entry says
//   which something, how many bytes each item takes, and how many copies exist. The run's counts of each something
//   are known from the scene, so the bytes of an entry are the count times the bytes of one item times the copies.
//   The sum counts every entry as alive at once: it is the peak, a bound that never sits below the real need.
//   Sums use saturating arithmetic: a count too large for 64 bits gives the largest 64-bit number, never a small
//   wrapped one, so a huge scene is always reported as too big.
//   A run reserves its memory as one block per processor and builds each buffer as a piece of it (base/run_memory.h).
//   A piece takes its bytes rounded up to a multiple of 256, so the bytes to reserve round each copy of each entry
//   the same way: ComputeRunMemoryBytes, with the one rounding function ComputeRunMemoryPieceBytes.
//
// Words:
//   memory estimate  the list of entries of a module or a run: every array it will allocate
//   entry            one line of a memory estimate: one array (or a group of identical arrays) and its size
//   place            one position of a grid array: a cell place, a face place or a corner place
// Owns:    CountedPer, MemoryWhere, KeptFor, MemoryEstimateEntry, MemoryCounts, ComputeMemoryBytes,
//          kRunMemoryPieceAlignmentBytes, ComputeRunMemoryPieceBytes, ComputeRunMemoryBytes
// Uses:    CPU-only guard (base/cpu_only.h)
// Used by: every module that declares a memory estimate, the step's run plan, base/run_memory.h
// Read:    docs/method-paper/fluidity-method.typ <sec-memory-estimate>
//          docs/method-paper/fluidity-method.typ <sec-run-memory>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace fluidity {

// What the items of an entry are counted by. Each value picks one field of MemoryCounts, except kGivenCount.
enum class CountedPer : std::uint8_t {
  kCellBlock,  // One item per cell block: MemoryCounts::cell_blocks.
  kCellPlace,  // One item per cell place: MemoryCounts::cell_places.
  kFacePlace,  // One item per face place: MemoryCounts::face_places.
  kCornerPlace,  // One item per corner place: MemoryCounts::corner_places.
  kParticle,  // One item per particle: MemoryCounts::particles.
  kGivenCount,  // The entry states its own item count in MemoryEstimateEntry::given_count.
};

// In which memory an entry lives.
enum class MemoryWhere : std::uint8_t {
  kRunOn,  // In the memory of the processor the run uses: GPU memory in a GPU run, CPU memory in a CPU run.
  kCpu,  // Always in CPU memory, even in a GPU run.
  // In the memory of the processor the run uses, but held by its driver outside every RunMemory: the stack the GPU
  // gives each resident thread of a GPU function. The run counts it against the processor's memory and never reserves
  // it in its block; a CPU run has none.
  kDriverStack,
};

// How long the array of an entry is kept. It names the lifetime for the reader; it does not change the sum.
enum class KeptFor : std::uint8_t {
  kWholeRun,  // Allocated when the run is built and kept until it ends.
  kOneStepTry,  // Allocated for one try of a step. A rejected try is thrown away, and the next try starts again.
  kOneCall,  // Allocated inside one function call and given back when it returns.
  kOneSnapshot,  // Allocated while one saved copy of the run state is built or written.
};

// One line of a memory estimate: one array, or a group of identical arrays, that a module will allocate.
struct MemoryEstimateEntry {
  std::string_view name;  // What the array holds, such as "grid: face piece kinds". Must point at a string literal.
  CountedPer counted_per = CountedPer::kGivenCount;  // Which count of MemoryCounts gives the number of items.
  std::uint64_t given_count = 0;  // Items. Read only when counted_per is kGivenCount.
  std::uint32_t bytes_each = 0;  // Bytes of one item.
  std::uint32_t copies = 1;  // Arrays of the same shape: 2 when a committed and a try copy exist.
  KeptFor kept_for = KeptFor::kWholeRun;  // The lifetime of the array. Does not change the sum.
  MemoryWhere where = MemoryWhere::kRunOn;  // In the run's memory, or always in CPU memory.
};

// The counts of items in one scene. The grid fills the first four from its blocks.
struct MemoryCounts {
  std::uint64_t cell_blocks = 0;  // Cell blocks in the grid.
  std::uint64_t cell_places = 0;  // 64 per block.
  std::uint64_t face_places = 0;  // 240 per block.
  std::uint64_t corner_places = 0;  // 125 per block.
  std::uint64_t particles = 0;  // Particles in the run.
};

// Sums the bytes of every entry of a memory estimate, wherever it lives.
//
// Method: each entry gives item count x bytes_each x copies, where the item count is the field of `counts` that
//   counted_per names, or given_count for kGivenCount. Every multiplication and addition saturates. kept_for is
//   ignored: the sum is the peak with every entry alive at once.
// Units: bytes.
// Example: 1000 particles in an entry of 24 bytes with 2 copies give 1000 x 24 x 2 = 48000.
// Fails: never. Any overflow gives std::numeric_limits<std::uint64_t>::max(). A counted_per value outside the
//        enum counts as zero items. An entry with copies = 0 counts nothing.
// Read: docs/method-paper/fluidity-method.typ <sec-memory-estimate>
std::uint64_t ComputeMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts);

// Sums the bytes of the entries kept in `where` only. The step adds the kRunOn sum and the largest kDriverStack entry
// (ComputeLargestEntryBytes) and compares that with the processor's memory, and compares the kCpu sum with CPU memory.
// Saturation, kept_for and counted_per behave as in the two-argument form.
std::uint64_t
ComputeMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts, MemoryWhere where);

// Computes the bytes of the largest single entry kept in `where`: 0 when no entry is kept there.
//
// Why: the GPU driver keeps one stack per thread, sized for the largest need of any function launched, so the
//   entries of MemoryWhere::kDriverStack share that one stack. The total takes the largest of them, not their sum,
//   which would count more than the driver takes.
// Method: each entry's bytes are counted as in ComputeMemoryBytes, with saturation; the largest of them is returned.
// Units: bytes.
// Example: kDriverStack entries of 100, 700 and 300 bytes give 700, where their sum would give 1,100. Entries kept
//   in another `where` are ignored.
// Fails: never. Any overflow gives std::numeric_limits<std::uint64_t>::max().
// Requires: nothing.
std::uint64_t
ComputeLargestEntryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts, MemoryWhere where);

// Bytes. Every piece of a RunMemory starts at a multiple of this from the start of its block, and takes its byte
// count rounded up to a multiple of it. 256 is the alignment the CUDA manual promises for cudaMalloc, so a piece is
// aligned as a GPU buffer of its own would be; the CPU uses the same value, so a plan's bytes do not depend on the
// processor.
inline constexpr std::size_t kRunMemoryPieceAlignmentBytes = 256;

// Computes the bytes a piece of a RunMemory takes for `byte_count` bytes: byte_count rounded up to a multiple of
// kRunMemoryPieceAlignmentBytes. The one rounding rule of RunMemory: building a buffer and planning its bytes both
// call it.
//
// Example: 0 gives 0, 1 gives 256, 256 gives 256, 257 gives 512, 24,000 gives 24,064.
// Fails: never. A byte count whose rounding would not fit in std::size_t gives the largest std::size_t, which no
//        block can hold.
// Read: docs/method-paper/fluidity-method.typ <sec-run-memory>
std::size_t ComputeRunMemoryPieceBytes(std::size_t byte_count);

// Computes the bytes a RunMemory needs to hold one buffer per copy of every entry kept in `where`: the sum of
// copies x ComputeRunMemoryPieceBytes(item count x bytes_each). Building those buffers from a RunMemory of exactly
// this size hands out every byte of it.
//
// Method: as ComputeMemoryBytes with `where`, except that each copy's bytes are rounded up to a piece first, since
//   every copy is its own buffer and so its own piece. Every multiplication, rounding and addition saturates.
//   A GPU run reserves one block per processor: the kRunOn sum on the GPU, the kCpu sum on the CPU. A CPU run
//   reserves one block for the sum of both.
//   Contract for every module: each copy of an entry is exactly one buffer of item count x bytes_each bytes. A
//   module that builds one copy as several buffers must write one entry per buffer: three buffers of 8 bytes per
//   item round three times, so one entry of 24 bytes per item would count too little.
// Units: bytes.
// Example: 1000 particles in a kRunOn entry of 24 bytes with 2 copies: each copy is 24,000 bytes, rounded up to
//   24,064, so 2 x 24,064 = 48,128. A kCpu entry adds nothing to the kRunOn sum. 10 particles as three entries of
//   8 bytes give 3 x 256 = 768, while one entry of 24 bytes gives 256.
// Fails: never. Any overflow gives std::numeric_limits<std::uint64_t>::max().
// Read: docs/method-paper/fluidity-method.typ <sec-run-memory>
std::uint64_t
ComputeRunMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts, MemoryWhere where);

}  // namespace fluidity
