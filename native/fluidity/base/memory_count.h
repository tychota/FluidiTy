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
//
// Words:
//   memory estimate  the list of entries of a module or a run: every array it will allocate
//   entry            one line of a memory estimate: one array (or a group of identical arrays) and its size
//   place            one position of a grid array: a cell place, a face place or a corner place
// Owns:    CountedPer, MemoryWhere, KeptFor, MemoryEstimateEntry, MemoryCounts, ComputeMemoryBytes
// Uses:    CPU-only guard (base/cpu_only.h)
// Used by: every module that declares a memory estimate, the step's run plan
// Read:    docs/method-paper/fluidity-method.typ <sec-memory-estimate>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

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

// Sums the bytes of the entries kept in `where` only. The step compares the kRunOn sum with the processor's memory
// and the kCpu sum with CPU memory. Saturation, kept_for and counted_per behave as in the two-argument form.
std::uint64_t
ComputeMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts, MemoryWhere where);

}  // namespace fluidity
