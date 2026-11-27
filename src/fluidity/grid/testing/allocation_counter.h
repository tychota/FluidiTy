// Test-only: counts every call of the global allocation functions in the test program that lists its source.
//
// Purpose: let a test show that a call allocates nothing, by reading the number of allocations before and after it.
//
// Idea: a C++ program may replace the global operator new and operator delete. testing/allocation_counter.cc
//   replaces every form, plain, nothrow and aligned, single and array, and counts each call of an operator new. A CPU
//   Buffer allocates through the aligned nothrow form (base/buffer_cpu.cc), so the count sees it. A replacement holds
//   for the whole program, so the source is listed in the SOURCES of each test program that counts allocations and
//   never in a library: a program that does not count must not get it. Nor may a program that replaces some forms
//   itself, such as grid/staggered_grid.out_of_memory.test.cc, list it.
//   Example: a test reads ComputeAllocationCount() as 1,204, builds a grid into its buffers, and reads 1,204 again.
//
// Owns:    ComputeAllocationCount
// Uses:    CPU-only guard (base/cpu_only.h)
// Used by: the tests of grid/ that show a call allocates nothing
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <version>

namespace fluidity::grid {

// Computes the number of calls of any global operator new form since the program started, in every thread.
//
// Example: a read of 10, then two std::vector<int> of one element each, then a read gives 12. A request too large to
//   hold, of any form, is counted and fails as its form says: null for a nothrow form, std::bad_alloc otherwise.
std::uint64_t ComputeAllocationCount();

// The iterator debug level of this build's standard library: the value of _ITERATOR_DEBUG_LEVEL of the MSVC library
// (2 in a Debug build, 0 in a Release build), and 0 where the library has no such macro. The standard headers
// included above define it first.
#ifdef _ITERATOR_DEBUG_LEVEL
inline constexpr int kIteratorDebugLevel = _ITERATOR_DEBUG_LEVEL;
#else
inline constexpr int kIteratorDebugLevel = 0;
#endif

// The std::vector objects a call constructs, for ComputeVectorAllocationCount.
struct VectorObjectCounts {
  // Vectors that hold a block of memory.
  std::uint64_t with_memory = 0;
  // Vectors that hold none because they were built by taking the block of another vector (the vector inside a
  // std::expected that a function returns from a local vector, for one).
  std::uint64_t without_memory = 0;
};

// Computes the number of allocations that the vectors of `vectors` make when they are constructed, at one iterator
// debug level.
//
// Why: the MSVC library, when _ITERATOR_DEBUG_LEVEL is above 0, gives each vector object a small heap block, its
//   container proxy, that the iterators of the vector point to. The proxy is allocated when the vector object is
//   constructed, empty or not, and a vector built by a move gets its own. So one vector with memory costs 2
//   allocations there and 1 elsewhere, and a vector without memory costs 1 there and none elsewhere. A test that reads
//   ComputeAllocationCount around code that builds vectors states its count through this function, so the count stays
//   exact on every build.
//   Example: at level 0, 2 vectors with memory and 1 without cost 2; at level 2 they cost 5.
constexpr std::uint64_t ComputeVectorAllocationCount(const VectorObjectCounts& vectors, int iterator_debug_level) {
  const std::uint64_t proxy_count_per_vector = iterator_debug_level > 0 ? 1U : 0U;
  const std::uint64_t vector_count = vectors.with_memory + vectors.without_memory;
  return vectors.with_memory + (vector_count * proxy_count_per_vector);
}

// ComputeVectorAllocationCount at the iterator debug level of this build.
constexpr std::uint64_t ComputeVectorAllocationCount(const VectorObjectCounts& vectors) {
  return ComputeVectorAllocationCount(vectors, kIteratorDebugLevel);
}

}  // namespace fluidity::grid
