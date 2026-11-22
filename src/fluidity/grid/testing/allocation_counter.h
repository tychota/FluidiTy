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

namespace fluidity::grid {

// Computes the number of calls of any global operator new form since the program started, in every thread.
//
// Example: a read of 10, then two std::vector<int> of one element each, then a read gives 12. A request too large to
//   hold, of any form, is counted and fails as its form says: null for a nothrow form, std::bad_alloc otherwise.
std::uint64_t ComputeAllocationCount();

}  // namespace fluidity::grid
