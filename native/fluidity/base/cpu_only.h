// The guard that keeps a CPU-only header out of GPU code.
//
// Purpose: turn a mistake into a build error at the line that made it: a shared or GPU file that includes a header
//   which uses the C++ library containers, strings or other host-only facilities.
//
// Idea: a header that needs <expected>, <string> or anything that allocates cannot be built for the GPU. Such a
//   header includes this one first. The CUDA compiler defines __CUDACC__, so under it this file stops the build and
//   the message names the fix. Any other compiler passes over it. The file declares nothing.
//
// Words:
//   CPU-only header  a header that GPU code must never include
// Owns:    nothing, only the guard
// Uses:    nothing
// Used by: every CPU-only header, starting with base/check_can_run_on.h
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

#if defined(__CUDACC__)
#error "A CPU-only header was included in GPU code. Split the header that included it."
#endif
