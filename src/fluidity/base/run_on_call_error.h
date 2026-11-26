// The name and text of the last CUDA error that a call of a GPU half returned.
//
// Purpose: keep what RunOnCallStatus drops, so a test or a tool can print which CUDA error made a call fail.
//
// Idea: every GPU half turns a CUDA error into the status kCallFailed or kOutOfMemory (base/run_on_call.h), which says
//   that a call failed and not which error it was. The helpers of base/gpu_launch.cuh write the name and the text of
//   each failed call here, by ApplyLastRunOnCallError, and a test reads them back with FindLastRunOnCallError. Each
//   thread has its own error, so tests that run at the same time do not mix them. A success never clears it: it
//   holds the last failure, so it is read right after a status that is not kDone, and a test that expects a clean
//   start calls ApplyNoRunOnCallError first. The public status values do not change. The functions use no CUDA type,
//   so a test runs them on the CPU.
//   Example: after ApplyLastRunOnCallError("cudaErrorInvalidValue", "invalid argument"), FindLastRunOnCallError() has
//   has_error true, name "cudaErrorInvalidValue" and text "invalid argument".
//
// Words:
//   call error  the name and text CUDA gives for one failed call (cudaGetErrorName, cudaGetErrorString)
// Owns:    kMaxRunOnCallErrorChars, RunOnCallError, ApplyLastRunOnCallError, ApplyNoRunOnCallError,
//          FindLastRunOnCallError
// Uses:    nothing from this code base
// Used by: base/gpu_launch.cuh, the runners of the tests of the GPU halves
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

#include <cstddef>

namespace fluidity {

// The most characters kept of a name or a text, the end mark not counted. A longer one is cut.
inline constexpr std::size_t kMaxRunOnCallErrorChars = 127;

// The last failed call of the calling thread. A plain struct of arrays, so it never allocates.
struct RunOnCallError {
  bool has_error = false;  // False until a call fails, and after ApplyNoRunOnCallError.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): fixed text that is copied by value and never allocates.
  char name[kMaxRunOnCallErrorChars + 1] = {};  // CUDA's name of the error, such as "cudaErrorInvalidValue".
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): as name.
  char text[kMaxRunOnCallErrorChars + 1] = {};  // CUDA's text for the error, such as "invalid argument".
};

// Writes `name` and `text` as the last failed call of the calling thread, cut to kMaxRunOnCallErrorChars each.
// A null pointer counts as an empty text.
// Example: a name of 200 characters keeps its first 127.
void ApplyLastRunOnCallError(const char* name, const char* text);

// Clears the last failed call of the calling thread: has_error becomes false and both texts empty.
void ApplyNoRunOnCallError();

// Finds the last failed call of the calling thread, by value.
// Example: with no failure since the thread started or since ApplyNoRunOnCallError, has_error is false.
RunOnCallError FindLastRunOnCallError();

}  // namespace fluidity
