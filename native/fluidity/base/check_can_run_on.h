// The check that the chosen processor can run here.
//
// Purpose: tell the caller before any work starts whether its choice of CPU or GPU can run on this machine, and
//   say why not in words and in a stable code.
//
// Idea: the caller makes the choice (RunOn). This check only answers yes or no for that choice. It never picks a
//   processor, and a failed GPU choice is never turned into a CPU run: the caller sees the error and decides.
//   Asked for the CPU, the answer is always yes. Asked for the GPU, the answer is no in a build that has no GPU
//   code. A build with GPU code also runs the shared math on the GPU and compares it with the CPU.
//   Every module's error enum has the same pair of functions as RunOnError: ToMessage for people, ToName for
//   programs, and results are std::expected.
//
// Words:
//   code  the stable snake_case name of an error value, such as "gpu_not_built"
// Owns:    CheckCanRunOn, ToMessage(RunOnError), ToName(RunOnError)
// Uses:    RunOn, RunOnError (base/run_on.h), CPU-only guard (base/cpu_only.h)
// Used by: every function asked to run on the GPU, which checks first; the command line, the Blender add-on,
//          every test that runs on both processors
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>, <proc-check-can-run-on>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
// .clang-format keeps it first.
#include "fluidity/base/cpu_only.h"

#include <expected>
#include <string_view>

#include "fluidity/base/run_on.h"

namespace fluidity {

// Checks that an operation asked to run on `run_on` can run on this machine. Changes nothing.
//
// The caller made the choice. This function never picks a processor and never turns a GPU choice that cannot run
// into a CPU run: the caller sees the error and decides.
//
// Method: the CPU always succeeds. The GPU fails with kGpuNotBuilt in a build with no GPU code. In a build with GPU
//   code it succeeds only when a GPU is found and its results equal the CPU's.
// Example: CheckCanRunOn(RunOn::kCpu) succeeds on every machine. In a build made without the CUDA preset,
//   CheckCanRunOn(RunOn::kGpu) returns kGpuNotBuilt, and ToName gives "gpu_not_built".
// Fails: kGpuNotBuilt     the GPU was asked for in a build with no GPU code
//        kNoGpuFound      the build has GPU code but no GPU or driver answers
//        kGpuCallFailed   a GPU call made by the check returned an error
//        kGpuMathDiffers  the shared math gave different bits on the GPU
//        Never fails for RunOn::kCpu. Never returns kNone as an error.
// Read: docs/method-paper/fluidity-method.typ <proc-check-can-run-on>
std::expected<void, RunOnError> CheckCanRunOn(RunOn run_on);

// Converts an error to a plain sentence that says what the user can do. Never empty; each error has its own text.
// An out-of-range value gives the text of an unknown error.
std::string_view ToMessage(RunOnError error);

// Converts an error to its code: the enumerator without the k, in snake_case, such as "gpu_not_built". The command
// line and the Blender add-on use it, so a code never changes once published. An out-of-range value gives "unknown".
std::string_view ToName(RunOnError error);

}  // namespace fluidity
