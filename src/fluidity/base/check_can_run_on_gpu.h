// The GPU half of CheckCanRunOn, private to base/: the declaration of what base/check_can_run_on_gpu.cu defines.
//
// Purpose: declare the one function that runs the shared math on the GPU, so base/check_can_run_on_cpu.cc can call
//   it and the GPU file that defines it can include the same declaration.
//
// Idea: the function evaluates the shared vector and matrix helpers on the GPU with fixed inputs and compares the
//   results bit for bit with the CPU's. It exists only in a build with GPU code, so base/check_can_run_on_cpu.cc
//   names it only under `#if defined(FLUIDITY_HAS_GPU)`. This file uses the C++20 subset and includes no CPU-only
//   header, so the GPU file can include it. Callers use base/check_can_run_on.h.
//
// Owns:    CheckGpuMath
// Uses:    RunOnError (base/run_on.h)
// Used by: base/check_can_run_on_cpu.cc, and base/check_can_run_on_gpu.cu, which defines it
// Read:    docs/method-paper/fluidity-method.typ <proc-check-can-run-on>

#pragma once

#include "fluidity/base/run_on.h"

namespace fluidity {

// Checks that the GPU gives the same numbers as the CPU for the shared math.
//
// Method: evaluates the shared vector and matrix helpers on the GPU with fixed inputs and compares every result
//   bit for bit with the CPU's. `error` must not be null. Returns true when all results are equal; otherwise
//   returns false and sets *error. *error is unchanged on success.
// Fails: kNoGpuFound      no GPU or driver answers
//        kGpuCallFailed   a GPU call returned an error
//        kGpuMathDiffers  a result differs from the CPU's in at least one bit
//        Never sets kNone or kGpuNotBuilt.
// Read: docs/method-paper/fluidity-method.typ <proc-check-can-run-on>
bool CheckGpuMath(RunOnError* error);

}  // namespace fluidity
