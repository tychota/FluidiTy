// The GPU half of the ordered sums, private to base/: the declarations of what base/ordered_sums_gpu.cu defines.
//
// Purpose: let base/ordered_sums_cpu.cc reach the GPU reductions without including a CUDA header.
//
// Idea: each function reduces `count` doubles in GPU memory to one double, copies it to `result` and returns kDone,
//   or returns how it failed. The working arrays come from the caller's SumBuffers, split by
//   base/ordered_sums_cpu.cc into GpuSumBufferAreas, so no function here allocates. They exist only in a build with
//   GPU code, so base/ordered_sums_cpu.cc names them only under `#if defined(FLUIDITY_HAS_GPU)`. The caller has
//   already checked sizes and handled count 0, and the buffers are on the GPU. This file uses the C++20 subset and
//   includes no CPU-only header, so base/ordered_sums_gpu.cu, which defines the functions, includes it too. Callers
//   use base/ordered_sums.h.
//
// Owns:    GpuSumBufferAreas, ComputeGpuTreeSum, ComputeGpuTreeDot, ComputeGpuLargestMagnitude
// Uses:    RunOnCallStatus (base/run_on_call.h)
// Used by: base/ordered_sums_cpu.cc, and base/ordered_sums_gpu.cu, which defines them
// Read:    docs/method-paper/fluidity-method.typ <proc-tree-sum>

#pragma once

#include <cstddef>

#include "fluidity/base/run_on_call.h"

namespace fluidity {

// The arrays of a SumBuffers on the GPU, each in GPU memory, for up to max_count values. The CPU half sizes them
// from base/tree_sum_layers.h, so they hold:
struct GpuSumBufferAreas {
  double* products = nullptr;  // max_count entries: the products of a dot product.
  double* layer_values = nullptr;  // ComputeLayerValueCount(max_count) entries: the two arrays of the layers.
};

// Sums values[0..count) of GPU memory in the order of math::TreeSum, one layer of the tree per GPU call, in
// areas.layer_values. count is 1 or more and at most the max_count of the sum buffers. Writes the sum to the CPU
// double *result on success only.
RunOnCallStatus ComputeGpuTreeSum(const double* values, std::size_t count, GpuSumBufferAreas areas, double* result);

// Sums first[i] * second[i] of GPU memory for i below count: writes the rounded products to areas.products, then sums
// them as ComputeGpuTreeSum does. count is 1 or more and at most the max_count of the sum buffers. Writes to *result
// on success only.
RunOnCallStatus ComputeGpuTreeDot(
    const double* first,
    const double* second,
    std::size_t count,
    GpuSumBufferAreas areas,
    double* result
);

// The largest magnitude of values[0..count) of GPU memory, NaN when any value is NaN. count is 1 or more and at most
// the max_count of the sum buffers. Writes to *result on success only.
RunOnCallStatus
ComputeGpuLargestMagnitude(const double* values, std::size_t count, GpuSumBufferAreas areas, double* result);

}  // namespace fluidity
