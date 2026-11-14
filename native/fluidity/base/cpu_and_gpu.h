// The mark for a function that both the CPU and the GPU can call.
//
// Purpose: let one function in a shared header compile for the CPU and for the GPU.
//
// Idea: the CUDA compiler builds a function for the GPU only when the function is marked `__host__ __device__`.
//   Other compilers do not know these two words. FLUIDITY_CPU_AND_GPU expands to them under the CUDA compiler
//   and to nothing otherwise, so the same header compiles with both.
//
// Words:
//   shared header  a header that both CPU code and GPU code include
// Owns:    FLUIDITY_CPU_AND_GPU
// Uses:    nothing
// Used by: every shared header, starting with base/vec3.h
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

// The CUDA compiler defines __CUDACC__ in both of its passes, the CPU pass and the GPU pass. Both must see the
// mark, or the CPU pass and the GPU pass would disagree about where the function may be called.
#if defined(__CUDACC__)
#define FLUIDITY_CPU_AND_GPU __host__ __device__
#else
#define FLUIDITY_CPU_AND_GPU
#endif
