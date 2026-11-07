#pragma once

#include <cstddef>

#include "fluidity/base/types.h"

namespace fluidity {

// Launches 1 to 32 samples on the CUDA stream carried by the opaque handle.
// The device buffer has count + 2 Vec2 slots.
// The kernel writes the first count slots, leaves the sentinel at count alone, and writes __CUDA_ARCH__
// in the x component at count + 1. Returns the immediate CUDA launch result.
int LaunchProbeKernel(Vec2* device_output, std::size_t count, void* stream);

}  // namespace fluidity
