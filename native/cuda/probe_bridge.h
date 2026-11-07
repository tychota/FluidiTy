#pragma once

#include <cstddef>

#include "fluidity/base/types.h"

namespace fluidity {

// Launches 1 to 32 samples on the CUDA stream carried by the opaque handle.
// The device buffer has sample_count + 2 Vec2 slots and must outlive stream completion.
// The kernel writes the samples and __CUDA_ARCH__ in the x component at sample_count + 1.
// It leaves the sentinel at sample_count untouched. Returns the immediate CUDA launch result.
int LaunchProbeKernel(Vec2* device_output, std::size_t sample_count, void* stream);

}  // namespace fluidity
