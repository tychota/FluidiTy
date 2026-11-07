#pragma once

#include <cstddef>

#include "fluidity/base/types.h"

namespace fluidity::cuda {

// One block writes sample_count arithmetic results and the architecture marker at sample_count + 1.
// The caller provides sample_count + 2 device slots. The slot at sample_count is left untouched.
__global__ void ProbeKernel(Vec2* device_output, std::size_t sample_count);

}  // namespace fluidity::cuda
