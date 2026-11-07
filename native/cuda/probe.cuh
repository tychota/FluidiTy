#pragma once

#include <cstddef>

#include "fluidity/base/types.h"

namespace fluidity::cuda {

// One block writes count arithmetic results and the architecture marker at count + 1.
// The caller provides count + 2 device slots. The slot at count is left untouched.
__global__ void ProbeKernel(Vec2* device_output, std::size_t count);

}  // namespace fluidity::cuda
