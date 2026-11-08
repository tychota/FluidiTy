#pragma once

#if defined(__CUDACC__)
#define FLUIDITY_HD __host__ __device__
#else
#define FLUIDITY_HD
#endif
