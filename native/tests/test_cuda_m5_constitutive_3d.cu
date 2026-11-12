#include <cuda_runtime.h>

#include <cstdio>

#include "m5_contact_3d_cases.h"
#include "m5_deformation_3d_cases.h"

namespace {

__global__ void CheckConstitutiveCasesOnDevice(int* result) {
  *result = fluidity::m5_test::CheckConstitutiveCases();
  if (*result == 0) {
    *result = fluidity::m5_test::CheckContactCases();
  }
  if (*result == 0) {
    *result = fluidity::m5_test::CheckDeformationCases();
  }
}

bool CheckCuda(cudaError_t status) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
  return false;
}

}  // namespace

int main() {
  int* device_result = nullptr;
  if (!CheckCuda(cudaMalloc(&device_result, sizeof(int)))) {
    return 1;
  }
  CheckConstitutiveCasesOnDevice<<<1, 1>>>(device_result);
  int result = -1;
  const bool ran = CheckCuda(cudaGetLastError())
      && CheckCuda(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
  const bool freed = CheckCuda(cudaFree(device_result));
  if (!ran || !freed) {
    return 1;
  }
  if (result != 0) {
    std::fprintf(stderr, "M5 invariant %d failed on CUDA\n", result);
  }
  return result;
}
