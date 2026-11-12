#include <cuda_runtime.h>

#include <cstdio>

#include "fluidity/base/sampled_solid_liquid_geometry_3d.h"
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

__global__ void CheckSampledSolidGeometryOnDevice(int* result) {
  fluidity::boundary_internal::SampledSolidLiquidCell cell{.cell_width = 1.0};
  for (int corner = 0; corner < 8; ++corner) {
    const double x = corner & 1;
    const double y = (corner >> 1) & 1;
    cell.liquid_distance[corner] = y - 0.75;
    cell.solid_distance[corner] = (x + y - 1.25) / sqrt(2.0);
  }
  const auto first = MeasureSampledSolidLiquidCell(cell);
  for (int corner = 0; corner < 8; ++corner) {
    const double x = corner & 1;
    const double y = (corner >> 1) & 1;
    cell.solid_distance[corner] = (x + y - 0.25) / sqrt(2.0);
  }
  const auto second = MeasureSampledSolidLiquidCell(cell);
  *result = 0;
  if (!first.ok() || !second.ok() || fabs(first->wet_volume - 0.125) > 2e-14
      || fabs(second->wet_volume - 0.71875) > 2e-14 || fabs(first->open_area[1] - 0.75) > 2e-14
      || first->open_area[1] != second->open_area[0]) {
    *result = 90;
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
  bool ran = CheckCuda(cudaGetLastError())
      && CheckCuda(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
  if (ran && result == 0) {
    CheckSampledSolidGeometryOnDevice<<<1, 1>>>(device_result);
    ran = CheckCuda(cudaGetLastError())
        && CheckCuda(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
  }
  const bool freed = CheckCuda(cudaFree(device_result));
  if (!ran || !freed) {
    return 1;
  }
  if (result != 0) {
    std::fprintf(stderr, "M5 invariant %d failed on CUDA\n", result);
  }
  return result;
}
