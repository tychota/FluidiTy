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
    return;
  }
  const auto weights = fluidity::boundary_internal::MeasureSampledSolidFaceWeights(1.0, {-0.5, 0.5, -0.5, 0.5});
  if (!weights.ok()) {
    *result = 91;
    return;
  }
  constexpr double kExpectedWeights[4]{11.0 / 48, 1.0 / 48, 7.0 / 48, 5.0 / 48};
  constexpr double kNormalVelocities[4]{2, 5, -2, 1};
  double wall_flux = 0;
  for (int corner = 0; corner < 4; ++corner) {
    if (fabs((*weights)[corner] - kExpectedWeights[corner]) > 2e-14) {
      *result = 92;
      return;
    }
    wall_flux += (*weights)[corner] * kNormalVelocities[corner];
  }
  if (fabs(wall_flux - 0.375) > 2e-14) {
    *result = 93;
    return;
  }
  using fluidity::boundary_internal::GeometryInput;
  using fluidity::boundary_internal::GeometryMode;
  GeometryInput query{.mode = GeometryMode::kSampledSolidFace, .geometry = {{}, 1.0}};
  for (int corner = 0; corner < 8; corner += 2) {
    const double y = (corner >> 1) & 1;
    query.derivative_distances[corner] = y - 0.25;
    query.corner_distances[corner] = 2.0 + 4.0 * y;
  }
  const auto geometry = fluidity::boundary_internal::ComputeSampledSolidGeometry(query);
  if (!geometry.ok() || fabs(geometry->prescribed_wall_flux - 0.625) > 2e-14
      || fabs(geometry->open_face_area - 0.75) > 2e-14) {
    *result = 94;
    return;
  }
  const auto reaction_weights =
      fluidity::boundary_internal::MeasureSampledSolidFaceWeights(1.0, {-0.25, 0.75, -0.25, 0.75});
  if (!reaction_weights.ok()) {
    *result = 95;
    return;
  }
  const auto forces = fluidity::boundary_internal::ComputeSampledSolidFacePressureForces(*reaction_weights, 13.0, 5.0);
  if (!forces.ok()) {
    *result = 96;
    return;
  }
  double normal_force = 0;
  double solid_power = 0;
  for (int corner = 0; corner < 4; ++corner) {
    normal_force += (*forces)[corner];
    solid_power += (*forces)[corner] * (2.0 + 4.0 * (corner & 1));
  }
  if (fabs(normal_force - 2.0) > 2e-14 || fabs(solid_power - 8.0 * geometry->prescribed_wall_flux) > 2e-14) {
    *result = 97;
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
