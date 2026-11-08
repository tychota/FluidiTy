#include <cuda_runtime_api.h>

#include <span>

#include "../src/liquid_boundary_backend.h"
#include "fluidity/host/cuda_liquid_boundary_3d.h"
#include "liquid_boundary_3d_bridge.h"
#include "operator_runtime.h"
namespace fluidity {
absl::StatusOr<CudaLiquidWallContactResult3D> CudaResolveLiquidWallContact3D(const LiquidWallContactInputs3D& inputs) {
  const auto staged = boundary_internal::PackContact(inputs);
  cuda_internal::BoundaryReport3D report{};
  CudaOperatorMeasurement measurement{};
  absl::Status cleanup;
  absl::Status status;
  {
    cuda_internal::CudaOperatorRuntime runtime(cleanup);
    const auto execute = [&]() -> absl::Status {
      auto current = runtime.Initialize();
      if (!current.ok()) {
        return current;
      }
      boundary_internal::ContactInput* device_input = nullptr;
      cuda_internal::BoundaryReport3D* device_output = nullptr;
      current = runtime.Upload(std::span(&staged, 1), device_input);
      if (!current.ok()) {
        return current;
      }
      current = runtime.Allocate(1, device_output);
      if (!current.ok()) {
        return current;
      }
      current = runtime.RecordKernelStart();
      if (!current.ok()) {
        return current;
      }
      current = cuda_internal::CheckCuda(
          static_cast<cudaError_t>(
              cuda_internal::LaunchLiquidWallContact3D(device_input, device_output, runtime.stream())
          ),
          "Launch liquid wall contact"
      );
      if (!current.ok()) {
        return current;
      }
      current = runtime.SynchronizeAndMeasureKernel();
      if (!current.ok()) {
        return current;
      }
      current = runtime.Download(device_output, std::span(&report, 1));
      if (!current.ok()) {
        return current;
      }
      const auto measured = runtime.ValidateKernelMeasurement(report.compiled_architecture);
      if (!measured.ok()) {
        return measured.status();
      }
      measurement = *measured;
      return boundary_internal::ArithmeticStatus(report.error);
    };
    status = execute();
  }
  status.Update(cleanup);
  if (!status.ok()) {
    return status;
  }
  return CudaLiquidWallContactResult3D{boundary_internal::UnpackContact(report.contact), measurement};
}
}  // namespace fluidity
