#include <cuda_runtime_api.h>

#include <span>

#include "../src/liquid_boundary_backend.h"
#include "fluidity/host/cuda_liquid_boundary_3d.h"
#include "liquid_boundary_3d_bridge.h"
#include "operator_runtime.h"
namespace fluidity {
namespace {
using boundary_internal::GeometryInput;
using boundary_internal::GeometryOutput;
using cuda_internal::CheckCuda;
// Two reusable allocations bound staging independently of cell/face count.
// Upload storage and reports outlive runtime cleanup, including early failures.
struct GeometryStaging {
  GeometryInput input{};
  cuda_internal::BoundaryReport3D report{};
};
class GeometryExecution {
 public:
  GeometryExecution(cuda_internal::CudaOperatorRuntime& runtime, GeometryStaging& staging)
      : runtime_(runtime), staging_(staging) {
  }
  absl::Status Initialize() {
    auto status = runtime_.Initialize();
    if (!status.ok()) {
      return status;
    }
    status = runtime_.Allocate(1, input_);
    if (!status.ok()) {
      return status;
    }
    status = runtime_.Allocate(1, output_);
    if (!status.ok()) {
      return status;
    }
    // Empty or fully sealed inputs still execute a device marker; no numerical
    // output can accidentally claim an unexecuted architecture.
    const auto marker = Compute(GeometryInput{});
    return marker.status();
  }
  static absl::StatusOr<GeometryOutput> Compute(void* context, const GeometryInput& input) {
    return static_cast<GeometryExecution*>(context)->Compute(input);
  }
  absl::StatusOr<GeometryOutput> Compute(const GeometryInput& input) {
    staging_.input = input;
    auto status = CheckCuda(
        cudaMemcpyAsync(input_, &staging_.input, sizeof(staging_.input), cudaMemcpyHostToDevice, runtime_.stream()),
        "Upload liquid geometry query"
    );
    if (!status.ok()) {
      return status;
    }
    status = runtime_.RecordKernelStart();
    if (!status.ok()) {
      return status;
    }
    status = CheckCuda(
        static_cast<cudaError_t>(cuda_internal::LaunchLiquidBoundaryGeometry3D(input_, output_, runtime_.stream())),
        "Launch liquid geometry"
    );
    if (!status.ok()) {
      return status;
    }
    status = runtime_.SynchronizeAndMeasureKernel();
    if (!status.ok()) {
      return status;
    }
    status = runtime_.Download(output_, std::span(&staging_.report, 1));
    if (!status.ok()) {
      return status;
    }
    const auto measured = runtime_.ValidateKernelMeasurement(staging_.report.compiled_architecture);
    if (!measured.ok()) {
      return measured.status();
    }
    measurement_.compiled_architecture = measured->compiled_architecture;
    measurement_.kernel_ms += measured->kernel_ms;
    status = boundary_internal::ArithmeticStatus(staging_.report.error);
    if (!status.ok()) {
      return status;
    }
    return staging_.report.geometry;
  }
  CudaOperatorMeasurement measurement() const {
    return measurement_;
  }

 private:
  cuda_internal::CudaOperatorRuntime& runtime_;
  GeometryInput* input_ = nullptr;
  cuda_internal::BoundaryReport3D* output_ = nullptr;
  GeometryStaging& staging_;
  CudaOperatorMeasurement measurement_{};
};
}  // namespace
absl::StatusOr<CudaLiquidBoundaryGeometry3D> CudaFreezeLiquidBoundaryGeometry3D(
    const LiquidBoundaryGeometryInputs3D& inputs
) {
  GeometryStaging staging;
  absl::Status cleanup;
  absl::StatusOr<CudaLiquidBoundaryGeometry3D> result = absl::InternalError("Unstarted geometry");
  // Keep staging alive until the runtime has synchronized and released buffers.
  // Successful callbacks synchronize before the staging object can be reused.
  {
    cuda_internal::CudaOperatorRuntime runtime(cleanup);
    GeometryExecution execution(runtime, staging);
    const auto run = [&]() -> absl::StatusOr<CudaLiquidBoundaryGeometry3D> {
      auto status = execution.Initialize();
      if (!status.ok()) {
        return status;
      }
      const auto geometry = boundary_internal::FreezeGeometry(inputs, {GeometryExecution::Compute, &execution});
      if (!geometry.ok()) {
        return geometry.status();
      }
      return CudaLiquidBoundaryGeometry3D{*geometry, execution.measurement()};
    };
    result = run();
  }
  if (!cleanup.ok()) {
    return cleanup;
  }
  return result;
}
absl::StatusOr<CudaLiquidPressureProblem3D>
CudaBuildLiquidPressureLinks3D(const FrozenLiquidBoundaryGeometry3D& geometry, const LiquidPressureFields3D& fields) {
  GeometryStaging staging;
  absl::Status cleanup;
  absl::StatusOr<CudaLiquidPressureProblem3D> result = absl::InternalError("Unstarted links");
  {
    cuda_internal::CudaOperatorRuntime runtime(cleanup);
    GeometryExecution execution(runtime, staging);
    const auto run = [&]() -> absl::StatusOr<CudaLiquidPressureProblem3D> {
      auto status = execution.Initialize();
      if (!status.ok()) {
        return status;
      }
      const auto problem = boundary_internal::BuildLinks(geometry, fields, {GeometryExecution::Compute, &execution});
      if (!problem.ok()) {
        return problem.status();
      }
      return CudaLiquidPressureProblem3D{*problem, execution.measurement()};
    };
    result = run();
  }
  if (!cleanup.ok()) {
    return cleanup;
  }
  return result;
}
}  // namespace fluidity
