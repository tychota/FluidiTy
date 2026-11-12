#include <utility>
#include <vector>

#include "../src/mac_viscous_pressure_3d.h"
#include "../src/viscosity_velocity_block_3d.h"
#include "fluidity/host/cuda_viscosity.h"
#include "operator_runtime.h"
#include "viscosity_velocity_block_3d.cuh"

namespace fluidity {
namespace {

using cuda_internal::ViscousVelocityDevice3D;
using cuda_internal::ViscousVelocityError3D;
using cuda_internal::ViscousVelocityFace3D;
using cuda_internal::ViscousVelocityState3D;
using cuda_internal::ViscousVelocityTerm3D;

absl::Status Execute(
    std::span<const ViscousVelocityFace3D> faces,
    std::span<const ViscousVelocityTerm3D> terms,
    std::span<const std::size_t> row_offsets,
    ViscousVelocityDevice3D device,
    CudaMacViscousVelocityBlockSolution3D& output,
    cuda_internal::CudaOperatorRuntime& runtime
) {
  if (auto status = runtime.Initialize(); !status.ok()) {
    return status;
  }
  ViscousVelocityFace3D* device_faces = nullptr;
  ViscousVelocityTerm3D* device_terms = nullptr;
  std::size_t* device_offsets = nullptr;
  if (auto status = runtime.Upload(faces, device_faces); !status.ok()) {
    return status;
  }
  if (auto status = runtime.Upload(terms, device_terms); !status.ok()) {
    return status;
  }
  if (auto status = runtime.Upload(row_offsets, device_offsets); !status.ok()) {
    return status;
  }
  device.faces = device_faces;
  device.terms = device_terms;
  device.row_offsets = device_offsets;
  for (double** buffer :
       {&device.velocity, &device.residual, &device.preconditioned, &device.direction, &device.applied}) {
    if (auto status = runtime.Allocate(device.face_count, *buffer); !status.ok()) {
      return status;
    }
  }
  if (auto status = runtime.Allocate(1, device.state); !status.ok()) {
    return status;
  }
  if (auto status = runtime.RecordKernelStart(); !status.ok()) {
    return status;
  }
  if (auto status = cuda_internal::CheckCuda(
          cuda_internal::LaunchViscousVelocityBlock3D(device, runtime.stream()),
          "CUDA viscous velocity launch"
      );
      !status.ok()) {
    return status;
  }
  if (auto status = runtime.SynchronizeAndMeasureKernel(); !status.ok()) {
    return status;
  }
  ViscousVelocityState3D state{};
  if (auto status = runtime.Download(device.state, std::span(&state, 1)); !status.ok()) {
    return status;
  }
  const auto measurement = runtime.ValidateKernelMeasurement(state.compiled_architecture);
  if (!measurement.ok()) {
    return measurement.status();
  }
  switch (state.error) {
    case ViscousVelocityError3D::kNone:
      break;
    case ViscousVelocityError3D::kRange:
      return absl::OutOfRangeError("CUDA velocity-block arithmetic is unrepresentable");
    case ViscousVelocityError3D::kCurvature:
      return absl::AbortedError("CUDA velocity-block curvature is not finite and positive");
    case ViscousVelocityError3D::kExhausted:
      return absl::AbortedError("CUDA velocity-block iteration budget exhausted");
  }
  if (auto status = runtime.Download(device.velocity, std::span(output.solution.velocity)); !status.ok()) {
    return status;
  }
  output.solution.iterations = state.iterations;
  output.solution.max_residual = state.max_residual;
  output.measurement = *measurement;
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<CudaMacViscousVelocityBlockSolution3D> CudaSolveMacViscousVelocityBlock3D(
    const MacViscousVelocityBlockInputs3D& inputs,
    std::span<const double> rhs,
    MacViscousVelocityBlockOptions3D options
) {
  const auto prepared = viscosity_internal::PrepareVelocityBlock3D(inputs, rhs, options);
  if (!prepared.ok()) {
    return prepared.status();
  }
  std::vector<ViscousVelocityFace3D> faces;
  faces.reserve(rhs.size());
  for (std::size_t face = 0; face < rhs.size(); ++face) {
    faces.push_back({
        .inertia = inputs.face_inertias[face],
        .diagonal = prepared->diagonal[face],
        .rhs = rhs[face],
    });
  }
  std::vector<std::size_t> row_offsets{0};
  std::vector<ViscousVelocityTerm3D> terms;
  for (const auto& row : prepared->weighted_rows) {
    for (const auto& term : std::span(row.terms).first(row.term_count)) {
      terms.push_back({
          .face_slot = term.face_slot,
          .weighted_coefficient = term.coefficient,
      });
    }
    row_offsets.push_back(terms.size());
  }
  ViscousVelocityDevice3D device{};
  device.face_count = faces.size();
  device.row_count = prepared->weighted_rows.size();
  device.max_iterations = options.max_iterations;
  device.max_rhs = prepared->max_rhs;
  device.tolerance = options.relative_tolerance * prepared->max_rhs;
  CudaMacViscousVelocityBlockSolution3D output{};
  output.solution.velocity.resize(faces.size());
  absl::Status cleanup;
  absl::Status execution;
  {
    cuda_internal::CudaOperatorRuntime runtime(cleanup);
    execution = Execute(faces, terms, row_offsets, device, output, runtime);
  }
  execution.Update(cleanup);
  if (!execution.ok()) {
    return execution;
  }
  return output;
}

absl::StatusOr<MacViscousPressureProjection3D> CudaProjectMacViscousPressure3D(
    const MacPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options
) {
  return pressure_internal::ProjectMacViscousPressureWithVelocitySolver3D(
      inputs,
      strain_rows,
      options,
      [](const MacViscousVelocityBlockInputs3D& block,
         std::span<const double> rhs,
         MacViscousVelocityBlockOptions3D controls) -> absl::StatusOr<MacViscousVelocityBlockSolution3D> {
        auto result = CudaSolveMacViscousVelocityBlock3D(block, rhs, controls);
        if (!result.ok()) {
          return result.status();
        }
        return std::move(result->solution);
      }
  );
}

}  // namespace fluidity
