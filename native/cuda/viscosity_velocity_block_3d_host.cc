#include <utility>
#include <vector>

#include "../src/mac_viscous_pressure_3d.h"
#include "../src/viscosity_velocity_block_3d.h"
#include "fluidity/host/cuda_viscosity.h"
#include "liquid_arena_dispatch_3d.h"
#include "liquid_pressure_stage_3d.h"
#include "operator_runtime.h"
#include "viscosity_velocity_block_3d.cuh"

namespace fluidity {
namespace {

using cuda_internal::ViscousVelocityDevice3D;
using cuda_internal::ViscousVelocityError3D;
using cuda_internal::ViscousVelocityFace3D;
using cuda_internal::ViscousVelocityState3D;
using cuda_internal::ViscousVelocityTerm3D;

absl::Status CheckVelocityState(const ViscousVelocityState3D& state) {
  switch (state.error) {
    case ViscousVelocityError3D::kNone:
      return absl::OkStatus();
    case ViscousVelocityError3D::kRange:
      return absl::OutOfRangeError("CUDA velocity-block arithmetic is unrepresentable");
    case ViscousVelocityError3D::kCurvature:
      return absl::AbortedError("CUDA velocity-block curvature is not finite and positive");
    case ViscousVelocityError3D::kExhausted:
      return absl::AbortedError("CUDA velocity-block iteration budget exhausted");
  }
  return absl::InternalError("invalid CUDA velocity-block status");
}

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
  if (auto status = CheckVelocityState(state); !status.ok()) {
    return status;
  }
  if (auto status = runtime.Download(device.velocity, std::span(output.solution.velocity)); !status.ok()) {
    return status;
  }
  output.solution.iterations = state.iterations;
  output.solution.max_residual = state.max_residual;
  output.measurement = *measurement;
  return absl::OkStatus();
}

absl::StatusOr<MacViscousVelocityBlockSolution3D> SolveArenaVelocityBlock(
    CudaLiquidArena3D& arena,
    const MacViscousVelocityBlockInputs3D& inputs,
    std::span<const double> rhs,
    MacViscousVelocityBlockOptions3D options
) {
  using cuda_internal::CheckedLiquidRange3D;
  using cuda_internal::CopyLiquidRange3D;
  using cuda_internal::FinishLiquidDispatch3D;
  using Range = CudaLiquidRange3D;
  const auto prepared = viscosity_internal::PrepareVelocityBlock3D(inputs, rhs, options);
  if (!prepared.ok()) {
    return prepared.status();
  }
  const std::size_t face_count = rhs.size();
  std::size_t term_count = 0;
  for (const auto& row : prepared->weighted_rows) {
    term_count += row.term_count;
  }
  // The inviscid and viscous pressure paths run exclusively. Borrow their
  // common pressure ranges before touching pinned staging or the device.
  auto faces = CheckedLiquidRange3D<ViscousVelocityFace3D>(arena, Range::kPressureLinks, face_count);
  auto terms = CheckedLiquidRange3D<ViscousVelocityTerm3D>(arena, Range::kPressureCoefficients, term_count);
  auto offsets = CheckedLiquidRange3D<std::size_t>(arena, Range::kPressureIndices, prepared->weighted_rows.size() + 1);
  auto scratch = CheckedLiquidRange3D<double>(arena, Range::kPressureCoarseScratch, 5 * face_count);
  auto state = CheckedLiquidRange3D<ViscousVelocityState3D>(arena, Range::kPressureState, 1);
  for (const auto& status : {faces.status(), terms.status(), offsets.status(), scratch.status(), state.status()}) {
    if (!status.ok()) {
      return status;
    }
  }
  for (std::size_t face = 0; face < face_count; ++face) {
    faces->pinned[face] = {
        .inertia = inputs.face_inertias[face],
        .diagonal = prepared->diagonal[face],
        .rhs = rhs[face],
    };
  }
  std::size_t term_index = 0;
  offsets->pinned[0] = 0;
  for (std::size_t row_index = 0; row_index < prepared->weighted_rows.size(); ++row_index) {
    const auto& row = prepared->weighted_rows[row_index];
    for (const auto& term : std::span(row.terms).first(row.term_count)) {
      terms->pinned[term_index++] = {
          .face_slot = term.face_slot,
          .weighted_coefficient = term.coefficient,
      };
    }
    offsets->pinned[row_index + 1] = term_index;
  }
  const ViscousVelocityDevice3D device{
      .faces = faces->device,
      .terms = terms->device,
      .row_offsets = offsets->device,
      .face_count = face_count,
      .row_count = prepared->weighted_rows.size(),
      .max_iterations = options.max_iterations,
      .max_rhs = prepared->max_rhs,
      .tolerance = options.relative_tolerance * prepared->max_rhs,
      .velocity = scratch->device,
      .residual = scratch->device + face_count,
      .preconditioned = scratch->device + 2 * face_count,
      .direction = scratch->device + 3 * face_count,
      .applied = scratch->device + 4 * face_count,
      .state = state->device,
  };
  if (auto status = CopyLiquidRange3D(arena, *faces, true); !status.ok()) {
    return status;
  }
  if (auto status = CopyLiquidRange3D(arena, *terms, true); !status.ok()) {
    return status;
  }
  if (auto status = CopyLiquidRange3D(arena, *offsets, true); !status.ok()) {
    return status;
  }
  if (auto status = arena.BeginMeasuredDispatch(); !status.ok()) {
    return status;
  }
  if (auto status = arena.RecordRuntimeResult(
          cuda_internal::LaunchViscousVelocityBlock3D(device, static_cast<cudaStream_t>(arena.stream()))
      );
      !status.ok()) {
    return status;
  }
  if (auto status = CopyLiquidRange3D(arena, *state, false); !status.ok()) {
    return status;
  }
  const cuda_internal::LiquidTypedRange3D<double> velocity{scratch->device, scratch->pinned.first(face_count)};
  if (auto status = CopyLiquidRange3D(arena, velocity, false); !status.ok()) {
    return status;
  }
  if (auto status = FinishLiquidDispatch3D(arena); !status.ok()) {
    return status;
  }
  const auto& report = state->pinned.front();
  if (report.compiled_architecture != 1200) {
    return absl::FailedPreconditionError("CUDA viscous arena requires an SM120 image");
  }
  if (auto status = CheckVelocityState(report); !status.ok()) {
    return status;
  }
  return MacViscousVelocityBlockSolution3D{
      .velocity = std::vector<double>(velocity.pinned.begin(), velocity.pinned.end()),
      .iterations = report.iterations,
      .max_residual = report.max_residual,
  };
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

absl::StatusOr<MacViscousPressureProjectionView3D> cuda_internal::ProjectArenaViscousPressure3D(
    CudaLiquidArena3D& arena,
    const pressure_internal::UniformViscousPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    MacPressureWorkspace3D& workspace
) {
  return pressure_internal::ProjectWorkspaceMacViscousPressureWithVelocitySolver3D(
      inputs,
      strain_rows,
      options,
      workspace,
      [&](const MacViscousVelocityBlockInputs3D& block,
          std::span<const double> rhs,
          MacViscousVelocityBlockOptions3D controls) {
        return SolveArenaVelocityBlock(arena, block, rhs, controls);
      }
  );
}

}  // namespace fluidity
