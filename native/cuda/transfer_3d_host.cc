#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/cuda_transfers_3d.h"
#include "fluidity/host/views.h"
#include "operator_runtime.h"
#include "transfer_3d_bridge.h"

namespace fluidity {
namespace {

using cuda_internal::CheckCuda;
using cuda_internal::CudaOperatorRuntime;
using cuda_internal::DeviceMacDepositParticle3D;
using cuda_internal::DeviceMacDepositQuantization3D;
using cuda_internal::DeviceMacFaceQuantaAccumulator3D;
using cuda_internal::DeviceMacParticleGatherResult3D;
using cuda_internal::DeviceMacParticleStencil3D;
using cuda_internal::MacTransferArithmeticError3D;
using cuda_internal::MacTransferKernelReport3D;

constexpr std::array kFaceNormals{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
constexpr std::size_t kMaxQuantizedParticleCapacity = std::size_t{1} << 20;

bool IsFinite(Vec3 vector) {
  return std::isfinite(vector.x) && std::isfinite(vector.y) && std::isfinite(vector.z);
}

bool IsFinite(Mat3 matrix) {
  return IsFinite(matrix.x_row) && IsFinite(matrix.y_row) && IsFinite(matrix.z_row);
}

bool IsPositiveNormalPowerOfTwo(double scale) {
  int exponent;
  return std::isnormal(scale) && std::frexp(scale, &exponent) == 0.5;
}

// Resolve every slot before transferring any input to the device. A zero
// weight sample still needs an owner and retains its derivative and offset.
absl::Status ResolveParticleFaceStencils(
    const SparseMacGrid3D& mac_grid,
    const MacTransferParticle3D& particle,
    DeviceMacParticleStencil3D& resolved_particle_stencil
) {
  for (std::size_t axis_index = 0; axis_index < kFaceNormals.size(); ++axis_index) {
    const auto stencil = BuildOwnedMacFaceStencil3D(
        mac_grid,
        kFaceNormals[axis_index],
        particle.sorted_world_position,
        particle.trial_world_position
    );
    if (!stencil.ok()) {
      return stencil.status();
    }
    std::ranges::copy(*stencil, resolved_particle_stencil.face_samples_by_normal[axis_index]);
  }
  return absl::OkStatus();
}

// Synchronize, confirm the image marker and read the arithmetic status before
// copying any numerical results out of device scratch.
absl::StatusOr<CudaOperatorMeasurement> ReadCompletedKernelReport(
    CudaOperatorRuntime& runtime,
    const MacTransferKernelReport3D* device_kernel_report,
    const char* range_diagnostic
) {
  if (auto status = runtime.SynchronizeAndMeasureKernel(); !status.ok()) {
    return status;
  }
  MacTransferKernelReport3D kernel_report;
  if (auto status = runtime.Download(device_kernel_report, std::span(&kernel_report, 1)); !status.ok()) {
    return status;
  }
  const auto measurement = runtime.ValidateKernelMeasurement(kernel_report.compiled_architecture);
  if (!measurement.ok()) {
    return measurement.status();
  }
  if (kernel_report.arithmetic_error == static_cast<int>(MacTransferArithmeticError3D::kOutOfRange)) {
    return absl::OutOfRangeError(range_diagnostic);
  }
  if (kernel_report.arithmetic_error != static_cast<int>(MacTransferArithmeticError3D::kNone)) {
    return absl::InternalError("CUDA 3D transfer kernel reported an unknown error");
  }
  return measurement;
}

absl::StatusOr<CudaOperatorMeasurement> RunAndDownloadDeposit(
    CudaOperatorRuntime& runtime,
    std::span<const DeviceMacDepositParticle3D> particles,
    DeviceMacDepositQuantization3D scales,
    std::span<DeviceMacFaceQuantaAccumulator3D> face_accumulators
) {
  if (auto status = runtime.Initialize(); !status.ok()) {
    return status;
  }
  DeviceMacDepositParticle3D* device_particles = nullptr;
  if (auto status = runtime.Upload(particles, device_particles); !status.ok()) {
    return status;
  }
  DeviceMacFaceQuantaAccumulator3D* device_face_accumulators = nullptr;
  // The empty-batch kernel clears the destination itself.
  const auto allocation_status = particles.empty()
      ? runtime.Allocate(face_accumulators.size(), device_face_accumulators)
      : runtime.AllocateZeroed(face_accumulators.size(), device_face_accumulators);
  if (!allocation_status.ok()) {
    return allocation_status;
  }
  MacTransferKernelReport3D* device_kernel_report = nullptr;
  if (auto status = runtime.AllocateZeroed(1, device_kernel_report); !status.ok()) {
    return status;
  }
  if (auto status = runtime.RecordKernelStart(); !status.ok()) {
    return status;
  }
  const auto launch_error = cuda_internal::LaunchDeposit3D(
      device_particles,
      particles.size(),
      scales,
      device_face_accumulators,
      face_accumulators.size(),
      device_kernel_report,
      runtime.stream()
  );
  if (auto status = CheckCuda(static_cast<cudaError_t>(launch_error), "LaunchDeposit3D"); !status.ok()) {
    return status;
  }
  const auto measurement = ReadCompletedKernelReport(
      runtime,
      device_kernel_report,
      "CUDA 3D deposit kernel contribution is nonfinite or exceeds the scaled magnitude bound"
  );
  if (!measurement.ok()) {
    return measurement.status();
  }
  if (auto status = runtime.Download(device_face_accumulators, face_accumulators); !status.ok()) {
    return status;
  }
  return measurement;
}

absl::StatusOr<CudaOperatorMeasurement> RunAndDownloadGather(
    CudaOperatorRuntime& runtime,
    std::span<const DeviceMacParticleStencil3D> particle_stencils,
    std::span<const double> old_face_velocities,
    std::span<const double> new_face_velocities,
    double cell_width,
    std::span<DeviceMacParticleGatherResult3D> particle_results
) {
  if (auto status = runtime.Initialize(); !status.ok()) {
    return status;
  }
  DeviceMacParticleStencil3D* device_particle_stencils = nullptr;
  if (auto status = runtime.Upload(particle_stencils, device_particle_stencils); !status.ok()) {
    return status;
  }
  double* device_old_face_velocities = nullptr;
  if (auto status = runtime.Upload(old_face_velocities, device_old_face_velocities); !status.ok()) {
    return status;
  }
  double* device_new_face_velocities = nullptr;
  if (auto status = runtime.Upload(new_face_velocities, device_new_face_velocities); !status.ok()) {
    return status;
  }
  DeviceMacParticleGatherResult3D* device_particle_results = nullptr;
  if (auto status = runtime.Allocate(particle_results.size(), device_particle_results); !status.ok()) {
    return status;
  }
  MacTransferKernelReport3D* device_kernel_report = nullptr;
  if (auto status = runtime.AllocateZeroed(1, device_kernel_report); !status.ok()) {
    return status;
  }
  if (auto status = runtime.RecordKernelStart(); !status.ok()) {
    return status;
  }
  const auto launch_error = cuda_internal::LaunchGather3D(
      device_particle_stencils,
      particle_stencils.size(),
      {device_old_face_velocities, device_new_face_velocities},
      cell_width,
      device_particle_results,
      device_kernel_report,
      runtime.stream()
  );
  if (auto status = CheckCuda(static_cast<cudaError_t>(launch_error), "LaunchGather3D"); !status.ok()) {
    return status;
  }
  const auto measurement =
      ReadCompletedKernelReport(runtime, device_kernel_report, "CUDA 3D gather kernel velocity or slope is nonfinite");
  if (!measurement.ok()) {
    return measurement.status();
  }
  if (auto status = runtime.Download(device_particle_results, particle_results); !status.ok()) {
    return status;
  }
  return measurement;
}

}  // namespace

absl::Status CudaDepositMacQuanta3D(
    const SparseMacGrid3D& mac_grid,
    std::span<const MacTransferParticle3D> particles,
    MacDepositQuantization3D quantization,
    std::span<MacFaceDepositQuanta3D> face_deposits,
    CudaOperatorMeasurement& measurement
) {
  if (!IsPositiveNormalPowerOfTwo(quantization.mass_per_quantum)
      || !IsPositiveNormalPowerOfTwo(quantization.pure_momentum_per_quantum)) {
    return absl::InvalidArgumentError("quantization scales must be positive normal powers of two");
  }
  if (quantization.declared_particle_capacity > kMaxQuantizedParticleCapacity) {
    return absl::ResourceExhaustedError("declared particle capacity exceeds the transfer limit");
  }
  if (particles.size() > quantization.declared_particle_capacity) {
    return absl::ResourceExhaustedError("particle count exceeds declared capacity");
  }
  if (auto status = ValidateExtent(face_deposits.size(), mac_grid.face_count()); !status.ok()) {
    return status;
  }
  if (auto status = CheckWritableViews(
          {std::as_bytes(face_deposits), std::as_bytes(std::span(&measurement, 1))},
          {std::as_bytes(particles)}
      );
      !status.ok()) {
    return status;
  }
  if (particles.empty() && face_deposits.empty()) {
    return absl::OkStatus();
  }

  std::vector<DeviceMacDepositParticle3D> resolved_deposit_particles(particles.size());
  for (std::size_t particle_index = 0; particle_index < particles.size(); ++particle_index) {
    const auto& particle = particles[particle_index];
    if (!std::isfinite(particle.particle_mass) || particle.particle_mass < 0.0) {
      return absl::InvalidArgumentError("particle mass must be finite and nonnegative");
    }
    if (!IsFinite(particle.particle_velocity) || !IsFinite(particle.particle_velocity_slopes)) {
      return absl::InvalidArgumentError("particle velocity and affine fit must be finite");
    }
    auto& resolved_deposit_particle = resolved_deposit_particles[particle_index];
    if (auto status = ResolveParticleFaceStencils(mac_grid, particle, resolved_deposit_particle.stencil);
        !status.ok()) {
      return status;
    }
    resolved_deposit_particle.particle_mass = particle.particle_mass;
    resolved_deposit_particle.particle_velocity = particle.particle_velocity;
    resolved_deposit_particle.particle_velocity_slopes = particle.particle_velocity_slopes;
  }
  const std::size_t nonzero_particle_capacity = std::max(std::size_t{1}, quantization.declared_particle_capacity);
  const int particle_capacity_exponent = static_cast<int>(std::bit_width(nonzero_particle_capacity - 1));
  const DeviceMacDepositQuantization3D scales{
      .mass_per_quantum = quantization.mass_per_quantum,
      .pure_momentum_per_quantum = quantization.pure_momentum_per_quantum,
      .scaled_contribution_magnitude_limit = std::ldexp(1.0, 62 - particle_capacity_exponent),
  };
  std::vector<DeviceMacFaceQuantaAccumulator3D> face_accumulators(face_deposits.size());
  // Keep the runtime inside this scope so its cleanup status is available
  // before any caller-visible output is published.
  absl::Status cleanup_status;
  const auto staged_measurement = [&] {
    CudaOperatorRuntime runtime(cleanup_status);
    return RunAndDownloadDeposit(runtime, resolved_deposit_particles, scales, face_accumulators);
  }();
  if (!staged_measurement.ok()) {
    return staged_measurement.status();
  }
  if (!cleanup_status.ok()) {
    return cleanup_status;
  }

  for (std::size_t face_slot = 0; face_slot < face_deposits.size(); ++face_slot) {
    const auto& face_accumulator = face_accumulators[face_slot];
    // Decode the device's modulo-2^64 sum as the same signed representation.
    face_deposits[face_slot] = {
        .mass_quanta_sum = std::bit_cast<std::int64_t>(face_accumulator.mass_quanta_sum),
        .pure_momentum_quanta_sum = std::bit_cast<std::int64_t>(face_accumulator.pure_momentum_quanta_sum),
        .stress_force_quanta_sum = 0,
    };
  }
  measurement = *staged_measurement;
  return absl::OkStatus();
}

absl::Status CudaGatherMac3D(
    const SparseMacGrid3D& mac_grid,
    std::span<const MacTransferParticle3D> particles,
    std::span<const double> old_face_velocities,
    std::span<const double> new_face_velocities,
    std::span<MacParticleGatherResult3D> particle_gather_results,
    CudaOperatorMeasurement& measurement
) {
  if (auto status = CheckExtents(mac_grid.face_count(), {old_face_velocities.size(), new_face_velocities.size()});
      !status.ok()) {
    return status;
  }
  if (auto status = ValidateExtent(particle_gather_results.size(), particles.size()); !status.ok()) {
    return status;
  }
  if (auto status = CheckWritableViews(
          {std::as_bytes(particle_gather_results), std::as_bytes(std::span(&measurement, 1))},
          {std::as_bytes(particles), std::as_bytes(old_face_velocities), std::as_bytes(new_face_velocities)}
      );
      !status.ok()) {
    return status;
  }
  const auto is_finite_face_velocity = [](double face_velocity) {
    return std::isfinite(face_velocity);
  };
  if (!std::ranges::all_of(old_face_velocities, is_finite_face_velocity)
      || !std::ranges::all_of(new_face_velocities, is_finite_face_velocity)) {
    return absl::InvalidArgumentError("old and new face velocity fields must be finite");
  }
  if (particles.empty()) {
    return absl::OkStatus();
  }

  const auto particle_stencil_bytes = CheckedExtentProduct(particles.size(), sizeof(DeviceMacParticleStencil3D));
  if (!particle_stencil_bytes.ok()) {
    return particle_stencil_bytes.status();
  }
  std::vector<DeviceMacParticleStencil3D> particle_stencils(particles.size());
  for (std::size_t particle_index = 0; particle_index < particles.size(); ++particle_index) {
    if (auto status =
            ResolveParticleFaceStencils(mac_grid, particles[particle_index], particle_stencils[particle_index]);
        !status.ok()) {
      return status;
    }
  }
  std::vector<DeviceMacParticleGatherResult3D> staged_gathers(particles.size());
  // Keep the runtime inside this scope so its cleanup status is available
  // before any caller-visible output is published.
  absl::Status cleanup_status;
  const auto staged_measurement = [&] {
    CudaOperatorRuntime runtime(cleanup_status);
    return RunAndDownloadGather(
        runtime,
        particle_stencils,
        old_face_velocities,
        new_face_velocities,
        mac_grid.geometry().cell_width,
        staged_gathers
    );
  }();
  if (!staged_measurement.ok()) {
    return staged_measurement.status();
  }
  if (!cleanup_status.ok()) {
    return cleanup_status;
  }

  for (std::size_t particle_index = 0; particle_index < particle_gather_results.size(); ++particle_index) {
    const auto& staged_particle_result = staged_gathers[particle_index];
    particle_gather_results[particle_index] = {
        .new_pic_velocity = staged_particle_result.new_pic_velocity,
        .grid_velocity_increment = staged_particle_result.grid_velocity_increment,
        .fitted_velocity_slopes = staged_particle_result.fitted_velocity_slopes,
        .velocity_gradient = staged_particle_result.velocity_gradient,
    };
  }
  measurement = *staged_measurement;
  return absl::OkStatus();
}

}  // namespace fluidity
