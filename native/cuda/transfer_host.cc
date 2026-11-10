#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>

#include "fluidity/host/basis.h"
#include "fluidity/host/cuda_transfers.h"
#include "fluidity/host/views.h"
#include "operator_runtime.h"
#include "transfer_bridge.h"

namespace fluidity {
namespace cuda_internal {
namespace {

bool IsFinite(Vec2 vector) {
  return std::isfinite(vector.x) && std::isfinite(vector.y);
}

bool IsFinite(Mat2 matrix) {
  return std::isfinite(matrix.xx) && std::isfinite(matrix.xy) && std::isfinite(matrix.yx) && std::isfinite(matrix.yy);
}

}  // namespace

absl::Status
PrepareTransferStencil(const Grid2D& mac_grid, Vec2 particle_world_position, TransferStencil& particle_stencil) {
  if (!IsFinite(particle_world_position)) {
    return absl::InvalidArgumentError("particle position must be finite");
  }
  const Vec2 particle_position_from_origin = particle_world_position - mac_grid.origin;
  if (!IsFinite(particle_position_from_origin)) {
    return absl::OutOfRangeError("particle position relative to grid origin is nonfinite");
  }

  for (std::size_t velocity_component_index = 0; velocity_component_index < kTransferComponents;
       ++velocity_component_index) {
    const bool is_x_normal = velocity_component_index == 0;
    const Vec2 lattice_offset = is_x_normal ? Vec2{0.0, 0.5} : Vec2{0.5, 0.0};
    const auto quadratic_basis = QuadraticStencil2D(particle_position_from_origin, mac_grid.h, lattice_offset);
    if (!quadratic_basis.ok()) {
      return quadratic_basis.status();
    }

    const std::int64_t face_column_count = static_cast<std::int64_t>(mac_grid.nx) + (is_x_normal ? 1 : 0);
    const std::int64_t face_row_count = static_cast<std::int64_t>(mac_grid.ny) + (is_x_normal ? 0 : 1);
    const std::size_t orientation_face_offset =
        is_x_normal ? 0 : (static_cast<std::size_t>(mac_grid.nx) + 1) * mac_grid.ny;
    for (std::size_t stencil_slot_index = 0; stencil_slot_index < kTransferSlots; ++stencil_slot_index) {
      const Node2D lattice_node = quadratic_basis->nodes[stencil_slot_index];
      // Complete support includes zero-weight nodes at exact half-cell ties.
      const bool node_on_face_lattice = lattice_node.x >= 0 && lattice_node.y >= 0 && lattice_node.x < face_column_count
          && lattice_node.y < face_row_count;
      if (!node_on_face_lattice) {
        return absl::InvalidArgumentError("particle does not have complete MAC support");
      }
      const auto face_slot =
          orientation_face_offset + static_cast<std::size_t>(lattice_node.y * face_column_count + lattice_node.x);
      const auto face_geometry = FaceGeometry(mac_grid, face_slot);
      if (!face_geometry.ok()) {
        return face_geometry.status();
      }
      const Vec2 particle_to_face_offset = face_geometry->world_position - particle_world_position;
      if (!IsFinite(particle_to_face_offset)) {
        return absl::OutOfRangeError("face position relative to particle is nonfinite");
      }
      particle_stencil.component_slots[velocity_component_index][stencil_slot_index] = {
          .face_slot = face_slot,
          .basis_weight = quadratic_basis->weights[stencil_slot_index],
          .basis_gradient = quadratic_basis->gradients[stencil_slot_index],
          .particle_to_face_offset = particle_to_face_offset,
      };
    }
  }
  return absl::OkStatus();
}

namespace {

bool IsBinaryScale(double scale) {
  int exponent;
  return scale > 0.0 && std::isnormal(scale) && std::frexp(scale, &exponent) == 0.5;
}

absl::Status PrepareDeposit(
    const Grid2D& mac_grid,
    ParticleView2D particle_inputs,
    std::span<const double> particle_reference_volumes,
    std::span<const Mat2> particle_kirchhoff_stresses,
    std::span<DepositParticle> prepared_particles
) {
  for (std::size_t particle_index = 0; particle_index < prepared_particles.size(); ++particle_index) {
    const double particle_mass = particle_inputs.masses[particle_index];
    const double particle_reference_volume = particle_reference_volumes[particle_index];
    const bool mass_is_valid = std::isfinite(particle_mass) && particle_mass >= 0.0;
    const bool reference_volume_is_valid = std::isfinite(particle_reference_volume) && particle_reference_volume >= 0.0;
    if (!mass_is_valid || !reference_volume_is_valid) {
      return absl::InvalidArgumentError("particle mass and reference volume must be finite and nonnegative");
    }
    const Vec2 particle_velocity = particle_inputs.velocities[particle_index];
    const Mat2 velocity_fit_slopes = particle_inputs.velocity_fit_slopes[particle_index];
    const Mat2 kirchhoff_stress = particle_kirchhoff_stresses[particle_index];
    if (!IsFinite(particle_velocity) || !IsFinite(velocity_fit_slopes) || !IsFinite(kirchhoff_stress)) {
      return absl::InvalidArgumentError("particle velocity, velocity fit slopes and Kirchhoff stress must be finite");
    }
    auto& prepared_particle = prepared_particles[particle_index];
    if (auto status =
            PrepareTransferStencil(mac_grid, particle_inputs.positions[particle_index], prepared_particle.stencil);
        !status.ok()) {
      return status;
    }
    prepared_particle.mass = particle_mass;
    prepared_particle.velocity = particle_velocity;
    prepared_particle.velocity_fit_slopes = velocity_fit_slopes;
    prepared_particle.reference_volume = particle_reference_volume;
    prepared_particle.kirchhoff_stress = kirchhoff_stress;
  }
  return absl::OkStatus();
}

absl::Status CollectDeposit(
    CudaOperatorRuntime& runtime,
    std::span<const DepositParticle> particles,
    CudaDepositScales scales,
    std::span<DepositFaceAccumulator> face_accumulators,
    CudaOperatorMeasurement& staged_measurement
) {
  if (auto status = runtime.Initialize(); !status.ok()) {
    return status;
  }
  DepositParticle* device_particles = nullptr;
  if (auto status = runtime.Upload(particles, device_particles); !status.ok()) {
    return status;
  }
  DepositFaceAccumulator* device_face_accumulators = nullptr;
  if (auto status = runtime.AllocateZeroed(face_accumulators.size(), device_face_accumulators); !status.ok()) {
    return status;
  }
  DepositKernelState* device_deposit_state = nullptr;
  if (auto status = runtime.AllocateZeroed(1, device_deposit_state); !status.ok()) {
    return status;
  }

  if (auto status = runtime.RecordKernelStart(); !status.ok()) {
    return status;
  }
  const int deposit_launch_code = LaunchDeposit(
      device_particles,
      particles.size(),
      {
          .mass_per_quantum = scales.mass,
          .pure_momentum_per_quantum = scales.pure_momentum,
          .stress_force_per_quantum = scales.stress_force,
      },
      device_face_accumulators,
      face_accumulators.size(),
      device_deposit_state,
      runtime.stream()
  );
  if (auto status = CheckCuda(static_cast<cudaError_t>(deposit_launch_code), "LaunchDeposit"); !status.ok()) {
    return status;
  }
  if (auto status = runtime.SynchronizeAndMeasureKernel(); !status.ok()) {
    return status;
  }

  DepositKernelState downloaded_deposit_state;
  if (auto status = runtime.Download(device_deposit_state, std::span(&downloaded_deposit_state, 1)); !status.ok()) {
    return status;
  }
  const auto measured_kernel = runtime.ValidateKernelMeasurement(downloaded_deposit_state.compiled_architecture);
  if (!measured_kernel.ok()) {
    return measured_kernel.status();
  }
  if (downloaded_deposit_state.error == static_cast<int>(DepositError::kOutOfRange)) {
    return absl::OutOfRangeError("CUDA deposit contribution is nonfinite or exceeds the scaled magnitude bound");
  }
  if (downloaded_deposit_state.error != static_cast<int>(DepositError::kNone)) {
    return absl::InternalError("CUDA deposit encountered an invalid prepared face index");
  }
  if (auto status = runtime.Download(device_face_accumulators, face_accumulators); !status.ok()) {
    return status;
  }
  staged_measurement = *measured_kernel;
  return absl::OkStatus();
}

absl::Status CollectGather(
    CudaOperatorRuntime& runtime,
    std::span<const TransferStencil> particle_stencils,
    std::span<const double> old_face_velocities,
    std::span<const double> new_face_velocities,
    std::span<GatherParticleResult> staged_particle_results,
    CudaOperatorMeasurement& staged_measurement
) {
  if (auto initialization_status = runtime.Initialize(); !initialization_status.ok()) {
    return initialization_status;
  }
  TransferStencil* device_particle_stencils = nullptr;
  if (auto upload_status = runtime.Upload(particle_stencils, device_particle_stencils); !upload_status.ok()) {
    return upload_status;
  }
  double* device_old_face_velocities = nullptr;
  if (auto upload_status = runtime.Upload(old_face_velocities, device_old_face_velocities); !upload_status.ok()) {
    return upload_status;
  }
  double* device_new_face_velocities = nullptr;
  if (auto upload_status = runtime.Upload(new_face_velocities, device_new_face_velocities); !upload_status.ok()) {
    return upload_status;
  }
  GatherParticleResult* device_particle_results = nullptr;
  if (auto allocation_status = runtime.Allocate(staged_particle_results.size(), device_particle_results);
      !allocation_status.ok()) {
    return allocation_status;
  }
  GatherKernelState* device_gather_state = nullptr;
  if (auto allocation_status = runtime.AllocateZeroed(1, device_gather_state); !allocation_status.ok()) {
    return allocation_status;
  }

  if (auto kernel_start_status = runtime.RecordKernelStart(); !kernel_start_status.ok()) {
    return kernel_start_status;
  }
  const int gather_launch_code = LaunchGather(
      device_particle_stencils,
      particle_stencils.size(),
      {
          .old_face_velocities = device_old_face_velocities,
          .new_face_velocities = device_new_face_velocities,
          .face_count = new_face_velocities.size(),
      },
      device_particle_results,
      device_gather_state,
      runtime.stream()
  );
  if (auto launch_status = CheckCuda(static_cast<cudaError_t>(gather_launch_code), "LaunchGather");
      !launch_status.ok()) {
    return launch_status;
  }
  if (auto kernel_completion_status = runtime.SynchronizeAndMeasureKernel(); !kernel_completion_status.ok()) {
    return kernel_completion_status;
  }

  GatherKernelState downloaded_gather_state;
  if (auto download_status = runtime.Download(device_gather_state, std::span(&downloaded_gather_state, 1));
      !download_status.ok()) {
    return download_status;
  }
  const auto measured_kernel = runtime.ValidateKernelMeasurement(downloaded_gather_state.compiled_architecture);
  if (!measured_kernel.ok()) {
    return measured_kernel.status();
  }
  if ((downloaded_gather_state.error_bits & static_cast<unsigned int>(GatherError::kInvalidFaceIndex)) != 0) {
    return absl::InternalError("CUDA gather encountered an invalid prepared face index");
  }
  if ((downloaded_gather_state.error_bits & static_cast<unsigned int>(GatherError::kNonfiniteArithmetic)) != 0) {
    return absl::OutOfRangeError("CUDA gather arithmetic is nonfinite");
  }
  if ((downloaded_gather_state.error_bits & static_cast<unsigned int>(GatherError::kUnsafeAffineMoment)) != 0) {
    return absl::FailedPreconditionError("CUDA gather affine moment is numerically singular");
  }
  if (auto download_status = runtime.Download(device_particle_results, staged_particle_results);
      !download_status.ok()) {
    return download_status;
  }
  staged_measurement = *measured_kernel;
  return absl::OkStatus();
}

}  // namespace
}  // namespace cuda_internal

absl::Status CudaP2G(
    const Grid2D& mac_grid,
    ParticleView2D particle_inputs,
    std::span<const double> particle_reference_volumes,
    std::span<const Mat2> particle_kirchhoff_stresses,
    CudaDepositScales quantization_scales,
    CudaDepositViews deposit_outputs,
    CudaOperatorMeasurement& kernel_measurement
) {
  const std::size_t particle_count = particle_inputs.positions.size();
  if (particle_count > cuda_internal::kMaxTransferParticles) {
    return absl::ResourceExhaustedError("CUDA deposit particle capacity exceeded");
  }
  const auto face_count = FaceCount(mac_grid);
  if (!face_count.ok()) {
    return face_count.status();
  }
  if (*face_count > cuda_internal::kMaxTransferFaces) {
    return absl::ResourceExhaustedError("CUDA deposit face capacity exceeded");
  }

  const auto particle_extent_status = CheckExtents(
      particle_count,
      {particle_inputs.masses.size(),
       particle_inputs.velocities.size(),
       particle_inputs.velocity_fit_slopes.size(),
       particle_reference_volumes.size(),
       particle_kirchhoff_stresses.size()}
  );
  if (!particle_extent_status.ok()) {
    return particle_extent_status;
  }
  const auto face_extents = CheckExtents(
      *face_count,
      {deposit_outputs.mass_quanta.size(),
       deposit_outputs.pure_momentum_quanta.size(),
       deposit_outputs.stress_force_quanta.size()}
  );
  if (!face_extents.ok()) {
    return face_extents;
  }
  const auto output_alias_status = CheckWritableViews(
      {
          std::as_bytes(deposit_outputs.mass_quanta),
          std::as_bytes(deposit_outputs.pure_momentum_quanta),
          std::as_bytes(deposit_outputs.stress_force_quanta),
          std::as_bytes(std::span(&kernel_measurement, 1)),
      },
      {
          std::as_bytes(std::span(&mac_grid, 1)),
          std::as_bytes(particle_inputs.positions),
          std::as_bytes(particle_inputs.masses),
          std::as_bytes(particle_inputs.velocities),
          std::as_bytes(particle_inputs.velocity_fit_slopes),
          std::as_bytes(particle_reference_volumes),
          std::as_bytes(particle_kirchhoff_stresses),
      }
  );
  if (!output_alias_status.ok()) {
    return output_alias_status;
  }
  if (!cuda_internal::IsBinaryScale(quantization_scales.mass)
      || !cuda_internal::IsBinaryScale(quantization_scales.pure_momentum)
      || !cuda_internal::IsBinaryScale(quantization_scales.stress_force)) {
    return absl::InvalidArgumentError("CUDA deposit scales must be positive normal powers of two");
  }

  // Keep bounded staging off the host stack and alive until pending CUDA work
  // has completed, including cleanup on every early failure.
  std::unique_ptr<cuda_internal::DepositParticle[]> prepared_particles;
  if (particle_count != 0) {
    prepared_particles.reset(new (std::nothrow) cuda_internal::DepositParticle[particle_count]);
    if (prepared_particles == nullptr) {
      return absl::ResourceExhaustedError("could not allocate CUDA deposit particle staging");
    }
  }
  const auto particle_staging = std::span(prepared_particles.get(), particle_count);
  const auto preparation = cuda_internal::PrepareDeposit(
      mac_grid,
      particle_inputs,
      particle_reference_volumes,
      particle_kirchhoff_stresses,
      particle_staging
  );
  if (!preparation.ok()) {
    return preparation;
  }
  const std::unique_ptr<cuda_internal::DepositFaceAccumulator[]> face_accumulators{
      new (std::nothrow) cuda_internal::DepositFaceAccumulator[*face_count],
  };
  if (face_accumulators == nullptr) {
    return absl::ResourceExhaustedError("could not allocate CUDA deposit face staging");
  }

  CudaOperatorMeasurement staged_measurement;
  absl::Status status;
  absl::Status cleanup_status;
  {
    cuda_internal::CudaOperatorRuntime runtime(cleanup_status);
    status = cuda_internal::CollectDeposit(
        runtime,
        particle_staging,
        quantization_scales,
        std::span(face_accumulators.get(), *face_count),
        staged_measurement
    );
  }
  status.Update(cleanup_status);
  if (!status.ok()) {
    return status;
  }

  static_assert(std::numeric_limits<unsigned long long>::digits == 64);
  static_assert(std::bit_cast<unsigned long long>(std::int64_t{-1}) == std::numeric_limits<unsigned long long>::max());
  for (std::size_t face_slot = 0; face_slot < *face_count; ++face_slot) {
    deposit_outputs.mass_quanta[face_slot] = std::bit_cast<std::int64_t>(face_accumulators[face_slot].mass_quanta_sum);
    deposit_outputs.pure_momentum_quanta[face_slot] =
        std::bit_cast<std::int64_t>(face_accumulators[face_slot].pure_momentum_quanta_sum);
    deposit_outputs.stress_force_quanta[face_slot] =
        std::bit_cast<std::int64_t>(face_accumulators[face_slot].stress_force_quanta_sum);
  }
  kernel_measurement = staged_measurement;
  return absl::OkStatus();
}

absl::Status CudaG2P(
    const Grid2D& mac_grid,
    std::span<const Vec2> particle_world_positions,
    std::span<const double> old_face_velocities,
    std::span<const double> new_face_velocities,
    CudaGatherViews gather_outputs,
    CudaOperatorMeasurement& kernel_measurement
) {
  const std::size_t particle_count = particle_world_positions.size();
  if (particle_count > cuda_internal::kMaxTransferParticles) {
    return absl::ResourceExhaustedError("CUDA gather particle capacity exceeded");
  }
  const auto face_count = FaceCount(mac_grid);
  if (!face_count.ok()) {
    return face_count.status();
  }
  if (*face_count > cuda_internal::kMaxTransferFaces) {
    return absl::ResourceExhaustedError("CUDA gather face capacity exceeded");
  }

  const auto face_extents = CheckExtents(*face_count, {old_face_velocities.size(), new_face_velocities.size()});
  if (!face_extents.ok()) {
    return face_extents;
  }
  const auto particle_extent_status = CheckExtents(
      particle_count,
      {
          gather_outputs.pic_velocities.size(),
          gather_outputs.grid_velocity_increments.size(),
          gather_outputs.velocity_fit_slopes.size(),
          gather_outputs.velocity_gradients.size(),
      }
  );
  if (!particle_extent_status.ok()) {
    return particle_extent_status;
  }
  const auto disjoint_gather_views = CheckWritableViews(
      {
          std::as_bytes(gather_outputs.pic_velocities),
          std::as_bytes(gather_outputs.grid_velocity_increments),
          std::as_bytes(gather_outputs.velocity_fit_slopes),
          std::as_bytes(gather_outputs.velocity_gradients),
          std::as_bytes(std::span(&kernel_measurement, 1)),
      },
      {
          std::as_bytes(std::span(&mac_grid, 1)),
          std::as_bytes(particle_world_positions),
          std::as_bytes(old_face_velocities),
          std::as_bytes(new_face_velocities),
      }
  );
  if (!disjoint_gather_views.ok()) {
    return disjoint_gather_views;
  }
  const auto finite_face_velocity = [](double face_velocity) {
    return std::isfinite(face_velocity);
  };
  if (!std::ranges::all_of(old_face_velocities, finite_face_velocity)
      || !std::ranges::all_of(new_face_velocities, finite_face_velocity)) {
    return absl::InvalidArgumentError("old and new face velocities must be finite");
  }
  if (particle_world_positions.empty()) {
    return absl::OkStatus();
  }

  // Keep upload and result staging alive through cleanup and off the host stack.
  const std::unique_ptr<cuda_internal::TransferStencil[]> particle_stencils{
      new (std::nothrow) cuda_internal::TransferStencil[particle_count],
  };
  if (particle_stencils == nullptr) {
    return absl::ResourceExhaustedError("could not allocate CUDA gather stencil staging");
  }
  for (std::size_t particle_index = 0; particle_index < particle_count; ++particle_index) {
    const auto stencil_status = cuda_internal::PrepareTransferStencil(
        mac_grid,
        particle_world_positions[particle_index],
        particle_stencils[particle_index]
    );
    if (!stencil_status.ok()) {
      return stencil_status;
    }
  }
  const std::unique_ptr<cuda_internal::GatherParticleResult[]> staged_particle_results{
      new (std::nothrow) cuda_internal::GatherParticleResult[particle_count],
  };
  if (staged_particle_results == nullptr) {
    return absl::ResourceExhaustedError("could not allocate CUDA gather result staging");
  }

  CudaOperatorMeasurement staged_measurement;
  absl::Status gather_status;
  absl::Status cleanup_status;
  {
    cuda_internal::CudaOperatorRuntime runtime(cleanup_status);
    gather_status = cuda_internal::CollectGather(
        runtime,
        std::span(particle_stencils.get(), particle_count),
        old_face_velocities,
        new_face_velocities,
        std::span(staged_particle_results.get(), particle_count),
        staged_measurement
    );
  }
  gather_status.Update(cleanup_status);
  if (!gather_status.ok()) {
    return gather_status;
  }

  for (std::size_t particle_index = 0; particle_index < particle_count; ++particle_index) {
    const auto& staged_particle_result = staged_particle_results[particle_index];
    gather_outputs.pic_velocities[particle_index] = staged_particle_result.pic_velocity;
    gather_outputs.grid_velocity_increments[particle_index] = staged_particle_result.grid_velocity_increment;
    gather_outputs.velocity_fit_slopes[particle_index] = staged_particle_result.velocity_fit_slopes;
    gather_outputs.velocity_gradients[particle_index] = staged_particle_result.velocity_gradient;
  }
  kernel_measurement = staged_measurement;
  return absl::OkStatus();
}

}  // namespace fluidity
