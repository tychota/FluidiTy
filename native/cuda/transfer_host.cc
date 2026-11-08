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

absl::Status PrepareTransferStencil(const Grid2D& grid, Vec2 position, TransferStencil& stencil) {
  if (!IsFinite(position)) {
    return absl::InvalidArgumentError("particle position must be finite");
  }
  const Vec2 relative_position = position - grid.origin;
  if (!IsFinite(relative_position)) {
    return absl::OutOfRangeError("position relative to grid origin is nonfinite");
  }

  for (std::size_t component = 0; component < kTransferComponents; ++component) {
    const bool is_x_normal = component == 0;
    const Vec2 lattice_offset = is_x_normal ? Vec2{0.0, 0.5} : Vec2{0.5, 0.0};
    const auto basis = QuadraticStencil2D(relative_position, grid.h, lattice_offset);
    if (!basis.ok()) {
      return basis.status();
    }

    const std::int64_t column_count = static_cast<std::int64_t>(grid.nx) + (is_x_normal ? 1 : 0);
    const std::int64_t row_count = static_cast<std::int64_t>(grid.ny) + (is_x_normal ? 0 : 1);
    const std::size_t orientation_offset = is_x_normal ? 0 : (static_cast<std::size_t>(grid.nx) + 1) * grid.ny;
    for (std::size_t slot = 0; slot < kTransferSlots; ++slot) {
      const Node2D node = basis->nodes[slot];
      // Complete support includes zero-weight nodes at exact half-cell ties.
      const bool has_face = node.x >= 0 && node.y >= 0 && node.x < column_count && node.y < row_count;
      if (!has_face) {
        return absl::InvalidArgumentError("particle does not have complete MAC support");
      }
      const auto face_index = orientation_offset + static_cast<std::size_t>(node.y * column_count + node.x);
      const auto geometry = FaceGeometry(grid, face_index);
      if (!geometry.ok()) {
        return geometry.status();
      }
      const Vec2 offset = geometry->position - position;
      if (!IsFinite(offset)) {
        return absl::OutOfRangeError("face position relative to particle is nonfinite");
      }
      stencil.components[component][slot] = {
          .face_index = face_index,
          .weight = basis->weights[slot],
          .gradient = basis->gradients[slot],
          .offset = offset,
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
    const Grid2D& grid,
    ParticleView2D particles,
    std::span<const double> reference_volumes,
    std::span<const Mat2> stresses,
    std::span<DepositParticle> prepared
) {
  for (std::size_t index = 0; index < prepared.size(); ++index) {
    const double mass = particles.masses[index];
    const double volume = reference_volumes[index];
    const bool valid_mass_and_volume = std::isfinite(mass) && mass >= 0.0 && std::isfinite(volume) && volume >= 0.0;
    if (!valid_mass_and_volume) {
      return absl::InvalidArgumentError("particle mass and reference volume must be finite and nonnegative");
    }
    const Vec2 velocity = particles.velocities[index];
    const Mat2 affine = particles.affine[index];
    const Mat2 stress = stresses[index];
    if (!IsFinite(velocity) || !IsFinite(affine) || !IsFinite(stress)) {
      return absl::InvalidArgumentError("particle velocity, affine fit and stress must be finite");
    }
    auto& particle = prepared[index];
    if (auto status = PrepareTransferStencil(grid, particles.positions[index], particle.stencil); !status.ok()) {
      return status;
    }
    particle.mass = mass;
    particle.velocity = velocity;
    particle.affine = affine;
    particle.reference_volume = volume;
    particle.stress = stress;
  }
  return absl::OkStatus();
}

absl::Status CollectDeposit(
    OperatorRuntime& runtime,
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
  DepositKernelState* device_state = nullptr;
  if (auto status = runtime.AllocateZeroed(1, device_state); !status.ok()) {
    return status;
  }

  if (auto status = runtime.BeginKernel(); !status.ok()) {
    return status;
  }
  const int launch_code = LaunchDeposit(
      device_particles,
      particles.size(),
      {
          .mass = scales.mass,
          .pure_momentum = scales.momentum,
          .stress_force = scales.force,
      },
      device_face_accumulators,
      face_accumulators.size(),
      device_state,
      runtime.stream()
  );
  if (auto status = CheckCuda(static_cast<cudaError_t>(launch_code), "LaunchDeposit"); !status.ok()) {
    return status;
  }
  if (auto status = runtime.FinishKernel(); !status.ok()) {
    return status;
  }

  DepositKernelState state;
  if (auto status = runtime.Download(device_state, std::span(&state, 1)); !status.ok()) {
    return status;
  }
  const auto measurement = runtime.Measurement(state.compiled_architecture);
  if (!measurement.ok()) {
    return measurement.status();
  }
  if (state.error == static_cast<int>(DepositError::kOutOfRange)) {
    return absl::OutOfRangeError("CUDA deposit contribution is nonfinite or exceeds the scaled magnitude bound");
  }
  if (state.error != static_cast<int>(DepositError::kNone)) {
    return absl::InternalError("CUDA deposit encountered an invalid prepared face index");
  }
  if (auto status = runtime.Download(device_face_accumulators, face_accumulators); !status.ok()) {
    return status;
  }
  staged_measurement = *measurement;
  return absl::OkStatus();
}

}  // namespace
}  // namespace cuda_internal

absl::Status CudaP2G(
    const Grid2D& grid,
    ParticleView2D particles,
    std::span<const double> reference_volumes,
    std::span<const Mat2> stresses,
    CudaDepositScales scales,
    CudaDepositViews outputs,
    CudaOperatorMeasurement& measurement
) {
  const std::size_t particle_count = particles.positions.size();
  if (particle_count > cuda_internal::kMaxTransferParticles) {
    return absl::ResourceExhaustedError("CUDA deposit particle capacity exceeded");
  }
  const auto face_count = FaceCount(grid);
  if (!face_count.ok()) {
    return face_count.status();
  }
  if (*face_count > cuda_internal::kMaxTransferFaces) {
    return absl::ResourceExhaustedError("CUDA deposit face capacity exceeded");
  }

  const auto particle_extents = CheckExtents(
      particle_count,
      {particles.masses.size(),
       particles.velocities.size(),
       particles.affine.size(),
       reference_volumes.size(),
       stresses.size()}
  );
  if (!particle_extents.ok()) {
    return particle_extents;
  }
  const auto face_extents =
      CheckExtents(*face_count, {outputs.mass.size(), outputs.momentum.size(), outputs.force.size()});
  if (!face_extents.ok()) {
    return face_extents;
  }
  const auto disjoint = CheckWritableViews(
      {
          std::as_bytes(outputs.mass),
          std::as_bytes(outputs.momentum),
          std::as_bytes(outputs.force),
          std::as_bytes(std::span(&measurement, 1)),
      },
      {
          std::as_bytes(std::span(&grid, 1)),
          std::as_bytes(particles.positions),
          std::as_bytes(particles.masses),
          std::as_bytes(particles.velocities),
          std::as_bytes(particles.affine),
          std::as_bytes(reference_volumes),
          std::as_bytes(stresses),
      }
  );
  if (!disjoint.ok()) {
    return disjoint;
  }
  if (!cuda_internal::IsBinaryScale(scales.mass) || !cuda_internal::IsBinaryScale(scales.momentum)
      || !cuda_internal::IsBinaryScale(scales.force)) {
    return absl::InvalidArgumentError("CUDA deposit scales must be positive normal powers of two");
  }

  // Keep bounded staging off the host stack and alive until pending CUDA work
  // has completed, including cleanup on every early failure.
  std::unique_ptr<cuda_internal::DepositParticle[]> prepared;
  if (particle_count != 0) {
    prepared.reset(new (std::nothrow) cuda_internal::DepositParticle[particle_count]);
    if (prepared == nullptr) {
      return absl::ResourceExhaustedError("could not allocate CUDA deposit particle staging");
    }
  }
  const auto particle_staging = std::span(prepared.get(), particle_count);
  const auto preparation =
      cuda_internal::PrepareDeposit(grid, particles, reference_volumes, stresses, particle_staging);
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
    cuda_internal::OperatorRuntime runtime(cleanup_status);
    status = cuda_internal::CollectDeposit(
        runtime,
        particle_staging,
        scales,
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
  for (std::size_t face = 0; face < *face_count; ++face) {
    outputs.mass[face] = std::bit_cast<std::int64_t>(face_accumulators[face].mass_sum);
    outputs.momentum[face] = std::bit_cast<std::int64_t>(face_accumulators[face].pure_momentum_sum);
    outputs.force[face] = std::bit_cast<std::int64_t>(face_accumulators[face].stress_force_sum);
  }
  measurement = staged_measurement;
  return absl::OkStatus();
}

}  // namespace fluidity
