#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <span>
#include <vector>

#include "../src/liquid_boundary_backend.h"
#include "../src/liquid_boundary_transfer_3d_internal.h"
#include "fluidity/host/cuda_liquid_boundary_3d.h"
#include "liquid_boundary_transfer_3d_bridge.h"
#include "liquid_boundary_transfer_3d_internal.h"
#include "operator_runtime.h"
namespace fluidity {
namespace {
constexpr std::array kAxes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
constexpr std::array kRows{&Mat3::x_row, &Mat3::y_row, &Mat3::z_row};
bool IsFinite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double Dot(Vec3 lhs, Vec3 rhs) {
  return (lhs.x * rhs.x + lhs.y * rhs.y) + lhs.z * rhs.z;
}

absl::Status ValidateCapacity(std::size_t face_count, std::size_t particle_count) {
  if (face_count > kLiquidBoundaryFaceCapacity3D || particle_count > kLiquidBoundaryParticleCapacity3D) {
    return absl::ResourceExhaustedError("liquid boundary transfer exceeds bounded face/particle capacity");
  }
  return absl::OkStatus();
}

absl::Status ValidateQuantization(MacDepositQuantization3D quantization, std::size_t particle_count) {
  for (double scale : {quantization.mass_per_quantum, quantization.pure_momentum_per_quantum}) {
    int exponent = 0;
    if (!std::isnormal(scale) || std::frexp(scale, &exponent) != 0.5) {
      return absl::InvalidArgumentError("quantization scales must be positive normal powers of two");
    }
  }
  if (quantization.declared_particle_capacity > kLiquidBoundaryParticleCapacity3D
      || particle_count > quantization.declared_particle_capacity) {
    return absl::ResourceExhaustedError("particle count or declared capacity exceeds boundary transfer capacity");
  }
  return absl::OkStatus();
}

bool IsUsable(LiquidFaceValidity3D validity) {
  return validity == LiquidFaceValidity3D::kValid || validity == LiquidFaceValidity3D::kPredicted;
}

absl::Status
ValidateHistory(const LiquidBoundaryVelocityHistory3D& history, std::size_t face_count, double stage_time) {
  if (history.velocity_sample_time != stage_time || history.wall_sample_time != stage_time) {
    return absl::InvalidArgumentError("velocity and wall history must match their requested stage time");
  }
  if (history.face_velocities.size() != face_count || history.validity.size() != face_count
      || history.face_tags.size() != face_count || history.wall_samples.size() != face_count) {
    return absl::InvalidArgumentError("each history span must cover every owned face");
  }
  for (std::size_t slot = 0; slot < face_count; ++slot) {
    const auto validity = history.validity[slot];
    if (validity != LiquidFaceValidity3D::kMissing && validity != LiquidFaceValidity3D::kLowMass
        && !IsUsable(validity)) {
      return absl::InvalidArgumentError("unknown face validity");
    }
    if (IsUsable(validity) && !std::isfinite(history.face_velocities[slot])) {
      return absl::InvalidArgumentError("valid and predicted face velocities must be finite");
    }
    const auto& wall = history.wall_samples[slot];
    if (wall.available) {
      if (!IsFinite(wall.wall_velocity) || !IsFinite(wall.unit_normal)) {
        return absl::InvalidArgumentError("prescribed wall velocity and normal must be finite");
      }
      if (std::abs(Dot(wall.unit_normal, wall.unit_normal) - 1.0) > 1e-12) {
        return absl::InvalidArgumentError("prescribed wall normal must have unit length");
      }
    }
  }
  return absl::OkStatus();
}

struct StagedTransfer {
  cuda_internal::BoundaryTransferQuery3D query{};
  std::vector<cuda_internal::BoundaryTransferFace3D> faces;
  std::vector<cuda_internal::BoundaryTransferParticle3D> particles;
  std::vector<cuda_internal::BoundaryTransferFaceResult3D> face_results;
  std::vector<cuda_internal::BoundaryTransferParticleResult3D> particle_results;
  cuda_internal::BoundaryTransferReport3D report{};
};
// Every upload and download owner belongs to the caller, outliving runtime cleanup.
absl::StatusOr<CudaOperatorMeasurement> Execute(StagedTransfer& staged) {
  staged.query.face_count = staged.faces.size();
  staged.query.particle_count = staged.particles.size();
  staged.face_results.resize(staged.faces.size());
  staged.particle_results.resize(staged.particles.size());
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
      cuda_internal::BoundaryTransferQuery3D* query = nullptr;
      cuda_internal::BoundaryTransferFace3D* faces = nullptr;
      cuda_internal::BoundaryTransferParticle3D* particles = nullptr;
      cuda_internal::BoundaryTransferFaceResult3D* face_results = nullptr;
      cuda_internal::BoundaryTransferParticleResult3D* particle_results = nullptr;
      cuda_internal::BoundaryTransferReport3D* report = nullptr;
      current = runtime.Upload(std::span<const cuda_internal::BoundaryTransferQuery3D>(&staged.query, 1), query);
      if (!current.ok()) {
        return current;
      }
      current = runtime.Upload(std::span<const cuda_internal::BoundaryTransferFace3D>(staged.faces), faces);
      if (!current.ok()) {
        return current;
      }
      current = runtime.Upload(std::span<const cuda_internal::BoundaryTransferParticle3D>(staged.particles), particles);
      if (!current.ok()) {
        return current;
      }
      current = runtime.Allocate(staged.faces.size(), face_results);
      if (!current.ok()) {
        return current;
      }
      current = runtime.Allocate(staged.particles.size(), particle_results);
      if (!current.ok()) {
        return current;
      }
      current = runtime.Allocate(1, report);
      if (!current.ok()) {
        return current;
      }
      current = runtime.RecordKernelStart();
      if (!current.ok()) {
        return current;
      }
      current = cuda_internal::CheckCuda(
          static_cast<cudaError_t>(cuda_internal::LaunchLiquidBoundaryTransfer3D(
              query,
              faces,
              particles,
              face_results,
              particle_results,
              report,
              runtime.stream()
          )),
          "Launch boundary transfer"
      );
      if (!current.ok()) {
        return current;
      }
      current = runtime.SynchronizeAndMeasureKernel();
      if (!current.ok()) {
        return current;
      }
      current = runtime.Download(report, std::span(&staged.report, 1));
      if (!current.ok()) {
        return current;
      }
      current = runtime.Download(face_results, std::span(staged.face_results));
      if (!current.ok()) {
        return current;
      }
      current = runtime.Download(particle_results, std::span(staged.particle_results));
      if (!current.ok()) {
        return current;
      }
      const auto measured = runtime.ValidateKernelMeasurement(staged.report.compiled_architecture);
      if (!measured.ok()) {
        return measured.status();
      }
      measurement = *measured;
      return boundary_internal::ArithmeticStatus(staged.report.error);
    };
    status = execute();
  }
  status.Update(cleanup);
  if (!status.ok()) {
    return status;
  }
  return measurement;
}
// Resolve every component/node before compatibility filtering, preserving slot order.
absl::Status StageParticles(
    const SparseMacGrid3D& grid,
    std::span<const LiquidBoundaryParticle3D> particles,
    StagedTransfer& staged,
    bool deposit
) {
  staged.particles.resize(particles.size());
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const auto& particle = particles[index];
    if (!IsFinite(particle.transfer.particle_velocity)) {
      return absl::InvalidArgumentError("particle velocity must be finite");
    }
    if (deposit) {
      if (!std::isfinite(particle.transfer.particle_mass) || particle.transfer.particle_mass < 0) {
        return absl::InvalidArgumentError("particle mass must be finite and nonnegative");
      }
      if (particle.affine_enabled) {
        for (auto row : kRows) {
          if (!IsFinite(particle.transfer.particle_velocity_slopes.*row)) {
            return absl::InvalidArgumentError("enabled slopes must be finite");
          }
        }
      }
    }
    staged.particles[index].particle = particle;
    for (std::size_t component = 0; component < 3; ++component) {
      const auto stencil = BuildOwnedMacFaceStencil3D(
          grid,
          kAxes[component],
          particle.transfer.sorted_world_position,
          particle.transfer.trial_world_position
      );
      if (!stencil.ok()) {
        return stencil.status();
      }
      std::copy(stencil->begin(), stencil->end(), staged.particles[index].stencils[component]);
    }
  }
  return absl::OkStatus();
}
}  // namespace
absl::StatusOr<CudaLiquidBoundaryDeposits3D> CudaDepositCompatibleMacQuanta3D(
    const LiquidBoundaryDepositInputs3D& inputs
) {
  auto status = ValidateCapacity(inputs.mac_grid.face_count(), inputs.particles.size());
  if (!status.ok()) {
    return status;
  }
  status = ValidateQuantization(inputs.quantization, inputs.particles.size());
  if (!status.ok()) {
    return status;
  }
  if (inputs.face_tags.size() != inputs.mac_grid.face_count()) {
    return absl::InvalidArgumentError("deposit tags must cover all faces");
  }
  StagedTransfer staged;
  staged.query.quantization = inputs.quantization;
  const auto capacity = std::max(std::size_t{1}, inputs.quantization.declared_particle_capacity);
  staged.query.scaled_contribution_magnitude_limit =
      std::ldexp(1.0, 62 - static_cast<int>(std::bit_width(capacity - 1)));
  staged.faces.resize(inputs.mac_grid.face_count());
  for (std::size_t slot = 0; slot < staged.faces.size(); ++slot) {
    staged.faces[slot].deposit_tag = inputs.face_tags[slot];
  }
  status = StageParticles(inputs.mac_grid, inputs.particles, staged, true);
  if (!status.ok()) {
    return status;
  }
  const auto measurement = Execute(staged);
  if (!measurement.ok()) {
    return measurement.status();
  }
  LiquidBoundaryFaceDeposits3D result;
  for (const auto& face : staged.face_results) {
    result.face_quanta.push_back(face.face_quanta);
  }
  result.compatible_stencils.resize(inputs.particles.size());
  for (std::size_t particle = 0; particle < inputs.particles.size(); ++particle) {
    for (std::size_t component = 0; component < 3; ++component) {
      std::copy_n(
          staged.particle_results[particle].stencils[component],
          27,
          result.compatible_stencils[particle][component].begin()
      );
    }
  }
  return CudaLiquidBoundaryDeposits3D{std::move(result), *measurement};
}
absl::StatusOr<CudaNormalizedLiquidMacFaces3D> CudaNormalizeLiquidMacFaces3D(
    std::span<const MacFaceDepositQuanta3D> quanta,
    MacDepositQuantization3D quantization,
    double minimum_face_mass
) {
  auto status = ValidateCapacity(quanta.size(), 0);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQuantization(quantization, 0);
  if (!status.ok()) {
    return status;
  }
  if (!std::isfinite(minimum_face_mass) || minimum_face_mass < 0) {
    return absl::InvalidArgumentError("minimum mass must be finite and nonnegative");
  }
  StagedTransfer staged;
  staged.query.operation = cuda_internal::BoundaryTransferOperation3D::kNormalize;
  staged.query.quantization = quantization;
  staged.query.minimum_face_mass = minimum_face_mass;
  staged.faces.resize(quanta.size());
  for (std::size_t slot = 0; slot < quanta.size(); ++slot) {
    staged.faces[slot].face_quanta = quanta[slot];
  }
  const auto measurement = Execute(staged);
  if (!measurement.ok()) {
    return measurement.status();
  }
  NormalizedLiquidMacFaces3D result;
  for (const auto& face : staged.face_results) {
    result.face_masses.push_back(face.decoded_face_mass);
    result.face_velocities.push_back(face.normalized_face_velocity);
    result.validity.push_back(face.validity);
  }
  return CudaNormalizedLiquidMacFaces3D{std::move(result), *measurement};
}
absl::StatusOr<CudaLiquidBoundaryGatherResults3D> CudaGatherLiquidBoundaryParticles3D(
    const LiquidBoundaryGatherInputs3D& inputs
) {
  auto status = ValidateCapacity(inputs.mac_grid.face_count(), inputs.particles.size());
  if (!status.ok()) {
    return status;
  }
  if (!std::isfinite(inputs.old_time) || !std::isfinite(inputs.new_time) || inputs.new_time <= inputs.old_time) {
    return absl::InvalidArgumentError("stage times must be finite and increasing");
  }
  status = ValidateHistory(inputs.old_history, inputs.mac_grid.face_count(), inputs.old_time);
  if (!status.ok()) {
    return status;
  }
  status = ValidateHistory(inputs.new_history, inputs.mac_grid.face_count(), inputs.new_time);
  if (!status.ok()) {
    return status;
  }
  StagedTransfer staged;
  staged.query.operation = cuda_internal::BoundaryTransferOperation3D::kGather;
  staged.query.same_history_samples = internal::SameBoundaryVelocitySamples3D(inputs.old_history, inputs.new_history);
  staged.query.cell_width = inputs.mac_grid.geometry().cell_width;
  staged.faces.resize(inputs.mac_grid.face_count());
  std::size_t slot = 0;
  for (auto block : inputs.mac_grid.active_block_indices()) {
    for (auto axis : kAxes) {
      for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
          for (int x = 0; x < 4; ++x) {
            auto& face = staged.faces[slot++];
            face.axis = axis;
            face.anchor_in_cell_widths = {
                4.0 * block.x + x + (axis == FaceNormalAxis::kX ? 0 : 0.5),
                4.0 * block.y + y + (axis == FaceNormalAxis::kY ? 0 : 0.5),
                4.0 * block.z + z + (axis == FaceNormalAxis::kZ ? 0 : 0.5),
            };
          }
        }
      }
    }
  }
  for (std::size_t face = 0; face < staged.faces.size(); ++face) {
    const auto pack_history = [face](const LiquidBoundaryVelocityHistory3D& history) {
      return cuda_internal::BoundaryTransferHistory3D{
          history.face_velocities[face],
          history.validity[face],
          history.face_tags[face],
          history.wall_samples[face],
      };
    };
    staged.faces[face].old_history = pack_history(inputs.old_history);
    staged.faces[face].new_history = pack_history(inputs.new_history);
  }

  status = StageParticles(inputs.mac_grid, inputs.particles, staged, false);
  if (!status.ok()) {
    return status;
  }
  const auto measurement = Execute(staged);
  if (!measurement.ok()) {
    return measurement.status();
  }
  LiquidBoundaryGatherResults3D result;
  for (const auto& particle : staged.particle_results) {
    result.particles.push_back(particle.gathered);
    result.old_pic_fallback_particle_count += particle.gathered.old_pic_fallback ? 1 : 0;
    result.new_pic_fallback_particle_count += particle.gathered.new_pic_fallback ? 1 : 0;
    result.low_mass_fallback_particle_count += particle.gathered.low_mass_fallback ? 1 : 0;
  }
  return CudaLiquidBoundaryGatherResults3D{std::move(result), *measurement};
}
namespace cuda_internal {
absl::StatusOr<internal::LiquidGhostAffineFit3D> CudaFitLiquidGhostAffine3D(
    std::span<const internal::LiquidGhostFitSample3D> samples
) {
  if (samples.size() > 125) {
    return absl::ResourceExhaustedError("ghost sample capacity exceeded");
  }
  StagedTransfer staged;
  staged.query.operation = cuda_internal::BoundaryTransferOperation3D::kFitGhost;
  staged.query.fit_count = samples.size();
  std::copy(samples.begin(), samples.end(), staged.query.fit_samples);
  const auto measurement = Execute(staged);
  if (!measurement.ok()) {
    return measurement.status();
  }
  return staged.report.fit;
}
}  // namespace cuda_internal
}  // namespace fluidity
