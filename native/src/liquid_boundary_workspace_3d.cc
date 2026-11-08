#include "fluidity/host/liquid_boundary_workspace_3d.h"

#include <algorithm>
#include <limits>

#include "fluidity/base/liquid_wall_contact_math_3d.h"
#include "liquid_boundary_backend.h"
namespace fluidity {
namespace {
template <typename T>
absl::Status Reserve(LiquidRecords3D<T>& records, std::size_t capacity) {
  auto storage = LiquidRecords3D<T>::Create(capacity);
  if (!storage.ok()) {
    return storage.status();
  }
  records = std::move(*storage);
  return absl::OkStatus();
}
}  // namespace
absl::StatusOr<LiquidBoundaryWorkspaceBytes3D> QueryLiquidBoundaryWorkspaceBytes3D(LiquidStateCapacities3D capacities) {
  LiquidBoundaryWorkspaceBytes3D bytes;
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  const auto add = [&](std::size_t count, std::size_t stride, std::size_t& region) {
    if (count > (limit - bytes.total_bytes) / stride) {
      return false;
    }
    region += count * stride;
    bytes.total_bytes += count * stride;
    return true;
  };
  // Two geometry/link/deposit/normalization/gather banks; four full histories.
  if (!add(
          capacities.cells,
          2
              * (27 * sizeof(LiquidCenterDistance3D) + sizeof(FrozenLiquidCell3D) + 6 * sizeof(FrozenLiquidFace3D)
                 + sizeof(bool)),
          bytes.geometry_bytes
      )
      || !add(capacities.cells, 2 * sizeof(MacPressureCell3D), bytes.pressure_link_bytes)
      || !add(capacities.faces, 2 * sizeof(MacPressureFace3D), bytes.pressure_link_bytes)
      || !add(capacities.faces, 2 * sizeof(MacFaceDepositQuanta3D), bytes.deposit_bytes)
      || !add(capacities.particles, 2 * sizeof(LiquidCompatibleStencils3D), bytes.deposit_bytes)
      || !add(capacities.faces, 2 * (2 * sizeof(double) + sizeof(LiquidFaceValidity3D)), bytes.normalization_bytes)
      || !add(
          capacities.faces,
          4
              * (sizeof(double) + sizeof(LiquidFaceValidity3D) + sizeof(LiquidBoundaryFaceTag3D)
                 + sizeof(LiquidPrescribedWallSample3D)),
          bytes.history_bytes
      )
      || !add(capacities.particles, 2 * sizeof(LiquidBoundaryParticleGather3D), bytes.gather_bytes)
      || !add(
          capacities.faces,
          sizeof(LiquidBoundaryWorkspace3D::OwnedFaceAnchor) + 2 * sizeof(double),
          bytes.gather_bytes
      )) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  return bytes;
}
absl::StatusOr<LiquidBoundaryWorkspace3D> LiquidBoundaryWorkspace3D::Create(LiquidStateCapacities3D capacities) {
  const auto bytes = QueryLiquidBoundaryWorkspaceBytes3D(capacities);
  if (!bytes.ok()) {
    return bytes.status();
  }
  LiquidBoundaryWorkspace3D result;
  result.capacities_ = capacities;
  result.bytes_ = *bytes;
  for (auto& bank : result.geometry_) {
    if (const auto status = Reserve(bank.centers, 27 * capacities.cells); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.cells, capacities.cells); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.faces, 6 * capacities.cells); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.needs_curvature, capacities.cells); !status.ok()) {
      return status;
    }
  }
  for (auto& bank : result.links_) {
    if (const auto status = Reserve(bank.cells, capacities.cells); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.faces, capacities.faces); !status.ok()) {
      return status;
    }
  }
  for (auto& bank : result.deposits_) {
    if (const auto status = Reserve(bank.quanta, capacities.faces); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.stencils, capacities.particles); !status.ok()) {
      return status;
    }
  }
  for (auto& bank : result.normalized_) {
    if (const auto status = Reserve(bank.masses, capacities.faces); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.velocities, capacities.faces); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.validity, capacities.faces); !status.ok()) {
      return status;
    }
  }
  for (auto& time_level : result.histories_) {
    for (auto& bank : time_level) {
      if (const auto status = Reserve(bank.velocities, capacities.faces); !status.ok()) {
        return status;
      }
      if (const auto status = Reserve(bank.validity, capacities.faces); !status.ok()) {
        return status;
      }
      if (const auto status = Reserve(bank.tags, capacities.faces); !status.ok()) {
        return status;
      }
      if (const auto status = Reserve(bank.walls, capacities.faces); !status.ok()) {
        return status;
      }
    }
  }
  for (auto& bank : result.gathered_) {
    if (const auto status = Reserve(bank, capacities.particles); !status.ok()) {
      return status;
    }
  }
  if (const auto status = Reserve(result.anchors_, capacities.faces); !status.ok()) {
    return status;
  }
  if (const auto status = Reserve(result.filled_old_, capacities.faces); !status.ok()) {
    return status;
  }
  if (const auto status = Reserve(result.filled_new_, capacities.faces); !status.ok()) {
    return status;
  }
  return result;
}
std::size_t LiquidBoundaryWorkspace3D::used_bytes() const {
  std::size_t bytes = 0;
  for (const auto& bank : geometry_) {
    bytes += bank.centers.size() * sizeof(LiquidCenterDistance3D);
    bytes += bank.cells.size() * sizeof(FrozenLiquidCell3D);
    bytes += bank.faces.size() * sizeof(FrozenLiquidFace3D);
    bytes += bank.needs_curvature.size() * sizeof(bool);
  }
  for (const auto& bank : links_) {
    bytes += bank.cells.size() * sizeof(MacPressureCell3D);
    bytes += bank.faces.size() * sizeof(MacPressureFace3D);
  }
  for (const auto& bank : deposits_) {
    bytes += bank.quanta.size() * sizeof(MacFaceDepositQuanta3D);
    bytes += bank.stencils.size() * sizeof(LiquidCompatibleStencils3D);
  }
  for (const auto& bank : normalized_) {
    bytes += bank.masses.size() * sizeof(double);
    bytes += bank.velocities.size() * sizeof(double);
    bytes += bank.validity.size() * sizeof(LiquidFaceValidity3D);
  }
  for (const auto& level : histories_) {
    for (const auto& bank : level) {
      bytes += bank.velocities.size() * sizeof(double);
      bytes += bank.validity.size() * sizeof(LiquidFaceValidity3D);
      bytes += bank.tags.size() * sizeof(LiquidBoundaryFaceTag3D);
      bytes += bank.walls.size() * sizeof(LiquidPrescribedWallSample3D);
    }
  }
  for (const auto& bank : gathered_) {
    bytes += bank.size() * sizeof(LiquidBoundaryParticleGather3D);
  }
  return bytes + anchors_.size() * sizeof(OwnedFaceAnchor) + (filled_old_.size() + filled_new_.size()) * sizeof(double);
}
void LiquidBoundaryWorkspace3D::RecordHighWater() {
  high_water_bytes_ = std::max(high_water_bytes_, used_bytes());
}
std::size_t LiquidBoundaryWorkspace3D::high_water_bytes() const {
  return high_water_bytes_;
}
LiquidBoundaryWorkspaceBytes3D LiquidBoundaryWorkspace3D::bytes() const {
  return bytes_;
}
absl::StatusOr<FrozenLiquidBoundaryGeometryView3D>
FreezeLiquidBoundaryGeometry3D(const LiquidBoundaryGeometryViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
  return workspace.Freeze(inputs);
}
absl::StatusOr<LiquidPressureProblemView3D> BuildLiquidPressureLinks3D(
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    const LiquidPressureFields3D& fields,
    LiquidBoundaryWorkspace3D& workspace
) {
  return workspace.BuildLinks(geometry, fields);
}
absl::StatusOr<LiquidBoundaryFaceDepositsView3D>
DepositCompatibleMacQuanta3D(const LiquidBoundaryDepositViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
  return workspace.Deposit(inputs);
}
absl::StatusOr<NormalizedLiquidMacFacesView3D> NormalizeLiquidMacFaces3D(
    std::span<const MacFaceDepositQuanta3D> quanta,
    MacDepositQuantization3D quantization,
    double minimum_face_mass,
    LiquidBoundaryWorkspace3D& workspace
) {
  return workspace.Normalize(quanta, quantization, minimum_face_mass);
}
absl::StatusOr<LiquidBoundaryGatherView3D>
GatherLiquidBoundaryParticles3D(const LiquidBoundaryGatherViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
  return workspace.Gather(inputs);
}
absl::StatusOr<LiquidWallContactResult3D> ResolveLiquidWallContactBounded3D(const LiquidWallContactInputs3D& inputs) {
  const auto result = boundary_internal::ResolveContact(boundary_internal::PackContact(inputs));
  if (!result.ok()) {
    return boundary_internal::ArithmeticStatus(result.status(), true);
  }
  return boundary_internal::UnpackContact(*result);
}
}  // namespace fluidity
