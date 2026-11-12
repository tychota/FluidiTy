#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "fluidity/host/gas_substep_3d.h"
#include "fluidity/host/transfers_3d.h"

namespace fluidity {
namespace {

constexpr std::array kCoordinates{&Vec3::x, &Vec3::y, &Vec3::z};
constexpr std::array kCellIndices{&CellIndex3D::x, &CellIndex3D::y, &CellIndex3D::z};
constexpr std::array kSlopeRows{&Mat3::x_row, &Mat3::y_row, &Mat3::z_row};

auto CellKey(CellIndex3D cell) {
  return std::tuple(cell.z, cell.y, cell.x);
}

bool ContainsCell(std::span<const GasCell3D> cells, CellIndex3D cell) {
  const auto key = CellKey(cell);
  const auto found = std::ranges::lower_bound(cells, key, {}, [](const GasCell3D& candidate) {
    return CellKey(candidate.cell_index);
  });
  return found != cells.end() && CellKey(found->cell_index) == key;
}

absl::StatusOr<std::int32_t> AxisCellIndex(UniformMacGeometry3D geometry, std::size_t axis, double position) {
  const double index = std::floor((position - geometry.world_origin.*kCoordinates[axis]) / geometry.cell_width);
  if (!std::isfinite(index) || index < std::numeric_limits<std::int32_t>::min()
      || index > std::numeric_limits<std::int32_t>::max()) {
    return absl::OutOfRangeError("gas marker left the signed lattice");
  }
  return static_cast<std::int32_t>(index);
}

// Resolves marker motion one axis at a time, matching the discrete MAC walls.
// True means the updated marker remains in the domain.
absl::StatusOr<bool> AdvectMarker(
    const SparseMacGrid3D& grid,
    double step_duration,
    std::span<const GasCell3D> cells,
    std::span<const double> face_apertures,
    GasApicMarker3D& marker
) {
  const auto geometry = grid.geometry();
  CellIndex3D cell{};
  for (std::size_t axis = 0; axis < kCoordinates.size(); ++axis) {
    const auto index = AxisCellIndex(geometry, axis, marker.world_position.*kCoordinates[axis]);
    if (!index.ok()) {
      return index.status();
    }
    cell.*kCellIndices[axis] = *index;
  }
  if (!ContainsCell(cells, cell)) {
    return absl::InvalidArgumentError("gas marker starts outside the fixed domain");
  }
  for (std::size_t axis = 0; axis < kCoordinates.size(); ++axis) {
    const double displacement = step_duration * (marker.velocity.*kCoordinates[axis]);
    if (!std::isfinite(displacement) || std::abs(displacement) > 0.5 * geometry.cell_width) {
      return absl::FailedPreconditionError("gas marker exceeds half-cell motion bound");
    }
    const double next_position = marker.world_position.*kCoordinates[axis] + displacement;
    const auto next_index = AxisCellIndex(geometry, axis, next_position);
    if (!next_index.ok()) {
      return next_index.status();
    }
    const auto old_index = cell.*kCellIndices[axis];
    if (*next_index != old_index) {
      auto face_cell = cell;
      face_cell.*kCellIndices[axis] = std::max(old_index, *next_index);
      const auto face_slot = grid.FaceSlot({
          static_cast<FaceNormalAxis>(axis),
          {face_cell.x, face_cell.y, face_cell.z},
      });
      if (!face_slot.ok()) {
        return face_slot.status();
      }
      if (face_apertures[*face_slot] == 0.0) {
        marker.velocity.*kCoordinates[axis] = 0.0;
        marker.velocity_slopes.*kSlopeRows[axis] = {};
        continue;
      }
      cell.*kCellIndices[axis] = *next_index;
      if (!ContainsCell(cells, cell)) {
        return false;
      }
    }
    marker.world_position.*kCoordinates[axis] = next_position;
  }
  return true;
}

}  // namespace

absl::StatusOr<GasApicSubstep3D> AdvanceGasApicSubstep3D(
    const SparseMacGrid3D& grid,
    double step_duration,
    std::span<const GasCell3D> cells,
    std::span<const MacPressureFace3D> faces,
    std::span<const GasCellSource3D> sources,
    GasScalars ambient,
    double ambient_pressure,
    Vec3 gravity,
    std::span<const GasApicMarker3D> markers,
    std::span<const double> old_face_velocities,
    std::span<const double> face_accelerations,
    absl::FunctionRef<absl::StatusOr<MacPressureProjection3D>(const MacPressureInputs3D&, MacPressureSolverOptions3D)>
        project
) {
  if (old_face_velocities.size() != grid.face_count()
      || (!face_accelerations.empty() && face_accelerations.size() != faces.size())) {
    return absl::InvalidArgumentError("gas APIC face field extent does not match its layout");
  }
  if (!std::ranges::all_of(old_face_velocities, [](double velocity) {
        return std::isfinite(velocity);
      })) {
    return absl::InvalidArgumentError("gas APIC prior face field must be finite");
  }
  std::vector<MacTransferParticle3D> particles;
  particles.reserve(markers.size());
  for (const auto& marker : markers) {
    if (!std::isfinite(marker.weight) || marker.weight <= 0.0) {
      return absl::InvalidArgumentError("gas APIC marker weight must be positive and finite");
    }
    particles.push_back({
        .sorted_world_position = marker.world_position,
        .trial_world_position = marker.world_position,
        .particle_mass = marker.weight,
        .particle_velocity = marker.velocity,
        .particle_velocity_slopes = marker.velocity_slopes,
    });
  }
  std::vector<MacFaceDeposit3D> deposits(grid.face_count());
  if (const auto status = DepositMac3D(grid, particles, deposits); !status.ok()) {
    return status;
  }
  std::vector<MacPressureFace3D> predicted(faces.begin(), faces.end());
  std::vector<std::size_t> pressure_face_slots;
  pressure_face_slots.reserve(faces.size());
  std::vector<double> face_apertures(grid.face_count());
  for (std::size_t index = 0; index < faces.size(); ++index) {
    const auto slot = grid.FaceSlot(faces[index].face_index);
    if (!slot.ok()) {
      return slot.status();
    }
    pressure_face_slots.push_back(*slot);
    face_apertures[*slot] = faces[index].open_area;
    const auto& deposit = deposits[*slot];
    const double velocity = deposit.deposited_mass > 0.0 ? deposit.deposited_pure_momentum / deposit.deposited_mass
                                                         : old_face_velocities[*slot];
    const double acceleration = face_accelerations.empty() ? 0.0 : face_accelerations[index];
    predicted[index].predicted_normal_velocity =
        faces[index].open_area > 0.0 ? velocity + step_duration * acceleration : 0.0;
  }
  auto gas =
      AdvanceGasSubstep3D(grid, step_duration, cells, predicted, sources, ambient, ambient_pressure, gravity, project);
  if (!gas.ok()) {
    return gas.status();
  }
  GasApicSubstep3D update{
      .gas = std::move(*gas),
      .markers = {},
      .face_velocities = {old_face_velocities.begin(), old_face_velocities.end()},
  };
  for (std::size_t index = 0; index < faces.size(); ++index) {
    const auto slot = pressure_face_slots[index];
    update.face_velocities[slot] =
        face_apertures[slot] > 0.0 ? update.gas.projection.corrected_face_velocities[index] : 0.0;
  }
  std::vector<MacParticleGatherResult3D> gathered(markers.size());
  if (const auto status = GatherMac3D(grid, particles, old_face_velocities, update.face_velocities, gathered);
      !status.ok()) {
    return status;
  }
  update.markers.reserve(markers.size());
  for (std::size_t index = 0; index < markers.size(); ++index) {
    auto marker = markers[index];
    marker.velocity = gathered[index].new_pic_velocity;
    marker.velocity_slopes = gathered[index].fitted_velocity_slopes;
    const auto inside = AdvectMarker(grid, step_duration, cells, face_apertures, marker);
    if (!inside.ok()) {
      return inside.status();
    }
    if (*inside) {
      update.markers.push_back(marker);
    }
  }
  return update;
}

}  // namespace fluidity
