#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "fluidity/host/gas_substep_3d.h"

namespace fluidity {
namespace {

constexpr std::array kLatticeCoordinates{
    &FaceLatticeIndex3D::x,
    &FaceLatticeIndex3D::y,
    &FaceLatticeIndex3D::z,
};

bool IsDomainCell(FaceLatticeIndex3D index, const std::array<std::int32_t, 3>& extent) {
  return index.x >= 0 && index.x < extent[0] && index.y >= 0 && index.y < extent[1] && index.z >= 0
      && index.z < extent[2];
}

bool PositiveFinite(double value) {
  return value > 0.0 && std::isfinite(value);
}

}  // namespace

absl::StatusOr<GasUniformDomain3D> CreateGasUniformDomain3D(const GasUniformDomainOptions3D& options) {
  constexpr std::size_t kMaxPressureCells = 4096;
  std::size_t cell_count = 1;
  for (const auto axis_extent : options.extent) {
    if (axis_extent <= 0) {
      return absl::InvalidArgumentError("gas domain extent must be positive");
    }
    const auto axis_cell_count = static_cast<std::size_t>(axis_extent);
    if (axis_cell_count > kMaxPressureCells / cell_count) {
      return absl::ResourceExhaustedError("gas domain exceeds the bounded pressure cell count");
    }
    cell_count *= axis_cell_count;
  }
  const auto& ambient = options.ambient;
  const std::array velocity{options.initial_velocity.x, options.initial_velocity.y, options.initial_velocity.z};
  if (!PositiveFinite(options.ambient_pressure) || !PositiveFinite(ambient.carrier) || !PositiveFinite(ambient.enthalpy)
      || !std::isfinite(ambient.fuel) || ambient.fuel < 0.0 || !std::isfinite(ambient.soot) || ambient.soot < 0.0
      || !std::isfinite(velocity[0]) || !std::isfinite(velocity[1]) || !std::isfinite(velocity[2])) {
    return absl::InvalidArgumentError("invalid gas ambient state or initial velocity");
  }

  const auto [nx, ny, nz] = options.extent;
  std::vector<BlockIndex3D> blocks;
  // Allocate the quadratic halo for any marker inside the domain, including
  // support at negative lattice coordinates and high-side exterior faces.
  for (std::int32_t z = -1; z <= (nz + 2) / 4; ++z) {
    for (std::int32_t y = -1; y <= (ny + 2) / 4; ++y) {
      for (std::int32_t x = -1; x <= (nx + 2) / 4; ++x) {
        blocks.push_back({x, y, z});
      }
    }
  }
  auto grid = SparseMacGrid3D::Create(options.geometry, blocks, blocks.size());
  if (!grid.ok()) {
    return grid.status();
  }
  const double width = options.geometry.cell_width;
  const double area = width * width;
  const double volume = area * width;
  if (!PositiveFinite(area) || !PositiveFinite(volume)) {
    return absl::OutOfRangeError("gas domain cell measures are not representable");
  }
  GasUniformDomain3D domain{
      .grid = std::move(*grid),
      .cells = {},
      .faces = {},
      .markers = {},
      .face_velocities = {},
  };
  domain.cells.reserve(cell_count);
  domain.markers.reserve(cell_count);
  for (std::int32_t z = 0; z < nz; ++z) {
    for (std::int32_t y = 0; y < ny; ++y) {
      for (std::int32_t x = 0; x < nx; ++x) {
        const CellIndex3D cell{x, y, z};
        const auto center = domain.grid.CellWorldCenter(cell);
        if (!center.ok()) {
          return center.status();
        }
        domain.cells.push_back({
            .cell_index = cell,
            .volume = volume,
            .density = ambient,
            .absolute_pressure = options.ambient_pressure,
        });
        domain.markers.push_back({
            .world_position = *center,
            .weight = 1.0,
            .velocity = options.initial_velocity,
            .velocity_slopes = {},
        });
      }
    }
  }
  domain.face_velocities.reserve(domain.grid.face_count());
  const auto grid_view = domain.grid.view();
  for (std::size_t slot = 0; slot < grid_view.face_count(); ++slot) {
    const auto face = grid_view.FaceIndex(slot);
    if (!face.ok()) {
      return face.status();
    }
    const auto axis = static_cast<std::size_t>(face->axis);
    const auto head = face->lattice_index;
    auto tail = head;
    --(tail.*kLatticeCoordinates[axis]);
    const bool has_tail = IsDomainCell(tail, options.extent);
    const bool has_head = IsDomainCell(head, options.extent);
    double normal_velocity = velocity[axis];
    if (has_tail || has_head) {
      MacPressureFace3D pressure_face{
          .face_index = *face,
          .open_area = area,
          .inverse_density = 1.0 / ambient.carrier,
          .pressure_gradient_distance = width,
          .predicted_normal_velocity = normal_velocity,
          .wall_normal_velocity = 0.0,
          .pressure_link = InternalMacPressureLink3D{.pressure_jump = 0.0},
      };
      if (has_tail != has_head) {
        const std::size_t boundary = 2 * axis + static_cast<std::size_t>(has_tail);
        if (options.open_boundary[boundary]) {
          pressure_face.pressure_gradient_distance = 0.5 * width;
          pressure_face.pressure_link = DirichletMacPressureLink3D{.exterior_pressure = 0.0};
        } else {
          normal_velocity = 0.0;
          pressure_face.open_area = 0.0;
          pressure_face.predicted_normal_velocity = 0.0;
          pressure_face.pressure_link = SealedMacPressureLink3D{};
        }
      }
      domain.faces.push_back(pressure_face);
    }
    domain.face_velocities.push_back(normal_velocity);
  }
  return domain;
}

}  // namespace fluidity
