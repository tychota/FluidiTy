#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

#include "fluidity/base/liquid_boundary_geometry_math_3d.h"
#include "wetted_piston_3d_internal.h"

namespace fluidity {
namespace {

absl::Status Status(absl::StatusCode code) {
  return absl::Status(code, {});
}

template <typename T>
absl::Status Reserve(LiquidRecords3D<T>& records, std::size_t count) {
  auto allocated = LiquidRecords3D<T>::Create(count);
  if (!allocated.ok()) {
    return allocated.status();
  }
  records = std::move(*allocated);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<WettedPistonWorkspace3D> WettedPistonWorkspace3D::Create(LiquidStateCapacities3D capacities) {
  using wetted_piston_internal::Disk;
  using wetted_piston_internal::Interval;
  using wetted_piston_internal::PatchIndex;
  using wetted_piston_internal::PatchMeasure;
  const auto disk_capacity = std::max<std::size_t>(1, capacities.particles);
  constexpr std::size_t kByteLimit = std::numeric_limits<std::ptrdiff_t>::max();
  constexpr std::size_t kDiskBytes = sizeof(Vec3) + sizeof(Disk) + 2 * sizeof(Interval);
  constexpr std::size_t kFaceBytes =
      sizeof(PatchIndex) + sizeof(PatchMeasure) + 2 * (sizeof(WettedPistonPatch3D) + sizeof(std::optional<double>));
  if (disk_capacity > kByteLimit / kDiskBytes
      || capacities.faces > (kByteLimit - disk_capacity * kDiskBytes) / kFaceBytes) {
    return Status(absl::StatusCode::kResourceExhausted);
  }
  WettedPistonWorkspace3D workspace;
  for (auto& publication : workspace.publications_) {
    if (!Reserve(publication.patches, capacities.faces).ok()
        || !Reserve(publication.face_fluxes, capacities.faces).ok()) {
      return Status(absl::StatusCode::kResourceExhausted);
    }
  }
  if (!Reserve(workspace.indices_, capacities.faces).ok() || !Reserve(workspace.measures_, capacities.faces).ok()
      || !Reserve(workspace.centers_, disk_capacity).ok() || !Reserve(workspace.disks_, disk_capacity).ok()
      || !Reserve(workspace.intervals_, 2 * disk_capacity).ok()) {
    return Status(absl::StatusCode::kResourceExhausted);
  }
  workspace.capacities_ = capacities;
  workspace.reserved_bytes_ = disk_capacity * kDiskBytes + capacities.faces * kFaceBytes;
  return workspace;
}

std::size_t WettedPistonWorkspace3D::reserved_bytes() const {
  return reserved_bytes_;
}

namespace wetted_piston_internal {
namespace {
constexpr std::int64_t kFirstLatticeCell = std::numeric_limits<std::int32_t>::min();
constexpr std::int64_t kPastLastLatticeCell = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;

absl::Status ArithmeticStatus(Error error) {
  switch (error) {
    case Error::kNone:
      return absl::OkStatus();
    case Error::kInvalid:
      return Status(absl::StatusCode::kInvalidArgument);
    case Error::kPrecondition:
      return Status(absl::StatusCode::kFailedPrecondition);
    case Error::kRange:
      return Status(absl::StatusCode::kOutOfRange);
  }
  return Status(absl::StatusCode::kInternal);
}

bool FinitePosition(Vec3 position) {
  return std::isfinite(position.x) && std::isfinite(position.y) && std::isfinite(position.z);
}

bool PatchLess(PatchIndex a, PatchIndex b) {
  return std::tie(a.z, a.y) < std::tie(b.z, b.y);
}

auto CellOrder(CellIndex3D cell) {
  return std::tuple{cell.z, cell.y, cell.x};
}

bool HasSelectedCell(std::span<const FrozenLiquidCell3D> cells, CellIndex3D index) {
  return std::ranges::binary_search(cells, CellOrder(index), {}, [](const FrozenLiquidCell3D& cell) {
    return CellOrder(cell.cell_index);
  });
}

// Signed half-open owner with no floor() of a rounded world reconstruction.
absl::StatusOr<std::int32_t> FindPistonOwnerLatticeX(const Field& field) {
  const auto compare = [&](std::int64_t index) {
    return boundary_internal::TankPlaneOffsetSign(
        field.tank.start_low_x,
        {field.geometry.world_origin.x, static_cast<double>(index), field.geometry.cell_width}
    );
  };
  auto low = kFirstLatticeCell;
  auto high = kPastLastLatticeCell;
  const auto low_side = compare(low);
  const auto high_side = compare(high);
  if (!low_side.ok() || !high_side.ok() || *low_side < 0 || *high_side >= 0) {
    return Status(absl::StatusCode::kOutOfRange);
  }
  while (high - low > 1) {
    const auto middle = low + (high - low) / 2;
    const auto side = compare(middle);
    if (!side.ok()) {
      return Status(absl::StatusCode::kOutOfRange);
    }
    if (*side < 0) {
      high = middle;
    } else {
      low = middle;
    }
  }
  return static_cast<std::int32_t>(low);
}

absl::Status AddIndex(LiquidRecords3D<PatchIndex>& indices, PatchIndex index) {
  const auto current = indices.values();
  const auto position = std::ranges::lower_bound(current, index, PatchLess);
  if (position != current.end() && *position == index) {
    return absl::OkStatus();
  }
  if (current.size() == indices.capacity()) {
    return Status(absl::StatusCode::kResourceExhausted);
  }
  const auto ordinal = static_cast<std::size_t>(position - current.begin());
  const auto status = indices.Resize(current.size() + 1);
  if (!status.ok()) {
    return status;
  }
  auto expanded = indices.values();
  std::move_backward(expanded.begin() + ordinal, expanded.end() - 1, expanded.end());
  expanded[ordinal] = index;
  return absl::OkStatus();
}

struct OuterRectangle {
  double low_y;
  double high_y;
  double low_z;
  double high_z;
};

absl::Status AddRectangle(LiquidRecords3D<PatchIndex>& indices, OuterRectangle rectangle) {
  if (!(rectangle.low_y < rectangle.high_y && rectangle.low_z < rectangle.high_z)) {
    return absl::OkStatus();
  }
  if (rectangle.low_y < kFirstLatticeCell || rectangle.low_z < kFirstLatticeCell
      || rectangle.high_y > kPastLastLatticeCell || rectangle.high_z > kPastLastLatticeCell) {
    return Status(absl::StatusCode::kOutOfRange);
  }
  const auto first_y = static_cast<std::int64_t>(std::floor(rectangle.low_y));
  const auto first_z = static_cast<std::int64_t>(std::floor(rectangle.low_z));
  const auto end_y = static_cast<std::int64_t>(std::ceil(rectangle.high_y));
  const auto end_z = static_cast<std::int64_t>(std::ceil(rectangle.high_z));
  const auto rows = static_cast<std::uint64_t>(end_y - first_y);
  const auto columns = static_cast<std::uint64_t>(end_z - first_z);
  if (rows > indices.capacity() || columns > indices.capacity() / rows) {
    return Status(absl::StatusCode::kResourceExhausted);
  }
  for (auto z = first_z; z < end_z; ++z) {
    for (auto y = first_y; y < end_y; ++y) {
      const auto status = AddIndex(indices, {static_cast<std::int32_t>(y), static_cast<std::int32_t>(z)});
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

// A circumscribed square provides a conservative, deduplicated broad phase.
// Plane support is finite because the tank floor and the actual height bound y.
absl::Status AddSupport(const Field& field, std::span<const Disk> disks, LiquidRecords3D<PatchIndex>& indices) {
  const double h = field.geometry.cell_width;
  const auto origin = field.geometry.world_origin;
  const auto floor = LatticeOffset(field.tank.floor_y, origin.y, 0, h);
  const auto low_z = LatticeOffset(field.tank.low_z, origin.z, 0, h);
  const auto high_z = LatticeOffset(field.tank.high_z, origin.z, 0, h);
  if (!Finite(floor) || !Finite(low_z) || !Finite(high_z)) {
    return Status(absl::StatusCode::kOutOfRange);
  }
  if (field.is_plane) {
    if (field.plane_height_y <= field.tank.floor_y) {
      return absl::OkStatus();
    }
    const auto surface = LatticeOffset(field.plane_height_y, origin.y, 0, h);
    if (!Finite(surface)) {
      return Status(absl::StatusCode::kOutOfRange);
    }
    return AddRectangle(indices, {floor.lower, surface.upper, low_z.lower, high_z.upper});
  }

  for (const auto& disk : disks) {
    if (disk.radius_squared.upper == 0) {
      continue;
    }
    const auto radius = Root(disk.radius_squared);
    const auto y_support = Add(disk.lattice_y, {-radius.upper, radius.upper});
    const auto z_support = Add(disk.lattice_z, {-radius.upper, radius.upper});
    if (!Finite(y_support) || !Finite(z_support)) {
      return Status(absl::StatusCode::kOutOfRange);
    }
    const auto status = AddRectangle(
        indices,
        {
            .low_y = std::max(y_support.lower, floor.lower),
            .high_y = y_support.upper,
            .low_z = std::max(z_support.lower, low_z.lower),
            .high_z = std::min(z_support.upper, high_z.upper),
        }
    );
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::size_t> FindReceiverFaceOrdinal(
    const FrozenLiquidBoundaryGeometryView3D& frozen,
    std::int32_t piston_cell_x,
    PatchIndex patch,
    double velocity
) {
  const FaceIndex3D face_index{FaceNormalAxis::kX, {piston_cell_x, patch.y, patch.z}};
  const auto slot = frozen.mac_grid.FaceSlot(face_index);
  if (!slot.ok()) {
    return Status(absl::StatusCode::kFailedPrecondition);
  }
  const auto face = std::ranges::lower_bound(frozen.faces, *slot, {}, &FrozenLiquidFace3D::face_slot);
  const bool has_tail =
      piston_cell_x > kFirstLatticeCell && HasSelectedCell(frozen.cells, {piston_cell_x - 1, patch.y, patch.z});
  const double h = frozen.mac_grid.geometry().cell_width;
  if (face == frozen.faces.end() || face->face_slot != *slot || has_tail || face->open_area != 0
      || face->blocked_area != h * h || face->wall_normal_velocity != velocity || !face->prescribed_wall_flux
      || !std::isfinite(*face->prescribed_wall_flux) || (velocity == 0 && *face->prescribed_wall_flux != 0)
      || (velocity > 0 && *face->prescribed_wall_flux <= 0) || (velocity < 0 && *face->prescribed_wall_flux >= 0)) {
    return Status(absl::StatusCode::kFailedPrecondition);
  }
  return static_cast<std::size_t>(face - frozen.faces.begin());
}

bool IsFaceOrderValid(const FrozenLiquidBoundaryGeometryView3D& frozen) {
  std::size_t prior_slot = 0;
  bool first = true;
  for (const auto& face : frozen.faces) {
    const auto slot = frozen.mac_grid.FaceSlot(face.face_index);
    if (!slot.ok() || *slot != face.face_slot || (!first && prior_slot >= *slot)) {
      return false;
    }
    first = false;
    prior_slot = *slot;
  }
  return true;
}

bool IsCellOrderValid(std::span<const FrozenLiquidCell3D> cells) {
  for (std::size_t ordinal = 0; ordinal < cells.size(); ++ordinal) {
    const auto& cell = cells[ordinal];
    if (!std::isfinite(cell.fluid_volume) || cell.fluid_volume <= 0
        || (ordinal > 0 && CellOrder(cells[ordinal - 1].cell_index) >= CellOrder(cell.cell_index))) {
      return false;
    }
  }
  return true;
}

// Assign each patch in (z,y) order. The column's highest original receiver
// controls attachment even when the column contains other selected rows.
absl::StatusOr<std::size_t> AssignPatches(
    std::span<WettedPistonPatch3D> patches,
    std::span<const PatchMeasure> measures,
    std::span<std::optional<double>> fluxes,
    double velocity
) {
  std::size_t attached = 0;
  for (std::size_t begin = 0; begin < patches.size();) {
    std::size_t end = begin;
    std::optional<std::size_t> highest;
    while (end < patches.size() && patches[end].lattice_z == patches[begin].lattice_z) {
      if (patches[end].receiver_face_ordinal) {
        highest = end;
      }
      ++end;
    }
    for (std::size_t index = begin; index < end; ++index) {
      auto& patch = patches[index];
      const auto& measure = measures[index];
      if (patch.area.upper == 0) {
        continue;
      }
      auto* receiver = &patch;
      if (!patch.receiver_face_ordinal) {
        if (!highest || static_cast<std::int64_t>(patch.lattice_y) - patches[*highest].lattice_y != 1
            || !measure.all_wet_sources_cross_lower_edge) {
          return Status(absl::StatusCode::kFailedPrecondition);
        }
        receiver = &patches[*highest];
        patch.receiver_face_ordinal = receiver->receiver_face_ordinal;
        ++attached;
      }
      receiver->receiver_area = Add(receiver->receiver_area, patch.area);
      receiver->receiver_flux = Add(receiver->receiver_flux, WettedAreaToSignedWallFlux(patch.area, velocity));
      auto& receiver_flux = fluxes[*receiver->receiver_face_ordinal];
      *receiver_flux += measure.midpoint_flux;
      if (!Finite(receiver->receiver_area) || !Finite(receiver->receiver_flux) || !std::isfinite(*receiver_flux)) {
        return Status(absl::StatusCode::kOutOfRange);
      }
    }
    begin = end;
  }
  return attached;
}

template <typename Left, typename Right>
bool Overlap(std::span<Left> a, std::span<Right> b) {
  if (a.empty() || b.empty()) {
    return false;
  }
  const auto left = reinterpret_cast<std::uintptr_t>(a.data());
  const auto right = reinterpret_cast<std::uintptr_t>(b.data());
  return left <= right ? right - left < a.size_bytes() : left - right < b.size_bytes();
}

}  // namespace

struct WorkspaceAccess {
  static absl::StatusOr<WettedPistonView3D>
  Run(const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& workspace, Backend backend) {
    const auto& frozen = inputs.frozen;
    const auto geometry = frozen.mac_grid.geometry();
    const auto& capacities = workspace.capacities_;
    if (workspace.centers_.capacity() == 0) {
      return Status(absl::StatusCode::kFailedPrecondition);
    }
    if (frozen.faces.size() > capacities.faces || frozen.cells.size() > capacities.cells
        || inputs.particles.size() > capacities.particles) {
      return Status(absl::StatusCode::kResourceExhausted);
    }
    const auto aliases_inputs = [&](auto scratch) {
      return Overlap(scratch, std::span{&inputs, 1}) || Overlap(scratch, std::span{&inputs.field, 1})
          || Overlap(scratch, std::span{&frozen, 1}) || Overlap(scratch, frozen.faces) || Overlap(scratch, frozen.cells)
          || Overlap(scratch, inputs.particles) || Overlap(scratch, frozen.mac_grid.active_block_indices());
    };
    const auto candidate = 1 - workspace.live_publication_;
    auto& output = workspace.publications_[candidate];
    const auto& previous = workspace.publications_[workspace.live_publication_];
    if (aliases_inputs(output.patches.values()) || aliases_inputs(output.face_fluxes.values())
        || aliases_inputs(previous.patches.values()) || aliases_inputs(previous.face_fluxes.values())
        || aliases_inputs(workspace.indices_.values()) || aliases_inputs(workspace.measures_.values())
        || aliases_inputs(workspace.centers_.values()) || aliases_inputs(workspace.disks_.values())
        || aliases_inputs(workspace.intervals_.values())) {
      return Status(absl::StatusCode::kInvalidArgument);
    }
    const auto tank_status = ArithmeticStatus(boundary_internal::CheckTankStep(inputs.tank, 0));
    if (!tank_status.ok()) {
      return tank_status;
    }
    const double h = geometry.cell_width;
    const auto nominal_area = WorldArea(Point(1), h);
    if (!Finite(nominal_area) || nominal_area.lower <= 0) {
      return Status(absl::StatusCode::kOutOfRange);
    }
    if (!IsFaceOrderValid(frozen) || !IsCellOrderValid(frozen.cells)) {
      return Status(absl::StatusCode::kInvalidArgument);
    }

    Field field{.geometry = geometry, .tank = inputs.tank};
    const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&inputs.field);
    const auto* particle_union = std::get_if<LiquidParticleSphereUnion3D>(&inputs.field);
    const auto* plane = std::get_if<LiquidHorizontalPlaneSurface3D>(&inputs.field);
    if (plane) {
      field.is_plane = true;
      field.plane_height_y = plane->world_surface_height_y;
      if (!std::isfinite(field.plane_height_y)) {
        return Status(absl::StatusCode::kInvalidArgument);
      }
    } else {
      if (!sphere && !particle_union) {
        return Status(absl::StatusCode::kInvalidArgument);
      }
      field.sphere_radius = sphere ? sphere->radius : particle_union->radius;
      if (!std::isfinite(field.sphere_radius) || field.sphere_radius <= 0
          || (sphere && !FinitePosition(sphere->world_center))) {
        return Status(absl::StatusCode::kInvalidArgument);
      }
    }
    for (const auto& particle : inputs.particles) {
      if (!FinitePosition(particle.world_position)) {
        return Status(absl::StatusCode::kInvalidArgument);
      }
    }

    const auto piston_cell_x = FindPistonOwnerLatticeX(field);
    if (!piston_cell_x.ok()) {
      return piston_cell_x.status();
    }
    auto& indices = workspace.indices_;
    static_cast<void>(indices.Resize(0));
    const double velocity = inputs.tank.piston_velocity_x;
    for (const auto& cell : frozen.cells) {
      if (cell.cell_index.x == *piston_cell_x) {
        const auto status = AddIndex(indices, {cell.cell_index.y, cell.cell_index.z});
        if (!status.ok()) {
          return status;
        }
      }
    }

    auto& centers = workspace.centers_;
    auto& disks = workspace.disks_;
    const std::size_t source_count = sphere ? 1 : (particle_union ? inputs.particles.size() : 0);
    static_cast<void>(centers.Resize(source_count));
    static_cast<void>(disks.Resize(source_count));
    if (sphere) {
      centers[0] = sphere->world_center;
    } else {
      for (std::size_t index = 0; index < source_count; ++index) {
        centers[index] = inputs.particles[index].world_position;
      }
    }
    int architecture = 0;
    const auto sections_status =
        backend.sections(backend.context, field, centers.values(), disks.values(), architecture);
    if (!sections_status.ok()) {
      return sections_status;
    }
    for (const auto& disk : disks.values()) {
      const auto status = ArithmeticStatus(disk.error);
      if (!status.ok()) {
        return status;
      }
      if (!Finite(disk.radius_squared) || disk.radius_squared.lower < 0 || !Finite(disk.lattice_y)
          || !Finite(disk.lattice_z)) {
        return Status(absl::StatusCode::kOutOfRange);
      }
    }
    const auto support_status = AddSupport(field, disks.values(), indices);
    if (!support_status.ok()) {
      return support_status;
    }

    const auto patch_count = indices.size();
    static_cast<void>(output.patches.Resize(patch_count));
    static_cast<void>(output.face_fluxes.Resize(frozen.faces.size()));
    std::ranges::fill(output.face_fluxes.values(), std::nullopt);
    for (std::size_t index = 0; index < patch_count; ++index) {
      const auto patch = indices[index];
      auto& record = output.patches[index];
      record = {.lattice_y = patch.y, .lattice_z = patch.z};
      if (HasSelectedCell(frozen.cells, {*piston_cell_x, patch.y, patch.z})) {
        const auto receiver = FindReceiverFaceOrdinal(frozen, *piston_cell_x, patch, velocity);
        if (!receiver.ok()) {
          return receiver.status();
        }
        record.receiver_face_ordinal = *receiver;
        output.face_fluxes[*receiver] = 0;
      }
    }
    for (std::size_t ordinal = 0; ordinal < frozen.faces.size(); ++ordinal) {
      const auto original_flux = frozen.faces[ordinal].prescribed_wall_flux;
      if (original_flux && *original_flux != 0 && !output.face_fluxes[ordinal]) {
        return Status(absl::StatusCode::kFailedPrecondition);
      }
    }

    auto& measures = workspace.measures_;
    static_cast<void>(measures.Resize(patch_count));
    auto& intervals = workspace.intervals_;
    static_cast<void>(intervals.Resize(2 * source_count));
    const auto patch_status = backend.patches(
        backend.context,
        field,
        disks.values(),
        indices.values(),
        measures.values(),
        intervals.values(),
        architecture
    );
    if (!patch_status.ok()) {
      return patch_status;
    }
    Bounds whole_area;
    for (std::size_t index = 0; index < patch_count; ++index) {
      const auto& measure = measures[index];
      const auto status = ArithmeticStatus(measure.error);
      if (!status.ok()) {
        return status;
      }
      if (!Finite(measure.area) || measure.area.lower < 0 || measure.area.upper > nominal_area.upper
          || !std::isfinite(measure.midpoint_flux)) {
        return Status(absl::StatusCode::kOutOfRange);
      }
      output.patches[index].area = measure.area;
      whole_area = Add(whole_area, measure.area);
    }
    const auto whole_flux = WettedAreaToSignedWallFlux(whole_area, velocity);
    if (!Finite(whole_area) || !Finite(whole_flux)) {
      return Status(absl::StatusCode::kOutOfRange);
    }
    if (whole_area.upper > 0) {
      // Compare the upper-rounded width to a lower-rounded 1/1000 of the
      // lower bound, so rounding cannot qualify a wider enclosure.
      const double width = boundary_internal::AddUp(whole_area.upper, -whole_area.lower);
      const double limit = DirectedQuotient(whole_area.lower, 1000, false);
      if (whole_area.lower <= 0 || width > limit) {
        return Status(absl::StatusCode::kFailedPrecondition);
      }
    }
    const auto attached =
        AssignPatches(output.patches.values(), measures.values(), output.face_fluxes.values(), velocity);
    if (!attached.ok()) {
      return attached.status();
    }
    Bounds assigned_area;
    Bounds assigned_flux;
    for (const auto& patch : output.patches.values()) {
      if (patch.receiver_area.upper == 0) {
        continue;
      }
      const auto enclosure = patch.receiver_flux;
      const auto assigned = *output.face_fluxes[*patch.receiver_face_ordinal];
      if (!Finite(enclosure) || assigned < enclosure.lower || assigned > enclosure.upper) {
        return Status(absl::StatusCode::kOutOfRange);
      }
      assigned_area = Add(assigned_area, patch.receiver_area);
      assigned_flux = Add(assigned_flux, enclosure);
    }
    const bool area_agrees =
        Finite(assigned_area) && assigned_area.upper >= whole_area.lower && assigned_area.lower <= whole_area.upper;
    const bool flux_agrees =
        Finite(assigned_flux) && assigned_flux.upper >= whole_flux.lower && assigned_flux.lower <= whole_flux.upper;
    if (!area_agrees || !flux_agrees) {
      return Status(absl::StatusCode::kOutOfRange);
    }

    workspace.live_publication_ = candidate;
    return WettedPistonView3D{
        .patches = output.patches.values(),
        .face_fluxes = output.face_fluxes.values(),
        .whole_area = whole_area,
        .whole_flux = whole_flux,
        .attached_patch_count = *attached,
        .compiled_architecture = architecture,
    };
  }
};

absl::StatusOr<WettedPistonView3D>
Compute(const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& workspace, Backend backend) {
  return WorkspaceAccess::Run(inputs, workspace, backend);
}
}  // namespace wetted_piston_internal

absl::StatusOr<WettedPistonView3D>
ComputeWettedPiston3D(const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& workspace) {
  return wetted_piston_internal::Compute(
      inputs,
      workspace,
      {
          .sections =
              [](void*,
                 const wetted_piston_internal::Field& field,
                 std::span<const Vec3> centers,
                 std::span<wetted_piston_internal::Disk> disks,
                 int&) {
                for (std::size_t index = 0; index < centers.size(); ++index) {
                  disks[index] = wetted_piston_internal::Section(field, centers[index]);
                }
                return absl::OkStatus();
              },
          .patches =
              [](void*,
                 const wetted_piston_internal::Field& field,
                 std::span<const wetted_piston_internal::Disk> disks,
                 std::span<const wetted_piston_internal::PatchIndex> indices,
                 std::span<wetted_piston_internal::PatchMeasure> measures,
                 std::span<wetted_piston_internal::Interval> intervals,
                 int&) {
                for (std::size_t index = 0; index < indices.size(); ++index) {
                  measures[index] = wetted_piston_internal::MeasurePatch(
                      field,
                      indices[index],
                      disks.data(),
                      disks.size(),
                      intervals.first(disks.size()).data(),
                      intervals.last(disks.size()).data()
                  );
                }
                return absl::OkStatus();
              },
      }
  );
}
}  // namespace fluidity
