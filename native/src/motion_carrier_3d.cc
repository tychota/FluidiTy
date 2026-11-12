#include "fluidity/host/motion_carrier_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

#include "liquid_boundary_transfer_3d_internal.h"
#include "liquid_substep_3d_internal.h"

namespace fluidity {
namespace {
constexpr std::array kAxes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};

bool Usable(LiquidFaceValidity3D validity) {
  return validity == LiquidFaceValidity3D::kValid || validity == LiquidFaceValidity3D::kPredicted;
}
}  // namespace

FrozenMotionCarrier3D::FrozenMotionCarrier3D(
    LiquidObstacle3D obstacle,
    std::variant<UniformCarrier, AdaptiveCarrier> carrier
)
    : obstacle_(std::move(obstacle)), carrier_(std::move(carrier)) {
}

absl::StatusOr<FrozenMotionCarrier3D> FrozenMotionCarrier3D::Create(
    const LiquidState3D& state,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& options
) {
  const auto valid = ValidateLiquidState3D(state);
  if (!valid.ok()) {
    return valid;
  }
  const auto obstacle = liquid_step_internal::ObstacleAtTime(scene.obstacle, state.clock.time);
  if (!state.adaptive_cells.values().empty()) {
    std::vector<AdaptiveMacCellKey3D> keys;
    keys.reserve(state.adaptive_cells.size());
    for (const auto& cell : state.adaptive_cells.values()) {
      keys.push_back(cell.key);
    }
    const auto topology = AdaptiveMacTopology3D::Create(keys, state.geometry.cell_width);
    if (!topology.ok()) {
      return topology.status();
    }
    std::optional<Vec3> solid_upper;
    std::optional<PlanarTop> open_top;
    if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle)) {
      solid_upper = Vec3{tank->high_x, 0, tank->high_z};
      const auto* plane = std::get_if<LiquidHorizontalPlaneSurface3D>(&scene.field);
      if (plane && tank->piston_velocity_x == 0 && std::isfinite(plane->world_surface_height_y)) {
        open_top = PlanarTop{
            .height_m = plane->world_surface_height_y,
            .continuation_width_m = 0.5 * state.geometry.cell_width,
        };
      }
    }
    auto transfer = GradedMacTransfer3D::Create(*topology, state.geometry.world_origin, solid_upper);
    if (!transfer.ok()) {
      return transfer.status();
    }
    std::vector<LiquidAdaptiveFaceState3D> faces(
        state.adaptive_faces.values().begin(),
        state.adaptive_faces.values().end()
    );
    for (auto& face : faces) {
      if (!Usable(face.validity)) {
        face.validity = LiquidFaceValidity3D::kMissing;
      }
    }
    return FrozenMotionCarrier3D(
        obstacle,
        AdaptiveCarrier{.transfer = std::move(*transfer), .faces = std::move(faces), .open_top = open_top}
    );
  }
  std::vector<BlockIndex3D> blocks;
  for (const auto& block : state.blocks.values()) {
    blocks.push_back(block.block_index);
  }
  const bool initial = state.clock == LiquidClock3D{} && state.faces.size() == 0;
  std::vector<LiquidBoundaryParticle3D> particles;
  if (initial) {
    for (const auto& particle : state.particles.values()) {
      particles.push_back({
          .transfer =
              {particle.world_position,
               particle.world_position,
               particle.mass,
               particle.world_velocity,
               particle.velocity_slopes},
          .side_tag = particle.side_tag,
          .affine_enabled = particle.affine_enabled,
      });
      for (const auto axis : kAxes) {
        const auto stencil =
            BuildMacFaceGeometryStencil3D(state.geometry, axis, particle.world_position, particle.world_position);
        if (!stencil.ok()) {
          return stencil.status();
        }
        for (const auto& node : *stencil) {
          blocks.push_back(MacOwnerBlock3D(node.face_index.lattice_index));
        }
      }
    }
  }
  const auto key = [](BlockIndex3D value) {
    return std::tuple(value.z, value.y, value.x);
  };
  std::sort(blocks.begin(), blocks.end(), [&](auto lhs, auto rhs) {
    return key(lhs) < key(rhs);
  });
  blocks.erase(
      std::unique(
          blocks.begin(),
          blocks.end(),
          [&](auto lhs, auto rhs) {
            return key(lhs) == key(rhs);
          }
      ),
      blocks.end()
  );
  auto grid = SparseMacGrid3D::Create(state.geometry, blocks, 128);
  if (!grid.ok()) {
    return grid.status();
  }
  std::vector<Face> faces;
  std::vector<LiquidBoundaryFaceTag3D> tags(grid->face_count());
  const auto tagged = liquid_step_internal::FillFaceTags(grid->view(), obstacle, tags);
  if (!tagged.ok()) {
    return tagged;
  }
  for (std::size_t slot = 0; slot < grid->face_count(); ++slot) {
    const auto index = grid->view().FaceIndex(slot);
    if (!index.ok()) {
      return index.status();
    }
    const auto anchor = grid->FaceWorldAnchor(*index);
    if (!anchor.ok()) {
      return anchor.status();
    }
    faces.push_back({
        .position = *anchor,
        .axis = index->axis,
        .velocity = 0,
        .validity = LiquidFaceValidity3D::kMissing,
        .tag = tags[slot],
    });
  }
  if (initial && !particles.empty()) {
    const MacDepositQuantization3D quantization{
        options.mass_per_quantum,
        options.momentum_per_quantum,
        state.particles.size(),
    };
    const auto deposited = DepositCompatibleMacQuanta3D({*grid, particles, tags, quantization});
    if (!deposited.ok()) {
      return deposited.status();
    }
    const auto normalized = NormalizeLiquidMacFaces3D(deposited->face_quanta, quantization, options.minimum_face_mass);
    if (!normalized.ok()) {
      return normalized.status();
    }
    for (std::size_t slot = 0; slot < faces.size(); ++slot) {
      faces[slot].velocity = normalized->face_velocities[slot];
      faces[slot].validity = normalized->validity[slot];
    }
  } else {
    for (const auto& source : state.faces.values()) {
      if (source.face_slot >= faces.size()) {
        return absl::DataLossError("motion face is outside the owned layout");
      }
      faces[source.face_slot].velocity = source.carrier_velocity;
      faces[source.face_slot].validity = source.validity;
      faces[source.face_slot].tag = {source.side_tag, source.structurally_eligible};
    }
  }
  return FrozenMotionCarrier3D(obstacle, UniformCarrier{.grid = std::move(*grid), .faces = std::move(faces)});
}

absl::StatusOr<MotionCarrierSample3D> FrozenMotionCarrier3D::Sample(Vec3 position, std::int64_t side) const {
  if (!liquid_step_internal::Finite(position)) {
    return absl::InvalidArgumentError("motion position must be finite");
  }
  if (liquid_step_internal::SolidDistance(obstacle_, position) < 0) {
    return absl::FailedPreconditionError("motion query lies inside the obstacle");
  }
  if (const auto* graded = std::get_if<AdaptiveCarrier>(&carrier_)) {
    auto sample_position = position;
    bool continue_top = false;
    if (const auto& top = graded->open_top) {
      const double top_clearance_m = position.y - top->height_m;
      continue_top = top_clearance_m >= 0 && top_clearance_m <= top->continuation_width_m;
      if (continue_top) {
        sample_position.y = std::nextafter(top->height_m, -std::numeric_limits<double>::infinity());
      }
    }
    const auto sampled = graded->transfer.Gather({.world_position = sample_position, .side_tag = side}, graded->faces);
    if (!sampled.ok()) {
      return sampled.status();
    }
    auto velocity = sampled->velocity;
    if (continue_top && sampled->affine_enabled) {
      const double offset_m = position.y - sample_position.y;
      const auto& slopes = sampled->velocity_slopes;
      velocity = {
          std::fma(offset_m, slopes.x_row.y, velocity.x),
          std::fma(offset_m, slopes.y_row.y, velocity.y),
          std::fma(offset_m, slopes.z_row.y, velocity.z),
      };
      if (!liquid_step_internal::Finite(velocity)) {
        return absl::OutOfRangeError("graded motion continuation overflowed");
      }
    }
    return MotionCarrierSample3D{
        .velocity = velocity,
        .continued = continue_top || !sampled->affine_enabled,
        .rank_fallback = !sampled->affine_enabled,
    };
  }
  const auto& uniform = std::get<UniformCarrier>(carrier_);
  MotionCarrierSample3D result{.velocity = {}};
  const double width = uniform.grid.geometry().cell_width;
  for (std::size_t component = 0; component < kAxes.size(); ++component) {
    const auto stencil = BuildMacFaceGeometryStencil3D(uniform.grid.geometry(), kAxes[component], position, position);
    if (!stencil.ok()) {
      return stencil.status();
    }
    double velocity = 0;
    bool complete = true;
    for (const auto& node : *stencil) {
      if (node.weight == 0) {
        continue;
      }
      const auto slot = uniform.grid.FaceSlot(node.face_index);
      if (!slot.ok() || !Usable(uniform.faces[*slot].validity) || !uniform.faces[*slot].tag.structurally_eligible
          || uniform.faces[*slot].tag.side_tag != side) {
        complete = false;
        break;
      }
      velocity += node.weight * uniform.faces[*slot].velocity;
    }
    if (!complete) {
      result.continued = true;
      std::array<internal::LiquidGhostFitSample3D, 125> donors;
      std::size_t count = 0;
      double nearest_distance = std::numeric_limits<double>::infinity();
      double nearest_velocity = 0;
      for (const auto& face : uniform.faces) {
        if (face.axis != kAxes[component] || !Usable(face.validity) || !face.tag.structurally_eligible
            || face.tag.side_tag != side) {
          continue;
        }
        const Vec3 offset{
            (face.position.x - position.x) / width,
            (face.position.y - position.y) / width,
            (face.position.z - position.z) / width
        };
        const double squared = (offset.x * offset.x + offset.y * offset.y) + offset.z * offset.z;
        if (squared > 4) {
          continue;
        }
        if (count == donors.size()) {
          return absl::ResourceExhaustedError("motion donor bound exceeded");
        }
        donors[count++] = {offset, face.velocity};
        if (squared < nearest_distance) {
          nearest_distance = squared;
          nearest_velocity = face.velocity;
        }
      }
      if (count == 0) {
        return absl::FailedPreconditionError("motion query has no valid carrier halo");
      }
      const auto fitted = internal::FitLiquidGhostAffine3D(std::span(donors).first(count));
      if (!fitted.ok()) {
        return fitted.status();
      }
      result.rank_fallback = result.rank_fallback || !fitted->full_rank;
      velocity = fitted->full_rank ? fitted->velocity_at_target : nearest_velocity;
    }
    if (!std::isfinite(velocity)) {
      return absl::OutOfRangeError("motion interpolation is not finite");
    }
    result.velocity.*kComponents[component] = velocity;
  }
  return result;
}
}  // namespace fluidity
