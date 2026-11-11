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

FrozenMotionCarrier3D::FrozenMotionCarrier3D(SparseMacGrid3D grid, LiquidObstacle3D obstacle, std::vector<Face> faces)
    : grid_(std::move(grid)), obstacle_(std::move(obstacle)), faces_(std::move(faces)) {
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
  const auto obstacle = liquid_step_internal::ObstacleAtTime(scene.obstacle, state.clock.time);
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
  return FrozenMotionCarrier3D(std::move(*grid), obstacle, std::move(faces));
}

absl::StatusOr<MotionCarrierSample3D> FrozenMotionCarrier3D::Sample(Vec3 position, std::int64_t side) const {
  if (!liquid_step_internal::Finite(position)) {
    return absl::InvalidArgumentError("motion position must be finite");
  }
  if (liquid_step_internal::SolidDistance(obstacle_, position) < 0) {
    return absl::FailedPreconditionError("motion query lies inside the obstacle");
  }
  MotionCarrierSample3D result{.velocity = {}};
  const double width = grid_.geometry().cell_width;
  for (std::size_t component = 0; component < kAxes.size(); ++component) {
    const auto stencil = BuildMacFaceGeometryStencil3D(grid_.geometry(), kAxes[component], position, position);
    if (!stencil.ok()) {
      return stencil.status();
    }
    double velocity = 0;
    bool complete = true;
    for (const auto& node : *stencil) {
      if (node.weight == 0) {
        continue;
      }
      const auto slot = grid_.FaceSlot(node.face_index);
      if (!slot.ok() || !Usable(faces_[*slot].validity) || !faces_[*slot].tag.structurally_eligible
          || faces_[*slot].tag.side_tag != side) {
        complete = false;
        break;
      }
      velocity += node.weight * faces_[*slot].velocity;
    }
    if (!complete) {
      result.continued = true;
      std::array<internal::LiquidGhostFitSample3D, 125> donors;
      std::size_t count = 0;
      double nearest_distance = std::numeric_limits<double>::infinity();
      double nearest_velocity = 0;
      for (const auto& face : faces_) {
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
