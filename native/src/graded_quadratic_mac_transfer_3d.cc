#include "fluidity/host/graded_quadratic_mac_transfer_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "mac_deposit_3d_internal.h"
#include "mac_gather_3d_internal.h"

namespace fluidity {
namespace {

using FaceKey = std::tuple<FaceNormalAxis, std::int64_t, std::int64_t, std::int64_t, std::uint8_t>;

FaceKey CanonicalKey(const AdaptiveMacInteriorFace3D& face) {
  const auto index = face.face_index.lattice_index;
  return {face.face_index.axis, index.z, index.y, index.x, face.width_exponent};
}

// Map the uniform level's signed lattice to the frozen finest lattice.
absl::StatusOr<FaceKey> CheckedFinestKey(FaceIndex3D level_face, std::uint8_t width_exponent) {
  const std::int64_t lattice_width = std::int64_t{1} << width_exponent;
  const auto index = level_face.lattice_index;
  const std::array finest{index.x * lattice_width, index.y * lattice_width, index.z * lattice_width};
  for (const auto coordinate : finest) {
    if (coordinate < std::numeric_limits<std::int32_t>::min()
        || coordinate > std::numeric_limits<std::int32_t>::max()) {
      return absl::OutOfRangeError("quadratic face index exceeds the signed finest lattice");
    }
  }
  return FaceKey{level_face.axis, finest[2], finest[1], finest[0], width_exponent};
}

absl::Status CheckCarrier(const LiquidAdaptiveFaceState3D& carrier, std::int64_t side_tag) {
  if (!std::isfinite(carrier.carrier_velocity) || carrier.validity > LiquidFaceValidity3D::kLowMass) {
    return absl::InvalidArgumentError("quadratic support requires finite carriers and known validity");
  }
  if (!carrier.structurally_eligible || carrier.side_tag != side_tag
      || carrier.validity != LiquidFaceValidity3D::kValid) {
    return absl::FailedPreconditionError("quadratic support requires valid eligible carriers on the particle side");
  }
  return absl::OkStatus();
}

absl::Status CheckOpenArea(double open_area_m2, double face_area_m2) {
  if (!std::isfinite(open_area_m2) || open_area_m2 < 0 || open_area_m2 > face_area_m2) {
    return absl::InvalidArgumentError("quadratic support aperture is outside its canonical face area");
  }
  if (open_area_m2 != face_area_m2) {
    return absl::FailedPreconditionError("quadratic support requires a fully open face");
  }
  return absl::OkStatus();
}

bool MatchesAnchorDisplacement(Vec3 anchor, Vec3 position, Vec3 displacement) {
  return anchor.x - position.x == displacement.x && anchor.y - position.y == displacement.y
      && anchor.z - position.z == displacement.z;
}

struct PairedQuadraticSupport {
  const GradedQuadraticMacSupportSnapshot3D& committed;
  const GradedQuadraticMacSupportSnapshot3D& candidate;
  double committed_cell_width_m;
  LiquidResamplingAdmissibility3D certify;
  void* context;
};

// Resampling supplies the accepted width for the old roster and the other
// width for its proposal. Bind checks the actual containing leaf in each.
absl::Status CheckPairedQuadraticSupport(
    const LiquidParticleState3D& particle,
    double cell_width_m,
    LiquidUniformResamplingBasis3D basis,
    void* context
) {
  const auto& paired = *static_cast<const PairedQuadraticSupport*>(context);
  const auto& snapshot = cell_width_m == paired.committed_cell_width_m ? paired.committed : paired.candidate;
  const GradedQuadraticMacParticle3D transfer_particle{
      .transfer =
          {
              .sorted_world_position = particle.world_position,
              .trial_world_position = particle.world_position,
              .particle_mass = particle.mass,
              .particle_velocity = particle.world_velocity,
              .particle_velocity_slopes = particle.velocity_slopes,
          },
      .side_tag = particle.side_tag,
  };
  const auto bound = snapshot.transfer.Bind(transfer_particle, snapshot.open_areas_m2, snapshot.carrier_faces);
  if (!bound.ok()) {
    return bound.status();
  }
  if (bound->cell_width_m != cell_width_m) {
    return absl::FailedPreconditionError("resampling support has the wrong physical leaf width");
  }
  return paired.certify(particle, cell_width_m, basis, paired.context);
}

}  // namespace

GradedQuadraticMacTransfer3D::GradedQuadraticMacTransfer3D(
    AdaptiveMacTopology3D topology,
    AdaptiveMacFaceOwnership3D ownership,
    GradedMacTransfer3D anchors,
    Vec3 world_origin
)
    : topology_(std::move(topology)),
      ownership_(std::move(ownership)),
      anchors_(std::move(anchors)),
      world_origin_(world_origin) {
}

absl::StatusOr<GradedQuadraticMacTransfer3D>
GradedQuadraticMacTransfer3D::Create(AdaptiveMacTopology3D topology, Vec3 world_origin) {
  auto ownership = BuildAdaptiveMacFaceOwnership3D(topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  auto anchors = GradedMacTransfer3D::Create(topology, world_origin);
  if (!anchors.ok()) {
    return anchors.status();
  }
  return GradedQuadraticMacTransfer3D(std::move(topology), std::move(*ownership), std::move(*anchors), world_origin);
}

absl::StatusOr<GradedQuadraticMacStencil3D> GradedQuadraticMacTransfer3D::Bind(
    const GradedQuadraticMacParticle3D& particle,
    std::span<const double> open_areas_m2,
    std::span<const LiquidAdaptiveFaceState3D> carrier_faces
) const {
  const auto anchors = anchors_.faces();
  if (open_areas_m2.size() != anchors.size() || carrier_faces.size() != anchors.size()) {
    return absl::InvalidArgumentError("quadratic aperture and carrier extents must cover the canonical faces");
  }
  const auto sorted_leaf = anchors_.ParticleLeafOrdinal(particle.transfer.sorted_world_position);
  if (!sorted_leaf.ok()) {
    return sorted_leaf.status();
  }
  const auto trial_leaf = anchors_.ParticleLeafOrdinal(particle.transfer.trial_world_position);
  if (!trial_leaf.ok()) {
    return trial_leaf.status();
  }
  const auto leaves = topology_.cells();
  const auto& leaf = leaves[*trial_leaf];
  const auto width_exponent = leaf.key.width_exponent;
  if (leaves[*sorted_leaf].key.width_exponent != width_exponent) {
    return absl::FailedPreconditionError("quadratic transfer endpoints must use the same adaptive level");
  }

  const auto interior_faces = topology_.interior_faces();
  const UniformMacGeometry3D level_geometry{world_origin_, leaf.cell_width};
  GradedQuadraticMacStencil3D bound{.cell_width_m = leaf.cell_width, .rows = {}};
  for (const auto component : internal::kNormalComponentAccess) {
    const auto geometry = BuildMacFaceGeometryStencil3D(
        level_geometry,
        component.axis,
        particle.transfer.sorted_world_position,
        particle.transfer.trial_world_position
    );
    if (!geometry.ok()) {
      return geometry.status();
    }
    auto& row = bound.rows[static_cast<std::size_t>(component.axis)];
    for (std::size_t node = 0; node < row.size(); ++node) {
      const auto& sample = (*geometry)[node];
      const auto key = CheckedFinestKey(sample.face_index, width_exponent);
      if (!key.ok()) {
        return key.status();
      }
      const auto face = std::ranges::lower_bound(interior_faces, *key, {}, CanonicalKey);
      if (face == interior_faces.end() || CanonicalKey(*face) != *key) {
        return absl::FailedPreconditionError("quadratic tensor node has no matching canonical interior face");
      }
      if (leaves[face->tail_cell_ordinal].key.width_exponent != width_exponent
          || leaves[face->head_cell_ordinal].key.width_exponent != width_exponent) {
        return absl::FailedPreconditionError("quadratic support requires both incident leaves at the particle level");
      }
      const auto ordinal = static_cast<std::size_t>(face - interior_faces.begin());
      const auto slot = ownership_.interior_face_slots[ordinal];
      if (!MatchesAnchorDisplacement(
              anchors[slot].world_anchor,
              particle.transfer.trial_world_position,
              sample.face_offset_from_particle
          )) {
        return absl::OutOfRangeError("quadratic sample displacement differs from its physical adaptive anchor");
      }
      const auto aperture = CheckOpenArea(open_areas_m2[slot], face->face_area);
      if (!aperture.ok()) {
        return aperture;
      }
      const auto carrier = CheckCarrier(carrier_faces[slot], particle.side_tag);
      if (!carrier.ok()) {
        return carrier;
      }
      row[node] = {
          .face_slot = slot,
          .weight = sample.weight,
          .face_offset_from_particle = sample.face_offset_from_particle,
          .weight_gradient = sample.weight_gradient,
      };
    }
  }
  return bound;
}

absl::StatusOr<std::vector<MacFaceDeposit3D>> GradedQuadraticMacTransfer3D::Deposit(
    std::span<const GradedQuadraticMacParticle3D> particles,
    std::span<const double> open_areas_m2,
    std::span<const LiquidAdaptiveFaceState3D> carrier_faces
) const {
  const auto face_count = anchors_.faces().size();
  if (open_areas_m2.size() != face_count || carrier_faces.size() != face_count) {
    return absl::InvalidArgumentError("quadratic aperture and carrier extents must cover the canonical faces");
  }
  std::vector<MacFaceDeposit3D> deposits(face_count);
  for (const auto& particle : particles) {
    const auto input = internal::ValidateMacDepositParticle(particle.transfer);
    if (!input.ok()) {
      return input;
    }
    const auto bound = Bind(particle, open_areas_m2, carrier_faces);
    if (!bound.ok()) {
      return bound.status();
    }
    for (const auto component : internal::kNormalComponentAccess) {
      const auto status = internal::AccumulateMacParticleComponent(
          bound->rows[static_cast<std::size_t>(component.axis)],
          particle.transfer,
          component,
          std::span(deposits),
          internal::AccumulateMacFaceDeposit
      );
      if (!status.ok()) {
        return status;
      }
    }
  }
  return deposits;
}

absl::StatusOr<MacParticleGatherResult3D> GradedQuadraticMacTransfer3D::Gather(
    const GradedQuadraticMacParticle3D& particle,
    std::span<const double> open_areas_m2,
    std::span<const LiquidAdaptiveFaceState3D> old_carrier_faces,
    std::span<const LiquidAdaptiveFaceState3D> new_carrier_faces
) const {
  if (new_carrier_faces.size() != anchors_.faces().size()) {
    return absl::InvalidArgumentError("quadratic new carrier extent must cover the canonical faces");
  }
  const auto bound = Bind(particle, open_areas_m2, old_carrier_faces);
  if (!bound.ok()) {
    return bound.status();
  }
  for (const auto& row : bound->rows) {
    for (const auto& sample : row) {
      const auto checked = CheckCarrier(new_carrier_faces[sample.face_slot], particle.side_tag);
      if (!checked.ok()) {
        return checked;
      }
    }
  }
  const auto old_velocity = [&](std::size_t slot) {
    return old_carrier_faces[slot].carrier_velocity;
  };
  const auto new_velocity = [&](std::size_t slot) {
    return new_carrier_faces[slot].carrier_velocity;
  };
  MacParticleGatherResult3D gathered{};
  for (const auto component : internal::kNormalComponentAccess) {
    const auto row = internal::GatherMacVelocityComponentFromSlots(
        bound->rows[static_cast<std::size_t>(component.axis)],
        bound->cell_width_m,
        old_velocity,
        new_velocity
    );
    if (!row.ok()) {
      return row.status();
    }
    gathered.new_pic_velocity.*component.velocity_component = row->new_pic_velocity;
    gathered.grid_velocity_increment.*component.velocity_component = row->grid_velocity_increment;
    gathered.fitted_velocity_slopes.*component.velocity_slope_row = row->fitted_velocity_slope_row;
    gathered.velocity_gradient.*component.velocity_slope_row = row->velocity_gradient_row;
  }
  return gathered;
}

absl::StatusOr<LiquidUniformSplit3D> SplitGradedQuadraticLiquidParticle3D(
    const LiquidParticleState3D& parent,
    std::uint64_t first_child_id,
    double parent_cell_width_m,
    const GradedQuadraticMacSupportSnapshot3D& committed,
    const GradedQuadraticMacSupportSnapshot3D& candidate,
    LiquidResamplingAdmissibility3D certify,
    void* context
) {
  if (certify == nullptr) {
    return absl::InvalidArgumentError("split requires particle admissibility certification");
  }
  PairedQuadraticSupport paired{committed, candidate, parent_cell_width_m, certify, context};
  return SplitUniformLiquidParticle3D(
      parent,
      first_child_id,
      parent_cell_width_m,
      LiquidUniformResamplingBasis3D::kQuadratic,
      CheckPairedQuadraticSupport,
      &paired
  );
}

absl::StatusOr<LiquidUniformMerge3D> MergeGradedQuadraticLiquidParticles3D(
    std::span<const LiquidParticleState3D> children,
    std::uint64_t target_id,
    double fine_cell_width_m,
    const GradedQuadraticMacSupportSnapshot3D& committed,
    const GradedQuadraticMacSupportSnapshot3D& candidate,
    LiquidResamplingAdmissibility3D certify,
    void* context
) {
  if (certify == nullptr) {
    return absl::InvalidArgumentError("merge requires particle admissibility certification");
  }
  PairedQuadraticSupport paired{committed, candidate, fine_cell_width_m, certify, context};
  return MergeUniformLiquidParticles3D(
      children,
      target_id,
      fine_cell_width_m,
      LiquidUniformResamplingBasis3D::kQuadratic,
      CheckPairedQuadraticSupport,
      &paired
  );
}
}  // namespace fluidity
