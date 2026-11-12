#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/str_cat.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity::test {

inline constexpr double kGradedPhysicalWetTransitionDurationS = 0.125;

struct GradedPhysicalWetTransitionControl3D {
  double start_height_m;
  double end_height_m;
  std::array<double, 2> sources_s_inv;
};

inline constexpr GradedPhysicalWetTransitionControl3D GradedPhysicalWetTransitionControl(bool wetting) {
  return wetting ? GradedPhysicalWetTransitionControl3D{0, 0.125, {-0.5, 0}}
                 : GradedPhysicalWetTransitionControl3D{0.125, 0, {-0.25, 16}};
}

// The replay identity authenticates the explicit endpoint and canonical
// source fields in addition to the independently authored start pose.
inline LiquidDigest3D GradedPhysicalWetTransitionFingerprint(bool wetting) {
  const auto control = GradedPhysicalWetTransitionControl(wetting);
  const std::array values{
      control.start_height_m,
      control.end_height_m,
      control.sources_s_inv[0],
      control.sources_s_inv[1],
      kGradedPhysicalWetTransitionDurationS,
  };
  std::array<std::uint8_t, 3 + sizeof(values)> bytes{1, 3, static_cast<std::uint8_t>(wetting ? 28 : 29)};
  std::size_t next = 3;
  for (double value : values) {
    const auto bits = std::bit_cast<std::uint64_t>(value);
    for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
      bytes[next++] = static_cast<std::uint8_t>(bits >> (8 * byte));
    }
  }
  return LiquidContentDigest3D(bytes);
}

// Two signed coarse/fine leaves straddle y=0. A single particle crosses
// the shared subface; the other retains liquid in the coarse leaf.
inline absl::Status
PopulateGradedPhysicalWetTransitionFixture3D(LiquidState3D& state, LiquidSceneInputs3D& scene, bool wetting) {
  scene.field = LiquidHorizontalPlaneSurface3D{GradedPhysicalWetTransitionControl(wetting).start_height_m};
  scene.obstacle = TranslatingHalfspace3D{
      .unit_normal = {0, 1, 0},
      .start_plane_coordinate = -4,
      .wall_velocity = {},
  };
  scene.world_acceleration = {};
  scene.liquid_density = 1;
  constexpr std::array keys{AdaptiveMacCellKey3D{{-2, -2, 0}, 1}, AdaptiveMacCellKey3D{{-2, 0, 0}, 0}};
  const auto topology = AdaptiveMacTopology3D::Create(keys, state.geometry.cell_width);
  if (!topology.ok()) {
    return topology.status();
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  const std::size_t face_count = ownership->interior_face_slots.size() + ownership->exterior_faces.size();
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(2);
  auto cells = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(keys.size());
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(face_count);
  if (!particles.ok() || !cells.ok() || !faces.ok() || !particles->Resize(2).ok() || !cells->Resize(keys.size()).ok()
      || !faces->Resize(face_count).ok()) {
    return absl::ResourceExhaustedError("Graded physical wet transition fixture storage failed");
  }
  (*particles)[0] = {
      .id = 9,
      .world_position = {-1.5, -1.25, 0.5},
      .mass = 0.0625,
      .reference_volume = 0.0625,
      .pic_initialized = true,
  };
  (*particles)[1] = {
      .id = 3,
      .world_position = {-1.5, wetting ? -0.0625 : 0.0625, 0.5},
      .mass = 0.0625,
      .reference_volume = 0.0625,
      .pic_initialized = true,
  };
  const double coarse_inventory = wetting ? 0.125 : 0.0625;
  (*cells)[0] = {
      .key = keys[0],
      .inventory =
          {
              .mass_kg = coarse_inventory,
              .reference_volume_m3 = coarse_inventory,
              .current_volume_m3 = coarse_inventory,
          },
  };
  const double fine_inventory = wetting ? 0 : 0.0625;
  (*cells)[1] = {
      .key = keys[1],
      .occupancy = wetting ? LiquidCellOccupancy3D::kAir : LiquidCellOccupancy3D::kLiquid,
      .inventory =
          {
              .mass_kg = fine_inventory,
              .reference_volume_m3 = fine_inventory,
              .current_volume_m3 = fine_inventory,
          },
  };
  for (auto& face : faces->values()) {
    face.validity = LiquidFaceValidity3D::kValid;
  }
  state.next_particle_id = 10;
  state.particles = std::move(*particles);
  state.adaptive_cells = std::move(*cells);
  state.adaptive_faces = std::move(*faces);
  return ValidateLiquidState3D(state);
}

inline absl::Status AdvanceGradedPhysicalWetTransition3D(
    LiquidSubstepScheduler3D& scheduler,
    bool wetting,
    bool rollback_only,
    absl::FunctionRef<absl::Status()> check_device_epoch
) {
  constexpr double kDuration = kGradedPhysicalWetTransitionDurationS;
  constexpr std::array keys{AdaptiveMacCellKey3D{{-2, -2, 0}, 1}, AdaptiveMacCellKey3D{{-2, 0, 0}, 0}};
  const auto& initial = scheduler.committed();
  const auto topology = AdaptiveMacTopology3D::Create(keys, initial.geometry.cell_width);
  if (!topology.ok()) {
    return topology.status();
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  const auto outlet = std::ranges::find_if(ownership->exterior_faces, [](const auto& face) {
    return face.cell_ordinal == 0 && face.face_index.axis == FaceNormalAxis::kY && !face.cell_is_tail;
  });
  if (outlet == ownership->exterior_faces.end() || ownership->interior_face_slots.size() != 1) {
    return absl::FailedPreconditionError("Wet transition fixture lacks the signed coarse/fine pressure geometry");
  }
  std::vector<std::optional<LiquidAdaptivePhysicalExteriorPressure3D>> exterior(initial.adaptive_faces.size());
  exterior[outlet->face_slot] = LiquidAdaptivePhysicalExteriorPressure3D{.pressure_pa = 0, .distance_m = 1};
  const auto control = GradedPhysicalWetTransitionControl(wetting);
  const LiquidAdaptivePhysicalGeometryRequest3D request{
      .accepted_clock = initial.clock,
      .target_divergences_s_inv = control.sources_s_inv,
      .exterior_pressures = exterior,
      .end_surface_height_m = control.end_height_m,
  };
  const auto begun = scheduler.BeginFrame(kDuration);
  if (!begun.ok()) {
    return begun;
  }
  auto stale = request;
  ++stale.accepted_clock.epoch;
  if (scheduler.AttemptWithAdaptivePhysicalGeometry(kDuration, stale).status().code()
      != absl::StatusCode::kFailedPrecondition) {
    return absl::FailedPreconditionError("Wet transition fixture accepted a stale endpoint request");
  }
  if (rollback_only) {
    const auto before = EncodeLiquidState3D(initial);
    if (!before.ok()) {
      return before.status();
    }
    scheduler.SetStageFaultHookForTesting({
        .before_stage =
            [](void*, LiquidPhysicalStage3D stage, double) {
              return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject wet transition at acceptance")
                                                             : absl::OkStatus();
            },
    });
    const auto rejected = scheduler.AttemptWithAdaptivePhysicalGeometry(kDuration, request);
    scheduler.SetStageFaultHookForTesting({});
    if (!rejected.ok()) {
      return absl::Status(
          rejected.status().code(),
          absl::StrCat(
              "Wet transition fault attempt failed at stage ",
              static_cast<int>(scheduler.diagnostics().failed_stage),
              " with ",
              rejected.status().message()
          )
      );
    }
    const auto after_rejection = EncodeLiquidState3D(scheduler.committed());
    if (!after_rejection.ok()) {
      return after_rejection.status();
    }
    if (!std::ranges::equal(before->values(), after_rejection->values())) {
      return absl::DataLossError("Rejected wet transition did not preserve the accepted state");
    }
    if (rejected->disposition != LiquidAttemptDisposition3D::kRetryDt
        || scheduler.diagnostics().failed_stage != LiquidPhysicalStage3D::kAccept) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Wet transition did not reach late rollback. Stage ",
          static_cast<int>(scheduler.diagnostics().failed_stage),
          " status ",
          static_cast<int>(scheduler.diagnostics().failure_code)
      ));
    }
    return check_device_epoch();
  }
  const auto accepted = scheduler.AttemptWithAdaptivePhysicalGeometry(kDuration, request);
  if (!accepted.ok()) {
    return absl::Status(
        accepted.status().code(),
        absl::StrCat(
            "Wet transition attempt failed at stage ",
            static_cast<int>(scheduler.diagnostics().failed_stage),
            " with ",
            accepted.status().message()
        )
    );
  }
  if (accepted->disposition != LiquidAttemptDisposition3D::kAccepted) {
    return absl::Status(scheduler.diagnostics().failure_code, "Wet transition physical attempt was refused");
  }
  const auto& published = scheduler.committed();
  if (published.adaptive_cells.size() != 2 || published.particles.size() != 2 || published.next_particle_id != 10) {
    return absl::DataLossError("Wet transition changed the canonical roster");
  }
  const auto& coarse = published.adaptive_cells[0];
  const auto& fine = published.adaptive_cells[1];
  const auto& mover = published.particles[1];
  // In drying, wet pressure-sealed neighboring anchors contribute zero
  // velocity to PIC while the shared subface carries the prescribed -2 m/s.
  const double target_speed_y = wetting ? 1 : -0.8425385510708052;
  const double start_y = wetting ? -0.0625 : 0.0625;
  const double target_y = std::fma(kDuration, target_speed_y, start_y);
  if (published.particles[0].id != 9 || mover.id != 3 || mover.mass != 0.0625 || mover.reference_volume != 0.0625
      || mover.volume_ratio != 1 || mover.world_position.x != -1.5 || mover.world_position.z != 0.5
      || std::abs(mover.world_position.y - target_y) > 1e-12 || coarse.inventory.mass_kg != (wetting ? 0.0625 : 0.125)
      || fine.inventory.mass_kg != (wetting ? 0.0625 : 0)
      || coarse.inventory.current_volume_m3 + fine.inventory.current_volume_m3 != 0.125
      || coarse.inventory.reference_volume_m3 + fine.inventory.reference_volume_m3 != 0.125
      || fine.occupancy != (wetting ? LiquidCellOccupancy3D::kLiquid : LiquidCellOccupancy3D::kAir)
      || fine.previous_pressure != 0) {
    return absl::DataLossError("Wet transition did not publish the expected stable material");
  }
  if (std::abs(mover.world_velocity.y - target_speed_y) > 1e-12
      || (wetting && std::abs(fine.inventory.momentum_kg_m_per_s.y - 0.0625) > 1e-12)
      || (!wetting && coarse.inventory.momentum_kg_m_per_s.y >= -0.0625)) {
    return absl::DataLossError("Wet transition lost the projected physical impulse");
  }
  const auto shared = published.adaptive_faces[ownership->interior_face_slots[0]];
  if (shared.validity != LiquidFaceValidity3D::kMissing || shared.carrier_velocity != 0) {
    return absl::DataLossError("Wet transition reused a carrier across a changed pressure boundary");
  }
  const auto finished = scheduler.FinishFrame();
  if (!finished.ok()) {
    return finished;
  }
  const auto continuation = scheduler.BeginFrame(2 * kDuration);
  if (!continuation.ok()) {
    return continuation;
  }
  const auto old_surface = scheduler.AttemptWithAdaptivePhysicalGeometry(
      kDuration,
      {
          .accepted_clock = scheduler.committed().clock,
          .target_divergences_s_inv = control.sources_s_inv,
          .exterior_pressures = exterior,
      }
  );
  if (old_surface.status().code() != absl::StatusCode::kFailedPrecondition) {
    return absl::DataLossError("Wet transition admitted a continuation with the old resolved plane");
  }
  return check_device_epoch();
}

}  // namespace fluidity::test
