#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/functional/function_ref.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity::test {

inline constexpr double kSampledFrameDuration = 0.125;
inline constexpr std::array<GradedLiquidCornerSample3D, 8> kSampledCorners{{
    {{-1, -1, -1}, -1, 1},
    {{0, -1, -1}, -1, 1},
    {{-1, 0, -1}, -1, 1},
    {{0, 0, -1}, -1, 1},
    {{-1, -1, 0}, 1, 1},
    {{0, -1, 0}, 0.5, 1},
    {{-1, 0, 0}, 0.5, 1},
    {{0, 0, 0}, 1, 1},
}};
inline constexpr std::array<std::uint8_t, 1> kSampledAir{0};
inline constexpr std::array<double, 1> kSampledSources{0};
inline constexpr LiquidAdaptivePhysicalExteriorPressure3D kSampledOutlet{1, 0.25};

inline LiquidDigest3D GradedSampledRuntimeFingerprint() {
  std::array<std::uint8_t, 3 + 20 * 8> bytes{1, 3, 31};
  std::size_t next = 3;
  const auto append = [&](double value) {
    const auto bits = std::bit_cast<std::uint64_t>(value);
    for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
      bytes[next++] = static_cast<std::uint8_t>(bits >> (8 * byte));
    }
  };
  append(kSampledFrameDuration);
  append(kSampledSources[0]);
  append(kSampledOutlet.pressure_pa);
  append(kSampledOutlet.distance_m);
  for (const auto& sample : kSampledCorners) {
    append(sample.liquid_distance_m);
    append(sample.solid_distance_m);
  }
  return LiquidContentDigest3D(bytes);
}

inline absl::Status PopulateGradedSampledRuntimeFixture3D(LiquidState3D& state, LiquidSceneInputs3D& scene) {
  scene.field = LiquidHorizontalPlaneSurface3D{-0.9};
  scene.obstacle = TranslatingHalfspace3D{.unit_normal = {0, 1, 0}, .start_plane_coordinate = -4};
  scene.world_acceleration = {};
  scene.liquid_density = 1;
  auto cells = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(1);
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(6);
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(1);
  if (!cells.ok() || !faces.ok() || !particles.ok() || !cells->Resize(1).ok() || !faces->Resize(6).ok()
      || !particles->Resize(1).ok()) {
    return absl::ResourceExhaustedError("Sampled replay fixture storage failed");
  }
  (*cells)[0] = {
      .key = {{-1, -1, -1}, 0},
      .inventory = {.mass_kg = 0.125, .reference_volume_m3 = 0.125, .current_volume_m3 = 0.125},
  };
  (*particles)[0] = {
      .id = 1,
      .world_position = {-0.5, -0.5, -0.75},
      .mass = 0.125,
      .reference_volume = 0.125,
  };
  state.adaptive_cells = std::move(*cells);
  state.adaptive_faces = std::move(*faces);
  state.particles = std::move(*particles);
  state.next_particle_id = 2;
  return ValidateLiquidState3D(state);
}

inline absl::Status AdvanceGradedSampledRuntime3D(
    LiquidSubstepScheduler3D& scheduler,
    bool first_frame,
    absl::FunctionRef<absl::Status()> check_device
) {
  const auto begin = scheduler.BeginFrame(scheduler.committed().clock.time + kSampledFrameDuration);
  if (!begin.ok()) {
    return begin;
  }
  std::array<std::optional<LiquidAdaptivePhysicalExteriorPressure3D>, 6> exterior;
  exterior[4] = kSampledOutlet;  // Canonical low-Z slot, fully wet.
  const auto attempt = [&](double duration) {
    return scheduler.AttemptWithAdaptivePhysicalGeometry(
        duration,
        {
            .accepted_clock = scheduler.committed().clock,
            .target_divergences_s_inv = kSampledSources,
            .exterior_pressures = exterior,
            .sampled = LiquidAdaptiveSampledPhysicalGeometry3D{kSampledCorners, kSampledCorners, kSampledAir},
        }
    );
  };
  if (first_frame) {
    const auto before = EncodeLiquidState3D(scheduler.committed());
    if (!before.ok()) {
      return before.status();
    }
    const auto implicit_plane = scheduler.AttemptWithAdaptivePhysicalGeometry(
        kSampledFrameDuration,
        {.accepted_clock = scheduler.committed().clock, .target_divergences_s_inv = kSampledSources}
    );
    if (implicit_plane.status().code() != absl::StatusCode::kFailedPrecondition) {
      return absl::DataLossError("Sampled replay admitted the resolved plane in place of the sampled field");
    }
    auto incompatible_end = kSampledCorners;
    for (auto& sample : incompatible_end) {
      sample.liquid_distance_m = -1;
    }
    const auto changed_support = scheduler.AttemptWithAdaptivePhysicalGeometry(
        kSampledFrameDuration,
        {
            .accepted_clock = scheduler.committed().clock,
            .target_divergences_s_inv = kSampledSources,
            .exterior_pressures = exterior,
            .sampled = LiquidAdaptiveSampledPhysicalGeometry3D{kSampledCorners, incompatible_end, kSampledAir},
        }
    );
    if (changed_support.status().code() != absl::StatusCode::kFailedPrecondition) {
      return absl::DataLossError("Sampled replay admitted unsupported endpoint carrier coverage");
    }
    scheduler.SetStageFaultHookForTesting({
        .before_stage =
            [](void*, LiquidPhysicalStage3D stage, double) {
              return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject sampled acceptance")
                                                             : absl::OkStatus();
            },
    });
    const auto rejected = attempt(kSampledFrameDuration);
    scheduler.SetStageFaultHookForTesting({});
    const auto after = EncodeLiquidState3D(scheduler.committed());
    if (!rejected.ok() || !after.ok() || !rejected->retry_duration
        || scheduler.diagnostics().failed_stage != LiquidPhysicalStage3D::kAccept
        || !std::ranges::equal(before->values(), after->values())) {
      return absl::DataLossError("Sampled replay did not reach late rollback");
    }
    if (const auto status = check_device(); !status.ok()) {
      return status;
    }
    const auto retry = attempt(*rejected->retry_duration);
    if (!retry.ok() || retry->disposition != LiquidAttemptDisposition3D::kAccepted) {
      return absl::FailedPreconditionError("Sampled replay retry failed");
    }
  }
  const auto final = attempt(first_frame ? kSampledFrameDuration / 2 : kSampledFrameDuration);
  if (!final.ok() || final->disposition != LiquidAttemptDisposition3D::kAccepted) {
    return absl::FailedPreconditionError("Sampled physical replay attempt failed");
  }
  const auto& measurement = scheduler.diagnostics().accepted_measurement;
  if (!measurement || std::abs(measurement->step_start_pressure_cell_volume_m3 - 119.0 / 216.0) > 1e-14) {
    return absl::DataLossError("Sampled physical replay pressure volume differs");
  }
  if (std::abs(scheduler.committed().adaptive_cells[0].previous_pressure - kSampledOutlet.pressure_pa) > 1e-12) {
    return absl::DataLossError("Sampled physical replay did not solve the explicit pressure boundary");
  }
  return scheduler.FinishFrame();
}

}  // namespace fluidity::test
