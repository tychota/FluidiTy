#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "../src/liquid_substep_3d_internal.h"
#include "absl/status/status.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/liquid_substep_3d.h"
#include "gtest/gtest.h"

namespace fluidity::liquid_step_internal {
namespace {

constexpr double kCellWidth = 0.025;
constexpr double kPhysicalRadius = 0.125;
constexpr double kSeedCovarianceScale =
    kPhysicalRadius * kPhysicalRadius / (5.177536231884058 * kCellWidth * kCellWidth);

std::vector<LiquidParticleState3D> DropSeed() {
  std::vector<LiquidParticleState3D> particles;
  const double stretch_xz = 1 / std::sqrt(1.03);
  for (int z = -5; z < 5; ++z) {
    for (int y = -5; y < 5; ++y) {
      for (int x = -5; x < 5; ++x) {
        const double x_offset = x + 0.5;
        const double y_offset = y + 0.5;
        const double z_offset = z + 0.5;
        if (x_offset * x_offset + y_offset * y_offset + z_offset * z_offset < 25) {
          particles.push_back({
              .id = particles.size() + 1,
              .world_position =
                  {
                      kCellWidth * x_offset * stretch_xz,
                      kCellWidth * y_offset * 1.03,
                      kCellWidth * z_offset * stretch_xz,
                  },
              .mass = 0.01482107041435402,
              .reference_volume = 1.4821070414354021e-05,
          });
        }
      }
    }
  }
  return particles;
}

absl::StatusOr<LiquidState3D> DropState() {
  LiquidState3D seed;
  seed.geometry = {{0, 0, 0}, kCellWidth};
  const auto particles = DropSeed();
  auto records = LiquidRecords3D<LiquidParticleState3D>::Create(particles.size());
  if (!records.ok()) {
    return records.status();
  }
  if (const auto status = records->Resize(particles.size()); !status.ok()) {
    return status;
  }
  std::ranges::copy(particles, records->values().begin());
  seed.particles = std::move(*records);
  seed.next_particle_id = particles.size() + 1;
  return seed;
}

TEST(IsolatedDrop3D, AuthoredSeedRetainsWetCentersCornersAndCurvature) {
  const auto particles = DropSeed();
  ASSERT_EQ(particles.size(), 552);
  const auto fit = PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale);
  ASSERT_TRUE(fit.ok()) << fit.status();

  int wet_centers = 0;
  int wet_corners = 0;
  int air_links = 0;
  double minimum_curvature = std::numeric_limits<double>::infinity();
  double maximum_curvature = 0;
  constexpr std::array<Vec3, 6> kNeighbors{
      Vec3{-kCellWidth, 0, 0},
      Vec3{kCellWidth, 0, 0},
      Vec3{0, -kCellWidth, 0},
      Vec3{0, kCellWidth, 0},
      Vec3{0, 0, -kCellWidth},
      Vec3{0, 0, kCellWidth},
  };
  for (int z = -7; z <= 7; ++z) {
    for (int y = -7; y <= 7; ++y) {
      for (int x = -7; x <= 7; ++x) {
        const auto corner = fit->QuadraticGridLevelSet({kCellWidth * x, kCellWidth * y, kCellWidth * z});
        ASSERT_TRUE(corner.ok()) << corner.status();
        wet_corners += *corner < 0;
        const Vec3 center{kCellWidth * (x + 0.5), kCellWidth * (y + 0.5), kCellWidth * (z + 0.5)};
        const auto level = fit->QuadraticGridLevelSet(center);
        ASSERT_TRUE(level.ok()) << level.status();
        if (*level >= 0) {
          continue;
        }
        ++wet_centers;
        for (const Vec3 offset : kNeighbors) {
          const Vec3 neighbor{center.x + offset.x, center.y + offset.y, center.z + offset.z};
          const auto neighbor_level = fit->QuadraticGridLevelSet(neighbor);
          ASSERT_TRUE(neighbor_level.ok()) << neighbor_level.status();
          if (*neighbor_level >= 0) {
            const auto curvature = fit->AirLinkCurvature(center, neighbor);
            ASSERT_TRUE(curvature.ok()) << curvature.status();
            ++air_links;
            minimum_curvature = std::min(minimum_curvature, *curvature);
            maximum_curvature = std::max(maximum_curvature, *curvature);
          }
        }
      }
    }
  }
  EXPECT_EQ(wet_centers, 528);
  EXPECT_EQ(wet_corners, 511);
  EXPECT_EQ(air_links, 472);
  EXPECT_NEAR(minimum_curvature, 15.561424751112174, 5e-10);
  EXPECT_NEAR(maximum_curvature, 16.94204543523493, 5e-10);
}

TEST(IsolatedDrop3D, UsesMassAndFixedSeedScaleForRotatedDrop) {
  auto particles = DropSeed();
  const auto transform = [](Vec3 point) {
    return Vec3{0.02 + 0.6 * point.x - 0.8 * point.y, 0.03 + 0.8 * point.x + 0.6 * point.y, -0.04 + point.z};
  };
  for (auto& particle : particles) {
    particle.volume_ratio = particle.world_position.y > 0 ? 1.2 : 0.9;
    particle.world_position = transform(particle.world_position);
  }
  const auto fit = PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale);
  ASSERT_TRUE(fit.ok()) << fit.status();
  const auto pole = fit->QuadraticGridLevelSet(transform({0, 0.12875, 0}));
  ASSERT_TRUE(pole.ok()) << pole.status();
  EXPECT_NEAR(*pole, 0, 2e-15);
  const auto curvature = fit->AirLinkCurvature(transform({0, 0.12, 0}), transform({0, 0.14, 0}));
  ASSERT_TRUE(curvature.ok()) << curvature.status();
  EXPECT_NEAR(*curvature, 16.9744, 5e-10);
}

TEST(IsolatedDrop3D, ParticleDistanceDistinguishesTheBetaBandAndMedialAxis) {
  const auto particles = DropSeed();
  const auto fit = PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale);
  ASSERT_TRUE(fit.ok()) << fit.status();
  const auto center_distance = fit->ParticleSignedDistance({0, 0, 0});
  const auto inner_distance = fit->ParticleSignedDistance({0, 0.103, 0});
  const auto outer_distance = fit->ParticleSignedDistance({0, 0.14, 0});
  const auto inner_level = fit->QuadraticGridLevelSet({0, 0.103, 0});
  ASSERT_TRUE(center_distance.ok()) << center_distance.status();
  ASSERT_TRUE(inner_distance.ok()) << inner_distance.status();
  ASSERT_TRUE(outer_distance.ok()) << outer_distance.status();
  ASSERT_TRUE(inner_level.ok()) << inner_level.status();
  EXPECT_NEAR(*center_distance, -0.12316615977053665, 2e-15);
  EXPECT_NEAR(*inner_distance, -0.02575, 2e-15);
  EXPECT_NEAR(*outer_distance, 0.01125, 2e-15);
  EXPECT_LT(*inner_distance, -kCellWidth);
  EXPECT_GE(*inner_level, -kCellWidth);
  EXPECT_EQ(fit->AirLinkCurvature({0, 0, 0}, {0, 0.1, 0}).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(IsolatedDrop3D, RejectsMalformedAndDisconnectedInputs) {
  auto particles = DropSeed();
  particles.back().mass = 0;
  EXPECT_EQ(
      PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  particles = DropSeed();
  particles.back().world_position = particles.front().world_position;
  EXPECT_EQ(
      PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  particles = DropSeed();
  particles.back().world_position.x += 1;
  EXPECT_EQ(
      PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(IsolatedDrop3D, RejectsConnectedPlanarCovariance) {
  const std::array particles{
      LiquidParticleState3D{.world_position = {0, 0, 0}},
      LiquidParticleState3D{.world_position = {kCellWidth, 0, 0}},
      LiquidParticleState3D{.world_position = {0, kCellWidth, 0}},
      LiquidParticleState3D{.world_position = {kCellWidth, kCellWidth, 0}},
  };
  EXPECT_EQ(
      PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(IsolatedDrop3D, ResolvedSceneActivatesPressureAndReplaysFromCheckpoint) {
  const auto seed = DropState();
  ASSERT_TRUE(seed.ok()) << seed.status();

  LiquidResolvedSceneView3D authored;
  authored.geometry = seed->geometry;
  authored.scene.field = LiquidIsolatedCovarianceDrop3D{kPhysicalRadius, kSeedCovarianceScale};
  authored.scene.surface_tension = 1;
  authored.scene.world_acceleration = {};
  const auto encoded = EncodeLiquidResolvedScene3D(authored);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  const auto repeated = EncodeLiquidResolvedScene3D(decoded->View());
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  EXPECT_TRUE(std::ranges::equal(encoded->values(), repeated->values()));
  const auto* field = std::get_if<LiquidIsolatedCovarianceDrop3D>(&decoded->scene.field);
  ASSERT_NE(field, nullptr);
  EXPECT_DOUBLE_EQ(field->physical_radius, kPhysicalRadius);
  EXPECT_DOUBLE_EQ(field->seed_covariance_scale, kSeedCovarianceScale);

  auto options = decoded->options;
  options.capacities = {1024, 128, 4096, 24576, 0};
  auto direct = LiquidSubstepScheduler3D::Create(*seed, decoded->scene, options);
  ASSERT_TRUE(direct.ok()) << direct.status();
  constexpr double kStepDuration = 1.0 / 120;
  ASSERT_TRUE(direct->BeginFrame(kStepDuration).ok());
  const auto first = direct->Attempt(kStepDuration);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_EQ(first->disposition, LiquidAttemptDisposition3D::kAccepted) << direct->diagnostics().failure_code;
  const auto& diagnostics = direct->diagnostics();
  ASSERT_TRUE(diagnostics.accepted_measurement);
  const auto& measurement = *diagnostics.accepted_measurement;
  ASSERT_TRUE(measurement.step_start_capillary_bound_seconds);
  EXPECT_DOUBLE_EQ(*measurement.step_start_capillary_bound_seconds, 0.035261848971734774);
  EXPECT_NEAR(measurement.step_start_pressure_cell_volume_m3, 0.0074696016768942049, 2e-15);
  EXPECT_EQ(diagnostics.lower_theta_regularization_count, 48);
  ASSERT_EQ(direct->committed().cells.size(), 528);
  const auto [min_pressure, max_pressure] =
      std::ranges::minmax_element(direct->committed().cells.values(), {}, &LiquidCellState3D::previous_pressure);
  EXPECT_NEAR(min_pressure->previous_pressure, 15.588011741638184, 2e-6);
  EXPECT_NEAR(max_pressure->previous_pressure, 16.720998764038086, 2e-6);

  ASSERT_TRUE(direct->FinishFrame().ok());
  const auto checkpoint = EncodeLiquidState3D(direct->committed());
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  const auto restored = DecodeLiquidState3D(checkpoint->values());
  ASSERT_TRUE(restored.ok()) << restored.status();
  auto replay = LiquidSubstepScheduler3D::Create(*restored, decoded->scene, options);
  ASSERT_TRUE(replay.ok()) << replay.status();
  ASSERT_TRUE(direct->BeginFrame(2 * kStepDuration).ok());
  ASSERT_TRUE(replay->BeginFrame(2 * kStepDuration).ok());
  const auto direct_next = direct->Attempt(kStepDuration);
  ASSERT_TRUE(direct_next.ok()) << direct_next.status();
  EXPECT_EQ(direct_next->disposition, LiquidAttemptDisposition3D::kAccepted);
  const auto replay_next = replay->Attempt(kStepDuration);
  ASSERT_TRUE(replay_next.ok()) << replay_next.status();
  EXPECT_EQ(replay_next->disposition, LiquidAttemptDisposition3D::kAccepted);
  const auto direct_bytes = EncodeLiquidState3D(direct->committed());
  const auto replay_bytes = EncodeLiquidState3D(replay->committed());
  ASSERT_TRUE(direct_bytes.ok()) << direct_bytes.status();
  ASSERT_TRUE(replay_bytes.ok()) << replay_bytes.status();
  EXPECT_TRUE(std::ranges::equal(direct_bytes->values(), replay_bytes->values()));
}

TEST(IsolatedDrop3D, ProjectedChangeRetainsArchivedDropAtFrameThree) {
  const auto seed = DropState();
  ASSERT_TRUE(seed.ok()) << seed.status();

  LiquidSceneInputs3D scene;
  scene.field = LiquidIsolatedCovarianceDrop3D{kPhysicalRadius, kSeedCovarianceScale};
  scene.surface_tension = 1;
  scene.world_acceleration = {};
  LiquidStepOptions3D options;
  options.capacities = {1024, 128, 4096, 24576, 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(*seed, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  for (int frame = 1; frame <= 3; ++frame) {
    const auto result = scheduler->AdvanceFrame(frame / 24.0);
    ASSERT_TRUE(result.ok()) << result;
  }
  const auto& state = scheduler->committed();
  ASSERT_EQ(state.clock.accepted_step_ordinal, 15);
  ASSERT_EQ(state.particles.size(), 552);
  Vec3 centroid{};
  double volume = 0;
  for (const auto& particle : state.particles.values()) {
    const double material_volume = particle.reference_volume * particle.volume_ratio;
    centroid.x += material_volume * particle.world_position.x;
    centroid.y += material_volume * particle.world_position.y;
    centroid.z += material_volume * particle.world_position.z;
    volume += material_volume;
  }
  centroid = {centroid.x / volume, centroid.y / volume, centroid.z / volume};
  double quadrupole = 0;
  for (const auto& particle : state.particles.values()) {
    const double material_volume = particle.reference_volume * particle.volume_ratio;
    const Vec3 position = particle.world_position;
    const double x = position.x - centroid.x;
    const double y = position.y - centroid.y;
    const double z = position.z - centroid.z;
    quadrupole += material_volume * (y * y - (x * x + z * z) / 2);
  }
  EXPECT_NEAR(quadrupole / volume, 0.00028263630776289309, 3e-12);
}

TEST(IsolatedDrop3D, GroupedProjectedChangeMatchesScalarAndReplaysRejectedRun) {
  const auto seed = DropState();
  ASSERT_TRUE(seed.ok()) << seed.status();
  LiquidSceneInputs3D scene;
  scene.field = LiquidIsolatedCovarianceDrop3D{kPhysicalRadius, kSeedCovarianceScale};
  scene.surface_tension = 1;
  scene.world_acceleration = {};
  LiquidStepOptions3D options;
  options.capacities = {1024, 128, 4096, 24576, 0};
  auto scalar = LiquidSubstepScheduler3D::Create(*seed, scene, options);
  ASSERT_TRUE(scalar.ok()) << scalar.status();
  ASSERT_TRUE(scalar->AdvanceFrame(1.0 / 120).ok());
  const auto scalar_bytes = EncodeLiquidState3D(scalar->committed());
  ASSERT_TRUE(scalar_bytes.ok()) << scalar_bytes.status();

  enum class GatherHistory {
    kRestrictedOld,
    kFullOld,
    kProjected,
    kOrdinary
  };
  struct Audit {
    std::vector<GatherHistory> histories;
    std::vector<std::size_t> run_sizes;
    std::size_t scalar_calls = 0;
    bool reject_projected = false;
    bool rejected = false;
  };
  for (const bool reject_projected : {false, true}) {
    SCOPED_TRACE(reject_projected);
    Audit audit{.reject_projected = reject_projected};
    auto backend = CpuLiquidSubstepBackend3D();
    backend.context = &audit;
    backend.gather =
        [](void* context, const LiquidBoundaryGatherViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
          ++static_cast<Audit*>(context)->scalar_calls;
          return GatherLiquidBoundaryParticles3D(inputs, workspace);
        };
    backend.gather_run = [](void* context,
                            const LiquidBoundaryGatherViewInputs3D& inputs,
                            LiquidBoundaryWorkspace3D& workspace) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
      auto& audit = *static_cast<Audit*>(context);
      const bool reads_old = inputs.new_history.face_velocities.data() == inputs.old_history.face_velocities.data();
      const bool full_old = inputs.new_history.validity.data() == inputs.old_history.validity.data();
      const bool continues_change = !audit.histories.empty() && audit.histories.back() == GatherHistory::kFullOld;
      const auto history = reads_old ? (full_old ? GatherHistory::kFullOld : GatherHistory::kRestrictedOld)
                                     : (continues_change ? GatherHistory::kProjected : GatherHistory::kOrdinary);
      audit.histories.push_back(history);
      audit.run_sizes.push_back(inputs.particles.size());
      if (history == GatherHistory::kProjected && audit.reject_projected && !audit.rejected) {
        audit.rejected = true;
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      return GatherLiquidBoundaryParticles3D(inputs, workspace);
    };
    auto grouped = LiquidSubstepScheduler3D::Create(*seed, scene, options, backend);
    ASSERT_TRUE(grouped.ok()) << grouped.status();
    const auto status = grouped->AdvanceFrame(1.0 / 120);
    ASSERT_TRUE(status.ok()) << status;
    const auto restricted = std::ranges::find(audit.histories, GatherHistory::kRestrictedOld);
    ASSERT_NE(restricted, audit.histories.end());
    ASSERT_GE(audit.histories.end() - restricted, 3);
    EXPECT_EQ(restricted[1], GatherHistory::kFullOld);
    EXPECT_EQ(restricted[2], GatherHistory::kProjected);
    const auto offset = static_cast<std::size_t>(restricted - audit.histories.begin());
    EXPECT_GE(audit.run_sizes[offset], 2);
    EXPECT_EQ(audit.run_sizes[offset], audit.run_sizes[offset + 1]);
    EXPECT_EQ(audit.run_sizes[offset], audit.run_sizes[offset + 2]);
    EXPECT_EQ(audit.rejected, reject_projected);
    if (reject_projected) {
      EXPECT_GT(audit.scalar_calls, 0);
    } else {
      EXPECT_LT(audit.scalar_calls, seed->particles.size());
    }
    const auto grouped_bytes = EncodeLiquidState3D(grouped->committed());
    ASSERT_TRUE(grouped_bytes.ok()) << grouped_bytes.status();
    EXPECT_TRUE(std::ranges::equal(scalar_bytes->values(), grouped_bytes->values()));
  }
}
TEST(IsolatedDrop3D, PressureLiftsUseTheSignedLatticeAndRejectSubnormalProducts) {
  // This cube's exact, scaled covariance fits a sphere of radius 1/8 m.
  std::array<LiquidParticleState3D, 8> particles{};
  std::size_t ordinal = 0;
  for (const double z : {-0.0078125, 0.0078125}) {
    for (const double y : {-0.0078125, 0.0078125}) {
      for (const double x : {0.1171875, 0.1328125}) {
        particles[ordinal++].world_position = {x, y, z};
      }
    }
  }
  const auto fit = PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, 256);
  ASSERT_TRUE(fit.ok()) << fit.status();
  Staging staging;
  staging.isolated_drop = *fit;
  auto faces = LiquidRecords3D<MacPressureFace3D>::Create(1);
  ASSERT_TRUE(faces.ok()) << faces.status();
  staging.pressure_face_scratch.emplace(std::move(*faces));

  constexpr int kHead = -98;
  const double origin_x = -kCellWidth * (kHead - 0.5);
  const UniformMacGeometry3D translated{{origin_x, -0.5 * kCellWidth, -0.5 * kCellWidth}, kCellWidth};
  ASSERT_EQ(origin_x + kCellWidth * (kHead - 0.5), 0);
  ASSERT_GT((origin_x + kCellWidth * (kHead + 0.5)) - kCellWidth, 1e-17);
  MacPressureFace3D face{
      .face_index = {FaceNormalAxis::kX, {kHead, 0, 0}},
      .pressure_link = DirichletMacPressureLink3D{0},
  };
  LiquidPressureProblemView3D pressure{.faces = std::span(&face, 1)};
  const auto status = SetIsolatedDropPressureLifts(translated, 1, 0, staging, pressure);
  EXPECT_TRUE(status.ok()) << status;
  if (status.ok()) {
    EXPECT_NEAR(std::get<DirichletMacPressureLink3D>(pressure.faces[0].pressure_link).exterior_pressure, 16, 1e-9);
  }

  face.face_index.lattice_index.x = 1;
  pressure.faces = std::span(&face, 1);
  const UniformMacGeometry3D dyadic{{-0.015625, -0.015625, -0.015625}, 0.03125};
  const auto unrepresentable =
      SetIsolatedDropPressureLifts(dyadic, std::numeric_limits<double>::denorm_min(), 1, staging, pressure);
  EXPECT_EQ(unrepresentable.code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(pressure.faces.data(), &face);
}

TEST(IsolatedDrop3D, OverflowingProjectedChangeRejectsGatherBeforePublishing) {
  const auto seed = DropState();
  ASSERT_TRUE(seed.ok()) << seed.status();
  const auto before = EncodeLiquidState3D(*seed);
  ASSERT_TRUE(before.ok()) << before.status();
  LiquidSceneInputs3D scene;
  scene.field = LiquidIsolatedCovarianceDrop3D{kPhysicalRadius, kSeedCovarianceScale};
  scene.surface_tension = 1;
  scene.world_acceleration = {};
  LiquidStepOptions3D options;
  options.capacities = {1024, 128, 4096, 24576, 0};
  struct FaultGather {
    LiquidSubstepBackend3D cpu = CpuLiquidSubstepBackend3D();
    LiquidBoundaryParticleGather3D result{};
    std::size_t old_gathers = 0;
  } fault;
  auto backend = fault.cpu;
  backend.context = &fault;
  backend.gather = [](void* context,
                      const LiquidBoundaryGatherViewInputs3D& inputs,
                      LiquidBoundaryWorkspace3D& workspace) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
    auto& fault = *static_cast<FaultGather*>(context);
    auto gathered = fault.cpu.gather(nullptr, inputs, workspace);
    if (!gathered.ok()) {
      return gathered.status();
    }
    fault.result = gathered->particles[0];
    const bool reads_old = inputs.new_history.face_velocities.data() == inputs.old_history.face_velocities.data();
    fault.old_gathers += reads_old;
    const double maximum = std::numeric_limits<double>::max();
    fault.result.new_pic_velocity.x = reads_old ? -maximum : maximum;
    gathered->particles = std::span(&fault.result, 1);
    return gathered;
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(*seed, scene, options, backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(1.0 / 24).ok());
  const auto outcome = scheduler->Attempt(1.0 / 120);
  ASSERT_TRUE(outcome.ok()) << outcome.status();
  EXPECT_EQ(outcome->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_GT(fault.old_gathers, 0);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kGather);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kOutOfRange);
  const auto after = EncodeLiquidState3D(scheduler->committed());
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_TRUE(std::ranges::equal(before->values(), after->values()));
}

}  // namespace
}  // namespace fluidity::liquid_step_internal
