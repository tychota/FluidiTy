#include <cmath>
#include <set>

#include "fluidity/host/surface_reconstruction_3d.h"
#include "gtest/gtest.h"
#include "liquid_cavity_scheduler_cases.h"

namespace fluidity {
namespace {
LiquidState3D Seed() {
  LiquidState3D state;
  state.particles = *LiquidRecords3D<LiquidParticleState3D>::Create(1);
  EXPECT_TRUE(state.particles.Resize(1).ok());
  state.particles[0].id = 1;
  state.particles[0].world_position = {2.5, 2.5, 2.5};
  state.next_particle_id = 2;
  return state;
}
TEST(SurfaceReconstruction3D, EvaluatesAndCachesMixedSurfaceFractions) {
  auto seed = Seed();
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(2);
  ASSERT_TRUE(particles.ok()) << particles.status();
  ASSERT_TRUE(particles->Resize(2).ok());
  (*particles)[0] = seed.particles[0];
  (*particles)[0].reference_volume = 0.25;
  (*particles)[0].phase_id = 4;
  (*particles)[0].constituent_count = 1;
  (*particles)[0].constituents[0] = {8, 1.0};
  (*particles)[1] = (*particles)[0];
  (*particles)[1].id = 2;
  (*particles)[1].reference_volume = 0.75;
  (*particles)[1].constituents[0] = {14, 1.0};
  seed.particles = std::move(*particles);
  seed.next_particle_id = 3;
  const auto motion_bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(motion_bytes.ok()) << motion_bytes.status();
  const auto source = DecodeMotionFrame3D(motion_bytes->values());
  ASSERT_TRUE(source.ok()) << source.status();
  constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{{8, 4, 1000.0}, {14, 4, 1020.0}}};
  LiquidSceneInputs3D scene;
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 1.1};
  scene.constituent_materials = kMaterials;
  LiquidStepOptions3D step;
  step.capacities = {16, 64, 128, 12288, 1};
  const SurfaceReconstructionOptions3D options{.spacing_m = 0.25, .sigma_m = 0.25};
  const auto reconstructed = ReconstructSurfaceFrame3D(*source, scene, step, nullptr, options);
  ASSERT_TRUE(reconstructed.ok()) << reconstructed.status();
  const auto encoded = EncodeSurfaceFrame3D(reconstructed->frame);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeSurfaceFrame3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  ASSERT_FALSE(decoded->mesh.vertices.empty());
  for (const auto& vertex : decoded->mesh.vertices) {
    EXPECT_EQ(vertex.phase_id, 4);
    ASSERT_EQ(vertex.constituent_count, 2);
    EXPECT_EQ(vertex.constituents[0].substance_id, 8);
    EXPECT_DOUBLE_EQ(vertex.constituents[0].volume_fraction, 0.25);
    EXPECT_EQ(vertex.constituents[1].substance_id, 14);
    EXPECT_DOUBLE_EQ(vertex.constituents[1].volume_fraction, 0.75);
  }
}

TEST(SurfaceReconstruction3D, PublishesSharedSheetFromTwoParticlePhases) {
  auto seed = Seed();
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(2);
  ASSERT_TRUE(particles.ok()) << particles.status();
  ASSERT_TRUE(particles->Resize(2).ok());
  (*particles)[0] = seed.particles[0];
  (*particles)[0].world_velocity.x = 0.24;
  (*particles)[0].mass = 1000;
  (*particles)[0].phase_id = 4;
  (*particles)[0].constituent_count = 1;
  (*particles)[0].constituents[0] = {8, 1.0};
  (*particles)[1] = (*particles)[0];
  (*particles)[1].id = 2;
  (*particles)[1].world_position = {2.92, 2.64, 2.57};
  (*particles)[1].mass = 820;
  (*particles)[1].phase_id = 5;
  (*particles)[1].constituents[0] = {25, 1.0};
  seed.particles = std::move(*particles);
  seed.next_particle_id = 3;
  const auto motion_bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(motion_bytes.ok()) << motion_bytes.status();
  const auto source = DecodeMotionFrame3D(motion_bytes->values());
  ASSERT_TRUE(source.ok()) << source.status();
  constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{{8, 4, 1000.0}, {25, 5, 820.0}}};
  constexpr std::array<LiquidPhasePair3D, 1> kPairs{{{4, 5, 0.0}}};
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{1.1};
  scene.constituent_materials = kMaterials;
  scene.phase_pairs = kPairs;
  scene.world_acceleration = {};
  LiquidStepOptions3D step;
  step.capacities = {16, 64, 128, 12288, 1};
  const SurfaceReconstructionOptions3D options{.spacing_m = 0.25, .sigma_m = 0.25};
  const auto reconstructed = ReconstructSurfaceFrame3D(*source, scene, step, nullptr, options);
  ASSERT_TRUE(reconstructed.ok()) << reconstructed.status();
  ASSERT_TRUE(reconstructed->frame.phase_interface);
  const auto& sheet = *reconstructed->frame.phase_interface;
  EXPECT_EQ(sheet.phase_a, 4);
  EXPECT_EQ(sheet.phase_b, 5);
  ASSERT_FALSE(sheet.side_compositions.empty());
  ASSERT_EQ(sheet.side_compositions.size() % 2, 0);
  bool sampled_interior = false;
  for (std::size_t record = 0; record < sheet.side_compositions.size(); record += 2) {
    const auto& water = sheet.side_compositions[record];
    const auto& oil = sheet.side_compositions[record + 1];
    EXPECT_EQ(water.vertex_ordinal, oil.vertex_ordinal);
    sampled_interior |= water.vertex_ordinal >= reconstructed->frame.mesh.vertices.size();
    ASSERT_EQ(water.constituent_count, 1);
    ASSERT_EQ(oil.constituent_count, 1);
    EXPECT_EQ(water.phase_id, 4);
    EXPECT_EQ(oil.phase_id, 5);
    EXPECT_EQ(water.constituents[0], (LiquidConstituentFraction3D{8, 1}));
    EXPECT_EQ(oil.constituents[0], (LiquidConstituentFraction3D{25, 1}));
  }
  EXPECT_TRUE(sampled_interior);
  const auto encoded_sheet = EncodeSurfaceFrame3D(reconstructed->frame);
  ASSERT_TRUE(encoded_sheet.ok()) << encoded_sheet.status();
  const auto decoded_sheet = DecodeSurfaceFrame3D(encoded_sheet->values());
  ASSERT_TRUE(decoded_sheet.ok()) << decoded_sheet.status();
  ASSERT_TRUE(decoded_sheet->phase_interface);
  EXPECT_EQ(decoded_sheet->phase_interface->side_compositions.size(), sheet.side_compositions.size());
  EXPECT_FALSE(sheet.triangles.empty());
  EXPECT_NE(std::ranges::find(sheet.exterior_sides, 0), sheet.exterior_sides.end());
  EXPECT_NE(std::ranges::find(sheet.exterior_sides, 1), sheet.exterior_sides.end());
  EXPECT_NEAR(reconstructed->corrected_metrics.signed_volume, 2.0, 0.01);
  auto scheduler = LiquidSubstepScheduler3D::Create(seed, scene, step);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  auto capture = MotionCapture3D::Create(scheduler->committed(), 1.0 / 24, 16);
  ASSERT_TRUE(capture.ok()) << capture.status();
  const LiquidProgressObserver3D observer{
      .context = &*capture,
      .observe =
          [](void* context, const LiquidState3D& state, const LiquidStepDiagnostics3D&, bool rollback) {
            return static_cast<MotionCapture3D*>(context)->Observe(state, rollback);
          },
  };
  const auto advanced = scheduler->AdvanceFrame(1.0 / 24, observer);
  ASSERT_TRUE(advanced.ok()) << advanced;
  const auto captured = capture->Finish(scheduler->committed());
  ASSERT_TRUE(captured.ok()) << captured.status();
  const auto moving = DecodeMotionFrame3D(captured->values());
  ASSERT_TRUE(moving.ok()) << moving.status();
  const auto next = ReconstructSurfaceFrame3D(*moving, scene, step, &reconstructed->frame, options);
  ASSERT_TRUE(next.ok()) << next.status();
  ASSERT_TRUE(next->frame.phase_interface);
  EXPECT_EQ(next->frame.phase_interface->triangles, sheet.triangles);
  EXPECT_EQ(next->retained_ids, reconstructed->frame.mesh.vertices.size());
  for (std::size_t index = 0; index < next->frame.mesh.vertices.size(); ++index) {
    const auto& vertex = next->frame.mesh.vertices[index];
    EXPECT_EQ(vertex.id, reconstructed->frame.mesh.vertices[index].id);
    EXPECT_NEAR(vertex.geometric_velocity.x, 0.24, 1e-8);
  }

  seed.particles[1].world_velocity.x += 0.75;
  auto deforming = LiquidSubstepScheduler3D::Create(seed, scene, step);
  ASSERT_TRUE(deforming.ok()) << deforming.status();
  auto deformation_capture = MotionCapture3D::Create(deforming->committed(), 1.0 / 24, 16);
  ASSERT_TRUE(deformation_capture.ok()) << deformation_capture.status();
  auto deformation_observer = observer;
  deformation_observer.context = &*deformation_capture;
  const auto deformed_step = deforming->AdvanceFrame(1.0 / 24, deformation_observer);
  ASSERT_TRUE(deformed_step.ok()) << deformed_step;
  const auto deformed_bytes = deformation_capture->Finish(deforming->committed());
  ASSERT_TRUE(deformed_bytes.ok()) << deformed_bytes.status();
  const auto deformed_source = DecodeMotionFrame3D(deformed_bytes->values());
  ASSERT_TRUE(deformed_source.ok()) << deformed_source.status();
  const auto deformed = ReconstructSurfaceFrame3D(*deformed_source, scene, step, &reconstructed->frame, options);
  ASSERT_TRUE(deformed.ok()) << deformed.status();
  ASSERT_TRUE(deformed->frame.phase_interface);
  EXPECT_NE(deformed->frame.mesh.triangles, reconstructed->frame.mesh.triangles);
  EXPECT_GT(deformed->retained_ids, 0);
  EXPECT_LT(deformed->retained_ids, deformed->frame.mesh.vertices.size());
  std::set<std::uint64_t> ids;
  for (const auto& vertex : deformed->frame.mesh.vertices) {
    EXPECT_TRUE(ids.insert(vertex.id).second);
    if (vertex.id >= reconstructed->frame.mesh.next_id) {
      EXPECT_DOUBLE_EQ(vertex.age_seconds, 0);
      EXPECT_DOUBLE_EQ(vertex.geometric_velocity.x, 0);
    }
  }
  const auto encoded_deformation = EncodeSurfaceFrame3D(deformed->frame);
  ASSERT_TRUE(encoded_deformation.ok()) << encoded_deformation.status();
  const auto replayed = ReconstructSurfaceFrame3D(*deformed_source, scene, step, &reconstructed->frame, options);
  ASSERT_TRUE(replayed.ok()) << replayed.status();
  const auto replayed_bytes = EncodeSurfaceFrame3D(replayed->frame);
  ASSERT_TRUE(replayed_bytes.ok()) << replayed_bytes.status();
  EXPECT_TRUE(std::ranges::equal(replayed_bytes->values(), encoded_deformation->values()));
}
TEST(SurfaceReconstruction3D, DeepTankBandDoesNotProduceAnInternalShell) {
  constexpr int kWidth = 8;
  constexpr std::size_t kBandCount = 296;
  constexpr double kTankVolume = 512.0;
  LiquidState3D initial;
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(kBandCount);
  ASSERT_TRUE(particles.ok()) << particles.status();
  ASSERT_TRUE(particles->Resize(kBandCount).ok());
  std::size_t count = 0;
  for (int z = 0; z < kWidth; ++z) {
    for (int y = 0; y < kWidth; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        if (x == 0 || y == 0 || z == 0 || x == kWidth - 1 || y == kWidth - 1 || z == kWidth - 1) {
          ASSERT_LT(count, kBandCount);
          (*particles)[count] = {
              .id = count + 1,
              .world_position = {x + 0.5, y + 0.5, z + 0.5},
          };
          ++count;
        }
      }
    }
  }
  ASSERT_EQ(count, kBandCount);
  initial.next_particle_id = count + 1;
  initial.particles = std::move(*particles);
  const auto step_options = cavity_test::ShellOptions(kBandCount);
  LiquidSceneInputs3D scene;
  scene.field = LiquidHorizontalPlaneSurface3D{.world_surface_height_y = kWidth};
  scene.obstacle = AxisAlignedLiquidTank3D{
      .start_low_x = 0.0,
      .high_x = kWidth,
      .floor_y = 0.0,
      .low_z = 0.0,
      .high_z = kWidth,
      .piston_velocity_x = 0.0,
  };
  scene.world_acceleration = {};
  scene.cavity = LiquidSceneCavity3D{.atmospheric_absolute_pressure = 100000.0};
  const auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, step_options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  EXPECT_EQ(scheduler->committed().next_cavity_id, 1);
  const auto encoded = EncodeMotionSeed3D(scheduler->committed());
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto motion = DecodeMotionFrame3D(encoded->values());
  ASSERT_TRUE(motion.ok()) << motion.status();
  const SurfaceReconstructionOptions3D options{.spacing_m = 0.5, .sigma_m = 0.5};
  const auto reconstructed = ReconstructSurfaceFrame3D(*motion, scene, step_options, nullptr, options);
  ASSERT_TRUE(reconstructed.ok()) << reconstructed.status();
  EXPECT_EQ(reconstructed->raw_metrics.positive_volume_components, 1);
  EXPECT_EQ(reconstructed->raw_metrics.negative_volume_components, 0);
  EXPECT_EQ(reconstructed->corrected_metrics.positive_volume_components, 1);
  EXPECT_EQ(reconstructed->corrected_metrics.negative_volume_components, 0);
  ASSERT_EQ(reconstructed->frame.volumes.size(), 1);
  EXPECT_DOUBLE_EQ(reconstructed->frame.volumes.front().material_m3, kTankVolume);
  EXPECT_NEAR(reconstructed->corrected_metrics.signed_volume, kTankVolume, 0.005 * kTankVolume);
}

TEST(SurfaceReconstruction3D, CachedCavityRetainsAnInwardBoundary) {
  const auto shell = cavity_test::ParticleShell();
  ASSERT_TRUE(shell.ok()) << shell.status();
  const auto step_options = cavity_test::ShellOptions(shell->particles.size());
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{.radius = 0.9};
  scene.world_acceleration = {};
  scene.cavity = LiquidSceneCavity3D{.atmospheric_absolute_pressure = 100000.0};
  auto scheduler = LiquidSubstepScheduler3D::Create(*shell, scene, step_options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_EQ(scheduler->committed().clock, LiquidClock3D{});
  ASSERT_EQ(scheduler->committed().next_cavity_id, 2);
  const auto bytes = EncodeMotionSeed3D(scheduler->committed());
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  const auto motion = DecodeMotionFrame3D(bytes->values());
  ASSERT_TRUE(motion.ok()) << motion.status();
  const SurfaceReconstructionOptions3D options{.spacing_m = 1.0, .sigma_m = 0.75};
  const auto reconstructed = ReconstructSurfaceFrame3D(*motion, scene, step_options, nullptr, options);
  ASSERT_TRUE(reconstructed.ok()) << reconstructed.status();
  EXPECT_EQ(reconstructed->raw_metrics.positive_volume_components, 1);
  EXPECT_EQ(reconstructed->raw_metrics.negative_volume_components, 1);
  EXPECT_EQ(reconstructed->corrected_metrics.positive_volume_components, 1);
  EXPECT_EQ(reconstructed->corrected_metrics.negative_volume_components, 1);
  ASSERT_EQ(reconstructed->frame.volumes.size(), 1);
  EXPECT_NEAR(
      reconstructed->frame.volumes[0].corrected_m3,
      static_cast<double>(shell->particles.size()),
      0.005 * static_cast<double>(shell->particles.size())
  );
}

TEST(SurfaceReconstruction3D, ClipsSeedAgainstInclinedStationaryBeach) {
  auto seed = Seed();
  seed.particles[0].world_velocity.x = 2;
  const auto bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  const auto motion = DecodeMotionFrame3D(bytes->values());
  ASSERT_TRUE(motion.ok()) << motion.status();
  LiquidSceneInputs3D scene;
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 1.1};
  const TranslatingHalfspace3D beach{
      .unit_normal = {-0.6, 0.8, 0},
      .start_plane_coordinate = 0.125,
      .wall_velocity = {},
  };
  scene.obstacle = beach;
  scene.field = LiquidParticleSphereUnion3D{1.1};
  scene.world_acceleration = {};
  LiquidStepOptions3D step;
  step.maximum_step_duration = 0.03;
  step.capacities = {16, 64, 128, 12288, 1};
  const SurfaceReconstructionOptions3D options{.spacing_m = 0.125, .sigma_m = 0.25};
  const auto result = ReconstructSurfaceFrame3D(*motion, scene, step, nullptr, options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NEAR(result->corrected_metrics.signed_volume, 1, 0.005);
  EXPECT_EQ(result->corrected_metrics.boundary_edges, 0);
  EXPECT_EQ(result->corrected_metrics.self_intersections, 0);
  std::size_t contact_count = 0;
  for (const auto& vertex : result->frame.mesh.vertices) {
    const double clearance_m = beach.unit_normal.x * vertex.position.x + beach.unit_normal.y * vertex.position.y
        - beach.start_plane_coordinate;
    EXPECT_GE(clearance_m, 0);
    contact_count += vertex.contact;
  }
  EXPECT_GT(contact_count, 0);
  auto scheduler = LiquidSubstepScheduler3D::Create(seed, scene, step);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  auto capture = MotionCapture3D::Create(seed, 1.0 / 24, 1);
  ASSERT_TRUE(capture.ok()) << capture.status();
  const LiquidProgressObserver3D observer{
      .context = &*capture,
      .observe =
          [](void* context, const LiquidState3D& state, const LiquidStepDiagnostics3D&, bool rollback) {
            return static_cast<MotionCapture3D*>(context)->Observe(state, rollback);
          },
  };
  const auto advanced = scheduler->AdvanceFrame(1.0 / 24, observer);
  ASSERT_TRUE(advanced.ok()) << advanced;
  const auto captured = capture->Finish(scheduler->committed());
  ASSERT_TRUE(captured.ok()) << captured.status();
  const auto interval = DecodeMotionFrame3D(captured->values());
  ASSERT_TRUE(interval.ok()) << interval.status();
  const auto next = ReconstructSurfaceFrame3D(*interval, scene, step, &result->frame, options);
  ASSERT_TRUE(next.ok()) << next.status();
  EXPECT_TRUE(next->used_tracker);
  EXPECT_EQ(next->retained_ids, result->frame.mesh.vertices.size());
}

TEST(SurfaceReconstruction3D, ReconstructsAndEncodesMeasuredSeedThenTracksAcceptedFrame) {
  const auto seed = Seed();
  const auto encoded_seed = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(encoded_seed.ok());
  const auto motion_seed = DecodeMotionFrame3D(encoded_seed->values());
  ASSERT_TRUE(motion_seed.ok());
  LiquidSceneInputs3D scene;
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 1.1};
  scene.world_acceleration = {};
  LiquidStepOptions3D step;
  step.capacities = {16, 64, 128, 12288, 1};
  const SurfaceReconstructionOptions3D options{.spacing_m = 0.125, .sigma_m = 0.25};
  auto first = ReconstructSurfaceFrame3D(*motion_seed, scene, step, nullptr, options);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_FALSE(first->used_tracker);
  EXPECT_LE(std::abs(first->corrected_metrics.signed_volume - 1), 0.005);
  EXPECT_EQ(first->frame.volumes.size(), 1);
  auto bytes = EncodeSurfaceFrame3D(first->frame);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  auto checkpoint = DecodeSurfaceFrame3D(bytes->values());
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  auto scheduler = LiquidSubstepScheduler3D::Create(seed, scene, step);
  ASSERT_TRUE(scheduler.ok());
  for (std::uint64_t frame = 1; frame <= 24; ++frame) {
    SCOPED_TRACE(frame);
    auto capture = MotionCapture3D::Create(scheduler->committed(), static_cast<double>(frame) / 24, 16);
    ASSERT_TRUE(capture.ok());
    const LiquidProgressObserver3D observer{
        .context = &*capture,
        .observe =
            [](void* context, const LiquidState3D& state, const LiquidStepDiagnostics3D&, bool rollback) {
              return static_cast<MotionCapture3D*>(context)->Observe(state, rollback);
            },
    };
    ASSERT_TRUE(scheduler->AdvanceFrame(static_cast<double>(frame) / 24, observer).ok());
    auto captured = capture->Finish(scheduler->committed());
    ASSERT_TRUE(captured.ok());
    auto interval = DecodeMotionFrame3D(captured->values());
    ASSERT_TRUE(interval.ok());
    if (frame == 1) {
      ASSERT_GT(interval->samples.size(), 2);
      const auto clock = interval->samples[1].clock;
      interval->samples[1].clock.frame_ordinal = 1;
      interval->samples[1].clock.substep_ordinal = 0;
      EXPECT_EQ(
          ReconstructSurfaceFrame3D(*interval, scene, step, &*checkpoint, options).status().code(),
          absl::StatusCode::kInvalidArgument
      );
      interval->samples[1].clock = clock;
    }
    auto next = ReconstructSurfaceFrame3D(*interval, scene, step, &*checkpoint, options);
    ASSERT_TRUE(next.ok()) << next.status();
    EXPECT_TRUE(next->used_tracker);
    EXPECT_EQ(next->retained_ids, checkpoint->mesh.vertices.size());
    EXPECT_EQ(next->frame.mesh.triangles, checkpoint->mesh.triangles);
    EXPECT_EQ(next->frame.clock.frame_ordinal, frame);
    EXPECT_LE(std::abs(next->corrected_metrics.signed_volume - 1), 0.005);
    EXPECT_TRUE(EncodeSurfaceFrame3D(next->frame).ok());
    for (std::size_t index = 0; index < next->frame.mesh.vertices.size(); ++index) {
      const auto& vertex = next->frame.mesh.vertices[index];
      const auto old = checkpoint->mesh.vertices[index].position;
      EXPECT_EQ(vertex.id, checkpoint->mesh.vertices[index].id);
      EXPECT_EQ(vertex.position.x, old.x);
      EXPECT_EQ(vertex.position.y, old.y);
      EXPECT_EQ(vertex.position.z, old.z);
      EXPECT_NEAR(vertex.age_seconds, static_cast<double>(frame) / 24, 1e-12);
      EXPECT_NEAR(vertex.geometric_velocity.x, (vertex.position.x - old.x) * 24, 1e-12);
      EXPECT_NEAR(vertex.fluid_velocity.x, 0, 1e-12);
    }
    auto next_bytes = EncodeSurfaceFrame3D(next->frame);
    ASSERT_TRUE(next_bytes.ok());
    checkpoint = DecodeSurfaceFrame3D(next_bytes->values());
    ASSERT_TRUE(checkpoint.ok());
  }
}
TEST(SurfaceReconstruction3D, RejectsMissingSeedAndUnsupportedContactBeforePublishingAnything) {
  MotionFrame3D empty;
  EXPECT_EQ(ReconstructSurfaceFrame3D(empty, {}, {}, nullptr, {}).status().code(), absl::StatusCode::kInvalidArgument);
  const auto seed = Seed();
  const auto encoded = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(encoded.ok());
  const auto motion = DecodeMotionFrame3D(encoded->values());
  ASSERT_TRUE(motion.ok());
  LiquidSceneInputs3D scene;
  scene.obstacle = TranslatingHalfspace3D{
      .unit_normal = {0, 1, 0},
      .start_plane_coordinate = 0,
      .wall_velocity = {0, 0.1, 0},
  };
  EXPECT_EQ(
      ReconstructSurfaceFrame3D(*motion, scene, {}, nullptr, {}).status().code(),
      absl::StatusCode::kUnimplemented
  );
}

TEST(SurfaceReconstruction3D, EnforcesSampleBudgetBeforeValidatingOrAllocatingDerivedGeometry) {
  MotionFrame3D source;
  source.samples.resize(66);
  EXPECT_EQ(
      ReconstructSurfaceFrame3D(source, {}, {}, nullptr, {}).status().code(),
      absl::StatusCode::kResourceExhausted
  );
}

TEST(SurfaceReconstruction3D, EnforcesSourceRecordBudgetAndReportsMaterialUnderflow) {
  auto seed = Seed();
  auto bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(bytes.ok());
  auto source = DecodeMotionFrame3D(bytes->values());
  ASSERT_TRUE(source.ok());
  SurfaceReconstructionOptions3D options{.spacing_m = 0.125, .sigma_m = 0.25};
  options.limits.cache.maximum_records.particles = 0;
  EXPECT_EQ(
      ReconstructSurfaceFrame3D(*source, {}, {}, nullptr, options).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  seed.particles[0].reference_volume = 1e-200;
  seed.particles[0].volume_ratio = 1e-200;
  bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(bytes.ok());
  source = DecodeMotionFrame3D(bytes->values());
  ASSERT_TRUE(source.ok());
  EXPECT_EQ(ReconstructSurfaceFrame3D(*source, {}, {}, nullptr, {}).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(SurfaceReconstruction3D, SeedsSeparatedComponentsAndRejectsOverlappingScaffolds) {
  auto seed = Seed();
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(2);
  ASSERT_TRUE(particles.ok());
  ASSERT_TRUE(particles->Resize(2).ok());
  (*particles)[0] = seed.particles[0];
  (*particles)[1] = seed.particles[0];
  (*particles)[1].id = 2;
  (*particles)[1].world_position.x += 3;
  seed.particles = std::move(*particles);
  seed.next_particle_id = 3;
  auto bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(bytes.ok());
  auto source = DecodeMotionFrame3D(bytes->values());
  ASSERT_TRUE(source.ok());
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{.radius = 0.1};
  const SurfaceReconstructionOptions3D options{.spacing_m = 0.125, .sigma_m = 0.25};
  const auto result = ReconstructSurfaceFrame3D(*source, scene, {}, nullptr, options);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->frame.volumes.size(), 2);
  EXPECT_EQ(result->corrected_metrics.components, 2);
  EXPECT_EQ(result->raw_metrics.positive_volume_components, 2);
  EXPECT_EQ(result->raw_metrics.negative_volume_components, 0);
  EXPECT_EQ(result->corrected_metrics.positive_volume_components, 2);
  EXPECT_EQ(result->corrected_metrics.negative_volume_components, 0);
  EXPECT_NEAR(result->frame.volumes[0].corrected_m3, 1, 0.005);
  EXPECT_NEAR(result->frame.volumes[1].corrected_m3, 1, 0.005);
  EXPECT_EQ(result->frame.volumes[0].component, 1);
  EXPECT_EQ(result->frame.volumes[1].component, 2);

  // Both particles stay disconnected in the source, but the reconstructed
  // volumes cannot fit at this separation without overlapping.
  source->samples.front().particles[1].world_position.x = 2.71;
  const auto overlapping = ReconstructSurfaceFrame3D(*source, scene, {}, nullptr, options);
  EXPECT_EQ(overlapping.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(SurfaceReconstruction3D, RejectsCoincidentNewComponentInsteadOfReusingSurfaceIds) {
  auto seed = Seed();
  auto bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(bytes.ok());
  auto initial = DecodeMotionFrame3D(bytes->values());
  ASSERT_TRUE(initial.ok());
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{.radius = 1.1};
  SurfaceReconstructionOptions3D options{.spacing_m = 0.125, .sigma_m = 0.25};
  auto first = ReconstructSurfaceFrame3D(*initial, scene, {}, nullptr, options);
  ASSERT_TRUE(first.ok()) << first.status();
  auto capture = MotionCapture3D::Create(seed, 1.0 / 24, 1);
  ASSERT_TRUE(capture.ok());
  seed.particles[0].id = 2;
  seed.next_particle_id = 3;
  seed.clock = {.time = 1.0 / 24, .frame_ordinal = 1, .accepted_step_ordinal = 1, .epoch = 1};
  ASSERT_TRUE(capture->Observe(seed, false).ok());
  auto interval = capture->Finish(seed);
  ASSERT_TRUE(interval.ok());
  auto source = DecodeMotionFrame3D(interval->values());
  ASSERT_TRUE(source.ok());
  options.maximum_nodes = 1;
  const auto result = ReconstructSurfaceFrame3D(*source, scene, {}, &first->frame, options);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_NE(result.status().message().find("lineage"), std::string_view::npos);
}
}  // namespace
}  // namespace fluidity
