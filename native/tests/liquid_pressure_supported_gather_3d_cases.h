#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "fluidity/host/liquid_substep_backend_3d.h"
#include "gtest/gtest.h"

namespace fluidity::pressure_supported_gather_test {

constexpr double kDuration = 0x1p-6;
constexpr Vec3 kCenter{-1.5, 2.5, 2.5};
constexpr double kTargetDivergence = 0.75;
constexpr double kForcedYVelocity = -0.125;

inline LiquidState3D Initial(std::size_t particle_count = 1) {
  LiquidState3D state;
  state.particles = *LiquidRecords3D<LiquidParticleState3D>::Create(particle_count);
  EXPECT_TRUE(state.particles.Resize(particle_count).ok());
  state.next_particle_id = particle_count + 1;
  state.particles[0].id = 1;
  state.particles[0].world_position = kCenter;
  state.particles[0].pic_initialized = true;
  if (particle_count == 2) {
    state.particles[1] = state.particles[0];
    state.particles[1].id = 2;
    state.particles[1].world_position.x = 14.5;
  }
  return state;
}

inline LiquidSceneInputs3D Scene() {
  LiquidSceneInputs3D scene;
  // All eight corners of the center cell are wet, while all adjacent
  // centers are dry. The six open air links have identical conductance.
  scene.field = LiquidAnalyticSphere3D{kCenter, 0.9375};
  scene.world_acceleration = {0, -8, 0};
  scene.target_divergence = kTargetDivergence;
  return scene;
}

struct Audit {
  explicit Audit(std::size_t face_capacity)
      : incident(face_capacity),
        old_validity(face_capacity),
        new_validity(face_capacity),
        old_velocity(face_capacity),
        new_velocity(face_capacity) {
  }
  LiquidSubstepBackend3D actual;
  std::vector<unsigned char> incident;
  std::vector<LiquidFaceValidity3D> old_validity;
  std::vector<LiquidFaceValidity3D> new_validity;
  std::vector<double> old_velocity;
  std::vector<double> new_velocity;
  LiquidBoundaryParticleGather3D gathered;
  std::size_t frozen_cell_count = 0;
  std::size_t frozen_face_count = 0;
  std::size_t gather_count = 0;
  bool gather_failed = false;
  Vec3 failed_position{};
};

inline thread_local Audit* active_audit = nullptr;

inline LiquidSubstepBackend3D Observe(Audit& audit, LiquidSubstepBackend3D backend) {
  audit.actual = backend;
  backend.freeze =
      [](void* context, const LiquidBoundaryGeometryViewInputs3D& input, LiquidBoundaryWorkspace3D& workspace) {
        auto& observed = *active_audit;
        const auto frozen = observed.actual.freeze(context, input, workspace);
        if (frozen.ok()) {
          observed.frozen_cell_count = frozen->cells.size();
          observed.frozen_face_count = frozen->faces.size();
          std::ranges::fill(observed.incident, false);
          for (const auto& face : frozen->faces) {
            observed.incident[face.face_slot] = true;
          }
        }
        return frozen;
      };
  backend.gather = [](void* context, const LiquidBoundaryGatherViewInputs3D& input, LiquidBoundaryWorkspace3D& workspace
                   ) {
    auto& observed = *active_audit;
    ++observed.gather_count;
    std::ranges::copy(input.old_history.validity, observed.old_validity.begin());
    std::ranges::copy(input.new_history.validity, observed.new_validity.begin());
    std::ranges::copy(input.old_history.face_velocities, observed.old_velocity.begin());
    std::ranges::copy(input.new_history.face_velocities, observed.new_velocity.begin());
    const auto gathered = observed.actual.gather(context, input, workspace);
    if (gathered.ok()) {
      observed.gathered = gathered->particles[0];
    } else {
      observed.gather_failed = true;
      observed.failed_position = input.particles[0].transfer.trial_world_position;
    }
    return gathered;
  };
  return backend;
}

inline void ExpectSymmetricPressureCell(const LiquidSubstepScheduler3D& scheduler, const Audit& audit) {
  const auto& committed = scheduler.committed();
  ASSERT_EQ(audit.gather_count, 1);
  ASSERT_EQ(audit.frozen_cell_count, 1);
  ASSERT_EQ(audit.frozen_face_count, 6);
  ASSERT_EQ(committed.cells.size(), 1);
  EXPECT_EQ(committed.cells[0].cell_index.x, -2);
  EXPECT_EQ(committed.cells[0].cell_index.y, 2);
  EXPECT_EQ(committed.cells[0].cell_index.z, 2);
  EXPECT_NEAR(committed.cells[0].previous_pressure, 7500, 1e-10);

  std::size_t force_only_faces = 0;
  std::size_t force_only_new_donors = 0;
  bool found_zero_weight_face = false;
  std::size_t zero_mass_faces = 0;
  for (const auto& face : committed.faces.values()) {
    const std::size_t slot = face.face_slot;
    EXPECT_DOUBLE_EQ(audit.old_velocity[slot], 0);
    EXPECT_DOUBLE_EQ(audit.new_velocity[slot], face.carrier_velocity);
    if (audit.incident[slot]) {
      EXPECT_EQ(face.validity, LiquidFaceValidity3D::kValid);
      EXPECT_EQ(audit.old_validity[slot], LiquidFaceValidity3D::kValid);
      EXPECT_EQ(audit.new_validity[slot], LiquidFaceValidity3D::kValid);
      continue;
    }
    const double forced_velocity = face.face_index.axis == FaceNormalAxis::kY ? kForcedYVelocity : 0;
    EXPECT_DOUBLE_EQ(face.carrier_velocity, forced_velocity);
    EXPECT_EQ(audit.old_validity[slot], face.validity);
    if (face.validity == LiquidFaceValidity3D::kValid) {
      ++force_only_faces;
      force_only_new_donors += audit.new_validity[slot] != LiquidFaceValidity3D::kMissing;
    } else if (face.validity == LiquidFaceValidity3D::kMissing) {
      ++zero_mass_faces;
      EXPECT_EQ(audit.new_validity[slot], LiquidFaceValidity3D::kMissing);
      const auto lattice = face.face_index.lattice_index;
      found_zero_weight_face |=
          face.face_index.axis == FaceNormalAxis::kX && lattice.x == 0 && lattice.y == 2 && lattice.z == 2;
    }
  }
  EXPECT_EQ(force_only_faces, 48);
  EXPECT_EQ(force_only_new_donors, 0);
  EXPECT_GT(zero_mass_faces, 0);
  EXPECT_TRUE(found_zero_weight_face);

  // Each pressure face receives an outward 1/8 m/s correction. The two
  // retained samples per component are rank deficient. Nearest continuation
  // assigns each missing sample on a contributing normal plane that plane's
  // corrected velocity, so the complete tangential basis sums to one.
  const auto& gathered = audit.gathered;
  EXPECT_FALSE(gathered.old_pic_fallback);
  EXPECT_TRUE(gathered.new_pic_fallback);
  EXPECT_FALSE(gathered.low_mass_fallback);
  EXPECT_FALSE(gathered.affine_enabled);
  const std::array pic{gathered.new_pic_velocity.x, gathered.new_pic_velocity.y, gathered.new_pic_velocity.z};
  const std::array increment{
      gathered.grid_velocity_increment.x,
      gathered.grid_velocity_increment.y,
      gathered.grid_velocity_increment.z,
  };
  const std::array residual{
      gathered.particle_minus_old_pic_velocity.x,
      gathered.particle_minus_old_pic_velocity.y,
      gathered.particle_minus_old_pic_velocity.z,
  };
  const std::array rows{
      gathered.velocity_gradient.x_row,
      gathered.velocity_gradient.y_row,
      gathered.velocity_gradient.z_row,
  };
  for (std::size_t axis = 0; axis < rows.size(); ++axis) {
    const double expected_pic = axis == 1 ? kForcedYVelocity : 0;
    EXPECT_NEAR(pic[axis], expected_pic, 1e-12);
    EXPECT_NEAR(increment[axis], expected_pic, 1e-12);
    EXPECT_DOUBLE_EQ(residual[axis], 0);
    const std::array gradient{rows[axis].x, rows[axis].y, rows[axis].z};
    for (std::size_t derivative_axis = 0; derivative_axis < gradient.size(); ++derivative_axis) {
      EXPECT_NEAR(gradient[derivative_axis], axis == derivative_axis ? 0.25 : 0, 1e-12);
    }
  }
  ASSERT_EQ(committed.particles.size(), 1);
  const auto& particle = committed.particles[0];
  EXPECT_NEAR(particle.world_velocity.y, kForcedYVelocity, 1e-12);
  EXPECT_NEAR(particle.world_position.y, kCenter.y + kDuration * kForcedYVelocity, 1e-12);
  EXPECT_NEAR(particle.volume_ratio, std::exp(kDuration * kTargetDivergence), 1e-12);
  EXPECT_FALSE(particle.affine_enabled);
}

inline void ExpectVector(Vec3 actual, Vec3 expected, double tolerance = 1e-12) {
  EXPECT_NEAR(actual.x, expected.x, tolerance);
  EXPECT_NEAR(actual.y, expected.y, tolerance);
  EXPECT_NEAR(actual.z, expected.z, tolerance);
}

inline void
ExpectSameGather(const LiquidBoundaryParticleGather3D& mixed, const LiquidBoundaryParticleGather3D& isolated) {
  ExpectVector(mixed.new_pic_velocity, isolated.new_pic_velocity, 0);
  ExpectVector(mixed.grid_velocity_increment, isolated.grid_velocity_increment, 0);
  ExpectVector(mixed.particle_minus_old_pic_velocity, isolated.particle_minus_old_pic_velocity, 0);
  for (const auto row : {&Mat3::x_row, &Mat3::y_row, &Mat3::z_row}) {
    ExpectVector(mixed.fitted_velocity_slopes.*row, isolated.fitted_velocity_slopes.*row, 0);
    ExpectVector(mixed.velocity_gradient.*row, isolated.velocity_gradient.*row, 0);
  }
  EXPECT_EQ(mixed.affine_enabled, isolated.affine_enabled);
  EXPECT_EQ(mixed.old_pic_fallback, isolated.old_pic_fallback);
  EXPECT_EQ(mixed.new_pic_fallback, isolated.new_pic_fallback);
  EXPECT_EQ(mixed.low_mass_fallback, isolated.low_mass_fallback);
}

inline std::size_t MaskedCarrierCount(const LiquidState3D& state, const Audit& audit) {
  std::size_t masked = 0;
  for (const auto& face : state.faces.values()) {
    if (audit.new_validity[face.face_slot] != face.validity) {
      EXPECT_FALSE(audit.incident[face.face_slot]);
      EXPECT_TRUE(face.structurally_eligible);
      EXPECT_EQ(face.validity, LiquidFaceValidity3D::kValid);
      EXPECT_EQ(audit.new_validity[face.face_slot], LiquidFaceValidity3D::kMissing);
      ++masked;
    }
  }
  return masked;
}

inline void ExpectMixedPressureDoesNotChangeUnsupportedGather(
    LiquidSubstepBackend3D mixed_operations,
    LiquidSubstepBackend3D no_pressure_operations,
    LiquidStepOptions3D options
) {
  const auto initial = Initial(2);
  auto mixed_scene = Scene();
  mixed_scene.target_divergence = 0;
  auto no_pressure_scene = mixed_scene;
  no_pressure_scene.field = LiquidAnalyticSphere3D{kCenter, 0.25};
  Audit mixed_audit(options.capacities.faces);
  Audit no_pressure_audit(options.capacities.faces);
  auto mixed = LiquidSubstepScheduler3D::Create(initial, mixed_scene, options, Observe(mixed_audit, mixed_operations));
  ASSERT_TRUE(mixed.ok()) << mixed.status();
  auto no_pressure = LiquidSubstepScheduler3D::Create(
      initial,
      no_pressure_scene,
      options,
      Observe(no_pressure_audit, no_pressure_operations)
  );
  ASSERT_TRUE(no_pressure.ok()) << no_pressure.status();

  active_audit = &mixed_audit;
  ASSERT_TRUE(mixed->AdvanceFrame(kDuration).ok());
  active_audit = &no_pressure_audit;
  ASSERT_TRUE(no_pressure->AdvanceFrame(kDuration).ok());
  ASSERT_EQ(mixed_audit.frozen_cell_count, 1);
  ASSERT_EQ(no_pressure_audit.frozen_cell_count, 0);
  ASSERT_EQ(mixed_audit.gather_count, 2);
  ASSERT_EQ(no_pressure_audit.gather_count, 2);
  EXPECT_EQ(mixed->diagnostics().new_pic_fallback_count, 1);
  EXPECT_EQ(no_pressure->diagnostics().new_pic_fallback_count, 0);
  EXPECT_EQ(mixed->diagnostics().old_pic_fallback_count, 0);
  EXPECT_EQ(no_pressure->diagnostics().old_pic_fallback_count, 0);
  EXPECT_EQ(MaskedCarrierCount(mixed->committed(), mixed_audit), 0);
  EXPECT_EQ(MaskedCarrierCount(no_pressure->committed(), no_pressure_audit), 0);

  // The later canonical gather is the disjoint particle. Compare both
  // full gather receipts and exact committed particle data to the empty solve.
  ExpectSameGather(mixed_audit.gathered, no_pressure_audit.gathered);
  EXPECT_EQ(mixed->committed().particles[1], no_pressure->committed().particles[1]);
  for (const auto* scheduler : {&*mixed, &*no_pressure}) {
    for (const auto& particle : scheduler->committed().particles.values()) {
      ExpectVector(particle.world_velocity, {0, kForcedYVelocity, 0});
      EXPECT_NEAR(particle.world_position.y, kCenter.y + kDuration * kForcedYVelocity, 1e-12);
      EXPECT_NEAR(particle.volume_ratio, 1, 1e-12);
    }
  }
}

inline void ExpectSignedBoundaryMembership(LiquidSubstepBackend3D operations, LiquidStepOptions3D options, double x) {
  SCOPED_TRACE(x);
  auto initial = Initial();
  initial.particles[0].world_position.x = x;
  Audit audit(options.capacities.faces);
  active_audit = &audit;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options, Observe(audit, operations));
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(kDuration);
  ASSERT_TRUE(status.ok()) << status;
  ASSERT_EQ(audit.frozen_cell_count, 1);
  EXPECT_EQ(MaskedCarrierCount(scheduler->committed(), audit) > 0, x < -1);
}

inline void ExpectTranslatedMembership(LiquidSubstepBackend3D operations, LiquidStepOptions3D options) {
  auto initial = Initial();
  // The exact high plane of cell 9 is strictly above origin + 1 since the
  // binary64 value of 0.1 is above 1/10. Rounded division gives exactly 10.
  constexpr double origin_x = 0x1p40;
  initial.geometry = {{origin_x, 0, 0}, 0.1};
  initial.particles[0].world_position = {origin_x + 1, 0.25, 0.25};
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{origin_x + 0.95, 0.25, 0.25}, 0.094};
  scene.target_divergence = 0;
  Audit audit(options.capacities.faces);
  active_audit = &audit;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, Observe(audit, operations));
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(kDuration);
  ASSERT_TRUE(status.ok()) << status;
  ASSERT_EQ(audit.frozen_cell_count, 1);
  ASSERT_EQ(scheduler->committed().cells[0].cell_index.x, 9);
  EXPECT_GT(MaskedCarrierCount(scheduler->committed(), audit), 0);
}

inline void ExpectPressureMaskKeepsLowMass(LiquidSubstepBackend3D operations, LiquidStepOptions3D options) {
  options.minimum_face_mass = 0x1p-7;
  Audit audit(options.capacities.faces);
  active_audit = &audit;
  auto scheduler = LiquidSubstepScheduler3D::Create(Initial(), Scene(), options, Observe(audit, operations));
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(kDuration);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(MaskedCarrierCount(scheduler->committed(), audit), 24);
  std::size_t low_mass = 0;
  for (const auto& face : scheduler->committed().faces.values()) {
    if (face.validity == LiquidFaceValidity3D::kLowMass) {
      ++low_mass;
      EXPECT_FALSE(audit.incident[face.face_slot]);
      EXPECT_EQ(audit.old_validity[face.face_slot], LiquidFaceValidity3D::kLowMass);
      EXPECT_EQ(audit.new_validity[face.face_slot], LiquidFaceValidity3D::kLowMass);
    }
  }
  EXPECT_EQ(low_mass, 24);
  EXPECT_TRUE(audit.gathered.low_mass_fallback);
  EXPECT_FALSE(audit.gathered.affine_enabled);
  ExpectVector(audit.gathered.particle_minus_old_pic_velocity, {});
}

inline LiquidState3D NoDonorInitial() {
  auto state = Initial(3);
  for (std::size_t index = 1; index < state.particles.size(); ++index) {
    auto& particle = state.particles[index];
    particle = state.particles[0];
    particle.id = index + 1;
    particle.mass = 8;
    particle.world_position.y = index == 1 ? 1.5 : 3.5;
  }
  return state;
}

inline void
ExpectNoDonorRollback(const LiquidSubstepScheduler3D& scheduler, const LiquidState3D& initial, const Audit& audit) {
  ASSERT_EQ(audit.frozen_cell_count, 1);
  EXPECT_EQ(scheduler.diagnostics().failed_stage, LiquidPhysicalStage3D::kGather);
  EXPECT_EQ(scheduler.diagnostics().failure_code, absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(scheduler.diagnostics().accepted_attempts, 0);
  EXPECT_EQ(scheduler.diagnostics().dt_retries, 0);
  EXPECT_EQ(scheduler.committed(), initial);
  ASSERT_TRUE(audit.gather_failed);
  EXPECT_EQ(audit.failed_position.y, kCenter.y);
}

inline void ExpectCompleteNonaffineGradient(LiquidSubstepBackend3D backend, LiquidStateCapacities3D capacities) {
  constexpr UniformMacGeometry3D geometry{{}, 1};
  constexpr std::array blocks{BlockIndex3D{0, 0, 0}};
  const auto grid = SparseMacGridView3D::Create(geometry, blocks);
  ASSERT_TRUE(grid.ok()) << grid.status();
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const auto count = grid->face_count();
  const std::vector<LiquidBoundaryFaceTag3D> tags(count);
  const std::vector<LiquidPrescribedWallSample3D> walls(count);
  const std::vector<LiquidFaceValidity3D> validity(count, LiquidFaceValidity3D::kValid);
  std::vector<double> old_velocity(count);
  std::vector<double> new_velocity(count);

  // Discrete curl of psi(x,y) = (x-1)^3 * (y-1). Its x/y face
  // differences cancel cell by cell. At particle-local x=1/4, the quadratic
  // spline differentiates u to 1, but interpolates the y slope of v to -19/16.
  const auto cube = [](double x) {
    return x * x * x;
  };
  for (std::size_t slot = 0; slot < count; ++slot) {
    const auto face = grid->FaceIndex(slot);
    ASSERT_TRUE(face.ok());
    const auto anchor = grid->FaceWorldAnchor(*face);
    ASSERT_TRUE(anchor.ok());
    const double x = anchor->x - 1;
    const double y = anchor->y - 1;
    switch (face->axis) {
      case FaceNormalAxis::kX:
        old_velocity[slot] = 0.25;
        new_velocity[slot] = cube(x);
        break;
      case FaceNormalAxis::kY:
        old_velocity[slot] = -0.125;
        new_velocity[slot] = -(cube(x + 0.5) - cube(x - 0.5)) * y;
        break;
      case FaceNormalAxis::kZ:
        old_velocity[slot] = 0.5;
        new_velocity[slot] = 0;
        break;
    }
  }
  constexpr Vec3 position{1.25, 1.375, 1.25};
  constexpr Vec3 velocity{0.5, -0.25, 0.75};
  const LiquidBoundaryParticle3D transfer_particle{
      .transfer =
          {
              .sorted_world_position = position,
              .trial_world_position = position,
              .particle_mass = 1,
              .particle_velocity = velocity,
              .particle_velocity_slopes = {},
          },
  };
  const std::array particles{transfer_particle};
  auto inputs = LiquidBoundaryGatherViewInputs3D{
      .mac_grid = *grid,
      .particles = particles,
      .old_time = 0,
      .new_time = kDuration,
      .old_history = {0, 0, old_velocity, validity, tags, walls},
      .new_history = {kDuration, kDuration, new_velocity, validity, tags, walls},
  };
  const auto gather = backend.gather(backend.context, inputs, *workspace);
  ASSERT_TRUE(gather.ok()) << gather.status();
  const auto& result = gather->particles[0];
  constexpr Vec3 expected_pic{0.25, -57.0 / 128, 0};
  constexpr Vec3 expected_retained{0.25, -0.125, 0.25};
  ExpectVector(result.new_pic_velocity, expected_pic);
  ExpectVector(result.grid_velocity_increment, {0, -41.0 / 128, -0.5});
  ExpectVector(result.particle_minus_old_pic_velocity, expected_retained);
  ExpectVector(result.velocity_gradient.x_row, {1, 0, 0});
  ExpectVector(result.velocity_gradient.y_row, {-0.5625, -1.1875, 0});
  ExpectVector(result.velocity_gradient.z_row, {});
  EXPECT_TRUE(result.affine_enabled);
  EXPECT_FALSE(result.old_pic_fallback);
  EXPECT_FALSE(result.new_pic_fallback);
  EXPECT_FALSE(result.low_mass_fallback);

  auto state = Initial();
  state.particles[0].world_position = position;
  state.particles[0].world_velocity = velocity;
  constexpr std::array liquid_distances{-2.0};
  const auto update = backend.update_particles(
      backend.context,
      {
          .gathered = gather->particles,
          .liquid_distances = liquid_distances,
          .obstacle = {},
          .cell_width = 1,
          .step_duration = kDuration,
          .particle_energy_blend = 0.6,
      },
      state.particles.values()
  );
  ASSERT_TRUE(update.ok()) << update;
  const auto& particle = state.particles[0];
  EXPECT_NEAR(particle.volume_ratio, std::exp(-0.1875 * kDuration), 1e-12);
  ExpectVector(particle.world_velocity, {0.3985, -57.0 / 128 - 0.07425, 0.1485});
  ExpectVector(particle.world_position, {position.x + kDuration * 0.25, position.y - kDuration * 57 / 128, position.z});

  const std::vector<LiquidFaceValidity3D> unavailable(count, LiquidFaceValidity3D::kMissing);
  inputs.new_history.validity = unavailable;
  const auto no_donor = backend.gather(backend.context, inputs, *workspace);
  EXPECT_EQ(no_donor.status().code(), absl::StatusCode::kFailedPrecondition);
  ExpectVector(result.new_pic_velocity, expected_pic);
}

}  // namespace fluidity::pressure_supported_gather_test
