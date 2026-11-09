#pragma once

#include <algorithm>
#include <array>
#include <vector>

#include "fluidity/host/liquid_substep_backend_3d.h"
#include "gtest/gtest.h"

namespace fluidity::liquid_wall_samples_test {

constexpr std::array<std::int64_t, 5> kSides{1, 1, -1, -1, 1};
constexpr double kWallSpeed = 0.1;

inline LiquidSceneInputs3D SlabScene() {
  LiquidSceneInputs3D scene;
  // Keep the only wet pressure cell outside the slab. Both particle sides
  // occupy one bin so the canonical gather order can revisit a side.
  scene.field = LiquidAnalyticSphere3D{{3.5, 2.5, 2.5}, 0.25};
  scene.obstacle = TranslatingSlab3D{{1, 0, 0}, 2.5, 0.05, {kWallSpeed, 0, 0}};
  return scene;
}

inline Vec3 ParticlePosition(std::size_t index) {
  return {kSides[index] > 0 ? 2.6 : 2.4, 2.5 + 0.0625 * static_cast<double>(index), 2.5};
}

inline LiquidState3D SlabParticles(std::size_t count = kSides.size()) {
  LiquidState3D state;
  state.particles = *LiquidRecords3D<LiquidParticleState3D>::Create(count);
  EXPECT_TRUE(state.particles.Resize(count).ok());
  state.next_particle_id = count + 1;
  for (std::size_t index = 0; index < count; ++index) {
    auto& particle = state.particles[index];
    particle.id = index + 1;
    particle.world_position = ParticlePosition(index);
    particle.side_tag = kSides[index];
    particle.pic_initialized = true;
  }
  return state;
}

enum class Operation {
  kWall,
  kGather,
  kField
};
struct Call {
  Operation operation;
  std::int64_t side;
  double coordinate;
  double world_y = 0;
  bool operator==(const Call&) const = default;
};

inline bool SameWallSample(const LiquidPrescribedWallSample3D& left, const LiquidPrescribedWallSample3D& right) {
  return left.available == right.available && left.side_tag == right.side_tag
      && left.unit_normal.x == right.unit_normal.x && left.unit_normal.y == right.unit_normal.y
      && left.unit_normal.z == right.unit_normal.z && left.wall_velocity.x == right.wall_velocity.x
      && left.wall_velocity.y == right.wall_velocity.y && left.wall_velocity.z == right.wall_velocity.z;
}

// Pass the original context to the real operations. Reserve audit memory
// before attempts so the same observer can be used by allocation controls.
struct Audit {
  explicit Audit(std::size_t face_capacity)
      : samples{
            std::vector<LiquidPrescribedWallSample3D>(face_capacity),
            std::vector<LiquidPrescribedWallSample3D>(face_capacity)
        } {
  }
  LiquidSubstepBackend3D actual;
  std::array<std::vector<LiquidPrescribedWallSample3D>, 2> samples;
  std::array<const LiquidPrescribedWallSample3D*, 2> sample_addresses{};
  std::array<Call, 96> calls{};
  std::size_t call_count = 0;
  std::size_t gather_count = 0;
  std::size_t wall_count = 0;
  std::size_t history_mismatches = 0;
  std::size_t old_fallbacks = 0;
  std::size_t new_fallbacks = 0;
  double fail_at_wall_coordinate = 0;

  void Record(Call call) {
    if (call_count < calls.size()) {
      calls[call_count] = call;
    }
    ++call_count;
  }
  void Reset() {
    call_count = gather_count = wall_count = history_mismatches = old_fallbacks = new_fallbacks = 0;
    sample_addresses = {};
  }
};

inline thread_local Audit* current_audit = nullptr;

inline LiquidSubstepBackend3D Observe(Audit& audit, LiquidSubstepBackend3D operations) {
  audit.actual = operations;
  current_audit = &audit;
  operations.sample_faces = [](void* context,
                               SparseMacGridView3D grid,
                               const LiquidObstacle3D& obstacle,
                               std::int64_t side,
                               std::span<LiquidBoundaryFaceTag3D> tags,
                               std::span<LiquidPrescribedWallSample3D> walls) {
    auto& audit = *current_audit;
    if (walls.empty()) {
      return audit.actual.sample_faces(context, grid, obstacle, side, tags, walls);
    }
    const double pose = std::get<TranslatingSlab3D>(obstacle).start_center_coordinate;
    audit.Record({Operation::kWall, side, pose});
    const std::size_t history = audit.wall_count++ % 2;
    if (side == -1 && pose == audit.fail_at_wall_coordinate) {
      return absl::Status(absl::StatusCode::kAborted, {});
    }
    const auto status = audit.actual.sample_faces(context, grid, obstacle, side, tags, walls);
    if (status.ok()) {
      std::ranges::copy(walls, audit.samples[history].begin());
      audit.sample_addresses[history] = walls.data();
    }
    return status;
  };
  operations.gather =
      [](void* context, const LiquidBoundaryGatherViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
        auto& audit = *current_audit;
        ++audit.gather_count;
        if (inputs.particles.size() == 1) {
          const auto& particle = inputs.particles.front();
          const auto position = particle.transfer.trial_world_position;
          audit.Record({Operation::kGather, particle.side_tag, position.x, position.y});
          audit.history_mismatches += inputs.old_history.wall_sample_time != inputs.old_time;
          audit.history_mismatches += inputs.new_history.wall_sample_time != inputs.new_time;
          audit.history_mismatches += inputs.old_history.wall_samples.data() == inputs.new_history.wall_samples.data();
          const std::array histories{inputs.old_history.wall_samples, inputs.new_history.wall_samples};
          for (std::size_t history = 0; history < histories.size(); ++history) {
            const auto walls = histories[history];
            audit.history_mismatches += walls.data() != audit.sample_addresses[history];
            audit.history_mismatches +=
                !std::ranges::equal(walls, std::span(audit.samples[history]).first(walls.size()), SameWallSample);
            bool available = false;
            for (const auto& wall : walls) {
              if (wall.available) {
                available = true;
                audit.history_mismatches += wall.side_tag != particle.side_tag;
                audit.history_mismatches += wall.unit_normal.x != static_cast<double>(particle.side_tag);
                audit.history_mismatches += wall.wall_velocity.x != kWallSpeed;
              }
            }
            audit.history_mismatches += !available;
          }
        }
        const auto result = audit.actual.gather(context, inputs, workspace);
        if (result.ok()) {
          audit.old_fallbacks += result->old_pic_fallback_particle_count;
          audit.new_fallbacks += result->new_pic_fallback_particle_count;
        }
        return result;
      };
  operations.sample_field =
      [](void* context, const LiquidSceneField3D& field, std::span<const LiquidParticleState3D> particles, Vec3 position
      ) {
        auto& audit = *current_audit;
        audit.Record({Operation::kField, 0, position.x, position.y});
        return audit.actual.sample_field(context, field, particles, position);
      };
  return operations;
}

inline void ExpectCalls(const Audit& audit, std::span<const Call> expected) {
  ASSERT_EQ(audit.call_count, expected.size());
  ASSERT_LE(audit.call_count, audit.calls.size());
  for (std::size_t index = 0; index < std::min(audit.call_count, expected.size()); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(audit.calls[index].operation, expected[index].operation);
    EXPECT_EQ(audit.calls[index].side, expected[index].side);
    EXPECT_DOUBLE_EQ(audit.calls[index].coordinate, expected[index].coordinate);
    EXPECT_DOUBLE_EQ(audit.calls[index].world_y, expected[index].world_y);
  }
  EXPECT_EQ(audit.history_mismatches, 0);
}

inline std::array<Call, 16> SlabCalls(double new_pose) {
  return {{
      {Operation::kWall, 1, 2.5},
      {Operation::kWall, 1, new_pose},
      {Operation::kGather, 1, 2.6, 2.5},
      {Operation::kField, 0, 2.6, 2.5},
      {Operation::kGather, 1, 2.6, 2.5625},
      {Operation::kField, 0, 2.6, 2.5625},
      {Operation::kWall, -1, 2.5},
      {Operation::kWall, -1, new_pose},
      {Operation::kGather, -1, 2.4, 2.625},
      {Operation::kField, 0, 2.4, 2.625},
      {Operation::kGather, -1, 2.4, 2.6875},
      {Operation::kField, 0, 2.4, 2.6875},
      {Operation::kWall, 1, 2.5},
      {Operation::kWall, 1, new_pose},
      {Operation::kGather, 1, 2.6, 2.75},
      {Operation::kField, 0, 2.6, 2.75},
  }};
}

inline void ExpectGrowthRefreshes(const Audit& audit) {
  ASSERT_EQ(audit.call_count, 18);
  EXPECT_EQ(audit.wall_count, 6);
  EXPECT_EQ(audit.gather_count, 6);
  EXPECT_EQ(audit.history_mismatches, 0);
  const auto first_step = std::span(audit.calls).first(6);
  const auto replayed_step = std::span(audit.calls).subspan(6, 6);
  const auto expected = SlabCalls(2.5005);
  for (std::size_t index = 0; index < first_step.size(); ++index) {
    EXPECT_EQ(first_step[index], expected[index]);
    EXPECT_EQ(replayed_step[index], expected[index]);
  }
  const Call next_old{Operation::kWall, 1, 2.5005};
  const Call next_new{Operation::kWall, 1, 2.501};
  EXPECT_EQ(audit.calls[12], next_old);
  EXPECT_EQ(audit.calls[13], next_new);
}
}  // namespace fluidity::liquid_wall_samples_test
