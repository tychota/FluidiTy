#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "fluidity/host/liquid_resampling_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

LiquidParticleState3D Parent() {
  return {
      .id = 3,
      .tag = 17,
      .world_position = {-1.0, 0.75, -0.5},
      .world_velocity = {2.0, -1.0, 3.0},
      .velocity_slopes = {{1.0, 2.0, 3.0}, {4.0, -2.0, 6.0}, {7.0, 8.0, 3.0}},
      .mass = 8.0,
      .reference_volume = 4.0,
      .volume_ratio = 1.5,
      .side_tag = -42,
      .pic_initialized = true,
  };
}

struct Certificate {
  int calls = 0;
  int refuse_call = -1;
  std::array<double, 9> widths{};
  std::array<std::uint64_t, 9> ids{};
};

absl::Status
Certify(const LiquidParticleState3D& particle, double cell_width, LiquidUniformResamplingBasis3D, void* context) {
  auto& certificate = *static_cast<Certificate*>(context);
  const int call = certificate.calls++;
  certificate.widths[call] = cell_width;
  certificate.ids[call] = particle.id;
  if (call == certificate.refuse_call) {
    return absl::FailedPreconditionError("support not admissible");
  }
  return absl::OkStatus();
}

std::array<double, 3> Components(Vec3 vector) {
  return {vector.x, vector.y, vector.z};
}

TEST(LiquidUniformSplit3D, PreservesInventoriesAndMatchesAnalyticCubicDeficit) {
  const LiquidParticleState3D parent = Parent();
  Certificate certificate;
  auto result = SplitUniformLiquidParticle3D(
      parent,
      50,
      2.0,
      LiquidUniformResamplingBasis3D::kCubicNormal,
      Certify,
      &certificate
  );
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->next_particle_id, 58);
  EXPECT_DOUBLE_EQ(result->kinetic_affine_energy_deficit, 14.0);
  EXPECT_EQ(certificate.calls, 9);
  EXPECT_DOUBLE_EQ(certificate.widths[0], 2.0);
  EXPECT_EQ(certificate.ids[0], parent.id);

  double mass = 0.0;
  double reference_volume = 0.0;
  double current_volume = 0.0;
  double energy = 0.0;
  std::array<double, 3> momentum{};
  std::array<std::array<double, 3>, 3> offset_covariance{};
  std::array<std::array<double, 3>, 3> spatial_momentum{};
  const auto position = Components(parent.world_position);
  const auto velocity = Components(parent.world_velocity);
  const std::array slopes{
      Components(parent.velocity_slopes.x_row),
      Components(parent.velocity_slopes.y_row),
      Components(parent.velocity_slopes.z_row)
  };
  const double displacement = std::sqrt(3.0) / 2.0;
  for (std::size_t index = 0; index < result->children.size(); ++index) {
    const auto& child = result->children[index];
    const auto child_position = Components(child.world_position);
    const auto child_velocity = Components(child.world_velocity);
    EXPECT_EQ(child.id, 50 + index);
    EXPECT_EQ(certificate.ids[index + 1], child.id);
    EXPECT_DOUBLE_EQ(certificate.widths[index + 1], 1.0);
    auto unchanged_fields = child;
    unchanged_fields.id = parent.id;
    unchanged_fields.world_position = parent.world_position;
    unchanged_fields.world_velocity = parent.world_velocity;
    unchanged_fields.mass = parent.mass;
    unchanged_fields.reference_volume = parent.reference_volume;
    EXPECT_EQ(unchanged_fields, parent);

    mass += child.mass;
    reference_volume += child.reference_volume;
    current_volume += child.volume_ratio * child.reference_volume;
    std::array<double, 3> offset{};
    for (std::size_t axis = 0; axis < offset.size(); ++axis) {
      offset[axis] = child_position[axis] - position[axis];
      EXPECT_NEAR(offset[axis], (index & (1u << axis)) ? displacement : -displacement, 1e-14);
    }
    for (std::size_t row = 0; row < velocity.size(); ++row) {
      double expected_velocity = velocity[row];
      energy += child.mass * child_velocity[row] * child_velocity[row] / 2.0;
      momentum[row] += child.mass * child_velocity[row];
      for (std::size_t axis = 0; axis < offset.size(); ++axis) {
        const double variance = row == axis ? 1.0 / 3.0 : 1.0 / 4.0;
        expected_velocity += slopes[row][axis] * offset[axis];
        energy += child.mass * slopes[row][axis] * slopes[row][axis] * variance / 2.0;
        offset_covariance[row][axis] += offset[row] * offset[axis] / 8.0;
        spatial_momentum[row][axis] +=
            child.mass * (child_position[axis] * child_velocity[row] + slopes[row][axis] * variance);
      }
      EXPECT_NEAR(child_velocity[row], expected_velocity, 1e-13);
    }
  }
  EXPECT_DOUBLE_EQ(mass, parent.mass);
  EXPECT_DOUBLE_EQ(reference_volume, parent.reference_volume);
  EXPECT_DOUBLE_EQ(current_volume, parent.reference_volume * parent.volume_ratio);

  double parent_energy = 0.0;
  for (std::size_t row = 0; row < velocity.size(); ++row) {
    EXPECT_NEAR(momentum[row], parent.mass * velocity[row], 1e-13);
    parent_energy += parent.mass * velocity[row] * velocity[row] / 2.0;
    for (std::size_t axis = 0; axis < position.size(); ++axis) {
      const double variance = row == axis ? 4.0 / 3.0 : 1.0;
      parent_energy += parent.mass * slopes[row][axis] * slopes[row][axis] * variance / 2.0;
      EXPECT_NEAR(offset_covariance[row][axis], row == axis ? 0.75 : 0.0, 1e-14);
      if (row != axis) {
        const double expected = parent.mass
            * (position[axis] * velocity[row] - position[row] * velocity[axis] + slopes[row][axis] - slopes[axis][row]);
        EXPECT_NEAR(spatial_momentum[row][axis] - spatial_momentum[axis][row], expected, 1e-12);
      }
    }
  }
  EXPECT_NEAR(parent_energy - energy, result->kinetic_affine_energy_deficit, 1e-11);
}

TEST(LiquidUniformSplit3D, RejectsUncertifiedParentAndChildWithoutConsumingIds) {
  const auto parent = Parent();
  for (int refuse_call : {0, 6}) {
    Certificate certificate{.refuse_call = refuse_call};
    auto result = SplitUniformLiquidParticle3D(
        parent,
        50,
        2.0,
        LiquidUniformResamplingBasis3D::kCubicNormal,
        Certify,
        &certificate
    );
    EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_EQ(certificate.calls, refuse_call + 1);
    EXPECT_EQ(parent, Parent());
  }
  Certificate certificate;
  auto retry = SplitUniformLiquidParticle3D(
      parent,
      50,
      2.0,
      LiquidUniformResamplingBasis3D::kCubicNormal,
      Certify,
      &certificate
  );
  ASSERT_TRUE(retry.ok()) << retry.status();
  EXPECT_EQ(retry->children.front().id, 50);
}

TEST(LiquidUniformSplit3D, RejectsInvalidAndUnrepresentableInput) {
  auto parent = Parent();
  Certificate certificate;
  constexpr auto basis = LiquidUniformResamplingBasis3D::kCubicNormal;
  EXPECT_EQ(
      SplitUniformLiquidParticle3D(parent, parent.id, 2.0, basis, Certify, &certificate).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      SplitUniformLiquidParticle3D(parent, 50, 2.0, basis, nullptr, nullptr).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      SplitUniformLiquidParticle3D(
          parent,
          std::numeric_limits<std::uint64_t>::max() - 7,
          2.0,
          basis,
          Certify,
          &certificate
      )
          .status()
          .code(),
      absl::StatusCode::kOutOfRange
  );
  parent.affine_enabled = false;
  EXPECT_EQ(
      SplitUniformLiquidParticle3D(parent, 50, 2.0, basis, Certify, &certificate).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  parent = Parent();
  parent.mass = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(
      SplitUniformLiquidParticle3D(parent, 50, 2.0, basis, Certify, &certificate).status().code(),
      absl::StatusCode::kOutOfRange
  );
  parent = Parent();
  parent.world_position.x = 1e18;
  EXPECT_EQ(
      SplitUniformLiquidParticle3D(parent, 50, 2.0, basis, Certify, &certificate).status().code(),
      absl::StatusCode::kOutOfRange
  );
  parent = Parent();
  parent.world_velocity.x = 1e18;
  EXPECT_EQ(
      SplitUniformLiquidParticle3D(parent, 50, 2.0, basis, Certify, &certificate).status().code(),
      absl::StatusCode::kOutOfRange
  );
  parent = Parent();
  parent.reference_volume = std::numeric_limits<double>::max();
  parent.volume_ratio = 2.0;
  EXPECT_EQ(
      SplitUniformLiquidParticle3D(parent, 50, 2.0, basis, Certify, &certificate).status().code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(certificate.calls, 0);

  constexpr auto last_start = std::numeric_limits<std::uint64_t>::max() - 8;
  auto last_split = SplitUniformLiquidParticle3D(Parent(), last_start, 2.0, basis, Certify, &certificate);
  ASSERT_TRUE(last_split.ok()) << last_split.status();
  EXPECT_EQ(last_split->children.back().id, std::numeric_limits<std::uint64_t>::max() - 1);
  EXPECT_EQ(last_split->next_particle_id, std::numeric_limits<std::uint64_t>::max());
}

TEST(LiquidUniformSplit3D, ReportsZeroForQuadraticAndOffDiagonalCubicRows) {
  auto parent = Parent();
  Certificate certificate;
  auto quadratic =
      SplitUniformLiquidParticle3D(parent, 50, 2.0, LiquidUniformResamplingBasis3D::kQuadratic, Certify, &certificate);
  ASSERT_TRUE(quadratic.ok()) << quadratic.status();
  EXPECT_DOUBLE_EQ(quadratic->kinetic_affine_energy_deficit, 0.0);

  parent.velocity_slopes.x_row.x = 0.0;
  parent.velocity_slopes.y_row.y = 0.0;
  parent.velocity_slopes.z_row.z = 0.0;
  certificate = {};
  auto cubic = SplitUniformLiquidParticle3D(
      parent,
      50,
      2.0,
      LiquidUniformResamplingBasis3D::kCubicNormal,
      Certify,
      &certificate
  );
  ASSERT_TRUE(cubic.ok()) << cubic.status();
  EXPECT_DOUBLE_EQ(cubic->kinetic_affine_energy_deficit, 0.0);
}

}  // namespace
}  // namespace fluidity
