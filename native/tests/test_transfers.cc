#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>

#include "absl/status/status.h"
#include "fixture_data.h"
#include "fluidity/host/transfers.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr Grid2D kGrid{8, 8, 1.0, {10.0, -3.0}};
constexpr std::size_t kXFaceCount = 72;
constexpr std::size_t kFaceCount = 144;
constexpr double kSentinel = -29.0;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

void ExpectNear(double actual, double expected) {
  EXPECT_NEAR(actual, expected, 1e-12 * std::max(1.0, std::abs(expected)));
}

void ExpectVec(Vec2 actual, Vec2 expected) {
  ExpectNear(actual.x, expected.x);
  ExpectNear(actual.y, expected.y);
}

void ExpectMat(Mat2 actual, Mat2 expected) {
  ExpectNear(actual.xx, expected.xx);
  ExpectNear(actual.xy, expected.xy);
  ExpectNear(actual.yx, expected.yx);
  ExpectNear(actual.yy, expected.yy);
}

TEST(Transfer, LocalAffineFixture) {
  const auto cases = fixtures::AllFixtures();
  const auto fixture = std::ranges::find(cases, "affine_apic", &fixtures::FixtureRecord::name);
  ASSERT_NE(fixture, cases.end());
  const auto position = fixtures::FindArray(*fixture, "position");
  const auto sample_positions = fixtures::FindArray(*fixture, "sample_positions");
  const auto weights = fixtures::FindArray(*fixture, "weights");
  const auto values = fixtures::FindArray(*fixture, "values");
  const auto expected_value = fixtures::FindArray(*fixture, "expected_value");
  const auto expected_affine = fixtures::FindArray(*fixture, "expected_affine");
  for (const auto* record : {&position, &sample_positions, &weights, &values, &expected_value, &expected_affine}) {
    ASSERT_TRUE(record->ok()) << record->status();
    ASSERT_EQ((**record)->dtype, fixtures::DType::kFloat64);
  }
  ASSERT_EQ((*position)->f64_data.size(), 2);
  ASSERT_EQ((*sample_positions)->f64_data.size(), 8);
  ASSERT_EQ((*expected_value)->f64_data.size(), 1);
  ASSERT_EQ((*expected_affine)->f64_data.size(), 2);

  std::array<Vec2, 4> samples;
  for (std::size_t sample = 0; sample < samples.size(); ++sample) {
    samples[sample] = {(*sample_positions)->f64_data[2 * sample], (*sample_positions)->f64_data[2 * sample + 1]};
  }
  const Vec2 particle_position{(*position)->f64_data[0], (*position)->f64_data[1]};
  const auto fit = FitAffine2D(particle_position, samples, (*weights)->f64_data, (*values)->f64_data);
  ASSERT_TRUE(fit.ok()) << fit.status();
  const auto expect_fixture_value = [&](double actual, double expected) {
    EXPECT_NEAR(actual, expected, fixture->atol + fixture->rtol * std::abs(expected));
  };
  expect_fixture_value(fit->interpolated_value, (*expected_value)->f64_data[0]);
  expect_fixture_value(fit->spatial_slopes.x, (*expected_affine)->f64_data[0]);
  expect_fixture_value(fit->spatial_slopes.y, (*expected_affine)->f64_data[1]);
}

TEST(Transfer, FitUsesFullCenteredMoment) {
  const std::array<Vec2, 3> samples{{{1.0, 1.0}, {2.0, 1.0}, {3.0, 2.0}}};
  const std::array weights{0.25, 0.25, 0.5};
  const std::array values{1.0, 3.0, 1.0};
  const auto fit = FitAffine2D({-2.0, 3.0}, samples, weights, values);
  ASSERT_TRUE(fit.ok()) << fit.status();
  ExpectNear(fit->interpolated_value, 1.5);
  ExpectVec(fit->spatial_slopes, {2.0, -4.0});
}

TEST(Transfer, FitKeepsRepresentableLargeCoupledRow) {
  constexpr std::array<Vec2, 4> samples{{{0.5, 0.5}, {-0.5, -0.5}, {0.25, -0.25}, {-0.25, 0.25}}};
  constexpr std::array weights{0.25, 0.25, 0.25, 0.25};
  constexpr std::array values{1.3e308, -1.3e308, 0.0, 0.0};
  // The moment has diagonal 5/32 and off-diagonal 3/32. The RHS points along
  // its (1, 1) eigenvector with eigenvalue 1/4.
  const auto fit = FitAffine2D({0.0, 0.0}, samples, weights, values);
  ASSERT_TRUE(fit.ok()) << fit.status();
  EXPECT_EQ(fit->interpolated_value, 0.0);
  ExpectVec(fit->spatial_slopes, {1.3e308, 1.3e308});
}

TEST(Transfer, FitPreservesIndependentCoefficientsAcrossExponentRange) {
  constexpr std::array<Vec2, 4> samples{{{0.5, 0.0}, {-0.5, 0.0}, {0.0, 0.5}, {0.0, -0.5}}};
  constexpr std::array weights{0.25, 0.25, 0.25, 0.25};
  constexpr std::array values{8e307, -8e307, 1e-100, -1e-100};
  const auto fit = FitAffine2D({0.0, 0.0}, samples, weights, values);
  ASSERT_TRUE(fit.ok()) << fit.status();
  ExpectNear(fit->spatial_slopes.x, 1.6e308);
  EXPECT_NEAR(fit->spatial_slopes.y, 2e-100, 2e-112);

  constexpr std::array overflowing_values{1e308, -1e308, 1e-100, -1e-100};
  const auto overflow = FitAffine2D({0.0, 0.0}, samples, weights, overflowing_values);
  EXPECT_EQ(overflow.status().code(), absl::StatusCode::kOutOfRange);
}

TEST(Transfer, FitKeepsSmallRepresentableRowWithLargeMoment) {
  constexpr std::array<Vec2, 4> samples{{{1e100, 0.0}, {-1e100, 0.0}, {0.0, 1e94}, {0.0, -1e94}}};
  constexpr std::array weights{0.25, 0.25, 0.25, 0.25};
  constexpr std::array values{1e100, -1e100, 1e-213, -1e-213};
  // The diagonal moments are .5e200 and .5e188. Direct RHS normalization
  // rounds the smaller RHS to subnormal precision before dividing by the determinant.
  const auto fit = FitAffine2D({0.0, 0.0}, samples, weights, values);
  ASSERT_TRUE(fit.ok()) << fit.status();
  ExpectNear(fit->spatial_slopes.x, 1.0);
  EXPECT_NEAR(fit->spatial_slopes.y, 1e-307, 1e-319);
}

TEST(Transfer, StaggeredAffine) {
  // Use explicit face indices and positions on a nonsquare grid to check layout
  // independently of FaceGeometry and the stencil's flattened face indices.
  constexpr Grid2D grid{7, 6, 0.75, {10.0, -3.0}};
  constexpr std::size_t x_face_count = 8 * 6;
  constexpr std::size_t face_count = x_face_count + 7 * 7;
  // The first particle is exactly on the lowest complete-support position of both shifted lattices.
  const std::array<Vec2, 2> positions{{{10.75, -2.25}, {12.3, 0.2}}};
  std::array<double, face_count> old_face_velocities;
  std::array<double, face_count> new_face_velocities;
  for (std::size_t row = 0; row < 6; ++row) {
    for (std::size_t column = 0; column < 8; ++column) {
      const std::size_t face = 8 * row + column;
      const double x = 10.0 + 0.75 * static_cast<double>(column);
      const double y = -3.0 + 0.75 * (static_cast<double>(row) + 0.5);
      new_face_velocities[face] = 1.0 + 2.0 * x - 3.0 * y;
      old_face_velocities[face] = new_face_velocities[face] - 0.75;
    }
  }
  for (std::size_t row = 0; row < 7; ++row) {
    for (std::size_t column = 0; column < 7; ++column) {
      const std::size_t face = x_face_count + 7 * row + column;
      const double x = 10.0 + 0.75 * (static_cast<double>(column) + 0.5);
      const double y = -3.0 + 0.75 * static_cast<double>(row);
      new_face_velocities[face] = -2.0 + 0.5 * x + 4.0 * y;
      old_face_velocities[face] = new_face_velocities[face] + 1.25;
    }
  }
  std::array<Vec2, 2> pic_velocities;
  std::array<Vec2, 2> grid_velocity_increments;
  std::array<Mat2, 2> velocity_fit_slopes;
  std::array<Mat2, 2> velocity_gradients;
  std::array<double, 24> scratch;
  const auto status =
      G2P(grid,
          positions,
          old_face_velocities,
          new_face_velocities,
          pic_velocities,
          grid_velocity_increments,
          velocity_fit_slopes,
          velocity_gradients,
          scratch);
  ASSERT_TRUE(status.ok()) << status;
  ExpectVec(pic_velocities[0], {29.25, -5.625});
  ExpectVec(pic_velocities[1], {25.0, 4.95});
  for (std::size_t particle = 0; particle < positions.size(); ++particle) {
    SCOPED_TRACE(particle);
    ExpectVec(grid_velocity_increments[particle], {0.75, -1.25});
    ExpectMat(velocity_fit_slopes[particle], {2.0, -3.0, 0.5, 4.0});
    ExpectMat(velocity_gradients[particle], {2.0, -3.0, 0.5, 4.0});
  }
}

TEST(Transfer, MassAndAngularMoments) {
  constexpr Grid2D grid{8, 8, 1.0};
  const std::array<Vec2, 3> positions{{{3.2, 3.4}, {4.3, 4.1}, {2.0, 3.0}}};
  const std::array masses{2.0, 3.0, 0.0};
  const std::array<Vec2, 3> velocities{{{1.0, 2.0}, {-2.0, 1.0}, {7.0, 8.0}}};
  const std::array<Mat2, 3> velocity_fit_slopes{{{0.0, -2.0, 2.0, 0.0}, {0.0, 1.0, -1.0, 0.0}, {1.0, 2.0, 3.0, 4.0}}};
  const ParticleView2D particles{
      .positions = positions,
      .masses = masses,
      .velocities = velocities,
      .velocity_fit_slopes = velocity_fit_slopes,
  };
  std::array<double, kFaceCount> face_mass;
  std::array<double, kFaceCount> face_momentum;
  std::array<double, 2 * kFaceCount> scratch;
  const auto status = P2G(grid, particles, face_mass, face_momentum, scratch);
  ASSERT_TRUE(status.ok()) << status;

  const auto mass_split = face_mass.begin() + kXFaceCount;
  const auto momentum_split = face_momentum.begin() + kXFaceCount;
  ExpectNear(std::accumulate(face_mass.begin(), mass_split, 0.0), 5.0);
  ExpectNear(std::accumulate(mass_split, face_mass.end(), 0.0), 5.0);
  ExpectNear(std::accumulate(face_momentum.begin(), momentum_split, 0.0), -4.0);
  ExpectNear(std::accumulate(momentum_split, face_momentum.end(), 0.0), 7.0);

  double angular_momentum = 0.0;
  for (std::size_t face = 0; face < face_momentum.size(); ++face) {
    const auto geometry = FaceGeometry(grid, face);
    ASSERT_TRUE(geometry.ok()) << geometry.status();
    const double moment_arm = face < kXFaceCount ? -geometry->world_position.y : geometry->world_position.x;
    angular_momentum += moment_arm * face_momentum[face];
  }
  // Orbital angular momentum is 43.5 and the affine spin adds 0.5 (h^2/4 per row).
  ExpectNear(angular_momentum, 44.0);
}

TEST(Transfer, NonlinearGradientDiffersFromAffine) {
  constexpr Grid2D grid{8, 8, 1.0};
  const std::array<Vec2, 1> positions{{{3.25, 3.25}}};
  std::array<double, kFaceCount> faces;
  for (std::size_t face = 0; face < faces.size(); ++face) {
    const auto geometry = FaceGeometry(grid, face);
    ASSERT_TRUE(geometry.ok()) << geometry.status();
    const double coordinate = face < kXFaceCount ? geometry->world_position.x : geometry->world_position.y;
    faces[face] = coordinate * coordinate * coordinate;
  }
  std::array<Vec2, 1> pic_velocities;
  std::array<Vec2, 1> grid_velocity_increments;
  std::array<Mat2, 1> velocity_fit_slopes;
  std::array<Mat2, 1> velocity_gradients;
  std::array<double, 12> scratch;
  const auto status =
      G2P(grid,
          positions,
          faces,
          faces,
          pic_velocities,
          grid_velocity_increments,
          velocity_fit_slopes,
          velocity_gradients,
          scratch);
  ASSERT_TRUE(status.ok()) << status;
  // Along each face normal the samples are 8, 27, 64, with weights (1,22,9)/32
  // and derivatives (-1,-2,3)/4. The centered variance of the positions is 1/4.
  ExpectVec(pic_velocities[0], {36.8125, 36.8125});
  ExpectMat(velocity_fit_slopes[0], {34.1875, 0.0, 0.0, 34.1875});
  ExpectMat(velocity_gradients[0], {32.5, 0.0, 0.0, 32.5});
}

TEST(Transfer, IdenticalGridRetainsParticleResidual) {
  const std::array<Vec2, 1> positions{{{13.0, 0.0}}};
  std::array<double, kFaceCount> faces;
  faces.fill(2.0);
  std::array<Vec2, 1> pic_velocities;
  std::array<Vec2, 1> grid_velocity_increments;
  std::array<Mat2, 1> velocity_fit_slopes;
  std::array<Mat2, 1> velocity_gradients;
  std::array<double, 12> scratch;
  const auto status =
      G2P(kGrid,
          positions,
          faces,
          faces,
          pic_velocities,
          grid_velocity_increments,
          velocity_fit_slopes,
          velocity_gradients,
          scratch);
  ASSERT_TRUE(status.ok()) << status;
  ExpectVec(pic_velocities[0], {2.0, 2.0});
  ExpectVec(grid_velocity_increments[0], {0.0, 0.0});

  constexpr Vec2 old_particle_velocity{3.0, 0.0};
  const Vec2 residual = old_particle_velocity - pic_velocities[0];
  const auto liquid = BlendHistory2D(pic_velocities[0], residual, 0.5, MaterialPhase::kLiquid, 0.5);
  ASSERT_TRUE(liquid.ok()) << liquid.status();
  ExpectVec(liquid->stored_particle_velocity, {2.495, 1.01});
  ExpectVec(liquid->advection_velocity, {2.12375, 1.7525});
}

TEST(Transfer, PhaseHistoryPolicy) {
  struct Case {
    MaterialPhase phase;
    Vec2 stored_particle_velocity;
    Vec2 advection_velocity;
  };
  constexpr Vec2 pic_velocity{1.0, 2.0};
  constexpr Vec2 residual{2.0, -1.0};
  constexpr Vec2 blended_stored{2.584, 1.208};
  constexpr Vec2 blended_advecting{1.6336, 1.6832};
  constexpr std::array cases{
      Case{MaterialPhase::kGas, pic_velocity, pic_velocity},
      Case{MaterialPhase::kElastic, blended_stored, pic_velocity},
      Case{MaterialPhase::kGranular, blended_stored, blended_advecting},
      Case{MaterialPhase::kSnow, blended_stored, blended_advecting},
  };
  for (const Case& test : cases) {
    SCOPED_TRACE(static_cast<int>(test.phase));
    const auto result = BlendHistory2D(pic_velocity, residual, 0.8, test.phase, 0.5);
    ASSERT_TRUE(result.ok()) << result.status();
    ExpectVec(result->stored_particle_velocity, test.stored_particle_velocity);
    ExpectVec(result->advection_velocity, test.advection_velocity);
  }
}

TEST(Transfer, EmptyBatches) {
  std::array<double, kFaceCount> face_mass;
  std::array<double, kFaceCount> face_momentum;
  face_mass.fill(kSentinel);
  face_momentum.fill(kSentinel);
  std::array<double, 2 * kFaceCount> scratch;
  const auto deposit = P2G(kGrid, {}, face_mass, face_momentum, scratch);
  ASSERT_TRUE(deposit.ok()) << deposit;
  EXPECT_EQ(face_mass, (std::array<double, kFaceCount>{}));
  EXPECT_EQ(face_momentum, (std::array<double, kFaceCount>{}));
  const auto gather = G2P(kGrid, {}, face_mass, face_momentum, {}, {}, {}, {}, {});
  EXPECT_TRUE(gather.ok()) << gather;
}

TEST(Transfer, ScratchCountsCheckExtents) {
  const auto deposit = P2GScratchDoubles(kGrid);
  ASSERT_TRUE(deposit.ok()) << deposit.status();
  EXPECT_EQ(*deposit, 2 * kFaceCount);
  const auto invalid_grid = P2GScratchDoubles({});
  EXPECT_EQ(invalid_grid.status().code(), absl::StatusCode::kInvalidArgument);

  constexpr std::uint32_t max_dimension = std::numeric_limits<std::uint32_t>::max();
  const auto face_scratch_overflow = P2GScratchDoubles({max_dimension, max_dimension / 2, 1.0});
  EXPECT_EQ(face_scratch_overflow.status().code(), absl::StatusCode::kOutOfRange);
  const auto gather = G2PScratchDoubles(3);
  ASSERT_TRUE(gather.ok()) << gather.status();
  EXPECT_EQ(*gather, 36);
  const auto empty = G2PScratchDoubles(0);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(*empty, 0);
  const auto particle_scratch_overflow = G2PScratchDoubles(std::numeric_limits<std::size_t>::max() / 12 + 1);
  EXPECT_EQ(particle_scratch_overflow.status().code(), absl::StatusCode::kOutOfRange);
}

TEST(Transfer, DepositDoesNotPublishOnError) {
  std::array<Vec2, 2> positions{{{13.25, 0.25}, {10.5, 0.25}}};
  std::array masses{1.0, 1.0};
  std::array<Vec2, 2> velocities{{{2.0, -3.0}, {3.0, 4.0}}};
  const std::array<Mat2, 2> velocity_fit_slopes{};
  ParticleView2D particles{
      .positions = positions,
      .masses = masses,
      .velocities = velocities,
      .velocity_fit_slopes = velocity_fit_slopes,
  };
  std::array<double, kFaceCount> face_mass;
  std::array<double, kFaceCount> face_momentum;
  // The mass sentinel is a valid input mass for the aliasing case.
  face_mass.fill(1.0);
  face_momentum.fill(kSentinel);
  const auto original_mass = face_mass;
  const auto original_momentum = face_momentum;
  std::array<double, 2 * kFaceCount> scratch;
  const auto expect_error = [&](absl::Status status, absl::StatusCode expected) {
    EXPECT_EQ(status.code(), expected) << status;
    EXPECT_EQ(face_mass, original_mass);
    EXPECT_EQ(face_momentum, original_momentum);
  };

  expect_error(P2G(kGrid, particles, face_mass, face_momentum, scratch), absl::StatusCode::kInvalidArgument);
  positions[1] = {14.0, 1.0};
  masses[1] = -1.0;
  expect_error(P2G(kGrid, particles, face_mass, face_momentum, scratch), absl::StatusCode::kInvalidArgument);
  masses[1] = 10.0;
  velocities[1].x = kInfinity;
  expect_error(P2G(kGrid, particles, face_mass, face_momentum, scratch), absl::StatusCode::kInvalidArgument);
  velocities[1].x = std::numeric_limits<double>::max();
  expect_error(P2G(kGrid, particles, face_mass, face_momentum, scratch), absl::StatusCode::kOutOfRange);
  velocities[1].x = 3.0;

  expect_error(
      P2G(kGrid, particles, face_mass, face_momentum, std::span(scratch).subspan(1)),
      absl::StatusCode::kInvalidArgument
  );
  expect_error(P2G(kGrid, particles, face_mass, face_mass, scratch), absl::StatusCode::kInvalidArgument);
  particles.masses = std::span(face_mass).last(2);
  expect_error(P2G(kGrid, particles, face_mass, face_momentum, scratch), absl::StatusCode::kInvalidArgument);
  particles.masses = std::span(masses).first(1);
  expect_error(P2G(kGrid, particles, face_mass, face_momentum, scratch), absl::StatusCode::kInvalidArgument);
}

TEST(Transfer, GatherDoesNotPublishOnError) {
  // x=nx-1 in grid coordinates places a zero-weight node beyond the shifted face lattice.
  std::array<Vec2, 2> positions{{{13.25, 0.25}, {17.0, 0.25}}};
  std::array<double, kFaceCount> old_face_velocities{};
  std::array<double, kFaceCount> new_face_velocities{};
  // The velocity sentinel is also a valid position for the input/output aliasing case.
  constexpr Vec2 velocity_sentinel{13.0, 1.0};
  constexpr Mat2 matrix_sentinel{kSentinel, kSentinel, kSentinel, kSentinel};
  std::array pic_velocities{velocity_sentinel, velocity_sentinel};
  std::array grid_velocity_increments{velocity_sentinel, velocity_sentinel};
  std::array velocity_fit_slopes{matrix_sentinel, matrix_sentinel};
  std::array velocity_gradients{matrix_sentinel, matrix_sentinel};
  std::array<double, 24> scratch;
  const auto expect_error = [&](absl::Status status, absl::StatusCode expected) {
    EXPECT_EQ(status.code(), expected) << status;
    for (std::size_t particle = 0; particle < positions.size(); ++particle) {
      SCOPED_TRACE(particle);
      EXPECT_EQ(pic_velocities[particle].x, velocity_sentinel.x);
      EXPECT_EQ(pic_velocities[particle].y, velocity_sentinel.y);
      EXPECT_EQ(grid_velocity_increments[particle].x, velocity_sentinel.x);
      EXPECT_EQ(grid_velocity_increments[particle].y, velocity_sentinel.y);
      const auto [affine_xx, affine_xy, affine_yx, affine_yy] = velocity_fit_slopes[particle];
      const auto [gradient_xx, gradient_xy, gradient_yx, gradient_yy] = velocity_gradients[particle];
      for (const double value :
           {affine_xx, affine_xy, affine_yx, affine_yy, gradient_xx, gradient_xy, gradient_yx, gradient_yy}) {
        EXPECT_EQ(value, kSentinel);
      }
    }
  };
  expect_error(
      G2P(kGrid,
          positions,
          old_face_velocities,
          new_face_velocities,
          pic_velocities,
          grid_velocity_increments,
          velocity_fit_slopes,
          velocity_gradients,
          scratch),
      absl::StatusCode::kInvalidArgument
  );
  positions[1] = {14.0, 1.0};
  // Even an unsampled face must satisfy the finite-input contract.
  old_face_velocities.front() = kInfinity;
  expect_error(
      G2P(kGrid,
          positions,
          old_face_velocities,
          new_face_velocities,
          pic_velocities,
          grid_velocity_increments,
          velocity_fit_slopes,
          velocity_gradients,
          scratch),
      absl::StatusCode::kInvalidArgument
  );
  old_face_velocities.front() = 0.0;
  expect_error(
      G2P(kGrid,
          positions,
          old_face_velocities,
          new_face_velocities,
          pic_velocities,
          grid_velocity_increments,
          velocity_fit_slopes,
          velocity_gradients,
          std::span(scratch).subspan(1)),
      absl::StatusCode::kInvalidArgument
  );
  expect_error(
      G2P(kGrid,
          positions,
          old_face_velocities,
          new_face_velocities,
          pic_velocities,
          pic_velocities,
          velocity_fit_slopes,
          velocity_gradients,
          scratch),
      absl::StatusCode::kInvalidArgument
  );
  expect_error(
      G2P(kGrid,
          positions,
          old_face_velocities,
          new_face_velocities,
          pic_velocities,
          grid_velocity_increments,
          velocity_fit_slopes,
          velocity_gradients,
          std::span(old_face_velocities).last(24)),
      absl::StatusCode::kInvalidArgument
  );
  expect_error(
      G2P(kGrid,
          std::span(pic_velocities),
          old_face_velocities,
          new_face_velocities,
          pic_velocities,
          grid_velocity_increments,
          velocity_fit_slopes,
          velocity_gradients,
          scratch),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(Transfer, LocalFitRejectsInvalidAndSingularMoments) {
  constexpr Vec2 position{0.0, 0.0};
  constexpr std::array<Vec2, 3> collinear_samples{{{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}}};
  constexpr std::array weights{0.25, 0.25, 0.5};
  constexpr std::array values{1.0, 2.0, 3.0};
  EXPECT_EQ(
      FitAffine2D(position, collinear_samples, weights, values).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  constexpr std::array bad_weights{0.25, 0.25, 0.51};
  EXPECT_EQ(
      FitAffine2D(position, collinear_samples, bad_weights, values).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  constexpr std::array negative_weights{-0.25, 0.75, 0.5};
  EXPECT_EQ(
      FitAffine2D(position, collinear_samples, negative_weights, values).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  constexpr std::array nonfinite_values{1.0, kInfinity, 3.0};
  EXPECT_EQ(
      FitAffine2D(position, collinear_samples, weights, nonfinite_values).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  constexpr std::array<Vec2, 3> large_samples{{{1e200, 1e200}, {1e200, -1e200}, {-1e200, 0.0}}};
  EXPECT_EQ(FitAffine2D(position, large_samples, weights, values).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(Transfer, LocalGradientValidatesAndUsesProvidedDerivatives) {
  constexpr std::array values{2.0, 3.0};
  constexpr std::array<Vec2, 2> gradients{{{-1.5, 4.0}, {2.5, -1.0}}};
  const auto result = VelocityGradient2D(values, gradients);
  ASSERT_TRUE(result.ok()) << result.status();
  ExpectVec(*result, {4.5, 5.0});

  EXPECT_EQ(VelocityGradient2D(values, {}).status().code(), absl::StatusCode::kInvalidArgument);
  constexpr std::array<Vec2, 2> nonfinite_gradients{{{-1.5, 4.0}, {kInfinity, -1.0}}};
  EXPECT_EQ(VelocityGradient2D(values, nonfinite_gradients).status().code(), absl::StatusCode::kInvalidArgument);
  constexpr std::array large_values{2.0, std::numeric_limits<double>::max()};
  EXPECT_EQ(VelocityGradient2D(large_values, gradients).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(Transfer, HistoryRejectsInvalidInputAndOverflow) {
  constexpr Vec2 velocity{2.0, 3.0};
  EXPECT_EQ(
      BlendHistory2D(velocity, velocity, 1.1, MaterialPhase::kLiquid, 1.0).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      BlendHistory2D(velocity, velocity, 1.0, MaterialPhase::kLiquid, -0.1).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      BlendHistory2D(velocity, velocity, 1.0, static_cast<MaterialPhase>(-1), 1.0).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      BlendHistory2D(velocity, {kInfinity, 3.0}, 1.0, MaterialPhase::kGas, 1.0).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  constexpr Vec2 large_velocity{std::numeric_limits<double>::max(), 0.0};
  EXPECT_EQ(
      BlendHistory2D(large_velocity, large_velocity, 1.0, MaterialPhase::kLiquid, 1.0).status().code(),
      absl::StatusCode::kOutOfRange
  );
}

}  // namespace
}  // namespace fluidity
