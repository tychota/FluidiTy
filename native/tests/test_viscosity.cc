#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "fixture_data.h"
#include "fluidity/host/viscosity.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr double kTolerance = 1e-12;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kMax = std::numeric_limits<double>::max();
constexpr double kSentinel = -31.0;

struct ViscosityResult {
  explicit ViscosityResult(std::size_t face_velocity_count) : updated_face_velocities(face_velocity_count, kSentinel) {
  }

  ViscosityOutputsView Outputs() {
    return {.updated_face_velocities = updated_face_velocities, .diagnostics = diagnostics};
  }

  void ExpectUnchanged() const {
    EXPECT_TRUE(std::ranges::all_of(updated_face_velocities, [](double value) {
      return value == kSentinel;
    }));
    EXPECT_EQ(diagnostics.method, ViscosityMethod::kImplicit);
    EXPECT_EQ(diagnostics.spectral_bound, kSentinel);
    EXPECT_EQ(diagnostics.energy_before, kSentinel);
    EXPECT_EQ(diagnostics.energy_after, kSentinel);
    EXPECT_EQ(diagnostics.wall_work, kSentinel);
    EXPECT_EQ(diagnostics.numerical_dissipation, kSentinel);
  }

  std::vector<double> updated_face_velocities;
  ViscosityDiagnostics diagnostics{
      .method = ViscosityMethod::kImplicit,
      .spectral_bound = kSentinel,
      .energy_before = kSentinel,
      .energy_after = kSentinel,
      .wall_work = kSentinel,
      .numerical_dissipation = kSentinel,
  };
};

struct TwoFaceCase {
  ViscosityInputsView Inputs() const {
    return {
        .face_velocities = face_velocities,
        .face_inertias = face_inertias,
        .strain_operator = strain_operator,
        .strain_row_count = 1,
        .strain_quadrature_weights = strain_quadrature_weights,
        .boundary_strain = boundary_strain,
        .dt = 0.1,
    };
  }

  std::array<double, 2> face_velocities{1.0, -1.0};
  std::array<double, 2> face_inertias{0.01, 1.0};
  std::array<double, 2> strain_operator{1.0, -1.0};
  std::array<double, 1> strain_quadrature_weights{1.0};
  std::array<double, 1> boundary_strain{0.0};
};

std::span<const double> FixtureFloats(const fixtures::FixtureRecord& fixture, std::string_view field) {
  const auto record = fixtures::FindArray(fixture, field);
  if (!record.ok()) {
    ADD_FAILURE() << record.status();
    return {};
  }
  if ((*record)->dtype != fixtures::DType::kFloat64) {
    ADD_FAILURE() << field << " must contain float64 data";
    return {};
  }
  return (*record)->f64_data;
}

TEST(Viscosity, StiffFixture) {
  const auto fixtures = fixtures::AllFixtures();
  const auto fixture = std::ranges::find(fixtures, "stiff_viscosity", &fixtures::FixtureRecord::name);
  ASSERT_NE(fixture, fixtures.end());
  const auto strain_operator_record = fixtures::FindArray(*fixture, "strain_operator");
  ASSERT_TRUE(strain_operator_record.ok()) << strain_operator_record.status();
  ASSERT_EQ((*strain_operator_record)->dtype, fixtures::DType::kFloat64);
  ASSERT_EQ((*strain_operator_record)->shape.size(), 2);
  const auto dt = FixtureFloats(*fixture, "dt");
  ASSERT_EQ(dt.size(), 1);

  const ViscosityInputsView inputs{
      .face_velocities = FixtureFloats(*fixture, "velocity"),
      .face_inertias = FixtureFloats(*fixture, "mass"),
      .strain_operator = (*strain_operator_record)->f64_data,
      .strain_row_count = (*strain_operator_record)->shape[0],
      .strain_quadrature_weights = FixtureFloats(*fixture, "strain_weight"),
      .boundary_strain = FixtureFloats(*fixture, "boundary_strain"),
      .dt = dt.front(),
  };
  ASSERT_EQ(inputs.face_velocities.size(), (*strain_operator_record)->shape[1]);
  ViscosityResult result(inputs.face_velocities.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kImplicit);

  const auto expected = FixtureFloats(*fixture, "expected_velocity");
  ASSERT_EQ(expected.size(), 2);
  ASSERT_EQ(result.updated_face_velocities.size(), expected.size());
  for (std::size_t column = 0; column < expected.size(); ++column) {
    const double tolerance = fixture->atol + fixture->rtol * std::abs(expected[column]);
    EXPECT_NEAR(result.updated_face_velocities[column], expected[column], tolerance);
  }
}

TEST(Viscosity, MacStrainRowsAndQuadrature) {
  const Grid2D grid{2, 2, 0.5};
  const auto shape = MacStrainExtents2D(grid);
  ASSERT_TRUE(shape.ok()) << shape.status();
  EXPECT_EQ(shape->strain_row_count, 9);
  EXPECT_EQ(shape->face_velocity_count, 12);
  ASSERT_EQ(shape->coefficient_count, 108);

  std::vector<double> strain_operator(shape->coefficient_count, kSentinel);
  std::vector<double> strain_quadrature_weights(shape->strain_row_count, kSentinel);
  const auto status = AssembleMacStrain2D(grid, 3.0, strain_operator, strain_quadrature_weights);
  ASSERT_TRUE(status.ok()) << status;

  std::array<double, 108> expected{};
  expected[0 * 12 + 0] = -2.0;
  expected[0 * 12 + 1] = 2.0;
  expected[1 * 12 + 1] = -2.0;
  expected[1 * 12 + 2] = 2.0;
  expected[2 * 12 + 3] = -2.0;
  expected[2 * 12 + 4] = 2.0;
  expected[3 * 12 + 4] = -2.0;
  expected[3 * 12 + 5] = 2.0;
  expected[4 * 12 + 6] = -2.0;
  expected[4 * 12 + 8] = 2.0;
  expected[5 * 12 + 7] = -2.0;
  expected[5 * 12 + 9] = 2.0;
  expected[6 * 12 + 8] = -2.0;
  expected[6 * 12 + 10] = 2.0;
  expected[7 * 12 + 9] = -2.0;
  expected[7 * 12 + 11] = 2.0;
  expected[8 * 12 + 1] = -1.0;
  expected[8 * 12 + 4] = 1.0;
  expected[8 * 12 + 8] = -1.0;
  expected[8 * 12 + 9] = 1.0;
  EXPECT_TRUE(std::ranges::equal(strain_operator, expected));
  constexpr std::array expected_weight{1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3.0};
  EXPECT_TRUE(std::ranges::equal(strain_quadrature_weights, expected_weight));
}

TEST(Viscosity, MacStrainOfRigidMotionAndCouette) {
  const Grid2D grid{3, 2, 0.5, {0.25, -0.75}};
  const auto shape = MacStrainExtents2D(grid);
  ASSERT_TRUE(shape.ok()) << shape.status();
  std::vector<double> strain_operator(shape->coefficient_count);
  std::vector<double> strain_quadrature_weights(shape->strain_row_count);
  const auto status = AssembleMacStrain2D(grid, 1.0, strain_operator, strain_quadrature_weights);
  ASSERT_TRUE(status.ok()) << status;

  std::vector<double> translation(shape->face_velocity_count);
  std::vector<double> rotation(shape->face_velocity_count);
  std::vector<double> couette(shape->face_velocity_count);
  const std::size_t x_faces = 8;
  for (std::size_t face_index = 0; face_index < shape->face_velocity_count; ++face_index) {
    const auto face = FaceGeometry(grid, face_index);
    ASSERT_TRUE(face.ok()) << face.status();
    const bool is_x_normal = face_index < x_faces;
    translation[face_index] = is_x_normal ? 1.0 : -2.0;
    rotation[face_index] = is_x_normal ? -3.0 * face->world_position.y : 3.0 * face->world_position.x;
    couette[face_index] = is_x_normal ? 2.0 * face->world_position.y : 0.0;
  }
  for (std::size_t row = 0; row < shape->strain_row_count; ++row) {
    SCOPED_TRACE(row);
    const auto coefficients =
        std::span(strain_operator).subspan(row * shape->face_velocity_count, shape->face_velocity_count);
    EXPECT_NEAR(
        std::inner_product(coefficients.begin(), coefficients.end(), translation.begin(), 0.0),
        0.0,
        kTolerance
    );
    EXPECT_NEAR(std::inner_product(coefficients.begin(), coefficients.end(), rotation.begin(), 0.0), 0.0, kTolerance);
    const double expected_shear = row < 12 ? 0.0 : 1.0;
    EXPECT_NEAR(
        std::inner_product(coefficients.begin(), coefficients.end(), couette.begin(), 0.0),
        expected_shear,
        kTolerance
    );
  }
}

TEST(Viscosity, TensorCouplesBothFaceOrientationsSymmetrically) {
  const Grid2D grid{2, 2, 0.5};
  const auto shape = MacStrainExtents2D(grid);
  ASSERT_TRUE(shape.ok()) << shape.status();
  std::vector<double> strain_operator(shape->coefficient_count);
  std::vector<double> strain_quadrature_weights(shape->strain_row_count);
  const auto strain_status = AssembleMacStrain2D(grid, 3.0, strain_operator, strain_quadrature_weights);
  ASSERT_TRUE(strain_status.ok()) << strain_status;
  std::vector<double> face_velocities(shape->face_velocity_count);
  const std::vector<double> face_inertias(shape->face_velocity_count, 1.0);
  const std::vector<double> boundary_strain(shape->strain_row_count);
  const ViscosityInputsView inputs{
      .face_velocities = face_velocities,
      .face_inertias = face_inertias,
      .strain_operator = strain_operator,
      .strain_row_count = shape->strain_row_count,
      .strain_quadrature_weights = strain_quadrature_weights,
      .boundary_strain = boundary_strain,
      .dt = 0.01,
      .mode = ViscosityMode::kExplicit,
  };

  ViscosityResult result(shape->face_velocity_count);
  ViscosityWorkspace workspace;
  std::vector<double> stiffness(shape->face_velocity_count * shape->face_velocity_count);
  for (std::size_t column = 0; column < shape->face_velocity_count; ++column) {
    face_velocities[column] = 1.0;
    const auto status = ViscousStep(inputs, result.Outputs(), workspace);
    ASSERT_TRUE(status.ok()) << status;
    for (std::size_t row = 0; row < shape->face_velocity_count; ++row) {
      stiffness[row * shape->face_velocity_count + column] =
          (face_velocities[row] - result.updated_face_velocities[row]) / inputs.dt;
    }
    face_velocities[column] = 0.0;
  }
  for (std::size_t row = 0; row < shape->face_velocity_count; ++row) {
    for (std::size_t column = 0; column < shape->face_velocity_count; ++column) {
      EXPECT_NEAR(
          stiffness[row * shape->face_velocity_count + column],
          stiffness[column * shape->face_velocity_count + row],
          kTolerance
      );
    }
  }
  EXPECT_NEAR(stiffness[1 * shape->face_velocity_count + 8], 3.0, kTolerance);
  EXPECT_NEAR(stiffness[1 * shape->face_velocity_count + 9], -3.0, kTolerance);
}

TEST(Viscosity, MassScaledBoundSelectsWholeImplicitStep) {
  const TwoFaceCase problem;
  ViscosityResult result(problem.face_velocities.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kImplicit);
  EXPECT_NEAR(result.diagnostics.spectral_bound, 110.0, kTolerance);
  EXPECT_NEAR(result.updated_face_velocities[0], -89.0 / 111.0, kTolerance);
  EXPECT_NEAR(result.updated_face_velocities[1], -109.0 / 111.0, kTolerance);
}

TEST(Viscosity, RejectsExplicitAboveMassScaledLimit) {
  const TwoFaceCase problem;
  auto inputs = problem.Inputs();
  inputs.mode = ViscosityMode::kExplicit;
  ViscosityResult result(problem.face_velocities.size());
  ViscosityWorkspace workspace;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kFailedPrecondition);
  result.ExpectUnchanged();
}

TEST(Viscosity, PermitsExplicitAtMassScaledLimit) {
  const TwoFaceCase problem;
  auto inputs = problem.Inputs();
  inputs.dt = 0.99 * 2.0 / 110.0;
  ViscosityResult result(problem.face_velocities.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kExplicit);
  EXPECT_NEAR(result.diagnostics.spectral_bound, 110.0, kTolerance);
  EXPECT_NEAR(result.updated_face_velocities[0], 1.0 - 200.0 * inputs.dt, kTolerance);
  EXPECT_NEAR(result.updated_face_velocities[1], -1.0 + 2.0 * inputs.dt, kTolerance);
  EXPECT_LT(result.diagnostics.energy_after, result.diagnostics.energy_before);
}

void CheckMovingWallWork(ViscosityMode mode, ViscosityMethod method, double expected_velocity) {
  const std::array face_velocities{0.25};
  const std::array face_inertias{2.0};
  const std::array strain_operator{2.0};
  const std::array strain_quadrature_weights{3.0};
  const std::array boundary_strain{1.5};
  const ViscosityInputsView inputs{
      .face_velocities = face_velocities,
      .face_inertias = face_inertias,
      .strain_operator = strain_operator,
      .strain_row_count = 1,
      .strain_quadrature_weights = strain_quadrature_weights,
      .boundary_strain = boundary_strain,
      .dt = 0.1,
      .mode = mode,
  };
  ViscosityResult result(1);
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  const auto& diagnostics = result.diagnostics;
  EXPECT_EQ(diagnostics.method, method);
  EXPECT_NEAR(result.updated_face_velocities[0], expected_velocity, kTolerance);
  EXPECT_NEAR(diagnostics.spectral_bound, 6.0, kTolerance);
  EXPECT_NEAR(diagnostics.energy_before, 0.25 * 0.25, kTolerance);
  EXPECT_NEAR(diagnostics.energy_after, expected_velocity * expected_velocity, kTolerance);
  const double increment = expected_velocity - face_velocities[0];
  EXPECT_NEAR(diagnostics.numerical_dissipation, increment * increment, kTolerance);

  const bool explicit_step = method == ViscosityMethod::kExplicit;
  const double work_velocity = explicit_step ? face_velocities[0] : expected_velocity;
  EXPECT_NEAR(diagnostics.wall_work, inputs.dt * 9.0 * work_velocity, kTolerance);
  const double material_dissipation = inputs.dt * 12.0 * work_velocity * work_velocity;
  const double increment_sign = explicit_step ? -1.0 : 1.0;
  EXPECT_NEAR(
      diagnostics.energy_after - diagnostics.energy_before + increment_sign * diagnostics.numerical_dissipation
          + material_dissipation,
      diagnostics.wall_work,
      kTolerance
  );
}

TEST(Viscosity, ExplicitMovingWallEnergyIdentity) {
  CheckMovingWallWork(ViscosityMode::kExplicit, ViscosityMethod::kExplicit, 0.55);
}

TEST(Viscosity, ImplicitMovingWallEnergyIdentity) {
  CheckMovingWallWork(ViscosityMode::kImplicit, ViscosityMethod::kImplicit, 0.4375);
}

TEST(Viscosity, ZeroViscosityPreservesVelocity) {
  const Grid2D grid{1, 1, 1.0};
  std::array<double, 8> strain_operator;
  std::array<double, 2> strain_quadrature_weights;
  const auto strain_status = AssembleMacStrain2D(grid, 0.0, strain_operator, strain_quadrature_weights);
  ASSERT_TRUE(strain_status.ok()) << strain_status;
  EXPECT_EQ(strain_quadrature_weights, (std::array{0.0, 0.0}));

  const std::array face_velocities{1.0, -2.0, 3.0, -4.0};
  const std::array face_inertias{0.5, 2.0, 3.0, 1.0};
  const std::array boundary_strain{2.0, -3.0};
  const ViscosityInputsView inputs{
      .face_velocities = face_velocities,
      .face_inertias = face_inertias,
      .strain_operator = strain_operator,
      .strain_row_count = strain_quadrature_weights.size(),
      .strain_quadrature_weights = strain_quadrature_weights,
      .boundary_strain = boundary_strain,
      .dt = 3.0,
      .mode = ViscosityMode::kImplicit,
  };
  ViscosityResult result(face_velocities.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_TRUE(std::ranges::equal(result.updated_face_velocities, face_velocities));
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kImplicit);
  EXPECT_EQ(result.diagnostics.spectral_bound, 0.0);
  EXPECT_EQ(result.diagnostics.energy_before, result.diagnostics.energy_after);
  EXPECT_EQ(result.diagnostics.wall_work, 0.0);
  EXPECT_EQ(result.diagnostics.numerical_dissipation, 0.0);
}

TEST(Viscosity, ZeroTimeStepPreservesVelocity) {
  TwoFaceCase problem;
  problem.boundary_strain[0] = 9.0;
  auto inputs = problem.Inputs();
  inputs.dt = 0.0;
  inputs.mode = ViscosityMode::kImplicit;
  ViscosityResult result(problem.face_velocities.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_TRUE(std::ranges::equal(result.updated_face_velocities, problem.face_velocities));
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kImplicit);
  EXPECT_NEAR(result.diagnostics.spectral_bound, 110.0, kTolerance);
  EXPECT_EQ(result.diagnostics.energy_before, result.diagnostics.energy_after);
  EXPECT_EQ(result.diagnostics.wall_work, 0.0);
  EXPECT_EQ(result.diagnostics.numerical_dissipation, 0.0);
}

TEST(Viscosity, AcceptsOver64StrainRows) {
  const Grid2D grid{5, 5, 1.0};
  const auto shape = MacStrainExtents2D(grid);
  ASSERT_TRUE(shape.ok()) << shape.status();
  ASSERT_EQ(shape->strain_row_count, 66);
  ASSERT_EQ(shape->face_velocity_count, 60);
  ASSERT_EQ(shape->coefficient_count, 66 * 60);
  std::vector<double> strain_operator(shape->coefficient_count);
  std::vector<double> strain_quadrature_weights(shape->strain_row_count);
  const auto strain_status = AssembleMacStrain2D(grid, 1.0, strain_operator, strain_quadrature_weights);
  ASSERT_TRUE(strain_status.ok()) << strain_status;
  const std::vector<double> face_velocities(shape->face_velocity_count, 1.0);
  const std::vector<double> face_inertias(shape->face_velocity_count, 1.0);
  std::vector<double> boundary_strain(shape->strain_row_count);
  boundary_strain.back() = 1.0;
  const ViscosityInputsView inputs{
      .face_velocities = face_velocities,
      .face_inertias = face_inertias,
      .strain_operator = strain_operator,
      .strain_row_count = shape->strain_row_count,
      .strain_quadrature_weights = strain_quadrature_weights,
      .boundary_strain = boundary_strain,
      .dt = 0.01,
      .mode = ViscosityMode::kExplicit,
  };
  ViscosityResult result(shape->face_velocity_count);
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_NEAR(result.updated_face_velocities[28], 1.02, kTolerance);
  EXPECT_NEAR(result.updated_face_velocities[54], 1.02, kTolerance);
}

TEST(Viscosity, RejectsOver64VelocityUnknowns) {
  const auto largest = MacStrainExtents2D(Grid2D{1, 21, 1.0});
  ASSERT_TRUE(largest.ok()) << largest.status();
  EXPECT_EQ(largest->face_velocity_count, 64);
  const Grid2D unsupported{6, 6, 1.0};
  EXPECT_EQ(MacStrainExtents2D(unsupported).status().code(), absl::StatusCode::kUnimplemented);

  const std::array<double, 65> face_velocities{};
  std::array<double, 65> face_inertias;
  face_inertias.fill(1.0);
  const ViscosityInputsView inputs{
      .face_velocities = face_velocities,
      .face_inertias = face_inertias,
      .strain_operator = {},
      .strain_row_count = 0,
      .strain_quadrature_weights = {},
      .boundary_strain = {},
      .dt = 0.1,
  };
  ViscosityResult result(face_velocities.size());
  ViscosityWorkspace workspace;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kUnimplemented);
  result.ExpectUnchanged();
}

TEST(Viscosity, RejectsInvalidMacStrainInputsWithoutWriting) {
  std::array<double, 8> strain_operator;
  strain_operator.fill(kSentinel);
  std::array strain_quadrature_weights{kSentinel, kSentinel};
  const Grid2D grid{1, 1, 1.0};
  EXPECT_EQ(
      AssembleMacStrain2D(grid, -1.0, strain_operator, strain_quadrature_weights).code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      AssembleMacStrain2D(grid, kInfinity, strain_operator, strain_quadrature_weights).code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      AssembleMacStrain2D(Grid2D{0, 1, 1.0}, 1.0, strain_operator, strain_quadrature_weights).code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      AssembleMacStrain2D(Grid2D{1, 1, kMax}, 1.0, strain_operator, strain_quadrature_weights).code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(
      AssembleMacStrain2D(grid, kMax, strain_operator, strain_quadrature_weights).code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(
      AssembleMacStrain2D(grid, 1.0, strain_operator, std::span(strain_quadrature_weights).first(1)).code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      AssembleMacStrain2D(grid, 1.0, strain_operator, std::span(strain_operator).last(2)).code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(strain_quadrature_weights, (std::array{kSentinel, kSentinel}));
  EXPECT_TRUE(std::ranges::all_of(strain_operator, [](double value) {
    return value == kSentinel;
  }));
}

TEST(Viscosity, MacQuadratureRetainsRepresentableWeight) {
  const Grid2D grid{1, 1, 0.5};
  std::array<double, 8> strain_operator;
  std::array<double, 2> strain_quadrature_weights;
  const auto status = AssembleMacStrain2D(grid, kMax, strain_operator, strain_quadrature_weights);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(strain_quadrature_weights, (std::array{kMax / 2.0, kMax / 2.0}));
}

TEST(Viscosity, RejectsInvalidViscousStepInputsWithoutWriting) {
  TwoFaceCase problem;
  ViscosityResult result(problem.face_velocities.size());
  ViscosityWorkspace workspace;
  auto inputs = problem.Inputs();

  problem.face_velocities[0] = kInfinity;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  problem.face_velocities[0] = 1.0;
  problem.face_inertias[0] = 0.0;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  problem.face_inertias[0] = 1.0;
  problem.strain_operator[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  problem.strain_operator[0] = 1.0;
  problem.strain_quadrature_weights[0] = -1.0;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  problem.strain_quadrature_weights[0] = 1.0;
  problem.boundary_strain[0] = kInfinity;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  problem.boundary_strain[0] = 0.0;

  inputs.dt = -1.0;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  inputs.dt = kInfinity;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  inputs = problem.Inputs();
  inputs.explicit_stability_margin = 1.0;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  inputs.explicit_stability_margin = 0.0;
  inputs.mode = static_cast<ViscosityMode>(-1);
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  inputs = problem.Inputs();
  inputs.strain_row_count = 2;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  inputs.strain_row_count = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Viscosity, RejectsWritableInputAndWorkspaceAliases) {
  TwoFaceCase problem;
  auto inputs = problem.Inputs();
  ViscosityResult result(problem.face_velocities.size());
  ViscosityWorkspace workspace;
  const ViscosityOutputsView overlapping_input{
      .updated_face_velocities = problem.face_velocities,
      .diagnostics = result.diagnostics,
  };
  EXPECT_EQ(ViscousStep(inputs, overlapping_input, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(problem.face_velocities, (std::array{1.0, -1.0}));

  auto scratch_tail = std::span(workspace.solve.working_rhs).last(2);
  std::ranges::fill(scratch_tail, kSentinel);
  const ViscosityOutputsView overlapping_workspace{
      .updated_face_velocities = scratch_tail,
      .diagnostics = result.diagnostics,
  };
  EXPECT_EQ(ViscousStep(inputs, overlapping_workspace, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(scratch_tail.front(), kSentinel);
  EXPECT_EQ(scratch_tail.back(), kSentinel);
  inputs.face_inertias = scratch_tail;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();
}

TEST(Viscosity, RejectsOutputRecordAndInputObjectAliases) {
  const std::array data{1.0};
  ViscosityInputsView inputs{
      .face_velocities = data,
      .face_inertias = data,
      .strain_operator = data,
      .strain_row_count = 1,
      .strain_quadrature_weights = data,
      .boundary_strain = data,
      .dt = 0.1,
  };
  ViscosityResult result(1);
  ViscosityWorkspace workspace;
  const ViscosityOutputsView overlapping_diagnostics{
      .updated_face_velocities = std::span(&result.diagnostics.spectral_bound, 1),
      .diagnostics = result.diagnostics,
  };
  EXPECT_EQ(ViscousStep(inputs, overlapping_diagnostics, workspace).code(), absl::StatusCode::kInvalidArgument);
  const ViscosityOutputsView overlapping_input_object{
      .updated_face_velocities = std::span(&inputs.dt, 1),
      .diagnostics = result.diagnostics,
  };
  EXPECT_EQ(ViscousStep(inputs, overlapping_input_object, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(inputs.dt, 0.1);
  result.ExpectUnchanged();
}

TEST(Viscosity, RejectsOverflowWithoutPublishingVelocityOrDiagnostics) {
  TwoFaceCase problem;
  ViscosityResult result(problem.face_velocities.size());
  ViscosityWorkspace workspace;
  const auto inputs = problem.Inputs();
  problem.strain_quadrature_weights[0] = kMax;
  problem.strain_operator[0] = 2.0;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();

  problem.strain_quadrature_weights[0] = 1.0;
  problem.strain_operator[0] = 1.0;
  problem.face_velocities[0] = kMax;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Viscosity, RetainsRepresentableStiffnessAndWallLoad) {
  const std::array face_velocities{0.0};
  const std::array face_inertias{1.0};
  const std::array strain_operator{0x1p-700};
  const std::array strain_quadrature_weights{0x1p+1000};
  const std::array boundary_strain{0x1p-300};
  const ViscosityInputsView inputs{
      .face_velocities = face_velocities,
      .face_inertias = face_inertias,
      .strain_operator = strain_operator,
      .strain_row_count = 1,
      .strain_quadrature_weights = strain_quadrature_weights,
      .boundary_strain = boundary_strain,
      .dt = 1.0,
  };
  ViscosityResult result(1);
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kExplicit);
  EXPECT_EQ(result.diagnostics.spectral_bound, 0x1p-400);
  EXPECT_EQ(result.updated_face_velocities[0], 1.0);
}

TEST(Viscosity, ExplicitStepRetainsRepresentableVelocityIncrements) {
  constexpr std::array stiffness_exponents{-1000, 1000};
  for (const int exponent : stiffness_exponents) {
    SCOPED_TRACE(exponent);
    const std::array face_velocities{std::ldexp(1.0, exponent / 10)};
    const std::array face_inertias{1.0};
    const std::array strain_operator{1.0};
    const std::array strain_quadrature_weights{std::ldexp(1.0, exponent)};
    const std::array boundary_strain{0.0};
    const ViscosityInputsView inputs{
        .face_velocities = face_velocities,
        .face_inertias = face_inertias,
        .strain_operator = strain_operator,
        .strain_row_count = 1,
        .strain_quadrature_weights = strain_quadrature_weights,
        .boundary_strain = boundary_strain,
        .dt = std::ldexp(1.0, -exponent),
        .mode = ViscosityMode::kExplicit,
    };
    ViscosityResult result(1);
    ViscosityWorkspace workspace;
    const auto status = ViscousStep(inputs, result.Outputs(), workspace);
    EXPECT_TRUE(status.ok()) << status;
    if (!status.ok()) {
      continue;
    }
    EXPECT_EQ(result.diagnostics.spectral_bound, strain_quadrature_weights[0]);
    EXPECT_EQ(result.updated_face_velocities[0], 0.0);
    EXPECT_EQ(result.diagnostics.energy_after, 0.0);
    EXPECT_EQ(result.diagnostics.numerical_dissipation, result.diagnostics.energy_before);
  }
}

void CheckImplicitVelocityRange(
    double old_velocity,
    double quadrature,
    double wall_strain,
    double dt,
    double expected_velocity
) {
  const std::array face_velocities{old_velocity};
  const std::array face_inertias{0x1p-1000};
  const std::array strain_operator{1.0};
  const std::array strain_quadrature_weights{quadrature};
  const std::array boundary_strain{wall_strain};
  const ViscosityInputsView inputs{
      .face_velocities = face_velocities,
      .face_inertias = face_inertias,
      .strain_operator = strain_operator,
      .strain_row_count = 1,
      .strain_quadrature_weights = strain_quadrature_weights,
      .boundary_strain = boundary_strain,
      .dt = dt,
      .mode = ViscosityMode::kImplicit,
  };
  ViscosityResult result(1);
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kImplicit);
  EXPECT_NEAR(result.updated_face_velocities[0] / expected_velocity, 1.0, kTolerance);
}

TEST(Viscosity, ImplicitStepRetainsWallCancellationResidual) {
  constexpr double sqrt_mass = 0x1.0000008p+0;
  const std::array face_inertias{sqrt_mass * sqrt_mass};
  const std::array strain_operator{1.0};
  const std::array strain_quadrature_weights{1.0};
  constexpr std::array velocity_exponents{0, 80};
  for (const int exponent : velocity_exponents) {
    SCOPED_TRACE(exponent);
    const std::array face_velocities{std::ldexp(0x1.ffffffffffffep-1, exponent)};
    const std::array boundary_strain{std::ldexp(-0x1.0000007ffffffp+0, exponent)};
    const ViscosityInputsView inputs{
        .face_velocities = face_velocities,
        .face_inertias = face_inertias,
        .strain_operator = strain_operator,
        .strain_row_count = 1,
        .strain_quadrature_weights = strain_quadrature_weights,
        .boundary_strain = boundary_strain,
        .dt = sqrt_mass,
        .mode = ViscosityMode::kImplicit,
    };
    ViscosityResult result(1);
    ViscosityWorkspace workspace;
    const auto status = ViscousStep(inputs, result.Outputs(), workspace);
    ASSERT_TRUE(status.ok()) << status;
    // The rounded wall term cancels the leading product bits. The exact
    // numerator is -2^(exponent-77) after dividing the equation by sqrt_mass.
    const double expected_velocity = std::ldexp(-0x1.ffffff8000002p-79, exponent);
    EXPECT_NEAR(result.updated_face_velocities[0] / expected_velocity, 1.0, kTolerance);
  }
}

TEST(Viscosity, ImplicitStepRetainsOldVelocityWithTinyMass) {
  CheckImplicitVelocityRange(0x1p-600, 0x1p-1000, 0.0, 1.0, 0x1p-601);
}

TEST(Viscosity, ImplicitStepRetainsTinyWallLiftWithTinyMass) {
  CheckImplicitVelocityRange(0.0, 0x1p-1000, 1.0, 0x1p-1000, 0x1p-1000);
}

TEST(Viscosity, ImplicitStepRetainsSolutionWithTinyMass) {
  CheckImplicitVelocityRange(1.0, 0x1p-400, 0.0, 1.0, 0x1p-600);
}

TEST(Viscosity, ImplicitStepRetainsVelocitiesAcrossMassScales) {
  const std::array face_velocities{0x1p-600, 0x1p-500};
  const std::array face_inertias{0x1p-1000, 0x1p+1000};
  const std::array strain_operator{1.0, 0.0, 0.0, 1.0};
  const std::array boundary_strain{0.0, 0.0};
  const ViscosityInputsView inputs{
      .face_velocities = face_velocities,
      .face_inertias = face_inertias,
      .strain_operator = strain_operator,
      .strain_row_count = 2,
      .strain_quadrature_weights = face_inertias,
      .boundary_strain = boundary_strain,
      .dt = 1.0,
      .mode = ViscosityMode::kImplicit,
  };
  ViscosityResult result(face_velocities.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kImplicit);
  EXPECT_NEAR(result.updated_face_velocities[0] / 0x1p-601, 1.0, kTolerance);
  EXPECT_NEAR(result.updated_face_velocities[1] / 0x1p-501, 1.0, kTolerance);
}

TEST(Viscosity, AcceptsZeroImplicitVelocityWithBalancedCoupling) {
  const std::array face_velocities{2.0, 5.0};
  const std::array face_inertias{1.0, 1.0};
  const std::array strain_operator{1.0, 1.0, 1.0, 0.0, 0.0, 1.0};
  const std::array strain_quadrature_weights{2.0, 1.0, 2.0};
  const std::array boundary_strain{0.0, 0.0, 0.0};
  const ViscosityInputsView inputs{
      .face_velocities = face_velocities,
      .face_inertias = face_inertias,
      .strain_operator = strain_operator,
      .strain_row_count = 3,
      .strain_quadrature_weights = strain_quadrature_weights,
      .boundary_strain = boundary_strain,
      .dt = 1.0,
      .mode = ViscosityMode::kImplicit,
  };
  // A = [[4, 2], [2, 5]] has an exact Cholesky factor. The second face
  // supplies the entire first equation even though its RHS is nonzero.
  ViscosityResult result(face_velocities.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.updated_face_velocities[0], 0.0);
  EXPECT_EQ(result.updated_face_velocities[1], 1.0);
}

TEST(Viscosity, RejectsImplicitSolutionLostAcrossScales) {
  const std::array face_velocities{1.0, 0.0};
  const std::array face_inertias{0x1p-1000, 0x1p-200};
  const std::array strain_operator{1.0, 0.0, 0.0, 1.0};
  const std::array strain_quadrature_weights{1.0, 0x1p+800};
  const std::array boundary_strain{0.0, 0x1p-200};
  const ViscosityInputsView inputs{
      .face_velocities = face_velocities,
      .face_inertias = face_inertias,
      .strain_operator = strain_operator,
      .strain_row_count = 2,
      .strain_quadrature_weights = strain_quadrature_weights,
      .boundary_strain = boundary_strain,
      .dt = 1.0,
      .mode = ViscosityMode::kImplicit,
  };
  // Both equations have the same mass-scaled diagonal. In physical units
  // the expected velocities round to 2^-1000 and 2^-200. A common shift
  // that keeps the second RHS finite cannot keep the first solution normal.
  ViscosityResult result(face_velocities.size());
  ViscosityWorkspace workspace;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Viscosity, RejectsRoundedSubnormalStiffness) {
  TwoFaceCase problem;
  problem.strain_quadrature_weights[0] = std::numeric_limits<double>::min();
  problem.strain_operator[0] = 0.5;
  ViscosityResult result(problem.face_velocities.size());
  ViscosityWorkspace workspace;
  EXPECT_EQ(ViscousStep(problem.Inputs(), result.Outputs(), workspace).code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

}  // namespace
}  // namespace fluidity
