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

struct Result {
  explicit Result(std::size_t columns) : velocity(columns, kSentinel) {
  }

  ViscosityOutputsView Outputs() {
    return {.velocity = velocity, .diagnostics = diagnostics};
  }

  void ExpectUnchanged() const {
    EXPECT_TRUE(std::ranges::all_of(velocity, [](double value) {
      return value == kSentinel;
    }));
    EXPECT_EQ(diagnostics.method, ViscosityMethod::kImplicit);
    EXPECT_EQ(diagnostics.spectral_bound, kSentinel);
    EXPECT_EQ(diagnostics.energy_before, kSentinel);
    EXPECT_EQ(diagnostics.energy_after, kSentinel);
    EXPECT_EQ(diagnostics.wall_work, kSentinel);
    EXPECT_EQ(diagnostics.numerical_dissipation, kSentinel);
  }

  std::vector<double> velocity;
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
        .velocity = velocity,
        .mass = mass,
        .strain = strain,
        .rows = 1,
        .weight = weight,
        .boundary_strain = boundary_strain,
        .dt = 0.1,
    };
  }

  std::array<double, 2> velocity{1.0, -1.0};
  std::array<double, 2> mass{0.01, 1.0};
  std::array<double, 2> strain{1.0, -1.0};
  std::array<double, 1> weight{1.0};
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
  const auto strain = fixtures::FindArray(*fixture, "strain_operator");
  ASSERT_TRUE(strain.ok()) << strain.status();
  ASSERT_EQ((*strain)->dtype, fixtures::DType::kFloat64);
  ASSERT_EQ((*strain)->shape.size(), 2);
  const auto dt = FixtureFloats(*fixture, "dt");
  ASSERT_EQ(dt.size(), 1);

  const ViscosityInputsView inputs{
      .velocity = FixtureFloats(*fixture, "velocity"),
      .mass = FixtureFloats(*fixture, "mass"),
      .strain = (*strain)->f64_data,
      .rows = (*strain)->shape[0],
      .weight = FixtureFloats(*fixture, "strain_weight"),
      .boundary_strain = FixtureFloats(*fixture, "boundary_strain"),
      .dt = dt.front(),
  };
  ASSERT_EQ(inputs.velocity.size(), (*strain)->shape[1]);
  Result result(inputs.velocity.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kImplicit);

  const auto expected = FixtureFloats(*fixture, "expected_velocity");
  ASSERT_EQ(expected.size(), 2);
  for (std::size_t column = 0; column < expected.size(); ++column) {
    const double tolerance = fixture->atol + fixture->rtol * std::abs(expected[column]);
    EXPECT_NEAR(result.velocity[column], expected[column], tolerance);
  }
}

TEST(Viscosity, MacStrainRowsAndQuadrature) {
  const Grid2D grid{2, 2, 0.5};
  const auto shape = MacStrainShape2D(grid);
  ASSERT_TRUE(shape.ok()) << shape.status();
  EXPECT_EQ(shape->rows, 9);
  EXPECT_EQ(shape->columns, 12);
  ASSERT_EQ(shape->coefficients, 108);

  std::vector<double> strain(shape->coefficients, kSentinel);
  std::vector<double> weight(shape->rows, kSentinel);
  const auto status = MacStrain2D(grid, 3.0, strain, weight);
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
  EXPECT_TRUE(std::ranges::equal(strain, expected));
  constexpr std::array expected_weight{1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3.0};
  EXPECT_TRUE(std::ranges::equal(weight, expected_weight));
}

TEST(Viscosity, MacStrainOfRigidMotionAndCouette) {
  const Grid2D grid{3, 2, 0.5, {0.25, -0.75}};
  const auto shape = MacStrainShape2D(grid);
  ASSERT_TRUE(shape.ok()) << shape.status();
  std::vector<double> strain(shape->coefficients);
  std::vector<double> weight(shape->rows);
  const auto status = MacStrain2D(grid, 1.0, strain, weight);
  ASSERT_TRUE(status.ok()) << status;

  std::vector<double> translation(shape->columns);
  std::vector<double> rotation(shape->columns);
  std::vector<double> couette(shape->columns);
  const std::size_t x_faces = 8;
  for (std::size_t face_index = 0; face_index < shape->columns; ++face_index) {
    const auto face = FaceGeometry(grid, face_index);
    ASSERT_TRUE(face.ok()) << face.status();
    const bool is_x_normal = face_index < x_faces;
    translation[face_index] = is_x_normal ? 1.0 : -2.0;
    rotation[face_index] = is_x_normal ? -3.0 * face->position.y : 3.0 * face->position.x;
    couette[face_index] = is_x_normal ? 2.0 * face->position.y : 0.0;
  }
  for (std::size_t row = 0; row < shape->rows; ++row) {
    SCOPED_TRACE(row);
    const auto coefficients = std::span(strain).subspan(row * shape->columns, shape->columns);
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
  const auto shape = MacStrainShape2D(grid);
  ASSERT_TRUE(shape.ok()) << shape.status();
  std::vector<double> strain(shape->coefficients);
  std::vector<double> weight(shape->rows);
  const auto strain_status = MacStrain2D(grid, 3.0, strain, weight);
  ASSERT_TRUE(strain_status.ok()) << strain_status;
  std::vector<double> velocity(shape->columns);
  const std::vector<double> mass(shape->columns, 1.0);
  const std::vector<double> wall(shape->rows);
  const ViscosityInputsView inputs{
      .velocity = velocity,
      .mass = mass,
      .strain = strain,
      .rows = shape->rows,
      .weight = weight,
      .boundary_strain = wall,
      .dt = 0.01,
      .mode = ViscosityMode::kExplicit,
  };

  Result result(shape->columns);
  ViscosityWorkspace workspace;
  std::vector<double> stiffness(shape->columns * shape->columns);
  for (std::size_t column = 0; column < shape->columns; ++column) {
    velocity[column] = 1.0;
    const auto status = ViscousStep(inputs, result.Outputs(), workspace);
    ASSERT_TRUE(status.ok()) << status;
    for (std::size_t row = 0; row < shape->columns; ++row) {
      stiffness[row * shape->columns + column] = (velocity[row] - result.velocity[row]) / inputs.dt;
    }
    velocity[column] = 0.0;
  }
  for (std::size_t row = 0; row < shape->columns; ++row) {
    for (std::size_t column = 0; column < shape->columns; ++column) {
      EXPECT_NEAR(stiffness[row * shape->columns + column], stiffness[column * shape->columns + row], kTolerance);
    }
  }
  EXPECT_NEAR(stiffness[1 * shape->columns + 8], 3.0, kTolerance);
  EXPECT_NEAR(stiffness[1 * shape->columns + 9], -3.0, kTolerance);
}

TEST(Viscosity, MassScaledBoundSelectsWholeImplicitStep) {
  const TwoFaceCase problem;
  Result result(problem.velocity.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kImplicit);
  EXPECT_NEAR(result.diagnostics.spectral_bound, 110.0, kTolerance);
  EXPECT_NEAR(result.velocity[0], -89.0 / 111.0, kTolerance);
  EXPECT_NEAR(result.velocity[1], -109.0 / 111.0, kTolerance);
}

TEST(Viscosity, RejectsExplicitAboveMassScaledLimit) {
  const TwoFaceCase problem;
  auto inputs = problem.Inputs();
  inputs.mode = ViscosityMode::kExplicit;
  Result result(problem.velocity.size());
  ViscosityWorkspace workspace;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kFailedPrecondition);
  result.ExpectUnchanged();
}

TEST(Viscosity, PermitsExplicitAtMassScaledLimit) {
  const TwoFaceCase problem;
  auto inputs = problem.Inputs();
  inputs.dt = 0.99 * 2.0 / 110.0;
  Result result(problem.velocity.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kExplicit);
  EXPECT_NEAR(result.diagnostics.spectral_bound, 110.0, kTolerance);
  EXPECT_NEAR(result.velocity[0], 1.0 - 200.0 * inputs.dt, kTolerance);
  EXPECT_NEAR(result.velocity[1], -1.0 + 2.0 * inputs.dt, kTolerance);
  EXPECT_LT(result.diagnostics.energy_after, result.diagnostics.energy_before);
}

void CheckMovingWallWork(ViscosityMode mode, ViscosityMethod method, double expected_velocity) {
  const std::array velocity{0.25};
  const std::array mass{2.0};
  const std::array strain{2.0};
  const std::array weight{3.0};
  const std::array wall{1.5};
  const ViscosityInputsView inputs{
      .velocity = velocity,
      .mass = mass,
      .strain = strain,
      .rows = 1,
      .weight = weight,
      .boundary_strain = wall,
      .dt = 0.1,
      .mode = mode,
  };
  Result result(1);
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  const auto& diagnostics = result.diagnostics;
  EXPECT_EQ(diagnostics.method, method);
  EXPECT_NEAR(result.velocity[0], expected_velocity, kTolerance);
  EXPECT_NEAR(diagnostics.spectral_bound, 6.0, kTolerance);
  EXPECT_NEAR(diagnostics.energy_before, 0.25 * 0.25, kTolerance);
  EXPECT_NEAR(diagnostics.energy_after, expected_velocity * expected_velocity, kTolerance);
  const double increment = expected_velocity - velocity[0];
  EXPECT_NEAR(diagnostics.numerical_dissipation, increment * increment, kTolerance);

  const bool explicit_step = method == ViscosityMethod::kExplicit;
  const double work_velocity = explicit_step ? velocity[0] : expected_velocity;
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
  std::array<double, 8> strain;
  std::array<double, 2> weight;
  const auto strain_status = MacStrain2D(grid, 0.0, strain, weight);
  ASSERT_TRUE(strain_status.ok()) << strain_status;
  EXPECT_EQ(weight, (std::array{0.0, 0.0}));

  const std::array velocity{1.0, -2.0, 3.0, -4.0};
  const std::array mass{0.5, 2.0, 3.0, 1.0};
  const std::array wall{2.0, -3.0};
  const ViscosityInputsView inputs{
      .velocity = velocity,
      .mass = mass,
      .strain = strain,
      .rows = weight.size(),
      .weight = weight,
      .boundary_strain = wall,
      .dt = 3.0,
      .mode = ViscosityMode::kImplicit,
  };
  Result result(velocity.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_TRUE(std::ranges::equal(result.velocity, velocity));
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
  Result result(problem.velocity.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_TRUE(std::ranges::equal(result.velocity, problem.velocity));
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kImplicit);
  EXPECT_NEAR(result.diagnostics.spectral_bound, 110.0, kTolerance);
  EXPECT_EQ(result.diagnostics.energy_before, result.diagnostics.energy_after);
  EXPECT_EQ(result.diagnostics.wall_work, 0.0);
  EXPECT_EQ(result.diagnostics.numerical_dissipation, 0.0);
}

TEST(Viscosity, AcceptsOver64StrainRows) {
  const Grid2D grid{5, 5, 1.0};
  const auto shape = MacStrainShape2D(grid);
  ASSERT_TRUE(shape.ok()) << shape.status();
  ASSERT_EQ(shape->rows, 66);
  ASSERT_EQ(shape->columns, 60);
  ASSERT_EQ(shape->coefficients, 66 * 60);
  std::vector<double> strain(shape->coefficients);
  std::vector<double> weight(shape->rows);
  const auto strain_status = MacStrain2D(grid, 1.0, strain, weight);
  ASSERT_TRUE(strain_status.ok()) << strain_status;
  const std::vector<double> velocity(shape->columns, 1.0);
  const std::vector<double> mass(shape->columns, 1.0);
  std::vector<double> wall(shape->rows);
  wall.back() = 1.0;
  const ViscosityInputsView inputs{
      .velocity = velocity,
      .mass = mass,
      .strain = strain,
      .rows = shape->rows,
      .weight = weight,
      .boundary_strain = wall,
      .dt = 0.01,
      .mode = ViscosityMode::kExplicit,
  };
  Result result(shape->columns);
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_NEAR(result.velocity[28], 1.02, kTolerance);
  EXPECT_NEAR(result.velocity[54], 1.02, kTolerance);
}

TEST(Viscosity, RejectsOver64VelocityUnknowns) {
  const auto largest = MacStrainShape2D(Grid2D{1, 21, 1.0});
  ASSERT_TRUE(largest.ok()) << largest.status();
  EXPECT_EQ(largest->columns, 64);
  const Grid2D unsupported{6, 6, 1.0};
  EXPECT_EQ(MacStrainShape2D(unsupported).status().code(), absl::StatusCode::kUnimplemented);

  const std::array<double, 65> velocity{};
  std::array<double, 65> mass;
  mass.fill(1.0);
  const ViscosityInputsView inputs{
      .velocity = velocity,
      .mass = mass,
      .strain = {},
      .rows = 0,
      .weight = {},
      .boundary_strain = {},
      .dt = 0.1,
  };
  Result result(velocity.size());
  ViscosityWorkspace workspace;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kUnimplemented);
  result.ExpectUnchanged();
}

TEST(Viscosity, RejectsInvalidMacStrainInputsWithoutWriting) {
  std::array<double, 8> strain;
  strain.fill(kSentinel);
  std::array weight{kSentinel, kSentinel};
  const Grid2D grid{1, 1, 1.0};
  EXPECT_EQ(MacStrain2D(grid, -1.0, strain, weight).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(MacStrain2D(grid, kInfinity, strain, weight).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(MacStrain2D(Grid2D{0, 1, 1.0}, 1.0, strain, weight).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(MacStrain2D(Grid2D{1, 1, kMax}, 1.0, strain, weight).code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(MacStrain2D(grid, kMax, strain, weight).code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(MacStrain2D(grid, 1.0, strain, std::span(weight).first(1)).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(MacStrain2D(grid, 1.0, strain, std::span(strain).last(2)).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(weight, (std::array{kSentinel, kSentinel}));
  EXPECT_TRUE(std::ranges::all_of(strain, [](double value) {
    return value == kSentinel;
  }));
}

TEST(Viscosity, MacQuadratureRetainsRepresentableWeight) {
  const Grid2D grid{1, 1, 0.5};
  std::array<double, 8> strain;
  std::array<double, 2> weight;
  const auto status = MacStrain2D(grid, kMax, strain, weight);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(weight, (std::array{kMax / 2.0, kMax / 2.0}));
}

TEST(Viscosity, RejectsInvalidViscousStepInputsWithoutWriting) {
  TwoFaceCase problem;
  Result result(problem.velocity.size());
  ViscosityWorkspace workspace;
  auto inputs = problem.Inputs();

  problem.velocity[0] = kInfinity;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  problem.velocity[0] = 1.0;
  problem.mass[0] = 0.0;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  problem.mass[0] = 1.0;
  problem.strain[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  problem.strain[0] = 1.0;
  problem.weight[0] = -1.0;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  problem.weight[0] = 1.0;
  problem.boundary_strain[0] = kInfinity;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  problem.boundary_strain[0] = 0.0;

  inputs.dt = -1.0;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  inputs.dt = kInfinity;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  inputs = problem.Inputs();
  inputs.safety = 1.0;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  inputs.safety = 0.0;
  inputs.mode = static_cast<ViscosityMode>(-1);
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  inputs = problem.Inputs();
  inputs.rows = 2;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  inputs.rows = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Viscosity, RejectsWritableInputAndWorkspaceAliases) {
  TwoFaceCase problem;
  auto inputs = problem.Inputs();
  Result result(problem.velocity.size());
  ViscosityWorkspace workspace;
  const ViscosityOutputsView overlapping_input{
      .velocity = problem.velocity,
      .diagnostics = result.diagnostics,
  };
  EXPECT_EQ(ViscousStep(inputs, overlapping_input, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(problem.velocity, (std::array{1.0, -1.0}));

  auto scratch_tail = std::span(workspace.solve.working_rhs).last(2);
  std::ranges::fill(scratch_tail, kSentinel);
  const ViscosityOutputsView overlapping_workspace{
      .velocity = scratch_tail,
      .diagnostics = result.diagnostics,
  };
  EXPECT_EQ(ViscousStep(inputs, overlapping_workspace, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(scratch_tail.front(), kSentinel);
  EXPECT_EQ(scratch_tail.back(), kSentinel);
  inputs.mass = scratch_tail;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();
}

TEST(Viscosity, RejectsOutputRecordAndInputObjectAliases) {
  const std::array data{1.0};
  ViscosityInputsView inputs{
      .velocity = data,
      .mass = data,
      .strain = data,
      .rows = 1,
      .weight = data,
      .boundary_strain = data,
      .dt = 0.1,
  };
  Result result(1);
  ViscosityWorkspace workspace;
  const ViscosityOutputsView overlapping_diagnostics{
      .velocity = std::span(&result.diagnostics.spectral_bound, 1),
      .diagnostics = result.diagnostics,
  };
  EXPECT_EQ(ViscousStep(inputs, overlapping_diagnostics, workspace).code(), absl::StatusCode::kInvalidArgument);
  const ViscosityOutputsView overlapping_input_object{
      .velocity = std::span(&inputs.dt, 1),
      .diagnostics = result.diagnostics,
  };
  EXPECT_EQ(ViscousStep(inputs, overlapping_input_object, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(inputs.dt, 0.1);
  result.ExpectUnchanged();
}

TEST(Viscosity, RejectsOverflowWithoutPublishingVelocityOrDiagnostics) {
  TwoFaceCase problem;
  Result result(problem.velocity.size());
  ViscosityWorkspace workspace;
  const auto inputs = problem.Inputs();
  problem.weight[0] = kMax;
  problem.strain[0] = 2.0;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();

  problem.weight[0] = 1.0;
  problem.strain[0] = 1.0;
  problem.velocity[0] = kMax;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Viscosity, RetainsRepresentableStiffnessAndWallLoad) {
  const std::array velocity{0.0};
  const std::array mass{1.0};
  const std::array strain{0x1p-700};
  const std::array weight{0x1p+1000};
  const std::array wall{0x1p-300};
  const ViscosityInputsView inputs{
      .velocity = velocity,
      .mass = mass,
      .strain = strain,
      .rows = 1,
      .weight = weight,
      .boundary_strain = wall,
      .dt = 1.0,
  };
  Result result(1);
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kExplicit);
  EXPECT_EQ(result.diagnostics.spectral_bound, 0x1p-400);
  EXPECT_EQ(result.velocity[0], 1.0);
}

TEST(Viscosity, ExplicitStepRetainsRepresentableVelocityIncrements) {
  constexpr std::array stiffness_exponents{-1000, 1000};
  for (const int exponent : stiffness_exponents) {
    SCOPED_TRACE(exponent);
    const std::array velocity{std::ldexp(1.0, exponent / 10)};
    const std::array mass{1.0};
    const std::array strain{1.0};
    const std::array weight{std::ldexp(1.0, exponent)};
    const std::array wall{0.0};
    const ViscosityInputsView inputs{
        .velocity = velocity,
        .mass = mass,
        .strain = strain,
        .rows = 1,
        .weight = weight,
        .boundary_strain = wall,
        .dt = std::ldexp(1.0, -exponent),
        .mode = ViscosityMode::kExplicit,
    };
    Result result(1);
    ViscosityWorkspace workspace;
    const auto status = ViscousStep(inputs, result.Outputs(), workspace);
    EXPECT_TRUE(status.ok()) << status;
    if (!status.ok()) {
      continue;
    }
    EXPECT_EQ(result.diagnostics.spectral_bound, weight[0]);
    EXPECT_EQ(result.velocity[0], 0.0);
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
  const std::array velocity{old_velocity};
  const std::array mass{0x1p-1000};
  const std::array strain{1.0};
  const std::array weight{quadrature};
  const std::array wall{wall_strain};
  const ViscosityInputsView inputs{
      .velocity = velocity,
      .mass = mass,
      .strain = strain,
      .rows = 1,
      .weight = weight,
      .boundary_strain = wall,
      .dt = dt,
      .mode = ViscosityMode::kImplicit,
  };
  Result result(1);
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kImplicit);
  EXPECT_NEAR(result.velocity[0] / expected_velocity, 1.0, kTolerance);
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
  const std::array velocity{0x1p-600, 0x1p-500};
  const std::array mass{0x1p-1000, 0x1p+1000};
  const std::array strain{1.0, 0.0, 0.0, 1.0};
  const std::array wall{0.0, 0.0};
  const ViscosityInputsView inputs{
      .velocity = velocity,
      .mass = mass,
      .strain = strain,
      .rows = 2,
      .weight = mass,
      .boundary_strain = wall,
      .dt = 1.0,
      .mode = ViscosityMode::kImplicit,
  };
  Result result(velocity.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.diagnostics.method, ViscosityMethod::kImplicit);
  EXPECT_NEAR(result.velocity[0] / 0x1p-601, 1.0, kTolerance);
  EXPECT_NEAR(result.velocity[1] / 0x1p-501, 1.0, kTolerance);
}

TEST(Viscosity, AcceptsZeroImplicitVelocityWithBalancedCoupling) {
  const std::array velocity{2.0, 5.0};
  const std::array mass{1.0, 1.0};
  const std::array strain{1.0, 1.0, 1.0, 0.0, 0.0, 1.0};
  const std::array weight{2.0, 1.0, 2.0};
  const std::array wall{0.0, 0.0, 0.0};
  const ViscosityInputsView inputs{
      .velocity = velocity,
      .mass = mass,
      .strain = strain,
      .rows = 3,
      .weight = weight,
      .boundary_strain = wall,
      .dt = 1.0,
      .mode = ViscosityMode::kImplicit,
  };
  // A = [[4, 2], [2, 5]] has an exact Cholesky factor. The second face
  // supplies the entire first equation even though its RHS is nonzero.
  Result result(velocity.size());
  ViscosityWorkspace workspace;
  const auto status = ViscousStep(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.velocity[0], 0.0);
  EXPECT_EQ(result.velocity[1], 1.0);
}

TEST(Viscosity, RejectsImplicitSolutionLostAcrossScales) {
  const std::array velocity{1.0, 0.0};
  const std::array mass{0x1p-1000, 0x1p-200};
  const std::array strain{1.0, 0.0, 0.0, 1.0};
  const std::array weight{1.0, 0x1p+800};
  const std::array wall{0.0, 0x1p-200};
  const ViscosityInputsView inputs{
      .velocity = velocity,
      .mass = mass,
      .strain = strain,
      .rows = 2,
      .weight = weight,
      .boundary_strain = wall,
      .dt = 1.0,
      .mode = ViscosityMode::kImplicit,
  };
  // Both equations have the same mass-scaled diagonal. In physical units
  // the expected velocities round to 2^-1000 and 2^-200. A common shift
  // that keeps the second RHS finite cannot keep the first solution normal.
  Result result(velocity.size());
  ViscosityWorkspace workspace;
  EXPECT_EQ(ViscousStep(inputs, result.Outputs(), workspace).code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Viscosity, RejectsRoundedSubnormalStiffness) {
  TwoFaceCase problem;
  problem.weight[0] = std::numeric_limits<double>::min();
  problem.strain[0] = 0.5;
  Result result(problem.velocity.size());
  ViscosityWorkspace workspace;
  EXPECT_EQ(ViscousStep(problem.Inputs(), result.Outputs(), workspace).code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

}  // namespace
}  // namespace fluidity
