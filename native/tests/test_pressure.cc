#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "fixture_data.h"
#include "fluidity/host/pressure.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kSentinel = -29.0;
constexpr std::size_t kCountSentinel = 103;
constexpr std::size_t kGaugeSentinel = 104;

struct Tolerances {
  double absolute;
  double relative;
};

constexpr Tolerances kHandTolerance{1e-12, 1e-12};

struct PressureCase {
  explicit PressureCase(Grid2D dimensions)
      : grid(dimensions),
        cells(static_cast<std::size_t>(grid.nx) * grid.ny),
        faces(2 * cells + grid.nx + grid.ny),
        velocity(faces),
        aperture(faces, 1.0),
        mobility(faces, 1.0),
        distance(faces, 1.0),
        wall_velocity(faces),
        dirichlet_mask(faces),
        dirichlet_pressure(faces),
        jump(faces),
        source(cells),
        fluid_volume(cells, grid.h * grid.h),
        bulk_modulus(cells, kInfinity),
        previous_pressure(cells) {
  }

  PressureInputsView Inputs() const {
    return {
        .grid = grid,
        .dt = dt,
        .velocity = velocity,
        .aperture = aperture,
        .mobility = mobility,
        .distance = distance,
        .wall_velocity = wall_velocity,
        .dirichlet_mask = dirichlet_mask,
        .dirichlet_pressure = dirichlet_pressure,
        .jump = jump,
        .source = source,
        .fluid_volume = fluid_volume,
        .bulk_modulus = bulk_modulus,
        .previous_pressure = previous_pressure,
    };
  }

  Grid2D grid;
  std::size_t cells;
  std::size_t faces;
  double dt = 1.0;
  std::vector<double> velocity;
  std::vector<double> aperture;
  std::vector<double> mobility;
  std::vector<double> distance;
  std::vector<double> wall_velocity;
  std::vector<std::uint8_t> dirichlet_mask;
  std::vector<double> dirichlet_pressure;
  std::vector<double> jump;
  std::vector<double> source;
  std::vector<double> fluid_volume;
  std::vector<double> bulk_modulus;
  std::vector<double> previous_pressure;
};

struct Result {
  explicit Result(const PressureCase& problem)
      : pressure(problem.cells, kSentinel),
        velocity(problem.faces, kSentinel),
        original_matrix(problem.cells * problem.cells, kSentinel),
        rhs(problem.cells, kSentinel),
        residual(problem.cells, kSentinel),
        constraint_defect(problem.cells, kSentinel),
        gauges(problem.cells, kGaugeSentinel) {
  }

  PressureOutputsView Outputs() {
    return {
        .pressure = pressure,
        .velocity = velocity,
        .original_matrix = original_matrix,
        .rhs = rhs,
        .residual = residual,
        .constraint_defect = constraint_defect,
        .gauges = gauges,
        .gauge_count = gauge_count,
    };
  }

  void ExpectUnchanged() const {
    for (const auto* field : {&pressure, &velocity, &original_matrix, &rhs, &residual, &constraint_defect}) {
      EXPECT_TRUE(std::ranges::all_of(*field, [](double value) {
        return value == kSentinel;
      }));
    }
    EXPECT_TRUE(std::ranges::all_of(gauges, [](std::size_t cell) {
      return cell == kGaugeSentinel;
    }));
    EXPECT_EQ(gauge_count, kCountSentinel);
  }

  std::vector<double> pressure;
  std::vector<double> velocity;
  std::vector<double> original_matrix;
  std::vector<double> rhs;
  std::vector<double> residual;
  std::vector<double> constraint_defect;
  std::vector<std::size_t> gauges;
  std::size_t gauge_count = kCountSentinel;
};

void ExpectField(
    std::span<const double> actual,
    std::span<const double> expected,
    Tolerances tolerances = kHandTolerance
) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    SCOPED_TRACE(index);
    const double tolerance = tolerances.absolute + tolerances.relative * std::abs(expected[index]);
    EXPECT_NEAR(actual[index], expected[index], tolerance);
  }
}

void ExpectGauges(const Result& result, std::span<const std::size_t> expected) {
  ASSERT_EQ(result.gauge_count, expected.size());
  EXPECT_TRUE(std::ranges::equal(std::span(result.gauges).first(expected.size()), expected));
  for (std::size_t unused : std::span(result.gauges).subspan(expected.size())) {
    EXPECT_EQ(unused, kGaugeSentinel);
  }
}

void ExpectResidualIdentity(const PressureCase& problem, const Result& result, double tolerance) {
  for (std::size_t cell = 0; cell < problem.cells; ++cell) {
    SCOPED_TRACE(cell);
    EXPECT_NEAR(result.constraint_defect[cell], -result.residual[cell] / problem.fluid_volume[cell], tolerance);
  }
}

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

void CheckFixture(
    std::string_view name,
    std::span<const double> expected_matrix,
    std::span<const double> expected_rhs,
    std::span<const std::size_t> expected_gauges
) {
  const auto records = fixtures::AllFixtures();
  const auto fixture = std::ranges::find(records, name, &fixtures::FixtureRecord::name);
  ASSERT_NE(fixture, records.end());
  const auto grid = fixtures::FindArray(*fixture, "grid");
  ASSERT_TRUE(grid.ok()) << grid.status();
  ASSERT_EQ((*grid)->dtype, fixtures::DType::kInt64);
  ASSERT_EQ((*grid)->i64_data.size(), 2);
  const auto spacing = FixtureFloats(*fixture, "spacing");
  const auto origin = FixtureFloats(*fixture, "origin");
  const auto dt = FixtureFloats(*fixture, "dt");
  ASSERT_EQ(spacing.size(), 1);
  ASSERT_EQ(origin.size(), 2);
  ASSERT_EQ(dt.size(), 1);
  PressureCase problem({
      .nx = static_cast<std::uint32_t>((*grid)->i64_data[0]),
      .ny = static_cast<std::uint32_t>((*grid)->i64_data[1]),
      .h = spacing[0],
      .origin = {origin[0], origin[1]},
  });
  problem.dt = dt[0];
  const auto velocity = FixtureFloats(*fixture, "velocity");
  ASSERT_EQ(velocity.size(), problem.faces);
  std::ranges::copy(velocity, problem.velocity.begin());

  // Materialize the reference defaults, including half-cell distance at boundaries.
  for (std::size_t face = 0; face < problem.faces; ++face) {
    const auto geometry = FaceGeometry(problem.grid, face);
    ASSERT_TRUE(geometry.ok()) << geometry.status();
    problem.distance[face] = geometry->distance;
  }
  if (name == "two_cell_pressure") {
    const auto jump = FixtureFloats(*fixture, "jump");
    ASSERT_EQ(jump.size(), problem.faces);
    std::ranges::copy(jump, problem.jump.begin());
  } else {
    const auto source = FixtureFloats(*fixture, "source");
    const auto bulk_modulus = FixtureFloats(*fixture, "bulk_modulus");
    ASSERT_EQ(source.size(), problem.cells);
    ASSERT_EQ(bulk_modulus.size(), 1);
    std::ranges::copy(source, problem.source.begin());
    std::ranges::fill(problem.bulk_modulus, bulk_modulus[0]);
  }

  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  const Tolerances tolerances{fixture->atol, fixture->rtol};
  ExpectField(result.pressure, FixtureFloats(*fixture, "expected_pressure"), tolerances);
  ExpectField(result.velocity, FixtureFloats(*fixture, "expected_velocity"), tolerances);
  ExpectField(result.original_matrix, expected_matrix, tolerances);
  ExpectField(result.rhs, expected_rhs, tolerances);
  ExpectField(result.residual, std::vector<double>(problem.cells), tolerances);
  ExpectResidualIdentity(problem, result, tolerances.absolute);
  ExpectGauges(result, expected_gauges);
}

TEST(Pressure, TwoCellFixtureUsesPositiveInternalJump) {
  CheckFixture(
      "two_cell_pressure",
      std::array{1.0, -1.0, -1.0, 1.0},
      std::array{-5.0, 5.0},
      std::array<std::size_t, 1>{0}
  );
}

TEST(Pressure, CompliantFixtureRetainsMeanPressure) {
  CheckFixture("compliant_pressure", std::array{0.5}, std::array{3.0}, {});
}

TEST(Pressure, DirichletPressureLiftsOnceAtBothBoundaryOrientations) {
  // Area 2, open area 1 and dt*mobility/distance = 2 give k = 2.
  // Each case starts with outward fluid flux 2 and outward wall flux 0.25.
  // The source balances the wall flux, leaving b = -2 + 2*7 = 12.
  for (const std::size_t face : {0, 1, 2, 3}) {
    SCOPED_TRACE(face);
    const double outward_sign = face % 2 == 0 ? -1.0 : 1.0;
    PressureCase problem({1, 1, 2.0, {10.0, -3.0}});
    problem.dt = 0.5;
    problem.aperture[face] = 0.5;
    problem.mobility[face] = 3.0;
    problem.distance[face] = 0.75;
    problem.velocity[face] = outward_sign * 2.0;
    problem.wall_velocity[face] = outward_sign * 0.25;
    problem.dirichlet_mask[face] = 1;
    problem.dirichlet_pressure[face] = 7.0;
    problem.source[0] = 0.0625;

    Result result(problem);
    PressureWorkspace workspace;
    const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
    ASSERT_TRUE(status.ok()) << status;
    ExpectField(result.original_matrix, std::array{2.0});
    ExpectField(result.rhs, std::array{12.0});
    ExpectField(result.pressure, std::array{6.0});
    ExpectField(result.velocity, std::array{0.0, 0.0, 0.0, 0.0});
    ExpectField(result.residual, std::array{0.0});
    ExpectField(result.constraint_defect, std::array{0.0});
    ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
    ExpectGauges(result, {});
  }
}

TEST(Pressure, ADirichletAnchorReachesOnlyItsConnectedCells) {
  PressureCase problem({3, 1, 1.0});
  problem.dirichlet_mask[0] = 1;
  problem.dirichlet_pressure[0] = 3.0;
  problem.aperture[2] = 0.0;
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectField(result.original_matrix, std::array{2.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0});
  ExpectField(result.rhs, std::array{3.0, 0.0, 0.0});
  ExpectField(result.pressure, std::array{3.0, 3.0, 0.0});
  ExpectField(result.velocity, std::vector<double>(problem.faces));
  ExpectField(result.residual, std::array{0.0, 0.0, 0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGauges(result, std::array<std::size_t, 1>{2});
}

TEST(Pressure, DisconnectedCellsGetIndependentGauges) {
  PressureCase problem({2, 1, 1.0});
  problem.aperture[1] = 0.0;
  problem.aperture[0] = 0.0;
  problem.dirichlet_mask[0] = 1;
  problem.dirichlet_pressure[0] = 7.0;
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectGauges(result, std::array<std::size_t, 2>{0, 1});
  ExpectField(result.original_matrix, std::array{0.0, 0.0, 0.0, 0.0});
  ExpectField(result.rhs, std::array{0.0, 0.0});
  ExpectField(result.pressure, std::array{0.0, 0.0});
}

TEST(Pressure, NonDirichletBoundaryKeepsPrescribedFlux) {
  PressureCase problem({1, 1, 1.0});
  problem.velocity[1] = 2.0;
  problem.source[0] = 2.0;
  problem.jump[1] = 4.0;
  problem.dirichlet_pressure[1] = 7.0;
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectGauges(result, std::array<std::size_t, 1>{0});
  ExpectField(result.velocity, std::array{0.0, 2.0, 0.0, 0.0});
  ExpectField(result.original_matrix, std::array{0.0});
  ExpectField(result.rhs, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
}

TEST(Pressure, DisconnectedNeumannSourcesCannotCancelGlobally) {
  PressureCase problem({2, 1, 1.0});
  problem.aperture[1] = 0.0;
  problem.source = {1.0, -1.0};
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  result.ExpectUnchanged();
}

TEST(Pressure, IncompatiblePrescribedNeumannFluxIsRejected) {
  PressureCase problem({1, 1, 1.0});
  problem.velocity[0] = 1.0;
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  result.ExpectUnchanged();
}

TEST(Pressure, BlockedMovingWallContributesFluxAtTinyVolume) {
  PressureCase problem({1, 1, 1.0});
  problem.aperture[0] = 0.0;
  problem.velocity[0] = 4.0;
  problem.wall_velocity[0] = 1.0;
  problem.fluid_volume[0] = 1e-12;
  problem.bulk_modulus[0] = 2.0;
  problem.source[0] = 3.0;

  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectField(result.original_matrix, std::array{5e-13}, {0.0, 1e-14});
  ExpectField(result.rhs, std::array{1.0 + 3e-12});
  ExpectField(result.pressure, std::array{2e12 + 6.0}, {0.0, 1e-14});
  ExpectField(result.velocity, std::array{4.0, 0.0, 0.0, 0.0});
  ExpectField(result.residual, std::array{0.0});
  ExpectField(result.constraint_defect, std::array{0.0}, {1e-3, 0.0});
  ExpectResidualIdentity(problem, result, 1e-3);
  ExpectGauges(result, {});
}

TEST(Pressure, ComplianceLiftsPreviousPressure) {
  PressureCase problem({1, 1, 1.0});
  problem.dt = 2.0;
  problem.fluid_volume[0] = 3.0;
  problem.bulk_modulus[0] = 3.0;
  problem.previous_pressure[0] = 4.0;
  problem.source[0] = 2.0;
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectField(result.original_matrix, std::array{0.5});
  ExpectField(result.rhs, std::array{8.0});
  ExpectField(result.pressure, std::array{16.0});
  ExpectField(result.residual, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGauges(result, {});
}

TEST(Pressure, ComplianceLiftSurvivesCancellationOfIntegratedSourceAndFlux) {
  PressureCase problem({1, 1, 1.0});
  problem.source[0] = 0x1p55;
  problem.velocity[1] = 0x1p55;
  problem.bulk_modulus[0] = 2.0;
  problem.previous_pressure[0] = 3.0;
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectField(result.original_matrix, std::array{0.5});
  ExpectField(result.rhs, std::array{1.5});
  ExpectField(result.pressure, std::array{3.0});
  ExpectField(result.velocity, problem.velocity);
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGauges(result, {});
}

TEST(Pressure, AcceptedCompatibilityRoundoffRemainsInOriginalResidual) {
  constexpr double kTinyVolume = 0x1p-40;
  PressureCase problem({2, 1, 1.0});
  problem.velocity[1] = 1.0;
  problem.source[0] = 1.0;
  problem.fluid_volume[0] = kTinyVolume;

  Result result(problem);
  PressureWorkspace workspace;
  // The integrated imbalance is below 1e-12 times the RHS 1-norm, but the
  // physical defect is order one because that imbalance sits in the tiny gauge cell.
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectField(result.rhs, std::array{-1.0 + kTinyVolume, 1.0}, {0.0, 0.0});
  ExpectField(result.pressure, std::array{0.0, 1.0});
  ExpectField(result.residual, std::array{kTinyVolume, 0.0}, {1e-16, 0.0});
  ExpectField(result.constraint_defect, std::array{-1.0, 0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGauges(result, std::array<std::size_t, 1>{0});
}

TEST(Pressure, CompatibilityToleranceScalesWithSmallRhs) {
  PressureCase problem({2, 1, 1.0});
  problem.velocity[1] = 1e-20;
  problem.source[0] = 1e-29;
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  result.ExpectUnchanged();
}

TEST(Pressure, RejectsInteriorDirichletAndJumpOnDirichlet) {
  PressureCase problem({2, 1, 1.0});
  Result result(problem);
  PressureWorkspace workspace;
  problem.dirichlet_mask[1] = 1;
  auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();

  problem.dirichlet_mask = {1, 0, 0, 0, 0, 0, 0};
  problem.jump[0] = 3.0;
  status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();
}

TEST(Pressure, RejectsInvalidPhysicalCoefficients) {
  PressureCase problem({1, 1, 1.0});
  Result result(problem);
  PressureWorkspace workspace;
  const auto expect_invalid = [&] {
    const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    result.ExpectUnchanged();
  };
  problem.aperture[0] = 1.01;
  expect_invalid();
  problem.aperture[0] = 1.0;
  problem.mobility[0] = kInfinity;
  expect_invalid();
  problem.mobility[0] = 1.0;
  problem.distance[0] = 0.0;
  expect_invalid();
  problem.distance[0] = 1.0;
  problem.fluid_volume[0] = 0.0;
  expect_invalid();
  problem.fluid_volume[0] = 1.0;
  problem.bulk_modulus[0] = -kInfinity;
  expect_invalid();
  problem.bulk_modulus[0] = kInfinity;
  problem.dirichlet_mask[0] = 2;
  expect_invalid();
  problem.dirichlet_mask[0] = 0;
  problem.dt = std::numeric_limits<double>::quiet_NaN();
  expect_invalid();
}

TEST(Pressure, RejectsIncorrectInputAndOutputExtents) {
  PressureCase problem({2, 1, 1.0});
  Result result(problem);
  PressureWorkspace workspace;
  auto inputs = problem.Inputs();
  auto outputs = result.Outputs();
  inputs.velocity = inputs.velocity.first(problem.faces - 1);
  auto status = SolvePressure2D(inputs, outputs, workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();

  inputs = problem.Inputs();
  outputs.original_matrix = outputs.original_matrix.first(3);
  status = SolvePressure2D(inputs, outputs, workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();

  outputs.original_matrix = result.original_matrix;
  outputs.gauges = {};
  status = SolvePressure2D(inputs, outputs, workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();
}

TEST(Pressure, AllowsSharedReadOnlyFields) {
  PressureCase problem({1, 1, 1.0});
  auto inputs = problem.Inputs();
  inputs.wall_velocity = inputs.velocity;
  inputs.jump = inputs.velocity;
  inputs.dirichlet_pressure = inputs.velocity;
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectGauges(result, std::array<std::size_t, 1>{0});
}

TEST(Pressure, RejectsAnyWritableOverlapIncludingGaugeCountAndScratchTail) {
  PressureCase problem({1, 1, 1.0});
  Result result(problem);
  PressureWorkspace workspace;
  const auto expect_overlap = [&](const PressureInputsView& inputs, PressureOutputsView outputs) {
    const auto status = SolvePressure2D(inputs, outputs, workspace);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    result.ExpectUnchanged();
  };
  auto outputs = result.Outputs();
  outputs.pressure = problem.source;
  expect_overlap(problem.Inputs(), outputs);
  EXPECT_EQ(problem.source[0], 0.0);

  outputs.pressure = outputs.rhs;
  expect_overlap(problem.Inputs(), outputs);

  outputs.pressure = std::span(workspace.pressure).last(1);
  expect_overlap(problem.Inputs(), outputs);

  auto inputs = problem.Inputs();
  inputs.source = std::span(workspace.rhs).last(1);
  expect_overlap(inputs, result.Outputs());

  const auto expect_count_overlap = [&](std::size_t& aliased_count) {
    expect_overlap(
        problem.Inputs(),
        {
            .pressure = result.pressure,
            .velocity = result.velocity,
            .original_matrix = result.original_matrix,
            .rhs = result.rhs,
            .residual = result.residual,
            .constraint_defect = result.constraint_defect,
            .gauges = result.gauges,
            .gauge_count = aliased_count,
        }
    );
  };
  expect_count_overlap(result.gauges[0]);
  expect_count_overlap(workspace.gauges.back());
}

TEST(Pressure, DirichletConductanceRetainsRepresentableSmallOpenArea) {
  PressureCase problem({1, 1, 1.0});
  problem.dt = 0x1p-600;
  problem.aperture[0] = 0x1p-600;
  problem.mobility[0] = 0x1p600;
  problem.dirichlet_mask[0] = 1;
  problem.dirichlet_pressure[0] = 0x1p600;

  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], 0x1p-600);
  EXPECT_EQ(result.rhs[0], 1.0);
  EXPECT_EQ(result.pressure[0], 0x1p600);
  ExpectField(result.velocity, std::array{0.0, 0.0, 0.0, 0.0});
  ExpectField(result.residual, std::array{0.0});
  ExpectField(result.constraint_defect, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGauges(result, {});
}

TEST(Pressure, CorrectionScaleRecoversAfterTinyProduct) {
  PressureCase problem({1, 1, 0x1p100});
  problem.dt = 0x1p-900;
  problem.mobility[0] = 0x1p-200;
  problem.distance[0] = 0x1p-100;
  problem.dirichlet_mask[0] = 1;
  problem.velocity[0] = 1.0;
  problem.fluid_volume[0] = 1.0;

  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], 0x1p-900);
  EXPECT_EQ(result.rhs[0], 0x1p100);
  EXPECT_EQ(result.pressure[0], 0x1p1000);
  ExpectField(result.velocity, std::array{0.0, 0.0, 0.0, 0.0});
  ExpectField(result.residual, std::array{0.0});
  ExpectField(result.constraint_defect, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGauges(result, {});
}

TEST(Pressure, CorrectionScaleRecoversAfterHugeProduct) {
  PressureCase problem({1, 1, 1.0});
  problem.dt = 0x1p600;
  problem.mobility[0] = 0x1p600;
  problem.distance[0] = 0x1p600;
  problem.dirichlet_mask[0] = 1;
  problem.dirichlet_pressure[0] = 1.0;

  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], 0x1p600);
  EXPECT_EQ(result.rhs[0], 0x1p600);
  EXPECT_EQ(result.pressure[0], 1.0);
  ExpectField(result.velocity, std::array{0.0, 0.0, 0.0, 0.0});
  ExpectField(result.residual, std::array{0.0});
  ExpectField(result.constraint_defect, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGauges(result, {});
}

TEST(Pressure, ConductanceUsesTheSameScaleAsVelocityCorrection) {
  constexpr double kOpenArea = 0x1.8p-475;
  PressureCase problem({1, 1, 1.0});
  problem.dt = 0x1p-600;
  problem.aperture[0] = kOpenArea;
  problem.mobility[0] = 0x1p600;
  problem.dirichlet_mask[0] = 1;
  problem.velocity[0] = 1.0;
  problem.fluid_volume[0] = kOpenArea;

  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], kOpenArea);
  EXPECT_EQ(result.rhs[0], kOpenArea);
  ExpectField(result.pressure, std::array{1.0});
  ExpectField(result.velocity, std::array{0.0, 0.0, 0.0, 0.0});
  ExpectField(result.constraint_defect, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGauges(result, {});
}

TEST(Pressure, RoundedZeroComplianceDoesNotInventAGauge) {
  PressureCase problem({1, 1, 1.0});
  std::ranges::fill(problem.aperture, 0.0);
  problem.fluid_volume[0] = 0x1p-600;
  problem.bulk_modulus[0] = 0x1p600;
  problem.previous_pressure[0] = 0x1p600;

  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Pressure, RoundedZeroActiveFaceCoefficientsAreRejected) {
  struct Coefficients {
    std::string_view name;
    double spacing;
    double dt;
    double aperture;
    double mobility;
  };
  constexpr std::array cases{
      Coefficients{"open area", 0x1p-500, 1.0, 0x1p-600, 1.0},
      Coefficients{"velocity scale", 1.0, 0x1p-1000, 1.0, 0x1p-100},
      Coefficients{"conductance", 0x1p-500, 0x1p-600, 1.0, 1.0},
  };
  for (const auto& coefficients : cases) {
    SCOPED_TRACE(coefficients.name);
    PressureCase problem({1, 1, coefficients.spacing});
    problem.dt = coefficients.dt;
    problem.aperture[0] = coefficients.aperture;
    problem.mobility[0] = coefficients.mobility;
    problem.dirichlet_mask[0] = 1;

    Result result(problem);
    PressureWorkspace workspace;
    const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
    EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
    result.ExpectUnchanged();
  }
}

TEST(Pressure, RejectsSubnormalConductanceEvenWithNormalCompliance) {
  constexpr double kFluidVolume = 0x1.8p-475;
  PressureCase problem({1, 1, 1.0});
  problem.aperture = {kFluidVolume, 0.0, 0.0, 0.0};
  problem.mobility[0] = 0x1p-600;
  problem.dirichlet_mask[0] = 1;
  problem.dirichlet_pressure[0] = 0x1p600;
  problem.fluid_volume[0] = kFluidVolume;
  problem.bulk_modulus[0] = kFluidVolume;

  // Conductance rounds from 0.75 * denorm_min to denorm_min. The normal
  // compliance keeps the resulting matrix well conditioned.
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Pressure, RejectsSubnormalComplianceEvenWithNormalConductance) {
  PressureCase problem({1, 1, 1.0});
  problem.aperture = {1.0, 0.0, 0.0, 0.0};
  problem.dirichlet_mask[0] = 1;
  problem.fluid_volume[0] = 0x1.8p-475;
  problem.bulk_modulus[0] = 0x1p600;
  problem.previous_pressure[0] = 0x1p600;

  // Compliance rounds from 0.75 * denorm_min to denorm_min. The normal
  // conductance keeps the resulting matrix well conditioned.
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Pressure, AcceptsMinimumNormalConductanceWithNormalCompliance) {
  constexpr double kFluidVolume = 0x1p-422;
  PressureCase problem({1, 1, 1.0});
  problem.aperture = {kFluidVolume, 0.0, 0.0, 0.0};
  problem.mobility[0] = 0x1p-600;
  problem.dirichlet_mask[0] = 1;
  problem.dirichlet_pressure[0] = 0x1p600;
  problem.fluid_volume[0] = kFluidVolume;
  problem.bulk_modulus[0] = kFluidVolume;

  // The conductance is exactly 2^-1022. Its Dirichlet lift is V and
  // compliance is one, so the represented system gives p = V.
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], 1.0);
  EXPECT_EQ(result.rhs[0], kFluidVolume);
  EXPECT_EQ(result.pressure[0], kFluidVolume);
  EXPECT_EQ(result.velocity[0], 1.0);
  EXPECT_EQ(result.residual[0], 0.0);
  EXPECT_EQ(result.constraint_defect[0], 0.0);
  ExpectGauges(result, {});
}

TEST(Pressure, AcceptsMinimumNormalComplianceWithNormalConductance) {
  constexpr double kFluidVolume = 0x1p-422;
  PressureCase problem({1, 1, 1.0});
  problem.aperture = {1.0, 0.0, 0.0, 0.0};
  problem.dirichlet_mask[0] = 1;
  problem.fluid_volume[0] = kFluidVolume;
  problem.bulk_modulus[0] = 0x1p600;
  problem.previous_pressure[0] = 0x1p600;

  // The compliance is exactly 2^-1022. Its previous-pressure lift is V and
  // conductance is one, so the represented system gives p = V.
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], 1.0);
  EXPECT_EQ(result.rhs[0], kFluidVolume);
  EXPECT_EQ(result.pressure[0], kFluidVolume);
  EXPECT_EQ(result.velocity[0], -kFluidVolume);
  EXPECT_EQ(result.residual[0], 0.0);
  EXPECT_EQ(result.constraint_defect[0], 0.0);
  ExpectGauges(result, {});
}

TEST(Pressure, CorrectionOverflowLeavesAllOutputsUnchanged) {
  PressureCase problem({1, 1, 1.0});
  // Finite conductance near one, but correcting the open fluid velocity overflows.
  problem.dirichlet_mask[0] = 1;
  problem.aperture[0] = 1e-308;
  problem.mobility[0] = 1e308;
  problem.source[0] = 4.0;
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Pressure, UnsafeSpdSolveLeavesAllOutputsUnchanged) {
  PressureCase problem({2, 1, 1.0});
  problem.aperture[1] = 0.0;
  problem.bulk_modulus = {1.0, 1e20};
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  result.ExpectUnchanged();
}

TEST(Pressure, SupportsSixtyFourCellsAndOneHundredNinetyThreeFaces) {
  PressureCase problem({64, 1, 1.0});
  Result result(problem);
  PressureWorkspace workspace;
  std::ranges::fill(problem.aperture, 0.0);
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.velocity.size(), 193);
  ASSERT_EQ(result.gauge_count, 64);
  for (std::size_t cell = 0; cell < result.gauge_count; ++cell) {
    EXPECT_EQ(result.gauges[cell], cell);
  }
  ExpectField(result.pressure, std::vector<double>(64));
  ExpectField(result.velocity, std::vector<double>(193));
}

TEST(Pressure, RejectsLargerSystemsBeforePublication) {
  PressureCase problem({65, 1, 1.0});
  Result result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented);
  result.ExpectUnchanged();
}

}  // namespace
}  // namespace fluidity
