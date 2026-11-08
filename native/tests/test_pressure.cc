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
        predicted_face_velocities(faces),
        face_open_fractions(faces, 1.0),
        face_inverse_densities(faces, 1.0),
        pressure_gradient_distances(faces, 1.0),
        wall_face_velocities(faces),
        dirichlet_face_mask(faces),
        dirichlet_face_pressures(faces),
        face_pressure_jumps(faces),
        target_cell_divergence(cells),
        cell_fluid_volumes(cells, grid.h * grid.h),
        cell_bulk_moduli(cells, kInfinity),
        previous_cell_pressures(cells) {
  }

  PressureInputsView Inputs() const {
    return {
        .grid = grid,
        .dt = dt,
        .predicted_face_velocities = predicted_face_velocities,
        .face_open_fractions = face_open_fractions,
        .face_inverse_densities = face_inverse_densities,
        .pressure_gradient_distances = pressure_gradient_distances,
        .wall_face_velocities = wall_face_velocities,
        .dirichlet_face_mask = dirichlet_face_mask,
        .dirichlet_face_pressures = dirichlet_face_pressures,
        .face_pressure_jumps = face_pressure_jumps,
        .target_cell_divergence = target_cell_divergence,
        .cell_fluid_volumes = cell_fluid_volumes,
        .cell_bulk_moduli = cell_bulk_moduli,
        .previous_cell_pressures = previous_cell_pressures,
    };
  }

  Grid2D grid;
  std::size_t cells;
  std::size_t faces;
  double dt = 1.0;
  std::vector<double> predicted_face_velocities;
  std::vector<double> face_open_fractions;
  std::vector<double> face_inverse_densities;
  std::vector<double> pressure_gradient_distances;
  std::vector<double> wall_face_velocities;
  std::vector<std::uint8_t> dirichlet_face_mask;
  std::vector<double> dirichlet_face_pressures;
  std::vector<double> face_pressure_jumps;
  std::vector<double> target_cell_divergence;
  std::vector<double> cell_fluid_volumes;
  std::vector<double> cell_bulk_moduli;
  std::vector<double> previous_cell_pressures;
};

struct PressureResult {
  explicit PressureResult(const PressureCase& problem)
      : cell_pressures(problem.cells, kSentinel),
        corrected_face_velocities(problem.faces, kSentinel),
        original_matrix(problem.cells * problem.cells, kSentinel),
        original_rhs(problem.cells, kSentinel),
        original_residuals(problem.cells, kSentinel),
        cell_constraint_defects(problem.cells, kSentinel),
        gauge_cell_indices(problem.cells, kGaugeSentinel) {
  }

  PressureOutputsView Outputs() {
    return {
        .cell_pressures = cell_pressures,
        .corrected_face_velocities = corrected_face_velocities,
        .original_matrix = original_matrix,
        .original_rhs = original_rhs,
        .original_residuals = original_residuals,
        .cell_constraint_defects = cell_constraint_defects,
        .gauge_cell_indices = gauge_cell_indices,
        .gauge_count = gauge_count,
    };
  }

  void ExpectUnchanged() const {
    for (const auto* field :
         {&cell_pressures,
          &corrected_face_velocities,
          &original_matrix,
          &original_rhs,
          &original_residuals,
          &cell_constraint_defects}) {
      EXPECT_TRUE(std::ranges::all_of(*field, [](double value) {
        return value == kSentinel;
      }));
    }
    EXPECT_TRUE(std::ranges::all_of(gauge_cell_indices, [](std::size_t cell) {
      return cell == kGaugeSentinel;
    }));
    EXPECT_EQ(gauge_count, kCountSentinel);
  }

  std::vector<double> cell_pressures;
  std::vector<double> corrected_face_velocities;
  std::vector<double> original_matrix;
  std::vector<double> original_rhs;
  std::vector<double> original_residuals;
  std::vector<double> cell_constraint_defects;
  std::vector<std::size_t> gauge_cell_indices;
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

void ExpectGaugeCellIndices(const PressureResult& result, std::span<const std::size_t> expected) {
  ASSERT_EQ(result.gauge_count, expected.size());
  EXPECT_TRUE(std::ranges::equal(std::span(result.gauge_cell_indices).first(expected.size()), expected));
  for (std::size_t unused : std::span(result.gauge_cell_indices).subspan(expected.size())) {
    EXPECT_EQ(unused, kGaugeSentinel);
  }
}

void ExpectResidualIdentity(const PressureCase& problem, const PressureResult& result, double tolerance) {
  for (std::size_t cell = 0; cell < problem.cells; ++cell) {
    SCOPED_TRACE(cell);
    EXPECT_NEAR(
        result.cell_constraint_defects[cell],
        -result.original_residuals[cell] / problem.cell_fluid_volumes[cell],
        tolerance
    );
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
  std::ranges::copy(velocity, problem.predicted_face_velocities.begin());

  // Materialize the reference defaults, including half-cell distance at boundaries.
  for (std::size_t face = 0; face < problem.faces; ++face) {
    const auto geometry = FaceGeometry(problem.grid, face);
    ASSERT_TRUE(geometry.ok()) << geometry.status();
    problem.pressure_gradient_distances[face] = geometry->pressure_gradient_distance;
  }
  if (name == "two_cell_pressure") {
    const auto jump = FixtureFloats(*fixture, "jump");
    ASSERT_EQ(jump.size(), problem.faces);
    std::ranges::copy(jump, problem.face_pressure_jumps.begin());
  } else {
    const auto source = FixtureFloats(*fixture, "source");
    const auto bulk_modulus = FixtureFloats(*fixture, "bulk_modulus");
    ASSERT_EQ(source.size(), problem.cells);
    ASSERT_EQ(bulk_modulus.size(), 1);
    std::ranges::copy(source, problem.target_cell_divergence.begin());
    std::ranges::fill(problem.cell_bulk_moduli, bulk_modulus[0]);
  }

  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  const Tolerances tolerances{fixture->atol, fixture->rtol};
  ExpectField(result.cell_pressures, FixtureFloats(*fixture, "expected_pressure"), tolerances);
  ExpectField(result.corrected_face_velocities, FixtureFloats(*fixture, "expected_velocity"), tolerances);
  ExpectField(result.original_matrix, expected_matrix, tolerances);
  ExpectField(result.original_rhs, expected_rhs, tolerances);
  ExpectField(result.original_residuals, std::vector<double>(problem.cells), tolerances);
  ExpectResidualIdentity(problem, result, tolerances.absolute);
  ExpectGaugeCellIndices(result, expected_gauges);
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
    problem.face_open_fractions[face] = 0.5;
    problem.face_inverse_densities[face] = 3.0;
    problem.pressure_gradient_distances[face] = 0.75;
    problem.predicted_face_velocities[face] = outward_sign * 2.0;
    problem.wall_face_velocities[face] = outward_sign * 0.25;
    problem.dirichlet_face_mask[face] = 1;
    problem.dirichlet_face_pressures[face] = 7.0;
    problem.target_cell_divergence[0] = 0.0625;

    PressureResult result(problem);
    PressureWorkspace workspace;
    const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
    ASSERT_TRUE(status.ok()) << status;
    ExpectField(result.original_matrix, std::array{2.0});
    ExpectField(result.original_rhs, std::array{12.0});
    ExpectField(result.cell_pressures, std::array{6.0});
    ExpectField(result.corrected_face_velocities, std::array{0.0, 0.0, 0.0, 0.0});
    ExpectField(result.original_residuals, std::array{0.0});
    ExpectField(result.cell_constraint_defects, std::array{0.0});
    ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
    ExpectGaugeCellIndices(result, {});
  }
}

TEST(Pressure, ADirichletAnchorReachesOnlyItsConnectedCells) {
  PressureCase problem({3, 1, 1.0});
  problem.dirichlet_face_mask[0] = 1;
  problem.dirichlet_face_pressures[0] = 3.0;
  problem.face_open_fractions[2] = 0.0;
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectField(result.original_matrix, std::array{2.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0});
  ExpectField(result.original_rhs, std::array{3.0, 0.0, 0.0});
  ExpectField(result.cell_pressures, std::array{3.0, 3.0, 0.0});
  ExpectField(result.corrected_face_velocities, std::vector<double>(problem.faces));
  ExpectField(result.original_residuals, std::array{0.0, 0.0, 0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGaugeCellIndices(result, std::array<std::size_t, 1>{2});
}

TEST(Pressure, DisconnectedCellsGetIndependentGauges) {
  PressureCase problem({2, 1, 1.0});
  problem.face_open_fractions[1] = 0.0;
  problem.face_open_fractions[0] = 0.0;
  problem.dirichlet_face_mask[0] = 1;
  problem.dirichlet_face_pressures[0] = 7.0;
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectGaugeCellIndices(result, std::array<std::size_t, 2>{0, 1});
  ExpectField(result.original_matrix, std::array{0.0, 0.0, 0.0, 0.0});
  ExpectField(result.original_rhs, std::array{0.0, 0.0});
  ExpectField(result.cell_pressures, std::array{0.0, 0.0});
}

TEST(Pressure, NonDirichletBoundaryKeepsPrescribedFlux) {
  PressureCase problem({1, 1, 1.0});
  problem.predicted_face_velocities[1] = 2.0;
  problem.target_cell_divergence[0] = 2.0;
  problem.face_pressure_jumps[1] = 4.0;
  problem.dirichlet_face_pressures[1] = 7.0;
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectGaugeCellIndices(result, std::array<std::size_t, 1>{0});
  ExpectField(result.corrected_face_velocities, std::array{0.0, 2.0, 0.0, 0.0});
  ExpectField(result.original_matrix, std::array{0.0});
  ExpectField(result.original_rhs, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
}

TEST(Pressure, DisconnectedNeumannSourcesCannotCancelGlobally) {
  PressureCase problem({2, 1, 1.0});
  problem.face_open_fractions[1] = 0.0;
  problem.target_cell_divergence = {1.0, -1.0};
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  result.ExpectUnchanged();
}

TEST(Pressure, IncompatiblePrescribedNeumannFluxIsRejected) {
  PressureCase problem({1, 1, 1.0});
  problem.predicted_face_velocities[0] = 1.0;
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  result.ExpectUnchanged();
}

TEST(Pressure, BlockedMovingWallContributesFluxAtTinyVolume) {
  PressureCase problem({1, 1, 1.0});
  problem.face_open_fractions[0] = 0.0;
  problem.predicted_face_velocities[0] = 4.0;
  problem.wall_face_velocities[0] = 1.0;
  problem.cell_fluid_volumes[0] = 1e-12;
  problem.cell_bulk_moduli[0] = 2.0;
  problem.target_cell_divergence[0] = 3.0;

  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectField(result.original_matrix, std::array{5e-13}, {0.0, 1e-14});
  ExpectField(result.original_rhs, std::array{1.0 + 3e-12});
  ExpectField(result.cell_pressures, std::array{2e12 + 6.0}, {0.0, 1e-14});
  ExpectField(result.corrected_face_velocities, std::array{4.0, 0.0, 0.0, 0.0});
  ExpectField(result.original_residuals, std::array{0.0});
  ExpectField(result.cell_constraint_defects, std::array{0.0}, {1e-3, 0.0});
  ExpectResidualIdentity(problem, result, 1e-3);
  ExpectGaugeCellIndices(result, {});
}

TEST(Pressure, ComplianceLiftsPreviousPressure) {
  PressureCase problem({1, 1, 1.0});
  problem.dt = 2.0;
  problem.cell_fluid_volumes[0] = 3.0;
  problem.cell_bulk_moduli[0] = 3.0;
  problem.previous_cell_pressures[0] = 4.0;
  problem.target_cell_divergence[0] = 2.0;
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectField(result.original_matrix, std::array{0.5});
  ExpectField(result.original_rhs, std::array{8.0});
  ExpectField(result.cell_pressures, std::array{16.0});
  ExpectField(result.original_residuals, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGaugeCellIndices(result, {});
}

TEST(Pressure, ComplianceLiftSurvivesCancellationOfIntegratedSourceAndFlux) {
  PressureCase problem({1, 1, 1.0});
  problem.target_cell_divergence[0] = 0x1p55;
  problem.predicted_face_velocities[1] = 0x1p55;
  problem.cell_bulk_moduli[0] = 2.0;
  problem.previous_cell_pressures[0] = 3.0;
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectField(result.original_matrix, std::array{0.5});
  ExpectField(result.original_rhs, std::array{1.5});
  ExpectField(result.cell_pressures, std::array{3.0});
  ExpectField(result.corrected_face_velocities, problem.predicted_face_velocities);
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGaugeCellIndices(result, {});
}

TEST(Pressure, AcceptedCompatibilityRoundoffRemainsInOriginalResidual) {
  constexpr double kTinyVolume = 0x1p-40;
  PressureCase problem({2, 1, 1.0});
  problem.predicted_face_velocities[1] = 1.0;
  problem.target_cell_divergence[0] = 1.0;
  problem.cell_fluid_volumes[0] = kTinyVolume;

  PressureResult result(problem);
  PressureWorkspace workspace;
  // The integrated imbalance is below 1e-12 times the RHS 1-norm, but the
  // physical defect is order one because that imbalance sits in the tiny gauge cell.
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectField(result.original_rhs, std::array{-1.0 + kTinyVolume, 1.0}, {0.0, 0.0});
  ExpectField(result.cell_pressures, std::array{0.0, 1.0});
  ExpectField(result.original_residuals, std::array{kTinyVolume, 0.0}, {1e-16, 0.0});
  ExpectField(result.cell_constraint_defects, std::array{-1.0, 0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGaugeCellIndices(result, std::array<std::size_t, 1>{0});
}

TEST(Pressure, CompatibilityToleranceScalesWithSmallRhs) {
  PressureCase problem({2, 1, 1.0});
  problem.predicted_face_velocities[1] = 1e-20;
  problem.target_cell_divergence[0] = 1e-29;
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  result.ExpectUnchanged();
}

TEST(Pressure, RejectsInteriorDirichletAndJumpOnDirichlet) {
  PressureCase problem({2, 1, 1.0});
  PressureResult result(problem);
  PressureWorkspace workspace;
  problem.dirichlet_face_mask[1] = 1;
  auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();

  problem.dirichlet_face_mask = {1, 0, 0, 0, 0, 0, 0};
  problem.face_pressure_jumps[0] = 3.0;
  status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();
}

TEST(Pressure, RejectsInvalidPhysicalCoefficients) {
  PressureCase problem({1, 1, 1.0});
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto expect_invalid = [&] {
    const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    result.ExpectUnchanged();
  };
  problem.face_open_fractions[0] = 1.01;
  expect_invalid();
  problem.face_open_fractions[0] = 1.0;
  problem.face_inverse_densities[0] = kInfinity;
  expect_invalid();
  problem.face_inverse_densities[0] = 1.0;
  problem.pressure_gradient_distances[0] = 0.0;
  expect_invalid();
  problem.pressure_gradient_distances[0] = 1.0;
  problem.cell_fluid_volumes[0] = 0.0;
  expect_invalid();
  problem.cell_fluid_volumes[0] = 1.0;
  problem.cell_bulk_moduli[0] = -kInfinity;
  expect_invalid();
  problem.cell_bulk_moduli[0] = kInfinity;
  problem.dirichlet_face_mask[0] = 2;
  expect_invalid();
  problem.dirichlet_face_mask[0] = 0;
  problem.dt = std::numeric_limits<double>::quiet_NaN();
  expect_invalid();
}

TEST(Pressure, RejectsIncorrectInputAndOutputExtents) {
  PressureCase problem({2, 1, 1.0});
  PressureResult result(problem);
  PressureWorkspace workspace;
  auto inputs = problem.Inputs();
  auto outputs = result.Outputs();
  inputs.predicted_face_velocities = inputs.predicted_face_velocities.first(problem.faces - 1);
  auto status = SolvePressure2D(inputs, outputs, workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();

  inputs = problem.Inputs();
  outputs.original_matrix = outputs.original_matrix.first(3);
  status = SolvePressure2D(inputs, outputs, workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();

  outputs.original_matrix = result.original_matrix;
  outputs.gauge_cell_indices = {};
  status = SolvePressure2D(inputs, outputs, workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  result.ExpectUnchanged();
}

TEST(Pressure, AllowsSharedReadOnlyFields) {
  PressureCase problem({1, 1, 1.0});
  auto inputs = problem.Inputs();
  inputs.wall_face_velocities = inputs.predicted_face_velocities;
  inputs.face_pressure_jumps = inputs.predicted_face_velocities;
  inputs.dirichlet_face_pressures = inputs.predicted_face_velocities;
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(inputs, result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  ExpectGaugeCellIndices(result, std::array<std::size_t, 1>{0});
}

TEST(Pressure, RejectsAnyWritableOverlapIncludingGaugeCountAndScratchTail) {
  PressureCase problem({1, 1, 1.0});
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto expect_overlap = [&](const PressureInputsView& inputs, PressureOutputsView outputs) {
    const auto status = SolvePressure2D(inputs, outputs, workspace);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    result.ExpectUnchanged();
  };
  auto outputs = result.Outputs();
  outputs.cell_pressures = problem.target_cell_divergence;
  expect_overlap(problem.Inputs(), outputs);
  EXPECT_EQ(problem.target_cell_divergence[0], 0.0);

  outputs.cell_pressures = outputs.original_rhs;
  expect_overlap(problem.Inputs(), outputs);

  outputs.cell_pressures = std::span(workspace.cell_pressures).last(1);
  expect_overlap(problem.Inputs(), outputs);

  auto inputs = problem.Inputs();
  inputs.target_cell_divergence = std::span(workspace.original_rhs).last(1);
  expect_overlap(inputs, result.Outputs());

  const auto expect_count_overlap = [&](std::size_t& aliased_count) {
    expect_overlap(
        problem.Inputs(),
        {
            .cell_pressures = result.cell_pressures,
            .corrected_face_velocities = result.corrected_face_velocities,
            .original_matrix = result.original_matrix,
            .original_rhs = result.original_rhs,
            .original_residuals = result.original_residuals,
            .cell_constraint_defects = result.cell_constraint_defects,
            .gauge_cell_indices = result.gauge_cell_indices,
            .gauge_count = aliased_count,
        }
    );
  };
  expect_count_overlap(result.gauge_cell_indices[0]);
  expect_count_overlap(workspace.gauge_cell_indices.back());
}

TEST(Pressure, DirichletConductanceRetainsRepresentableSmallOpenArea) {
  PressureCase problem({1, 1, 1.0});
  problem.dt = 0x1p-600;
  problem.face_open_fractions[0] = 0x1p-600;
  problem.face_inverse_densities[0] = 0x1p600;
  problem.dirichlet_face_mask[0] = 1;
  problem.dirichlet_face_pressures[0] = 0x1p600;

  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], 0x1p-600);
  EXPECT_EQ(result.original_rhs[0], 1.0);
  EXPECT_EQ(result.cell_pressures[0], 0x1p600);
  ExpectField(result.corrected_face_velocities, std::array{0.0, 0.0, 0.0, 0.0});
  ExpectField(result.original_residuals, std::array{0.0});
  ExpectField(result.cell_constraint_defects, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGaugeCellIndices(result, {});
}

TEST(Pressure, CorrectionScaleRecoversAfterTinyProduct) {
  PressureCase problem({1, 1, 0x1p100});
  problem.dt = 0x1p-900;
  problem.face_inverse_densities[0] = 0x1p-200;
  problem.pressure_gradient_distances[0] = 0x1p-100;
  problem.dirichlet_face_mask[0] = 1;
  problem.predicted_face_velocities[0] = 1.0;
  problem.cell_fluid_volumes[0] = 1.0;

  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], 0x1p-900);
  EXPECT_EQ(result.original_rhs[0], 0x1p100);
  EXPECT_EQ(result.cell_pressures[0], 0x1p1000);
  ExpectField(result.corrected_face_velocities, std::array{0.0, 0.0, 0.0, 0.0});
  ExpectField(result.original_residuals, std::array{0.0});
  ExpectField(result.cell_constraint_defects, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGaugeCellIndices(result, {});
}

TEST(Pressure, CorrectionScaleRecoversAfterHugeProduct) {
  PressureCase problem({1, 1, 1.0});
  problem.dt = 0x1p600;
  problem.face_inverse_densities[0] = 0x1p600;
  problem.pressure_gradient_distances[0] = 0x1p600;
  problem.dirichlet_face_mask[0] = 1;
  problem.dirichlet_face_pressures[0] = 1.0;

  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], 0x1p600);
  EXPECT_EQ(result.original_rhs[0], 0x1p600);
  EXPECT_EQ(result.cell_pressures[0], 1.0);
  ExpectField(result.corrected_face_velocities, std::array{0.0, 0.0, 0.0, 0.0});
  ExpectField(result.original_residuals, std::array{0.0});
  ExpectField(result.cell_constraint_defects, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGaugeCellIndices(result, {});
}

TEST(Pressure, ConductanceUsesTheSameScaleAsVelocityCorrection) {
  constexpr double kOpenArea = 0x1.8p-475;
  PressureCase problem({1, 1, 1.0});
  problem.dt = 0x1p-600;
  problem.face_open_fractions[0] = kOpenArea;
  problem.face_inverse_densities[0] = 0x1p600;
  problem.dirichlet_face_mask[0] = 1;
  problem.predicted_face_velocities[0] = 1.0;
  problem.cell_fluid_volumes[0] = kOpenArea;

  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], kOpenArea);
  EXPECT_EQ(result.original_rhs[0], kOpenArea);
  ExpectField(result.cell_pressures, std::array{1.0});
  ExpectField(result.corrected_face_velocities, std::array{0.0, 0.0, 0.0, 0.0});
  ExpectField(result.cell_constraint_defects, std::array{0.0});
  ExpectResidualIdentity(problem, result, kHandTolerance.absolute);
  ExpectGaugeCellIndices(result, {});
}

TEST(Pressure, RoundedZeroComplianceDoesNotInventAGauge) {
  PressureCase problem({1, 1, 1.0});
  std::ranges::fill(problem.face_open_fractions, 0.0);
  problem.cell_fluid_volumes[0] = 0x1p-600;
  problem.cell_bulk_moduli[0] = 0x1p600;
  problem.previous_cell_pressures[0] = 0x1p600;

  PressureResult result(problem);
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
    problem.face_open_fractions[0] = coefficients.aperture;
    problem.face_inverse_densities[0] = coefficients.mobility;
    problem.dirichlet_face_mask[0] = 1;

    PressureResult result(problem);
    PressureWorkspace workspace;
    const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
    EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
    result.ExpectUnchanged();
  }
}

TEST(Pressure, RejectsSubnormalConductanceEvenWithNormalCompliance) {
  constexpr double kFluidVolume = 0x1.8p-475;
  PressureCase problem({1, 1, 1.0});
  problem.face_open_fractions = {kFluidVolume, 0.0, 0.0, 0.0};
  problem.face_inverse_densities[0] = 0x1p-600;
  problem.dirichlet_face_mask[0] = 1;
  problem.dirichlet_face_pressures[0] = 0x1p600;
  problem.cell_fluid_volumes[0] = kFluidVolume;
  problem.cell_bulk_moduli[0] = kFluidVolume;

  // Conductance rounds from 0.75 * denorm_min to denorm_min. The normal
  // compliance keeps the resulting matrix well conditioned.
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Pressure, RejectsSubnormalComplianceEvenWithNormalConductance) {
  PressureCase problem({1, 1, 1.0});
  problem.face_open_fractions = {1.0, 0.0, 0.0, 0.0};
  problem.dirichlet_face_mask[0] = 1;
  problem.cell_fluid_volumes[0] = 0x1.8p-475;
  problem.cell_bulk_moduli[0] = 0x1p600;
  problem.previous_cell_pressures[0] = 0x1p600;

  // Compliance rounds from 0.75 * denorm_min to denorm_min. The normal
  // conductance keeps the resulting matrix well conditioned.
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Pressure, AcceptsMinimumNormalConductanceWithNormalCompliance) {
  constexpr double kFluidVolume = 0x1p-422;
  PressureCase problem({1, 1, 1.0});
  problem.face_open_fractions = {kFluidVolume, 0.0, 0.0, 0.0};
  problem.face_inverse_densities[0] = 0x1p-600;
  problem.dirichlet_face_mask[0] = 1;
  problem.dirichlet_face_pressures[0] = 0x1p600;
  problem.cell_fluid_volumes[0] = kFluidVolume;
  problem.cell_bulk_moduli[0] = kFluidVolume;

  // The conductance is exactly 2^-1022. Its Dirichlet lift is V and
  // compliance is one, so the represented system gives p = V.
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], 1.0);
  EXPECT_EQ(result.original_rhs[0], kFluidVolume);
  EXPECT_EQ(result.cell_pressures[0], kFluidVolume);
  EXPECT_EQ(result.corrected_face_velocities[0], 1.0);
  EXPECT_EQ(result.original_residuals[0], 0.0);
  EXPECT_EQ(result.cell_constraint_defects[0], 0.0);
  ExpectGaugeCellIndices(result, {});
}

TEST(Pressure, AcceptsMinimumNormalComplianceWithNormalConductance) {
  constexpr double kFluidVolume = 0x1p-422;
  PressureCase problem({1, 1, 1.0});
  problem.face_open_fractions = {1.0, 0.0, 0.0, 0.0};
  problem.dirichlet_face_mask[0] = 1;
  problem.cell_fluid_volumes[0] = kFluidVolume;
  problem.cell_bulk_moduli[0] = 0x1p600;
  problem.previous_cell_pressures[0] = 0x1p600;

  // The compliance is exactly 2^-1022. Its previous-pressure lift is V and
  // conductance is one, so the represented system gives p = V.
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.original_matrix[0], 1.0);
  EXPECT_EQ(result.original_rhs[0], kFluidVolume);
  EXPECT_EQ(result.cell_pressures[0], kFluidVolume);
  EXPECT_EQ(result.corrected_face_velocities[0], -kFluidVolume);
  EXPECT_EQ(result.original_residuals[0], 0.0);
  EXPECT_EQ(result.cell_constraint_defects[0], 0.0);
  ExpectGaugeCellIndices(result, {});
}

TEST(Pressure, CorrectionOverflowLeavesAllOutputsUnchanged) {
  PressureCase problem({1, 1, 1.0});
  // Finite conductance near one, but correcting the open fluid velocity overflows.
  problem.dirichlet_face_mask[0] = 1;
  problem.face_open_fractions[0] = 1e-308;
  problem.face_inverse_densities[0] = 1e308;
  problem.target_cell_divergence[0] = 4.0;
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  result.ExpectUnchanged();
}

TEST(Pressure, UnsafeSpdSolveLeavesAllOutputsUnchanged) {
  PressureCase problem({2, 1, 1.0});
  problem.face_open_fractions[1] = 0.0;
  problem.cell_bulk_moduli = {1.0, 1e20};
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  result.ExpectUnchanged();
}

TEST(Pressure, SupportsSixtyFourCellsAndOneHundredNinetyThreeFaces) {
  PressureCase problem({64, 1, 1.0});
  PressureResult result(problem);
  PressureWorkspace workspace;
  std::ranges::fill(problem.face_open_fractions, 0.0);
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(result.corrected_face_velocities.size(), 193);
  ASSERT_EQ(result.gauge_count, 64);
  for (std::size_t cell = 0; cell < result.gauge_count; ++cell) {
    EXPECT_EQ(result.gauge_cell_indices[cell], cell);
  }
  ExpectField(result.cell_pressures, std::vector<double>(64));
  ExpectField(result.corrected_face_velocities, std::vector<double>(193));
}

TEST(Pressure, RejectsLargerSystemsBeforePublication) {
  PressureCase problem({65, 1, 1.0});
  PressureResult result(problem);
  PressureWorkspace workspace;
  const auto status = SolvePressure2D(problem.Inputs(), result.Outputs(), workspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented);
  result.ExpectUnchanged();
}

}  // namespace
}  // namespace fluidity
