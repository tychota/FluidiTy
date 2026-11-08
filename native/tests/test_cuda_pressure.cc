#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "fixture_data.h"
#include "fluidity/host/cuda_pressure.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr double kUnwrittenValue = -29.0;
constexpr std::size_t kUnwrittenGaugeCellIndex = 104;
constexpr std::size_t kUnwrittenGaugeCount = 103;
constexpr CudaOperatorMeasurement kUnwrittenMeasurement{-4, -7.0F};

struct PressureCase {
  explicit PressureCase(Grid2D mac_grid)
      : grid(mac_grid),
        cell_count(static_cast<std::size_t>(grid.nx) * grid.ny),
        face_count(2 * cell_count + grid.nx + grid.ny),
        predicted_face_velocities(face_count),
        face_apertures(face_count, 1.0),
        face_inverse_densities(face_count, 1.0),
        face_gradient_distances(face_count, grid.h),
        wall_face_velocities(face_count),
        dirichlet_face_mask(face_count),
        dirichlet_face_pressures(face_count),
        oriented_face_pressure_jumps(face_count),
        target_cell_divergences(cell_count),
        cell_fluid_volumes(cell_count, grid.h * grid.h),
        cell_bulk_moduli(cell_count, std::numeric_limits<double>::infinity()),
        previous_cell_pressures(cell_count) {
  }

  PressureInputsView Inputs() const {
    return {
        .grid = grid,
        .dt = step_duration,
        .velocity = predicted_face_velocities,
        .aperture = face_apertures,
        .mobility = face_inverse_densities,
        .distance = face_gradient_distances,
        .wall_velocity = wall_face_velocities,
        .dirichlet_mask = dirichlet_face_mask,
        .dirichlet_pressure = dirichlet_face_pressures,
        .jump = oriented_face_pressure_jumps,
        .source = target_cell_divergences,
        .fluid_volume = cell_fluid_volumes,
        .bulk_modulus = cell_bulk_moduli,
        .previous_pressure = previous_cell_pressures,
    };
  }

  Grid2D grid;
  std::size_t cell_count;
  std::size_t face_count;
  double step_duration = 1.0;
  std::vector<double> predicted_face_velocities;
  std::vector<double> face_apertures;
  std::vector<double> face_inverse_densities;
  std::vector<double> face_gradient_distances;
  std::vector<double> wall_face_velocities;
  std::vector<std::uint8_t> dirichlet_face_mask;
  std::vector<double> dirichlet_face_pressures;
  std::vector<double> oriented_face_pressure_jumps;
  std::vector<double> target_cell_divergences;
  std::vector<double> cell_fluid_volumes;
  std::vector<double> cell_bulk_moduli;
  std::vector<double> previous_cell_pressures;
};

struct PressureResult {
  explicit PressureResult(const PressureCase& pressure_case)
      : cell_pressures(pressure_case.cell_count, kUnwrittenValue),
        corrected_face_velocities(pressure_case.face_count, kUnwrittenValue),
        original_matrix(pressure_case.cell_count * pressure_case.cell_count, kUnwrittenValue),
        original_rhs(pressure_case.cell_count, kUnwrittenValue),
        original_residuals(pressure_case.cell_count, kUnwrittenValue),
        cell_constraint_defects(pressure_case.cell_count, kUnwrittenValue),
        gauge_cell_indices(pressure_case.cell_count, kUnwrittenGaugeCellIndex) {
  }

  PressureOutputsView Outputs() {
    return {
        .pressure = cell_pressures,
        .velocity = corrected_face_velocities,
        .original_matrix = original_matrix,
        .rhs = original_rhs,
        .residual = original_residuals,
        .constraint_defect = cell_constraint_defects,
        .gauges = gauge_cell_indices,
        .gauge_count = gauge_count,
    };
  }

  void ExpectUnchanged() const {
    for (const auto* output_field : {
             &cell_pressures,
             &corrected_face_velocities,
             &original_matrix,
             &original_rhs,
             &original_residuals,
             &cell_constraint_defects,
         }) {
      EXPECT_TRUE(std::ranges::all_of(*output_field, [](double output_value) {
        return output_value == kUnwrittenValue;
      }));
    }
    EXPECT_EQ(gauge_count, kUnwrittenGaugeCount);
    EXPECT_EQ(gauge_cell_indices, std::vector(gauge_cell_indices.size(), kUnwrittenGaugeCellIndex));
    EXPECT_EQ(kernel_measurement.compiled_architecture, kUnwrittenMeasurement.compiled_architecture);
    EXPECT_EQ(kernel_measurement.kernel_ms, kUnwrittenMeasurement.kernel_ms);
  }

  std::vector<double> cell_pressures;
  std::vector<double> corrected_face_velocities;
  std::vector<double> original_matrix;
  std::vector<double> original_rhs;
  std::vector<double> original_residuals;
  std::vector<double> cell_constraint_defects;
  std::vector<std::size_t> gauge_cell_indices;
  std::size_t gauge_count = kUnwrittenGaugeCount;
  CudaOperatorMeasurement kernel_measurement = kUnwrittenMeasurement;
};

void ExpectField(
    std::span<const double> actual_values,
    std::span<const double> expected_values,
    double absolute_tolerance = 2e-12,
    double relative_tolerance = 2e-12
) {
  ASSERT_EQ(actual_values.size(), expected_values.size());
  for (std::size_t entry_index = 0; entry_index < expected_values.size(); ++entry_index) {
    SCOPED_TRACE(entry_index);
    const double expected_value = expected_values[entry_index];
    EXPECT_NEAR(
        actual_values[entry_index],
        expected_value,
        absolute_tolerance + relative_tolerance * std::abs(expected_value)
    );
  }
}

void ExpectGaugeCellIndices(const PressureResult& pressure_result, std::span<const std::size_t> expected_gauge_cells) {
  ASSERT_EQ(pressure_result.gauge_count, expected_gauge_cells.size());
  ASSERT_LE(pressure_result.gauge_count, pressure_result.gauge_cell_indices.size());
  const auto gauge_slots = std::span(pressure_result.gauge_cell_indices);
  EXPECT_TRUE(std::ranges::equal(gauge_slots.first(pressure_result.gauge_count), expected_gauge_cells));
  for (const std::size_t unused_gauge_cell_index : gauge_slots.subspan(pressure_result.gauge_count)) {
    EXPECT_EQ(unused_gauge_cell_index, kUnwrittenGaugeCellIndex);
  }
}

void ExpectCpuParity(const PressureCase& pressure_case, const PressureResult& cuda_result) {
  PressureResult cpu_result(pressure_case);
  PressureWorkspace cpu_workspace;
  const auto cpu_status = SolvePressure2D(pressure_case.Inputs(), cpu_result.Outputs(), cpu_workspace);
  ASSERT_TRUE(cpu_status.ok()) << cpu_status;
  ExpectField(cuda_result.cell_pressures, cpu_result.cell_pressures);
  ExpectField(cuda_result.corrected_face_velocities, cpu_result.corrected_face_velocities);
  ExpectField(cuda_result.original_matrix, cpu_result.original_matrix);
  ExpectField(cuda_result.original_rhs, cpu_result.original_rhs);
  ExpectField(cuda_result.original_residuals, cpu_result.original_residuals);
  ExpectField(cuda_result.cell_constraint_defects, cpu_result.cell_constraint_defects);
  ASSERT_LE(cpu_result.gauge_count, cpu_result.gauge_cell_indices.size());
  ExpectGaugeCellIndices(cuda_result, std::span(cpu_result.gauge_cell_indices).first(cpu_result.gauge_count));
}

std::span<const double> FixtureFloats(const fixtures::FixtureRecord& fixture, std::string_view field_name) {
  const auto array_record = fixtures::FindArray(fixture, field_name);
  if (!array_record.ok()) {
    ADD_FAILURE() << array_record.status();
    return {};
  }
  if ((*array_record)->dtype != fixtures::DType::kFloat64) {
    ADD_FAILURE() << field_name << " must be float64";
    return {};
  }
  return (*array_record)->f64_data;
}

void CheckPressureFixture(
    std::string_view fixture_name,
    std::string_view operator_name,
    std::span<const double> expected_original_matrix,
    std::span<const double> expected_original_rhs,
    std::span<const std::size_t> expected_gauge_cells
) {
  const auto fixture_records = fixtures::AllFixtures();
  const auto fixture_record_iterator = std::ranges::find(fixture_records, fixture_name, &fixtures::FixtureRecord::name);
  ASSERT_NE(fixture_record_iterator, fixture_records.end());
  const auto fixture_grid = fixtures::FindArray(*fixture_record_iterator, "grid");
  ASSERT_TRUE(fixture_grid.ok()) << fixture_grid.status();
  ASSERT_EQ((*fixture_grid)->dtype, fixtures::DType::kInt64);
  const auto grid_dimensions = (*fixture_grid)->i64_data;
  const auto grid_spacing = FixtureFloats(*fixture_record_iterator, "spacing");
  const auto grid_origin = FixtureFloats(*fixture_record_iterator, "origin");
  const auto step_duration = FixtureFloats(*fixture_record_iterator, "dt");
  ASSERT_EQ(grid_dimensions.size(), 2);
  ASSERT_EQ(grid_spacing.size(), 1);
  ASSERT_EQ(grid_origin.size(), 2);
  ASSERT_EQ(step_duration.size(), 1);
  PressureCase pressure_case({
      .nx = static_cast<std::uint32_t>(grid_dimensions[0]),
      .ny = static_cast<std::uint32_t>(grid_dimensions[1]),
      .h = grid_spacing[0],
      .origin = {grid_origin[0], grid_origin[1]},
  });
  pressure_case.step_duration = step_duration[0];
  const auto predicted_face_velocities = FixtureFloats(*fixture_record_iterator, "velocity");
  ASSERT_EQ(predicted_face_velocities.size(), pressure_case.face_count);
  std::ranges::copy(predicted_face_velocities, pressure_case.predicted_face_velocities.begin());
  for (std::size_t face_index = 0; face_index < pressure_case.face_count; ++face_index) {
    const auto face_geometry = FaceGeometry(pressure_case.grid, face_index);
    ASSERT_TRUE(face_geometry.ok()) << face_geometry.status();
    pressure_case.face_gradient_distances[face_index] = face_geometry->distance;
  }
  if (fixture_name == "two_cell_pressure") {
    const auto oriented_face_pressure_jumps = FixtureFloats(*fixture_record_iterator, "jump");
    ASSERT_EQ(oriented_face_pressure_jumps.size(), pressure_case.face_count);
    std::ranges::copy(oriented_face_pressure_jumps, pressure_case.oriented_face_pressure_jumps.begin());
  } else {
    const auto target_cell_divergences = FixtureFloats(*fixture_record_iterator, "source");
    const auto bulk_modulus = FixtureFloats(*fixture_record_iterator, "bulk_modulus");
    ASSERT_EQ(target_cell_divergences.size(), pressure_case.cell_count);
    ASSERT_EQ(bulk_modulus.size(), 1);
    std::ranges::copy(target_cell_divergences, pressure_case.target_cell_divergences.begin());
    std::ranges::fill(pressure_case.cell_bulk_moduli, bulk_modulus.front());
  }

  PressureResult cuda_result(pressure_case);
  const auto cuda_status =
      CudaSolvePressure2D(pressure_case.Inputs(), cuda_result.Outputs(), cuda_result.kernel_measurement);
  ASSERT_TRUE(cuda_status.ok()) << cuda_status;
  ExpectField(
      cuda_result.cell_pressures,
      FixtureFloats(*fixture_record_iterator, "expected_pressure"),
      fixture_record_iterator->atol,
      fixture_record_iterator->rtol
  );
  ExpectField(
      cuda_result.corrected_face_velocities,
      FixtureFloats(*fixture_record_iterator, "expected_velocity"),
      fixture_record_iterator->atol,
      fixture_record_iterator->rtol
  );
  ExpectField(
      cuda_result.original_matrix,
      expected_original_matrix,
      fixture_record_iterator->atol,
      fixture_record_iterator->rtol
  );
  ExpectField(
      cuda_result.original_rhs,
      expected_original_rhs,
      fixture_record_iterator->atol,
      fixture_record_iterator->rtol
  );
  const std::vector<double> zero_cell_values(pressure_case.cell_count, 0.0);
  ExpectField(
      cuda_result.original_residuals,
      zero_cell_values,
      fixture_record_iterator->atol,
      fixture_record_iterator->rtol
  );
  ExpectField(
      cuda_result.cell_constraint_defects,
      zero_cell_values,
      fixture_record_iterator->atol,
      fixture_record_iterator->rtol
  );
  ExpectGaugeCellIndices(cuda_result, expected_gauge_cells);
  ExpectCpuParity(pressure_case, cuda_result);
  const auto& kernel_measurement = cuda_result.kernel_measurement;
  ASSERT_EQ(kernel_measurement.compiled_architecture, 1200);
  ASSERT_TRUE(std::isfinite(kernel_measurement.kernel_ms));
  ASSERT_GE(kernel_measurement.kernel_ms, 0.0F);
  std::cout << "operator=" << operator_name << " compiled_architecture=" << kernel_measurement.compiled_architecture
            << " kernel_ms=" << std::fixed << std::setprecision(9) << kernel_measurement.kernel_ms << '\n';
}

TEST(CudaPressure, NeumannJumpMatchesFixture) {
  CheckPressureFixture(
      "two_cell_pressure",
      "pressure_neumann",
      std::array{1.0, -1.0, -1.0, 1.0},
      std::array{-5.0, 5.0},
      std::array<std::size_t, 1>{0}
  );
}

TEST(CudaPressure, ComplianceRetainsMeanPressure) {
  CheckPressureFixture("compliant_pressure", "pressure_compliance", std::array{0.5}, std::array{3.0}, {});

  // The finite-modulus cell anchors the whole component. For dt=0.5,
  // the internal conductance is 1 and the second cell's compliance is 0.5.
  PressureCase mixed_compliance_case({2, 1, 1.0});
  mixed_compliance_case.step_duration = 0.5;
  mixed_compliance_case.face_inverse_densities[1] = 2.0;
  mixed_compliance_case.target_cell_divergences = {1.0, 2.0};
  mixed_compliance_case.cell_fluid_volumes = {2.0, 1.0};
  mixed_compliance_case.cell_bulk_moduli[1] = 4.0;
  mixed_compliance_case.previous_cell_pressures[1] = 5.0;
  PressureResult cuda_result(mixed_compliance_case);
  const auto cuda_status =
      CudaSolvePressure2D(mixed_compliance_case.Inputs(), cuda_result.Outputs(), cuda_result.kernel_measurement);
  ASSERT_TRUE(cuda_status.ok()) << cuda_status;
  ExpectField(cuda_result.original_matrix, std::array{1.0, -1.0, -1.0, 1.5});
  ExpectField(cuda_result.original_rhs, std::array{2.0, 4.5});
  ExpectField(cuda_result.cell_pressures, std::array{15.0, 13.0});
  EXPECT_NEAR(cuda_result.corrected_face_velocities[1], 2.0, 2e-12);
  ExpectGaugeCellIndices(cuda_result, {});
  ExpectCpuParity(mixed_compliance_case, cuda_result);
}

TEST(CudaPressure, LaunchIgnoresPriorRuntimeError) {
  const PressureCase pressure_case({1, 1, 1.0});
  PressureResult baseline_result(pressure_case);
  const auto baseline_status =
      CudaSolvePressure2D(pressure_case.Inputs(), baseline_result.Outputs(), baseline_result.kernel_measurement);
  ASSERT_TRUE(baseline_status.ok()) << baseline_status;

  ASSERT_EQ(cudaSetDevice(-1), cudaErrorInvalidDevice);
  ASSERT_EQ(cudaPeekAtLastError(), cudaErrorInvalidDevice);

  PressureResult cuda_result(pressure_case);
  const auto cuda_status =
      CudaSolvePressure2D(pressure_case.Inputs(), cuda_result.Outputs(), cuda_result.kernel_measurement);
  EXPECT_EQ(cudaGetLastError(), cudaErrorInvalidDevice);
  EXPECT_TRUE(cuda_status.ok()) << cuda_status;
}

TEST(CudaPressure, DirichletAndWallFluxMatchAnalyticAndCpu) {
  PressureCase pressure_case({1, 1, 1.0});
  pressure_case.face_apertures = {0.5, 0.0, 0.0, 0.0};
  pressure_case.face_gradient_distances[0] = 0.5;
  pressure_case.dirichlet_face_mask[0] = 1;
  pressure_case.dirichlet_face_pressures[0] = 3.0;
  pressure_case.wall_face_velocities[1] = 1.0;

  PressureResult cuda_result(pressure_case);
  const auto cuda_status =
      CudaSolvePressure2D(pressure_case.Inputs(), cuda_result.Outputs(), cuda_result.kernel_measurement);
  ASSERT_TRUE(cuda_status.ok()) << cuda_status;
  ExpectField(cuda_result.original_matrix, std::array{1.0});
  ExpectField(cuda_result.original_rhs, std::array{2.0});
  ExpectField(cuda_result.cell_pressures, std::array{2.0});
  ExpectField(cuda_result.corrected_face_velocities, std::array{2.0, 0.0, 0.0, 0.0});
  ExpectField(cuda_result.original_residuals, std::array{0.0});
  ExpectField(cuda_result.cell_constraint_defects, std::array{0.0});
  ExpectGaugeCellIndices(cuda_result, {});
  ExpectCpuParity(pressure_case, cuda_result);
}

TEST(CudaPressure, CapacityAndIncompatibilityPreserveOutputs) {
  PressureCase disconnected_pressure_case({8, 1, 1.0});
  disconnected_pressure_case.face_apertures[4] = 0.0;
  disconnected_pressure_case.target_cell_divergences = {1.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0, -2.0};

  // A small accepted imbalance on the first gauge still appears in the original
  // equation's residual and flux defect. The pinned solve alone would report zero.
  constexpr double gauge_imbalance = 0x1p-46;
  disconnected_pressure_case.target_cell_divergences[0] += gauge_imbalance;
  PressureResult compatible_result(disconnected_pressure_case);
  const auto compatible_status = CudaSolvePressure2D(
      disconnected_pressure_case.Inputs(),
      compatible_result.Outputs(),
      compatible_result.kernel_measurement
  );
  ASSERT_TRUE(compatible_status.ok()) << compatible_status;
  ExpectGaugeCellIndices(compatible_result, std::array<std::size_t, 2>{0, 4});
  EXPECT_NEAR(compatible_result.original_residuals[0], gauge_imbalance, 0x1p-49);
  EXPECT_NEAR(compatible_result.cell_constraint_defects[0], -gauge_imbalance, 0x1p-49);
  ExpectCpuParity(disconnected_pressure_case, compatible_result);

  const auto expect_rejection = [](const PressureCase& rejected_case, absl::StatusCode expected_status) {
    PressureResult rejected_result(rejected_case);
    const auto rejected_status =
        CudaSolvePressure2D(rejected_case.Inputs(), rejected_result.Outputs(), rejected_result.kernel_measurement);
    EXPECT_EQ(rejected_status.code(), expected_status) << rejected_status;
    rejected_result.ExpectUnchanged();
  };

  disconnected_pressure_case.target_cell_divergences = {1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0};
  expect_rejection(disconnected_pressure_case, absl::StatusCode::kFailedPrecondition);

  disconnected_pressure_case.dirichlet_face_mask[0] = 1;
  disconnected_pressure_case.oriented_face_pressure_jumps[0] = 1.0;
  expect_rejection(disconnected_pressure_case, absl::StatusCode::kInvalidArgument);
  expect_rejection(PressureCase({9, 1, 1.0}), absl::StatusCode::kResourceExhausted);

  // A positive subnormal is still too small for a pressure matrix coefficient.
  constexpr double subnormal_coefficient = 0x1p-1023;
  PressureCase subnormal_conductance_case({2, 1, 1.0});
  subnormal_conductance_case.face_apertures[1] = subnormal_coefficient;
  expect_rejection(subnormal_conductance_case, absl::StatusCode::kOutOfRange);
  PressureCase subnormal_compliance_case({1, 1, 1.0});
  subnormal_compliance_case.cell_fluid_volumes[0] = subnormal_coefficient;
  subnormal_compliance_case.cell_bulk_moduli[0] = 1.0;
  expect_rejection(subnormal_compliance_case, absl::StatusCode::kOutOfRange);

  const PressureCase aliased_pressure_case({1, 1, 1.0});
  PressureResult aliased_result(aliased_pressure_case);
  auto aliased_outputs = aliased_result.Outputs();
  aliased_outputs.rhs = aliased_outputs.pressure;
  const auto alias_status =
      CudaSolvePressure2D(aliased_pressure_case.Inputs(), aliased_outputs, aliased_result.kernel_measurement);
  EXPECT_EQ(alias_status.code(), absl::StatusCode::kInvalidArgument);
  aliased_result.ExpectUnchanged();
}

}  // namespace
}  // namespace fluidity
