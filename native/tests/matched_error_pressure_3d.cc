#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/host/adaptive_mac_topology_3d.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"
#include "mac_pressure_3d_system.h"

namespace fluidity {
namespace {

constexpr double kFinestCellWidth = 0.25;  // m.
constexpr double kDuration = 0.5;  // s.
constexpr double kInverseDensity = 0.5;  // m³/kg.
constexpr double kPressureSlope = 0.75;  // Pa/m in the x direction.
constexpr double kPressureIntercept = 0.125;  // Pa at x = 0.
constexpr double kPressureErrorLimit = 1e-4;  // Pa, after removing each layout's pressure gauge.
constexpr double kDivergenceErrorLimit = 2e-5;  // s⁻¹.
constexpr double kDomainVolume = 8.0;  // m³.
constexpr std::size_t kRepetitions = 9;

enum class Layout {
  kUniformFinest,
  kTwoLevel,
};

struct BenchmarkCase {
  AdaptiveMacTopology3D topology;
  pressure_internal::PreparedMacPressureSystem3D system;
  MacPressureWorkspace3D workspace;
  std::size_t coarse_cells;
  std::size_t mixed_faces;
};

struct Measurement {
  double elapsed_ms;
  double pressure_error_pa;
  double divergence_error_per_s;
  double corrected_velocity_m_per_s;
  std::size_t iterations;
};

using Measurements = std::array<Measurement, kRepetitions>;

double ExactPressure(const AdaptiveMacLeaf3D& cell) {
  const double center_x = kFinestCellWidth * cell.key.lower_corner.x + 0.5 * cell.cell_width;
  return kPressureIntercept + kPressureSlope * center_x;
}

// Both layouts tile the same sealed cube. In the two-level layout x < 0.5 m
// uses aligned coarse cells, and the rightmost quarter remains at finest width.
absl::StatusOr<BenchmarkCase> PrepareCase(Layout layout) {
  std::vector<AdaptiveMacCellKey3D> keys;
  std::size_t coarse_count = 0;
  for (int z = -4; z < 4; ++z) {
    for (int y = -4; y < 4; ++y) {
      for (int x = -4; x < 4; ++x) {
        const bool coarse = layout == Layout::kTwoLevel && x < 2;
        if (coarse && (x % 2 != 0 || y % 2 != 0 || z % 2 != 0)) {
          continue;
        }
        keys.push_back({{x, y, z}, static_cast<std::uint8_t>(coarse)});
        coarse_count += coarse;
      }
    }
  }

  auto topology = AdaptiveMacTopology3D::Create(keys, kFinestCellWidth);
  if (!topology.ok()) {
    return topology.status();
  }
  const auto leaves = topology->cells();
  std::vector<MacPressureCell3D> cells;
  cells.reserve(leaves.size());
  double total_volume = 0;
  for (const auto& leaf : leaves) {
    total_volume += leaf.cell_volume;
    cells.push_back({
        .cell_index = leaf.key.lower_corner,
        .fluid_volume = leaf.cell_volume,
        .target_divergence = 0.0,
        .bulk_modulus = std::numeric_limits<double>::infinity(),
        .previous_pressure = 0.0,
    });
  }

  std::vector<pressure_internal::ExplicitMacPressureFace3D> faces;
  faces.reserve(topology->interior_faces().size());
  std::size_t mixed_faces = 0;
  for (const auto& face : topology->interior_faces()) {
    const auto& tail = leaves[face.tail_cell_ordinal];
    const auto& head = leaves[face.head_cell_ordinal];
    mixed_faces += tail.key.width_exponent != head.key.width_exponent;
    // The continuum normal gradient and the two-point gradient agree for this
    // x-affine field, even when coarse/fine centers are offset tangentially.
    const double normal_gradient = face.face_index.axis == FaceNormalAxis::kX ? kPressureSlope : 0.0;
    faces.push_back({
        .face_slot = faces.size(),
        .tail_cell_ordinal = face.tail_cell_ordinal,
        .head_cell_ordinal = face.head_cell_ordinal,
        .open_area = face.face_area,
        .inverse_density = kInverseDensity,
        .pressure_gradient_distance = face.cell_center_distance,
        .predicted_normal_velocity = kDuration * kInverseDensity * normal_gradient,
    });
  }
  const bool uniform = layout == Layout::kUniformFinest;
  if (total_volume != kDomainVolume || cells.size() != (uniform ? 512 : 176) || faces.size() != (uniform ? 1344 : 456)
      || coarse_count != (uniform ? 0 : 48) || mixed_faces != (uniform ? 0 : 64)) {
    return absl::InternalError("manufactured cube topology does not match the authored domain");
  }

  auto system = pressure_internal::PrepareExplicitMacPressureSystem3D(kDuration, cells, faces);
  if (!system.ok()) {
    return system.status();
  }
  auto workspace = MacPressureWorkspace3D::Create(cells.size(), faces.size());
  if (!workspace.ok()) {
    return workspace.status();
  }
  return BenchmarkCase{
      .topology = std::move(*topology),
      .system = std::move(*system),
      .workspace = std::move(*workspace),
      .coarse_cells = coarse_count,
      .mixed_faces = mixed_faces,
  };
}

absl::StatusOr<Measurement> MeasureProjection(BenchmarkCase& fixture) {
  const auto start = std::chrono::steady_clock::now();
  const auto projection = fixture.workspace.ProjectPrepared(fixture.system);
  const auto stop = std::chrono::steady_clock::now();
  if (!projection.ok()) {
    return projection.status();
  }
  const auto leaves = fixture.topology.cells();
  const auto faces = fixture.topology.interior_faces();
  if (projection->gauge_cell_ordinals.size() != 1 || projection->diagnostics.iteration_count == 0
      || projection->cell_pressures.size() != leaves.size()
      || projection->corrected_face_velocities.size() != faces.size()) {
    return absl::InternalError("expected a nontrivial zero-start projection on one sealed component");
  }

  Measurement measurement{
      .elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count(),
      .pressure_error_pa = 0.0,
      .divergence_error_per_s = 0.0,
      .corrected_velocity_m_per_s = 0.0,
      .iterations = projection->diagnostics.iteration_count,
  };
  // Reconstruct physical divergence from authored areas and corrected velocities
  // in original face order, independently of the solver's stopping diagnostics.
  std::vector<double> corrected_outward_flux(leaves.size(), 0.0);
  for (std::size_t slot = 0; slot < faces.size(); ++slot) {
    const auto& face = faces[slot];
    const double velocity = projection->corrected_face_velocities[slot];
    const double flux = face.face_area * velocity;
    corrected_outward_flux[face.tail_cell_ordinal] += flux;
    corrected_outward_flux[face.head_cell_ordinal] -= flux;
    measurement.corrected_velocity_m_per_s = std::max(measurement.corrected_velocity_m_per_s, std::abs(velocity));
  }

  const double exact_gauge = ExactPressure(leaves[projection->gauge_cell_ordinals.front()]);
  for (std::size_t ordinal = 0; ordinal < leaves.size(); ++ordinal) {
    const double exact_gauged_pressure = ExactPressure(leaves[ordinal]) - exact_gauge;
    const double pressure_error = std::abs(projection->cell_pressures[ordinal] - exact_gauged_pressure);
    const double divergence = std::abs(corrected_outward_flux[ordinal] / leaves[ordinal].cell_volume);
    if (!std::isfinite(pressure_error) || !std::isfinite(divergence)) {
      return absl::InternalError("manufactured pressure error is nonfinite");
    }
    measurement.pressure_error_pa = std::max(measurement.pressure_error_pa, pressure_error);
    measurement.divergence_error_per_s = std::max(measurement.divergence_error_per_s, divergence);
  }
  return measurement;
}

bool MeetsErrorBudget(const Measurement& measurement) {
  return measurement.pressure_error_pa <= kPressureErrorLimit
      && measurement.divergence_error_per_s <= kDivergenceErrorLimit;
}

void PrintCase(const char* name, const BenchmarkCase& fixture, const Measurements& measurements) {
  std::array<double, kRepetitions> elapsed{};
  double pressure_error = 0;
  double divergence_error = 0;
  double velocity_error = 0;
  std::size_t min_iterations = measurements.front().iterations;
  std::size_t max_iterations = min_iterations;
  for (std::size_t sample = 0; sample < measurements.size(); ++sample) {
    const auto& measurement = measurements[sample];
    elapsed[sample] = measurement.elapsed_ms;
    pressure_error = std::max(pressure_error, measurement.pressure_error_pa);
    divergence_error = std::max(divergence_error, measurement.divergence_error_per_s);
    velocity_error = std::max(velocity_error, measurement.corrected_velocity_m_per_s);
    min_iterations = std::min(min_iterations, measurement.iterations);
    max_iterations = std::max(max_iterations, measurement.iterations);
  }
  std::ranges::sort(elapsed);
  std::cout << "    {\n"
            << "      \"layout\": \"" << name << "\",\n"
            << "      \"cells\": " << fixture.topology.cells().size() << ",\n"
            << "      \"coarse_cells\": " << fixture.coarse_cells << ",\n"
            << "      \"interior_faces\": " << fixture.topology.interior_faces().size() << ",\n"
            << "      \"mixed_level_faces\": " << fixture.mixed_faces << ",\n"
            << "      \"pressure_linf_error_pa\": " << pressure_error << ",\n"
            << "      \"divergence_linf_error_per_s\": " << divergence_error << ",\n"
            << "      \"corrected_velocity_linf_m_per_s\": " << velocity_error << ",\n"
            << "      \"min_iterations\": " << min_iterations << ",\n"
            << "      \"max_iterations\": " << max_iterations << ",\n"
            << "      \"workspace_reserved_bytes\": " << fixture.workspace.bytes().total_bytes << ",\n"
            << "      \"workspace_high_water_bytes\": " << fixture.workspace.high_water_bytes() << ",\n"
            << "      \"median_project_ms\": " << elapsed[elapsed.size() / 2] << ",\n"
            << "      \"project_ms\": [";
  for (std::size_t sample = 0; sample < measurements.size(); ++sample) {
    if (sample > 0) {
      std::cout << ", ";
    }
    std::cout << measurements[sample].elapsed_ms;
  }
  std::cout << "]\n    }";
}

absl::Status Benchmark() {
  auto uniform = PrepareCase(Layout::kUniformFinest);
  if (!uniform.ok()) {
    return uniform.status();
  }
  auto adaptive = PrepareCase(Layout::kTwoLevel);
  if (!adaptive.ok()) {
    return adaptive.status();
  }
  const std::array fixtures{&*uniform, &*adaptive};
  for (auto* fixture : fixtures) {
    const auto warmup = MeasureProjection(*fixture);
    if (!warmup.ok()) {
      return warmup.status();
    }
    if (!MeetsErrorBudget(*warmup)) {
      return absl::FailedPreconditionError("warm-up exceeded the common pressure or divergence error budget");
    }
  }

  std::array<Measurements, 2> measurements{};
  for (std::size_t repeat = 0; repeat < kRepetitions; ++repeat) {
    for (std::size_t slot = 0; slot < fixtures.size(); ++slot) {
      const std::size_t layout = (repeat + slot) % fixtures.size();
      const auto measurement = MeasureProjection(*fixtures[layout]);
      if (!measurement.ok()) {
        return measurement.status();
      }
      measurements[layout][repeat] = *measurement;
    }
  }
  const bool passed =
      std::ranges::all_of(measurements[0], MeetsErrorBudget) && std::ranges::all_of(measurements[1], MeetsErrorBudget);
  std::cout << std::setprecision(10) << "{\n  \"benchmark\": \"manufactured_sealed_cube_pressure\",\n"
            << "  \"domain_volume_m3\": " << kDomainVolume << ",\n"
            << "  \"pressure_linf_limit_pa\": " << kPressureErrorLimit << ",\n"
            << "  \"divergence_linf_limit_per_s\": " << kDivergenceErrorLimit << ",\n"
            << "  \"passed\": " << std::boolalpha << passed << ",\n"
            << "  \"layouts\": [\n";
  PrintCase("uniform_finest", *uniform, measurements[0]);
  std::cout << ",\n";
  PrintCase("balanced_two_level", *adaptive, measurements[1]);
  std::cout << "\n  ]\n}\n";
  if (!passed) {
    return absl::FailedPreconditionError("projection exceeded the common pressure or divergence error budget");
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace fluidity

int main() {
  const auto status = fluidity::Benchmark();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
