#include "fluidity/host/mac_pressure_3d.h"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mac_pressure_3d_multigrid.h"
#include "mac_pressure_3d_system.h"

namespace fluidity {
namespace {

using pressure_internal::MacPressureHierarchy3D;
using pressure_internal::MacPressureLevel3D;
using pressure_internal::MacPressureStoppingCriteria3D;
using pressure_internal::PreparedMacPressureSystem3D;

// Cast only finite values; a nonzero FP64 residual or coarse correction must
// not disappear on entering the fine FP32 iteration.
absl::StatusOr<float> FineValue(double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
    return absl::OutOfRangeError("pressure iterative value exceeds the finite FP32 range");
  }
  const float rounded = static_cast<float>(value);
  if (!std::isfinite(rounded) || (value != 0.0 && rounded == 0.0F)) {
    return absl::OutOfRangeError("pressure iterative value is not representable in FP32");
  }
  return rounded;
}

// The fine call uses float for every multiply/add and difference. Coarse calls
// use double. Edge differences retain the graph's constant mode and avoid
// cancellation between independently rounded diagonal and off-diagonal terms.
template <std::floating_point Scalar>
absl::Status
ApplyLevel(const MacPressureLevel3D& level, std::span<const Scalar> pressure, std::span<Scalar> integrated_flux) {
  for (std::size_t unknown = 0; unknown < pressure.size(); ++unknown) {
    Scalar row_flux = static_cast<Scalar>(level.anchor_diagonal[unknown]) * pressure[unknown];
    for (std::size_t offset = level.row_offsets[unknown]; offset < level.row_offsets[unknown + 1]; ++offset) {
      const auto neighbor = level.neighbors[offset];
      const Scalar pressure_difference = pressure[unknown] - pressure[neighbor.unknown_ordinal];
      const Scalar edge_flux =
          static_cast<Scalar>(level.edges[neighbor.edge_ordinal].conductance) * pressure_difference;
      row_flux += edge_flux;
    }
    if (!std::isfinite(row_flux)) {
      return absl::OutOfRangeError("pressure graph product is unrepresentable");
    }
    integrated_flux[unknown] = row_flux;
  }
  return absl::OkStatus();
}

// A fixed linear symmetric V-cycle: one 2/3 weighted Jacobi sweep before and
// after P A_c^-1 Pᵀ correction. Both sweeps use the same mirrored operator and
// diagonal. Terminal blocks use the precomputed FP64 Cholesky factors.
template <std::floating_point Scalar>
absl::StatusOr<std::vector<Scalar>>
ApplyVcycle(const MacPressureHierarchy3D& hierarchy, std::size_t level_ordinal, std::span<const Scalar> residual) {
  const auto& level = hierarchy.levels[level_ordinal];
  std::vector<Scalar> correction(residual.size(), Scalar{0});
  if (level_ordinal + 1 == hierarchy.levels.size()) {
    for (const auto& block : level.terminal_blocks) {
      const std::size_t count = block.unknown_ordinals.size();
      std::vector<double> block_correction(count);
      for (std::size_t row = 0; row < count; ++row) {
        double forward_rhs = residual[block.unknown_ordinals[row]];
        for (std::size_t column = 0; column < row; ++column) {
          forward_rhs -= block.cholesky_lower[row * count + column] * block_correction[column];
        }
        block_correction[row] = forward_rhs / block.cholesky_lower[row * count + row];
      }
      for (std::size_t reverse_row = count; reverse_row > 0; --reverse_row) {
        const std::size_t row = reverse_row - 1;
        double backward_rhs = block_correction[row];
        for (std::size_t column = row + 1; column < count; ++column) {
          backward_rhs -= block.cholesky_lower[column * count + row] * block_correction[column];
        }
        block_correction[row] = backward_rhs / block.cholesky_lower[row * count + row];
        if (!std::isfinite(block_correction[row])) {
          return absl::OutOfRangeError("pressure terminal solve is unrepresentable");
        }
        if constexpr (std::same_as<Scalar, float>) {
          const auto fine_correction = FineValue(block_correction[row]);
          if (!fine_correction.ok()) {
            return fine_correction.status();
          }
          correction[block.unknown_ordinals[row]] = *fine_correction;
        } else {
          correction[block.unknown_ordinals[row]] = block_correction[row];
        }
      }
    }
    return correction;
  }

  constexpr Scalar kJacobiWeight = Scalar{2} / Scalar{3};
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    correction[unknown] = kJacobiWeight * (residual[unknown] / static_cast<Scalar>(level.diagonal[unknown]));
  }
  std::vector<Scalar> correction_flux(residual.size());
  auto status = ApplyLevel<Scalar>(level, correction, correction_flux);
  if (!status.ok()) {
    return status;
  }
  std::vector<double> coarse_residual(hierarchy.levels[level_ordinal + 1].coordinates.size(), 0.0);
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    const Scalar remaining_flux = residual[unknown] - correction_flux[unknown];
    coarse_residual[level.aggregate_ordinals[unknown]] += static_cast<double>(remaining_flux);
  }
  for (const double flux : coarse_residual) {
    if (!std::isfinite(flux)) {
      return absl::OutOfRangeError("pressure restricted residual is unrepresentable");
    }
  }
  const auto coarse_correction = ApplyVcycle<double>(hierarchy, level_ordinal + 1, coarse_residual);
  if (!coarse_correction.ok()) {
    return coarse_correction.status();
  }
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    const double coarse_pressure = (*coarse_correction)[level.aggregate_ordinals[unknown]];
    if constexpr (std::same_as<Scalar, float>) {
      const auto fine_pressure = FineValue(coarse_pressure);
      if (!fine_pressure.ok()) {
        return fine_pressure.status();
      }
      correction[unknown] += *fine_pressure;
    } else {
      correction[unknown] += coarse_pressure;
    }
  }
  status = ApplyLevel<Scalar>(level, correction, correction_flux);
  if (!status.ok()) {
    return status;
  }
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    const Scalar remaining_flux = residual[unknown] - correction_flux[unknown];
    correction[unknown] += kJacobiWeight * (remaining_flux / static_cast<Scalar>(level.diagonal[unknown]));
    if (!std::isfinite(correction[unknown])) {
      return absl::OutOfRangeError("pressure Jacobi correction is unrepresentable");
    }
  }
  return correction;
}

// Every dot visits unknowns in ascending ordinal order and multiplies after
// promotion to FP64, independently of the FP32 iterative vector operations.
absl::StatusOr<double> Dot(std::span<const float> left, std::span<const float> right) {
  double product_sum = 0.0;
  for (std::size_t unknown = 0; unknown < left.size(); ++unknown) {
    const double product = static_cast<double>(left[unknown]) * static_cast<double>(right[unknown]);
    product_sum += product;
  }
  if (!std::isfinite(product_sum)) {
    return absl::OutOfRangeError("pressure inner product is unrepresentable");
  }
  return product_sum;
}

absl::Status CopyTrueResidual(
    const MacPressureProjection3D& projection,
    const MacPressureHierarchy3D& hierarchy,
    std::span<float> residual
) {
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    const auto fine_residual = FineValue(projection.true_residuals[hierarchy.unknown_to_cell[unknown]]);
    if (!fine_residual.ok()) {
      return fine_residual.status();
    }
    residual[unknown] = *fine_residual;
  }
  return absl::OkStatus();
}

bool RecursiveResidualPasses(
    const PreparedMacPressureSystem3D& system,
    const MacPressureHierarchy3D& hierarchy,
    std::span<const float> residual,
    const MacPressureStoppingCriteria3D& criteria
) {
  double residual_norm = 0.0;
  double maximum_local_residual = 0.0;
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    const double flux_residual = residual[unknown];
    residual_norm = std::hypot(residual_norm, flux_residual);
    const double local_residual =
        std::abs(flux_residual) / system.cells()[hierarchy.unknown_to_cell[unknown]].fluid_volume;
    maximum_local_residual = std::max(maximum_local_residual, local_residual);
  }
  return residual_norm <= criteria.integrated_residual_limit && maximum_local_residual <= criteria.volume_scaled_limit;
}

// The FP32 correction is relative to a frozen FP64 pressure base. On residual
// replacement, absorb it into the base, zero the correction, and restart PCG.
// This preserves fine FP32 iteration while permitting an FP64 true-residual
// gate tighter than the rounding error of an absolute FP32 pressure vector.
absl::StatusOr<MacPressureProjection3D> EvaluateCorrection(
    const PreparedMacPressureSystem3D& system,
    const MacPressureHierarchy3D& hierarchy,
    std::span<const double> base_pressure,
    std::span<const float> correction,
    std::size_t iterations
) {
  std::vector<double> candidate(base_pressure.begin(), base_pressure.end());
  for (std::size_t unknown = 0; unknown < correction.size(); ++unknown) {
    candidate[hierarchy.unknown_to_cell[unknown]] += static_cast<double>(correction[unknown]);
    if (!std::isfinite(candidate[hierarchy.unknown_to_cell[unknown]])) {
      return absl::OutOfRangeError("pressure candidate is unrepresentable");
    }
  }
  auto projection = pressure_internal::EvaluateMacPressure3D(system, candidate);
  if (!projection.ok()) {
    return projection.status();
  }
  projection->diagnostics.iteration_count = iterations;
  projection->diagnostics.multigrid_level_count = hierarchy.levels.size();
  return projection;
}

}  // namespace

absl::StatusOr<MacPressureProjection3D>
ProjectMacPressure3D(const MacPressureInputs3D& inputs, MacPressureSolverOptions3D options) {
  if (options.max_iterations == 0 || options.max_iterations > 256) {
    return absl::InvalidArgumentError("pressure iteration budget must be in [1,256]");
  }
  const auto system = pressure_internal::PrepareMacPressureSystem3D(inputs);
  if (!system.ok()) {
    return system.status();
  }
  const auto hierarchy = pressure_internal::BuildMacPressureHierarchy3D(*system);
  if (!hierarchy.ok()) {
    return hierarchy.status();
  }
  const auto criteria = pressure_internal::MacPressureStoppingCriteria(*system);
  if (!criteria.ok()) {
    return criteria.status();
  }
  auto base_pressure = pressure_internal::InitialMacPressure3D(*system);
  if (!base_pressure.ok()) {
    return base_pressure.status();
  }
  const std::size_t count = hierarchy->unknown_to_cell.size();
  std::vector<float> correction(count, 0.0F);
  auto projection = EvaluateCorrection(*system, *hierarchy, *base_pressure, correction, 0);
  if (!projection.ok()) {
    return projection.status();
  }
  if (pressure_internal::MeetsMacPressureStoppingCriteria3D(projection->diagnostics, *criteria)) {
    return projection;
  }
  if (count == 0) {
    return absl::AbortedError("pressure original gauge rows fail the stopping gates");
  }

  std::vector<float> residual(count);
  auto status = CopyTrueResidual(*projection, *hierarchy, residual);
  if (!status.ok()) {
    return status;
  }
  std::vector<float> direction(count);
  std::vector<float> direction_flux(count);
  double previous_preconditioned_product = 0.0;
  bool restart_direction = true;
  for (std::size_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
    const auto preconditioned = ApplyVcycle<float>(*hierarchy, 0, residual);
    if (!preconditioned.ok()) {
      return preconditioned.status();
    }
    const auto preconditioned_product = Dot(residual, *preconditioned);
    if (!preconditioned_product.ok()) {
      return preconditioned_product.status();
    }
    if (*preconditioned_product <= 0.0) {
      return absl::AbortedError("pressure preconditioned residual lost positive curvature");
    }
    if (restart_direction) {
      direction = *preconditioned;
    } else {
      const auto beta = FineValue(*preconditioned_product / previous_preconditioned_product);
      if (!beta.ok()) {
        return beta.status();
      }
      for (std::size_t unknown = 0; unknown < count; ++unknown) {
        const float previous_direction = *beta * direction[unknown];
        direction[unknown] = (*preconditioned)[unknown] + previous_direction;
      }
    }
    status = ApplyLevel<float>(hierarchy->levels.front(), direction, direction_flux);
    if (!status.ok()) {
      return status;
    }
    const auto curvature = Dot(direction, direction_flux);
    if (!curvature.ok()) {
      return curvature.status();
    }
    if (*curvature <= 0.0) {
      return absl::AbortedError("pressure direction has nonpositive curvature");
    }
    const auto alpha = FineValue(*preconditioned_product / *curvature);
    if (!alpha.ok()) {
      return alpha.status();
    }
    for (std::size_t unknown = 0; unknown < count; ++unknown) {
      const float pressure_increment = *alpha * direction[unknown];
      const float flux_increment = *alpha * direction_flux[unknown];
      correction[unknown] += pressure_increment;
      residual[unknown] -= flux_increment;
      if (!std::isfinite(correction[unknown]) || !std::isfinite(residual[unknown])) {
        return absl::OutOfRangeError("pressure iterative update is unrepresentable");
      }
    }
    restart_direction = false;
    previous_preconditioned_product = *preconditioned_product;
    const bool last_iteration = iteration == options.max_iterations;
    if (last_iteration || RecursiveResidualPasses(*system, *hierarchy, residual, *criteria)) {
      projection = EvaluateCorrection(*system, *hierarchy, *base_pressure, correction, iteration);
      if (!projection.ok()) {
        return projection.status();
      }
      if (pressure_internal::MeetsMacPressureStoppingCriteria3D(projection->diagnostics, *criteria)) {
        return projection;
      }
      if (last_iteration) {
        break;
      }
      *base_pressure = projection->cell_pressures;
      std::fill(correction.begin(), correction.end(), 0.0F);
      status = CopyTrueResidual(*projection, *hierarchy, residual);
      if (!status.ok()) {
        return status;
      }
      restart_direction = true;
    }
  }
  return absl::AbortedError("pressure iteration budget exhausted before all true stopping gates passed");
}

}  // namespace fluidity
