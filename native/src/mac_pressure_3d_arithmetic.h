#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <variant>

#include "mac_pressure_3d_system.h"

namespace fluidity::pressure_internal::arithmetic {
// Shared FP64 integrated assembly and independent physical correction kernels.
constexpr std::size_t kMaxPressureCells = 4096;
constexpr std::size_t kMaxPressureFaces = 24576;
constexpr double kCompatibilityRoundoff = 32.0 * std::numeric_limits<double>::epsilon();
constexpr double kIdentityRoundoff = 64.0 * std::numeric_limits<double>::epsilon();
constexpr std::array kFaceAxes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};

inline bool IsFinitePositive(double value) {
  return std::isfinite(value) && value > 0.0;
}

// Normal products have the usual FP64 relative error bound. In the
// subnormal range an inexact product can erase an O(1) defect after division
// by a small cell volume. Test exactness on normalized mantissas so the FMA
// residual itself cannot underflow.
inline bool IsRepresentablePressureProduct(double coefficient, double quantity, double product) {
  if (!std::isfinite(product)) {
    return false;
  }
  if (coefficient == 0.0 || quantity == 0.0 || std::abs(product) >= std::numeric_limits<double>::min()) {
    return true;
  }
  if (product == 0.0) {
    return false;
  }
  int coefficient_exponent = 0;
  int quantity_exponent = 0;
  const double coefficient_mantissa = std::frexp(coefficient, &coefficient_exponent);
  const double quantity_mantissa = std::frexp(quantity, &quantity_exponent);
  const double normalized_product = std::scalbn(product, -(coefficient_exponent + quantity_exponent));
  return std::fma(coefficient_mantissa, quantity_mantissa, -normalized_product) == 0.0;
}

inline auto CellSortKey(CellIndex3D index) {
  return std::tuple{index.z, index.y, index.x};
}

// Deterministic Neumaier summation retains small physical terms next to large
// cancelling fluxes. Callers check both signed and absolute sums for finiteness.
struct CompensatedSum {
  double sum = 0.0;
  double correction = 0.0;

  void Add(double term) {
    const double next_sum = sum + term;
    if (std::abs(sum) >= std::abs(term)) {
      correction += (sum - next_sum) + term;
    } else {
      correction += (term - next_sum) + sum;
    }
    sum = next_sum;
  }

  double Total() const {
    return sum + correction;
  }
};

struct FluxRow {
  CompensatedSum flux;
  CompensatedSum absolute_terms;

  void Add(double signed_term) {
    flux.Add(signed_term);
    absolute_terms.Add(std::abs(signed_term));
  }
};

template <typename Accumulate>
void ForEachIncidentCell(const FrozenMacPressureLink3D& link, Accumulate accumulate) {
  if (link.tail_cell_ordinal.has_value()) {
    accumulate(*link.tail_cell_ordinal, 1.0);
  }
  if (link.head_cell_ordinal.has_value()) {
    accumulate(*link.head_cell_ordinal, -1.0);
  }
}
// Accumulate only boundary flux explicitly supplied by the geometry. The
// oriented contributions of a two-cell link cancel in the domain total.
inline void AccumulateInwardExplicitWallFlux(
    const FrozenMacPressureLink3D& link,
    bool explicitly_prescribed,
    CompensatedSum& total
) {
  if (!explicitly_prescribed || link.tail_cell_ordinal.has_value() == link.head_cell_ordinal.has_value()) {
    return;
  }
  ForEachIncidentCell(link, [&](std::size_t, double outward_sign) {
    total.Add(-outward_sign * link.prescribed_wall_flux);
  });
}

inline absl::StatusOr<PreparedMacPressureCell3D>
FreezeCell(const MacPressureCell3D& cell, double step_duration, double full_cell_volume) {
  if (!IsFinitePositive(cell.fluid_volume) || cell.fluid_volume > full_cell_volume) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (!std::isfinite(cell.target_divergence) || !std::isfinite(cell.previous_pressure)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  double compliance = 0.0;
  if (cell.bulk_modulus != std::numeric_limits<double>::infinity()) {
    if (!IsFinitePositive(cell.bulk_modulus)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const double modulus_times_step_duration = cell.bulk_modulus * step_duration;
    if (!IsFinitePositive(modulus_times_step_duration)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    compliance = cell.fluid_volume / modulus_times_step_duration;
    if (!IsFinitePositive(compliance)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
  }

  const double integrated_target_flux = cell.fluid_volume * cell.target_divergence;
  if (!IsRepresentablePressureProduct(cell.fluid_volume, cell.target_divergence, integrated_target_flux)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  return PreparedMacPressureCell3D{
      .cell_index = cell.cell_index,
      .fluid_volume = cell.fluid_volume,
      .integrated_target_flux = integrated_target_flux,
      .compliance = compliance,
      .previous_pressure = cell.previous_pressure,
  };
}

inline std::optional<std::size_t> FindCellOrdinal(std::span<const MacPressureCell3D> cells, CellIndex3D cell_index) {
  const auto key = CellSortKey(cell_index);
  const auto cell = std::ranges::lower_bound(cells, key, {}, [](const MacPressureCell3D& row) {
    return CellSortKey(row.cell_index);
  });
  if (cell == cells.end() || CellSortKey(cell->cell_index) != key) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(cell - cells.begin());
}

inline std::optional<std::size_t>
FindTailCellOrdinal(std::span<const MacPressureCell3D> cells, FaceIndex3D face_index) {
  const auto [x, y, z] = face_index.lattice_index;
  std::array coordinates{x, y, z};
  // FaceSlot has validated the normal. An unrepresentable exterior tail at
  // INT32_MIN is absent, while the low face itself is still on the lattice.
  const auto normal = static_cast<std::size_t>(face_index.axis);
  const std::int64_t tail_coordinate = static_cast<std::int64_t>(coordinates[normal]) - 1;
  if (tail_coordinate < std::numeric_limits<std::int32_t>::min()) {
    return std::nullopt;
  }
  coordinates[normal] = static_cast<std::int32_t>(tail_coordinate);
  return FindCellOrdinal(cells, {coordinates[0], coordinates[1], coordinates[2]});
}

inline absl::StatusOr<FrozenMacPressureLink3D> FreezeLinkFromIncidence(
    const MacPressureFace3D& face,
    std::size_t face_slot,
    double full_face_area,
    double step_duration,
    std::optional<std::size_t> tail_cell_ordinal,
    std::optional<std::size_t> head_cell_ordinal
) {
  if (!std::isfinite(face.open_area) || face.open_area < 0.0 || face.open_area > full_face_area) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (!std::isfinite(face.inverse_density) || !std::isfinite(face.pressure_gradient_distance)
      || !std::isfinite(face.predicted_normal_velocity) || !std::isfinite(face.wall_normal_velocity)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }

  const bool has_both_endpoints = tail_cell_ordinal.has_value() && head_cell_ordinal.has_value();
  const bool has_one_endpoint = tail_cell_ordinal.has_value() != head_cell_ordinal.has_value();
  if (const auto* internal = std::get_if<InternalMacPressureLink3D>(&face.pressure_link)) {
    if (!has_both_endpoints || !std::isfinite(internal->pressure_jump)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  } else if (const auto* boundary = std::get_if<DirichletMacPressureLink3D>(&face.pressure_link)) {
    if (!has_one_endpoint || face.open_area == 0.0 || !std::isfinite(boundary->exterior_pressure)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  } else {
    const bool is_sealed = std::holds_alternative<SealedMacPressureLink3D>(face.pressure_link);
    if (!is_sealed || face.open_area != 0.0) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }

  double pressure_to_velocity_scale = 0.0;
  double conductance = 0.0;
  if (face.open_area > 0.0) {
    if (face.inverse_density <= 0.0 || face.pressure_gradient_distance <= 0.0) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const double step_scaled_mobility = step_duration * face.inverse_density;
    pressure_to_velocity_scale = step_scaled_mobility / face.pressure_gradient_distance;
    conductance = face.open_area * pressure_to_velocity_scale;
    if (!IsFinitePositive(pressure_to_velocity_scale) || !IsFinitePositive(conductance)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
  }
  const double blocked_area = full_face_area - face.open_area;
  double prescribed_wall_flux = 0;
  if (face.prescribed_wall_flux.has_value()) {
    prescribed_wall_flux = *face.prescribed_wall_flux;
    if (!std::isfinite(prescribed_wall_flux) || (blocked_area == 0 && prescribed_wall_flux != 0)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  } else {
    prescribed_wall_flux = blocked_area * face.wall_normal_velocity;
    if (!IsRepresentablePressureProduct(blocked_area, face.wall_normal_velocity, prescribed_wall_flux)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
  }
  return FrozenMacPressureLink3D{
      .face_slot = face_slot,
      .tail_cell_ordinal = tail_cell_ordinal,
      .head_cell_ordinal = head_cell_ordinal,
      .open_area = face.open_area,
      .blocked_area = blocked_area,
      .pressure_to_velocity_scale = pressure_to_velocity_scale,
      .conductance = conductance,
      .predicted_normal_velocity = face.predicted_normal_velocity,
      .prescribed_wall_flux = prescribed_wall_flux,
      .pressure_link = face.pressure_link,
  };
}

template <typename Inputs>
absl::StatusOr<FrozenMacPressureLink3D>
FreezeLink(const MacPressureFace3D& face, std::size_t face_slot, double full_face_area, const Inputs& inputs) {
  const auto [x, y, z] = face.face_index.lattice_index;
  return FreezeLinkFromIncidence(
      face,
      face_slot,
      full_face_area,
      inputs.step_duration,
      FindTailCellOrdinal(inputs.pressure_cells, face.face_index),
      FindCellOrdinal(inputs.pressure_cells, {x, y, z})
  );
}

// The original RHS uses one signed flux per physical contribution. Track
// their magnitudes before cancellation for the final identity roundoff bound.
inline absl::Status AssembleIntegratedRhsInto(
    std::span<const PreparedMacPressureCell3D> cells,
    std::span<const FrozenMacPressureLink3D> links,
    std::span<FluxRow> rows
) {
  std::fill(rows.begin(), rows.end(), FluxRow{});
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    rows[cell].Add(cells[cell].integrated_target_flux);
    const auto& coefficients = cells[cell];
    const double prior_compliance_flux = coefficients.compliance * coefficients.previous_pressure;
    if (!IsRepresentablePressureProduct(
            coefficients.compliance,
            coefficients.previous_pressure,
            prior_compliance_flux
        )) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    rows[cell].Add(prior_compliance_flux);
  }
  for (const auto& link : links) {
    const double predicted_fluid_flux = link.open_area * link.predicted_normal_velocity;
    if (!IsRepresentablePressureProduct(link.open_area, link.predicted_normal_velocity, predicted_fluid_flux)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    ForEachIncidentCell(link, [&](std::size_t cell, double outward_sign) {
      rows[cell].Add(-outward_sign * predicted_fluid_flux);
      rows[cell].Add(-outward_sign * link.prescribed_wall_flux);
    });
    if (const auto* internal = std::get_if<InternalMacPressureLink3D>(&link.pressure_link)) {
      const double jump_flux = link.conductance * internal->pressure_jump;
      if (!IsRepresentablePressureProduct(link.conductance, internal->pressure_jump, jump_flux)) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      ForEachIncidentCell(link, [&](std::size_t cell, double outward_sign) {
        rows[cell].Add(-outward_sign * jump_flux);
      });
    } else if (const auto* boundary = std::get_if<DirichletMacPressureLink3D>(&link.pressure_link)) {
      // The Dirichlet lift has the same sign at either orientation because the
      // known pressure at the missing endpoint carries the opposite incidence.
      const double exterior_pressure_flux = link.conductance * boundary->exterior_pressure;
      if (!IsRepresentablePressureProduct(link.conductance, boundary->exterior_pressure, exterior_pressure_flux)) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      ForEachIncidentCell(link, [&](std::size_t cell, double) {
        rows[cell].Add(exterior_pressure_flux);
      });
    }
  }
  return absl::OkStatus();
}

struct ComponentBalance {
  bool anchored = false;
  FluxRow physical_balance;
};
inline std::size_t FindComponentRoot(std::span<const std::size_t> parent, std::size_t cell) {
  while (parent[cell] != cell) {
    cell = parent[cell];
  }
  return cell;
}

// Union only positive internal edges, always toward the lower ordinal. Each
// root is then the first row in its component and the candidate gauge.
inline absl::StatusOr<std::size_t> SelectGaugesInto(
    std::span<const PreparedMacPressureCell3D> cells,
    std::span<const FrozenMacPressureLink3D> links,
    std::span<std::size_t> component_ordinals,
    std::span<ComponentBalance> balances,
    std::span<std::size_t> gauges
) {
  std::iota(component_ordinals.begin(), component_ordinals.end(), std::size_t{0});
  for (const auto& link : links) {
    if (link.conductance > 0.0 && link.tail_cell_ordinal && link.head_cell_ordinal) {
      const auto tail_root = FindComponentRoot(component_ordinals, *link.tail_cell_ordinal);
      const auto head_root = FindComponentRoot(component_ordinals, *link.head_cell_ordinal);
      component_ordinals[std::max(tail_root, head_root)] = std::min(tail_root, head_root);
    }
  }

  std::fill(balances.begin(), balances.end(), ComponentBalance{});
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const auto root = FindComponentRoot(component_ordinals, cell);
    component_ordinals[cell] = root;
    balances[root].anchored |= cells[cell].compliance > 0.0;
  }
  for (const auto& link : links) {
    if (std::holds_alternative<DirichletMacPressureLink3D>(link.pressure_link)) {
      ForEachIncidentCell(link, [&](std::size_t cell, double) {
        balances[component_ordinals[cell]].anchored = true;
      });
    }
  }

  // Internal predicted and jump fluxes cancel on a whole component. A blocked
  // face contributes prescribed wall flux only when it crosses the component
  // boundary. Test physical compatibility before any gauge is selected.
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    auto& balance = balances[component_ordinals[cell]];
    if (!balance.anchored) {
      balance.physical_balance.Add(cells[cell].integrated_target_flux);
    }
  }
  for (const auto& link : links) {
    const bool stays_in_component = link.tail_cell_ordinal && link.head_cell_ordinal
        && component_ordinals[*link.tail_cell_ordinal] == component_ordinals[*link.head_cell_ordinal];
    if (stays_in_component) {
      continue;
    }
    ForEachIncidentCell(link, [&](std::size_t cell, double outward_sign) {
      auto& balance = balances[component_ordinals[cell]];
      if (!balance.anchored) {
        balance.physical_balance.Add(-outward_sign * link.prescribed_wall_flux);
      }
    });
  }

  std::size_t gauge_count = 0;
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const auto& balance = balances[cell];
    if (component_ordinals[cell] != cell || balance.anchored) {
      continue;
    }
    const double physical_balance = balance.physical_balance.flux.Total();
    const double absolute_physical_terms = balance.physical_balance.absolute_terms.Total();
    if (!std::isfinite(physical_balance) || !std::isfinite(absolute_physical_terms)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    if (std::abs(physical_balance) > kCompatibilityRoundoff * absolute_physical_terms) {
      return absl::Status(absl::StatusCode::kFailedPrecondition, {});
    }
    gauges[gauge_count++] = cell;
  }
  return gauge_count;
}

// Apply the original integrated operator as R diag(k) R^T plus compliance.
// Missing endpoints have zero unknown pressure because exterior pressures are
// already lifted into the RHS. All rows contribute, including gauge rows.
template <typename System>
inline absl::Status ApplyOriginalPressureOperatorInto(
    const System& system,
    std::span<const double> candidate_cell_pressures,
    std::span<FluxRow> rows
) {
  std::fill(rows.begin(), rows.end(), FluxRow{});
  for (std::size_t cell = 0; cell < rows.size(); ++cell) {
    const double compliance = system.cells()[cell].compliance;
    const double pressure = candidate_cell_pressures[cell];
    const double operator_compliance_flux = compliance * pressure;
    if (!IsRepresentablePressureProduct(compliance, pressure, operator_compliance_flux)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    rows[cell].Add(operator_compliance_flux);
  }
  for (const auto& link : system.links()) {
    if (link.conductance == 0.0) {
      continue;
    }
    const double tail_pressure = link.tail_cell_ordinal ? candidate_cell_pressures[*link.tail_cell_ordinal] : 0.0;
    const double head_pressure = link.head_cell_ordinal ? candidate_cell_pressures[*link.head_cell_ordinal] : 0.0;
    const double pressure_difference = tail_pressure - head_pressure;
    const double operator_flux = link.conductance * pressure_difference;
    if (!IsRepresentablePressureProduct(link.conductance, pressure_difference, operator_flux)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    ForEachIncidentCell(link, [&](std::size_t cell, double outward_sign) {
      rows[cell].Add(outward_sign * operator_flux);
    });
  }
  return absl::OkStatus();
}

// Accumulate actual corrected fluid flux, blocked wall flux and compliance
// without using the algebraic RHS or the original operator product.
template <typename System>
inline absl::Status CorrectFacesAndAccumulatePhysicalDefectInto(
    const System& system,
    std::span<const double> candidate_cell_pressures,
    std::span<double> corrected_face_velocities,
    std::span<FluxRow> rows
) {
  std::fill(rows.begin(), rows.end(), FluxRow{});
  for (std::size_t cell = 0; cell < rows.size(); ++cell) {
    const auto& coefficients = system.cells()[cell];
    rows[cell].Add(-coefficients.integrated_target_flux);
    if (coefficients.compliance > 0.0) {
      const double pressure_change = candidate_cell_pressures[cell] - coefficients.previous_pressure;
      const double compliance_change_flux = coefficients.compliance * pressure_change;
      if (!IsRepresentablePressureProduct(coefficients.compliance, pressure_change, compliance_change_flux)) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      rows[cell].Add(compliance_change_flux);
    }
  }

  const auto links = system.links();
  for (std::size_t face = 0; face < links.size(); ++face) {
    const auto& link = links[face];
    double corrected_normal_velocity = link.predicted_normal_velocity;
    if (link.open_area > 0.0) {
      double exterior_pressure = 0.0;
      double pressure_jump = 0.0;
      if (const auto* boundary = std::get_if<DirichletMacPressureLink3D>(&link.pressure_link)) {
        exterior_pressure = boundary->exterior_pressure;
      } else if (const auto* internal = std::get_if<InternalMacPressureLink3D>(&link.pressure_link)) {
        pressure_jump = internal->pressure_jump;
      }
      const double tail_pressure =
          link.tail_cell_ordinal ? candidate_cell_pressures[*link.tail_cell_ordinal] : exterior_pressure;
      const double head_pressure =
          link.head_cell_ordinal ? candidate_cell_pressures[*link.head_cell_ordinal] : exterior_pressure;
      const double lifted_pressure_difference = tail_pressure - head_pressure + pressure_jump;
      const double normal_velocity_correction = link.pressure_to_velocity_scale * lifted_pressure_difference;
      if (!IsRepresentablePressureProduct(
              link.pressure_to_velocity_scale,
              lifted_pressure_difference,
              normal_velocity_correction
          )) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      corrected_normal_velocity += normal_velocity_correction;
    }
    if (!std::isfinite(corrected_normal_velocity)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    corrected_face_velocities[face] = corrected_normal_velocity;
    const double corrected_fluid_flux = link.open_area * corrected_normal_velocity;
    if (!IsRepresentablePressureProduct(link.open_area, corrected_normal_velocity, corrected_fluid_flux)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    ForEachIncidentCell(link, [&](std::size_t cell, double outward_sign) {
      rows[cell].Add(outward_sign * corrected_fluid_flux);
      rows[cell].Add(outward_sign * link.prescribed_wall_flux);
    });
  }
  return absl::OkStatus();
}

}  // namespace fluidity::pressure_internal::arithmetic
