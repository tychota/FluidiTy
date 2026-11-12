#include "fluidity/host/gas_transport.h"

#include <array>
#include <cmath>
#include <vector>

#include "absl/status/status.h"

namespace fluidity {
namespace {

constexpr std::array kScalarChannels{
    &GasScalars::carrier,
    &GasScalars::fuel,
    &GasScalars::soot,
    &GasScalars::enthalpy,
};

bool HasFiniteScalars(const GasScalars& scalars) {
  for (const auto channel : kScalarChannels) {
    if (!std::isfinite(scalars.*channel)) {
      return false;
    }
  }
  return true;
}

bool HasNegativeScalars(const GasScalars& scalars) {
  for (const auto channel : kScalarChannels) {
    if (scalars.*channel < 0.0) {
      return true;
    }
  }
  return false;
}

struct CellFlux {
  double outgoing_volume_rate = 0.0;
  GasScalars incoming_rate;
};

}  // namespace

absl::StatusOr<GasTransportUpdate> TransportGasScalars(
    double step_duration,
    std::span<const GasTransportCell> cells,
    std::span<const GasTransportFace> faces,
    GasScalars ambient
) {
  if (!std::isfinite(step_duration) || step_duration <= 0.0 || cells.empty() || !HasFiniteScalars(ambient)
      || HasNegativeScalars(ambient)) {
    return absl::InvalidArgumentError("gas transport requires a positive step, cells and valid ambient densities");
  }
  for (const auto& cell : cells) {
    if (!std::isfinite(cell.volume) || cell.volume <= 0.0 || !HasFiniteScalars(cell.density)
        || HasNegativeScalars(cell.density) || !HasFiniteScalars(cell.source_rate)) {
      return absl::InvalidArgumentError("invalid gas transport cell volume, density or source");
    }
  }

  std::vector<CellFlux> cell_fluxes(cells.size());
  GasTransportUpdate update;
  for (const auto& face : faces) {
    if (face.tail == face.head || (face.tail && *face.tail >= cells.size()) || (face.head && *face.head >= cells.size())
        || !std::isfinite(face.volume_flux)) {
      return absl::InvalidArgumentError("invalid gas transport face endpoints or flux");
    }
    const bool forward = face.volume_flux >= 0.0;
    const auto& donor = forward ? face.tail : face.head;
    const auto& receiver = forward ? face.head : face.tail;
    const double volume_rate = std::abs(face.volume_flux);
    const auto& donor_density = donor ? cells[*donor].density : ambient;
    if (donor) {
      cell_fluxes[*donor].outgoing_volume_rate += volume_rate;
    }
    for (const auto channel : kScalarChannels) {
      const double transported_rate = volume_rate * (donor_density.*channel);
      if (receiver) {
        cell_fluxes[*receiver].incoming_rate.*channel += transported_rate;
      } else {
        update.outward_rate.*channel += transported_rate;
      }
      if (!donor) {
        update.outward_rate.*channel -= transported_rate;
      }
    }
  }
  if (!HasFiniteScalars(update.outward_rate)) {
    return absl::OutOfRangeError("gas transport boundary flux overflowed");
  }

  update.density.reserve(cells.size());
  for (std::size_t ordinal = 0; ordinal < cells.size(); ++ordinal) {
    const auto& cell = cells[ordinal];
    const auto& flux = cell_fluxes[ordinal];
    const double outgoing_fraction = step_duration * flux.outgoing_volume_rate / cell.volume;
    if (!std::isfinite(outgoing_fraction)) {
      return absl::OutOfRangeError("gas transport outgoing volume overflowed");
    }
    if (outgoing_fraction > 1.0) {
      return absl::FailedPreconditionError("gas transport exceeds donor-cell outflow CFL");
    }
    // Evaluate the retained donor fraction directly so the transport step
    // preserves nonnegativity even at the unit outflow CFL.
    const double retained_fraction = 1.0 - outgoing_fraction;
    GasScalars density;
    for (const auto channel : kScalarChannels) {
      const double incoming_density = step_duration * (flux.incoming_rate.*channel) / cell.volume;
      const double source_density = step_duration * (cell.source_rate.*channel);
      density.*channel = retained_fraction * (cell.density.*channel) + incoming_density + source_density;
    }
    if (!HasFiniteScalars(density)) {
      return absl::OutOfRangeError("updated gas density overflowed");
    }
    if (HasNegativeScalars(density)) {
      return absl::FailedPreconditionError("gas transport source depletes cell density");
    }
    update.density.push_back(density);
  }
  return update;
}

}  // namespace fluidity
