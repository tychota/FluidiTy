#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "absl/status/statusor.h"

namespace fluidity {

struct GasScalars {
  double carrier = 0.0;  // kg/m³ of fixed-composition carrier, independent of APIC marker weight.
  double fuel = 0.0;  // kg/m³.
  double soot = 0.0;  // kg/m³.
  double enthalpy = 0.0;  // J/m³.
};

struct GasTransportCell {
  double volume;  // Fixed gas volume in m³.
  GasScalars density;
  GasScalars source_rate;  // Signed change in density per second, including pressure work for enthalpy.
};

struct GasTransportFace {
  std::optional<std::size_t> tail;
  std::optional<std::size_t> head;
  double volume_flux;  // Projected aperture times velocity in m³/s, positive from tail to head.
};

struct GasTransportUpdate {
  std::vector<GasScalars> density;  // In cell input order.
  GasScalars outward_rate;  // Signed net boundary flux in kg/s or J/s by channel.
};

// Advances fixed-grid cell densities by donor-cell fluxes and explicit sources.
// Each face appears once and at least one endpoint must be a valid cell ordinal.
// A missing endpoint supplies ambient density when it is upwind. Sealed faces
// have zero flux and may be omitted. Densities must be finite and nonnegative.
// Returns InvalidArgument for invalid data, FailedPrecondition for excess
// outflow CFL or source depletion, and OutOfRange for nonfinite arithmetic.
// Errors carry no update.
absl::StatusOr<GasTransportUpdate> TransportGasScalars(
    double step_duration,
    std::span<const GasTransportCell> cells,
    std::span<const GasTransportFace> faces,
    GasScalars ambient
);

}  // namespace fluidity
