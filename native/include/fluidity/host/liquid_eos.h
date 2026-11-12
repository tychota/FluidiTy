#pragma once

#include "absl/status/statusor.h"

namespace fluidity {

enum class LiquidEosLaw {
  kHencky,
  kTait
};

struct LiquidEos3D {
  double reference_density;  // kg/m³ at J = 1.
  double reference_sound_speed;  // m/s. Positive infinity selects incompressibility.
  LiquidEosLaw law = LiquidEosLaw::kHencky;
};

struct LiquidEosPressureState3D {
  double volume_ratio;  // Current volume divided by reference volume.
  double log_volume_ratio;
  double density;  // kg/m³.
  double bulk_modulus;  // Pa, frozen at the supplied pressure.
  double sound_speed;  // m/s at the supplied pressure.
};

// Evaluates the liquid EOS and its tangent at a finite gauge pressure in Pa.
// Tait uses exponent seven and requires pressure > -reference_bulk_modulus/7.
// Infinite reference sound speed returns J=1 and infinite modulus and sound
// speed. Invalid input or pressure domain returns InvalidArgument.
// Unrepresentable coefficients or state return OutOfRange. Errors carry no state.
absl::StatusOr<LiquidEosPressureState3D> EvaluateLiquidEosAtPressure3D(LiquidEos3D eos, double gauge_pressure);

}  // namespace fluidity
