#include "fluidity/host/liquid_eos.h"

#include <cmath>
#include <limits>

namespace fluidity {
namespace {

constexpr double kTaitExponent = 7.0;

bool PositiveFinite(double value) {
  return std::isfinite(value) && value > 0.0;
}

// Multiplication in exponent space avoids intermediate range loss when the
// density compensates an unusually large or small reference speed.
double ReferenceBulkModulus(LiquidEos3D eos) {
  int density_exponent;
  int speed_exponent;
  const double density_fraction = std::frexp(eos.reference_density, &density_exponent);
  const double speed_fraction = std::frexp(eos.reference_sound_speed, &speed_exponent);
  return std::scalbn(density_fraction * speed_fraction * speed_fraction, density_exponent + 2 * speed_exponent);
}

}  // namespace

absl::StatusOr<LiquidEosPressureState3D> EvaluateLiquidEosAtPressure3D(LiquidEos3D eos, double gauge_pressure) {
  if (eos.law != LiquidEosLaw::kHencky && eos.law != LiquidEosLaw::kTait) {
    return absl::InvalidArgumentError("unknown liquid EOS law");
  }
  if (!PositiveFinite(eos.reference_density) || !(eos.reference_sound_speed > 0.0) || !std::isfinite(gauge_pressure)) {
    return absl::InvalidArgumentError("invalid liquid EOS reference data or gauge pressure");
  }
  if (std::isinf(eos.reference_sound_speed)) {
    return LiquidEosPressureState3D{
        .volume_ratio = 1.0,
        .log_volume_ratio = 0.0,
        .density = eos.reference_density,
        .bulk_modulus = std::numeric_limits<double>::infinity(),
        .sound_speed = std::numeric_limits<double>::infinity(),
    };
  }
  double bulk_modulus = ReferenceBulkModulus(eos);
  if (!PositiveFinite(bulk_modulus)) {
    return absl::OutOfRangeError("liquid reference bulk modulus is unrepresentable");
  }
  double log_volume_ratio = 0.0;
  if (eos.law == LiquidEosLaw::kHencky) {
    log_volume_ratio = -gauge_pressure / bulk_modulus;
  } else {
    const double reference_bulk_modulus = bulk_modulus;
    // FMA evaluates K0 + n*p directly. Dividing K0 by n and multiplying
    // back would lose bits at the tensile limit and for subnormal scales.
    bulk_modulus = std::fma(kTaitExponent, gauge_pressure, reference_bulk_modulus);
    if (bulk_modulus <= 0.0) {
      return absl::InvalidArgumentError("Tait pressure must exceed the negative pressure scale");
    }
    if (!PositiveFinite(bulk_modulus)) {
      return absl::OutOfRangeError("Tait tangent bulk modulus is unrepresentable");
    }
    const double pressure_scale = reference_bulk_modulus / kTaitExponent;
    const double relative_pressure = std::isnormal(pressure_scale)
        ? gauge_pressure / pressure_scale
        : (gauge_pressure / reference_bulk_modulus) * kTaitExponent;
    // Log1p retains small predictor strain. Near the tensile limit use the
    // fused tangent to avoid rounding 1 + p/B to zero or losing its low bits.
    const double log_pressure_ratio = relative_pressure < -0.5 || !std::isfinite(relative_pressure)
        ? std::log(bulk_modulus) - std::log(reference_bulk_modulus)
        : std::log1p(relative_pressure);
    log_volume_ratio = -log_pressure_ratio / kTaitExponent;
  }
  const double volume_ratio = std::exp(log_volume_ratio);
  const double density = eos.reference_density / volume_ratio;
  const double sound_speed = std::sqrt(bulk_modulus) / std::sqrt(density);
  if (!PositiveFinite(volume_ratio) || !PositiveFinite(density) || !PositiveFinite(sound_speed)) {
    return absl::OutOfRangeError("liquid EOS state is unrepresentable");
  }
  return LiquidEosPressureState3D{
      .volume_ratio = volume_ratio,
      .log_volume_ratio = log_volume_ratio,
      .density = density,
      .bulk_modulus = bulk_modulus,
      .sound_speed = sound_speed,
  };
}

}  // namespace fluidity
