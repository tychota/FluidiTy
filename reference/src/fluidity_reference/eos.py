"""Exact-infinite and finite liquid equations of state for reference fixtures."""

from dataclasses import dataclass, field
from fractions import Fraction
import math
import sys

import numpy as np

from .pressure import bounded_band_source


_TAIT_EXPONENT = 7.0
_LOG_FLOAT_MAX = math.log(sys.float_info.max)


def _positive_finite(value):
    return math.isfinite(value) and value > 0.0


def _tait_log_volume_ratio(pressure, pressure_scale, exponent):
    shifted_pressure = pressure_scale + pressure
    if shifted_pressure <= 0.0:
        raise ValueError('Tait pressure must exceed the negative pressure scale')
    relative_pressure = pressure / pressure_scale
    if relative_pressure < -0.5 or not math.isfinite(relative_pressure):
        log_pressure_ratio = math.log(shifted_pressure) - math.log(pressure_scale)
    else:
        log_pressure_ratio = math.log1p(relative_pressure)
    return -log_pressure_ratio / exponent


@dataclass(frozen=True)
class LiquidEOSState:
    volume_ratio: float
    log_volume_ratio: float
    density: float
    bulk_modulus: float
    sound_speed: float


@dataclass(frozen=True)
class LiquidEOS:
    """Liquid EOS in SI units, using gauge pressure and reference density."""

    reference_density: float
    reference_sound_speed: float
    law: str = 'hencky'
    reference_bulk_modulus: float = field(init=False)

    def __post_init__(self):
        if self.law not in ('hencky', 'tait'):
            raise ValueError('unknown liquid EOS')
        if not _positive_finite(self.reference_density) or not self.reference_sound_speed > 0.0:
            raise ValueError('invalid reference density or sound speed')
        speed = self.reference_sound_speed
        bulk_modulus = self.reference_density * speed * speed
        if speed != math.inf and not _positive_finite(bulk_modulus):
            raise ValueError('unrepresentable reference bulk modulus')
        if self.law == 'tait' and bulk_modulus / _TAIT_EXPONENT == 0.0:
            raise ValueError('unrepresentable Tait pressure scale')
        object.__setattr__(self, 'reference_bulk_modulus', bulk_modulus)

    def pressure(self, volume_ratio):
        """Return gauge pressure in Pa for a finite EOS and positive J."""
        if self.reference_sound_speed == math.inf:
            raise ValueError('incompressible pressure is determined by the constraint')
        if not _positive_finite(volume_ratio):
            raise ValueError('invalid volume ratio')
        log_volume_ratio = math.log(volume_ratio)
        if self.law == 'hencky':
            pressure = -self.reference_bulk_modulus * log_volume_ratio
        else:
            pressure_scale = self.reference_bulk_modulus / _TAIT_EXPONENT
            exponent = -_TAIT_EXPONENT * log_volume_ratio
            try:
                if exponent >= _LOG_FLOAT_MAX:
                    pressure = math.exp(exponent + math.log(pressure_scale)) - pressure_scale
                else:
                    pressure = pressure_scale * math.expm1(exponent)
            except OverflowError as exc:
                raise ValueError('unrepresentable Tait pressure') from exc
            if pressure <= -pressure_scale:
                raise ValueError('unrepresentable Tait pressure')
        if not math.isfinite(pressure):
            raise ValueError('unrepresentable pressure')
        return pressure

    def at_pressure(self, pressure):
        """Return the state and tangent acoustic coefficients at gauge pressure in Pa."""
        if not math.isfinite(pressure):
            raise ValueError('invalid pressure')
        if self.reference_sound_speed == math.inf:
            return LiquidEOSState(1.0, 0.0, self.reference_density, math.inf, math.inf)
        bulk_modulus = self.reference_bulk_modulus
        if self.law == 'hencky':
            log_volume_ratio = -pressure / bulk_modulus
        else:
            pressure_scale = bulk_modulus / _TAIT_EXPONENT
            log_volume_ratio = _tait_log_volume_ratio(pressure, pressure_scale, _TAIT_EXPONENT)
            bulk_modulus = _TAIT_EXPONENT * (pressure_scale + pressure)
        try:
            volume_ratio = math.exp(log_volume_ratio)
        except OverflowError as exc:
            raise ValueError('unrepresentable volume ratio') from exc
        if not _positive_finite(volume_ratio) or not _positive_finite(bulk_modulus):
            raise ValueError('unrepresentable volume ratio or bulk modulus')
        density = self.reference_density / volume_ratio
        if not _positive_finite(density):
            raise ValueError('unrepresentable density')
        sound_speed = math.sqrt(bulk_modulus) / math.sqrt(density)
        if not _positive_finite(sound_speed):
            raise ValueError('unrepresentable sound speed')
        return LiquidEOSState(volume_ratio, log_volume_ratio, density, bulk_modulus, sound_speed)


def volume_ratio_sensor(weights, particle_volume_ratios, reference_volumes):
    """Return the same-phase, reference-volume-weighted J sample or None.

    The tiny reference reduces exact representations of the binary64 inputs
    and rounds the resulting positive mean once.
    """
    if not len(weights) == len(particle_volume_ratios) == len(reference_volumes):
        raise ValueError('inconsistent volume-ratio sensor extents')
    weighted_volume = Fraction()
    weighted_ratio = Fraction()
    for weight, ratio, volume in zip(weights, particle_volume_ratios, reference_volumes):
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError('invalid sensor quadrature weight')
        if not _positive_finite(ratio) or not _positive_finite(volume):
            raise ValueError('invalid particle volume ratio or reference volume')
        contribution = Fraction(float(weight)) * Fraction(float(volume))
        weighted_volume += contribution
        weighted_ratio += contribution * Fraction(float(ratio))
    if weighted_volume == 0:
        return None
    return float(weighted_ratio / weighted_volume)


def eos_drift_source(eos, predicted_pressures, sensed_volume_ratios, fluid_volumes,
                     correction_time, maximum_rate, *, sealed):
    """Compute bounded EOS-relative drift in one connected pressure component.

    Pressures are gauge Pa, volumes are m³, correction time is seconds,
    and maximum rate is s⁻¹. None denotes a cell without a particle sensor.
    Only a sealed exact-incompressible component balances the drift to zero
    mean. The physical source and wall flux are checked independently.
    """
    pressures = np.asarray(predicted_pressures, dtype=float)
    volumes = np.asarray(fluid_volumes, dtype=float)
    if pressures.shape != (len(sensed_volume_ratios),) or volumes.shape != pressures.shape:
        raise ValueError('inconsistent EOS drift extents')
    if not np.all(np.isfinite(pressures)) or not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError('invalid pressure predictor or cell volume')
    if not _positive_finite(correction_time) or not math.isfinite(maximum_rate) or maximum_rate < 0.0:
        raise ValueError('invalid drift correction time or rate')

    band = np.array([ratio is not None for ratio in sensed_volume_ratios], dtype=bool)
    errors = np.zeros_like(pressures)
    for cell in np.flatnonzero(band):
        ratio = sensed_volume_ratios[cell]
        if not _positive_finite(ratio):
            raise ValueError('invalid sensed volume ratio')
        target = eos.at_pressure(float(pressures[cell])).log_volume_ratio
        errors[cell] = (target - math.log(ratio)) / correction_time
    if sealed and eos.reference_sound_speed == math.inf:
        return bounded_band_source(errors, volumes, band, maximum_rate)
    return np.clip(errors, -maximum_rate, maximum_rate)
