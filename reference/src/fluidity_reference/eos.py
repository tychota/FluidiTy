"""Exact-infinite and finite liquid equations of state for reference fixtures."""

from dataclasses import dataclass, field
import math
import sys


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
