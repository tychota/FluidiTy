"""Polytropic pressure for a resolved closed-air component."""

from dataclasses import dataclass
import math


_GAMMA = 1.4
_RADIUS_FACTOR = math.cbrt(3.0 / (4.0 * math.pi))


def _positive(value, name):
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f'invalid {name}')


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f'invalid {name}')


@dataclass(frozen=True)
class CavityState:
    absolute_pressure: float
    boundary_pressure: float
    equivalent_radius: float
    step_limit: float

    def is_resolved(self, cell_width):
        """Apply the candidate two-cell-radius policy in meters."""
        _positive(cell_width, 'cell width')
        return self.equivalent_radius >= 2.0 * cell_width


@dataclass(frozen=True)
class CavityReference:
    """Frozen polytropic inventory for a cavity between topology changes."""

    initial_volume: float
    initial_pressure: float

    def __post_init__(self):
        _positive(self.initial_volume, 'initial volume')
        _positive(self.initial_pressure, 'initial absolute pressure')

    @classmethod
    def submerged_sphere(cls, volume, ambient_pressure, hydrostatic_pressure, surface_tension):
        """Initialize a sphere in mechanical equilibrium, using absolute Pa."""
        _positive(volume, 'initial volume')
        _positive(ambient_pressure, 'ambient pressure')
        _nonnegative(hydrostatic_pressure, 'hydrostatic pressure')
        _nonnegative(surface_tension, 'surface tension')
        radius = _RADIUS_FACTOR * math.cbrt(volume)
        return cls(volume, ambient_pressure + hydrostatic_pressure + surface_tension * (2.0 / radius))

    def at_volume(self, volume, ambient_pressure, liquid_density, surface_tension, liquid_curvature):
        """Evaluate the single pressure lift and acoustic bound in SI units.

        Curvature is signed outward from the liquid. It is negative on a
        spherical cavity. The resulting boundary pressure is gauge pressure.
        """
        _positive(volume, 'volume')
        _positive(ambient_pressure, 'ambient pressure')
        _positive(liquid_density, 'liquid density')
        _nonnegative(surface_tension, 'surface tension')
        if not math.isfinite(liquid_curvature):
            raise ValueError('invalid liquid curvature')

        relative_compression = (self.initial_volume - volume) / volume
        try:
            if -0.5 <= relative_compression <= 0.5:
                pressure_increment = self.initial_pressure * math.expm1(
                    _GAMMA * math.log1p(relative_compression))
                absolute_pressure = self.initial_pressure + pressure_increment
                gauge_pressure = (self.initial_pressure - ambient_pressure) + pressure_increment
            else:
                log_volume_ratio = math.log(self.initial_volume) - math.log(volume)
                absolute_pressure = math.exp(math.log(self.initial_pressure) + _GAMMA * log_volume_ratio)
                gauge_pressure = absolute_pressure - ambient_pressure
        except OverflowError as exc:
            raise ValueError('unrepresentable cavity pressure') from exc
        _positive(absolute_pressure, 'cavity absolute pressure')

        boundary_pressure = gauge_pressure + surface_tension * liquid_curvature
        if not math.isfinite(boundary_pressure):
            raise ValueError('unrepresentable cavity boundary pressure')
        equivalent_radius = _RADIUS_FACTOR * math.cbrt(volume)
        step_limit = 0.5 * equivalent_radius * math.sqrt(liquid_density) / (
            math.sqrt(3.0 * _GAMMA) * math.sqrt(absolute_pressure))
        _positive(step_limit, 'cavity timestep bound')
        return CavityState(absolute_pressure, boundary_pressure, equivalent_radius, step_limit)
