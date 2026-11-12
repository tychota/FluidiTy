"""Volume-weighted liquid composition and one oriented pressure link per phase pair."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType


def _valid_id(value):
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class Mixture:
    volume: float
    mass: float
    density: float
    volume_fractions: Mapping[int, float]


def mix_constituent_volumes(volumes_by_substance, densities_by_substance):
    """Mix current constituent volumes within one phase under additive volumes.

    Volumes are in m³ and densities in kg/m³. Return fractions keyed by
    stable substance ID in ascending order. Call separately for each phase.
    """
    if not all(_valid_id(substance) for substance in volumes_by_substance):
        raise ValueError('invalid substance ID')
    volumes = {}
    masses = []
    for substance in sorted(volumes_by_substance):
        volume = volumes_by_substance[substance]
        if not math.isfinite(volume) or volume < 0.0:
            raise ValueError('invalid constituent volume')
        density = densities_by_substance.get(substance, 0.0)
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError('invalid or missing constituent density')
        volumes[substance] = volume
        masses.append(volume * density)
    try:
        volume = math.fsum(volumes.values())
        mass = math.fsum(masses)
    except OverflowError as exc:
        raise ValueError('unrepresentable mixture volume or mass') from exc
    if volume <= 0.0 or mass <= 0.0 or not math.isfinite(mass):
        raise ValueError('invalid or unrepresentable mixture volume or mass')
    fractions = MappingProxyType({substance: value / volume for substance, value in volumes.items()})
    density = mass / volume
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError('unrepresentable mixture density')
    return Mixture(volume, mass, density, fractions)


@dataclass(frozen=True)
class InterfacePressureLink:
    mobility: float
    jump: float


@dataclass(frozen=True)
class InterfacePair:
    """One canonical phase-pair record, with curvature signed from a to b."""

    phase_a: int
    phase_b: int
    surface_tension: float

    def __post_init__(self):
        if not (_valid_id(self.phase_a) and _valid_id(self.phase_b) and self.phase_a < self.phase_b):
            raise ValueError('phase IDs must be distinct and in ascending order')
        if not math.isfinite(self.surface_tension) or self.surface_tension < 0.0:
            raise ValueError('invalid surface tension')

    def pressure_link(self, tail_phase, tail_fraction, densities_by_phase, curvature_a_to_b,
                      normal_viscous_jump_a_to_b=0.0):
        """Return mobility in m³/kg and the tail-to-head pressure jump in Pa."""
        if tail_phase not in (self.phase_a, self.phase_b):
            raise ValueError('tail phase does not belong to pair')
        if not math.isfinite(tail_fraction) or not 0.0 <= tail_fraction <= 1.0:
            raise ValueError('invalid tail fraction')
        if not math.isfinite(curvature_a_to_b) or not math.isfinite(normal_viscous_jump_a_to_b):
            raise ValueError('invalid curvature or viscous stress jump')

        forward = tail_phase == self.phase_a
        head_phase = self.phase_b if forward else self.phase_a
        tail_density = densities_by_phase.get(tail_phase, 0.0)
        head_density = densities_by_phase.get(head_phase, 0.0)
        if not all(math.isfinite(density) and density > 0.0 for density in (tail_density, head_density)):
            raise ValueError('invalid or missing phase density')
        resistance = tail_fraction * tail_density + (1.0 - tail_fraction) * head_density
        jump_a_to_b = normal_viscous_jump_a_to_b - self.surface_tension * curvature_a_to_b
        if resistance <= 0.0:
            raise ValueError('unrepresentable interface resistance')
        mobility = 1.0 / resistance
        if not math.isfinite(mobility) or mobility <= 0.0 or not math.isfinite(jump_a_to_b):
            raise ValueError('unrepresentable interface mobility or jump')
        return InterfacePressureLink(mobility, jump_a_to_b if forward else -jump_a_to_b)
