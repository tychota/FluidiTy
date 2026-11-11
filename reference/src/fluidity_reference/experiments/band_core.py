"""Independent 3D qualification of manufactured band/core ownership transactions."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from math import dist, fsum, isfinite, sqrt
from types import MappingProxyType

from ..resampling import reseed_cell
from ..state import TrialRejected


SHAPE = (18, 4, 24)
CELLS = tuple(product(*(range(size) for size in SHAPE)))
BAND = 6.0
OVERLAP = 2.0
SUPPORT_RADIUS = sqrt(2**2 + 1.5**2 + 1.5**2)  # mixed MAC, in units of h
SEED_OFFSETS = tuple(product((-0.25, 0.25), repeat=3))
SEED_RADIUS = dist((0, 0, 0), SEED_OFFSETS[0])
LINEAGE = 17
DENSITY = 1000.0
COMPOSITION = (0.25, 0.75)
ENTHALPY_DENSITY = 2000.0
AFFINE = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.125), (0.0, 0.0, 0.0))
Cell = tuple[int, int, int]
Vector = tuple[float, float, float]


@dataclass(frozen=True)
class OccupancySample:
    fraction: float
    # Minimum signed plane/cavity distance, positive on the liquid side.
    depth: float


@dataclass(frozen=True)
class Tank:
    top: float
    sides: tuple[float, float] | None = None
    cavity: tuple[Cell, Cell] | None = None

    def depth(self, point):
        distance = self.top - point[2]
        if self.sides is not None:
            low, high = self.sides
            distance = min(distance, point[0] - low, high - point[0])
        if self.cavity is not None:
            low, high = self.cavity
            if all(a < x < b for x, a, b in zip(point, low, high)):
                return -min(min(x - a, b - x) for x, a, b in zip(point, low, high))
            nearest = tuple(min(max(x, a), b) for x, a, b in zip(point, low, high))
            distance = min(distance, dist(point, nearest))
        return distance

    def occupancy(self):
        low, high = self.sides or (0, SHAPE[0])
        samples = []
        for cell in CELLS:
            x, _, z = cell
            width = max(0, min(x + 1, high) - max(x, low))
            height = min(1, max(0, self.top - z))
            fraction = width * height
            if self.cavity is not None:
                lo, hi = self.cavity
                if all(a <= i < b for i, a, b in zip(cell, lo, hi)):
                    fraction = 0.0
            samples.append((cell, OccupancySample(fraction, self.depth(_center(cell)))))
        return tuple(samples)

    def volume(self):
        low, high = self.sides or (0, SHAPE[0])
        volume = (high - low) * SHAPE[1] * self.top
        if self.cavity is not None:
            low, high = self.cavity
            dx, dy, dz = (b - a for a, b in zip(low, high))
            volume -= dx * dy * dz
        return volume


@dataclass(frozen=True)
class Motion:
    before: Tank
    after: Tank | None
    displacement_h: float
    incoming_volume: float = 0.0
    outgoing_volume: float = 0.0
    source_volume: float = 0.0


def fixtures():
    """Return four independent unit-width, one-second manufactured tank motions."""
    deep = Tank(18.25)
    cavity = Tank(18.25, cavity=((6, 1, 8), (8, 3, 10)))
    return {
        "deep_tank": Motion(deep, deep, 0),
        "translating": Motion(Tank(18.25, (0.75, 16.75)), Tank(18.25, (1.75, 17.75)),
                              1, incoming_volume=73, outgoing_volume=-73),
        "descending": Motion(deep, Tank(17.25), 1, outgoing_volume=-80, source_volume=8),
        "closed_cavity": Motion(cavity, cavity, 0),
    }


@dataclass(frozen=True)
class Inventory:
    mass: float
    current_volume: float
    reference_volume: float
    momentum: Vector
    constituent_volume: tuple[float, float]
    enthalpy: float


@dataclass(frozen=True)
class Particle:
    id: int
    position: Vector
    velocity: Vector
    affine: tuple[Vector, Vector, Vector]
    mass: float
    current_volume: float
    reference_volume: float
    constituent_volume: tuple[float, float]
    enthalpy: float


@dataclass(frozen=True)
class FaceSample:
    particle_velocity: float
    eulerian_velocity: float
    particle_weight: float
    velocity: float
    pic_reset: bool


@dataclass(frozen=True)
class State:
    occupancy: tuple[tuple[Cell, OccupancySample], ...]
    core: Mapping[Cell, Inventory]
    particles: Mapping[Cell, tuple[Particle, ...]]
    old_grid: Mapping[tuple[int, int, int, int], FaceSample]
    target_volume: float
    lineage: int
    accepted_time: int
    next_id: int

    @property
    def geometric_volume(self):
        return fsum(sample.fraction for _, sample in self.occupancy)

    @property
    def geometric_drift(self):
        return self.geometric_volume - self.target_volume

    @property
    def old_grid_time(self):
        return self.accepted_time - 1 if self.old_grid else None

    def digest(self):
        """Hash ordered fixture state including ownership, particles, history and IDs."""
        return sha256(repr(self).encode("utf-8")).hexdigest()


def _center(cell):
    return tuple(index + 0.5 for index in cell)


def _eulerian_velocity(point):
    # Manufactured affine destination support, with exact dyadic coefficients.
    return (0.5, point[2] / 8, 0.0)


def _material(cell, volume):
    mass = DENSITY * volume
    return Inventory(mass, volume, volume, tuple(mass * v for v in _eulerian_velocity(_center(cell))),
                     tuple(volume * fraction for fraction in COMPOSITION), volume * ENTHALPY_DENSITY)


def _sum_inventory(records):
    records = tuple(records)
    return Inventory(
        mass=fsum(row.mass for row in records),
        current_volume=fsum(row.current_volume for row in records),
        reference_volume=fsum(row.reference_volume for row in records),
        momentum=tuple(fsum(row.momentum[axis] for row in records) for axis in range(3)),
        constituent_volume=tuple(fsum(row.constituent_volume[i] for row in records) for i in range(2)),
        enthalpy=fsum(row.enthalpy for row in records),
    )


def _particle_inventory(particles):
    return _sum_inventory(
        Inventory(p.mass, p.current_volume, p.reference_volume, tuple(p.mass * v for v in p.velocity),
                  p.constituent_volume, p.enthalpy)
        for p in particles
    )


def inventory(state):
    """Sum owned extensive values across actual carriers, including particle quadrature."""
    return _sum_inventory((*state.core.values(), *map(_particle_inventory, state.particles.values())))


def _reseed(cell, owned, next_id):
    if owned.current_volume != 1:
        raise TrialRejected("reseed requires a fully liquid unit cell")
    count = len(SEED_OFFSETS)
    volume = owned.current_volume
    composition = tuple(amount / volume for amount in owned.constituent_volume)
    # The selected incompressible EOS gives J = 1 at every pressure.
    values = reseed_cell(volume, owned.mass / volume, 1.0, count, composition)
    particles = []
    for i, offset in enumerate(SEED_OFFSETS):
        position = tuple(x + delta for x, delta in zip(_center(cell), offset))
        particles.append(Particle(
            id=next_id + i,
            position=position,
            velocity=_eulerian_velocity(position),
            affine=AFFINE,
            mass=float(values["mass"][i]),
            current_volume=float(values["current_volume"][i]),
            reference_volume=float(values["reference_volume"][i]),
            constituent_volume=tuple(float(f * values["current_volume"][i])
                                     for f in values["composition"][i]),
            enthalpy=owned.enthalpy / count,
        ))
    if _particle_inventory(particles) != owned:
        raise TrialRejected("reseed inventory imbalance")
    return tuple(particles)


def _snapshot(core, particles, old_grid, occupancy, target, lineage, time, next_id):
    return State(occupancy, MappingProxyType(dict(sorted(core.items()))),
                 MappingProxyType(dict(sorted(particles.items()))), MappingProxyType(dict(sorted(old_grid.items()))),
                 target, lineage, time, next_id)


def _band(sample):
    return sample.fraction == 1 and sample.depth <= BAND


def initialize(tank):
    """Build the accepted state with all interior-only particle samples stripped."""
    occupancy = tank.occupancy()
    core, particles = {}, {}
    next_id = 0
    for cell, sample in occupancy:
        if sample.fraction == 0:
            continue
        owned = _material(cell, sample.fraction)
        if _band(sample):
            particles[cell] = _reseed(cell, owned, next_id)
            next_id += len(particles[cell])
        else:
            core[cell] = owned
    return _snapshot(core, particles, {}, occupancy, tank.volume(), LINEAGE, 0, next_id)


def _old_grid(tank, occupancy, particles):
    """Blend analytic carrier velocities at wet-adjacent MAC faces before publishing history."""
    faces = {}
    for axis in range(3):
        ranges = [range(size + (i == axis)) for i, size in enumerate(SHAPE)]
        for face in product(*ranges):
            low = tuple(index - (i == axis) for i, index in enumerate(face))
            neighbors = (low, face)
            if not any(cell in occupancy and occupancy[cell].fraction > 0 for cell in neighbors):
                continue
            anchor = tuple(index + (0 if i == axis else 0.5) for i, index in enumerate(face))
            core_velocity = _eulerian_velocity(anchor)[axis]
            particle_velocity = core_velocity + 0.25
            depth = tank.depth(anchor)
            requested = min(1.0, max(0.0, (BAND - depth) / OVERLAP))
            # Deliberately suppress one otherwise supported deposit to exercise PIC fallback.
            low_mass = (axis, *face) == (2, 9, 1, 17)
            available = not low_mass and any(cell in particles for cell in neighbors)
            weight = requested if available else 0.0
            velocity = weight * particle_velocity + (1.0 - weight) * core_velocity
            faces[(axis, *face)] = FaceSample(particle_velocity, core_velocity, weight, velocity,
                                             requested > 0 and not available)
    return faces


def _validate_motion(motion):
    before, after = motion.before, motion.after
    if before.cavity != after.cavity or (before.sides is None) != (after.sides is None):
        raise TrialRejected("motion changes the manufactured topology")
    before_planes = (before.top, *(before.sides or ()))
    after_planes = (after.top, *(after.sides or ()))
    if not all(isfinite(plane) for plane in (*before_planes, *after_planes)):
        raise TrialRejected("motion has nonfinite interface planes")
    displacement = max(abs(b - a) for a, b in zip(before_planes, after_planes))
    if not isfinite(motion.displacement_h) or not displacement <= motion.displacement_h <= 1:
        raise TrialRejected("motion exceeds its declared or certified one-cell bound")


def advance(accepted, motion, *, fail_at=None):
    """Build a trial from authoritative occupancy, publishing only after both stages pass."""
    if not accepted.occupancy or motion.after is None:
        raise TrialRejected("authoritative occupancy required")
    if accepted.occupancy != motion.before.occupancy():
        raise TrialRejected("accepted occupancy does not match the interval")
    _validate_motion(motion)
    if (not all(isfinite(value) for value in
                (motion.incoming_volume, motion.outgoing_volume, motion.source_volume))
            or motion.incoming_volume < 0 or motion.outgoing_volume > 0):
        raise TrialRejected("invalid signed volume ledger")
    occupancy = motion.after.occupancy()
    previous = dict(accepted.occupancy)
    old_grid = _old_grid(motion.before, previous, accepted.particles)
    core, particles = dict(accepted.core), dict(accepted.particles)
    for cell, sample in occupancy:
        old_fraction = previous[cell].fraction
        if cell in particles and _band(sample):
            continue
        if cell in particles:
            core[cell] = _particle_inventory(particles.pop(cell))
        if sample.fraction != old_fraction:
            # The manufactured occupancy update changes only swept cells. The
            # material state there has prescribed density, affine velocity and
            # composition. The global physical target comes from a separate ledger.
            if sample.fraction > 0:
                core[cell] = _material(cell, sample.fraction)
            else:
                core.pop(cell)
        if fail_at == "transfer" and (cell in accepted.particles or sample.fraction != old_fraction):
            raise TrialRejected("injected transfer failure")

    next_id = accepted.next_id
    for cell, sample in occupancy:
        if cell in core and _band(sample):
            particles[cell] = _reseed(cell, core.pop(cell), next_id)
            next_id += len(particles[cell])
            if fail_at == "reseed":
                raise TrialRejected("injected reseed failure")

    target = fsum((accepted.target_volume, motion.incoming_volume, motion.outgoing_volume,
                   motion.source_volume))
    result = _snapshot(core, particles, old_grid, occupancy, target, accepted.lineage,
                       accepted.accepted_time + 1, next_id)
    wet = {cell for cell, sample in occupancy if sample.fraction > 0}
    if (result.core.keys() & result.particles.keys()
            or (result.core.keys() | result.particles.keys()) != wet):
        raise TrialRejected("wet cells must have exactly one inventory owner")
    expected = _sum_inventory(_material(cell, sample.fraction) for cell, sample in occupancy)
    if inventory(result) != expected:
        raise TrialRejected("owned material inventory imbalance")
    return result


def _neighbors(cell):
    for axis in range(3):
        for step in (-1, 1):
            yield tuple(index + (step if i == axis else 0) for i, index in enumerate(cell))


def air_regions(state):
    """Return exterior-connected and enclosed dry cells using 6-neighbor connectivity."""
    if not state.occupancy:
        raise TrialRejected("authoritative occupancy required")
    dry = {cell for cell, sample in state.occupancy if sample.fraction == 0}
    exterior = {cell for cell in dry if cell[2] == SHAPE[2] - 1}
    queue = deque(sorted(exterior))
    while queue:
        cell = queue.popleft()
        for adjacent in _neighbors(cell):
            if adjacent in dry and adjacent not in exterior:
                exterior.add(adjacent)
                queue.append(adjacent)
    return frozenset(exterior), frozenset(dry - exterior)


def qualify():
    """Run all fixture intervals and report measured ownership, geometry and ledger checks."""
    report = {}
    for name, motion in fixtures().items():
        before = initialize(motion.before)
        after = advance(before, motion)
        exterior, closed = air_regions(after)
        expected_closed = set()
        if motion.after.cavity is not None:
            low, high = motion.after.cavity
            expected_closed = set(product(*(range(a, b) for a, b in zip(low, high))))
        if closed != expected_closed:
            raise AssertionError("air classification disagrees with manufactured geometry")
        wet = {cell for cell, sample in after.occupancy if sample.fraction > 0}
        exterior_faces = sum(neighbor in wet for cell in exterior for neighbor in _neighbors(cell))
        cavity_faces = sum(neighbor in wet for cell in closed for neighbor in _neighbors(cell))
        expected_exterior_faces = 144 if name == "translating" else 72
        expected_cavity_faces = 24 if name == "closed_cavity" else 0
        if (exterior_faces, cavity_faces) != (expected_exterior_faces, expected_cavity_faces):
            raise AssertionError("wet/dry adjacency contains an unexpected surface")
        support_margin = BAND - SUPPORT_RADIUS - motion.displacement_h - SEED_RADIUS
        overlap_motion_margin = OVERLAP - motion.displacement_h - SEED_RADIUS
        if min(support_margin, overlap_motion_margin) <= 0:
            raise AssertionError("band support or overlap movement margin exhausted")
        if after.geometric_drift != 0 or inventory(after).current_volume != after.target_volume:
            raise AssertionError("geometric or owned inventory drift")
        report[name] = {
            "band_depth_h": BAND,
            "overlap_h": OVERLAP,
            "support_margin_h": support_margin,
            "overlap_motion_margin_h": overlap_motion_margin,
            "geometric_drift": after.geometric_drift,
            "inventory_defect": inventory(after).current_volume - after.target_volume,
            "volume_delta": after.target_volume - before.target_volume,
            "incoming_volume": motion.incoming_volume,
            "outgoing_volume": motion.outgoing_volume,
            "source_volume": motion.source_volume,
            "lineage": after.lineage,
            "core_cells": len(after.core),
            "band_cells": len(after.particles),
            "reseeded_cells": len(after.particles.keys() - before.particles.keys()),
            "retired_cells": len(before.particles.keys() - after.particles.keys()),
            "closed_air_cells": len(closed),
            "exterior_surface_faces": exterior_faces,
            "cavity_surface_faces": cavity_faces,
            "overlap_faces": sum(0 < face.particle_weight < 1 for face in after.old_grid.values()),
            "pic_reset_faces": sum(face.pic_reset for face in after.old_grid.values()),
        }
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(qualify(), indent=2))
