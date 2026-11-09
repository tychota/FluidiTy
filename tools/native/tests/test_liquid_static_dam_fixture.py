"""Exercise the authenticated static-sphere and dam-break native fixtures."""

import json
import math
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def native_tools():
    fixture = os.environ.get("FLUIDITY_ACCEPTANCE_FIXTURE")
    solver = os.environ.get("FLUIDITY_ACCEPTANCE_SOLVER")
    if not fixture or not solver:
        pytest.skip("set FLUIDITY_ACCEPTANCE_FIXTURE and FLUIDITY_ACCEPTANCE_SOLVER for native fixture tests")
    return Path(fixture), Path(solver)


def native_records(executable, *arguments, success=True):
    result = subprocess.run(
        [str(executable), *map(str, arguments)], capture_output=True, text=True, timeout=60,
    )
    if success:
        assert result.returncode == 0, result.stderr
        return [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode != 0
    assert result.stdout == ""
    return result.stderr


def author_scene(native_tools, case, path):
    fixture, solver = native_tools
    authored, = native_records(fixture, "write-scene", case, "cpu", solver, path)
    assert authored["case"] == case
    assert authored["backend"] == "cpu"
    seed_path = path.with_suffix(".flrs.flst")
    seed_bytes = seed_path.read_bytes()
    assert authored["seed_byte_length"] == len(seed_bytes)
    seed, = native_records(fixture, "read-seed", seed_path, authored["seed_digest"])
    assert seed["state_digest"] == authored["seed_digest"]
    return authored, seed, seed_path


def assert_resolved_identity(resolved, authored):
    for field in ("resolved_fingerprint", "execution_id", "backend", "solver_digest",
                  "seed_digest", "seed_byte_length", "energy_blend"):
        assert resolved[field] == authored[field]


def assert_initial_particles(seed, positions, reference_volume):
    assert seed["frame"] == seed["time_seconds"] == seed["substep"] == seed["accepted_step"] == seed["epoch"] == 0
    assert seed["random_seed"] == 0
    assert seed["next_particle_id"] == len(positions) + 1
    particles = seed["particles"]
    assert [particle["id"] for particle in particles] == list(range(1, len(positions) + 1))
    assert [particle["world_position"] for particle in particles] == positions
    for particle in particles:
        assert particle["world_velocity"] == [0, 0, 0]
        assert particle["velocity_slopes"] == [[0, 0, 0]] * 3
        assert particle["affine_enabled"] is True
        assert particle["pic_initialized"] is False
        assert particle["reference_volume"] == pytest.approx(reference_volume, rel=1e-14)
        assert particle["mass"] == pytest.approx(1000 * reference_volume, rel=1e-14)
        assert particle["volume_ratio"] == 1


def assert_scene_defaults(resolved):
    assert resolved["world_origin_m"] == [0, 0, 0]
    assert resolved["density_kg_per_m3"] == 1000
    assert resolved["ambient_gauge_pa"] == 0
    assert resolved["random_seed"] == 0
    assert resolved["incompressible"] is True
    assert resolved["source_count"] == 0
    assert resolved["maximum_dt_seconds"] == 0.01


@pytest.mark.parametrize(("case", "cells_per_radius", "count"), [
    ("static-sphere-coarse", 3, 136),
    ("static-sphere-fine", 5, 552),
])
def test_native_static_sphere_authenticates_exact_field_and_ordered_seed(native_tools, tmp_path, case,
                                                                        cells_per_radius, count):
    fixture, _ = native_tools
    scene_path = tmp_path / "scene.flrs"
    authored, seed, _ = author_scene(native_tools, case, scene_path)
    fingerprint = authored["resolved_fingerprint"]
    resolved, = native_records(fixture, "read-static-sphere-scene", scene_path, fingerprint)
    assert resolved["event"] == "resolved_static_sphere"
    assert_resolved_identity(resolved, authored)
    assert_scene_defaults(resolved)

    radius = 0.125
    h = radius / cells_per_radius
    assert resolved["sphere_center_m"] == [0, 0, 0]
    assert resolved["sphere_radius_m"] == radius
    assert resolved["cell_size_m"] == h
    assert resolved["gravity_m_per_s2"] == [0, 0, 0]
    assert resolved["surface_tension_n_per_m"] == 1
    assert resolved["energy_blend"] == 0.6
    lattice = range(1 - 2 * cells_per_radius, 2 * cells_per_radius, 2)
    positions = [
        [x * (h / 2), y * (h / 2), z * (h / 2)]
        for z in lattice for y in lattice for x in lattice
        if x * x + y * y + z * z < (2 * cells_per_radius)**2
    ]
    assert len(positions) == count
    assert_initial_particles(seed, positions, 4 * math.pi * radius**3 / (3 * count))
    assert "fingerprint" in native_records(
        fixture, "read-static-sphere-scene", scene_path, "0" * 64, success=False,
    )


def test_static_sphere_reader_requires_an_analytic_sphere_without_obstacles(native_tools, tmp_path):
    fixture, solver = native_tools
    tank_path = tmp_path / "hydrostatic.flrs"
    authored, = native_records(fixture, "write-scene", "hydrostatic-column", "cpu", solver, tank_path)
    assert "analytic sphere" in native_records(
        fixture, "read-static-sphere-scene", tank_path, authored["resolved_fingerprint"], success=False,
    )


def test_native_dam_break_authenticates_one_seed_across_distinct_energy_blends(native_tools, tmp_path):
    fixture, solver = native_tools
    h = 0.0625
    positions = [
        [(x + 0.5) * h, (y + 0.5) * h, (z + 0.5) * h]
        for z in range(4) for y in range(8) for x in range(8)
    ]
    seeds = set()
    fingerprints = set()
    execution_ids = set()
    blend_cases = (("pic", 0), ("default", 0.6), ("retained", 1))
    for case_index, (suffix, blend) in enumerate(blend_cases):
        scene_path = tmp_path / f"{suffix}.flrs"
        authored, seed, seed_path = author_scene(native_tools, f"dam-break-{suffix}", scene_path)
        seeds.add(seed_path.read_bytes())
        fingerprints.add(authored["resolved_fingerprint"])
        execution_ids.add(authored["execution_id"])
        assert_initial_particles(seed, positions, h**3)

        # Leave room for digest-named blob files on Windows.
        cache_path = tmp_path / f"c{case_index}"
        records = native_records(
            solver, "run", "--stage", "simulate", "--scene", scene_path, "--initial-state", seed_path,
            "--backend", "cpu", "--cache", cache_path, "--frame-end", 1,
        )
        assert records[-1]["event"] == "result"
        assert records[-1]["frame"] == 1
        resolved, = native_records(fixture, "read-tank-scene", cache_path, authored["resolved_fingerprint"])
        assert resolved["event"] == "resolved_tank"
        assert_resolved_identity(resolved, authored)
        assert_scene_defaults(resolved)
        assert resolved["cell_size_m"] == h
        assert resolved["gravity_m_per_s2"] == [0, -9.81, 0]
        assert resolved["surface_tension_n_per_m"] == 0
        assert resolved["energy_blend"] == blend
        assert resolved["sphere_union_radius_m"] == 0.9 * h
        assert [resolved[key] for key in ("tank_low_x_m", "tank_high_x_m", "tank_floor_y_m", "tank_low_z_m",
                                          "tank_high_z_m", "piston_velocity_x_m_per_s")] == [0, 2, 0, 0, 0.25, 0]
        assert "surface_height_y_m" not in resolved
    assert len(seeds) == 1
    assert len(fingerprints) == len(execution_ids) == 3
