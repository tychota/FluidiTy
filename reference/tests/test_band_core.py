from dataclasses import replace
from math import isclose

import numpy as np
import pytest

from fluidity_reference.experiments.band_core import (
    TrialRejected,
    advance,
    air_regions,
    fixtures,
    initialize,
    inventory,
    qualify,
)


def test_manufactured_cases_report_margin_and_component_balance():
    report = qualify()
    assert set(report) == {"deep_tank", "translating", "descending", "closed_cavity"}
    for row in report.values():
        assert row["band_depth_h"] == 6
        assert row["overlap_h"] == 2
        assert row["support_margin_h"] > 0
        assert row["overlap_motion_margin_h"] > 0
        assert row["geometric_drift"] == 0
        assert row["inventory_defect"] == 0
        assert row["lineage"] == 17
        assert row["overlap_faces"] > 0
        assert row["pic_reset_faces"] > 0
        assert row["core_cells"] > 0
        assert row["band_cells"] > 0
    for name in ("translating", "descending"):
        assert report[name]["reseeded_cells"] > 0
        assert report[name]["retired_cells"] > 0
    assert report["translating"]["volume_delta"] == 0
    assert report["descending"]["volume_delta"] == -72
    assert report["closed_cavity"]["closed_air_cells"] == 8
    assert report["closed_cavity"]["cavity_surface_faces"] == 24
    assert report["deep_tank"]["exterior_surface_faces"] == 72
    assert report["deep_tank"]["closed_air_cells"] == 0


def test_cavity_sign_and_full_cell_reseed_survive_stripped_core():
    cavity = fixtures()["closed_cavity"]
    state = initialize(cavity.before)
    assert (15, 1, 3) in state.core
    assert (15, 1, 3) not in state.particles
    assert dict(state.occupancy)[(15, 1, 3)].fraction == 1
    assert dict(state.occupancy)[(6, 1, 8)].fraction == 0
    exterior, closed = air_regions(state)
    assert (6, 1, 8) not in exterior
    assert closed == frozenset(
        (x, y, z) for x in (6, 7) for y in (1, 2) for z in (8, 9)
    )

    motion = fixtures()["translating"]
    old = initialize(motion.before)
    new = advance(old, motion)
    wet = dict(new.occupancy)
    assert set(new.core).isdisjoint(new.particles)
    assert set(new.core) | set(new.particles) == {
        cell for cell, sample in wet.items() if sample.fraction > 0
    }
    assert any(0 < wet[cell].fraction < 1 for cell in new.core)
    assert all(wet[cell].fraction == 1 for cell in new.particles)
    created = set(new.particles) - set(old.particles)
    assert created
    ids = []
    for cell, particles in new.particles.items():
        assert len(particles) == 8
        ids.extend(particle.id for particle in particles)
        for particle in particles:
            assert all(i < x < i + 1 for i, x in zip(cell, particle.position))
            assert particle.current_volume == particle.reference_volume == 0.125
            assert particle.mass == 125
            assert particle.constituent_volume == (0.03125, 0.09375)
            assert particle.enthalpy == 250
            if cell in created:
                np.testing.assert_allclose(particle.velocity, (0.5, particle.position[2] / 8, 0))
                np.testing.assert_allclose(
                    particle.affine, ((0, 0, 0), (0, 0, 0.125), (0, 0, 0)), atol=1e-12
                )
    assert len(set(ids)) == len(ids)
    assert inventory(new) == inventory(old)
    assert new.target_volume == old.target_volume


def test_blended_old_grid_and_low_mass_pic_fallback():
    motion = fixtures()["descending"]
    state = advance(initialize(motion.before), motion)
    blended = [sample for sample in state.old_grid.values() if 0 < sample.particle_weight < 1]
    assert blended
    for sample in blended:
        assert isclose(
            sample.velocity,
            sample.particle_weight * sample.particle_velocity
            + (1 - sample.particle_weight) * sample.eulerian_velocity,
            abs_tol=1e-12,
        )
    fallback = [sample for sample in state.old_grid.values() if sample.pic_reset]
    assert fallback
    assert all(sample.velocity == sample.eulerian_velocity for sample in fallback)
    assert all(sample.particle_weight == 0 for sample in fallback)


def test_missing_occupancy_and_trial_failures_leave_accepted_state_untouched():
    motion = fixtures()["descending"]
    accepted = initialize(motion.before)
    initial_digest = accepted.digest()
    expected = advance(accepted, motion)
    for point in ("transfer", "reseed"):
        with pytest.raises(TrialRejected, match=point):
            advance(accepted, motion, fail_at=point)
        assert accepted.digest() == initial_digest
        assert advance(accepted, motion).digest() == expected.digest()
    with pytest.raises(TrialRejected, match="occupancy"):
        advance(accepted, replace(motion, after=None))
    with pytest.raises(TrialRejected, match="occupancy"):
        advance(replace(accepted, occupancy=()), motion)
    with pytest.raises(TrialRejected, match="motion"):
        advance(accepted, replace(motion, displacement_h=2))
    assert accepted.digest() == initial_digest

    drift = advance(accepted, replace(motion, source_volume=motion.source_volume + 0.5))
    assert drift.geometric_volume == expected.geometric_volume
    assert drift.target_volume == expected.target_volume + 0.5
    assert drift.geometric_drift == -0.5

def test_malformed_motion_cannot_understate_geometric_displacement():
    motion = fixtures()["descending"]
    accepted = initialize(motion.before)
    with pytest.raises(TrialRejected, match="motion"):
        advance(accepted, replace(motion, displacement_h=0))
    with pytest.raises(TrialRejected, match="motion"):
        advance(accepted, replace(motion, after=replace(motion.after, top=16.25)))


def test_two_intervals_preserve_lineage_and_replay_after_reseed_rejection():
    forward = fixtures()["translating"]
    reverse = replace(forward, before=forward.after, after=forward.before)
    original = replace(initialize(forward.before), lineage=23)
    first = advance(original, forward)
    before_rejection = first.digest()
    with pytest.raises(TrialRejected, match="reseed"):
        advance(first, reverse, fail_at="reseed")
    assert first.digest() == before_rejection
    replay = advance(first, reverse)
    assert replay.digest() == advance(advance(original, forward), reverse).digest()
    assert replay.accepted_time == 2
    assert replay.old_grid_time == 1
    assert replay.next_id > first.next_id > original.next_id
    assert replay.lineage == original.lineage
    assert replay.occupancy == original.occupancy
    assert inventory(replay) == inventory(original)
