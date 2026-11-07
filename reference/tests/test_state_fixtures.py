import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from fluidity_reference.state import ReferenceState, TrialRejected, trial_state, accept_trial
from fluidity_reference.fixtures import write_fixture


def test_trial_rejection_and_acceptance_are_independent():
    source = np.array([1., 2.])
    state = ReferenceState({'x': source}, {'step': 0})
    def update(arrays, counters):
        arrays['x'][0] = 3
        counters['step'] += 1
    trial = trial_state(state, update)
    with pytest.raises(TrialRejected):
        accept_trial(state, trial, lambda _: False)
    assert state.arrays['x'].tobytes() == source.tobytes()
    assert state.counters['step'] == 0
    accepted = accept_trial(state, trial, lambda _: True)
    source[0] = 9
    assert accepted.arrays['x'].tolist() == [3, 2]
    with pytest.raises(ValueError):
        accepted.arrays['x'].setflags(write=True)


def test_fixture_canonical_and_rejects_bad_values(tmp_path):
    p = write_fixture(tmp_path, 'sample', {'x': np.array([1., 2.])}, {'units': {'x': 'm'}})
    raw = p.read_bytes()
    payload = json.loads(raw)
    assert payload['schema_version'] == 1
    assert payload['arrays']['x'] == {'dtype': 'float64', 'shape': [2], 'data': [1., 2.], 'units': 'm'}
    assert raw == (json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()
    assert hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        write_fixture(tmp_path, 'bad', {'x': np.array([np.inf])}, {'units': {'x': 'm'}})
    with pytest.raises(ValueError):
        write_fixture(tmp_path, 'bad', {'x': np.array([1.])}, {})


def test_cli_exports_deterministic_hand_checked_fixtures(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    for path in (a, b):
        subprocess.run([sys.executable, '-m', 'fluidity_reference.cli', 'export', str(path)], check=True)
    assert (a / 'manifest.json').read_bytes() == (b / 'manifest.json').read_bytes()
    manifest = json.loads((a / 'manifest.json').read_text())
    assert len(manifest['fixtures']) == 4
    for name, digest in manifest['fixtures'].items():
        assert hashlib.sha256((a / name).read_bytes()).hexdigest() == digest
    assert subprocess.run([sys.executable, '-m', 'fluidity_reference.cli', 'export', str(a)], capture_output=True).returncode != 0
    subprocess.run([sys.executable, '-m', 'fluidity_reference.cli', 'export', str(a), '--overwrite'], check=True)


def test_manufactured_values_against_reference_operators():
    from scipy.sparse import csr_matrix
    from fluidity_reference.cli import examples
    from fluidity_reference.grid import Grid2D
    from fluidity_reference.pressure import PressureInputs, solve_pressure
    from fluidity_reference.transfers import local_apic_fit
    from fluidity_reference.viscosity import viscous_step
    cases = examples()
    p = cases['two_cell_pressure'][0]
    result = solve_pressure(PressureInputs(Grid2D(2, 1, 1.), p['dt'][0], p['velocity'], jump=p['jump']))
    np.testing.assert_allclose(result.pressure, p['expected_pressure'])
    np.testing.assert_allclose(result.velocity, p['expected_velocity'], atol=1e-14)
    p = cases['compliant_pressure'][0]
    result = solve_pressure(PressureInputs(Grid2D(1, 1, 1.), p['dt'][0], p['velocity'], bulk_modulus=p['bulk_modulus'], source=p['source']))
    np.testing.assert_allclose(result.pressure, p['expected_pressure'])
    np.testing.assert_allclose(result.velocity, p['expected_velocity'])
    p = cases['affine_apic'][0]
    value, affine = local_apic_fit(p['position'], p['sample_positions'], p['weights'], p['values'])
    np.testing.assert_allclose(value, p['expected_value'][0])
    np.testing.assert_allclose(affine, p['expected_affine'])
    p = cases['stiff_viscosity'][0]
    result = viscous_step(p['velocity'], p['mass'], csr_matrix(p['strain_operator']), p['strain_weight'], p['boundary_strain'], p['dt'][0])
    assert result.method == 'implicit'
    np.testing.assert_allclose(result.velocity, p['expected_velocity'])
