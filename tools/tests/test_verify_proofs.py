"""Process doubles exercise orchestration, never independent kernel evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('verify', Path(__file__).parents[1] / 'verify_proofs.py')
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def test_axiom_audit_requires_all_targets_and_allowlist():
    good = "\n".join(f"'{name}' depends on axioms: [propext]" for name in verify.TARGETS)
    verify.audit_axioms(good)
    with pytest.raises(ValueError):
        verify.audit_axioms(good.replace('propext', 'sorryAx', 1))
    with pytest.raises(ValueError):
        verify.audit_axioms(good.split('\n', 1)[1])


def test_checker_arguments_are_fail_closed():
    args = verify.checker_command('/checker', '/prelude', '/proof')
    assert args == ['/checker', '--prelude', '/prelude', '--axioms', 'propext,Classical.choice,Quot.sound', '--no-cache', '/proof']


def test_receipt_binds_checker_and_pins(tmp_path):
    checker = tmp_path / 'checker'
    checker.write_bytes(b'checker')
    receipt = {'schema_version': 1, 'nano_lean_commit': verify.PINS['nano_lean_commit'], 'rust_version': verify.PINS['rust_version'], 'checker_sha256': verify.sha256(checker)}
    verify.validate_receipt(receipt, checker)
    receipt['checker_sha256'] = 'wrong'
    with pytest.raises(ValueError):
        verify.validate_receipt(receipt, checker)


def test_missing_or_empty_export_fails(tmp_path):
    with pytest.raises(ValueError):
        verify.require_nonempty(tmp_path / 'missing')
    empty = tmp_path / 'empty'
    empty.touch()
    with pytest.raises(ValueError):
        verify.require_nonempty(empty)


def test_process_doubles_fail_closed(tmp_path):
    """Fake executables test command wiring and failures, not proof validity."""
    import os
    import subprocess
    import sys

    bindir = tmp_path / 'bin'
    bindir.mkdir()
    driver = '''import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if os.environ.get("FAIL_TOOL") == name: sys.exit(7)
if name == "lean": print(os.environ.get("LEAN_VERSION", "Lean (version 4.34.1, test, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release)"))
elif name == "exporter":
    if "--version" in args: print(os.environ.get("EXPORTER_VERSION", "olean-export 0.2.0"))
    elif os.environ.get("FAIL_EXPORT"): sys.exit(8)
    elif not os.environ.get("EMPTY_EXPORT"): pathlib.Path(args[args.index("-o") + 1]).write_bytes(b"fake export")
elif name == "lake":
    if "lean" in args:
        names = json.loads(os.environ["FAKE_MODULE_TARGETS"])
        for key, targets in names.items():
            if args[-1].endswith(key):
                for target in targets: print("'Fluidity." + target + "' depends on axioms: [" + os.environ.get("FAKE_AXIOM", "propext") + "]")
    elif pathlib.Path(args[1]).name == "exporter":
        sys.exit(subprocess.run([sys.executable, *args[1:]]).returncode)
elif name == "checker":
    assert "--no-cache" in args and "--prelude" in args and "--axioms" in args
'''
    for name in ('lean', 'lake', 'exporter', 'checker'):
        (bindir / name).write_text(driver)
        (bindir / name).chmod(0o755)
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({'schema_version': 1, 'nano_lean_commit': verify.PINS['nano_lean_commit'], 'rust_version': verify.PINS['rust_version'], 'checker_sha256': verify.sha256(bindir / 'checker')}))
    launcher = tmp_path / 'launch_verifier.py'
    launcher.write_text('''
import pathlib
import runpy
import subprocess
import sys

bin_dir = pathlib.Path(sys.argv.pop(1))
verifier = sys.argv.pop(1)
real_run = subprocess.run

def run_double(argv, **kwargs):
    assert pathlib.Path(argv[0]).parent == bin_dir
    return real_run([sys.executable, *argv], **kwargs)

subprocess.run = run_double
runpy.run_path(verifier, run_name="__main__")
''')
    output = tmp_path / 'output'
    cmd = [
        sys.executable, '-X', 'utf8', str(launcher), str(bindir), str(ROOT_SCRIPT),
        '--lean-bin', str(bindir), '--exporter', str(bindir / 'exporter'),
        '--checker', str(bindir / 'checker'), '--checker-provenance', str(receipt),
        '--output', str(output),
    ]

    fake_modules = json.dumps({name + '.lean': targets for name, targets in verify.MODULE_TARGETS.items()})

    def invoke(**kwargs):
        return subprocess.run(cmd, env={**os.environ, "FAKE_MODULE_TARGETS": fake_modules, **kwargs}, capture_output=True,
                              text=True, encoding='utf-8')

    success = invoke()
    assert success.returncode == 0, (success.stdout, success.stderr)
    assert (output / 'evidence.json').is_file()
    for env in ({'EXPORTER_VERSION': 'wrong'}, {'LEAN_VERSION': 'wrong'}, {'FAKE_AXIOM': 'sorryAx'}, {'FAIL_EXPORT': '1'}, {'FAIL_TOOL': 'checker'}, {'EMPTY_EXPORT': '1'}):
        result = invoke(**env)
        assert result.returncode != 0, (env, result.stdout, result.stderr)
        assert not (tmp_path / 'output/evidence.json').exists()
    (bindir / 'checker').unlink()
    assert invoke().returncode != 0


ROOT_SCRIPT = Path(__file__).parents[1] / 'verify_proofs.py'

def test_uniform_transfer_contract_inventory():
    required = {
        'Fluidity.affine_reconstruction_of_moments',
        'Fluidity.equal_samples_grid_increment_and_residual',
        'Fluidity.mac_deposit_conserves_row_moments',
        'Fluidity.finite_face_sum_eq_weighted_forward_difference',
        'Fluidity.cubic_centered_derivative_polynomials',
        'Fluidity.quantized_sum_error_bound',
        'Fluidity.uniform_mac_tile_contains_stencil',
    }
    assert required <= set(verify.TARGETS)


def test_uniform_basis_and_cap_lemmas_are_independently_exported():
    required = {
        'Fluidity.quadratic_tensor_moments',
        'Fluidity.quadratic_shifted_basis',
        'Fluidity.fixed_point_cap_partial_bound',
        'Fluidity.uniform_mac_tile_product',
    }
    assert required <= set(verify.TARGETS)


@pytest.mark.parametrize('missing', [
    'Fluidity.affine_reconstruction_of_moments',
    'Fluidity.equal_samples_grid_increment_and_residual',
    'Fluidity.mac_deposit_conserves_row_moments',
    'Fluidity.finite_face_sum_eq_weighted_forward_difference',
    'Fluidity.cubic_centered_derivative_polynomials',
    'Fluidity.quantized_sum_error_bound',
    'Fluidity.uniform_mac_tile_contains_stencil',
])
def test_new_contract_missing_closure_fails_audit(missing):
    output = '\n'.join(
        f"'{name}' depends on axioms: [propext, Classical.choice, Quot.sound]"
        for name in verify.TARGETS if name != missing
    )
    with pytest.raises(ValueError, match='missing target'):
        verify.audit_axioms(output)

PRESSURE_CONTRACTS = {
    'Fluidity.pressure_kernel_eq_component_constants',
    'Fluidity.pressure_gauge_positive_definite',
    'Fluidity.pressure_solvable_iff_component_compatible',
    'Fluidity.pressure_corrected_flux_defect_eq_neg_residual',
}


def test_pressure_topology_and_defect_inventory():
    assert PRESSURE_CONTRACTS <= set(verify.TARGETS)


@pytest.mark.parametrize('missing', sorted(PRESSURE_CONTRACTS))
def test_pressure_contract_missing_closure_fails_audit(missing):
    output = '\n'.join(
        f"'{name}' depends on axioms: [propext, Classical.choice, Quot.sound]"
        for name in verify.TARGETS if name != missing
    )
    with pytest.raises(ValueError, match='missing target'):
        verify.audit_axioms(output)


TRANSACTION_CONTRACTS = {
    'Fluidity.liquid_reject_preserves_committed_state',
    'Fluidity.liquid_finite_rejects_preserve_committed_state',
    'Fluidity.liquid_accept_advances_clock_once',
    'Fluidity.liquid_same_state_key_replays_trial',
    'Fluidity.liquid_adaptation_preserves_input_snapshot',
}


def test_liquid_transaction_contract_inventory():
    assert TRANSACTION_CONTRACTS <= set(verify.TARGETS)
    assert len(verify.TARGETS) == 38


def test_m2_trilinear_volume_contracts_are_exported_and_audited():
    required = {
        'Fluidity.trilinear_volume_weight_partition',
        'Fluidity.trilinear_volume_weight_nonnegative',
        'Fluidity.trilinear_deposition_conserves_inventory',
        'Fluidity.trilinear_batch_conserves_inventory',
    }
    assert required <= set(verify.TARGETS)


@pytest.mark.parametrize('missing', [
    'Fluidity.trilinear_volume_weight_partition',
    'Fluidity.trilinear_volume_weight_nonnegative',
    'Fluidity.trilinear_deposition_conserves_inventory',
    'Fluidity.trilinear_batch_conserves_inventory',
])
def test_m2_volume_missing_closure_fails_audit(missing):
    output = '\n'.join(
        f"'{name}' does not depend on any axioms" for name in verify.TARGETS if name != missing
    )
    with pytest.raises(ValueError, match='missing target'):
        verify.audit_axioms(output)


def test_axiom_free_closures_are_checked():
    output = '\n'.join(f"'{name}' does not depend on any axioms" for name in verify.TARGETS)
    assert verify.audit_axioms(output) == {name: [] for name in verify.TARGETS}


@pytest.mark.parametrize('missing', sorted(TRANSACTION_CONTRACTS))
def test_liquid_transaction_missing_closure_fails_audit(missing):
    output = '\n'.join(
        f"'{name}' does not depend on any axioms" for name in verify.TARGETS if name != missing
    )
    with pytest.raises(ValueError, match='missing target'):
        verify.audit_axioms(output)


def test_axiom_audit_rejects_malformed_and_mixed_duplicate_lines():
    target = 'Fluidity.liquid_reject_preserves_committed_state'
    output = '\n'.join(
        f"info: Fluidity/Proof.lean:1:0: '{name}' does not depend on any axioms" for name in verify.TARGETS
    )
    assert verify.audit_axioms(output)[target] == []
    malformed = output.replace(f"'{target}' does not depend on any axioms", f"'{target}' does not depend on any axioms?")
    with pytest.raises(ValueError, match='missing target'):
        verify.audit_axioms(malformed)
    duplicate = output + f"\ninfo: Fluidity/Proof.lean:2:0: '{target}' depends on axioms: [propext]"
    with pytest.raises(ValueError, match='duplicate axiom target'):
        verify.audit_axioms(duplicate)
